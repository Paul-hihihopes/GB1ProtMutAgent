"""LLM 接入层。

两种后端，对上层 Agent 完全透明：
  * "api"     : 任意 OpenAI 兼容接口(OpenAI / DeepSeek / Qwen / 本地 vLLM 均可)，
                通过环境变量 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL 配置。
  * "offline" : 内置的确定性推理引擎。它按照与真实 LLM 完全相同的 prompt 规范
                产出同样 schema 的 JSON，使整套流程在没有 API Key、没有网络的
                环境下依然可复现。

无论走哪个后端，prompt 都会被完整构造并记录在 transcript 中，
因此报告里可以直接展示 Agent 的"思考过程"。
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Callable
from uuid import uuid4

from .config import RESULT_DIR, load_env, rel
load_env()   # 读取项目根目录的 .env（若存在）

DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "deepseek-flash")


def _repair_truncated(fragment: str) -> dict | list | None:
    """抢救被 max_tokens 截断的 JSON。

    真实大模型在输出长列表时经常被截断，得到的是一段合法前缀。
    这里逐字符跟踪括号/字符串状态，回退到最后一个完整的元素边界，
    再把没闭合的括号补齐，从而保住已经生成的那部分内容。
    """
    stack, in_str, esc = [], False, False
    safe_cut = None          # 最后一个「元素结束」的位置（逗号或右括号之后）
    for i, c in enumerate(fragment):
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "{[":
            stack.append(c)
        elif c in "}]":
            if stack:
                stack.pop()
            safe_cut = i + 1
        elif c == "," and len(stack) <= 2:
            safe_cut = i          # 截到逗号之前，丢掉未写完的那个元素
    if safe_cut is None:
        return None

    head = fragment[:safe_cut].rstrip().rstrip(",")
    # 重新统计截断点之后还有哪些括号没闭合
    stack, in_str, esc = [], False, False
    for c in head:
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "{[":
            stack.append(c)
        elif c in "}]" and stack:
            stack.pop()
    closing = "".join("}" if c == "{" else "]" for c in reversed(stack))
    try:
        return json.loads(head + closing)
    except json.JSONDecodeError:
        return None


def _extract_json(text: str) -> dict | list | None:
    """从 LLM 回复里稳健地抠出 JSON。

    依次尝试：``` 代码块 -> 完整括号匹配 -> 截断修复。
    """
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    # 完整匹配失败：多半是被 max_tokens 截断，尝试抢救前缀
    for opener in ("{", "["):
        start = text.find(opener)
        if start != -1:
            repaired = _repair_truncated(text[start:])
            if repaired is not None:
                return repaired
    return None


@dataclass
class LLMTurn:
    """一次完整的 Agent-LLM 交互记录。"""
    step: str
    role: str
    backend: str          # 本次结果的真实来源: "api" / "api(truncated-repaired)" / "offline"
    model: str
    system: str
    user: str
    response: str
    parsed: dict | list | None
    latency_s: float
    round_id: int = 0
    reasoning: str = ""   # 推理型模型返回的思维链（offline 后端为空）
    strategy: str = ""
    seed: int | None = None
    timestamp: str = ""
    run_id: str = ""
    phase: str = "standalone"
    call_index: int = 0
    step_call_index: int = 0

    def to_record(self) -> dict:
        """文件导出与交互 Demo 使用同一份可追溯记录。"""
        return {"step": self.step, "role": self.role, "round": self.round_id,
                "backend": self.backend, "model": self.model,
                "system": self.system, "user": self.user,
                "response": self.response, "reasoning": self.reasoning,
                "strategy": self.strategy, "seed": self.seed, "timestamp": self.timestamp,
                "latency_s": self.latency_s, "run_id": self.run_id, "phase": self.phase,
                "call_index": self.call_index, "step_call_index": self.step_call_index}


@dataclass
class LLMClient:
    backend: str = "auto"          # auto / api / offline
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key: str | None = None
    temperature: float = 0.3
    # 推理型模型的 max_tokens 是「思维链 + 正式回答」的总预算，
    # 需要为思维链留出足够空间。
    max_tokens: int = 32768
    timeout: int = 300
    max_retries: int = 3
    json_mode: bool = True          # 走 OpenAI 兼容的 response_format=json_object
    allow_fallback: bool = True
    strategy: str = ""
    seed: int | None = None
    run_id: str = field(default_factory=lambda: uuid4().hex)
    phase: str = "standalone"
    transcript: list[LLMTurn] = field(default_factory=list)
    n_api_calls: int = 0            # 真正由大模型返回并成功解析的次数
    n_repaired: int = 0             # 其中被截断后抢救回来的次数
    n_offline_calls: int = 0        # 回退到离线引擎的次数
    n_retries: int = 0
    n_truncated: int = 0            # finish_reason == "length" 的次数
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0       # 推理型模型的思维链 token（含在 completion 内）
    errors: list[str] = field(default_factory=list)
    last_finish_reason: str = ""
    last_reasoning: str = ""        # 最近一次调用的思维链，便于在报告里展示

    def __post_init__(self):
        if self.backend not in {"auto", "api", "offline"}:
            raise ValueError("backend 必须为 auto、api 或 offline")
        self.api_key = self.api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if self.backend == "auto":
            self.backend = "api" if self.api_key else "offline"
        if self.backend == "api" and not self.api_key:
            if not self.allow_fallback:
                raise RuntimeError("API 实验需要配置 LLM_API_KEY")
            print("[LLM] 未检测到 API Key，自动回退到内置离线推理引擎。")
            self.backend = "offline"

    def begin_run(self, phase: str) -> str:
        """开始一次独立运行，保留已有交互与总用量。"""
        self.run_id, self.phase = uuid4().hex, phase
        return self.run_id

    # 描述
    @property
    def description(self) -> str:
        if self.backend == "api":
            return f"OpenAI 兼容接口 · {self.model} @ {self.base_url}"
        return "离线规则引擎（Rule-Agent，使用确定性统计与搜索规则）"

    def banner(self) -> None:
        print(f"[LLM] 后端 = {self.backend}  |  {self.description}")
        if self.backend == "offline":
            print("[LLM] 提示：设置环境变量 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL 后，"
                  "同一套代码会自动切换到真实大模型。")

    # 调用
    def chat(self, step: str, role: str, system: str, user: str,
             offline_handler: Callable[[], dict | list] | None = None,
             round_id: int = 0,
             validate: Callable[[dict | list], bool] | None = None) -> dict | list:
        """统一入口：返回结构化结果(dict/list)，并把整段交互写入 transcript。

        validate 用来检查大模型返回的 JSON 是否含有下游真正需要的字段；
        不通过则判定为无效输出，继续重试或回退到离线引擎。
        """
        t0 = time.time()
        text, parsed, source, reasoning = "", None, "offline", ""

        if self.backend == "api":
            for attempt in range(1, self.max_retries + 1):
                try:
                    text = self._call_api(system, user)
                    truncated = self.last_finish_reason == "length"
                    cand = _extract_json(text)
                    if cand is not None and (validate is None or validate(cand)):
                        parsed = cand
                        reasoning = self.last_reasoning
                        self.n_api_calls += 1
                        if truncated:
                            self.n_repaired += 1
                            source = "api(truncated-repaired)"
                            print(f"[LLM] {step}: 输出被 max_tokens 截断，"
                                  f"已从合法前缀中抢救出结构化结果。")
                        else:
                            source = "api"
                        break
                    msg = (f"{step}: 第 {attempt} 次返回无法解析为可用 JSON"
                           + ("（finish_reason=length，输出被截断）" if truncated else ""))
                except Exception as e:
                    msg = f"{step}: 第 {attempt} 次调用失败 {type(e).__name__}: {e}"
                self.errors.append(msg)
                if attempt < self.max_retries:
                    self.n_retries += 1
                    time.sleep(1.2 * attempt)
                else:
                    print(f"[LLM] {msg}")

        if parsed is None:
            if self.backend == "api" and not self.allow_fallback:
                raise RuntimeError(f"步骤 {step} 未获得通过校验的 API 响应，实验已停止")
            if offline_handler is None:
                raise RuntimeError(f"步骤 {step} 无法获得结构化结果，且未提供离线处理器")
            parsed = offline_handler()
            if validate is not None and not validate(parsed):
                raise ValueError(f"步骤 {step} 的规则引擎输出不符合模块数据契约")
            text = json.dumps(parsed, ensure_ascii=False, indent=2)
            self.n_offline_calls += 1
            source = "offline"

        run_turns = [t for t in self.transcript if t.run_id == self.run_id]
        turn = LLMTurn(step=step, role=role, backend=source, model=self.model,
                       system=system, user=user, response=text, parsed=parsed,
                       latency_s=round(time.time() - t0, 3), round_id=round_id,
                       reasoning=reasoning, strategy=self.strategy, seed=self.seed,
                       timestamp=datetime.now(timezone.utc).isoformat(),
                       run_id=self.run_id, phase=self.phase, call_index=len(run_turns) + 1,
                       step_call_index=1 + sum(t.step == step and t.round_id == round_id for t in run_turns))
        self.transcript.append(turn)
        return parsed

    def _call_api(self, system: str, user: str) -> str:
        import urllib.request
        body_dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.json_mode:
            body_dict["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(body_dict).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        usage = body.get("usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens", 0))
        self.completion_tokens += int(usage.get("completion_tokens", 0))
        self.reasoning_tokens += int((usage.get("completion_tokens_details") or {})
                                     .get("reasoning_tokens", 0))
        choice = body["choices"][0]
        self.last_finish_reason = str(choice.get("finish_reason") or "")
        if self.last_finish_reason == "length":
            self.n_truncated += 1
        msg = choice.get("message") or {}
        self.last_reasoning = str(msg.get("reasoning_content") or "")
        return msg.get("content") or ""

    # 统计
    def usage_report(self) -> dict:
        total = self.n_api_calls + self.n_offline_calls
        return {
            "backend": self.backend,
            "model": self.model if self.backend == "api" else "offline-engine",
            "total_calls": total,
            "llm_calls": self.n_api_calls,
            "offline_fallbacks": self.n_offline_calls,
            "llm_ratio": self.n_api_calls / total if total else 0.0,
            "repaired": self.n_repaired,
            "truncated": self.n_truncated,
            "retries": self.n_retries,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }

    def print_usage(self) -> None:
        u = self.usage_report()
        print(f"[LLM 用量] 后端={u['backend']}  模型={u['model']}")
        print(f"[LLM 用量] 总调用 {u['total_calls']} 次，其中真实大模型返回并成功解析 "
              f"{u['llm_calls']} 次（{u['llm_ratio']:.1%}），"
              f"回退到离线引擎 {u['offline_fallbacks']} 次，重试 {u['retries']} 次")
        print(f"[LLM 用量] 输出被截断 {u['truncated']} 次，其中 {u['repaired']} 次从合法前缀抢救成功")
        print(f"[LLM 用量] token 消耗：prompt {u['prompt_tokens']:,}，"
              f"completion {u['completion_tokens']:,}"
              f"（其中思维链 {u['reasoning_tokens']:,}），"
              f"合计 {u['prompt_tokens'] + u['completion_tokens']:,}")
        if self.errors:
            print(f"[LLM 用量] 共记录 {len(self.errors)} 条异常，最近 3 条：")
            for e in self.errors[-3:]:
                print(f"           {e}")

    # 记录
    def last(self, step: str | None = None) -> LLMTurn | None:
        for t in reversed(self.transcript):
            if step is None or t.step == step:
                return t
        return None

    def print_turn(self, turn: LLMTurn, show_prompt: bool = True,
                   max_prompt_chars: int = 1400) -> None:
        print("=" * 100)
        print(f"[Agent 模块] {turn.role}   [步骤] {turn.step}   [轮次] R{turn.round_id}   "
              f"[后端] {turn.backend}   [耗时] {turn.latency_s}s")
        print(f"[运行] {turn.run_id}   [阶段] {turn.phase}   [调用] {turn.call_index}   "
              f"[本轮模块调用] {turn.step_call_index}")
        print("=" * 100)
        if show_prompt:
            print("--- SYSTEM PROMPT " + "-" * 82)
            print(turn.system.strip())
            print("--- USER PROMPT " + "-" * 84)
            u = turn.user.strip()
            print(u if len(u) <= max_prompt_chars else u[:max_prompt_chars] + f"\n... (共 {len(u)} 字符，已截断)")
        if turn.reasoning:
            print("--- 模型思维链 (reasoning_content) " + "-" * 64)
            r = turn.reasoning.strip()
            print(r if len(r) <= 1800 else r[:1800] + f"\n... (共 {len(r)} 字符，已截断)")
        print("--- 模型输出 " + "-" * 87)
        print(turn.response.strip()[:4000])
        print()

    def dump_transcript(self, path=None) -> str:
        path = path or (RESULT_DIR / "agent_transcript.json")
        payload = [t.to_record() for t in self.transcript]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[保存] Agent 交互记录({len(payload)} 次) -> {rel(path)}")
        return str(path)
