# 面向蛋白质定向进化的科学智能体

本项目在 GB1 四位点突变文库上实现小规模虚拟定向进化：历史实验统计 → 假设生成 → 候选设计 → 模型评分 → 科学审查 → 虚拟实验 → 更新模型。实验比较随机搜索、模型直接推荐、LLM Agent、知识增强 LLM Agent，另提供离线规则方法、配对消融和可交互的 Console Demo。

## 运行环境与命令

在项目根目录使用已安装依赖的 Python 运行。Console Demo 可安装轻量依赖 `requirements-console.txt`；完整 Notebook 实验需要下方的完整环境。模型推理和离线 Agent 无需 API 凭据，真实 LLM 调用需要网络及 `.env` 中的有效凭据。

实验环境为 Windows、Python 3.12.7。`requirements-lock.txt` 记录生成随附结果时的直接依赖版本；`requirements.txt` 提供兼容范围。图表中文字体使用微软雅黑或黑体；其他系统需提供可用中文字体并在 `src/config.py` 中配置。建议在独立环境中执行：

```bash
python -m pip install -r requirements-lock.txt
python -m ipykernel install --user --name gb1-evolution --display-name "GB1 Evolution"
python -m unittest discover -s tests -v
```

**不需要 API 的完整运行**：

```bash
python scripts/run_notebook.py --kernel gb1-evolution --backend offline --output-dir runs/offline
```

**真实 LLM 实验**：将 `.env.example` 复制为 `.env`，设置 `LLM_API_KEY`、`LLM_BASE_URL` 和 `LLM_MODEL`，然后执行：

```bash
python scripts/run_notebook.py --kernel gb1-evolution --backend api
```

API 模式会向配置的服务发送实验摘要及候选设计请求，消耗对应服务的额度。正式实验采用严格模式：请求失败或结构化输出不合格时重试，重试耗尽后停止，不以离线结果代替 LLM 结果。`--backend auto` 根据是否配置凭据选择后端。每轮通常包含四个 LLM 模块调用，审稿否决后的补选可能增加调用次数。

默认输出到项目根目录下的 `results/`、`img/`、`models/`、`data/processed/` 和 `notebooks/`。`--output-dir` 可保存独立实验，避免覆盖随附结果。Jupyter 中也可直接运行 `notebooks/protein_directed_evolution.ipynb` 的 33 个代码单元。

## Console Demo

启动交互程序（已有 Python 环境先安装 `requirements.txt`）：

```bash
python demo.py --backend offline
```

菜单提供交互进化、fitness 预测和模型信息。输入起点后显示解析组合、突变记号、已测状态及实测或预测 fitness；选择 knowledge 与 batch size（4–24）运行一轮，查看五模块结果、候选预测与 Oracle 真实值、统计及 LLM 使用量。继续下一轮时可输入新起点，留空则选本批真实最优；这会将本批送检数据加入历史并重新训练模型。输入 `q` 返回菜单，输入 `0` 退出。也可运行 `python demo.py --backend api` 使用 `.env` 中的真实 LLM 凭据；`--backend auto` 根据配置自动选择。Console 的回退会标注实际后端，不计入正式实验。

`.env`、解压后的原始 CSV、临时实验 `runs/` 不进入 Git。原始 ZIP 可以直接读取，不要求手动解压。

## 数据来源与任务边界

- 原始实验：[Wu et al., 2016, *Adaptation in protein fitness landscapes is facilitated by indirect paths*](https://elifesciences.org/articles/16965)，eLife 5:e16965。
- 数据整理：[FLIP 的 GB1 数据说明](https://github.com/J-SNACKKB/FLIP/blob/main/splits/gb1/README.md)，使用 `four_mutations_full_data.csv.zip`，原始文件随项目保存在 `data/raw/`。
- 目标：提高蛋白 G 的 B1 结构域（GB1）的 IgG Fc 结合适应度；使用数据集提供的 fitness，野生型归一化为 1。
- 仅考虑 39、40、41、54 四个位点，野生型组合为 `VDGV`。清洗后 149,361 个变体，覆盖理论 `20^4` 空间的 93.35%。虚拟实验只在有实测记录的组合中搜索；缺失组合不能获得真实实验分数。

GB1 结构域为 **56 aa**。FLIP 原始 FASTA 是根据结构推定的 **265 aa 融合序列表示**，不能把它全部称为 GB1 结构域。本项目默认展示前 56 aa；保留原始表示用于数据追溯。完整序列示例：

```fasta
>WT_VDGV
MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE
>G41A_VDAV
MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDAEWTYDDATKTFTVTE
>G41A_V54A_VDAA
MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDAEWTYDDATKTFTATE
```

如需重新获取数据：

```bash
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/J-SNACKKB/FLIP/main/splits/gb1/four_mutations_full_data.csv.zip', 'data/raw/four_mutations_full_data.csv.zip')"
```

## 数据划分与信息隔离

| 集合 | 数量 | 用途 |
|---|---:|---|
| 训练集 | 1,735 | 低阶突变，用于拟合 baseline |
| 验证集 | 433 | 从低阶突变按阶数分层抽取，用于选择模型 |
| 初始已测集合 | 2,168 | 训练集与验证集合并，作为每次定向进化的初始历史 |
| 高阶测试集／候选池 | 147,193 | 三点和四点突变，用于外推评估与虚拟实验回读 |

baseline 的测试指标由仅在 1,735 条训练数据上拟合的模型给出。模型种类按验证集 Spearman 选择；随后各策略用全部 2,168 条初始已测记录训练起始模型。每轮选择 12 条此前未测的候选，回读真实 fitness 后加入历史并重新训练，连续运行 3 轮，每个策略合计 36 次新实验。

Agent 只能读取当轮历史记录、由历史计算的统计量和模型预测；候选池只提供序列成员资格。真实 fitness 在选定实验批次后由 Oracle 返回。全局最优信息只用于事后评估与展示，不注入设计提示词。每个策略使用独立实验状态。

该协议是固定文库上的主动搜索，不是整个过程中始终封闭的测试集评估：已被选中测量的高阶变体随后进入训练集。逐轮模型相关系数在固定抽样的 20,000 条高阶评估池中剔除已测候选后计算，不参与候选选择；不同轮次的评价集合因此略有变化。

## 适应度模型与指标

特征为 one-hot 80 维、氨基酸性质 28 维、位点对描述符 30 维，共 138 维。所有策略使用相同特征，知识增强的比较主要针对 Agent 提示、候选生成与规则审查。标签为 `log10(fitness + 0.01)`。

| 模型 | 验证 Spearman | 测试 Spearman | 测试 MSE（log） | recall@100 |
|---|---:|---:|---:|---:|
| XGBoost | 0.9025 | 0.4893 | 0.1552 | 0.19 |
| ExtraTrees | 0.8971 | 0.4625 | 0.1257 | 0.15 |
| RandomForest | 0.8911 | 0.4562 | 0.1633 | 0.03 |
| MLP | 0.8579 | 0.4586 | 0.2460 | 0.11 |
| Ridge | 0.8505 | 0.4720 | 0.6482 | 0.02 |

Spearman 衡量排序；Pearson、MSE、RMSE、MAE、R² 均在 log 标签尺度计算，完整结果见 `results/model_comparison.csv`。`recall@k` 是预测 Top-k 与真实 Top-k 的交集大小除以 k；此处两组大小相等，所以 precision@k 与 recall@k 相同。候选推荐和进化曲线使用原始 fitness 尺度。

低阶验证到高阶测试的表现下降，说明外推更困难；训练集最大值 5.39 不是所有回归模型的输出上限。辅助 ExtraTrees 的树间预测标准差作为 UCB 探索的启发式代理，单位是 log fitness，不是经校准的置信区间。

## 五模块 Agent

| 模块 | 输入与输出 |
|---|---|
| Data Analyst | 汇总历史实验、Top variants、单点效应；知识模式额外提供双突变相互作用 |
| Hypothesis Generator | 输出假设、目标位点、候选残基、依据及待验证的方向 |
| Mutation Designer | LLM 输出少量种子方案；确定性枚举、重组和探索扩展到最多 600 条候选 |
| Fitness Evaluator | 预测候选适应度与树间分歧，计算 UCB |
| Scientific Critic | 规则审查、多样性选择和逐条评议；否决候选后补选并再次审查 |

送检批次必须满足预算、序列唯一、此前未测及文库成员条件。Critic 的 `reject` 候选不进入实验；达到审查轮数上限仍无法凑齐批次时停止该轮。Top-5 在实验回读前确定，不按真实 fitness 事后挑选。

LLM 输出通过字段、类型、位点、氨基酸和候选覆盖校验。每条候选保留 `source_type`；只有满足假设目标残基条件时才保留对应 `hypothesis_id`，未直接关联的枚举候选不会挂在任意假设名下。来源统计只描述生成来源，不等于各模块的独立因果贡献。

`results/agent_transcript.json` 保存实际提示词、结构化返回、服务提供的推理文本、后端、策略、轮次、随机种子和时间。新运行还记录独立 `run_id`、运行阶段 `phase`、调用序号及本轮模块调用序号，campaign 导出使用相同运行编号。随附旧版记录开头为单轮演示，随后是两种 Agent 的主实验，尚无这些新标识；分析旧记录时须按原始调用顺序区分重复轮次并保留补选审查。推荐理由是模型给出的解释，需与可查的实验数据核对，不能直接视为已证实的机理。

## 知识增强与上位效应

知识库包含 20 种氨基酸的性质、BLOSUM62、结构位置教学注释、10 条设计规则和关系图。规则包括合法字母表、总突变数、单轮改动跨度、替换幅度、脯氨酸／半胱氨酸风险、电荷变化及历史单点证据。结构注释和惩罚权重是启发式先验，未在此项目中通过结构模拟验证。“低功能”指分析阈值 `fitness < 0.1`，不表示生物体死亡；全库零适应度比例单独按 `fitness == 0` 统计。

提示词直接使用规则库配置生成规则说明。单轮相对设计骨架通常改动不超过 2 个位点，这是软约束：超限候选会扣分并标注风险，仍可探索；序列合法性、总突变上限和禁用的 β 股脯氨酸属于硬约束。WT Demo 的低阶变体均已测过，因此新方案会涉及三点或四点替换。审稿模块接收候选全部突变的知识、历史单点与双突变实测值及通过规则的依据，组合证据只描述对应背景。

令 `f0`、`fa`、`fb`、`fab` 分别为 WT、单点 a、单点 b 和双突变适应度。只要 `(fa-f0)*(fab-fb)<0` 或 `(fb-f0)*(fab-fa)<0`，就存在**符号上位效应**。正向上位效应还单独以 `log10(fab+0.01) - log10(fa+0.01) - log10(fb+0.01) + log10(f0+0.01)` 描述。

例如 G41A 单点为 **0.120576**，V54A 为 **1.372949**；偏移 log 加性模型还原的组合期望约 **0.168792**，实测 VDAA 为 **4.157984**。G41A 在 WT 背景有害，在 V54A 背景有益，符合效应方向翻转。

只有历史双突变显示有害单点在伙伴背景中转为有益时，才记录有方向的补偿证据。该证据可降低相应规则惩罚，但不能保证高阶背景中的效果。知识图谱用于组织已测事实和提示文本，不将测试真值作为知识导入。

## 实验结果与可复核产物

主实验比较 `Random`、`Model-Greedy`、`LLM-Agent`、`LLM-Agent+KB`，每种策略一次 campaign。完整逐轮 Top-5、12 条实验批次、预测和真实 fitness、来源、理由及审查结果见 `results/campaigns.json`。

随附结果使用 DeepSeek `deepseek-flash`。单轮演示与两种 Agent 的主实验合计 **33 次真实 API 返回，0 次离线回退、0 次重试**。主实验中知识增强 Agent 共否决 9 条候选，均经补选后组成完整实验批次。

这些随附 API 结果是运行清单所记录版本的历史实验。后续修复补齐了审稿证据、统一软约束提示词，并修正 Demo 统计和日志展示；历史响应与指标保留原样。当前提示词的 API 表现需重新运行实验评估，可使用 `--output-dir` 保存新结果。

| 主实验策略 | 最终最优 | 最优变体 | 相对初始最优的增量 | 命中率 | 批次平均 fitness |
|---|---:|---|---:|---:|---:|
| LLM-Agent+KB | 8.762 | FWAA | +3.371 | 25.0% | 4.430 |
| LLM-Agent | 8.762 | FWAA | +3.371 | 19.4% | 4.231 |
| Model-Greedy | 5.772 | IWGF | +0.381 | 11.1% | 4.304 |
| Random | 5.391 | VWGF | +0.000 | 0.0% | 0.204 |

两种 LLM Agent 都在第 2 轮找到 FWAA。每种方法仅一次真实 API campaign，此表不能说明知识增强在其他运行中始终更优。两个 Agent 各有 8 条送检候选直接来自 LLM 种子设计，其余来自假设扩展、重组、枚举或探索。

- `results/campaign_summary.csv`：主实验最终最优、改善幅度、命中率、批次均值等。
- `results/campaign_rounds.csv`：主实验每轮累计最优、批次质量和模型表现。
- `results/replicate_summary.csv`：8 个种子的离线规则方法与基线汇总。
- `results/ablation_summary.csv`、`ablation_rounds.csv`：相同种子下启用／关闭补偿证据的完整三轮消融。
- `results/source_summary.csv`：主实验送检候选的来源分布。
- `results/run_manifest.json`：数据摘要、代码摘要、后端、参数、随机种子和软件版本。
- `img/`：13 张实验图；已执行 Notebook 包含输出和案例分析。

离线重复实验的随机种子为 11、22、33、44、55、66、77、88，名称为 `Rule-Agent` 与 `Rule-Agent+KB`。离线规则处理器不调用 LLM，其结果不能当作真实 LLM 的重复实验。

| 离线策略 | 最终最优均值 ± 标准差 | 找到文库全局最优 | 批次平均 fitness |
|---|---:|---:|---:|
| Rule-Agent+KB | 8.762 ± 0.000 | 8/8 | 4.266 |
| Rule-Agent | 8.672 ± 0.253 | 7/8 | 4.401 |
| Model-Greedy | 6.927 ± 1.522 | 3/8 | 4.366 |
| Random | 5.391 ± 0.000 | 0/8 | 0.094 |

配对消融中，启用与关闭补偿证据均为 **8/8** 找到全局最优；启用时批次均值 **4.266**，关闭时 **4.275**。首次达到目标的累计实验数按整批计数，两组均值分别为 **28.5** 与 **18.0**。因此这一机制在本次实验中没有带来搜索速度优势，不能预设知识越多效果越好。`assays_to_target` 只在成功重复中求平均，失败重复不赋予虚构实验次数，须结合成功率解读。

命中率定义为 36 条新测候选中超过初始最优 `VWGF=5.390733` 的比例；成功率定义为独立 campaign 在预算内找到文库最优 `FWAA=8.761966` 的比例。累计最优天然不下降，是否改进还应看逐轮批次均值、命中率与失败样例。

主实验中的具体案例（均来自 LLM-Agent+KB）：

| 案例 | 轮次 | 预测 fitness | 真实 fitness | 分析 |
|---|---:|---:|---:|---|
| FWAA | 2 | 5.144 | 8.762 | LLM 种子候选获谨慎接受，实验找到文库最优 |
| IYAA | 2 | 8.290 | 4.426 | 获接受但明显高估，预测高分不能保证组合收益 |
| FWAG | 3 | 1.823 | 6.509 | 随机探索补充的候选被低估，探索保留了潜在机会 |

推荐文字也存在事实偏差：LWAA 的审查理由将 V54A 称为“未知”，而该单点的 fitness 1.373 已在初始历史中。这说明输出格式校验不能代替生物学事实核查；保留原始返回供检查，不能把解释文本当作实验结论。预测失败可能涉及高阶相互作用与分布差异，当前证据不足以确认具体结构原因。

## 交互规则

Demo 支持 `WT`／留空、四位点组合（如 `VDGV`）、突变记号（如 `D40W + V54F`），以及 GB1 56 aa 序列或 FLIP 265 aa 参考背景下的完整序列。完整输入必须与参考序列在其他位点保持一致；拒绝非标准残基、终止符、非法长度及重复指定同一位点。

输入骨架与历史最优分开记录：已测骨架使用其真实 fitness，尚未测量的骨架只使用模型预测参照。生成方案后才回读所选候选的真实值。Console 首轮从初始历史出发；继续下一轮会提交上一批的测量、累计实验预算并重新训练模型。完整批量实验也可由 Notebook 执行。

只有起始序列已测量时，Demo 才统计“优于起始序列”。未测起点的结构化结果中 `n_above_start` 为 `None`，`n_above_reference` 表示超过预测参照的数量，`comparison_basis` 标注比较依据；这不能作为真实改善的证据。不确定度单独以 log 尺度展示。

## 结果解释与局限

1. 与模型贪心的比较同时改变候选生成、UCB、多样性和审查；不能把总体差异单独归因于 LLM。要估计 LLM 的独立贡献，需要在相同候选和选择规则下替换假设模块，并进行更多真实 API 配对重复。
2. 真实 LLM 主实验每种配置只有一次，不能据此估计稳定成功率。离线多种子结果只支持离线方法的比较。固定种子控制本地模型与搜索过程，不控制远程模型采样；API 重新执行可能生成不同方案，随附 transcript 用于核查本次结果。
3. GB1 是常见公开基准，LLM 的预训练知识可能影响结果；本项目没有排除基准记忆。
4. 双突变补偿不保证三点、四点组合仍然有益；规则阈值可能压制有价值的探索。Notebook 分析高估失败与低估候选，不只展示高分方案。
5. 该任务只覆盖 GB1 四个位点和固定文库，没有模拟测量噪声、合成可行性、表达成本或真实湿实验，也不能自动泛化到任意蛋白。
6. 结构化假设与推荐理由展示了可追踪的研究流程，但不能据此声称 Agent 已掌握科学家的独立推理能力。

可拓展方向包括更公平的模块消融、更多真实 LLM 重复、校准不确定度、显式实验噪声和迁移到第二个蛋白任务。

## 代码与资源说明

根目录 `demo.py` 是 Console 启动入口，`src/console_demo.py` 提供输入解析、实验环境、结构化单轮结果及终端交互。`src/data.py` 负责数据与 Oracle，`models.py` 负责训练评估，`agent.py` 实现五模块流程，`schemas.py` 定义模块输出契约，`knowledge.py` 实现知识与规则，`evolution.py` 管理多轮实验，`llm.py` 接入兼容接口。`scripts/run_notebook.py` 执行整套流程，`tests/` 检查数据边界、科学定义、模块契约和实验预算。最终五页报告位于 `output/pdf/面向蛋白质定向进化的科学智能体.pdf`。

使用的外部资源包括 FLIP／Wu et al. 数据、BLOSUM62（Henikoff & Henikoff 1992）、氨基酸性质参数（Kyte–Doolittle、Zamyatnin、Chou–Fasman、Vihinen）、scikit-learn、XGBoost、NumPy、pandas、SciPy、matplotlib 和 NetworkX。未训练大型蛋白语言模型；Agent 使用自建 prompt + Python 函数流程。代码与文档编写使用了 ChatGPT／Codex AI 编程助手辅助，实验推荐的 API 模型和实际调用记录以运行清单及 transcript 为准。
