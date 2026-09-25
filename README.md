# 面向蛋白质定向进化的科学智能体

本项目在 GB1 四位点突变文库上实现小规模虚拟定向进化：历史实验统计 → 假设生成 → 候选设计 → 模型评分 → 科学审查 → 虚拟实验 → 更新模型。实验比较随机搜索、模型直接推荐、LLM Agent、知识增强 LLM Agent，另提供离线规则方法、配对消融和可交互的 Console Demo。

## 运行环境与命令

在项目根目录使用已安装依赖的 Python 运行。完整 Notebook 实验需要下方的requirements环境。模型推理和离线 Agent 无需 API 凭据，真实 LLM 调用需要网络及 `.env` 中的有效凭据。

实验环境为 Windows、Python 3.12.7。建议在独立环境中执行：

```bash
python -m pip install -r requirements.txt
python -m ipykernel install --user --name gb1-evolution --display-name "GB1 Evolution"
python -m unittest discover -s tests -v
```

**Offline 完整运行**：

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


## 代码与资源说明

根目录 `demo.py` 是 Console 启动入口，`src/console_demo.py` 提供输入解析、实验环境、结构化单轮结果及终端交互。`src/data.py` 负责数据与 Oracle，`models.py` 负责训练评估，`agent.py` 实现五模块流程，`schemas.py` 定义模块输出契约，`knowledge.py` 实现知识与规则，`evolution.py` 管理多轮实验，`llm.py` 接入兼容接口。`scripts/run_notebook.py` 执行整套流程，`tests/` 检查数据边界、科学定义、模块契约和实验预算。最终五页报告位于 `output/pdf/面向蛋白质定向进化的科学智能体.pdf`。

使用的外部资源包括 FLIP／Wu et al. 数据、BLOSUM62（Henikoff & Henikoff 1992）、氨基酸性质参数（Kyte–Doolittle、Zamyatnin、Chou–Fasman、Vihinen）、scikit-learn、XGBoost、NumPy、pandas、SciPy、matplotlib 和 NetworkX。未训练大型蛋白语言模型；Agent 使用自建 prompt + Python 函数流程。代码与文档编写使用了 ChatGPT／Codex AI 编程助手辅助，实验推荐的 API 模型和实际调用记录以运行清单及 transcript 为准。
