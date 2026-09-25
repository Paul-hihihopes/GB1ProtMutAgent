"""面向蛋白质定向进化的科学智能体 —— 核心代码包。

模块一览
    config     全局配置、路径、浅色系绘图风格
    data       数据加载/清洗/划分 + 虚拟湿实验 Oracle
    knowledge  氨基酸理化性质、BLOSUM62、突变规则库、知识图谱
    features   4 位点组合的特征编码
    models     适应度预测模型与评估指标
    llm        LLM 接入层（OpenAI 兼容 API / 内置离线推理引擎）
    agent      五模块科学智能体
    baselines  随机突变 / 模型贪心 两个对照基线
    evolution  多轮虚拟定向进化闭环
"""
__version__ = "1.0.0"
