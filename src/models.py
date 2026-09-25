"""适应度预测模型与评估指标。

输入  : 4 位点组合字符串
输出  : log10(fitness + eps) 的预测值(单调变换，不影响排序指标)
评估  : Spearman / Pearson / MSE / MAE / R² / Top-k 召回率 / Top-k 精确率 / NDCG
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import MODEL_DIR, RANDOM_SEED, rel
from .data import from_log_fitness
from .features import encode, encode_dataframe, feature_names

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:                                    # pragma: no cover
    HAS_XGB = False


# 指标
def topk_metrics(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> dict:
    """Top-k 相关指标。

    recall@k    : 真实 Top-k 中有多少被预测 Top-k 捕获
    precision@k : 预测 Top-k 里有多少确实属于真实 Top-k（与 recall@k 在 k 相同时数值相同）
    max_in_topk : 预测 Top-k 中的最大真实适应度（定向进化最关心的指标）
    mean_in_topk: 预测 Top-k 的平均真实适应度
    ndcg@k      : 排序质量
    """
    k = min(k, len(y_true))
    order_pred = np.argsort(-y_pred)[:k]
    true_topk = set(np.argsort(-y_true)[:k].tolist())
    hit = len(true_topk.intersection(order_pred.tolist()))

    # NDCG（用真实值的 min-max 归一化作为增益）
    rel = y_true - y_true.min()
    rel = rel / (rel.max() + 1e-12)
    dcg = np.sum(rel[order_pred] / np.log2(np.arange(2, k + 2)))
    ideal = np.sort(rel)[::-1][:k]
    idcg = np.sum(ideal / np.log2(np.arange(2, k + 2)))

    return {
        f"recall@{k}": hit / k,
        f"precision@{k}": hit / k,
        f"max_true_in_top{k}": float(y_true[order_pred].max()),
        f"mean_true_in_top{k}": float(y_true[order_pred].mean()),
        f"ndcg@{k}": float(dcg / (idcg + 1e-12)),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       ks: tuple[int, ...] = (10, 50, 100)) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = {
        "n": int(len(y_true)),
        "spearman": float(spearmanr(y_true, y_pred).correlation),
        "pearson": float(pearsonr(y_true, y_pred)[0]),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    for k in ks:
        out.update(topk_metrics(y_true, y_pred, k))
    return out


def print_metrics(name: str, m: dict, ks: tuple[int, ...] = (10, 50, 100)) -> None:
    print(f"  {name:<16} n={m['n']:>7,} | Spearman={m['spearman']:.4f} "
          f"Pearson={m['pearson']:.4f} MSE={m['mse']:.4f} MAE={m['mae']:.4f} R²={m['r2']:.4f}")
    frag = "  ".join(f"recall@{k}={m[f'recall@{k}']:.3f}" for k in ks if f"recall@{k}" in m)
    print(f"  {'':<16} {frag}")


# 模型库
def build_model(kind: str, seed: int = RANDOM_SEED):
    kind = kind.lower()
    if kind == "ridge":
        return make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0, random_state=seed))
    if kind == "randomforest":
        return RandomForestRegressor(n_estimators=400, max_depth=None, min_samples_leaf=2,
                                     n_jobs=-1, random_state=seed)
    if kind == "extratrees":
        return ExtraTreesRegressor(n_estimators=500, min_samples_leaf=1,
                                   n_jobs=-1, random_state=seed)
    if kind == "mlp":
        return make_pipeline(StandardScaler(with_mean=False),
                             MLPRegressor(hidden_layer_sizes=(256, 128), activation="relu",
                                          alpha=1e-3, learning_rate_init=1e-3, max_iter=800,
                                          early_stopping=True, n_iter_no_change=30,
                                          random_state=seed))
    if kind == "xgboost":
        if not HAS_XGB:
            raise ImportError("未安装 xgboost，请 pip install xgboost 或改用 'randomforest'")
        return XGBRegressor(n_estimators=700, max_depth=6, learning_rate=0.05,
                            subsample=0.85, colsample_bytree=0.85,
                            reg_lambda=1.0, min_child_weight=2,
                            n_jobs=-1, random_state=seed, tree_method="hist")
    raise ValueError(f"未知模型类型: {kind}")


MODEL_ZOO = ["ridge", "randomforest", "extratrees", "mlp"] + (["xgboost"] if HAS_XGB else [])

MODEL_DISPLAY = {
    "ridge": "Ridge 线性回归",
    "randomforest": "随机森林",
    "extratrees": "极端随机树",
    "mlp": "多层感知机 MLP",
    "xgboost": "XGBoost 梯度提升树",
}


# 封装
@dataclass
class FitnessModel:
    """适应度预测器：封装编码 + 回归 + 不确定度估计。"""
    kind: str = "xgboost" if HAS_XGB else "randomforest"
    use_physchem: bool = True
    use_pairwise: bool = True
    seed: int = RANDOM_SEED
    model: object = field(default=None, repr=False)
    n_train: int = 0
    fitted: bool = False
    n_unc_trees: int = 60          # 用于估计认知不确定度的辅助树数量
    unc_model: object = field(default=None, repr=False)

    # 训练 / 预测
    def fit(self, df: pd.DataFrame, target: str = "log_fitness", verbose: bool = False):
        X = encode_dataframe(df, use_physchem=self.use_physchem, use_pairwise=self.use_pairwise)
        y = df[target].values
        self.model = build_model(self.kind, self.seed)
        self.model.fit(X, y)
        # 辅助的随机化树集成：用子模型之间的预测分歧作为认知不确定度。
        # XGBoost / Ridge / MLP 本身不提供方差估计，统一用这个代理。
        self.unc_model = ExtraTreesRegressor(n_estimators=self.n_unc_trees, min_samples_leaf=1,
                                             max_features=0.6, bootstrap=True,
                                             n_jobs=-1, random_state=self.seed)
        self.unc_model.fit(X, y)
        self.n_train = len(df)
        self.fitted = True
        if verbose:
            print(f"[模型] {MODEL_DISPLAY.get(self.kind, self.kind)} 训练完成，"
                  f"样本={self.n_train:,}，特征={X.shape[1]}；"
                  f"另训练 {self.n_unc_trees} 棵随机化树用于不确定度估计")
        return self

    def predict(self, variants) -> np.ndarray:
        """返回 log 空间的预测值。"""
        if not self.fitted:
            raise RuntimeError("模型尚未训练")
        X = encode(variants, use_physchem=self.use_physchem, use_pairwise=self.use_pairwise)
        return np.asarray(self.model.predict(X), dtype=float)

    def predict_fitness(self, variants) -> np.ndarray:
        """返回还原到原始 fitness 尺度的预测值(野生型 = 1.0)。"""
        return from_log_fitness(self.predict(variants))

    def predict_with_uncertainty(self, variants) -> tuple[np.ndarray, np.ndarray]:
        """返回 (预测均值, 认知不确定度)。

        不确定度 = 辅助随机化树集成中各棵树预测值的标准差。
        该值作为 UCB 探索的启发式代理，未经过置信区间校准，
        不能保证在所有未知组合上都随实际预测误差增大。
        """
        mu = self.predict(variants)
        X = encode(variants, use_physchem=self.use_physchem, use_pairwise=self.use_pairwise)
        est = getattr(self.unc_model, "estimators_", None)
        if not est:
            est = getattr(self.model, "estimators_", None)
        if est:
            preds = np.stack([e.predict(X) for e in est])
            return mu, preds.std(axis=0)
        return mu, np.zeros_like(mu)

    # 评估
    def evaluate(self, df: pd.DataFrame, target: str = "log_fitness",
                 ks: tuple[int, ...] = (10, 50, 100)) -> dict:
        pred = self.predict(df["variant"].tolist())
        return regression_metrics(df[target].values, pred, ks=ks)

    # 可解释性
    def feature_importance(self, top: int = 20) -> pd.DataFrame:
        names = feature_names(self.use_physchem, self.use_pairwise)
        m = self.model
        if hasattr(m, "feature_importances_"):
            imp = m.feature_importances_
        elif hasattr(m, "steps") and hasattr(m.steps[-1][1], "coef_"):
            imp = np.abs(m.steps[-1][1].coef_)
        else:
            return pd.DataFrame(columns=["feature", "importance"])
        df = pd.DataFrame({"feature": names[:len(imp)], "importance": imp})
        return df.sort_values("importance", ascending=False).head(top).reset_index(drop=True)

    def position_importance(self) -> pd.DataFrame:
        """汇总单点特征重要性，并将位点对特征的重要性均分给两个位点。"""
        from .config import MUT_POSITIONS
        names = feature_names(self.use_physchem, self.use_pairwise)
        m = self.model
        if hasattr(m, "feature_importances_"):
            imp = m.feature_importances_
        elif hasattr(m, "steps") and hasattr(m.steps[-1][1], "coef_"):
            imp = np.abs(m.steps[-1][1].coef_)
        else:
            return pd.DataFrame(columns=["position", "importance"])
        agg = {p: 0.0 for p in MUT_POSITIONS}
        for n, v in zip(names[:len(imp)], imp):
            if n.startswith("pair_"):
                positions = [int(p) for p in n.split("_")[1].split("x")]
            else:
                positions = [p for p in MUT_POSITIONS if f"P{p}_" in n]
            for p in positions:
                agg[p] += float(v) / len(positions)
        tot = sum(agg.values()) + 1e-12
        return pd.DataFrame({"position": list(agg), "importance": list(agg.values()),
                             "share": [v / tot for v in agg.values()]})

    # 持久化
    def save(self, name: str = "fitness_model") -> str:
        path = MODEL_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[保存] 模型 -> {rel(path)}")
        return str(path)

    @staticmethod
    def load(name: str = "fitness_model") -> "FitnessModel":
        with open(MODEL_DIR / f"{name}.pkl", "rb") as f:
            return pickle.load(f)


# 模型选择
def compare_models(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                   kinds: list[str] | None = None, verbose: bool = True) -> pd.DataFrame:
    """在相同特征下比较多个 baseline，按验证集 Spearman 选优。"""
    kinds = kinds or MODEL_ZOO
    rows = []
    for kind in kinds:
        fm = FitnessModel(kind=kind).fit(train)
        mv = fm.evaluate(val, ks=(10, 50))
        mt = fm.evaluate(test, ks=(10, 50, 100))
        rows.append({
            "model": kind,
            "model_cn": MODEL_DISPLAY.get(kind, kind),
            "val_spearman": mv["spearman"], "val_pearson": mv["pearson"],
            "val_mse": mv["mse"], "val_r2": mv["r2"],
            "test_spearman": mt["spearman"], "test_pearson": mt["pearson"],
            "test_mse": mt["mse"], "test_mae": mt["mae"], "test_r2": mt["r2"],
            "test_recall@10": mt["recall@10"], "test_recall@50": mt["recall@50"],
            "test_recall@100": mt["recall@100"],
            "test_max_true_in_top100": mt["max_true_in_top100"],
            "test_ndcg@100": mt["ndcg@100"],
        })
        if verbose:
            print(f"  [{MODEL_DISPLAY.get(kind, kind):<16}] "
                  f"验证 Spearman={mv['spearman']:.4f} | 测试 Spearman={mt['spearman']:.4f} "
                  f"Pearson={mt['pearson']:.4f} MSE={mt['mse']:.4f} "
                  f"recall@100={mt['recall@100']:.3f}")
    return pd.DataFrame(rows).sort_values("val_spearman", ascending=False).reset_index(drop=True)
