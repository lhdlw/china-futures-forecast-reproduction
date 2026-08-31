# -*- coding: utf-8 -*-
"""
复现论文2 —— 第一步：特征构建（忠实复现）
Liang & Jia (2022) "China futures price forecasting based on online search and information transfer"

严格按论文 Section 2.2 + 3.1 的预处理顺序实现 ICPI：

  1. 关键词筛选  —— 只用 Table 4 最终选定的 9 个关键词（4 个谷歌 + 5 个百度）
  2. 组合项构造  —— "涨价,降价" 用 Antweiler & Frank (2004) 式(2): pos=(1+涨价)/(1+降价)
  3. 对数变换    —— 论文"logarithmically processed"降噪，用 log1p（兼容搜索量为0）
  4. 时滞平移    —— 按 Table 4 的 Adjustment days 逐词平移（负数=提前）
  5. 标准化      —— 论文未明说，但 Table 6 特征值/方差贡献率反推出做了 z-score
                    （总方差=9，即每个词方差为1）
  6. PCA         —— 取"特征值>1"的主成分（论文得到4个，累计方差 79.01%）
  7. 式(4)加权   —— ICPI = Σ w_k·PC_k，w_k = 方差贡献率 / 前N个主成分累计贡献率
  8. 式(1)归一化 —— 全数据 min-max 到 [0,1]

输出：
  data/features/icpi.csv               ICPI 时间序列（全品种共用，与期货无关）
  data/features/<symbol>_features.csv  各品种 [open, ICPI] 归一化特征
"""
import os
import glob
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

START, END = "2016-04-01", "2021-04-30"
FUTURES_DIR = "data/futures"
BDI_FILE = "data/baidu_index/bdi_all_keywords.csv"
GT_DIR = "data/google_trends"
OUT_DIR = "data/features"
os.makedirs(OUT_DIR, exist_ok=True)

EIGEN_THRESHOLD = 1.0   # 主成分特征值阈值（论文取>1）

# ----------------------------------------------------------------------
# Table 4：最终 9 个关键词。
# 每个元素 = (内部名, 来源, 源列名/文件, 调整天数)
# 调整天数：论文 Table 4 "Adjustment days"，负数=提前(序列上移)，正数=滞后。
# ----------------------------------------------------------------------
#   来源  英文                拼音            中文/文件键        调整天数
KEYWORDS = [
    ("jia",             "GT", "decrease in price",  -3),   # Decrease in price (jiangjia)
    ("jiagexiajiang",   "GT", "price decrease",      -2),   # Price decrease
    ("jiageshangzhang", "GT", "price rise",           0),   # Price rise
    ("jiage",           "GT", "price",               -8),   # Price
    ("jia_bdi",         "BDI", "价",                  0),   # Price 1
    ("shangzhang",      "BDI", "上涨",                0),   # Rise
    ("xiadie",          "BDI", "下跌",               -7),   # Drop
    ("zhangjia",        "BDI", "涨价",               -4),   # Rise in price
    ("zhangjia_jiangjia", "BDI", "__combo__",        -4),   # Rise in price, decrease in price
]


def load_raw_keywords():
    """加载原始搜索数据，返回 {内部名: Series}（含构造的组合项）。

    组合项 "zhangjia_jiangjia" 用式(2)：(1+涨价)/(1+降价)。
    """
    bdi = pd.read_csv(BDI_FILE, parse_dates=["date"]).set_index("date")
    bdi = bdi[(bdi.index >= START) & (bdi.index <= END)]

    gt = {}
    for f in sorted(glob.glob(os.path.join(GT_DIR, "gt_*.csv"))):
        df = pd.read_csv(f, parse_dates=["date"]).set_index("date")
        col = [c for c in df.columns if c != "date"][0]
        gt[col] = df[col].rename(col)

    raw = {}
    for name, src, key, adj in KEYWORDS:
        if key == "__combo__":
            # Antweiler & Frank (2004) 式(2): pos = (1+posterm)/(1+negterm)
            posterm = bdi["涨价"]     # Rise in price (正向)
            negterm = bdi["降价"]     # Decrease in price (负向)
            s = (1.0 + posterm) / (1.0 + negterm)
        elif src == "BDI":
            s = bdi[key]
        else:  # GT
            s = gt[key]
        raw[name] = s
    return pd.DataFrame(raw)


def build_icpi():
    """对数 -> 时滞平移 -> 标准化 -> PCA(特征值>1) -> 式(4)加权 -> ICPI。"""
    df = load_raw_keywords()

    # 3. 对数变换（降噪，兼容0值）
    x = np.log1p(df)

    # 4. 时滞平移（Table 4 adjustment days；负数=提前）
    for name, src, key, adj in KEYWORDS:
        if adj != 0:
            x[name] = x[name].shift(adj)
    x = x.dropna()          # 平移产生的尾部 NaN 丢弃

    # 5. 标准化（z-score；论文 Table 6 反推隐含此步）
    z = (x - x.mean()) / x.std()

    # 6. PCA，取特征值>1的主成分
    pca_full = PCA().fit(z)
    n = int((pca_full.explained_variance_ > EIGEN_THRESHOLD).sum())
    n = max(1, n)
    pca = PCA(n_components=n)
    comps = pca.fit_transform(z)

    # 7. 式(4)加权：w_k = 方差贡献率 / 累计贡献率（归一化）
    w_raw = pca.explained_variance_ratio_[:n]
    w = w_raw / w_raw.sum()
    icpi = pd.Series((comps * w).sum(axis=1), index=z.index, name="ICPI")

    info = {
        "n_components": n,
        "eigenvalues": np.round(pca_full.explained_variance_[:n], 3),
        "variance_ratio": np.round(w_raw, 4),
        "cum_variance": float(w_raw.sum()),
        "weights_eq4": np.round(w, 4),
    }
    return icpi, info


def build_dataset(symbol, icpi):
    """按品种对齐期货开盘价与 ICPI，式(1) min-max 归一化到 [0,1]。"""
    fp = glob.glob(os.path.join(FUTURES_DIR, symbol + "_*.csv"))[0]
    fut = pd.read_csv(fp, parse_dates=["date"]).set_index("date").sort_index()
    fut = fut[(fut.index >= START) & (fut.index <= END)]

    df = pd.DataFrame({"open": fut["open"]})
    df["ICPI"] = icpi.reindex(df.index)
    df = df.dropna()

    for col in ["open", "ICPI"]:                       # 式(1)
        mn, mx = df[col].min(), df[col].max()
        df[col + "_norm"] = (df[col] - mn) / (mx - mn)

    df = df.dropna()
    out = df[["open", "ICPI", "open_norm", "ICPI_norm"]]
    out.to_csv(os.path.join(OUT_DIR, symbol + "_features.csv"), encoding="utf-8-sig")
    return df


if __name__ == "__main__":
    icpi, info = build_icpi()
    icpi.to_csv(os.path.join(OUT_DIR, "icpi.csv"), encoding="utf-8-sig")

    print("=" * 64)
    print("ICPI 构建结果")
    print("=" * 64)
    print(f"日期范围        : {icpi.index.min().date()} ~ {icpi.index.max().date()}  共 {len(icpi)} 天")
    print(f"主成分数(特征值>1): {info['n_components']}")
    print(f"各主成分特征值   : {info['eigenvalues']}")
    print(f"方差贡献率       : {info['variance_ratio']}")
    print(f"累计方差         : {info['cum_variance']:.2%}  (论文 79.01%)")
    print(f"式(4)权重        : {info['weights_eq4']}  (论文 [0.434 0.228 0.194 0.144])")
    print()

    for sym in ["C0", "A0", "V0", "JD0", "RB0"]:
        try:
            df = build_dataset(sym, icpi)
            print(f"[{sym}] 特征 {df.shape}  区间 {df.index.min().date()}~{df.index.max().date()}")
        except Exception as e:
            print(f"[{sym}] 失败: {type(e).__name__}: {e}")
