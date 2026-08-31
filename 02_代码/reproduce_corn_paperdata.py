# -*- coding: utf-8 -*-
"""
按论文玉米数据重跑：把价格+搜索数据统一截断到 2020-09-30（匹配论文 Table 1 统计量），
完整重跑 ICPI -> TE -> GWO-CNN-LSTM，对比论文 Table 11 玉米结果（MAE=15.25 / R²=0.831）。

复用 reproduce_features 的 KEYWORDS、reproduce_te 的 KSG、reproduce_model 的 GWO-CNN-LSTM。
"""
import os
import glob
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

import reproduce_features as F
import reproduce_te as TE
import reproduce_model as M

torch.manual_seed(0)
np.random.seed(0)

START = "2016-04-01"
END = "2020-09-30"          # 论文玉米 Table 1 统计量匹配的截止日
SYMBOL = "C0"
PAPER = {"仅价格": (21.0855, 0.7471), "价格+ICPI+TE": (15.2499, 0.8312)}  # (MAE, R²)


def build_icpi_trunc():
    """对数->时滞平移->z-score->PCA(特征值>1)->式(4)加权，全部截断到 END。"""
    bdi = pd.read_csv(F.BDI_FILE, parse_dates=["date"]).set_index("date")
    bdi = bdi[(bdi.index >= START) & (bdi.index <= END)]

    gt = {}
    for f in sorted(glob.glob(os.path.join(F.GT_DIR, "gt_*.csv"))):
        df = pd.read_csv(f, parse_dates=["date"]).set_index("date")
        col = [c for c in df.columns if c != "date"][0]
        gt[col] = df[col].rename(col)

    raw = {}
    for name, src, key, adj in F.KEYWORDS:
        if key == "__combo__":
            s = (1.0 + bdi["涨价"]) / (1.0 + bdi["降价"])
        elif src == "BDI":
            s = bdi[key]
        else:
            s = gt[key]
        raw[name] = s

    df = pd.DataFrame(raw)
    df = df[(df.index >= START) & (df.index <= END)]   # 双平台都截断
    x = np.log1p(df)
    for name, src, key, adj in F.KEYWORDS:
        if adj != 0:
            x[name] = x[name].shift(adj)
    x = x.dropna()
    z = (x - x.mean()) / x.std()

    pca_full = PCA().fit(z)
    n = int((pca_full.explained_variance_ > F.EIGEN_THRESHOLD).sum())
    n = max(1, n)
    pca = PCA(n_components=n)
    comps = pca.fit_transform(z)
    w_raw = pca.explained_variance_ratio_[:n]
    w = w_raw / w_raw.sum()
    icpi = pd.Series((comps * w).sum(axis=1), index=z.index, name="ICPI")
    return icpi, {"n": n, "cum_var": float(w_raw.sum())}


def build_dataset(icpi, features):
    """构造监督数据集，features 为特征名列表。"""
    fp = glob.glob(os.path.join(F.FUTURES_DIR, SYMBOL + "_*.csv"))[0]
    fut = pd.read_csv(fp, parse_dates=["date"]).set_index("date").sort_index()
    fut = fut[(fut.index >= START) & (fut.index <= END)]

    df = pd.DataFrame({"open": fut["open"]})
    df["ICPI"] = icpi.reindex(df.index)
    df = df.dropna()

    for col in ["open", "ICPI"]:
        mn, mx = df[col].min(), df[col].max()
        df[col + "_norm"] = (df[col] - mn) / (mx - mn)

    # TE 净流（在归一化序列上滚动窗口 60/步长 1）
    ic = df["ICPI_norm"]
    op = df["open_norm"]
    te_df = TE.rolling_te(ic, op)
    te_norm = (te_df["TE_net"] - te_df["TE_net"].min()) / (te_df["TE_net"].max() - te_df["TE_net"].min())

    df = pd.DataFrame({"open": df["open"], "open_norm": df["open_norm"]})
    cols = ["open_norm"]
    if "ICPI" in features:
        df["ICPI_norm"] = ic
        cols.append("ICPI_norm")
    if "TE" in features:
        df["TE_norm"] = te_norm
        cols.append("TE_norm")
    df = df.dropna()

    return df[cols].values, df["open_norm"].values, df["open"].values, cols


def run(features):
    X, y_norm, open_raw, cols = build_dataset(icpi, features)

    def objective(w):
        p = M.decode(w)
        mae, _ = M.fit_eval(p, X, y_norm, open_raw, M.SEARCH_EPOCHS)
        return mae

    best = M.gwo(objective, len(M.PARAM_KEYS), M.GWO_POP, M.GWO_ITER)
    best_p = M.decode(best)

    metrics = []
    for r in range(M.REPEATS):
        _, m = M.fit_eval(best_p, X, y_norm, open_raw, M.FINAL_EPOCHS, seed=r)
        if m:
            metrics.append(m)
    avg = {k: float(np.mean([m[k] for m in metrics])) for k in ["MAE", "RMSE", "MAPE", "R2"]}
    return best_p, avg, cols


if __name__ == "__main__":
    icpi, info = build_icpi_trunc()
    print(f"[数据截断] {START} ~ {END}，ICPI 主成分数={info['n']}，累计方差={info['cum_var']:.2%}", flush=True)

    results = {}
    for label, feats in [("仅价格", ["open"]), ("价格+ICPI+TE", ["open", "ICPI", "TE"])]:
        print(f"[跑模型] {label} ...", flush=True)
        bp, res, cols = run(feats)
        results[label] = res
        pmae, pr2 = PAPER[label]
        print(f"  我们: MAE={res['MAE']:8.4f}  R2={res['R2']:6.4f}  |  论文: MAE={pmae:8.4f}  R2={pr2:6.4f}", flush=True)

    print("\n==== 对比 ====", flush=True)
    for label in ["仅价格", "价格+ICPI+TE"]:
        res = results[label]
        pmae, pr2 = PAPER[label]
        print(f"{label:14s}  我们 MAE={res['MAE']:7.2f} R2={res['R2']:.4f}  vs  论文 MAE={pmae:7.2f} R2={pr2:.4f}  "
              f"(MAE 差 {res['MAE']/pmae:.1f}x)", flush=True)
