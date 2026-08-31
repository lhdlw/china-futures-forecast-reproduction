# -*- coding: utf-8 -*-
"""
消融实验：定位 ICPI / TE 两个特征各自对预测的影响。

对指定品种，分别跑 4 组输入特征：
    仅价格          -> open
    价格 + ICPI     -> open + ICPI
    价格 + TE       -> open + TE
    价格 + ICPI+TE  -> open + ICPI + TE

复用 reproduce_model 的 CNNLSTM / GWO / fit_eval，预算与 reproduce_model 一致
（FINAL_EPOCHS / REPEATS / GWO_* / SEARCH_EPOCHS 均从 reproduce_model 读）。
结果写入 data/features/ablation_<symbol>_ep<FINAL_EPOCHS>_rep<REPEATS>.txt

用法：
    python ablation.py            # 默认只跑玉米 C0
    python ablation.py C0 A0      # 指定品种
"""
import os
import sys
import numpy as np
import pandas as pd
import torch

import reproduce_model as M      # 复用 CNNLSTM/gwo/fit_eval/decode/BOUNDS 等

torch.manual_seed(0)
np.random.seed(0)

ABLATIONS = [
    ("仅价格",       ["open"]),
    ("价格+ICPI",    ["open", "ICPI"]),
    ("价格+TE",      ["open", "TE"]),
    ("价格+ICPI+TE", ["open", "ICPI", "TE"]),
]


def load_dataset(symbol, features):
    """按特征子集构造数据集（features 为特征名列表，如 ['open','ICPI']）。"""
    feat = pd.read_csv(os.path.join(M.FEAT_DIR, f"{symbol}_features.csv"),
                       parse_dates=["date"]).set_index("date")
    te = pd.read_csv(os.path.join(M.FEAT_DIR, f"{symbol}_te.csv"),
                     parse_dates=["date"]).set_index("date")

    df = feat[["open", "open_norm"]].copy()
    te_norm = (te["TE_net"] - te["TE_net"].min()) / (te["TE_net"].max() - te["TE_net"].min())

    cols = ["open_norm"]
    if "ICPI" in features:
        df["ICPI_norm"] = feat["ICPI_norm"]
        cols.append("ICPI_norm")
    if "TE" in features:
        df["TE_norm"] = te_norm          # pandas 按日期对齐，TE 缺失段自动 NaN
        cols.append("TE_norm")

    df = df.dropna()
    return df[cols].values, df["open_norm"].values, df["open"].values, cols


def run_ablation(symbol, features):
    X, y_norm, open_raw, cols = load_dataset(symbol, features)

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
    symbols = sys.argv[1:] if len(sys.argv) > 1 else ["C0"]
    RESULT_FILE = os.path.join(M.FEAT_DIR, f"ablation_{'_'.join(symbols)}_ep{M.FINAL_EPOCHS}_rep{M.REPEATS}.txt")

    header = f"{'品种':6s} {'输入特征':14s} {'MAE':>9s} {'RMSE':>9s} {'MAPE':>8s} {'R2':>7s}   最优超参"
    sep = "-" * 108

    def report(line):
        print(line, flush=True)
        with open(RESULT_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write(header + "\n" + sep + "\n")

    report(header)
    report(sep)
    for sym in symbols:
        for label, feats in ABLATIONS:
            report(f"[{sym}] 消融：{label} ...")
            bp, res, cols = run_ablation(sym, feats)
            report(f"{sym:6s} {label:14s} {res['MAE']:9.4f} {res['RMSE']:9.4f} "
                   f"{res['MAPE']:8.4f} {res['R2']:7.4f}   {bp}")
        report(sep)
    report(f"完成。结果已保存到 {RESULT_FILE}")
