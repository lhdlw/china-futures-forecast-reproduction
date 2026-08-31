# -*- coding: utf-8 -*-
"""
复现论文2 —— 第二步：传递熵 (Transfer Entropy, TE) 特征
Liang & Jia (2022) Section 3.2 + 4.2

论文要点：
  - TE 用 Kraskov (KSG) 估计器（连续变量的 k 近邻熵估计），见式(11)
  - 式(11) 用"一维延迟向量"(i=j=1)，即历史长度=1：TE(K→L) 用 l_{t-1}, k_{t-1} 预测 l_t
  - 因果检测：窗口 180 天 / 步长 5 天（Section 4.2.1）
  - 预测特征：窗口 60 天 / 步长 1 天（Section 4.2.2）
  - 两个方向都算：TE(ICPI→期货) 与 TE(期货→ICPI)，并给净流 TE_net

本脚本实现 KSG 传递熵：
  TE(X→Y) = ψ(k) + ψ(k+1) + < ψ(n_{y_{t-1}}+1) − ψ(n_{y_t,y_{t-1}}+1) − ψ(n_{y_{t-1},x_{t-1}}+1) >
  其中 ε_t = 全空间(y_t, y_{t-1}, x_{t-1})中到第 k 近邻的 Chebyshev 距离，
  n_{...} 为对应子空间内距离 < ε_t 的点数(不含自身)。

输出：data/features/<symbol>_te.csv  列 [date, TE_icpi_to_open, TE_open_to_icpi, TE_net]
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy.special import digamma

START, END = "2016-04-01", "2021-04-30"
FUTURES_DIR = "data/futures"
ICPI_FILE = "data/features/icpi.csv"
OUT_DIR = "data/features"

WINDOW = 60      # 预测特征用滚动窗口(天)
STEP = 1
K = 4            # KSG k 近邻数（常用 4）


def ksg_te(source, target, k=K):
    """KSG 传递熵：从 source(X) 到 target(Y)，历史长度=1。

    TE(X→Y) = I(y_t ; x_{t-1} | y_{t-1})
    输入为等长一维数组，返回标量 TE（纳特）。
    """
    y_now = np.asarray(target[1:], dtype=float)
    y_past = np.asarray(target[:-1], dtype=float)
    x_past = np.asarray(source[:-1], dtype=float)
    m = y_now.shape[0]

    full = np.column_stack([y_now, y_past, x_past])

    # 每个点到其第 k 近邻的 Chebyshev 距离 ε_t（不含自身）
    eps = np.empty(m)
    for t in range(m):
        d = np.max(np.abs(full - full[t]), axis=1)
        d[t] = np.inf                 # 排除自身
        eps[t] = np.partition(d, k)[k]

    # 子空间内距离 < ε_t 的点数（严格小于，不含自身）
    n_yp = np.empty(m)      # y_{t-1}         (1D)
    n_yyp = np.empty(m)     # (y_t, y_{t-1})  (2D)
    n_ypxp = np.empty(m)    # (y_{t-1}, x_{t-1}) (2D)
    for t in range(m):
        n_yp[t] = np.sum(np.abs(y_past - y_past[t]) < eps[t]) - 1
        n_yyp[t] = np.sum(np.max(np.abs(np.column_stack([y_now, y_past]) - np.column_stack([y_now[t], y_past[t]])), axis=1) < eps[t]) - 1
        n_ypxp[t] = np.sum(np.max(np.abs(np.column_stack([y_past, x_past]) - np.column_stack([y_past[t], x_past[t]])), axis=1) < eps[t]) - 1

    te = (
        digamma(k) + digamma(k + 1)
        + np.mean(digamma(n_yp + 1) - digamma(n_yyp + 1) - digamma(n_ypxp + 1))
    )
    return te


def rolling_te(icpi, open_px):
    """在归一化序列上做滚动窗口 TE，返回两个方向 + 净流的 Series。"""
    dates = icpi.index
    n = len(icpi)
    te_a, te_b, te_net = [], [], []
    out_dates = []
    for t in range(WINDOW - 1, n, STEP):
        si = icpi.values[t - WINDOW + 1: t + 1]
        so = open_px.values[t - WINDOW + 1: t + 1]
        tab = ksg_te(si, so)       # ICPI → 期货
        tba = ksg_te(so, si)       # 期货 → ICPI
        te_a.append(tab)
        te_b.append(tba)
        te_net.append(tab - tba)
        out_dates.append(dates[t])
    idx = pd.DatetimeIndex(out_dates)
    return pd.DataFrame({
        "TE_icpi_to_open": pd.Series(te_a, index=idx),
        "TE_open_to_icpi": pd.Series(te_b, index=idx),
        "TE_net": pd.Series(te_net, index=idx),
    })


if __name__ == "__main__":
    icpi = pd.read_csv(ICPI_FILE, parse_dates=["date"]).set_index("date")["ICPI"]

    for sym in ["C0", "A0", "V0", "JD0", "RB0"]:
        fp = glob.glob(os.path.join(FUTURES_DIR, sym + "_*.csv"))[0]
        fut = pd.read_csv(fp, parse_dates=["date"]).set_index("date").sort_index()
        fut = fut[(fut.index >= START) & (fut.index <= END)]

        # 式(1) 全局 min-max 归一化后再算 TE（论文 Section 2.1：因果分析前归一化）
        ic = (icpi - icpi.min()) / (icpi.max() - icpi.min())
        op = (fut["open"] - fut["open"].min()) / (fut["open"].max() - fut["open"].min())

        common = ic.index.intersection(op.index)
        te = rolling_te(ic.reindex(common), op.reindex(common))
        te.index.name = "date"

        te.to_csv(os.path.join(OUT_DIR, f"{sym}_te.csv"), encoding="utf-8-sig")
        print(f"[{sym}] TE  {te.shape}  区间 {te.index.min().date()}~{te.index.max().date()}  "
              f"净流均值={te['TE_net'].mean():+.4f}  net>0占比={(te['TE_net']>0).mean():.1%}")
