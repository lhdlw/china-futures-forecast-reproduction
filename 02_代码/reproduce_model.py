# -*- coding: utf-8 -*-
"""
复现论文2 —— 第三步：GWO-CNN-LSTM 模型
Liang & Jia (2022) Section 3.3 + 3.4 + 4.2.2

忠实实现：
  - 结构：1D-CNN(ReLU) → MaxPool → LSTM(多层) → 全连接，预测次日开盘价
  - GWO 优化 6 个超参：学习率、时间步长(lookback)、卷积核大小、卷积步长、
    每层 LSTM 神经元数、隐藏层数（对应论文"learning rate, time-step, size and
    step size of the convolution kernel, number of cells per layer, number of hidden layers"）
  - 80/20 时间序划分（不 shuffle），评价 MAE/RMSE/MAPE/R²（式30-33）
  - 对比 [仅价格] vs [价格+ICPI+TE]

⚠️ 计算量取舍（论文 → 本实现，CPU 可跑完）：
  训练 epochs      500  →  FINAL_EPOCHS(默认100)
  重复次数取平均   10   →  REPEATS(默认3)
  GWO 种群/迭代    未写明 → GWO_POP(默认8) × GWO_ITER(默认10)，适应度用 SEARCH_EPOCHS(默认15)
  MaxPool 池化大小 未写明 → 固定 kernel=2,stride=2
"""
import os
import glob
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)

START, END = "2016-04-01", "2021-04-30"
FUTURES_DIR = "data/futures"
FEAT_DIR = "data/features"
SYMBOLS = ["C0", "A0", "V0", "JD0", "RB0"]

# ---- 计算预算（论文原值见注释）----
FINAL_EPOCHS = 500      # 论文 500（原折中值 100）
REPEATS = 10            # 论文 10（原折中值 3）
GWO_POP = 10
GWO_ITER = 15
SEARCH_EPOCHS = 15
TRAIN_RATIO = 0.8
BATCH_SIZE = 64

# ---- GWO 搜索空间（按论文列出的6个超参）----
# 每个维度用 [0,1] 归一化编码，评估时映射回真实范围（整数项四舍五入）
BOUNDS = {
    "lr":        (1e-4, 1e-2, "log"),   # 学习率
    "lookback":  (5, 30, "int"),        # 时间步长
    "kernel":    (2, 7, "int"),         # 卷积核大小
    "stride":    (1, 3, "int"),         # 卷积步长
    "hidden":    (8, 64, "int"),        # LSTM 每层神经元数
    "layers":    (1, 3, "int"),         # LSTM 隐藏层数
}
PARAM_KEYS = list(BOUNDS.keys())


def decode(x):
    """[0,1]^d -> 真实超参 dict"""
    p = {}
    for i, k in enumerate(PARAM_KEYS):
        lo, hi, kind = BOUNDS[k]
        v = x[i] * (hi - lo) + lo
        if kind == "log":
            v = 10 ** (x[i] * (np.log10(hi) - np.log10(lo)) + np.log10(lo))
        elif kind == "int":
            v = int(round(v))
        p[k] = v
    return p


class CNNLSTM(nn.Module):
    """1D-CNN(ReLU) -> MaxPool -> LSTM -> FC，输入 [batch, lookback, n_features]"""
    def __init__(self, n_features, lookback, kernel, stride, hidden, layers):
        super().__init__()
        self.conv = nn.Conv1d(n_features, hidden, kernel_size=kernel, stride=stride)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: [batch, lookback, n_features] -> [batch, n_features, lookback]
        x = x.transpose(1, 2)
        x = self.relu(self.conv(x))
        x = self.pool(x) if x.shape[-1] >= 2 else x
        x = x.transpose(1, 2)                     # [batch, L, hidden]
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])             # 取最后时刻 -> [batch, 1]


def conv_out_len(lookback, kernel, stride):
    """卷积+池化后的序列长度，非法返回 0"""
    lc = (lookback - kernel) // stride + 1
    if lc < 1:
        return 0
    return max(lc // 2, 1)


def load_dataset(symbol, with_icpi_te):
    """构造监督数据集：X=[过去lookback天的特征], y=[次日开盘价(归一化)]。"""
    feat = pd.read_csv(os.path.join(FEAT_DIR, f"{symbol}_features.csv"), parse_dates=["date"]).set_index("date")
    te = pd.read_csv(os.path.join(FEAT_DIR, f"{symbol}_te.csv"), parse_dates=["date"]).set_index("date")

    cols = ["open_norm"]
    df = feat[["open", "open_norm"]].copy()
    if with_icpi_te:
        te_norm = (te["TE_net"] - te["TE_net"].min()) / (te["TE_net"].max() - te["TE_net"].min())
        df["ICPI_norm"] = feat["ICPI_norm"]
        df["TE_norm"] = te_norm
        cols = ["open_norm", "ICPI_norm", "TE_norm"]
    df = df.dropna()

    X = df[cols].values        # [T, n_features]
    y_norm = df["open_norm"].values
    open_raw = df["open"].values
    return X, y_norm, open_raw, cols


def make_windows(X, y, lookback):
    """滑动窗口：X[i] = 特征[t-lookback..t-1], y[i] = 次日开盘价[t]。"""
    n = len(X)
    xs, ys = [], []
    for t in range(lookback, n):
        xs.append(X[t - lookback:t])
        ys.append(y[t])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def fit_eval(params, X, y_norm, open_raw, epochs, seed=0):
    """按超参训练一次，返回 (验证集MAE[归一化], 各指标[原始价]) 用于适应度/最终评价。"""
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(X)
    ntr = int(n * TRAIN_RATIO)
    lookback = params["lookback"]
    nfeat = X.shape[1]

    # 卷积(未池化)后的序列长度，至少保留 4 个时刻，否则 LSTM 退化成前馈网络
    lc = (lookback - params["kernel"]) // params["stride"] + 1
    if lc < 4:
        return 1e9, None

    Xw, yw = make_windows(X, y_norm, lookback)
    if len(Xw) < 2:
        return 1e9, None
    ntr = min(int(len(Xw) * TRAIN_RATIO), len(Xw) - 1)
    Xtr, ytr = Xw[:ntr], yw[:ntr]
    Xva, yva = Xw[ntr:], yw[ntr:]
    raw_va = open_raw[lookback + ntr:]           # 验证集对应的原始开盘价

    model = CNNLSTM(nfeat, lookback, params["kernel"], params["stride"],
                    params["hidden"], params["layers"])
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"])
    lossf = nn.MSELoss()

    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr).unsqueeze(1)
    Xva_t = torch.tensor(Xva)
    nbatch = max(len(Xtr) // BATCH_SIZE, 1)

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for b in range(nbatch):
            idx = perm[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            opt.zero_grad()
            pred = model(Xtr_t[idx])
            loss = lossf(pred, ytr_t[idx])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred_va = model(Xva_t).squeeze().numpy()
    mae_norm = float(np.mean(np.abs(pred_va - yva)))

    if raw_va is None:
        return mae_norm, None

    # 反归一化到原始价（用全序列 open 的 min/max，式(1)逆变换）
    open_min, open_max = open_raw.min(), open_raw.max()
    pred_raw = pred_va * (open_max - open_min) + open_min
    y_raw = yva * (open_max - open_min) + open_min

    mae = float(np.mean(np.abs(pred_raw - y_raw)))
    rmse = float(np.sqrt(np.mean((pred_raw - y_raw) ** 2)))
    mape = float(np.mean(np.abs(pred_raw - y_raw) / np.abs(y_raw)))
    ss_res = float(np.sum((pred_raw - y_raw) ** 2))
    ss_tot = float(np.sum((y_raw - y_raw.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return mae_norm, {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def gwo(objective, dim, pop, iters, seed=0):
    """标准灰狼优化器（论文式19-29），返回最优解向量。"""
    rng = np.random.default_rng(seed)
    wolves = rng.random((pop, dim))
    fitness = np.array([objective(w) for w in wolves])

    order = np.argsort(fitness)
    alpha = wolves[order[0]].copy()
    beta = wolves[order[1]].copy()
    delta = wolves[order[2]].copy()
    a_best = fitness[order[0]]

    for t in range(iters):
        a = 2 - 2 * t / max(iters - 1, 1)
        for i in range(pop):
            for j in range(dim):
                r1, r2 = rng.random(), rng.random()
                A1 = 2 * a * r1 - a; C1 = 2 * r2
                D_alpha = abs(C1 * alpha[j] - wolves[i, j])
                X1 = alpha[j] - A1 * D_alpha

                r1, r2 = rng.random(), rng.random()
                A2 = 2 * a * r1 - a; C2 = 2 * r2
                D_beta = abs(C2 * beta[j] - wolves[i, j])
                X2 = beta[j] - A2 * D_beta

                r1, r2 = rng.random(), rng.random()
                A3 = 2 * a * r1 - a; C3 = 2 * r2
                D_delta = abs(C3 * delta[j] - wolves[i, j])
                X3 = delta[j] - A3 * D_delta

                wolves[i, j] = np.clip((X1 + X2 + X3) / 3, 0, 1)

        fitness = np.array([objective(w) for w in wolves])
        order = np.argsort(fitness)
        alpha = wolves[order[0]].copy()
        beta = wolves[order[1]].copy()
        delta = wolves[order[2]].copy()
        if fitness[order[0]] < a_best:
            a_best = fitness[order[0]]
    return alpha


def run_symbol(symbol, with_icpi_te):
    X, y_norm, open_raw, cols = load_dataset(symbol, with_icpi_te)

    def objective(w):
        p = decode(w)
        mae, _ = fit_eval(p, X, y_norm, open_raw, SEARCH_EPOCHS)
        return mae

    best = gwo(objective, len(PARAM_KEYS), GWO_POP, GWO_ITER)
    best_p = decode(best)

    # 最终评价：最优超参下重复训练 REPEATS 次取平均
    metrics_list = []
    for r in range(REPEATS):
        _, m = fit_eval(best_p, X, y_norm, open_raw, FINAL_EPOCHS, seed=r)
        if m:
            metrics_list.append(m)
    avg = {k: float(np.mean([m[k] for m in metrics_list])) for k in ["MAE", "RMSE", "MAPE", "R2"]}
    return best_p, avg


if __name__ == "__main__":
    # 可选参数：python reproduce_model.py C0 A0 ...  只跑指定品种（默认全部）
    run_symbols = sys.argv[1:] if len(sys.argv) > 1 else SYMBOLS
    RESULT_FILE = os.path.join(FEAT_DIR, f"model_results_ep{FINAL_EPOCHS}_rep{REPEATS}.txt")

    header = f"{'品种':6s} {'输入特征':20s} {'MAE':>9s} {'RMSE':>9s} {'MAPE':>8s} {'R2':>7s}   最优超参"
    sep = "-" * 110

    def report(line):
        print(line, flush=True)
        with open(RESULT_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # 清空旧结果文件
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write(header + "\n" + sep + "\n")

    report(header)
    report(sep)
    for sym in run_symbols:
        report(f"[{sym}] 开始 GWO 搜索：仅价格 ...")
        bp_base, res_base = run_symbol(sym, with_icpi_te=False)
        report(f"{sym:6s} {'[仅价格]':20s} {res_base['MAE']:9.4f} {res_base['RMSE']:9.4f} "
               f"{res_base['MAPE']:8.4f} {res_base['R2']:7.4f}   {bp_base}")

        report(f"[{sym}] 开始 GWO 搜索：价格+ICPI+TE ...")
        bp_full, res_full = run_symbol(sym, with_icpi_te=True)
        report(f"{sym:6s} {'[价格+ICPI+TE]':20s} {res_full['MAE']:9.4f} {res_full['RMSE']:9.4f} "
               f"{res_full['MAPE']:8.4f} {res_full['R2']:7.4f}   {bp_full}")

        dmae = (res_base['MAE'] - res_full['MAE']) / res_base['MAE'] * 100
        report(f"{'':6s} {'→ 改善':20s} MAE {dmae:+.1f}%")
        report(sep)
    report(f"全部完成。结果已保存到 {RESULT_FILE}")
