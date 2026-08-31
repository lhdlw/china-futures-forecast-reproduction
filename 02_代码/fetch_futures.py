# -*- coding: utf-8 -*-
"""抓取论文2所需的中国期货主力连续日线数据（AkShare / 新浪源）。

品种映射（论文 Liang & Jia 2022）:
    CSI300 股指期货 -> IF0
    玉米             -> C0
    大豆(豆一)       -> A0
    PVC             -> V0
    鸡蛋             -> JD0
    螺纹钢           -> RB0
数据区间按论文: 2016-04 ~ 2021-04。
"""
import os
import akshare as ak
import pandas as pd

START = "2016-04-01"
END = "2021-04-30"

SYMBOLS = {
    "IF0": "CSI300股指期货",
    "C0": "玉米",
    "A0": "大豆",
    "V0": "PVC",
    "JD0": "鸡蛋",
    "RB0": "螺纹钢",
}

OUT_DIR = "data/futures"
os.makedirs(OUT_DIR, exist_ok=True)

summary = []
for sym, name in SYMBOLS.items():
    try:
        df = ak.futures_zh_daily_sina(symbol=sym)
        df = df.rename(columns={
            "date": "date", "open": "open", "high": "high", "low": "low",
            "close": "close", "volume": "volume", "hold": "hold", "settle": "settle",
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= START) & (df["date"] <= END)].sort_values("date").reset_index(drop=True)
        path = os.path.join(OUT_DIR, f"{sym}_{name}.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        summary.append((sym, name, len(df), str(df["date"].min().date()), str(df["date"].max().date())))
        print(f"[OK] {sym:5s} {name:12s} rows={len(df):4d}  {df['date'].min().date()} -> {df['date'].max().date()}")
    except Exception as e:
        summary.append((sym, name, 0, "ERROR", repr(e)[:80]))
        print(f"[ERR] {sym} {name}: {repr(e)[:120]}")

sdf = pd.DataFrame(summary, columns=["symbol", "name", "rows", "start", "end"])
sdf.to_csv(os.path.join(OUT_DIR, "_summary.csv"), index=False, encoding="utf-8-sig")
print("\n==== 汇总 ====")
print(sdf.to_string(index=False))
