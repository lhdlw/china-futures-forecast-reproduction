# -*- coding: utf-8 -*-
"""整理百度指数数据：把 10 个 Excel 解析成统一 CSV，归档原始文件。

不硬编码中文列名（避免编码/全角字符坑），改为按列位置读取：
  百度指数导出的列顺序固定为：[关键词, 省/市, 时间, 搜索指数pc+移动, 搜索指数pc, 搜索指数移动]
  第3列(索引2)=时间，第4列(索引3)=综合搜索指数。

输出：
  data/baidu_index/bdi_<关键词>.csv       每个关键词一个 CSV（date, 搜索指数）
  data/baidu_index/bdi_all_keywords.csv   合并宽表（date + 10 个关键词）
  data/baidu_index/raw/                   原始 Excel 归档
"""
import os
import glob
import pandas as pd

SRC_DIR = "data"
OUT_DIR = "data/baidu_index"
RAW_DIR = os.path.join(OUT_DIR, "raw")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(SRC_DIR, "*.xlsx")))
if not files:
    print("data/ 下没找到 Excel，请检查文件位置")
    raise SystemExit

all_frames = {}
for f in files:
    df = pd.read_excel(f, sheet_name=0)
    kw = str(df.iloc[0, 0])            # 第1行第1列 = 关键词
    date_col = df.columns[2]           # 第3列 = 时间
    idx_col = df.columns[3]            # 第4列 = 搜索指数pc+移动（综合）
    sub = df[[date_col, idx_col]].copy()
    sub.columns = ["date", kw]
    sub["date"] = pd.to_datetime(sub["date"])
    sub = sub.sort_values("date").reset_index(drop=True)
    sub.to_csv(os.path.join(OUT_DIR, f"bdi_{kw}.csv"), index=False, encoding="utf-8-sig")
    all_frames[kw] = sub
    print(f"[OK] {kw}  ({len(sub)} rows)")

# 合并宽表
keys = sorted(all_frames)
combined = all_frames[keys[0]][["date"]].copy()
for kw in keys:
    combined = combined.merge(all_frames[kw], on="date", how="outer")
combined = combined.sort_values("date").reset_index(drop=True)
combined.to_csv(os.path.join(OUT_DIR, "bdi_all_keywords.csv"), index=False, encoding="utf-8-sig")
print(f"[OK] merged wide table bdi_all_keywords.csv  {combined.shape}")

# 归档原始 Excel
for f in files:
    os.rename(f, os.path.join(RAW_DIR, os.path.basename(f)))
print(f"[OK] {len(files)} raw xlsx moved to data/baidu_index/raw/")
