# -*- coding: utf-8 -*-
"""
抓取论文2所需的谷歌趋势（Google Trends）英文关键词数据 —— 跨段重标定版。

⚠️ 前提：谷歌趋势需科学上网。英文词(price)/中文词(价格)有真实量，拼音(jiage)几乎为0，
        论文原文写明 "GT's English keywords were used"，故用英文关键词。

关键修正（对齐论文引用 Xu et al. 2019 的做法）：
    Google Trends 单次请求仅返回 ≤~270 天日度数据，且每段各自归一化到 0–100
    （以本段峰值为 100）。若直接按不重叠分块拼接，每段的"100"代表不同的绝对搜索量，
    段边界会出现人工跳变。
    修法：分块时让相邻段重叠 overlap 天，在重叠区用两段的比例算缩放因子，
    把后一段拉到前一段（已锚定）的尺度上再拼接 —— 即"重叠锚点重标定"。

用法：
    python fetch_google_trends.py              # 重抓全部关键词
    python fetch_google_trends.py price        # 只抓指定关键词
"""
import time
import os
import sys
import pandas as pd
from pytrends.request import TrendReq

START = "2016-04-01"
END = "2021-04-30"
CHUNK_DAYS = 240     # 单段天数（<270 才返回日度）
OVERLAP = 60         # 相邻段重叠天数（用于锚点重标定）


def get_proxy():
    """从 Windows 系统代理设置读取代理地址（梯子常为系统代理模式，端口非常规）。
    requests/pytrends 不读系统代理，必须显式传入。返回代理 URL 列表或 []。"""
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                           r"Software\Microsoft\Windows\CurrentVersion\Internet Settings")
        enable, _ = winreg.QueryValueEx(k, "ProxyEnable")
        server, _ = winreg.QueryValueEx(k, "ProxyServer")
        winreg.CloseKey(k)
        if enable and server:
            server = server.replace("=", ":").split(";")[0].strip()
            if "://" not in server:
                server = "http://" + server
            print(f"[proxy] 检测到系统代理: {server}", flush=True)
            return [server]
    except Exception as e:
        print(f"[proxy] 读注册表失败({e})，用默认端口", flush=True)
    # 兜底：常见梯子端口
    for port in ("4088", "7890", "10809", "1080", "8118", "8889"):
        print(f"[proxy] 未从注册表读到，尝试 127.0.0.1:{port}", flush=True)
        return [f"http://127.0.0.1:{port}"]
    return []

# 论文 Table 3 谷歌趋势英文关键词（最终 PCA 只用到其中 4 个，见 FETCH_KEYS）
KEYWORDS = {
    "price": "价格",
    "decrease in price": "降价",
    "drop in price": "价格下跌",
    "price decrease": "价格下降",
    "price hike": "涨价",
    "price rise": "价格上涨",
    "price shrink": "价格缩减",
    "raise price": "提高价格",
    "reduce price": "降低价格",
}
# 论文 Table 4 最终进 PCA 的 4 个谷歌词（默认只重抓这 4 个，其余 5 个未用到）
NEEDED_KEYS = ["price", "decrease in price", "price decrease", "price rise"]

OUT_DIR = "data/google_trends"
os.makedirs(OUT_DIR, exist_ok=True)


def chunk_ranges(start, end, days=CHUNK_DAYS, overlap=OVERLAP):
    """重叠分块：每段 days 天，相邻段重叠 overlap 天。"""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    cur = s
    while cur < e:
        t = min(cur + pd.Timedelta(days=days), e)
        out.append((cur.strftime("%Y-%m-%d"), t.strftime("%Y-%m-%d")))
        if t >= e:                     # 已覆盖到末尾，退出（否则 cur=e-overlap 恒 <e 会死循环）
            break
        cur = t - pd.Timedelta(days=overlap)
    return out


def fetch_chunk(pt, kw, s, e):
    """抓一个分块，返回非全0的日度DataFrame；失败返回None。"""
    for attempt in range(5):
        try:
            pt.build_payload([kw], timeframe=f"{s} {e}", geo="")
            df = pt.interest_over_time()
            if df is None or df.empty:
                return None
            df = df.drop(columns=["isPartial"], errors="ignore")
            if (df[kw] > 0).sum() == 0:      # 全0 = 被限流或无效
                time.sleep(10 + 8 * attempt)
                continue
            return df
        except Exception as ex:
            sys.stderr.write(f"    retry {attempt+1}/5 ({s}~{e}): {type(ex).__name__}\n")
            time.sleep(10 + 8 * attempt)
    return None


def rescale_stitch(parts, col):
    """重叠锚点重标定 + 拼接（Xu et al. 2019 跨段重标定）。
    前一段为锚，后续段用重叠区比例缩放到前一段尺度，再拼上非重叠部分。
    """
    parts = sorted(parts, key=lambda d: d.index.min())
    result = parts[0][col].astype(float)
    for cur in parts[1:]:
        cur = cur[col].astype(float)
        overlap = result.index.intersection(cur.index)
        if len(overlap) >= 2 and cur.loc[overlap].sum() > 0:
            factor = result.loc[overlap].sum() / cur.loc[overlap].sum()
        else:
            factor = 1.0
        cur_scaled = cur * factor
        new_part = cur_scaled[cur_scaled.index > result.index.max()]
        result = pd.concat([result, new_part])
    result.index.name = "date"
    return result


def main():
    pt = TrendReq(hl="en-US", tz=360, timeout=(10, 30), retries=2, backoff_factor=0.5,
                  proxies=get_proxy())
    keys = sys.argv[1:] if len(sys.argv) > 1 else NEEDED_KEYS
    ranges = chunk_ranges(START, END)
    print(f"重叠分块数={len(ranges)}（每段{CHUNK_DAYS}天，重叠{OVERLAP}天），关键词={keys}")

    report = []
    for kw in keys:
        parts, failed = [], []
        for (s, e) in ranges:
            df = fetch_chunk(pt, kw, s, e)
            if df is not None:
                parts.append(df)
            else:
                failed.append(s)
            time.sleep(4)          # 限流保护
        if parts:
            full = rescale_stitch(parts, kw)
            full = full[(full.index >= START) & (full.index <= END)]
            safe = kw.replace(" ", "_").replace("/", "_")
            out = os.path.join(OUT_DIR, f"gt_{safe}.csv")
            full.to_frame(kw).to_csv(out, encoding="utf-8-sig")
            nz = (full > 0).sum()
            # 平滑性检验：相邻日差的最大值（跳变应远小于原来直接拼接的 0-100 量级）
            max_jump = full.diff().abs().max()
            msg = f"[OK] {kw:18s} {len(full):4d}行 非0={nz:4d} 最大日跳变={max_jump:6.2f} 缺口块={failed}"
            report.append(msg)
            print(msg, flush=True)
        else:
            msg = f"[FAIL] {kw:18s} 全部块失败"
            report.append(msg)
            print(msg, flush=True)

    print("\n==== 汇总 ====")
    for r in report:
        print(r)


if __name__ == "__main__":
    main()
