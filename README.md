# 基于网络搜索与信息传递的中国期货价格预测 —— 复现

> **非官方复现（unofficial reproduction）** · 代码为独立实现，未使用原论文作者的代码。

## 复现对象

Liang, J., & Jia, G. (2022). *China futures price forecasting based on online search and information transfer.* **Data Science and Management**, 5, 187–198.

- DOI：https://doi.org/10.1016/j.dsm.2022.09.002
- 期刊：Data Science and Management（KeAi / Elsevier，开放获取）

```bibtex
@article{liang2022china,
  title   = {China futures price forecasting based on online search and information transfer},
  author  = {Liang, Jingyi and Jia, Guozhu},
  journal = {Data Science and Management},
  volume  = {5},
  pages   = {187--198},
  year    = {2022},
  doi     = {10.1016/j.dsm.2022.09.002}
}
```

## 这个仓库是什么

对上述论文的完整复现管线：

```
百度指数(中文9词) + 谷歌趋势(英文4词)
        │ 对数 → 时滞平移 → z-score → PCA(特征值>1) → 式(4)加权
        ▼
      ICPI（网络消费者物价指数）
        │ 传递熵 TE（KSG 估计，窗口60/步长1）
        ▼
  [开盘价] vs [开盘价+ICPI+TE] → GWO-CNN-LSTM 预测
```

## 复现结论（诚实汇报）

1. **流程忠实复现成功**：关键词映射、预处理顺序、PCA、传递熵、GWO-CNN-LSTM 均与论文对齐。
2. **论文"加入 ICPI+TE 能提升预测精度"的结论未能复现**：5 个品种中 4 个融合模型反而比"仅价格"更差（玉米 -31.6%、大豆 -157%、PVC -2.6%、螺纹钢 -13%），仅鸡蛋 +7.3% 小幅改善。
3. **根因已定位**：本复现得到的 ICPI 只有 3 个主成分 / 68.33% 方差（论文 4 个 / 79.01%）；Google Trends 相对热度数据会回溯修订，2026 年抓取的与论文 2021 年抓取的同一批词数值无法对上，属数据层难以完全还原的差异。

> 详细过程、全部数字与排查证据见 [`01_报告与图表/复现报告.pdf`](01_报告与图表/复现报告.pdf)。

## 目录结构

```
├── 00_说明_请先读我.md   成果包总览
├── 01_报告与图表/        复现报告（PDF/Word/HTML）+ 对比图
├── 02_代码/              8 个复现脚本（管线说明见 代码说明.md）
├── 03_数据/
│   ├── futures_期货价格/       期货日线（来源 AkShare / 新浪财经）
│   └── features_特征与结果/    ICPI / 传递熵 / 各品种特征 / 模型结果（中间产物）
└── 04_论文与笔记/        精读笔记与对比表（原文 PDF 未包含，见该目录 README）
```

## 如何运行

见 [`02_代码/代码说明.md`](02_代码/代码说明.md) 与 [`00_说明_请先读我.md`](00_说明_请先读我.md)。

```bash
pip install pandas numpy scikit-learn torch pytrends akshare
# 脚本相对路径按 data/ 书写：把 03_数据 重命名为 data，与脚本放在同级
```

## 数据与版权说明

- **代码**：独立实现，MIT License（见 [LICENSE](LICENSE)）。
- **百度指数 / 谷歌趋势原始数据**：因版权与服务条款限制，**未包含在本仓库**。采集方法见 [`04_论文与笔记/百度指数采集说明.md`](04_论文与笔记/百度指数采集说明.md) 与 `02_代码/fetch_google_trends.py`。
- **期货价格**：来源 AkShare（新浪财经接口），字段与来源见 [`03_数据/数据说明.md`](03_数据/数据说明.md)。
- **原论文 PDF**：未包含，请通过 DOI 获取。

## 致谢

感谢原论文作者 Liang & Jia (2022) 的公开研究。
