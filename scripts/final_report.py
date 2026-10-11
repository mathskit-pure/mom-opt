# -*- coding: utf-8 -*-
"""最终候选：相位平均口径下的完整绩效对比 + 成本敏感性 + 分年度。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from engine import Data, run, perf

MODE, POLICY = "open1", "cash"
d = Data()
PD = pd.DatetimeIndex(d.dates)
base = PD.get_indexer([pd.Timestamp("2019-05-06")], method="bfill")[0]
pd.set_option("display.width", 320)

CANDS = [(17, 8, 90, "① 推荐 N17/K8/L90"),
         (18, 7, 90, "② 并列最优 N18/K7/L90"),
         (17, 8, 100, "③ 次优 N17/K8/L100"),
         (16, 8, 100, "④ 次优 N16/K8/L100"),
         (15, 10, 50, "⑤ 你原定 N15/K10/L50"),
         (10, 5, 12, "⑥ 现网口径 N10/K5/L12")]


def phase_stats(N, K, L, cost=0.001):
    vs, cums, vols, mdds, anns = [], [], [], [], []
    for o in range(K):
        nav, st = run(d, N, K, L, mode=MODE, policy=POLICY, cost=cost,
                      start=str(PD[base + o].date()))
        p = perf(nav)
        if not p:
            continue
        vs.append(p["夏普比率"]); cums.append(p["累计收益"]); vols.append(p["年化波动"])
        mdds.append(p["最大回撤"]); anns.append(p["年化收益"])
    return dict(夏普=float(np.mean(vs)), 夏普std=float(np.std(vs)), 最差=float(np.min(vs)),
                累计=float(np.mean(cums)), 年化=float(np.mean(anns)),
                波动=float(np.mean(vols)), 回撤=float(np.mean(mdds)))


print("=== 相位平均绩效（单边 10bp）===")
rows = []
for N, K, L, tag in CANDS:
    s = phase_stats(N, K, L)
    nav, st = run(d, N, K, L, mode=MODE, policy=POLICY, cost=0.001)
    rows.append({"方案": tag, "N": N, "K": K, "L": L,
                 "相位夏普": round(s["夏普"], 4), "std": round(s["夏普std"], 4),
                 "最差相位": round(s["最差"], 3), "相位累计": round(s["累计"] * 100, 1),
                 "年化": round(s["年化"] * 100, 2), "波动": round(s["波动"] * 100, 2),
                 "回撤": round(s["回撤"] * 100, 2),
                 "年均换手": round(st["turnover"] / max(st["n_period"], 1) * (252.0 / K), 1),
                 "涨停阻断率": round(st["blocked"] / max(st["target"], 1) * 100, 3)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== 成本敏感性（相位平均夏普）===")
rows = []
for N, K, L, tag in CANDS:
    r = {"方案": tag}
    for c in (0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003):
        r["%dbp" % int(c * 1e4)] = round(phase_stats(N, K, L, cost=c)["夏普"], 4)
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== 分年度收益（相位 0，10bp，%）===")
yrs = {}
for N, K, L, tag in CANDS:
    nav, _ = run(d, N, K, L, mode=MODE, policy=POLICY, cost=0.001)
    s = pd.Series(nav, index=d.dates)
    yrs[tag] = s.pct_change().groupby(s.index.year).apply(lambda g: (1 + g).prod() - 1)
b = pd.read_csv(r"C:\Users\12066\WorkBuddy\2026-10-05-23-13-30\data\bench_all.csv", index_col=0)
b.index = pd.to_datetime(b.index)
for c in ["沪深300", "创业板50", "科创50"]:
    s = b[c].dropna()
    yrs[c] = s.groupby(s.index.year).apply(lambda g: g.iloc[-1] / g.iloc[0] - 1)
print((pd.DataFrame(yrs) * 100).round(2).to_string())

print("\n=== 关键子区间（相位 0，10bp）===")
SEGS = [("2019-05-06", "2021-12-31", "牛市段"),
        ("2022-01-04", "2023-12-29", "熊市段"),
        ("2024-01-02", "2026-10-08", "近三年")]
rows = []
for N, K, L, tag in CANDS:
    nav, _ = run(d, N, K, L, mode=MODE, policy=POLICY, cost=0.001)
    s = pd.Series(nav, index=d.dates)
    r = {"方案": tag}
    for a, b_, nm in SEGS:
        x = s[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b_))]
        v = x.values / x.values[0]
        r[nm] = "%.1f%% / %.2f" % (perf(v)["累计收益"] * 100, perf(v)["夏普比率"])
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
