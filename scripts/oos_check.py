# -*- coding: utf-8 -*-
"""分年度表现 + 2022 年以来 / 2024 年以来子区间表现（解释样本外折价来自哪里）。"""
import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from engine import Data, run, perf

COST = 0.001
d = Data()
pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 60)

CFGS = [
    ("全样本最优 N10K12L40", 10, 12, 40, "close1", "redistribute"),
    ("open1 N10K12L40", 10, 12, 40, "open1", "redistribute"),
    ("IS最优 open1 N5K10L30", 5, 10, 30, "open1", "wait"),
    ("IS最优 close0 N5K15L50", 5, 15, 50, "close0", "substitute"),
    ("原口径 N10K5L12 行业≤2 不设涨停", 10, 5, 12, "close0", "redistribute"),
]


def sub_perf(s, d0):
    v = s[s.index >= pd.Timestamp(d0)].values
    return perf(v / v[0]) if len(v) > 50 else {}


rows_y, rows_p = [], []
for tag, N, K, L, mode, pol in CFGS:
    if tag.startswith("原口径"):
        nav, st = run(d, N, K, L, mode=mode, policy=pol, cost=COST, cap=2, ignore_limit=True)
    else:
        nav, st = run(d, N, K, L, mode=mode, policy=pol, cost=COST)
    s = pd.Series(nav, index=d.dates)
    r = s.pct_change()
    y = r.groupby(r.index.year).apply(lambda g: (1 + g).prod() - 1)
    y.name = tag
    rows_y.append(y)
    p_all = perf(nav, tag)
    p22 = sub_perf(s, "2022-01-04")
    p24 = sub_perf(s, "2024-01-02")
    rows_p.append({
        "方案": tag,
        "全期累计": p_all["累计收益"], "全期夏普": p_all["夏普比率"],
        "全期回撤": p_all["最大回撤"],
        "2022起累计": p22.get("累计收益"), "2022起夏普": p22.get("夏普比率"),
        "2024起累计": p24.get("累计收益"), "2024起夏普": p24.get("夏普比率"),
        "2024起回撤": p24.get("最大回撤"),
        "年均换手": st["turnover"] / max(st["n_period"], 1) * (252.0 / K),
    })

print("=" * 130)
print("【分年度收益】")
print("=" * 130)
Y = pd.DataFrame(rows_y)
b = pd.read_csv(r"C:\Users\12066\WorkBuddy\2026-10-05-23-13-30\data\bench_all.csv", index_col=0)
b.index = pd.to_datetime(b.index)
for c in ["沪深300", "创业板50", "科创50"]:
    if c in b.columns:
        s = b[c].dropna()
        s.name = c
        Y = Y.T.join(s.groupby(s.index.year).apply(lambda g: g.iloc[-1] / g.iloc[0] - 1)).T
print((Y * 100).round(2).to_string())

print("\n" + "=" * 130)
print("【子区间表现】")
print("=" * 130)
P = pd.DataFrame(rows_p)
for c in P.columns:
    if c != "方案":
        P[c] = P[c].astype(float)
print(P.round(4).to_string(index=False))
