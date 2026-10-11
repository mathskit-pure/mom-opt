# -*- coding: utf-8 -*-
"""
最终寻优（份额记账 + 收盘盯市 + 相位平均）

对每个 K，在全部 K 个调仓相位上跑一遍取平均，消除「恰好从哪天开始调仓」的偶然性。
维度：N(动量周期) × K(调仓间隔) × L(持仓数) × 执行时点 × 涨停补仓策略
"""
import sys, os, time, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from engine import Data, run, perf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "output")
COST = 0.001
NS = [8, 10, 12, 15]
KS = [5, 8, 10, 12, 15, 20]
LS = [20, 30, 40, 50, 60]
MODES = ["open1", "close1", "close0"]
POLS = ["redistribute", "substitute", "wait"]

d = Data()
PD = pd.DatetimeIndex(d.dates)
base = PD.get_indexer([pd.Timestamp("2019-05-06")], method="bfill")[0]

rows = []
t0 = time.time()
nrun = 0
for K in KS:
    for N in NS:
        for L in LS:
            for mode in MODES:
                nav0, st0 = run(d, N, K, L, mode=mode, policy="redistribute", cost=COST)
                for pol in POLS:
                    vals, cums, vols, mdds = [], [], [], []
                    for o in range(K):
                        st = str(PD[base + o].date())
                        nav, s = run(d, N, K, L, mode=mode, policy=pol, cost=COST, start=st)
                        p = perf(nav)
                        vals.append(p["夏普比率"]); cums.append(p["累计收益"])
                        vols.append(p["年化波动"]); mdds.append(p["最大回撤"])
                        nrun += 1
                    rows.append({
                        "N": N, "K": K, "L": L, "执行时点": mode, "补仓策略": pol,
                        "相位期望夏普": float(np.mean(vals)),
                        "相位标准差": float(np.std(vals)),
                        "最差相位夏普": float(np.min(vals)),
                        "相位期望累计": float(np.mean(cums)),
                        "相位期望波动": float(np.mean(vols)),
                        "相位期望回撤": float(np.mean(mdds)),
                        "单相位夏普": perf(nav0)["夏普比率"],
                        "单相位累计": perf(nav0)["累计收益"],
                        "年均换手": st0["turnover"] / max(st0["n_period"], 1) * (252.0 / K),
                        "阻断率": st0["blocked"] / max(st0["target"], 1),
                    })
    print(f"  K={K} 完成 {time.time()-t0:.0f}s（累计 {nrun} 次回测）", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "final_phase_grid.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 60)

print("\n" + "=" * 140)
print("【A】相位期望夏普 Top20")
print("=" * 140)
print(df.sort_values("相位期望夏普", ascending=False).head(20)[
    ["N", "K", "L", "执行时点", "补仓策略", "相位期望夏普", "相位标准差",
     "最差相位夏普", "相位期望累计", "相位期望波动", "相位期望回撤", "年均换手", "阻断率"]
].to_string(index=False))

print("\n【B】边际剖面（相位期望夏普，其它维度取平均）")
for dim in ["N", "K", "L", "执行时点", "补仓策略"]:
    g = df.groupby(dim)["相位期望夏普"].agg(["mean", "max"])
    print(f"\n-- {dim} --")
    print(g.round(4).to_string())

print("\n【C】执行时点 × 补仓策略 交叉表（相位期望夏普，对 N/K/L 取平均）")
print(df.pivot_table(index="执行时点", columns="补仓策略",
                     values="相位期望夏普", aggfunc="mean").round(4).to_string())

print("\n【D】执行时点边际剖面（按 K 分层，对 N/L/策略 取平均）")
print(df.pivot_table(index="K", columns="执行时点",
                     values="相位期望夏普", aggfunc="mean").round(4).to_string())
print("\n【E】执行时点边际剖面（按 L 分层）")
print(df.pivot_table(index="L", columns="执行时点",
                     values="相位期望夏普", aggfunc="mean").round(4).to_string())
print("\n【F】执行时点边际剖面（按 N 分层）")
print(df.pivot_table(index="N", columns="执行时点",
                     values="相位期望夏普", aggfunc="mean").round(4).to_string())
print(f"\n总回测次数 {nrun}，用时 {time.time()-t0:.0f}s")
