# -*- coding: utf-8 -*-
"""
相位平均网格寻优（抗日历对齐过拟合）
对每个 K，把回测起点在 [0, K) 全部相位上跑一遍取平均夏普，得到「相位期望夏普」。
这样得到的参数不再依赖「恰好从哪天开始调仓」。
"""
import sys, os, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from engine import Data, run, perf

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
COST = 0.001
NS = [5, 8, 10, 12, 15, 20]
KS = [5, 8, 10, 12, 15, 20]
LS = [20, 30, 40]
MODES = ["open1", "close1", "close0"]
POL = "redistribute"

d = Data()
PD = pd.DatetimeIndex(d.dates)
base = PD.get_indexer([pd.Timestamp("2019-05-06")], method="bfill")[0]

rows = []
t0 = time.time()
for K in KS:
    for N in NS:
        for L in LS:
            for mode in MODES:
                vals, cums = [], []
                for o in range(K):
                    st = str(PD[base + o].date())
                    nav, s = run(d, N, K, L, mode=mode, policy=POL, cost=COST, start=st)
                    p = perf(nav)
                    vals.append(p["夏普比率"])
                    cums.append(p["累计收益"])
                rows.append({"N": N, "K": K, "L": L, "执行时点": mode,
                             "相位期望夏普": float(np.mean(vals)),
                             "相位标准差": float(np.std(vals)),
                             "最差相位夏普": float(np.min(vals)),
                             "相位期望累计": float(np.mean(cums))})
    print(f"  K={K} 完成  {time.time()-t0:.0f}s", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "grid_phase.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 260)
print("\n===== 相位平均后 夏普 Top15 =====")
print(df.sort_values("相位期望夏普", ascending=False).head(15).to_string(index=False))

print("\n===== 各执行时点的最优（按相位期望夏普）=====")
for mode in MODES:
    s = df[df["执行时点"] == mode].sort_values("相位期望夏普", ascending=False)
    print(f"\n-- {mode} --")
    print(s.head(5).to_string(index=False))

print("\n===== 边际剖面（相位平均）：其它维度取该列最优 =====")
for dim, others in [("K", ["N", "L"]), ("N", ["K", "L"]), ("L", ["N", "K"])]:
    g = df.groupby([dim, "执行时点"])["相位期望夏普"].mean().unstack()
    print(f"\n-- 按 {dim}（对 {others} 取平均）--")
    print(g.round(4).to_string())
