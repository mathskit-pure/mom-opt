# -*- coding: utf-8 -*-
"""相位平均网格（第二轮）：扩展持仓数 L，并顺带对三种补仓策略做相位平均比较。"""
import sys, os, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from engine import Data, run, perf

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
COST = 0.001
NS = [8, 10, 12, 15]
KS = [8, 10, 12, 15, 20]
LS = [30, 40, 50, 60, 80]
MODES = ["open1", "close1", "close0"]

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
                    nav, _ = run(d, N, K, L, mode=mode, policy="redistribute",
                                 cost=COST, start=st)
                    p = perf(nav)
                    vals.append(p["夏普比率"])
                    cums.append(p["累计收益"])
                rows.append({"N": N, "K": K, "L": L, "执行时点": mode,
                             "相位期望夏普": float(np.mean(vals)),
                             "相位标准差": float(np.std(vals)),
                             "最差相位夏普": float(np.min(vals)),
                             "相位期望累计": float(np.mean(cums))})
    print(f"  K={K} 完成 {time.time()-t0:.0f}s", flush=True)

df = pd.DataFrame(rows)
old = pd.read_csv(os.path.join(OUT, "grid_phase.csv"))
df = pd.concat([old, df], ignore_index=True)
df.to_csv(os.path.join(OUT, "grid_phase.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 260)

print("\n===== 相位平均夏普 Top15 =====")
print(df.sort_values("相位期望夏普", ascending=False).head(15).to_string(index=False))
print("\n===== 相位标准差最小（最稳）Top10（夏普≥0.85）=====")
s = df[df["相位期望夏普"] >= 0.85].sort_values("相位标准差")
print(s.head(10).to_string(index=False))
print("\n===== 边际剖面（相位平均）=====")
for dim in ["L", "K", "N"]:
    g = df.groupby([dim, "执行时点"])["相位期望夏普"].mean().unstack()
    print(f"\n-- 按 {dim}（其它维度取平均）--")
    print(g.round(4).to_string())

# ---------- 补仓策略的相位平均比较（在若干代表参数上）----------
print("\n\n===== 补仓策略 × 执行时点：相位平均夏普（N=10/12, K=10/15, L=40/50）=====")
combos = [(10, 10, 40), (10, 15, 40), (12, 10, 40), (12, 15, 50)]
res = []
for mode in MODES:
    for pol in ["redistribute", "substitute", "wait"]:
        vs = []
        for N, K, L in combos:
            v = []
            for o in range(K):
                st = str(PD[base + o].date())
                v.append(perf(run(d, N, K, L, mode=mode, policy=pol,
                                  cost=COST, start=st)[0])["夏普比率"])
            vs.append(np.mean(v))
        res.append({"执行时点": mode, "补仓策略": pol,
                    **{f"N{a}K{b}L{c}": round(v, 4)
                       for (a, b, c), v in zip(combos, vs)},
                    "平均": round(float(np.mean(vs)), 4)})
R = pd.DataFrame(res)
print(R.to_string(index=False))
print("\n同执行时点内，相对 redistribute 的平均差：")
for mode in MODES:
    sub = R[R["执行时点"] == mode].set_index("补仓策略")["平均"]
    print(f"  {mode}: " + "  ".join(
        f"{p} {sub[p]-sub['redistribute']:+.4f}" for p in ["substitute", "wait"]))
R.to_csv(os.path.join(OUT, "policy_phase.csv"), index=False, encoding="utf-8-sig")
