# -*- coding: utf-8 -*-
"""第三轮：锁定 K=7~10，把 N 扫到 14–19、L 扫到 60–110，确认峰值位置。"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from engine import Data, run, perf

NS = [14, 15, 16, 17, 18, 19]
KS = [7, 8, 9, 10]
LS = [60, 70, 80, 90, 100, 110]
MODE, POLICY, COST = "open1", "cash", 0.001

d = Data()
PD = pd.DatetimeIndex(d.dates)
base = PD.get_indexer([pd.Timestamp("2019-05-06")], method="bfill")[0]
starts = {o: str(PD[base + o].date()) for o in range(max(KS))}

rows, t0 = [], time.time()
for N in NS:
    for L in LS:
        for K in KS:
            sh, cums, vols, mdds = [], [], [], []
            for o in range(K):
                nav, st = run(d, N, K, L, mode=MODE, policy=POLICY, cost=COST,
                              start=starts[o])
                p = perf(nav)
                if not p:
                    continue
                sh.append(p["夏普比率"]); cums.append(p["累计收益"])
                vols.append(p["年化波动"]); mdds.append(p["最大回撤"])
            if not sh:
                continue
            rows.append({"N": N, "K": K, "L": L,
                         "相位夏普": float(np.mean(sh)), "相位夏普std": float(np.std(sh)),
                         "最差相位": float(np.min(sh)), "累计收益": float(np.mean(cums)),
                         "年化波动": float(np.mean(vols)), "最大回撤": float(np.mean(mdds))})
    print("  N=%-3d 完成（%.0fs）" % (N, time.time() - t0), flush=True)

G = pd.DataFrame(rows)
G.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output",
                      "grid_phase_n.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print("\n=== TOP 20 ===")
print(G.sort_values("相位夏普", ascending=False).head(20).round(4).to_string(index=False))
print("\n=== 按 N 分层 ==="); print(G.groupby("N")["相位夏普"].agg(["mean", "max"]).round(4).to_string())
print("\n=== 按 K 分层 ==="); print(G.groupby("K")["相位夏普"].agg(["mean", "max"]).round(4).to_string())
print("\n=== 按 L 分层 ==="); print(G.groupby("L")["相位夏普"].agg(["mean", "max"]).round(4).to_string())
print("\n=== N×K 交叉（对 L 平均）===")
print(G.pivot_table(index="N", columns="K", values="相位夏普").round(4).to_string())
print("\n=== N×L 交叉（对 K 平均）===")
print(G.pivot_table(index="N", columns="L", values="相位夏普").round(4).to_string())
print("\n共 %d 组，耗时 %.0fs" % (len(G), time.time() - t0))
