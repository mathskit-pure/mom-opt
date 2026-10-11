# -*- coding: utf-8 -*-
"""第二轮细化网格：L 上探到 120，N 细化到 12–20，K 细化到 6–13。"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from engine import Data, run, perf

NS = [12, 13, 15, 17, 20]
KS = [6, 7, 8, 9, 10, 11, 12, 13]
LS = [40, 50, 60, 70, 80, 100, 120]
MODE, POLICY, COST = "open1", "cash", 0.001

d = Data()
PD = pd.DatetimeIndex(d.dates)
base = PD.get_indexer([pd.Timestamp("2019-05-06")], method="bfill")[0]
starts = {o: str(PD[base + o].date()) for o in range(max(KS))}

rows, t0, done = [], time.time(), 0
total = len(NS) * len(LS) * sum(KS)
for N in NS:
    for L in LS:
        for K in KS:
            sh, cums, vols, mdds = [], [], [], []
            for o in range(K):
                nav, st = run(d, N, K, L, mode=MODE, policy=POLICY,
                              cost=COST, start=starts[o])
                p = perf(nav)
                if not p:
                    continue
                sh.append(p["夏普比率"]); cums.append(p["累计收益"])
                vols.append(p["年化波动"]); mdds.append(p["最大回撤"])
                done += 1
            if not sh:
                continue
            rows.append({"N": N, "K": K, "L": L,
                         "相位夏普": float(np.mean(sh)),
                         "相位夏普std": float(np.std(sh)),
                         "最差相位": float(np.min(sh)),
                         "累计收益": float(np.mean(cums)),
                         "年化波动": float(np.mean(vols)),
                         "最大回撤": float(np.mean(mdds))})
    print("  N=%-3d 完成（%d/%d，%.0fs）" % (N, done, total, time.time() - t0), flush=True)

G = pd.DataFrame(rows)
G.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output",
                      "grid_phase_fine.csv"), index=False, encoding="utf-8-sig")

pd.set_option("display.width", 250)
print("\n=== 相位平均夏普 TOP 25 ===")
print(G.sort_values("相位夏普", ascending=False).head(25).round(4).to_string(index=False))
print("\n=== 按 N 分层 ===")
print(G.groupby("N")["相位夏普"].agg(["mean", "max"]).round(4).to_string())
print("\n=== 按 K 分层 ===")
print(G.groupby("K")["相位夏普"].agg(["mean", "max", "std"]).round(4).to_string())
print("\n=== 按 L 分层 ===")
print(G.groupby("L")["相位夏普"].agg(["mean", "max"]).round(4).to_string())
print("\n=== 稳健 TOP15（std<0.10 且 最差相位>0.88）===")
R = G[(G["相位夏普std"] < 0.10) & (G["最差相位"] > 0.88)]
print(R.sort_values("相位夏普", ascending=False).head(15)
      [["N", "K", "L", "相位夏普", "相位夏普std", "最差相位", "累计收益", "年化波动", "最大回撤"]]
      .round(4).to_string(index=False))
print("\n共 %d 组，耗时 %.0fs" % (len(G), time.time() - t0))
