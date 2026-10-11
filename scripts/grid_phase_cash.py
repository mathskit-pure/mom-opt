# -*- coding: utf-8 -*-
"""
相位平均网格寻优（最终口径）
============================
口径：无行业上限 / T+1 开盘执行 / 涨停无法建仓则权重空置留现金 / 单边 10bp 成本
寻优：对每个 (N,K,L)，把回测起点平移 0..K-1 个交易日，跑 K 次取夏普均值。
      这样"恰好从哪天开始调仓"的日历对齐假象会被平均掉。
"""
import sys, os, itertools, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from engine import Data, run, perf

NS = [5, 8, 10, 12, 15, 20, 25, 30]
KS = [2, 3, 5, 8, 10, 12, 15, 20]
LS = [10, 20, 30, 40, 50, 60, 80]
MODE, POLICY, COST = "open1", "cash", 0.001

d = Data()
PD = pd.DatetimeIndex(d.dates)
base = PD.get_indexer([pd.Timestamp("2019-05-06")], method="bfill")[0]
starts = {o: str(PD[base + o].date()) for o in range(max(KS))}
print("回测区间 %s → %s（%d 个交易日）" % (PD[base].date(), PD[-1].date(), d.T - base))
print("网格 %d×%d×%d = %d 组，相位平均共 %d 次回测"
      % (len(NS), len(KS), len(LS), len(NS) * len(KS) * len(LS),
         len(NS) * len(LS) * sum(KS)))

rows, t0 = [], time.time()
done = 0
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
                sh.append(p["夏普比率"])
                cums.append(p["累计收益"])
                vols.append(p["年化波动"])
                mdds.append(p["最大回撤"])
                done += 1
            if not sh:
                continue
            rows.append({"N": N, "K": K, "L": L,
                         "相位夏普": float(np.mean(sh)),
                         "相位夏普std": float(np.std(sh)),
                         "最差相位": float(np.min(sh)),
                         "最好相位": float(np.max(sh)),
                         "累计收益": float(np.mean(cums)),
                         "年化波动": float(np.mean(vols)),
                         "最大回撤": float(np.mean(mdds))})
    print("  N=%-3d 完成（%d/%d 次，%.0fs）" % (N, done, total, time.time() - t0),
          flush=True)

G = pd.DataFrame(rows)
G.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "output", "grid_phase_cash.csv"),
         index=False, encoding="utf-8-sig")

pd.set_option("display.width", 250)
print("\n=== 相位平均夏普 TOP 20 ===")
print(G.sort_values("相位夏普", ascending=False).head(20).round(4).to_string(index=False))

print("\n=== 按 N 分层（对 K/L 取平均）===")
print(G.groupby("N")["相位夏普"].agg(["mean", "max"]).round(4).to_string())
print("\n=== 按 K 分层 ===")
print(G.groupby("K")["相位夏普"].agg(["mean", "max", "std"]).round(4).to_string())
print("\n=== 按 L 分层 ===")
print(G.groupby("L")["相位夏普"].agg(["mean", "max"]).round(4).to_string())

print("\n=== TOP20 明细（含稳健性）===")
T = G.sort_values("相位夏普", ascending=False).head(20)
print(T[["N", "K", "L", "相位夏普", "相位夏普std", "最差相位", "累计收益",
         "年化波动", "最大回撤"]].round(4).to_string(index=False))

print("\n=== 稳健筛选：std < 0.13 且 最差相位 > 0.85 的 TOP15 ===")
R = G[(G["相位夏普std"] < 0.13) & (G["最差相位"] > 0.85)]
print(R.sort_values("相位夏普", ascending=False).head(15)
      [["N", "K", "L", "相位夏普", "相位夏普std", "最差相位", "累计收益", "最大回撤"]]
      .round(4).to_string(index=False))

print("\n共 %d 组，耗时 %.0fs" % (len(G), time.time() - t0))
