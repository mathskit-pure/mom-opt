# -*- coding: utf-8 -*-
"""
网格寻优：动量周期 N × 调仓间隔 K × 持仓数 L × 执行时点 × 涨停补仓策略
目标：夏普比率最大化（无申万三级行业数量限制，保留次新股过滤 60 个交易日）
"""
import sys, os, time, itertools, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from engine import Data, run, perf

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

NS = [5, 10, 15, 20, 30, 40, 60]
KS = [1, 2, 3, 5, 10, 15, 20]
LS = [5, 8, 10, 12, 15, 20, 25, 30]
MODES = ["open1", "close1", "close0"]
POLICIES = ["redistribute", "substitute", "wait"]
COST = 0.001          # 单边 10bp

d = Data()
rows = []
t0 = time.time()
cnt = 0
total = len(NS) * len(KS) * len(LS) * len(MODES) * len(POLICIES)
for N in NS:
    for K in KS:
        for L in LS:
            for mode in MODES:
                for pol in POLICIES:
                    nav, st = run(d, N, K, L, mode=mode, policy=pol, cost=COST)
                    p = perf(nav)
                    if not p:
                        continue
                    rows.append({
                        "N": N, "K": K, "L": L, "执行时点": mode, "补仓策略": pol,
                        "累计收益": p["累计收益"], "年化收益": p["年化收益"],
                        "年化波动": p["年化波动"], "最大回撤": p["最大回撤"],
                        "夏普比率": p["夏普比率"], "卡玛比率": p["卡玛比率"],
                        "阻断率": st["blocked"] / max(st["target"], 1),
                        "阻断_全天": st["blocked_all"], "阻断_尾盘": st["blocked_late"],
                        "等待成功": st["wait_filled"], "等待失败": st["wait_failed"],
                        "递补次数": st["sub_used"],
                        "年均换手": st["turnover"] / max(st["n_period"], 1) * (252.0 / K),
                        "成本拖累": st["cost"] / max(st["n_period"], 1) * (252.0 / K),
                        "平均持仓": st["avg_n"] / max(st["n_period"], 1),
                    })
                    cnt += 1
                    if cnt % 200 == 0:
                        print(f"  {cnt}/{total}  {time.time()-t0:.0f}s", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "grid_full.csv"), index=False, encoding="utf-8-sig")
print(f"\n完成 {len(df)} 组，耗时 {time.time()-t0:.0f}s → {os.path.join(OUT,'grid_full.csv')}")

pd.set_option("display.width", 250)
for mode in MODES:
    sub = df[df["执行时点"] == mode].sort_values("夏普比率", ascending=False)
    print(f"\n===== {mode} 夏普 Top10 =====")
    print(sub.head(10)[["N", "K", "L", "补仓策略", "累计收益", "年化收益", "年化波动",
                        "最大回撤", "夏普比率", "阻断率", "年均换手"]].to_string(index=False))
