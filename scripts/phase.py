# -*- coding: utf-8 -*-
"""
相位敏感性检验：把回测起点在调仓网格上平移 o 个交易日重跑，看最优参数是否稳定。
若某组参数只在某一相位上表现好 → 是日历对齐造成的假象，不能用。
"""
import sys, os, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from engine import Data, run, perf

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
COST = 0.001
d = Data()
PD = pd.DatetimeIndex(d.dates)
base = PD.get_indexer([pd.Timestamp("2019-05-06")], method="bfill")[0]
pd.set_option("display.width", 260)

# ---------- 快速验证：K=12 尖峰是否依赖相位 ----------
print("=" * 100)
print("【快速验证】N=10 L=40 close1/redistribute，不同起点相位下的夏普")
print("=" * 100)
rows = []
for K in [8, 10, 11, 12, 13, 14, 15, 20]:
    r = {"K": K}
    vals = []
    for o in range(6):
        st = str(PD[base + o].date())
        v = perf(run(d, 10, K, 40, mode="close1", policy="redistribute",
                     cost=COST, start=st)[0])["夏普比率"]
        r[f"相位{o}"] = round(v, 3)
        vals.append(v)
    r["均值"] = round(float(np.mean(vals)), 3)
    r["标准差"] = round(float(np.std(vals)), 3)
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
