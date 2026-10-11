# -*- coding: utf-8 -*-
"""参数剖面：固定其它维度，看夏普随单个参数变化的形状（判断是否尖峰过拟合）。"""
import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from engine import Data, run, perf

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
COST = 0.001
d = Data()
pd.set_option("display.width", 260)


def sh(N, K, L, mode, pol, c=COST):
    return perf(run(d, N, K, L, mode=mode, policy=pol, cost=c)[0])["夏普比率"]


print("=" * 100)
print("【剖面 A】N=10、L=40 下，夏普随调仓间隔 K 的变化")
print("=" * 100)
rows = []
for K in [3, 5, 8, 10, 12, 15, 20, 25, 30, 40]:
    rows.append({"K": K, **{f"{m}/{p[:4]}": round(sh(10, K, 40, m, p), 4)
                            for m in ["open1", "close1", "close0"]
                            for p in ["redistribute", "substitute", "wait"]}})
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 100)
print("【剖面 B】N=10、K=12 下，夏普随持仓数 L 的变化")
print("=" * 100)
rows = []
for L in [5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80]:
    rows.append({"L": L, **{f"{m}/{p[:4]}": round(sh(10, 12, L, m, p), 4)
                            for m in ["open1", "close1", "close0"]
                            for p in ["redistribute", "substitute", "wait"]}})
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 100)
print("【剖面 C】K=12、L=40 下，夏普随动量周期 N 的变化")
print("=" * 100)
rows = []
for N in [3, 5, 8, 10, 12, 15, 20, 30, 40, 60]:
    rows.append({"N": N, **{f"{m}/{p[:4]}": round(sh(N, 12, 40, m, p), 4)
                            for m in ["open1", "close1", "close0"]
                            for p in ["redistribute", "substitute", "wait"]}})
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 100)
print("【剖面 D】交易成本敏感性（N=10 K=12 L=40）")
print("=" * 100)
rows = []
for c in [0.0, 0.0005, 0.001, 0.002, 0.003]:
    rows.append({"单边成本bp": c * 1e4, **{f"{m}/{p[:4]}": round(sh(10, 12, 40, m, p, c), 4)
                                           for m in ["open1", "close1", "close0"]
                                           for p in ["redistribute", "substitute", "wait"]}})
print(pd.DataFrame(rows).to_string(index=False))
