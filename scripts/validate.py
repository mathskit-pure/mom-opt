# -*- coding: utf-8 -*-
"""对照校验：复现项目原版 backtest.py（行业上限 2、忽略涨停、T 日收盘执行）。"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from engine import Data, run, perf

d = Data()
# 原版：N=10 K=5 L=12 行业上限2、不管涨停、T日收盘执行
for cap, il, mode in [(2, True, "close0"), (None, True, "close0"), (2, False, "close0"),
                      (None, False, "close0")]:
    nav, st = run(d, 10, 5, 12, mode=mode, policy="redistribute", cap=cap, ignore_limit=il)
    p = perf(nav)
    print(f"cap={str(cap):>4} 忽略涨停={str(il):<5} {mode}  累计 {p['累计收益']:>8.2%} "
          f"年化 {p['年化收益']:>7.2%} 波动 {p['年化波动']:>6.2%} 回撤 {p['最大回撤']:>7.2%} "
          f"夏普 {p['夏普比率']:.3f}")
print("\n项目已发布数值（截至 2026-10-08）：动量轮动策略 累计 +553.83%")
