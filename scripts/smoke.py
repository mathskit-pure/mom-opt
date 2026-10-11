# -*- coding: utf-8 -*-
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from engine import Data, run, perf

d = Data()
t0 = time.time()
for mode in ("close0", "close1", "open1"):
    for pol in ("redistribute", "substitute", "wait"):
        nav, st = run(d, 10, 5, 12, mode=mode, policy=pol)
        p = perf(nav, f"{mode}/{pol}")
        print(f"{mode:>6} {pol:<13} 累计 {p['累计收益']:>8.2%} 年化 {p['年化收益']:>7.2%} "
              f"波动 {p['年化波动']:>6.2%} 回撤 {p['最大回撤']:>7.2%} 夏普 {p['夏普比率']:.3f} "
              f"| 目标 {st['target']} 阻断 {st['blocked']} ({st['blocked']/max(st['target'],1):5.2%}) "
              f"全天 {st['blocked_all']:>3} 尾盘 {st['blocked_late']:>4} "
              f"等待成 {st['wait_filled']:>3} 败 {st['wait_failed']:>2} 递补 {st['sub_used']:>3} "
              f"均现金 {st['cash_w']/max(st['n_period'],1):.4f}", flush=True)
print(f"耗时 {time.time()-t0:.1f}s")
