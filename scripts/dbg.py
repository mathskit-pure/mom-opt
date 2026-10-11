# -*- coding: utf-8 -*-
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from engine import Data, run

d = Data()
N, K, L = 10, 5, 12
mom, adj = d.mom_adj(N)
tradable = (~d.seal_close) & d.has
start_pos = int(np.searchsorted(d.dates.values, np.datetime64("2019-05-06")))
reb = list(range(start_pos, d.T - 1, K))

shown = 0
for j, p in enumerate(reb):
    e = p + 1
    ok = d.pool_mask[p] & (d.listed[p] >= 60)
    m, a = mom[p], adj[p]
    cand = np.where(ok & np.isfinite(m) & np.isfinite(a) & (m > 0))[0]
    if not len(cand):
        continue
    order = cand[np.argsort(-a[cand])]
    top = order[:L]
    bm = ~tradable[e][top]
    if not bm.any():
        continue
    # 只有出现阻断时才打印
    wa = a[top]
    w_top = wa / wa.sum()
    keep = ~bm
    w_red = w_top.copy()
    w_red[keep] /= w_red[keep].sum()
    w_red[bm] = 0
    extra = [c for c in order[L:] if tradable[e][c]]
    need = int(bm.sum())
    take = extra[:need]
    w_sub = w_top.copy()
    fi = np.where(bm)[0]
    for k2, c in enumerate(take):
        w_sub[fi[k2]] = a[c]
    w_sub = w_sub / w_sub.sum()
    print(f"\n=== 决策 {d.dates[p].date()} 执行 {d.dates[e].date()} 阻断 {need} 只 ===")
    print(" idx  code       adj    w原    w分配   w递补   封板")
    for k2, c in enumerate(top):
        print(f" {k2:>3} {d.codes[c]:>9} {a[c]:6.3f} {w_top[k2]:6.2%} {w_red[k2]:6.2%} "
              f"{w_sub[k2]:6.2%}   {'是' if bm[k2] else ''}")
    for k2, c in enumerate(take):
        print(f"  补 {d.codes[c]:>9} {a[c]:6.3f}        {'':6} {w_sub[fi[k2]]:6.2%}")
    shown += 1
    if shown >= 4:
        break
