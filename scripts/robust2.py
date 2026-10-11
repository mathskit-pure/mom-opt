# -*- coding: utf-8 -*-
"""
稳健性验证：候选参数的成本敏感性 / 分年度 / 子样本 / 相位分布。
口径：无行业上限 · T+1 开盘执行 · 涨停空置 · 单边 10bp（敏感性段另标）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from engine import Data, run, perf

MODE, POLICY = "open1", "cash"
d = Data()
PD = pd.DatetimeIndex(d.dates)
base = PD.get_indexer([pd.Timestamp("2019-05-06")], method="bfill")[0]
pd.set_option("display.width", 320)


def phase_sharpe(N, K, L, cost=0.001, start=None):
    b = base if start is None else PD.get_indexer([pd.Timestamp(start)], method="bfill")[0]
    st_map = {o: str(PD[b + o].date()) for o in range(K)}
    vs, cums = [], []
    for o in range(K):
        nav, _ = run(d, N, K, L, mode=MODE, policy=POLICY, cost=cost, start=st_map[o])
        p = perf(nav)
        if p:
            vs.append(p["夏普比率"]); cums.append(p["累计收益"])
    return np.array(vs), np.mean(cums)


def seg_perf(nav, d0, d1):
    s = pd.Series(nav, index=d.dates)
    s = s[(s.index >= pd.Timestamp(d0)) & (s.index <= pd.Timestamp(d1))]
    if len(s) < 20:
        return np.nan, np.nan
    v = s.values / s.values[0]
    return perf(v)["累计收益"], perf(v)["夏普比率"]


CANDS = [(17, 8, 100), (17, 8, 80), (17, 8, 70), (17, 7, 100), (17, 8, 60),
         (15, 8, 60), (15, 10, 50), (12, 8, 50), (10, 5, 12)]

print("=== ① 成本敏感性（相位平均夏普）===")
rows = []
for N, K, L in CANDS:
    r = {"N": N, "K": K, "L": L}
    for c in (0.0, 0.0005, 0.001, 0.002, 0.003):
        v, _ = phase_sharpe(N, K, L, cost=c)
        r["%dbp" % int(c * 1e4)] = round(float(v.mean()), 4)
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== ② 相位分布（10bp，min / 均值 / max / std）===")
rows = []
for N, K, L in CANDS:
    v, cum = phase_sharpe(N, K, L)
    rows.append({"N": N, "K": K, "L": L, "最差": round(float(v.min()), 3),
                 "均值": round(float(v.mean()), 4), "最好": round(float(v.max()), 3),
                 "std": round(float(v.std()), 3),
                 "相位累计均值": round(cum * 100, 1)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== ③ 子样本（相位 0 全段，10bp）===")
SEGS = [("2019-05-06", "2021-12-31"), ("2022-01-04", "2023-12-29"),
        ("2024-01-02", "2026-10-08"), ("2019-05-06", "2026-10-08")]
rows = []
for N, K, L in CANDS:
    nav, st = run(d, N, K, L, mode=MODE, policy=POLICY, cost=0.001)
    r = {"N": N, "K": K, "L": L}
    for a, b in SEGS:
        cu, sh = seg_perf(nav, a, b)
        r["%s~%s" % (a[:7], b[:7])] = "%.1f%%/%.2f" % (cu * 100, sh) if cu == cu else "—"
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== ④ 分年度收益（相位 0，10bp）===")
yrs = {}
for N, K, L in CANDS:
    nav, _ = run(d, N, K, L, mode=MODE, policy=POLICY, cost=0.001)
    s = pd.Series(nav, index=d.dates)
    yrs["N%d/K%d/L%d" % (N, K, L)] = s.pct_change().groupby(s.index.year).apply(
        lambda g: (1 + g).prod() - 1)
b = pd.read_csv(r"C:\Users\12066\WorkBuddy\2026-10-05-23-13-30\data\bench_all.csv", index_col=0)
b.index = pd.to_datetime(b.index)
for c in ["沪深300", "创业板50", "科创50"]:
    s = b[c].dropna()
    yrs[c] = s.groupby(s.index.year).apply(lambda g: g.iloc[-1] / g.iloc[0] - 1)
print((pd.DataFrame(yrs) * 100).round(2).to_string())

print("\n=== ⑤ 全段绩效（相位 0，10bp）===")
rows = []
for N, K, L in CANDS:
    nav, st = run(d, N, K, L, mode=MODE, policy=POLICY, cost=0.001)
    p = perf(nav)
    p["年均换手"] = st["turnover"] / max(st["n_period"], 1) * (252.0 / K)
    p["阻断率"] = st["blocked"] / max(st["target"], 1)
    p["空置权重"] = st["cash_w"] / max(st["n_period"], 1)
    rows.append({"N": N, "K": K, "L": L, **{k: v for k, v in p.items() if k != "组合"}})
P = pd.DataFrame(rows)
print(P.round(4).to_string(index=False))
