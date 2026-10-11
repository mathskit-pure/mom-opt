# -*- coding: utf-8 -*-
"""候选方案详解：相位稳定性、分年度、成本敏感性、集中度、与基准对比。"""
import sys, os, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from engine import Data, run, perf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "output")
SRC = r"C:\Users\12066\WorkBuddy\2026-10-05-23-13-30"
COST = 0.001
d = Data()
PD = pd.DatetimeIndex(d.dates)
base = PD.get_indexer([pd.Timestamp("2019-05-06")], method="bfill")[0]
pd.set_option("display.width", 320)
pd.set_option("display.max_columns", 60)

CAND = [
    (15, 8, 60), (12, 8, 50), (15, 10, 50), (15, 10, 60), (12, 10, 50),
    (12, 8, 60), (12, 12, 50), (15, 12, 50), (10, 10, 50), (12, 10, 40),
]

print("=" * 150)
print("【候选方案（open1 + 依次递补）相位稳定性】")
print("=" * 150)
rows = []
for N, K, L in CAND:
    vals = []
    for o in range(K):
        st = str(PD[base + o].date())
        vals.append(perf(run(d, N, K, L, mode="open1", policy="substitute",
                             cost=COST, start=st)[0])["夏普比率"])
    nav, s = run(d, N, K, L, mode="open1", policy="substitute", cost=COST)
    p = perf(nav)
    rows.append({"N": N, "K": K, "L": L, "相位期望夏普": np.mean(vals),
                 "相位std": np.std(vals), "最差相位": np.min(vals),
                 "单相位夏普": p["夏普比率"], "累计收益": p["累计收益"],
                 "年化": p["年化收益"], "波动": p["年化波动"], "回撤": p["最大回撤"],
                 "年均换手": s["turnover"] / max(s["n_period"], 1) * (252.0 / K),
                 "阻断率": s["blocked"] / max(s["target"], 1)})
F = pd.DataFrame(rows).sort_values("相位期望夏普", ascending=False)
print(F.to_string(index=False))

BEST = (12, 8, 50)
print(f"\n\n主推方案：N={BEST[0]} K={BEST[1]} L={BEST[2]} / T+1开盘 / 依次递补")

print("\n" + "=" * 150)
print("【补仓策略在主推参数下的三法对比（相位平均）】")
print("=" * 150)
rows = []
N, K, L = BEST
for pol in ["redistribute", "substitute", "wait"]:
    for mode in ["open1", "close1", "close0"]:
        vals, cums = [], []
        for o in range(K):
            st = str(PD[base + o].date())
            nav, _ = run(d, N, K, L, mode=mode, policy=pol, cost=COST, start=st)
            vals.append(perf(nav)["夏普比率"]); cums.append(perf(nav)["累计收益"])
        rows.append({"执行时点": mode, "补仓策略": pol,
                     "相位期望夏普": np.mean(vals), "相位std": np.std(vals),
                     "相位期望累计": np.mean(cums)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 150)
print("【主推方案 分年度 / 子区间】")
print("=" * 150)
nav, s = run(d, N, K, L, mode="open1", policy="substitute", cost=COST)
sv = pd.Series(nav, index=d.dates)
r = sv.pct_change()
yr = r.groupby(r.index.year).apply(lambda g: (1 + g).prod() - 1)
navs = {"主推 open1/递补": sv}
for tag, mm, pp in [("close1/递补", "close1", "substitute"),
                    ("open1/权重分配", "open1", "redistribute"),
                    ("open1/等开板", "open1", "wait")]:
    nv, _ = run(d, N, K, L, mode=mm, policy=pp, cost=COST)
    navs[tag] = pd.Series(nv, index=d.dates)
Y = pd.DataFrame({k: v.pct_change().groupby(v.index.year).apply(lambda g: (1 + g).prod() - 1)
                  for k, v in navs.items()})
b = pd.read_csv(os.path.join(SRC, "data", "bench_all.csv"), index_col=0)
b.index = pd.to_datetime(b.index)
for c in ["沪深300", "创业板50", "科创50"]:
    s2 = b[c].dropna()
    Y[c] = s2.groupby(s2.index.year).apply(lambda g: g.iloc[-1] / g.iloc[0] - 1)
print((Y * 100).round(2).to_string())

print("\n【全期 & 子区间】")
rows = []
for k, v in navs.items():
    p = perf(v.values)
    for tag, d0 in [("2022起", "2022-01-04"), ("2024起", "2024-01-02")]:
        pass
    v22 = v[v.index >= pd.Timestamp("2022-01-04")].values
    v24 = v[v.index >= pd.Timestamp("2024-01-02")].values
    rows.append({"方案": k, "全期累计": p["累计收益"], "全期夏普": p["夏普比率"],
                 "全期回撤": p["最大回撤"],
                 "2022起累计": perf(v22 / v22[0])["累计收益"],
                 "2022起夏普": perf(v22 / v22[0])["夏普比率"],
                 "2024起累计": perf(v24 / v24[0])["累计收益"],
                 "2024起夏普": perf(v24 / v24[0])["夏普比率"]})
for c in ["沪深300", "创业板50", "科创50"]:
    s2 = b[c].dropna(); s2 = s2[s2.index >= d.dates[0]]
    p = perf((s2 / s2.iloc[0]).values)
    rows.append({"方案": c, "全期累计": p["累计收益"], "全期夏普": p["夏普比率"],
                 "全期回撤": p["最大回撤"],
                 "2022起累计": np.nan, "2022起夏普": np.nan,
                 "2024起累计": np.nan, "2024起夏普": np.nan})
print(pd.DataFrame(rows).round(4).to_string(index=False))

print("\n" + "=" * 150)
print("【成本敏感性（主推方案）】")
print("=" * 150)
rows = []
for c in [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003]:
    r2 = {"单边成本bp": c * 1e4}
    for mm in ["open1", "close1"]:
        p = perf(run(d, N, K, L, mode=mm, policy="substitute", cost=c)[0])
        r2[f"{mm}夏普"] = p["夏普比率"]; r2[f"{mm}累计"] = p["累计收益"]
    rows.append(r2)
print(pd.DataFrame(rows).round(4).to_string(index=False))

# 持仓集中度与阻断统计
print("\n" + "=" * 150)
print("【最后一个调仓日的持仓与集中度 / 涨停阻断统计】")
print("=" * 150)
nav, s = run(d, N, K, L, mode="open1", policy="substitute", cost=COST)
print(f"目标槽位 {s['target']}，涨停阻断 {s['blocked']}（{s['blocked']/max(s['target'],1):.2%}）"
      f"｜全天封死 {s['blocked_all']}｜盘中/尾盘封板 {s['blocked_late']}")
print(f"递补成功 {s['sub_used']} 次｜平均持仓 {s['avg_n']/max(s['n_period'],1):.1f} 只"
      f"｜年均单边换手 {s['turnover']/max(s['n_period'],1)*(252.0/K):.1f} 倍")

# 保存净值
pd.DataFrame({k: v for k, v in navs.items()}).to_csv(
    os.path.join(OUT, "final_nav.csv"), encoding="utf-8-sig")
print(f"\n净值已保存 → {os.path.join(OUT, 'final_nav.csv')}")
