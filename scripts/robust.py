# -*- coding: utf-8 -*-
"""
稳健性检验：
  1) 样本内/样本外切分（IS 2019-05→2023-12，OOS 2024-01→今）
  2) 三种补仓策略两两对比（同一参数下谁更好、胜率多少）
  3) 执行时点对比（T+1开盘 / T+1收盘 / T日收盘）
  4) 参数高原：夏普 Top 区域的邻域稳定性
  5) 滚动前进（walk-forward）：每年用过去 3 年选参，下一年实盘
"""
import sys, os, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from engine import Data, run, perf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "output")
COST = 0.001
IS_END = "2023-12-31"
OOS_START = "2024-01-02"

d = Data()
grid = pd.read_csv(os.path.join(OUT, "grid_full.csv"))
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 50)


def sharpe_window(nav, d0, d1):
    s = pd.Series(nav, index=d.dates)
    m = (s.index >= pd.Timestamp(d0)) & (s.index <= pd.Timestamp(d1))
    v = s[m].values
    if len(v) < 50:
        return np.nan
    return perf(v / v[0])["夏普比率"]


# ---------------------------------------------------------------- 1) IS/OOS
print("=" * 100)
print("【1】样本内 / 样本外切分（IS 2019-05-06→2023-12-31，OOS 2024-01-02→2026-10-08）")
print("=" * 100)
# 先按 IS 夏普选参（分执行时点×补仓策略），再看 OOS 表现
res = []
for mode in ["open1", "close1", "close0"]:
    for pol in ["redistribute", "substitute", "wait"]:
        best = None
        for N in [5, 10, 15, 20, 30]:
            for K in [5, 8, 10, 12, 15, 20, 25]:
                for L in [20, 25, 30, 40, 50]:
                    nav, st = run(d, N, K, L, mode=mode, policy=pol, cost=COST)
                    s_is = sharpe_window(nav, "2019-05-06", IS_END)
                    if best is None or s_is > best[0]:
                        best = (s_is, N, K, L, nav)
        s_is, N, K, L, nav = best
        s_oos = sharpe_window(nav, OOS_START, d.dates[-1].strftime("%Y-%m-%d"))
        p_all, p_oos = perf(nav), perf(
            pd.Series(nav, index=d.dates)[d.dates >= pd.Timestamp(OOS_START)].values /
            pd.Series(nav, index=d.dates)[d.dates >= pd.Timestamp(OOS_START)].values[0])
        res.append({"执行时点": mode, "补仓策略": pol, "N": N, "K": K, "L": L,
                    "IS夏普": s_is, "OOS夏普": s_oos,
                    "OOS累计": p_oos["累计收益"], "OOS回撤": p_oos["最大回撤"],
                    "全期夏普": p_all["夏普比率"]})
r1 = pd.DataFrame(res).sort_values("OOS夏普", ascending=False)
print(r1.to_string(index=False))

# ------------------------------------------------- 2) 补仓策略两两对比
print("\n" + "=" * 100)
print("【2】三种补仓策略两两对比（同参数、同执行时点，全期夏普）")
print("=" * 100)
combos = [(N, K, L) for N in [5, 10, 15, 20] for K in [8, 10, 12, 15, 20]
          for L in [20, 25, 30, 40, 50]]
for mode in ["open1", "close1", "close0"]:
    tbl = {}
    for pol in ["redistribute", "substitute", "wait"]:
        tbl[pol] = [perf(run(d, N, K, L, mode=mode, policy=pol, cost=COST)[0])["夏普比率"]
                    for N, K, L in combos]
    t = pd.DataFrame(tbl)
    print(f"\n-- {mode}（{len(combos)} 组参数）平均夏普 --")
    print(t.mean().round(4).to_string())
    for a, b in [("redistribute", "substitute"), ("redistribute", "wait"),
                 ("substitute", "wait")]:
        diff = t[a] - t[b]
        print(f"   {a:<13} vs {b:<13} 平均差 {diff.mean():+.4f}  胜率 {(diff>0).mean():.1%}")

# ------------------------------------------------- 3) 执行时点对比
print("\n" + "=" * 100)
print("【3】执行时点对比（同参数、同补仓策略）")
print("=" * 100)
rows = []
for pol in ["redistribute", "substitute", "wait"]:
    vals = {}
    for mode in ["open1", "close1", "close0"]:
        vals[mode] = [perf(run(d, N, K, L, mode=mode, policy=pol, cost=COST)[0])["夏普比率"]
                      for N, K, L in combos]
    t = pd.DataFrame(vals)
    rows.append({"补仓策略": pol, **{f"{m}均夏普": t[m].mean() for m in t.columns}})
    for a, b in [("open1", "close1"), ("open1", "close0")]:
        diff = t[a] - t[b]
        rows[-1][f"{a}>{b}胜率"] = f"{(diff>0).mean():.1%}"
print(pd.DataFrame(rows).to_string(index=False))

# ------------------------------------------------- 4) 参数高原
print("\n" + "=" * 100)
print("【4】夏普 Top20 参数分布（是否集中在某一片区域）")
print("=" * 100)
top = grid.sort_values("夏普比率", ascending=False).head(20)
print(top[["N", "K", "L", "执行时点", "补仓策略", "夏普比率", "年化收益", "最大回撤"]]
      .to_string(index=False))
print("\nTop50 中各参数取值频次：")
t50 = grid.sort_values("夏普比率", ascending=False).head(50)
for col in ["N", "K", "L", "执行时点", "补仓策略"]:
    print(f"  {col}: {dict(t50[col].value_counts())}")

# ------------------------------------------------- 5) 滚动前进
print("\n" + "=" * 100)
print("【5】滚动前进（每年末用过去 3 年数据选夏普最优参数，次年按此参数交易）")
print("=" * 100)
NS = [5, 10, 15, 20, 30]
KS = [5, 8, 10, 12, 15, 20, 25]
LS = [20, 25, 30, 40, 50]
years = [2022, 2023, 2024, 2025]
for mode in ["open1", "close1"]:
    for pol in ["redistribute", "wait"]:
        nav_wf = None
        picks = []
        seg_nav = []
        for y in years:
            is0 = f"{y-3}-01-01"
            is1 = f"{y-1}-12-31"
            best = None
            for N in NS:
                for K in KS:
                    for L in LS:
                        nav, _ = run(d, N, K, L, mode=mode, policy=pol, cost=COST)
                        s = sharpe_window(nav, is0, is1)
                        if best is None or s > best[0]:
                            best = (s, N, K, L, nav)
            s, N, K, L, nav = best
            picks.append((y, N, K, L, round(s, 3)))
            seg_nav.append(pd.Series(nav, index=d.dates)[
                (d.dates >= pd.Timestamp(f"{y}-01-01")) &
                (d.dates <= pd.Timestamp(f"{y}-12-31"))])
        cat = pd.concat(seg_nav)
        v = (cat / cat.iloc[0]).values
        p = perf(v)
        print(f"{mode:>6} {pol:<13} 选参 {picks} → OOS(2022-01→今) 累计 {p['累计收益']:.2%} "
              f"年化 {p['年化收益']:.2%} 夏普 {p['夏普比率']:.3f} 回撤 {p['最大回撤']:.2%}")
