# -*- coding: utf-8 -*-
"""最终方案详解：分年度、基准对比、涨停阻断统计、建议方案的完整绩效。"""
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
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 60)


def yearly(nav):
    s = pd.Series(nav, index=d.dates)
    r = s.pct_change()
    out = []
    for y, g in r.groupby(r.index.year):
        v = (1 + g).prod() - 1
        out.append({"年份": y, "收益": v})
    return pd.DataFrame(out)


def bench():
    b = pd.read_csv(os.path.join(SRC, "data", "bench_all.csv"), index_col=0)
    b.index = pd.to_datetime(b.index)
    return b


def report(N, K, L, mode, pol, tag=""):
    nav, st = run(d, N, K, L, mode=mode, policy=pol, cost=COST)
    p = perf(nav, tag or f"N{N}K{K}L{L}")
    p["调仓次数"] = st["n_period"]
    p["阻断率"] = st["blocked"] / max(st["target"], 1)
    p["阻断_全天"] = st["blocked_all"]
    p["阻断_尾盘"] = st["blocked_late"]
    p["年均单边换手"] = st["turnover"] / max(st["n_period"], 1) * (252.0 / K)
    p["年均成本拖累"] = st["cost"] / max(st["n_period"], 1) * (252.0 / K)
    p["平均持仓只数"] = st["avg_n"] / max(st["n_period"], 1)
    return nav, p, st


if __name__ == "__main__":
    cfgs = json.load(open(os.path.join(OUT, "best_config.json"), encoding="utf-8"))
    print("=" * 110)
    print("【最终推荐方案】")
    print("=" * 110)
    rows = []
    navs = {}
    for c in cfgs:
        nav, p, st = report(c["N"], c["K"], c["L"], c["mode"], c["policy"], c.get("tag"))
        rows.append(p)
        navs[c.get("tag") or f"{c['mode']}/{c['policy']}"] = nav
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    print("\n【分年度收益】")
    ys = {}
    for k, v in navs.items():
        ys[k] = yearly(v).set_index("年份")["收益"]
    b = bench()
    for c in ["沪深300", "创业板50", "科创50"]:
        if c in b.columns:
            s = b[c].dropna()
            ys[c] = s.groupby(s.index.year).apply(lambda g: g.iloc[-1] / g.iloc[0] - 1)
    Y = pd.DataFrame(ys)
    print((Y * 100).round(2).to_string())

    print("\n【基准对比（全期）】")
    brows = []
    for k, v in navs.items():
        brows.append(perf(v, k))
    for c in ["沪深300", "创业板50", "科创50"]:
        if c in b.columns:
            s = b[c].dropna()
            s = s[s.index >= d.dates[0]]
            brows.append(perf((s / s.iloc[0]).values, c))
    print(pd.DataFrame(brows)[["组合", "累计收益", "年化收益", "年化波动",
                               "最大回撤", "夏普比率", "卡玛比率"]].to_string(index=False))

    # 涨停阻断统计
    print("\n【涨停阻断事件统计（推荐方案）】")
    c0 = cfgs[0]
    nav, p, st = report(c0["N"], c0["K"], c0["L"], c0["mode"], c0["policy"])
    print(f"  目标持仓槽位 {st['target']}，被涨停阻断 {st['blocked']} 次 "
          f"（{st['blocked']/max(st['target'],1):.2%}）")
    print(f"  ├ 全天封死（一字/全天未开板）：{st['blocked_all']} 次")
    print(f"  └ 盘中/尾盘封板至收盘　　　：{st['blocked_late']} 次")
    print(f"  等待开板补买成功 {st['wait_filled']} 次 / 等到期末仍未开板 {st['wait_failed']} 次")
    print(f"  递补次数 {st['sub_used']}，平均现金仓位 {st['cash_w']/max(st['n_period'],1):.4f}")

    # 全池涨停背景频率
    n = d.T * d.C
    print(f"\n【全标的池涨停背景频率（2019-01-02→{d.dates[-1].date()}）】")
    print(f"  收盘封板 {d.seal_close.sum()} 次（{d.seal_close.mean():.3%}）"
          f"｜全天封死 {d.seal_all.sum()}｜尾盘/盘中封板 {d.seal_late.sum()}"
          f"｜触板未封 {d.touch.sum()}｜跌停 {d.limit_dn.sum()}")

    # 保存净值
    out = {}
    for k, v in navs.items():
        out[k] = pd.Series(v, index=d.dates)
    pd.DataFrame(out).to_csv(os.path.join(OUT, "final_nav.csv"), encoding="utf-8-sig")
    print(f"\n净值已保存 → {os.path.join(OUT,'final_nav.csv')}")
