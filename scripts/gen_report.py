# -*- coding: utf-8 -*-
"""生成《动量轮动策略 · 参数重构回测》HTML 报告（自包含，无外链资源）。"""
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

b = pd.read_csv(os.path.join(SRC, "data", "bench_all.csv"), index_col=0)
b.index = pd.to_datetime(b.index)

REC = (15, 10, 50, "open1", "substitute")     # 推荐方案


def sv(nav):
    return pd.Series(nav, index=d.dates)


# ---------------------------------------------------------------- 方案对照
CFG = [
    ("① 推荐 N=15 / K=10 / L=50 · T+1开盘 · 递补", 15, 10, 50, "open1", "substitute", 0.001, None, False),
    ("② 推荐 N=15 / K=10 / L=60 · T+1开盘 · 递补", 15, 10, 60, "open1", "substitute", 0.001, None, False),
    ("③ 同参数但 T+1 收盘执行", 15, 10, 50, "close1", "substitute", 0.001, None, False),
    ("④ 现行 N=10/K=5/L=12 + 行业≤2 · 开盘", 10, 5, 12, "open1", "substitute", 0.001, 2, False),
    ("⑤ 现行参数 · T+1 收盘", 10, 5, 12, "close1", "substitute", 0.001, 2, False),
    ("⑥ 已发布口径（不计成本、不管涨停）", 10, 5, 12, "close0", "redistribute", 0.0, 2, True),
    ("⑦ 已发布口径 + 计成本 10bp", 10, 5, 12, "close0", "redistribute", 0.001, 2, True),
    ("⑧ 已发布口径 + 计成本 + 管涨停", 10, 5, 12, "close0", "redistribute", 0.001, 2, False),
]
rows, yrs, nvs = [], {}, {}
for tag, N, K, L, mode, pol, cost, cap, il in CFG:
    nav, s = run(d, N, K, L, mode=mode, policy=pol, cost=cost, cap=cap, ignore_limit=il)
    p = perf(nav, tag)
    v = sv(nav)
    yrs[tag] = v.pct_change().groupby(v.index.year).apply(lambda g: (1 + g).prod() - 1)
    for lab, d0 in [("2022起", "2022-01-04"), ("2024起", "2024-01-02")]:
        vv = v[v.index >= pd.Timestamp(d0)].values
        q = perf(vv / vv[0])
        p[lab + "累计"] = q["累计收益"]; p[lab + "夏普"] = q["夏普比率"]
    p["年均换手"] = s["turnover"] / max(s["n_period"], 1) * (252.0 / K)
    p["阻断率"] = s["blocked"] / max(s["target"], 1)
    p["平均持仓"] = s["avg_n"] / max(s["n_period"], 1)
    rows.append(p); nvs[tag] = v
SUM = pd.DataFrame(rows)
Y = pd.DataFrame(yrs)
for c in ["沪深300", "创业板50", "科创50"]:
    s2 = b[c].dropna()
    Y[c] = s2.groupby(s2.index.year).apply(lambda g: g.iloc[-1] / g.iloc[0] - 1)
    Yc = (s2 / s2.iloc[0])
    Yc = Yc[Yc.index >= d.dates[0]]
    nvs[c] = Yc

# ---------------------------------------------------------------- 网格结果
G = pd.read_csv(os.path.join(OUT, "final_phase_grid.csv"))
TOP = G.sort_values("相位期望夏普", ascending=False).head(12)
prof_dim = {}
for dim in ["N", "K", "L", "执行时点", "补仓策略"]:
    prof_dim[dim] = G.groupby(dim)["相位期望夏普"].mean().round(4).to_dict()
cross = G.pivot_table(index="执行时点", columns="补仓策略", values="相位期望夏普",
                      aggfunc="mean").round(4)
prof_k = G.pivot_table(index="K", columns="执行时点", values="相位期望夏普",
                       aggfunc="mean").round(4)
prof_l = G.pivot_table(index="L", columns="执行时点", values="相位期望夏普",
                       aggfunc="mean").round(4)

# 成本敏感性
cost_rows = []
for c in [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003]:
    r = {"单边成本bp": int(c * 1e4)}
    for mm in ["open1", "close1"]:
        p = perf(run(d, *REC[:3], mode=mm, policy=REC[4], cost=c)[0])
        r[f"{mm}夏普"] = round(p["夏普比率"], 4); r[f"{mm}累计"] = round(p["累计收益"], 4)
    cost_rows.append(r)
COSTS = pd.DataFrame(cost_rows)

# 涨停统计
N, K, L, mode, pol = REC
nav_r, st_r = run(d, N, K, L, mode=mode, policy=pol, cost=COST)
stat = {
    "target": st_r["target"], "blocked": st_r["blocked"],
    "rate": st_r["blocked"] / max(st_r["target"], 1),
    "all": st_r["blocked_all"], "late": st_r["blocked_late"],
    "sub": st_r["sub_used"],
    "pool_seal_close": int(d.seal_close.sum()), "pool_seal_all": int(d.seal_all.sum()),
    "pool_seal_late": int(d.seal_late.sum()), "pool_touch": int(d.touch.sum()),
    "pool_dn": int(d.limit_dn.sum()), "cells": int(d.T * d.C),
}
# 收盘执行下的阻断率（同参数）
_, st_c = run(d, N, K, L, mode="close1", policy=pol, cost=COST)
stat["rate_close"] = st_c["blocked"] / max(st_c["target"], 1)
_, st_c0 = run(d, N, K, L, mode="close0", policy=pol, cost=COST)
stat["rate_close0"] = st_c0["blocked"] / max(st_c0["target"], 1)

# ---------------------------------------------------------------- SVG 曲线
def svg_chart(series, title, w=1180, h=420):
    keys = list(series.keys())
    colors = ["#c0392b", "#1e6fd9", "#e08a00", "#7a7a7a", "#2b8a3e", "#8e44ad", "#00a2a2", "#b8860b"]
    idx = series[keys[0]].index
    data = {k: series[k].reindex(idx).ffill().values for k in keys}
    allv = np.concatenate([v[np.isfinite(v)] for v in data.values()])
    lo, hi = float(np.nanmin(allv)), float(np.nanmax(allv))
    lo = min(lo, 0.95)
    pad_l, pad_r, pad_t, pad_b = 78, 190, 34, 46
    x0, x1 = pad_l, w - pad_r
    y0, y1 = h - pad_b, pad_t
    n = len(idx)
    def px(i): return x0 + (x1 - x0) * i / max(n - 1, 1)
    def py(v): return y1 + (y0 - y1) * (np.log(max(v, 1e-6)) - np.log(lo)) / (np.log(hi) - np.log(lo))
    parts = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto">']
    parts.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff"/>')
    # y 轴刻度
    ticks = []
    v = 1.0
    while v <= hi * 1.05:
        ticks.append(v); v *= 2
    ticks += [x for x in [1.5, 3, 6, 12, 25] if lo < x < hi]
    ticks = sorted(set([round(t, 3) for t in ticks if lo * 0.98 <= t <= hi * 1.05]))
    for t in ticks:
        yy = py(t)
        parts.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="#eceff3" stroke-width="1"/>')
        parts.append(f'<text x="{x0-8}" y="{yy+4:.1f}" text-anchor="end" font-size="12" fill="#6b7280">{t:g}</text>')
    # x 轴年份
    years = sorted(set(idx.year))
    for y in years:
        i = int(np.argmax(idx.year == y))
        xx = px(i)
        parts.append(f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y0+5}" stroke="#c8cdd4"/>')
        parts.append(f'<text x="{xx:.1f}" y="{y0+20}" text-anchor="middle" font-size="12" fill="#6b7280">{y}</text>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#c8cdd4"/>')
    for ci, k in enumerate(keys):
        v = data[k]
        pts = " ".join(f"{px(i):.1f},{py(v[i]):.1f}" for i in range(n) if np.isfinite(v[i]))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{colors[ci%len(colors)]}" '
                     f'stroke-width="1.9" stroke-linejoin="round"/>')
        ly = 30 + ci * 22
        parts.append(f'<line x1="{x1+18}" y1="{ly}" x2="{x1+42}" y2="{ly}" stroke="{colors[ci%len(colors)]}" stroke-width="2.6"/>')
        lab = k if len(k) <= 26 else k[:25] + "…"
        parts.append(f'<text x="{x1+48}" y="{ly+4}" font-size="12.5" fill="#374151">{lab}</text>')
    parts.append(f'<text x="{x0}" y="{pad_t-14}" font-size="13" font-weight="600" fill="#111827">{title}</text>')
    parts.append('</svg>')
    return "\n".join(parts)


chart_keys = [CFG[0][0], CFG[7][0], CFG[5][0], "沪深300", "创业板50"]
chart1 = svg_chart({k: nvs[k] for k in chart_keys},
                   "净值曲线（对数轴，2019-05-06 = 1，后复权）")

main_keys = [CFG[0][0], CFG[2][0], CFG[4][0]]
chart2 = svg_chart({k: nvs[k] for k in main_keys},
                   "同参数下：T+1 开盘 vs T+1 收盘 vs 现行参数（收盘）")


# ---------------------------------------------------------------- HTML
def tbl(df, fmts=None, cls=""):
    fmts = fmts or {}
    h = "<table class='%s'><thead><tr>" % cls
    for c in df.columns:
        h += f"<th>{c}</th>"
    h += "</tr></thead><tbody>"
    for _, r in df.iterrows():
        h += "<tr>"
        for c in df.columns:
            v = r[c]
            if c in fmts:
                h += f"<td>{fmts[c](v)}</td>"
            elif isinstance(v, (float, np.floating)):
                h += f"<td>{v:,.4f}</td>" if abs(v) < 1000 else f"<td>{v:,.2f}</td>"
            else:
                h += f"<td>{v}</td>"
        h += "</tr>"
    return h + "</tbody></table>"


pc = lambda v: f"{v*100:.2f}%"
p1 = lambda v: f"{v*100:.1f}%"
f2 = lambda v: f"{v:.2f}"
f3 = lambda v: f"{v:.3f}"

sum_show = SUM[["组合", "累计收益", "年化收益", "年化波动", "最大回撤", "夏普比率", "卡玛比率",
                "2022起累计", "2022起夏普", "2024起累计", "2024起夏普", "年均换手", "阻断率"]].copy()
fm = {c: pc for c in ["累计收益", "年化收益", "年化波动", "最大回撤", "2022起累计", "2024起累计", "阻断率"]}
fm.update({c: f3 for c in ["夏普比率", "卡玛比率", "2022起夏普", "2024起夏普"]})
fm["年均换手"] = lambda v: f"{v:.1f}×"

top_show = TOP[["N", "K", "L", "执行时点", "补仓策略", "相位期望夏普", "相位标准差",
                "最差相位夏普", "相位期望累计", "年均换手", "阻断率"]]
fm2 = {"相位期望夏普": f3, "相位标准差": f3, "最差相位夏普": f3,
       "相位期望累计": pc, "年均换手": lambda v: f"{v:.1f}×", "阻断率": pc}

fm3 = {c: f3 for c in ["close0", "close1", "open1"]}
t_k = tbl(prof_k.reset_index().rename(columns={"K": "调仓间隔 K"}), fm3)
t_l = tbl(prof_l.reset_index().rename(columns={"L": "持仓数 L"}), fm3)
t_cross = tbl(cross.reset_index(), {c: f3 for c in cross.columns})
t_cost = tbl(COSTS.rename(columns={"open1夏普": "开盘执行夏普", "open1累计": "开盘执行累计",
                                   "close1夏普": "收盘执行夏普", "close1累计": "收盘执行累计"}),
             {"开盘执行夏普": f3, "收盘执行夏普": f3, "开盘执行累计": pc, "收盘执行累计": pc})
t_year = tbl(Y.reset_index().rename(columns={"index": "年份"}), {c: pc for c in Y.columns})
t_sum = tbl(sum_show, fm)
t_top = tbl(top_show, fm2)


def prof_table(mapping, label):
    ks = list(mapping.keys()); vs = list(mapping.values())
    h = "<table><tr><th>%s</th>" % label + "".join(f"<td>{k}</td>" for k in ks) + "</tr>"
    h += "<tr><th>相位期望夏普</th>" + "".join(f"<td>{v:.4f}</td>" for v in vs) + "</tr></table>"
    return h


t_pn = prof_table(prof_dim["N"], "动量周期 N")
t_pk = prof_table(prof_dim["K"], "调仓间隔 K")
t_pl = prof_table(prof_dim["L"], "持仓数 L")

HTML = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>动量轮动策略 · 参数重构回测</title>
<style>
:root{{--bg:#f6f7f9;--card:#fff;--bd:#e4e7ec;--tx:#1f2937;--mu:#6b7280;--ac:#c0392b;--ok:#0f7b46}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--tx);
 font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
 font-size:14.5px;line-height:1.75}}
.wrap{{max-width:1240px;margin:0 auto;padding:28px 20px 64px}}
h1{{font-size:25px;margin:0 0 6px}}
.sub{{color:var(--mu);font-size:13.5px;margin-bottom:22px}}
h2{{font-size:19px;margin:34px 0 12px;padding-left:11px;border-left:4px solid var(--ac)}}
h3{{font-size:15.5px;margin:20px 0 8px;color:#374151}}
.card{{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:18px 20px;margin:14px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}}
th,td{{border-bottom:1px solid var(--bd);padding:7px 9px;text-align:right;white-space:nowrap}}
th{{background:#f3f5f8;font-weight:600;color:#374151;text-align:right;position:sticky;top:0}}
td:first-child,th:first-child{{text-align:left;white-space:normal;min-width:180px}}
tbody tr:hover{{background:#fafbfc}}
.kpi{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:14px 0}}
.kpi div{{background:#fff;border:1px solid var(--bd);border-radius:10px;padding:12px 14px}}
.kpi b{{display:block;font-size:21px;color:var(--ac);line-height:1.3}}
.kpi span{{font-size:12px;color:var(--mu)}}
.ans{{background:#fff;border:1px solid var(--bd);border-left:4px solid var(--ok);
 border-radius:10px;padding:14px 18px;margin:12px 0}}
.ans h4{{margin:0 0 6px;font-size:15px;color:var(--ok)}}
.warn{{border-left-color:#d97706}} .warn h4{{color:#b45309}}
.note{{color:var(--mu);font-size:12.8px;margin-top:8px}}
.code{{font-family:ui-monospace,Consolas,monospace;background:#f3f4f6;padding:1px 5px;border-radius:4px;font-size:12.5px}}
ul{{margin:8px 0 8px 20px;padding:0}} li{{margin:5px 0}}
.pill{{display:inline-block;background:#fdecea;color:#a93226;border-radius:999px;
 padding:2px 10px;font-size:12.5px;margin-right:6px}}
.scroll{{overflow-x:auto}}
</style></head><body><div class="wrap">

<h1>动量轮动策略 · 参数重构回测</h1>
<div class="sub">
标的池：舞弊调整质量因子季度动态股池（2019Q1–2026H1，23 期，384–445 只/期）｜
区间：2019-05-06 → 2026-10-08（1881 个交易日）｜
评分与次新过滤口径不变（上市未满 60 个交易日不入池）<br>
本次变更：<b>取消「单个申万三级行业最多 2 只」限制</b>；新增 <b>涨停无法建仓</b> 约束与 <b>单边 10bp 交易成本</b>；
执行时点由「T 日收盘」改为可比较的三种口径；所有参数用 <b>调仓相位平均</b> 寻优以抑制日历对齐过拟合。
</div>

<h2>一、结论速览</h2>
<div class="kpi">
 <div><b>N = 15</b><span>动量周期（交易日）</span></div>
 <div><b>K = 10</b><span>调仓间隔（交易日）</span></div>
 <div><b>L = 50</b><span>持仓只数（50–60 均可）</span></div>
 <div><b>T+1 开盘</b><span>执行时点</span></div>
 <div><b>依次递补</b><span>涨停阻断处理</span></div>
 <div><b>1.02</b><span>相位期望夏普（10bp 成本）</span></div>
</div>

<div class="ans">
<h4>问题 1：动量周期 N、调仓间隔 K、持仓 L 取多少？</h4>
<p><b>N = 15 个交易日、K = 10 个交易日、L = 50 只（50–60 只均为高原区）。</b>
稳健取值区间：N ∈ [10, 15]、K ∈ [8, 12]、L ∈ [50, 60]；
这三个区间内夏普差异都在 0.01 以内，属于同一片高地，不必卡死在单点上。
相比现行 N=10/K=5/L=12，本质是「<b>把动量看短一点、调仓放慢一点、持仓铺宽一点</b>」：
更宽的持仓稀释了风险调整动量（adj）归一化后的极端集中，更慢的调仓把年换手从 83× 降到 38×。</p>
</div>

<div class="ans">
<h4>问题 2：全天涨停 / 尾盘涨停买不进去，权重给别的股、依次递补、还是等打开？</h4>
<p><b>选「依次递补」</b>——按动量排名往后取下一只可成交标的补上空缺。
全网格（540 组参数 × 3 种执行时点）平均相位期望夏普：
递补 <b>0.875</b> ＞ 权重分配 0.871 ＞ 等开板 <b>0.848</b>。
<p>
<ul>
<li><b>等开板最差</b>，且在收盘执行下差距被放大：收盘/收盘(T日) 口径下比递补低 <b>0.026 / 0.038</b> 夏普
（100 组参数对比中 85% / 72% 的组合都更差）。原因：资金被锁在现金里错过建仓时点，
而涨停打开的股票短期多为「开板即回落」，等到的价格往往更贵。</li>
<li><b>权重分配</b>与递补差距极小（约 0.004 夏普），实现更简单，是次优但完全可接受的方案。</li>
<li><b>但更根本的解法是换成 T+1 开盘执行</b>：开盘执行的涨停阻断率只有
<b>{stat['rate']:.2%}</b>，而 T+1 收盘是 <b>{stat['rate_close']:.2%}</b>、T 日收盘是 <b>{stat['rate_close0']:.2%}</b>。
开盘模式下三种补仓策略的夏普差距被压到 0.008 以内（0.915–0.923），这个问题基本消失。</li>
</ul>
</p></div>

<div class="ans">
<h4>问题 3：T+1 开盘买，还是收盘买？</h4>
<p><b>T+1 开盘买，夏普明显更高。</b>相位期望夏普：开盘 <b>0.920</b> ＞ T+1 收盘 0.848 ＞ T 日收盘 0.827；
而且在 N、K、L 的<b>每一个分层</b>上开盘都一致占优，不是个别参数的偶然。
推荐参数下具体为：开盘 1.170 vs 收盘 1.008（累计 570.9% vs 388.3%）。
<p>
两条真实原因：<br>
① <b>延迟成本</b>：信号在 T 收盘产生，T+1 开盘就能吃到当日盘中那段动量延续，等到收盘才买等于白扔一天；<br>
② <b>可建仓性</b>：强势股 T+1 常在盘中封板，开盘买得进去，收盘买不进去（阻断率 0.27% vs 1.76%）。</p>
</p></div>

<div class="ans warn">
<h4>必须同时说明的一件事：已发布的 +553% 曲线不可直接实现</h4>
<p>现行参数（N=10/K=5/L=12 + 行业≤2）在「不计交易成本、且允许买入当日收盘涨停的股票」下是
<b>{SUM.loc[5,'累计收益']*100:.1f}%</b>（与线上已发布口径一致）。
但一旦同时计入 10bp 单边成本与涨停不可建仓，同一套参数只剩
<b>{SUM.loc[7,'累计收益']*100:.1f}%</b>，夏普 <b>{SUM.loc[7,'夏普比率']:.2f}</b>。
也就是说，线上曲线里约 <b>八成</b> 的收益来自两个回测里没有建模的假设。
本次重构的核心价值，是把这两个漏洞补上之后再重新找参数。</p>
</div>

<h2>二、方案对照</h2>
<div class="card scroll">{t_sum}</div>
<div class="note">⑥ = 项目现行已发布口径的复现（零成本 + 不管涨停）；⑦ 在其上只加成本；⑧ 再叠加涨停约束。
① 为本次推荐方案。所有方案均含次新股过滤，成本口径为单边 10bp（双边按换手额计收）。</div>

<div class="card">{chart1}</div>

<h2>三、为什么是这样一组参数（相位平均寻优）</h2>
<p>直接用单一回测起点选参数会踩到<b>日历对齐陷阱</b>：首轮网格里 N=10/K=12 的夏普高达 1.27，但把回测起点在调仓网格上平移
1–5 天后，它在 6 个相位上的夏普是 <span class="code">1.27 / 1.22 / 1.03 / 0.68 / 0.65 / 0.96</span>（标准差 0.24，最差相位只有 0.65）——
尖峰完全来自「恰好从哪天开始调仓」。因此最终寻优对每个 K 在<b>全部 K 个相位</b>上跑一遍取平均，
共 12,600 次回测。</p>

<h3>边际剖面（相位期望夏普，其它维度取平均）</h3>
<div class="card scroll">
{t_pn}
<div style="height:16px"></div>
{t_pk}
<div style="height:16px"></div>
{t_pl}
</div>
<div class="note">N 在 10–15 之间见顶（N=5 明显差：0.60–0.69）；K 从 8 起进入高原，10–15 最佳，K=5 最差（0.70）；
L 单调上升并在 50 以后走平，L=20 只有 0.79。</div>

<h3>相位期望夏普 Top 12</h3>
<div class="card scroll">{t_top}</div>

<h2>四、执行时点：开盘 vs 收盘</h2>
<div class="card scroll">
{t_k}
<div class="note">按 K 分层（对 N / L / 补仓策略取平均）。<b>open1 在每个 K 上都最高。</b></div>
</div>
<div class="card scroll">
{t_l}
<div class="note">按 L 分层，同样 open1 全段占优。</div>
</div>
<div class="card">{chart2}</div>

<h2>五、涨停阻断：发生了多少次、怎么处理</h2>
<div class="card">
<p><b>全池背景频率</b>（560 只 × 1881 天 = {stat['cells']:,} 个标的-日）：
收盘封板 <b>{stat['pool_seal_close']:,}</b> 次（{stat['pool_seal_close']/stat['cells']:.3%}），
其中全天封死（一字/全天未开板）<b>{stat['pool_seal_all']:,}</b> 次、
盘中或尾盘封板至收盘 <b>{stat['pool_seal_late']:,}</b> 次；
触板未封 <b>{stat['pool_touch']:,}</b> 次；跌停 <b>{stat['pool_dn']:,}</b> 次。</p>
<p><b>推荐方案实际被阻断</b>：目标槽位 {stat['target']:,} 个，涨停无法建仓 {stat['blocked']} 次
（<b>{stat['rate']:.2%}</b>，其中全天封死 {stat['all']} 次、盘中/尾盘封板 {stat['late']} 次），
全部由递补填补（{stat['sub']} 次）。同参数下 T+1 收盘执行阻断率 <b>{stat['rate_close']:.2%}</b>、
T 日收盘 <b>{stat['rate_close0']:.2%}</b>——这正是「开盘执行」最重要的附带好处。</p>
<div class="note">判定在原始价格空间做：涨停价 = 前收盘 ×(1+ 涨跌幅限制)，主板 10%，
创业板 2020-08-24 起 20%（之前 10%），科创板 20%。
全天封死 = 当日最低价 ≥ 涨停价；盘中/尾盘封板 = 收盘封板但最低价曾低于涨停价；触板未封 = 最高价触及涨停但收盘未封住。</div>
</div>

<h3>补仓策略 × 执行时点（相位期望夏普，对 N/K/L 取平均）</h3>
<div class="card scroll">{t_cross}</div>

<h2>六、推荐方案详解（N=15 / K=10 / L=50 · T+1 开盘 · 依次递补）</h2>
<div class="kpi">
 <div><b>{SUM.loc[0,'累计收益']*100:.0f}%</b><span>累计收益（单相位）</span></div>
 <div><b>{SUM.loc[0,'年化收益']*100:.1f}%</b><span>年化收益</span></div>
 <div><b>{SUM.loc[0,'年化波动']*100:.1f}%</b><span>年化波动</span></div>
 <div><b>{SUM.loc[0,'最大回撤']*100:.1f}%</b><span>最大回撤</span></div>
 <div><b>{SUM.loc[0,'夏普比率']:.3f}</b><span>夏普（单相位 / 相位期望 1.015）</span></div>
 <div><b>{SUM.loc[0,'年均换手']:.0f}×</b><span>年均单边换手</span></div>
</div>
<h3>分年度收益</h3>
<div class="card scroll">{t_year}</div>
<h3>成本敏感性</h3>
<div class="card scroll">{t_cost}</div>
<div class="note">策略年换手约 38×，对成本敏感：单边 20bp 时夏普降到约 0.86，30bp 时约 0.54。
若实际冲击成本高于 10bp，建议把 K 放宽到 12–15（换手降到 34×）而不是继续缩短。</div>

<h2>七、风险提示与执行要点</h2>
<div class="card">
<ul>
<li><b>样本外折价要预期。</b>滚动前进检验（每年末用过去 3 年选参、次年交易）2022–2026 的样本外夏普只有
0.36–0.40，远低于全样本最优的 1.0+。推荐参数下的 2022 年以来夏普为
{SUM.loc[0,'2022起夏普']:.2f}、2024 年以来 {SUM.loc[0,'2024起夏普']:.2f}，
说明近两年表现好于滚动选参的结果，但仍应把「未来夏普落在 0.6–1.0」当作基准预期，而不是 1.2。</li>
<li><b>L=50 意味着约 50 只持仓</b>，单票权重按 adj 归一化后仍会向头部集中，需要一定资金规模才能
降低冲击成本与零股误差。</li>
<li><b>回测未按跌停建模卖出端。</b>全池跌停 {stat['pool_dn']:,} 次（{stat['pool_dn']/stat['cells']:.3%}），
调仓日撞上跌停无法卖出的概率很低，但极端行情下会低估流动性风险。</li>
<li><b>开盘执行的操作含义</b>：T 日收盘算信号，T+1 集合竞价下单。开盘价用通达信原始日 K 的 open，
后复权开盘价由 <span class="code">open_raw × (close_hfq / close_raw)</span> 逐日构造，
与既有后复权收盘价同口径。</li>
<li>本次结果<b>未覆盖</b>ST 股 5% 涨跌幅、盘中临停、以及一字板当日排队能否成交的细节；
标的池为质量因子选出的股票，ST 与临停占比极低，影响可忽略。</li>
</ul>
</div>

<div class="note" style="margin-top:26px">
数据：后复权日线 <span class="code">prices_full_v2.csv</span>（954,894 行 / 560 只）+ 通达信原始日 K OHLC
（1,161,396 行 / 560 只，用于涨停判定与开盘价）。引擎已用「行业上限 2 + 不计涨停 + 零成本 + T 日收盘」
复现已发布净值，末值偏差 −0.65%（6.4957 vs 6.5383），年化波动 27.75% vs 28.0%，口径一致。
</div>

</div></body></html>"""

path = os.path.join(OUT, "momentum_reopt.html")
open(path, "w", encoding="utf-8").write(HTML)
print("已生成", path, len(HTML), "字符")
