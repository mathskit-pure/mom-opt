# -*- coding: utf-8 -*-
"""生成《动量轮动策略 · 参数重构回测》分析报告（HTML）。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from engine import Data, run, perf

MODE, POLICY, COST = "open1", "cash", 0.001
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
pd.set_option("display.width", 320)

d = Data()
PD = pd.DatetimeIndex(d.dates)
base = PD.get_indexer([pd.Timestamp("2019-05-06")], method="bfill")[0]

CANDS = [(17, 8, 90, "① 推荐 N17 / K8 / L90"),
         (18, 7, 90, "② 并列最优 N18 / K7 / L90"),
         (17, 8, 100, "③ 次优 N17 / K8 / L100"),
         (16, 8, 100, "④ 次优 N16 / K8 / L100"),
         (15, 10, 50, "⑤ 你原定 N15 / K10 / L50"),
         (10, 5, 12, "⑥ 现网口径 N10 / K5 / L12")]


def phase_stats(N, K, L, cost=COST):
    vs, cums, vols, mdds, anns = [], [], [], [], []
    for o in range(K):
        nav, _ = run(d, N, K, L, mode=MODE, policy=POLICY, cost=cost,
                     start=str(PD[base + o].date()))
        p = perf(nav)
        if not p:
            continue
        vs.append(p["夏普比率"]); cums.append(p["累计收益"]); vols.append(p["年化波动"])
        mdds.append(p["最大回撤"]); anns.append(p["年化收益"])
    return dict(sh=float(np.mean(vs)), std=float(np.std(vs)), worst=float(np.min(vs)),
                best=float(np.max(vs)), cum=float(np.mean(cums)), ann=float(np.mean(anns)),
                vol=float(np.mean(vols)), mdd=float(np.mean(mdds)),
                seq=[float(x) for x in vs])


# ── 主表 ────────────────────────────────────────────────────────────
main_rows, seq_map = [], {}
for N, K, L, tag in CANDS:
    s = phase_stats(N, K, L)
    nav, st = run(d, N, K, L, mode=MODE, policy=POLICY, cost=COST)
    seq_map[tag] = s["seq"]
    main_rows.append({"方案": tag, "N": N, "K": K, "L": L,
                      "相位平均夏普": s["sh"], "相位std": s["std"],
                      "最差相位": s["worst"], "最好相位": s["best"],
                      "相位平均累计": s["cum"], "年化": s["ann"], "波动": s["vol"],
                      "回撤": s["mdd"],
                      "年均换手": st["turnover"] / max(st["n_period"], 1) * (252.0 / K),
                      "涨停阻断率": st["blocked"] / max(st["target"], 1)})
MAIN = pd.DataFrame(main_rows)

# ── 成本敏感性 ──────────────────────────────────────────────────────
cost_rows = []
for N, K, L, tag in CANDS:
    r = {"方案": tag}
    for c in (0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003):
        r["%dbp" % int(c * 1e4)] = phase_stats(N, K, L, cost=c)["sh"]
    cost_rows.append(r)
COST_DF = pd.DataFrame(cost_rows)

# ── 分年度 ──────────────────────────────────────────────────────────
yrs = {}
for N, K, L, tag in CANDS:
    nav, _ = run(d, N, K, L, mode=MODE, policy=POLICY, cost=COST)
    s = pd.Series(nav, index=d.dates)
    yrs[tag] = s.pct_change().groupby(s.index.year).apply(lambda g: (1 + g).prod() - 1)
b = pd.read_csv(r"C:\Users\12066\WorkBuddy\2026-10-05-23-13-30\data\bench_all.csv", index_col=0)
b.index = pd.to_datetime(b.index)
for c in ["沪深300", "创业板50", "科创50"]:
    s = b[c].dropna()
    yrs[c] = s.groupby(s.index.year).apply(lambda g: g.iloc[-1] / g.iloc[0] - 1)
YEAR = pd.DataFrame(yrs)

# ── 子区间 ──────────────────────────────────────────────────────────
SEGS = [("2019-05-06", "2021-12-31", "牛市段 19.05–21.12"),
        ("2022-01-04", "2023-12-29", "熊市段 22.01–23.12"),
        ("2024-01-02", "2026-10-08", "近三年 24.01–26.10")]
seg_rows = []
for N, K, L, tag in CANDS:
    nav, _ = run(d, N, K, L, mode=MODE, policy=POLICY, cost=COST)
    s = pd.Series(nav, index=d.dates)
    r = {"方案": tag}
    for a, b_, nm in SEGS:
        x = s[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b_))]
        v = x.values / x.values[0]
        r[nm] = "%.1f%% / %.2f" % (perf(v)["累计收益"] * 100, perf(v)["夏普比率"])
    seg_rows.append(r)
SEG = pd.DataFrame(seg_rows)

# ── 网格分层 ────────────────────────────────────────────────────────
G1 = pd.read_csv(os.path.join(ROOT, "output", "grid_phase_cash.csv"))
G3 = pd.read_csv(os.path.join(ROOT, "output", "grid_phase_n.csv"))
TOP = G3.sort_values("相位夏普", ascending=False).head(15)
prof_K = G3.groupby("K")["相位夏普"].agg(["mean", "max"]).round(4)
prof_L = G3.groupby("L")["相位夏普"].agg(["mean", "max"]).round(4)
prof_N = G3.groupby("N")["相位夏普"].agg(["mean", "max"]).round(4)

WF = pd.read_csv(os.path.join(ROOT, "output", "walkforward.csv"))


# ── HTML 工具 ───────────────────────────────────────────────────────
def tbl(df, fmts=None, hl=None):
    fmts = fmts or {}
    h = "<table><thead><tr>"
    for c in df.columns:
        h += "<th>%s</th>" % c
    h += "</tr></thead><tbody>"
    for _, r in df.iterrows():
        cls = " class='hi'" if (hl and str(r.iloc[0]).startswith(hl)) else ""
        h += "<tr%s>" % cls
        for c in df.columns:
            v = r[c]
            if c in fmts:
                s = fmts[c](v)
            elif isinstance(v, (float, np.floating)):
                s = "%.4f" % v
            else:
                s = str(v)
            num = isinstance(v, (float, np.floating)) and not isinstance(v, bool)
            if num and c not in ("方案", "N", "K", "L"):
                pos = ("up" if v > 0 else ("dn" if v < 0 else "")) \
                    if c in ("累计", "年化", "回撤", "相位平均累计", "涨跌") else ""
                s = "<span class='%s'>%s</span>" % (pos, s) if pos else s
            h += "<td>%s</td>" % s
        h += "</tr>"
    return h + "</tbody></table>"


pc = lambda v: "%.2f%%" % (v * 100)
f3 = lambda v: "%.3f" % v
f4 = lambda v: "%.4f" % v
f1 = lambda v: "%.1f" % v

MAIN_SHOW = MAIN.copy()
MAIN_SHOW["相位平均夏普"] = MAIN_SHOW["相位平均夏普"].map(f4)
MAIN_SHOW["相位std"] = MAIN_SHOW["相位std"].map(f4)
MAIN_SHOW["最差相位"] = MAIN_SHOW["最差相位"].map(f3)
MAIN_SHOW["最好相位"] = MAIN_SHOW["最好相位"].map(f3)
MAIN_SHOW["相位平均累计"] = MAIN_SHOW["相位平均累计"].map(pc)
MAIN_SHOW["年化"] = MAIN_SHOW["年化"].map(pc)
MAIN_SHOW["波动"] = MAIN_SHOW["波动"].map(pc)
MAIN_SHOW["回撤"] = MAIN_SHOW["回撤"].map(pc)
MAIN_SHOW["年均换手"] = MAIN_SHOW["年均换手"].map(lambda v: "%.1f×" % v)
MAIN_SHOW["涨停阻断率"] = MAIN_SHOW["涨停阻断率"].map(lambda v: "%.3f%%" % (v * 100))

COST_SHOW = COST_DF.copy()
for c in COST_DF.columns[1:]:
    COST_SHOW[c] = COST_DF[c].map(f4)

YEAR_SHOW = (YEAR * 100).round(2)
YEAR_SHOW.index.name = "年份"
YEAR_SHOW = YEAR_SHOW.reset_index()
for c in YEAR_SHOW.columns[1:]:
    YEAR_SHOW[c] = YEAR_SHOW[c].map(lambda v: "—" if pd.isna(v) else "%.2f%%" % v)

TOP_SHOW = TOP[["N", "K", "L", "相位夏普", "相位夏普std", "最差相位", "累计收益", "年化波动", "最大回撤"]].copy()
TOP_SHOW["相位夏普"] = TOP_SHOW["相位夏普"].map(f4)
TOP_SHOW["相位夏普std"] = TOP_SHOW["相位夏普std"].map(f4)
TOP_SHOW["最差相位"] = TOP_SHOW["最差相位"].map(f3)
TOP_SHOW["累计收益"] = TOP_SHOW["累计收益"].map(pc)
TOP_SHOW["年化波动"] = TOP_SHOW["年化波动"].map(pc)
TOP_SHOW["最大回撤"] = TOP_SHOW["最大回撤"].map(pc)


def prof_html(s, name):
    h = "<table><thead><tr><th>%s</th>" % name
    for i in s.index:
        h += "<td>%s</td>" % i
    h += "</tr></thead><tbody><tr><th>相位平均夏普（均值）</th>"
    for v in s["mean"]:
        h += "<td>%.4f</td>" % v
    h += "</tr><tr><th>该层最高</th>"
    for v in s["max"]:
        h += "<td>%.4f</td>" % v
    return h + "</tr></tbody></table>"


# ── 相位分布条形图 ──────────────────────────────────────────────────
def seq_svg(seq_map):
    rows = []
    y = 30
    allv = [v for s in seq_map.values() for v in s]
    vmin, vmax = min(allv), max(allv)
    for tag, seq in seq_map.items():
        rows.append("<text x='0' y='%d' font-size='11' fill='#374151'>%s</text>"
                    % (y + 4, tag))
        n = len(seq)
        for i, v in enumerate(seq):
            x = 168 + i * 46
            w = 40
            col = "#16a34a" if v >= np.mean(seq) else "#dc2626"
            rows.append("<rect x='%d' y='%d' width='%d' height='14' rx='2' fill='%s' "
                        "opacity='0.85'/>" % (x, y - 10, w, col))
            rows.append("<text x='%d' y='%d' font-size='9' fill='#6b7280' "
                        "text-anchor='middle'>%.2f</text>" % (x + w / 2, y + 12, v))
        rows.append("<text x='%d' y='%d' font-size='10' fill='#111827'>均值 %.3f</text>"
                    % (168 + n * 46 + 8, y + 4, np.mean(seq)))
        y += 34
    return ("<svg viewBox='0 0 720 %d' width='100%%' style='max-width:720px'>%s</svg>"
            % (y + 10, "".join(rows)))


WF_SHOW = WF.copy()
for c in ["训练期夏普", "样本外夏普", "N17K8L90夏普", "N15K10L50夏普"]:
    WF_SHOW[c] = WF_SHOW[c].map(f3)
for c in ["样本外收益", "N17K8L90收益", "N15K10L50收益"]:
    WF_SHOW[c] = WF_SHOW[c].map(lambda v: "%.1f%%" % v)

HTML = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>动量轮动策略 · 参数重构回测（相位平均寻优）</title>
<style>
:root{{--bg:#f7f8fa;--card:#fff;--bd:#e5e7eb;--tx:#111827;--mut:#6b7280;--up:#dc2626;--dn:#16a34a}}
*{{box-sizing:border-box}}
body{{margin:0;padding:28px 20px 60px;background:var(--bg);color:var(--tx);
 font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
 font-size:14px;line-height:1.7}}
.wrap{{max-width:1180px;margin:0 auto}}
h1{{font-size:24px;margin:0 0 6px}}
h2{{font-size:18px;margin:34px 0 12px;padding-left:10px;border-left:4px solid #2563eb}}
h3{{font-size:15px;margin:20px 0 8px;color:#374151}}
.sub{{color:var(--mut);font-size:13px;margin-bottom:18px}}
.card{{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:16px 18px;margin-bottom:14px}}
.scroll{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border:1px solid var(--bd);padding:6px 10px;text-align:right;white-space:nowrap}}
th{{background:#f3f4f6;font-weight:600;text-align:center}}
td:first-child,th:first-child{{text-align:left}}
tbody tr:nth-child(even){{background:#fafafa}}
tr.hi{{background:#eff6ff!important}}
tr.hi td{{font-weight:600}}
.up{{color:var(--up)}} .dn{{color:var(--dn)}}
.note{{color:var(--mut);font-size:12.5px;margin-top:10px;line-height:1.75}}
.kv{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:14px 0}}
.kv div{{background:#fff;border:1px solid var(--bd);border-radius:10px;padding:12px 14px}}
.kv b{{display:block;font-size:22px;margin-bottom:2px}}
.kv span{{color:var(--mut);font-size:12px}}
.warn{{background:#fffbeb;border:1px solid #fcd34d;border-radius:10px;padding:14px 16px;margin:14px 0}}
.ok{{background:#eff6ff;border:1px solid #93c5fd;border-radius:10px;padding:14px 16px;margin:14px 0}}
code{{background:#f3f4f6;padding:1px 5px;border-radius:4px;font-size:12.5px}}
</style></head><body><div class="wrap">

<h1>动量轮动策略 · 参数重构回测</h1>
<div class="sub">标的池：舞弊调整质量因子季度动态股池（2019Q1–2026H1，23 期，384–445 只/期）　·
区间：2019-05-06 → 2026-10-08（1802 个交易日）　·　生成于 2026-10-10</div>

<div class="ok">
<b>本次口径变更</b>：取消「单个申万三级行业最多 2 只」限制；新增<b>涨停无法建仓约束</b>（开盘涨停 → 该股权重本期空置、以现金持有）与<b>单边 10bp 交易成本</b>；执行时点由 T 日收盘改为 <b>T+1 开盘</b>；全部参数用<b>调仓相位平均</b>寻优。
评分口径与次新过滤不变（上市未满 60 个交易日不入池）。
</div>

<h2>一、结论：N = 17、K = 8、L = 90</h2>
<div class="kv">
<div><b>17 天</b><span>动量周期 N</span></div>
<div><b>8 个交易日</b><span>调仓间隔 K</span></div>
<div><b>90 只</b><span>持仓数量 L</span></div>
<div><b>1.0407</b><span>相位平均夏普（单边 10bp）</span></div>
</div>
<div class="card">
按该参数从 2019-05-06 单一起点跑全段：累计 <b>+422.5%</b>、年化 <b>24.8%</b>、年化波动 <b>23.2%</b>、
最大回撤 <b>−29.7%</b>、夏普 <b>1.074</b>、卡玛 <b>0.835</b>，年均换手 <b>36.5×</b>。
</div>
<div class="card">
相位平均夏普最高的单点其实是 <b>N=18 / K=7 / L=90（1.0457）</b>，与 N17/K8/L90 只差 0.005，
但后者的<b>最差相位 0.962 明显更高</b>（前者 0.937）、标准差更小，抗"起点偏移"能力更强，故取 N17/K8/L90。
真正稳健的不是某个点，而是一块高原：<b>N ∈ [16, 18]、K ∈ [7, 8]、L ∈ [80, 110]</b>，
区间内相位平均夏普都在 1.035–1.046，彼此相差不到 0.01 —— 不必卡死单点。
</div>
<div class="warn">
<b>关于你原定的 N=15 / K=10 / L=50：</b>相位平均夏普 <b>1.0148</b>，比推荐值低 0.026。
它在<b>单一相位起点</b>（2019-05-06）下能跑到夏普 1.177、累计 +576.9%，看起来比推荐值（1.074 / +422.5%）更好，
但把起点平移 0–9 个交易日后，它的相位分布是 <b>0.864 ~ 1.212（std 0.118）</b>—— 那个 1.177 恰好踩在最好的相位上。
推荐值的相位分布是 <b>0.962 ~ 1.15（std 0.065）</b>。你要的就是相位平均，所以按 1.0407 > 1.0148 排序。
<br><br>如果你更看重"实际会走出来的那一条曲线"而不是相位平均，N15/K10/L50 也并非不可 ——
它胜在年化更高（29.2% vs 24.8%）、卡玛更好（0.926 vs 0.835），但代价是结果更依赖起点运气。
</div>

<h2>二、候选方案对比</h2>
<div class="card scroll">{tbl(MAIN_SHOW, hl="①")}
<div class="note">「相位平均」＝把回测起点依次平移 0…K−1 个交易日共跑 K 遍，取夏普均值。
「涨停阻断率」＝被开盘涨停挡掉的目标权重占比；T+1 开盘执行下仅 0.2%–0.7%，
远低于收盘执行的 1.8%、T 日收盘的 5.5%。「空置权重」以现金形式保留在组合内，收益计 0。</div>
</div>

<h3>相位分布（每个柱 = 一个调仓起点）</h3>
<div class="card scroll">{seq_svg(seq_map)}
<div class="note">绿色 = 高于该方案自身均值，红色 = 低于。柱子越参差说明参数越依赖"恰好从哪天开始调仓"。
⑥ 现网口径 N10/K5/L12 在计入成本与涨停约束后，相位 std 高达 0.150、最差相位 0.361。</div>
</div>

<h2>三、寻优过程</h2>
<div class="card">
共跑 <b>三轮相位平均网格</b>，合计 <b>872 组参数、8108 次回测</b>：
第一轮 8×8×7（N 5–30 / K 2–20 / L 10–80）→ 发现峰值在 N≈15、K=8、L 触及上边界 80；
第二轮把 L 上探到 120、N 细化到 12–20 → 峰值移到 N=17；
第三轮把 N 扫到 14–19、L 扫到 60–110 → 确认峰值 <b>N=16~18 / K=7~8 / L=90~100</b>。
</div>
<h3>单参数剖面（第三轮，对其它维度取平均）</h3>
<div class="card scroll">
{prof_html(prof_N, "动量周期 N")}
<div class="note" style="margin-top:12px"></div>
{prof_html(prof_K, "调仓间隔 K")}
<div class="note" style="margin-top:12px"></div>
{prof_html(prof_L, "持仓数量 L")}
<div class="note">N：12 以下与 20 以上都明显掉档，16–18 是峰值。<br>
K：8 最高，7 次之，10 起单调下滑 —— 调仓太快会被 10bp 成本吃掉。<br>
L：80–110 是平台；L 越大波动越低（22.6% → 22.9%），但累计收益也越低。</div>
</div>
<h3>第三轮网格 TOP 15</h3>
<div class="card scroll">{tbl(TOP_SHOW)}
<div class="note">按相位平均夏普降序。前 15 名全部落在 N∈[16,18] × K∈[7,8] × L∈[90,110]，高原非常集中。</div>
</div>

<h2>四、稳健性</h2>
<h3>成本敏感性（相位平均夏普）</h3>
<div class="card scroll">{tbl(COST_SHOW)}
<div class="note">10bp 是本次基准假设（佣金+印花税+冲击）。即便把成本翻到 20bp，推荐值仍有 0.887；
而现网口径 N10/K5/L12 在 30bp 时只剩 0.029 —— 它是靠高换手（83×/年）吃饭的，成本一涨就垮。
推荐值年均换手 36×，只有现网口径的 43%。</div>
</div>
<h3>关键子区间</h3>
<div class="card scroll">{tbl(SEG)}
<div class="note">格式：区间累计收益 / 区间夏普。2022–2023 熊市段所有参数都亏钱，
差别在于亏多少：推荐值 −13.9%，现网口径 −37.6%。</div>
</div>
<h3>分年度收益（相位 0，10bp）</h3>
<div class="card scroll">{tbl(YEAR_SHOW)}
<div class="note">2022 年沪深300 −21.27%、创业板50 −27.78%，策略同样下跌，属系统性风险；
2024 年以来策略连续三年跑赢三大宽基。</div>
</div>

<h2>五、滚动前进样本外检验</h2>
<div class="card scroll">{tbl(WF_SHOW)}
<div class="note">做法：只用「截至第 t 年末」的数据做相位平均寻优，拿选出的参数去跑第 t+1 年。
训练期夏普均值 0.833 → 样本外均值 <b>0.739</b>，衰减约 11%；同期固定 N17/K8/L90 的逐年夏普均值 0.802。
2022 年样本外 −0.99 是主要拖累（该年沪深300 −21.27%，所有动量参数都无法幸免）。</div>
</div>
<div class="warn">
<b>预期管理：</b>相位平均夏普 1.04 是<b>全样本回测</b>值，含参数选择偏差。
滚动前进给出的样本外基准是 <b>0.74–0.80</b>，所以对未来实盘的合理预期应设在 <b>0.6–0.9</b>，
而不是 1.04。2022 年那种系统性下跌年份，该策略会跟着亏 15%–20%。
</div>

<h2>六、口径说明</h2>
<div class="card">
<b>信号</b>：<code>Adj = N 日平均收益 / √(N 日收益方差)</code>；筛原始动量为正，按 Adj 降序取前 L 只；权重按 Adj 归一化。<br>
<b>时点</b>：T 日收盘算信号 → T+1 <b>开盘</b>成交；持有至下一执行日开盘卖出；组合一律按每日收盘价盯市，
保证不同执行时点的日收益同为 close-to-close，波动率可比。<br>
<b>涨停判定</b>：按原始价格空间计算涨停价（主板 10%、创业板/科创板 20%、北交所 30%，创业板 2020-08-24 前后不同），
开盘价 ≥ 涨停价 − 0.006 元即判定无法建仓。<br>
<b>空置处理</b>：被阻断的权重<b>不递补、不摊给其它个股</b>，直接以现金持有至下一调仓日（收益计 0）。<br>
<b>成本</b>：按每期换手额（目标权重与漂移后权重之差）单边计收 10bp。<br>
<b>相位平均</b>：对每个 K，把回测起点平移 0…K−1 个交易日共跑 K 遍取均值，用于抑制"恰好从哪天开始调仓"的日历对齐过拟合。
</div>

<div class="note" style="margin-top:24px">
数据：通达信日 K（560 只标的，1.16M 行 OHLC）+ 舞弊调整质量因子季度股池。
回测窗口 2019-05-06 → 2026-10-08 为 <b>1802</b> 个交易日（1881 是含 2019 年初的全表天数）。
</div>
</div></body></html>"""

out = os.path.join(ROOT, "output", "momentum_reopt_v2.html")
open(out, "w", encoding="utf-8").write(HTML)
print("已写出：%s（%d chars）" % (out, len(HTML)))
print("表格数 %d，外链 %s" % (HTML.count("<table"),
                              __import__("re").findall(r'(?:src|href)=["\']https?:', HTML) or "无"))
