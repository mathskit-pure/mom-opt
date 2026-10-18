# -*- coding: utf-8 -*-
"""
按「新口径」重算页面所需的全部回测产物
=====================================
口径变更（2026-10-10 定稿）：
  · 动量周期 N=15、调仓间隔 K=10、持仓 L=50
  · 取消「单个申万三级行业最多 2 只」限制
  · 执行时点 T+1 开盘（open1）
  · 开盘涨停无法建仓 → 该股权重本期空置、以现金持有
  · 单边 10bp 交易成本
  · 参数由「调仓相位平均」寻优确定（抑制日历对齐过拟合）
  · 回测区间 2019-05-06 → 最新交易日

产物（写入原项目 data/）：
  nav.csv              动量轮动策略 / 等权 / 质量Top28 / 三大宽基
  holdings.csv         逐期持仓明细（含建仓状态）
  removed.csv          逐期剔除明细
  backtest_summary.csv 绩效汇总（口径与页面 JS calc() 严格一致）
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from engine import Data, run           # noqa: E402
from paths import SRC          # noqa: E402  路径：本机 Windows 与云端 Linux 通用

sys.path.insert(0, os.path.join(str(SRC), "scripts"))
try:
    import quality_v2 as Q2     # 质量 Top28 v2 选股/加权规则（pool-core/scripts）
except Exception as _e:        # 规则文件缺失时退回旧权重，不阻断整链
    Q2 = None
    print("! 未加载 quality_v2（%s），质量组合退回旧权重" % _e)


def load_turnover():
    """换手率面板：{报告期: {code: (adv20, turn)}}；缺失返回空字典（自动降级）"""
    if Q2 is None:
        return {}
    f = os.path.join(str(SRC), "data", "turnover_panel.csv")
    if not os.path.exists(f):
        print("! 未找到 turnover_panel.csv，质量组合退回旧权重")
        return {}
    t = pd.read_csv(f, encoding="utf-8-sig", dtype={"code": str})
    out = {}
    for per, g in t.groupby("报告期"):
        out[per] = {r["code"]: (float(r["adv20"]), float(r["turn"])) for _, r in g.iterrows()}
    print("换手面板：%d 期 · %d 只" % (len(out), t["code"].nunique()))
    return out

OUT = os.path.join(SRC, "data")

N, K, L = 15, 10, 50
MODE, POLICY, COST = "open1", "cash", 0.001
START = "2019-05-06"
MIN_LISTED = 60
LQ, MAX_PER_IND_Q, TOP_W, N_TOP_W = 28, 2, 0.16, 3


# ── 页面口径绩效（与页面 JS calc() 逐行一致：首日收益按 0 计入 n）──
def page_perf(vals, label):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if len(vals) < 10:
        return {}
    n = len(vals)
    r = [0.0] + [vals[i] / vals[i - 1] - 1.0 for i in range(1, n)]
    eq, peak, mdd, mu = 1.0, 1.0, 0.0, 0.0
    for x in r:
        eq *= (1 + x)
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1)
    mu = sum(r) / n
    sq = sum((x - mu) ** 2 for x in r)
    vol = np.sqrt(sq / n) * np.sqrt(252)
    ann = eq ** (252.0 / n) - 1
    return {"组合": label, "累计收益": eq - 1, "年化收益": ann, "年化波动": vol,
            "最大回撤": mdd, "夏普比率": (ann / vol) if vol else np.nan,
            "卡玛比率": (ann / abs(mdd)) if mdd else np.nan, "交易日数": n}


def qual_weights(n):
    if n <= 0:
        return []
    if n <= N_TOP_W:
        return [1.0 / n] * n
    rest = (1.0 - TOP_W * N_TOP_W) / (n - N_TOP_W)
    return [TOP_W] * N_TOP_W + [rest] * (n - N_TOP_W)


def active_period(d, asof):
    cur = None
    for a, period in asof:
        if pd.Timestamp(a) <= d:
            cur = period
    return cur


def main():
    d = Data()
    print("价格矩阵 %d×%d，%s → %s" % (d.T, d.C, d.dates[0].date(), d.dates[-1].date()))

    # ── 1) 动量轮动策略（新口径）───────────────────────────────────
    nav, st, hold_rows, rm_rows = run(d, N, K, L, mode=MODE, policy=POLICY,
                                      cost=COST, detail=True)
    strat = pd.Series(nav, index=d.dates)
    print("动量策略：%d 期，阻断率 %.3f%%，年均换手 %.1f×，平均持仓 %.1f 只，"
          "空置权重均值 %.3f%%"
          % (st["n_period"], st["blocked"] / max(st["target"], 1) * 100,
             st["turnover"] / max(st["n_period"], 1) * (252.0 / K),
             st["avg_n"] / max(st["n_period"], 1),
             st["cash_w"] / max(st["n_period"], 1) * 100))

    # ── 2) 收盘价矩阵：等权组合 / 质量Top28 / 基准 ─────────────────
    close = d.close
    ret = close.pct_change()
    listed = close.notna().cumsum()
    dates = close.index
    start_pos = int(np.searchsorted(dates.values, np.datetime64(START)))
    reb_pos = list(range(start_pos, len(dates), K))

    pools = json.load(open(os.path.join(SRC, "data", "pools.json"), encoding="utf-8"))
    asof = sorted([(v["asof"], k) for k, v in pools.items() if v.get("asof")])
    pool_by_period = {k: [m["code"] for m in v["members"]] for k, v in pools.items()}
    score_by_period = {k: sorted(((m["code"], float(m["FAQ_score"]))
                                  for m in v["members"] if m.get("FAQ_score") is not None),
                                 key=lambda x: -x[1]) for k, v in pools.items()}
    meta = json.load(open(os.path.join(SRC, "scripts", "meta.json"), encoding="utf-8"))
    sw3 = {c: (meta.get(c, {}).get("sw3") or "未分类") for c in close.columns}
    name = {c: (meta.get(c, {}).get("name") or c) for c in close.columns}
    score_map_by_period = {k: dict(v) for k, v in score_by_period.items()}
    turn_by_period = load_turnover()

    def price_at(code, d):
        """取 d 日（或之前最近交易日）收盘价，用于计算持仓区间涨跌幅。"""
        s = close[code]
        v = s.get(d, np.nan) if d in s.index else np.nan
        if pd.notna(v):
            return float(v)
        ss = s.loc[:d].dropna()
        return float(ss.iloc[-1]) if len(ss) else np.nan

    def active_pool(dd):
        cur = active_period(dd, asof)
        return [c for c in pool_by_period.get(cur, []) if c in close.columns] if cur else []

    # 等权：同股池、同调仓频率（K=10）、全部等权持有
    We = pd.DataFrame(0.0, index=dates, columns=close.columns)
    for j, pos in enumerate(reb_pos):
        end_pos = reb_pos[j + 1] - 1 if j + 1 < len(reb_pos) else len(dates) - 1
        dd = dates[pos]
        pool = [c for c in active_pool(dd) if listed.iloc[pos].get(c, 0) >= MIN_LISTED]
        if pool:
            We.iloc[pos:end_pos + 1, [We.columns.get_loc(c) for c in pool]] = 1.0 / len(pool)

    # 质量Top28：季度换仓、买入持有
    Wq = pd.DataFrame(0.0, index=dates, columns=close.columns)
    last_per, prev_qw = None, pd.Series(0.0, index=close.columns)
    q_prev_set, q_entry_of = set(), {}      # 质量组合：上季持仓 / 个股本轮建仓日
    qual_rows, qual_removed = [], []
    for pos in range(start_pos, len(dates)):
        dd = dates[pos]
        per = active_period(dd, asof)
        if per != last_per:
            last_per = per
            ok_q = {c for c in close.columns if listed.iloc[pos].get(c, 0) >= MIN_LISTED}
            cand = [(c, s) for c, s in score_by_period.get(per, [])
                    if c in close.columns and c in ok_q]
            if Q2 is not None:
                picked_v2, v2meta = Q2.select_v2(cand, sw3, turn_by_period.get(per, {}),
                                                 lq=LQ, max_per_ind=MAX_PER_IND_Q)
            else:
                picked_v2, v2meta = [], {"degrade": "v1", "floor": None}
            if picked_v2 and v2meta["degrade"] == "ok":
                qcodes = [c for c, _, _ in picked_v2]
                qw = [ww for _, ww, _ in picked_v2]
                qpct = {c: p for c, _, p in picked_v2}
                print("  %s %s 持仓 %d 只（门槛 %.0f 万）"
                      % (dd.date(), per, len(qcodes), (v2meta["floor"] or 0) / 1e4))
            else:                       # 降级：退回旧口径（前3名16%）
                qcodes, cnt = [], {}
                for c, s in cand:
                    if len(qcodes) >= LQ:
                        break
                    ind = sw3.get(c, "未分类")
                    if cnt.get(ind, 0) >= MAX_PER_IND_Q:
                        continue
                    cnt[ind] = cnt.get(ind, 0) + 1
                    qcodes.append(c)
                qw = qual_weights(len(qcodes))
                qpct = {}
                print("  [降级] %s %s（%s）" % (dd.date(), per, v2meta["degrade"]))
            w = pd.Series(0.0, index=close.columns)
            for c, ww in zip(qcodes, qw):
                w[c] = ww
            qset = set(qcodes)

            # 本季被剔除：上季持有、本季不再持有 → 建仓日至剔除日累计涨跌幅
            for c in sorted(q_prev_set - qset):
                e = q_entry_of.pop(c, None)
                p0 = price_at(c, e) if e is not None else np.nan
                p1 = price_at(c, dd)
                chg = ((p1 / p0 - 1)
                       if (p0 and p1 and np.isfinite(p0) and np.isfinite(p1)) else np.nan)
                qual_removed.append({"调仓日": dd.date(), "代码": c,
                                     "名称": name.get(c, c), "申万三级": sw3.get(c, "未分类"),
                                     "建仓日": e.date() if e is not None else None,
                                     "区间涨跌幅": chg})

            smap = score_map_by_period.get(per, {})
            for rank, c in enumerate(qcodes, 1):
                if c not in q_prev_set:
                    q_entry_of[c] = dd
                qual_rows.append({"报告期": per, "调仓日": dd.date(), "次序": rank,
                                  "代码": c, "名称": name.get(c, c),
                                  "申万三级": sw3.get(c, "未分类"),
                                  "FAQ得分": smap.get(c),
                                  "换手分位": qpct.get(c),
                                  "建仓权重": qw[rank - 1] if qcodes else 0,
                                  "状态": "新增" if c not in q_prev_set else "持有"})
            q_prev_set = qset
        else:
            r = ret.iloc[pos].reindex(close.columns).fillna(0.0)
            w = prev_qw * (1.0 + r)
            s = float(w.sum())
            w = (w / s) if s else prev_qw
        Wq.iloc[pos] = w.values
        prev_qw = w

    mask = dates >= pd.Timestamp(START)
    equal = (We.shift(1) * ret).sum(axis=1)[mask]
    qual = (Wq.shift(1) * ret).sum(axis=1)[mask]

    # ── 3) 组装 nav.csv ────────────────────────────────────────────
    strat_s = strat[mask]
    # 注意：engine 返回的 nav 是净值水平（起点 1.0），不是日收益，不能 cumprod
    nav_df = pd.DataFrame({
        "策略": strat_s / strat_s.iloc[0],
        "动量轮动策略（等权）": (1 + equal).cumprod(),
        "质量Top28组合": (1 + qual).cumprod(),
    })
    # 起始日补 1.0（与页面首日对齐）
    first = nav_df.index[0]
    nav_df.loc[first] = 1.0
    nav_df = nav_df.sort_index()

    # 基准：先在完整交易日轴上对齐再前后填充，最后按自己的首个有效值定基。
    # 旧产物的基准列没有 NaN（科创50 未上市期回填为 1.0），必须保持一致，
    # 否则页面统计表 calc() 会在缺口处撞上 None 直接崩。
    b = pd.read_csv(os.path.join(SRC, "data", "bench_all.csv"), index_col=0)
    b.index = pd.to_datetime(b.index)
    for c in ["沪深300", "创业板50", "科创50"]:
        s = b[c].dropna().sort_index()
        s = s.reindex(nav_df.index).ffill().bfill()
        nav_df[c] = s / s.iloc[0]
    nav_df = nav_df.dropna(how="all")
    print("nav.csv：%s → %s，%d 行，列 %s"
          % (nav_df.index[0].date(), nav_df.index[-1].date(), len(nav_df), list(nav_df.columns)))

    # ── 4) 绩效汇总 ────────────────────────────────────────────────
    rows = [page_perf(nav_df["策略"].values, "动量轮动策略"),
            page_perf(nav_df["动量轮动策略（等权）"].values, "动量轮动策略（等权）"),
            page_perf(nav_df["质量Top28组合"].values, "质量Top28组合")]
    for c in ["沪深300", "创业板50", "科创50"]:
        v = nav_df[c].dropna().values
        rows.append(page_perf(v, c))
    summary = pd.DataFrame(rows)

    # ── 5) 写出 ────────────────────────────────────────────────────
    nav_df.to_csv(os.path.join(OUT, "nav.csv"), encoding="utf-8-sig")

    # 持仓明细：列顺序与原页面消费端保持一致（权重 = 实际权重；涨停空置者为 0）
    hold_df = pd.DataFrame(hold_rows)
    hold_df["权重"] = hold_df["实际权重"]
    hold_df = hold_df[["调仓日", "执行日", "次序", "代码", "名称", "申万三级",
                       "原始动量", "调整动量", "目标权重", "实际权重", "权重",
                       "建仓状态", "状态"]]
    hold_df.to_csv(os.path.join(OUT, "holdings.csv"), index=False, encoding="utf-8-sig")

    rm_df = pd.DataFrame(rm_rows)
    rm_df = rm_df[["调仓日", "代码", "名称", "申万三级", "建仓日", "剔除日", "区间涨跌幅"]]
    rm_df.to_csv(os.path.join(OUT, "removed.csv"), index=False, encoding="utf-8-sig")
    summary.to_csv(os.path.join(OUT, "backtest_summary.csv"),
                   index=False, encoding="utf-8-sig")

    # 质量 Top28 明细（4.2 持仓变动表消费）：与旧口径产物列完全一致
    q_df = pd.DataFrame(qual_rows)
    for col in ("换手分位", "建仓权重"):
        if col not in q_df.columns:
            q_df[col] = np.nan
    q_df = q_df[["报告期", "调仓日", "次序", "代码", "名称", "申万三级",
                 "FAQ得分", "换手分位", "建仓权重", "状态"]]
    q_df.to_csv(os.path.join(OUT, "quality_top28.csv"), index=False, encoding="utf-8-sig")
    qr_df = pd.DataFrame(qual_removed)
    qr_df = qr_df[["调仓日", "代码", "名称", "申万三级", "建仓日", "区间涨跌幅"]]
    qr_df.to_csv(os.path.join(OUT, "quality_removed.csv"), index=False, encoding="utf-8-sig")
    print("质量Top28 明细 %d 行 · 剔除记录 %d 行" % (len(q_df), len(qr_df)))

    pd.set_option("display.width", 250)
    print("\n=== 绩效汇总（页面口径）===")
    print(summary.round(4).to_string(index=False))
    print("\n持仓明细 %d 行 · 剔除记录 %d 行" % (len(hold_rows), len(rm_rows)))
    print("调仓日 %d 个：%s → %s"
          % (len(set(r["调仓日"] for r in hold_rows)),
             min(r["调仓日"] for r in hold_rows), max(r["调仓日"] for r in hold_rows)))
    print("\n最新一期持仓前 8 只：")
    last_day = max(r["调仓日"] for r in hold_rows)
    for r in [x for x in hold_rows if x["调仓日"] == last_day][:8]:
        print("  %-6s %-8s 目标 %.2f%% 实际 %.2f%% %s"
              % (r["代码"], r["名称"], r["目标权重"] * 100, r["实际权重"] * 100, r["建仓状态"]))
    nb = sum(1 for r in hold_rows if r["建仓状态"] != "正常")
    print("建仓受阻合计 %d 笔 / %d 笔（%.3f%%）"
          % (nb, len(hold_rows), nb / max(len(hold_rows), 1) * 100))


if __name__ == "__main__":
    main()
