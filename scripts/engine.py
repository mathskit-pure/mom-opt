# -*- coding: utf-8 -*-
"""
动量轮动策略回测引擎（无行业数量限制版）
======================================
与原 scripts/backtest.py 的差异：
  1) 取消「单个申万三级行业最多 2 只」约束（MAX_PER_IND = ∞）
  2) 引入真实的可成交性判定：涨停封板 → 无法建仓
  3) 执行时点可选：T+1 开盘 / T+1 收盘
  4) 涨停阻断后的补仓策略可选：权重分配 / 依次递补 / 等待开板
  5) 保留次新股过滤（MIN_LISTED = 60 个交易日）

信号口径与原版一致：
  mom = N 日平均收益率；var = N 日收益率方差；adj = mom / sqrt(var)
  原始动量 > 0 → 按 adj 降序取前 L 只 → 权重 = adj 归一化

时点口径（本次统一为「T 日决策、T+1 执行」）：
  决策日 T（用 T 收盘数据算动量） → 执行日 E = T+1
  开盘模式：以 open_hfq[E] 成交；收盘模式：以 close_hfq[E] 成交
  持有至下一次执行日 E'（下一决策日的次日）以同一口径价格卖出
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = r"C:\Users\12066\WorkBuddy\2026-10-09-23-54-07\mom_opt"
SRC = r"C:\Users\12066\WorkBuddy\2026-10-05-23-13-30"

START = "2019-05-06"
MIN_LISTED = 60


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #
def limit_ratio(code: str, date: pd.Timestamp) -> float:
    """涨跌幅限制：主板 10%；创业板 2020-08-24 起 20%；科创板 20%。"""
    if code.startswith("sh688"):
        return 0.20
    if code.startswith("sz300") or code.startswith("sz301"):
        return 0.20 if date >= pd.Timestamp("2020-08-24") else 0.10
    if code.startswith("bj"):
        return 0.30
    return 0.10


class Data:
    def __init__(self):
        px = pd.read_csv(os.path.join(SRC, "data", "prices_full_v2.csv"),
                         dtype={"code": str})
        px["date"] = pd.to_datetime(px["date"])
        px["close"] = pd.to_numeric(px["close"], errors="coerce")
        px = px.dropna(subset=["close"]).drop_duplicates(["code", "date"])
        self.close = px.pivot(index="date", columns="code", values="close").sort_index()

        raw = pd.read_csv(os.path.join(BASE, "data", "ohlc_raw.csv"), dtype={"code": str})
        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw.drop_duplicates(["code", "date"])
        codes = list(self.close.columns)
        raw = raw[raw["code"].isin(codes)]
        for col in ("open", "high", "low", "close"):
            m = raw.pivot(index="date", columns="code", values=col).sort_index()
            m = m.reindex(index=self.close.index, columns=codes)
            setattr(self, col + "_raw", m)

        # 后复权开盘价：用 hfq_close / raw_close 逐日构造复权因子
        factor = self.close / self.close_raw
        self.open = self.open_raw * factor
        self.high = self.high_raw * factor
        self.low = self.low_raw * factor

        self.dates = self.close.index
        self.codes = codes
        self.T = len(self.dates)
        self.C = len(codes)
        self.code_i = {c: i for i, c in enumerate(codes)}

        # 停牌 = 当日无数据
        self.has = self.close.notna().values
        self.close_f = self.close.ffill().values
        self.open_f = self.open.ffill().values

        self._build_limit_flags()
        self._build_pool_mask()
        self.listed = self.close.notna().cumsum().values

    # ------------------------------------------------------------------ #
    def mom_adj(self, N):
        """N 日动量与风险调整动量（带缓存）。"""
        if getattr(self, "_ma", None) is None:
            self._ma = {}
        if N in self._ma:
            return self._ma[N]
        # 与项目原版同口径：停牌/缺失日收益为 NaN（滚动窗口内出现 NaN 即剔除该股），
        # 而非按 0 处理；仅在市值盯市时用 ffill 价格。
        c = self.close.values
        with np.errstate(invalid="ignore"):
            ret = np.vstack([np.full((1, self.C), np.nan), c[1:] / c[:-1] - 1.0])
        with np.errstate(invalid="ignore"):
            mom = pd.DataFrame(ret).rolling(N, min_periods=N).mean().values
            var = pd.DataFrame(ret).rolling(N, min_periods=N).var(ddof=1).values
            adj = mom / np.sqrt(var)
        self._ma[N] = (mom, adj)
        return mom, adj

    # ------------------------------------------------------------------ #
    def _build_limit_flags(self):
        """涨停判定（原始价格空间）。

        seal_all  : 全天封板（一字/开盘涨停后未开板）→ low >= 涨停价
        seal_close: 收盘封板（含全天封板与尾盘封板）→ close >= 涨停价
        seal_open : 开盘即涨停 → open >= 涨停价
        """
        c_raw = self.close_raw.values
        o_raw = self.open_raw.values
        l_raw = self.low_raw.values
        prev = np.vstack([np.full((1, self.C), np.nan), c_raw[:-1]])

        ratio = np.zeros((self.T, self.C))
        for j, c in enumerate(self.codes):
            if c.startswith("sh688") or c.startswith("bj"):
                ratio[:, j] = 0.20
            elif c.startswith("sz300") or c.startswith("sz301"):
                ratio[:, j] = np.where(self.dates >= pd.Timestamp("2020-08-24"), 0.20, 0.10)
            else:
                ratio[:, j] = 0.10

        lim = np.round(prev * (1.0 + ratio), 2)
        eps = 0.006          # 一分钱容差
        with np.errstate(invalid="ignore"):
            self.seal_close = (np.round(c_raw, 2) >= lim - eps) & np.isfinite(prev)
            self.seal_open = (np.round(o_raw, 2) >= lim - eps) & np.isfinite(prev)
            self.seal_all = (np.round(l_raw, 2) >= lim - eps) & np.isfinite(prev)
            self.limit_dn = (np.round(c_raw, 2) <= np.round(prev * (1.0 - ratio), 2) + eps) \
                & np.isfinite(prev)
        # 尾盘封板 = 收盘封板 且 并非全天封板
        self.seal_late = self.seal_close & ~self.seal_all
        # 触板未封 = 盘中摸到涨停但收盘没封住
        h_raw = self.high_raw.values
        self.touch = (np.round(h_raw, 2) >= lim - eps) & ~self.seal_close & np.isfinite(prev)

    # ------------------------------------------------------------------ #
    def _build_pool_mask(self):
        pools = json.load(open(os.path.join(SRC, "data", "pools.json"), encoding="utf-8"))
        asof = sorted([(v["asof"], k) for k, v in pools.items() if v.get("asof")])
        pool_by_period = {k: [m["code"] for m in v["members"]] for k, v in pools.items()}
        mask = np.zeros((self.T, self.C), dtype=bool)
        cur = np.zeros(self.C, dtype=bool)
        ai = 0
        for i, d in enumerate(self.dates):
            while ai < len(asof) and pd.Timestamp(asof[ai][0]) <= d:
                per = asof[ai][1]
                cur = np.zeros(self.C, dtype=bool)
                for c in pool_by_period.get(per, []):
                    if c in self.code_i:
                        cur[self.code_i[c]] = True
                ai += 1
            mask[i] = cur
        self.pool_mask = mask
        self.asof = asof
        self.pool_by_period = pool_by_period

        meta = json.load(open(os.path.join(SRC, "scripts", "meta.json"), encoding="utf-8"))
        sw3 = [str(meta.get(c, {}).get("sw3") or "未分类") for c in self.codes]
        uniq = {s: i for i, s in enumerate(sorted(set(sw3)))}
        self.ind = np.array([uniq[s] for s in sw3])
        self.sw3name = sw3
        self.name = [str(meta.get(c, {}).get("name") or c) for c in self.codes]


# --------------------------------------------------------------------------- #
# 绩效
# --------------------------------------------------------------------------- #
def perf(nav: np.ndarray, label="", rf: float = 0.0):
    r = nav[1:] / nav[:-1] - 1.0
    r = r[np.isfinite(r)]
    if len(r) < 10:
        return {}
    eq = nav[-1]
    n = len(r)
    ann = eq ** (252.0 / n) - 1.0
    vol = r.std(ddof=0) * np.sqrt(252.0)
    mdd = (nav / np.maximum.accumulate(nav) - 1.0).min()
    ex = r - rf / 252.0
    sharpe = (ex.mean() / ex.std(ddof=0) * np.sqrt(252.0)) if ex.std(ddof=0) > 0 else np.nan
    calmar = ann / abs(mdd) if mdd else np.nan
    return {"组合": label, "累计收益": eq - 1.0, "年化收益": ann, "年化波动": vol,
            "最大回撤": mdd, "夏普比率": sharpe, "卡玛比率": calmar, "交易日数": n}


# --------------------------------------------------------------------------- #
# 单次回测
# --------------------------------------------------------------------------- #
MODES = {
    # mode: (滞后交易日, 价格字段)
    "open1":  (1, "open"),    # T 日决策、T+1 开盘执行
    "close1": (1, "close"),   # T 日决策、T+1 收盘执行
    "close0": (0, "close"),   # T 日决策、T 日收盘执行（项目原口径）
}


def run(d: Data, N: int, K: int, L: int, mode: str = "close1",
        policy: str = "redistribute", max_wait: int = None, start=START,
        cap: int = None, ignore_limit: bool = False, cost: float = 0.001,
        detail: bool = False, end=None):
    """mode   : 'open1' | 'close1' | 'close0'
       policy : 'cash' | 'redistribute' | 'substitute' | 'wait'
       cap    : 单个申万三级行业最多几只（None = 不限制）
       cost   : 单边交易成本（佣金+印花税+冲击），按换手额双边计收
       detail : True 时额外返回逐期持仓明细（供页面表格使用）
    """
    lag, field = MODES[mode]
    # 成交价按执行时点取（开盘/收盘），但组合一律以「每日收盘价」盯市，
    # 保证三种执行时点的日收益序列同为 close-to-close，波动率可比
    pexec = d.open_f if field == "open" else d.close_f
    pmark = d.close_f
    # 开盘模式：开盘即涨停 → 买不到；收盘模式：收盘封板 → 买不到
    base_seal = d.seal_open if field == "open" else d.seal_close
    tradable = (~base_seal) & d.has
    if ignore_limit:                                   # 校验用：连停牌也当作可成交
        tradable = np.ones_like(d.has)

    mom, adj = d.mom_adj(N)

    start_pos = int(np.searchsorted(d.dates.values, np.datetime64(start)))
    end_pos = (d.T - 1) if end is None else \
        int(np.searchsorted(d.dates.values, np.datetime64(end)))
    reb = list(range(start_pos, end_pos - lag + 1, K))  # 决策日

    nav = np.ones(d.T)
    stat = {"blocked": 0, "blocked_all": 0, "blocked_late": 0, "target": 0,
            "wait_filled": 0, "wait_failed": 0, "sub_used": 0, "cash_w": 0.0,
            "n_period": 0, "turnover": 0.0, "cost": 0.0, "avg_n": 0.0}
    # 明细收集（detail=True）
    detail_rows, removed_rows = [], []
    entry_exec, prev_hold = {}, set()

    def flush_removed(cur_codes, e_now):
        """上期持有、本期不再持有 → 记录「建仓执行日 → 本期执行日」的开盘价涨跌幅。"""
        if not detail:
            return
        for c in sorted(prev_hold - set(cur_codes)):
            ee = entry_exec.pop(c, None)
            p0 = d.open_f[ee][c] if ee is not None else np.nan
            p1 = d.open_f[e_now][c]
            chg = (p1 / p0 - 1.0) if (np.isfinite(p0) and np.isfinite(p1) and p0 > 0) else np.nan
            removed_rows.append({
                "调仓日": str(d.dates[p].date()), "代码": d.codes[c],
                "名称": d.name[c], "申万三级": d.sw3name[c],
                "建仓日": (str(d.dates[max(ee - lag, 0)].date()) if ee is not None else None),
                "剔除日": str(d.dates[e_now].date()), "区间涨跌幅": chg})

    prev_codes = np.array([], dtype=int)
    prev_shares = np.zeros(0)        # 上一期持有的份额（以 V_buy 为单位）
    prev_cash = np.zeros(0)          # 上一期未成交、停在现金里的部分
    V_pre = 1.0                      # 上一期持仓在本期执行价下的市值

    for j, p in enumerate(reb):
        e = p + lag
        e2 = (reb[j + 1] + lag) if j + 1 < len(reb) else end_pos
        if e2 <= e:
            break
        stat["n_period"] += 1

        # 上期持仓按本期执行价平仓后的可用资金
        if len(prev_codes):
            vv = prev_shares * pexec[e][prev_codes] + prev_cash
            vv = np.where(np.isfinite(vv), vv, prev_cash)
            V_pre = float(vv.sum())
            wd = {int(c): vv[k2] / V_pre for k2, c in enumerate(prev_codes)} if V_pre > 0 else {}
        else:
            wd = {}

        ok = d.pool_mask[p] & (d.listed[p] >= MIN_LISTED)
        m = mom[p]
        a = adj[p]
        cand = np.where(ok & np.isfinite(m) & np.isfinite(a) & (m > 0))[0]
        if len(cand) == 0:
            flush_removed([], e)
            prev_hold = set()
            nav[e:e2 + 1] = V_pre
            V_pre = nav[e2]
            prev_codes = np.array([], dtype=int)
            prev_shares = np.zeros(0)
            prev_cash = np.zeros(0)
            continue
        order = cand[np.argsort(-a[cand])]
        if cap:                                        # 单行业最多 cap 只
            picked, cnt = [], {}
            for c in order:
                if len(picked) >= L:
                    break
                i = d.ind[c]
                if cnt.get(i, 0) >= cap:
                    continue
                cnt[i] = cnt.get(i, 0) + 1
                picked.append(c)
            top = np.array(picked, dtype=int)
        else:
            top = order[:L]
        if len(top) == 0:
            flush_removed([], e)
            prev_hold = set()
            nav[e:e2 + 1] = V_pre
            V_pre = nav[e2]
            continue
        w_raw = a[top]
        tot = w_raw.sum()
        if not np.isfinite(tot) or tot <= 0:
            w_raw = np.ones(len(top))
            tot = len(top)
        w_top = w_raw / tot

        tr_e = tradable[e]
        blocked_mask = ~tr_e[top]
        n_block = int(blocked_mask.sum())
        stat["target"] += len(top)
        stat["blocked"] += n_block
        stat["blocked_all"] += int((d.seal_all[e][top] & blocked_mask).sum())
        stat["blocked_late"] += int((d.seal_late[e][top] & blocked_mask).sum())

        codes = list(top)
        w = w_top.copy()
        # 现金兜底权重：建仓被阻断（或递延未成交）时，这部分资金以现金形式继续留在
        # 组合里（收益为 0），而不是凭空消失。w 是「实际投到个股」的权重，
        # wcash 是「没投出去、趴在账上」的权重，两者按槽位互补。
        wcash = np.zeros(len(top))

        if n_block == 0:
            pass
        elif policy == "cash":
            # 权重空置：被阻断的槽位直接留现金，不递补、也不摊给其余个股
            w[blocked_mask] = 0.0
            wcash[blocked_mask] = w_top[blocked_mask]
        elif policy == "redistribute":
            keep = ~blocked_mask
            s = w[keep].sum()
            if s > 0:
                w[keep] = w[keep] / s
            w[blocked_mask] = 0.0      # 资金已摊给其余个股 → 不留现金
        elif policy == "substitute":
            # 依次递补：按排名从 L 位之后取可成交的替代，权重统一按 adj 重新归一化
            extra = [c for c in order[L:] if tr_e[c]]
            need = int(blocked_mask.sum())
            take = extra[:need]
            fill_idx = np.where(blocked_mask)[0]
            for k2, c in enumerate(take):
                codes[fill_idx[k2]] = c
            stat["sub_used"] += len(take)
            codes = np.array(codes)
            w = a[codes].astype(float).copy()          # 统一用原始 adj 重新定权
            if len(take) < need:                       # 递补不足 → 空出的槽位留现金
                w[fill_idx[len(take):]] = 0.0
                wcash[fill_idx[len(take):]] = w_top[fill_idx[len(take):]]
            s = w.sum()
            if s > 0:
                w = w / s
            else:
                w = np.zeros(len(codes))
        elif policy == "wait":
            wcash = w_top.copy()       # 全部递延，成交前按现金计

        codes = np.array(codes)
        # 建仓日
        if policy == "wait":
            mw = max_wait if max_wait is not None else (e2 - e)
            hi = min(e + mw, e2)                       # 最晚等待日
            fill = np.full(len(codes), -1, dtype=int)
            for k2, c in enumerate(codes):
                if tr_e[c]:
                    fill[k2] = e
                    continue
                seg = tradable[e + 1:hi + 1, c] if hi > e else np.array([], dtype=bool)
                if seg.any():
                    fill[k2] = e + 1 + int(np.argmax(seg))
            stat["wait_filled"] += int((fill > e).sum())
            stat["wait_failed"] += int((fill < 0).sum())
        else:
            fill = np.where(tr_e[codes], e, -1)

        stat["cash_w"] += float(wcash[fill < 0].sum())
        stat["avg_n"] += float((fill >= 0).sum())

        # ---------- 明细 ----------
        if detail:
            cur = [int(c) for c in codes]
            day_s = str(d.dates[p].date())
            for k2, c in enumerate(cur):
                if c not in entry_exec:
                    entry_exec[c] = e
                detail_rows.append({
                    "调仓日": day_s, "执行日": str(d.dates[e].date()), "次序": k2 + 1,
                    "代码": d.codes[c], "名称": d.name[c], "申万三级": d.sw3name[c],
                    "原始动量": float(m[c]), "调整动量": float(a[c]),
                    "目标权重": float(w_top[k2]), "实际权重": float(w[k2]),
                    "建仓状态": "正常" if fill[k2] >= 0 else "涨停空置",
                    "状态": "新增" if c not in prev_hold else "持有"})
            flush_removed(cur, e)
            prev_hold = set(cur)

        # ---------- 换手与交易成本 ----------
        tg = {int(c): w[k2] for k2, c in enumerate(codes)}
        turn = sum(abs(tg.get(c, 0.0) - wd.get(c, 0.0))
                   for c in set(list(wd.keys()) + list(tg.keys())))
        stat["turnover"] += turn
        cost_now = turn * cost
        stat["cost"] += cost_now
        V_buy = V_pre * max(1.0 - cost_now, 0.0)

        # ---------- 建仓（份额记账）+ 逐日盯市（收盘价）----------
        pf = np.array([pexec[fill[k2], codes[k2]] if fill[k2] >= 0 else np.nan
                       for k2 in range(len(codes))])
        shares = np.zeros(len(codes))
        imm = fill == e                                    # 当日立即成交
        shares[imm] = w[imm] * V_buy / pf[imm]
        # 递延（等开板）：成交日把该笔现金按当日执行价换成份额；未成交前按现金计。
        # 未成交槽位的现金按 wcash 计（cash 口径下 = 原始目标权重），不能用置零后的 w
        cash0 = np.where(imm, 0.0, np.where(fill >= 0, w, wcash) * V_buy)
        for k2 in np.where(fill > e)[0]:
            px = pexec[fill[k2], codes[k2]]
            if np.isfinite(px) and px > 0:
                shares[k2] = cash0[k2] / px
                cash0[k2] = 0.0
            else:
                fill[k2] = -1                              # 无法按价成交 → 视作未成交

        sub = pmark[e:e2 + 1][:, codes]                    # (days, n) 收盘价
        day_idx = np.arange(e, e2 + 1)[:, None]
        with np.errstate(invalid="ignore"):
            ratio = sub / pf[None, :]
        mask = (fill[None, :] >= 0) & (day_idx >= fill[None, :])
        ratio = np.where(mask, np.where(np.isfinite(ratio), ratio, 1.0), 0.0)
        # 市值 = 目标权重×建仓资金 ×(收盘价/成交价)
        hold = ratio * (w * V_buy)[None, :]
        # 现金部分：永远没成交 → 全程计现金；递延成交 → 成交日之前计现金
        cash_part = np.where(
            (fill[None, :] < 0) |
            ((fill[None, :] > 0) & (day_idx < np.maximum(fill, 0)[None, :])),
            (wcash * V_buy)[None, :], 0.0)
        nav[e:e2 + 1] = hold.sum(axis=1) + cash_part.sum(axis=1)

        prev_codes, prev_shares, prev_cash = codes, shares, cash0

    nav[:reb[0] + lag] = 1.0
    nav[end_pos:] = nav[end_pos]      # 截断时尾部填平，便于直接算绩效
    if detail:
        return nav, stat, detail_rows, removed_rows
    return nav, stat
