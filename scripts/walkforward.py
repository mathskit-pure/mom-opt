# -*- coding: utf-8 -*-
"""
滚动前进（walk-forward）样本外检验
==================================
做法：用「截至第 t 年末」的全部数据做相位平均寻优（在同一网格里选夏普最高的参数），
      然后用选出的参数跑第 t+1 年，记录该年的样本外夏普。
目的：回答「相位平均夏普 1.04 里有多少是可持续的」。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from engine import Data, run, perf

MODE, POLICY, COST = "open1", "cash", 0.001
NS = [14, 16, 17, 18, 20]
KS = [7, 8, 9, 10]
LS = [70, 80, 90, 100]

d = Data()
PD = pd.DatetimeIndex(d.dates)
base = PD.get_indexer([pd.Timestamp("2019-05-06")], method="bfill")[0]
pd.set_option("display.width", 300)

# 训练期结束 → 验证年
FOLDS = [("2021-12-31", 2022), ("2022-12-30", 2023), ("2023-12-29", 2024),
         ("2024-12-31", 2025), ("2025-12-31", 2026)]


def phase_mean(N, K, L, end):
    vs = []
    for o in range(K):
        st = str(PD[base + o].date())
        nav, _ = run(d, N, K, L, mode=MODE, policy=POLICY, cost=COST, start=st, end=end)
        p = perf(nav)
        if p:
            vs.append(p["夏普比率"])
    return float(np.mean(vs)) if vs else np.nan


def year_sharpe(N, K, L, yr):
    """第 yr 年的样本外夏普（用该年 1 月 1 日起的净值重算）。"""
    a = "%d-01-01" % yr
    b = "%d-12-31" % yr if yr < 2026 else "2026-10-08"
    nav, _ = run(d, N, K, L, mode=MODE, policy=POLICY, cost=COST)
    s = pd.Series(nav, index=d.dates)
    s = s[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))]
    if len(s) < 30:
        return np.nan, np.nan
    v = s.values / s.values[0]
    return perf(v)["夏普比率"], perf(v)["累计收益"]


rows = []
for end, yr in FOLDS:
    best, bkey = -9, None
    for N in NS:
        for K in KS:
            for L in LS:
                v = phase_mean(N, K, L, end)
                if v == v and v > best:
                    best, bkey = v, (N, K, L)
    oos_sh, oos_cum = year_sharpe(*bkey, yr=yr)
    # 对照组：固定 N=17/K=8/L=90 与 用户指定 N=15/K=10/L=50
    f_sh, f_cum = year_sharpe(17, 8, 90, yr)
    u_sh, u_cum = year_sharpe(15, 10, 50, yr)
    rows.append({"训练截止": end, "验证年": yr,
                 "训练期选出的参数": "N%d/K%d/L%d" % bkey, "训练期夏普": round(best, 3),
                 "样本外夏普": round(float(oos_sh), 3), "样本外收益": round(float(oos_cum) * 100, 1),
                 "N17K8L90夏普": round(float(f_sh), 3), "N17K8L90收益": round(float(f_cum) * 100, 1),
                 "N15K10L50夏普": round(float(u_sh), 3), "N15K10L50收益": round(float(u_cum) * 100, 1)})
    print("  %s → %d 年：选出 %s（训练夏普 %.3f），样本外 %.3f"
          % (end, yr, bkey, best, oos_sh), flush=True)

W = pd.DataFrame(rows)
print("\n=== 滚动前进样本外检验 ===")
print(W.to_string(index=False))
print("\n样本外夏普均值 %.3f（选参组） / %.3f（固定 N17K8L90） / %.3f（固定 N15K10L50）"
      % (W["样本外夏普"].mean(), W["N17K8L90夏普"].mean(), W["N15K10L50夏普"].mean()))
print("训练期夏普均值 %.3f → 样本外衰减 %.0f%%"
      % (W["训练期夏普"].mean(),
         (1 - W["样本外夏普"].mean() / W["训练期夏普"].mean()) * 100))
W.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output",
                      "walkforward.csv"), index=False, encoding="utf-8-sig")
