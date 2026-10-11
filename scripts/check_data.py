# -*- coding: utf-8 -*-
import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from engine import Data

d = Data()
print("矩阵:", d.close.shape, d.dates.min().date(), "→", d.dates.max().date())
raw_cov = np.isfinite(d.close_raw.values).mean()
print(f"原始日K覆盖率: {raw_cov:.4f}")
print(f"复权因子异常(<=0): {(np.isfinite(d.close.values/d.close_raw.values) & ((d.close.values/d.close_raw.values)<=0)).sum()}")

n = d.T * d.C
print("\n=== 涨停统计（全池 2019-01-02 → 最新）===")
print(f"收盘封板   : {d.seal_close.sum():>7} ({d.seal_close.sum()/n*100:.3f}%)")
print(f"  其中全天封板: {d.seal_all.sum():>7} ({d.seal_all.sum()/max(d.seal_close.sum(),1)*100:.1f}% of 封板)")
print(f"  其中尾盘封板: {d.seal_late.sum():>7} ({d.seal_late.sum()/max(d.seal_close.sum(),1)*100:.1f}% of 封板)")
print(f"开盘涨停   : {d.seal_open.sum():>7}")
print(f"触板未封   : {d.touch.sum():>7}")
print(f"跌停       : {d.limit_dn.sum():>7}")

# 后复权开盘价合理性：|open/close - 1| 分布
r = (d.open.values / d.close.values)
r = r[np.isfinite(r)]
print("\n=== 后复权 open/close 比值分布 ===")
for q in [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]:
    print(f"  P{q:<5}: {np.percentile(r, q):.4f}")
print(f"  比值落在 [0.85,1.15] 占比: {((r>=0.85)&(r<=1.15)).mean()*100:.2f}%")

# 抽查一只
c = "sz300017"
j = d.code_i[c]
df = pd.DataFrame({"date": d.dates, "raw_c": d.close_raw.values[:, j],
                   "raw_o": d.open_raw.values[:, j], "hfq_c": d.close_f[:, j],
                   "hfq_o": d.open_f[:, j], "seal_close": d.seal_close[:, j],
                   "seal_all": d.seal_all[:, j]}).dropna()
print("\n=== 抽查", c, "===")
print(df.tail(5).to_string(index=False))
k = df.index[df["seal_close"]]
print("封板样本:")
print(df.loc[k[:3]].to_string(index=False))
