# -*- coding: utf-8 -*-
"""
从通达信拉取池内个股的原始（不复权）日 K OHLC。

用途：
  1) 涨停/跌停判定必须在「原始价格空间」做（涨停价 = 前收盘 × (1+涨跌幅限制)）；
  2) 「T+1 开盘买入」需要当日开盘价。

后复权开盘价的构造：
  factor[t] = close_hfq[t] / close_raw[t]   →   open_hfq[t] = open_raw[t] * factor[t]
后复权是乘法链，因此该因子逐日吸收了除权除息造成的跳空，与既有 close_hfq 完全同口径。
"""
import argparse
import itertools
import os
import queue
import sys
import threading
import time

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = r"C:\Users\12066\WorkBuddy\2026-10-09-23-54-07\mom_opt"
SRC = r"C:\Users\12066\WorkBuddy\2026-10-05-23-13-30"
OUT = os.path.join(BASE, "data", "ohlc_raw.csv")

HOSTS = [(f"117.34.114.{i}", 7709) for i in (13, 14, 15, 16, 17, 18, 20, 27)]
TIMEOUT = 8
WORKERS = 8
BARS_PER_CALL = 800
# 需要覆盖到 2019-01-01；到 2026-10 约 1900 个交易日，取 2400 根足够
TOTAL_BARS = 2400
N_BARS = TOTAL_BARS      # 本次实际要取的根数（增量模式会调小）

from pytdx.hq import TdxHq_API


def market_of(code: str) -> int:
    if code.startswith("sh"):
        return 1
    if code.startswith("sz"):
        return 0
    if code.startswith("bj"):
        return 2
    return 1


def fetch_one(code: str):
    mkt = market_of(code)
    sym = code[2:]
    rows = []
    cycle = itertools.cycle(HOSTS)
    for attempt in range(4):
        host = next(cycle)
        api = TdxHq_API(heartbeat=False, auto_retry=False, raise_exception=False)
        try:
            api.connect(host[0], host[1], time_out=TIMEOUT)
            got = []
            start = 0
            while start < N_BARS:
                b = api.get_security_bars(9, mkt, sym, start, BARS_PER_CALL)
                if not b:
                    break
                got.extend(b)
                if len(b) < BARS_PER_CALL:
                    break
                start += BARS_PER_CALL
            if got:
                for r in got:
                    d = r.get("datetime")
                    if hasattr(d, "strftime"):
                        ds = d.strftime("%Y-%m-%d")
                    else:
                        ds = str(d)[:10]
                    rows.append({
                        "code": code,
                        "date": ds,
                        "open": float(r["open"]),
                        "high": float(r["high"]),
                        "low": float(r["low"]),
                        "close": float(r["close"]),
                        "vol": float(r.get("vol", 0) or 0),
                    })
                break
        except Exception:
            pass
        finally:
            try:
                api.disconnect()
            except Exception:
                pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--incr", action="store_true",
                    help="增量：已存在代码的日 K 只补齐末尾新交易日，不全量重取")
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()

    codes = [c.strip() for c in open(os.path.join(BASE, "data", "codes.txt"),
                                     encoding="utf-8") if c.strip()]
    if args.incr:
        todo = list(codes)                      # 增量：所有代码都要检查末尾
        global N_BARS
        N_BARS = 60          # 只需末尾增量；60 根足够覆盖假期+停牌
        print("增量模式：%d 只标的" % len(todo), flush=True)
    else:
        # 全量：已完成的跳过
        done = set()
        if os.path.exists(OUT):
            try:
                prev = pd.read_csv(OUT, dtype={"code": str}, usecols=["code"])
                done = set(prev["code"].unique())
            except Exception:
                done = set()
        todo = [c for c in codes if c not in done]
        print(f"标的 {len(codes)} 只，待取 {len(todo)} 只", flush=True)
    if not todo:
        return

    q = queue.Queue()
    for c in todo:
        q.put(c)
    lock = threading.Lock()
    results = []
    counter = {"n": 0}

    def worker():
        while True:
            try:
                c = q.get_nowait()
            except queue.Empty:
                return
            rows = fetch_one(c)
            with lock:
                results.extend(rows)
                counter["n"] += 1
                if counter["n"] % 20 == 0:
                    print(f"  {counter['n']}/{len(todo)}  累计行 {len(results)}", flush=True)

    ts = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print(f"取数完成 {time.time()-t0:.0f}s，{len(results)} 行", flush=True)

    df = pd.DataFrame(results)
    if df.empty:
        print("无数据", flush=True)
        return
    df = df.drop_duplicates(["code", "date"]).sort_values(["code", "date"])

    if args.incr:
        old = pd.read_csv(OUT, dtype={"code": str})
        old["date"] = pd.to_datetime(old["date"]).dt.strftime("%Y-%m-%d")
        # 只保留「比该代码已有最后日期更新」的行，历史部分一律不动，
        # 避免重取时的复权/撮合差异污染已发布曲线。
        lastmap = old.groupby("code")["date"].max().to_dict()
        key = df["code"].map(lastmap)
        df = df[(key.isna()) | (df["date"] > key)]
        if df.empty:
            print("原始 OHLC 已最新，无需追加", flush=True)
            return
        all_df = pd.concat([old, df], ignore_index=True)
        all_df = (all_df.drop_duplicates(["code", "date"], keep="last")
                        .sort_values(["code", "date"]))
        all_df.to_csv(OUT, index=False, encoding="utf-8")
        print("增量追加 %d 行 → 合计 %d 行" % (len(df), len(all_df)), flush=True)
        return

    header = not os.path.exists(OUT)
    df.to_csv(OUT, mode="a", index=False, header=header, encoding="utf-8")
    print("已写入", OUT, len(df), "行", flush=True)


if __name__ == "__main__":
    main()
