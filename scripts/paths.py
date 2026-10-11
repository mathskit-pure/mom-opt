"""mom_opt 统一路径解析：本机 Windows 与云端 Linux 共用一份代码。

环境变量优先，其次按平台给默认值：
  Windows 本机  —— 沿用既有 WorkBuddy 目录，保证改造后本机行为不变
  Linux 云端    —— 由 bootstrap 克隆到 /workspace 下的兄弟目录

可用变量：MOM_OPT_DIR（本工程根）、POOL_BASE（主项目 pool-core 根）
"""
from __future__ import annotations

import os
from pathlib import Path

# 本工程根目录（scripts/ 的上一级）
BASE = Path(os.environ.get("MOM_OPT_DIR") or Path(__file__).resolve().parent.parent)

_WIN_POOL = r"C:\Users\12066\WorkBuddy\2026-10-05-23-13-30"

# 主项目（舞弊调整质量因子 pool-core）：股池与代码清单的来源
SRC = Path(os.environ.get("POOL_BASE") or (_WIN_POOL if os.name == "nt" else BASE.parent / "pool-core"))

DATA = BASE / "data"
SCRIPTS = BASE / "scripts"

if __name__ == "__main__":
    print(f"BASE = {BASE}")
    print(f"SRC  = {SRC}")
    print(f"DATA = {DATA}")
