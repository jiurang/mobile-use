"""执行快照（LangGraph checkpointer）的路径与开关。

它是 resume 的缓存，不是会话真相：事件日志说「发生了什么」，checkpoint 说
「图跑到哪一步」。两者都丢了，才需要整任务重跑。
"""

from __future__ import annotations

import os
from pathlib import Path

_DISABLED_VALUES = frozenset({"0", "false", "off", "no"})


def checkpoint_enabled() -> bool:
    """``MOBILE_USE_CHECKPOINT=0`` 关闭执行快照；默认开。"""
    value = (os.environ.get("MOBILE_USE_CHECKPOINT") or "").strip().lower()
    return value not in _DISABLED_VALUES


def checkpoint_db_path() -> Path:
    """快照库路径：默认当前工作目录下的 ``.mobile-use/checkpoints.db``。"""
    override = (os.environ.get("MOBILE_USE_CHECKPOINT_DB") or "").strip()
    if override:
        return Path(override).expanduser()
    return Path.cwd() / ".mobile-use" / "checkpoints.db"
