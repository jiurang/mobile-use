"""时间相关的格式化工具。"""

from datetime import datetime


def convert_timestamp_to_str(ts: float) -> str:
    dt = datetime.fromtimestamp(ts)
    return dt.strftime("%Y-%m-%dT%H-%M-%S")


def format_task_started_at(dt: datetime | None) -> str | None:
    """把「任务开始时间」格式化成一行人话。

    格式：``2026-10-26 22:16:33+08:00``（本地时间 + 时区偏移）。

    为什么带时区偏移
    ----------------
    提示词里**同时**有任务开始时间（机器侧）和设备当前时间（手机侧）。
    两者是不是同一个钟，模型只能靠时区偏移自己判断 —— 不给偏移，
    「已经过去多久」这种推理就无从谈起。

    naive 时间（没有 tzinfo）按**本地时区**解释 —— 那是 ``datetime.now()``
    的默认语义。

    Args:
        dt: 任务开始时间。``None`` ⇒ 不知道，返回 ``None``。

    Returns:
        格式化后的字符串，或 ``None``（时间未知）。
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.astimezone()
    return dt.isoformat(sep=" ", timespec="seconds")
