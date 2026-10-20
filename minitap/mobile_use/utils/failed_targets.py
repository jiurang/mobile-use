"""失败目标的账本：让「刚才那个目标已经不在屏上了」进入模型的下一轮上下文。

为什么需要它
------------
真机实测（2026-10-20，Redmi Turbo 4，exec 340）里，点击失败**全部**是同一类：

    no element with resource_id='com.kuaishou.nebula.corona_base_plugin:id/
    lv_bottom_like_view_container' on the current screen

也就是说，失败不是「点歪了」，而是**目标已经不在屏幕上了**（横屏全屏页被
视频播完带走了，模型思考了 17 秒才动手）。模型看到这条报错时的真实反应是
误诊成「索引 48 指向的是弹幕文本元素」——它把「东西没了」读成了「指错人了」。

所以账本要做两件事：

1. **把事实说清楚**：下一轮提示词里直接列出「这些目标已经不在屏上了，不要再用」。
2. **不让它原地打转**：同一个目标连续失败到阈值后，工具**不再执行**，
   直接要求换目标（省掉一次 1.2s 的 live dump，也避免点了别的东西）。

键（key）的取法
--------------
`index` + 元素身份（resource-id / 可见文字）拼在一起：

    index=48|id=com.x:id/like_button|text=点赞

只用 index 不行：索引每轮都会重新编号，同一个 48 在下一轮可能是完全不同的
元素，只看 index 会把「模型换了个目标」误判成「模型在重复犯同一个错」。
带上身份之后，只有「模型报了一模一样的目标」才会算连续失败。
"""

from __future__ import annotations

from pydantic import BaseModel

# 连续失败到第几次就不再执行、改为要求模型换目标
FAILURE_BREAKER_THRESHOLD = 2

# 账本最多留几条（超出就把最老的挤掉）
MAX_TRACKED_FAILURES = 5


class FailedTarget(BaseModel):
    """一次失败的目标，以及它连着失败了几次。"""

    key: str
    index: int | None = None
    resource_id: str | None = None
    label: str = ""
    count: int = 1
    reason: str = ""


def make_key(
    index: int | None,
    resource_id: str | None,
    label: str | None,
) -> str:
    """给一个目标算账本键。

    索引 + 身份一起进键：索引每轮重编号，只看它会把「换了目标」当成「重复犯错」。
    """
    parts: list[str] = []
    if index is not None:
        parts.append(f"index={index}")
    if resource_id:
        parts.append(f"id={resource_id}")
    if label:
        parts.append(f"text={label}")
    return "|".join(parts) or "unknown"


def describe(index: int | None, resource_id: str | None, label: str | None) -> str:
    """给人（和模型）看的一行描述。"""
    bits: list[str] = []
    if index is not None:
        bits.append(f"[{index}]")
    if label:
        bits.append(f'"{label}"')
    if resource_id:
        bits.append(resource_id.split("/")[-1])
    return " ".join(bits) if bits else "(unknown target)"


def record_failure(
    streak: list[FailedTarget] | None,
    *,
    key: str,
    index: int | None = None,
    resource_id: str | None = None,
    label: str | None = None,
    reason: str = "",
) -> list[FailedTarget]:
    """把一次失败记进账本，返回新的账本。

    同键连续失败 → 计数 +1；换了个目标 → 新起一条，**旧的清掉**
    （「连续」的意思就是上一次也失败在同一个目标上）。
    """
    items = list(streak or [])
    if items and items[-1].key == key:
        last = items[-1]
        items[-1] = last.model_copy(
            update={"count": last.count + 1, "reason": reason or last.reason}
        )
    else:
        items = [
            FailedTarget(
                key=key,
                index=index,
                resource_id=resource_id,
                label=label or "",
                count=1,
                reason=reason,
            )
        ]
    return items[-MAX_TRACKED_FAILURES:]


def breaker_reason(streak: list[FailedTarget] | None, key: str) -> str | None:
    """这个目标是不是已经连续失败到阈值了？是的话给出拒绝理由。"""
    items = streak or []
    if not items:
        return None
    last = items[-1]
    if last.key != key or last.count < FAILURE_BREAKER_THRESHOLD:
        return None
    return (
        f"target {describe(last.index, last.resource_id, last.label)} already failed "
        f"{last.count} times in a row ({last.reason}). It is not on the current screen "
        f"anymore. Do not report the same target again: re-read the screen and pick a "
        f"different element, scroll, go back, or complete the subgoal differently."
    )


def render_failures(streak: list[FailedTarget] | None) -> str:
    """渲染成给模型看的一行。空账本返回空串。"""
    items = streak or []
    if not items:
        return ""
    bits = [f"{describe(t.index, t.resource_id, t.label)} (failed {t.count}x)" for t in items]
    return "Targets that failed and are no longer on screen — do not reuse them: " + "; ".join(bits)
