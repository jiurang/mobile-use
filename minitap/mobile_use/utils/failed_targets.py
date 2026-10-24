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

from typing import TypedDict

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


class ActionFailureFact(TypedDict):
    """「上一步失败的那个目标」的身份 —— **只有身份，没有结论**。

    为什么是 TypedDict 而不是 ``dict[str, str]``：形状固定就这两个字段，
    而 ``dict[str, str]`` 让每个读取点都只能靠 ``.get("label")`` 猜键名 ——
    键名拼错了类型检查器也不吭声（规则：types-typeddict-over-dict-any）。

    ⚠️ **不带 index**：索引每轮重编号（同一个 48 在下一屏可能是完全不同的
    元素），把它写进给模型看的事实里会让模型去点一个完全无关的元素。
    """

    label: str
    resource_id: str


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


def _identity_bits(label: str | None, resource_id: str | None) -> list[str]:
    """目标身份的片段：可读文字 + resource-id 尾段。

    这是「身份长什么样」的**唯一定义** —— `describe`（带 index）和
    `describe_target`（不带 index）共用它，两边不会各写一份然后走样
    （规则：simplify-extract-after-duplication）。
    """
    bits: list[str] = []
    if label:
        bits.append(f'"{label}"')
    if resource_id:
        bits.append(resource_id.split("/")[-1])
    return bits


def describe(index: int | None, resource_id: str | None, label: str | None) -> str:
    """给人（和模型）看的一行描述。"""
    bits = [f"[{index}]"] if index is not None else []
    bits.extend(_identity_bits(label, resource_id))
    return " ".join(bits) if bits else "(unknown target)"


def describe_target(*, label: str | None, resource_id: str | None) -> str:
    """目标的身份 —— **不带 index**。

    为什么另写一个不带 index 的：索引**每轮重编号**（同一个 48 在下一屏可能是
    完全不同的元素）。把 index 写进给模型看的事实里，模型会以为「下一屏的
    48 号还是刚才那个」—— 于是去点一个完全无关的元素。

    ⇒ **给模型看的失败事实只带身份**（可读文字 + resource-id 尾段），
    不带 index。带 index 的 `describe()` 只留给**给人看**的地方
    （Reporter 的失败清单、遥测）。

    ⚠️ 这条只改**给模型的措辞**；`FailedTarget.index` 字段照旧保留 ——
    账本的键（`make_key`）还要用它区分「换了目标」和「重复犯错」。

    ⚠️ **keyword-only**：`describe()` 的参数顺序是 (index, resource_id, label)，
    而这里只有 (label, resource_id) —— 两处顺序相反，位置参数传错不会报错，
    只会渲染出另一个元素的身份（规则：api-keyword-only-params）。
    """
    bits = _identity_bits(label, resource_id)
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


# ---------------------------------------------------------------------------
# 跨子目标的失败事实（append-only）
# ---------------------------------------------------------------------------
#
# 为什么要和上面的熔断账本分开
# --------------------------
# `record_failure`（上面那个）的语义是「**连续**失败」：
#
#   - 同键 → count + 1
#   - **换键 → 整表替换**（`items = [新条目]`，旧的清掉）
#   - 而且任何一次**成功点击**都会把整个账本清空（`tools/mobile/tap.py` 的成功分支）
#
# 也就是说它是「**当下**这个目标连着失败了几次」的即时保护，
# 换个目标失败一次，之前那条就没了 —— 连熔断都会失效。
#
# 但「跨子目标的失败事实」要的是另一回事：
# 「刚才那个子目标里，索引 48 的 like 按钮点不到，它已经不在屏上了」——
# 这个事实必须活过子目标边界，让下一个子目标的 Executor 知道别再报它。
#
# 所以这里用**只增不减**的账本，和熔断账本物理隔开。
# 熔断器继续读 `state.failed_targets`（行为一个字没改），
# 汇报的事实层读这个新账本。


def append_failure(
    streak: list[dict] | None,
    *,
    key: str,
    index: int | None = None,
    resource_id: str | None = None,
    label: str | None = None,
    reason: str = "",
) -> list[dict]:
    """把一次失败**追加**到事实账本，返回新的账本。

    和 `record_failure` 的区别就是**不换键替换、不清空**：
    每一条失败都留着，跨子目标可见。

    条目用普通 dict 而不是 `FailedTarget`：`FailedTarget` 的 `count`
    表达的是「连续失败次数」，而这里的每条都是**独立的一次失败事件**，
    语义不同，混用会读错。
    """
    items = list(streak or [])
    items.append(
        {
            "key": key,
            "index": index,
            "resource_id": resource_id,
            "label": label or "",
            "reason": reason,
        }
    )
    return items


def render_failure_facts(streak: list[dict] | None) -> str:
    """把事实账本渲染成给模型看的一行（空账本返回空串）。

    和 `render_failures` 的区别：那条是给「**当下**的熔断」用的，
    只显示最后一条（因为熔断只看最后一次失败）。
    这里要把**全部**失败列出来 —— 模型要据此避开所有试过失败的目标。
    """
    items = streak or []
    if not items:
        return ""
    bits = [
        f"{describe(t.get('index'), t.get('resource_id'), t.get('label'))}"
        f" ({t.get('reason', '')[:80]})"
        for t in items
    ]
    return (
        "Targets that already failed in this task — do not reuse them, "
        "pick a different element: " + "; ".join(bits)
    )


def render_failures(streak: list[FailedTarget] | None) -> str:
    """渲染成给模型看的一行。空账本返回空串。"""
    items = streak or []
    if not items:
        return ""
    bits = [f"{describe(t.index, t.resource_id, t.label)} (failed {t.count}x)" for t in items]
    return "Targets that failed and are no longer on screen — do not reuse them: " + "; ".join(bits)
