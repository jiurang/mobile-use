"""进展检测：判断 executor 是不是在原地打转。

为什么要有它
------------
2026-10-26 真机实测（mobile-manager execution 880，任务「淘宝红包签到」）：
executor 在「去头条刷热点领现金(0/1)」这一条上卡了 **70 步** ——

    launch_app 今日头条 → 等 15~31 秒 → 滑屏 → Back（头条拦返回，弹「再按一次退出」）
    → launch_app 淘宝 → 核对「还是 0/1，已攒仍是 51415」→ 再 launch_app 今日头条 → ……

它的动作**全都执行成功** ⇒ 失败账本里一条事实都没有，熔断不触发；
它也**从来没发过 done**（`done` 的语义是「目标达成」）⇒ 图里没有出口，
看门狗（cortex / orchestrator）**从来没被叫醒** —— 最后一个 `cortex` node_start
停在第 3 个子目标下发之前，之后 2436 个事件全是 executor 空转，
直到撞 `recursion_limit` 崩掉（45 分钟 / 198 次调用 / 409 万 input token）。

⇒ 缺的不是「步数上限」，是**「没有进展」这件事本身没人说**。

这个模块只做一件事：把「没有进展」变成**可观察的事实**。

它**不做任何硬拦截**（不截断、不腰斩、不改结果）—— 只产出信号，
由上层决定拿它怎么办。理由见 `doc/executor-autonomy-plan.md` 的 N34：
限制会把「打转」腰斩成「走不通」，制造新因果；而**说清楚**没有代价。

两个判据
--------
1. **同屏重复**（``screen``）：最近若干步里出现了同一个屏幕指纹 ≥ N 次。
   屏幕指纹 = 元素表的**结构**（resource_id / 静态文字，数字归一化）——
   「同一个页面、同样的内容」是「没进展」最直接的证据。
2. **同动作重复**（``action``）：最近若干步里出现了同一个动作指纹 ≥ N 次。

两个都只产出**信号**，不产出裁决。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from minitap.mobile_use.utils.screen_fingerprint import normalize_resource_id

#: 判「重复」时看最近多少步。
#:
#: 为什么是 8：太短会把「连着点两下同一个按钮」当成打转；
#: 太长会让「早就换页面了」也算进来。8 步 ≈ 一个子目标里一小段尝试。
DEFAULT_WINDOW = 8

#: 同一指纹在窗口里出现多少次才算「重复」。
#:
#: 为什么是 3：出现 2 次是巧合（点两下、翻两页），3 次才是模式。
DEFAULT_REPEAT_THRESHOLD = 3

#: 同一子目标里最多提醒几次「你在打转」。
#:
#: 提醒是给模型看的**信息**，不是限制；但每次提醒都要多花一次调用的钱，
#: 而且反复念叨会让模型烦躁、乱改策略。所以有上限。
MAX_NUDGES_PER_SUBGOAL = 3

#: 两次提醒之间至少隔多少步。
NUDGE_COOLDOWN_STEPS = 5

_DIGITS = re.compile(r"\d+")
_WHITESPACE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    """数字归一化 + 空白折叠。

    为什么归一化数字：页面上的计数器（「已攒 51415 元宝」）每时每刻都在变，
    留着它，**同一个页面每次的指纹都不一样** ⇒ 永远检测不到「回到了同一页」。
    数字是**状态**，不是**身份** —— 身份归一化掉，状态变化才不会被误判成换页。
    """
    return _WHITESPACE.sub(" ", _DIGITS.sub("#", text or "")).strip()


def screen_state_fingerprint(
    elements: list[Any] | None, ocr_results: list[Any] | None = None
) -> str:
    """屏幕指纹：这一屏「是哪个页面」。

    用**结构**（resource-id / 静态文字）而不是完整文本 ——
    完整文本每次都在变（时间、计数、滚动位置），变了就永远认不出「同一页」。

    * UI 元素 → **归一化后的 id 尾段**（复用
      ``utils/screen_fingerprint.normalize_resource_id``：
      ``like_button`` 和 ``like_element_click_layout`` 都归一化成 ``like``
      ⇒ app 改版换了 id 尾段也不影响「是不是同一页」的判断）
    * OCR 行 → 归一化后的文字（数字全部换成 ``#``）

    指纹相同 ⇒ 认为是同一屏。
    """
    parts: list[str] = []

    for el in elements or []:
        get = el.get if isinstance(el, dict) else lambda k, d=None: getattr(el, k, d)
        rid = normalize_resource_id(str(get("resource_id") or ""))
        label = str(get("label") or "")
        token = rid or _normalize_text(label)[:40]
        if token:
            parts.append(token)

    for row in ocr_results or []:
        get = row.get if isinstance(row, dict) else lambda k, d=None: getattr(row, k, d)
        token = _normalize_text(get("text") or "")[:40]
        if token:
            parts.append(token)

    if not parts:
        return ""
    digest = hashlib.sha1("|".join(sorted(parts)).encode("utf-8", "ignore")).hexdigest()
    return digest[:12]


def action_fingerprint(action: dict | None) -> str:
    """动作指纹：这一步「做的是哪件事」。

    归一化掉会变的部分（``index``、坐标、时长），只留**身份**：
    工具名 + 它指向的那个东西（app 名 / 按键 / resource_id / 文字）。

    ``index`` 为什么要归一化掉：元素表每屏重新编号，同一个按钮在两屏上是
    两个 index；留着它，「重复点同一个按钮」就永远认不出来。
    """
    if not action:
        return ""
    name = str(action.get("action") or "").strip()
    if not name:
        return ""

    detail = ""
    for key in ("app_name", "key", "resource_id", "text", "url"):
        value = action.get(key)
        if value:
            detail = f"{key}={value}"
            break
    else:
        target = action.get("target")
        if isinstance(target, dict):
            for key in ("resource_id", "text"):
                if target.get(key):
                    detail = f"{key}={target[key]}"
                    break

    return f"{name}:{detail}" if detail else name


@dataclass(frozen=True)
class StallSignal:
    """「没有进展」这个事实的描述。

    ⚠️ 它**只是事实**，不含裁决 —— 「所以该怎么办」是上层的事。
    """

    kind: str  # "screen" | "action"
    fingerprint: str  # 重复的那个指纹
    repeats: int  # 窗口内出现了几次
    window: int  # 窗口多大
    steps_in_subgoal: int  # 这个子目标一共走了多少步
    nudges_sent: int  # 已经提醒过几次

    @property
    def description(self) -> str:
        if self.kind == "screen":
            return (
                f"最近 {self.window} 步里，你回到了 {self.repeats} 次**同一个页面**"
                f"（屏幕指纹 {self.fingerprint}）—— 页面上的东西没有变化。"
            )
        return (
            f"最近 {self.window} 步里，你重复了 {self.repeats} 次**同一个动作**"
            f"（{self.fingerprint}）—— 结果没有变化。"
        )


def detect_stall(
    progress: list[dict] | None,
    *,
    window: int = DEFAULT_WINDOW,
    repeat_threshold: int = DEFAULT_REPEAT_THRESHOLD,
) -> StallSignal | None:
    """看最近 ``window`` 步有没有「同屏重复」或「同动作重复」。

    ``progress`` 是本子目标内每步一条的记录（``{"action":…, "screen":…}``）。
    返回 ``None`` 表示没有可观察的重复。

    ⚠️ **只报信号，不下裁决。** 重复可能完全正常（连点两下、翻三页），
    所以阈值取得保守（窗口内 ≥3 次）；而且这个信号**只用来提醒模型**
    「你可以用 failed 明确报告做不到」，**不截断**、**不改结果**。
    """
    entries = [e for e in (progress or []) if isinstance(e, dict)]
    if len(entries) < window:
        return None

    recent = entries[-window:]
    steps_in_subgoal = len(entries)
    nudges_sent = sum(1 for e in entries if e.get("nudged"))

    for kind, key in (("screen", "screen"), ("action", "action")):
        counts: dict[str, int] = {}
        for e in recent:
            fp = str(e.get(key) or "")
            if not fp:
                continue
            counts[fp] = counts.get(fp, 0) + 1
        if not counts:
            continue
        fingerprint, repeats = max(counts.items(), key=lambda kv: kv[1])
        if repeats >= repeat_threshold:
            return StallSignal(
                kind=kind,
                fingerprint=fingerprint,
                repeats=repeats,
                window=window,
                steps_in_subgoal=steps_in_subgoal,
                nudges_sent=nudges_sent,
            )

    return None


def should_nudge(signal: StallSignal, *, cooldown: int = NUDGE_COOLDOWN_STEPS) -> bool:
    """这次要不要**再**提醒一次。

    * 同一个子目标里最多提醒 ``MAX_NUDGES_PER_SUBGOAL`` 次 —— 提醒是信息，
      不是限制；但反复念叨会让模型乱改策略。
    * 两次提醒之间至少隔 ``cooldown`` 步 —— 否则连续几步都在提醒，
      模型会把提示词里那几行当成噪音。
    """
    if signal.nudges_sent >= MAX_NUDGES_PER_SUBGOAL:
        return False
    return signal.steps_in_subgoal - signal.nudges_sent * cooldown >= cooldown


__all__ = [
    "DEFAULT_REPEAT_THRESHOLD",
    "DEFAULT_WINDOW",
    "MAX_NUDGES_PER_SUBGOAL",
    "NUDGE_COOLDOWN_STEPS",
    "StallSignal",
    "action_fingerprint",
    "detect_stall",
    "screen_state_fingerprint",
    "should_nudge",
]
