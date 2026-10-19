import random
import string

from minitap.mobile_use.agents.planner.types import Subgoal, SubgoalStatus
from datetime import datetime, UTC


def get_current_subgoal(subgoals: list[Subgoal]) -> Subgoal | None:
    return next((s for s in subgoals if s.status == SubgoalStatus.PENDING), None)


def get_subgoals_by_ids(subgoals: list[Subgoal], ids: list[str]) -> list[Subgoal]:
    return [s for s in subgoals if s.id in ids]


def get_next_subgoal(subgoals: list[Subgoal]) -> Subgoal | None:
    return next((s for s in subgoals if s.status == SubgoalStatus.NOT_STARTED), None)


def nothing_started(subgoals: list[Subgoal]) -> bool:
    return all(s.status == SubgoalStatus.NOT_STARTED for s in subgoals)


def complete_current_subgoal(subgoals: list[Subgoal]) -> list[Subgoal]:
    current_subgoal = get_current_subgoal(subgoals)
    if not current_subgoal:
        return subgoals
    current_subgoal.status = SubgoalStatus.SUCCESS
    current_subgoal.ended_at = datetime.now(UTC)
    return subgoals


def complete_subgoals_by_ids(subgoals: list[Subgoal], ids: list[str]) -> list[Subgoal]:
    for subgoal in subgoals:
        if subgoal.id in ids:
            subgoal.status = SubgoalStatus.SUCCESS
            subgoal.ended_at = datetime.now(UTC)
    return subgoals


def fail_current_subgoal(subgoals: list[Subgoal]) -> list[Subgoal]:
    current_subgoal = get_current_subgoal(subgoals)
    if not current_subgoal:
        return subgoals
    current_subgoal.status = SubgoalStatus.FAILURE
    current_subgoal.ended_at = datetime.now(UTC)
    return subgoals


def all_completed(subgoals: list[Subgoal]) -> bool:
    return all(s.status == SubgoalStatus.SUCCESS for s in subgoals)


def one_of_them_is_failure(subgoals: list[Subgoal]) -> bool:
    return any(s.status == SubgoalStatus.FAILURE for s in subgoals)


def start_next_subgoal(subgoals: list[Subgoal]) -> list[Subgoal]:
    next_subgoal = get_next_subgoal(subgoals)
    if not next_subgoal:
        return subgoals
    next_subgoal.status = SubgoalStatus.PENDING
    next_subgoal.started_at = datetime.now(UTC)
    return subgoals


def render_plan_skeleton(subgoals: list[Subgoal]) -> str:
    """计划骨架：只含 ID 与描述，**不含状态**。

    `Subgoal` 的状态是原地改写的（见上面 complete_subgoals_by_ids /
    start_next_subgoal / fail_current_subgoal），而 cortex 每轮都会重发这段
    文本。一旦把状态写进去，提示词前缀就会随状态流转而变，前缀缓存随之失效。
    状态改由 render_subgoal_status 以「计划序固定槽位 + 替换式」单独呈现，并排在
    thoughts 之后（见 cortex.py），所以它怎么变都不会击穿 thoughts 前缀。
    """
    if not subgoals:
        return "(no plan)"
    return "\n".join(f"- [ID:{subgoal.id}] {subgoal.description}" for subgoal in subgoals)


def render_subgoal_status(subgoals: list[Subgoal]) -> str:
    """当前计划的完成情况**索引**：按计划序列出每个子目标的状态。

    只输出「ID + 状态」，描述不重复 —— 描述在计划日志骨架（render_plan_skeleton）
    里，这里做索引即可。

    按**计划序**而非「完成序」：计划序是固定槽位，状态在固定槽位上原地替换，永远
    不会有新行插到行中间（乱序完成也不会）。cortex 把它排在 thoughts **之后**，
    所以它每轮怎么变都不会碰到 thoughts 那段唯一增长的前缀。

    空计划返回 "none"（不留空串漂移：这条消息在 cortex 里恒发，条数本身也是
    前缀的一部分）。
    """
    if not subgoals:
        return "none"
    labels = {
        SubgoalStatus.SUCCESS: "done",
        SubgoalStatus.FAILURE: "failed",
        SubgoalStatus.PENDING: "active",
        SubgoalStatus.NOT_STARTED: "pending",
    }
    return " | ".join(f"[{subgoal.id}] {labels[subgoal.status]}" for subgoal in subgoals)


def generate_id(length: int = 6) -> str:
    """Generates a small and distinct random string ID."""
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choice(chars) for _ in range(length))
