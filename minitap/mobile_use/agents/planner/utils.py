"""计划（``list[Subgoal]``）的读写原语。

所有函数都原地改写传入的 plan 并把它返回去，调用方既可以写
``plan = fail_current_subgoal(plan)``，也可以直接调用后继续用原来的变量。
返回的永远是同一个对象，不是副本。

状态和时间戳只在这里落。``status``、``started_at``、``ended_at`` 由这一处的
几个函数改写，别处只读。

给模型看的文本拆成两块：``render_plan_skeleton`` 渲染计划骨架，
``render_subgoal_status`` 渲染完成情况。状态每轮都变，计划日志只增不改，
两块分开渲染，前缀缓存才不会被状态流转击穿。
"""

import random
import string
from datetime import UTC, datetime

from minitap.mobile_use.agents.planner.types import Subgoal, SubgoalStatus


def get_current_subgoal(subgoals: list[Subgoal]) -> Subgoal | None:
    """正在执行的那个子目标，也就是状态为 PENDING 的那个。"""
    return next((s for s in subgoals if s.status == SubgoalStatus.PENDING), None)


def get_subgoals_by_ids(subgoals: list[Subgoal], ids: list[str]) -> list[Subgoal]:
    """按 ID 取出若干个子目标，保持传入的 ID 顺序。"""
    return [s for s in subgoals if s.id in ids]


def get_next_subgoal(subgoals: list[Subgoal]) -> Subgoal | None:
    """下一个该开始的子目标，也就是状态为 NOT_STARTED 的那个。"""
    return next((s for s in subgoals if s.status == SubgoalStatus.NOT_STARTED), None)


def nothing_started(subgoals: list[Subgoal]) -> bool:
    """计划里还没有任何子目标动过。"""
    return all(s.status == SubgoalStatus.NOT_STARTED for s in subgoals)


def complete_subgoals_by_ids(subgoals: list[Subgoal], ids: list[str]) -> list[Subgoal]:
    """把指定 ID 的子目标标成完成，并记下结束时间。"""
    for subgoal in subgoals:
        if subgoal.id in ids:
            subgoal.status = SubgoalStatus.SUCCESS
            subgoal.ended_at = datetime.now(UTC)
    return subgoals


def fail_current_subgoal(subgoals: list[Subgoal]) -> list[Subgoal]:
    """把正在执行的那个子目标标成失败，并记下结束时间。"""
    current_subgoal = get_current_subgoal(subgoals)
    if not current_subgoal:
        return subgoals
    current_subgoal.status = SubgoalStatus.FAILURE
    current_subgoal.ended_at = datetime.now(UTC)
    return subgoals


def all_completed(subgoals: list[Subgoal]) -> bool:
    """计划里的子目标是否全部完成。"""
    return all(s.status == SubgoalStatus.SUCCESS for s in subgoals)


def one_of_them_is_failure(subgoals: list[Subgoal]) -> bool:
    """计划里是否有子目标失败。这是触发重规划的信号。"""
    return any(s.status == SubgoalStatus.FAILURE for s in subgoals)


def start_next_subgoal(subgoals: list[Subgoal]) -> list[Subgoal]:
    """把下一个该开始的子目标置为进行中，并记下开始时间。"""
    next_subgoal = get_next_subgoal(subgoals)
    if not next_subgoal:
        return subgoals
    next_subgoal.status = SubgoalStatus.PENDING
    next_subgoal.started_at = datetime.now(UTC)
    return subgoals


def render_plan_skeleton(subgoals: list[Subgoal]) -> str:
    """计划骨架：只含 ID 与描述，不含状态。

    ``Subgoal`` 的状态是原地改写的（见 complete_subgoals_by_ids、
    start_next_subgoal、fail_current_subgoal），而 cortex 每轮都会重发这段
    文本。一旦把状态写进去，提示词前缀就会随状态流转而变，前缀缓存随之失效。
    状态改由 render_subgoal_status 以「计划序固定槽位加替换式」单独呈现，并排在
    thoughts 之后（见 cortex.py），所以它怎么变都不会击穿 thoughts 前缀。
    """
    if not subgoals:
        return "(no plan)"
    return "\n".join(f"- [ID:{subgoal.id}] {subgoal.description}" for subgoal in subgoals)


def render_subgoal_status(subgoals: list[Subgoal]) -> str:
    """当前计划的完成情况索引：按计划序列出每个子目标的状态。

    只输出 ID 和状态，不重复描述。描述在计划日志骨架（render_plan_skeleton）
    里，这里做索引即可。

    按计划序排，不按完成序排。计划序是固定槽位，状态在固定槽位上原地替换，
    不会有新行插到行中间，乱序完成也不会。cortex 把它排在 thoughts 之后，
    所以它每轮怎么变都不会碰到 thoughts 那段唯一增长的前缀。

    空计划返回 "none"，不留空串。这条消息在 cortex 里恒发，条数本身也是前缀
    的一部分。
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
    """生成一个短的随机 ID：默认 6 位，小写字母加数字。"""
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choice(chars) for _ in range(length))
