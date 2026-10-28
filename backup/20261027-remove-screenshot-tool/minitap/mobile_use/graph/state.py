from typing import Annotated, Any

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel

from minitap.mobile_use.agents.executor.types import ExecutorReport
from minitap.mobile_use.agents.planner.types import Subgoal
from minitap.mobile_use.config import AgentNode
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.services.ocr.types import OcrText
from minitap.mobile_use.utils.failed_targets import ActionFailureFact, FailedTarget
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.recorder import record_interaction
from minitap.mobile_use.utils.screen_elements import ScreenElement

logger = get_logger(__name__)


def take_last(a, b):
    return b


def merge_dicts(a, b):
    """Merge two dicts; keys in b overwrite keys in a."""
    return {**a, **b}


class State(BaseModel):
    messages: Annotated[list[AnyMessage], "Sequential messages", add_messages]
    remaining_steps: Annotated[int | None, "Remaining steps before the task is completed"] = None

    # planner related keys
    initial_goal: Annotated[str, "Initial goal given by the user"]

    # orchestrator related keys
    subgoal_plan: Annotated[list[Subgoal], "The current plan, made of subgoals"]

    # planner related keys（计划日志）
    # 计划本身会被 replan 换掉（completed + new），直接把它渲染给 cortex 会让提示词
    # 从被改写那一行起整段失效。日志只追加：每生成一版计划追加一段，旧段永不改动。
    subgoal_plan_log: Annotated[
        str,
        "Append-only log of every generated plan; the last PLAN section is the active one",
    ] = ""
    plan_version: Annotated[int, "How many plans have been generated so far"] = 0

    # contextor related keys
    latest_ui_hierarchy: Annotated[
        list[dict] | None, "Latest UI hierarchy of the device", take_last
    ]
    latest_ocr_results: Annotated[
        list[OcrText] | None,
        "Latest OCR results of the device screen (text + bounds + confidence)",
        take_last,
    ]
    focused_app_info: Annotated[str | None, "Focused app info", take_last]
    device_date: Annotated[str | None, "Date of the device", take_last]
    screen_state: Annotated[
        str | None,
        "Screen power state: 'on' / 'off' / 'unknown'. "
        "A screen that is off shows a black screenshot and an element tree that "
        "only holds systemui leftovers.",
        take_last,
    ] = None

    # 屏幕元素表：UI 层级树 + OCR 文本合并后的统一索引表。
    #
    # 为什么它必须活到 executor：`Target.index` 要还原成「模型当时看到的那
    # 个元素」，而索引只在**那一次裁剪**上成立。执行时重新 dump 再重新编号
    # 是不行的 —— 屏可能已经变了，编号会和模型看到的不一致。
    # 所以这张表和 `latest_ocr_results` 一样，cortex 不清空它。
    latest_screen_elements: Annotated[
        list[ScreenElement] | None,
        "Screen element table (UI tree + OCR, one shared 1-based index)",
        take_last,
    ] = None

    # 上一次「动过屏幕」的动作发生在什么时候（epoch 秒）。
    #
    # contextor 用它算「距离上一个动作多久了」：不够久就先等一等再抓屏。
    # 真机实测换页后新控件要 1~2s 才挂上，抓早了会拿到一棵控件没挂全的树。
    # 只有真的动过屏幕的动作才记时间 —— 记笔记之类的动作不记。
    last_action_at: Annotated[
        float | None,
        "Epoch seconds when the last screen-changing action was dispatched",
        take_last,
    ] = None

    # 连续失败的目标账本（见 utils/failed_targets.py）。
    #
    # 它同时是**熔断器**和**提示词素材**：连着失败同一目标到阈值后，工具
    # 不再执行它（避免原地打转、也避免点了别的东西）；下一轮提示词里会把
    # 这些目标列出来，告诉模型「这些已经不在屏上了，不要再用」。
    failed_targets: Annotated[
        list[FailedTarget],
        "Consecutive failed targets (the last one may be blocking further attempts)",
        take_last,
    ] = []

    # cortex related keys
    structured_decisions: Annotated[
        str | None,
        "Structured decisions made by the cortex, for the executor to follow",
        take_last,
    ]
    complete_subgoals_by_ids: Annotated[
        list[str],
        "List of subgoal IDs to complete",
        take_last,
    ]

    # executor related keys
    executor_messages: Annotated[list[AnyMessage], "Sequential Executor messages", add_messages]
    cortex_last_thought: Annotated[str | None, "Last thought of the cortex for the executor"]

    # Executor 最近一次的**当屏截图**（base64，不带 data URI 前缀）。
    #
    # 为什么它不在 `executor_messages` 里
    # ------------------------------
    # **图片块会进 provider 的前缀缓存键**（真 API 实测，MiniMax-M3.1-Flash）：
    # 把请求中段的图片删掉或换一张，命中量立刻掉到「图之前」的位置。
    #
    # ⇒ 图一旦写进局部历史，它就被钉在历史中段；下一次拍图要「顶替」它时，
    # 那个位置的字节变了 ⇒ **前缀从历史中段起整段作废**。
    # 真机实测（execution 893）：14 次顶替让 47,181 个本可命中的 token 变满价。
    #
    # ⇒ 图单独存，**装配 prompt 时放到最后一条消息里**（元素表旁边 —— 那是
    # 每轮都变的易变区）。局部历史一个字节都不动 ⇒ 前缀永远稳定。
    #
    # 生命周期：拍图时覆盖（只留最新一张）；Cortex 下发新子目标时清空
    # （换目标 ⇒ executor 失忆 ⇒ 旧截图对应的是上一个目标的屏幕）。
    executor_screenshot: Annotated[
        str | None,
        "Latest on-screen screenshot (base64 JPEG, no data URI prefix)",
    ] = None

    # ---- Executor 自主执行（方案 v2，doc/executor-autonomy-plan.md §8）----
    #
    # 语义变化：`executor_messages` 从「Cortex 每轮清空」变成
    # 「**Cortex 下发新子目标时**清空」—— 它现在是 Executor 在**本子目标内**的
    # 局部历史（动作 + 工具反馈交替）。工具层写这个 key 的代码**不用改**
    # （`EXECUTOR_MESSAGES_KEY` 常量不变），清空责任从 Cortex 转移到 Cortex 的
    # 「下发目标」那一刻。

    # 本次子目标的目标陈述（Cortex → Executor）。Cortex 每次下发新目标时，
    # 同时清空 executor_messages，让 Executor 的局部历史从零开始。
    executor_goal: Annotated[
        str | None,
        "The goal statement for the current subgoal, issued by Cortex",
    ] = None

    # Executor 本轮认为目标已达成。Executor 节点写，graph 的条件边读：
    # True → 走 Reporter 收尾；False → 走 executor_tools 执行动作。
    executor_done: Annotated[
        bool,
        "Whether the executor believes the current subgoal is achieved",
    ] = False

    # 本子目标已执行的步数（熔断用）。Executor 节点每轮 +1。
    executor_steps: Annotated[
        int,
        "Steps executed so far in the current subgoal (circuit breaker)",
    ] = 0

    # Executor **明确放弃**了当前子目标（模型自己说「做不到」）。
    #
    # 为什么要有这个字段：`executor_done` 只有一个含义 ——「别再动了，收尾」，
    # 它**不区分**「做成了」和「做不到」。而 Reporter 的 outcome 是**代码判定**的
    # （`agents/executor/types.py::determine_outcome`），`done=True` 一律映射成
    # ``completed`` ⇒ 「做不到」这个状态在契约里**不可表达**。
    #
    # 真机实测（2026-10-26，mobile-manager execution 880）：executor 在
    # 「去头条刷热点领现金(0/1)」这一条上卡了 70 步、**从来没发过 done**，
    # 于是图里没有出口、看门狗没被叫醒，最后撞 `recursion_limit` 崩掉。
    #
    # ⇒ 有这个字段，模型就能说「这条做不到」；outcome 判成 ``failed``，
    # Cortex 换招或放弃，**而不是整个任务崩**。
    executor_failed: Annotated[
        bool,
        "Executor explicitly gave up on the current subgoal (cannot be achieved)",
    ] = False

    # 本子目标内的**进展指纹**（每步一条），供「有没有在原地打转」的检测用。
    #
    # 一条 = ``{"action": <动作指纹>, "screen": <屏幕指纹>, "nudged": <bool>}``。
    # 由 Executor 节点追加，Cortex 下发新目标时清空（和 ``executor_messages``
    # 同一个生命周期：属于「本子目标内」）。
    #
    # ⚠️ 这里存的是**指纹**（哈希），不是原始内容 —— 它的唯一用途是
    # 「和上一次比一比是不是同一屏/同一个动作」，不需要原文。
    executor_progress: Annotated[
        list[dict],
        "Per-step progress fingerprints inside the current subgoal (loop detection)",
    ] = []

    # 全任务累计的**动作步**数（跨子目标累加，换目标时**不**归零）。
    #
    # 为什么不能直接用 `executor_steps`：它是「本子目标内」的步数，Cortex 换目标时
    # 把它归零（`agents/cortex/cortex.py:429`），跨子目标会重置 —— 拿它当
    # 「整个任务跑了多少步」，读到的永远是 0~12 这种小数字。
    #
    # 为什么不能退回 `len(agents_thoughts)`：改造后 **Contextor 每个动作步写 1 条
    # thought**（`agents/contextor/contextor.py:191`），Cortex / Orchestrator 每个子目标
    # 再各写几条 —— thoughts 的**条数**是「感知 + 监督的次数」，和「跑了多少个动作步」
    # 根本不是一个东西（方案 §11.5 M13）。thoughts 仍然是「推理过程」的载体，
    # 口径不能混用。
    #
    # 口径：**executor 节点每跑一轮 +1**，和 `executor_steps` 的增量严格一致
    # （同一个节点、同一个时机写），所以恒有
    # `total_steps` ≡ Σ(各子目标的 `executor_steps`)。
    #
    # reducer 是 `take_last`（覆盖语义），累加由节点自己算 —— 和 `agents_thoughts`
    # 一个模式（那个字段的追加也是在节点的 `asanitize_update` 里算的）。
    total_steps: Annotated[
        int,
        "Cumulative action steps executed across all subgoals (never reset)",
        take_last,
    ] = 0

    # 失败事实账本（append-only，**与 failed_targets 熔断账本分开**）。
    #
    # 为什么分开：`failed_targets` 的语义是「同屏同目标连续失败」的即时保护，
    # 它**成功即清**（tools/mobile/tap.py:227）且**换键即整表替换**
    # （utils/failed_targets.py:99-115）—— 跨子目标的事实留不住。
    # 这个字段只增不减，承载「跨子目标的失败事实」。
    executor_failures: Annotated[
        list[dict],
        "Append-only ledger of failed targets across subgoals (facts, never rewritten)",
    ] = []

    # 「上一步的动作失败了，目标是谁」—— **瞬时**事实，供 Contextor 并进
    # 「[上一步的结果]」那一行。
    #
    # 为什么要有这个字段：失败事实要**记在产生它的那一步旁边**（模型读历史时
    # 不用跨块对账），而那一行是 Contextor 写的。工具层拿不到 Contextor 的行、
    # Contextor 也拿不到工具层手里的目标身份 ⇒ 只能靠一个瞬时字段把
    # 「上一步失败了、目标是谁」递过去。
    #
    # 生命周期：工具层**每次动作都写**（失败写事实、成功写 None）；
    # Contextor 读完之后写回 None ⇒ 不会被下一轮重复并进。
    #
    # ⚠️ **只记事实，不记结论**：内容只有「哪个目标失败了」。
    # 「换不换目标」是相对于目标的判断，框架不知道目标（见
    # `agents/contextor/contextor.py` 的 `render_action_result`）。
    last_action_failure: Annotated[
        ActionFailureFact | None,
        "Failure fact of the last action (None when it succeeded); consumed by the Contextor",
        take_last,
    ] = None

    # Reporter 的产出（Reporter → Cortex / Orchestrator）。
    executor_report: Annotated[
        ExecutorReport | None,
        "The report of the last subgoal attempt",
    ] = None

    # common keys
    agents_thoughts: Annotated[
        list[str],
        "All thoughts and reasons that led to actions (why a tool was called, expected outcomes..)",
        take_last,
    ]

    # scratchpad for explicit memory
    scratchpad: Annotated[
        dict[str, str],
        "Persistent key-value storage for notes the agent can save and retrieve",
        merge_dicts,
    ] = {}

    async def asanitize_update(
        self,
        ctx: MobileUseContext,
        update: dict[str, Any],
        agent: AgentNode | None = None,
    ) -> dict[str, Any]:
        """
        Sanitizes the state update to ensure it is valid and apply side effect logic where required.
        The agent is required if the update contains the "agents_thoughts" key.
        """
        updated_agents_thoughts: str | list[str] | None = update.get("agents_thoughts", None)
        if updated_agents_thoughts is not None:
            if isinstance(updated_agents_thoughts, str):
                updated_agents_thoughts = [updated_agents_thoughts]
            elif isinstance(updated_agents_thoughts, list):
                updated_agents_thoughts = [t for t in updated_agents_thoughts if t is not None]
            else:
                raise ValueError("agents_thoughts must be a str or list[str]")

            if agent is None:
                raise ValueError("Agent is required when updating the 'agents_thoughts' key")
            update["agents_thoughts"] = await _add_agent_thoughts(
                ctx=ctx,
                old=self.agents_thoughts,
                new=updated_agents_thoughts,
                agent=agent,
            )
        return update


async def _add_agent_thoughts(
    ctx: MobileUseContext,
    old: list[str],
    new: list[str],
    agent: AgentNode,
) -> list[str]:
    if ctx.on_agent_thought:
        for thought in new:
            await ctx.on_agent_thought(agent, thought)

    named_thoughts = [f"[{agent}] {thought}" for thought in new]
    if (
        ctx.execution_setup
        and ctx.execution_setup.traces_path is not None
        and ctx.execution_setup.trace_name is not None
    ):
        await record_interaction(ctx, response=AIMessage(content=str(named_thoughts)))
    return old + named_thoughts
