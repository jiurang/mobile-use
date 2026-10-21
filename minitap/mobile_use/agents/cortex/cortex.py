import asyncio
import os
from functools import cache
from pathlib import Path

from jinja2 import Template
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from minitap.mobile_use.agents.cortex.types import CortexOutput
from minitap.mobile_use.agents.planner.utils import render_subgoal_status
from minitap.mobile_use.constants import EXECUTOR_MESSAGES_KEY
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.controllers.controller_factory import create_device_controller
from minitap.mobile_use.graph.state import State
from minitap.mobile_use.services.llm import get_llm, invoke_llm_with_timeout_message, with_fallback
from minitap.mobile_use.services.telemetry import telemetry
from minitap.mobile_use.tools.index import (
    EXECUTOR_WRAPPERS_TOOLS,
    VIDEO_RECORDING_WRAPPERS,
    format_tools_list,
)
from minitap.mobile_use.utils.conversations import get_screenshot_message_for_llm
from minitap.mobile_use.utils.decorators import wrap_with_callbacks
from minitap.mobile_use.utils.failed_targets import render_failures
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.ocr_view import render_ocr_results
from minitap.mobile_use.utils.screen_elements import (
    ScreenElement,
    build_screen_elements,
    render_screen_elements,
)
from minitap.mobile_use.utils.thoughts_compaction import render_thoughts_history
from minitap.mobile_use.utils.ui_hierarchy_view import (
    normalize_hierarchy,
    render_hierarchy_and_labels,
)

logger = get_logger(__name__)

# 推理历史**只追加、不截断**。
#
# state.py 的 reducer 本来就只增不减（`old + named_thoughts`）；这里之前套了一个
# `[-25:]` 滑窗来封顶 token，代价是历史每轮整体左移 —— 提示词前缀每轮都会变，
# 前缀缓存永远命中不了。两件事互斥，取「前缀单调」这一条：缓存命中的部分按
# cache-read 计价，长任务里比滑窗更省。
#
# 后果是 token 随轮次线性增长（长任务里 thoughts 可占 cortex 上下文一半以上）。
# 要压这块成本，杠杆不在截断，而在别处：工具 schema 结构手术、截图降分辨率，
# 见 doc/token-context-analysis.md 五·一 / 5.3。
CORTEX_PROMPT_ENV = "MOBILE_USE_CORTEX_PROMPT"

# 模型偶尔会用这些占位串表示“没有内容”，统一当作 None 处理。
EMPTY_STRING_TOKENS = frozenset({"{}", "[]", "null", "None", ""})


@cache
def _compile_template(template_text: str) -> Template:
    """按模板文本缓存编译结果，避免每一步都重新编译 Jinja 模板。"""
    return Template(template_text)


def _normalize_optional_str(value: str | None) -> str | None:
    if value is None or value.strip() in EMPTY_STRING_TOKENS:
        return None
    return value


def load_cortex_prompt() -> str:
    """读 cortex 的 system prompt 模板。

    MOBILE_USE_CORTEX_PROMPT 可以指向另一个 .md（例如同目录的 cortex.zh.md），
    用来做中/英文提示词的 A/B 实测 —— 同一份 context 分别渲染两种语言，
    再各自真跑一遍，比较 token 与成功率。

    不设这个变量时读的还是同目录的 cortex.md，行为与从前逐字节一致，
    所以对生产是 no-op。路径不可读时只 warning 并回落到默认模板：
    指针写错不该让整条任务挂掉，降级 + 上报即可。
    """
    override = os.getenv(CORTEX_PROMPT_ENV)
    if override:
        path = Path(override).expanduser()
        try:
            if path.is_file():
                return path.read_text(encoding="utf-8")
            logger.warning(
                f"[cortex] {CORTEX_PROMPT_ENV}={override!r} 不是可读文件，回落到默认 cortex.md"
            )
        except OSError as e:
            logger.warning(
                f"[cortex] 读取 {CORTEX_PROMPT_ENV}={override!r} 失败（{e}），回落到 cortex.md"
            )
    return Path(__file__).parent.joinpath("cortex.md").read_text(encoding="utf-8")


async def _invoke_cortex(
    llm,
    messages: list[BaseMessage],
    ctx: MobileUseContext,
    known_subgoal_ids: set[str],
    used_fallback: bool = False,
) -> CortexOutput:
    """调用 LLM 并校验输出。

    先过滤不存在的子目标 ID，再判断响应是否为空。空响应（没有 decisions、
    没有推理、没有完成原因、也没有有效的待完成子目标）无法推进任务，直接抛错
    让 with_fallback 切到备用模型，避免白耗 step 直到 recursion_limit。
    """
    response = await invoke_llm_with_timeout_message(
        llm.ainvoke(messages),
        agent_name="cortex",
        task_id=ctx.trace_id,
        used_fallback=used_fallback,
    )
    if response is None:
        raise ValueError("Cortex LLM returned no response")

    response.decisions = _normalize_optional_str(response.decisions)
    response.decisions_reason = _normalize_optional_str(response.decisions_reason)
    response.goals_completion_reason = _normalize_optional_str(response.goals_completion_reason)

    unknown_subgoal_ids = [
        subgoal_id
        for subgoal_id in response.complete_subgoals_by_ids
        if subgoal_id not in known_subgoal_ids
    ]
    if unknown_subgoal_ids:
        logger.warning(f"[cortex] Ignoring unknown subgoal IDs: {unknown_subgoal_ids}")
    response.complete_subgoals_by_ids = [
        subgoal_id
        for subgoal_id in response.complete_subgoals_by_ids
        if subgoal_id in known_subgoal_ids
    ]

    if (
        response.decisions is None
        and response.decisions_reason is None
        and response.goals_completion_reason is None
        and not response.complete_subgoals_by_ids
        # 「我要图」也是一个**合法**的应答：它没有决策，但它要的是**再问一次**，
        # 不是错误。少了这个判断，on_demand 模式下模型每次要图都会在这里抛
        # "empty response"，整轮失败、agent 白烧步数（真机实测 51 步没跑完）。
        and not response.need_screenshot
    ):
        raise ValueError("Cortex LLM returned an empty response")
    return response


def should_request_screenshot(
    response: CortexOutput,
    *,
    on_demand: bool,
    has_screenshot: bool,
) -> bool:
    """模型说「看不清」时，要不要补一张图、同一轮再问一次。

    三个条件缺一不可：

    - `on_demand` —— 只有按需模式才有这个流程（每轮都带图的老模式不需要）。
    - `has_screenshot` —— 手上得有图可补；contextor 没抓到就补不了。
    - `not response.decisions` —— **模型已经给出动作就别补图**。它既然能给出
      动作，说明它其实判得出来（或者决定先动手），这时候塞一张图只会打乱
      它的判断，还多花一次调用。

    单独抽成纯函数是为了能单测 —— 这条分支是 ③ 的全部行为。
    """
    return bool(
        on_demand and has_screenshot and response.need_screenshot and not response.decisions
    )


class CortexNode:
    def __init__(self, ctx: MobileUseContext):
        self.ctx = ctx

    @wrap_with_callbacks(
        before=lambda: logger.info("Starting Cortex Agent..."),
        on_success=lambda _: logger.success("Cortex Agent"),
        on_failure=lambda _: logger.error("Cortex Agent"),
    )
    async def __call__(self, state: State):
        executor_feedback = get_executor_agent_feedback(state)

        current_locked_app_package = (
            self.ctx.execution_setup.get_locked_app_package() if self.ctx.execution_setup else None
        )

        executor_wrappers = list(EXECUTOR_WRAPPERS_TOOLS)
        if self.ctx.video_recording_enabled:
            executor_wrappers.extend(VIDEO_RECORDING_WRAPPERS)

        # system prompt 整段静态：不再接 initial_goal / 计划 / feedback 这类每轮
        # 变化的值。它因此是一个整体可复用的前缀，不会被任何单轮字段击穿。
        system_message = _compile_template(load_cortex_prompt()).render(
            platform=self.ctx.device.mobile_platform.value,
            executor_tools_list=format_tools_list(ctx=self.ctx, wrappers=executor_wrappers),
            locked_app_package=current_locked_app_package,
            perception=self.ctx.perception,
        )

        # 消息按「首次变化时刻」从早到晚排序：恒定 → 只追加 → 每轮变。
        # 越靠前的块越稳定，能被前缀缓存覆盖的公共前缀就越长。
        messages: list[BaseMessage] = [
            SystemMessage(content=system_message),
            # 任务内恒定：设备基本信息、目标。这块逐字节不变，永远是前缀的一部分。
            HumanMessage(
                content="\n".join(
                    [
                        "Here are my device info:",
                        self.ctx.device.to_str().rstrip(),
                        f"Initial goal: {state.initial_goal}",
                    ]
                )
            ),
        ]

        # 这些 thought 绝不能塞成 assistant 回合。state.py:113 给每条加了 "[role] "
        # 前缀，模型会把它们当成 few-shot 照抄，于是无视 tool_choice 改用纯文本回答，
        # _parse() 再把这段中文当 JSON 解析 -> pydantic json_invalid -> 整个 goal 挂掉。
        # 压成单个 user 回合后，MiniMax-M3 的 tool_calls 实测恢复 2/2。
        # 见 v/probe_cortex_toolcall_ab.py、v/probe_cortex_history_len.py。
        #
        # 压缩：只压发给 LLM 的这份，state 里的原文不动（trace / TUI / steps_count /
        # outputter 都要全量）。压的是 cortex 的散文，executor 工具结果和
        # orchestrator 里程碑是**事实**，逐字保留。
        #
        # 关键：压缩是**位置无关**的（每条按同一条规则渲染），所以已经发出去的
        # 字节永不改变 -> 追加语义保留 -> 前缀缓存不因压缩而击穿。
        # 见 utils/thoughts_compaction.py 与 doc/thoughts-compaction-plan.md。
        thoughts = state.agents_thoughts
        if thoughts:
            compaction_enabled = self.ctx.perception.thoughts_compaction
            body = render_thoughts_history(thoughts, enabled=compaction_enabled)
            header = "Agent thoughts history (newest last):"
            if compaction_enabled:
                header += (
                    "\n(Older cortex reasoning is compressed to its first and last "
                    "sentence. Executor results and subgoal milestones are verbatim.)"
                )
            messages.append(HumanMessage(content=header + "\n" + body))

        # 计划日志（子目标的 ID + 描述）排在 thoughts **之后**。
        #
        # 为什么不能放在 thoughts 前面
        # --------------------------
        # 计划日志是**只追加**的：replan 时尾部追加一段 "PLAN vN"，旧段一字不改。
        # 但「只追加」不等于「不击穿」—— 只要它排在 thoughts 前面，追加点之后的
        # 全部重算，而 thoughts 恰恰在它后面。
        #
        # 实测（2026-10-21，MiniMax M3.1-Flash 真 API，122 条 thoughts）：
        #   计划日志在 thoughts 前 -> replan 当轮 cached 2,329，满价重算 8,261
        #   计划日志在 thoughts 后 -> replan 当轮 cached 10,411，满价重算 183
        # 一次 replan 差 8,078 token，而这个差随 thoughts 变长而变大。
        #
        # 挪到 thoughts 之后的代价：thoughts 每轮增长，断点落在 thoughts 末尾，
        # 于是「计划日志 + 尾部块」每轮要满价重算一次（约 150 + 500 token）。
        # 摊到一次 replan 上（省 8,000+）是划算的，而且 replan 越多越划算。
        #
        # 语义没变：内容与之前逐字相同，只是从 head 块里拆出来独立成一条消息，
        # 「最后一段 PLAN 是当前计划」这句话依然成立。
        messages.append(
            HumanMessage(
                content="Subgoal plan (append-only history; "
                "the last PLAN section is the active one):\n"
                + (state.subgoal_plan_log or "(no plan)")
            )
        )

        # 完成情况是**索引**、且排在 thoughts 之后：它每轮都可能变（子目标完成 /
        # 推进 / replan），但按**计划序**排列、在固定槽位上原地替换，不会有新行插到
        # 行中间。放在 thoughts 后面，它怎么变都不会碰到 thoughts 那段唯一增长的前缀。
        # 固定成一条消息（无子目标时也发，值填 none），条数漂移本身也是前缀的一部分。
        messages.append(
            HumanMessage(content="Subgoal status: " + render_subgoal_status(state.subgoal_plan))
        )

        # 已失败的目标：模型下一轮最需要知道的一件事。
        #
        # 真机实测里，模型看到「目标不在屏上了」会误诊成「索引指错人了」，
        # 然后在同一个目标上反复打转。这条把事实摆出来：这些目标已经没了，
        # 别再用，换一个现在还在的元素。
        failed_line = render_failures(state.failed_targets)
        if failed_line:
            messages.append(HumanMessage(content=failed_line))

        # 以下都是每轮必变的内容，必须排在最后。executor feedback 过去长在
        # system prompt 里，等于每轮把整段 system prompt 的缓存打掉。
        messages.append(HumanMessage(content="Executor feedback:\n" + executor_feedback))
        device_state_lines = []
        if state.device_date:
            device_state_lines.append(f"Device date: {state.device_date}")
        if state.focused_app_info:
            device_state_lines.append(f"Focused app info: {state.focused_app_info}")
        if state.screen_state == "off":
            # 屏幕灭着的时候，截图是全黑的、元素表里只剩 systemui 的残留 ——
            # 模型看到的是「屏上什么都没有」。不告诉它，它就会在黑屏上反复
            # 尝试（2026-10-21 真机实测：30 分钟息屏超时让一个任务从 30 步涨到
            # 51 步还没跑完）。
            device_state_lines.append(
                "Screen is OFF (the device is asleep). The screenshot is black and the "
                "element list only holds systemui leftovers, so there is nothing to "
                "act on. Call wake_screen FIRST, then read the screen again on the "
                "next turn."
            )
        if device_state_lines:
            messages.append(HumanMessage(content="\n".join(device_state_lines)))

        # ---- 感知块 ----
        # Android 形状下，层级树和 OCR 合成**一张**带统一索引的表：
        # 一个编号空间，模型报编号、代码反解，不再抄 id 和坐标。
        # 层级树形状不认识时（iOS / 云设备）保持改造前的行为：原始 dump + 独立 OCR 块。
        screen_elements: list[ScreenElement] = []
        tree_rows: list[dict] | None = None
        if self.ctx.perception.ui_hierarchy and state.latest_ui_hierarchy:
            tree_rows = normalize_hierarchy(state.latest_ui_hierarchy)

        if tree_rows is None:
            if self.ctx.perception.ui_hierarchy and state.latest_ui_hierarchy:
                legacy_tree_text, _ = render_hierarchy_and_labels(state.latest_ui_hierarchy)
                if legacy_tree_text:
                    messages.append(HumanMessage(content=legacy_tree_text))
            if self.ctx.perception.ocr and state.latest_ocr_results:
                # 不去重：OCR 行的 [i] 是 Target.ocr_index 的取值，
                # 剔掉任何一条都会让模型手里的行号和 results 的下标对不上。
                legacy_ocr_text = render_ocr_results(state.latest_ocr_results)
                if legacy_ocr_text:
                    messages.append(HumanMessage(content=legacy_ocr_text))
        else:
            screen_elements = build_screen_elements(
                tree_rows=tree_rows,
                ocr_results=(state.latest_ocr_results if self.ctx.perception.ocr else None),
                foreground_package=state.focused_app_info,
            )
            screen_text = render_screen_elements(screen_elements)
            if screen_text:
                messages.append(HumanMessage(content=screen_text))

        perception = self.ctx.perception
        # 截图通道三态：True=每轮都带 / False=不带 / "on_demand"=模型要才给。
        on_demand_screenshot = perception.screenshot == "on_demand"
        always_screenshot = perception.screenshot is True

        async def append_screenshot() -> None:
            """把当前屏幕的截图追加到 messages 末尾。

            用的是 `state.latest_screenshot` —— 那是 contextor 刚抓的当屏截图。
            cortex 决策时 executor 还没动过屏幕，所以它就是**当前**画面，
            不用重抓（重抓会多花一次 adb 往返，还可能拿到不同帧）。
            """
            if not state.latest_screenshot:
                return
            controller = create_device_controller(self.ctx)
            # 压缩是同步 PIL 操作，放到线程里避免阻塞事件循环。
            compressed_image_base64 = await asyncio.to_thread(
                controller.get_compressed_b64_screenshot, state.latest_screenshot
            )
            messages.append(get_screenshot_message_for_llm(compressed_image_base64))

        if always_screenshot:
            await append_screenshot()

        llm = get_llm(ctx=self.ctx, name="cortex", temperature=1).with_structured_output(
            CortexOutput
        )
        llm_fallback = get_llm(
            ctx=self.ctx, name="cortex", use_fallback=True, temperature=1
        ).with_structured_output(CortexOutput)
        known_subgoal_ids = {subgoal.id for subgoal in state.subgoal_plan}

        async def ask_cortex() -> CortexOutput:
            return await with_fallback(
                main_call=lambda: _invoke_cortex(llm, messages, self.ctx, known_subgoal_ids),
                fallback_call=lambda: _invoke_cortex(
                    llm_fallback, messages, self.ctx, known_subgoal_ids, used_fallback=True
                ),
            )

        response: CortexOutput = await ask_cortex()

        # 按需取图：模型说「元素表和 OCR 不够用」→ 补一张图，**同一轮**再问一次。
        #
        # 为什么要同一轮：多一轮往返意味着多一次完整的 LLM 调用（prompt 更大、
        # 还要重算一次），比每轮都带图更贵。同一轮内补图只有真正需要图的那
        # 少数轮次才多花这一次。
        if should_request_screenshot(
            response,
            on_demand=on_demand_screenshot,
            has_screenshot=bool(state.latest_screenshot),
        ):
            await append_screenshot()
            response = await ask_cortex()

        valid_subgoal_ids = response.complete_subgoals_by_ids

        thought_parts = []
        if response.decisions_reason:
            thought_parts.append(f"Decisions reason: {response.decisions_reason}")
        if response.goals_completion_reason:
            thought_parts.append(f"Goals completion reason: {response.goals_completion_reason}")

        agent_thought = "\n\n".join(thought_parts)

        # Capture cortex decision telemetry (only non-sensitive flags)
        telemetry.capture_cortex_decision(
            task_id=self.ctx.trace_id,
            has_decisions=response.decisions is not None,
            has_goals_completion=response.goals_completion_reason is not None,
            completed_subgoals_count=len(valid_subgoal_ids),
        )

        update = {
            "structured_decisions": response.decisions,
            "complete_subgoals_by_ids": valid_subgoal_ids,
            "latest_ui_hierarchy": None,
            # latest_ocr_results 故意不清空：渲染给模型的 [i] 就是
            # Target.ocr_index 的取值，executor 工具执行时需要同一份
            # results 才能把索引还原成 bounds。图序是
            # contextor -> cortex -> executor -> executor_tools -> ... -> contextor，
            # 所以 executor_tools 跑的时候这份感知数据正是当屏的，
            # 下一次 contextor 会整体覆盖它。
            # latest_ui_hierarchy 仍然清空 —— 它没有索引契约，
            # 工具侧走 controller.get_ui_elements() 现取即可。
            # latest_screen_elements 同样不清空，而且理由更强：它**就是**
            # 索引契约本身。Target.index 要还原成「模型当时看到的那一行」，
            # 只有这一次裁剪的结果算数 —— 执行时重新 dump 再重新编号是不行的，
            # 屏可能已经变了，编号会和模型看到的不一致。
            "latest_screen_elements": screen_elements,
            "latest_screenshot": None,
            "focused_app_info": None,
            "device_date": None,
            # Executor related fields
            EXECUTOR_MESSAGES_KEY: [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
            "cortex_last_thought": agent_thought or None,
        }
        # 没有任何推理内容时不要把空字符串写进历史（会触发回调并污染 trace）。
        if agent_thought:
            update["agents_thoughts"] = [agent_thought]

        return await state.asanitize_update(
            ctx=self.ctx,
            update=update,
            agent="cortex",
        )


def get_executor_agent_feedback(state: State) -> str:
    if state.structured_decisions is None:
        return "None."
    feedback_parts = [
        str(message) for message in state.executor_messages if isinstance(message, ToolMessage)
    ]
    # 走 LLM 兜底路径时 executor 可能只回一条纯文本（没有 tool_calls），
    # 这段内容同样是 cortex 需要看到的执行结果，不能丢。
    last_ai_content = next(
        (
            message.content.strip()
            for message in reversed(state.executor_messages)
            if isinstance(message, AIMessage)
            and isinstance(message.content, str)
            and message.content.strip()
        ),
        None,
    )
    if last_ai_content:
        feedback_parts.append(last_ai_content)
    return (
        f"Latest UI decisions:\n{state.structured_decisions}"
        + "\n\n"
        + f"Executor feedback:\n{feedback_parts}"
    )
