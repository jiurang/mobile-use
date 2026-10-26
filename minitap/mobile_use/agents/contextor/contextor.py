"""Contextor：感知节点 —— 抓屏、判稳、建「屏幕元素表」。

它同时是索引契约的产地：模型报的 ``Target.index`` 只能在**本次裁剪**上还原，
所以抓屏和建表必须发生在同一个节点里，中间不能插入任何动作。

``ctx.perception`` 的两条通道：``ui_hierarchy`` 关掉则不建表（工具侧走 legacy
OCR 路径）；``ocr`` 关掉则 OCR 引擎根本不跑。截图不再进模型，只供 OCR 和坐标
校验使用。

⚠️ 先判稳再建表，但「静 ≠ 全」：懒加载控件可能在静默期之后才挂上，这是已知
代价 —— 验收看「步数」和「失败目标数」。
"""

import asyncio
import time
from pathlib import Path
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage

from minitap.mobile_use.agents.contextor.types import AppLockVerificationOutput, ContextorOutput
from minitap.mobile_use.agents.planner.types import Subgoal
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.constants import EXECUTOR_MESSAGES_KEY
from minitap.mobile_use.controllers.controller_factory import create_device_controller
from minitap.mobile_use.controllers.platform_specific_commands_controller import (
    get_device_date,
)
from minitap.mobile_use.graph.state import State
from minitap.mobile_use.prompt_language import load_prompt_template, pick
from minitap.mobile_use.services.llm import get_llm, invoke_llm_with_timeout_message, with_fallback
from minitap.mobile_use.services.ocr import get_ocr_engine
from minitap.mobile_use.services.ocr.types import OcrEngine, OcrText
from minitap.mobile_use.utils.app_launch_utils import launch_app_with_retries
from minitap.mobile_use.utils.app_names import app_display_name, resolve_app_display_name
from minitap.mobile_use.utils.decorators import wrap_with_callbacks
from minitap.mobile_use.utils.failed_targets import ActionFailureFact, describe_target
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.screen_elements import (
    ScreenElement,
    build_screen_elements,
    foreground_package_from_tree,
    is_noise_text,
    screen_fingerprint,
    visible_texts,
)
from minitap.mobile_use.utils.screen_stability import (
    is_stable,
    structural_change_count,
    tree_fingerprint,
)
from minitap.mobile_use.utils.ui_hierarchy_view import normalize_hierarchy

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 「上一步的结果」—— 写进执行器的局部历史
# ---------------------------------------------------------------------------
#
# 执行器的局部历史里只有调用和工具回执，没有「结果」（屏长什么样、前台 app
# 是谁）。实测它因此把 Home 按了 13 次，每次都以为会回到桌面。
#
# ⚠️ 只陈述事实，不下结论：「这个动作没用」「重复了就换招」都是相对目标的
# 判断，框架做不了（刷短视频时连滑 10 次是正常浏览），判定留给模型。


def render_action_result(
    *,
    previous_elements: list[ScreenElement] | None,
    previous_app: str | None,
    current_elements: list[ScreenElement] | None,
    current_app: str | None,
    previous_app_name: str | None = None,
    current_app_name: str | None = None,
    failure: ActionFailureFact | None = None,
    max_new_texts: int = 6,
) -> str | None:
    """一句话描述「上一步的动作之后，屏变成了什么样」。

    三样事实：动作是否失败（失败的是哪个目标）、前台 app 是否变化、屏幕内容
    是否变化及新增了哪些文字。不说「有没有进展」「重复了几次」—— 那是相对
    目标的结论，框架不知道目标。

    Returns:
        没有可比的前一屏时返回 None。
    """
    if not previous_elements and previous_app is None:
        return None

    parts: list[str] = []

    # ---- 0. 上一步的动作失败了（事实，不是结论）----
    if failure:
        label = failure["label"]
        resource_id = failure["resource_id"]
        if label or resource_id:
            target = describe_target(label=label, resource_id=resource_id)
            parts.append(f"{target} 这一步失败了")
        else:
            # 目标只有索引，而索引每轮重编号 ⇒ 不写进去；也不拿
            # "(unknown target)" 当占位符（模型会去找叫这个名字的元素）。
            parts.append("这一步失败了（没能认出是哪个目标）")

    # ---- 1. 前台 app：比较用包名（精确稳定），展示用应用名 ----
    if previous_app != current_app:
        if current_app:
            parts.append(f"前台应用变成了 {current_app_name or app_display_name(current_app)}")
        else:
            parts.append("前台应用读不到了")
    elif current_app:
        parts.append(f"前台应用没变（还是 {current_app_name or app_display_name(current_app)}）")

    # ---- 2. 屏幕内容：新文字取原文差集里的非噪音项（时钟/电量不算）----
    prev_fp = screen_fingerprint(previous_elements)
    cur_fp = screen_fingerprint(current_elements)
    prev_texts = set(visible_texts(previous_elements))
    new_texts = [
        t for t in visible_texts(current_elements) if t not in prev_texts and not is_noise_text(t)
    ]
    if prev_fp == cur_fp:
        parts.append("屏幕内容没变")
    elif new_texts:
        shown = "、".join(new_texts[:max_new_texts])
        if len(new_texts) > max_new_texts:
            shown += f" 等 {len(new_texts)} 条"
        parts.append(f"屏幕上新出现了：{shown}")
    else:
        parts.append("屏幕内容变了（没有新的可读文字）")

    if not parts:
        return None
    return "[上一步的结果] " + "；".join(parts)


class ContextorNode:
    def __init__(self, ctx: MobileUseContext):
        self.ctx: MobileUseContext = ctx

    @wrap_with_callbacks(
        before=lambda: logger.info("Starting Contextor Agent"),
        on_success=lambda _: logger.success("Contextor Agent"),
        on_failure=lambda _: logger.error("Contextor Agent"),
    )
    async def __call__(self, state: State) -> dict[str, Any]:
        device_controller = create_device_controller(self.ctx)

        # 屏幕沉降：先等下限（防「动作刚发出、屏还没开始变」的竞态），再确认
        # 已经定下来。动画不用外部等 —— ``dump_hierarchy`` 自带 waitForIdle。
        await self._wait_for_settle(state)

        # 稳定性采样：
        # * ``fingerprint``（默认）：连续两次全量树 dump 比指纹。
        # * ``structural``：只采一次当结构基线，真正的比对放在建表那次 dump
        #   之后 ⇒ 常态省一次 dump，也避开视频页因点赞数跳动而永远判不稳。
        stability_sample: list[dict] | None = None
        if self._structural_mode():
            stability_sample = await device_controller.get_hierarchy_only()
            if not stability_sample:
                logger.warning(
                    "[contextor] Structural stability: could not sample the tree; skipping check"
                )
        else:
            await self._confirm_tree_stable(device_controller, state)

        # OCR 走 on_screenshot 回调，在 u2 采集线程里就地跑。通道关掉时 OCR
        # 根本不跑（不是「跑完再丢掉」：实测 381~443ms/步，结果还是被丢掉）。
        ocr_enabled = self.ctx.perception.ocr
        if not ocr_enabled:
            logger.info("[contextor] OCR channel is OFF; skipping OCR entirely")

        ocr_box: list[list[OcrText] | None] = []
        ocr_done = False

        def _run_ocr_inline(screenshot_base64: str) -> None:
            nonlocal ocr_done
            try:
                ocr_box.append(self._recognize_ocr(screenshot_base64))
            finally:
                ocr_done = True

        # 屏幕是不是灭的：息屏时 uiautomator 仍会返回 19 个 systemui 残留元素，
        # 所以「树为空 ⇒ 灭屏」永远不触发。改为无条件读 `dumpsys power` 的
        # mWakefulness（系统权威状态）。
        screen_state: str | None = None
        power_task: asyncio.Task[str] | None = None
        power_reader = getattr(device_controller, "get_screen_power_state", None)
        if power_reader is not None:
            power_task = asyncio.create_task(power_reader())
        # 电源状态和设备日期都和抓屏**并发**：它们只是 adb 往返（~100ms / 88ms），
        # 叠进 dump 的等待窗口里就净增 ≈ 0。日期必须读**设备**时钟，设备时区
        # 可能和 PC 不同，否则 prompt 里的 "Device date" 会说错。
        date_task = asyncio.create_task(asyncio.to_thread(get_device_date, self.ctx))
        if power_reader is not None:
            # create_task 只是排进日程表，不会立刻开跑 —— 显式让出一次控制权，
            # 否则两个 adb 调用退化成串行。
            await asyncio.sleep(0)
        device_date: str = ""
        try:
            try:
                if ocr_enabled:
                    device_data = await device_controller.get_screen_data(
                        on_screenshot=_run_ocr_inline
                    )
                else:
                    # OCR 通道关着 ⇒ 不挂回调，控制器也就不跑 OCR 引擎。
                    device_data = await device_controller.get_screen_data()
            except TypeError:
                # 自定义控制器没跟上新签名时不能整轮挂掉，退回串行路径。
                logger.warning("Controller does not support on_screenshot; using serial path")
                device_data = await device_controller.get_screen_data()
        finally:
            # 抓屏炸了也要回收这两个 task，否则事件循环关闭时报
            # "Task exception was never retrieved"，把真正的异常盖掉。
            if power_task is not None:
                screen_state = await power_task
            device_date = await date_task

        if screen_state == "off":
            logger.warning("[contextor] Screen is OFF — agent will be told to wake it up.")

        if ocr_enabled and not ocr_done:
            # 回调没被调用（平台不支持 / 提前抛错）—— 补跑一次。
            ocr_box.append(await self._run_ocr(device_data.base64))

        # structural 模式：基线 = 上面那次采样，current = 建表用的这棵树。
        if stability_sample is not None:
            await self._verify_structural_stability(
                device_controller=device_controller,
                state=state,
                baseline=stability_sample,
                current=cast(list[dict], device_data.elements),
            )

        # 前台包名从刚抓回来的树里推，省掉一次串行的 `dumpsys window`（127ms）。
        current_app_package = foreground_package_from_tree(device_data.elements)
        agent_outcome: str | None = None
        if self.ctx.execution_setup and self.ctx.execution_setup.app_lock_status:
            locked_app_package = self.ctx.execution_setup.app_lock_status.locked_app_package
            should_verify_app_lock = (
                self.ctx.execution_setup.app_lock_status.locked_app_initial_launch_success
            )
            if should_verify_app_lock:
                if current_app_package:
                    try:
                        verification = await self._handle_app_lock_verification(
                            state=state,
                            current_app_package=current_app_package,
                            locked_app_package=locked_app_package,
                        )
                        agent_outcome = verification.to_optional_message()
                    except Exception as e:
                        logger.error(f"Failed to verify app lock: {e}")
                else:
                    logger.warning(
                        f"App lock: current app unknown, skipping ({locked_app_package})"
                    )
            else:
                logger.warning(f"App lock: initial launch failed, skipping ({locked_app_package})")

        ocr_results = ocr_box[0] if ocr_box else None

        # 屏幕元素表：UI 层级树 + OCR 合成**一张**统一索引的表。表就是索引契约
        # 本身，所以必须和抓屏在同一个节点里建（见模块 docstring）。
        #
        # `ui_hierarchy` 是「要不要把层级树给模型看」的开关，关掉则表里没有 UI 行。
        # getattr 兜底：测试替身可能没这个字段，缺省按 PerceptionConfig 的默认值。
        use_ui_hierarchy = getattr(self.ctx.perception, "ui_hierarchy", True)
        tree_rows: list[dict[str, Any]] | None = None
        if use_ui_hierarchy and device_data.elements:
            tree_rows = normalize_hierarchy(device_data.elements)

        screen_elements: list[ScreenElement] = []
        if tree_rows is not None:
            screen_elements = build_screen_elements(
                tree_rows=tree_rows,
                ocr_results=(ocr_results if self.ctx.perception.ocr else None),
                foreground_package=current_app_package,
            )
        # tree_rows 为 None = 层级树形状不认识（iOS / 云设备）→ 元素表留空，
        # 工具侧改走 `latest_ocr_results` 的老路径，**不用改**。
        # 这里是「不猜」：猜错就是静默剥夺别的平台的感知能力。

        # 「上一步的结果」写进执行器的局部历史：只有动作执行过才有「上一步」，
        # 第一次抓屏不写。`last_action_failure` 是工具层递过来的瞬时事实，
        # 并进这一行 ⇒ 失败事实落在产生它的那一步旁边，模型不用跨块对账。
        # getattr 兜底：测试替身（精简过的 state）可能没这几个字段。
        last_failure = getattr(state, "last_action_failure", None)
        previous_app_package = getattr(state, "focused_app_info", None)
        # 展示名：切 app 时才走一次 adb（进程内缓存），读不到退回内置表/包名。
        current_app_name = await resolve_app_display_name(self.ctx, current_app_package)
        previous_app_name = await resolve_app_display_name(self.ctx, previous_app_package)
        result_line = render_action_result(
            previous_elements=getattr(state, "latest_screen_elements", None),
            previous_app=previous_app_package,
            previous_app_name=previous_app_name,
            current_elements=screen_elements,
            current_app=current_app_package,
            current_app_name=current_app_name,
            failure=last_failure,
        )

        update: dict[str, Any] = {
            # 索引契约本身，不能被清空：工具靠它把 Target.index 还原成当屏坐标，
            # 只有这一次裁剪算数。下一次 contextor 会整体覆盖它。
            "latest_screen_elements": screen_elements,
            # 层级树没有索引契约，建完表就清空。留着会让工具侧按 id / 文字查找
            # 的分支复活 —— 那是行为变更。
            "latest_ui_hierarchy": None,
            "latest_ocr_results": ocr_results,
            "focused_app_info": current_app_package,
            "screen_size": (device_data.width, device_data.height),
            "device_date": device_date,
            "screen_state": screen_state,
            "agents_thoughts": [agent_outcome],
            # 瞬时事实用完即清，避免下一轮重复并进。
            "last_action_failure": None,
        }
        if result_line:
            update[EXECUTOR_MESSAGES_KEY] = [HumanMessage(content=result_line)]

        return await state.asanitize_update(
            ctx=self.ctx,
            update=update,
            agent="contextor",
        )

    async def _wait_for_settle(self, state: State) -> float:
        """抓屏之前等屏幕沉降到配置的下限，返回实际等待的秒数。

        它只是「起步下限」；「这一屏定下来没有」由 :meth:`_confirm_tree_stable` 判。
        """
        delay_ms = self.ctx.perception.settle_delay_ms
        if delay_ms <= 0 or state.last_action_at is None:
            return 0.0

        elapsed = time.time() - state.last_action_at
        remaining = delay_ms / 1000.0 - elapsed
        if remaining <= 0:
            return 0.0

        logger.info(
            f"Waiting {remaining:.1f}s for the screen to settle "
            f"(last action was {elapsed:.1f}s ago, floor is {delay_ms}ms)"
        )
        await asyncio.sleep(remaining)
        return remaining

    def _structural_mode(self) -> bool:
        """走不走「结构摘要」这套 video-safe 判据。"""
        return getattr(self.ctx.perception, "stability_mode", "fingerprint") == "structural"

    async def _verify_structural_stability(
        self,
        *,
        device_controller,
        state: State,
        baseline: list[dict],
        current: list[dict],
    ) -> bool:
        """结构摘要判稳（video-safe）。

        现行指纹含 text，视频页每秒都在变（点赞数、倒计时、进度）⇒ 永远判不稳，
        跑满重试还是「照用」。新判据只比**可交互元素的身份 + 位置**，不看内容：
        视频页判稳，弹窗 / 翻页 / 懒加载挂上则判不稳。

        判不稳也照用：执行时按 resource-id 重新定位，元素挪了位置也能点对；
        真正的致命问题是「模型报了屏上根本没有的东西」，那和稳不稳无关。
        """
        changed = structural_change_count(baseline, current)
        if changed == 0:
            logger.info(
                "[contextor] Structure stable (video-safe digest): "
                f"{len(current)} elements, interactive structure unchanged"
            )
            return True

        extra = getattr(self.ctx.perception, "stability_extra_dumps", 0)
        previous = current
        for attempt in range(extra):
            delay_ms = self.ctx.perception.settle_delay_ms
            if delay_ms > 0:
                await asyncio.sleep(delay_ms / 1000.0)
            sample = await device_controller.get_hierarchy_only()
            if not sample:
                break
            if structural_change_count(previous, sample) == 0:
                logger.info(
                    f"[contextor] Structure stable after {attempt + 1} extra dump(s) "
                    f"({changed} interactive element(s) had changed)"
                )
                return True
            previous = sample

        logger.info(
            "[contextor] Interactive structure still changing "
            f"({changed} element(s)); proceeding with the latest capture "
            "(execution re-resolves by resource-id)"
        )
        return False

    async def _confirm_tree_stable(self, device_controller, state: State) -> bool:
        """连续两次树级 dump 指纹相同，才算这一屏定下来了。

        沉降下限降到 200ms 后，「抓早了」会拿到过渡态（页面导航实测
        49 → 91 → 91 → 228 个元素），这一步把它挡掉。

        代价很低：只 dump 树（150~250ms），不截图不跑 OCR；只读步骤完全跳过。
        ⚠️ 已知误判面：「静 ≠ 全」，懒加载控件在静默期之后才挂上时挡不住。
        """
        max_retries = getattr(self.ctx.perception, "settle_max_retries", 0)
        if max_retries <= 0 or getattr(state, "last_action_at", None) is None:
            return False

        previous: str | None = None
        for attempt in range(max_retries + 1):
            elements = await device_controller.get_hierarchy_only()
            if not elements:
                # 拿不到树就当「没法确认」—— 别在这里把流程卡住。
                logger.warning("[contextor] Could not dump hierarchy for stability check")
                return False

            current = tree_fingerprint(elements)
            if is_stable(previous, current):
                logger.info(
                    f"[contextor] Screen stable after {attempt} stability check(s) "
                    f"({len(elements)} elements)"
                )
                return True

            previous = current
            if attempt < max_retries:
                # 还在变 ⇒ 再等一个下限，再采一次。
                delay_ms = self.ctx.perception.settle_delay_ms
                if delay_ms > 0:
                    await asyncio.sleep(delay_ms / 1000.0)

        logger.info(
            "[contextor] Screen still changing after "
            f"{max_retries} stability check(s); proceeding with the latest capture"
        )
        return False

    def _resolve_ocr_engine(self, screenshot_base64: str | None) -> OcrEngine | None:
        """OCR 的开关 + 可用性检查。两条执行路径共用。"""
        if not self.ctx.perception.ocr:
            return None
        if not screenshot_base64:
            logger.warning("No screenshot available; skipping OCR")
            return None
        return get_ocr_engine(self.ctx)

    def _recognize_ocr(self, screenshot_base64: str) -> list[OcrText] | None:
        """**同步**跑 OCR，只能在不跑事件循环的线程里调用。

        复用 `get_screen_data()` 的原分辨率截图（压缩过的图会把框算歪）。
        OCR 是增强感知，失败只降级不中断。
        """
        engine = self._resolve_ocr_engine(screenshot_base64)
        if engine is None:
            return None

        try:
            started = time.perf_counter()
            results = engine.recognize(screenshot_base64)
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.info(f"OCR: {len(results)} texts in {elapsed_ms:.0f}ms (engine={engine.name})")
            return results
        except Exception as e:
            logger.warning(f"OCR failed, continuing without it: {type(e).__name__}: {e}")
            return None

    async def _run_ocr(self, screenshot_base64: str | None) -> list[OcrText] | None:
        """回调没生效时的兜底路径：显式丢到线程池，免得阻塞事件循环。"""
        return await asyncio.to_thread(self._recognize_ocr, screenshot_base64 or "")

    async def _handle_app_lock_verification(
        self, state: State, current_app_package: str, locked_app_package: str
    ) -> AppLockVerificationOutput:
        """Verify app lock compliance and decide whether to relaunch the locked app."""
        if not self.ctx.execution_setup or not self.ctx.execution_setup.app_lock_status:
            return AppLockVerificationOutput(
                package_name=locked_app_package,
                reasoning="App lock feature is not setup",
                status="error",
            )

        app_lock_status = self.ctx.execution_setup.app_lock_status
        locked_app_package = app_lock_status.locked_app_package

        if current_app_package == locked_app_package:
            logger.info(f"App lock verified: current app matches locked app ({locked_app_package})")
            return AppLockVerificationOutput(
                package_name=locked_app_package,
                status="already_in_foreground",
            )

        logger.warning(f"App lock: want {locked_app_package}, got {current_app_package}")

        decision: ContextorOutput = await self._invoke_contextor_llm(
            initial_goal=state.initial_goal,
            subgoal_plan=state.subgoal_plan,
            agents_thoughts=state.agents_thoughts,
            locked_app_package=locked_app_package,
            current_app_package=current_app_package,
        )

        if decision.should_relaunch_app:
            logger.info(f"Relaunching locked app: {locked_app_package}")
            success, error = await launch_app_with_retries(self.ctx, app_package=locked_app_package)
            if not success:
                logger.error(f"Failed to relaunch {locked_app_package}: {error}")
                return AppLockVerificationOutput(
                    package_name=locked_app_package,
                    reasoning=f"Failed to relaunch app: {error}",
                    status="error",
                )
            return AppLockVerificationOutput(
                package_name=locked_app_package,
                reasoning=decision.reasoning,
                status="relaunched",
            )

        logger.info(f"Allowing app deviation to: {current_app_package}")
        return AppLockVerificationOutput(
            package_name=locked_app_package,
            reasoning=decision.reasoning,
            status="allowed_deviation",
        )

    async def _invoke_contextor_llm(
        self,
        initial_goal: str,
        subgoal_plan: list[Subgoal],
        agents_thoughts: list[str],
        locked_app_package: str,
        current_app_package: str,
    ) -> ContextorOutput:
        """Invoke the LLM to decide whether to relaunch the locked app."""

        # ⚠️ 取**最老**的 25 条（``[:25]``），planner 那边取的是**最新**的
        # 25 条（``[-25:]``）。同名不同向 —— 别照抄 planner 的写法。
        MAX_AGENTS_THOUGHTS = 25

        system_message = load_prompt_template(
            Path(__file__).parent.joinpath("contextor.md"),
            self.ctx.prompt_language,
        ).render(
            task_goal=initial_goal,
            subgoal_plan="\n".join([str(subgoal) for subgoal in subgoal_plan]),
            locked_app_package=locked_app_package,
            current_app_package=current_app_package,
            agents_thoughts=agents_thoughts[:MAX_AGENTS_THOUGHTS],
        )

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(
                content=pick(
                    "Please make your decision.",
                    "请做出你的判断。",
                    language=self.ctx.prompt_language,
                )
            ),
        ]

        llm = get_llm(ctx=self.ctx, name="contextor").with_structured_output(ContextorOutput)
        llm_fallback = get_llm(
            ctx=self.ctx, name="contextor", use_fallback=True
        ).with_structured_output(ContextorOutput)

        response = cast(
            ContextorOutput,
            await with_fallback(
                main_call=lambda: invoke_llm_with_timeout_message(
                    llm.ainvoke(messages),
                    agent_name="contextor",
                    task_id=self.ctx.trace_id,
                ),
                fallback_call=lambda: invoke_llm_with_timeout_message(
                    llm_fallback.ainvoke(messages),
                    agent_name="contextor",
                    task_id=self.ctx.trace_id,
                    used_fallback=True,
                ),
            ),
        )

        return response
