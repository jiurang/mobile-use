import asyncio
import time
from pathlib import Path
from typing import Any, cast

from jinja2 import Template
from langchain_core.messages import HumanMessage, SystemMessage

from minitap.mobile_use.agents.contextor.types import AppLockVerificationOutput, ContextorOutput
from minitap.mobile_use.agents.planner.types import Subgoal
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.controllers.controller_factory import create_device_controller
from minitap.mobile_use.controllers.platform_specific_commands_controller import (
    get_current_foreground_package_async,
    get_device_date,
)
from minitap.mobile_use.graph.state import State
from minitap.mobile_use.services.llm import get_llm, invoke_llm_with_timeout_message, with_fallback
from minitap.mobile_use.services.ocr import get_ocr_engine
from minitap.mobile_use.services.ocr.types import OcrEngine, OcrText
from minitap.mobile_use.utils.app_launch_utils import launch_app_with_retries
from minitap.mobile_use.utils.decorators import wrap_with_callbacks
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.screen_elements import ScreenElement, build_screen_elements
from minitap.mobile_use.utils.ui_hierarchy_view import normalize_hierarchy

logger = get_logger(__name__)


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

        # ---- 屏幕沉降等待（动作后至少等这么久才抓屏）----
        #
        # 上滑换页之后，交互控件是**延迟挂载**的：实测新页面的 like 类
        # resource-id 在动作后 2.0s / 2.5s 才出现在层级树里。抓早了会拿到
        # 一棵「控件还没挂上」的树，看起来像「这个按钮不存在」。
        #
        # 它是**下限**不是固定延迟：距离上一个动屏动作已经够久时（真机实测
        # 决策延迟中位 19.2s，绝大多数轮次都属于这种情况）一个字都不等。
        await self._wait_for_settle(state)

        # OCR 走 on_screenshot 回调，在 u2 采集线程里就地跑。
        # dump_hierarchy 真机要 ~1.2s，OCR 只要 ~0.4s，截图一到位就开始，
        # 整段推理藏进 dump 的等待窗口里，关键路径不增加。
        ocr_box: list[list[OcrText] | None] = []
        ocr_done = False

        def _run_ocr_inline(screenshot_base64: str) -> None:
            nonlocal ocr_done
            try:
                ocr_box.append(self._recognize_ocr(screenshot_base64))
            finally:
                ocr_done = True

        # ---- 屏幕是不是灭的 ----
        #
        # 息屏时 uiautomator **仍然**会返回元素 —— 2026-10-21 真机实测
        # （Redmi Turbo 4 / Android 16）熄屏后拿到的是 **19 个 systemui 残留**
        # （通知面板容器、状态栏、keyguard 容器……），截图才是全黑的。
        #
        # 「层级树为空 ⇒ 屏幕灭了」这个判据因此**永远不会触发** —— 它是被实测
        # 证伪的，不是推断错的。唤醒功能等于没接上。所以这里改成**无条件**读
        # 电源状态（`dumpsys power` 的 `mWakefulness`），那是系统的权威状态。
        #
        # 代价被并发吃掉了：`dumpsys power` 要 ~100ms，和抓屏**同时**跑，
        # 这 100ms 就藏进了 dump 的等待窗口里（真机实测 698ms → 589ms，
        # 8 轮元素数稳定，没有连接池被抽干）。
        screen_state: str | None = None
        power_task: asyncio.Task[str] | None = None
        power_reader = getattr(device_controller, "get_screen_power_state", None)
        if power_reader is not None:
            power_task = asyncio.create_task(power_reader())
            # `create_task` 只是把协程排进日程表，**不会**立刻开跑 —— 要等当前
            # 协程让出控制权。这一行显式让一次，电源状态那个 adb 调用才会真正
            # 启动，然后和下面的抓屏重叠跑（否则就退化成串行 +100ms）。
            await asyncio.sleep(0)
        try:
            try:
                device_data = await device_controller.get_screen_data(on_screenshot=_run_ocr_inline)
            except TypeError:
                # 自定义控制器没跟上新签名时不能整轮挂掉，退回原来的串行路径。
                logger.warning("Controller does not support on_screenshot; using serial path")
                device_data = await device_controller.get_screen_data()
        finally:
            # 抓屏炸了也要把电源状态收回来，不能留一个悬空的 task。
            if power_task is not None:
                screen_state = await power_task

        if screen_state == "off":
            logger.warning("[contextor] Screen is OFF — agent will be told to wake it up.")

        if not ocr_done:
            # 回调没被调用（平台不支持 / 提前抛错）——补跑一次。
            ocr_box.append(await self._run_ocr(device_data.base64))

        current_app_package = await get_current_foreground_package_async(self.ctx)
        device_date = get_device_date(self.ctx)
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

        # ---- 屏幕元素表：UI 层级树 + OCR 合成**一张**统一索引的表 ----
        #
        # 为什么建在这里（以前在 cortex 里建）：这张表**就是索引契约**本身 ——
        # 模型报的 `Target.index` 要还原成「它当时看到的那一行」，编号只在
        # **那一次裁剪**上成立（屏可能已经变了，执行时重新 dump 再重新编号，
        # 编号会和模型看到的不一致）。
        #
        # 所以建表必须和抓屏在**同一个节点**里做完：抓屏和建表之间不可能有
        # 任何动作插进来，工具读到的就一定是模型看到的那一屏的表。

        # `perception.ui_hierarchy` 是「要不要把层级树给模型看」的开关（不是抓不抓，
        # 控制器照样把树返回来）。关掉时整条层级树通道都不走，表里也不会有 UI 行。
        # 用 getattr 兜底：精简过的 perception 对象（测试替身）可能没这个字段，
        # 缺省按 PerceptionConfig 的默认值 True（= 开）处理。
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
        # tree_rows is None = 层级树形状不认识（iOS / 云设备）→ 建不出表，
        # 元素表留空。工具侧会因此走 `latest_ocr_results` 的老路径
        # （tools/utils.py:384-398），那边**不用改**。
        # 这里是「不猜」：猜错就是静默剥夺别的平台的感知能力。

        return await state.asanitize_update(
            ctx=self.ctx,
            update={
                # 索引契约本身，**不能被清空**：工具要把 Target.index 还原成
                # 当屏那一行的真实坐标，只有这一次裁剪的结果算数。
                # 下一次 contextor 会整体覆盖它。
                "latest_screen_elements": screen_elements,
                # 建完表就清空层级树：它没有索引契约，工具侧走
                # controller.get_ui_elements() 现取即可。留着它会让工具侧
                # 按 id / 文字查找的分支复活（tools/utils.py:178,197）——
                # 那是行为变更，不在本次范围内。
                "latest_ui_hierarchy": None,
                "latest_ocr_results": ocr_results,
                "latest_screenshot": device_data.base64,
                "focused_app_info": current_app_package,
                "screen_size": (device_data.width, device_data.height),
                "device_date": device_date,
                "screen_state": screen_state,
                "agents_thoughts": [agent_outcome],
            },
            agent="contextor",
        )

    async def _wait_for_settle(self, state: State) -> float:
        """抓屏之前，等屏幕沉降到配置的下限。

        返回实际等待的秒数（0 表示没等）。
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

    def _resolve_ocr_engine(self, screenshot_base64: str | None) -> OcrEngine | None:
        """OCR 的开关 + 可用性检查。两条执行路径共用。"""
        if not self.ctx.perception.ocr:
            return None
        if not screenshot_base64:
            logger.warning("No screenshot available; skipping OCR")
            return None
        return get_ocr_engine(self.ctx)

    def _recognize_ocr(self, screenshot_base64: str) -> list[OcrText] | None:
        """**同步**跑 OCR。只能在已经不在事件循环上的线程里调用。

        复用 `get_screen_data()` 已经取到的**原分辨率**截图，不再单独截一次 ——
        OCR 的框要精确到能当点击目标用，拿压缩过的图会把框算歪。

        失败只降级不中断：OCR 是增强感知，层级树和截图还在。
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

        MAX_AGENTS_THOUGHTS = 25

        system_message = Template(
            Path(__file__).parent.joinpath("contextor.md").read_text(encoding="utf-8")
        ).render(
            task_goal=initial_goal,
            subgoal_plan="\n".join([str(subgoal) for subgoal in subgoal_plan]),
            locked_app_package=locked_app_package,
            current_app_package=current_app_package,
            agents_thoughts=agents_thoughts[:MAX_AGENTS_THOUGHTS],
        )

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content="Please make your decision."),
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
