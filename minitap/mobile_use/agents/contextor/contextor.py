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
    get_device_date,
)
from minitap.mobile_use.graph.state import State
from minitap.mobile_use.services.llm import get_llm, invoke_llm_with_timeout_message, with_fallback
from minitap.mobile_use.services.ocr import get_ocr_engine
from minitap.mobile_use.services.ocr.types import OcrEngine, OcrText
from minitap.mobile_use.utils.app_launch_utils import launch_app_with_retries
from minitap.mobile_use.utils.decorators import wrap_with_callbacks
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.screen_elements import (
    ScreenElement,
    build_screen_elements,
    foreground_package_from_tree,
)
from minitap.mobile_use.utils.screen_stability import (
    is_stable,
    structural_change_count,
    structural_digest,
    tree_fingerprint,
)
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

        # ---- 屏幕沉降：等下限 + 确认这一屏已经定下来 ----
        #
        # 🔴 **2026-10-22 改：原来是「纯等 3 秒」，现在是「等 200ms + 确认稳定」。**
        #
        # 原来那个 3 秒的依据（「控件延迟挂载，抓早了拿到半成品」）经真机实测
        # 拆成了两半：
        #
        # * **「防动画」那部分不需要外部下限** —— ``dump_hierarchy`` 自带
        #   waitForIdle，屏幕还在动的时候它**自己会等**（实测滚动场景里
        #   dump#0 耗时 609ms，那 609ms 就是它在等）。外部下限在做同一件事，
        #   只是更粗、更贵。
        # * **「懒加载」那部分外部下限也挡不住** —— 它挡的是「静默期内才挂上
        #   的控件」，而那需要时间维度上的多次采样才能发现。
        #
        # 所以现在分两步：
        #
        # 1. **等下限**（``settle_delay_ms``，默认 200ms）—— 防「动作刚发出、
        #    屏幕还没开始变」的竞态。
        # 2. **确认稳定**（``_confirm_tree_stable``）—— 连续两次树级 dump 比
        #    指纹，相同才算定下来了。**只 dump 树，不截图不跑 OCR**
        #    （树级 dump 实测 150~250ms，整段 get_screen_data 要 600ms）。
        #
        # ⚠️ 「静 ≠ 全」：新页面的控件是懒加载的，实测 dump#1/dump#2 内容已经
        # 一样（91 个元素）但 228 才是完整的 ⇒ 稳定确认**会漏判**这种情况。
        # 这是已知代价，验收看「步数 / 失败目标数」，不看「点错率」。
        await self._wait_for_settle(state)

        # ---- 稳定性采样 ----
        #
        # * ``fingerprint``（默认，现行）：连续两次**全量树** dump 比指纹。
        # * ``structural``：**只采一次**，这次采样当「结构基线」；真正的比对放在
        #   建表那次 dump 之后（``get_screen_data`` 自带一次树 dump）⇒
        #   常态 **2 次 dump**（现行 3~4 次）。
        #
        # 为什么要合并：现行路径把稳定性确认的采样**扔掉**，然后 ``get_screen_data``
        # 又 dump 一次建表 —— 同一棵屏被抓两遍。视频页上更贵：全量指纹含 text，
        # 点赞数每秒在跳 ⇒ 永远判不稳 ⇒ 3 次采样全跑满，最后还是「照用」。
        stability_sample: list[dict] | None = None
        if self._structural_mode():
            stability_sample = await device_controller.get_hierarchy_only()
            if not stability_sample:
                logger.warning(
                    "[contextor] Structural stability: could not sample the tree; skipping check"
                )
        else:
            await self._confirm_tree_stable(device_controller, state)

        # OCR 走 on_screenshot 回调，在 u2 采集线程里就地跑。
        #
        # ⚠️ **2026-10-22 改：OCR 通道关掉时，OCR 根本不跑。**
        # 以前 `perception.ocr` 只管「跑完之后要不要给模型」，OCR 引擎照样
        # 每步跑一遍 —— 真机实测 **381~443ms/步**，占整段感知的一大截，
        # 而结果被丢掉。Manager 把通道关掉的意思是「这条链路不要了」，
        # 不是「照跑但别给」。
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
        # 设备日期：**和抓屏并发**。原来它是抓屏之后的**串行** adb 往返
        # （`device.shell("date")`，实测中位 88ms），全额计入关键路径。
        # 改成 to_thread 之后它和 dump 的等待窗口重叠，净增 ≈ 0。
        #
        # ⚠️ 保留读**设备**时钟而不是用本机时钟：设备时区可能和 PC 不同，
        # 那会让 prompt 里的 "Device date" 说错时间。（iOS 那条路本来就用
        # 本机时钟，那是另一回事。）
        date_task = asyncio.create_task(asyncio.to_thread(get_device_date, self.ctx))
        if power_reader is not None:
            # `create_task` 只是把协程排进日程表，**不会**立刻开跑 —— 要等当前
            # 协程让出控制权。这一行显式让一次，电源状态 / 日期那两个 adb 调用
            # 才会真正启动，然后和下面的抓屏重叠跑（否则就退化成串行 +100ms）。
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
                # 自定义控制器没跟上新签名时不能整轮挂掉，退回原来的串行路径。
                logger.warning("Controller does not support on_screenshot; using serial path")
                device_data = await device_controller.get_screen_data()
        finally:
            # 抓屏炸了也要把电源状态和日期收回来，不能留两个悬空的 task
            # （悬空 task 会在事件循环关闭时报 "Task exception was never
            # retrieved"，把真正的异常盖掉）。
            if power_task is not None:
                screen_state = await power_task
            device_date = await date_task

        if screen_state == "off":
            logger.warning("[contextor] Screen is OFF — agent will be told to wake it up.")

        if ocr_enabled and not ocr_done:
            # 回调没被调用（平台不支持 / 提前抛错）——补跑一次。
            ocr_box.append(await self._run_ocr(device_data.base64))

        # 结构稳定性判定（``structural`` 模式）：基线 = 上面那次采样，
        # 当前 = 建表用的这棵树（``get_screen_data`` 自带的那次 dump）。
        if stability_sample is not None:
            await self._verify_structural_stability(
                device_controller=device_controller,
                state=state,
                baseline=stability_sample,
                current=cast(list[dict], device_data.elements),
            )

        # 前台包名：**从刚抓回来的树里推**，不再单独跑一次
        # `dumpsys window | grep mCurrentFocus`（那是 127ms 的串行 adb 往返，
        # 实测省掉它整段感知快 40%）。详见
        # `utils/screen_elements.foreground_package_from_tree`。
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

        ⚠️ **它现在只是「起步下限」，不再是完整的沉降保证。**
        「这一屏定下来没有」由 :meth:`_confirm_tree_stable` 判。
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

        为什么视频页必须换判据
        ----------------------
        现行指纹含 ``text``。**视频页每秒都在变** —— 点赞数、倒计时、进度、
        推荐流内容 ⇒ 指纹**永远不相等** ⇒ ``_confirm_tree_stable`` 每次都跑满
        重试次数，最后还是「判定不稳，照用」：**三倍的 dump，换同一个结果**。

        新判据只比「**可交互元素的身份 + 位置**」，**不看内容**。依据是本仓库
        已经写下的结论（``tools/live_target.py``）：

            OCR 的**内容会变**（点赞数在跳、时钟在走）
            **位置比内容稳**

        视频页上这套判据会怎么判
        ----------------------
        * 点赞数 / 倒计时在跳 → 那些 text 变了，但**身份+位置没变** → **判稳** ✓
        * 视频画面在动 → 画面不是可交互元素 → 摘要不变 → **判稳** ✓
        * 弹窗 / 翻页 → 可交互元素增删或位移 → 摘要变 → **判不稳** ✓
          （要挡的正是这个 —— 过渡态的按钮位置还没最终确定）
        * 懒加载控件挂上 → 可交互元素变多 → 摘要变 → **判不稳** → 补采一次 ✓

        判不稳时为什么可以照用
        --------------------
        执行时是**按 resource-id 重新定位**的（``tools/live_target.py``）：元素
        挪了位置也能点对；元素真没了，live_target 会给出精确失败原因（"不再在屏
        上"而不是"指错了"），让模型重读屏幕。所以「位置没定下来」不是致命的，
        致命的是「模型报了一个屏上根本没有的东西」—— 那和稳不稳无关。
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
        """确认这一屏已经定下来 —— 连续两次树级 dump 的指纹相同。

        为什么要有这一步
        ----------------
        沉降下限从 3000ms 降到 200ms 之后，「抓早了」的风险回来了。实测
        （Redmi Turbo 4 / Android 16）动作后立刻抓屏会拿到**过渡态**：

            页面导航（点进 WLAN）：49 → 91 → 91 → **228** 个元素
            列表滚动：           292 → 267 → 240 → **239** 个元素
            返回键：             **141**（一次就稳）

        这一步就是**把「还在动」挡掉**：连续两次采样内容一样 ⇒ 屏定下来了。

        代价（为什么说它便宜）
        --------------------
        * **只 dump 树**，不截图、不跑 OCR —— 树级 dump 实测 150~250ms，
          整段 ``get_screen_data`` 要 600ms（截图 166 + PNG 编码 79 + OCR 381）。
        * **不做无谓的重抓**：没动过屏的步骤（读笔记、等一下）**完全跳过**。
        * 两次采样**连着发**，中间不额外等 —— ``dump_hierarchy`` 自带
          waitForIdle，屏幕还在动的时候它自己会等。

        什么时候做
        ----------
        **上一次动作动过屏**（``last_action_at`` 有值）才做。只读步骤不动屏，
        屏早就是稳的，没必要多花两次 dump。

        ⚠️ **已知误判面：「静 ≠ 全」**
        懒加载的控件可能在静默期之后才挂上（实测 dump#1/dump#2 内容一样
        但都不是完整的）。这一挡不住。**验收看「步数」和「失败目标数」**，
        不看「点错率」。
        """
        max_retries = getattr(self.ctx.perception, "settle_max_retries", 0)
        if max_retries <= 0 or getattr(state, "last_action_at", None) is None:
            return False

        previous: str | None = None
        for attempt in range(max_retries + 1):
            elements = await device_controller.get_hierarchy_only()
            if not elements:
                # 拿不到树就当「没法确认」——保留现状，别在这里把流程卡住。
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
