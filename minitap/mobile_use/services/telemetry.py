"""Telemetry service for mobile-use SDK using PostHog."""

import hashlib
import json
import platform
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import uuid_utils

from minitap.mobile_use.config import settings
from minitap.mobile_use.utils.llm_usage import extract_usage_full
from minitap.mobile_use.utils.logger import get_logger

if TYPE_CHECKING:
    # posthog is only needed when telemetry is actually enabled, which the APK
    # (MOBILE_USE_TELEMETRY_ENABLED=false) never is. It sits on the import path
    # of services.llm, so a top-level import made every Android deployment
    # depend on it.
    from posthog import Posthog

logger = get_logger(__name__)


POSTHOG_API_KEY = "phc_MTwMcqOjMpTdTdrYwQUlsWaKkB7C8MPAw9YyZhRv8B8"
POSTHOG_HOST = "https://eu.i.posthog.com"
EVENT_PREFIX = "mobile_use_"

TELEMETRY_CONFIG_DIR = Path.home() / ".minitap"
TELEMETRY_CONFIG_FILE = TELEMETRY_CONFIG_DIR / "telemetry.json"

# 2026-10-21（Executor 自主执行改造）：cortex 不再产出 decisions / goals_completion，
# 这两个字段在新架构下**恒为 False**。继续上报它们，等于在看板里放一个
# 「看着有数、其实是常数」的假指标 —— 所以新的 cortex_decision 事件不再带这两个字段，
# 只在 payload 里留一份 deprecated_fields 名单说明它们去了哪。
DEPRECATED_CORTEX_DECISION_FIELDS = ("has_decisions", "has_goals_completion")


def subgoal_id_for(goal: str, index: int) -> str:
    """给子目标生成一个**不含目标原文**的稳定标识。

    事件里不能出现目标陈述（那是用户数据，屏幕内容、账号名都在里面），
    但又需要把同一子目标的「步数 / 汇报 / 重发次数」归到同一条线上 ——
    所以用「序号 + 目标文本的短哈希」当 id：同一段目标文本永远得到同一个 id
    （被 Cortex 换招重发时也稳定），而 id 本身反推不出目标内容。
    """
    digest = hashlib.sha256(goal.encode("utf-8")).hexdigest()[:8]
    return f"sg{index}-{digest}"


class TelemetryConfig:
    """Manages telemetry consent and configuration persistence."""

    def __init__(self):
        self._distinct_id: str | None = None
        self._enabled: bool | None = None

    @property
    def config_exists(self) -> bool:
        return TELEMETRY_CONFIG_FILE.exists()

    def load(self) -> dict:
        """Load telemetry config from disk."""
        if not self.config_exists:
            return {}
        try:
            with open(TELEMETRY_CONFIG_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def save(self, enabled: bool, distinct_id: str) -> None:
        """
        Save telemetry config to disk.

        Only persists if enabled=True. If disabled, we don't persist
        so the user will be asked again next session.
        """
        TELEMETRY_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        if enabled:
            config = {"enabled": enabled, "distinct_id": distinct_id}
            with open(TELEMETRY_CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=2)
        else:
            # Don't persist denial - user will be asked again next session
            # But still save distinct_id for consistency if they enable later
            config = {"distinct_id": distinct_id}
            with open(TELEMETRY_CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=2)

    @property
    def distinct_id(self) -> str:
        """Get or generate a persistent anonymous distinct ID."""
        if self._distinct_id is None:
            config = self.load()
            self._distinct_id = config.get("distinct_id") or str(uuid_utils.uuid4())
        return self._distinct_id

    @property
    def enabled(self) -> bool | None:
        """
        Check if telemetry is enabled.

        Priority:
        1. Environment variable MOBILE_USE_TELEMETRY_ENABLED (true/false/1/0)
        2. Persisted config file
        3. None if not configured (needs consent prompt)
        """
        if self._enabled is not None:
            return self._enabled

        telemetry_enabled = settings.MOBILE_USE_TELEMETRY_ENABLED
        if telemetry_enabled is not None:
            self._enabled = telemetry_enabled
            return self._enabled

        config = self.load()
        if "enabled" in config:
            self._enabled = config["enabled"]
            return self._enabled

        return None

    def set_enabled(self, enabled: bool) -> None:
        """Set telemetry enabled state and persist to config."""
        self._enabled = enabled
        self.save(enabled=enabled, distinct_id=self.distinct_id)


class TelemetryService:
    """PostHog telemetry service for mobile-use SDK."""

    _instance: "TelemetryService | None" = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self):
        self._config = TelemetryConfig()
        self._initialized = False
        self._client: Posthog | None = None
        self._session_id: str | None = None
        self._session_context: dict = {}

    @classmethod
    def get_instance(cls) -> "TelemetryService":
        """Get singleton instance of TelemetryService (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = TelemetryService()
        return cls._instance

    @property
    def is_enabled(self) -> bool:
        """Check if telemetry is enabled."""
        return self._config.enabled is True

    @property
    def needs_consent(self) -> bool:
        """Check if user consent is needed (not yet configured)."""
        return self._config.enabled is None

    def initialize(self) -> None:
        """Initialize PostHog client if telemetry is enabled."""
        if self._initialized:
            return

        if not self.is_enabled:
            logger.debug("Telemetry disabled, skipping PostHog initialization")
            return

        try:
            from posthog import Posthog

            self._client = Posthog(
                project_api_key=POSTHOG_API_KEY,
                host=POSTHOG_HOST,
                debug=False,
            )
            self._initialized = True
            logger.debug("PostHog telemetry initialized")
        except Exception as e:
            logger.debug(f"Failed to initialize PostHog: {e}")
            self._initialized = False

    def set_consent(self, enabled: bool) -> None:
        """Set user consent for telemetry."""
        self._config.set_enabled(enabled)
        if enabled:
            self.initialize()
        else:
            if self._client:
                self._client.disabled = True

    def start_session(self, context: dict | None = None) -> str:
        """
        Start a new telemetry session for CLI usage.

        Args:
            context: Initial session context (e.g., goal, platform, device_id)

        Returns:
            The session ID
        """
        self._session_id = str(uuid_utils.uuid7())
        self._session_context = context or {}

        if self.is_enabled:
            self.capture(
                "session_started",
                {
                    "$session_id": self._session_id,
                    **self._session_context,
                },
            )

        return self._session_id

    def update_session_context(self, context: dict) -> None:
        """Update the current session context with additional data."""
        self._session_context.update(context)

    def end_session(self, success: bool = True, error: str | None = None) -> None:
        """
        End the current telemetry session.

        Args:
            success: Whether the session completed successfully
            error: Error message if session failed
        """
        if not self._session_id:
            return

        if self.is_enabled:
            self.capture(
                "session_ended",
                {
                    "$session_id": self._session_id,
                    "success": success,
                    "error": error,
                    **self._session_context,
                },
            )

        self._session_id = None
        self._session_context = {}

    def capture_action(self, action: str, details: dict | None = None) -> None:
        """
        Capture an action within the current session.

        Args:
            action: The action name (e.g., "screenshot_taken", "tap_performed")
            details: Additional action details
        """
        properties = {
            "action": action,
            **(details or {}),
        }

        if self._session_id:
            properties["$session_id"] = self._session_id
            properties.update(self._session_context)

        self.capture("action", properties)

    def capture(self, event: str, properties: dict | None = None) -> None:
        """Capture a telemetry event."""
        if not self.is_enabled:
            return

        # Lazy initialization for SDK usage (non-CLI)
        # If user has previously consented via config/env, auto-initialize
        if not self._initialized:
            self.initialize()
            if not self._initialized:
                return

        try:
            all_properties = {
                "sdk_version": self._get_sdk_version(),
                "python_version": platform.python_version(),
                "os": platform.system(),
                "os_version": platform.release(),
                **(properties or {}),
            }

            # Include session ID if available
            if self._session_id and "$session_id" not in all_properties:
                all_properties["$session_id"] = self._session_id

            if self._client:
                prefixed_event = f"{EVENT_PREFIX}{event}"
                self._client.capture(
                    distinct_id=self._config.distinct_id,
                    event=prefixed_event,
                    properties=all_properties,
                )
        except Exception as e:
            logger.debug(f"Failed to capture telemetry event: {e}")

    def capture_exception(
        self,
        exception: Exception,
        context: dict | None = None,
    ) -> None:
        """
        Capture an exception event to PostHog Error Tracking.

        Uses PostHog's native capture_exception method for proper
        integration with the Error Tracking dashboard.

        Args:
            exception: The exception to capture
            context: Additional context properties
        """
        if not self.is_enabled:
            return

        # Lazy init for SDK usage
        if not self._initialized:
            self.initialize()
            if not self._initialized:
                return

        try:
            if self._client:
                # Use PostHog's native capture_exception for Error Tracking
                self._client.capture_exception(
                    exception,
                    distinct_id=self._config.distinct_id,
                    properties={
                        "sdk_version": self._get_sdk_version(),
                        "source": "mobile_use",
                        **(context or {}),
                    },
                )
        except Exception as e:
            logger.debug(f"Failed to capture exception telemetry: {e}")

    def capture_llm_usage(
        self,
        task_id: str,
        agent: str,
        model: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        duration_seconds: float | None = None,
        used_fallback: bool = False,
    ) -> None:
        """记录一次 LLM 调用的 token 用量。

        这是成本核算的**唯一数据源**。之前 `doc/token-context-analysis.md` 里的所有
        token 数字都是按公式离线估算的（例如截图按 Gemini tile 公式
        `(w//768+1)*(h//768+1)*258`），无法验证真实端到端用量，也就无法证明
        截图降分辨率这类优化到底省了多少。接上之后才是真实 API 返回的 usage。

        `cached_tokens` 是**本次输入中命中前缀缓存的 token 数**（input 的子集）。
        单独记是因为它是第三个数：并进 input 会算错成本，丢掉它则遥测里没有
        「命中率」这个指标 —— 而提示词分层的收益只能靠它证明。
        2026-10-21 起才有这个字段；旧 trace 里没有，按 0 处理。

        Args:
            task_id: 任务 trace id。
            agent: 调用方 agent 名（cortex / executor / planner ...）。
            model: 模型标识。
            input_tokens / output_tokens: provider 报告的用量；读不到时传 0。
            cached_tokens: provider 报告的前缀缓存命中量；读不到时传 0。
            duration_seconds: 本次调用耗时。
            used_fallback: 是否走了 fallback 模型。
        """
        self.capture(
            "llm_usage",
            {
                "task_id": task_id,
                "agent": agent,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_tokens": cached_tokens,
                "total_tokens": input_tokens + output_tokens,
                "duration_seconds": duration_seconds,
                "used_fallback": used_fallback,
            },
        )

    def capture_task_started(
        self,
        task_id: str,
        platform: str,
        has_locked_app: bool = False,
    ) -> None:
        """Capture task started event."""
        self.capture(
            "task_started",
            {
                "task_id": task_id,
                "device_platform": platform,
                "has_locked_app": has_locked_app,
            },
        )

    def capture_task_completed(
        self,
        task_id: str,
        success: bool,
        steps_count: int,
        duration_seconds: float,
        cancelled: bool = False,
    ) -> None:
        """Capture task completed event."""
        self.capture(
            "task_completed",
            {
                "task_id": task_id,
                "success": success,
                "cancelled": cancelled,
                "steps_count": steps_count,
                "duration_seconds": duration_seconds,
            },
        )

    def capture_agent_initialized(self, platform: str, device_id: str | None = None) -> None:
        """Capture agent initialization event."""
        self.capture(
            "agent_initialized",
            {
                "device_platform": platform,
                "has_device_id": device_id is not None,
            },
        )

    def capture_cortex_decision(
        self,
        task_id: str,
        has_decisions: bool | None = None,
        has_goals_completion: bool | None = None,
        completed_subgoals_count: int = 0,
        *,
        has_subgoal: bool | None = None,
        subgoal_completed: bool | None = None,
        decision_source: str | None = None,
    ) -> None:
        """Capture cortex agent decision event (only non-sensitive flags).

        ⚠️ 语义变更（2026-10-21，Executor 自主执行改造）
        ---------------------------------------------------
        旧架构里 cortex 每轮都要产出两样东西：`decisions`（下一步动作）
        和 `goals_completion`（目标完成情况），所以事件带 `has_decisions` /
        `has_goals_completion` 两个标志位。

        **新架构里 cortex 不再产出这两个字段**（决策下沉给 executor），
        照旧上报的话这两个值**恒为 False** —— 看板会显示
        「cortex 从不产出决策」，那是个**假信号**，比没有数据更糟。

        所以这里做了两件事：
        1. 事件改报新架构下**真的有信息量**的字段：
           `has_subgoal`（cortex 下发了下一个子目标）/ `subgoal_completed`
           （上一个子目标被判完成）/ `completed_subgoals_count`；
        2. 旧的 `has_decisions` / `has_goals_completion` **仍然接受入参**
           （老调用点不会因为签名变化而崩），但**不再上报**，
           只在 payload 里留 `deprecated_fields` 名单说明去向。

        ⇒ 下游看板：`has_decisions` / `has_goals_completion` 这两个属性
        **会变成「没有数据」**（不是「恒为 false」）—— 这是刻意的，
        一眼能看出「指标没了、该换新字段」，不会被假数据骗过去。
        替代口径：`has_subgoal` / `subgoal_completed`。

        ⚠️ `decision_source`（2026-10-21，L3 规则换招）
        ----------------------------------------------
        失败路径上「换个路径再来」现在由**代码规则**做（省一次 LLM 调用），
        不问模型。这个决定和「模型判的」不是一回事 —— 判断力不同，
        混在一起看会得出「cortex 判完成率很高」这种被稀释过的假结论。
        所以这里带上来源标志：`"llm"` = 模型判的，`"rule"` = 规则换招。
        """
        if has_decisions is not None or has_goals_completion is not None:
            logger.debug(
                "capture_cortex_decision 收到已废弃字段 "
                "has_decisions/has_goals_completion，新架构下不再上报，"
                "请改用 has_subgoal / subgoal_completed"
            )

        self.capture(
            "cortex_decision",
            {
                "task_id": task_id,
                "has_subgoal": has_subgoal,
                "subgoal_completed": subgoal_completed,
                "completed_subgoals_count": completed_subgoals_count,
                # "llm" = 模型判的；"rule" = 规则换招（失败路径，不调 LLM）
                "decision_source": decision_source,
                # 事件结构版本：给下游一个判断「这条数据是新口径还是老口径」的依据
                "schema_version": 2,
                "deprecated_fields": list(DEPRECATED_CORTEX_DECISION_FIELDS),
            },
        )

    def capture_executor_dispatch(
        self,
        task_id: str,
        dispatched: bool = True,
        *,
        action_count: int = 0,
        rejection_code: str | None = None,
    ) -> None:
        """Capture whether the executor action was accepted by the deterministic dispatcher.

        ⚠️ 语义变更（2026-10-21，Executor 自主执行改造）
        ---------------------------------------------------
        旧架构里这条事件量的是「有没有走确定性直连快路径」：落空了就得退回
        LLM 再问一次，命中率低说明 cortex.md 的输出格式约定有问题。

        **新架构里 executor 永远走确定性直连**（没有 LLM 兜底分支了：
        dispatcher 拒绝动作时是把拒绝理由回灌给 executor 让它重选），
        所以「走没走直连」**恒为 True** —— 又是那个「看着有数、其实是常数」
        的假指标。

        现在这个字段表示的是**这一步的动作有没有被 dispatcher 接受**：
        被拒绝时（工具名不存在 / 必填参数缺失）`dispatched=False`，
        拒绝理由会回灌给 executor 让它重新决策。
        换句话说，它从「直连命中率」变成了「**executor 输出合法率**」——
        后者在新架构下是有信息量的（也正是 §10 R6 要单独统计的解析失败率）。

        ⇒ 下游看板：原来那张「直连命中率」卡片已经**失效**
        （新架构下必然 100%）。请改看 `dispatched=false` 的占比。
        `action_count` 是这一步产出的动作数，`rejection_code` 是拒绝原因的
        **短代码**（如 `unknown_tool` / `invalid_payload`）——
        只能是代码，不能塞拒绝理由原文（那里面可能带用户数据）。

        Args:
            task_id: 任务 trace id。
            dispatched: 这一步的动作有没有被 dispatcher 接受。
            action_count: 这一步产出的动作数。
            rejection_code: 拒绝原因短代码；没被拒绝时留空。
        """
        self.capture(
            "executor_dispatch",
            {
                "task_id": task_id,
                "dispatched": dispatched,
                "action_count": action_count,
                "rejection_code": rejection_code,
                "schema_version": 2,
            },
        )

    def capture_executor_action(
        self,
        task_id: str,
        tool_name: str,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Capture executor tool action event."""
        self.capture(
            "executor_action",
            {
                "task_id": task_id,
                "tool_name": tool_name,
                "success": success,
                "error": error,
            },
        )

    # ------------------------------------------------------------------
    # Executor 自主执行架构（2026-10-21）新增的三个事件
    # ------------------------------------------------------------------

    def capture_executor_step(
        self,
        task_id: str,
        step_index: int,
        actions_count: int = 0,
        done: bool = False,
        input_tokens: int = 0,
        output_tokens: int = 0,
        *,
        response: Any = None,
    ) -> None:
        """记录 Executor 自主循环里的一步（一次 LLM 决策）。

        新架构里决策权下沉到 executor：cortex 每子目标才跑一次，
        executor 每步都要自己看元素表选一个动作。所以「每步花了多少 token」
        成了成本核算的主战场 —— 方案 §11.2 的关键判据
        「executor 每轮 output < 300 tok」就落在这个事件上
        （它同时也是「墙钟时间能不能翻盘」的唯一杠杆，见方案 §2.3）。

        ⚠️ `input_tokens` / `output_tokens` 为 0 表示**这次没读到用量**
        （provider 没报，或结构化输出路径把原始响应藏起来了），
        **不是「这次免费」** —— 这个区别和 `capture_llm_usage` 一样要紧。
        能拿到原始响应时把它传进 `response=`，本方法用
        `utils/llm_usage.extract_usage_full` 统一解析
        （与 TUI、services/llm.py 共用同一份实现，usage 形状最容易漂移）。

        Args:
            task_id: 任务 trace id。
            step_index: 本子目标内的步序号。
            actions_count: 这一步产出的动作数（executor 一次通常只出 1 个）。
            done: 这一步是否判定目标达成。
            input_tokens / output_tokens: 本步的 token 用量；读不到时留 0。
            response: 可选的 LLM 原始响应；给了就优先从它解析用量。
        """
        if response is not None:
            usage = extract_usage_full(response)
            if usage is not None:
                input_tokens = usage.input_tokens
                output_tokens = usage.output_tokens

        self.capture(
            "executor_step",
            {
                "task_id": task_id,
                "step_index": step_index,
                "actions_count": actions_count,
                "done": done,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )

    def capture_reporter_report(
        self,
        task_id: str,
        outcome: str,
        summary_len: int = 0,
        has_evidence: bool = False,
        failed_count: int = 0,
    ) -> None:
        """记录一次汇报（reporter 把执行日志压成「结论层 + 事实层」）。

        `outcome` 取值：`completed` / `failed` / `blocked` —— 由代码判定
        （`determine_outcome`），不是 LLM 说了算。这条事件是方案 §11.6 里
        「outcome 分布」指标的来源：用来验证熔断真的在工作，
        也用来算「误判完成率」（outcome=completed 但实际没完成）。

        ⚠️ 这里**只有长度和计数，没有正文**：summary 和 evidence 都是 LLM 写的
        用户数据（目标陈述、屏幕内容都在里面），不能进事件。
        `summary_len` / `has_evidence` / `failed_count` 足够算出
        摘要长度分布、有证据率、失败率。

        Args:
            task_id: 任务 trace id。
            outcome: completed / failed / blocked。
            summary_len: 摘要字数（不是摘要本身）。
            has_evidence: 有没有带上证据。
            failed_count: 失败目标条数。
        """
        self.capture(
            "reporter_report",
            {
                "task_id": task_id,
                "outcome": outcome,
                "summary_len": summary_len,
                "has_evidence": has_evidence,
                "failed_count": failed_count,
            },
        )

    def capture_subgoal_progress(
        self,
        task_id: str,
        subgoal_id: str,
        steps: int = 0,
        outcome: str = "unknown",
        retry_count: int = 0,
    ) -> None:
        """记录一个子目标跑完时的结果（步数 / 结果 / 被重发了几次）。

        这是方案 §11.6「steps_per_subgoal」指标的来源 —— 步数上限只能靠它校准
        （§4.3 的实测步数区间就是从这里来的）。

        `retry_count` = 当前子目标被**重发**过几次（也就是 cortex 把它重新
        下发了几次）。

        ⚠️ 口径：它是「下发次数 − 1」，**不是**「outcome 不是 completed 的
        次数」—— 重发的三条路径里有两条（规则换招、撞步数上限）根本不产生
        失败事实，按后者算会**恒为 0**（真机实测踩过：实际重发 5 次，报 0）。
        详见 `agents/reporter/reporter.py` 的 `_retry_count`。

        ⚠️ `subgoal_id` 必须是**不含目标原文**的标识：目标陈述是用户数据。
        用 `subgoal_id_for(goal, index)` 生成。

        Args:
            task_id: 任务 trace id。
            subgoal_id: 子目标的不含原文的标识（见 `subgoal_id_for`）。
            steps: 这个子目标跑了多少步。
            outcome: completed / failed / blocked / unknown。
            retry_count: 这个子目标被重发的次数。
        """
        self.capture(
            "subgoal_progress",
            {
                "task_id": task_id,
                "subgoal_id": subgoal_id,
                "steps": steps,
                "outcome": outcome,
                "retry_count": retry_count,
            },
        )

    def capture_replan(
        self,
        task_id: str,
        plan_version: int = 0,
        failed_subgoals: int = 0,
        completed_subgoals: int = 0,
        total_subgoals: int = 0,
    ) -> None:
        """记录一次**重规划**（计划级失败 → Planner 重新拆计划）。

        这是方案 §11.6「replan 触发计数」指标的来源 —— 它的用途只有一个：
        **验证看门狗还活着（R16）**。

        为什么这个计数能验证看门狗
        ------------------------
        R16 说的是：「计划级失败（子目标根本做不到）不能没人管」。新拓扑里
        它的处理链是::

            reporter → orchestrator（看门狗）→ convergence → **replan** → planner

        也就是说，**每一次真正走到 replan，就证明这条链是通的**。反过来，
        如果一个任务跑了很久却**一次 replan 都没有**，那要么是它一次没失败过
        （正常），要么是**看门狗断了**（子目标失败了但没人接手，Cortex 一直
        重发到撞 `recursion_limit`）—— 这两种情况光看别的指标分不开，
        只有一个 replan 计数能分开。

        ⚠️ 埋点位置在 **Planner 节点**（它正好算得出 `needs_replan`），
        不是在 `convergence_gate`（那是纯函数，拿不到 task_id）。

        ⚠️ 这里**不带任何目标原文**：子目标描述是用户数据。
        只报数量。

        Args:
            task_id: 任务 trace id。
            plan_version: 这一版计划的版本号（第几次计划）。
            failed_subgoals: 计划里被标记失败的子目标数。
            completed_subgoals: 计划里已完成的子目标数。
            total_subgoals: 计划里的子目标总数。
        """
        self.capture(
            "replan",
            {
                "task_id": task_id,
                "plan_version": plan_version,
                "failed_subgoals": failed_subgoals,
                "completed_subgoals": completed_subgoals,
                "total_subgoals": total_subgoals,
            },
        )

    def flush(self) -> None:
        """Flush pending events to PostHog."""
        if self.is_enabled and self._initialized and self._client:
            try:
                self._client.flush()
            except Exception as e:
                logger.debug(f"Failed to flush telemetry: {e}")

    def shutdown(self) -> None:
        """Shutdown telemetry service."""
        self.flush()
        if self._client:
            try:
                self._client.shutdown()
            except Exception:
                pass

    def _get_sdk_version(self) -> str:
        """Get the mobile-use SDK version."""
        try:
            from importlib.metadata import version

            return version("minitap-mobile-use")
        except Exception:
            return "unknown"


telemetry = TelemetryService.get_instance()
