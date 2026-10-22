"""宏的数据模型：一步 = 一次工具调用；一个宏 = 一串步骤 + 前置条件。

（模型部分刻意不 import 任何设备相关的东西 —— 它只是数据。）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: 这些工具的参数里有 ``target``，需要「在当屏重新定位」。
TARGET_TOOLS = frozenset(
    {"tap", "long_press", "input_text", "clear_text", "erase_one_char"}
)

#: 一步的默认「等目标出现」时长（秒）。
DEFAULT_WAIT_TIMEOUT_S = 8.0
#: 一步的默认重试次数（「元素不在屏上」时等它出现）。
DEFAULT_MAX_RETRIES = 6


@dataclass
class MacroStep:
    """宏的一步。"""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    #: 这一步失败要不要整个宏停下来。默认停（半截执行比不执行更危险）。
    optional: bool = False
    #: 「元素不在屏上」时最多重试几次（每次间隔 0.6s）。
    max_retries: int = DEFAULT_MAX_RETRIES
    #: 目标出现前的最长等待（秒）。
    wait_timeout_s: float = DEFAULT_WAIT_TIMEOUT_S
    #: **降级链的第 2 级**：录制时目标的位置。``resource_id`` 失效时用它找。
    #:
    #: 为什么要有它：真机实测（2026-10-22）快手 4 小时内把同一个「搜索」入口的
    #: id 从 ``right_tv`` 改成 ``thanos_home_top_search`` —— **id 会变，位置基本不变**。
    #:
    #: ⚠️ 它**不在** ``args`` 里：``args`` 会原样发给工具，多余的键会被参数校验
    #: 当成「未知参数」拒掉。
    fallback_bounds: dict[str, Any] | None = None
    #: **降级链的第 3 级**：录制时的上下文指纹（邻居序列）。
    #:
    #: 位置也救不了的时候靠它 —— 屏幕尺寸变了会让中下部的元素整体平移 230px，
    #: 但**邻居**（相邻控件的文字或归一化 id）没变。详见
    #: :mod:`minitap.mobile_use.utils.screen_fingerprint`。
    fallback_context: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MacroStep:
        if not isinstance(raw, dict):
            raise ValueError("each step must be an object")
        tool = raw.get("tool")
        if not isinstance(tool, str) or not tool.strip():
            raise ValueError("each step needs a 'tool' name")
        args = raw.get("args")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            # ⚠️ 不能写 ``raw.get("args") or {}``：空列表 / 空串是 falsy，
            # 会被静默变成 {}，参数校验就形同虚设。
            raise ValueError("step 'args' must be an object")
        fallback = raw.get("fallback")
        fallback = fallback if isinstance(fallback, dict) else {}
        bounds = fallback.get("bounds")
        context = fallback.get("context")
        return cls(
            tool=tool,
            args=dict(args),
            optional=bool(raw.get("optional", False)),
            max_retries=int(raw.get("max_retries", DEFAULT_MAX_RETRIES)),
            wait_timeout_s=float(raw.get("wait_timeout_s", DEFAULT_WAIT_TIMEOUT_S)),
            fallback_bounds=bounds if isinstance(bounds, dict) else None,
            fallback_context=context if isinstance(context, dict) else None,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"tool": self.tool, "args": self.args}
        if self.optional:
            out["optional"] = True
        fallback: dict[str, Any] = {}
        if self.fallback_bounds:
            fallback["bounds"] = self.fallback_bounds
        if self.fallback_context:
            fallback["context"] = self.fallback_context
        if fallback:
            out["fallback"] = fallback
        return out

    def needs_target(self) -> bool:
        return self.tool in TARGET_TOOLS and isinstance(self.args.get("target"), dict)

    def wanted_selector(self) -> str | None:
        target = self.args.get("target")
        if not isinstance(target, dict):
            return None
        resource_id = target.get("resource_id")
        return f"resource_id={resource_id}" if isinstance(resource_id, str) and resource_id else None


@dataclass
class Macro:
    """一个可回放的宏。"""

    name: str
    steps: list[MacroStep] = field(default_factory=list)
    when_to_use: str = ""
    app_package: str | None = None
    macro_id: int | None = None
    #: 录制这个宏的设备。**必须匹配当前设备**才跑 ——
    #: 真机实测：手机上连着两台设备时，执行面会自动挑「第一台」，
    #: 宏在**另一台**上跑就会一路找不到元素。
    device_serial: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Macro:
        steps_raw = raw.get("steps")
        if steps_raw is None:
            steps_raw = []
        if not isinstance(steps_raw, list):
            raise ValueError("'steps' must be a list")
        return cls(
            name=str(raw.get("name") or "untitled"),
            steps=[MacroStep.from_dict(s) for s in steps_raw],
            when_to_use=str(raw.get("when_to_use") or ""),
            app_package=raw.get("app_package"),
            macro_id=raw.get("id"),
            device_serial=raw.get("device_serial"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "when_to_use": self.when_to_use,
            "app_package": self.app_package,
            "device_serial": self.device_serial,
            "steps": [s.to_dict() for s in self.steps],
        }


@dataclass
class MacroRunResult:
    """回放结果。**降级记录**在里面 —— 调用方要知道宏是不是还在有效期内。"""

    ok: bool
    steps_total: int
    steps_executed: int = 0
    steps_skipped: int = 0
    failed_index: int | None = None
    failed_tool: str | None = None
    reason: str = ""
    #: 每一次降级：{"step": 2, "tool": "tap", "wanted": "resource_id=...", "got": "..."}
    degradations: list[dict[str, Any]] = field(default_factory=list)
    #: 每一步的结果摘要（给模型看，要短）。
    steps: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        parts = [
            f"{self.steps_executed}/{self.steps_total} 步执行成功",
            f"跳过 {self.steps_skipped} 步",
        ]
        if self.degradations:
            parts.append(f"{len(self.degradations)} 处降级")
        if not self.ok:
            parts.append(f"第 {self.failed_index} 步（{self.failed_tool}）失败：{self.reason}")
        return "；".join(parts)
