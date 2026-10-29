from langchain_core.tools import BaseTool

from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.tools.mobile.back import back_wrapper
from minitap.mobile_use.tools.mobile.erase_one_char import erase_one_char_wrapper
from minitap.mobile_use.tools.mobile.focus_and_clear_text import focus_and_clear_text_wrapper
from minitap.mobile_use.tools.mobile.focus_and_input_text import focus_and_input_text_wrapper
from minitap.mobile_use.tools.mobile.launch_app import launch_app_wrapper
from minitap.mobile_use.tools.mobile.long_press_on import long_press_on_wrapper
# 🔴 **2026-10-23 临时停用宏（脚本）工具。**
#
# 三个宏工具（`find_macros` / `save_macro` / `run_macro`）不再注册给 Executor，
# 提示词里对应的工具表行也一并摘掉了（`agents/executor/prompts/system/40-tools.md`）。
#
# **只摘 LLM 面向的这一层**，宏引擎本体（`minitap/mobile_use/macros/`）和三个工具的
# 实现（`tools/mobile/macros.py`）都**原封不动** —— 恢复只要把下面三行注释解开。
#
# 备份 / 恢复说明：`backup/20261023-disable-macro-tools/README.md`
# from minitap.mobile_use.tools.mobile.macros import (
#     find_macros_wrapper,
#     run_macro_wrapper,
#     save_macro_wrapper,
# )
from minitap.mobile_use.tools.mobile.open_link import open_link_wrapper
from minitap.mobile_use.tools.mobile.press_key import press_key_wrapper
from minitap.mobile_use.tools.mobile.scan_visual_subjects import scan_visual_subjects_wrapper
from minitap.mobile_use.tools.mobile.stop_app import stop_app_wrapper
from minitap.mobile_use.tools.mobile.swipe import swipe_wrapper
from minitap.mobile_use.tools.mobile.tap import tap_wrapper
from minitap.mobile_use.tools.mobile.video_recording import (
    start_video_recording_wrapper,
    stop_video_recording_wrapper,
)
from minitap.mobile_use.tools.mobile.wait_for_delay import wait_for_delay_wrapper
from minitap.mobile_use.tools.mobile.wake_screen import wake_screen_wrapper
# 🔴 **2026-10-23 临时停用笔记（scratchpad）工具。**
#
# 三个笔记工具（`save_note` / `read_note` / `list_notes`）不再注册给 Executor，
# 提示词里对应的工具表行也一并摘掉了（`agents/executor/prompts/system/40-tools.md`）。
#
# **只摘 LLM 面向的这一层**：`tools/scratchpad.py` 的实现和 `graph/state.py` 里的
# `scratchpad` 字段都**原封不动** —— 恢复只要把下面几行注释解开。
#
# ⚠️ 连带后果：planner 不再被教「用笔记工具做跨 app 数据搬运」
# （`agents/planner/planner.md` 那段示例已删）⇒ 「把 A app 的内容搬到 B app」
# 这类任务 planner 规划不出来。
#
# 备份 / 恢复说明：`backup/20261023-disable-note-tools/README.md`
# from minitap.mobile_use.tools.scratchpad import (
#     list_notes_wrapper,
#     read_note_wrapper,
#     save_note_wrapper,
# )
from minitap.mobile_use.tools.tool_wrapper import CompositeToolWrapper, ToolWrapper

EXECUTOR_WRAPPERS_TOOLS = [
    back_wrapper,
    open_link_wrapper,
    tap_wrapper,
    long_press_on_wrapper,
    swipe_wrapper,
    focus_and_input_text_wrapper,
    erase_one_char_wrapper,
    launch_app_wrapper,
    stop_app_wrapper,
    focus_and_clear_text_wrapper,
    press_key_wrapper,
    wait_for_delay_wrapper,
    # 屏幕是灭的时候用它点亮 —— 息屏时 OCR 认不出字、层级树只剩 systemui 残留，agent 会盲操作
    wake_screen_wrapper,
    # 🔴 **2026-10-27 移除截图工具。**
    #
    # `take_screenshot`（模型自己喊一声要看图）已从 Executor 的工具集里**整体摘掉**：
    # 实现文件 `tools/mobile/take_screenshot.py` 一并删除，提示词里的工具表行、
    # 「图片感知」通道、`state.executor_screenshot` 字段全部清干净。
    #
    # 备份 / 恢复说明：`backup/20261027-remove-screenshot-tool/README.md`
    #
    # ⚠️ 2026-10-28 的 `scan_visual_subjects` **不是它的复活**：那个工具不接受
    # 「看什么」的描述，只返回当屏全部视觉主体的编号表；图仍然不进 Executor 的
    # prompt。它按 `ctx.perception.vision` 开关注册，见 `executor_wrappers()`。
    # Scratchpad tools for persistent memory
    # 🔴 2026-10-23 临时停用 —— 见本文件顶部的说明。恢复：解开下面三行注释。
    # save_note_wrapper,
    # read_note_wrapper,
    # list_notes_wrapper,
    # 宏（脚本）：查 / 存 / 跑。跑宏是**零次 LLM 调用**的一串已决定的动作。
    # 🔴 2026-10-23 临时停用 —— 见本文件顶部的说明。恢复：解开下面三行注释。
    # find_macros_wrapper,
    # save_macro_wrapper,
    # run_macro_wrapper,
]

VIDEO_RECORDING_WRAPPERS = [
    start_video_recording_wrapper,
    stop_video_recording_wrapper,
]

#: 按需视觉扫描。**不放进 `EXECUTOR_WRAPPERS_TOOLS`**：它受
#: `ctx.perception.vision` 开关控制，统一由 `executor_wrappers()` 拼装。
VISION_WRAPPERS = [scan_visual_subjects_wrapper]


def executor_wrappers(ctx: MobileUseContext) -> list[ToolWrapper]:
    """Executor 这次能用的工具集合。

    Executor 节点和它的 prompt 必须用**同一份**：工具表里写着、实际却调不到
    （或反过来）会让模型反复报一个不存在的工具。
    """
    wrappers = list(EXECUTOR_WRAPPERS_TOOLS)
    if ctx.video_recording_enabled:
        wrappers.extend(VIDEO_RECORDING_WRAPPERS)
    if ctx.perception.vision:
        wrappers.extend(VISION_WRAPPERS)
    return wrappers


def get_tools_from_wrappers(
    ctx: "MobileUseContext",
    wrappers: list[ToolWrapper],
) -> list[BaseTool]:
    tools: list[BaseTool] = []
    for wrapper in wrappers:
        if isinstance(wrapper, CompositeToolWrapper):
            tools.extend(wrapper.composite_tools_fn_getter(ctx))
            continue

        tools.append(wrapper.tool_fn_getter(ctx))
    return tools


def format_tools_list(ctx: MobileUseContext, wrappers: list[ToolWrapper]) -> str:
    """给 system prompt 用的工具名清单，**按名称排序**。

    为什么排序
    ----------
    这串名字会进 system prompt（cortex.py / planner.py），而 system prompt 是
    前缀缓存的最前段。实测（2026-10-20，MiniMax M3.1-Flash）：
    工具清单在 cortex system prompt 的第 3950/6861 字符处（约 58% 深度），
    把顺序换掉，缓存从 10,477 token 掉到 1,169 —— 破在 system 内部 58% 的位置，
    后面全部重算。

    现在的顺序来自 `get_tools_from_wrappers`，而它是遍历模块级固定 list
    （`EXECUTOR_WRAPPERS_TOOLS`），**当前是稳定的**。但稳定性是「碰巧」的：
    哪天有人把注册改成 dict / set 遍历，或调整 wrapper 顺序，进程重启后
    system prompt 就会换序，整个缓存静默归零 —— 而这类改动看起来完全无害。

    按名称排序把「碰巧稳定」变成「按定义稳定」，一行成本换一个单点消除。

    注意：这里只排**名字清单**。真正的 tools schema 数组（`get_tools_from_wrappers`
    的返回值）不在这里动 —— MiniMax 的缓存键不含 tools（实测去掉 tools 数组
    缓存量不变），改它的顺序没有缓存收益，却会改变模型选工具时的平局顺序。
    """
    names = [tool.name for tool in get_tools_from_wrappers(ctx, wrappers)]
    return ", ".join(sorted(names))
