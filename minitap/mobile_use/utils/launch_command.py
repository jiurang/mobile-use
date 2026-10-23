"""发启动命令，并**看它的输出**判断到底成没成。

为什么不能直接信 adbutils 的 `app_start`
----------------------------------------
adbutils 的 ``AdbDevice.app_start`` 内部是::

    monkey -p <pkg> -c android.intent.category.LAUNCHER 1

而 adbutils 的 ``shell()`` **只返回输出字符串、不检查错误输出** ⇒
monkey 失败时**不抛异常**，``app_start()`` 照常返回。

真机实测（Redmi Turbo 4 / Android 16，2026-10-24）::

    $ adb shell monkey -p com.definitely.not.here.xyz -c android.intent.category.LAUNCHER 1
    ** No activities found to run, monkey aborted.

    $ adb shell monkey -p com.android.settings -c android.intent.category.LAUNCHER 1
    Events injected: 1

⇒ 直接信 ``app_start()`` 的返回值，会把「**根本没启动**」报成「启动成功」。
⇒ 这里自己发命令、看输出。
"""

from __future__ import annotations

#: 启动命令输出里出现这些 ⇒ 启动没成。
#:
#: - ``No activities found`` / ``monkey aborted`` —— monkey 的失败措辞
#:   （包不存在，或包里没有可启动的 Activity）。
#: - ``Error:`` —— ``am start`` 的失败措辞
#:   （``Error: Activity class {...} does not exist.``）。
#: - ``Exception occurred`` —— monkey 自己的异常。
#:
#: ⚠️ **只认失败，不认成功**：``am start`` 成功时会打印
#: ``Warning: Activity not started, ... currently running top-most instance``
#: —— 那是**成功**（app 已经在前台），不能当失败。
LAUNCH_FAILURE_MARKERS = (
    "No activities found",
    "monkey aborted",
    "Error:",
    "Exception occurred",
)


def launch_failure_reason(output: str | bytes | None) -> str | None:
    """从启动命令的输出里找出失败原因；没失败返回 None。

    Args:
        output: 启动命令的原始输出（adbutils 的 `shell()` 返回 str，
            某些路径返回 bytes —— 都接）。

    Returns:
        失败时是**一行**原文（给模型看，别改写）；成功时 None。
    """
    if not output:
        return None
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")

    for marker in LAUNCH_FAILURE_MARKERS:
        index = output.find(marker)
        if index == -1:
            continue
        # 取包含这个标记的那一行 —— monkey 的错误在单独一行上，
        # 上面还跟着几行「bash arg: -p ...」的彩色回显。
        start = output.rfind("\n", 0, index) + 1
        end = output.find("\n", index)
        line = output[start : end if end != -1 else len(output)].strip()
        return line or marker
    return None
