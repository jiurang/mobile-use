import asyncio
import re
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from adbutils import AdbClient, AdbDevice

from minitap.mobile_use.clients.ui_automator_client import UIAutomatorClient
from minitap.mobile_use.controllers.device_controller import (
    MobileDeviceController,
    ScreenDataResponse,
)
from minitap.mobile_use.controllers.types import Bounds, CoordinatesSelectorRequest, TapOutput
from minitap.mobile_use.utils.image_compress import compress_screenshot_base64
from minitap.mobile_use.utils.launch_command import launch_failure_reason
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.video import (
    ANDROID_MAX_RECORDING_DURATION_SECONDS,
    DEFAULT_MAX_DURATION_SECONDS,
    VIDEO_READY_DELAY_SECONDS,
    RecordingSession,
    VideoRecordingResult,
    cleanup_video_segments,
    concatenate_videos,
    get_active_session,
    has_active_session,
    remove_active_session,
    set_active_session,
)

logger = get_logger(__name__)


class AndroidDeviceController(MobileDeviceController):
    def __init__(
        self,
        device_id: str,
        adb_client: AdbClient,
        ui_adb_client: UIAutomatorClient,
        device_width: int,
        device_height: int,
    ):
        self.device_id = device_id
        self.adb_client = adb_client
        self.ui_adb_client = ui_adb_client
        self.device_width = device_width
        self.device_height = device_height
        self._device: AdbDevice | None = None

    @property
    def device(self) -> AdbDevice:
        if self._device is None:
            self._device = self.adb_client.device(serial=self.device_id)
        return self._device

    async def tap(
        self,
        coords: CoordinatesSelectorRequest,
        long_press: bool = False,
        long_press_duration: int = 1000,
    ) -> TapOutput:
        try:
            if long_press:
                cmd = (
                    f"input swipe {coords.x} {coords.y} {coords.x} {coords.y} {long_press_duration}"
                )
            else:
                cmd = f"input tap {coords.x} {coords.y}"

            self.device.shell(cmd)
            return TapOutput(error=None)
        except Exception as e:
            return TapOutput(error=f"ADB tap failed: {str(e)}")

    async def swipe(
        self,
        start: CoordinatesSelectorRequest,
        end: CoordinatesSelectorRequest,
        duration: int = 400,
    ) -> str | None:
        try:
            cmd = f"input touchscreen swipe {start.x} {start.y} {end.x} {end.y} {duration}"
            self.device.shell(cmd)
            return None
        except Exception as e:
            return f"ADB swipe failed: {str(e)}"

    async def get_screen_data(
        self, on_screenshot: Callable[[str], None] | None = None
    ) -> ScreenDataResponse:
        """Get screen data using the UIAutomator2 client"""
        try:
            logger.info("Using UIAutomator2 for screen data retrieval")
            # to_thread: UIAutomatorClient is a sync client (adb subprocess +
            # HTTP to the on-device u2 agent), and this runs on the same event
            # loop as the TUI. Calling it inline freezes the UI for the whole
            # hierarchy dump + screenshot.
            #
            # on_screenshot 在这个工作线程里就地执行，所以里面可以直接跑
            # CPU 密集的 OCR —— 它本来就不该阻塞事件循环，而此时 u2 正在
            # 并发地 dump 层级树，OCR 白得。
            ui_data = await asyncio.to_thread(
                self.ui_adb_client.get_screen_data, on_screenshot
            )
            return ScreenDataResponse(
                base64=ui_data.base64,
                elements=ui_data.elements,
                width=ui_data.width,
                height=ui_data.height,
                platform="android",
            )
        except Exception as e:
            logger.error(f"Failed to get screen data: {e}")
            raise

    async def screenshot(self) -> str:
        try:
            return (await self.get_screen_data()).base64
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            raise

    async def input_text(self, text: str) -> bool:
        try:
            self.ui_adb_client.send_text(text)
            return True
        except Exception as e:
            logger.warning(f"UIAutomator2 send_text failed: {e}, falling back to ADB shell")
            return self._input_text_adb_fallback(text)

    def _input_text_adb_fallback(self, text: str) -> bool:
        """Fallback method using ADB shell input text command."""
        try:
            parts = text.split("%s")
            for i, part in enumerate(parts):
                to_write = ""
                for char in part:
                    if char == " ":
                        to_write += "%s"
                    elif char in ["&", "<", ">", "|", ";", "(", ")", "$", "`", "\\", '"', "'"]:
                        to_write += f"\\{char}"
                    else:
                        to_write += char

                if to_write:
                    self.device.shell(f"input text '{to_write}'")

                if i < len(parts) - 1:
                    self.device.shell("input keyevent 62")

            return True
        except Exception as e:
            logger.error(f"Failed to input text via ADB fallback: {e}")
            return False

    async def launch_app(self, package_or_bundle_id: str) -> bool:
        """启动 app。

        ⚠️ **不用 `device.app_start()`**（2026-10-24）

        adbutils 的 `app_start` 内部是 monkey，而 **monkey 失败时不抛异常**
        （adbutils 的 `shell()` 只返回输出字符串、不检查错误输出）。真机实测：
        包不存在时输出 `** No activities found to run, monkey aborted.`，
        但 `app_start()` 照常返回 ⇒ 调用方会把「**根本没启动**」当成「启动了」。
        ⇒ 这里自己发命令、看输出（`utils/launch_command.py`）。

        失败时**抛异常**（消息里带原因原文）：调用方 `launch_app_with_retries`
        会把原因回灌给模型 —— 只回一句「失败了」，模型没法决定下一步该干什么。
        """
        try:
            output = self.device.shell(
                [
                    "monkey",
                    "-p",
                    package_or_bundle_id,
                    "-c",
                    "android.intent.category.LAUNCHER",
                    "1",
                ]
            )
        except Exception as e:
            logger.error(f"Failed to launch app {package_or_bundle_id}: {e}")
            raise RuntimeError(f"launch failed for {package_or_bundle_id}: {e}") from e

        reason = launch_failure_reason(output)
        if reason:
            logger.error(f"Failed to launch app {package_or_bundle_id}: {reason}")
            raise RuntimeError(f"launch failed for {package_or_bundle_id}: {reason}")
        return True

    async def terminate_app(self, package_or_bundle_id: str | None) -> bool:
        try:
            if package_or_bundle_id is None:
                current_app = self._get_current_foreground_package()
                if current_app:
                    logger.info(f"Stopping currently running app: {current_app}")
                    self.device.app_stop(current_app)
                else:
                    logger.warning("No foreground app detected")
                    return False
            else:
                self.device.app_stop(package_or_bundle_id)
            return True
        except Exception as e:
            logger.error(f"Failed to terminate app {package_or_bundle_id}: {e}")
            return False

    async def open_url(self, url: str) -> bool:
        try:
            self.device.shell(f"am start -a android.intent.action.VIEW -d {url}")
            return True
        except Exception as e:
            logger.error(f"Failed to open URL {url}: {e}")
            return False

    async def press_back(self) -> bool:
        try:
            self.device.shell("input keyevent 4")
            return True
        except Exception as e:
            logger.error(f"Failed to press back: {e}")
            return False

    async def press_home(self) -> bool:
        try:
            self.device.shell("input keyevent 3")
            return True
        except Exception as e:
            logger.error(f"Failed to press home: {e}")
            return False

    async def get_screen_power_state(self) -> str:
        """读屏幕的电源状态。

        返回 ``"on"`` / ``"off"`` / ``"unknown"``。

        判据是 ``dumpsys power`` 的 ``mWakefulness``：屏幕亮着是 ``Awake``，
        息屏（自动超时或按了电源键）是 ``Asleep``。真机实测（Redmi Turbo 4 /
        Android 16）：

            亮屏  -> mWakefulness=Awake
            熄屏  -> mWakefulness=Asleep
            KEYCODE_WAKEUP 之后 -> mWakefulness=Awake

        **为什么需要它**：屏幕一黑，uiautomator 的层级树是**空的**、截图是全黑，
        agent 于是看不到任何东西，会在黑屏上反复尝试、原地打转（2026-10-21
        真机实测：30 分钟息屏超时让一个任务从 30 步涨到 51 步还没跑完）。
        有了这个状态，harness 就能告诉模型「屏幕是灭的，先唤醒」。
        """
        try:
            raw = await asyncio.to_thread(self.device.shell, "dumpsys power")
        except Exception as e:
            logger.warning(f"Failed to read screen power state: {e}")
            return "unknown"
        out = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        if "mWakefulness=Awake" in out:
            return "on"
        if "mWakefulness=Asleep" in out or "mWakefulness=Dozing" in out:
            return "off"
        return "unknown"

    async def wake_up(self) -> bool:
        """唤醒屏幕并尝试划掉锁屏。

        两步都是必要的：

        - ``KEYCODE_WAKEUP`` 把屏幕点亮（息屏时普通的 tap/swipe 是**没用的** ——
          事件发给了一个没显示的屏幕）。
        - ``wm dismiss-keyguard`` 划掉无密码锁屏。**有 PIN 的话划不掉**，
          这时只能靠人解锁 —— 返回 True 但屏幕可能还是锁着的，所以调用方
          还要再读一次状态确认。
        """
        try:
            await asyncio.to_thread(self.device.shell, "input keyevent 224")  # KEYCODE_WAKEUP
            await asyncio.to_thread(self.device.shell, "wm dismiss-keyguard")
            return True
        except Exception as e:
            logger.error(f"Failed to wake up the screen: {e}")
            return False

    async def press_enter(self) -> bool:
        try:
            self.device.shell("input keyevent 66")
            return True
        except Exception as e:
            logger.error(f"Failed to press enter: {e}")
            return False

    async def get_ui_hierarchy(self) -> list[dict]:
        try:
            device_data = await self.get_screen_data()
            return device_data.elements
        except Exception as e:
            logger.error(f"Failed to get UI hierarchy: {e}")
            return []

    async def get_hierarchy_only(self) -> list[dict] | None:
        """**只要层级树** —— 不截图、不跑 OCR。

        沉降稳定性确认用这条：判定「屏定下来没有」只需要树的内容。
        走 ``get_screen_data`` 会连带做一次截图（166ms）+ PNG 编码（79ms）
        + OCR（381ms），而那些在这一步完全用不上。

        实测一次树级 dump 约 150~250ms（Redmi Turbo 4 / Android 16）。

        ⚠️ ``dump_hierarchy`` 自带 waitForIdle —— 屏幕还在动的时候它会
        **自己等**（实测滚动场景里它等了 609ms）。所以这个方法返回的树
        一定是「静」的，但不保证「全」（懒加载的控件可能还没挂上）。
        """
        from minitap.mobile_use.clients.ui_automator_client import (
            _parse_hierarchy_xml_to_elements,
        )

        try:
            xml = await asyncio.to_thread(self.ui_adb_client.get_hierarchy)
            if not xml:
                return None
            return _parse_hierarchy_xml_to_elements(xml)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to dump hierarchy only: {e}")
            return None

    def find_element(
        self,
        ui_hierarchy: list[dict],
        resource_id: str | None = None,
        text: str | None = None,
        index: int = 0,
    ) -> tuple[dict | None, Bounds | None, str | None]:
        if not resource_id and not text:
            return None, None, "No resource_id or text provided"

        matches = []
        for element in ui_hierarchy:
            if resource_id and element.get("resource-id") == resource_id:
                matches.append(element)
            elif text and (element.get("text") == text or element.get("accessibilityText") == text):
                matches.append(element)

        if not matches:
            criteria = f"resource_id='{resource_id}'" if resource_id else f"text='{text}'"
            return None, None, f"No element found with {criteria}"

        if index >= len(matches):
            criteria = f"resource_id='{resource_id}'" if resource_id else f"text='{text}'"
            return (
                None,
                None,
                f"Index {index} out of range for {criteria} (found {len(matches)} matches)",
            )

        element = matches[index]
        bounds = self._extract_bounds(element)

        return element, bounds, None

    def _get_current_foreground_package(self) -> str | None:
        try:
            result = self.device.shell("dumpsys window | grep mCurrentFocus")

            # Convert to string if bytes
            if isinstance(result, bytes):
                result_str = result.decode("utf-8")
            elif isinstance(result, str):
                result_str = result
            else:
                return None

            if result_str and "=" in result_str:
                parts = result_str.split("/")
                if len(parts) > 0:
                    package = parts[0].split()[-1]
                    return package if package else None
            return None
        except Exception as e:
            logger.error(f"Failed to get current foreground package: {e}")
            return None

    def _extract_bounds(self, element: dict) -> Bounds | None:
        bounds_str = element.get("bounds")
        if not bounds_str or not isinstance(bounds_str, str):
            return None

        try:
            match = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str)
            if match:
                return Bounds(
                    x1=int(match.group(1)),
                    y1=int(match.group(2)),
                    x2=int(match.group(3)),
                    y2=int(match.group(4)),
                )
        except (ValueError, IndexError):
            return None

        return None

    async def erase_text(self, nb_chars: int | None = None) -> bool:
        try:
            chars_to_delete = nb_chars if nb_chars is not None else 50
            for _ in range(chars_to_delete):
                self.device.shell("input keyevent KEYCODE_DEL")
            return True
        except Exception as e:
            logger.error(f"Failed to erase text: {e}")
            return False

    async def cleanup(self) -> None:
        pass

    def get_compressed_b64_screenshot(self, image_base64: str, quality: int = 50) -> str:
        return compress_screenshot_base64(image_base64, quality=quality)

    async def _start_android_segment(
        self, session: RecordingSession
    ) -> asyncio.subprocess.Process | None:
        """Start a single Android recording segment."""
        segment_path = f"/sdcard/screen_recording_{session.android_segment_index}.mp4"
        cmd = f"screenrecord --time-limit {ANDROID_MAX_RECORDING_DURATION_SECONDS} {segment_path}"

        process = await asyncio.create_subprocess_shell(
            f'adb -s {self.device_id} shell "{cmd}"',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        session.android_device_path = segment_path
        session.process = process
        return process

    async def _android_auto_restart_loop(
        self,
        session: RecordingSession,
        max_duration_seconds: int,
    ) -> None:
        """Background task that auto-restarts Android recording when segment ends."""
        total_elapsed = 0

        while total_elapsed < max_duration_seconds:
            process = session.process
            if process is None:
                break

            try:
                await asyncio.wait_for(
                    process.wait(),
                    timeout=ANDROID_MAX_RECORDING_DURATION_SECONDS + 5,
                )
            except TimeoutError:
                pass
            except asyncio.CancelledError:
                return

            if not has_active_session(self.device_id):
                return

            total_elapsed = time.time() - session.start_time
            if total_elapsed >= max_duration_seconds:
                break

            await asyncio.sleep(VIDEO_READY_DELAY_SECONDS)

            temp_dir = tempfile.mkdtemp(prefix="mobile_use_video_segment_")
            local_segment = Path(temp_dir) / f"segment_{session.android_segment_index}.mp4"

            try:
                self.device.sync.pull(session.android_device_path, str(local_segment))
                self.device.shell(f"rm -f {session.android_device_path}")
                session.android_video_segments.append(local_segment)
                logger.info(
                    f"Saved Android segment {session.android_segment_index} to {local_segment}"
                )
            except Exception as e:
                error_msg = f"Failed to pull segment {session.android_segment_index}: {e}"
                logger.warning(error_msg)
                session.errors.append(error_msg)

            session.android_segment_index += 1

            try:
                await self._start_android_segment(session)
                logger.info(
                    f"Auto-restarted Android recording (segment {session.android_segment_index})"
                )
            except Exception as e:
                error_msg = f"Failed to restart Android recording: {e}"
                logger.error(error_msg)
                session.errors.append(error_msg)
                break

    async def start_video_recording(
        self,
        max_duration_seconds: int = DEFAULT_MAX_DURATION_SECONDS,
    ) -> VideoRecordingResult:
        """Start screen recording on Android device using adb shell screenrecord."""
        if has_active_session(self.device_id):
            return VideoRecordingResult(
                success=False,
                message=f"Recording already in progress for device {self.device_id}",
            )

        try:
            session = RecordingSession(
                device_id=self.device_id,
                start_time=time.time(),
                android_video_segments=[],
                android_segment_index=0,
            )

            set_active_session(self.device_id, session)
            await self._start_android_segment(session)

            restart_task = asyncio.create_task(
                self._android_auto_restart_loop(session, max_duration_seconds)
            )
            session.android_restart_task = restart_task

            logger.info(f"Started Android screen recording on {self.device_id}")
            return VideoRecordingResult(
                success=True,
                message=(
                    f"Recording started (max {max_duration_seconds}s, "
                    f"auto-restarts every {ANDROID_MAX_RECORDING_DURATION_SECONDS}s)."
                ),
            )

        except Exception as e:
            logger.error(f"Failed to start Android recording: {e}")
            remove_active_session(self.device_id)
            return VideoRecordingResult(
                success=False,
                message=f"Failed to start recording: {e}",
            )

    async def stop_video_recording(self) -> VideoRecordingResult:
        """Stop Android recording, pull all segments, and concatenate them."""
        session = get_active_session(self.device_id)

        if not session:
            return VideoRecordingResult(
                success=False,
                message=f"No active recording for device {self.device_id}",
            )

        try:
            if session.android_restart_task:
                session.android_restart_task.cancel()
                try:
                    await session.android_restart_task
                except asyncio.CancelledError:
                    pass

            process = session.process
            if process is not None:
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except TimeoutError:
                    process.kill()
                    await process.wait()

            self.device.shell("pkill -2 screenrecord || true")
            await asyncio.sleep(VIDEO_READY_DELAY_SECONDS)

            temp_dir = tempfile.mkdtemp(prefix="mobile_use_video_")
            final_segment = Path(temp_dir) / f"segment_{session.android_segment_index}.mp4"

            try:
                self.device.sync.pull(session.android_device_path, str(final_segment))
                self.device.shell(f"rm -f {session.android_device_path}")
                session.android_video_segments.append(final_segment)
            except Exception as e:
                logger.warning(f"Failed to pull final segment: {e}")

            for i in range(session.android_segment_index + 1):
                self.device.shell(f"rm -f /sdcard/screen_recording_{i}.mp4")

            all_segments = session.android_video_segments

            if not all_segments:
                remove_active_session(self.device_id)
                return VideoRecordingResult(
                    success=False,
                    message="No video segments were captured",
                )

            output_dir = tempfile.mkdtemp(prefix="mobile_use_video_final_")
            output_path = Path(output_dir) / "recording.mp4"

            if len(all_segments) == 1:
                all_segments[0].rename(output_path)
            else:
                success = await concatenate_videos(all_segments, output_path)
                if not success:
                    output_path = all_segments[-1]
                    logger.warning("Concatenation failed, using last segment only")

            cleanup_video_segments(all_segments, keep_path=output_path)

            errors = session.errors.copy()
            remove_active_session(self.device_id)

            duration = time.time() - session.start_time
            segment_count = len(all_segments)
            logger.info(
                f"Stopped Android recording after {duration:.1f}s, "
                f"{segment_count} segment(s), saved to {output_path}"
            )

            message = f"Recording stopped after {duration:.1f}s ({segment_count} segments)"
            if errors:
                message += f". Warnings during recording: {'; '.join(errors)}"

            return VideoRecordingResult(
                success=True,
                message=message,
                video_path=output_path,
            )

        except Exception as e:
            logger.error(f"Failed to stop Android recording: {e}")
            remove_active_session(self.device_id)
            return VideoRecordingResult(
                success=False,
                message=f"Failed to stop recording: {e}",
            )
