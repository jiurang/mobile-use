from abc import abstractmethod
from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel

from minitap.mobile_use.controllers.types import Bounds, CoordinatesSelectorRequest, TapOutput
from minitap.mobile_use.utils.video import VideoRecordingResult


class ScreenDataResponse(BaseModel):
    base64: str
    elements: list
    width: int
    height: int
    platform: str


class MobileDeviceController(Protocol):
    @abstractmethod
    async def tap(
        self,
        coords: CoordinatesSelectorRequest,
        long_press: bool = False,
        long_press_duration: int = 1000,
    ) -> TapOutput:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    async def swipe(
        self,
        start: CoordinatesSelectorRequest,
        end: CoordinatesSelectorRequest,
        duration: int = 400,
    ) -> str | None:
        """
        Swipe from start to end coordinates.
        Returns error message on failure, None on success.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    async def screenshot(self) -> str:
        """Take a screenshot and return raw image data."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    async def input_text(self, text: str) -> bool:
        """
        Input text at the currently focused field.
        Returns True on success, False on failure.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    async def launch_app(self, package_or_bundle_id: str) -> bool:
        """
        Launch an application by package name (Android) or bundle ID (iOS).
        Returns True on success, False on failure.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    async def terminate_app(self, package_or_bundle_id: str | None) -> bool:
        """
        Terminate an application.
        Returns True on success, False on failure.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    async def open_url(self, url: str) -> bool:
        """
        Open a URL.
        Returns True on success, False on failure.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    async def press_back(self) -> bool:
        """
        Press the back button (Android) or equivalent gesture (iOS).
        Returns True on success, False on failure.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    async def press_home(self) -> bool:
        """
        Press the home button.
        Returns True on success, False on failure.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    async def press_enter(self) -> bool:
        """
        Press the enter/return key.
        Returns True on success, False on failure.
        """
        raise NotImplementedError("Subclasses must implement this method")

    async def get_screen_power_state(self) -> str:
        """
        Read the screen power state.

        Returns:
            "on" / "off" / "unknown". A screen that is off yields an empty
            element tree and a black screenshot, so the agent would act blind
            unless it is told.

        默认实现返回 ``"unknown"`` —— 含义是「不知道」，**不是「亮着」**，
        所以 cortex 不会因此误判成息屏。实现不了这一项的控制器（iOS、云端
        旧实现、测试替身）都走这个默认值，不需要各自实现一遍。
        """
        return "unknown"

    async def wake_up(self) -> bool:
        """
        Wake the screen up and dismiss a keyless lock screen.

        Returns:
            True on success, False on failure. A PIN-protected screen cannot be
            dismissed programmatically, so callers should re-read the power
            state to confirm.

        默认实现返回 ``False`` —— 实现不了就如实说做不到。
        """
        return False

    @abstractmethod
    async def get_ui_hierarchy(self) -> list[dict]:
        """
        Get the UI element hierarchy.
        Returns a list of UI elements with their properties.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def find_element(
        self,
        ui_hierarchy: list[dict],
        resource_id: str | None = None,
        text: str | None = None,
        index: int = 0,
    ) -> tuple[dict | None, Bounds | None, str | None]:
        """
        Find a UI element in the hierarchy.

        Returns:
            Tuple of (element_dict, bounds, error_message)
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources (e.g., stop companion processes)."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    async def erase_text(self, nb_chars: int | None = None) -> bool:
        """
        Erase the last nb_chars characters.
        Returns True on success, False on failure.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    async def get_screen_data(
        self, on_screenshot: "Callable[[str], None] | None" = None
    ) -> "ScreenDataResponse":
        """
        Get screen data including screenshot (base64), UI hierarchy elements,
        screen dimensions, and platform.

        Args:
            on_screenshot: 可选回调，在截图拿到之后、层级树还在采集的窗口里
                同步调用一次。Android 实测 `dump_hierarchy` 要 ~1.2s，OCR 只要
                ~0.4s，把 OCR 塞进这个窗口等于白送。实现方不支持时忽略即可。

        Returns:
            ScreenDataResponse with base64 screenshot, elements, width, height, platform
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_compressed_b64_screenshot(self, image_base64: str, quality: int = 50) -> str:
        """
        Compress a base64 image.
        Returns the compressed base64 image.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    async def start_video_recording(
        self,
        max_duration_seconds: int = 900,
    ) -> VideoRecordingResult:
        """
        Start screen recording on the device.

        Args:
            max_duration_seconds: Maximum recording duration in seconds.

        Returns:
            VideoRecordingResult with success status and message.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    async def stop_video_recording(self) -> VideoRecordingResult:
        """
        Stop screen recording and retrieve the video file.

        Returns:
            VideoRecordingResult with success status, message, and video_path if successful.
        """
        raise NotImplementedError("Subclasses must implement this method")
