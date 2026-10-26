"""包名 → 应用名：优先读设备 APK 资源，读不到退回内置表（本文件测试的是后者）。"""

from __future__ import annotations

from minitap.mobile_use.utils.app_names import app_display_name, parse_badging_label


class TestAppDisplayName:
    def test_known_packages_get_chinese_names(self):
        assert app_display_name("com.dragon.read") == "番茄免费小说"
        assert app_display_name("com.phoenix.read") == "红果免费短剧"
        assert app_display_name("com.miui.home") == "桌面"

    def test_unknown_package_falls_back_to_package_name(self):
        """没收录就不猜 —— 显示包名比显示一个猜错的名字强。"""
        assert app_display_name("com.example.unknown") == "com.example.unknown"

    def test_none_stays_none(self):
        assert app_display_name(None) is None


class TestParseBadgingLabel:
    """`aapt2 dump badging` 的标签解析：中文优先，没有就用默认标签。"""

    def test_prefers_the_most_specific_chinese_label(self):
        output = (
            "package: name='com.example'\n"
            "application-label:'Tomato'\n"
            "application-label-zh:'番茄'\n"
            "application-label-zh-CN:'番茄免费小说'\n"
        )
        assert parse_badging_label(output) == "番茄免费小说"

    def test_falls_back_to_the_default_label(self):
        assert parse_badging_label("application-label:'Settings'\n") == "Settings"

    def test_no_label_returns_none(self):
        assert parse_badging_label("package: name='com.example'\n") is None
