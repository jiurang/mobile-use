"""包名 → 应用中文名（只用于**给人/给模型看的文案**）。

优先从设备上读真名
------------------
`pm list packages` / `dumpsys package` 只给包名 —— `android:label` 是 APK
资源里的字符串。所以这里绕一步，**不拉整个 APK**：

1. `pm path <pkg>` 拿 base APK 路径；
2. 在设备上 `unzip -p` 只解两条（``AndroidManifest.xml`` / ``resources.arsc``，
   实测红果短剧 424KB + 4.5MB），base64 传回来（番茄小说 51MB、红果短剧
   150MB，整包拉不起）；
3. 主机上拼一个最小 APK 交给 ``aapt2 dump badging``，读
   ``application-label-zh*``，没有中文标签就用 ``application-label``。

⚠️ 第 3 步依赖主机的 Android SDK（aapt2）。**找不到 aapt2**（比如 mobile-use
直接跑在手机上）就退回内置的 ``APP_NAMES_ZH``，再不行显示包名 —— 不猜。

⚠️ state 里的 ``focused_app_info`` 仍然存**包名**：app lock 比对、宏的
``app_package`` 前置条件都要精确的包名，中文名只活在渲染层。
"""

from __future__ import annotations

import asyncio
import base64
import glob
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from minitap.mobile_use.utils.logger import get_logger

if TYPE_CHECKING:
    from minitap.mobile_use.context import MobileUseContext

logger = get_logger(__name__)


#: 离线兜底：拿不到 APK 资源（没有 aapt2 / 云端设备 / iOS）时用。
#: 顺序无所谓，按「字节/腾讯/阿里/其它 + 小米系统应用」分组便于核对。
APP_NAMES_ZH: dict[str, str] = {
    # ---- 字节系 ----
    "com.dragon.read": "番茄免费小说",
    "com.phoenix.read": "红果免费短剧",
    "com.xs.fm": "番茄畅听",
    "com.ss.android.ugc.aweme": "抖音",
    "com.ss.android.ugc.aweme.lite": "抖音极速版",
    "com.ss.android.article.news": "今日头条",
    "com.ss.android.article.lite": "今日头条极速版",
    "com.smile.gifmaker": "快手",
    "com.kuaishou.nebula": "快手极速版",
    # ---- 腾讯系 ----
    "com.tencent.mm": "微信",
    "com.tencent.mobileqq": "QQ",
    "com.tencent.qqlive": "腾讯视频",
    "com.tencent.qqmusic": "QQ音乐",
    "com.tencent.weread": "微信读书",
    # ---- 阿里系 ----
    "com.eg.android.AlipayGphone": "支付宝",
    "com.taobao.taobao": "淘宝",
    "com.taobao.idlefish": "闲鱼",
    "com.tmall.wireless": "天猫",
    "com.xunmeng.pinduoduo": "拼多多",
    "com.alibaba.android.rimet": "钉钉",
    "com.youku.phone": "优酷",
    # ---- 其它常用 ----
    "com.sina.weibo": "微博",
    "com.xingin.xhs": "小红书",
    "tv.danmaku.bili": "哔哩哔哩",
    "com.zhihu.android": "知乎",
    "com.qiyi.video": "爱奇艺",
    "com.netease.cloudmusic": "网易云音乐",
    "com.sankuai.meituan": "美团",
    "me.ele": "饿了么",
    "com.autonavi.minimap": "高德地图",
    "com.baidu.BaiduMap": "百度地图",
    "com.baidu.searchbox": "百度",
    "ctrip.android.view": "携程旅行",
    "com.jingdong.app.mall": "京东",
    "com.duokan.reader": "多看阅读",
    "cn.wps.moffice_eng": "WPS Office",
    "com.iflytek.inputmethod.miui": "讯飞输入法",
    # ---- 小米系统应用 ----
    "com.miui.home": "桌面",
    "com.android.settings": "设置",
    "com.android.systemui": "系统界面",
    "com.miui.securitycenter": "手机管家",
    "com.miui.gallery": "相册",
    "com.miui.calculator": "计算器",
    "com.miui.notes": "笔记",
    "com.miui.compass": "指南针",
    "com.miui.weather2": "天气",
    "com.android.deskclock": "时钟",
    "com.android.calendar": "日历",
    "com.miui.player": "音乐",
    "com.miui.video": "小米视频",
    "com.miui.fm": "收音机",
    "com.miui.themestore": "主题商店",
    "com.xiaomi.shop": "小米商城",
    "com.xiaomi.smarthome": "米家",
    "com.xiaomi.gamecenter": "游戏中心",
    "com.mi.health": "运动健康",
    "com.miui.huanji": "小米换机",
    "com.android.email": "邮件",
    "com.android.soundrecorder": "录音机",
    "com.miui.screenrecorder": "屏幕录制",
    "com.xiaomi.scanner": "扫一扫",
    "com.android.providers.downloads.ui": "下载管理",
    "com.miui.mediaeditor": "相册编辑",
    "com.xiaomi.youpin": "小米有品",
    "com.tongcheng.android": "同程旅行",
}


def app_display_name(package: str | None) -> str | None:
    """**只查内置表**：包名 → 中文名；没收录就原样返回，绝不猜。"""
    if not package:
        return package
    return APP_NAMES_ZH.get(package, package)


async def resolve_app_display_name(ctx: MobileUseContext, package: str | None) -> str | None:
    """包名 → 展示名：设备上的真名 > 内置表 > 包名。

    结果按包名缓存在进程内 —— 切换 app 时才走一次 adb，同一步里重复问是纯查表。
    """
    if not package:
        return package
    label = await _device_app_label(ctx, package)
    return label or app_display_name(package)


#: 包名 → 设备上读到的 label（``None`` = 读不到，别再重试）。
_LABEL_CACHE: dict[str, str | None] = {}

#: aapt2 的查找结果缓存；``_aapt2_searched`` 区分「还没找」和「找不到」。
_AAPT2: str | None = None
_AAPT2_SEARCHED = False


async def _device_app_label(ctx: MobileUseContext, package: str) -> str | None:
    if package in _LABEL_CACHE:
        return _LABEL_CACHE[package]
    label: str | None = None
    try:
        label = await asyncio.to_thread(_read_device_label, ctx, package)
    except Exception as e:  # noqa: BLE001 - 读不到只是少一个名字，不能影响任务
        logger.warning(f"[app-name] Failed to read label for '{package}': {e}")
    _LABEL_CACHE[package] = label
    if label:
        logger.info(f"[app-name] {package} -> {label}")
    return label


def _read_device_label(ctx: MobileUseContext, package: str) -> str | None:
    """同步实现（在 thread 里跑）：摘 APK 的两条 + 本地 aapt2 解析。"""
    aapt2 = _find_aapt2()
    if not aapt2:
        return None
    # 云端设备没有可直连的 adb；iOS 没有 APK —— 都退回内置表。
    if getattr(ctx, "cloud_android_controller", None) is not None:
        return None
    if ctx.device.mobile_platform.value != "android":
        return None

    device = ctx.get_adb_client().device(serial=ctx.device.device_id)
    apk_path = _first_apk_path(device.shell(["pm", "path", package]) or "")
    if not apk_path:
        return None

    manifest = _extract_entry(device, apk_path, "AndroidManifest.xml")
    resources = _extract_entry(device, apk_path, "resources.arsc")
    if not manifest or not resources:
        return None

    with tempfile.TemporaryDirectory(prefix="mobile-use-label-") as tmp:
        mini_apk = Path(tmp) / "label.apk"
        with zipfile.ZipFile(mini_apk, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("AndroidManifest.xml", manifest)
            archive.writestr("resources.arsc", resources)
        return _dump_label(aapt2, mini_apk)


def _first_apk_path(pm_path_output: str) -> str | None:
    """``pm path`` 的输出里取 base APK（split 的 label 也在 base 里）。"""
    for line in pm_path_output.splitlines():
        line = line.strip()
        if line.startswith("package:"):
            return line[len("package:") :]
    return None


def _extract_entry(device: Any, apk_path: str, entry: str) -> bytes | None:
    """在设备上 ``unzip -p`` 只解一个条目，base64 传回来（不拉整个 APK）。"""
    encoded = device.shell(f'unzip -p "{apk_path}" {entry} | base64')
    if not encoded or not encoded.strip():
        return None
    return base64.b64decode(encoded)


def _dump_label(aapt2: str, apk: Path) -> str | None:
    result = subprocess.run(
        [aapt2, "dump", "badging", str(apk)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    if result.returncode != 0:
        return None
    return parse_badging_label(result.stdout)


#: ``aapt2 dump badging`` 的标签行，例如 ``application-label-zh:'红果免费短剧'``。
_LABEL_LINE_RE = re.compile(r"^application-label(?:-([\w-]+))?:\s*'(.*)'\s*$", re.MULTILINE)


def parse_badging_label(output: str) -> str | None:
    """从 badging 输出里挑标签：中文优先（zh-CN/zh…），没有就用默认标签。"""
    labels: dict[str, str] = {}
    for locale, value in _LABEL_LINE_RE.findall(output):
        if value:
            labels[locale] = value
    if not labels:
        return None
    chinese = [locale for locale in labels if locale.lower().startswith("zh")]
    if chinese:
        # 更具体的 locale（zh-CN、zh-Hans）优先
        return labels[max(chinese, key=len)]
    return labels.get("")


def _find_aapt2() -> str | None:
    """找主机的 aapt2：PATH > ANDROID_HOME/SDK > 各平台默认 SDK 目录。"""
    global _AAPT2, _AAPT2_SEARCHED
    if _AAPT2_SEARCHED:
        return _AAPT2
    _AAPT2_SEARCHED = True

    exe = "aapt2.exe" if os.name == "nt" else "aapt2"
    found = shutil.which("aapt2")
    if found:
        _AAPT2 = found
        return _AAPT2

    roots = [
        os.environ.get("ANDROID_HOME"),
        os.environ.get("ANDROID_SDK_ROOT"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Android", "Sdk"),
        os.path.expanduser("~/Library/Android/sdk"),
        os.path.expanduser("~/Android/Sdk"),
        "/usr/lib/android-sdk",
    ]
    candidates: list[str] = []
    for root in roots:
        if root:
            candidates.extend(glob.glob(os.path.join(root, "build-tools", "*", exe)))
    _AAPT2 = max(candidates, key=_build_tools_version) if candidates else None
    if _AAPT2:
        logger.info(f"[app-name] Using aapt2 at {_AAPT2}")
    else:
        logger.info("[app-name] aapt2 not found; app names fall back to the built-in table")
    return _AAPT2


def _build_tools_version(path: str) -> tuple[int, ...]:
    match = re.search(r"build-tools[\\/](\d+(?:\.\d+)*)", path)
    return tuple(int(part) for part in match.group(1).split(".")) if match else (0,)
