"""渠道目录导出：manager 侧那份 JSON 不能和 mobile-use 的目录对不上。

为什么要有这个测试
------------------
manager 的「添加渠道」表单读的是**导出**的那份 JSON
（``manager-server/src/main/resources/llm-channels.json``），而目录的真相在
mobile-use 这边。两边对不上的症状是：

* 目录里新增了一条渠道，表单里没有 → 选不到新渠道
* 目录里改了默认地址，表单里还是旧的 → 「我明明选了新地址，它还是打到老地方」
* 目录里删了一条渠道，表单里还能选 → 选了之后设备上起不来（provider 不合法）

⇒ 这条测试把两份锁在一起。改了目录就跑一次导出脚本
（``python scripts/export_channel_catalog.py``）。

manager 仓库不在旁边时**跳过**而不是失败 —— 两个仓库是各自独立 clone 的，
单跑 mobile-use 的测试不该因为邻居不在而红。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from minitap.mobile_use.services.channels import catalog_payload

ROOT = Path(__file__).resolve().parent.parent.parent

#: manager 仓库的位置。默认按「同级目录」找（这台机器上两个仓库是并排的）。
CANDIDATES = (
    ROOT.parent
    / "mobile-manager"
    / "manager-server"
    / "src"
    / "main"
    / "resources"
    / "llm-channels.json",
    Path("E:/MyCoding/mobile-manager/manager-server/src/main/resources/llm-channels.json"),
)


def _find_export() -> Path | None:
    for path in CANDIDATES:
        if path.exists():
            return path
    return None


EXPORT = _find_export()


@pytest.mark.skipif(EXPORT is None, reason="找不到 manager 仓库里的导出文件")
def test_the_exported_catalog_matches_the_source_of_truth():
    payload = json.loads(EXPORT.read_text(encoding="utf-8"))
    assert payload["channels"] == catalog_payload()


@pytest.mark.skipif(EXPORT is None, reason="找不到 manager 仓库里的导出文件")
def test_the_export_says_where_it_came_from():
    payload = json.loads(EXPORT.read_text(encoding="utf-8"))
    assert payload["generatedFrom"] == "minitap/mobile_use/services/channels.py"
    assert payload["version"] == 1
