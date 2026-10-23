"""把渠道目录导出成 manager 侧要的 JSON。

为什么要有这个导出
------------------
「渠道」这个概念在 mobile-use 里（它才是真正发请求的那一侧），但**添加渠道的
表单在 manager 侧**。两边需要的是同一份数据：

* 选哪个渠道（id + 显示名）
* 选哪个模型（推荐模型名）
* 默认地址是什么（不用手抄 URL）
* 要不要密钥、能不能改端点

以前这份清单在三个地方各抄一遍（mobile-use 的 ``LLMProvider``、manager-server 的
``KEYLESS_PROVIDERS``、manager-web 的 ``KNOWN_PROVIDERS``），对不上的症状是
「下拉框里有这个选项，设备上不认」。

现在：**mobile-use 是唯一真相**，导出一份 JSON 给 manager-server 读。
本脚本负责导出，以及 ``--check`` 模式（校验仓库里那份是不是最新的）。

跑法：
    .\\.venv\\Scripts\\python.exe scripts/export_channel_catalog.py            # 写文件
    .\\.venv\\Scripts\\python.exe scripts/export_channel_catalog.py --check    # 只校验
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "minitap"))

from minitap.mobile_use.services.channels import catalog_payload  # noqa: E402

#: 导出目标。默认是同级的 mobile-manager 仓库。
DEFAULT_TARGET = (
    ROOT.parent
    / "mobile-manager"
    / "manager-server"
    / "src"
    / "main"
    / "resources"
    / "llm-channels.json"
)

#: 载荷的版本号。格式变了才 +1，manager 侧按这个判断能不能读。
PAYLOAD_VERSION = 1


def build_payload() -> dict:
    return {
        "version": PAYLOAD_VERSION,
        # 指明这份数据的来源，免得有人以为它是手写的。
        "generatedFrom": "minitap/mobile_use/services/channels.py",
        "channels": catalog_payload(),
    }


def render(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument(
        "--check",
        action="store_true",
        help="只校验仓库里那份是不是最新的（内容不一致就退出码 1）",
    )
    args = parser.parse_args()

    payload = build_payload()
    text = render(payload)

    if args.check:
        if not args.target.exists():
            print(f"❌ 找不到 {args.target}")
            return 1
        current = args.target.read_text(encoding="utf-8")
        if current != text:
            print(
                f"❌ {args.target} 与渠道目录不一致。\n"
                f"   跑一下：.\\.venv\\Scripts\\python.exe scripts/export_channel_catalog.py"
            )
            return 1
        print(f"✅ {args.target} 与渠道目录一致（{len(payload['channels'])} 条渠道）")
        return 0

    args.target.parent.mkdir(parents=True, exist_ok=True)
    args.target.write_text(text, encoding="utf-8")
    print(f"✅ 已导出 {len(payload['channels'])} 条渠道 → {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
