"""修模板库第二轮：把被 LLM 命名撞车污染的「篝火」模板修回来，并补上「木桩」。

事故经过（2026-10-24）：
  MiniMax 视觉把「木桩」（棕色年轮桩）叫成了「篝火」——因为篝火图案下面
  就是棕色木柴。probe 脚本存模板时同名直接覆盖，于是「篝火」模板被换成了
  木桩的图。后果是：槽位里 3 张牌被认成 3 张篝火，而实际只有 1 张是篝火，
  消除永远不触发 —— 表现为「游戏规则不对」，真因是识别错了。

修法：
  1. 篝火：从 game_010.png 里已知位置（网格 (396,1561)）裁回真篝火。
  2. 木桩：用颜色规则认（低饱和棕占比高、无高饱和红焰），命名「木桩」。
  3. match_tile 改成「图像相关度 ∧ 颜色直方图」双判据（见 game_vision）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import game_vision as gv  # noqa: E402

GAME_010 = Path(r"E:\MyCoding\mobile-use\probe_out\game_010.png")
RUN3_STEP29 = Path(r"E:\MyCoding\mobile-use\probe_out\autoplay_run3\step_029.png")


def crop_like(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """和 game_vision.tile_crop 一样的裁剪规则（去边框 2px、去底部 14%）。"""
    x0, y0 = x + 2, y + 2
    x1, y1 = x + w - 2, y + int(h * 0.86)
    patch = img[y0:y1, x0:x1]
    return cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA)


def is_stump(crop: np.ndarray) -> bool:
    """木桩：低饱和棕为主，没有篝火那种高饱和红焰。"""
    b, g, r = crop[:, :, 0].astype(int), crop[:, :, 1].astype(int), crop[:, :, 2].astype(int)
    fire_red = int(((r > 170) & (g < 110) & (b < 110)).sum())
    brown = int(((r > 90) & (r < 185) & (g > 50) & (g < 140) & (b > 25) & (b < 100)).sum())
    return brown > 1200 and fire_red < 250


def main() -> int:
    templates = gv.load_templates()

    # 1) 篝火：从 game_010 的 (396,1561) 裁回（fix_templates 自检里它是篝火 1.000）
    im10 = cv2.imread(str(GAME_010))
    templates["篝火"] = crop_like(im10, 396, 1561, 135, 138)
    print("篝火 <- game_010 (396,1561)")

    # 2) 木桩：从 step_029 的槽位里按颜色规则挑出来
    im29 = cv2.imread(str(RUN3_STEP29))
    found = 0
    for t in gv.detect_tiles(im29):
        c = gv.tile_crop(im29, t)
        if is_stump(c):
            templates["木桩"] = c
            found += 1
            print(f"木桩 <- tile at ({t['x']},{t['y']})")
    if found == 0:
        print("!! 没找到木桩样本")

    gv.save_templates(templates)
    print("template library:", list(templates))

    # 3) 自检：用 step_029 复算每张牌，确认木桩不再被当成篝火
    print("\n-- 自检 step_029 --")
    for t in gv.detect_tiles(im29):
        c = gv.tile_crop(im29, t)
        name, score = gv.match_tile(c, templates)
        ncc = (
            float(cv2.matchTemplate(c, templates[name], cv2.TM_CCOEFF_NORMED)[0, 0])
            if name
            else -1
        )
        hc = (
            float(
                cv2.compareHist(
                    gv.color_hist(templates[name]), gv.color_hist(c), cv2.HISTCMP_CORREL
                )
            )
            if name
            else -1
        )
        print(
            f"  ({t['x']:>4},{t['y']:>4}) -> {name} "
            f"score={score:.3f} (ncc={ncc:.3f} hist={hc:.3f})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
