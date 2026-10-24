"""修模板库：从已知帧里裁回被污染的样本，并补上新牌型。

问题复盘（2026-10-24）：
  MiniMax 把「蓝色瓶子」认成了「刷子」（两个都是蓝色主体），而
  probe_game_tiles.py 存模板时是 `templates[name] = crop` —— 同名直接覆盖，
  于是刷子的模板被换成了瓶子的图，库被污染。

修法：
  - 「刷子」样本从 game_005.png 裁回（那帧里 9 号牌确定是蓝柄锤）。
  - 「蓝瓶」用颜色规则从当前帧认：瓶身有大片近白像素，刷子有棕色刷头。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import game_vision as gv  # noqa: E402

GAME_005 = Path(r"E:\MyCoding\mobile-use\probe_out\game_005.png")
CURRENT = Path(r"E:\MyCoding\mobile-use\probe_out\game_010.png")


def is_bottle(crop: np.ndarray) -> bool:
    """蓝色瓶子：瓶身近白像素多；蓝柄锤：棕色刷头多。"""
    b, g, r = crop[:, :, 0].astype(int), crop[:, :, 1].astype(int), crop[:, :, 2].astype(int)
    white = int(((r > 215) & (g > 215) & (b > 205)).sum())
    brown = int(((r > 120) & (g > 50) & (g < 150) & (b < 100)).sum())
    return white > brown * 2


def main() -> int:
    templates = gv.load_templates()

    # 1) 刷子：从 game_005 里已知的蓝柄锤位置裁回
    #    裁剪规则必须和 game_vision.tile_crop 一致（去边框 2px、去底部 14%）
    im5 = cv2.imread(str(GAME_005))
    brush = im5[904:1020, 691:822]
    templates["刷子"] = cv2.resize(brush, (64, 64), interpolation=cv2.INTER_AREA)
    print("刷子 <- game_005 (691,904)-(822,1020)")

    # 2) 蓝瓶：从当前帧找未识别的牌，用颜色规则认
    cur = gv.screencap()
    cv2.imwrite(r"E:\MyCoding\mobile-use\probe_out\fix_templates_frame.png", cur)
    tiles = gv.detect_tiles(cur)
    grid, slot = gv.split_grid_slot(cur, tiles)
    found = 0
    for t in grid + slot:
        crop = gv.tile_crop(cur, t)
        name, score = gv.match_tile(crop, templates)
        if name and score >= 0.80:
            continue
        if is_bottle(crop):
            templates["蓝瓶"] = crop
            found += 1
            print(f"蓝瓶 <- tile at ({t['x']},{t['y']}) score_vs_other={score:.2f}")
    if found == 0:
        print("!! 没找到蓝瓶样本")

    gv.save_templates(templates)
    print("template library:", list(templates))

    # 3) 自检：每张牌必须匹配到自己这一类，且不会匹配到别的
    cv2.imwrite(r"E:\MyCoding\mobile-use\probe_out\fix_templates_frame.png", cur)
    tiles = gv.detect_tiles(cur)
    for t in tiles:
        crop = gv.tile_crop(cur, t)
        name, score = gv.match_tile(crop, templates)
        print(f"  tile({t['x']},{t['y']}) -> {name} ({score:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
