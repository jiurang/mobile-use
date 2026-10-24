"""探针：抓一帧 -> 检测牌位 -> 拼画廊图 -> （可选）调 LLM 标注 -> 建/更新模板库。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
import game_vision as gv  # noqa: E402

OUT = Path(r"E:\MyCoding\mobile-use\probe_out")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="store_true", help="调 LLM 标注新牌型")
    ap.add_argument("--tag", default="probe")
    args = ap.parse_args()

    img = gv.screencap()
    cv2.imwrite(str(OUT / f"{args.tag}_screen.png"), img)
    tiles = gv.detect_tiles(img)
    grid, slot = gv.split_grid_slot(img, tiles)
    print(f"detected {len(tiles)} bright tiles: grid={len(grid)} slot={len(slot)}")
    for t in grid:
        print(
            f"  r{t.get('row')}-c{t.get('col')} x={t['x']} y={t['y']} "
            f"{t['w']}x{t['h']} center=({t['cx']},{t['cy']})"
        )

    crops = [gv.tile_crop(img, t) for t in grid]
    gallery, boxes = gv.build_gallery(crops)
    gpath = OUT / f"{args.tag}_gallery.png"
    cv2.imwrite(str(gpath), gallery)
    print(f"gallery -> {gpath} ({gallery.shape[1]}x{gallery.shape[0]})")

    templates = gv.load_templates()
    print(f"templates in library: {list(templates)}")

    if args.label:
        # 已有模板先匹配，匹配不上的（低相关度）才送去 LLM
        all_tiles = grid + slot
        unknown_idx: list[int] = []
        unknown_crops: list = []
        for i, t in enumerate(all_tiles):
            c = gv.tile_crop(img, t)
            name, score = gv.match_tile(c, templates)
            if name and score >= 0.93:
                t["kind"] = name
            else:
                unknown_idx.append(i)
                unknown_crops.append(c)
        known = sum(1 for t in all_tiles if t.get("kind"))
        print(f"known by template: {known}, unknown: {len(unknown_idx)}")
        if unknown_crops:
            ugallery, _ = gv.build_gallery(unknown_crops)
            upath = OUT / f"{args.tag}_unknown_gallery.png"
            cv2.imwrite(str(upath), ugallery)
            print(f"unknown gallery -> {upath}")
            labels = gv.llm_label_gallery(ugallery, len(unknown_crops))
            print("llm labels:", labels)
            labels = gv.unify_labels(unknown_crops, labels)
            # 存模板：同名但长得不像 -> LLM 命名撞车了（「蓝瓶」被叫成「刷子」），
            # 这时**不能覆盖**已有模板，否则把旧类型污染掉（旧样本被换成新样本）。
            for j, i in enumerate(unknown_idx):
                name = labels.get(j + 1)
                if not name:
                    continue
                if name in templates:
                    same, score = gv.match_tile(unknown_crops[j], {name: templates[name]})
                    if score < 0.85:
                        new_name = f"{name}-{len(templates)}"
                        print(
                            f"  name clash: '{name}' exists but sample only "
                            f"{score:.2f} similar -> '{new_name}'"
                        )
                        name = new_name
                templates[name] = unknown_crops[j]
            gv.save_templates(templates)
            print("template library now:", list(templates))

    # 识别汇总
    counts = {}
    for t in grid + slot:
        k = t.get("kind") or "?"
        counts[k] = counts.get(k, 0) + 1
    print("counts:", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
