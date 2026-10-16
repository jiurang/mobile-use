"""算清楚：目标高度定在哪，才能让缩放后的截图落在最少的 tile 上。

宽度先压到 768 以内 -> 1 列。高度决定行数：
    h <= 768    -> 1 行
    h <= 1536   -> 2 行
    h <= 2304   -> 3 行

真机 1220x2712 等比缩到宽 768 时，高 = 768 * 2712/1220 = 1706 -> 3 行 = 774 token。
要落到 2 行（516 token），高度必须 <= 1536，即缩放比 <= 1536/2712 = 0.566。
"""

TILE = 768

SRC_W, SRC_H = 1220, 2712


def tiles(w: int, h: int) -> tuple[int, int, int]:
    cols = w // TILE + 1
    rows = h // TILE + 1
    return cols, rows, cols * rows * 258


print(f"原始 {SRC_W}x{SRC_H}: tiles={tiles(SRC_W, SRC_H)}")

print("\n等比缩放，宽压到 <=768 时：")
for w in (768, 700, 640, 610, 560, 500, 440):
    h = round(SRC_H * w / SRC_W)
    print(f"  宽{w:4d} -> {w}x{h:4d}  tiles={tiles(w, h)}")

print("\n要让高度 <=1536（2 行），最大宽度：")
w_max = int(1536 * SRC_W / SRC_H)
h_at_max = round(SRC_H * w_max / SRC_W)
print(f"  宽必须 <= {w_max}，此时高 = {h_at_max}")
print(f"  验证: {w_max}x{h_at_max} tiles={tiles(w_max, h_at_max)}")
