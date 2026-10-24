"""
游戏牌面视觉层：检测牌位 + 识别牌型。

为什么不用 OCR / a11y 树：这是今日头条小程序里的纯画布游戏，
`uiautomator dump` 直接失败（实测 exit 1），一个控件都没有。
牌面全是插画（篝火/胡萝卜/玉米/草/蘑菇/毛线球/锄头/洋葱/剪刀…），
文字识别完全用不上。

所以这层做两件事：
1. **检测**：卡底是浅米色 (246,254,209)，用连通域找出每一张「亮牌」
   （被压住的牌底是暗灰色，不会命中这个阈值）。
2. **识别**：牌型靠**模板库**。同类型牌的裁剪图几乎逐像素一致，
   所以模板库存每种牌的一张标准裁剪图，匹配用 NCC 相关度。
   遇到模板库里没有的新牌型，才叫一次 LLM（MiniMax 视觉）给画廊图编号，
   把结果写成新模板 —— 一次调用解决一整批新牌型，不是每步都调模型。

模板库落盘在 probe_out/tile_templates/，跨次运行复用。
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

DEVICE = "89R8ZXNBGQIRJRLF"
TEMPLATE_DIR = Path(r"E:\MyCoding\mobile-use\probe_out\tile_templates")
TEMPLATE_INDEX = TEMPLATE_DIR / "index.json"

# 屏幕 1220x2712
SCREEN_W, SCREEN_H = 1220, 2712
# 顶部 y<430 是小程序状态栏 / 今日头条胶囊按钮，白底会被当成卡底，要排除
MIN_TILE_Y = 430
# 槽位栏（底部木架）
SLOT_ROI = (60, 2180, 1180, 2470)

# 卡底色范围（RGB）：浅米色
CARD_BG_LO = (225, 212, 145)
CARD_BG_HI = (255, 255, 228)


def screencap() -> np.ndarray:
    out = subprocess.run(
        ["adb", "-s", DEVICE, "exec-out", "screencap", "-p"],
        capture_output=True,
        check=True,
    ).stdout
    img = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("screencap decode failed")
    return img


def tap(x: int, y: int) -> None:
    subprocess.run(["adb", "-s", DEVICE, "shell", "input", "tap", str(x), str(y)], check=True)


# --------------------------------------------------------------------------- 检测


def detect_tiles(img: np.ndarray) -> list[dict]:
    """找出所有「亮牌」（可点的牌）。返回按 (行, 列) 排序的列表。"""
    b, g, r = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
    card = (
        (r >= CARD_BG_LO[0])
        & (r <= CARD_BG_HI[0])
        & (g >= CARD_BG_LO[1])
        & (g <= CARD_BG_HI[1])
        & (b >= CARD_BG_LO[2])
        & (b <= CARD_BG_HI[2])
    )
    card[:MIN_TILE_Y, :] = False
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(card.astype(np.uint8), 8)
    tiles: list[dict] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 4000 or w < 50 or h < 50:
            continue
        tiles.append(
            {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "cx": int(x + w / 2),
                "cy": int(y + h / 2),
            }
        )
    # 按行聚类（y 差 < 60 视为同一行），行内按 x 排序
    tiles.sort(key=lambda t: (t["y"], t["x"]))
    rows: list[list[dict]] = []
    for t in tiles:
        if rows and abs(t["y"] - rows[-1][0]["y"]) < 60:
            rows[-1].append(t)
        else:
            rows.append([t])
    out = []
    for row in rows:
        row.sort(key=lambda t: t["x"])
        out.extend(row)
    # 重新编号：行号 + 列号
    for ri, row in enumerate(rows):
        for ci, t in enumerate(row):
            t["row"], t["col"] = ri, ci
    return out


def split_grid_slot(img: np.ndarray, tiles: list[dict]) -> tuple[list[dict], list[dict]]:
    """把牌分成「网格区」和「槽位区」。"""
    grid, slot = [], []
    for t in tiles:
        (slot if SLOT_ROI[1] <= t["cy"] <= SLOT_ROI[3] else grid).append(t)
    slot.sort(key=lambda t: t["x"])
    return grid, slot


def tile_crop(img: np.ndarray, t: dict, pad: int = 2) -> np.ndarray:
    """裁出牌的图标区（去掉底部那条绿边），统一缩放。"""
    x0 = max(0, t["x"] + pad)
    y0 = max(0, t["y"] + pad)
    x1 = min(img.shape[1], t["x"] + t["w"] - pad)
    y1 = min(img.shape[0], t["y"] + int(t["h"] * 0.86))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((64, 64, 3), np.uint8)
    return cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- 模板库


def load_templates() -> dict[str, np.ndarray]:
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, np.ndarray] = {}
    if TEMPLATE_INDEX.exists():
        for name, fname in json.loads(TEMPLATE_INDEX.read_text(encoding="utf-8")).items():
            p = TEMPLATE_DIR / fname
            if p.exists():
                out[name] = cv2.imread(str(p))
    return out


def save_templates(templates: dict[str, np.ndarray]) -> None:
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    index: dict[str, str] = {}
    for i, (name, img) in enumerate(sorted(templates.items())):
        fname = f"tpl_{i:03d}.png"
        cv2.imwrite(str(TEMPLATE_DIR / fname), img)
        index[name] = fname
    TEMPLATE_INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def color_hist(crop: np.ndarray) -> np.ndarray:
    """HSV 颜色直方图（16x16x16），作为牌型的第二判据。

    为什么需要它：**图像相关度分不开「篝火」和「木桩」** ——
    篝火图案下面就是棕色木柴，木桩是棕色年轮，两者的轮廓相关度能到 0.885
    （实测）。而且连 MiniMax 视觉也会把木桩叫成「篝火」，说明这俩在
    像素层面是真的接近。所以再加一道颜色分布：篝火顶部有高饱和红焰，
    木桩整体是低饱和棕 —— 直方图一判就分开了。
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 16, 16], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def match_tile(crop: np.ndarray, templates: dict[str, np.ndarray]) -> tuple[str | None, float]:
    """把一张牌和模板库比对，返回 (牌名, 得分)。

    得分 = min(图像相关度, 颜色直方图相关度) —— 两个判据都得过。
    实测：同一张牌 0.98~1.00；篝火 vs 木桩 图像 0.885 / 颜色 ~0.6 -> 合成 ~0.6。
    """
    if not templates:
        return None, 0.0
    ch = color_hist(crop)
    best_name, best_score = None, -1.0
    for name, tpl in templates.items():
        if tpl is None or tpl.shape != crop.shape:
            continue
        ncc = float(cv2.matchTemplate(crop, tpl, cv2.TM_CCOEFF_NORMED)[0, 0])
        hist_corr = float(cv2.compareHist(color_hist(tpl), ch, cv2.HISTCMP_CORREL))
        score = min(ncc, hist_corr)
        if score > best_score:
            best_name, best_score = name, score
    return best_name, best_score


# --------------------------------------------------------------------------- LLM 标注


def _load_dotenv() -> dict[str, str]:
    env: dict[str, str] = {}
    env_path = Path(r"E:\MyCoding\mobile-use\.env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            env[k.strip()] = v
    return env


def llm_label_gallery(gallery: np.ndarray, count: int) -> dict[int, str]:
    """把画廊图丢给 MiniMax 视觉模型，拿到 {编号: 牌名}。"""
    env = _load_dotenv()
    key = env.get("MINIMAX_API_KEY")
    base = (env.get("MINIMAX_BASE_URL") or "https://api.minimaxi.com/v1").rstrip("/")
    model = env.get("MINIMAX_MODEL") or "MiniMax-M3.1-Flash-Preview"
    if not key:
        raise RuntimeError("MINIMAX_API_KEY not found")

    ok, buf = cv2.imencode(".png", gallery)
    if not ok:
        raise RuntimeError("gallery encode failed")
    b64 = base64.b64encode(buf.tobytes()).decode()
    prompt = (
        f"这是一张游戏牌面的画廊图，每格左上角有一个白色数字编号（共 {count} 张牌）。\n"
        "请逐格识别牌面上画的图案是什么（例如：篝火、胡萝卜、玉米、草、蘑菇、毛线球、"
        "锄头、红水桶、剪刀、蓝手套、白菜、刷子、黄色滚筒…）。\n"
        "命名规则：**同一个图案必须用完全一样的名字**；不同图案才用不同名字。\n"
        "名字用简短中文（2-4 个字）。\n"
        '严格返回 JSON，不要任何多余文字：{"cards":[{"id":<编号>,"name":"<简短中文名>"},...]}'
        f"，id 必须覆盖 1 到 {count} 全部编号。"
    )
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        "max_completion_tokens": 8000,
    }
    import httpx

    last_err = None
    for attempt in range(3):
        try:
            r = httpx.post(
                f"{base}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                timeout=180.0,
            )
            r.raise_for_status()
            data = r.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"no choices: {json.dumps(data, ensure_ascii=False)[:300]}")
            message = choices[0].get("message") or {}
            content = message.get("content")
            if not content:
                # MiniMax 偶尔回空 content（thinking 吃光预算 / 命中审核），
                # 这时把能看到的字段都打出来再重试，别让 KeyError 变成黑箱。
                raise RuntimeError(
                    "empty content; finish_reason="
                    f"{choices[0].get('finish_reason')}, "
                    f"message_keys={list(message)}, "
                    f"base_resp={data.get('base_resp')}"
                )
            m = re.search(r"\{.*\}", content, re.S)
            if not m:
                raise RuntimeError(f"no json in reply: {content[:400]}")
            obj = json.loads(m.group(0))
            result: dict[int, str] = {}
            for item in obj.get("cards", []):
                result[int(item["id"])] = str(item["name"]).strip()
            missing = [i for i in range(1, count + 1) if i not in result]
            if missing:
                raise RuntimeError(f"labels missing ids: {missing}")
            return result
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"  llm attempt {attempt + 1} failed: {e}", flush=True)
            time.sleep(3)
    raise RuntimeError(f"LLM labeling failed: {last_err}")


def unify_labels(crops: list[np.ndarray], labels: dict[int, str]) -> dict[int, str]:
    """一致性自检：长得几乎一样的牌必须同名。

    牌型识别不需要「名字对」，只需要「同一张牌每次都叫同一个名字」——
    名字错会导致该消的消不掉。但 LLM 偶尔会把同一图案分两次命名
    （例如 10 号叫「黄蘑菇」、16 号叫「蜂蜜勺」），那就永远凑不齐 3 张。

    判据：两张牌裁剪图的 NCC 相关度 > 0.93 => 它们是同一张牌，必须同名。
    冲突时以出现次数多的名字为准，把少数派改过来。
    """
    from collections import Counter

    n = len(crops)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = crops[i], crops[j]
            if a.shape != b.shape:
                continue
            score = float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0, 0])
            if score <= 0.93:
                continue
            ni, nj = labels.get(i + 1), labels.get(j + 1)
            if not ni or not nj or ni == nj:
                continue
            counts = Counter(labels.values())
            if counts[nj] > counts[ni]:
                ni, nj = nj, ni
            print(
                f"  unify: #{i + 1}({labels[i + 1]}) ~= #{j + 1}({labels[j + 1]})"
                f" -> 统一为 {ni}",
                flush=True,
            )
            labels[j + 1] = ni
    return labels


def build_gallery(
    crops: list[np.ndarray], cols: int = 6
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """把牌面裁剪图拼成带编号的画廊。返回 (画廊图, 每格在画廊中的 bbox)。"""
    cell = 150
    label_h = 34
    rows = (len(crops) + cols - 1) // cols
    gallery = np.full((rows * (cell + label_h), cols * cell, 3), 245, np.uint8)
    boxes: list[tuple[int, int, int, int]] = []
    for i, c in enumerate(crops):
        r, cc = divmod(i, cols)
        x0 = cc * cell
        y0 = r * (cell + label_h)
        tile = cv2.resize(c, (cell, cell), interpolation=cv2.INTER_CUBIC)
        gallery[y0 + label_h : y0 + label_h + cell, x0 : x0 + cell] = tile
        cv2.rectangle(gallery, (x0, y0), (x0 + cell - 1, y0 + label_h + cell - 1), (60, 60, 60), 2)
        cv2.rectangle(gallery, (x0, y0), (x0 + 40, y0 + label_h - 2), (40, 40, 40), -1)
        cv2.putText(
            gallery, str(i + 1), (x0 + 8, y0 + label_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA,
        )
        boxes.append((x0 + label_h // 2, y0 + label_h, x0 + cell, y0 + label_h + cell))
    return gallery, boxes
