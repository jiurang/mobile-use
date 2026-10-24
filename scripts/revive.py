"""复活助手：槽位满失败后，点「获得(1/1)」看广告复活，再关掉广告回到游戏。

广告页的判定不看文字（OCR 不划算），看**中央是不是游戏界面**：
游戏背景是浅绿 (206,251,151)，广告是全屏视频。判据 = 中央区域浅绿像素占比。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import game_vision as gv  # noqa: E402

BTN_GET_ITEM = (602, 1556)  # 失败弹窗里的「获得(1/1)」
BTN_CLOSE_AD = (1153, 184)  # 广告页右上角 ✕
OUT = r"E:\MyCoding\mobile-use\probe_out"


def is_game_screen(img: np.ndarray) -> bool:
    """中央区域是不是游戏界面（浅绿背景）。"""
    roi = img[900:1600, 300:900]
    if roi.size == 0:
        return False
    b, g, r = roi[:, :, 0].astype(int), roi[:, :, 1].astype(int), roi[:, :, 2].astype(int)
    green_bg = ((g > 200) & (r > 150) & (r < 245) & (b > 100) & (b < 215)).mean()
    return green_bg > 0.25


def is_ad_screen(img: np.ndarray) -> bool:
    """右上角是不是黑的 —— 广告页顶部是黑色操作条，游戏页右上角是浅绿底。

    这一条必须有：广告页的关闭 ✕ (1153,184) 和**游戏页**右上角的「◉」按钮
    在同一个坐标。盲点 ✕ 会在回到游戏后把「当前赛况」面板点开（踩过）。
    """
    roi = img[60:220, 900:1200]
    if roi.size == 0:
        return False
    return (roi.max(axis=2) < 90).mean() > 0.45


def has_continue_button(img: np.ndarray) -> bool:
    """广告中途会弹「再看 N 秒可领奖励 / 继续观看 / 坚持退出」。

    橙色「继续观看」按钮的特征：R>200, 100<G<185, 40<B<120 的大块区域。
    出现它就点掉，否则广告会一直停在中间态。
    """
    roi = img[1400:1750, 300:920]
    if roi.size == 0:
        return False
    b, g, r = roi[:, :, 0].astype(int), roi[:, :, 1].astype(int), roi[:, :, 2].astype(int)
    orange = ((r > 200) & (g > 100) & (g < 185) & (b > 40) & (b < 120)).mean()
    return orange > 0.10


def has_continue_popup(img: np.ndarray) -> bool:
    """广告中途的「再看 N 秒可领奖励」二次确认弹窗 = 中央一大块白色面板。

    不用文字判据（"N秒后可领奖励" 和 "领取成功" 的白字位置重叠，
    按白色像素分不开，实测被这个误判坑了三次）。白色圆角面板是唯一
    稳定的视觉特征。
    """
    roi = img[860:1880, 200:1020]
    if roi.size == 0:
        return False
    white = (roi.min(axis=2) > 225).mean()
    return white > 0.25


def has_reward_text(img: np.ndarray) -> bool:
    """广告播完时右上角出现「领取成功」。只看那一小块（避开倒计时文字）。"""
    roi = img[150:225, 930:1100]
    if roi.size == 0:
        return False
    return (roi.min(axis=2) > 210).mean() > 0.10


def board_tiles(img: np.ndarray) -> int:
    """棋盘上能看到的牌数 —— 用来确认真的回到了游戏。"""
    return len(gv.detect_tiles(img))


def main() -> int:
    """复活流程（固定时序，不靠文字判据）。

    教训：这个广告的三个状态在像素上高度相似 ——
      播放中   = 右上角「N秒后可领奖励」+ 黑底
      二次确认 = 中央白面板「再看N秒可领奖励」
      播完     = 右上角「领取成功」+ 黑底
    白字位置互相重叠，按颜色/白像素分不开（实测误判三次）。
    所以改成**固定时序**：等广告自己播完再点 ✕；万一被二次确认面板挡住，
    就点「继续观看」重新等一轮。
    """
    print("tap 获得(1/1) ...", flush=True)
    gv.tap(*BTN_GET_ITEM)

    for round_no in range(1, 5):
        time.sleep(38)  # 让广告播完（广告 ~30s）
        img = gv.screencap()
        if has_continue_popup(img):
            print(f"[{round_no}] 二次确认白面板 -> 点『继续观看』", flush=True)
            gv.tap(612, 1590)
            time.sleep(3)
            continue
        print(f"[{round_no}] 点 ✕ 关广告", flush=True)
        gv.tap(*BTN_CLOSE_AD)
        time.sleep(4)
        img = gv.screencap()
        cv2.imwrite(f"{OUT}/revive_done.png", img)
        n = board_tiles(img)
        print(f"[{round_no}] 关广告后可见牌数={n}", flush=True)
        if n >= 5:
            print("back to game", flush=True)
            return 0

    img = gv.screencap()
    cv2.imwrite(f"{OUT}/revive_done.png", img)
    print("gave up waiting for ad", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
