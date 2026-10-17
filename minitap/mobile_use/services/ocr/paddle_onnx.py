"""百度 PaddleOCR 的 ONNX 运行时实现（det + rec + 字典）。

对应模型目录：`minitap/model/ocr/{det.onnx, rec.onnx, keys.txt}`。

下面三个坑都是实测踩出来的，不是从文档抄的，改之前先读完再动：

1. **det 输出可能已经带 Sigmoid。**
   这批 `det.onnx` 的输出范围是 [0.0000, 1.0000]（已经过 sigmoid）。
   再套一次 sigmoid 会把整张图压到 0.5 附近，于是 `det_thresh=0.3` 全屏命中、
   框全部被 unclip 成垃圾，最终一个框都出不来。所以要先看范围再决定要不要
   套 sigmoid —— 见 `_to_prob`。

2. **pyclipper 1.4.0 的 `PyclipperOffset` 是「原始单位进、原始单位出」。**
   PaddleOCR 那套 `Execute(distance * 1e2)` 再 `/1e2` **不能抄**。
   实测：输入 (794,2550)-(914,2575)、distance=15.5 时正确结果是
   x[779,930] y[2535,2591]；一旦把 distance 传成 15.5e2，结果变成
   x[-758,2466]，再除以 100 就得到 x[-7.58, 24.66] —— 一堆挤在原点附近、
   带负坐标的框。坐标映射错成这样之后，现象只是「识别率低」，极难往回追。

3. **CTC 的最后一个类别是空格。**
   类别布局是 `['blank'] + dict + [' ']`。PaddleOCR 默认
   `use_space_char=False`，把这个空格类当 blank 跳过。只跳过 0 的话会
   `IndexError`，因为 dict 有 18708 条而类别数是 18710。

另外两处和上游不同、但实测更稳：

- **unclip 出来的是带圆角的多边形**（几十个点），`getPerspectiveTransform`
  只收 4 个点。这里按 PaddleOCR 的做法退化成外接矩形，并保留它对竖排
  文字的 90° 旋转处理。
- **polygon area 用鞋带公式**，不为一个面积计算引入 shapely。
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from minitap.mobile_use.services.ocr.types import OcrConfig, OcrText
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.ui_hierarchy import ElementBounds

if TYPE_CHECKING:
    import numpy as np
    import onnxruntime as ort

logger = get_logger(__name__)

# DB 检测的输入归一化参数（PaddleOCR NormalizeImage + toRGB）
_DET_MEAN = (0.485, 0.456, 0.406)
_DET_STD = (0.229, 0.224, 0.225)
_DET_MULTIPLE = 32

# rec 的输入固定高度，以及长宽比上限（PaddleOCR rec_image_shape=[3,48,320]）
_REC_HEIGHT = 48
_REC_MAX_WIDTH = 320


class OcrDependenciesMissing(RuntimeError):
    """缺少 OCR 运行时依赖时抛出，附带安装提示。"""


def _require_deps() -> tuple[Any, Any, Any]:
    """延迟导入重量级依赖。

    仓库主依赖里**没有** onnxruntime / opencv / pyclipper，它们属于可选的
    `[ocr]` extra。没装时不能把整个包 import 搞崩 —— 只有真的要用 OCR 才报错。
    """
    missing: list[str] = []
    try:
        import cv2
    except ImportError:
        cv2 = None  # type: ignore[assignment]
        missing.append("opencv-python-headless")
    try:
        import onnxruntime
    except ImportError:
        onnxruntime = None  # type: ignore[assignment]
        missing.append("onnxruntime")
    try:
        import pyclipper
    except ImportError:
        pyclipper = None  # type: ignore[assignment]
        missing.append("pyclipper")

    if missing:
        raise OcrDependenciesMissing(
            "OCR 感知需要以下依赖："
            + ", ".join(missing)
            + "。安装方式：uv pip install 'minitap-mobile-use[ocr]'"
            "（或 pip install " + " ".join(missing) + "）"
        )
    return cv2, onnxruntime, pyclipper


def resolve_model_dir(config: OcrConfig) -> Path:
    """按 配置 -> 环境变量 -> 仓库内默认 的顺序解析模型目录。"""
    if config.model_dir is not None:
        return Path(config.model_dir)

    env = os.getenv("MOBILE_USE_OCR_MODEL_DIR")
    if env:
        return Path(env)

    # config.py 里已定义 ROOT_DIR = 仓库根。这里再兜一层：包被单独安装时
    # ROOT_DIR 可能指向 site-packages，minitap/model 就在那底下。
    from minitap.mobile_use.config import ROOT_DIR

    return ROOT_DIR / "minitap" / "model" / "ocr"


# ---------------------------------------------------------------- det


def _to_prob(pred: np.ndarray) -> np.ndarray:
    """把 det 的原始输出转成概率图。

    导出图里可能已经包含了 Sigmoid，也可能只出 logits，两种都要认。
    见模块 docstring 的坑 1。
    """
    import numpy as np

    if float(pred.min()) >= 0.0 and float(pred.max()) <= 1.0:
        return pred[0, 0]
    return 1.0 / (1.0 + np.exp(-pred[0, 0]))


def _det_resize(img: np.ndarray, limit_side_len: int) -> tuple[np.ndarray, int, int]:
    """按 limit_side_len 缩放并对齐到 32 的倍数，返回 (图, 宽, 高)。"""

    h, w = img.shape[:2]
    ratio = 1.0
    if max(h, w) > limit_side_len:
        ratio = limit_side_len / max(h, w)
    # min(1.0, ratio) 会让本来就小的图不变大，PaddleOCR 是这么写的。
    ratio = min(1.0, ratio)
    rh = max(int(round(h * ratio / _DET_MULTIPLE) * _DET_MULTIPLE), _DET_MULTIPLE)
    rw = max(int(round(w * ratio / _DET_MULTIPLE) * _DET_MULTIPLE), _DET_MULTIPLE)

    import cv2

    return cv2.resize(img, (rw, rh)), rw, rh


def _det_normalize(img: np.ndarray) -> np.ndarray:
    import numpy as np

    # BGR -> RGB（PaddleOCR 的 NormalizeImage 默认 toRGB=True）
    rgb = img[:, :, ::-1].astype(np.float32) / 255.0
    x = (rgb - np.asarray(_DET_MEAN, np.float32)) / np.asarray(_DET_STD, np.float32)
    return x.transpose(2, 0, 1)[None].astype(np.float32)


def _order_points(points: np.ndarray) -> np.ndarray:
    """四点按 左上-右上-右下-左下 排。"""
    import numpy as np

    rect = np.zeros((4, 2), dtype=np.float32)
    total = points.sum(axis=1)
    rect[0] = points[np.argmin(total)]
    rect[2] = points[np.argmax(total)]
    diff = np.diff(points, axis=1).ravel()
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def _polygon_area(points: np.ndarray) -> float:
    """鞋带公式。"""
    import numpy as np

    x, y = points[:, 0], points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _perimeter(points: np.ndarray) -> float:
    import numpy as np

    return float(
        np.linalg.norm(points[0] - points[1])
        + np.linalg.norm(points[1] - points[2])
        + np.linalg.norm(points[2] - points[3])
        + np.linalg.norm(points[3] - points[0])
    )


def _unclip(points: np.ndarray, ratio: float, pyclipper: Any) -> np.ndarray | None:
    """DB 的框外扩。

    注意这里**没有**任何 1e2 缩放，见模块 docstring 的坑 2。
    pyclipper 1.4.0 没有导出 `EndType` 枚举，但导出了同值的 `ET_*` 常量，
    而 `AddPath` 强制要 3 个位置参数。1.3.0 在 win/py3.12 没有 wheel
    （会退化成源码编译并失败），所以别想着退回旧版绕开这件事。
    """
    import numpy as np

    pts = points.astype(np.float64)
    length = _perimeter(points)
    if length < 1e-6:
        return None

    offsetter = pyclipper.PyclipperOffset()
    offsetter.AddPath(pts.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = offsetter.Execute(_polygon_area(pts) * ratio / length)
    if not expanded or len(expanded[0]) < 3:
        return None
    return np.asarray(expanded[0], dtype=np.float32).reshape(-1, 2)


def _score_fast(points: np.ndarray, prob: np.ndarray) -> float:
    """DB fast 评分：框内概率图的均值。"""
    import numpy as np

    h, w = prob.shape
    x1 = int(np.clip(np.floor(points[:, 0].min()), 0, w - 1))
    x2 = int(np.clip(np.ceil(points[:, 0].max()), 0, w - 1))
    y1 = int(np.clip(np.floor(points[:, 1].min()), 0, h - 1))
    y2 = int(np.clip(np.ceil(points[:, 1].max()), 0, h - 1))
    return float(prob[y1 : y2 + 1, x1 : x2 + 1].mean())


def _detect(
    img: np.ndarray, det_sess: ort.InferenceSession, config: OcrConfig
) -> list[list[list[float]]]:
    import cv2
    import numpy as np
    import pyclipper

    height, width = img.shape[:2]
    resized, rw, rh = _det_resize(img, config.det_limit_side_len)
    pred = det_sess.run(None, {"x": _det_normalize(resized)})[0]
    prob = _to_prob(cast("np.ndarray", pred))

    binary = (prob > config.det_thresh).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[list[list[float]]] = []
    for contour in contours[:1000]:
        rect = cv2.minAreaRect(contour)
        if min(rect[1]) < 3 or max(rect[1]) > 4000:
            continue
        points = _order_points(cv2.boxPoints(rect))
        if _score_fast(points, prob) < config.det_box_thresh:
            continue
        expanded = _unclip(points, config.det_unclip_ratio, pyclipper)
        if expanded is None or _polygon_area(expanded) < 10:
            continue
        # 按列缩放回原图；stack 成 (N,2)，别写成 (2,N) 的元组。
        scaled = np.stack([expanded[:, 0] / rw * width, expanded[:, 1] / rh * height], axis=1)
        boxes.append([[float(px), float(py)] for px, py in scaled])

    boxes.sort(key=lambda b: (min(p[1] for p in b), min(p[0] for p in b)))
    return boxes


# ---------------------------------------------------------------- rec


def _crop(img: np.ndarray, box: list[list[float]]) -> np.ndarray | None:
    """把一个多边形框转成 rec 能吃的矩形裁剪图。

    unclip 出来的是 Minkowski 外扩后的圆角多边形（几十个点），而
    `getPerspectiveTransform` 只收 4 个点，所以这里退化成外接矩形。
    """
    import cv2
    import numpy as np

    points = np.asarray(box, dtype=np.float32)
    ih, iw = img.shape[:2]
    # 外扩框可以越界，先夹到图内。
    x1 = max(0.0, min(float(points[:, 0].min()), iw - 1.0))
    x2 = max(0.0, min(float(points[:, 0].max()), iw - 1.0))
    y1 = max(0.0, min(float(points[:, 1].min()), ih - 1.0))
    y2 = max(0.0, min(float(points[:, 1].max()), ih - 1.0))

    width, height = x2 - x1, y2 - y1
    if width < 1 or height < 1:
        return None

    crop = img[int(y1) : int(y2) + 1, int(x1) : int(x2) + 1]
    if crop.size == 0 or min(crop.shape[:2]) < 2:
        return None
    # 竖排（高远大于宽）转正，否则 CTC 会把逐字竖排读成乱序。
    if height > 1.5 * width:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return crop


def _rec_preprocess(crop: np.ndarray) -> np.ndarray:
    import cv2
    import numpy as np

    height, width = crop.shape[:2]
    target_w = min(int(np.ceil(height * (_REC_MAX_WIDTH / _REC_HEIGHT))), _REC_MAX_WIDTH)
    resized = cv2.resize(crop, (max(1, target_w), _REC_HEIGHT))
    rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
    return ((rgb - 0.5) / 0.5).transpose(2, 0, 1)[None].astype(np.float32)


def _ctc_decode(feat: np.ndarray, keys: list[str]) -> tuple[str, float]:
    """CTC 贪心解码。

    类别布局 `['blank'] + dict + [' ']`：0 是 blank，`len(keys)+1` 是空格
    （PaddleOCR 默认 `use_space_char=False`，按 blank 跳过）。见模块 docstring 的坑 3。
    """

    space_class = len(keys) + 1
    indices = feat.argmax(axis=-1)
    confidences = feat.max(axis=-1)

    chars: list[str] = []
    scores: list[float] = []
    previous = -1
    for position, index in enumerate(indices):
        if index == 0 or index == space_class or index == previous:
            previous = index
            continue
        if 1 <= index <= len(keys):
            chars.append(keys[index - 1])
            scores.append(float(confidences[position]))
        previous = index

    text = "".join(chars)
    return text, (sum(scores) / len(scores) if scores else 0.0)


# ---------------------------------------------------------------- engine


class PaddleOcrOnnxEngine:
    """百度 PaddleOCR（det + rec）的 ONNX 实现。"""

    name = "paddle_onnx"

    def __init__(self, config: OcrConfig | None = None):
        cv2, onnxruntime, _ = _require_deps()
        import numpy as np  # noqa: F401  （确认 numpy 在，_require_deps 没查它）

        self._config = config or OcrConfig()
        self._model_dir = resolve_model_dir(self._config)

        det_path = self._model_dir / "det.onnx"
        rec_path = self._model_dir / "rec.onnx"
        keys_path = self._model_dir / "keys.txt"
        missing = [p.name for p in (det_path, rec_path, keys_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"OCR 模型缺失：{', '.join(missing)}（期望目录 {self._model_dir}）。"
                " 可用 OcrConfig.model_dir 或 MOBILE_USE_OCR_MODEL_DIR 指定其他模型目录。"
            )

        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3  # 只留错误，避免每轮刷屏
        options.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
        providers = ["CPUExecutionProvider"]
        self._det_sess = onnxruntime.InferenceSession(str(det_path), options, providers=providers)
        self._rec_sess = onnxruntime.InferenceSession(str(rec_path), options, providers=providers)

        # keys.txt 是 UTF-8。一行一个字符，不要用系统默认编码读 ——
        # 中文 Windows 上按 GBK 解会直接 UnicodeDecodeError。
        self._keys = keys_path.read_text(encoding="utf-8").splitlines()
        if not self._keys:
            raise ValueError(f"OCR 字典为空：{keys_path}")

        logger.info(
            f"OCR engine '{self.name}' ready | models={self._model_dir} "
            f"dict={len(self._keys)} entries"
        )

    def recognize(self, image_base64: str) -> list[OcrText]:
        import numpy as np

        image = self._decode(image_base64)
        if image is None:
            logger.warning("OCR: could not decode screenshot")
            return []

        boxes = _detect(image, self._det_sess, self._config)

        results: list[OcrText] = []
        for box in boxes:
            crop = _crop(image, box)
            if crop is None:
                continue
            feat = self._rec_sess.run(None, {"x": _rec_preprocess(crop)})[0][0]
            text, score = _ctc_decode(feat, self._keys)
            text = text.strip()
            if not text or score < self._config.rec_min_score:
                continue

            points = np.asarray(box, dtype=np.float32)
            x1, y1 = float(points[:, 0].min()), float(points[:, 1].min())
            x2, y2 = float(points[:, 0].max()), float(points[:, 1].max())
            results.append(
                OcrText(
                    text=text,
                    score=round(score, 4),
                    bounds=ElementBounds(
                        x=int(x1),
                        y=int(y1),
                        width=max(1, int(x2) - int(x1)),
                        height=max(1, int(y2) - int(y1)),
                    ),
                )
            )

        results.sort(key=lambda t: (t.bounds.y, t.bounds.x))
        if len(results) > self._config.max_texts:
            logger.warning(
                f"OCR found {len(results)} texts, keeping the top {self._config.max_texts} by score"
            )
            results.sort(key=lambda t: -t.score)
            results = results[: self._config.max_texts]
            results.sort(key=lambda t: (t.bounds.y, t.bounds.x))

        logger.debug(
            f"OCR: {len(boxes)} boxes -> {len(results)} texts "
            f"(min_score={self._config.rec_min_score})"
        )
        return results

    @staticmethod
    def _decode(image_base64: str) -> np.ndarray | None:
        """base64 -> BGR ndarray。

        `cv2.imdecode(np.fromfile(...))` 而不是 `cv2.imread`：Windows 上
        imread 对相对路径和非 ASCII 路径都不稳。
        """
        import cv2
        import numpy as np

        if image_base64.startswith("data:image"):
            image_base64 = image_base64.split(",", 1)[1]
        try:
            raw = base64.b64decode(image_base64)
        except Exception as e:
            logger.error(f"OCR: invalid base64 screenshot: {e}")
            return None
        return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
