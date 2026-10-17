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

## GPU 后端：实测结论（RTX 4060 Laptop / 驱动 566.26 / ORT-GPU 1.22 / Python 3.12）

先说结论：**GPU 化本身没用，GPU 化 + 批处理才有 2.44x**。

`rec.onnx` 的输入是 `[N, 3, 48, W]`，N 和 W 都是动态的，所以一批框可以一次喂进去。
但改造前是**逐框串行** `run()` 的，一屏 26 个框就是 26 次调用 —— **这才是瓶颈**。

稳态中位耗时（1220x2712 真机截图，25/26 框，跑 4 轮取中位）：

| 阶段 | CPU | CUDA | |
| --- | --- | --- | --- |
| det 前向 | 111.3ms | 30.1ms | **3.69x** |
| det 全程（含后处理） | 115.2ms | 34.3ms | 3.35x |
| rec 批量（1 次调用） | 113.2ms | 59.2ms | 1.91x |
| rec 逐框（1 次调用） | 6.2ms | 40.5ms | **0.15x** |
| **合计** | **228.4ms** | **93.6ms** | **2.44x** |

最后两行是全部关键：**CUDA 每次 `run()` 的固定开销约 40ms，CPU 只要 6.2ms**
（H2D/D2H 拷贝 + kernel launch + 同步）。所以逐框串行时 GPU 会被这些开销吃掉，
反而比 CPU 慢 4 倍（实测 741ms vs 178ms）；批量化把 26 次调用压成 1 次之后，
GPU 才真正开始发挥 —— det 前向上它确实快 3.7 倍。

DirectML 更差：批量化后仍然打不过 CPU（139.8ms vs 118.8ms），逐框时慢 4 倍。
所以 `device=auto` **只探 CUDA，不探 DML**：DirectML 要额外装
`onnxruntime-directml`，却换不来速度，不该被 auto 悄悄选中。

另外两个数据：

- **det 的后处理只占 4ms**（115.2 - 111.3），findContours / unclip / pyclipper
  都不是瓶颈。所以 det 不需要为 GPU 做任何改造，模型搬上 GPU 就够了。
- **单轮比稳态慢得多**（端到端单轮 238ms vs 稳态 94ms），差的是首次调用的
  内存分配与 kernel 加载。这也是 `_warmup` 存在的原因。

CUDA EP 另有两条硬约束，都实测撞过：

1. **ORT 版本和 CUDA 大版本必须配套。** `onnxruntime-gpu` 1.31 要
   **CUDA 13 + cuDNN 9**，报 `cublasLt64_13.dll` 缺失；而 1.19~1.22 是
   CUDA 12 系列。选错版本就是一堆缺 dll，且降级是**静默**的。
2. **Windows 上 onnxruntime 不会自动搜 CUDA dll 目录**，而且
   `os.add_dll_directory` **对它无效**（它用原生 `LoadLibrary`）。
   必须在进程内改 PATH —— 见 `register_cuda_dll_dirs`。

**ORT 拿到 GPU provider 后如果初始化失败，会静默退回 CPU 并只打日志**（实测
`SESSION OK -> ['CPUExecutionProvider']`）。所以每次建完 session 都要用
`get_providers()` 校验实际落点，见 `verify_provider` —— 不校验等于没改。
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from minitap.mobile_use.services.ocr.types import (
    DEVICE_PROVIDERS as _DEVICE_PROVIDERS,
)
from minitap.mobile_use.services.ocr.types import (
    OcrConfig,
    OcrText,
    resolve_device,
    resolve_rec_batch,
)
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
# 批量 pad 时的填充值：归一化之后纯黑是 (0 - 0.5) / 0.5 = -1。
# 别用 0 —— 那对应中灰，会被 CTC 当成内容读出多余字符。
_REC_PAD_VALUE = -1.0


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


# ---------------------------------------------------------------- device / provider

# device 名 -> provider 名的映射，以及 `resolve_device`，都在 types.py 里 ——
# registry 的实例缓存 key 也要用**最终生效**的后端，不能直接读 `config.device`。
#
# GPU provider 后面必须跟 CPU provider：ORT 是在**节点级**回退的，只给 GPU 时
# 一旦某个算子没有 GPU 实现就直接报错；两个都给，它会自动把那几个节点放 CPU。

# auto 模式只探 CUDA，不探 DML。见模块 docstring 的实测表：DirectML 批量化后
# 仍然打不过 CPU，还要额外装 onnxruntime-directml，不该被 auto 悄悄选中。
_AUTO_ORDER: tuple[str, ...] = ("cuda",)

# `os.add_dll_directory` 的返回值必须持有住，否则目录立刻会被从搜索路径移除。
_DLL_DIR_HANDLES: list[Any] = []


def resolve_provider_options(config: OcrConfig, providers: list[str]) -> list[dict[str, str]]:
    """给每个 provider 配一份 options，长度必须和 `providers` 一致。

    `device_id` 在这里才真正生效。之前它只是个装饰性字段：写进了缓存 key、
    文档也写了「多卡机器上指定用哪张」，但**从来没传给 onnxruntime** ——
    多卡机器上改它完全没有作用，而且没有任何报错。

    只给 GPU provider 配：CPU EP 不认 `device_id`，给了会被 ORT 判为非法 key。
    """
    options: list[dict[str, str]] = [{} for _ in providers]
    for index, name in enumerate(providers):
        if name in (
            _DEVICE_PROVIDERS["cuda"],
            _DEVICE_PROVIDERS["dml"],
        ):
            options[index] = {"device_id": str(config.device_id)}
    return options


def resolve_providers(config: OcrConfig, onnxruntime: Any) -> tuple[list[str], str, bool]:
    """按配置解析 provider 列表。

    返回 `(providers, expected, explicit)`：
    - `providers`  按优先级排列，第一个是**期望**落到的 provider。
    - `expected`   期望落点，用来和 `session.get_providers()` 对账。
    - `explicit`   用户是不是点名要了某个后端（点名了但没拿到必须 warn）。

    解析不出来时不报错：降级到 CPU 并交给上层上报。OCR 是增强感知，
    不该因为显卡的事把整轮感知搞挂。
    """
    device = resolve_device(config)
    explicit = device in _DEVICE_PROVIDERS
    if explicit:
        order = [_DEVICE_PROVIDERS[device]]
    elif device != "auto":
        logger.warning(
            f"未知的 OCR device '{device}'，按 auto 处理。可选：{', '.join(_DEVICE_PROVIDERS)}"
        )
        order = [_DEVICE_PROVIDERS[name] for name in _AUTO_ORDER]
    else:
        order = [_DEVICE_PROVIDERS[name] for name in _AUTO_ORDER]

    available = set(onnxruntime.get_available_providers())
    providers = [ep for ep in order if ep in available]
    cpu = _DEVICE_PROVIDERS["cpu"]
    if not providers:
        providers = [cpu]
    elif cpu not in providers:
        providers.append(cpu)

    return providers, providers[0], explicit


def register_cuda_dll_dirs() -> list[str]:
    """把 `MOBILE_USE_CUDA_DLL_DIR` 里的目录接进本进程的 DLL 搜索路径。

    **只调 `os.add_dll_directory` 是不够的** —— 实测踩过：注册成功、日志也打了
    `cuda_dll_dirs=...`，但建 session 仍然报 `cublasLt64_12.dll` 缺失。

    原因是 ORT 的 provider dll 由 C++ 侧用原生 `LoadLibrary` 加载
    （`onnxruntime_pybind_state.cc`），**不走** Python 3.8+ 的
    `LOAD_LIBRARY_SEARCH_USER_DIRS` 标志，所以 `add_dll_directory` 注册的目录
    它根本看不见。原生 `LoadLibrary` 认的是进程启动时的 PATH，而 Python 3.8+
    又不再替 `ctypes` 拼 PATH。

    所以两件事都要做：
    1. 改 `os.environ["PATH"]` —— 对原生 `LoadLibrary` 生效，**这步是关键**；
    2. `os.add_dll_directory` —— 给走 Python ctypes 的加载路径兜底。

    `add_dll_directory` 返回的句柄必须持有住，否则目录立刻从搜索路径移除。

    多个目录用 `os.pathsep`（Windows 上是 `;`）分隔。只警告不抛 —— dll 找不到
    会表现为降级到 CPU，不需要在这里拦。
    """
    import os

    raw = os.getenv("MOBILE_USE_CUDA_DLL_DIR", "")
    dirs = [item.strip() for item in raw.split(os.pathsep) if item.strip()]
    if not dirs:
        return []

    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        for directory in dirs:
            try:
                _DLL_DIR_HANDLES.append(add_dll_directory(directory))
            except OSError as e:
                logger.warning(f"CUDA dll 目录不可用 {directory}: {type(e).__name__}: {e}")

    # 关键一步：对原生 LoadLibrary 生效。已存在的同名目录不要重复拼。
    existing = os.environ.get("PATH", "").split(os.pathsep)
    added = [d for d in dirs if d not in existing]
    if added:
        os.environ["PATH"] = os.pathsep.join([*added, os.environ.get("PATH", "")])

    return dirs


def verify_provider(session: Any, expected: str, explicit: bool, device_label: str) -> str:
    """核对 session 实际落在哪个 provider 上，返回实际 provider。

    这一步不能省：ORT 拿到 GPU provider 后如果初始化失败，会**静默**退回
    CPU 并只往 stderr 打一行日志（实测 `SESSION OK -> ['CPUExecutionProvider']`）。
    不对账的话，「配了 GPU」和「跑在 GPU 上」完全是两回事，而代码看起来一切正常。
    """
    actual = session.get_providers()[0] if session.get_providers() else "(none)"
    if actual == expected:
        logger.info(f"OCR inference backend: {actual}")
        return actual

    message = (
        f"OCR 推理后端降级：要的是 {expected}（device={device_label}），"
        f"实际拿到 {actual}。"
        + (
            " CUDA EP 通常是因为 onnxruntime-gpu 的版本和本机 CUDA 大版本不匹配"
            "（1.31 要 CUDA 13，1.19~1.22 是 CUDA 12），"
            "或缺 CUDA/cuDNN dll —— 可用 MOBILE_USE_CUDA_DLL_DIR 指向解压目录。"
            if expected == _DEVICE_PROVIDERS["cuda"]
            else ""
        )
    )
    (logger.warning if explicit else logger.info)(message)
    return actual


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


def _rec_row(crop: np.ndarray) -> np.ndarray:
    """crop -> `[3, 48, target_w]`，按**自己的**高宽比缩放。

    宽度取各自算出来的 target_w（所以窄框保持窄），这样 pad 成一批时
    每行字形的几何比例都不变。见 `_recognize_crops` 里 `pad` 和 `resize`
    的区别。
    """
    import cv2
    import numpy as np

    height, width = crop.shape[:2]
    target_w = min(int(np.ceil(height * (_REC_MAX_WIDTH / _REC_HEIGHT))), _REC_MAX_WIDTH)
    resized = cv2.resize(crop, (max(1, target_w), _REC_HEIGHT))
    rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
    return ((rgb - 0.5) / 0.5).transpose(2, 0, 1)


def _rec_normalize_rows(rows: list[np.ndarray]) -> np.ndarray:
    """一组同尺寸 BGR 图 -> `[N, 3, 48, W]` 归一化张量。"""
    import cv2
    import numpy as np

    stacked = np.stack([cv2.resize(r, (r.shape[1], _REC_HEIGHT)) for r in rows])
    rgb = stacked[:, :, :, ::-1].astype(np.float32) / 255.0
    return ((rgb - 0.5) / 0.5).transpose(0, 3, 1, 2).astype(np.float32)


def _rec_preprocess(crop: np.ndarray) -> np.ndarray:
    """单图版，带 batch 维。走 `serial` 路径时用。"""
    import numpy as np

    return _rec_row(crop)[None].astype(np.float32)


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

        dll_dirs = register_cuda_dll_dirs()
        device_label = resolve_device(self._config)
        providers, expected, explicit = resolve_providers(self._config, onnxruntime)
        provider_options = resolve_provider_options(self._config, providers)

        self._det_sess = onnxruntime.InferenceSession(
            str(det_path), options, providers=providers, provider_options=provider_options
        )
        det_actual = verify_provider(self._det_sess, expected, explicit, device_label)
        self._rec_sess = onnxruntime.InferenceSession(
            str(rec_path), options, providers=providers, provider_options=provider_options
        )
        rec_actual = verify_provider(self._rec_sess, expected, explicit, device_label)

        # 两个模型的算子集不同，可能一个落 GPU 一个落 CPU，两个都核一遍。
        self._device = det_actual if det_actual == rec_actual else f"{det_actual}+{rec_actual}"

        # keys.txt 是 UTF-8。一行一个字符，不要用系统默认编码读 ——
        # 中文 Windows 上按 GBK 解会直接 UnicodeDecodeError。
        self._keys = keys_path.read_text(encoding="utf-8").splitlines()
        if not self._keys:
            raise ValueError(f"OCR 字典为空：{keys_path}")

        warmup_ms = self._warmup()

        logger.info(
            f"OCR engine '{self.name}' ready | models={self._model_dir} "
            f"dict={len(self._keys)} entries | backend={self._device} "
            f"(requested device={device_label}) "
            + (f"| cuda_dll_dirs={','.join(dll_dirs)}" if dll_dirs else "")
            + (f" | warmup={warmup_ms:.0f}ms" if warmup_ms is not None else "")
        )

    @property
    def device(self) -> str:
        """**实际**跑到哪个后端，不是配置里请求的那个。

        两者可能不一致（GPU provider 初始化失败会静默退回 CPU），所以想确认
        显卡有没有被用上，要看这个，不要看 `config.device`。
        """
        return getattr(self, "_device", "CPUExecutionProvider")

    def _warmup(self) -> float | None:
        """用一张极小的合成图先跑一遍 det + rec。

        GPU 后端的第一次 `run()` 普遍比后续慢：DirectML 要 JIT 编译 shader，
        CUDA 要加载 cublas kernel。agent 的第一轮感知不该替这份开销买单，
        所以在构造函数里就付掉。

        预热失败不影响可用性（形状问题之类），只记 debug。
        """
        import time

        import numpy as np

        if self._device.startswith("CPU") and self._device.count("+") == 0:
            return None  # CPU 后端没有这个开销，不必白跑一次

        try:
            start = time.perf_counter()
            self._det_sess.run(None, {"x": np.zeros((1, 3, 64, 64), dtype=np.float32)})
            self._rec_sess.run(None, {"x": np.zeros((1, 3, 48, 32), dtype=np.float32)})
            return (time.perf_counter() - start) * 1000
        except Exception as e:  # noqa: BLE001 — 预热是优化，不是功能
            logger.debug(f"OCR warmup skipped: {type(e).__name__}: {e}")
            return None

    def _recognize_crops(self, crops: list[Any]) -> list[Any]:
        """把一批 crop 送进 rec，返回与输入同序的特征列表。

        `rec_batch` 有三种，差别是**调用次数**和**每行的几何**：

        - `serial`  逐框调用。与改造前逐字一致，CPU 上 26 框约 178ms。
        - `pad`     每行保持 `_rec_row` 算出的自己的宽度（**与 serial 逐像素
          相同**），右侧用黑色补齐到组内最大宽度，一次调用。CTC 靠 blank 跳过
          把补齐区读成空白。CPU 约 1.5x、CUDA 约 5.5x。
        - `resize`  把所有行强行拉到组内最大宽度，一次调用。矮框因此被**额外
          横向拉长**（实测 220px 的框拉成 320px，1.45x），长句会读错。

        为什么值得做：rec 的输入是 `[N, 3, 48, W]`，N 和 W 都是动态的，
        所以一批本来就能一次喂进去。改造前是逐框 `run()`，一屏 26 个框
        就是 26 次调用 —— **这才是瓶颈，不是算力**。GPU 尤其吃亏：每次
        `run()` 的 H2D/D2H 拷贝加同步约 40ms，而 CPU 只要 6.2ms，所以
        不批量化的话 CUDA 会比 CPU 慢 4 倍（实测 741ms vs 178ms）。

        注意 `_rec_row` 的目标宽度只由 crop 的**高度**决定 —— 这是 PaddleOCR
        rec 的固有设计（固定高 48 + 按比例算宽），不是这里引入的形变。
        `pad` 与 `serial` 的每行内容因此完全一致，差别仅在右侧多出的黑边。
        """
        import cv2
        import numpy as np

        if not crops:
            return []

        mode = resolve_rec_batch(self._config)
        if mode == "serial" or len(crops) == 1:
            return [self._rec_sess.run(None, {"x": _rec_preprocess(c)})[0][0] for c in crops]

        if mode == "resize":
            common_w = max(_rec_row(crop).shape[2] for crop in crops)
            stretched = [cv2.resize(crop, (common_w, _REC_HEIGHT)) for crop in crops]
            blob = _rec_normalize_rows(stretched)
        else:  # pad
            rows = [_rec_row(crop) for crop in crops]
            common_w = max(row.shape[2] for row in rows)
            batch = np.full((len(rows), 3, _REC_HEIGHT, common_w), _REC_PAD_VALUE, dtype=np.float32)
            for index, row in enumerate(rows):
                batch[index, :, :, : row.shape[2]] = row
            blob = batch

        out = self._rec_sess.run(None, {"x": blob})[0]
        return [out[index] for index in range(out.shape[0])]

    def recognize(self, image_base64: str) -> list[OcrText]:
        import numpy as np

        image = self._decode(image_base64)
        if image is None:
            logger.warning("OCR: could not decode screenshot")
            return []

        boxes = _detect(image, self._det_sess, self._config)

        # 先裁剪再批量识别：批量只省 run() 的次数，裁剪本来就得逐框做。
        pairs: list[tuple[list[list[float]], Any]] = []
        for box in boxes:
            crop = _crop(image, box)
            if crop is not None:
                pairs.append((box, crop))

        feats = self._recognize_crops([crop for _, crop in pairs])

        results: list[OcrText] = []
        for (box, _crop_unused), feat in zip(pairs, feats):
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
