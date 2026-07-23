"""Remove Visible Watermark (RAIW)节点 - 通用区域擦除器：通过图像修复功能，自动移除用户框选范围内的任意物体。

不受位置与内容限制。你只需框选矩形区域，擦除工具便会对框内全部内容自动进行图像修复，
能够去除任意可见标识、水印或物体，不受色彩、样式、所处位置影响。
区域框选由用户自行操作；图像修复运算在 CPU 上执行。
针对 Gemini、豆包这类固定生成引擎无法处理的痕迹，本工具作为通用兜底方案使用。

Erase Region (RAIW)节点 - 后端引擎选项说明：
- cv2（默认）：使用 cv2.inpaint（Telea / Navier-Stokes 算法）。
    优点：效率最高，响应即时，无需额外依赖；
    缺点：大面积、纹理复杂区域修复效果较差。
- migan选项（可选，需额外安装下载 migan 模型和安装环境支持）：
    基于 onnxruntime 运行 MI-GAN 模型（仓库地址：https://huggingface.co/andraniksargsyan/migan/tree/main，MIT 协议）。
    仅 CPU 运算，模型体积约 28 MB，单次调用耗时约 0.19 秒，适配小型瑕疵修复场景；
    优点：处理小幅痕迹时画质接近 LaMa模型。    
    与 lama 逻辑一致，推理前会截取蒙版周边带留白的局部区域（原生分辨率输入，MI-GAN 支持任意尺寸），
    峰值内存占用仅由瑕疵大小决定（约 0.6–0.9 GB），不会随原图尺寸线性增长，低配内存设备也可处理大图。
    建议本地模型库预先下载,本程序不支持通过 huggingface_hub 下载并缓存。
- lama选项（可选，需额外下载 lama 模型和安装环境支持）：
    必须安装 `pip install "remove-ai-watermarks[lama]"` 启用 LaMa 图像擦除后端；
    可选安装 `pip install "remove-ai-watermarks[esrgan]"` 为隐形水印节点配置 Real-ESRGAN 超分放大模块。
    基于 onnxruntime 运行 LaMa模型（仓库地址：https://huggingface.co/Carve/LaMa-ONNX/tree/main，Apache-2.0 协议）。
    仅 CPU 运算，适配各类分辨率，
    优点：纹理修复画质最优。（推荐使用lama_fp32.onnx模型）；
    但模型体积约 200 MB，峰值内存占用约 4.7 GB，低配设备难以承载。
    建议本地模型库预先下载lama.onnx、lama_fp32.onnx 模型文件并存储至 .\ComfyUI\models\lama 目录下，
    本程序不支持通过 huggingface_hub 下载并缓存。
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false, reportConstantRedefinition=false, reportUnnecessaryComparison=false
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

Backend = Literal["cv2", "lama", "migan"]

_LAMA_REPO = "Carve/LaMa-ONNX"
_LAMA_FILE = "lama_fp32.onnx"

_MIGAN_REPO = "andraniksargsyan/migan"
_MIGAN_FILE = "migan.onnx"

# Cached onnxruntime sessions (loading is expensive; reuse across calls).
_lama_session: object | None = None
_migan_session: object | None = None


def boxes_to_mask(
    shape: tuple[int, int],
    boxes: list[tuple[int, int, int, int]],
    dilate: int = 3,
) -> NDArray[Any]:
    """Build a uint8 mask (255 inside boxes) from ``(x, y, w, h)`` rectangles."""
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    for x, y, bw, bh in boxes:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + bw), min(h, y + bh)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    if dilate > 0 and mask.any():
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1)
        )
        mask = cv2.dilate(mask, k)
    return mask


def _padded_crop_box(
    mask: NDArray[Any], h: int, w: int, *, pad_frac: float, pad_min: int
) -> tuple[int, int, int, int] | None:
    """Bounding box of the set mask pixels, padded and clamped to the image."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    pad = max(
        pad_min,
        int(
            pad_frac
            * max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
        ),
    )
    x0, y0 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
    x1, y1 = (
        min(w, int(xs.max()) + 1 + pad),
        min(h, int(ys.max()) + 1 + pad),
    )
    return x0, y0, x1, y1


def erase_cv2(
    image_bgr: NDArray[Any],
    mask: NDArray[Any],
    *,
    method: Literal["telea", "ns"] = "telea",
    radius: int = 6,
) -> NDArray[Any]:
    """Inpaint ``mask`` with classical cv2 inpainting (CPU, no extra deps)."""
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    if image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
        bgr = cv2.inpaint(image_bgr[:, :, :3], mask, radius, flag)
        return np.dstack([bgr, image_bgr[:, :, 3]])
    return cv2.inpaint(image_bgr, mask, radius, flag)


def lama_available() -> bool:
    """True when the optional LaMa-ONNX backend can run (onnxruntime installed)."""
    try:
        from .optional_deps import module_available
    except ImportError:
        try:
            import onnxruntime

            return True
        except ImportError:
            return False
    return module_available("onnxruntime")


# ========== 修改①：_get_lama_session 彻底删除 HF Hub 下载，纯本地加载 ==========
def _get_lama_session() -> object:
    """Load (once) the big-LaMa ONNX session from local path only."""
    global _lama_session
    if _lama_session is not None:
        return _lama_session

    import onnxruntime as ort
    import os

    try:
        import folder_paths

        local_path = os.path.join(folder_paths.models_dir, "lama", _LAMA_FILE)
    except ImportError:
        local_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "models", "lama", _LAMA_FILE
        )
        local_path = os.path.abspath(local_path)

    if not os.path.exists(local_path):
        raise RuntimeError(
            f"LaMa model not found locally at {local_path}. "
            f"Please download the model manually and place it at this path."
        )

    logger.info("Loading local LaMa-ONNX model: %s", local_path)
    _lama_session = ort.InferenceSession(
        local_path, providers=["CPUExecutionProvider"]
    )
    return _lama_session


# ========== 官方 erase_lama（100% 还原，只改 mask_in 注释说明）==========
def erase_lama(image_bgr: NDArray[Any], mask: NDArray[Any]) -> NDArray[Any]:
    """Inpaint ``mask`` with big-LaMa via onnxruntime (CPU)."""
    if image_bgr.ndim == 2:
        bgr = erase_lama(cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR), mask)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
        bgr = erase_lama(np.ascontiguousarray(image_bgr[:, :, :3]), mask)
        return np.dstack([bgr, image_bgr[:, :, 3]])

    session = _get_lama_session()
    inp = session.get_inputs()
    img_name = inp[0].name
    mask_name = inp[1].name
    dims = inp[0].shape
    size = next(
        (d for d in reversed(dims) if isinstance(d, int) and d > 1), 512
    )

    h, w = image_bgr.shape[:2]
    box = _padded_crop_box(mask, h, w, pad_frac=0.4, pad_min=16)
    if box is None:
        return image_bgr.copy()
    cx0, cy0, cx1, cy1 = box
    crop = image_bgr[cy0:cy1, cx0:cx1]
    crop_mask = mask[cy0:cy1, cx0:cx1]
    ch, cw = crop.shape[:2]

    crop_rs = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    mask_rs = cv2.resize(
        crop_mask, (size, size), interpolation=cv2.INTER_NEAREST
    )
    img_in = cv2.cvtColor(crop_rs, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_in = np.transpose(img_in, (2, 0, 1))[None]  # (1,3,size,size)

    # ===== 核心修正：官方代码 mask 就是 [0,1]，不能乘 255！=====
    mask_in = (mask_rs > 127).astype(np.float32)[
        None, None
    ]  # (1,1,size,size), 1=hole

    out = session.run(None, {img_name: img_in, mask_name: mask_in})[0]
    out = np.asarray(out)[0]  # (3,size,size)
    out = np.transpose(out, (1, 2, 0))
    if float(out.max()) <= 1.5:
        out = out * 255.0
    out = np.clip(out, 0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    out_crop = cv2.resize(out_bgr, (cw, ch), interpolation=cv2.INTER_LINEAR)
    result = image_bgr.copy()
    region = result[cy0:cy1, cx0:cx1]
    paste = crop_mask > 127
    region[paste] = out_crop[paste]
    result[cy0:cy1, cx0:cx1] = region
    return result


# ========== 新增：单输入4通道 LaMa 支持（lama.onnx）==========
def _get_lama_session_by_name(model_name: str) -> object:
    """Load LaMa ONNX session by model file name (cached)."""
    global _lama_session_dict
    if model_name in _lama_session_dict:
        return _lama_session_dict[model_name]

    import onnxruntime as ort
    import os
    import folder_paths

    model_path = os.path.join(folder_paths.models_dir, "lama", model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"LaMa model not found locally: {model_path}. "
            f"Please download the model manually and place it at this path."
        )

    logger.info("Loading LaMa-ONNX model: %s", model_path)
    session = ort.InferenceSession(
        model_path, providers=["CPUExecutionProvider"]
    )
    _lama_session_dict[model_name] = session
    return session


_lama_session_dict: dict[str, Any] = {}


def erase_lama_merged(
    image_bgr: NDArray[Any], mask: NDArray[Any], model_name: str
) -> NDArray[Any]:
    """Erase region using merged-input LaMa ONNX model (single 4-channel input)."""
    if image_bgr.ndim == 2:
        bgr = erase_lama_merged(
            cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR), mask, model_name
        )
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
        bgr = erase_lama_merged(
            np.ascontiguousarray(image_bgr[:, :, :3]), mask, model_name
        )
        return np.dstack([bgr, image_bgr[:, :, 3]])

    if mask.sum() == 0 or mask.max() == 0:
        return image_bgr.copy()

    h, w = image_bgr.shape[:2]
    box = _padded_crop_box(mask, h, w, pad_frac=0.4, pad_min=16)
    if box is None:
        return image_bgr.copy()
    cx0, cy0, cx1, cy1 = box
    crop = image_bgr[cy0:cy1, cx0:cx1]
    crop_mask = mask[cy0:cy1, cx0:cx1]
    ch, cw = crop.shape[:2]

    session = _get_lama_session_by_name(model_name)
    inp = session.get_inputs()
    if not inp:
        logger.error(f"模型 {model_name} 没有定义有效的输入节点")
        return image_bgr.copy()

    input_shape = inp[0].shape
    target_h, target_w = 512, 512
    try:
        if len(input_shape) >= 4:
            h_dim = input_shape[2]
            w_dim = input_shape[3]
            if isinstance(h_dim, int) and isinstance(w_dim, int):
                target_h, target_w = h_dim, w_dim
            else:
                dims = [d for d in input_shape if isinstance(d, int) and d > 10]
                if len(dims) >= 2:
                    target_h, target_w = dims[-2], dims[-1]
                elif len(dims) == 1:
                    target_h = target_w = dims[0]
    except Exception as e:
        logger.warning(f"动态探测模型尺寸失败: {e}，使用默认值 512")

    crop_rs = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
    mask_rs = cv2.resize(
        crop_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST
    )

    # 单输入4通道官方逻辑：预挖空 + concat + 融合
    img_rgb = cv2.cvtColor(crop_rs, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    m = (mask_rs > 127).astype(np.float32)
    masked_img = img_rgb * (1.0 - m[..., None])
    masked_img = np.transpose(masked_img, (2, 0, 1))
    m_chw = m[None, ...]
    input_feats = np.concatenate([masked_img, m_chw], axis=0)
    input_feats = np.expand_dims(input_feats, axis=0)

    logger.info(
        f"[RAIW] 单输入 feed shape: {input_feats.shape}, mask均值={m.mean():.4f}"
    )
    out = session.run(None, {inp[0].name: input_feats})[0]

    # 后处理融合：只在 hole 区域用模型输出，其余保留原图
    out = out[0]
    img_chw = np.transpose(img_rgb, (2, 0, 1))
    recover = m_chw * out + (1.0 - m_chw) * img_chw
    recover = np.transpose(recover, (1, 2, 0))
    recover = np.clip(recover * 255.0, 0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(recover, cv2.COLOR_RGB2BGR)

    out_crop = cv2.resize(out_bgr, (cw, ch), interpolation=cv2.INTER_LINEAR)
    result = image_bgr.copy()
    region = result[cy0:cy1, cx0:cx1]
    paste = crop_mask > 127
    region[paste] = out_crop[paste]
    result[cy0:cy1, cx0:cx1] = region
    return result


def migan_available() -> bool:
    """True when the optional MI-GAN backend can run (onnxruntime installed)."""
    try:
        from .optional_deps import module_available
    except ImportError:
        try:
            import onnxruntime

            return True
        except ImportError:
            return False
    return module_available("onnxruntime")


# ========== 修改②：_get_migan_session 彻底删除 HF Hub 下载，纯本地加载 ==========
def _get_migan_session() -> object:
    """Load (once) the MI-GAN ONNX session from local path only."""
    global _migan_session
    if _migan_session is not None:
        return _migan_session

    import onnxruntime as ort
    import os

    try:
        import folder_paths

        local_path = os.path.join(folder_paths.models_dir, "migan", _MIGAN_FILE)
    except ImportError:
        local_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "models",
            "migan",
            _MIGAN_FILE,
        )
        local_path = os.path.abspath(local_path)

    if not os.path.exists(local_path):
        raise RuntimeError(
            f"MI-GAN model not found locally at {local_path}. "
            f"Please download the model manually and place it at this path."
        )

    logger.info("Loading local MI-GAN ONNX model: %s", local_path)
    _migan_session = ort.InferenceSession(
        local_path, providers=["CPUExecutionProvider"]
    )
    return _migan_session


def erase_migan(image_bgr: NDArray[Any], mask: NDArray[Any]) -> NDArray[Any]:
    """Inpaint ``mask`` (255 = erase) with MI-GAN via onnxruntime (CPU)."""
    if image_bgr.ndim == 2:
        bgr = erase_migan(cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR), mask)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
        bgr = erase_migan(np.ascontiguousarray(image_bgr[:, :, :3]), mask)
        return np.dstack([bgr, image_bgr[:, :, 3]])

    h, w = image_bgr.shape[:2]
    box = _padded_crop_box(mask, h, w, pad_frac=2.0, pad_min=256)
    if box is None:
        return image_bgr.copy()
    cx0, cy0, cx1, cy1 = box
    crop = np.ascontiguousarray(image_bgr[cy0:cy1, cx0:cx1])
    crop_mask = mask[cy0:cy1, cx0:cx1]
    ch, cw = crop.shape[:2]

    session = _get_migan_session()
    inp = session.get_inputs()
    img_name, mask_name = inp[0].name, inp[1].name

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(rgb, (2, 0, 1))[None].astype(np.uint8)
    known = (crop_mask <= 127).astype(np.uint8) * 255
    mask_in = known[None, None]

    out = session.run(None, {img_name: img_in, mask_name: mask_in})[0]
    res = np.transpose(np.asarray(out)[0], (1, 2, 0)).astype(np.uint8)
    if res.shape[:2] != (ch, cw):
        res = cv2.resize(res, (cw, ch), interpolation=cv2.INTER_LINEAR)
    out_bgr = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

    result = image_bgr.copy()
    region = result[cy0:cy1, cx0:cx1]
    hole = crop_mask > 127
    region[hole] = out_bgr[hole]
    result[cy0:cy1, cx0:cx1] = region
    return result


def erase(
    image_bgr: NDArray[Any],
    *,
    boxes: list[tuple[int, int, int, int]] | None = None,
    mask: NDArray[Any] | None = None,
    backend: Backend = "cv2",
    dilate: int = 3,
    cv2_method: Literal["telea", "ns"] = "telea",
    cv2_radius: int = 6,
    model_name: str | None = None,
) -> NDArray[Any]:
    """Erase the given boxes (or mask) via the chosen inpainting backend."""
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    if mask is None:
        if not boxes:
            return image_bgr.copy()
        mask = boxes_to_mask(image_bgr.shape[:2], boxes, dilate=dilate)
    if not mask.any():
        return image_bgr.copy()

    if backend == "migan":
        if not migan_available():
            raise RuntimeError("MI-GAN backend requires onnxruntime.")
        return erase_migan(image_bgr, mask)
    if backend == "lama":
        if not lama_available():
            raise RuntimeError("LaMa backend requires onnxruntime.")
        if model_name == "lama.onnx":
            return erase_lama_merged(image_bgr, mask, model_name)
        return erase_lama(image_bgr, mask)
    return erase_cv2(image_bgr, mask, method=cv2_method, radius=cv2_radius)