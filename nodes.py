"""封装 Comfyui-erase-watermarks 库的 ComfyUI 节点。
ComfyUI 以形状为 (B, H, W, C)、取值区间 [0, 1] 的 float32 类型 RGB 格式 Torch 张量传递图像。
而 Comfyui-erase-watermarks 库基于单张图像的 BGR uint8 格式 NumPy 数组运算（OpenCV 标准），
因此每个节点会在数据接口处完成格式转换，并逐帧批量处理图像。
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
import numpy as np
import torch
import folder_paths
import comfy
import inspect
from .lama_onnx_loader import LoadLaMaONNXModel

log = logging.getLogger(__name__)

CATEGORY = "Comfyui-erase-watermarks"

_local_dir = os.path.dirname(os.path.abspath(__file__))
if _local_dir not in sys.path:
    sys.path.insert(0, _local_dir)

# ==============================================================================
# Hugging Face 离线模式配置（默认关闭）
# ==============================================================================
# 当前状态：禁止联网（1），需手动将所有模型下载至本地。
#
# 两种模式的区别：
#   • 0（默认）→ 允许联网下载缺失模型，省心但可能受网络影响。
#   • 1（启用）→ 强制完全离线，需手动确保所有模型已下载。
#
# 如需启用严格离线模式，取消下方两行注释：
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 国内镜像加速（仅本地模型缺失且未开启离线时生效）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# --- tensor <-> numpy boundary helpers -------------------------------------

def _tensor_to_bgr_list(image: "torch.Tensor") -> list[np.ndarray[Any, Any]]:
    arr = image.detach().cpu().numpy()
    frames: list[np.ndarray[Any, Any]] = []
    for i in range(arr.shape[0]):
        rgb = np.clip(arr[i] * 255.0, 0, 255).astype(np.uint8)
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        rgb = rgb[..., :3]
        frames.append(rgb[..., ::-1].copy())
    return frames

def _bgr_list_to_tensor(frames: list[np.ndarray[Any, Any]]) -> "torch.Tensor":
    tensors = []
    for bgr in frames:
        rgb = bgr[..., :3][..., ::-1].copy()
        tensors.append(torch.from_numpy(rgb.astype(np.float32) / 255.0))
    return torch.stack(tensors, dim=0)

def _mask_to_uint8_list(mask: "torch.Tensor", count: int) -> list[np.ndarray[Any, Any]]:
    arr = mask.detach().cpu().numpy()
    if arr.ndim == 2:
        arr = arr[None, ...]
    out: list[np.ndarray[Any, Any]] = []
    for i in range(count):
        m = arr[i] if i < arr.shape[0] else arr[-1]
        out.append((m > 0.5).astype(np.uint8) * 255)
    return out


# --- 节点 1：去除可见水印（完美修复版）---
class ERAWRemoveVisibleWatermark:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        from remove_ai_watermarks import watermark_registry
        marks = ["auto", *watermark_registry.mark_keys()]
        return {
            "required": {
                "image": ("IMAGE",),
                "mark": (marks, {"default": "auto"}),
            },
            "optional": {
                "inpaint": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "remove"
    CATEGORY = CATEGORY

    def remove(self, image: "torch.Tensor", mark: str, inpaint: bool = True) -> tuple[Any, str]:
        from remove_ai_watermarks import watermark_registry

        frames = _tensor_to_bgr_list(image)
        out: list[np.ndarray[Any, Any]] = []
        infos: list[str] = []

        def _safe_remove(known_instance, bgr_img, inpaint_flag=True, force_flag=True):
            """根据底层 KnownMark 的真实函数签名，安全传参"""
            sig = inspect.signature(known_instance.remove)
            kwargs = {}
            if "inpaint" in sig.parameters:
                kwargs["inpaint"] = inpaint_flag
            if "force" in sig.parameters:
                kwargs["force"] = force_flag
            
            res = known_instance.remove(bgr_img, **kwargs)
            # 兼容返回 (image, info) 元组或单独图像
            if isinstance(res, tuple):
                return res[0]
            return res

        for bgr in frames:
            if mark == "auto":
                # 1. 使用标准 API 检测当前帧的水印类型
                detections = watermark_registry.detect_marks(bgr)
                fired = [d for d in detections if getattr(d, 'detected', False)]
                
                if not fired:
                    out.append(bgr)
                    infos.append("no visible mark detected")
                    continue
                
                # 2. 找到置信度最高的已匹配水印
                best = max(fired, key=lambda d: getattr(d, 'confidence', 0.0))
                known = watermark_registry.get_mark(best.key)
                result = _safe_remove(known, bgr, inpaint_flag=inpaint, force_flag=True)
                infos.append(f"removed {best.key} (conf {best.confidence:.2f})")
            else:
                # 指定具体水印类型（如 doubao / gemini）
                known = watermark_registry.get_mark(mark)
                result = _safe_remove(known, bgr, inpaint_flag=inpaint, force_flag=True)
                infos.append(f"removed {mark} (forced)")
                
            out.append(result)

        return (_bgr_list_to_tensor(out), " | ".join(infos))


# --- 节点 2：检测可见水印 ---
class ERAWDetectVisibleWatermark:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("STRING", "BOOLEAN", "FLOAT", "STRING")
    RETURN_NAMES = ("report", "detected", "confidence", "mark")
    FUNCTION = "detect"
    CATEGORY = CATEGORY

    def detect(self, image: "torch.Tensor") -> tuple[str, bool, float, str]:
        from remove_ai_watermarks import watermark_registry
        bgr = _tensor_to_bgr_list(image)[0]
        detections = watermark_registry.detect_marks(bgr)
        lines = [f"{d.label}: {'YES' if d.detected else 'no'} ({d.confidence:.2f})" for d in detections]
        fired = [d for d in detections if d.detected]
        best = max(detections, key=lambda d: d.confidence) if detections else None
        any_detected = bool(fired)
        confidence = best.confidence if best else 0.0
        mark_key = best.key if (best and best.detected) else ""
        return ("\n".join(lines), any_detected, confidence, mark_key)


# --- 节点 3：擦除指定区域（支持外部 LAMA_MODEL 对象）---
class ERAWEraseRegion:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
            "optional": {
                "backend": (["cv2", "lama"], {"default": "lama"}),
                "dilate": ("INT", {"default": 3, "min": 0, "max": 64}),
                "cv2_method": (["telea", "ns"], {"default": "telea"}),
                "cv2_radius": ("INT", {"default": 6, "min": 1, "max": 64}),
                # 外部 LAMA 模型对象（优先使用，忽略 backend 中的 lama 回退）
                "lama_model": ("LAMA_MODEL", {"optional": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "erase"
    CATEGORY = CATEGORY

    def erase(
        self, image: "torch.Tensor", mask: "torch.Tensor", backend: str = "lama",
        dilate: int = 3, cv2_method: str = "telea", cv2_radius: int = 6,
        lama_model: Any = None,
    ) -> tuple[Any]:
        from .region_eraser import erase as erase_region

        frames = _tensor_to_bgr_list(image)
        masks = _mask_to_uint8_list(mask, len(frames))
        out = []

        # 确定实际使用的 backend 和模型对象
        if lama_model is not None:
            # 外部提供了模型对象，强制使用 lama 后端，并传递该对象
            effective_backend = "lama"
            effective_model = lama_model
            log.info("[ERAW] 使用外部 LAMA_MODEL 对象")
        else:
            # 未提供外部模型，按 backend 选择
            effective_backend = backend
            # 若选择 lama，尝试加载默认模型路径，但这里我们无法在节点内加载模型，
            # 因为 erase_region 需要 model_name（路径或对象）。若 backend 为 lama 且无外部模型，
            # 我们应传递 None，让 erase_region 内部使用默认路径（如果它支持）。
            # 但 erase_region 的 model_name 参数为 None 时可能使用内置默认。
            # 为了兼容，我们传递默认路径字符串，但路径需存在。
            effective_model = None
            if backend == "lama":
                default_path = os.path.join(folder_paths.models_dir, "lama", "lama_fp32.onnx")
                if os.path.exists(default_path):
                    effective_model = default_path
                    log.info(f"[ERAW] 使用默认模型路径: {default_path}")
                else:
                    log.warning("[ERAW] 未找到默认模型，将降级为 cv2 后端")
                    effective_backend = "cv2"
                    effective_model = None

        for bgr, m in zip(frames, masks):
            result = erase_region(
                bgr,
                mask=m,
                backend=effective_backend,
                dilate=dilate,
                cv2_method=cv2_method,
                cv2_radius=cv2_radius,
                model_name=effective_model  # 传递模型对象或路径字符串
            )
            out.append(result)

        return (_bgr_list_to_tensor(out),)


# --- 节点 4：去除隐形水印 / SynthID (完全使用原生对象，无联网依赖) ---
class ERAWRemoveInvisibleWatermark:
    """通过 SDXL 扩散模型重绘移除隐形 AI 水印（SynthID）。

    需安装该库的 GPU 机器学习扩展依赖：
        pip install "Comfyui-erase-watermarks[gpu]"

    ComfyUI 张量不携带任何文件元数据，因此适配水印生成厂商的强度参数
    默认会采用未知厂商对应的预设值。将 strength 设为大于 0 的数值即可
    手动强制覆盖该默认配置。
    """
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        # 动态检测设备
        from invisible_engine_local import _get_device, SAMPLER_LIST, SCHEDULER_LIST
        ui_default_device = _get_device()  # 返回 'cuda' / 'mps' / 'cpu'

        return {
            "required": {
                "image": ("IMAGE",),
                "sdxl_model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "pipeline": (["controlnet", "sdxl"], {"default": "controlnet"}),
                "strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),   # 优化为 0.1
                "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
            },
            "optional": {
                "controlnet_model": ("CONTROL_NET",),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "scheduler_name": (SAMPLER_LIST, {"default": "euler"}),
                "scheduler": (SCHEDULER_LIST, {"default": "simple"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": ui_default_device}),
                "controlnet_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "humanize": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "unsharp": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3.0, "step": 0.1}),
                "max_resolution": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "min_resolution": ("INT", {"default": 1024, "min": 0, "max": 8192}),
                "adaptive_polish": ("BOOLEAN", {"default": True}),
                "upscaler": (["lanczos", "esrgan"], {"default": "lanczos"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "remove"
    CATEGORY = CATEGORY

    def remove(
        self, image: "torch.Tensor", sdxl_model: Any, clip: Any, vae: Any, pipeline: str = "controlnet", strength: float = 0.0, steps: int = 30,
        controlnet_model: Any = None,
        guidance_scale: float = 7.5, 
        scheduler_name: str = "euler", scheduler: str = "normal", 
        seed: int = 0, device: str = "auto", controlnet_scale: float = 1.0,
        humanize: float = 0.0, unsharp: float = 0.0, max_resolution: int = 0, min_resolution: int = 1024,
        adaptive_polish: bool = True, upscaler: str = "lanczos",
    ) -> tuple[Any]:
        import traceback

        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

        try:
            import invisible_engine_local
            from invisible_engine_local import InvisibleEngineLocal as InvisibleEngine, is_available
        except Exception as exc:
            log.error("Failed to import InvisibleEngineLocal", exc_info=True)
            raise RuntimeError(
                f"[ERAW] 导入隐形水印本地引擎失败，真实原因: {exc}\n"
                "建议：检查 invisible_engine_local.py 文件是否存在，以及相关依赖是否完整安装。"
            ) from exc

        if not is_available():
            raise RuntimeError(
                "Diffusion 依赖不可用。\n"
                "建议：请确保已安装 GPU/ML 扩展包：pip install 'Comfyui-erase-watermarks[gpu]'"
            )

        try:
            frames = _tensor_to_bgr_list(image)
            out: list[np.ndarray[Any, Any]] = []

            # 🌟 核心修正：以采样总步数初始化原生进度条（多帧时乘以帧数）
            total_steps_ui = steps * len(frames)
            pbar = comfy.utils.ProgressBar(total_steps_ui)

            with tempfile.TemporaryDirectory(prefix="ERAW_comfy_") as tmp:
                tmp_dir = Path(tmp)

                # 直接传递 ComfyUI 对象组合，彻底接管模型流转
                external_components = {
                    "sdxl_model": sdxl_model,
                    "clip_model": clip, 
                    "vae_model": vae,
                    "controlnet_model": controlnet_model
                }

                engine = InvisibleEngine(
                    model_id=None,
                    device=None if device == "auto" else device,
                    pipeline=pipeline,
                    controlnet_conditioning_scale=controlnet_scale,
                    progress_callback=lambda msg: log.info("invisible: %s", msg),
                    controlnet_model_path=None,
                    vae_model_path=None,
                    external_components=external_components, 
                    scheduler_name=scheduler_name,   
                    scheduler=scheduler,             
                )

                # 🌟 将 comfy 的 pbar 实例直接注入引擎，供底层采样回调调用
                engine.pbar = pbar

                for idx, bgr in enumerate(frames):
                    from remove_ai_watermarks import image_io
                    src = tmp_dir / f"in_{idx}.png"
                    dst = tmp_dir / f"out_{idx}.png"
                    image_io.imwrite(str(src), bgr)
                    
                    # 发送执行指令，保留原有签名
                    result_path = engine.remove_watermark(
                        image_path=src, output_path=dst, strength=strength,
                        num_inference_steps=steps, guidance_scale=guidance_scale, seed=seed,
                        humanize=humanize, unsharp=unsharp, max_resolution=max_resolution,
                        min_resolution=min_resolution, adaptive_polish=adaptive_polish, upscaler=upscaler,
                    )
                    cleaned = image_io.imread(str(result_path))
                    out.append(cleaned)
                    
            return (_bgr_list_to_tensor(out),)
        except Exception as e:
            log.error(f"[ERAW] ERROR: {e}", exc_info=True)
            raise


# --- 节点映射注册 ---
# 新增: LoadLaMaONNXModel 节点
NODE_CLASS_MAPPINGS = {
    "ERAWRemoveVisibleWatermark": ERAWRemoveVisibleWatermark,
    "ERAWDetectVisibleWatermark": ERAWDetectVisibleWatermark,
    "ERAWEraseRegion": ERAWEraseRegion,
    "ERAWRemoveInvisibleWatermark": ERAWRemoveInvisibleWatermark,
    "LoadLaMaONNXModel": LoadLaMaONNXModel,  
}
"""
# --- 注意：源码中的映射注册 优先于翻译节点（两者只要保留其一） ---
# 注意：NODE_DISPLAY_NAME_MAPPINGS 的优先级高于翻译插件，如果希望翻译生效，请勿在源码中硬编码中文标题
# 本节点包支持通过 ComfyUI-DD-Translation 插件进行界面汉化。

如需展示节点注册对应的中文名称，建议使用以下{简体中文}显示名映射配置。

NODE_DISPLAY_NAME_MAPPINGS = {
    "ERAWRemoveVisibleWatermark": "移除可见水印 (ERAW)",
    "ERAWDetectVisibleWatermark": "检测可见水印 (ERAW)",
    "ERAWEraseRegion": "遮罩擦除(支持任意元素) (ERAW)",
    "ERAWRemoveInvisibleWatermark": "移除不可见水印 (ERAW)",
    "LoadLaMaONNXModel": "加载 LaMa ONNX 模型", 
}
"""
"""
If you wish to display the registered English names of nodes, use the following display name mapping configuration.
如需展示节点注册对应的英文名称，建议使用以下{english}显示名映射配置。

NODE_DISPLAY_NAME_MAPPINGS = {
    "ERAWRemoveVisibleWatermark": "Remove Visible Watermark (ERAW)",
    "ERAWDetectVisibleWatermark": "Detect Visible Watermark (ERAW)",
    "ERAWEraseRegion": "Mask Erase (Supports All Elements) (ERAW)",
    "ERAWRemoveInvisibleWatermark": "Remove Invisible Watermark (ERAW)",
    "LoadLaMaONNXModel": "Load LaMa ONNX Model", 
}
"""