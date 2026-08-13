"""
ERAW InvisibleEngine 本地实现
重载 remove_ai_watermarks.invisible_engine，提供 ComfyUI 专用实现。

- 替换原有 diffusers 依赖为 ComfyUI 原生采样器
- 接入 ComfyUI 模型管理（SDXL/VAE/CLIP/ControlNet）
- 启用 FP16 半精度、异步 GPU 传输
- 本地模型优先加载，离线可用
"""
from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import torch
import numpy as np
from PIL import Image

import folder_paths
import huggingface_hub
import comfy.samplers
import comfy.utils

# 屏蔽无关提示输出
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

log = logging.getLogger(__name__)

# 采样器与调度器列表
SAMPLER_LIST = comfy.samplers.SAMPLER_NAMES
SCHEDULER_LIST = comfy.samplers.SCHEDULER_NAMES

# ==============================================================================
# 🌟 核心注入点：重写 hf_hub_download 并覆盖 remove_ai_watermarks 内部调用
# 彻底解决 huggingface_hub.errors.LocalEntryNotFoundError 离线缓存报错
# ==============================================================================
_orig_hf_hub_download = huggingface_hub.hf_hub_download

# ---------- 缓存优化：避免每次重复扫描 models 目录 ----------
_model_cache: dict[str, str] = {}
_cache_initialized = False

def _build_model_cache():
    """一次性扫描整个 ComfyUI models 目录，建立 文件名 -> 绝对路径 的映射缓存"""
    global _model_cache, _cache_initialized
    if _cache_initialized:
        return
    for root, _, files in os.walk(folder_paths.models_dir):
        for f in files:
            _model_cache[f] = os.path.join(root, f)
    _cache_initialized = True
    log.info(f"[ERAW 缓存] 已建立 models 目录文件索引，共 {len(_model_cache)} 个文件")

def _smart_local_hf_hub_download(repo_id=None, filename=None, **kwargs):
    """拦截 Hugging Face 请求，强制引导读取 ComfyUI 本地模型路径（带缓存优化）"""
    lama_dir = os.path.join(folder_paths.models_dir, "lama")

    # 1. 优先匹配 ComfyUI/models/lama/ 下的具体文件名（最快）
    if filename:
        target_path = os.path.join(lama_dir, filename)
        if os.path.exists(target_path):
            log.info(f"[ERAW 本地重定向] 成功锁定本地模型: {target_path}")
            return target_path

    # 2. 若 filename 为 ONNX 类型，直接匹配 lama 目录下的已有 .onnx 权重
    if os.path.exists(lama_dir):
        onnx_files = [f for f in os.listdir(lama_dir) if f.endswith(".onnx")]
        if onnx_files:
            target_path = os.path.join(lama_dir, onnx_files[0])
            log.info(f"[ERAW 本地重定向] 自动绑定 ONNX 引擎: {target_path}")
            return target_path

    # 3. 利用缓存进行快速检索（避免每次 os.walk）
    if filename:
        # 确保缓存已建立
        _build_model_cache()
        if filename in _model_cache:
            target_path = _model_cache[filename]
            if os.path.exists(target_path):
                log.info(f"[ERAW 本地重定向] 缓存命中: {target_path}")
                return target_path
            else:
                # 缓存中的文件已被删除，移除并重新扫描
                log.warning(f"[ERAW 缓存] 文件 {target_path} 已不存在，重新扫描 models 目录")
                _model_cache.clear()
                _cache_initialized = False
                _build_model_cache()
                if filename in _model_cache:
                    return _model_cache[filename]

    # 4. 保底：尝试离线查找 HF 本地缓存
    try:
        kw_offline = dict(kwargs)
        kw_offline["local_files_only"] = True
        return _orig_hf_hub_download(repo_id=repo_id, filename=filename, **kw_offline)
    except Exception:
        pass

    # 5. 回退至原始调用（可能会触发联网下载，但此时已无本地替代）
    return _orig_hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)

# 覆盖全局 huggingface_hub 句柄
huggingface_hub.hf_hub_download = _smart_local_hf_hub_download

def patch_imported_modules():
    for mod_name, mod in list(sys.modules.items()):
        if mod and mod_name.startswith("remove_ai_watermarks"):
            if hasattr(mod, "hf_hub_download"):
                setattr(mod, "hf_hub_download", _smart_local_hf_hub_download)

patch_imported_modules()


def is_available():
    return True

def _get_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"

class InvisibleEngineLocal:
    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        pipeline: str = "controlnet",
        hf_token: str | None = None,
        progress_callback = None,
        controlnet_conditioning_scale: float = 1.0,
        controlnet_model_path: str | None = None,
        vae_model_path: str | None = None,
        external_components: dict | None = None,
        scheduler_name: str = "euler",
        scheduler: str = "simple",
        use_fp16: bool = True,
    ):
        patch_imported_modules()
        self.pipeline = pipeline
        self.progress_callback = progress_callback
        self.scheduler_name = scheduler_name
        self.scheduler = scheduler
        self.use_fp16 = use_fp16
        try:
            self.controlnet_conditioning_scale = float(controlnet_conditioning_scale)
        except Exception:
            self.controlnet_conditioning_scale = 1.0

        self.device = device if device and device != "auto" else _get_device()
        self.torch_device = torch.device(self.device)
        
        self.external_components = external_components or {}
        self.model_patcher = self.external_components.get('sdxl_model')
        self.vae = self.external_components.get('vae_model')
        self.clip = self.external_components.get('clip_model')
        self.controlnet = self.external_components.get('controlnet_model')
        
        self.model_id = model_id
        self.controlnet_model_path = controlnet_model_path
        self.vae_model_path = vae_model_path

        self._model_dtype = torch.float16 if self.use_fp16 else torch.float32

    def _set_progress(self, msg):
        log.info(f"[ERAW] Progress: {msg}")
        if self.progress_callback:
            try:
                self.progress_callback(msg)
            except Exception:
                pass

    def preload(self):
        pass

    def _load_pipeline(self):
        pass

    def _encode_prompt(self, clip_model, prompt):
        try:
            tokens = clip_model.tokenize(prompt)
            cond, pooled = clip_model.encode_from_tokens(tokens, return_pooled=True)
            return [[cond, {"pooled_output": pooled}]]
        except Exception as e:
            log.error(f"[ERAW] 提示词编码失败: {e}")
            raise RuntimeError(
                f"CLIP 编码失败: {e}\n"
                "建议：检查 CLIP 模型是否正确连接，或尝试更换其他 CLIP 模型。"
            )

    def _vae_encode(self, vae_model, images):
        try:
            if images.dim() == 3:
                images = images.unsqueeze(0)
            images = images.to(self.torch_device, dtype=self._model_dtype, non_blocking=True)
            # 已移除 torch.cuda.synchronize()，让 GPU 异步执行
            result = vae_model.encode(images[:, :, :, :3])
            if isinstance(result, dict) and 'samples' in result:
                return result['samples']
            return result
        except Exception as e:
            log.error(f"[ERAW] VAE 编码失败: {e}")
            raise RuntimeError(
                f"VAE 编码失败: {e}\n"
                "可能原因：图像尺寸不是 64 的倍数。\n"
                "建议：在节点前添加「缩放图像（长边）」节点，并将长边设为 64 的倍数（如 1024、2048）。"
            )

    def _vae_decode(self, vae_model, latents):
        try:
            return vae_model.decode(latents)
        except Exception as e:
            log.error(f"[ERAW] VAE 解码失败: {e}")
            raise RuntimeError(
                f"VAE 解码失败: {e}\n"
                "可能原因：VAE 模型损坏或不兼容。\n"
                "建议：检查 VAE 模型是否正确加载，尝试更换另一个 VAE 模型。"
            )

    def _apply_controlnet(self, positive, negative, control_net, control_hint_tensor, strength):
        if control_net is None:
            return positive, negative
        try:
            def apply_to_conditioning(cond):
                result = []
                for t in cond:
                    d = t[1].copy()
                    if hasattr(control_net, 'set_cond_hint'):
                        c_net = control_net.copy().set_cond_hint(control_hint_tensor, strength, (0.0, 1.0))
                        d['control'] = c_net
                        d['control_apply_to_uncond'] = False
                    n = [t[0], d]
                    result.append(n)
                return result
            return apply_to_conditioning(positive), apply_to_conditioning(negative)
        except Exception as e:
            log.warning(f"[ERAW] ControlNet 注入失败: {e}")
            return positive, negative

    def _native_sample(self, model_patcher, positive, negative, latent_image, 
                       vae, strength, num_inference_steps, guidance_scale, 
                       seed=None, controlnet=None, control_image=None):
        try:
            import comfy.sample
            import comfy.model_management
            
            scheduler = self.scheduler
            sampler = self.scheduler_name
            
            device = comfy.model_management.intermediate_device()
            dtype = self._model_dtype
            noise = torch.randn(latent_image.shape, device=device, dtype=dtype)
            if seed is not None:
                generator = torch.Generator(device='cpu').manual_seed(seed)
                noise = torch.randn(latent_image.shape, generator=generator, device=device, dtype=dtype)
            
            if controlnet is not None and control_image is not None:
                positive, negative = self._apply_controlnet(
                    positive, negative, controlnet, control_image, self.controlnet_conditioning_scale
                )
            
            # 官方标准采样回调：更新 UI 进度条
            def callback(step, x0, x, total_steps):
                if hasattr(self, 'pbar') and self.pbar is not None:
                    try:
                        self.pbar.update(1)
                    except Exception:
                        pass

            samples = comfy.sample.sample(
                model_patcher, noise, num_inference_steps, guidance_scale,
                sampler, scheduler, positive, negative, latent_image,
                denoise=strength, disable_noise=False, seed=seed,
                callback=callback
            )
            return samples
        except Exception as e:
            log.error(f"[ERAW] 采样失败: {e}")
            raise RuntimeError(
                f"采样失败: {e}\n"
                "可能原因：采样器/调度器组合不兼容，或步数设置过高。\n"
                "建议：尝试切换采样器（如 euler）和调度器（如 normal），或适当减少步数（如 20）。"
            )

    def remove_watermark(
        self,
        image_path: Path,
        output_path: Path | None = None,
        strength: float | None = None,
        num_inference_steps: int = 100,
        guidance_scale: float | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> Path:
        from remove_ai_watermarks import image_io
        
        if output_path is None:
            output_path = image_path

        log.info(f"[DEBUG] remove_watermark 接收到的 strength 原始值: {strength} (type: {type(strength)})")

        self._set_progress("Loading input image")
        init_image = Image.open(image_path).convert("RGB")
        
        try:
            raw_strength = float(strength) if strength is not None else 0.1
        except Exception:
            raw_strength = 0.1
        
        if raw_strength < 0.0:
            raw_strength = 0.0
        elif raw_strength > 1.0:
            raw_strength = 1.0

        if raw_strength == 0.0:
            log.warning(f"[操作提示] Strength=0.0，节点将执行直通操作（原图输出），**不会去除任何隐式水印**。")
        elif 0.0 < raw_strength <= 0.15:
            log.info(f"[保脸最优] Strength={raw_strength:.3f}，处于极低扰动区间，人脸特征无可见变化，且能有效破坏隐式水印。")
        elif raw_strength >= 0.3:
            log.warning(f"[保脸警告] Strength={raw_strength:.3f} 较高，可能会导致人脸特征轻微改变，建议降至 0.04 ~ 0.1。")
        
        strength_val = raw_strength

        try:
            num_inference_steps = int(num_inference_steps) if num_inference_steps is not None else 30
            num_inference_steps = max(1, min(100, num_inference_steps))
        except Exception:
            num_inference_steps = 30
            
        try:
            guidance_scale = float(guidance_scale) if guidance_scale is not None else 7.5
        except Exception:
            guidance_scale = 7.5

        self._set_progress("Preprocessing image with async GPU upload")
        img_np = np.array(init_image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(
            self.torch_device, dtype=self._model_dtype, non_blocking=True
        )
        # 已移除 torch.cuda.synchronize()

        if self.vae is None or self.model_patcher is None:
            raise RuntimeError(
                "节点缺少 VAE 或 SDXL MODEL 连线。\n"
                "建议：在 ComfyUI 面板中正确接入 VAE 和 SDXL 模型节点。"
            )
        if self.clip is None:
            raise RuntimeError(
                "节点缺少 CLIP 连线。\n"
                "建议：在 ComfyUI 面板中接入 CLIP 模型。"
            )
        
        self._set_progress("Encoding prompts")
        positive_prompt = "best quality, high quality, sharp, detailed, photographic"
        negative_prompt = "blurry, lowres, deformed, distorted text, garbled text, watermark, jpeg artifacts"
        positive = self._encode_prompt(self.clip, positive_prompt)
        negative = self._encode_prompt(self.clip, negative_prompt)
        
        self._set_progress("Encoding to latent space")
        latents = self._vae_encode(self.vae, img_tensor)
        
        control_tensor = None
        if self.pipeline == "controlnet" and self.controlnet is not None:
            try:
                self._set_progress("Processing ControlNet Canny edges")
                import kornia.filters as K
                gray = img_tensor.permute(0, 3, 1, 2).mean(dim=1, keepdim=True)
                gray = gray.float()
                edges = K.canny(gray, low_threshold=0.1, high_threshold=0.2)[0]
                if self._model_dtype == torch.float16:
                    edges = edges.half()
                control_tensor = edges.expand(-1, 3, -1, -1).contiguous()
                log.info("[ERAW] Canny 已在GPU完成 (Kornia)")
            except ImportError:
                raise RuntimeError(
                    "未安装 kornia 库，无法启用 GPU 加速的 Canny 边缘检测。\n"
                    "建议：在终端执行 pip install kornia 以安装。"
                )
            except Exception as e:
                log.error(f"[ERAW] ControlNet 预处理失败: {e}")
                raise RuntimeError(
                    f"ControlNet 预处理失败: {e}\n"
                    "建议：检查 kornia 版本是否兼容，或尝试降级/升级。"
                )

        self._set_progress(f"Sampling with strength={strength_val:.3f}")
        result_latents = self._native_sample(
            self.model_patcher, positive, negative, latents, 
            self.vae, strength_val, num_inference_steps, 
            guidance_scale, seed, 
            self.controlnet, control_tensor
        )
        
        self._set_progress("Decoding from latent space")
        result_img_tensor = self._vae_decode(self.vae, result_latents)
        
        result_img_tensor = torch.clamp(result_img_tensor, 0.0, 1.0)
        img_cpu = result_img_tensor.squeeze(0).cpu().numpy()
        img_uint8 = (img_cpu * 255).astype(np.uint8)
        cleaned_img = Image.fromarray(img_uint8)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._set_progress(f"Saving to {output_path.name}...")
        
        cleaned_bgr = np.asarray(cleaned_img.convert("RGB"))[:, :, ::-1].copy()
        if not image_io.imwrite(str(output_path), cleaned_bgr):
            cleaned_img.save(output_path)
        
        return output_path

# ==============================================================================
# 🔗 保证类名兼容导入
# ==============================================================================
InvisibleEngine = InvisibleEngineLocal