# Comfyui-Erase-Watermarks

**ComfyUI-Erase-Watermarks 是一个 ComfyUI 自定义节点包，用于移除 AI 生成图像中的可见水印与不可见水印（SynthID）。基于 remove-ai-watermarks 核心算法进行了深度重构与功能增强，重写了完整的 ComfyUI 集成层，致力于提供更顺畅、更可控的 AI 水印擦除体验。**

[![GitHub license](https://img.shields.io/github/license/yourusername/Comfyui-Erase-Watermarks)](LICENSE)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-blue)](https://github.com/comfyanonymous/ComfyUI)

---

## 📖 目录

- [功能概览](#-功能概览)
- [亮点](#-亮点)
- [版本更新日志](#-版本更新日志)
- [安装说明](#-安装说明)
- [依赖环境](#-依赖环境)
- [模型下载与存放](#-模型下载与存放)
- [节点说明](#-节点说明)
- [工作流程示例](#-工作流程示例)
- [翻译支持](#-翻译支持)
- [推荐一起安装的扩展](#-推荐一起安装的扩展)
- [常见问题](#-常见问题)
- [版权与许可证](#-版权与许可证)
- [致谢](#-致谢)

---

## 🎯 功能概览

**ComfyUI-Erase-Watermarks** 基于 [`Comfyui-Erase-Watermarks`](https://github.com/wiltodelta/ComfyUI-Comfyui-Erase-Watermarks) 核心算法进行了深度重构与功能增强，重写了完整的 ComfyUI 集成层，致力于提供更顺畅、更可控的 AI 水印擦除体验。

**相比原库，本次重构在用户体验方面做了显著优化：**

- **更灵活的流程编排**：扩展了工作流节点的输入接口，支持与更多自定义节点无缝对接，适配复杂场景。
- **更广泛的模型兼容**：引入 LaMa ONNX 模型加载器，支持加载 lama 多种格式的擦除模型，让用户的选择不再受限。
- **更直观的操作反馈**：UI 进度条与控制台日志实时同步更新，让每一步处理状态清晰可见，避免误判和等待焦虑。

| 节点名称                       | 功能描述                                                     |
| :----------------------------- | :----------------------------------------------------------- |
| **移除可见水印 (RAIW)**        | 自动检测并移除图像中的可见 AI 水印（支持 Gemini、豆包、即梦、三星等）。 |
| **检测可见水印 (RAIW)**        | 检测图像中是否存在已知类型的可见 AI 水印，并返回置信度。     |
| **区域擦除可见水印 (RAIW)**    | 通过用户提供的 MASK（遮罩）精确擦除指定区域的水印或任意内容，支持 OpenCV 和 LaMa 两种后端。 |
| **移除不可见水印 (RAIW)**      | 基于 SDXL 扩散模型，移除 SynthID 等像素级隐形 AI 水印。      |
| **加载 LaMa ONNX 模型 (新增)** | 从 `ComfyUI/models/lama/` 目录加载 `.onnx` 格式的 LaMa 模型，并输出模型对象供其他节点使用。 |

---

## ✨ **亮点：**

- **深度重构**：基于 `Comfyui-Erase-Watermarks` 核心算法，重写完整 ComfyUI 集成层。
- **新增 LaMa ONNX 模型加载器**：支持 ONNX 格式模型加载，提升模型兼容性与 ComfyUI 工作流可扩展性。
- **扩展节点输入接口**：提供更灵活的连接能力，适配复杂工作流。
- **进度反馈同步**：UI 进度条与控制台日志实时同步，操作反馈更直观。

------

## 📦 版本更新日志

### v1.1.0 (2026-07-23)

**新增功能：**
- ✨ **新增「加载 LaMa ONNX 模型」节点**：支持从 `models/lama` 目录下拉选择 `.onnx` 格式的 LaMa 模型，输出 `LAMA_MODEL` 类型对象，可与「区域擦除可见水印」节点的 `lama_model` 端口直接连线。
- ✨ **「遮罩擦除(支持任意元素) 」节点新增 `lama_model` 输入端口**：支持外部传入 ONNX 模型对象，实现模块化工作流设计。
- ✨ **「移除不可见水印」节点新增 SDXL/CLIP/VAE/ControNet模型的输入接口**：支持外部传入模型对象，实现模块化工作流设计。

**优化与修复：**

- 🔧 优化「移除不可见水印」节点：增加采样器（Sampler）和调度器（Scheduler）下拉菜单，支持从 ComfyUI 核心动态获取完整列表。
- 🔧 优化进度条同步：修复 UI 进度条与后端采样进度不同步的问题。
- 🔧 增强错误提示：所有异常信息均附带解决建议，提升用户体验。
- 🔧 离线模型加载优化：通过 Monkey-Patching 方式重写 `huggingface_hub.hf_hub_download`，优先从 ComfyUI 本地 `models` 目录加载模型，彻底解决 `LocalEntryNotFoundError` 问题。

**性能优化：**
- ⚡ 使用 FP16 半精度加速推理（默认开启）。
- ⚡ 移除 `torch.cuda.synchronize()` 同步调用，充分利用 CUDA 异步执行。
- ⚡ `hf_hub_download` 拦截增加缓存机制，避免重复扫描 `models` 目录。

### v1.0.0 (初始版本)

- 实现「移除可见水印」「检测可见水印」「区域擦除」「移除不可见水印」四个核心节点。
- 支持 ControlNet 结构保持。
- 支持 FP16 加速。

---

## 🔧 安装说明

### 方法一：通过 ComfyUI Manager 安装（推荐）

1. 在 ComfyUI 中打开 **ComfyUI Manager**。
2. 切换到 **Install Custom Nodes** 选项卡。
3. 搜索 **`Comfyui-Erase-Watermarks`**。
4. 点击 **Install** 安装。

### 方法二：手动安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/Comfyui-Erase-Watermarks.git
cd Comfyui-Erase-Watermarks
```

### 方法三：下载 ZIP

1. 访问本仓库页面，点击 **Code → Download ZIP**。
2. 解压到 `ComfyUI/custom_nodes/` 目录下，确保文件夹名称为 `Comfyui-Erase-Watermarks`。

安装完成后，**重启 ComfyUI** 即可在节点菜单的 `Comfyui-Erase-Watermarks` 分类下找到所有节点。

---

## 📋 依赖环境

### 基础依赖（必需）

| 依赖包          | 版本要求        | 说明                |
| :-------------- | :-------------- | :------------------ |
| Python          | >= 3.10         | 推荐 3.12.x         |
| PyTorch         | >= 2.0          | 与 ComfyUI 环境一致 |
| torchvision     | 与 PyTorch 匹配 | —                   |
| numpy           | >= 1.24         | 推荐 1.26.4         |
| Pillow          | >= 10.0         | —                   |
| huggingface_hub | >= 0.20         | 用于模型下载        |

### 可选依赖（按需安装）

| 依赖包                            | 用途                                     | 安装命令                                      |
| :-------------------------------- | :--------------------------------------- | :-------------------------------------------- |
| **onnxruntime-gpu**               | LaMa ONNX 模型 GPU 推理（推荐）          | `pip install onnxruntime-gpu`                 |
| **onnxruntime**                   | LaMa ONNX 模型 CPU 推理                  | `pip install onnxruntime`                     |
| **kornia**                        | GPU 加速 Canny 边缘检测（ControlNet 用） | `pip install kornia`                          |
| **Comfyui-Erase-Watermarks[gpu]** | 不可见水印移除的扩散模型依赖             | `pip install "Comfyui-Erase-Watermarks[gpu]"` |

### 一键安装所有依赖

```bash
pip install onnxruntime-gpu kornia "Comfyui-Erase-Watermarks[gpu]"
```

> **注意**：如果使用 CPU 或 CUDA 版本不兼容，请将 `onnxruntime-gpu` 替换为 `onnxruntime`。

---

## 📥 模型下载与存放

### 1. LaMa 模型（用于「区域擦除可见水印」节点）

LaMa 的 ONNX 模型由社区从原始 PyTorch 模型移植而来。

**推荐下载地址：**

- Hugging Face: [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX)
- 推荐下载 **`lama_fp32.onnx`**（FP32 版本，兼容性最好）

**存放目录：**
```
ComfyUI/models/lama/lama_fp32.onnx
```

### 2. SDXL 模型（用于「移除不可见水印」节点）

| 模型类型               | 推荐模型                                      | 下载源                                                       |
| :--------------------- | :-------------------------------------------- | :----------------------------------------------------------- |
| **SDXL Base**          | `stable-diffusion-xl-base-1.0`                | Hugging Face / ModelScope                                    |
| **SDXL VAE**           | `sdxl_vae_fp16_fix.safetensors`               | 推荐使用修复版，避免 fp16 精度下产生 NaN                     |
| **ControlNet (Canny)** | `xinsir/controlnet-canny-sdxl-1.0`（V1 版本） | [Hugging Face 仓库](https://huggingface.co/xinsir/controlnet-canny-sdxl-1.0) |

### 📂 ControlNet (Canny) 文件重命名建议

为避免 `diffusion_pytorch_model.safetensors` 在本地模型库中重名，建议按以下规则重命名：

| 原始文件名                               | 建议重命名为                                            |
| :--------------------------------------- | :------------------------------------------------------ |
| `diffusion_pytorch_model.safetensors`    | `xinsir-controlnet-canny-sdxl-v1.safetensors`           |
| `diffusion_pytorch_model_V2.safetensors` | `xinsir-controlnet-canny-sdxl-v2.safetensors`（如备用） |

### 📁 存放目录

| 模型            | 存放路径                                                    |
| :-------------- | :---------------------------------------------------------- |
| SDXL Base       | `ComfyUI/models/checkpoints/stable-diffusion-xl-base-1.0/`  |
| SDXL VAE        | `ComfyUI/models/vae/sdxl-vae/sdxl_vae_fp16_fix.safetensors` |
| ControlNet (V1) | `ComfyUI/models/controlnet/xinsir/controlnet-canny-sdxl/`   |

### 📌 ControlNet 版本选择说明

| 版本             | 对应文件名                               | 实测表现                                                | 推荐场景                         |
| :--------------- | :--------------------------------------- | :------------------------------------------------------ | :------------------------------- |
| **V1（推荐）**   | `diffusion_pytorch_model.safetensors`    | 严格跟随 Canny 边缘，**精准锁定原图结构**，人物特征稳定 | ✅ **去水印场景推荐**             |
| **V2（不推荐）** | `diffusion_pytorch_model_V2.safetensors` | 通用线条控制，**实测易导致人物特征变化**                | ⚠️ 创意生成场景，不推荐用于去水印 |

### 💡 总结

> ✅ **去水印场景请下载 `diffusion_pytorch_model.safetensors`（V1），实测人物特征更稳定；V2 版本易导致面部变化，不推荐。**
>
> 📁 **建议重命名**模型文件，方便识别与管理。

### 3. 可见水印检测模型

首次使用 ComfyUI-Erase-Watermarks 扩展时，建议手动下载所需模型，以避免网络问题影响使用。如遇 Hugging Face 连接困难，可从国内镜像站 [ModelScope](https://modelscope.cn/) 搜索并下载对应模型文件，放置于 ComfyUI 的模型目录中：

```python
modelscope download --model jingqingdai/watermark-remover-lama
modelscope download --model stabilityai/stable-diffusion-xl-base-1.0 sd_xl_base_1.0.safetensors --local_dir ./dir
modelscope download --model AI-ModelScope/sdxl-vae-fp16-fix sdxl.vae.safetensors --local_dir ./dir
```

（ 本项目已配置为**本地优先加载**：默认从 ComfyUI 模型库读取，不强制联网，不依赖 Hugging Face 缓存，**真正离线可用**。）

---

## 🧩 节点说明

### 1. 移除可见水印 (RAIW)

自动检测并移除图像中的可见 AI 水印。

| 参数      | 类型    | 默认值 | 说明                                            |
| :-------- | :------ | :----- | :---------------------------------------------- |
| `image`   | IMAGE   | —      | 输入图像                                        |
| `mark`    | COMBO   | auto   | 水印类型（auto / doubao / gemini / samsung 等） |
| `inpaint` | BOOLEAN | True   | 是否使用修复（Inpainting）                      |

**输出：** `(IMAGE, STRING)` — 处理后的图像 + 处理信息

---

### 2. 检测可见水印 (RAIW)

检测图像中是否存在已知类型的可见水印。

| 参数    | 类型  | 默认值 | 说明     |
| :------ | :---- | :----- | :------- |
| `image` | IMAGE | —      | 输入图像 |

**输出：** `(STRING, BOOLEAN, FLOAT, STRING)` — 检测报告、是否检测到、置信度、水印标识

---

### 3. 区域擦除可见水印 (RAIW)

通过用户提供的 MASK 精确擦除指定区域。

| 参数         | 类型       | 默认值 | 说明                             |
| :----------- | :--------- | :----- | :------------------------------- |
| `image`      | IMAGE      | —      | 输入图像                         |
| `mask`       | MASK       | —      | 遮罩（白色区域为擦除目标）       |
| `backend`    | COMBO      | lama   | 修复后端（cv2 / lama）           |
| `遮罩扩展`   | INT        | 3      | 遮罩边缘向外扩展的像素数         |
| `cv2_method` | COMBO      | telea  | OpenCV 修复算法（telea / ns）    |
| `cv2_radius` | INT        | 6      | OpenCV 修复半径                  |
| `lama_model` | LAMA_MODEL | 可选   | 外部传入的 LaMa 模型对象（新增） |

**输出：** `(IMAGE,)`

---

### 4. 移除不可见水印 (RAIW)

基于 SDXL 扩散模型移除 SynthID 等隐形水印。

| 参数               | 类型    | 默认值     | 说明                                      |
| :----------------- | :------ | :--------- | :---------------------------------------- |
| `image`            | IMAGE   | —          | 输入图像                                  |
| `sdxl_model`       | MODEL   | —          | SDXL 基础模型                             |
| `clip`             | CLIP    | —          | CLIP 模型                                 |
| `vae`              | VAE     | —          | VAE 模型                                  |
| `pipeline`         | COMBO   | controlnet | 管线类型（controlnet / sdxl）             |
| `strength`         | FLOAT   | 0.1        | 重绘强度（0.04~0.15 保脸最优）            |
| `steps`            | INT     | 30         | 采样步数                                  |
| `guidance_scale`   | FLOAT   | 7.5        | CFG 引导比例                              |
| `scheduler_name`   | COMBO   | euler      | 采样器名称（自动从 ComfyUI 获取完整列表） |
| `scheduler`        | COMBO   | simple     | 调度器（自动从 ComfyUI 获取完整列表）     |
| `seed`             | INT     | 0          | 随机种子                                  |
| `device`           | COMBO   | auto       | 运行设备（auto / cuda / mps / cpu）       |
| `controlnet_scale` | FLOAT   | 1.0        | ControlNet 控制权重                       |
| `humanize`         | FLOAT   | 0.0        | 人性化微调                                |
| `unsharp`          | FLOAT   | 0.0        | 锐化强度                                  |
| `max_resolution`   | INT     | 0          | 最大分辨率限制                            |
| `min_resolution`   | INT     | 1024       | 最小分辨率限制                            |
| `adaptive_polish`  | BOOLEAN | True       | 自适应抛光                                |
| `upscaler`         | COMBO   | lanczos    | 放大算法                                  |

**输出：** `(IMAGE,)`

---

### 5. 加载 LaMa ONNX 模型 (新增) ⭐

从 `ComfyUI/models/lama/` 目录加载 `.onnx` 格式的 LaMa 模型。

| 参数         | 类型  | 默认值 | 说明                                            |
| :----------- | :---- | :----- | :---------------------------------------------- |
| `model_name` | COMBO | 自动   | 从 `models/lama/` 目录自动扫描所有 `.onnx` 文件 |

**输出：** `(LAMA_MODEL,)` — 可直接连接到「区域擦除可见水印」的 `lama_model` 端口。

---

## 🔄 工作流程示例

### 示例 1：使用 LaMa ONNX 模型进行图像遮罩区域擦除（支持任意可见元素擦除）

```
[原始图像] ────────────────→ [区域擦除可见水印 (RAIW)] → [输出]
[MASK 遮罩] ──────────────→          ↑
                                    │
[加载 LaMa ONNX 模型] ──→ [lama_model]
```

![Example Workflow 01](https://raw.githubusercontent.com/chinaxunmei/Comfyui-Erase-Watermarks/main/assets/Example-Workflow-01.png)



> ##### **说明**：通过「加载 LaMa ONNX 模型」节点加载 `lama_fp32.onnx`，输出 `LAMA_MODEL` 对象连接到「区域擦除可见水印」的 `lama_model` 端口，实现 ONNX 推理加速。

### 示例 2：自动检测AI可见水印（ERAW）

![Example Workflow 02](https://raw.githubusercontent.com/chinaxunmei/Comfyui-Erase-Watermarks/main/assets/Example-Workflow-02.png)

### 示例 3：自动移除AI可见水印（ERAW）

![Example Workflow 03](https://raw.githubusercontent.com/chinaxunmei/Comfyui-Erase-Watermarks/main/assets/Example-Workflow-03.png)



### 示例 4：移除不可见水印（ERAW）

![Example Workflow 04](https://raw.githubusercontent.com/chinaxunmei/Comfyui-Erase-Watermarks/main/assets/Example-Workflow-04.png)

> ### 注意：建议 64 倍数尺寸自动对齐

在「移除不可见水印」节点前，建议添加 **「缩放图像（长边）」节点** + **「数学表达式」节点**，自动将图像长边对齐到 64 的倍数（满足 VAE 编码要求）：

**数学表达式：**

```
floor(max(a, b) / 64) * 64
```

### 示例5 - 高级组合工作流：移除不可见水印 + 移除可见水印（推荐串联顺序）

```
[原始图像] → [移除不可见水印 (RAIW)] → [区域擦除可见水印 (RAIW)] → [输出]
                                        ↑
                                   [MASK 遮罩]
```

> **说明**：先处理不可见水印（全局重绘），再通过遮罩擦除可见水印（局部修复），既避免扩散模型放大修复痕迹，又保证遮罩区域已去除隐形水印。

---

## 🌐 翻译支持

#### 本节点包支持通过 **ComfyUI-DD-Translation** 插件进行界面汉化。

### 翻译文件放置在自定义扩展ComfyUI-DD-Translation中

| ComfyUI-DD-Translation 子目录 | 用途         | 文件示例                                |
| :---------------------------- | :----------- | :-------------------------------------- |
| `zh-CN/Categories/`           | 节点标题翻译 | `ComfyUI-Comfyui-Erase-Watermarks.json` |
| `zh-CN/Nodes/`                | 节点参数翻译 | `ComfyUI-Comfyui-Erase-Watermarks.json` |

### 翻译 JSON 示例

**Categories 目录（节点标题）：**

```json
{
  "Remove Visible Watermark (RAIW)": "移除可见水印 (RAIW)",
  "Detect Visible Watermark (RAIW)": "检测可见水印 (RAIW)",
  "Erase Region (RAIW)": "遮罩擦除(支持任意元素) (RAIW)",
  "Remove Invisible Watermark / SynthID (RAIW)": "移除不可见水印 (RAIW)",
  "Load LaMa ONNX Model": "加载 LaMa ONNX 模型"
}
```

**Nodes 目录（节点参数）：**
```json
{
  "LoadLaMaONNXModel": {
    "title": "加载 LaMa ONNX 模型",
    "widgets": {
      "model_name": "模型名称"
    }
  }
}
```

> **注意**：`NODE_DISPLAY_NAME_MAPPINGS` 的优先级高于翻译插件，如果希望翻译生效，请勿在源码中硬编码中文标题。

---

## 🔌 推荐一起安装的扩展

| 扩展名称                                                     | 用途                               | 安装方式 |
| :----------------------------------------------------------- | :--------------------------------- | :------- |
| **[ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)** | 插件管理与安装                     | 必装     |
| **[ComfyUI-DD-Translation](https://github.com/Dontdrunk/ComfyUI-DD-Translation)** | 界面汉化                           | 可选     |
| **[ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts)** | 数学表达式、图像尺寸获取等辅助工具 | 推荐     |

---

## ❓ 常见问题

### Q1: 「移除不可见水印」节点报错 `VAE 编码失败，请确认输入尺寸为 64 倍数`

**解决方案**：在节点前添加「缩放图像（长边）」节点，将长边设为 64 的倍数（如 1024、2048）。

---

### Q2: 「移除可见水印」节点报错 `LocalEntryNotFoundError`

**原因**：可见水印检测模型未下载，且网络被离线环境变量阻断。

**解决方案**：检查 `nodes.py` 顶部的 `os.environ["HF_HUB_OFFLINE"]` 是否被注释。首次运行需联网下载模型。

---

### Q3: 「加载 LaMa ONNX 模型」下拉列表为空

**原因**：`ComfyUI/models/lama/` 目录下没有 `.onnx` 文件。

**解决方案**：
1. 下载 `lama_fp32.onnx` 放入 `ComfyUI/models/lama/`。
2. 重启 ComfyUI。

---

### Q4: ONNX 推理报错 `InvalidArgument` 或维度不匹配

**原因**：ONNX 模型可能未设置动态维度（dynamic_axes），不支持不同尺寸输入。

**解决方案**：使用 `onnx-simplifier` 修复：
```bash
onnxsim lama_fp32.onnx lama_fp32_sim.onnx --dynamic-input-shape
```

---

### Q5: GPU 加速不生效

**检查步骤**：
1. 确认已安装 `onnxruntime-gpu` 而非 `onnxruntime`。
2. 检查 CUDA 版本是否与 ONNX Runtime 兼容。
3. 查看 ComfyUI 启动日志中 `session.get_providers()` 的输出。

---

## 📄 版权与许可证

本项目基于以下开源项目构建：

- **[Comfyui-Erase-Watermarks](https://github.com/Sanster/Comfyui-Erase-Watermarks)** — 核心水印移除算法库
- **[LaMa (advimman/lama)](https://github.com/advimman/lama)** — 图像修复模型
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** — 节点式 UI 框架

本项目采用 **MIT License** 开源许可证。详细信息请参阅 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- 感谢 [wiltodelta](https://github.com/wiltodelta) 开发的 `remove-ai-watermarks` 库。
- 感谢 [advimman](https://github.com/advimman) 团队开源的 LaMa 模型。
- 感谢 [Carve-Photos](https://huggingface.co/Carve) 提供的 ONNX 模型移植。
- 感谢 ComfyUI 社区所有贡献者和用户。

---

## 📬 反馈与贡献

- **Issue 提交**：https://github.com/chinaxunmei/Comfyui-Erase-Watermarks/issues
- **Pull Request**：欢迎提交 PR 改进功能或修复 Bug。

---

**⭐ 动动手指点一下 Star，就是对我最大的支持——不会有人知道，但我会记得。**