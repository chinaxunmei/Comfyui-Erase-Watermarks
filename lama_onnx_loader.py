import os
import folder_paths
import onnxruntime as ort
from comfy import model_management

# ============================================================================
# 1. 自定义类型（与 RAIWEraseRegion 的 INPUT_TYPES 中一致）
# ============================================================================
class LAMA_MODEL:
    def __init__(self, session, input_names, output_names):
        self.session = session
        self.input_names = input_names
        self.output_names = output_names
        self.metadata = {
            "input_names": input_names,
            "output_names": output_names,
            "providers": session.get_providers(),
        }

    def __repr__(self):
        return f"<LAMA_MODEL providers={self.metadata['providers']}>"


# ============================================================================
# 2. 加载节点（带下拉选择）
# ============================================================================
class LoadLaMaONNXModel:
    @classmethod
    def INPUT_TYPES(cls):
        # 获取 models/lama 目录
        lama_dir = os.path.join(folder_paths.models_dir, "lama")
        if os.path.isdir(lama_dir):
            # 列出所有 .onnx 文件（只显示文件名，不带路径）
            files = [f for f in os.listdir(lama_dir) if f.endswith('.onnx')]
            if not files:
                files = ["（无 ONNX 模型）"]
        else:
            files = ["（目录不存在）"]

        return {
            "required": {
                "model_name": (files, {"default": files[0] if files and "（" not in files[0] else ""}),
            }
        }

    RETURN_TYPES = ("LAMA_MODEL",)
    RETURN_NAMES = ("lama_model",)
    FUNCTION = "load"
    CATEGORY = "remove-ai-watermarks/LaMa"

    def load(self, model_name):
        # 检查是否选择了有效模型
        if not model_name or "（" in model_name:
            raise FileNotFoundError(
                f"未选择有效的 ONNX 模型。请确保 'models/lama' 目录下存在 .onnx 文件。"
            )

        # 构建完整路径
        model_path = os.path.join(folder_paths.models_dir, "lama", model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 配置 ONNX Runtime 会话
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_cpu_mem_arena = True

        # 选择执行提供器（优先 GPU）
        if model_management.should_use_fp16():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        try:
            session = ort.InferenceSession(model_path, sess_options, providers=providers)
        except Exception as e:
            print(f"⚠️ GPU 加载失败，尝试 CPU 加载: {e}")
            session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])

        # 获取输入输出信息
        input_names = [inp.name for inp in session.get_inputs()]
        output_names = [out.name for out in session.get_outputs()]

        print(f"✅ LaMa ONNX 模型加载成功: {model_name}")
        print(f"   - 输入: {input_names}")
        print(f"   - 输出: {output_names}")
        print(f"   - 执行提供器: {session.get_providers()}")

        return (LAMA_MODEL(session, input_names, output_names),)