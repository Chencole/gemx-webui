"""Process-local model paths and an explicit CUDA preflight."""
from pathlib import Path
import os
import tempfile

APP_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("MOTION_BACKEND_ROOT", APP_ROOT.parent / "GEM-X")).expanduser().resolve()
REPO = ROOT / "repo"


def configure_runtime():
    temporary = ROOT / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = os.environ["TMP"] = str(temporary)
    tempfile.tempdir = str(temporary)
    os.environ["TORCH_HOME"] = str(ROOT / "cache" / "torch")
    os.environ["HF_HOME"] = str(ROOT / "cache" / "huggingface")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # Preserve the user's CUDA_VISIBLE_DEVICES selection. Never hide the GPU.
    return temporary


def require_cuda():
    import torch
    import onnxruntime as ort
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable. Install the CUDA runtime dependencies; CPU fallback is disabled.")
    # Loading PyTorch first exposes its CUDA/cuDNN DLLs on Windows; ORT's
    # official preload helper also checks the process and wheel directories.
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("onnxruntime-gpu CUDAExecutionProvider is unavailable; CPU fallback is disabled.")
    return {"torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0), "onnxruntime": ort.__version__}


if __name__ == "__main__":
    import json
    print(json.dumps(require_cuda(), ensure_ascii=False, indent=2))
