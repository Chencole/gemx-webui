"""Lightweight bridge to GEM-X; importing this module never loads a model."""
from __future__ import annotations

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
# Sizes of the already installed official files; no models are loaded here.
MODEL_SIZES = {
    "repo/inputs/checkpoints/sam-3d-body-dinov3/sam3d_body.ckpt": 2109129346,
    "repo/inputs/checkpoints/sam-3d-body-dinov3/model_config.yaml": 1488,
    "repo/inputs/mhr_data/mhr_model.pt": 696110248,
    "repo/inputs/onnx/vitpose.onnx": 4279801,
    "repo/inputs/onnx/vitpose.onnx.data": 3389390848,
    "repo/inputs/onnx/gem_denoiser.onnx": 1508810,
    "repo/inputs/onnx/gem_denoiser.onnx.data": 199114752,
    "repo/inputs/onnx/gem_denoiser_no_imgfeat.onnx": 187936462,
    "repo/inputs/onnx/gem_denoiser_no_imgfeat.onnx.data": 1156259576,
    "repo/inputs/soma_assets/SOMA_neutral.npz": 26608121,
    "repo/inputs/soma_assets/correctives_model.pt": 70549157,
    "repo/inputs/soma_assets/MHR/mhr_model_lod6.pt": 2080751,
    "repo/inputs/soma_assets/MHR/base_body_lod6.obj": 39006,
    "repo/inputs/soma_assets/MHR/SOMA_wrap_lod1.obj": 1816801,
    "cache/torch/hub/checkpoints/yolox_x_8xb8-300e_humanart-a39d44ed.onnx": 396283264,
}
LIMITATIONS = [
    "直接调用现有 GEM-X CPU 版本；尚未进行完整视频推理验证。",
    "仅处理主要人物，使用静止相机假设；遮挡、模糊和很小的手部会影响结果。",
    "输出 SOMA 77 关节动作参数，不恢复剑等物体，也不恢复面部表情。",
    "当前是 ONNX regression 模式，不支持 seed 或 steps。",
    "原版 MP4 可能无法在浏览器播放，可下载后使用本地播放器。",
]


def backend_root() -> Path:
    configured = os.environ.get("MOTION_BACKEND_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (APP_ROOT.parent / "GEM-X").resolve()


def backend_python(root: Path) -> Path:
    # Preserve a venv's executable symlink path: resolving it would bypass venv.
    # A virtual environment copied from another OS is not executable here.
    return root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _missing(root: Path) -> list[str]:
    missing = []
    if not backend_python(root).is_file():
        missing.append("isolated Python environment")
    sources = ["cpu/run_cpu.py", "cpu/runtime_env.py", "cpu/sam3d_cpu.py", "cpu/skeleton_cpu.py", "repo/gem/network/stats_compose.py", "repo/third_party/sam-3d-body/sam_3d_body/build_models.py", "repo/third_party/soma/soma/soma.py", "repo/third_party/dinov3/hubconf.py"]
    missing.extend(path for path in sources if not (root / path).is_file())
    for relative, size in MODEL_SIZES.items():
        path = root / relative
        if not path.is_file() or path.stat().st_size != size:
            missing.append(relative)
    return missing


def inspect_backend() -> dict:
    """File/size readiness only; no subprocess, download, imports or inference."""
    root = backend_root()
    missing = _missing(root)
    message = "已找到现有 GEM-X；完整视频推理测试尚未执行。" if not missing else f"未找到完整的现有 GEM-X（缺少或不完整：{len(missing)} 项）。请用 MOTION_BACKEND_ROOT 指向已安装目录。"
    return {
        "ready": not missing,
        "message": message,
        "root": str(root),
        "python": str(backend_python(root)),
        "model": "NVIDIA GEM-X / SOMA 77 joints / CPU",
        "limitations": list(LIMITATIONS),
        "missing": missing,
    }


def _integer(config: dict, key: str, default: int, minimum: int, maximum: int) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{key} 必须是 {minimum}–{maximum} 之间的整数。")
    return value


def _boolean(config: dict, key: str, default: bool) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} 必须是 true 或 false。")
    return value


def build_command(job_dir: Path, config: dict, input_path: Path | None = None) -> list[str]:
    """Return argv for subprocess.Popen(..., shell=False), never a shell string."""
    if not isinstance(config, dict):
        raise ValueError("config 必须是对象。")
    unsupported = set(config) - {"threads", "start_frame", "max_frames", "render_skeleton", "use_image_features"}
    if unsupported:
        raise ValueError("GEM-X regression 不支持这些参数：" + ", ".join(sorted(unsupported)))
    threads = _integer(config, "threads", 4, 1, 64)
    start = _integer(config, "start_frame", 0, 0, 1_000_000_000)
    maximum = _integer(config, "max_frames", 120, 1, 3600)
    render = _boolean(config, "render_skeleton", True)
    image_features = _boolean(config, "use_image_features", True)
    if input_path is None:
        raise ValueError("请先上传人物视频。")
    video = Path(input_path).expanduser().resolve()
    if not video.is_file():
        raise ValueError("上传的视频文件不存在。")
    status = inspect_backend()
    if not status["ready"]:
        raise RuntimeError(status["message"])
    output = Path(job_dir).resolve() / "output"
    command = [
        status["python"], "-u", str(Path(status["root"]) / "cpu" / "run_cpu.py"),
        "--video", str(video), "--output", str(output),
        "--threads", str(threads), "--start-frame", str(start), "--max-frames", str(maximum),
    ]
    if render:
        command.append("--render-skeleton")
    if not image_features:
        command.append("--no-imgfeat")
    return command
