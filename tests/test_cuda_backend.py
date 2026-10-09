"""GPU selection and fail-fast checks require no models or GPU on the test host."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import backend


class CudaBackendTests(unittest.TestCase):
    def test_gpu_command_does_not_invoke_cpu_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "input ; literal.mp4"
            video.write_bytes(b"path fixture")
            with patch.dict(os.environ, {"MOTION_DEVICE": "cuda"}):
                status = {"ready": True, "root": str(root), "python": str(backend.backend_python(root))}
                with patch.object(backend, "inspect_backend", return_value=status):
                    command = backend.build_command(root / "job", {}, video)
            self.assertEqual(command[2], str(APP / "runtime" / "run_gpu.py"))
            self.assertIn(str(APP / "runtime" / ".venv"), command[0])
            self.assertIn(str(video.resolve()), command)
            self.assertNotIn("run_cpu.py", " ".join(command))

    def test_invalid_device_is_not_silently_cpu(self):
        with patch.dict(os.environ, {"MOTION_DEVICE": "gpu-typo"}):
            with self.assertRaises(ValueError):
                backend.backend_device()

    def runtime(self):
        spec = importlib.util.spec_from_file_location("gemx_test_runtime", APP / "runtime" / "runtime_env.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_no_gpu_fails_before_model_loading(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, {"torch": torch, "onnxruntime": SimpleNamespace()}):
            with self.assertRaisesRegex(RuntimeError, "CPU fallback is disabled"):
                self.runtime().require_cuda()

    def test_cpu_onnx_wheel_is_rejected_even_with_torch_cuda(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        ort = SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"], preload_dlls=Mock())
        with patch.dict(sys.modules, {"torch": torch, "onnxruntime": ort}):
            with self.assertRaisesRegex(RuntimeError, "CUDAExecutionProvider is unavailable"):
                self.runtime().require_cuda()
        ort.preload_dlls.assert_called_once()

    def test_runtime_keeps_cuda_device_selection(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "MOTION_BACKEND_ROOT": directory, "CUDA_VISIBLE_DEVICES": "0"
        }):
            runtime = self.runtime()
            previous = tempfile.tempdir
            try:
                runtime.configure_runtime()
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")
            finally:
                tempfile.tempdir = previous

    def test_onnx_runtime_rejects_failed_cuda_initialization(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"MOTION_BACKEND_ROOT": directory}):
            model = Path(directory) / "model.onnx"
            model.touch()
            runtime = self.runtime()
            ort = SimpleNamespace(
                SessionOptions=SimpleNamespace,
                ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=1),
                GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=1),
                InferenceSession=lambda *args, **kwargs: SimpleNamespace(
                    get_inputs=lambda: [], get_outputs=lambda: [],
                    get_providers=lambda: ["CPUExecutionProvider"]),
            )
            spec = importlib.util.spec_from_file_location("gemx_test_runner", APP / "runtime" / "run_gpu.py")
            runner = importlib.util.module_from_spec(spec)
            previous = tempfile.tempdir
            try:
                with patch.dict(sys.modules, {"runtime_env": runtime, "onnxruntime": ort}):
                    spec.loader.exec_module(runner)
                    with self.assertRaisesRegex(RuntimeError, "refusing CPU fallback"):
                        runner.CudaOnnx(model)
            finally:
                tempfile.tempdir = previous


if __name__ == "__main__":
    unittest.main()
