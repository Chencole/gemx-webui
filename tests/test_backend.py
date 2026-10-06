"""Existing-backend command contracts only; no inference or downloads."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import backend


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.video = self.root / "uploaded video ; literal.mp4"
        self.video.write_bytes(b"test path only; not a media fixture")
        installed = self.root / "GEM-X"
        self.status = {"ready": True, "root": str(installed), "python": str(backend.backend_python(installed)), "message": "ready"}

    def tearDown(self):
        self.temp.cleanup()

    def test_explicit_or_sibling_existing_backend(self):
        with patch.dict(os.environ, {"MOTION_BACKEND_ROOT": str(self.root)}):
            self.assertEqual(backend.backend_root(), self.root.resolve())
        with patch.object(backend, "APP_ROOT", self.root / "web"), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(backend.backend_root(), self.root / "GEM-X")

    def test_command_invokes_original_script_and_environment_only(self):
        with patch.object(backend, "inspect_backend", return_value=self.status):
            command = backend.build_command(self.root / "a job", {}, self.video)
        self.assertEqual(command, [
            self.status["python"], "-u", str(Path(self.status["root"]) / "cpu/run_cpu.py"),
            "--video", str(self.video.resolve()), "--output", str(self.root / "a job/output"),
            "--threads", "4", "--start-frame", "0", "--max-frames", "120", "--render-skeleton",
        ])
        self.assertFalse((self.root / "a job").exists())

    def test_existing_cli_frame_and_conditioning_flags(self):
        config = {"threads": 2, "start_frame": 17, "max_frames": 45, "render_skeleton": False, "use_image_features": False}
        with patch.object(backend, "inspect_backend", return_value=self.status):
            command = backend.build_command(self.root / "job", config, self.video)
        self.assertIn("--no-imgfeat", command)
        self.assertNotIn("--render-skeleton", command)
        self.assertEqual(command[command.index("--start-frame") + 1], "17")
        self.assertEqual(command[command.index("--max-frames") + 1], "45")

    def test_unsupported_and_invalid_options_are_rejected(self):
        for config in ({"seed": 42}, {"steps": 30}, {"transcode": True}, {"threads": True}, {"threads": "4"}, {"threads": 0}, {"max_frames": -1}, {"start_frame": -1}, {"render_skeleton": "false"}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                backend.build_command(self.root, config, self.video)

    def test_input_file_and_backend_are_required(self):
        for input_path in (None, self.root / "missing.mp4", self.root):
            with self.subTest(input_path=input_path), self.assertRaises(ValueError):
                backend.build_command(self.root, {}, input_path)
        with patch.object(backend, "inspect_backend", return_value={"ready": False, "message": "not installed"}):
            with self.assertRaisesRegex(RuntimeError, "not installed"):
                backend.build_command(self.root, {}, self.video)

    def test_inspection_is_read_only_and_needs_original_script(self):
        with patch.dict(os.environ, {"MOTION_BACKEND_ROOT": str(self.root)}):
            result = backend.inspect_backend()
        self.assertFalse(result["ready"])
        self.assertIn("cpu/run_cpu.py", result["missing"])
        self.assertFalse((self.root / ".venv").exists())
        self.assertFalse((self.root / "repo").exists())

    def test_backend_import_loads_no_model_or_extra_adapter(self):
        code = "import sys;sys.path.insert(0,sys.argv[1]);import backend;assert 'torch' not in sys.modules;assert 'onnxruntime' not in sys.modules;assert 'run_cpu' not in sys.modules"
        subprocess.run([sys.executable, "-c", code, str(APP)], check=True)


if __name__ == "__main__":
    unittest.main()
