"""Recover SOMA skeleton transforms from saved GEM predictions, without rerunning GEM.

Uses the official pinned SOMA rig and MHR identity model. Does not alter motion
predictions, smooth motion, average identities, or run YOLO/ViTPose/SAM3D.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from runtime_env import ROOT, REPO
COMPAT_ASSETS = REPO / "inputs/soma_assets_compat_e0f8ff0"
SOMA_COMMIT = "e0f8ff0ecfa3edbbb6058b1e0f08822ee2f84ee5"
PINNED_HASHES = {
    "SOMA_neutral.npz": "515f7d5bb74be4e370e9adf5e779760ec3581556374c0b33212a32d13ab3b53f",
    "correctives_model.pt": "3b0f82928dc25c952e0a23af59dbc9c02854ccb4c4c1c988f70bbffd7c284636",
    "MHR/mhr_model_lod6.pt": "28839b7ee0eb38eb3943cf26cff61a306de78f86227a0f17bb044252c7bc06ac",
    "MHR/base_body_lod6.obj": "cd46620ec46e968658527f25e2b54b8f2292bfd486eac59c596f8b5d5229b9e0",
    "MHR/SOMA_wrap_lod1.obj": "9ab7a2328d5b026c17ef1d09ed692ffeb68f96d3fc74970cba72e667dbe0f728",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_soma_model(asset_dir):
    """Use the complete official rig matching the installed SOMA source schema."""
    import numpy as np

    source = Path(asset_dir).resolve()
    required_rig = {"joint_parent_ids", "joint_names", "bind_shape", "bind_pose_world", "t_pose_world"}
    core = source / "SOMA_neutral.npz"
    if not core.is_file():
        raise FileNotFoundError(f"Missing SOMA assets: {core}")
    with np.load(core, allow_pickle=False) as data:
        compatible = required_rig.issubset(data.files)
    if not compatible:
        print("SOMA asset schema differs from pinned source; using separately preserved matching official rig", flush=True)
        source = COMPAT_ASSETS
    for relative, expected in PINNED_HASHES.items():
        path = source / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"Required official SOMA {SOMA_COMMIT[:8]} asset missing or mismatched: {path}")
    sys.path.insert(0, str(REPO / "third_party/soma"))
    from soma import SOMALayer, Unit

    model = SOMALayer(data_root=source, low_lod=True, device="cpu", identity_model_type="mhr", mode="dense", output_unit=Unit.METERS)
    model.eval()
    return model, source
