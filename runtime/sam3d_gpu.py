"""Official SAM-3D-Body primary pose tokens on CUDA.

Uses the upstream model, crop transforms, camera/ray conditioning and iterative
MHR decoder. CUDA placement and token exposure are explicit; weights and conditioning are unchanged.
No checkpoint download or model inference happens at import time.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import os
from pathlib import Path
import sys
import time
from runtime_env import ROOT, REPO, configure_runtime



# SAM3D treats *any defined value* as disabling the optional native Momentum
# implementation. Its provided TorchScript MHR then supplies the same body rig.
os.environ["MOMENTUM_ENABLED"] = "0"
configure_runtime()


def _log(message):
    print(time.strftime("[%H:%M:%S] ") + message, flush=True)


def _import_upstream(repo=REPO):
    vendor = Path(repo) / "third_party" / "sam-3d-body"
    if not (vendor / "sam_3d_body" / "build_models.py").is_file():
        raise FileNotFoundError(f"Official sam-3d-body source is missing: {vendor}")
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))

    import torch
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
    from sam_3d_body.data.utils.prepare_batch import prepare_batch
    from sam_3d_body.models.heads import mhr_head

    if mhr_head.MOMENTUM_ENABLED:
        raise RuntimeError(
            "SAM3D was imported before the TorchScript MHR environment was configured. "
            "Start a fresh process with MOMENTUM_ENABLED=0."
        )
    return torch, SAM3DBodyEstimator, load_sam_3d_body, prepare_batch


def validate_imports():
    """Import the model code without loading weights or running inference."""
    torch, _, _, _ = _import_upstream()
    return {
        "torch_version": torch.__version__,
        "source": str(REPO / "third_party" / "sam-3d-body"),
        "mhr_backend": "official TorchScript",
        "weights_loaded": False,
        "inference_run": False,
    }


def _collate_batches(batches, torch, np):
    # Matches GEM-X's SAM3DBExtractor, including the nested NoCollate metadata.
    collated = {}
    for key, value in batches[0].items():
        if isinstance(value, torch.Tensor):
            collated[key] = torch.cat([item[key] for item in batches])
        elif isinstance(value, np.ndarray):
            collated[key] = np.concatenate([item[key] for item in batches])
        else:
            collated[key] = [item[key] for item in batches]
    return collated


@contextmanager
def _local_dinov3_source(torch):
    """Use the pinned local DINOv3 source for SAM's exact torch.hub call."""
    source = REPO / "third_party" / "dinov3"
    if not (source / "hubconf.py").is_file():
        raise FileNotFoundError(f"Pinned official DINOv3 source is missing: {source}")
    original_load = torch.hub.load

    def load_from_local(repo_or_dir, model, *positional, **keywords):
        if str(repo_or_dir) == "facebookresearch/dinov3":
            if keywords.get("pretrained", False):
                raise RuntimeError("SAM should use the checkpoint's DINOv3 weights, not download separate pretrained weights")
            keywords["source"] = "local"
            return original_load(str(source), model, *positional, **keywords)
        return original_load(repo_or_dir, model, *positional, **keywords)

    torch.hub.load = load_from_local
    try:
        yield
    finally:
        torch.hub.load = original_load


def extract_pose_tokens_cuda(frames_rgb, boxes_xys, args):
    """Return the actual official primary body token as float32 (T, 1024).

    ``frames_rgb`` contains uint8 RGB video frames, already restricted to the
    requested frame range. ``boxes_xys`` contains (center_x, center_y, size)
    boxes in that same image resolution, as in the GEM-X extractor.
    """
    import numpy as np

    frames_rgb = np.asarray(frames_rgb)
    boxes_xys = np.asarray(boxes_xys, dtype=np.float32)
    if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3 or len(frames_rgb) == 0:
        raise ValueError("Expected at least one RGB frame with shape (T, H, W, 3)")
    if frames_rgb.dtype != np.uint8:
        raise ValueError("RGB frames must have uint8 values in [0, 255]")
    if boxes_xys.shape != (len(frames_rgb), 3):
        raise ValueError("Expected one (center_x, center_y, size) box per frame")
    if not np.isfinite(boxes_xys).all() or np.any(boxes_xys[:, 2] <= 0):
        raise ValueError("Bounding boxes must be finite and have positive size")

    batch_size = int(getattr(args, "batch_size", 1))
    threads = int(getattr(args, "threads", 4))
    if batch_size < 1 or threads < 1:
        raise ValueError("batch_size and threads must be positive")
    checkpoint = REPO / "inputs" / "checkpoints" / "sam-3d-body-dinov3" / "sam3d_body.ckpt"
    config = checkpoint.with_name("model_config.yaml")
    mhr = REPO / "inputs" / "mhr_data" / "mhr_model.pt"
    missing = [str(path) for path in (checkpoint, config, mhr) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required official SAM3D assets are missing: " + "; ".join(missing))

    torch, estimator_type, load_model, prepare_batch = _import_upstream()
    torch.set_num_threads(threads)
    _log("Loading official SAM3D and TorchScript MHR on CUDA (FP32, batch 1 by default)")
    with _local_dinov3_source(torch):
        model, model_cfg = load_model(
            checkpoint_path=str(checkpoint), device=torch.device("cuda:0"), mhr_path=str(mhr)
        )
    model.float().eval()
    model.backbone_dtype = torch.float32
    # This is the same primary-body-token configuration used by GEM-X.
    model.cfg.defrost()
    model.cfg.MODEL.DECODER.DO_HAND_DETECT_TOKENS = False
    model.cfg.freeze()
    estimator = estimator_type(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    )

    # Capture the decoder result already computed by forward_pose_branch.
    # Running the decoder again, as GEM-X's compatibility wrapper does, is
    # unnecessary in eval mode. All intermediate MHR/keypoint updates still run.
    captured = {}
    original_decoder = model.forward_decoder

    def decoder_with_token(*decoder_args, **decoder_kwargs):
        result = original_decoder(*decoder_args, **decoder_kwargs)
        captured["tokens"] = result[0]
        return result

    model.forward_decoder = decoder_with_token
    features = []
    try:
        with torch.no_grad():
            for start in range(0, len(frames_rgb), batch_size):
                stop = min(start + batch_size, len(frames_rgb))
                batches = []
                for frame, (cx, cy, size) in zip(
                    frames_rgb[start:stop], boxes_xys[start:stop]
                ):
                    half = size * 0.5
                    box = np.asarray(
                        [[cx - half, cy - half, cx + half, cy + half]],
                        dtype=np.float32,
                    )
                    batches.append(
                        prepare_batch(
                            frame, estimator.transform, box,
                            masks=None, masks_score=None,
                        )
                    )
                from sam_3d_body.utils import recursive_to
                batch = recursive_to(_collate_batches(batches, torch, np), "cuda:0")
                model._initialize_batch(batch)
                captured.clear()
                output = model.forward_step(batch, decoder_type="body")
                if "tokens" not in captured:
                    raise RuntimeError("Official SAM3D decoder did not expose pose tokens")
                tokens = captured.pop("tokens")
                if tokens.ndim != 3 or tokens.shape[0] != stop - start or tokens.shape[-1] != 1024:
                    raise RuntimeError(f"Unexpected SAM3D decoder token shape: {tuple(tokens.shape)}")
                primary = tokens[:, 0].detach().float().cpu().numpy().copy()
                if not np.isfinite(primary).all():
                    raise RuntimeError("SAM3D returned non-finite primary pose tokens")
                features.append(primary)
                _log(f"SAM3D CUDA pose tokens: {stop}/{len(frames_rgb)} frames")
                del output, tokens, batch, batches
    finally:
        model.forward_decoder = original_decoder
        captured.clear()
        del estimator, model, original_decoder, decoder_with_token
        gc.collect()
        torch.cuda.empty_cache()
    return np.ascontiguousarray(np.concatenate(features, axis=0), dtype=np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-imports", action="store_true",
        help="Check Python imports only; do not load weights or run inference",
    )
    options = parser.parse_args()
    if options.validate_imports:
        import json
        print(json.dumps(validate_imports(), indent=2))
    else:
        parser.print_help()
