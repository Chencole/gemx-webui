#!/usr/bin/env python
"""Native CUDA entrypoint for official GEM-X ONNX models.

This adapter does not alter or approximate neural-network weights. It mirrors
GEM-X's ONNX preprocessing and SOMA-v2 decoder with sequential CUDA stages.
Full image conditioning uses the sam3d_gpu module; --no-imgfeat is an
explicit official alternative and is never selected silently.
"""
from __future__ import annotations

import argparse
import ast
import gc
import json
import os
import sys
import time
from pathlib import Path
from runtime_env import ROOT, REPO, configure_runtime, require_cuda

sys.path.insert(0, str(REPO))
configure_runtime()


def log(message):
    print(time.strftime("[%H:%M:%S] ") + message, flush=True)


class CudaOnnx:
    def __init__(self, path, threads=4):
        import onnxruntime as ort

        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Required official ONNX file missing: {path}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Large ViT models otherwise retain buffers from each batch shape.
        options.enable_mem_pattern = False
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=[("CUDAExecutionProvider", {"device_id": 0, "arena_extend_strategy": "kSameAsRequested", "cudnn_conv_use_max_workspace": "0"}), "CPUExecutionProvider"]
        )
        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()
        self.path = path
        if "CUDAExecutionProvider" not in self.session.get_providers():
            raise RuntimeError("CUDAExecutionProvider could not start; refusing CPU fallback")
        log(f"Loaded {path.name} with CUDAExecutionProvider")

    def __call__(self, **inputs):
        import numpy as np

        feed = {}
        for item in self.inputs:
            if item.name not in inputs:
                raise ValueError(f"Missing ONNX input {item.name} in {self.path.name}")
            value = inputs[item.name]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            value = np.ascontiguousarray(value)
            if item.type == "tensor(float)":
                value = value.astype(np.float32, copy=False)
            elif item.type == "tensor(float16)":
                value = value.astype(np.float16, copy=False)
            feed[item.name] = value
        result = self.session.run(None, feed)
        return {item.name: value for item, value in zip(self.outputs, result)}


def read_video(path, start_frame, max_frames):
    import cv2
    import numpy as np

    reader = cv2.VideoCapture(str(path))
    if not reader.isOpened():
        raise RuntimeError(f"Cannot decode input video: {path}")
    fps = float(reader.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        raise RuntimeError("Video has no valid frame rate; re-encode it before inference")
    # Sequential skipping preserves the requested frame index even for codecs
    # whose random seeking jumps to keyframes or ignores CAP_PROP_POS_FRAMES.
    for _ in range(start_frame):
        if not reader.grab():
            reader.release()
            raise RuntimeError("Requested start frame is beyond the end of the video")
    frames = []
    while max_frames is None or len(frames) < max_frames:
        ok, frame = reader.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    reader.release()
    if not frames:
        raise RuntimeError("No frames could be read from the requested video range")
    return np.stack(frames), fps


def bbox_to_xys(boxes):
    """Same 192:256 aspect correction and 1.2 expansion as official GEM."""
    import numpy as np

    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
        raise ValueError("Bounding boxes must be finite arrays of shape (T, 4)")
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    if np.any(widths <= 0) or np.any(heights <= 0):
        raise ValueError("Bounding boxes must have positive width and height")
    size = np.maximum(heights, widths / (192 / 256)) * 1.2
    return np.concatenate([centers, size[:, None]], axis=1).astype(np.float32)


def detect_people(frames, args):
    import numpy as np
    from scipy.ndimage import uniform_filter1d

    if args.bbox:
        boxes = np.tile(np.asarray(args.bbox, dtype=np.float32), (len(frames), 1))
        return boxes, bbox_to_xys(boxes), np.full(len(frames), -2, dtype=np.int32)
    from gem.utils.yolox_detector import YOLOXDetector, detect_and_track

    detector_path = args.detector or ROOT / "cache" / "torch" / "hub" / "checkpoints" / "yolox_x_8xb8-300e_humanart-a39d44ed.onnx"
    if not detector_path.is_file():
        raise FileNotFoundError(f"Local YOLOX detector missing: {detector_path}. Run the WebUI environment panel to inspect the installation; this entrypoint does not download models.")
    detector = YOLOXDetector(onnx_path=str(detector_path), device="cuda")
    if "CUDAExecutionProvider" not in detector.sess.get_providers():
        raise RuntimeError("YOLOX CUDA provider unavailable; refusing CPU fallback")
    boxes, track_ids = detect_and_track(frames, detector)
    if np.all(track_ids < 0):
        raise RuntimeError("No person detected. Supply an explicit --bbox x1 y1 x2 y2 only if the video contains a visible person.")
    # Replicate official three-frame box smoothing, preserving the detection track.
    if len(boxes) > 3:
        boxes = uniform_filter1d(boxes, size=3, axis=0, mode="nearest")
    h, w = frames.shape[1:3]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
    del detector
    gc.collect()
    return boxes, bbox_to_xys(boxes), track_ids


def predict_keypoints(frames, boxes_xys, args):
    import numpy as np
    import torch
    from gem.utils.vitpose_extractor import get_batch, keypoints_from_heatmaps, flip_heatmap_soma77

    runner = CudaOnnx(args.model_dir / "vitpose.onnx", args.threads)
    results = []
    # Official get_batch expects BGR and converts to RGB internally.
    for start in range(0, len(frames), args.batch_size):
        end = min(start + args.batch_size, len(frames))
        bxy = torch.from_numpy(boxes_xys[start:end])
        crops, _ = get_batch(frames[start:end, :, :, ::-1], bxy)
        crops = crops[:, :, :, 32:224]
        result = runner(imgs=crops)
        heatmaps = result["heatmaps"]
        if not args.no_flip_test:
            flipped = runner(imgs=crops.flip(3))["heatmaps"]
            flipped = flip_heatmap_soma77(torch.from_numpy(flipped)).numpy()
            heatmaps = (heatmaps + flipped) * 0.5
        scale = np.stack([boxes_xys[start:end, 2] * 0.75, boxes_xys[start:end, 2]], axis=1) / 200
        points, confidence = keypoints_from_heatmaps(heatmaps, boxes_xys[start:end, :2], scale, use_udp=True)
        results.append(np.concatenate([points, confidence], axis=-1))
        log(f"ViTPose: {end}/{len(frames)} frames")
    del runner
    gc.collect()
    keypoints = np.concatenate(results).astype(np.float32)
    if keypoints.shape != (len(frames), 77, 3) or not np.isfinite(keypoints).all():
        raise RuntimeError(f"ViTPose returned invalid keypoints: {keypoints.shape}")
    return keypoints


def estimate_intrinsics(width, height, count):
    import numpy as np

    matrix = np.eye(3, dtype=np.float32)
    matrix[0, 0] = matrix[1, 1] = max(width, height)
    matrix[0, 2], matrix[1, 2] = width / 2, height / 2
    return np.repeat(matrix[None], count, axis=0)


def decode_predictions(pred_x, pred_cam, boxes_xys, intrinsics):
    """Official SOMA-v2 decode, with static-camera world-trajectory recovery.

    No model instance is needed: EnDecoder.decode_soma_v2 is a deterministic
    denormalization and 6D-rotation conversion. Reference:
    repo/gem/network/endecoder.py; configs/endecoder/v2_soma_local_cam.yaml.
    """
    import numpy as np
    import torch
    from gem.network.stats_compose import MM_V2_SOMA_METROSIM
    from gem.utils.rotation_conversions import matrix_to_axis_angle, rotation_6d_to_matrix
    from gem.utils.motion_utils import rollout_local_transl_vel

    x = torch.as_tensor(pred_x, dtype=torch.float32)
    if x.ndim == 3:
        if x.shape[0] != 1:
            raise ValueError("This entrypoint processes one tracked person at a time")
        x = x[0]
    if x.shape[-1] != 585:
        raise ValueError(f"Expected official SOMA-v2 feature dimension 585, got {x.shape}")
    mean = torch.tensor(MM_V2_SOMA_METROSIM["mean"], dtype=torch.float32)
    # clip_std=true in the official model configuration.
    std = torch.tensor(MM_V2_SOMA_METROSIM["std"], dtype=torch.float32).clamp(min=1)
    x = x * std + mean
    body = matrix_to_axis_angle(rotation_6d_to_matrix(x[:, :456].reshape(-1, 76, 6))).flatten(1)
    root_camera = matrix_to_axis_angle(rotation_6d_to_matrix(x[:, 570:576]))
    root_global = matrix_to_axis_angle(rotation_6d_to_matrix(x[:, 576:582]))
    scale = x[:, 501:570].clone()
    scale[:, 0].clamp_(0.7, 1.0)
    camera = torch.as_tensor(pred_cam, dtype=torch.float32).reshape(-1, 3)
    bxy = torch.as_tensor(boxes_xys, dtype=torch.float32)
    K = torch.as_tensor(intrinsics, dtype=torch.float32)
    sb = camera[:, 0] * bxy[:, 2] + 1e-9
    transl_camera = torch.stack([
        camera[:, 1] + 2 * (bxy[:, 0] - K[:, 0, 2]) / sb,
        camera[:, 2] + 2 * (bxy[:, 1] - K[:, 1, 2]) / sb,
        2 * K[:, 0, 0] / sb,
    ], dim=-1)
    # For identity camera angular velocity, official get_body_params_w_Rt_v2
    # reduces to R_global=R_gv and rollout_local_transl_vel. No SLAM is invented.
    transl_global = rollout_local_transl_vel(x[None, :, 582:585], root_global[None])[0]
    common = {"body_pose": body, "identity_coeffs": x[:, 456:501], "scale_params": scale}
    result = {
        "body_params_incam": {**common, "global_orient": root_camera, "transl": transl_camera},
        "body_params_global": {**common, "global_orient": root_global, "transl": transl_global},
        "K_fullimg": K,
    }
    for group in (result["body_params_incam"], result["body_params_global"]):
        for name, value in group.items():
            if not torch.isfinite(value).all():
                raise RuntimeError(f"Non-finite model output in {name}; refusing to save an invalid motion")
    return result


def infer_motion(keypoints, boxes_xys, intrinsics, features, args):
    import numpy as np

    model_name = "gem_denoiser_no_imgfeat.onnx" if args.no_imgfeat else "gem_denoiser.onnx"
    runner = CudaOnnx(args.model_dir / model_name, args.threads)
    length = len(keypoints)
    inputs = {
        "obs": keypoints[None],
        "bbx_xys": boxes_xys[None],
        "K_fullimg": intrinsics[None],
        "f_imgseq": features[None],
        "f_cam_angvel": np.tile(np.array([1, 0, 0, 0, 1, 0], np.float32), (1, length, 1)),
    }
    # Exported graphs may have a fixed temporal dimension. Pad only with the
    # final real frame; reject longer inputs rather than truncate motion.
    target_length = length
    for item in runner.inputs:
        if item.name == "obs" and len(item.shape) > 1 and isinstance(item.shape[1], int):
            target_length = item.shape[1]
    if length > target_length:
        raise ValueError(f"This ONNX graph supports {target_length} frames; input has {length}. Use --max-frames or a dynamic official export.")
    if target_length > length:
        inputs = {k: np.concatenate([v, np.repeat(v[:, -1:], target_length - length, axis=1)], axis=1) for k, v in inputs.items()}
        log(f"Fixed-length graph: padded {length} real frames to {target_length}, trimming outputs afterwards")
    result = runner(**inputs)
    pred_x, pred_cam = result["pred_x"][:, :length], result["pred_cam"][:, :length]
    del runner
    gc.collect()
    return pred_x, pred_cam


def skeleton_parents():
    # Read the official constant without importing video/render dependencies.
    tree = ast.parse((REPO / "gem/utils/kp2d_utils.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "PARENTS_77" for target in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("Official SOMA skeleton parent table was not found")


def write_keypoint_preview(frames, keypoints, output, fps):
    import cv2

    h, w = frames.shape[1:3]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video: {output}")
    parents = skeleton_parents()
    for index, (rgb, points) in enumerate(zip(frames, keypoints)):
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for joint, parent in enumerate(parents):
            if parent >= 0 and min(points[joint, 2], points[parent, 2]) >= 0.3:
                a, b = tuple(points[joint, :2].astype(int)), tuple(points[parent, :2].astype(int))
                cv2.line(frame, a, b, (40, 230, 80), 2, cv2.LINE_AA)
        cv2.putText(frame, f"2D observations | frame {index + 1}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


def main():
    parser = argparse.ArgumentParser(description="GEM-X official models on CUDA (static-camera videos)")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "cuda")
    parser.add_argument("--model-dir", type=Path, default=REPO / "inputs" / "onnx")
    parser.add_argument("--detector", type=Path)
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--no-imgfeat", action="store_true", help="Explicitly use official 2D-keypoints-only GEM variant (lower quality)")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-flip-test", action="store_true", help="Skip official keypoint flip test for speed; changes inference behavior")
    parser.add_argument("--features", type=Path, help="Previously computed official SAM3D pose_tokens .npy for exactly these frames")
    parser.add_argument("--render-skeleton", action="store_true", help="Use actual SOMA body-model CPU kinematics and render 3D joints")
    parser.add_argument("--soma-assets", type=Path, default=REPO / "inputs" / "soma_assets")
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1 or args.start_frame < 0 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("Threads, batch size and max frames must be positive; start frame must be nonnegative")
    if args.no_imgfeat and args.features:
        parser.error("--features and --no-imgfeat are mutually exclusive")
    args.video = args.video.resolve(strict=True)
    args.model_dir = args.model_dir.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    require_cuda()
    log("CUDA inference; global trajectory assumes a static camera")
    frames, fps = read_video(args.video, args.start_frame, args.max_frames)
    log(f"Read {len(frames)} frames at {fps:.3f} fps")
    if abs(fps - 30) > 1:
        log("Input is not 30 fps. Model temporal motion scale is trained around 30 fps; consider re-encoding to 30 fps.")
    boxes, boxes_xys, track_ids = detect_people(frames, args)
    np.savez_compressed(args.output / "detections.npz", bbx_xyxy=boxes, bbx_xys=boxes_xys, track_ids=track_ids)
    keypoints = predict_keypoints(frames, boxes_xys, args)
    np.save(args.output / "keypoints_2d.npy", keypoints)
    write_keypoint_preview(frames, keypoints, args.output / "keypoints_2d.mp4", fps)
    intrinsics = estimate_intrinsics(frames.shape[2], frames.shape[1], len(frames))
    if args.no_imgfeat:
        features = np.zeros((len(frames), 1024), dtype=np.float32)
        log("Using explicitly requested official no-image-feature model")
    elif args.features:
        features = np.load(args.features, allow_pickle=False).astype(np.float32)
    else:
        from sam3d_gpu import extract_pose_tokens_cuda
        features = extract_pose_tokens_cuda(frames, boxes_xys, args)
        np.save(args.output / "sam3d_pose_tokens.npy", features)
    if features.shape != (len(frames), 1024) or not np.isfinite(features).all():
        raise ValueError(f"Expected finite official SAM3D features of shape ({len(frames)}, 1024), got {features.shape}")
    pred_x, pred_cam = infer_motion(keypoints, boxes_xys, intrinsics, features, args)
    np.savez_compressed(args.output / "gem_raw_predictions.npz", pred_x=pred_x, pred_cam=pred_cam)
    prediction = decode_predictions(pred_x, pred_cam, boxes_xys, intrinsics)
    torch.save(prediction, args.output / "hpe_results.pt")
    export = {f"{space}_{name}": value.numpy() for space, group in prediction.items() if isinstance(group, dict) for name, value in group.items()}
    export.update(K_fullimg=intrinsics, fps=np.float32(fps), frame_indices=np.arange(args.start_frame, args.start_frame + len(frames)))
    np.savez_compressed(args.output / "motion_soma.npz", **export)
    if args.render_skeleton:
        from skeleton import render_soma_skeleton
        render_soma_skeleton(prediction, args, fps)
    metadata = {
        "source_video": str(args.video), "frame_start": args.start_frame,
        "frames": len(frames), "fps": fps, "provider": "CUDAExecutionProvider", "device": torch.cuda.get_device_name(0),
        "model": "official GEM-X SOMA-v2", "conditioning": "2D only (official no_imgfeat model)" if args.no_imgfeat else "2D and official SAM3D image features",
        "camera": "static assumption; no SLAM", "soma_joints": 77,
        "body_axis_angle_shape": list(prediction["body_params_incam"]["body_pose"].shape),
        "identity_coefficients": 45, "scale_parameters": 69,
        "keypoint_flip_test": not args.no_flip_test, "elapsed_seconds": round(time.perf_counter() - started, 3),
        "limitations": ["Single primary tracked person", "No object/sword reconstruction", "No facial expression estimation", "Global trajectory is a learned estimate under a static-camera assumption", "ONNX regression path does not apply the full PyTorch demo's optional contact postprocessing"],
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Saved real GEM-X output to {args.output} ({metadata['elapsed_seconds']:.1f}s)")


if __name__ == "__main__":
    main()
