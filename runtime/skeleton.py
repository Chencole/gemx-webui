"""Optional real SOMA forward kinematics and OpenCV 3D skeleton preview."""
from pathlib import Path
import sys


def render_soma_skeleton(prediction, args, fps):
    import cv2
    import numpy as np
    import torch

    from runtime_env import REPO
    repository = REPO
    sys.path.insert(0, str(repository / "third_party" / "soma"))
    from soma_assets import load_soma_model
    from run_gpu import log

    asset_dir = args.soma_assets.resolve()
    required = ["SOMA_neutral.npz", "correctives_model.pt", "MHR/mhr_model_lod6.pt",
                "MHR/base_body_lod6.obj", "MHR/SOMA_wrap_lod1.obj"]
    missing = [name for name in required if not (asset_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Optional 3D skeleton needs official SOMA assets in {asset_dir}: {missing}")
    model, asset_dir = load_soma_model(asset_dir)
    body = prediction["body_params_global"]
    joints = []
    with torch.inference_mode():
        for start in range(0, len(body["body_pose"]), args.batch_size):
            stop = start + args.batch_size
            identity = body["identity_coeffs"][start:stop].float().cpu()
            scales = body["scale_params"][start:stop].float().cpu()
            model.prepare_identity(identity, scales[:, 1:], repose_to_bind_pose=False, global_scale=scales[:, :1])
            poses = torch.cat([body["global_orient"][start:stop, None], body["body_pose"][start:stop].reshape(-1, 76, 3)], dim=1).float().cpu()
            result = model.pose(poses, transl=body["transl"][start:stop].float().cpu(), apply_correctives=False)
            value = result["joints"].detach().cpu().numpy()
            if value.shape[1:] != (77, 3) or not np.isfinite(value).all():
                raise RuntimeError("SOMA returned invalid 3D joints")
            joints.append(value)
            log(f"SOMA CPU skeleton: {min(stop, len(body['body_pose']))}/{len(body['body_pose'])}")
    joints = np.concatenate(joints)
    np.save(args.output / "joints_3d_global.npy", joints)
    np.save(args.output / "soma_parents.npy", np.asarray(model.parents, dtype=np.int32))
    write_skeleton_preview(joints, model.parents, args.output, fps)


def write_skeleton_preview(joints, parents, output, fps):
    """Render already recovered official joints without running any model."""
    import cv2
    import numpy as np

    output = Path(output)
    # Front and side orthographic views retain the model's actual metric geometry.
    # Shared fixed framing does not normalize each frame or remove model motion.
    width, height, margin = 1280, 720, 70
    writer = cv2.VideoWriter(str(output / "skeleton_3d.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create skeleton_3d.mp4")
    views = []
    for horizontal in (0, 2):
        uv = joints[..., [horizontal, 1]].copy()
        uv[..., 1] *= -1
        low, high = uv.min(axis=(0, 1)), uv.max(axis=(0, 1))
        center = (low + high) / 2
        scale = min((width / 2 - margin * 2) / max(high[0] - low[0], 0.01), (height - margin * 2) / max(high[1] - low[1], 0.01))
        views.append(((uv - center) * scale).astype(np.int32))
    for index in range(len(joints)):
        frame = np.full((height, width, 3), 25, dtype=np.uint8)
        for view_index, projected in enumerate(views):
            points = projected[index] + np.array([width // 4 + view_index * width // 2, height // 2])
            for joint, parent in enumerate(parents):
                if parent >= 0:
                    cv2.line(frame, tuple(points[joint]), tuple(points[parent]), (90, 225, 155), 2, cv2.LINE_AA)
            cv2.putText(frame, "Front (X/Y)" if view_index == 0 else "Side (Z/Y)", (30 + view_index * width // 2, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, f"GEM-X / SOMA actual 3D joints | static camera assumption | frame {index + 1}", (30, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
