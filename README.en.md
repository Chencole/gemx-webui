[简体中文](README.md) | [English](README.en.md)

# GEM-X Studio

A lightweight web interface for an existing local GEM-X CPU command-line installation. Upload a person video, select a frame range, inspect logs, cancel jobs, and preview or download the original SOMA motion data and skeleton videos.

The interface is currently in Chinese; documentation is available in Chinese and English.

**This project adds visual controls only. It does not install or replace models, or change inference, motion data, or postprocessing.**

## Project sources

- Official model and source: [NVIDIA / NVlabs GEM-X](https://github.com/NVlabs/GEM-X). The local installation uses source revision [32992550](https://github.com/NVlabs/GEM-X/tree/32992550dba114c62243fb55e361311972dce8f9).
- Official weights and model card: [nvidia/GEM-X](https://huggingface.co/nvidia/GEM-X).
- This community interface: [Chencole/gemx-webui](https://github.com/Chencole/gemx-webui). It is an independent community wrapper, not an official NVIDIA web product.
- The [Hugging Face showcase](https://huggingface.co/spaces/coooooooai/gemx-webui) is a free static page. It provides no online inference and does not connect to a visitor's computer.

`GEM-X/cpu/run_cpu.py` is a local CPU adaptation previously added to the existing installation. There is currently no separate public repository link to provide for that adaptation. It is not an official NVIDIA CPU distribution and is not included in this web interface repository. Neither the official repository nor this interface repository should be treated as its installation package.

## Prerequisites

You need Git, [uv](https://docs.astral.sh/uv/getting-started/installation/), and an already prepared local GEM-X CPU backend containing `cpu/run_cpu.py`, its own `.venv`, source code, and model files. The web interface's dependencies are separate from the model environment.

**Cloning this interface or the official GEM-X repository does not automatically provide that local CPU adaptation.** Without the matching backend, you can open the interface but cannot run inference. The page reports that the backend is unavailable and does not automatically install or download models.

## Get and start the interface

For a new checkout, run the following in the directory where you want to keep the project. These commands work in both Windows CMD and PowerShell:

```sh
git clone https://github.com/Chencole/gemx-webui.git GEMX-WebUI
cd GEMX-WebUI
uv run python app.py
```

If you already cloned the project, enter its directory and run `uv run python app.py`. Open **http://127.0.0.1:8771**.

By default, the interface looks for a sibling directory named `GEM-X`. If your existing backend is elsewhere, set `MOTION_BACKEND_ROOT` in the same terminal before starting the interface. Point it to the installation root containing `cpu`, `repo`, and `.venv`, not to the `repo` subdirectory.

Windows CMD, from the web interface's project directory:

```cmd
set "MOTION_BACKEND_ROOT=E:\dev\GEM-X"
uv run python app.py
```

PowerShell, from the web interface's project directory:

```powershell
$env:MOTION_BACKEND_ROOT = "E:\dev\GEM-X"
uv run python app.py
```

Replace the example path with your existing installation. The interface uses that backend's `.venv` Python to call its original `cpu/run_cpu.py`.

## Usage and shutdown

Upload a single-person video, choose the starting frame, frame count, CPU thread count, and existing inference options, then start the job. The current backend assumes a static camera. Occlusion, blur, and small hands can reduce quality; it does not reconstruct swords or other objects. The current regression mode has no seed or steps parameters.

Each job is stored in `data/<unique ID>/output`. You can inspect logs, cancel a job, and download individual files or a results ZIP. The ZIP excludes the uploaded source video. History is stored locally and separated by browser session. Restarting the server does not rerun interrupted jobs.

Main outputs include `motion_soma.npz`, `hpe_results.pt`, and `metadata.json`. The 2D keypoint preview is `keypoints_2d.mp4`; enabling skeleton rendering also produces `joints_3d_global.npy` and `skeleton_3d.mp4`.

Previews use the original outputs. If your browser cannot play an MP4, download it and open it with a local media player. The interface does not transcode videos or modify motion data.

**Closing the last local interface tab cancels jobs and stops the server after about three seconds.** Refreshing does not immediately stop it; all tabs must be closed. If the browser exits unexpectedly, heartbeat detection stops the server within about three minutes. You can also press Ctrl+C in the terminal. The server listens on `127.0.0.1` by default for local use.

## Port already in use — Windows error 10048

Another service is already using the default port, `8771`. If GEM-X Studio is already available at that address, use the existing page. Otherwise, press Ctrl+C in the old service's terminal, or choose another available port for the new instance:

```sh
uv run python app.py --port 8772
```

Then open **http://127.0.0.1:8772**.

## Checks and licensing

Lightweight checks cover the interface, argument passing, cancellation, downloads, and shutdown when tabs close. Full model inference has not been run; generation speed and motion quality have not been validated.

```sh
uv run python -m unittest discover -s tests -v
```

Original interface and invocation code in this repository is MIT licensed. Models and the existing environment retain their respective licenses; see [Third-party notices](THIRD_PARTY_NOTICES.md).
