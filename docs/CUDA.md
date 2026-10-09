# Windows CUDA 部署

此路径保留 GEM-X Studio 的上传、参数、任务历史、日志、取消与预览界面，使用社区 CUDA 适配。模型为官方 GEM-X ONNX、SAM-3D-Body 与 SOMA，权重不作转换或量化。默认 CPU 配置继续兼容原有 `GEM-X/cpu/run_cpu.py`。

## 安装布局

```text
E:\dev\GEMX-WebUI\          本可视化仓库
  .venv\                   轻量网页依赖
  runtime\.venv\           CUDA 推理依赖
E:\dev\GEM-X\
  repo\                    官方源码、三个第三方源码、inputs 模型
  cache\torch\hub\checkpoints\  YOLOX ONNX
```

Python 3.12.14 由 uv 在其常规管理目录中安装并跨项目复用。不要复制其他机器的 `.venv`。`runtime/uv.lock` 固定 CUDA 依赖，包含 PyTorch 2.10.0 CUDA 12.6 和 ONNX Runtime GPU 1.23.2；必须使用 NVIDIA 驱动支持的 CUDA 环境。PyTorch wheel 自带所需运行库，基础 ONNX 路径无需编译 CUDA 扩展。

上游源码版本：GEM-X `32992550dba114c62243fb55e361311972dce8f9`、SAM-3D-Body `b5c765a0d89d789985e186d396315e7590887b94`、SOMA-X `e0f8ff0ecfa3edbbb6058b1e0f08822ee2f84ee5`、DINOv3 `6876159a11b4df116f30f667f8c9888617df0751`。第三方源码分别位于 `repo/third_party/sam-3d-body`、`soma`、`dinov3`。

需要 `backend.py` 中列出的官方模型文件。SAM3D 使用 `inputs/checkpoints/sam-3d-body-dinov3` 和 `inputs/mhr_data`；ONNX 位于 `inputs/onnx`。与 SOMA 源码匹配的完整 rig 存放在 `inputs/soma_assets_compat_e0f8ff0`，由 `runtime/soma_assets.py` 验证 SHA-256。不要用不同版本的 SOMA rig 互相覆盖。迁移时可以复制这些已经校验的模型和源码文件；不会复制环境、用户视频或生成历史。

在本仓库目录的普通 PowerShell 中安装：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\Install-GPU.ps1
```

脚本使用已安装的 uv、复用兼容 Python，创建两个项目环境，检查真实 CUDA 与 SAM3D 导入。它不更改系统默认 Python，不下载或覆盖模型。环境检查通过只代表运行时可用，完整生成需下一步验证。

## 启动和验证

双击 `Start-GPU.cmd`，打开 `http://127.0.0.1:8771`。或在普通终端运行：

```powershell
$env:MOTION_DEVICE = 'cuda'
$env:MOTION_BACKEND_ROOT = 'E:\dev\GEM-X'
uv run --locked python app.py
```

需局域网访问时，运行 `Start-GPU.cmd --host 192.168.1.8` 并将 Windows 防火墙 8771 端口限定到需要访问的客户端 IP。没有登录系统，不适合直接开放到公网。关闭最后一个页面后仍沿用原来的停服行为。

先上传清晰、静止相机、单个人物的短视频，处理 8–16 帧，保留完整图像特征。8 GB 显存默认 batch 1，各模型依次加载释放；视频预处理、SOMA 运动学和骨骼绘制仍在 CPU 上。CUDA 初始化失败或显存不足会报错，不会静默切回 CPU、降低精度或禁用图像特征。其他显卡和更长视频仍需实际验证，不能由文件检查推断成功。

也可直接测试，不经过网页：

```powershell
$env:MOTION_BACKEND_ROOT = 'E:\dev\GEM-X'
uv run --project runtime --locked python runtime\run_gpu.py --video 'E:\videos\person.mp4' --output 'E:\dev\GEM-X\outputs\gpu-smoke' --max-frames 8 --batch-size 1 --render-skeleton
```

成功需同时确认进程退出 0，`metadata.json` 的 provider 是 `CUDAExecutionProvider`、device 是实际 NVIDIA GPU，以及 `motion_soma.npz`、骨骼坐标和预览包含所选帧数。模型采用静止相机 regression 路径，不进行相机 SLAM、武器追踪或机器人重定向。

## 已验证范围

2026-10-10 在 Windows 10、RTX 3060 Ti 8 GB、32 GB 内存上完成 8 帧测试：YOLOX、ViTPose、SAM3D FP32、GEM ONNX 和 SOMA 骨架预览全程成功，总耗时 67.232 秒。使用 batch 1，保留默认关键点翻转测试和完整图像特征，没有传入预计算特征或人工 bounding box，也没有启用 `--no-imgfeat`。

输入取自上述固定版本 [官方 README 的真人 GIF](https://github.com/NVlabs/GEM-X/blob/32992550dba114c62243fb55e361311972dce8f9/assets/demo_2d_keypoints.gif)，前 8 帧，480×270，保留原始 10 fps。该 GIF 已有关键点覆盖；它适合作为运行检查，不是干净视频的精度评测。模型训练时间尺度接近 30 fps，实际使用应优先采用清晰的 30 fps 视频。

独立结果检查确认 `CUDAExecutionProvider` 与实际 GPU 名称，有限的 8×1024 SAM3D 特征、8×77×3 SOMA 关节、8 帧 SOMA 参数，以及两路预览都能解码出 8 帧。未测量峰值显存，未验证长视频性能或动作精度。结束后模型进程退出，显存返回约 416 MiB 的桌面占用。

## 更新和卸载

更新本仓库后再次执行安装脚本；它按锁文件同步，不影响其他项目。模型和上游源码的版本升级应一起验证，不能只替换某个 rig 文件。卸载本项目时先关闭窗口，再删除本项目目录；若仍需 CPU 版本或其他应用，共用的 `GEM-X` 模型、uv 和托管 Python 可以保留。通用工具继续通过其原安装方式更新和卸载。

参考：[NVIDIA 官方 GEM-X](https://github.com/NVlabs/GEM-X)、[ONNX Runtime CUDA 与 PyTorch 兼容说明](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)。
