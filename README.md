---
title: GEM-X Studio
emoji: 🌀
colorFrom: purple
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Visual controls for an existing local motion backend.
---

[简体中文](README.md) | [English](README.en.md)

# GEM-X Studio

为已有的本机 GEM-X CPU 命令行版本提供轻量网页：上传人物视频，设置处理范围，查看日志、取消任务，并预览或下载原始 SOMA 动作数据和骨骼视频。

当前网页界面为中文；使用文档提供中英文。

**仅增加可视化调用层，不安装或替换模型，不修改推理、动作数据或后处理。**

## 项目来源

- 官方模型与源码：[NVIDIA / NVlabs GEM-X](https://github.com/NVlabs/GEM-X)。本机安装使用的源码版本为 [32992550](https://github.com/NVlabs/GEM-X/tree/32992550dba114c62243fb55e361311972dce8f9)。
- 官方权重与模型卡：[nvidia/GEM-X](https://huggingface.co/nvidia/GEM-X)。
- 本社区可视化封装：[Chencole/gemx-webui](https://github.com/Chencole/gemx-webui)。这是独立社区界面，并非 NVIDIA 官方网页产品。
- 本界面的 [Hugging Face 展示](https://huggingface.co/spaces/coooooooai/gemx-webui) 是免费静态页面，不提供在线推理，也不连接访问者本机。

`GEM-X/cpu/run_cpu.py` 是此前为已有安装添加的本地 CPU 适配。目前没有可提供的独立公开仓库链接；它不是 NVIDIA 官方 CPU 发行版，也不包含在这个网页仓库中。请勿将官方仓库或本界面仓库当成该适配的下载包。

## 前置条件

需要 Git、[uv](https://docs.astral.sh/uv/getting-started/installation/)，以及已经准备好的本机 GEM-X CPU 后端：包含 `cpu/run_cpu.py`、后端自己的 `.venv`、源码和模型文件。网页依赖与模型环境分开。

**只克隆本网页或官方 GEM-X 仓库，都不会自动获得上述本地 CPU 适配。** 没有对应后端时可以打开界面，但不能生成；页面会显示环境未就绪，不自动安装或下载模型。

## 获取与启动

初次获取网页，在希望存放项目的目录执行（CMD 和 PowerShell 均可）：

```sh
git clone https://github.com/Chencole/gemx-webui.git GEMX-WebUI
cd GEMX-WebUI
uv run python app.py
```

已克隆的用户只需进入项目目录，执行 `uv run python app.py`。打开 **http://127.0.0.1:8771**。

默认查找网页目录同级的 `GEM-X` 文件夹。若已有后端位于其他位置，在启动网页的同一终端设置 `MOTION_BACKEND_ROOT`；路径应指向同时包含 `cpu`、`repo` 和 `.venv` 的安装根目录，不是 `repo` 子目录。

Windows CMD（在网页项目目录中）：

```cmd
set "MOTION_BACKEND_ROOT=E:\dev\GEM-X"
uv run python app.py
```

PowerShell（在网页项目目录中）：

```powershell
$env:MOTION_BACKEND_ROOT = "E:\dev\GEM-X"
uv run python app.py
```

将示例路径改为自己的已有安装位置。界面直接使用后端 `.venv` 的 Python 调用原有 `cpu/run_cpu.py`。

## 使用与退出

上传单人视频，选择起始帧、处理帧数、CPU 线程数和原有推理选项，再点击生成。当前后端使用静止相机假设；遮挡、模糊和手部过小会影响结果，不恢复刀剑等物体。当前 regression 模式没有 seed 或 steps 参数。

每次任务保存在 `data/<独立 ID>/output`。可查看日志、取消任务、下载单个文件或结果 ZIP；ZIP 不包含上传的源视频。历史保存在本机并按浏览器会话区分，重启不会重跑中断任务。

主要输出为 `motion_soma.npz`、`hpe_results.pt` 和 `metadata.json`；二维关键点预览为 `keypoints_2d.mp4`，启用骨骼渲染后还会生成 `joints_3d_global.npy` 和 `skeleton_3d.mp4`。

预览直接使用原版输出。若浏览器无法播放 MP4，请下载后用本地播放器打开；网页不转码或修改动作数据。

**关闭最后一个本地界面标签页后，约 3 秒内取消任务并停止服务。** 刷新不会立即停止；多个标签页需全部关闭。浏览器异常退出时，约 3 分钟内通过心跳检测停止。也可以在终端按 Ctrl+C。服务默认仅监听 `127.0.0.1`，用于本机操作。

## 端口被占用（Windows 错误 10048）

这表示默认端口 `8771` 已有服务在使用。若该地址已打开 GEM-X Studio，直接使用已有页面即可；否则在旧服务终端按 Ctrl+C，或为新实例换一个空闲端口：

```sh
uv run python app.py --port 8772
```

然后打开 **http://127.0.0.1:8772**。

## 检查与许可

页面、参数传递、取消任务、文件下载和关闭停服已进行轻量测试。尚未运行完整模型推理，未验证生成速度或动作质量。

```sh
uv run python -m unittest discover -s tests -v
```

本仓库原创界面和调用代码为 MIT；模型与已有环境仍遵守各自许可，见 [第三方说明](THIRD_PARTY_NOTICES.md)。
