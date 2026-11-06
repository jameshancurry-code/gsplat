# AutoDL RTX 4090：最小 COLMAP 训练

适用：Python 3.12.3、PyTorch 2.8.0+cu128、CUDA Toolkit 12.8.93、RTX 4090 24GB；
仓库 `/root/autodl-tmp/gsplat-learning/gsplat`，数据盘约剩余 49GB。
**尚未在 AutoDL 上执行安装、GPU 编译或训练。** 不需要 OpenAI API 或 API key，不改变训练算法。

## 源码依据

- `setup.py` 要求 torch>=2.7，支持 `BUILD_NO_CUDA=1` 安装后按需 JIT。
- `examples/requirements.txt` 当前固定 torch 2.9.1 / torchvision 0.24.1，不能直接用于本机。
  本脚本锁定现有 torch 2.8.0+cu128，复用可通过 CUDA NMS 检查的 torchvision 0.23.0；
  缺失时才安装官方 torchvision 0.23.0+cu128。
- `examples/simple_trainer.py` 是入口；`examples/datasets/colmap.py` 使用 pycolmap、OpenCV、Pillow、
  imageio；`examples/exif.py` 顶层导入 piexif；`examples/utils.py` 顶层导入 sklearn 和 matplotlib。
- 关闭 viewer 仍会导入 viser / nerfview；TensorBoard、torchmetrics 也是必需导入。
  nerfview 使用 requirements 中的固定提交，通过 codeload 安装。
- Runner 总会初始化默认 AlexNet LPIPS，即使没有评估步骤也需要预训练权重。
- `gsplat/losses.py::ssim_loss` 有 PyTorch 回退实现，fused-ssim 非必需。
  本次不装 NCore、PPISP、fused-bilagrid、PLAS、torchpq、CuPy、nerfacc、文档/开发依赖或 FFmpeg。
  默认关闭压缩、视频和外观后处理，不使用这些可选功能。
- CUDA JIT 需要 GLM；本次只补齐 GLM，不下载只用于原生 C++ 测试的 GoogleTest。

## 安装

在 AutoDL 终端选择能导入现有 GPU torch 的 Python；需要时指定 `BASE_PYTHON=/完整路径/python`。
不要激活云端 CPU venv，不要运行 `pip install -r examples/requirements.txt`。

```bash
cd /root/autodl-tmp/gsplat-learning/gsplat
bash scripts/setup_autodl.sh
source /root/autodl-tmp/gsplat-autodl-state/activate.sh
```

默认创建 `/root/autodl-tmp/gsplat-autodl-venv`，以 `--system-site-packages` 复用兼容包。
新增安装只进入 venv；它是独立安装位置，**并不完全隔绝系统包**，系统包后续变动仍会影响它。
脚本拒绝不兼容的基础 torch、已有 torchvision 或 CUDA Toolkit，不替换系统 PyTorch/CUDA，
不使用 sudo，不装驱动。已有 torchvision 不匹配时先确认基础 Python，勿卸载系统包。

首次可指定 `ENV_DIR`、`STATE_DIR`（不含空格，放数据盘），重试保持一致。
`STATE_DIR` 保存 constraints、局部包版本清单、JIT 缓存、临时文件和 LPIPS 权重。
pip 禁用下载缓存，不复制整个 PyTorch；JIT 仅生成 sm_89，默认 `MAX_JOBS=2`，可在 source 前
`export MAX_JOBS=1` 降低编译内存压力。首次编译可能耗时较长、占数 GB；8GiB 预检只是最低安装余量。

不要求 GitHub clone：nerfview、缺失 GLM 均使用固定提交的 HTTPS codeload 归档。
Git checkout 的 GLM 版本从 HEAD gitlink 读取，无 Git 元数据的源码副本使用本次检查的版本。
已有完整 GLM 直接复用，不重置子模块；非空但不完整时停止，避免覆盖用户文件。
仍需能访问 PyPI、files.pythonhosted.org、codeload.github.com、download.pytorch.org
（可能重定向 download-r2.pytorch.org）。保持 TLS 校验，不使用 curl -k。
网络中断后修复并重复同一安装命令，完整包由 pip 复用；不自动清理用户数据/缓存。

## 数据与 GPU 验证

准备已有 COLMAP 重建，不是只有照片：

```text
/root/autodl-tmp/scene-small/
  images/              # 与 COLMAP 注册图像名称一致
  sparse/0/            # 或 sparse/
    cameras.bin        # 也支持同一重建的 COLMAP 文本格式
    images.bin
    points3D.bin
```

默认 SFM 初始化至少需 4 个有效点，并有非空训练/验证划分。建议使用小型、已去畸变的
PINHOLE/SIMPLE_PINHOLE 场景。先用 factor=1；如缩图，须让重建内参与之匹配。
当前 Parser 对 factor>1 要求已有 `images_<factor>/`，目录缺失时不会自动生成；
部分 JPEG 下采样路径还会生成额外 PNG，占更多空间。

```bash
source /root/autodl-tmp/gsplat-autodl-state/activate.sh
cd /root/autodl-tmp/gsplat-learning/gsplat
python scripts/verify_autodl.py --prepare-lpips \
  --data-dir /root/autodl-tmp/scene-small --data-factor 1
# 缓存权重后复查，不主动下载：
python scripts/verify_autodl.py --data-dir /root/autodl-tmp/scene-small
```

检查真实 CLI 导入、CUDA torchvision NMS、gsplat JIT、合成渲染前向/反向、SSIM、Adam 更新、
LPIPS 以及仓库 Parser/Dataset；失败时退出非零，打印阶段和 traceback。
LPIPS 首次下载 AlexNet 约 233MB。也可提前把可信来源的 `alexnet-owt-7be5be79.pth` 放入
`/root/autodl-tmp/gsplat-autodl-state/torch/hub/checkpoints/` 以复用已有权重。
不传 data-dir 会标记数据检查未执行。数据检查验证全部注册图像路径、解码首张训练/验证图像，
不会完整解码审计所有图像；合成渲染通过也不代表真实场景训练成功或收敛。

## 真实场景冒烟与训练

先跑 20 步，确认数据加载、loss、反向和最终 checkpoint 写入。
20 步低于默认增密起始步数，不验证完整 densification。使用单卡、默认策略、batch=1，
packed 是仓库已有的省显存选项。

```bash
source /root/autodl-tmp/gsplat-autodl-state/activate.sh
cd /root/autodl-tmp/gsplat-learning/gsplat/examples
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
  --data-dir /root/autodl-tmp/scene-small --data-factor 1 \
  --result-dir /root/autodl-tmp/results/scene-small-smoke \
  --disable-viewer --disable-video --packed \
  --max-steps 20 --eval-steps 20 --save-steps 20
```

确认退出码 0、loss 有限，结果目录 `ckpts/`、`stats/`、`renders/` 中有本次输出。
成功后换独立输出目录延长训练：

```bash
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
  --data-dir /root/autodl-tmp/scene-small --data-factor 1 \
  --result-dir /root/autodl-tmp/results/scene-small-7k \
  --disable-viewer --disable-video --packed \
  --max-steps 7000 --eval-steps 7000 --save-steps 7000
```

7k 是控制初次成本的试跑，不是标准 30k 完整结果，不据此宣称收敛。
默认不存 PLY、不生成视频、不缓存整套解码图像，仅保存少量 checkpoint。
24GB 是否够用取决于分辨率、点数和增密；不足时先降输入分辨率，或用已有 `--patch-size 512`
做小规模试跑，再评估质量影响。定期 `df -h /root/autodl-tmp` 与 `du -sh` 检查数据/结果/缓存，
不要盲目删除系统缓存。脚本不会自动下载训练数据或启动训练。

## 当前验证边界

云端仅检查 Bash/Python 语法、验证器 CLI 帮助和 CPU 环境拒绝路径；未验证 AutoDL 网络安装、
RTX 4090 编译、真实 COLMAP 数据和训练结果。以上命令需在 AutoDL 实际执行。
不修改或重新发布当前云环境配置。
