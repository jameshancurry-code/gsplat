#!/usr/bin/env python3
"""Fail-closed AutoDL checks. GPU success can only be established on that host."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback

REPO = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def runtime():
    import torch

    require(sys.prefix != sys.base_prefix, 'Activate the independent venv first')
    require(sys.version_info[:2] == (3, 12), f'Expected Python 3.12: {sys.version}')
    require(torch.__version__ == '2.8.0+cu128', f'Unexpected torch: {torch.__version__}')
    import torchvision
    from torch.utils.cpp_extension import CUDA_HOME

    require(torchvision.__version__.split('+')[0] == '0.23.0', f'Unexpected torchvision: {torchvision.__version__}')
    require(torch.version.cuda == '12.8', f'Unexpected torch CUDA: {torch.version.cuda}')
    require(torch.cuda.is_available(), 'CUDA is unavailable')
    require(torch.cuda.get_device_capability(0) == (8, 9), 'Expected sm_89 / RTX 4090')
    require(CUDA_HOME is not None, 'CUDA_HOME/toolkit not found')
    nvcc = Path(CUDA_HOME) / 'bin/nvcc'
    version = subprocess.check_output([str(nvcc), '--version'], text=True)
    require(re.search(r'release 12\.8\b', version), version)
    require(shutil.which('c++'), 'C++ compiler missing')
    require(shutil.which('ninja'), 'ninja missing')
    require(os.environ.get('TORCH_CUDA_ARCH_LIST') == '8.9', 'Source the generated activate.sh (TORCH_CUDA_ARCH_LIST=8.9)')
    import gsplat
    require(Path(gsplat.__file__).resolve().is_relative_to(REPO), f'Wrong gsplat checkout: {gsplat.__file__}')
    require((REPO / 'gsplat/cuda/csrc/third_party/glm/glm/glm.hpp').is_file(), 'GLM headers missing')
    # torchvision operators expose binary incompatibility that version strings miss.
    torchvision.ops.nms(torch.tensor([[0., 0., 1., 1.]], device='cuda'), torch.ones(1, device='cuda'), .5)
    print(json.dumps({
        'python': sys.version.split()[0], 'torch': torch.__version__,
        'torch_path': torch.__file__, 'torchvision': torchvision.__version__,
        'gpu': torch.cuda.get_device_name(0),
        'vram_GiB': round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        'nvcc': version.strip(), 'gsplat': gsplat.__file__,
        'extension_cache': os.environ.get('TORCH_EXTENSIONS_DIR'),
    }, indent=2))


def trainer_imports():
    # This runs the actual CLI import chain, including viewer, COLMAP, metrics,
    # compression and scene/stage imports; it does not construct Runner.
    subprocess.run([sys.executable, 'simple_trainer.py', 'default', '--help'],
                   cwd=REPO / 'examples', check=True, stdout=subprocess.DEVNULL)


def lpips(prepare):
    import torch
    from torchvision.models import AlexNet_Weights
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    filename = AlexNet_Weights.IMAGENET1K_V1.url.rsplit('/', 1)[-1]
    weight = Path(torch.hub.get_dir()) / 'checkpoints' / filename
    require(prepare or weight.is_file(),
            f'LPIPS needs {weight}. Run with --prepare-lpips to download official AlexNet weights (~233 MB).')
    metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()
    image = torch.rand(1, 3, 64, 64, device='cuda')
    with torch.no_grad():
        value = metric(image, image)
    require(torch.isfinite(value).all().item() and abs(value.item()) < 1e-5,
            f'LPIPS identity check failed: {value}')
    torch.cuda.synchronize()
    print(f'LPIPS initialized and exercised; pretrained weights: {weight}')


def render_backward():
    import torch
    from gsplat import rasterization
    from gsplat.losses import ssim_loss

    torch.manual_seed(42)
    device = 'cuda'
    means = torch.tensor([[0., 0., 3.], [.15, .05, 3.2], [-.15, -.05, 2.8]], device=device, requires_grad=True)
    quats = torch.tensor([[1., 0., 0., 0.]] * 3, device=device, requires_grad=True)
    scales = torch.full((3, 3), .12, device=device, requires_grad=True)
    opacity = torch.full((3,), .6, device=device, requires_grad=True)
    colors = torch.full((3, 3), .4, device=device, requires_grad=True)
    params = [means, quats, scales, opacity, colors]
    optimizer = torch.optim.Adam(params, lr=1e-3)
    before = colors.detach().clone()
    image, alpha, _ = rasterization(
        means=means, quats=quats, scales=scales, opacities=opacity, colors=colors,
        viewmats=torch.eye(4, device=device)[None],
        Ks=torch.tensor([[[50., 0., 32.], [0., 50., 32.], [0., 0., 1.]]], device=device),
        width=64, height=64, packed=True,
    )
    require(image.shape == (1, 64, 64, 3), f'Unexpected render shape: {image.shape}')
    require(torch.isfinite(image).all().item() and alpha.max().item() > 0, 'Empty or nonfinite render')
    loss = image.square().mean() + .2 * ssim_loss(image.permute(0, 3, 1, 2), torch.zeros(1, 3, 64, 64, device=device))
    loss.backward()
    for name, value in zip(('means', 'quats', 'scales', 'opacity', 'colors'), params):
        require(value.grad is not None and torch.isfinite(value.grad).all().item(), f'Invalid {name} gradient')
    require(colors.grad.abs().sum().item() > 0, 'Zero color gradient')
    optimizer.step()
    torch.cuda.synchronize()
    require(not torch.equal(before, colors.detach()), 'Adam did not update colors')
    print(f'CUDA rasterization forward/backward + SSIM + Adam passed; loss={loss.item():.6g}')


def dataset(path, factor, test_every):
    import numpy as np
    sys.path.insert(0, str(REPO / 'examples'))
    from datasets.colmap import Dataset, Parser

    require(factor >= 1 and test_every >= 2, 'factor >= 1 and test-every >= 2 required')
    parser = Parser(data_dir=str(path), factor=factor, normalize=True,
                    test_every=test_every, load_exposure=True)
    require(len(parser.points) >= 4, 'SFM initialization requires >=4 points for knn(points, 4)')
    require(np.isfinite(parser.points).all(), 'Nonfinite COLMAP points')
    for image in parser.image_paths:
        require(Path(image).is_file(), f'Missing registered image: {image}')
    for split in ('train', 'val'):
        data = Dataset(parser, split=split)
        require(len(data) > 0, f'Empty {split} split')
        sample = data[0]
        require(sample['image'].numel() > 0, f'Cannot decode {split} image')
        require(np.isfinite(sample['image'].numpy()).all(), f'Nonfinite {split} image')
        print(f'COLMAP {split}: {len(data)} images; first shape={tuple(sample["image"].shape)}')
    print(f'COLMAP: {len(parser.points)} points; factor={factor}; camera models parsed by repository Parser')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--prepare-lpips', action='store_true', help='Allow official pretrained AlexNet download')
    cli.add_argument('--data-dir', type=Path, help='COLMAP scene root (images/ and sparse/0/)')
    cli.add_argument('--data-factor', type=int, default=1)
    cli.add_argument('--test-every', type=int, default=8)
    args = cli.parse_args()
    checks = [('runtime', runtime), ('trainer imports/CLI', trainer_imports),
              ('CUDA forward/backward/optimizer', render_backward),
              ('LPIPS', lambda: lpips(args.prepare_lpips))]
    if args.data_dir:
        checks.append(('COLMAP', lambda: dataset(args.data_dir, args.data_factor, args.test_every)))
    failures = []
    for name, check in checks:
        print(f'\n[CHECK] {name}', flush=True)
        try:
            check()
            print(f'[PASS] {name}', flush=True)
        except Exception:
            failures.append(name)
            traceback.print_exc()
            # Invalid runtime is not a useful basis for expensive builds/downloads.
            if name == 'runtime':
                break
    if failures:
        print(f'FAILED: {", ".join(failures)}', file=sys.stderr)
        return 1
    if not args.data_dir:
        print('[UNRUN] COLMAP data checks: supply --data-dir before training.')
    print('Selected checks passed on THIS host. This is not a full training/convergence test.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
