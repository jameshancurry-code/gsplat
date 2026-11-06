#!/usr/bin/env bash
# Minimal simple_trainer.py COLMAP environment; never install into the base Python.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_PYTHON="${BASE_PYTHON:-python}"
ENV_DIR="${ENV_DIR:-/root/autodl-tmp/gsplat-autodl-venv}"
STATE_DIR="${STATE_DIR:-/root/autodl-tmp/gsplat-autodl-state}"
if [[ "$ENV_DIR" == *[[:space:]]* || "$STATE_DIR" == *[[:space:]]* ]]; then
    echo 'ENV_DIR and STATE_DIR must not contain whitespace.' >&2; exit 1
fi
command -v curl >/dev/null
command -v c++ >/dev/null
"$BASE_PYTHON" - <<'PY'
import sys, torch
assert sys.version_info[:2] == (3, 12), sys.version
assert torch.__version__ == '2.8.0+cu128', torch.__version__
assert torch.version.cuda == '12.8', torch.version.cuda
assert torch.cuda.is_available(), 'Base Python cannot access CUDA GPU'
assert torch.cuda.get_device_capability(0) == (8, 9), 'Expected RTX 4090 / sm_89'
print('Reusing base torch:', torch.__version__, torch.__file__)
print('GPU:', torch.cuda.get_device_name(0))
PY
NVCC="${CUDA_HOME:+$CUDA_HOME/bin/}nvcc"
command -v "$NVCC" >/dev/null
"$NVCC" --version | grep -q 'release 12.8' || { echo 'CUDA Toolkit 12.8 required' >&2; exit 1; }
mkdir -p "$STATE_DIR"
export TMPDIR="$STATE_DIR/tmp"
mkdir -p "$TMPDIR"
# Leave headroom for JIT objects, LPIPS (~233 MB), data and checkpoints.
"$BASE_PYTHON" - "$STATE_DIR" <<'PY'
import shutil, sys
free = shutil.disk_usage(sys.argv[1]).free / 2**30
print(f'Data disk free: {free:.1f} GiB')
assert free >= 8, 'Need at least 8 GiB free before setup (not a training size guarantee)'
PY
if [[ ! -e "$ENV_DIR" ]]; then
    "$BASE_PYTHON" -m venv --system-site-packages "$ENV_DIR"
fi
[[ -f "$ENV_DIR/pyvenv.cfg" ]] || { echo 'ENV_DIR is not a venv' >&2; exit 1; }
grep -qi '^include-system-site-packages = true' "$ENV_DIR/pyvenv.cfg" || {
    echo 'Use a new ENV_DIR or a venv created with --system-site-packages' >&2; exit 1;
}
source "$ENV_DIR/bin/activate"
unset PIP_TARGET PIP_PREFIX PIP_USER
export PIP_CONFIG_FILE=/dev/null
export PIP_REQUIRE_VIRTUALENV=true PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
CONSTRAINTS="$STATE_DIR/constraints.txt"
# Prevent resolver from shadowing torch/CUDA packages with other versions.
python - "$CONSTRAINTS" <<'PY'
import importlib.metadata as m
import sys, torch
assert torch.__version__ == '2.8.0+cu128', torch.__version__
assert not torch.__file__.startswith(sys.prefix + '/'), 'Torch must be reused from base environment'
locked = {'torch': '2.8.0+cu128', 'torchvision': '0.23.0+cu128'}
try:
    locked['torchvision'] = m.version('torchvision')
except m.PackageNotFoundError:
    pass
for d in m.distributions():
    name = d.metadata.get('Name', '').lower().replace('_', '-')
    if name.startswith('nvidia-') or name in ('triton', 'torchaudio'):
        locked[name] = d.version
with open(sys.argv[1], 'w') as f:
    for name, version in sorted(locked.items()):
        f.write(f'{name}=={version}\n')
PY
# Matching torchvision is required by torchmetrics LPIPS. Install only if absent;
# refuse to hide an existing incompatible/broken torchvision.
if python -c 'import importlib.util; raise SystemExit(importlib.util.find_spec("torchvision") is None)'; then
    python - <<'PY'
import torchvision
import torch
assert torchvision.__version__.split('+')[0] == '0.23.0', torchvision.__version__
torchvision.ops.nms(torch.tensor([[0., 0., 1., 1.]], device='cuda'), torch.ones(1, device='cuda'), .5)
PY
else
    python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cu128 'torchvision==0.23.0+cu128'
fi
python -m pip install -c "$CONSTRAINTS" \
    'setuptools>=64' wheel ninja 'numpy>=2,<3' jaxtyping nvtx 'rich>=12' \
    'pycolmap>=3.10' scipy scikit-learn 'imageio>=2.37.2' tqdm \
    'tyro>=0.8.8,!=1.0.9,!=1.0.10' viser 'torchmetrics==1.8.2' \
    opencv-python-headless Pillow piexif tensorboard pyyaml matplotlib
# Fixed source revision from examples/requirements.txt; no Git clone required.
python -m pip install -c "$CONSTRAINTS" --no-build-isolation \
    'https://codeload.github.com/nerfstudio-project/nerfview/tar.gz/4538024fe0d15fd1a0e4d760f3695fc44ca72787'
GLM_DIR="$REPO_DIR/gsplat/cuda/csrc/third_party/glm"
if [[ ! -f "$GLM_DIR/glm/glm.hpp" ]]; then
    # Read the gitlink when available; use the inspected revision for source ZIPs.
    GLM_REF=2d4c4b4dd31fde06cfffad7915c2b3006402322f
    if git -C "$REPO_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        GLM_REF="$(git -C "$REPO_DIR" rev-parse HEAD:gsplat/cuda/csrc/third_party/glm)"
    fi
    [[ "$GLM_REF" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid GLM gitlink' >&2; exit 1; }
    # Never overwrite a populated but incomplete user directory.
    if [[ -d "$GLM_DIR" ]] && [[ -n "$(ls -A "$GLM_DIR")" ]]; then
        echo "Incomplete nonempty GLM directory: $GLM_DIR; inspect it before retrying." >&2; exit 1
    fi
    GLM_TMP="$(mktemp -d "$TMPDIR/glm.XXXXXX")"
    trap 'rm -rf -- "$GLM_TMP"' EXIT
    curl --fail --location --retry 3 --connect-timeout 30 \
        "https://codeload.github.com/g-truc/glm/tar.gz/$GLM_REF" -o "$GLM_TMP/glm.tar.gz"
    python - "$GLM_TMP" "$GLM_DIR" <<'PY'
import pathlib, shutil, sys, tarfile
src, dst = map(pathlib.Path, sys.argv[1:])
with tarfile.open(src / 'glm.tar.gz') as archive:
    archive.extractall(src / 'unpacked', filter='data')
roots = list((src / 'unpacked').iterdir())
assert len(roots) == 1 and (roots[0] / 'glm/glm.hpp').is_file()
dst.mkdir(parents=True, exist_ok=True)
shutil.copytree(roots[0], dst, dirs_exist_ok=True)
PY
    rm -rf -- "$GLM_TMP"
    trap - EXIT
fi
cd "$REPO_DIR"
# Metadata build only; actual CUDA compilation happens during verification.
BUILD_NO_CUDA=1 python -m pip install --no-deps --no-build-isolation -e .
# A sourced helper is necessary: exports in this installer cannot reach its parent.
python - "$STATE_DIR" "$ENV_DIR" "$(dirname "$(dirname "$(readlink -f "$(command -v "$NVCC")")")")" <<'PY'
from pathlib import Path
import shlex, sys
state, venv, cuda = sys.argv[1:]
q = shlex.quote
Path(state, 'activate.sh').write_text(
    f'source {q(venv + "/bin/activate")}\n'
    f'export CUDA_HOME={q(str(Path(cuda).resolve()))}\n'
    'export PATH="$CUDA_HOME/bin:$PATH"\n'
    'export TORCH_CUDA_ARCH_LIST=8.9\n'
    'export MAX_JOBS="${MAX_JOBS:-2}"\n'
    f'export TORCH_EXTENSIONS_DIR={q(state + "/torch_extensions")}\n'
    f'export TORCH_HOME={q(state + "/torch")}\n'
    f'export TMPDIR={q(state + "/tmp")}\n'
    'export PIP_NO_CACHE_DIR=1 PIP_REQUIRE_VIRTUALENV=true\n'
)
PY
python -m pip freeze --local > "$STATE_DIR/installed-local.txt"
echo "Installed. Next: source $STATE_DIR/activate.sh"
echo "Then: python $REPO_DIR/scripts/verify_autodl.py --prepare-lpips --data-dir /path/to/scene"
echo 'This installer has NOT validated GPU compilation or training.'
