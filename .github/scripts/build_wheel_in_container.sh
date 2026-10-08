#!/bin/bash
# Builds one gsplat wheel inside a CUDA or ROCm dev container. Started by .github/workflows/wheels.yml with
# the source mounted at /src, the wheel directory at /out and uv at /usr/local/bin/uv.
# Usage: build_wheel_in_container.sh cuda|rocm
set -euxo pipefail
BACKEND=$1
TORCH=2.14.1
export UV_CACHE_DIR=/tmp/uv-cache MAX_JOBS=${MAX_JOBS:-4}

case "$BACKEND" in
  cuda)
    export GSPLAT_LOCAL_VERSION=cu126 CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
    uv venv --python 3.12 /tmp/venv
    uv pip install --python /tmp/venv/bin/python --index-url https://download.pytorch.org/whl/cu126 "torch==$TORCH"
    ;;
  rocm)
    export GSPLAT_LOCAL_VERSION=rocm714 ROCM_HOME=/opt/rocm PYTORCH_ROCM_ARCH=gfx1151
    uv venv --python /usr/bin/python3 /tmp/venv
    uv pip install --python /tmp/venv/bin/python --index-url https://download.pytorch.org/whl/rocm7.14 \
      "torch==$TORCH+rocm7.14"
    ;;
  *) echo "unknown backend: $BACKEND" >&2; exit 2 ;;
esac
uv pip install --python /tmp/venv/bin/python numpy==1.26.4 setuptools==74.0.0 wheel ninja jaxtyping rich
export PATH=/tmp/venv/bin:$PATH

# The extension links against torch's libraries, so the torch it builds against must be the one it runs with.
python - "$BACKEND" <<'EOF'
import sys, torch
assert sys.version_info[:2] == (3, 12), sys.version
assert torch.__version__.split("+")[0] == "2.14.1", torch.__version__
if sys.argv[1] == "cuda":
    assert torch.version.cuda == "12.6" and torch.version.hip is None, (torch.version.cuda, torch.version.hip)
else:
    assert torch.version.hip and torch.version.hip.startswith("7.14"), torch.version.hip
print("torch", torch.__version__, "cuda", torch.version.cuda, "hip", torch.version.hip)
EOF

# Build from a copy, so hipify and build/ write nothing into the mounted checkout.
cp -a /src /tmp/src
cd /tmp/src
start=$(date +%s)
python setup.py bdist_wheel -d /tmp/dist
echo "build took $(( $(date +%s) - start )) s"

WHEEL=gsplat-1.4.0+$GSPLAT_LOCAL_VERSION-cp312-cp312-linux_x86_64.whl
ls /tmp/dist
[ "$(ls /tmp/dist)" = "$WHEEL" ] || { echo "expected exactly $WHEEL" >&2; exit 1; }
python -m zipfile -l "/tmp/dist/$WHEEL" | grep -q 'gsplat/csrc.so' || { echo "gsplat/csrc.so missing" >&2; exit 1; }
cp "/tmp/dist/$WHEEL" /out/
