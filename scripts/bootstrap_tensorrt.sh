#!/usr/bin/env bash
# DGX Spark TensorRT bootstrap without sudo.
#
# The node exposes NVIDIA's Ubuntu 24.04 SBSA APT repository, but Developer has
# no passwordless sudo. Download the exact CUDA-13.0 TensorRT packages and
# extract them under ~/local instead of mutating the system package database.
# ONNX exporter dependencies are installed into the existing visual venv.
#
# Run on Spark as a background task; this may take longer than one minute.
set -euo pipefail

TRT_APT_VERSION="${TRT_APT_VERSION:-10.14.1.48-1+cuda13.0}"
TRT_SHORT_VERSION="${TRT_SHORT_VERSION:-10.14.1.48}"
TRT_ROOT="${TRT_ROOT:-$HOME/local/tensorrt-$TRT_SHORT_VERSION}"
DEB_DIR="${DEB_DIR:-$HOME/tmp/tensorrt-debs-$TRT_SHORT_VERSION}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/venv/bin/python}"
PYPI_MIRROR="${PYPI_MIRROR:-https://mirrors.aliyun.com/pypi/simple/}"

packages=(
  libnvinfer10
  libnvinfer-plugin10
  libnvinfer-vc-plugin10
  libnvinfer-lean10
  libnvinfer-dispatch10
  libnvonnxparsers10
  libnvinfer-bin
  python3-libnvinfer
)

mkdir -p "$DEB_DIR" "$TRT_ROOT"
cd "$DEB_DIR"

for package in "${packages[@]}"; do
  if ! compgen -G "${package}_*.deb" >/dev/null; then
    apt-get download "${package}=${TRT_APT_VERSION}" \
      || apt-get download "${package}=${TRT_APT_VERSION}"
  fi
done

for archive in ./*.deb; do
  dpkg-deb -x "$archive" "$TRT_ROOT"
done

"$PYTHON_BIN" -m pip install -U \
  "onnx==1.22.0" \
  "onnxscript==0.7.1" \
  -i "$PYPI_MIRROR" \
  || "$PYTHON_BIN" -m pip install -U \
       "onnx==1.22.0" \
       "onnxscript==0.7.1" \
       -i "$PYPI_MIRROR"

TRTEXEC="$(find "$TRT_ROOT" -type f -name trtexec -print -quit)"
[ -n "$TRTEXEC" ] || { echo "TRTEXEC_NOT_FOUND under $TRT_ROOT" >&2; exit 3; }

TRT_LIB="$TRT_ROOT/usr/lib/aarch64-linux-gnu"
TRT_PYTHON=""
for candidate in "$TRT_ROOT"/usr/lib/python*/dist-packages; do
  if [ -d "$candidate/tensorrt" ]; then
    TRT_PYTHON="$candidate"
    break
  fi
done
[ -d "$TRT_LIB" ] || { echo "TRT_LIB_NOT_FOUND: $TRT_LIB" >&2; exit 3; }
[ -n "$TRT_PYTHON" ] || { echo "TRT_PYTHON_NOT_FOUND under $TRT_ROOT" >&2; exit 3; }

LD_LIBRARY_PATH="$TRT_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$TRTEXEC" --version
PYTHONPATH="$TRT_PYTHON${PYTHONPATH:+:$PYTHONPATH}" \
LD_LIBRARY_PATH="$TRT_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$PYTHON_BIN" -c 'import onnx, onnxscript, tensorrt; print("onnx", onnx.__version__, "onnxscript", onnxscript.__version__, "tensorrt", tensorrt.__version__)'

echo "TRT_ROOT=$TRT_ROOT"
echo "TRT_LIB=$TRT_LIB"
echo "TRT_PYTHON=$TRT_PYTHON"
echo "TRTEXEC=$TRTEXEC"
echo "BOOTSTRAP_TENSORRT_DONE"
