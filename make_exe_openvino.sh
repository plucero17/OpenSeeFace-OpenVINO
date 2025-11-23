#!/bin/bash
set -euo pipefail

VENV_DIR="venv-openvino"
PYTHON_BINARY=${PYTHON_BINARY:-}
if [[ -z "${PYTHON_BINARY}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BINARY=$(command -v python3)
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BINARY=$(command -v python)
    else
        echo "Python 3 must be available on your path. Exiting." >&2
        exit 1
    fi
fi

echo "Creating isolated venv in ${VENV_DIR}"
"$PYTHON_BINARY" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip wheel
pip install "numpy==1.23.0" "opencv-python==4.5.4.60" pillow \
    onnxruntime openvino pyinstaller

SITE_PACKAGES=$(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")
OV_LIB_DIR="$SITE_PACKAGES/openvino/libs"
if [[ ! -d "$OV_LIB_DIR" ]]; then
    echo "Could not find OpenVINO libraries under $OV_LIB_DIR" >&2
    exit 1
fi

echo "Building OpenVINO binary"
pyinstaller --clean --onedir facetracker.py \
    --name facetracker_openvino \
    --add-data "ov-models:ov-models" \
    --add-data "models:models" \
    --add-binary "${OV_LIB_DIR}/*.so:openvino/libs"

echo "Done. Artifacts are available under dist/facetracker_openvino"
