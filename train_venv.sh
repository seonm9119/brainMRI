#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQUIREMENTS_FILE="${PROJECT_ROOT}/requirements.txt"

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
  echo "requirements file not found: ${REQUIREMENTS_FILE}" >&2
  exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "creating venv: ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip wheel "setuptools<82"
"${VENV_DIR}/bin/python" -m pip install -r "${REQUIREMENTS_FILE}"

"${VENV_DIR}/bin/python" - <<'PY'
import sys

print(f"python={sys.executable}")
try:
    import torch
except Exception as exc:
    print(f"torch_import_error={exc}")
else:
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device_name={torch.cuda.get_device_name(0)}")
PY

echo "venv ready: ${VENV_DIR}"
