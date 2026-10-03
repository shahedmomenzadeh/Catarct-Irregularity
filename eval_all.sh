#!/usr/bin/env bash
# ==============================================================================
# Cataract Surgery Irregularity Benchmark (CSI-Bench) Evaluation Runner
# Dispatches model families: API (Gemini/OpenAI), Qwen3-VL, HuluMed, Lingshu, Mage-VL
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/results"
DATASET_ROOT="${SCRIPT_DIR}/Irregularity_dataset"
mkdir -p "${OUTPUT_DIR}"

MODEL_FAMILY="${1:-api}"

echo "==========================================================================="
echo "  CATARACT SURGERY IRREGULARITY BENCHMARK RUNNER (CSI-Bench)"
echo "  3-Class Classification: Normal, Lens Irregularity, Pupil Contraction"
echo "  Protocol: 100% Deterministic, Silent Video, One-by-One Evaluation"
echo "==========================================================================="
echo "Model Family Selection : ${MODEL_FAMILY}"
echo "Output Directory       : ${OUTPUT_DIR}"
echo "Dataset Root           : ${DATASET_ROOT}"
echo ""

if [[ "${MODEL_FAMILY}" == "--dry-run" || "${MODEL_FAMILY}" == "dry-run" ]]; then
    python3 "${SCRIPT_DIR}/main.py" --dry-run --dataset-root "${DATASET_ROOT}"
    exit 0
fi

# Function: run API models
run_api() {
    echo "---------------------------------------------------------------------------"
    echo "[API Inference] ag/gemini-3.8-flash via 9router"
    echo "---------------------------------------------------------------------------"
    VENV_API="${SCRIPT_DIR}/.venv-api"
    if [ ! -d "${VENV_API}" ]; then
        uv venv "${VENV_API}" --python 3.11
    fi
    PY_BIN="${VENV_API}/bin/python"
    [ ! -f "${PY_BIN}" ] && PY_BIN="${VENV_API}/Scripts/python.exe"
    uv pip install --python "${PY_BIN}" -r "${SCRIPT_DIR}/requirements/requirements-api.txt"
    "${PY_BIN}" "${SCRIPT_DIR}/main.py" \
        --model-family api \
        --model-id "ag/gemini-3.8-flash" \
        --api-base-url "http://localhost:20128/v1" \
        --api-timeout 600 \
        --api-retries 5 \
        --output-dir "${OUTPUT_DIR}"
}

# Function: run Qwen3-VL
run_qwen3vl() {
    echo "---------------------------------------------------------------------------"
    echo "[Qwen3-VL Inference] 2B, 4B, 8B"
    echo "---------------------------------------------------------------------------"
    VENV_QWEN="${SCRIPT_DIR}/.venv-qwen3vl"
    if [ ! -d "${VENV_QWEN}" ]; then
        uv venv "${VENV_QWEN}" --python 3.12
    fi
    PY_BIN="${VENV_QWEN}/bin/python"
    [ ! -f "${PY_BIN}" ] && PY_BIN="${VENV_QWEN}/Scripts/python.exe"
    uv pip install --python "${PY_BIN}" -r "${SCRIPT_DIR}/requirements/requirements-qwen3vl.txt"
    "${PY_BIN}" "${SCRIPT_DIR}/main.py" --model-family qwen3vl --model-id "Qwen/Qwen3-VL-2B-Instruct" --output-dir "${OUTPUT_DIR}"
    "${PY_BIN}" "${SCRIPT_DIR}/main.py" --model-family qwen3vl --model-id "Qwen/Qwen3-VL-4B-Instruct" --output-dir "${OUTPUT_DIR}"
    "${PY_BIN}" "${SCRIPT_DIR}/main.py" --model-family qwen3vl --model-id "Qwen/Qwen3-VL-8B-Instruct" --output-dir "${OUTPUT_DIR}"
}

# Function: run HuluMed
run_hulumed() {
    echo "---------------------------------------------------------------------------"
    echo "[HuluMed Inference] 7B & 4B"
    echo "---------------------------------------------------------------------------"
    VENV_HULU="${SCRIPT_DIR}/.venv-hulumed"
    if [ ! -d "${VENV_HULU}" ]; then
        uv venv "${VENV_HULU}" --python 3.12
    fi
    PY_BIN="${VENV_HULU}/bin/python"
    [ ! -f "${PY_BIN}" ] && PY_BIN="${VENV_HULU}/Scripts/python.exe"
    uv pip install --python "${PY_BIN}" -r "${SCRIPT_DIR}/requirements/requirements-hulumed.txt"
    "${PY_BIN}" "${SCRIPT_DIR}/main.py" --model-family hulumed --model-id "ZJU-AI4H/Hulu-Med-7B" --frame-size 224 --output-dir "${OUTPUT_DIR}"
    "${PY_BIN}" "${SCRIPT_DIR}/main.py" --model-family hulumed --model-id "ZJU-AI4H/Hulu-Med-4B" --frame-size 224 --output-dir "${OUTPUT_DIR}"
}

# Function: run Lingshu
run_lingshu() {
    echo "---------------------------------------------------------------------------"
    echo "[Lingshu Inference] Lingshu-7B"
    echo "---------------------------------------------------------------------------"
    VENV_QWEN="${SCRIPT_DIR}/.venv-qwen3vl"
    if [ ! -d "${VENV_QWEN}" ]; then
        uv venv "${VENV_QWEN}" --python 3.12
    fi
    PY_BIN="${VENV_QWEN}/bin/python"
    [ ! -f "${PY_BIN}" ] && PY_BIN="${VENV_QWEN}/Scripts/python.exe"
    uv pip install --python "${PY_BIN}" -r "${SCRIPT_DIR}/requirements/requirements-qwen3vl.txt"
    "${PY_BIN}" "${SCRIPT_DIR}/main.py" --model-family lingshu --model-id "lingshu-medical-mllm/Lingshu-7B" --output-dir "${OUTPUT_DIR}"
}

# Function: run Mage-VL
run_magevl() {
    echo "---------------------------------------------------------------------------"
    echo "[Mage-VL Inference] microsoft/Mage-VL"
    echo "---------------------------------------------------------------------------"
    VENV_MAGE="${SCRIPT_DIR}/.venv-magevl"
    if [ ! -d "${VENV_MAGE}" ]; then
        uv venv "${VENV_MAGE}" --python 3.12
    fi
    PY_BIN="${VENV_MAGE}/bin/python"
    [ ! -f "${PY_BIN}" ] && PY_BIN="${VENV_MAGE}/Scripts/python.exe"
    uv pip install --python "${PY_BIN}" -r "${SCRIPT_DIR}/requirements/requirements-magevl.txt"
    "${PY_BIN}" "${SCRIPT_DIR}/main.py" --model-family mage_vl --model-id "microsoft/Mage-VL" --output-dir "${OUTPUT_DIR}"
}

case "${MODEL_FAMILY}" in
    api|openai_api|gemini)
        run_api
        ;;
    qwen3vl)
        run_qwen3vl
        ;;
    hulumed)
        run_hulumed
        ;;
    lingshu)
        run_lingshu
        ;;
    mage_vl)
        run_magevl
        ;;
    all)
        run_api
        run_hulumed
        run_qwen3vl
        run_lingshu
        run_magevl
        ;;
    *)
        echo "Unknown model family: ${MODEL_FAMILY}"
        exit 1
        ;;
esac

echo "Evaluation pipeline finished successfully."
