#!/usr/bin/env bash
# ==============================================================================
# eval_kaggle_qwen3vl.sh - Kaggle runner for Qwen3-VL family only (CSI-Bench)
#
#   !git clone https://github.com/shahedmomenzadeh/Catarct-Irregularity.git
#   %cd Catarct-Irregularity
#   %env HF_TOKEN=hf_xxxx   (dataset is public; optional)
#   !bash eval_kaggle_qwen3vl.sh
#
#   Env overrides: FRAMES=64 (default), LIMIT=2 (smoke test)
# ==============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/results"
DATASET_ROOT="${SCRIPT_DIR}/Irregularity_dataset"
mkdir -p "${OUTPUT_DIR}"

HF_DATASET="${HF_DATASET:-shahedm2001/Catarct-Irregularity}"
HF_TOKEN="${HF_TOKEN:-}"
FRAMES="${FRAMES:-64}"
MAX_PIXELS="${MAX_PIXELS:-307200}"
MIN_PIXELS="${MIN_PIXELS:-100352}"
LIMIT_ARG="${LIMIT:-}"
HF_HUB_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"

MODELS=(
  "Qwen/Qwen3-VL-2B-Instruct"
  "Qwen/Qwen3-VL-4B-Instruct"
  "Qwen/Qwen3-VL-8B-Instruct"
  "shahedm2001/qwen3-vl-2b-cataract-sft-stage2"
  "shahedm2001/qwen3-vl-2b-cataract-grpo"
)

N_OK=0; N_SKIP=0; N_FAIL=0

log()  { echo "[qwen3vl] $*"; }
disk_status() {
  echo "--- disk / gpu ---"
  df -h "${SCRIPT_DIR}" 2>/dev/null || df -h
  du -sh "${OUTPUT_DIR}" "${DATASET_ROOT}" 2>/dev/null || true
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv 2>/dev/null || echo "(nvidia-smi unavailable)"
  echo "------------------"
}

make_tag() {
  local base slug
  base="$(basename "$2")"
  slug="$(echo "$base" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
  echo "${1}_${slug}"
}

expected_cases() {
  local n
  n="$(ls -1 "${DATASET_ROOT}"/case_*.mp4 2>/dev/null | wc -l | tr -d ' ')"
  if [ -z "$n" ] || [ "$n" = "0" ]; then echo "41"; else echo "$n"; fi
}

is_finished() {
  local summary="${OUTPUT_DIR}/${1}_summary.json"
  local scores="${OUTPUT_DIR}/${1}_scores.jsonl"
  [ -f "$summary" ] || return 1
  [ -f "$scores" ] || return 1
  local total lines
  total="$(python3 -c "import json;print(json.load(open('$summary')).get('total_cases_evaluated',0))" 2>/dev/null)" || return 1
  lines="$(wc -l < "$scores" | tr -d ' ')"
  [ "$total" = "$2" ] && [ "$lines" = "$2" ]
}

download_dataset() {
  if ls "${DATASET_ROOT}"/case_*.mp4 1>/dev/null 2>&1 && ls "${DATASET_ROOT}"/case_*.jsonl 1>/dev/null 2>&1; then
    log "Dataset present -> skip download."
    return 0
  fi
  log "Installing huggingface_hub..."
  pip install -q --no-cache-dir huggingface_hub
  log "Downloading ${HF_DATASET} ..."
  if [ -n "$HF_TOKEN" ]; then
    HF_TOKEN="$HF_TOKEN" python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$HF_DATASET', repo_type='dataset', local_dir='$DATASET_ROOT', token='$HF_TOKEN')
"
  else
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$HF_DATASET', repo_type='dataset', local_dir='$DATASET_ROOT')
"
  fi
}

cleanup_weights() {
  local slug="${1//\//--}"
  log "Cleaning weights for $1 ..."
  rm -rf "${HF_HUB_CACHE}/models--${slug}" 2>/dev/null || true
  rm -rf "${HF_HUB_CACHE}/.locks" 2>/dev/null || true
  rm -rf /tmp/* 2>/dev/null || true
  pip cache purge 2>/dev/null || true
  disk_status
}

zip_model() {
  (
    cd "$OUTPUT_DIR" || exit 0
    rm -f "${1}.zip"
    if ls "${1}"_responses.jsonl "${1}"_scores.jsonl "${1}"_summary.json "${1}"_summary.md 1>/dev/null 2>&1; then
      if command -v zip >/dev/null 2>&1; then
        zip -j -q "${1}.zip" "${1}"_responses.jsonl "${1}"_scores.jsonl "${1}"_summary.json "${1}"_summary.md 2>/dev/null || true
      else
        python3 -c "
import zipfile, os
z = zipfile.ZipFile('${1}.zip', 'w', zipfile.ZIP_DEFLATED)
for f in ['${1}_responses.jsonl','${1}_scores.jsonl','${1}_summary.json','${1}_summary.md']:
    if os.path.exists(f): z.write(f, os.path.basename(f))
z.close()
"
      fi
      log "Zipped ${1}.zip"
    fi
  )
}

zip_all() {
  (
    cd "$SCRIPT_DIR" || exit 0
    rm -f results.zip
    if command -v zip >/dev/null 2>&1; then
      zip -r -q results.zip results -x "results/*.zip" 2>/dev/null || zip -r -q results.zip results 2>/dev/null || true
    else
      python3 -c "
import zipfile, os
z = zipfile.ZipFile('results.zip', 'w', zipfile.ZIP_DEFLATED)
for root, _, files in os.walk('results'):
    for f in files:
        if f.endswith('.zip'): continue
        p = os.path.join(root, f)
        z.write(p, p)
z.close()
"
    fi
    log "Updated results.zip"
  )
}

log "Qwen3-VL Kaggle runner | frames=${FRAMES}"
disk_status
download_dataset
log "Expected cases: $(expected_cases)"

log "Installing requirements (once)..."
pip install -q --no-cache-dir -r "${SCRIPT_DIR}/requirements/requirements-qwen3vl.txt"
pip install -q --no-cache-dir scipy scikit-learn 2>/dev/null || true
pip cache purge 2>/dev/null || true

for MODEL_ID in "${MODELS[@]}"; do
  TAG="$(make_tag qwen3vl "$MODEL_ID")"
  EXP="$(expected_cases)"
  if is_finished "$TAG" "$EXP"; then
    log "SKIP finished: $TAG"
    N_SKIP=$((N_SKIP + 1))
    zip_model "$TAG"
    continue
  fi
  log "RUN $MODEL_ID (tag=$TAG) ..."
  disk_status
  limit_flag=()
  if [ -n "$LIMIT_ARG" ]; then limit_flag=(--limit "$LIMIT_ARG"); fi
  python3 "${SCRIPT_DIR}/main.py" \
    --model-family qwen3vl \
    --model-id "$MODEL_ID" \
    --dataset-root "$DATASET_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    "${limit_flag[@]}" \
    --max-frames "$FRAMES" --max-pixels "$MAX_PIXELS" --min-pixels "$MIN_PIXELS"
  rc=$?
  if [ $rc -eq 0 ]; then log "DONE $TAG"; N_OK=$((N_OK + 1)); else log "FAIL $TAG (exit=$rc)"; N_FAIL=$((N_FAIL + 1)); fi
  cleanup_weights "$MODEL_ID"
  zip_model "$TAG"
  zip_all
done

zip_all
disk_status
log "Finished: ok=${N_OK} skipped=${N_SKIP} failed=${N_FAIL}"
