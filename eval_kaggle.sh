#!/usr/bin/env bash
# ==============================================================================
# eval_kaggle.sh - Kaggle runner for Cataract Irregularity Benchmark (CSI-Bench)
#
# Notebook usage:
#   !git clone https://github.com/shahedmomenzadeh/Catarct-Irregularity.git
#   %cd Catarct-Irregularity
#   # set token (use %env so it persists, !export does NOT persist across cells):
#   # %env HF_TOKEN=hf_xxxx
#   !bash eval_kaggle.sh [all|qwen3vl|hulumed|lingshu]
#
# What it does:
#   1. Downloads dataset from https://huggingface.co/datasets/shahedm2001/Catarct-Irregularity
#   2. Runs local inference (no api/gemini), default 64 frames per model (overridable)
#   3. pip-only env management: reinstalls reqs on family switch, cleans weights after each model
#   4. Skips fully-finished models (already in git-cloned results/), resumes partial ones
#   5. Zips per-model results + full results.zip after each model
# ==============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/results"
DATASET_ROOT="${SCRIPT_DIR}/Irregularity_dataset"
mkdir -p "${OUTPUT_DIR}"

# ---------------- Config (env-overridable) ------------------------------------
HF_DATASET="${HF_DATASET:-shahedm2001/Catarct-Irregularity}"
HF_TOKEN="${HF_TOKEN:-}"                       # optional; dataset is public
FAMILY_ARG="${1:-all}"

FRAMES="${FRAMES:-64}"                          # global default for all models
QWEN3_FRAMES="${QWEN3_FRAMES:-$FRAMES}"
HULU_FRAMES="${HULU_FRAMES:-$FRAMES}"
LINGSHU_FRAMES="${LINGSHU_FRAMES:-$FRAMES}"

HULU_SIZE="${HULU_SIZE:-224}"
QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS:-307200}"
QWEN_MIN_PIXELS="${QWEN_MIN_PIXELS:-100352}"
FPS_DEFAULT="${FPS_DEFAULT:-1.0}"
LIMIT_ARG="${LIMIT:-}"                          # e.g. LIMIT=2 for smoke test
HF_HUB_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"

QWEN3_MODELS=(
  "Qwen/Qwen3-VL-2B-Instruct"
  "Qwen/Qwen3-VL-4B-Instruct"
  "Qwen/Qwen3-VL-8B-Instruct"
)
HULU_MODELS=(
  "ZJU-AI4H/Hulu-Med-7B"
  "ZJU-AI4H/Hulu-Med-4B"
)
LINGSHU_MODELS=(
  "lingshu-medical-mllm/Lingshu-7B"
)

CURRENT_REQS=""   # tracks which requirements file is currently installed
N_OK=0; N_SKIP=0; N_FAIL=0

# ---------------- Helpers ------------------------------------------------------
log()  { echo "[eval_kaggle] $*"; }
disk_status() {
  echo "--- disk / gpu status ---"
  df -h "${SCRIPT_DIR}" 2>/dev/null || df -h
  du -sh "${OUTPUT_DIR}" "${DATASET_ROOT}" 2>/dev/null || true
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv 2>/dev/null || echo "(nvidia-smi unavailable)"
  echo "-------------------------"
}

# Replicates main.py tag: family + "_" + basename(model_id), "-"->"_", lowercase
make_tag() {
  local family="$1" model_id="$2"
  local base slug
  base="$(basename "$model_id")"
  slug="$(echo "$base" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
  echo "${family}_${slug}"
}

# Expected case count = number of mp4s in dataset (fallback 41)
expected_cases() {
  local n
  n="$(ls -1 "${DATASET_ROOT}"/case_*.mp4 2>/dev/null | wc -l | tr -d ' ')"
  if [ -z "$n" ] || [ "$n" = "0" ]; then echo "41"; else echo "$n"; fi
}

# Finished ONLY if summary exists with total==expected AND scores has expected lines.
# Partial runs return 1 (do not skip -> eval_common resumes via get_processed_ids).
is_finished() {
  local tag="$1" expected="$2"
  local summary="${OUTPUT_DIR}/${tag}_summary.json"
  local scores="${OUTPUT_DIR}/${tag}_scores.jsonl"
  [ -f "$summary" ] || return 1
  [ -f "$scores" ] || return 1
  local total lines
  total="$(python3 -c "import json;print(json.load(open('$summary')).get('total_cases_evaluated',0))" 2>/dev/null)" || return 1
  lines="$(wc -l < "$scores" | tr -d ' ')"
  [ "$total" = "$expected" ] && [ "$lines" = "$expected" ]
}

download_dataset() {
  local mp4n jsonln
  mp4n="$(ls -1 "${DATASET_ROOT}"/case_*.mp4 2>/dev/null | wc -l | tr -d ' ')"
  jsonln="$(ls -1 "${DATASET_ROOT}"/case_*.jsonl 2>/dev/null | wc -l | tr -d ' ')"
  if [ -n "$mp4n" ] && [ "$mp4n" != "0" ] && [ -n "$jsonln" ] && [ "$jsonln" != "0" ]; then
    log "Dataset present: ${mp4n} mp4, ${jsonln} jsonl -> skip download."
    return 0
  fi
  log "Installing huggingface_hub (pip, no-cache)..."
  pip install --no-cache-dir huggingface_hub
  log "Downloading ${HF_DATASET} -> ${DATASET_ROOT} ..."
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
  log "Download done."
  ls "${DATASET_ROOT}" | head -20
}

# pip-only, no uv. Reinstalls only on family switch. Never touches torch itself.
install_family_reqs() {
  local reqfile="$1"
  if [ "$CURRENT_REQS" = "$reqfile" ]; then
    log "Requirements already active: $reqfile -> skip reinstall."
    return 0
  fi
  log "Switching env -> $reqfile (pip uninstall + force-reinstall)..."
  # Uninstall version-sensitive pkgs (keep torch/torchvision to protect CUDA stack)
  pip uninstall -y transformers accelerate bitsandbytes qwen-vl-utils decord \
    opencv-python imageio ffmpeg-python einops scipy Pillow tqdm 2>/dev/null || true
  pip install --no-cache-dir --force-reinstall -r "${SCRIPT_DIR}/requirements/${reqfile}"
  pip cache purge 2>/dev/null || true
  rm -rf /tmp/pip-* /tmp/tmp* 2>/dev/null || true
  CURRENT_REQS="$reqfile"
  log "Env ready: $reqfile"
}

# Delete downloaded weights for one model to free Kaggle disk. Always runs.
cleanup_weights() {
  local model_id="$1"
  local slug="${model_id//\//--}"
  log "Cleaning weights for ${model_id} ..."
  rm -rf "${HF_HUB_CACHE}/models--${slug}" 2>/dev/null || true
  rm -rf "${HF_HUB_CACHE}/.locks" 2>/dev/null || true
  rm -rf /tmp/* 2>/dev/null || true
  pip cache purge 2>/dev/null || true
  disk_status
}

zip_model() {
  local tag="$1"
  (
    cd "$OUTPUT_DIR" || exit 0
    rm -f "${tag}.zip"
    # shellcheck disable=SC2046
    if ls ${tag}_responses.jsonl ${tag}_scores.jsonl ${tag}_summary.json ${tag}_summary.md 1>/dev/null 2>&1; then
      if command -v zip >/dev/null 2>&1; then
        zip -j -q "${tag}.zip" ${tag}_responses.jsonl ${tag}_scores.jsonl ${tag}_summary.json ${tag}_summary.md 2>/dev/null || true
      else
        python3 -c "
import zipfile, glob, os
z = zipfile.ZipFile('${tag}.zip', 'w', zipfile.ZIP_DEFLATED)
for f in ['${tag}_responses.jsonl','${tag}_scores.jsonl','${tag}_summary.json','${tag}_summary.md']:
    if os.path.exists(f): z.write(f, os.path.basename(f))
z.close()
"
      fi
      log "Zipped ${tag}.zip"
    else
      log "Nothing to zip for ${tag} yet."
    fi
  )
}

zip_all() {
  (
    cd "$SCRIPT_DIR" || exit 0
    rm -f results.zip
    if command -v zip >/dev/null 2>&1; then
      # raw result files only (exclude per-model zips to avoid nesting)
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
    ls -lh results.zip 2>/dev/null || true
  )
}

# Generic single-model runner: skip-if-finished, run, cleanup, zip.
run_model() {
  local family="$1" model_id="$2"
  shift 2
  # remaining args are extra main.py flags
  local tag expected rc
  tag="$(make_tag "$family" "$model_id")"
  expected="$(expected_cases)"
  if is_finished "$tag" "$expected"; then
    log "SKIP finished: $tag (${expected}/${expected})"
    N_SKIP=$((N_SKIP + 1))
    zip_model "$tag"
    return 0
  fi
  log "RUN ${family} ${model_id} (tag=${tag}, expected=${expected}) ..."
  disk_status
  limit_flag=()
  if [ -n "$LIMIT_ARG" ]; then limit_flag=(--limit "$LIMIT_ARG"); fi
  # shellcheck disable=SC2086
  python3 "${SCRIPT_DIR}/main.py" \
    --model-family "$family" \
    --model-id "$model_id" \
    --dataset-root "$DATASET_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    "${limit_flag[@]}" \
    $*
  rc=$?
  if [ $rc -eq 0 ]; then
    log "DONE ${tag}"
    N_OK=$((N_OK + 1))
  else
    log "FAIL ${tag} (exit=$rc) - continuing to next model"
    N_FAIL=$((N_FAIL + 1))
  fi
  cleanup_weights "$model_id"
  zip_model "$tag"
  zip_all
  return 0  # never abort the loop
}

run_qwen3vl() {
  install_family_reqs "requirements-qwen3vl.txt"
  local m
  for m in "${QWEN3_MODELS[@]}"; do
    run_model "qwen3vl" "$m" --max-frames "$QWEN3_FRAMES" --max-pixels "$QWEN_MAX_PIXELS" --min-pixels "$QWEN_MIN_PIXELS"
  done
}

run_hulumed() {
  install_family_reqs "requirements-hulumed.txt"
  local m
  for m in "${HULU_MODELS[@]}"; do
    run_model "hulumed" "$m" --max-frames "$HULU_FRAMES" --frame-size "$HULU_SIZE" --fps "$FPS_DEFAULT"
  done
}

run_lingshu() {
  install_family_reqs "requirements-qwen3vl.txt"  # same stack as qwen3vl
  local m
  for m in "${LINGSHU_MODELS[@]}"; do
    run_model "lingshu" "$m" --max-frames "$LINGSHU_FRAMES" --max-pixels "$QWEN_MAX_PIXELS" --min-pixels "$QWEN_MIN_PIXELS"
  done
}

# ---------------- Main ---------------------------------------------------------
case "${FAMILY_ARG}" in
  qwen3vl|hulumed|lingshu|all) ;;
  *)
    echo "Unknown family: ${FAMILY_ARG} (use: all|qwen3vl|hulumed|lingshu)"
    exit 1
    ;;
esac

log "CSI-Bench Kaggle runner | family=${FAMILY_ARG} | frames default=${FRAMES}"
log "HF dataset: ${HF_DATASET}"
disk_status

download_dataset
log "Expected cases: $(expected_cases)"
disk_status

case "${FAMILY_ARG}" in
  qwen3vl) run_qwen3vl ;;
  hulumed) run_hulumed ;;
  lingshu) run_lingshu ;;
  all)     run_qwen3vl; run_hulumed; run_lingshu ;;
esac

zip_all
disk_status
log "Finished: ok=${N_OK} skipped=${N_SKIP} failed=${N_FAIL}"
log "Download: results.zip + results/<tag>.zip"
