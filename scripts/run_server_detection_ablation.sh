#!/usr/bin/env bash
# Run on AutoDL / SeetaCloud GPU instance after SSH login.
# Usage:
#   cd /root/autodl-tmp/lung-nodule-mvp   # adjust if your path differs
#   bash scripts/run_server_detection_ablation.sh preprocess
#   bash scripts/run_server_detection_ablation.sh baseline
#   bash scripts/run_server_detection_ablation.sh all
#   bash scripts/run_server_detection_ablation.sh eval

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA_YAML="configs/luna16_subset.yaml"
LUNA_DIR="${LUNA_DIR:-data/LUNA16}"
ANNOTATIONS="${ANNOTATIONS:-data/LUNA16/annotations.csv}"
PROJECT="runs/detection_ablation"
PREFIX="${PREFIX:-luna16}"
EPOCHS="${EPOCHS:-200}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-4}"

activate_env() {
  if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck source=/dev/null
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate lung-nodule 2>/dev/null || conda activate base
  elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "$HOME/miniconda3/etc/profile.d/conda.sh
    conda activate lung-nodule 2>/dev/null || true
  fi
  echo "Python: $(which python)"
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
}

step_preprocess() {
  echo "=== LUNA16 preprocess ==="
  if [[ ! -f "$ANNOTATIONS" ]]; then
    echo "Missing annotations: $ANNOTATIONS"
    exit 1
  fi
  python scripts/preprocess_luna16.py \
    --data-dir "$LUNA_DIR" \
    --annotations "$ANNOTATIONS" \
    --output-dir datasets
  python scripts/inspect_yolo_dataset.py --root datasets/detection
}

step_baseline() {
  echo "=== Train baseline only ==="
  python train_detection_ablation.py \
    --experiment baseline \
    --data "$DATA_YAML" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --workers "$WORKERS" \
    --project "$PROJECT" \
    --prefix "$PREFIX"
}

step_all() {
  echo "=== Train all 8 ablations ==="
  python train_detection_ablation.py \
    --experiment all \
    --data "$DATA_YAML" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --workers "$WORKERS" \
    --project "$PROJECT" \
    --prefix "$PREFIX"
}

step_eval() {
  echo "=== Evaluate ablations ==="
  python eval_detection_ablation.py \
    --data "$DATA_YAML" \
    --project "$PROJECT" \
    --prefix "$PREFIX" \
    --output "$PROJECT/metrics_${PREFIX}.csv"
  echo "CSV: $PROJECT/metrics_${PREFIX}.csv"
}

step_smoke() {
  EPOCHS=3 BATCH=4 WORKERS=2 step_baseline
}

CMD="${1:-help}"
activate_env
case "$CMD" in
  preprocess) step_preprocess ;;
  baseline)   step_baseline ;;
  all)        step_all ;;
  eval)       step_eval ;;
  smoke)      step_smoke ;;
  *)
    echo "Commands: preprocess | smoke | baseline | all | eval"
    echo "Env overrides: LUNA_DIR, ANNOTATIONS, PREFIX, EPOCHS, BATCH, WORKERS"
    exit 1
    ;;
esac

echo "=== Done: $CMD ==="
