#!/usr/bin/env bash
# Reproduce every table and figure in results/ from scratch.
# Takes about 4 hours on an RTX 3050 laptop GPU (most of it: tuning, the GNN ablations and the sparsity suite).
#
#   bash scripts/run_all.sh            # full run
#   QUICK=1 bash scripts/run_all.sh    # 2 epochs per model, 1 seed, written to results_quick/ (pipeline check, ~20 min)
#
# Add WANDB=1 to also log the training runs to Weights & Biases.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
EXTRA=()
SET=()
if [[ "${QUICK:-0}" == "1" ]]; then
  EXTRA+=(--max-epochs 2 --seeds 42)
  SET=(paths.results_dir=results_quick paths.checkpoint_dir=checkpoints_quick paths.selected_config=results_quick/selected.yaml)
  mkdir -p results_quick
fi
if [[ "${WANDB:-0}" == "1" ]]; then EXTRA+=(--wandb); fi
S=()
if [[ ${#SET[@]} -gt 0 ]]; then S=(--set "${SET[@]}"); fi

$PY scripts/download_data.py
$PY scripts/prepare_data.py
$PY scripts/run_eda.py "${S[@]}"

for suite in tune main gnn_ablation hybrid item_cold sparsity onboarding; do
  $PY scripts/run_experiments.py "$suite" "${EXTRA[@]}" "${S[@]}"
done

$PY scripts/run_retrieval.py "${S[@]}"
$PY scripts/run_ranking.py "${S[@]}"
$PY scripts/run_ranking_temporal.py "${S[@]}"
$PY scripts/run_reranking.py "${S[@]}"
$PY scripts/evaluate_pipeline.py "${S[@]}"
$PY scripts/run_analysis.py "${S[@]}"
$PY scripts/make_tables.py "${S[@]}"
$PY scripts/make_figures.py "${S[@]}"
# copy the generated tables into README.md / REPORT.md (only for a full run)
if [[ "${QUICK:-0}" != "1" ]]; then $PY scripts/update_docs.py; fi

# PostgreSQL part (needs DATABASE_URL or `pip install pgserver`)
if [[ "${SKIP_DB:-0}" != "1" ]]; then
  $PY scripts/db_load.py "${S[@]}"
  $PY scripts/db_check.py "${S[@]}" || true
fi
