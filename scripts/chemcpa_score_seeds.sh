#!/usr/bin/env bash
# Score each trained chemCPA seed against our models on identical targets.
#
# chemcpa_compare.py and chemcpa_snr_strata.py write fixed filenames (comparison.json,
# snr_strata.csv) into --out, so running them per seed into results/chemcpa/ would overwrite
# seed 42's existing output. Each seed therefore gets its own subdirectory, and the files are
# then linked to the per-seed names scripts/chemcpa_seeds.py reads.
#
# Sequential on purpose: each prediction pass loads a 169 MB dataset plus the model, and this
# runs while other work may be resident.
#
# The two stages need DIFFERENT interpreters, which is the whole reason they are separate scripts:
# chemcpa_predict.py imports their package and must run in the chemCPA venv, while compare and
# strata run in ours. Running predict under the general venv fails with
# "No module named 'lightning'". cwd stays in this repo so `-m scripts.x` resolves either way.
set -u
cd /home/ruman/repos/Cell2Compound
export NUMBA_CACHE_DIR=/tmp/cell2compound_numba MPLCONFIGDIR=/tmp/cell2compound_mpl
export PYTHONPATH=/home/ruman/repos/chemCPA WANDB_MODE=offline
P=/home/ruman/venvs/general/bin/python
PC=/home/ruman/venvs/chemcpa/bin/python
R=results/chemcpa

for S in "$@"; do
  CKPT=$(ls -t /home/ruman/repos/chemCPA/training_output_s$S/*/last.ckpt 2>/dev/null | head -1)
  if [ -z "$CKPT" ]; then echo "SKIP seed $S: no checkpoint"; continue; fi
  echo "===== seed $S: $CKPT"
  mkdir -p "$R/s$S"

  $PC -m scripts.chemcpa_predict --checkpoint "$CKPT" --dataset "ours_sciplex_s$S" \
     --out "$R/ood_predictions_s$S.npz" || { echo "FAIL predict s$S"; continue; }

  $P -m scripts.chemcpa_compare --predictions "$R/ood_predictions_s$S.npz" \
     --seed "$S" --out "$R/s$S" || { echo "FAIL compare s$S"; continue; }

  $P -m scripts.chemcpa_snr_strata --predictions "$R/ood_predictions_s$S.npz" \
     --seed "$S" --out "$R/s$S" || echo "WARN strata s$S"

  cp "$R/s$S/comparison.json" "$R/comparison_s$S.json"
  [ -f "$R/s$S/snr_strata.csv" ] && cp "$R/s$S/snr_strata.csv" "$R/snr_strata_s$S.csv"
  echo "OK seed $S"
done

echo "===== aggregate ====="
$P -m scripts.chemcpa_seeds
