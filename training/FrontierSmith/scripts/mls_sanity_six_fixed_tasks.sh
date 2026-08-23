#!/usr/bin/env bash
# Verify the six previously-always-zero MLS CPU tasks now produce real scores.
# Each runs a known-good baseline algorithm through the identical in-container
# path an agent uses, so a pass means the task is genuinely unblocked.
set -uo pipefail
cd "${MLSBENCH_ROOT:-/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev}"
export MLSBENCH_NO_PREBUILT=1 MLSBENCH_SCHEDULER_MANAGED=1
export PYTHONPATH="${MLSBENCH_ROOT:-/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev}/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
P=/home/bl3615/miniconda3/bin/python
for spec in "ml-anomaly-detection|isolation_forest" \
            "ml-missing-data-imputation|mean_impute" \
            "ml-selective-deferral|confidence_thresholding" \
            "ml-subgroup-calibration-shift|temperature_scaling" \
            "optimization-multi-objective|nsga2" \
            "causal-observational-linear-non-gaussian|icalingam"; do
  T="${spec%%|*}"; N="${spec##*|}"
  echo "=== $T ($N) ==="
  $P -m mlsbench baseline "$T" --name "$N" --config configs/config.yaml --seed 42 2>&1 | tail -4
done
