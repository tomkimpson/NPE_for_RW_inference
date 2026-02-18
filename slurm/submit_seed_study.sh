#!/bin/bash
#
# Submit all seed study jobs: 4 models x 5 seeds = 20 jobs.
# Usage:  bash slurm/submit_seed_study.sh
#

MODELS="original A B C"
SEEDS="42 123 456 789 1024"

echo "========================================"
echo "SEED STUDY: Submitting 20 jobs"
echo "Models: ${MODELS}"
echo "Seeds:  ${SEEDS}"
echo "========================================"

COUNT=0
for MODEL in ${MODELS}; do
    for SEED in ${SEEDS}; do
        JOB_ID=$(sbatch --parsable slurm/run_seed_single.sh "${MODEL}" "${SEED}")
        echo "  Submitted: model=${MODEL}  seed=${SEED}  job_id=${JOB_ID}"
        COUNT=$((COUNT + 1))
    done
done

echo "========================================"
echo "Total jobs submitted: ${COUNT}"
echo "Monitor with: squeue -u \$USER"
echo "========================================"
