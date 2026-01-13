#!/bin/bash
#SBATCH --job-name=dh_sweep
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

#SBATCH --constraint=compute

#TOTAL=$(python -c "import params; print(int(params.dbins*params.hbins))")
#SBATCH --array=0-$((TOTAL-1)) scheduler.sh


set -euo pipefail

mkdir -p logs

echo "JobID=${SLURM_JOB_ID} ArrayTaskID=${SLURM_ARRAY_TASK_ID} Host=$(hostname)"
echo "SubmitDir=${SLURM_SUBMIT_DIR} PWD=$(pwd)"

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate libmobility

OUTBASE="/scratch/${USER}/"

python -u run.py --outdir "${OUTBASE}"
