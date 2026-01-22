#!/bin/bash
#SBATCH --job-name=mob_dz
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=15:00:00

mkdir -p logs

nvidia-smi

# conda activate libmobility

python - <<'PY'
import os
import numpy as np
import parameters as params
import subprocess

task = int(os.environ["SLURM_ARRAY_TASK_ID"])

dvals = np.asarray(params.d, dtype=float)
zvals = np.asarray(params.h, dtype=float)   

pairs = [(d, z) for d in dvals for z in zvals]
d, z = pairs[task]

cmd = ["python", "run.py", "--d", str(d), "--z", str(z)]
print("Task:", task, "d:", d, "z:", z)
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)
PY

