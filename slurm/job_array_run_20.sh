#!/bin/bash -e
#SBATCH --job-name=gemma_cases
#SBATCH --partition=genoa
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=H100:1
#SBATCH --array=0-49
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

CASES_PER_TASK=20
START_INDEX=$(( SLURM_ARRAY_TASK_ID * CASES_PER_TASK ))
PRESET=${PRESET:-test_2026}

echo "🧮 SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "📦 Processing cases $START_INDEX to $((START_INDEX + CASES_PER_TASK - 1))"
echo "📁 PRESET=${PRESET}"

module load Miniforge3/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /nesi/project/uoa04341/conda/nesi-env

python 5.process_cases.py \
  --preset "${PRESET}" \
  --start "${START_INDEX}" \
  --num "${CASES_PER_TASK}"
