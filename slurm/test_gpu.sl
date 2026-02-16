#!/bin/bash -e
#SBATCH --job-name=GPUJob   # job name (shows up in the queue)
#SBATCH --time=00-00:10:00  # Walltime (DD-HH:MM:SS)
#SBATCH --cpus-per-task=2   # number of CPUs per task (1 by default)

#SBATCH --gpus-per-node=A100:1

#SBATCH --output=/nesi/project/uoa04341/COMPSCI792/logs/GPUJob-%j.out
#SBATCH --error=/nesi/project/uoa04341/COMPSCI792/logs/GPUJob-%j.err
#SBATCH --mem=2G         # amount of memory per node (1 by default)
#SBATCH --mail-type=ALL          # When to send email: BEGIN, END, FAIL, ALL
#SBATCH --mail-user=mzak071@aucklanduni.ac.nz


# load CUDA module
module purge
module load CUDA/12.6.3
module load Python/3.11.6-foss-2023a

# display information about the available GPUs
nvidia-smi

# check the value of the CUDA_VISIBLE_DEVICES variable
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# check the value of the torch cuda 
echo -e "\n=== PyTorch test ==="
python - <<'PY'
import torch, os, subprocess, platform
print("Python        :", platform.python_version())
print("Torch         :", torch.__version__)
print("Torch CUDA    :", torch.version.cuda)
print("Available GPU :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name   :", torch.cuda.get_device_name(0))
PY
