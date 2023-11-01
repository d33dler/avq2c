#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=42:00:00
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=12
#SBATCH --array=1-12
module load Python/3.10.4-GCCcore-11.3.0

cd ..
source venv/bin/activate
cd executables
ulimit -n 2048
export LRU_CACHE_SIZE=8000
python execute.py --jobfile $1 --job_id ${SLURM_ARRAY_TASK_ID}
python execute.py --jobfile $1 --job_id ${SLURM_ARRAY_TASK_ID} --test