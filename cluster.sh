#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem=100G
#SBATCH --output=ray_headroom-%x.%j.out
#SBATCH --account=mmakar98
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=rjsingh@umich.edu
#SBATCH --error=/nfs/turbo/coe-soto/rjsingh/slurm/slurm-%A-%a.err

python scripts/ray_train_headroom.py \
    --models label_group_dro \
    --experiment acspubcov \
    --cache_dir  /nfs/turbo/coe-soto/tableshift/ \
    --results_dir /nfs/turbo/coe-soto/rjsingh/results \
    --use_cached \
    --num_samples 1 \
    --max_concurrent_trials 1