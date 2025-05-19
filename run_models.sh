#!/bin/bash
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=ray_headroom-%x.%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --account=mmakar98
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100gb
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=rjsingh@umich.edu
#SBATCH --error=/nfs/turbo/coe-rbg/rjsingh/slurm/slurm-%A-%a.err


python scripts/ray_train_headroom.py \
    --models aldro \
    --experiment acsincome \
    --cache_dir  /nfs/turbo/coe-rbg/mmakar/tableshift/ \
    --results_dir  /nfs/turbo/coe-rbg/rjsingh/results \
    --use_cached \
    --num_samples 20 \
    --max_concurrent_trials 1 \
    --split_mode train 