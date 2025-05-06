#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --account=precisionhealth_owned1
#SBATCH --partition=precisionhealth
#SBATCH --mem-per-gpu=200gb
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=mmakar@umich.edu

experiment=$1 
split=$2

errfile="/nfs/turbo/coe-rbg/rjsingh/slurm/catboost_${split}_${experiment}.err"
outfile="/nfs/turbo/coe-rbg/rjsingh/slurm_logs/catboost_${split}_${experiment}_${jobid}.out"

exec 2> "$errfile"
exec > "$outfile"

python scripts/train_catboost_optuna_headroom.py \
    --experiment $1 \
    --cache_dir  /nfs/turbo/coe-rbg/mmakar/tableshift/ \
    --results_dir /nfs/turbo/coe-rbg/rjsingh/results \
    --use_cached \
    --use_gpu \
    --num_samples 20 \
    --split_mode $2 