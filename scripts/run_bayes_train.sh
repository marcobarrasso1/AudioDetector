#!/bin/bash
#SBATCH --job-name=bayes_train
#SBATCH --partition=GPU
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=%x.log
#SBATCH --error=%x.log
#SBATCH --open-mode=append

# Account MaxWall is 2h -> the 40-epoch training runs as a chain of jobs.
# Each job does up to 8 epochs (resuming from <prefix>_last.pt automatically),
# then resubmits itself unless <prefix>.done exists.
#
# Usage:
#   sbatch --job-name=bayes_p01  run_bayes_train.sh bayes_p01  0.1 1.0
#   sbatch --job-name=bayes_cold run_bayes_train.sh bayes_cold 0.1 0.1
#
# args: 1=ckpt/log prefix   2=prior_sigma   3=kl_scale

PREFIX=${1:-bayes}
PRIOR=${2:-1.0}
KLSCALE=${3:-1.0}

cd "$SLURM_SUBMIT_DIR"
source ../newenv/bin/activate
module load cuda

python train_bayes.py \
    --epochs 40 \
    --max_epochs_this_run 8 \
    --batch_size 64 \
    --lr 3e-4 \
    --kl_warmup 5 \
    --kl_scale "$KLSCALE" \
    --prior_sigma "$PRIOR" \
    --val_every 3 \
    --num_workers 8 \
    --ckpt_prefix "$PREFIX" \
    --log_dir "runs/$PREFIX"

if [ ! -f "$PREFIX.done" ]; then
    echo "=== chaining next job ($PREFIX) ==="
    sbatch --job-name="$SLURM_JOB_NAME" run_bayes_train.sh "$PREFIX" "$PRIOR" "$KLSCALE"
else
    echo "=== training complete ($PREFIX) ==="
fi
