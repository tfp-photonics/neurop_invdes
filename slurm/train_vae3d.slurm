#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --job-name=vae3d
#SBATCH --output=%j_%x.log
#SBATCH --partition=accelerated

module load devel/cuda/11.6

CONDAENV=$TMP/conda
LOGS=$TMP/logs
CKPTS=$(ws_find fno)/checkpoints/

mkdir -p $LOGS
mkdir -p $CKPTS

source $(conda info --base)/etc/profile.d/conda.sh
srun --exclusive -n 1 -c 16 --mem=125400mb \
    conda env create -f environment.yml --prefix $CONDAENV

conda activate $CONDAENV
pip install -e . --no-deps

BATCH=64
STEPS=60000

srun --exclusive -n 1 -c 16 -G 1 --mem=125400mb python3 train_vae.py \
    --shape 128 128 128 \
    --batch-size $BATCH \
    --latent-dim 2048 \
    --steps $STEPS \
    --kld-weight 0.2 --kld-weight-annealing 1e-3 2e-3 0.4 \
    --bin-weight 1.0 --bin-weight-annealing 0.0 5e-3 0.75 \
    --channels-encode 8 16 32 64 128 256 \
    --channels-decode 256 128 64 32 16 8 \
    --log-dir $LOGS \
    --checkpoint-dir $CKPTS &

wait
