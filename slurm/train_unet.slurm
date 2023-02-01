#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --job-name=fno2d_unet
#SBATCH --output=%j_%x.log
#SBATCH --partition=accelerated

module load devel/cuda/11.6

CONDAENV=$TMP/conda
DATA=$(ws_find fno)/datasets/fields_2d_ez/
LOGS=$TMP/logs
CKPTS=$(ws_find fno)/checkpoints/

mkdir -p $LOGS
mkdir -p $CKPTS

source $(conda info --base)/etc/profile.d/conda.sh
srun --exclusive -n 1 -c 16 --mem=125400mb \
    conda env create -f environment.yml --prefix $CONDAENV

conda activate $CONDAENV

VAL=256
BATCH=32
EPOCHS=100

srun --exclusive -n 1 -c 16 -G 1 --mem=125400mb python3 train_surrogates.py \
    --name unet_1k \
    --model unet \
    --alpha 0.3 --num-down-conv 6 --hidden-dim 16 \
    --split 1024 $VAL --batch-size $BATCH --epochs $EPOCHS \
    --data-key permittivity --label-key fields \
    --lr 5e-5 --weight-decay 1e-6 --scheduler onecycle \
    --data-dir $DATA --log-dir $LOGS --checkpoint-dir $CKPTS &

srun --exclusive -n 1 -c 16 -G 1 --mem=125400mb python3 train_surrogates.py \
    --name unet_2k \
    --model unet \
    --alpha 0.3 --num-down-conv 6 --hidden-dim 16 \
    --split 2048 $VAL --batch-size $BATCH --epochs $EPOCHS \
    --data-key permittivity --label-key fields \
    --lr 5e-5 --weight-decay 1e-6 --scheduler onecycle \
    --data-dir $DATA --log-dir $LOGS --checkpoint-dir $CKPTS &

srun --exclusive -n 1 -c 16 -G 1 --mem=125400mb python3 train_surrogates.py \
    --name unet_4k \
    --model unet \
    --alpha 0.3 --num-down-conv 6 --hidden-dim 16 \
    --split 4096 $VAL --batch-size $BATCH --epochs $EPOCHS \
    --data-key permittivity --label-key fields \
    --lr 5e-5 --weight-decay 1e-6 --scheduler onecycle \
    --data-dir $DATA --log-dir $LOGS --checkpoint-dir $CKPTS &

srun --exclusive -n 1 -c 16 -G 1 --mem=125400mb python3 train_surrogates.py \
    --name unet_8k \
    --model unet \
    --alpha 0.3 --num-down-conv 6 --hidden-dim 16 \
    --split 8192 $VAL --batch-size $BATCH --epochs $EPOCHS \
    --data-key permittivity --label-key fields \
    --lr 5e-5 --weight-decay 1e-6 --scheduler onecycle \
    --data-dir $DATA --log-dir $LOGS --checkpoint-dir $CKPTS &

srun --exclusive -n 1 -c 16 -G 1 --mem=125400mb python3 train_surrogates.py \
    --name unet_16k \
    --model unet \
    --alpha 0.3 --num-down-conv 6 --hidden-dim 16 \
    --split 16384 $VAL --batch-size $BATCH --epochs $EPOCHS \
    --data-key permittivity --label-key fields \
    --lr 5e-5 --weight-decay 1e-6 --scheduler onecycle \
    --data-dir $DATA --log-dir $LOGS --checkpoint-dir $CKPTS &

wait
