#!/bin/bash -l
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --time=12:00:00
#SBATCH --job-name=fno3d
#SBATCH --output=%j_%x.log
#SBATCH --partition=accelerated

module load devel/cuda/11.6

CONDAENV=$TMP/conda
DATA=$(ws_find fno)/datasets/fields_3d_exeyez/
LOGS=$TMP/logs
CKPTS=$(ws_find fno)/checkpoints/

mkdir -p $LOGS
mkdir -p $CKPTS

source $(conda info --base)/etc/profile.d/conda.sh
srun --exclusive -n $SLURM_JOB_NUM_NODES \
    conda env create -f environment.yml --prefix $CONDAENV

conda activate $CONDAENV

TRAIN=8192
VAL=128
BATCH=4
EPOCHS=100

srun python3 train_surrogates.py \
    --name fno3d_8k \
    --model fno3d \
    --modes 12 --width 32 --blocks 10 --padding 2 \
    --split $TRAIN $VAL --batch-size $BATCH --epochs $EPOCHS \
    --data-key design --label-key fields \
    --lr 1e-3 --weight-decay 1e-6 --scheduler onecycle \
    --accelerator gpu --devices 4 --strategy ddp_find_unused_parameters_false \
    --num-workers 16 \
    --data-dir $DATA --log-dir $LOGS --checkpoint-dir $CKPTS
