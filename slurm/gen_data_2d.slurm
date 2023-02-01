#!/bin/sh
#SBATCH --job-name=gen_samples
#SBATCH --nodes=1
#SBATCH --ntasks=76
#SBATCH --time=03:00:00
#SBATCH --output=logs/%j_%x.log
#SBATCH --partition=cpuonly

CONDAENV=$TMP/conda

source $(conda info --base)/etc/profile.d/conda.sh
srun --exclusive -n 1 -c 16 --mem=125400mb \
    conda env create -f env_datagen.yml --prefix $CONDAENV

conda activate $CONDAENV

mpirun -n 76 --bind-to core --map-by core python python create_sample.py \
    --n_parallel 76 \
    --n_samples 17100 \
    --src_components ez \
    --out_components ez \
    --sim_res 25 \
    --sim_extent 5.12 5.12 \
    --design_extent 5.12 5.12 \
    --sigma 12 \
    --odir samples/fields_2d_ez \
    --uuid
