#!/bin/sh
#SBATCH --job-name=gen_samples
#SBATCH --nodes=1
#SBATCH --ntasks=76
#SBATCH --time=04:00:00
#SBATCH --array=1-232%40
#SBATCH --output=logs/gen_samples_%A_%a.log
#SBATCH --partition=cpuonly

CONDAENV=$TMP/conda

source $(conda info --base)/etc/profile.d/conda.sh
srun --exclusive -n 1 -c 16 --mem=125400mb \
    conda env create -f env_datagen.yml --prefix $CONDAENV

conda activate $CONDAENV

mpirun -n 76 --bind-to core --map-by core python create_sample.py \
    --n_parallel 19 \
    --n_samples 19 \
    --src_components ex \
    --out_components ex ey ez \
    --sim_res 25 \
    --sim_extent 5.12 5.12 5.12 \
    --design_extent 5.12 5.12 5.12 \
    --sigma 12 \
    --odir samples/fields_3d_exeyez \
    --uuid
