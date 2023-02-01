# Surrogate solvers for electromagnetic field inference and inverse design

This repository contains the code for the paper "A neural operator-based surrogate solver for free-form electromagnetic inverse design".

## Installation

To run the code, set up a [conda](https://docs.conda.io/en/latest/) environment from the provided environment file and activate it:

```bash
conda env create -f environment.yml -p your_env_prefix
conda activate your_env_prefix
```

For data generation, use the environment file `env_datagen.yml` provided [here](./data_generation/).

## Usage

We provide the [Slurm](https://slurm.schedmd.com/documentation.html) batch scripts that were used to perform all the computations in the paper [here](./slurm/).
While these probably won't run on your particular setup without modification, they should provide enough insight on how to use the data generation, training and inverse design scripts.

### FNO & UNet training

To train a surrogate solver, use the file `train_surrogates.py`, see `./train_surrogates.py --help` for a list of parameters.
In the simplest case, using just the defaults, you can train an `FNO-2D` model by simply providing a path to the dataset:
```bash
python3 train_surrogates.py --data-dir /path/to/data
```
Note that the dataloaders used here expect the following folder structure:
```
data
├── train
│   ├── train_data_0.h5
│   ├── train_data_1.h5
│   ├── etc.
└── test
    ├── test_data_0.h5
    ├── test_data_1.h5
    └── etc.
```

### VAE training

The VAE models can be trained using `train_vae.py`, see `./train_vae.py --help` for a list of parameters.
You can simply run the file without providing any arguments, which will train a 2D VAE.

### Inverse design

For inverse design, use the file `inverse_design_3d.py`.
We do not currently provide an implementation for 2D inverse design, but it should be straightforward to adapt the current 3D implementation.
Please note that this file needs pre-trained FNO and VAE models in [TorchScript](https://pytorch.org/docs/stable/jit.html) format.
