## Data generation

This directory contains the scripts for generating scattering datasets using [Meep](https://github.com/NanoComp/meep).
It is recommended to use a separate conda environment for data generation, the environment file is provided in `env_datagen.yml`.

Data generation happens in two steps.
First, individual samples are generated using `create_sample.py`, one file each, that contain a scattering geometry and associated fields.
Then, these samples are aggregated into a dataset via `create_dataset.py`.
The reason for this is that on the one hand an arbitrary number of samples with different field components can be generated, and on the other hand it is easier to generate samples in parallel if processes don't have to write to the same file/dataset.
Afterwards, datasets for training/validation/testing can be created by taking whatever is desired (number of samples, field components) from existing samples, *i.e.* samples have to be generated only once, and different datasets can be created from those samples.

For usage examples, refer to the [slurm scripts](../slurm) that were used to generate the datasets for the paper.

## Getting fields from samples / optimized devices

The simulation in `simulation.py` can be used to get fields from arbitrary scatterers as well.
The following code snippet is what was used to verify the inverse-designed devices:

```python
yopt = ...  # load device array

sim = FDTD(
        extent=[5.12, 5.12, 5.12],
        design_extent=[5.12, 5.12, 5.12],
        resolution=25,
        src_components=["ex"],
        out_components=["ex", "ey", "ez"],
)

fields = sim(yopt)
fields = np.stack([c[1:-1, 1:-1, 1:-1] for c in fields.values()], dtype="c8")
```