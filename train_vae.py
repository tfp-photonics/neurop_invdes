#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from fno_field_prediction.data import BlobData
from fno_field_prediction.models import VAE2d, VAE3d


def main(args):
    seed_everything(args.seed, workers=True)

    data = BlobData(
        args.shape,
        args.sigma,
        args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    module = VAE3d if len(args.shape) == 3 else VAE2d
    model = module(
        args.latent_dim,
        args.channels_encode,
        args.channels_decode,
        input_dim=args.shape[0],
        output_dim=args.shape[0],
        kld_weight=args.kld_weight,
        kld_weight_annealing=args.kld_weight_annealing,
        bin_weight=args.bin_weight,
        bin_cutoff=args.bin_cutoff,
        bin_weight_annealing=args.bin_weight_annealing,
        lr=args.lr,
        weight_decay=args.weight_decay,
        steps=args.steps,
    )

    callbacks = [
        LearningRateMonitor(),
    ]
    if args.checkpoint_dir is not None:
        callbacks.append(
            ModelCheckpoint(
                monitor="elbo",
                dirpath=f"{args.checkpoint_dir}",
                filename=f"{args.group}_latent{args.latent_dim}_" + "{step}_{elbo:.3e}",
                save_top_k=1,
                save_last=True,
                every_n_train_steps=1000,
            )
        )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        max_steps=args.steps,
        max_epochs=-1,
        callbacks=callbacks,
        fast_dev_run=args.dev,
        enable_progress_bar=False,
    )

    trainer.fit(model, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=912374122)

    VAE2d.add_model_specific_args(parser)

    parser.add_argument("--shape", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--sigma", type=int, default=12)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)

    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)

    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument(
        "--num-nodes", type=int, default=os.getenv("SLURM_JOB_NUM_NODES", 1)
    )
    parser.add_argument(
        "--num-workers", type=int, default=os.getenv("SLURM_CPUS_PER_TASK", 0)
    )
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    main(parser.parse_args())
