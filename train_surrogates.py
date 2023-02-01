#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from fno_field_prediction.data import FieldData
from fno_field_prediction.models import FNO2d, FNO3d, UNet


def main(args):
    seed_everything(args.seed, workers=True)

    data = FieldData(
        args.data_dir,
        args.batch_size,
        args.split,
        data_key=args.data_key,
        label_key=args.label_key,
        cache="2d" in args.model,  # 3d data too large to keep in memory
        num_workers=args.num_workers,
    )

    if "fno" in args.model:
        mc = FNO2d if "2d" in args.model else FNO3d
        model = mc(
            args.modes,
            args.width,
            args.blocks,
            args.padding,
            data.hparams.out_channels,
            args.lr,
            args.weight_decay,
            args.epochs,
            args.scheduler,
            data.hparams.get("lambda0", None),
            data.hparams.get("dl", None),
            args.with_maxwell_loss,
        )
    elif args.model == "unet":
        model = UNet(
            args.alpha,
            args.num_down_conv,
            args.hidden_dim,
            data.hparams.out_channels,
            args.lr,
            args.weight_decay,
            args.epochs,
            args.scheduler,
            data.hparams.lambda0,
            data.hparams.dl,
            args.with_maxwell_loss,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    callbacks = [
        LearningRateMonitor(log_momentum=True),
    ]
    if args.checkpoint_dir is not None:
        callbacks.append(
            ModelCheckpoint(
                monitor="val_loss",
                dirpath=args.checkpoint_dir,
                filename=f"{args.model}_{args.name}_{str(args.split[0])[:-3]}k"
                + "_{epoch:02d}_{val_loss:.3f}",
                save_top_k=1,
                mode="min",
            )
        )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        max_epochs=args.epochs,
        callbacks=callbacks,
        fast_dev_run=args.dev,
        enable_progress_bar=False,
    )

    trainer.fit(model, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", type=str, choices=["fno2d", "fno3d", "unet"], default="fno2d"
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["onecycle", "exponential", "none"],
        default="onecycle",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--with-maxwell-loss", action="store_true")
    FNO2d.add_model_specific_args(parser)
    UNet.add_model_specific_args(parser)

    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--data-key", type=str, default="design")
    parser.add_argument("--label-key", type=str, default="fields")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--split", type=int, nargs=2, default=[2048, 256])
    parser.add_argument("--epochs", type=int, default=100)
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
    parser.add_argument("--seed", type=int, default=912374122)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    main(parser.parse_args())
