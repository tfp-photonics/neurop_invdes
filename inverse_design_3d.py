#!/usr/bin/env python3

import argparse
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from einops import rearrange
from mpi4py import MPI
from torch.optim import AdamW
from tqdm import tqdm

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
DEVICE = f"cuda:{RANK % torch.cuda.device_count()}"


class onrank:
    def __init__(self, rank):
        self.noop = RANK != rank

    def __enter__(self):
        if self.noop:
            sys.settrace(lambda _: None)
            frame = sys._getframe(1)
            frame.f_trace = lambda _: exec("raise")

    def __exit__(self, *_):
        return self.noop


@torch.no_grad()
def asnumpy(x):
    return x.detach().cpu().numpy()


class FNO:
    def __init__(self, fp, device):
        model = torch.jit.load(fp, map_location=device)
        self.model = torch.jit.optimize_for_inference(model)

    @staticmethod
    def rearrange_fields(x):
        x = rearrange(x, "b (ExEyEz ReIm) x y z -> b ExEyEz x y z ReIm", ExEyEz=3)
        x = torch.view_as_complex(x.contiguous())
        return x

    def __call__(self, x):
        x = self.model(x)
        x = self.rearrange_fields(x)
        return x


class VAE:
    def __init__(self, fp, device):
        model = torch.jit.load(fp, map_location=device)
        self.encoder = torch.jit.optimize_for_inference(model.encoder)
        self.decoder = torch.jit.optimize_for_inference(model.decoder)
        self.latent = int(re.findall(r"latent(\d+)", str(fp))[0])


def enforce_symmetry(x, axes=[-2, -3]):
    s = x.shape
    for ax in axes:
        x = torch.index_select(x, ax, torch.arange(s[ax] // 2, device=x.device))
    for ax in axes:
        x = torch.cat([x, x.flip(ax)], ax)
    return x


def optimize_adamw(x0, vae, fno, mask, args):
    xopt = [x0i.detach().clone().requires_grad_() for x0i in x0]
    opt = AdamW(xopt, lr=args.lr, weight_decay=args.weight_decay)
    hist = []

    for _ in tqdm(range(args.max_its), ncols=80, disable=RANK != 0):
        opt.zero_grad()

        x = torch.stack(xopt)
        x = vae.decoder(x)
        x = enforce_symmetry(x, args.symmetry_axes)
        x = fno(x)
        x = torch.sum(torch.abs(x) ** 2, 1)
        x = x[:, mask]
        x = -torch.mean(x, -1)

        x.backward(torch.ones_like(x))
        opt.step()

        with torch.no_grad():
            for xi in xopt:
                xi.clamp_(*args.bounds)
            hist.append(asnumpy(-x))

    return torch.stack(xopt), np.stack(hist, -1)


def gather(ary, root=0):
    recv = None
    if RANK == 0:
        recv = np.empty([SIZE, *ary.shape], dtype=ary.dtype)
    COMM.Gather(ary, recv, root=root)
    if RANK == 0:
        recv = np.reshape(recv, (-1, *recv.shape[2:]))
    return recv


def main(args):
    if args.seed is not None:
        torch.manual_seed(args.seed + RANK)

    vae = VAE(args.vae_fp, DEVICE)
    fno = FNO(args.fno_fp, DEVICE)

    x0 = torch.normal(0, 1, (args.batch_size, vae.latent), device=DEVICE)
    x0 = torch.clamp(x0, *args.bounds)
    y0 = enforce_symmetry(vae.decoder(x0), args.symmetry_axes)
    z0 = fno(y0)

    mask = np.zeros(y0.shape[-3:], dtype="?")

    if args.objectve == "1point":
        mask[60:68, 60:68, 116:124] = 1
    elif args.objective == "4point":
        mask[94:98, 94:98, 120:124] = 1
        mask[30:34, 94:98, 120:124] = 1
        mask[94:98, 30:34, 120:124] = 1
        mask[30:34, 30:34, 120:124] = 1
    else:
        raise ValueError(f"Unkown objective type: {args.objective}")

    x0_ = gather(asnumpy(x0))
    y0_ = gather(asnumpy(y0))
    z0_ = gather(asnumpy(z0))
    with onrank(0), torch.no_grad(), h5py.File(args.output_file, "w") as f:
        for k, v in json.loads(json.dumps(vars(args), default=str)).items():
            f.attrs[k] = v
        f.create_dataset("mask", data=mask, dtype="?", compression="gzip")
        f.create_dataset("x0", data=x0_, dtype="f4", compression="gzip")
        f.create_dataset("y0", data=y0_, dtype="f4", compression="gzip")
        f.create_dataset("z0", data=z0_, dtype="c8", compression="gzip")

    del x0_, y0_, z0_
    del y0, z0
    torch.cuda.empty_cache()

    xopt, hist = optimize_adamw(x0, vae, fno, mask, args)
    yopt = enforce_symmetry(vae.decoder(xopt), args.symmetry_axes)
    zopt = fno(yopt)

    xopt_ = gather(asnumpy(xopt))
    yopt_ = gather(asnumpy(yopt))
    zopt_ = gather(asnumpy(zopt))
    hist_ = gather(hist)
    with onrank(0), torch.no_grad(), h5py.File(args.output_file, "a") as f:
        f.create_dataset("xopt", data=xopt_, dtype="f4", compression="gzip")
        f.create_dataset("yopt", data=yopt_, dtype="f4", compression="gzip")
        f.create_dataset("zopt", data=zopt_, dtype="c8", compression="gzip")
        f.create_dataset("hist", data=hist_, compression="gzip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=912350123)
    parser.add_argument("--vae-fp", type=Path, required=True)
    parser.add_argument("--fno-fp", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--max-its", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--symmetry-axes", type=int, nargs="+", default=[-2, -3])
    parser.add_argument("--bounds", type=int, nargs=2, default=(-3, 3))
    parser.add_argument(
        "--objective", type=str, choices=["1point", "4point"], default="1point"
    )
    main(parser.parse_args())
