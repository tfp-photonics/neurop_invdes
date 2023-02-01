#!/usr/bin/env python3

import argparse
from glob import glob
from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np
from mpi4py import MPI


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    samples = np.array_split(glob(f"{args.input_dir.resolve()}/*.h5"), size)
    fp = args.output_dir / f"{args.name}_{rank}.h5"

    with h5py.File(fp, "w") as of:
        for sample in samples[rank]:
            with h5py.File(sample, "r") as f:
                design = np.array(f["design"], dtype="?")
                fields = []
                oc = []
                for c in sorted(f["fields"].keys()):  # ensure we are in standard order
                    if args.components is None or c in args.components:
                        field = np.array(f["fields"][c])
                        field = field[tuple(field.ndim * [slice(1, -1)])]
                        fields.append(field)
                        oc.append(c)

            all_fields = np.zeros((2 * len(fields), *field.shape), dtype=args.dtype)
            for idx, field in enumerate(fields):
                all_fields[2 * idx] = np.real(field)
                all_fields[2 * idx + 1] = np.imag(field)

            of.attrs["components"] = ",".join(oc)
            grp = of.create_group(str(uuid4()))
            grp.create_dataset("design", data=design, compression="gzip")
            grp.create_dataset("fields", data=all_fields, compression="gzip")

    print(f"Successfully wrote {fp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, required=True)
    parser.add_argument("-i", "--input_dir", type=Path, required=True)
    parser.add_argument("-o", "--output_dir", type=Path, default="datasets/")
    parser.add_argument(
        "-c", "--components", type=str, nargs="+", default=["ex", "ey", "ez"]
    )
    parser.add_argument("-d", "--dtype", type=str, default="f4")
    args = parser.parse_args()
    main(args)
