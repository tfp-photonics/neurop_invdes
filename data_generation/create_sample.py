import argparse
from pathlib import Path
from uuid import uuid4

import h5py
import meep as mp
import numpy as np
from mpi4py import MPI
from scipy.ndimage import gaussian_filter
from simulation import FDTD


def get_group_masters():
    comm = MPI.COMM_WORLD
    num_workers = comm.Get_size()

    is_group_master = True if mp.my_rank() == 0 else False
    group_master_idx = np.zeros((num_workers,), dtype=np.bool_)

    smsg = [np.array([is_group_master]), ([1] * num_workers, [0] * num_workers)]
    rmsg = [group_master_idx, ([1] * num_workers, list(range(num_workers)))]

    comm.Alltoallv(smsg, rmsg)

    group_masters = np.arange(num_workers)[group_master_idx]

    return group_masters


def blob(shape, sigma, dtype="?"):
    design = np.random.uniform(0, 1, shape)
    design = gaussian_filter(design, sigma, mode="constant")
    return np.array(design > 0.5, dtype=dtype)


def main(args):
    mp.verbosity(0)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    mp.divide_parallel_processes(args.n_parallel)

    group_masters = get_group_masters()

    if args.from_dataset is not None:
        with h5py.File(args.from_dataset, "r") as f:
            samples = sorted(list(f.keys()))
            shape = np.array(f[samples[0]][args.dataset_key]).shape
        samples = {
            k: v
            for k, v in zip(
                group_masters,
                np.array_split(samples[: args.n_samples], args.n_parallel),
            )
        }
    else:
        if args.uuid:
            samples = {
                k: v
                for k, v in zip(
                    group_masters,
                    np.array_split(
                        [uuid4() for _ in range(args.n_samples)], args.n_parallel
                    ),
                )
            }
        else:
            samples = {
                k: v
                for k, v in zip(
                    group_masters,
                    np.array_split(np.arange(args.n_samples), args.n_parallel),
                )
            }
        shape = [int(args.sim_res * e) for e in args.design_extent]

    worker_map = {
        k: list(range(k + 1, k + size // args.n_parallel)) for k in group_masters
    }

    if mp.am_master():
        my_master = rank
    else:
        my_master = [k for k in worker_map.keys() if rank in worker_map[k]][0]

    for key in samples[my_master]:
        if mp.am_master():
            if args.from_dataset is not None:
                with h5py.File(args.from_dataset, "r") as f:
                    design = np.array(f[key][args.dataset_key], dtype="f4")
                    design = design > np.mean(design)
            else:
                while True:
                    design = blob(shape, args.sigma, dtype="?")
                    if not np.all(~design):
                        break
            for dest in worker_map[rank]:
                comm.Send(design, dest=dest)
        else:
            design = np.empty(shape, dtype="?")
            comm.Recv(design, source=[k for k, v in worker_map.items() if rank in v][0])

        sim = FDTD(
            extent=args.sim_extent,
            design_extent=args.design_extent,
            resolution=args.sim_res,
            src_components=args.src_components,
            out_components=args.out_components,
        )
        fields = sim(design)

        if mp.am_master():
            args.odir.mkdir(exist_ok=True, parents=True)
            if args.from_dataset is not None or args.uuid:
                fname = key
            else:
                fname = f"{key:0{len(str(args.n_samples))}d}"
            with h5py.File(args.odir / f"{fname}.h5", "w") as f:
                f.create_dataset("design", data=design, compression="gzip")
                g = f.create_group("fields")
                for k, v in fields.items():
                    g.create_dataset(k, data=v, compression="gzip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_parallel", type=int, default=19)
    parser.add_argument("--n_samples", type=int, default=19)
    parser.add_argument("--src_components", type=str, nargs="+", default=["ex"])
    parser.add_argument(
        "--out_components", type=str, nargs="+", default=["ex", "ey", "ez"]
    )
    parser.add_argument("--sim_res", type=int, default=25)
    parser.add_argument(
        "--sim_extent", type=float, nargs="+", default=[5.12, 5.12, 5.12]
    )
    parser.add_argument("--sigma", type=int, default=12)
    parser.add_argument(
        "--design_extent", type=float, nargs="+", default=[5.12, 5.12, 5.12]
    )
    parser.add_argument("--odir", type=Path, default="generated_samples/")
    parser.add_argument("--from_dataset", type=Path, default=None)
    parser.add_argument("--dataset_key", type=str, default=None)
    parser.add_argument("--uuid", action="store_true")
    args = parser.parse_args()
    main(args)
