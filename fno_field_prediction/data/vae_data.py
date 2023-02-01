import numpy as np
import pytorch_lightning as pl
import torch
from numpy.random import SeedSequence
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, IterableDataset


class Sampler(IterableDataset):
    def __iter__(self):
        num_workers, worker_id = self._get_worker_info()
        ss = SeedSequence(self.seed)
        rng = [self.default_rng(s) for s in ss.spawn(num_workers)][worker_id]
        return iter(self._make_iter(rng))

    def _get_worker_info(self):
        if (info := torch.utils.data.get_worker_info()) is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = info.num_workers
            worker_id = info.id
        return num_workers, worker_id

    def _make_iter(self, rng):
        while True:
            yield self._make(rng)


class BlobSampler(Sampler):
    def __init__(self, shape, sigma, seed=None):
        self.shape = shape
        self.sigma = sigma
        self.seed = seed

        self.default_rng = np.random.default_rng

    def _make(self, rng):
        while True:
            design = rng.random(self.shape, dtype="f4")
            design = gaussian_filter(design, self.sigma, mode="constant") > 0.5
            if not np.all(~design):
                break
        return torch.as_tensor(design, dtype=torch.float)


class Dataset(pl.LightningDataModule):
    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
        )


class BlobData(Dataset):
    def __init__(self, shape, sigma, batch_size, num_workers=4, seed=None):
        super().__init__()
        self.save_hyperparameters()
        self.save_hyperparameters({"geometry_type": "blobs"})

    def setup(self, stage=None):
        rng = np.random.default_rng(self.hparams.seed)
        train_seed = rng.integers(np.iinfo(np.uint64).max, dtype=np.uint64)
        self.train_data = BlobSampler(
            self.hparams.shape, self.hparams.sigma, train_seed
        )
