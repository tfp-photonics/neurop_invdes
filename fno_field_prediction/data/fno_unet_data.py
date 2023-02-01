from functools import cache
from pathlib import Path

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split


class FieldData(pl.LightningDataModule):
    def __init__(
        self,
        data_path,
        batch_size,
        split,
        cache,
        data_key="design",
        label_key="fields",
        num_workers=4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.save_data_attrs()

    def save_data_attrs(self):
        data = HDF5Dataset(
            self.hparams.data_path / "train",
            data_key=self.hparams.data_key,
            label_key=self.hparams.label_key,
            cache=False,
        )
        attrs = data.attrs
        attrs["out_channels"] = len(data[0][self.hparams.label_key])
        self.save_hyperparameters(attrs)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_data = HDF5Dataset(
                self.hparams.data_path / "train",
                data_key=self.hparams.data_key,
                label_key=self.hparams.label_key,
                cache=self.hparams.cache,
            )
            train, val, _ = random_split(
                train_data,
                (*self.hparams.split, len(train_data) - sum(self.hparams.split)),
            )
            self.train_data = train
            self.val_data = val
        if stage == "test" or stage is None:
            self.test_data = HDF5Dataset(
                self.hparams.data_path / "test",
                data_key=self.hparams.data_key,
                label_key=self.hparams.label_key,
                cache=self.hparams.cache,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.hparams.batch_size)


class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        recursive=False,
        data_key="data",
        label_key="label",
        cache=False,
        transform=None,
    ):
        super().__init__()

        self.data_key = data_key
        self.label_key = label_key
        self._get = self._cached_getitem if cache else self._getitem
        self.cache = cache
        self.transform = transform

        self.attrs = {}
        self.data_keys = []

        p = Path(path)
        if not p.is_dir():
            raise RuntimeError(f"Not a directory: {p}")
        pattern = "**/*.h5" if recursive else "*.h5"
        files = sorted(p.glob(pattern))
        if len(files) < 1:
            raise RuntimeError("No hdf5 datasets found")
        for f in files:
            with h5py.File(f.resolve(), "r") as h5f:
                self.attrs.update(h5f.attrs)
                self.data_keys += [[k, f] for k in h5f.keys() if k != "src"]

    def __getitem__(self, index):
        return self._get(index)

    def _getitem(self, index):
        uid, fp = self.data_keys[index]
        sample = self._from_file(fp, uid)
        if self.transform:
            sample = self.transform(sample)
        return sample

    @cache
    def _cached_getitem(self, index):
        uid, fp = self.data_keys[index]
        sample = self._from_file(fp, uid)
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.data_keys)

    def _from_file(self, fp, uid):
        with h5py.File(fp, "r") as f:
            x = torch.from_numpy(np.array(f[uid][self.data_key], dtype="f4"))
            y = torch.from_numpy(np.array(f[uid][self.label_key], dtype="f4"))
        return {self.data_key: x, self.label_key: y}
