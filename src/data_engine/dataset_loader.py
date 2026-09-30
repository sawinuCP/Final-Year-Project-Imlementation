"""Automated dataset ingestion and sliding-window chunking (Phase 1).

Responsibilities (spec Phase 1, task 2):
    * Automated downloader/loader for ETTh1, driven by
      ``configs/data/etth1.yaml`` (ETTm1 / Electricity / Weather follow the
      same pattern when scaling beyond the pilot dataset).
    * Sliding-window chunking producing
          X in R^{B x L x D},   Y in R^{B x H x D}      (default L=96, H=96)
          or univariate targets when ``features='MS'``.
    * Standard scalar normalization fitted STRICTLY on the training split:
          X_norm = (X - mu_train) / sigma_train
      (zero-variance channels are guarded by std -> 1.0)
    * Chronological split with lookback overlap for the val/test borders
      (standard ETT protocol: 12 months train / 4 months val / 4 months test,
      split_ratios = [0.6, 0.2, 0.2]).
    * Per-window ``domain_id`` (0/1) for domain-stratified FOIL residual alignment.

Numerics: windows are materialized as float32 tensors (full FP32 mandate).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import requests
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

__all__ = ["StandardScaler", "ETTh1Dataset", "build_dataloaders"]


class StandardScaler:
    """Train-split-fitted standard scalar: (X - mu_train) / sigma_train."""

    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.std[self.std == 0.0] = 1.0

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class ETTh1Dataset(Dataset):
    """
    Standard ETTh1 Dataset Loader.
    Supports:
      - features='M' : Multivariate inputs [L, D] -> Multivariate targets [H, D]
      - features='MS': Multivariate inputs [L, D] -> Univariate target (e.g. OT) [H, 1]
      - domain_id    : Emits temporal chunk regime (0 = first half of split,
                       1 = second half) for domain-stratified FOIL residual alignment.
    """

    def __init__(
        self,
        root_path="data/raw/ETTh1.csv",
        flag="train",
        size=(96, 96),
        features="M",
        data_url=None,
        target="OT",
        scale=True,
        split_ratios=(0.6, 0.2, 0.2),
    ):
        self.seq_len = size[0]
        self.pred_len = size[1]
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.root_path = root_path
        self.data_url = data_url
        self.split_ratios = split_ratios

        self._ensure_data_exists()
        self._read_data()

    def _ensure_data_exists(self):
        if not os.path.exists(self.root_path):
            os.makedirs(os.path.dirname(self.root_path), exist_ok=True)
            if self.data_url:
                print(f"Downloading ETTh1 dataset from {self.data_url}...")
                response = requests.get(self.data_url, timeout=60)
                response.raise_for_status()
                with open(self.root_path, "wb") as f:
                    f.write(response.content)
                print("Download complete.")
            else:
                raise FileNotFoundError(
                    f"File {self.root_path} not found and no URL provided."
                )

    def _read_data(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.root_path)

        num_train = int(len(df_raw) * self.split_ratios[0])
        num_test = int(len(df_raw) * self.split_ratios[2])
        num_vali = len(df_raw) - num_train - num_test

        border1s = [
            0,
            num_train - self.seq_len,
            len(df_raw) - num_test - self.seq_len,
        ]
        border2s = [num_train, num_train + num_vali, len(df_raw)]

        type_map = {"train": 0, "val": 1, "test": 2}
        border1 = border1s[type_map[self.flag]]
        border2 = border2s[type_map[self.flag]]

        cols_data = list(df_raw.columns[1:])
        df_data = df_raw[cols_data]

        if self.target in cols_data:
            self.target_idx = cols_data.index(self.target)
        else:
            self.target_idx = -1

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2].astype(np.float32)

        if self.features == "MS":
            self.data_y = data[
                border1 : border2, self.target_idx : self.target_idx + 1
            ].astype(np.float32)
        else:
            self.data_y = data[border1:border2].astype(np.float32)

        midpoint = len(self.data_x) // 2
        self.domain_ids = np.zeros(len(self.data_x), dtype=np.int64)
        self.domain_ids[midpoint:] = 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        domain_id = self.domain_ids[s_begin]

        return (
            torch.from_numpy(seq_x),
            torch.from_numpy(seq_y),
            torch.tensor(domain_id, dtype=torch.long),
        )

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


def build_dataloaders(config_path="configs/data/etth1.yaml",
                      batch_size=None, num_workers=None):
    """Build (train, val, test) DataLoaders from ``configs/data/etth1.yaml``.

    Overrides (``batch_size``, ``num_workers``) are useful for tests and
    CPU-only smoke runs (set ``num_workers=0`` on Windows).
    """

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["data"]

    batch_size = batch_size or cfg.get("batch_size", 32)
    num_workers = (
        num_workers if num_workers is not None else cfg.get("num_workers", 0)
    )
    split_ratios = tuple(cfg.get("split_ratios", (0.6, 0.2, 0.2)))

    loaders = {}
    for flag in ("train", "val", "test"):
        dataset = ETTh1Dataset(
            root_path=cfg.get("raw_path", "data/raw/ETTh1.csv"),
            flag=flag,
            size=(cfg.get("seq_len", 96), cfg.get("pred_len", 96)),
            features=cfg.get("features", "M"),
            data_url=cfg.get("data_url"),
            target=cfg.get("target_col", "OT"),
            split_ratios=split_ratios,
        )
        loaders[flag] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(flag == "train"),
            num_workers=num_workers,
            drop_last=(flag == "train"),
        )
    return loaders
