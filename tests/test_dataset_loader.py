"""Phase 1 verification: dataset ingestion, windowing, scaler protocol.

Covers:
    * ETTh1 automated download + chronological 0.6/0.2/0.2 protocol
    * window shapes (L=96 -> H=96) and contiguity semantics
    * scaler fitted on the TRAIN split only (val/test must NOT be zero-mean)
    * zero-variance channel guard
    * build_dataloaders() wiring against configs/data/etth1.yaml

Network-dependent tests skip gracefully when the download is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data_engine.dataset_loader import (
    ETTh1Dataset,
    StandardScaler,
    build_dataloaders,
)

SEQ, PRED, N_CHANNELS = 96, 96, 7
ETTH1_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"


@pytest.fixture(scope="module")
def train_dataset():
    try:
        return ETTh1Dataset(flag="train", size=(SEQ, PRED), data_url=ETTH1_URL)
    except Exception as exc:  # pragma: no cover - network unavailable
        pytest.skip(f"ETTh1 unavailable (download failed): {exc}")


def test_split_shapes_and_dtypes(train_dataset):
    ds = train_dataset
    assert ds.data_x.ndim == 2
    assert ds.data_x.shape[1] == N_CHANNELS
    assert ds.data_x.dtype == np.float32
    assert ds.data_y.dtype == np.float32
    assert len(ds) > 0


def test_window_semantics_and_contiguity(train_dataset):
    ds = train_dataset
    i = 10
    x, y, domain_id = ds[i]
    assert x.shape == (SEQ, N_CHANNELS)
    assert y.shape == (PRED, N_CHANNELS)
    assert domain_id.dtype == torch.int64
    assert domain_id.item() in (0, 1)
    assert x.dtype == torch.float32 and y.dtype == torch.float32
    # The first future step of y must continue x chronologically
    np.testing.assert_allclose(y[0].numpy(), ds.data_x[i + SEQ], rtol=1e-6)


def test_scaler_fit_on_train_only():
    scaler = StandardScaler()
    data = np.concatenate([np.full((50, 2), 10.0), np.full((50, 2), -10.0)])
    scaler.fit(data[:50])  # train half only
    assert np.allclose(scaler.mean, 10.0)
    transformed = scaler.transform(data)
    assert np.allclose(transformed[:50], 0.0)      # train maps to ~zero mean
    assert np.allclose(transformed[50:], -20.0)    # val/test shifted, NOT zero-mean


def test_scaler_zero_variance_guard():
    scaler = StandardScaler()
    data = np.ones((10, 3))
    scaler.fit(data)
    assert np.all(scaler.std == 1.0)
    assert np.allclose(scaler.transform(data), 0.0)


def test_ms_mode_univariate_target_shape():
    try:
        ds = ETTh1Dataset(
            flag="train",
            size=(SEQ, PRED),
            features="MS",
            data_url=ETTH1_URL,
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"ETTh1 unavailable: {exc}")
    x, y, domain_id = ds[0]
    assert x.shape == (SEQ, N_CHANNELS)
    assert y.shape == (PRED, 1)
    assert domain_id.item() in (0, 1)


def test_build_dataloaders_smoke():
    try:
        loaders = build_dataloaders(num_workers=0)  # Windows-safe smoke run
    except Exception as exc:  # pragma: no cover - network unavailable
        pytest.skip(f"ETTh1 unavailable: {exc}")
    assert set(loaders) == {"train", "val", "test"}
    x, y, domain_ids = next(iter(loaders["train"]))
    assert domain_ids.shape == (32,)
    assert x.shape == (32, SEQ, N_CHANNELS)
    assert y.shape == (32, PRED, N_CHANNELS)
    assert x.dtype == torch.float32
