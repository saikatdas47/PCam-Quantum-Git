"""Building blocks for the prospective study, not a main-study launcher.

CPU software verification is separate from scientific performance/timing runs.
Historical model definitions and executed source snapshots remain unchanged.
"""
import copy
import os
from pathlib import Path
import time

PROJECT = Path(__file__).resolve().parents[3]
os.environ['MPLCONFIGDIR'] = str(PROJECT / '.cache/matplotlib')
os.environ['XDG_CACHE_HOME'] = str(PROJECT / '.cache')

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


from torch_circuit import TwoLayerRYHead


def paired_circuits(dimensions, initialization_seed):
    """Use a private RNG context and copy both rotations and affine readout."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initialization_seed)
        on = TwoLayerRYHead(dimensions, cnot=True)
        off = TwoLayerRYHead(dimensions, cnot=False)
    off.load_state_dict(copy.deepcopy(on.state_dict()))
    return off, on


def make_orders(samples, epochs, batch_seed):
    if samples < 1 or epochs < 1:
        raise ValueError('Sample and epoch counts must be positive')
    generator = torch.Generator(device='cpu').manual_seed(batch_seed)
    return torch.stack([torch.randperm(samples, generator=generator) for _ in range(epochs)])


def make_optimizer(model, learning_rate):
    return torch.optim.Adam(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
        eps=1e-8, weight_decay=0.0, foreach=False,
    )


def train_block(model, optimizer, x, y, orders, first, last, batch_size=128):
    """006 timing scope; caller handles diagnostics/checkpoints outside this call."""
    if x.device.type != 'cpu' or x.dtype != torch.float64:
        raise ValueError('CPU float64 is the only currently verified backend')
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    steps = 0
    started = time.perf_counter()
    for epoch in range(first, last):
        for batch in torch.split(orders[epoch], batch_size):
            bx, by = x[batch], y[batch]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            steps += 1
    return time.perf_counter() - started, steps


def warm_then_reset(model, optimizer, x, y, batch_size=128):
    if optimizer.state:
        raise ValueError('Warm-up is only permitted before the first optimizer update')
    initial_model = copy.deepcopy(model.state_dict())
    initial_optimizer = copy.deepcopy(optimizer.state_dict())
    initial_rng = torch.get_rng_state().clone()
    warm = torch.arange(min(batch_size, len(x)))
    train_block(model, optimizer, x, y, torch.stack([warm, warm]), 0, 2, batch_size)
    model.load_state_dict(initial_model)
    optimizer.load_state_dict(initial_optimizer)
    optimizer.zero_grad(set_to_none=True)
    torch.set_rng_state(initial_rng)
    if not all(torch.equal(v, initial_model[k]) for k, v in model.state_dict().items()):
        raise RuntimeError('Warm-up changed the restored model')
    if optimizer.state_dict()['state'] != initial_optimizer['state']:
        raise RuntimeError('Warm-up changed restored optimizer state')
    if not torch.equal(torch.get_rng_state(), initial_rng):
        raise RuntimeError('Warm-up changed restored RNG')


class TrainingTransform:
    """Fit PCA/scaling on explicitly supplied training IDs; transform held-out data."""

    def __init__(self, dimensions):
        self.pca = PCA(n_components=dimensions, svd_solver='full')
        self.scaler = StandardScaler()
        self.fit_ids = None

    def fit(self, features, sample_ids):
        ids = [str(value) for value in sample_ids]
        if len(ids) != len(features) or len(set(ids)) != len(ids):
            raise ValueError('Training IDs must align with features and be unique')
        if not ids or any(not value.startswith('train:') for value in ids):
            raise ValueError('Only official-training IDs may fit this transform')
        if self.fit_ids is not None:
            raise ValueError('Create a new transform for each fold instead of refitting')
        projected = self.pca.fit_transform(np.asarray(features, dtype=np.float64))
        self.scaler.fit(projected)
        self.fit_ids = tuple(ids)
        return self

    def transform(self, features, mapping):
        if self.fit_ids is None:
            raise ValueError('Fit on training data before transforming')
        z = self.scaler.transform(self.pca.transform(np.asarray(features, dtype=np.float64)))
        if mapping == 'standard':
            return z
        if mapping == 'atan':
            return np.arctan(z)
        raise ValueError('Undeclared representation')
