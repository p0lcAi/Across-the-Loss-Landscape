from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, random_split
from torchvision import datasets, transforms


@dataclass
class DataConfig:
    dataset: str = "cifar100"
    root: str = "./data"
    batch_size: int = 128
    eval_batch_size: int = 256
    num_workers: int = 4
    val_fraction: float = 0.1
    image_size: int = 32
    augmentation: str = "cifar"
    pin_memory: bool = True
    persistent_workers: bool = True
    seed: int = 0
    hessian_batch_size: int = 128
    n_train: int = 4000
    n_test: int = 2000
    synthetic_noise: float = 0.15


class IndexedSubset(Dataset):
    def __init__(self, dataset: Dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.dataset[self.indices[i]]



def _build_transforms(cfg: DataConfig):
    if cfg.dataset.lower() != "cifar100":
        raise ValueError(f"Unsupported dataset: {cfg.dataset}")

    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    train_tf = transforms.Compose([
        transforms.RandomCrop(cfg.image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, test_tf

def make_two_moons(n: int, noise: float, seed: int):
    generator = torch.Generator().manual_seed(int(seed))

    n0 = n // 2
    n1 = n - n0

    t0 = torch.rand(n0, generator=generator) * torch.pi
    t1 = torch.rand(n1, generator=generator) * torch.pi

    x0 = torch.stack(
        [
            torch.cos(t0),
            torch.sin(t0),
        ],
        dim=1,
    )

    x1 = torch.stack(
        [
            1.0 - torch.cos(t1),
            0.5 - torch.sin(t1),
        ],
        dim=1,
    )

    x = torch.cat([x0, x1], dim=0)
    y = torch.cat(
        [
            torch.zeros(n0, dtype=torch.long),
            torch.ones(n1, dtype=torch.long),
        ],
        dim=0,
    )

    x = x + float(noise) * torch.randn(x.shape, generator=generator)

    perm = torch.randperm(n, generator=generator)
    x = x[perm]
    y = y[perm]

    x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-8)

    return x.float(), y.long()


def build_dataloaders(cfg: DataConfig) -> Tuple[Dict[str, DataLoader], int]:
    dataset_name = cfg.dataset.lower()

    if dataset_name in {"two_moons", "twomoons"}:
        x_train_full, y_train_full = make_two_moons(
            n=int(cfg.n_train),
            noise=float(cfg.synthetic_noise),
            seed=int(cfg.seed),
        )

        x_test, y_test = make_two_moons(
            n=int(cfg.n_test),
            noise=float(cfg.synthetic_noise),
            seed=int(cfg.seed) + 10_000,
        )

        n_total = x_train_full.shape[0]
        n_val = int(round(float(cfg.val_fraction) * n_total))
        n_val = max(1, min(n_total - 1, n_val))
        n_train = n_total - n_val

        generator = torch.Generator().manual_seed(int(cfg.seed))
        perm = torch.randperm(n_total, generator=generator)

        train_idx = perm[:n_train]
        val_idx = perm[n_train:]

        train_ds = TensorDataset(x_train_full[train_idx], y_train_full[train_idx])
        val_ds = TensorDataset(x_train_full[val_idx], y_train_full[val_idx])
        test_ds = TensorDataset(x_test, y_test)

        kwargs = dict(
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=(cfg.num_workers > 0 and cfg.persistent_workers),
        )

        loaders = {
            "train": DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False, **kwargs),
            "train_eval": DataLoader(train_ds, batch_size=cfg.eval_batch_size, shuffle=False, drop_last=False, **kwargs),
            "val": DataLoader(val_ds, batch_size=cfg.eval_batch_size, shuffle=False, drop_last=False, **kwargs),
            "test": DataLoader(test_ds, batch_size=cfg.eval_batch_size, shuffle=False, drop_last=False, **kwargs),
            "hessian": DataLoader(train_ds, batch_size=cfg.hessian_batch_size, shuffle=True, drop_last=False, **kwargs),
            "transition": DataLoader(val_ds, batch_size=cfg.eval_batch_size, shuffle=False, drop_last=False, **kwargs),
        }

        return loaders, 2
    train_tf, test_tf = _build_transforms(cfg)
    full_train = datasets.CIFAR100(root=cfg.root, train=True, transform=train_tf, download=True)
    full_train_eval = datasets.CIFAR100(root=cfg.root, train=True, transform=test_tf, download=True)
    test_ds = datasets.CIFAR100(root=cfg.root, train=False, transform=test_tf, download=True)

    n_total = len(full_train)
    n_val = int(round(cfg.val_fraction * n_total))
    n_val = max(1, min(n_total - 1, n_val))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(cfg.seed)
    train_subset_aug, val_subset_aug = random_split(full_train, [n_train, n_val], generator=generator)
    train_indices = train_subset_aug.indices
    val_indices = val_subset_aug.indices

    train_ds = IndexedSubset(full_train, train_indices)
    train_eval_ds = IndexedSubset(full_train_eval, train_indices)
    val_ds = IndexedSubset(full_train_eval, val_indices)

    kwargs = dict(
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.num_workers > 0 and cfg.persistent_workers),
    )
    loaders = {
        "train": DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False, **kwargs),
        "train_eval": DataLoader(train_eval_ds, batch_size=cfg.eval_batch_size, shuffle=False, drop_last=False, **kwargs),
        "val": DataLoader(val_ds, batch_size=cfg.eval_batch_size, shuffle=False, drop_last=False, **kwargs),
        "test": DataLoader(test_ds, batch_size=cfg.eval_batch_size, shuffle=False, drop_last=False, **kwargs),
        "hessian": DataLoader(train_eval_ds, batch_size=cfg.hessian_batch_size, shuffle=True, drop_last=False, **kwargs),
        "transition": DataLoader(val_ds, batch_size=cfg.eval_batch_size, shuffle=False, drop_last=False, **kwargs),
    }
    return loaders, 100
