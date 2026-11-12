"""Seen-node training windows, full-node testing, and single-variable normalization."""
import numpy as np
from .protocol import fit_scaler, window_starts


def observed_preparation(raw, manifest):
    """NumPy-only core, testable without torch; never fits on unseen values."""
    if list(raw.shape) != manifest["raw_shape"] or raw.shape[-1] != 1:
        raise ValueError("Expected manifest-matched single-variable input")
    starts = window_starts(len(raw), manifest["split_ratio"], manifest["window_size"])
    mean, std = fit_scaler(raw, manifest["seen_nodes"], starts["train"], manifest["window_size"])
    # Restrict the available training/validation tensor to seen nodes and its
    # final allowed time index, so it cannot accidentally access test-only rows.
    stop = int(starts["val"][-1]) + manifest["window_size"]
    observed = raw[:stop, manifest["seen_nodes"]].copy()
    return observed, starts, mean, std


def make_loaders(raw, manifest, batch_size, seed, eval_batch_size=None):
    import torch
    from torch.utils.data import DataLoader
    from .data_windows import TrafficWindows
    from .protocol import calendar_codes
    observed, starts, mean, std = observed_preparation(raw, manifest)
    common = dict(codes=calendar_codes(len(raw), manifest["steps_per_day"]),
                  mean=mean, std=std, in_steps=manifest["in_steps"], out_steps=manifest["out_steps"])
    datasets = {split: TrafficWindows(observed if split != "test" else raw, ids, **common)
                for split, ids in starts.items()}
    generator = torch.Generator().manual_seed(seed)
    loaders = {split: DataLoader(ds, batch_size=(eval_batch_size or batch_size) if split == "test" else batch_size, shuffle=split == "train",
                                generator=generator if split == "train" else None, num_workers=0)
               for split, ds in datasets.items()}
    return loaders, mean, std
