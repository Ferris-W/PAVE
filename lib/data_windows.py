"""Lazy single-variable windows with dataset-relative calendar features."""
import numpy as np
import torch
from torch.utils.data import Dataset


class TrafficWindows(Dataset):
    def __init__(self, values, starts, codes, mean, std, in_steps=12, out_steps=12, target_nodes=None):
        self.values, self.starts, self.codes = values, starts, codes
        self.mean, self.std = mean, std
        self.in_steps, self.out_steps = in_steps, out_steps
        self.target_nodes = target_nodes

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = int(self.starts[index])
        middle, end = start + self.in_steps, start + self.in_steps + self.out_steps
        physical = (self.values[start:middle] - self.mean) / self.std
        clock = np.broadcast_to(self.codes[start:middle, None, :], (*physical.shape[:2], 3))
        x = np.concatenate([physical, clock], axis=-1).astype(np.float32)
        y = self.values[middle:end, :, :1]
        if self.target_nodes is not None:
            y = y[:, self.target_nodes]
        # ST-FiT normalizes targets too. Round-trip with the same float32
        # statistics at evaluation (including y>=1 threshold boundary behavior).
        y = (y - self.mean[0]) / self.std[0]
        return torch.from_numpy(x), torch.from_numpy(y.copy())
