import numpy as np
from torch.utils.data import Dataset


class MixedDataset(Dataset):
    """
    Sample-level mixing: each sample is drawn from `main_ds` (multimodal),
    or with probability `aux_prob` from `aux_ds` (text-only).
    """

    def __init__(self, main_ds, aux_ds, aux_prob: float = 0.1, seed: int = 42):
        self.main_ds = main_ds
        self.aux_ds = aux_ds
        self.aux_prob = aux_prob
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.main_ds)

    def __getitem__(self, idx):
        if self.rng.random() < self.aux_prob:
            aux_idx = int(self.rng.integers(len(self.aux_ds)))
            return self.aux_ds[aux_idx]
        return self.main_ds[idx]