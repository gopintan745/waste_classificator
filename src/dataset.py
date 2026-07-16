import torch
from torch.utils.data import WeightedRandomSampler
import numpy as np
from PIL import Image
from pathlib import Path

class WasteDataset(torch.utils.data.Dataset):
    """Extended to support weighted sampling."""
    def __init__(self, root, transform=None, split="train"):
        self.root = Path(root) / split
        self.transform = transform
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            for img_path in (self.root / cls).glob("*.*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((img_path, self.class_to_idx[cls]))
        self.targets = [s[1] for s in self.samples]

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, label

    def get_sampler(self):
        """Returns a WeightedRandomSampler that balances classes per epoch."""
        class_counts = np.bincount(self.targets)
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[t] for t in self.targets]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self.samples),
            replacement=True
        )
