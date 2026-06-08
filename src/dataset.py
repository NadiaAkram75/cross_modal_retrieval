from pathlib import Path
import numpy as np

class BraTSDataset:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.npy"))
      
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return np.load(self.files[idx])