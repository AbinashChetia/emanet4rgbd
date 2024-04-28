import os
from PIL import Image
import numpy as np
import torch

class NYUDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, device):
        self.rgb = torch.from_numpy(dataset['rgb'].astype(np.float32)).to(device)
        self.depth = torch.from_numpy(dataset['depth'].astype(np.float32)).to(device)
        self.label = torch.from_numpy(dataset['label'].astype(np.float32)).to(device)
    
    def __len__(self):
        return len(self.rgb)
    
    def __getitem__(self, idx):
        return self.rgb[idx], self.depth[idx], self.label[idx]