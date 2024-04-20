import os
from PIL import Image
import numpy as np
import torch
from torch.utils import data

MEAN = torch.Tensor(np.array([0.485, 0.456, 0.406]))
STD = torch.Tensor(np.array([0.229, 0.224, 0.225]))

def get_image(rbg_path, depth_path, lbl_path=None):
    with open(rbg_path, 'rb') as f:
        rbg = Image.open(f).convert('RGB')
    with open(depth_path, 'rb') as f:
        depth = Image.open(f).convert('L')
    rbg = torch.FloatTensor(np.asarray(rbg)) / 255
    rbg = (rbg - MEAN) / STD
    rbg = rbg.permute(2, 0, 1).unsqueeze(dim=0)
    depth = torch.FloatTensor(np.asarray(depth))
    depth = depth.unsqueeze(dim=0).unsqueeze(dim=1)
    if lbl_path is not None:
        with open(lbl_path, 'rb') as f:
            lbl = Image.open(f).convert('P')
        lbl = torch.FloatTensor(np.asarray(lbl))
        lbl = lbl.unsqueeze(dim=0).unsqueeze(dim=1)
    else:
        lbl = None
    return rbg, depth, lbl

class MyDataset(data.Dataset):
    def __init__(self, root_path, split='train'):
        self.root_path = root_path
        self.split = split
        self.rbg_paths = []
        self.depth_paths = []
        self.lbl_paths = []
        with open(os.path.join(root_path, split + '.txt')) as f:
            for line in f:
                rbg_path, depth_path, lbl_path = line.strip().split()
                rbg_path = os.path.join(root_path, rbg_path)
                depth_path = os.path.join(root_path, depth_path)
                lbl_path = os.path.join(root_path, lbl_path)
                self.rbg_paths.append(rbg_path)
                self.depth_paths.append(depth_path)
                self.lbl_paths.append(lbl_path)
    
    def __len__(self):
        return len(self.rbg_paths)
    
    def __getitem__(self, index):
        rbg_path = self.rbg_paths[index]
        depth_path = self.depth_paths[index]
        lbl_path = self.lbl_paths[index]
        rbg, depth, lbl = get_image(rbg_path, depth_path, lbl_path)
        return rbg, depth, lbl

def collate_fn(batch):
    rbg = torch.cat([item[0] for item in batch], dim=0)
    depth = torch.cat([item[1] for item in batch], dim=0)
    lbl = torch.cat([item[2] for item in batch], dim=0)
    return rbg, depth, lbl