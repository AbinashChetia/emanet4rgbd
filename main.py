import numpy as np
import pickle as pk
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from network import RCFusionEMA
from dataset import NYUDataset

import warnings
warnings.filterwarnings('ignore')

DATASET_LOC = 'nyu_depth_v2_labeled_data.pkl'
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# DEVICE = torch.device('cpu')

MAX_DATA = 1000
TRAIN_TEST_RATIO = 0.8
BATCH_SIZE = 2
MAX_ITERS = 10
ITER_STEP = 1

nyu_depth_labeled_data = pk.load(open(DATASET_LOC, 'rb'))
nyu_dataset = {
    'rgb': nyu_depth_labeled_data['images'],
    'depth': nyu_depth_labeled_data['depths'],
    'label': nyu_depth_labeled_data['labels']
}

nyu_dataset = {
    'rgb': nyu_dataset['rgb'][:MAX_DATA],
    'depth': nyu_dataset['depth'][:MAX_DATA],
    'label': nyu_dataset['label'][:MAX_DATA]
}

nyu_train_dataset = {
    'rgb': nyu_dataset['rgb'][:int(len(nyu_dataset['rgb']) * TRAIN_TEST_RATIO)],
    'depth': nyu_dataset['depth'][:int(len(nyu_dataset['depth']) * TRAIN_TEST_RATIO)],
    'label': nyu_dataset['label'][:int(len(nyu_dataset['label']) * TRAIN_TEST_RATIO)]
}

nyu_val_dataset = {
    'rgb': nyu_dataset['rgb'][int(len(nyu_dataset['rgb']) * TRAIN_TEST_RATIO):],
    'depth': nyu_dataset['depth'][int(len(nyu_dataset['depth']) * TRAIN_TEST_RATIO):],
    'label': nyu_dataset['label'][int(len(nyu_dataset['label']) * TRAIN_TEST_RATIO):]
}

train_data = NYUDataset(nyu_train_dataset, device=DEVICE)
val_data = NYUDataset(nyu_val_dataset, device=DEVICE)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

model = RCFusionEMA(n_classes=64).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for iter in range(MAX_ITERS):
    for rgb, depth, label in train_loader:
        loss = model.train(rgb, depth, label, optimizer)
        print(f'Iter: {iter} | Training Loss: {loss}')
        if iter % ITER_STEP == 0:
            val_loss = 0
            for rgb, depth, label in val_loader:
                val_loss += model.validate(rgb, depth, label)
            print(f'\tValidation loss: {val_loss/len(val_loader)}')
        iter += 1
    break

save_model = input('Do you want to save the model? (y/n): ')
if save_model == 'y':
    torch.save(model.state_dict(), 'rcfusion_ema.pth')
    print('Model saved as rcfusion_ema.pth!')
else:
    print('Model not saved!')