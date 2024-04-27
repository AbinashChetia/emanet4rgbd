import numpy as np
import pickle as pk
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import network

DATASET_LOC = 'nyu_depth_v2_labeled_data.pkl'
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

TRAIN_TEST_RATIO = 0.8
BATCH_SIZE = 32

def collate_fn(batch):
    pass

def train():
    dataset = pk.load(open(DATASET_LOC, 'rb'))
    dataset_img = {'rgb': np.array(dataset['images'], dtype=np.float32), 'd': np.array(dataset['depths'], dtype=np.float32)}
    dataset_labels = np.array(dataset['labels'], dtype=int)
    
    train_test_ratio = TRAIN_TEST_RATIO
    last_train_idx = int(len(dataset_img['rgb']) * train_test_ratio)
    
    train_data = {'rgb': dataset_img['rgb'][:last_train_idx], 'd': dataset_img['d'][:last_train_idx]}
    train_labels = dataset_labels[:last_train_idx]

    val_data = {'rgb': dataset_img['rgb'][last_train_idx:], 'd': dataset_img['d'][last_train_idx:]}
    val_labels = dataset_labels[last_train_idx:]
    
    model = network.ResNetEMA(n_classes=100, emau_stages=3)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(list(zip(train_data['rgb'], train_data['d'], train_labels)), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(list(zip(val_data['rgb'], val_data['d'], val_labels)), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    for epoch in range(10):
        model.train()
        for idx, (data, labels) in enumerate(train_loader):
            data['rgb'] = data['rgb'].to(DEVICE)
            data['d'] = data['d'].to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            loss, _ = model(data, labels)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch}, Batch {idx}, Loss: {loss.item()}')

        model.eval()
        with torch.no_grad():
            total_loss = 0
            for idx, (data, labels) in enumerate(val_loader):
                data = data.to(DEVICE)
                labels = labels.to(DEVICE)
                loss, _ = model(data, labels)
                total_loss += loss.item()
            print(f'Epoch {epoch}, Validation Loss: {total_loss / len(val_loader)}')
    opt = input('Save model? (y/n)')
    if opt == 'y':
        torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    train()    