import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml, os
from models.ega_net import EGANet
from dataset.private_dataset import PrivateVoiceDataset
from dataset.augmentation import Compose, TimeMasking, FrequencyMasking
from utils.train_utils import compute_f1, save_model

with open('configs/default.yaml') as f:
    cfg = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EGANet(in_channels=1, num_classes=cfg['num_classes']).to(device)

train_set = PrivateVoiceDataset(
    root_dir='data/train',
    label_file='data/train_labels.csv',
    transform=Compose([TimeMasking(), FrequencyMasking()])
)
train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                 patience=cfg['lr_scheduler_patience'],
                                                 factor=cfg['lr_scheduler_factor'])

best_f1 = 0

for epoch in range(cfg['num_epochs']):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for wave, label in train_loader:
        wave, label = wave.to(device), label.to(device)
        output = model(wave)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.append(output)
        all_labels.append(label)

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    f1 = compute_f1(preds, labels)
    scheduler.step(total_loss)

    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}, F1: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        save_model(model, "best_model.pth")
        print("New best model saved.")
