import os
import torch
from torch.utils.data import Dataset
import torchaudio

class PrivateVoiceDataset(Dataset):
    def __init__(self, root_dir, label_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        with open(label_file, 'r') as f:
            for line in f:
                path, label = line.strip().split(',')
                self.samples.append((path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        waveform, _ = torchaudio.load(os.path.join(self.root_dir, path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, label
