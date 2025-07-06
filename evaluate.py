import torch
from torch.utils.data import DataLoader
from models.ega_net import EGANet
from dataset.private_dataset import PrivateVoiceDataset
from utils.train_utils import load_model, compute_f1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EGANet(in_channels=1, num_classes=3).to(device)
model = load_model(model, "best_model.pth", device)
model.eval()

test_set = PrivateVoiceDataset(
    root_dir="data/test",
    label_file="data/test_labels.csv"
)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

all_preds = []
all_labels = []

with torch.no_grad():
    for wave, label in test_loader:
        wave, label = wave.to(device), label.to(device)
        out = model(wave)
        all_preds.append(out)
        all_labels.append(label)

preds = torch.cat(all_preds)
labels = torch.cat(all_labels)
f1 = compute_f1(preds, labels)
print("Test F1 Score:", round(f1, 4))
