import torch
import numpy as np
from sklearn.metrics import f1_score

def compute_f1(pred, target):
    pred = torch.argmax(pred, dim=1).cpu().numpy()
    target = target.cpu().numpy()
    return f1_score(target, pred, average='macro')

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model
