import torch
import torch.nn as nn
from tqdm import tqdm
import config

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for img, lab in tqdm(loader):
        img, lab = img.to(config.DEVICE), lab.to(config.DEVICE)
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, lab)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = out.argmax(1)
        correct += (pred == lab).sum().item()
        total += lab.size(0)

    return total_loss / len(loader), correct / total

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for img, lab in tqdm(loader):
            img, lab = img.to(config.DEVICE), lab.to(config.DEVICE)
            out = model(img)
            loss = criterion(out, lab)

            total_loss += loss.item()
            pred = out.argmax(1)
            correct += (pred == lab).sum().item()
            total += lab.size(0)

    return total_loss / len(loader), correct / total