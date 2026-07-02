# train.py
"""
Simple transfer-learning trainer for bird species classification.
Assumes folders: data/train, data/val, data/test with one subfolder per species.
"""

import os
import json
from pathlib import Path
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ---------- CONFIG ----------
DATA_DIR = "data"           # root data folder (contains train, val, test)
MODEL_PATH = "model.pth"    # where best model will be saved
LABELS_PATH = "labels.json" # where class names will be saved
BATCH_SIZE = 32
NUM_EPOCHS = 8
LR = 1e-4
NUM_WORKERS = 4             # set 0 on Windows if issues
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------

def get_data_loaders(data_dir, batch_size, num_workers, image_size):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    # transformations: resize, augment for train, only resize for val
    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_ds.classes

def build_model(num_classes):
    # load pretrained ResNet-50 and replace final layer
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss()
    running_loss = 0.0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            running_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / total if total > 0 else 0
    avg_loss = running_loss / total if total > 0 else 0
    return acc, avg_loss

def train():
    train_loader, val_loader, classes = get_data_loaders(DATA_DIR, BATCH_SIZE, NUM_WORKERS, IMAGE_SIZE)
    num_classes = len(classes)
    print("Found classes:", num_classes)
    print(classes)

    model = build_model(num_classes)
    model = model.to(DEVICE)

    # OPTIONAL: freeze backbone to train only final layer (uncomment if you have few images)
    # for name, param in model.named_parameters():
    #     if "fc" not in name:
    #         param.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            pbar.set_postfix({"loss": f"{running_loss / ((pbar.n + 1) * train_loader.batch_size):.4f}"})

        val_acc, val_loss = evaluate(model, val_loader, DEVICE)
        print(f"\nEpoch {epoch} -> val_acc: {val_acc:.4f}, val_loss: {val_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved best model (val_acc={best_val_acc:.4f}) to {MODEL_PATH}")

    # save class names in order
    with open(LABELS_PATH, "w") as f:
        json.dump(classes, f)
    print("Training complete. Best val acc:", best_val_acc)

if __name__ == "__main__":
    # quick checks
    if not Path(DATA_DIR).exists():
        raise SystemExit(f"Data folder '{DATA_DIR}' not found. Put train/val/test inside it.")
    train()
