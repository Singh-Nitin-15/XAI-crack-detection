"""
Train ResNet18 for concrete crack classification.
Uses a synthetic dataset generated procedurally since Kaggle access is unavailable.
The synthetic dataset creates crack-like patterns for positive class and plain textures for negative.
"""
import os
import sys
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFilter

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = Path(__file__).parent / "dataset"
MODEL_PATH = Path(__file__).parent / "model.pth"
NUM_TRAIN = 800
NUM_VAL = 200
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_concrete_texture(size=IMG_SIZE):
    base_color = random.randint(140, 200)
    img = Image.new("RGB", (size, size), (base_color, base_color, base_color))
    pixels = img.load()
    for _ in range(size * size // 4):
        x, y = random.randint(0, size - 1), random.randint(0, size - 1)
        noise = random.randint(-30, 30)
        c = max(0, min(255, base_color + noise))
        pixels[x, y] = (c, c, c)
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    return img


def draw_crack_pattern(img):
    draw = ImageDraw.Draw(img)
    size = img.size[0]
    num_cracks = random.randint(1, 4)
    for _ in range(num_cracks):
        points = []
        x, y = random.randint(0, size), random.randint(0, size)
        segments = random.randint(5, 15)
        for _ in range(segments):
            x += random.randint(-20, 20)
            y += random.randint(-20, 20)
            x = max(0, min(size - 1, x))
            y = max(0, min(size - 1, y))
            points.append((x, y))
        if len(points) >= 2:
            width = random.randint(1, 3)
            dark = random.randint(10, 60)
            draw.line(points, fill=(dark, dark, dark), width=width)
    return img


def generate_dataset():
    for split, count in [("train", NUM_TRAIN), ("val", NUM_VAL)]:
        for label, name in [(0, "negative"), (1, "positive")]:
            folder = DATA_DIR / split / name
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                img = generate_concrete_texture()
                if label == 1:
                    img = draw_crack_pattern(img)
                img.save(folder / f"{name}_{i:04d}.jpg")
    print(f"Dataset generated: {NUM_TRAIN} train + {NUM_VAL} val per class")


class CrackDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for label, name in [(0, "negative"), (1, "positive")]:
            folder = Path(root_dir) / name
            if folder.exists():
                for f in folder.iterdir():
                    if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        self.samples.append((str(f), label))
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def train():
    print(f"Using device: {DEVICE}")

    if not (DATA_DIR / "train").exists():
        print("Generating synthetic crack dataset...")
        generate_dataset()

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = CrackDataset(DATA_DIR / "train", train_transform)
    val_ds = CrackDataset(DATA_DIR / "val", val_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = get_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total * 100
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, preds = outputs.max(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total * 100
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f} - Train Acc: {train_acc:.1f}% - Val Acc: {val_acc:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\nTraining complete. Best validation accuracy: {best_acc:.1f}%")
    print(f"Model saved to: {MODEL_PATH}")
    return best_acc


if __name__ == "__main__":
    train()
