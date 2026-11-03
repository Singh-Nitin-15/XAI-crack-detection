"""Evaluate model on a test dataset folder and print metrics."""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

MODEL_PATH = Path(__file__).parent / "model.pth"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


def evaluate(test_dir, batch_size=32):
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(test_dir, transform=tf)
    print(f"Classes: {ds.classes}")
    print(f"Test samples: {len(ds)}")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = load_model()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nAccuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"  [[{cm[0][0]:>4d}, {cm[0][1]:>4d}],")
    print(f"   [{cm[1][0]:>4d}, {cm[1][1]:>4d}]]")
    print(f"  (rows=actual, cols=predicted)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_dir", type=str, help="Path to test folder with class subfolders")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    evaluate(args.test_dir, args.batch_size)

