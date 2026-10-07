"""Part 1: CNN image classifier (PyTorch) on Fashion-MNIST."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CLASSES = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")
EPOCHS = 5
OUT = "outputs/cnn"


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28 -> 14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return loss_sum / n, correct / n


def main():
    os.makedirs(OUT, exist_ok=True)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    train_ds = datasets.FashionMNIST("data", train=True, download=True, transform=tf)
    test_ds = datasets.FashionMNIST("data", train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    hist = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for epoch in range(1, EPOCHS + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer)
        te = run_epoch(model, test_loader, criterion)
        for key, val in zip(hist, (*tr, *te)):
            hist[key].append(val)
        print(f"Epoch {epoch}/{EPOCHS} | train acc {tr[1]:.3f} | test acc {te[1]:.3f}")
    torch.save(model.state_dict(), f"{OUT}/cnn.pt")

    # Per-class accuracy with NumPy
    model.eval()
    preds, labels, images = [], [], None
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            preds.append(model(x).argmax(1).numpy())
            labels.append(y.numpy())
            if i == 0:
                images = x[:12]
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    print(f"\nFinal test accuracy: {(preds == labels).mean():.3f}")
    for c, name in enumerate(CLASSES):
        print(f"  {name:<12} {(preds[labels == c] == c).mean():.3f}")

    # Plot 1: training curves
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["test_loss"], label="test")
    ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(hist["train_acc"], label="train"); ax[1].plot(hist["test_acc"], label="test")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/training_curves.png", dpi=150); plt.close(fig)

    # Plot 2: sample predictions
    fig, axes = plt.subplots(3, 4, figsize=(8, 6.5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i, 0], cmap="gray"); ax.axis("off")
        ok = preds[i] == labels[i]
        ax.set_title(f"{CLASSES[preds[i]]}", color="green" if ok else "red", fontsize=8)
    fig.suptitle("Sample predictions (green = correct)")
    fig.tight_layout(); fig.savefig(f"{OUT}/sample_predictions.png", dpi=150); plt.close(fig)
    print(f"Saved results in ./{OUT}")


if __name__ == "__main__":
    main()
