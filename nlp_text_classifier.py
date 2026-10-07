"""Part 2: NLP text classifier. Pandas for exploration, TF-IDF features,
and a feed-forward neural network built in PyTorch."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

CATEGORIES = ["sci.space", "rec.sport.baseball", "comp.graphics", "talk.politics.misc"]
EPOCHS = 10
OUT = "outputs/nlp"


def main():
    os.makedirs(OUT, exist_ok=True)

    # 1. Load data into a Pandas DataFrame and explore it
    data = fetch_20newsgroups(subset="all", categories=CATEGORIES,
                              remove=("headers", "footers", "quotes"))
    df = pd.DataFrame({"text": data.data, "label": data.target})
    df["label_name"] = df["label"].map(lambda i: data.target_names[i])
    df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    df["word_count"] = df["text"].str.split().str.len()
    print(df.head())
    print("\nClass counts:\n", df["label_name"].value_counts())
    print("\nWord count stats:\n", df["word_count"].describe())

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    df["label_name"].value_counts().plot.bar(ax=ax[0], title="Documents per class")
    ax[1].hist(df["word_count"].clip(upper=1000), bins=40)
    ax[1].set_title("Words per document (clipped at 1000)")
    fig.tight_layout(); fig.savefig(f"{OUT}/data_exploration.png", dpi=150); plt.close(fig)

    # 2. Split, then convert text to TF-IDF vectors (fit on train only)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42)
    vec = TfidfVectorizer(max_features=5000, stop_words="english")
    Xtr = torch.tensor(vec.fit_transform(X_train).toarray(), dtype=torch.float32)
    Xte = torch.tensor(vec.transform(X_test).toarray(), dtype=torch.float32)
    ytr = torch.tensor(y_train.values, dtype=torch.long)
    yte = torch.tensor(y_test.values, dtype=torch.long)

    # 3. Neural network
    model = nn.Sequential(
        nn.Linear(Xtr.shape[1], 128), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(128, len(CATEGORIES)),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)

    losses, test_accs = [], []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            loss = criterion(model(xb), yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item() * len(yb)
        model.eval()
        with torch.no_grad():
            acc = (model(Xte).argmax(1) == yte).float().mean().item()
        losses.append(total / len(ytr)); test_accs.append(acc)
        print(f"Epoch {epoch}/{EPOCHS} | train loss {losses[-1]:.3f} | test acc {acc:.3f}")

    # 4. Evaluate and plot
    with torch.no_grad():
        preds = model(Xte).argmax(1).numpy()
    print("\n", classification_report(y_test.values, preds, target_names=data.target_names))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(losses); ax[0].set_title("Training loss"); ax[0].set_xlabel("Epoch")
    ax[1].plot(test_accs); ax[1].set_title("Test accuracy"); ax[1].set_xlabel("Epoch")
    fig.tight_layout(); fig.savefig(f"{OUT}/training_curves.png", dpi=150); plt.close(fig)

    # 5. Try your own sentences
    samples = ["NASA launched a new telescope into orbit around the Earth",
               "The pitcher threw a fastball and the batter hit a home run",
               "This GPU renders 3D graphics and textures very quickly",
               "The senator proposed a new tax bill in congress"]
    with torch.no_grad():
        out = model(torch.tensor(vec.transform(samples).toarray(), dtype=torch.float32))
    for s, p in zip(samples, out.argmax(1).numpy()):
        print(f"{data.target_names[p]:<22} <- {s}")
    print(f"Saved results in ./{OUT}")


if __name__ == "__main__":
    main()
