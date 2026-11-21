"""
Train a dedicated jersey-number reader (ResNet-18, two digit heads) on datasets/numbers (number_dataset.py) and compare
it with the current reader (Qwen3-VL-2B) on the hand-labelled test crops.

  python train_numbers.py                 # train, then score both readers on datasets/numbers/test.csv
  python train_numbers.py --compare-only  # only score (uses weights/number_reader.pt)

Heads: tens digit (none, 0-9) and units digit (0-9), so numbers never seen in training can still be read.
Training players are split by track: 10% of tracks are held out to choose the best epoch. The test game
(Australia vs India) is never used for training.
"""

import argparse
import csv
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2

DATA = "datasets/numbers"
WEIGHTS = "weights/number_reader.pt"
SIZE = 96


def split_label(label):
    label = str(label)
    return (0, int(label)) if len(label) == 1 else (1 + int(label[0]), int(label[1]))   # tens 0 = none


def join_label(tens, units):
    return str(units) if tens == 0 else f"{tens - 1}{units}"


def letterbox(img):
    """Keep the aspect ratio: scale the longer side to SIZE and pad with the image's mean colour."""
    h, w = img.shape[:2]
    s = SIZE / max(h, w)
    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_CUBIC)
    out = np.full((SIZE, SIZE, 3), img.reshape(-1, 3).mean(0), np.uint8)
    y, x = (SIZE - img.shape[0]) // 2, (SIZE - img.shape[1]) // 2
    out[y:y + img.shape[0], x:x + img.shape[1]] = img
    return out


class Crops(Dataset):
    def __init__(self, rows, train):
        self.rows, self.train = rows, train
        self.aug = v2.Compose([
            v2.RandomApply([v2.ColorJitter(0.35, 0.35, 0.35, 0.05)], p=0.8),
            v2.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.85, 1.15), shear=8),
            v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 1.6))], p=0.35),
            v2.RandomGrayscale(p=0.1),
        ])
        self.norm = v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = cv2.cvtColor(cv2.imread(r["path"]), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(letterbox(img)).permute(2, 0, 1).float() / 255
        if self.train:
            x = self.aug(x)
        tens, units = split_label(r["label"])
        return self.norm(x), tens, units


class Reader(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        net.fc = nn.Identity()
        self.body = net
        self.tens = nn.Linear(512, 11)
        self.units = nn.Linear(512, 10)

    def forward(self, x):
        f = self.body(x)
        return self.tens(f), self.units(f)


def predict(model, loader, device):
    model.eval()
    out = []
    with torch.no_grad():
        for x, _, _ in loader:
            t, u = model(x.to(device))
            pt, pu = t.softmax(1), u.softmax(1)
            ct, it = pt.max(1)
            cu, iu = pu.max(1)
            out += [(join_label(a, b), float(c1 * c2)) for a, b, c1, c2 in zip(it.tolist(), iu.tolist(), ct, cu)]
    return out


def read_rows(name):
    return list(csv.DictReader(open(os.path.join(DATA, f"{name}.csv"), newline="")))


def score(name, preds, labels, threshold=0.0):
    kept = [(p, l) for (p, c), l in zip(preds, labels) if p is not None and c >= threshold]
    right = sum(p == l for p, l in kept)
    print(f"  {name:34} answered {len(kept)}/{len(labels)} ({100 * len(kept) / max(1, len(labels)):.0f}%) | "
          f"correct {right}/{len(kept)} ({100 * right / max(1, len(kept)):.0f}% of answers, "
          f"{100 * right / max(1, len(labels)):.0f}% of all)")


def compare(device, weights=WEIGHTS, extra_tests=()):
    for name in extra_tests:
        rows = read_rows(name)
        model = Reader(pretrained=False).to(device)
        model.load_state_dict(torch.load(weights, map_location=device))
        print(); print(f"{name}: {len(rows)} crops")
        score("New reader", predict(model, DataLoader(Crops(rows, False), batch_size=128), device), [r["label"] for r in rows])
    test = read_rows("test")
    labels = [r["label"] for r in test]
    print(f"\nTest crops (Australia vs India, hand labels, never trained on): {len(test)}")
    model = Reader(pretrained=False).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    ours = predict(model, DataLoader(Crops(test, False), batch_size=64), device)
    for th in (0.0, 0.5, 0.8):
        score(f"New reader (confidence >= {th})", ours, labels, th)
    from jersey_numbers import NumberReader
    qwen = NumberReader()
    crops = [cv2.imread(r["path"]) for r in test]
    qp = []
    for i in range(0, len(crops), 16):
        qp += qwen(crops[i:i + 16])
    score("Current reader (Qwen3-VL-2B, 224 px)", qp, labels)
    wrong = [(l, p) for (p, _), l in zip(ours, labels) if p != l]
    print("  new reader mistakes (true -> read):", ", ".join(f"{l}->{p}" for l, p in wrong[:30]))


def train(args, device):
    rows = read_rows("train") + [r for name in args.extra for r in read_rows(name)]
    tracks = sorted({(r["run"], r["track"]) for r in rows})
    random.Random(0).shuffle(tracks)
    held = set(tracks[: max(1, len(tracks) // 10)])
    tr = [r for r in rows if (r["run"], r["track"]) not in held]
    va = [r for r in rows if (r["run"], r["track"]) in held]
    print(f"Training crops {len(tr)} | validation crops {len(va)} (held-out tracks {len(held)})")
    model = Reader().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(Crops(tr, True), batch_size=64, shuffle=True, num_workers=4, persistent_workers=True, drop_last=True)
    vloader = DataLoader(Crops(va, False), batch_size=128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(loader))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    best = -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, losses = time.time(), []
        for x, t, u in loader:
            x, t, u = x.to(device), t.to(device), u.to(device)
            pt, pu = model(x)
            loss = loss_fn(pt, t) + loss_fn(pu, u)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        preds = predict(model, vloader, device)
        acc = np.mean([p == r["label"] for (p, _), r in zip(preds, va)])
        if acc > best:
            best = acc
            torch.save(model.state_dict(), args.out)
        print(f"Epoch {epoch:2}/{args.epochs} | loss {np.mean(losses):.3f} | validation accuracy {acc:.3f}"
              f"{'  (best, saved)' if acc == best else ''} | {time.time() - started:.0f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument("--extra", nargs="*", default=[], help="more training sets, e.g. ebard_train")
    parser.add_argument("--extra-test", nargs="*", default=[], help="more test sets, e.g. ebard_test")
    parser.add_argument("--out", default=WEIGHTS)
    args = parser.parse_args()
    torch.manual_seed(0)
    random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.compare_only:
        train(args, device)
    compare(device, args.out, args.extra_test)


if __name__ == "__main__":
    main()
