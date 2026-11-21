"""
Fine-tune the Hugging Face RF-DETR Nano basketball model at 576px on several basketball datasets.

Made for a Google Colab A100, but it also runs on any CUDA GPU:
    python train_rfdetr.py --out /content/drive/MyDrive/basketball_rfdetr --fiba fiba_ausind

Data: three public Hugging Face datasets (NBA and college) plus our own FIBA frames from AUS-IND.
Classes are merged into ball / player / referee, the same order as the starting model.
Test scores are reported separately for each dataset, so a gain on FIBA can't hide a loss on NBA.
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image, ImageEnhance
from transformers import AutoImageProcessor, AutoModelForObjectDetection

BASE_MODEL = "koppolusameer/rfdetr-basketball-player-ball-referee-detection"
# Short name -> Hugging Face dataset. Each has train / valid / test folders in COCO format.
DATASETS = {
    "nba_celtics_knicks": "koppolusameer/basketball-coco-20260416",
    "nba_six_games": "koppolusameer/basketball-coco-player-detection",
    "college": "koppolusameer/basketball-coco-2025-04-02",
}
# Our FIBA frames are few, so each one is shown this many times per epoch
FIBA_REPEAT = 4
CLASSES = ["ball", "player", "referee"]
# Lower-case dataset category name -> our class index. Rims, numbers and scoreboard classes are dropped.
CLASS_MAP = {
    "ball": 0, "ball-in-basket": 0,
    "player": 1, "player-in-possession": 1, "player-jump-shot": 1, "player-layup-dunk": 1, "player-shot-block": 1,
    "player-dribble": 1, "player-layup": 1, "player-screen": 1, "player-fall": 1,
    "referee": 2, "ref": 2,
}


def load_split(root, split):
    """Return a list of (image_path, [[x, y, w, h, class], ...]) for one split."""
    with open(os.path.join(root, split, "_annotations.coco.json")) as f:
        coco = json.load(f)
    names = {c["id"]: c["name"].lower() for c in coco["categories"]}
    boxes = {img["id"]: [] for img in coco["images"]}
    for a in coco["annotations"]:
        cls = CLASS_MAP.get(names[a["category_id"]])
        if cls is not None and a["bbox"][2] > 1 and a["bbox"][3] > 1:
            boxes[a["image_id"]].append([*a["bbox"], cls])
    return [(os.path.join(root, split, img["file_name"]), boxes[img["id"]]) for img in coco["images"]]


def augment(image, boxes):
    """Random zoom-in crop (makes the small ball bigger), horizontal flip and lighting changes."""
    w, h = image.size
    if random.random() < 0.6:
        scale = random.uniform(0.55, 1.0)
        cw, ch = int(w * scale), int(h * scale)
        x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
        image = image.crop((x0, y0, x0 + cw, y0 + ch))
        kept = []
        for x, y, bw, bh, c in boxes:
            nx1, ny1 = max(x, x0), max(y, y0)
            nx2, ny2 = min(x + bw, x0 + cw), min(y + bh, y0 + ch)
            # Keep a box only if most of it is still inside the crop
            if nx2 - nx1 > 2 and ny2 - ny1 > 2 and (nx2 - nx1) * (ny2 - ny1) >= 0.6 * bw * bh:
                kept.append([nx1 - x0, ny1 - y0, nx2 - nx1, ny2 - ny1, c])
        boxes = kept
        w, h = cw, ch
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        boxes = [[w - x - bw, y, bw, bh, c] for x, y, bw, bh, c in boxes]
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.25))
    image = ImageEnhance.Color(image).enhance(random.uniform(0.7, 1.3))
    return image, boxes


class BasketballDataset(torch.utils.data.Dataset):
    def __init__(self, items, processor, train):
        self.items, self.processor, self.train = items, processor, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, boxes = self.items[i]
        image = Image.open(path).convert("RGB")
        if self.train:
            image, boxes = augment(image, boxes)
        target = {"image_id": i, "annotations": [
            {"bbox": b[:4], "category_id": b[4], "area": b[2] * b[3], "iscrowd": 0} for b in boxes]}
        enc = self.processor(images=image, annotations=target, return_tensors="pt")
        return enc["pixel_values"][0], enc["labels"][0]


def collate(batch):
    return {"pixel_values": torch.stack([b[0] for b in batch]), "labels": [b[1] for b in batch]}


def box_iou(a, b):
    """IoU between each box in a (N,4) and b (M,4), both xyxy."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def average_precision(preds, truths, cls):
    """AP at IoU 0.5 for one class. preds: per image (boxes, scores, labels). truths: per image (boxes, labels)."""
    scored, total = [], 0
    for img, ((pb, ps, pl), (tb, tl)) in enumerate(zip(preds, truths)):
        gt = tb[tl == cls]
        total += len(gt)
        p = pb[pl == cls]
        s = ps[pl == cls]
        order = np.argsort(-s)
        used = np.zeros(len(gt), bool)
        ious = box_iou(p[order], gt) if len(gt) and len(p) else np.zeros((len(p), 0))
        for rank, k in enumerate(order):
            hit = False
            if len(gt):
                j = int(np.argmax(np.where(used, -1, ious[rank])))
                if not used[j] and ious[rank, j] >= 0.5:
                    used[j] = hit = True
            scored.append((s[k], hit))
    if total == 0:
        return float("nan")
    scored.sort(key=lambda x: -x[0])
    hits = np.array([h for _, h in scored], dtype=float)
    tp, fp = np.cumsum(hits), np.cumsum(1 - hits)
    recall = tp / total
    precision = tp / np.maximum(tp + fp, 1e-9)
    # All-point interpolated AP
    precision = np.concatenate([[0], precision, [0]])
    recall = np.concatenate([[0], recall, [1]])
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    steps = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[steps + 1] - recall[steps]) * precision[steps + 1]))


@torch.no_grad()
def evaluate(model, processor, items, device, limit=None):
    model.eval()
    preds, truths = [], []
    for path, boxes in items[:limit]:
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            outputs = model(**inputs)
        res = processor.post_process_object_detection(outputs, threshold=0.01, target_sizes=[image.size[::-1]])[0]
        preds.append((res["boxes"].float().cpu().numpy(), res["scores"].float().cpu().numpy(), res["labels"].cpu().numpy()))
        tb = np.array([[x, y, x + w, y + h] for x, y, w, h, _ in boxes]).reshape(-1, 4)
        truths.append((tb, np.array([b[4] for b in boxes], dtype=int)))
    ap = {name: average_precision(preds, truths, i) for i, name in enumerate(CLASSES)}
    ap["mean"] = float(np.nanmean(list(ap.values())))
    return ap


def format_ap(ap):
    return " | ".join(f"{k} {100 * v:.1f}" for k, v in ap.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="rfdetr_basketball_576")
    parser.add_argument("--base", default=BASE_MODEL, help="Model to start from: Hugging Face id or a saved folder")
    parser.add_argument("--fiba", default=None, help="Folder with our FIBA dataset (train/valid/test)")
    parser.add_argument("--size", type=int, default=576)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke-test", action="store_true", help="Run a few steps only, to check the code works")
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    sources = {name: snapshot_download(repo, repo_type="dataset") for name, repo in DATASETS.items()}
    if args.fiba:
        sources["fiba_ausind"] = args.fiba
    data = {name: {s: load_split(root, s) for s in ("train", "valid", "test")} for name, root in sources.items()}
    for name, splits in data.items():
        print(f"{name}: " + ", ".join(f"{s} {len(v)}" for s, v in splits.items()))

    train_items = []
    for name, splits in data.items():
        train_items += splits["train"] * (FIBA_REPEAT if name == "fiba_ausind" else 1)
    # Pick the best epoch on the original NBA validation clips plus our FIBA validation frames
    valid_items = data["nba_celtics_knicks"]["valid"] + (data["fiba_ausind"]["valid"] if args.fiba else [])
    print(f"Training images per epoch: {len(train_items)} | validation images: {len(valid_items)}")

    processor = AutoImageProcessor.from_pretrained(BASE_MODEL, size={"height": args.size, "width": args.size})
    model = AutoModelForObjectDetection.from_pretrained(args.base).to(device)

    if args.smoke_test:
        train_items = random.sample(train_items, 8)
    eval_limit = 4 if args.smoke_test else None
    loader = torch.utils.data.DataLoader(
        BasketballDataset(train_items, processor, train=True), batch_size=args.batch, shuffle=True,
        num_workers=0 if args.smoke_test else args.workers, collate_fn=collate, drop_last=True)

    start_ap = evaluate(model, processor, valid_items, device, eval_limit)
    print(f"Starting model, validation AP50: {format_ap(start_ap)}")

    backbone = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": args.backbone_lr}, {"params": rest, "lr": args.lr}],
                                  weight_decay=1e-4)
    epochs = 1 if args.smoke_test else args.epochs
    total_steps = max(1, epochs * len(loader))
    warmup = min(len(loader), total_steps // 10)

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    best_score, history = -1.0, []
    for epoch in range(1, epochs + 1):
        model.train()
        started, losses = time.time(), []
        for step, batch in enumerate(loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(pixel_values=pixel_values, labels=labels).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            if args.smoke_test and step >= 2:
                break

        ap = evaluate(model, processor, valid_items, device, eval_limit)
        # The ball matters most for our analytics, so it counts double when picking the best epoch
        score = (ap["mean"] + ap["ball"]) / 2
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **ap})
        print(f"Epoch {epoch}/{epochs} | loss {np.mean(losses):.3f} | val AP50: {format_ap(ap)} | {time.time() - started:.0f}s")
        if score > best_score:
            best_score = score
            model.save_pretrained(os.path.join(args.out, "best"))
            processor.save_pretrained(os.path.join(args.out, "best"))
            print("  Saved new best model")
        with open(os.path.join(args.out, "history.json"), "w") as f:
            json.dump(history, f, indent=1)

    # Final comparison on each dataset's test images, which were never used for training or choosing the epoch
    best = AutoModelForObjectDetection.from_pretrained(os.path.join(args.out, "best")).to(device)
    start = AutoModelForObjectDetection.from_pretrained(args.base).to(device)
    report = {"input_size": args.size, "starting_model": args.base, "test_AP50": {}}
    print()
    for name, splits in data.items():
        before = evaluate(start, processor, splits["test"], device, eval_limit)
        after = evaluate(best, processor, splits["test"], device, eval_limit)
        report["test_AP50"][name] = {"starting_model": before, "fine_tuned": after}
        print(f"{name} test ({len(splits['test'])} images)")
        print(f"   starting model: {format_ap(before)}")
        print(f"   fine-tuned:     {format_ap(after)}")
    with open(os.path.join(args.out, "test_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
