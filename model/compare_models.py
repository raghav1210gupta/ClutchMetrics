"""
Compare our two RF-DETR models on frames with human labels, to pick the best one for the ball and for players.

    python compare_models.py

Models:
  v2      - our fine-tuned RF-DETR Nano 576px (weights/rfdetr_basketball_v2/best)
  5class  - RF-DETR Nano 384px with ball / number / player / referee / rim (weights/rfdetr_5class)

Tests:
  Knicks  - 195 frames of an NBA video where we clicked the ball by hand (not used to train either model)
  FIBA    - 79 AUS-IND test frames (1:30-2:00 of the clip, never used in training); ball clicked by hand

Metrics:
  Ball F1  - a detection counts if its centre is within 25 px of the clicked ball; best over confidence 0.1-0.5
  Ball AP50, Player AP50, Referee AP50 - standard detection AP at IoU 0.5 (FIBA test only).
  Note: FIBA player/referee boxes were drawn automatically by our v1 model, then spot-checked,
  so player AP there slightly favours v2. The ball labels are fully human.
Results are saved to runs/model_comparison/report.json
"""

import csv
import json
import os
import time

import cv2
import numpy as np
from PIL import Image

from basketball_pipeline import FINETUNED_MODEL, FinetunedDetector
from compare_detectors import sample_frames
from train_rfdetr import average_precision

FIVE_CLASS = "weights/rfdetr_5class/checkpoint_best_total.pth"
FIVE_NAMES = ["ball", "number", "player", "referee", "rim"]
CLASSES = ["ball", "player", "referee"]
KNICKS_VIDEO = "datasets/test_videos/knicks_76ers_highlights.mp4"
KNICKS_LABELS = "runs/detector_comparison/knicks_76ers_highlights/ball_labels.csv"
FIBA_TEST = "datasets/fiba_ausind/test"
RADIUS = 25
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]
OUT = "runs/model_comparison"


class FiveClass:
    def __init__(self):
        from rfdetr import RFDETRNano
        self.model = RFDETRNano(pretrain_weights=FIVE_CLASS, num_classes=5)

    def __call__(self, frame):
        d = self.model.predict(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), threshold=0.05)
        return [{"cls": FIVE_NAMES[c], "conf": float(p), "xyxy": [float(v) for v in b]}
                for b, c, p in zip(d.xyxy, d.class_id, d.confidence) if FIVE_NAMES[c] in CLASSES]


def ball_f1(predictions, labels):
    """labels: {key: (x, y) or None}. Best F1 over thresholds."""
    with_ball = sum(v is not None for v in labels.values())
    best = None
    for conf in THRESHOLDS:
        tp = fp = 0
        for key, truth in labels.items():
            hit = False
            for d in predictions[key]:
                if d["cls"] != "ball" or d["conf"] < conf:
                    continue
                x1, y1, x2, y2 = d["xyxy"]
                if truth and not hit and np.hypot((x1 + x2) / 2 - truth[0], (y1 + y2) / 2 - truth[1]) <= RADIUS:
                    hit = True
                else:
                    fp += 1
            tp += hit
        recall, precision = tp / max(1, with_ball), tp / max(1, tp + fp)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        if best is None or f1 > best["f1"]:
            best = {"f1": f1, "recall": recall, "precision": precision, "conf": conf,
                    "false_per_frame": fp / max(1, len(labels))}
    return best


def ap50(predictions, truths):
    preds, gts = [], []
    for key, boxes in truths.items():
        p = predictions[key]
        preds.append((np.array([d["xyxy"] for d in p]).reshape(-1, 4), np.array([d["conf"] for d in p]),
                      np.array([CLASSES.index(d["cls"]) for d in p])))
        gts.append((np.array([b for b, _ in boxes]).reshape(-1, 4), np.array([CLASSES.index(c) for _, c in boxes])))
    return {c: average_precision(preds, gts, i) for i, c in enumerate(CLASSES)}


def load_knicks():
    labels = {}
    with open(KNICKS_LABELS, newline="") as f:
        for r in csv.DictReader(f):
            if r["status"] != "exclude":
                labels[int(r["frame"])] = (float(r["x"]), float(r["y"])) if r["status"] == "ball" else None
    frames = {idx: frame for idx, frame in sample_frames(KNICKS_VIDEO, 250) if idx in labels}
    return frames, {k: v for k, v in labels.items() if k in frames}


def load_fiba():
    coco = json.load(open(os.path.join(FIBA_TEST, "_annotations.coco.json")))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    frames, boxes, balls = {}, {}, {}
    for img in coco["images"]:
        frames[img["id"]] = cv2.imread(os.path.join(FIBA_TEST, img["file_name"]))
        boxes[img["id"]], balls[img["id"]] = [], None
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        boxes[a["image_id"]].append(([x, y, x + w, y + h], names[a["category_id"]]))
        if names[a["category_id"]] == "ball":
            balls[a["image_id"]] = (x + w / 2, y + h / 2)
    return frames, boxes, balls


def run(model, frames):
    out, times = {}, []
    for key, frame in frames.items():
        t = time.time()
        out[key] = model(frame)
        times.append(time.time() - t)
    return out, 1000 * float(np.median(times[1:]))


def main():
    os.makedirs(OUT, exist_ok=True)
    knicks_frames, knicks_balls = load_knicks()
    fiba_frames, fiba_boxes, fiba_balls = load_fiba()
    print(f"Knicks: {len(knicks_frames)} frames ({sum(v is not None for v in knicks_balls.values())} with ball) | "
          f"FIBA test: {len(fiba_frames)} frames ({sum(v is not None for v in fiba_balls.values())} with ball)")

    report = {}
    for name, make in [("v2", lambda: FinetunedDetector(FINETUNED_MODEL)), ("5class", FiveClass)]:
        model = make()
        knicks_pred, ms = run(model, knicks_frames)
        fiba_pred, _ = run(model, fiba_frames)
        report[name] = {"ms_per_frame": ms, "knicks_ball": ball_f1(knicks_pred, knicks_balls),
                        "fiba_ball": ball_f1(fiba_pred, fiba_balls), "fiba_ap50": ap50(fiba_pred, fiba_boxes)}
        del model

    json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=1)
    print(f"\n{'':24}{'v2':>12}{'5class':>12}")
    rows = [("Knicks ball F1", lambda r: r["knicks_ball"]["f1"], "{:.2f}"),
            ("Knicks ball recall", lambda r: r["knicks_ball"]["recall"], "{:.0%}"),
            ("Knicks ball precision", lambda r: r["knicks_ball"]["precision"], "{:.0%}"),
            ("FIBA ball F1", lambda r: r["fiba_ball"]["f1"], "{:.2f}"),
            ("FIBA ball recall", lambda r: r["fiba_ball"]["recall"], "{:.0%}"),
            ("FIBA ball precision", lambda r: r["fiba_ball"]["precision"], "{:.0%}"),
            ("FIBA ball AP50", lambda r: r["fiba_ap50"]["ball"], "{:.2f}"),
            ("FIBA player AP50", lambda r: r["fiba_ap50"]["player"], "{:.2f}"),
            ("FIBA referee AP50", lambda r: r["fiba_ap50"]["referee"], "{:.2f}"),
            ("Speed (ms/frame)", lambda r: r["ms_per_frame"], "{:.0f}")]
    for label, get, fmt in rows:
        print(f"{label:24}{fmt.format(get(report['v2'])):>12}{fmt.format(get(report['5class'])):>12}")
    print(f"\nSaved {OUT}/report.json")


if __name__ == "__main__":
    main()
