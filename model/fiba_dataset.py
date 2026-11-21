"""
Build a small FIBA training set from the AUS-IND clip, so the detector learns FIBA courts and jerseys.

  1. Click the ball on each frame (resumes where you stopped):
       python fiba_dataset.py label
  2. Build the COCO dataset and zip it for Colab:
       python fiba_dataset.py build

The ball comes from your clicks. Players and referees are labelled automatically by our fine-tuned model.
Frames are split by time so the test frames come from a part of the clip never used for training:
train 0:10-1:20, val 1:20-1:30, test 1:30-2:00.
"""

import argparse
import csv
import json
import os
import shutil

import cv2
import numpy as np

from basketball_pipeline import FINETUNED_MODEL, FinetunedDetector

VIDEO = "runs/aus_ind_14m_to_16m_clip.mp4"
OUT_DIR = "datasets/fiba_ausind"
LABELS = os.path.join(OUT_DIR, "ball_clicks.csv")
# (split, start second, end second, number of frames)
SPLITS = [("train", 10, 80, 220), ("valid", 80, 90, 30), ("test", 90, 120, 100)]
PERSON_CONF = 0.5
CATEGORIES = [{"id": 0, "name": "ball"}, {"id": 1, "name": "player"}, {"id": 2, "name": "referee"}]


def frame_plan():
    """Return [(split, frame_index)] for every frame to label."""
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    plan = []
    for split, start, end, count in SPLITS:
        for idx in np.linspace(start * fps, end * fps, count, endpoint=False).astype(int):
            plan.append((split, int(idx)))
    return plan


def read_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None


def read_labels():
    labels = {}
    if os.path.exists(LABELS):
        with open(LABELS, newline="") as f:
            for row in csv.DictReader(f):
                labels[int(row["frame"])] = row
    return labels


def write_labels(labels):
    with open(LABELS, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "split", "status", "x", "y"])
        writer.writeheader()
        writer.writerows(labels[k] for k in sorted(labels))


def cmd_label(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    labels = read_labels()
    plan = frame_plan()
    cap = cv2.VideoCapture(VIDEO)
    window = "Click ball | n=not visible | x=replay/close-up | b=back | q=save+quit"
    clicked = []
    scale = 1280 / cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x / scale, y / scale))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    pos = 0
    while pos < len(plan):
        split, idx = plan[pos]
        if idx in labels:
            pos += 1
            continue
        frame = read_frame(cap, idx)
        if frame is None:
            pos += 1
            continue
        view = cv2.resize(frame, None, fx=scale, fy=scale)
        done = sum(1 for _, i in plan if i in labels)
        cv2.putText(view, f"{done + 1}/{len(plan)}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow(window, view)
        clicked.clear()
        while True:
            key = cv2.waitKey(30) & 0xFF
            row = None
            if clicked:
                x, y = clicked[0]
                row = {"frame": idx, "split": split, "status": "ball", "x": round(x, 1), "y": round(y, 1)}
            elif key == ord("n"):
                row = {"frame": idx, "split": split, "status": "no_ball", "x": "", "y": ""}
            elif key == ord("x"):
                row = {"frame": idx, "split": split, "status": "exclude", "x": "", "y": ""}
            elif key == ord("b"):
                # Go back to the most recently labelled frame and label it again
                previous = [p for p in range(pos - 1, -1, -1) if plan[p][1] in labels]
                if previous:
                    pos = previous[0]
                    labels.pop(plan[pos][1])
                    write_labels(labels)
                break
            elif key == ord("q"):
                pos = len(plan)
                break
            if row:
                labels[idx] = row
                write_labels(labels)
                pos += 1
                break
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {len(labels)}/{len(plan)} labels to {LABELS}")


def ball_box(click, detections, default_size):
    """Use the model's ball box if one is close to the click, otherwise a box of typical ball size."""
    x, y = click
    near = [d for d in detections if d["cls"] == "ball"
            and np.hypot((d["xyxy"][0] + d["xyxy"][2]) / 2 - x, (d["xyxy"][1] + d["xyxy"][3]) / 2 - y) <= default_size]
    if near:
        x1, y1, x2, y2 = max(near, key=lambda d: d["conf"])["xyxy"]
        return [x1, y1, x2 - x1, y2 - y1]
    return [x - default_size / 2, y - default_size / 2, default_size, default_size]


def cmd_build(args):
    labels = {k: v for k, v in read_labels().items() if v["status"] != "exclude"}
    if not labels:
        raise SystemExit("No labels yet. Run: python fiba_dataset.py label")
    detector = FinetunedDetector(FINETUNED_MODEL)
    cap = cv2.VideoCapture(VIDEO)

    detections, frames = {}, {}
    for idx in labels:
        frames[idx] = read_frame(cap, idx)
        detections[idx] = detector(frames[idx])
    cap.release()
    confident = [d["xyxy"] for dets in detections.values() for d in dets if d["cls"] == "ball" and d["conf"] >= 0.5]
    default_size = float(np.median([max(b[2] - b[0], b[3] - b[1]) for b in confident])) if confident else 28.0

    coco = {s: {"images": [], "annotations": [], "categories": CATEGORIES} for s, *_ in SPLITS}
    for s in coco:
        shutil.rmtree(os.path.join(OUT_DIR, s), ignore_errors=True)
        os.makedirs(os.path.join(OUT_DIR, s))
    ann_id = 0
    for idx, row in sorted(labels.items()):
        split = row["split"]
        name = f"ausind_{idx:06d}.jpg"
        cv2.imwrite(os.path.join(OUT_DIR, split, name), frames[idx], [cv2.IMWRITE_JPEG_QUALITY, 95])
        h, w = frames[idx].shape[:2]
        coco[split]["images"].append({"id": idx, "file_name": name, "width": w, "height": h})
        boxes = [(d["cls"], [d["xyxy"][0], d["xyxy"][1], d["xyxy"][2] - d["xyxy"][0], d["xyxy"][3] - d["xyxy"][1]])
                 for d in detections[idx] if d["cls"] in ("player", "referee") and d["conf"] >= PERSON_CONF]
        if row["status"] == "ball":
            boxes.append(("ball", ball_box((float(row["x"]), float(row["y"])), detections[idx], default_size)))
        for cls, bbox in boxes:
            cat = next(c["id"] for c in CATEGORIES if c["name"] == cls)
            coco[split]["annotations"].append({"id": ann_id, "image_id": idx, "category_id": cat, "bbox": bbox,
                                               "area": bbox[2] * bbox[3], "iscrowd": 0})
            ann_id += 1
    for split, data in coco.items():
        with open(os.path.join(OUT_DIR, split, "_annotations.coco.json"), "w") as f:
            json.dump(data, f)
        balls = sum(a["category_id"] == 0 for a in data["annotations"])
        print(f"{split}: {len(data['images'])} images, {balls} balls, {len(data['annotations']) - balls} people")

    archive = shutil.make_archive("fiba_ausind", "zip", OUT_DIR, ".")
    print(f"Typical ball size: {default_size:.0f}px. Saved {archive} - upload it to Google Drive (My Drive).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["label", "build"])
    args = parser.parse_args()
    {"label": cmd_label, "build": cmd_build}[args.command](args)


if __name__ == "__main__":
    main()
