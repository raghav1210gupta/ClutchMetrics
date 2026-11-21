"""
Build a jersey-number crop dataset for training a dedicated number reader.

  python number_dataset.py build --runs qat3 upload_0d063a08d734 demo45   # training crops (auto-labelled)
  python number_dataset.py test  --name aus_ind_v2                        # test crops from the 150 hand labels

Training crops: number patches (found by the 5-class model) on player tracks whose number the current system confirmed
by voting; the label is that confirmed number (about 90% right, good enough for training). At most PER_TRACK crops per
track, spread over its life, so long tracks do not dominate.
Test crops: the number patch inside each hand-labelled player box (label_numbers.py) whose number is visible; the label
is the hand label. The test game is never used for training.

Output: datasets/numbers/<split>/<label>_<run>_<frame>_<track>_<i>.jpg  +  datasets/numbers/<split>.csv
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import cv2
from PIL import Image

from basketball_pipeline import run_dir, sampled_frames
from jersey_numbers import MIN_NUMBER_HEIGHT, NUMBER_CLASS, NUMBER_CONF, NUMBER_MODEL, owner_of

OUT = "datasets/numbers"
PER_TRACK = 20


def crop_number(frame, box):
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * 0.25, (y2 - y1) * 0.2          # same margin as the live reader
    return frame[int(max(0, y1 - py)):int(y2 + py), int(max(0, x1 - px)):int(x2 + px)]


def detector():
    from rfdetr import RFDETRNano
    return RFDETRNano(pretrain_weights=NUMBER_MODEL, num_classes=5)


def number_boxes(model, frame):
    found = model.predict(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), threshold=NUMBER_CONF)
    return [b for b, c in zip(found.xyxy, found.class_id) if c == NUMBER_CLASS and b[3] - b[1] >= MIN_NUMBER_HEIGHT]


def cmd_build(args):
    model = detector()
    split_dir = os.path.join(OUT, "train")
    os.makedirs(split_dir, exist_ok=True)
    rows = []
    for name in args.runs:
        out_dir = run_dir(name)
        meta = json.load(open(os.path.join(out_dir, "meta.json")))
        players = json.load(open(os.path.join(out_dir, "players.json")))["tracks"]
        confirmed = {int(t): p["number"] for t, p in players.items() if p.get("status") == "confirmed" and p.get("number")}
        by_frame = defaultdict(list)
        frames_of = defaultdict(list)
        for r in csv.DictReader(open(os.path.join(out_dir, "tracks_named.csv"), newline="")):
            t = int(r["track"])
            if r["team"] in ("0", "1"):
                by_frame[int(r["frame"])].append((t, int(r["team"]), [float(r[k]) for k in ("x1", "y1", "x2", "y2")]))
                if t in confirmed:
                    frames_of[t].append(int(r["frame"]))
        # frames to look at: up to 3 x PER_TRACK evenly spread frames per confirmed track (not every patch is readable)
        wanted = defaultdict(set)
        for t, fs in frames_of.items():
            step = max(1, len(fs) // (3 * PER_TRACK))
            for f in fs[::step]:
                wanted[f].add(t)
        got = defaultdict(int)
        cap = cv2.VideoCapture(meta["video"])
        for idx, frame in sampled_frames(cap, meta["start"], meta["end"], meta["fps"]):
            if idx not in wanted:
                continue
            for box in number_boxes(model, frame):
                t = owner_of(box, by_frame[idx])
                if t not in wanted[idx] or got[t] >= PER_TRACK:
                    continue
                crop = crop_number(frame, box)
                if crop.size == 0:
                    continue
                label = confirmed[t]
                path = os.path.join(split_dir, f"{label}_{name}_{idx}_{t}_{got[t]}.jpg")
                cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append({"path": path, "label": label, "run": name, "frame": idx, "track": t})
                got[t] += 1
        cap.release()
        print(f"{name}: {sum(got.values())} crops from {len(got)} tracks ({len(confirmed)} confirmed)")
    with open(os.path.join(OUT, "train.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "run", "frame", "track"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(rows)} training crops to {OUT}/train.csv")


def cmd_test(args):
    out_dir = run_dir(args.name)
    meta = json.load(open(os.path.join(out_dir, "meta.json")))
    labels = list(csv.DictReader(open(os.path.join(out_dir, "number_labels.csv"), newline="")))
    visible = [r for r in labels if r["label"] not in ("none", "bad") and r.get("x1")]
    model = detector()
    split_dir = os.path.join(OUT, "test")
    os.makedirs(split_dir, exist_ok=True)
    cap = cv2.VideoCapture(meta["video"])
    rows, missing = [], 0
    for r in visible:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["frame"]))
        ok, frame = cap.read()
        if not ok:
            missing += 1
            continue
        player = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
        boxes = [b for b in number_boxes(model, frame) if owner_of(b, [(0, 0, player)]) == 0]
        if not boxes:
            missing += 1
            continue
        box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        path = os.path.join(split_dir, f"{r['label']}_{args.name}_{r['frame']}_{r['track']}.jpg")
        cv2.imwrite(path, crop_number(frame, box), [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append({"path": path, "label": r["label"], "run": args.name, "frame": r["frame"], "track": r["track"]})
    with open(os.path.join(OUT, "test.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "run", "frame", "track"])
        w.writeheader()
        w.writerows(rows)
    print(f"Test: {len(rows)} number crops from {len(visible)} hand-labelled visible numbers "
          f"({missing} where the number patch was not detected)")


def cmd_ebard(args):
    """E-BARD Jersey Number Recognition dataset (Hugging Face GabrieleGiudici/E-BARD-JerseyNumberRecognition,
    CC-BY-4.0): NBA broadcast torso crops, one hand label per 16-frame sequence. Takes PER_SEQUENCE frames spread
    over each labelled sequence; E-BARD's own test split is kept apart as a second test (ebard_test.csv)."""
    import zipfile
    z = zipfile.ZipFile(args.zip)
    names = [n for n in z.namelist() if n.startswith("annotated_crop/") and n.endswith(".png")]
    by_folder = defaultdict(list)
    for n in names:
        by_folder[n.rsplit("/", 1)[0]].append(n)
    for split_file, out_name in (("train.txt", "ebard_train"), ("val.txt", "ebard_train"), ("test.txt", "ebard_test")):
        rows = []
        if os.path.exists(os.path.join(OUT, f"{out_name}.csv")) and split_file == "val.txt":
            rows = list(csv.DictReader(open(os.path.join(OUT, f"{out_name}.csv"), newline="")))
        split_dir = os.path.join(OUT, out_name)
        os.makedirs(split_dir, exist_ok=True)
        for line in z.read(split_file).decode().splitlines():
            parts = line.split()
            if len(parts) < 3 or parts[-1] == "nan" or not parts[-1].isdigit():
                continue
            folder, label = parts[0].replace("annotated", "annotated_crop", 1), parts[-1]
            frames = sorted(by_folder.get(folder, []), key=lambda n: int(n.rsplit("_", 1)[1].split(".")[0]))
            if not frames:
                continue
            pick = frames[:: max(1, len(frames) // args.per_sequence)][: args.per_sequence]
            for n in pick:
                path = os.path.join(split_dir, f"{label}_{folder.split('/')[-1]}_{n.rsplit('/', 1)[1]}")
                with open(path, "wb") as f:
                    f.write(z.read(n))
                rows.append({"path": path, "label": label, "run": "ebard", "frame": n.rsplit("/", 1)[1], "track": folder})
        with open(os.path.join(OUT, f"{out_name}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "label", "run", "frame", "track"])
            w.writeheader()
            w.writerows(rows)
        print(f"{split_file}: {out_name}.csv now has {len(rows)} crops")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build", "test", "ebard"])
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--name", default="aus_ind_v2")
    parser.add_argument("--zip", default="datasets/ebard/all.zip")
    parser.add_argument("--per-sequence", type=int, default=6)
    args = parser.parse_args()
    {"build": cmd_build, "test": cmd_test, "ebard": cmd_ebard}[args.command](args)


if __name__ == "__main__":
    main()
