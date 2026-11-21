"""
Check how accurate the jersey numbers are, against numbers typed in by a person.

  1. Label 150 random player boxes (resumes where you stopped):
       python label_numbers.py label --name aus_ind_v2
     Type the jersey number you can see and press Enter.
     n = number not visible   x = not a player / two players mixed   b = back   q = save and quit
  2. Score our automatic numbers against your labels:
       python label_numbers.py score --name aus_ind_v2

Boxes are picked at random over the whole clip, so the score reflects the whole video.

Every label is kept as a dataset for later: number_labels.csv (frame, track, label, player box) and the player
picture at full resolution in number_labels/<frame>_<track>.jpg. These 150 are the fixed TEST set: never train on them.
"""

import argparse
import csv
import os
import random

import cv2

from basketball_pipeline import run_dir

SAMPLES = 150
SEED = 7


def load_named(name):
    with open(os.path.join(run_dir(name), "tracks_named.csv"), newline="") as f:
        return [r for r in csv.DictReader(f) if r["team"] in ("0", "1")]


def label_path(name):
    return os.path.join(run_dir(name), "number_labels.csv")


def read_labels(name):
    if not os.path.exists(label_path(name)):
        return {}
    with open(label_path(name), newline="") as f:
        return {(r["frame"], r["track"]): r for r in csv.DictReader(f)}


FIELDS = ["frame", "track", "label", "x1", "y1", "x2", "y2", "crop"]


def write_labels(name, labels):
    with open(label_path(name), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(labels.values())


def pick_samples(rows):
    rows = [r for r in rows if float(r["y2"]) - float(r["y1"]) >= 60]
    random.Random(SEED).shuffle(rows)
    return rows[:SAMPLES]


def cmd_label(args):
    import json
    meta = json.load(open(os.path.join(run_dir(args.name), "meta.json")))
    samples = pick_samples(load_named(args.name))
    labels = read_labels(args.name)
    cap = cv2.VideoCapture(meta["video"])
    window = "Type number + Enter | n=not visible | x=bad box | b=back | q=quit"
    pos = 0
    while pos < len(samples):
        r = samples[pos]
        key_id = (r["frame"], r["track"])
        if key_id in labels:
            pos += 1
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["frame"]))
        ok, frame = cap.read()
        if not ok:
            pos += 1
            continue
        x1, y1, x2, y2 = [int(float(r[k])) for k in ("x1", "y1", "x2", "y2")]
        h, w = y2 - y1, x2 - x1
        # Show the player with some surroundings, the player boxed in yellow
        cx1, cy1 = max(0, x1 - w), max(0, y1 - h // 2)
        cx2, cy2 = min(frame.shape[1], x2 + w), min(frame.shape[0], y2 + h // 2)
        view = frame[cy1:cy2, cx1:cx2].copy()
        cv2.rectangle(view, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (0, 255, 255), 2)
        scale = 520 / view.shape[0]
        view = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        typed = ""
        while True:
            shown = view.copy()
            done = sum(1 for s in samples if (s["frame"], s["track"]) in labels)
            cv2.rectangle(shown, (0, 0), (shown.shape[1], 34), (0, 0, 0), -1)
            cv2.putText(shown, f"{done + 1}/{len(samples)}  number: {typed}_", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(window, shown)
            key = cv2.waitKey(50) & 0xFF
            value = None
            if ord("0") <= key <= ord("9") and len(typed) < 2:
                typed += chr(key)
            elif key == 8:                       # Backspace
                typed = typed[:-1]
            elif key == 13 and typed:            # Enter
                value = typed
            elif key == ord("n"):
                value = "none"
            elif key == ord("x"):
                value = "bad"
            elif key == ord("b"):
                previous = [p for p in range(pos - 1, -1, -1)
                            if (samples[p]["frame"], samples[p]["track"]) in labels]
                if previous:
                    pos = previous[0]
                    labels.pop((samples[pos]["frame"], samples[pos]["track"]))
                    write_labels(args.name, labels)
                break
            elif key == ord("q"):
                pos = len(samples)
                break
            if value:
                # keep the player picture (full resolution, small margin) for training/testing later
                crops = os.path.join(run_dir(args.name), "number_labels")
                os.makedirs(crops, exist_ok=True)
                mx, my = w // 10, h // 20
                crop_name = f"{r['frame']}_{r['track']}.jpg"
                cv2.imwrite(os.path.join(crops, crop_name),
                            frame[max(0, y1 - my):min(frame.shape[0], y2 + my), max(0, x1 - mx):min(frame.shape[1], x2 + mx)],
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                labels[key_id] = {"frame": r["frame"], "track": r["track"], "label": value,
                                  "x1": x1, "y1": y1, "x2": x2, "y2": y2, "crop": f"number_labels/{crop_name}"}
                write_labels(args.name, labels)
                pos += 1
                break
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {len(labels)}/{len(samples)} labels to {label_path(args.name)}")


def cmd_score(args):
    labels = read_labels(args.name)
    if not labels:
        raise SystemExit("No labels yet. Run: python label_numbers.py label --name " + args.name)
    rows = load_named(args.name)
    ours = {(r["frame"], r["track"]): r["number"] for r in rows}
    by_box = {(r["frame"], int(float(r["x1"])), int(float(r["y1"]))): r["number"] for r in rows}
    for k, v in labels.items():
        if v.get("x1") not in (None, ""):       # same player box in a re-run, whatever its track id now is
            ours[k] = by_box.get((v["frame"], int(float(v["x1"])), int(float(v["y1"]))), ours.get(k, ""))
    visible = [(k, v["label"]) for k, v in labels.items() if v["label"] not in ("none", "bad")]
    hidden = [k for k, v in labels.items() if v["label"] == "none"]
    named = [(k, lab) for k, lab in visible if ours.get(k)]
    correct = sum(ours[k] == lab for k, lab in named)
    named_hidden = [k for k in hidden if ours.get(k)]
    correct_hidden_note = len(named_hidden)
    print(f"Labelled boxes: {len(labels)} | number visible: {len(visible)} | not visible: {len(hidden)}")
    print(f"Number visible -> we gave a number: {len(named)}/{len(visible)} ({100 * len(named) / max(1, len(visible)):.0f}%), "
          f"correct: {correct}/{len(named)} ({100 * correct / max(1, len(named)):.0f}%)")
    print(f"Number not visible -> we still gave a number (from voting/carrying/lineup): {correct_hidden_note}/{len(hidden)}")
    wrong = [(k, lab, ours[k]) for k, lab in named if ours[k] != lab]
    for (frame, track), lab, got in wrong:
        print(f"  wrong: frame {frame} track {track}: you saw {lab}, we said {got}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["label", "score"])
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    {"label": cmd_label, "score": cmd_score}[args.command](args)


if __name__ == "__main__":
    main()
