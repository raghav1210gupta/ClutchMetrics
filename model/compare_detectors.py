"""
Compare our YOLO detectors with the Roboflow RF-DETR basketball detector on one video.

  1. Run all models on evenly spaced frames (RF-DETR runs only if ROBOFLOW_API_KEY is set):
       python compare_detectors.py run --video path/to/game.mp4
  2. Hand-label the ball on those frames:
       python compare_detectors.py label --video path/to/game.mp4
  3. Score each model against the labels:
       python compare_detectors.py score --video path/to/game.mp4

RF-DETR uses Roboflow's hosted API, so its sampled frames are uploaded to Roboflow.
"""

import argparse
import base64
import csv
import json
import os
import time

import cv2
import numpy as np
import requests
from ultralytics import YOLO

YOLO_MODELS = {
    "broadcast_large": "weights/best_broadcast_large_yolo.pt",
    "deepsport_arena": "weights/deepsport_arena_best.pt",
}
RFDETR_MODEL = "basketball-player-detection-3-ycjdo/4"
RFDETR_URL = "https://serverless.roboflow.com/" + RFDETR_MODEL
MIN_CONF = 0.05
RETRIES = 5
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]
COLORS = {"player": (0, 215, 255), "ball": (0, 100, 255), "referee": (200, 200, 200)}


def normalize_class(name):
    """Map each model's class names onto player / ball / referee. Other classes are ignored."""
    name = name.lower()
    if name == "ball":
        return "ball"
    if name == "referee":
        return "referee"
    if name.startswith("player") or name.startswith("team_"):
        return "player"
    return None


def output_dir(args):
    stem = os.path.splitext(os.path.basename(args.video))[0]
    path = os.path.join(args.out, stem)
    os.makedirs(path, exist_ok=True)
    return path


def sample_frames(video, count):
    """Yield (frame_index, frame) for evenly spaced frames across the video."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Skip the first and last 2% to avoid intros and end cards
    indices = np.linspace(total * 0.02, total * 0.98, count).astype(int)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            yield int(idx), frame
    cap.release()


def run_yolo(model, frame, imgsz):
    res = model.predict(frame, imgsz=imgsz, conf=MIN_CONF, verbose=False)[0]
    dets = []
    for box in res.boxes:
        cls = normalize_class(model.names[int(box.cls[0])])
        if cls:
            dets.append({"cls": cls, "conf": float(box.conf[0]), "xyxy": [float(v) for v in box.xyxy[0]]})
    return dets


def run_rfdetr(frame, api_key):
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("Could not encode frame")
    body = base64.b64encode(jpg.tobytes())
    resp = None
    for attempt in range(RETRIES):
        try:
            resp = requests.post(
                RFDETR_URL,
                params={"api_key": api_key, "confidence": MIN_CONF},
                data=body,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=60,
            )
            if resp.status_code < 500 and resp.status_code != 429:
                break
            reason = f"HTTP {resp.status_code}"
        except requests.RequestException as e:
            # Keep only the error type, because the message contains the API key
            reason = type(e).__name__
        print(f"  RF-DETR request failed ({reason}), retry {attempt + 1}/{RETRIES}")
        time.sleep(2 ** attempt)
    else:
        raise SystemExit(f"RF-DETR request failed after {RETRIES} attempts: {reason}")
    if resp.status_code != 200:
        # Show the server's message, never the request URL
        raise SystemExit(f"RF-DETR returned HTTP {resp.status_code}: {resp.text[:300]}")
    dets = []
    for p in resp.json().get("predictions", []):
        cls = normalize_class(p["class"])
        if cls:
            x, y, w, h = p["x"], p["y"], p["width"], p["height"]
            dets.append({"cls": cls, "conf": float(p["confidence"]),
                         "xyxy": [x - w / 2, y - h / 2, x + w / 2, y + h / 2]})
    return dets


def draw(frame, dets, title, conf):
    out = frame.copy()
    for d in dets:
        if d["conf"] < conf:
            continue
        x1, y1, x2, y2 = map(int, d["xyxy"])
        color = COLORS[d["cls"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3 if d["cls"] == "ball" else 2)
        if d["cls"] == "ball":
            cv2.circle(out, ((x1 + x2) // 2, (y1 + y2) // 2), 22, color, 2)
    cv2.rectangle(out, (0, 0), (out.shape[1], 44), (20, 20, 20), -1)
    cv2.putText(out, title, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return out


def cmd_run(args):
    out = output_dir(args)
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    models = {name: YOLO(path) for name, path in YOLO_MODELS.items()}
    names = list(models) + (["rf_detr"] if api_key else [])
    if not api_key:
        print("ROBOFLOW_API_KEY is not set: skipping RF-DETR.")

    results = {name: {} for name in names}
    times = {name: [] for name in names}
    for n, (idx, frame) in enumerate(sample_frames(args.video, args.frames), 1):
        panels = []
        for name in names:
            start = time.time()
            if name == "rf_detr":
                dets = run_rfdetr(frame, api_key)
            else:
                dets = run_yolo(models[name], frame, args.imgsz)
            times[name].append(time.time() - start)
            results[name][str(idx)] = dets
            panels.append(draw(frame, dets, f"{name} | frame {idx} | conf>={args.view_conf}", args.view_conf))
        # Spread previews across the video, not just the opening frames
        if args.previews and (n - 1) % max(1, args.frames // args.previews) == 0:
            preview = cv2.resize(np.vstack(panels), None, fx=0.6, fy=0.6)
            cv2.imwrite(os.path.join(out, f"compare_{idx:07d}.jpg"), preview)
        if n % 25 == 0:
            print(f"  {n}/{args.frames} frames")

    with open(os.path.join(out, "detections.json"), "w") as f:
        json.dump(results, f)

    print(f"\nNo-label summary at conf>={args.view_conf} (bench and crowd players can inflate counts)")
    print(f"{'model':<18}{'median players':>15}{'frames with ball':>18}{'ms/frame':>10}")
    for name in names:
        by_frame = results[name].values()
        players = [sum(d["cls"] == "player" and d["conf"] >= args.view_conf for d in fr) for fr in by_frame]
        balls = [any(d["cls"] == "ball" and d["conf"] >= args.view_conf for d in fr) for fr in by_frame]
        # The first call includes model loading, so leave it out
        ms = 1000 * float(np.mean(times[name][1:] or times[name]))
        print(f"{name:<18}{np.median(players):>15.0f}{100 * np.mean(balls):>17.0f}%{ms:>10.0f}")
    print(f"\nSaved to {out}")


def read_labels(path):
    labels = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                labels[row["frame"]] = row
    return labels


def write_labels(path, labels):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "status", "x", "y"])
        writer.writeheader()
        writer.writerows(labels.values())


def cmd_label(args):
    out = output_dir(args)
    path = os.path.join(out, "ball_labels.csv")
    labels = read_labels(path)
    frames = list(sample_frames(args.video, args.frames))
    window = "Click ball | n=not visible | x=replay/close-up | b=back | q=save+quit"
    clicked = []
    scale = 1280 / frames[0][1].shape[1]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x / scale, y / scale))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    pos = 0
    while pos < len(frames):
        idx, frame = frames[pos]
        if str(idx) in labels and not args.relabel:
            pos += 1
            continue
        view = cv2.resize(frame, None, fx=scale, fy=scale)
        cv2.putText(view, f"{pos + 1}/{len(frames)}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow(window, view)
        clicked.clear()
        while True:
            key = cv2.waitKey(30) & 0xFF
            if clicked:
                x, y = clicked[0]
                labels[str(idx)] = {"frame": idx, "status": "ball", "x": round(x, 1), "y": round(y, 1)}
                pos += 1
                break
            if key == ord("n"):
                labels[str(idx)] = {"frame": idx, "status": "no_ball", "x": "", "y": ""}
                pos += 1
                break
            if key == ord("x"):
                labels[str(idx)] = {"frame": idx, "status": "exclude", "x": "", "y": ""}
                pos += 1
                break
            if key == ord("b"):
                pos = max(0, pos - 1)
                labels.pop(str(frames[pos][0]), None)
                break
            if key == ord("q"):
                pos = len(frames)
                break
        write_labels(path, labels)
    cv2.destroyAllWindows()
    print(f"Saved {len(labels)} labels to {path}")


def cmd_score(args):
    out = output_dir(args)
    labels = read_labels(os.path.join(out, "ball_labels.csv"))
    if not labels:
        raise SystemExit("No labels found. Run the label step first.")
    with open(os.path.join(out, "detections.json")) as f:
        results = json.load(f)

    used = {k: v for k, v in labels.items() if v["status"] != "exclude"}
    with_ball = sum(v["status"] == "ball" for v in used.values())
    print(f"Scoring {len(used)} frames, {with_ball} with a visible ball. Match radius: {args.radius}px.")
    print(f"{'model':<18}{'conf':>6}{'recall':>9}{'precision':>11}{'F1':>7}{'false/frame':>13}")

    for name, frames in results.items():
        best = None
        for conf in THRESHOLDS:
            tp = fp = 0
            scored = 0
            for frame_id, label in used.items():
                if frame_id not in frames:
                    continue
                scored += 1
                balls = [d for d in frames[frame_id] if d["cls"] == "ball" and d["conf"] >= conf]
                hit = False
                for d in balls:
                    x1, y1, x2, y2 = d["xyxy"]
                    if (label["status"] == "ball" and not hit and
                            np.hypot((x1 + x2) / 2 - float(label["x"]), (y1 + y2) / 2 - float(label["y"])) <= args.radius):
                        hit = True
                    else:
                        fp += 1
                tp += hit
            recall = tp / max(1, with_ball)
            precision = tp / max(1, tp + fp)
            f1 = 2 * precision * recall / max(1e-6, precision + recall)
            row = (f1, conf, recall, precision, fp / max(1, scored))
            print(f"{name:<18}{conf:>6.1f}{100 * recall:>8.1f}%{100 * precision:>10.1f}%{f1:>7.2f}{row[4]:>13.2f}")
            if best is None or row[0] > best[0]:
                best = row
        print(f"{'':<18}best F1 {best[0]:.2f} at conf {best[1]:.1f}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "label", "score"])
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames", type=int, default=250, help="Evenly spaced frames to test")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO input size")
    parser.add_argument("--view-conf", type=float, default=0.3, help="Confidence for previews and the summary")
    parser.add_argument("--previews", type=int, default=20, help="Side-by-side images to save")
    parser.add_argument("--radius", type=float, default=25, help="Ball match distance in source pixels")
    parser.add_argument("--relabel", action="store_true", help="Label again, replacing saved labels")
    parser.add_argument("--out", default="runs/detector_comparison")
    args = parser.parse_args()
    {"run": cmd_run, "label": cmd_label, "score": cmd_score}[args.command](args)


if __name__ == "__main__":
    main()
