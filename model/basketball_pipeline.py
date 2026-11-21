"""
Day 1 pipeline: detection, anonymous player tracking and team assignment.

  1. Save detections (safe to stop and run again, it continues where it stopped):
       python basketball_pipeline.py detect --video runs/aus_ind_14m_to_16m_clip.mp4 --name aus_ind --start 10 --end 120
  2. Track players, assign teams and render the annotated video:
       python basketball_pipeline.py track --video runs/aus_ind_14m_to_16m_clip.mp4 --name aus_ind
  3. Link the ball across frames and render the ball video:
       python basketball_pipeline.py ball --video runs/aus_ind_14m_to_16m_clip.mp4 --name aus_ind

Results are saved in runs/pipeline/<name>/. The default detector is our fine-tuned RF-DETR,
which runs locally. The hosted Roboflow model (--detector rf_detr) needs ROBOFLOW_API_KEY.
"""

import argparse
import csv
import json
import os
from collections import Counter

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from compare_detectors import YOLO_MODELS, normalize_class, run_rfdetr, run_yolo

FINETUNED_MODEL = "weights/rfdetr_basketball_v2/best"
MIN_CONF = 0.05

HIGH_CONF = 0.5
LOW_CONF = 0.15
MATCH_IOU = 0.2
BOX_EXPANSION = 0.3
MAX_LOST = 15
MIN_TRACK_FRAMES = 8
CUT_THRESHOLD = 0.45
BALL_DRAW_CONF = 0.3
BALL_MIN_CONF = 0.15
BALL_CONF_FLOOR = 0.25
BALL_LINK_WINDOW = 15
BALL_MAX_STEP = 70
BALL_RESTART_COST = 0.6
BALL_MAX_FILL = 10
BALL_TRAIL = 8
UNASSIGNED = -1
REFEREE = 2
REFEREE_COLOR = (200, 200, 200)
REFEREE_SHARE = 0.5
MIN_CONFIDENT_VOTES = 3
COLOR_OVERLAP_IOU = 0.15
COLOR_GATE = 38.0
UNASSIGNED_COLOR = (120, 120, 120)


def run_dir(name):
    path = os.path.join("runs", "pipeline", name)
    os.makedirs(path, exist_ok=True)
    return path


def sampled_frames(cap, start, end, fps):
    """Yield (frame_index, frame) at about `fps` between the start and end seconds."""
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, round(source_fps / fps))
    first = int(start * source_fps)
    last = int(end * source_fps) if end else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    for idx in range(first, last):
        if not cap.grab():
            break
        if (idx - first) % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                yield idx, frame


def cmd_detect(args):
    path = os.path.join(run_dir(args.name), "detections.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {json.loads(line)["frame"] for line in f}
        print(f"Continuing: {len(done)} frames already saved.")

    api_key = None
    model = None
    if args.detector == "rf_detr":
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            raise SystemExit("ROBOFLOW_API_KEY is not set in this PowerShell window.")
    elif args.detector == "finetuned":
        model = FinetunedDetector(FINETUNED_MODEL)
    else:
        from ultralytics import YOLO
        model = YOLO(YOLO_MODELS[args.detector])

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    with open(os.path.join(run_dir(args.name), "meta.json"), "w") as f:
        json.dump({"video": args.video, "start": args.start, "end": args.end, "fps": args.fps,
                   "detector": args.detector, "source_fps": source_fps}, f)

    total = int(((args.end or cap.get(cv2.CAP_PROP_FRAME_COUNT) / source_fps) - args.start) * args.fps)
    with open(path, "a") as out:
        for n, (idx, frame) in enumerate(sampled_frames(cap, args.start, args.end, args.fps), 1):
            if idx not in done:
                if args.detector == "rf_detr":
                    dets = run_rfdetr(frame, api_key)
                elif args.detector == "finetuned":
                    dets = model(frame)
                else:
                    dets = run_yolo(model, frame, args.imgsz)
                out.write(json.dumps({"frame": idx, "dets": dets}) + "\n")
                out.flush()
            if n % 50 == 0:
                print(f"  {n}/{total} frames")
    cap.release()
    print(f"Saved detections to {path}")


class FinetunedDetector:
    """Our RF-DETR Nano model, fine-tuned at 576px. Runs locally, so it needs no API key."""

    def __init__(self, path):
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(path)
        self.model = AutoModelForObjectDetection.from_pretrained(path).to(self.device).eval()

    def __call__(self, frame):
        from PIL import Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        with self.torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
        res = self.processor.post_process_object_detection(outputs, threshold=MIN_CONF, target_sizes=[image.size[::-1]])[0]
        dets = []
        for score, label, box in zip(res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist()):
            cls = normalize_class(self.model.config.id2label[label])
            if cls:
                dets.append({"cls": cls, "conf": score, "xyxy": box})
        return dets


def expanded_iou(a, b):
    """IoU after enlarging both boxes, which keeps fast players matched between sampled frames."""
    a = expand(a)
    b = expand(b)
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def box_iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def expand(box):
    dx = (box[2] - box[0]) * BOX_EXPANSION / 2
    dy = (box[3] - box[1]) * BOX_EXPANSION / 2
    return [box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy]


def estimate_camera_motion(prev_gray, gray):
    """Return a 2x3 matrix that moves points from the previous frame to this one."""
    points = cv2.goodFeaturesToTrack(prev_gray, 400, 0.01, 10)
    if points is None:
        return None
    moved, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, None)
    good = status.flatten() == 1
    if good.sum() < 10:
        return None
    # RANSAC ignores the moving players and keeps the camera's motion
    matrix, _ = cv2.estimateAffinePartial2D(points[good], moved[good], method=cv2.RANSAC)
    return matrix


def warp_box(box, matrix):
    corners = np.array([[box[0], box[1], 1.0], [box[2], box[3], 1.0]])
    (x1, y1), (x2, y2) = (matrix @ corners.T).T
    return [x1, y1, x2, y2]


class PlayerTracker:
    """Short-term tracker. It ends a track when unsure, so IDs here are anonymous and short-lived."""

    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def reset(self):
        self.tracks = {}

    def update(self, dets, camera):
        for track in self.tracks.values():
            if camera is not None:
                track["box"] = warp_box(track["box"], camera)
            track["box"] = [v + d for v, d in zip(track["box"], track["velocity"])]

        high = [d for d in dets if d["conf"] >= HIGH_CONF]
        low = [d for d in dets if LOW_CONF <= d["conf"] < HIGH_CONF]
        results = []
        unmatched = self._match(list(self.tracks), high, results)
        self._match(unmatched, low, results)

        matched = {tid for tid, _ in results}
        for tid in list(self.tracks):
            if tid not in matched:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > MAX_LOST:
                    del self.tracks[tid]

        used = {id(d) for _, d in results}
        for d in high:
            if id(d) not in used:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"box": list(d["xyxy"]), "velocity": [0.0] * 4, "lost": 0,
                                    "color": None if d.get("crowded") else d.get("color")}
                results.append((tid, d))
        return results

    def _match(self, track_ids, dets, results):
        if not track_ids or not dets:
            return track_ids
        cost = np.ones((len(track_ids), len(dets)))
        for i, tid in enumerate(track_ids):
            for j, d in enumerate(dets):
                cost[i, j] = 1.0 - expanded_iou(self.tracks[tid]["box"], d["xyxy"])
                if color_distance(self.tracks[tid]["color"], d.get("color")) > COLOR_GATE:
                    cost[i, j] = 1.0
        matched = set()
        for i, j in zip(*linear_sum_assignment(cost)):
            if cost[i, j] <= 1.0 - MATCH_IOU:
                tid = track_ids[i]
                track = self.tracks[tid]
                new_box = list(dets[j]["xyxy"])
                track["velocity"] = [0.7 * v + 0.3 * (n - o) for v, n, o in zip(track["velocity"], new_box, track["box"])]
                track["box"] = new_box
                track["lost"] = 0
                color = dets[j].get("color")
                if color is not None and not dets[j].get("crowded"):
                    track["color"] = color if track["color"] is None else 0.8 * track["color"] + 0.2 * color
                results.append((tid, dets[j]))
                matched.add(tid)
        return [t for t in track_ids if t not in matched]


def color_distance(a, b):
    """Lab colour distance with brightness down-weighted, since lighting changes across the court."""
    if a is None or b is None:
        return 0.0
    return float(np.sqrt(0.25 * (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2))


def is_cut(prev_hist, hist):
    return prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA) > CUT_THRESHOLD


def frame_histogram(frame):
    small = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([small], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def jersey_color(frame, box):
    """Median Lab colour of the chest area, or None if the crop is too small."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    crop = frame[max(0, int(y1 + 0.2 * h)):max(0, int(y1 + 0.5 * h)),
                 max(0, int(x1 + 0.25 * w)):max(0, int(x2 - 0.25 * w))]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    return np.median(lab, axis=0).astype(np.float32)


def referee_or_player(votes):
    """Decide from confident detections only: low-confidence 'player' guesses on a referee would outvote
    the confident 'referee' ones. Falls back to all detections when there are too few confident ones."""
    counts = votes["confident"] if sum(votes["confident"].values()) >= MIN_CONFIDENT_VOTES else votes["all"]
    return "referee" if counts["referee"] >= REFEREE_SHARE * max(1, sum(counts.values())) else "player"


def assign_teams(colors_by_track, roles):
    """Group player tracks into two teams by jersey colour. Returns track->team and BGR colours."""
    players = [t for t, colors in colors_by_track.items()
               if len(colors) >= MIN_TRACK_FRAMES and roles.get(t) != "referee"]
    if len(players) < 2:
        return {}, [(0, 215, 255), (255, 120, 0)]
    samples = np.array([np.median(colors_by_track[t], axis=0) for t in players], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, _, centers = cv2.kmeans(samples, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    # Team A is always the lighter jersey, so letters stay the same between runs
    centers = centers[np.argsort(-centers[:, 0])]
    teams = {}
    for t in players:
        colors = np.array(colors_by_track[t], dtype=np.float32)
        nearest = np.argmin(np.linalg.norm(colors[:, None, :] - centers[None, :, :], axis=2), axis=1)
        teams[t] = int(np.bincount(nearest, minlength=2).argmax())
    # Draw each team in its own jersey colour
    swatches = cv2.cvtColor(np.uint8(centers).reshape(2, 1, 3), cv2.COLOR_LAB2BGR).reshape(2, 3)
    return teams, [tuple(int(v) for v in s) for s in swatches]


def cmd_track(args):
    out_dir = run_dir(args.name)
    with open(os.path.join(out_dir, "meta.json")) as f:
        meta = json.load(f)
    detections = {}
    with open(os.path.join(out_dir, "detections.jsonl")) as f:
        for line in f:
            row = json.loads(line)
            detections[row["frame"]] = row["dets"]

    cap = cv2.VideoCapture(args.video)
    tracker = PlayerTracker()
    rows = []
    prev_gray = prev_hist = None
    colors_by_track, crowded_colors, class_votes = {}, {}, {}
    cuts = 0

    # First pass: track everyone and collect jersey colours
    for idx, frame in sampled_frames(cap, meta["start"], meta["end"], meta["fps"]):
        if idx not in detections:
            continue
        gray = cv2.cvtColor(cv2.resize(frame, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY)
        hist = frame_histogram(frame)
        camera = None
        if is_cut(prev_hist, hist):
            tracker.reset()
            cuts += 1
        elif prev_gray is not None:
            camera = estimate_camera_motion(prev_gray, gray)
            if camera is not None:
                camera = camera.copy()
                camera[:, 2] *= 2  # motion was measured at half resolution
        prev_gray, prev_hist = gray, hist

        people = [dict(d) for d in detections[idx] if d["cls"] in ("player", "referee") and d["conf"] >= LOW_CONF]
        for d in people:
            d["color"] = jersey_color(frame, d["xyxy"])
            # Overlapping players mix two jerseys in one crop
            d["crowded"] = any(box_iou(d["xyxy"], o["xyxy"]) > COLOR_OVERLAP_IOU for o in people if o is not d)
        for tid, det in tracker.update(people, camera):
            rows.append({"frame": idx, "track": tid, "box": det["xyxy"], "conf": det["conf"]})
            votes = class_votes.setdefault(tid, {"all": Counter(), "confident": Counter()})
            votes["all"][det["cls"]] += 1
            if det["conf"] >= HIGH_CONF:
                votes["confident"][det["cls"]] += 1
            if det["color"] is not None and det["cls"] == "player":
                (crowded_colors if det["crowded"] else colors_by_track).setdefault(tid, []).append(det["color"])

    # Fall back to crowded samples for tracks that were rarely seen on their own
    for tid, colors in crowded_colors.items():
        if len(colors_by_track.get(tid, [])) < MIN_TRACK_FRAMES:
            colors_by_track[tid] = colors_by_track.get(tid, []) + colors
    roles = {t: referee_or_player(votes) for t, votes in class_votes.items()}
    teams, team_colors = assign_teams(colors_by_track, roles)
    with open(os.path.join(out_dir, "team_colors.json"), "w") as f:
        json.dump(team_colors, f)

    def team_of(track):
        if roles.get(track) == "referee":
            return REFEREE
        return teams.get(track, UNASSIGNED)

    with open(os.path.join(out_dir, "tracks.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "seconds", "track", "team", "x1", "y1", "x2", "y2", "conf"])
        for r in rows:
            writer.writerow([r["frame"], round(r["frame"] / meta["source_fps"], 3), r["track"], team_of(r["track"]),
                             *[round(v, 1) for v in r["box"]], round(r["conf"], 3)])

    # Second pass: render the annotated video
    by_frame = {}
    for r in rows:
        by_frame.setdefault(r["frame"], []).append(r)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_step = max(1, round(meta["source_fps"] / meta["fps"]))
    video_path = os.path.join(out_dir, "annotated.mp4")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), meta["source_fps"] / source_step, (width, height))
    for idx, frame in sampled_frames(cap, meta["start"], meta["end"], meta["fps"]):
        if idx not in detections:
            continue
        for r in by_frame.get(idx, []):
            team = team_of(r["track"])
            if team == UNASSIGNED and len(colors_by_track.get(r["track"], [])) < MIN_TRACK_FRAMES:
                continue
            color = REFEREE_COLOR if team == REFEREE else UNASSIGNED_COLOR if team == UNASSIGNED else team_colors[team]
            x1, y1, x2, y2 = map(int, r["box"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            label = "REF" if team == REFEREE else f"{'AB'[team]}{r['track']}" if team in (0, 1) else f"?{r['track']}"
            # Dark label background keeps pale jersey colours readable on the floor
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_y = max(text_h + 8, y1)
            cv2.rectangle(frame, (x1, label_y - text_h - 8), (x1 + text_w + 8, label_y), (20, 20, 20), -1)
            cv2.putText(frame, label, (x1 + 4, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        balls = [d for d in detections[idx] if d["cls"] == "ball" and d["conf"] >= BALL_DRAW_CONF]
        if balls:
            x1, y1, x2, y2 = max(balls, key=lambda d: d["conf"])["xyxy"]
            cv2.circle(frame, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 14, (0, 100, 255), 3)
        writer.write(frame)
    writer.release()
    cap.release()

    per_frame = [sum(team_of(r["track"]) in (0, 1) for r in v) for v in by_frame.values()]
    lengths = Counter(r["track"] for r in rows)
    print(f"Frames: {len(detections)} | camera cuts: {cuts} | tracks: {len(lengths)} | median track length: {np.median(list(lengths.values())):.0f} frames")
    print(f"Team A tracks: {sum(v == 0 for v in teams.values())} | Team B tracks: {sum(v == 1 for v in teams.values())} | referee tracks: {sum(v == 'referee' for v in roles.values())}")
    print(f"Team players per frame: median {np.median(per_frame):.0f}, range {min(per_frame)}-{max(per_frame)}")
    print(f"Saved {video_path} and tracks.csv")


def camera_path(video, meta, frames):
    """For each frame, return a matrix to shared coordinates that cancel camera motion, and a shot number."""
    cap = cv2.VideoCapture(video)
    to_shared, shot = {}, {}
    current = np.eye(3)
    shot_id = 0
    prev_gray = prev_hist = None
    for idx, frame in sampled_frames(cap, meta["start"], meta["end"], meta["fps"]):
        if idx not in frames:
            continue
        gray = cv2.cvtColor(cv2.resize(frame, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY)
        hist = frame_histogram(frame)
        if is_cut(prev_hist, hist):
            shot_id += 1
        elif prev_gray is not None:
            motion = estimate_camera_motion(prev_gray, gray)
            if motion is not None:
                step = np.vstack([motion, [0, 0, 1]])
                step[:2, 2] *= 2  # motion was measured at half resolution
                current = current @ np.linalg.inv(step)
        prev_gray, prev_hist = gray, hist
        to_shared[idx] = current.copy()
        shot[idx] = shot_id
    cap.release()
    return to_shared, shot


def project(matrix, x, y):
    px, py, pw = matrix @ np.array([x, y, 1.0])
    return px / pw, py / pw


def printed_ball_pair(frame, box):
    """Look for the second matching basketball printed alongside this one on a light panel."""
    x1, y1, x2, y2 = map(round, box)
    w, h = x2 - x1, y2 - y1
    height, width = frame.shape[:2]
    if min(w, h) < 5 or x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        return False
    patch = frame[y1:y2, x1:x2]
    if patch.std(axis=(0, 1)).mean() < 9:
        return False
    for left, right in ((x1 - 4*w, x1), (x2, x2 + 4*w)):
        left, right = max(0, left), min(width, right)
        top, bottom = max(0, y1 - h//2), min(height, y2 + h//2)
        region = frame[top:bottom, left:right]
        if right-left < w or bottom-top < h:
            continue
        score = cv2.matchTemplate(region, patch, cv2.TM_CCOEFF_NORMED)
        _, similarity, _, (dx, dy) = cv2.minMaxLoc(score)
        if similarity < 0.68:
            continue
        other = left + dx
        gap = frame[min(y1+h//2, top+dy+h//2), min(x2, other+w):max(x1, other)]
        if gap.size and np.median(gap.min(axis=1)) > 125:
            return True
    return False


def link_ball_path(frames, detections, to_shared, shot, max_step):
    """Choose the ball detections that form the best physically possible path through the clip."""
    candidates = []
    for idx in frames:
        row = []
        for d in detections[idx]:
            x1, y1, x2, y2 = d["xyxy"]
            if d["cls"] == "ball" and d["conf"] >= BALL_MIN_CONF and max(x2 - x1, y2 - y1) <= max_step * 2:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                sx, sy = project(to_shared[idx], cx, cy)
                row.append({"frame": idx, "x": cx, "y": cy, "sx": sx, "sy": sy, "conf": d["conf"], "score": 0.0, "prev": None})
        candidates.append(row)

    best_so_far = None
    best_until = []
    for i, row in enumerate(candidates):
        for c in row:
            reward = c["conf"] - BALL_CONF_FLOOR
            c["score"], c["prev"] = reward, None
            for j in range(i - 1, max(i - BALL_LINK_WINDOW, 0) - 1, -1):
                gap = i - j
                for p in candidates[j]:
                    if shot[p["frame"]] != shot[c["frame"]]:
                        continue
                    dist = float(np.hypot(c["sx"] - p["sx"], c["sy"] - p["sy"]))
                    if dist > max_step * gap:
                        continue
                    score = p["score"] + reward - 0.3 * dist / (max_step * gap) - 0.01 * (gap - 1)
                    if score > c["score"]:
                        c["score"], c["prev"] = score, p
            # Allow the path to restart elsewhere, at a cost, when the ball was lost
            if i > 0 and best_until[i - 1] is not None:
                score = best_until[i - 1]["score"] + reward - BALL_RESTART_COST
                if score > c["score"]:
                    c["score"], c["prev"] = score, best_until[i - 1]
        for c in row:
            if best_so_far is None or c["score"] > best_so_far["score"]:
                best_so_far = c
        best_until.append(best_so_far)

    path = []
    node = best_so_far
    while node is not None:
        path.append(node)
        node = node["prev"]
    return path[::-1]


def fill_ball_gaps(frames, path, to_shared, shot, max_step):
    """Label each frame detected, filled (short physically possible gap) or unknown."""
    order = {idx: i for i, idx in enumerate(frames)}
    states = {idx: {"status": "unknown", "x": None, "y": None, "conf": 0.0, "segment": None} for idx in frames}
    restarts = segment = 0
    if path:
        states[path[0]["frame"]] = {"status": "detected", "x": path[0]["x"], "y": path[0]["y"],
                                    "conf": path[0]["conf"], "segment": segment}
    for a, b in zip(path, path[1:]):
        gap = order[b["frame"]] - order[a["frame"]]
        dist = float(np.hypot(b["sx"] - a["sx"], b["sy"] - a["sy"]))
        if shot[a["frame"]] != shot[b["frame"]] or dist > max_step * gap:
            restarts += 1
            segment += 1
        elif gap > BALL_MAX_FILL:
            segment += 1
        else:
            for k in range(1, gap):
                t = k / gap
                idx = frames[order[a["frame"]] + k]
                sx, sy = a["sx"] + t * (b["sx"] - a["sx"]), a["sy"] + t * (b["sy"] - a["sy"])
                x, y = project(np.linalg.inv(to_shared[idx]), sx, sy)
                states[idx] = {"status": "filled", "x": x, "y": y, "conf": 0.0, "segment": segment}
        states[b["frame"]] = {"status": "detected", "x": b["x"], "y": b["y"], "conf": b["conf"], "segment": segment}
    return states, restarts


def cmd_ball(args):
    out_dir = run_dir(args.name)
    with open(os.path.join(out_dir, "meta.json")) as f:
        meta = json.load(f)
    detections = {}
    with open(os.path.join(out_dir, "detections.jsonl")) as f:
        for line in f:
            row = json.loads(line)
            detections[row["frame"]] = row["dets"]
    with open(os.path.join(out_dir, "team_colors.json")) as f:
        team_colors = [tuple(c) for c in json.load(f)]
    by_frame = {}
    with open(os.path.join(out_dir, "tracks.csv"), newline="") as f:
        for r in csv.DictReader(f):
            by_frame.setdefault(int(r["frame"]), []).append(r)

    frames = sorted(detections)
    cap = cv2.VideoCapture(args.video)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_step = max(1, round(meta["source_fps"] / meta["fps"]))
    output_fps = meta["source_fps"] / source_step
    # A hard pass moves about 70 pixels per frame in 1080p video at 15 fps
    max_step = BALL_MAX_STEP * (width / 1920) * (15 / output_fps)

    to_shared, shot = camera_path(args.video, meta, set(frames))
    for idx, frame in sampled_frames(cap, meta["start"], meta["end"], meta["fps"]):
        if idx in detections:
            detections[idx] = [d for d in detections[idx] if d["cls"] != "ball"
                               or not printed_ball_pair(frame, d["xyxy"])]
    cap.release()
    cap = cv2.VideoCapture(args.video)
    path = link_ball_path(frames, detections, to_shared, shot, max_step)
    states, restarts = fill_ball_gaps(frames, path, to_shared, shot, max_step)

    with open(os.path.join(out_dir, "ball.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "seconds", "status", "x", "y", "conf"])
        for idx in frames:
            s = states[idx]
            writer.writerow([idx, round(idx / meta["source_fps"], 3), s["status"],
                             "" if s["x"] is None else round(s["x"], 1), "" if s["y"] is None else round(s["y"], 1),
                             round(s["conf"], 3)])

    video_path = os.path.join(out_dir, "annotated_ball.mp4")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
    trail, trail_segment = [], None
    for idx, frame in sampled_frames(cap, meta["start"], meta["end"], meta["fps"]):
        if idx not in detections:
            continue
        for r in by_frame.get(idx, []):
            team = int(r["team"])
            if team == UNASSIGNED:
                continue
            color = REFEREE_COLOR if team == REFEREE else team_colors[team]
            x1, y1, x2, y2 = (int(float(r[k])) for k in ("x1", "y1", "x2", "y2"))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            label = "REF" if team == REFEREE else f"Team {'AB'[team]}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_y = max(text_h + 8, y1)
            cv2.rectangle(frame, (x1, label_y - text_h - 8), (x1 + text_w + 8, label_y), (20, 20, 20), -1)
            cv2.putText(frame, label, (x1 + 4, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        s = states[idx]
        if s["x"] is None or s["segment"] != trail_segment:
            trail.clear()
            trail_segment = s["segment"]
        if s["x"] is not None:
            trail.append(project(to_shared[idx], s["x"], s["y"]))
            trail[:] = trail[-BALL_TRAIL:]
            inverse = np.linalg.inv(to_shared[idx])
            points = [tuple(int(v) for v in project(inverse, *p)) for p in trail]
            for p, q in zip(points, points[1:]):
                cv2.line(frame, p, q, (0, 140, 255), 3, cv2.LINE_AA)
            # Orange ring: detected by the model. Yellow ring: estimated between detections
            ring = (0, 100, 255) if s["status"] == "detected" else (0, 255, 255)
            cv2.circle(frame, (int(s["x"]), int(s["y"])), 18, ring, 3, cv2.LINE_AA)

        status = {"detected": "BALL: detected", "filled": "BALL: estimated", "unknown": "BALL: unknown"}[s["status"]]
        cv2.rectangle(frame, (20, 20), (300, 62), (20, 20, 20), -1)
        cv2.putText(frame, status, (32, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    cap.release()

    counts = Counter(states[i]["status"] for i in frames)
    raw = np.mean([any(d["cls"] == "ball" and d["conf"] >= BALL_DRAW_CONF for d in detections[i]) for i in frames])
    longest = current = 0
    for i in frames:
        current = current + 1 if states[i]["status"] == "unknown" else 0
        longest = max(longest, current)
    total = len(frames)
    print(f"Raw detections at conf>={BALL_DRAW_CONF}: {100 * raw:.0f}% of frames")
    print(f"Ball path: detected {100 * counts['detected'] / total:.0f}% | estimated {100 * counts['filled'] / total:.0f}% | unknown {100 * counts['unknown'] / total:.0f}%")
    print(f"Path restarts: {restarts} | longest unknown stretch: {longest / output_fps:.1f}s")
    print(f"Saved {video_path} and ball.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["detect", "track", "ball"])
    parser.add_argument("--video", required=True)
    parser.add_argument("--name", required=True, help="Name of this run's results folder")
    parser.add_argument("--start", type=float, default=0, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds")
    parser.add_argument("--fps", type=float, default=15, help="Frames per second to process")
    parser.add_argument("--detector", default="finetuned", choices=["finetuned", "rf_detr", *YOLO_MODELS])
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO input size")
    args = parser.parse_args()
    {"detect": cmd_detect, "track": cmd_track, "ball": cmd_ball}[args.command](args)


if __name__ == "__main__":
    main()
