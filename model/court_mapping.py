"""
Map the video onto a real FIBA court (28 x 15 m), then measure player positions, speed and distance.

Run after jersey_numbers.py on the same run name:
  1. Calibrate (about 5-10 minutes, resumes where you stopped):
       python court_mapping.py calibrate --name aus_ind_v2
     The tool shows ~30 keyframes. Yellow lines = where it thinks the court lines are.
       - Lines sit on the real court lines  -> press Enter.
       - Lines are off -> click a point on the video, then the same point on the court diagram
         (it snaps to the nearest landmark: corners, paint corners, centre line, 3-point corners...).
         After 4+ pairs the lines are redrawn in pink; press Enter when they fit.
       u = undo last pair   c = clear pairs   s = skip (close-up / court not visible)   b = back   q = save + quit
  2. Apply (automatic):
       python court_mapping.py apply --name aus_ind_v2

Between keyframes the camera motion (optical flow) carries the mapping, blended from both neighbouring keyframes.
Each player's feet (bottom-centre of the box) are mapped to metres, smoothed over 0.6 s, and turned into speed
(km/h) and distance (m). People outside the court most of the time (bench, staff) are flagged off-court.
Frames that cannot be mapped get no position and no speed - nothing is guessed.

Outputs in runs/pipeline/<name>/: court_keyframes.json, court.json, positions.csv, court_check.mp4

Note: we first tried an automatic court-landmark model (koppolusameer/yolo11n-basketball-court-keypoints).
On this FIBA clip it found too few points and mixed up the left and right halves of the court (also on NBA
frames), so its court lines landed metres away from the real ones. Manual keyframes are reliable.
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import cv2
import numpy as np

from basketball_pipeline import camera_path, run_dir, sampled_frames

COURT_LENGTH, COURT_WIDTH = 28.0, 15.0
KEYFRAME_PAN = 0.35         # new keyframe after the camera pans this share of the frame width...
KEYFRAME_ZOOM = 0.15        # ...or zooms this much (log scale)...
KEYFRAME_SECONDS = 8.0      # ...or after this long
MAX_CARRY_SECONDS = 8.0     # a keyframe's mapping is carried at most this far in time
SMOOTH_SECONDS = 0.6        # position smoothing window for speed
MAX_SPEED_KMH = 32.0        # faster than any sprint: treated as a mapping/track error
OFF_COURT_MARGIN = 1.0      # metres outside the lines still counted as on court
OFF_COURT_SHARE = 0.6       # tracks outside the court this share of the time are bench/staff
AUTO_MODEL = os.environ.get("COURT_MODEL", "weights/court_fiba/best.pt")   # our court-landmark model (court_dataset.py + YOLO pose)
AUTO_POINT_CONF = 0.5       # keep landmarks the model is at least this sure about
AUTO_POOL_SECONDS = 1.0     # combine landmarks from frames this close in time
AUTO_MIN_INLIERS = 6        # landmarks that must agree for a frame's mapping
AUTO_RANSAC_METRES = 0.5    # a landmark within this distance of the fitted mapping agrees
AUTO_MIN_MAPPED = 0.6       # below this share of frames mapped, ask for the manual fallback
VIEW_WIDTH = 860           # size of the video in the calibration window
DIAGRAM_SCALE = 17          # pixels per metre in the calibration diagram
DIAGRAM_PAD = 24


def landmarks():
    """Court points that are easy to see, in metres (x along the length, y across; y=0 is the far sideline)."""
    half = [(0, 0), (0, 15), (0, 0.9), (0, 14.1), (2.99, 0.9), (2.99, 14.1), (0, 5.05), (0, 9.95),
            (5.8, 5.05), (5.8, 9.95), (5.8, 7.5), (7.6, 7.5), (8.325, 7.5), (1.575, 7.5)]
    pts = half + [(COURT_LENGTH - x, y) for x, y in half]
    return pts + [(14, 0), (14, 15), (14, 7.5), (14, 5.7), (14, 9.3), (12.2, 7.5), (15.8, 7.5)]


def court_lines():
    """Court markings as polylines in metres (FIBA)."""
    L, W = COURT_LENGTH, COURT_WIDTH
    lines = [[(0, 0), (L, 0), (L, W), (0, W), (0, 0)], [(L / 2, 0), (L / 2, W)]]

    def circle(cx, cy, r, a0=0, a1=360):
        return [(cx + r * np.cos(np.radians(a)), cy + r * np.sin(np.radians(a))) for a in np.linspace(a0, a1, 40)]

    lines.append(circle(L / 2, W / 2, 1.8))
    for side in (0, 1):
        fx = (lambda x: x) if side == 0 else (lambda x: L - x)
        lines.append([(fx(0), 5.05), (fx(5.8), 5.05), (fx(5.8), 9.95), (fx(0), 9.95)])
        lines.append([(fx(x), y) for x, y in circle(5.8, 7.5, 1.8, -90, 90)])
        arc_end = np.degrees(np.arcsin((7.5 - 0.9) / 6.75))
        arc = circle(1.575, 7.5, 6.75, -arc_end, arc_end)
        lines.append([(fx(0), 0.9), (fx(2.99), 0.9)] + [(fx(x), y) for x, y in arc] + [(fx(2.99), 14.1), (fx(0), 14.1)])
    return lines


def to_court(H, x, y):
    p = H @ np.array([x, y, 1.0])
    return p[0] / p[2], p[1] / p[2]


def load_tracks(out_dir):
    by_track, frames = defaultdict(list), set()
    named = os.path.join(out_dir, "tracks_named.csv")
    path = named if os.path.exists(named) else os.path.join(out_dir, "tracks.csv")
    if not os.path.exists(path):
        # A video prepared only for calibration (no player tracking): every processed frame
        meta = json.load(open(os.path.join(out_dir, "meta.json")))
        cap = cv2.VideoCapture(meta["video"])
        frames = {idx for idx, _ in sampled_frames(cap, meta["start"], meta["end"], meta["fps"])}
        cap.release()
        return by_track, frames
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            frames.add(int(r["frame"]))
            by_track[int(r["track"])].append({
                "frame": int(r["frame"]), "track": int(r["track"]), "team": int(r["team"]),
                "number": r.get("number", ""), "box": [float(r[k]) for k in ("x1", "y1", "x2", "y2")]})
    return by_track, frames


def load_camera(out_dir, meta, frames):
    path = os.path.join(out_dir, "camera.json")
    if os.path.exists(path):
        raw = json.load(open(path))
        return {int(k): np.array(v) for k, v in raw["to_shared"].items()}, {int(k): v for k, v in raw["shot"].items()}
    print("Measuring camera motion (one pass over the video)...")
    to_shared, shot = camera_path(meta["video"], meta, frames)
    json.dump({"to_shared": {k: v.tolist() for k, v in to_shared.items()}, "shot": shot}, open(path, "w"))
    return to_shared, shot


def move_matrix(to_shared, a, b):
    """Matrix taking pixels of frame b to pixels of frame a."""
    return np.linalg.inv(to_shared[a]) @ to_shared[b]


def wide_shots(video, to_shared, shot, size):
    """Camera shots that show the court from the wide TV camera: in 3 sampled frames, at least 5 players are found
    and they are small in the picture. Close-ups, crowd and bench shots fail this and are left out of mapping."""
    from basketball_pipeline import FINETUNED_MODEL, FinetunedDetector
    detector = FinetunedDetector(FINETUNED_MODEL)
    by_shot = defaultdict(list)
    for f in sorted(to_shared):
        by_shot[shot[f]].append(f)
    cap = cv2.VideoCapture(video)
    wide = set()
    for s, frames in by_shot.items():
        votes = 0
        for f in (frames[len(frames) // 6], frames[len(frames) // 2], frames[-1 - len(frames) // 6]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue
            people = [d for d in detector(img) if d["cls"] in ("player", "referee") and d["conf"] >= 0.5]
            heights = [d["xyxy"][3] - d["xyxy"][1] for d in people]
            votes += int(len(people) >= 5 and np.median(heights) < 0.3 * size[1])
        if votes >= 2:
            wide.add(s)
    cap.release()
    return wide


def wide_keyframes(video, keys, frames, shot, size, fps):
    """Check every chosen keyframe picture: if it is not a wide court view (a close-up the cut detector missed),
    use the nearest wide view within a few seconds in the same stretch, else drop it. Near-duplicates are merged."""
    from basketball_pipeline import FINETUNED_MODEL, FinetunedDetector
    detector = FinetunedDetector(FINETUNED_MODEL)
    cap = cv2.VideoCapture(video)

    def is_wide(f):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            return False
        if fading(f):
            return False
        people = [d for d in detector(img) if d["cls"] in ("player", "referee") and d["conf"] >= 0.5]
        return len(people) >= 5 and np.median([d["xyxy"][3] - d["xyxy"][1] for d in people]) < 0.3 * size[1]

    def small(f):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f))
        ok, img = cap.read()
        return cv2.cvtColor(cv2.resize(img, (480, 270)), cv2.COLOR_BGR2GRAY) if ok else None

    def leftover(a, b):
        """Mean difference after undoing the camera movement from picture a to picture b."""
        p = cv2.goodFeaturesToTrack(a, 300, 0.01, 8)
        if p is None or len(p) < 10:
            return 255.0
        q, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p, None)
        ok = st.ravel() == 1
        M = cv2.estimateAffinePartial2D(p[ok], q[ok], method=cv2.RANSAC, ransacReprojThreshold=2)[0] if ok.sum() >= 6 else None
        if M is None:
            return 255.0
        seen = cv2.warpAffine(np.ones_like(a), M, a.shape[::-1]) > 0
        return float(np.abs(cv2.warpAffine(a, M, a.shape[::-1]).astype(np.float32) - b)[seen].mean())

    def fading(f):
        """A dissolve between two scenes (replay transitions) is a blend of two pictures: even after undoing the camera
        movement it differs from BOTH neighbours 0.25 s away (a frame right after a hard cut differs from one side only)."""
        g = max(1, int(0.25 * fps))
        a, b, c = small(f - g), small(f), small(f + g)
        if a is None or c is None:
            return False
        return min(leftover(a, b), leftover(c, b)) > 20

    chosen = []
    for k in keys:
        same = [f for f in frames if shot[f] == shot[k]]
        # k itself, then frames 0.5 s, 1 s, ... up to 6 s away, alternating after/before
        steps = [0] + [s * d for s in (1, -1) for d in np.arange(0.5, 6.01, 0.5)]
        steps = sorted(steps, key=abs)
        for d in steps:
            target = k + d * fps
            f = min(same, key=lambda x: abs(x - target)) if same else None
            if f is not None and abs(f - target) <= fps and is_wide(f):
                if not chosen or abs(f - chosen[-1]) > 1.5 * fps:
                    chosen.append(f)
                break
    cap.release()
    return sorted(set(chosen))


def limited_keyframes(to_shared, shot, size, source_fps, max_keys, min_shot_seconds=2.0, allowed_shots=None):
    """At most max_keys keyframes: skip very short camera shots (replays, close-ups), then pick keyframes less often
    until the total fits. If there are still too many shots, keep the longest ones."""
    by_shot = defaultdict(list)
    for f in sorted(to_shared):
        if allowed_shots is None or shot[f] in allowed_shots:
            by_shot[shot[f]].append(f)
    long_shots = {k: v for k, v in by_shot.items() if (v[-1] - v[0]) / source_fps >= min_shot_seconds}
    if not long_shots:
        return []
    frames = [f for v in long_shots.values() for f in v]
    # use as many keyframes as allowed (more keyframes = more accurate), from frequent to sparse
    best = None
    for pan_limit in (0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.2, 9):
        for seconds in (8, 10, 12, 16, 20, 25, 40, 1e9):
            for zoom_limit in (0.15, 0.25, 0.35, 0.5, 0.7, 1.0):
                keys = pick_keyframes(to_shared, shot, size, source_fps, pan_limit, seconds, frames, zoom_limit)
                if len(keys) <= max_keys and (best is None or len(keys) > len(best)):
                    best = keys
    if best:
        return best
    longest = sorted(long_shots.values(), key=len, reverse=True)[:max_keys]
    return sorted(v[0] for v in longest)


def pick_keyframes(to_shared, shot, size, source_fps, pan_limit=None, seconds=None, frames=None, zoom_limit=None):
    pan_limit = KEYFRAME_PAN if pan_limit is None else pan_limit
    seconds = KEYFRAME_SECONDS if seconds is None else seconds
    zoom_limit = KEYFRAME_ZOOM if zoom_limit is None else zoom_limit
    frames = sorted(frames if frames is not None else to_shared)
    keys = [frames[0]]
    w, h = size
    for f in frames[1:]:
        M = move_matrix(to_shared, f, keys[-1])
        c = M @ np.array([w / 2, h / 2, 1.0])
        pan = np.hypot(c[0] / c[2] - w / 2, c[1] / c[2] - h / 2) / w
        zoom = abs(np.log(np.sqrt(abs(np.linalg.det(M[:2, :2])))))
        if shot[f] != shot[keys[-1]] or pan > pan_limit or zoom > zoom_limit \
                or (f - keys[-1]) / source_fps > seconds:
            keys.append(f)
    return keys


def diagram_image():
    w = int(COURT_LENGTH * DIAGRAM_SCALE) + 2 * DIAGRAM_PAD
    h = int(COURT_WIDTH * DIAGRAM_SCALE) + 2 * DIAGRAM_PAD
    img = np.full((h, w, 3), (60, 95, 150), np.uint8)
    for line in court_lines():
        pts = np.array([(DIAGRAM_PAD + x * DIAGRAM_SCALE, DIAGRAM_PAD + y * DIAGRAM_SCALE) for x, y in line], np.int32)
        cv2.polylines(img, [pts], False, (255, 255, 255), 2, cv2.LINE_AA)
    for x, y in landmarks():
        cv2.circle(img, (int(DIAGRAM_PAD + x * DIAGRAM_SCALE), int(DIAGRAM_PAD + y * DIAGRAM_SCALE)), 4, (0, 220, 255), -1)
    cv2.putText(img, "far sideline", (w // 2 - 50, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, "LEFT basket", (DIAGRAM_PAD + 4, h // 2 - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    cv2.putText(img, "RIGHT basket", (w - DIAGRAM_PAD - 100, h // 2 - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    cv2.putText(img, "near sideline (camera side)", (w // 2 - 110, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def draw_lines(view, H, scale, color):
    inv = np.linalg.inv(H)
    for line in court_lines():
        pts = [to_court(inv, x, y) for x, y in line]
        pts = np.array([(px * scale, py * scale) for px, py in pts], np.float32)
        if np.all(np.isfinite(pts)) and np.abs(pts).max() < 1e5:
            cv2.polylines(view, [pts.astype(np.int32)], False, color, 2, cv2.LINE_AA)


def cmd_prepare(args):
    """For court mapping in the website: measure the camera motion, pick a small set of keyframes (very short shots
    skipped, at most --max-keyframes) and save their pictures + camera data into --out (the upload's folder)."""
    out_dir = run_dir(args.name)
    cap = cv2.VideoCapture(args.video)
    fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    meta_path = os.path.join(out_dir, "meta.json")
    if not os.path.exists(meta_path):
        json.dump({"video": args.video, "start": args.start, "end": args.end or round(count / fps, 2), "fps": 15.0,
                   "detector": "finetuned", "source_fps": fps}, open(meta_path, "w"))
    meta = json.load(open(meta_path))
    _, frames = load_tracks(out_dir)
    to_shared, shot = load_camera(out_dir, meta, frames)
    wide = wide_shots(args.video, to_shared, shot, size)
    keys = limited_keyframes(to_shared, shot, size, fps, args.max_keyframes or 12, allowed_shots=wide)
    keys = wide_keyframes(args.video, keys, sorted(to_shared), shot, size, fps)
    images = os.path.join(args.out, "keyframes")
    os.makedirs(images, exist_ok=True)
    width = min(1280, size[0])
    scale = width / size[0]
    cap = cv2.VideoCapture(args.video)
    for k in keys:
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, frame = cap.read()
        if ok:
            frame = cv2.resize(frame, (width, int(round(size[1] * scale))))
            cv2.imwrite(os.path.join(images, f"{k}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    cap.release()
    points = [p for p in landmarks() if p not in ((1.575, 7.5), (COURT_LENGTH - 1.575, 7.5))]  # no points in the air
    json.dump({"size": list(size), "image_scale": scale, "fps": fps, "landmarks": points, "court": [COURT_LENGTH, COURT_WIDTH],
               "keyframes": [{"frame": int(k), "t": round(k / fps - meta["start"], 2), "shot": int(shot[k]),
                              "to_shared": np.asarray(to_shared[k]).tolist()} for k in keys]},
              open(os.path.join(args.out, "calibration.json"), "w"))
    print(f"Keyframes to map: {len(keys)} (wide court views in {len(wide)} of {len(set(shot.values()))} camera shots)")


def cmd_calibrate(args):
    out_dir = run_dir(args.name)
    meta_path = os.path.join(out_dir, "meta.json")
    if args.video and not os.path.exists(meta_path):
        # New video, calibration only: no need to run player detection first
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"Could not open video: {args.video}")
        fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        json.dump({"video": args.video, "start": args.start, "end": args.end or round(frames / fps, 2), "fps": 15.0,
                   "detector": None, "source_fps": fps}, open(meta_path, "w"))
    meta = json.load(open(meta_path))
    _, frames = load_tracks(out_dir)
    cap = cv2.VideoCapture(meta["video"])
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    to_shared, shot = load_camera(out_dir, meta, frames)
    keys = (limited_keyframes(to_shared, shot, size, meta["source_fps"], args.max_keyframes) if args.max_keyframes
            else pick_keyframes(to_shared, shot, size, meta["source_fps"]))
    if args.add_at:     # extra pictures at chosen seconds (e.g. a wide shot the automatic choice missed)
        processed = sorted(frames)
        for sec in args.add_at:
            target = (meta["start"] + sec) * meta["source_fps"]
            keys = sorted(set(keys) | {min(processed, key=lambda f: abs(f - target))})
    path = os.path.join(out_dir, "court_keyframes.json")
    saved = {int(k): v for k, v in json.load(open(path)).items()} if os.path.exists(path) else {}
    print(f"{len(keys)} keyframes; {len(saved)} already done.")

    scale = VIEW_WIDTH / size[0]
    base_diagram = diagram_image()
    view_h = int(size[1] * scale)
    window = "Court calibration | Enter=accept  click video then diagram=add pair  u=undo  c=clear  s=skip  b=back  q=quit"
    state = {"pairs": [], "pending": None}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if x < VIEW_WIDTH and y < view_h:
            state["pending"] = (x / scale, y / scale)
        elif x >= VIEW_WIDTH and state["pending"] is not None:
            cx = (x - VIEW_WIDTH - DIAGRAM_PAD) / DIAGRAM_SCALE
            cy = (y - DIAGRAM_PAD) / DIAGRAM_SCALE
            nearest = min(landmarks(), key=lambda p: np.hypot(p[0] - cx, p[1] - cy))
            state["pairs"].append((state["pending"], nearest))
            state["pending"] = None

    # Resizable window, shrunk to fit the screen (Windows display scaling can otherwise push it off-screen)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    canvas_w = VIEW_WIDTH + base_diagram.shape[1]
    canvas_h = max(view_h, base_diagram.shape[0] + 160)
    try:
        import ctypes
        screen_w, screen_h = ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
    except Exception:
        screen_w, screen_h = 1280, 720
    fit = min(1.0, 0.95 * screen_w / canvas_w, 0.8 * screen_h / canvas_h)
    cv2.resizeWindow(window, int(canvas_w * fit), int(canvas_h * fit))
    cv2.setMouseCallback(window, on_mouse)
    pos = 0
    while pos < len(keys):
        idx = keys[pos]
        if idx in saved:
            pos += 1
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            pos += 1
            continue
        # Prediction: the nearest earlier calibrated keyframe, carried here by the camera motion
        # nearest earlier calibrated frame in the same camera shot (also frames reused from another calibration)
        earlier = sorted(k for k, v in saved.items() if isinstance(v, list) and k < idx and k in shot and shot[k] == shot[idx])
        predicted = np.array(saved[earlier[-1]]) @ move_matrix(to_shared, earlier[-1], idx) if earlier else None
        state["pairs"], state["pending"] = [], None
        action = None
        while action is None:
            view = cv2.resize(frame, (VIEW_WIDTH, view_h))
            fitted = None
            if len(state["pairs"]) >= 4:
                fitted, _ = cv2.findHomography(np.float32([p for p, _ in state["pairs"]]),
                                               np.float32([c for _, c in state["pairs"]]))
            if fitted is not None:
                draw_lines(view, fitted, scale, (255, 0, 255))
            elif predicted is not None:
                draw_lines(view, predicted, scale, (0, 255, 255))
            diagram = base_diagram.copy()
            for n, ((px, py), (cx, cy)) in enumerate(state["pairs"], 1):
                cv2.circle(view, (int(px * scale), int(py * scale)), 6, (0, 0, 255), -1)
                cv2.putText(view, str(n), (int(px * scale) + 6, int(py * scale) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                dx, dy = int(DIAGRAM_PAD + cx * DIAGRAM_SCALE), int(DIAGRAM_PAD + cy * DIAGRAM_SCALE)
                cv2.circle(diagram, (dx, dy), 7, (0, 0, 255), -1)
                cv2.putText(diagram, str(n), (dx + 6, dy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if state["pending"] is not None:
                px, py = state["pending"]
                cv2.circle(view, (int(px * scale), int(py * scale)), 8, (0, 255, 0), 2)
            canvas = np.zeros((max(view_h, diagram.shape[0] + 160), VIEW_WIDTH + diagram.shape[1], 3), np.uint8)
            canvas[:view_h, :VIEW_WIDTH] = view
            canvas[:diagram.shape[0], VIEW_WIDTH:] = diagram
            done = sum(1 for k in keys if k in saved)
            help_lines = [f"Keyframe {done + 1}/{len(keys)}  ({idx / meta['source_fps']:.1f}s)",
                          "Basket on the right of the video = RIGHT basket",
                          "Yellow = predicted lines, pink = from your pairs",
                          "Enter: accept   click video, then diagram: add pair",
                          "u: undo   c: clear   s: skip   b: back   q: quit",
                          "Pick points far apart (e.g. 2 paint corners + centre line ends)"]
            for n, text in enumerate(help_lines):
                cv2.putText(canvas, text, (VIEW_WIDTH + 10, diagram.shape[0] + 20 + 19 * n),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key == 13 and (fitted is not None or predicted is not None):
                action = ("save", (fitted if fitted is not None else predicted).tolist())
            elif key == ord("u") and state["pairs"]:
                state["pairs"].pop()
            elif key == ord("c"):
                state["pairs"], state["pending"] = [], None
            elif key == ord("s"):
                action = ("save", "skip")
            elif key == ord("b"):
                action = ("back", None)
            elif key == ord("q"):
                action = ("quit", None)
        kind, value = action
        if kind == "save":
            saved[idx] = value
            json.dump({str(k): v for k, v in saved.items()}, open(path, "w"))
            pos += 1
        elif kind == "back":
            previous = [p for p in range(pos - 1, -1, -1) if keys[p] in saved]
            if previous:
                pos = previous[0]
                saved.pop(keys[pos])
                json.dump({str(k): v for k, v in saved.items()}, open(path, "w"))
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {len(saved)}/{len(keys)} keyframes to {path}")


def recover_court_views(video, keyframes, shot, size, fps, carry):
    """Register returning broadcast views to a clicked court using features on its floor."""
    anchors = {k: np.asarray(v) for k, v in keyframes.items() if isinstance(v, list)}
    cap = cv2.VideoCapture(video)
    sift = cv2.SIFT_create(nfeatures=1400)
    matcher = cv2.BFMatcher()
    scale = min(1., 960 / size[0])
    def features(k):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, frame = cap.read()
        if not ok:
            return np.empty((0, 2)), None
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale)
        pts, desc = sift.detectAndCompute(gray, None)
        return np.float32([p.pt for p in pts]).reshape(-1, 2) / scale, desc

    refs = []
    for k, H in anchors.items():
        xy, desc = features(k)
        if desc is None:
            continue
        court = cv2.perspectiveTransform(xy[:, None], H.astype(float))[:, 0]
        inside = (court > [0, 0]).all(axis=1) & (court < [COURT_LENGTH, COURT_WIDTH]).all(axis=1)
        if inside.sum() >= 16:
            refs.append((court[inside], desc[inside]))
    recovered = dict(keyframes)
    last = {}
    skips = sorted(k for k,v in keyframes.items() if v == "skip")
    for k in sorted(shot):
        if (not refs or k in keyframes or k-last.get(shot[k], -1e12) < fps*4
                or any(shot[a] == shot[k] and abs(k-a) <= fps*carry
                       and not any(min(a,k) < b < max(a,k) for b in skips) for a in anchors)):
            continue
        last[shot[k]] = k
        xy, desc = features(k)
        if desc is None:
            continue
        best = 15
        for court, ref in refs:
            pairs = matcher.knnMatch(desc, ref, k=2)
            matches = [a for a,b in pairs if a.distance < .7*b.distance]
            if len(matches) <= best:
                continue
            src = np.float32([xy[m.queryIdx] for m in matches])
            dst = np.float32([court[m.trainIdx] for m in matches])
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, .18)
            if H is None or mask.sum() <= best or mask.mean() < .65:
                continue
            support = dst[mask.ravel().astype(bool)]
            if cv2.contourArea(cv2.convexHull(support)) < 12:
                continue
            best = int(mask.sum())
            recovered[k] = H.tolist()
    cap.release()
    return recovered


def frame_homographies(keyframes, to_shared, shot, size, source_fps, carry_seconds=MAX_CARRY_SECONDS):
    """Per frame: carry the previous and next calibrated keyframes here and blend them by time."""
    keys = sorted(k for k, v in keyframes.items() if isinstance(v, list))
    skips = sorted(k for k, v in keyframes.items() if v == "skip")
    w, h = size
    corners = [(0, 0), (w, 0), (w, h), (0, h)]
    out = {}
    for idx in sorted(to_shared):
        # A skipped keyframe (close-up) blocks carrying across it
        prev = [k for k in keys if k <= idx and shot[k] == shot[idx] and not any(k < s <= idx for s in skips)]
        nxt = [k for k in keys if k >= idx and shot[k] == shot[idx] and not any(idx <= s < k for s in skips)]
        cands = []
        for k in ([prev[-1]] if prev else []) + ([nxt[0]] if nxt else []):
            gap = abs(idx - k) / source_fps
            if gap <= carry_seconds:
                cands.append((np.array(keyframes[k]) @ move_matrix(to_shared, k, idx), gap))
        if not cands:
            continue
        if len(cands) == 1 or cands[0][1] + cands[1][1] == 0:
            out[idx] = cands[0][0]
            continue
        (Ha, ga), (Hb, gb) = cands
        wa = gb / (ga + gb)
        court = [tuple(wa * np.array(to_court(Ha, x, y)) + (1 - wa) * np.array(to_court(Hb, x, y))) for x, y in corners]
        out[idx] = cv2.getPerspectiveTransform(np.float32(corners), np.float32(court))
    return out


def smooth(values, half):
    return [sum(values[max(0, i - half): i + half + 1]) / len(values[max(0, i - half): i + half + 1])
            for i in range(len(values))]


def player_positions(by_track, homographies, source_fps, fps, shots=None):
    """Feet positions in metres, smoothed, with speed (km/h) and cumulative distance (m) per track."""
    half = max(1, int(SMOOTH_SECONDS * fps / 2))
    rows = []
    for track, items in by_track.items():
        items = [it for it in sorted(items, key=lambda it: it["frame"]) if it["frame"] in homographies]
        if not items:
            continue
        for it in items:
            x1, y1, x2, y2 = it["box"]
            it["x"], it["y"] = to_court(homographies[it["frame"]], (x1 + x2) / 2, y2)
        # A camera cut or missing projection breaks the movement measurement.
        segments = []
        for it in items:
            previous = segments[-1][-1]["frame"] if segments else None
            if (previous is None or it["frame"] - previous > 1.5 * source_fps / fps
                    or (shots and shots.get(previous) != shots.get(it["frame"]))):
                segments.append([])
            segments[-1].append(it)
        distance = 0.0
        for segment in segments:
            xs, ys = smooth([it["x"] for it in segment], half), smooth([it["y"] for it in segment], half)
            raw_speed = []
            for i, it in enumerate(segment):
                it["x"], it["y"] = xs[i], ys[i]
                speed = None
                if i:
                    dt = (it["frame"] - segment[i - 1]["frame"]) / source_fps
                    step = np.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
                    if 0 < dt <= 0.5 and step / dt * 3.6 <= MAX_SPEED_KMH:
                        speed = step / dt * 3.6
                        distance += step
                raw_speed.append(speed)
                it["distance"] = distance
            for i, it in enumerate(segment):
                near = [s for s in raw_speed[max(0, i - half): i + half + 1] if s is not None]
                it["speed"] = sum(near) / len(near) if near and raw_speed[i] is not None else None
        outside = [not (-OFF_COURT_MARGIN <= it["x"] <= COURT_LENGTH + OFF_COURT_MARGIN and
                        -OFF_COURT_MARGIN <= it["y"] <= COURT_WIDTH + OFF_COURT_MARGIN) for it in items]
        off_court = sum(outside) / len(outside) >= OFF_COURT_SHARE
        for it in items:
            it["off_court"] = off_court
            rows.append(it)
    return rows


def render_check(meta, homographies, rows_by_frame, out_dir, size):
    """Court lines drawn back onto the video: if they sit on the real lines, the mapping is right."""
    lines = court_lines()
    cap = cv2.VideoCapture(meta["video"])
    step = max(1, round(meta["source_fps"] / meta["fps"]))
    path = os.path.join(out_dir, "court_check.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), meta["source_fps"] / step, size)
    scale, pad = 14, 12
    for idx, frame in sampled_frames(cap, meta["start"], meta["end"], meta["fps"]):
        H = homographies.get(idx)
        if H is not None:
            draw_lines(frame, H, 1.0, (0, 255, 255))
        mw, mh = int(COURT_LENGTH * scale) + 2 * pad, int(COURT_WIDTH * scale) + 2 * pad
        x0, y0 = size[0] - mw - 20, 20
        cv2.rectangle(frame, (x0, y0), (x0 + mw, y0 + mh), (40, 40, 40), -1)
        for line in lines:
            pts = np.array([(x0 + pad + x * scale, y0 + pad + y * scale) for x, y in line], np.int32)
            cv2.polylines(frame, [pts], False, (200, 200, 200), 1, cv2.LINE_AA)
        for r in rows_by_frame.get(idx, []):
            color = (0, 0, 255) if r["off_court"] else (0, 220, 0)
            cv2.circle(frame, (int(x0 + pad + r["x"] * scale), int(y0 + pad + r["y"] * scale)), 4, color, -1)
        status = "court mapped" if H is not None else "court NOT mapped"
        cv2.putText(frame, status, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255) if H is not None else (0, 0, 255), 2)
        writer.write(frame)
    writer.release()
    cap.release()
    return path


def detect_court_points(meta, frames, out_dir):
    """Run our FIBA court-landmark model on every processed frame. Cached in court_auto_points.json."""
    path = os.path.join(out_dir, "court_auto_points.json")
    if os.path.exists(path):
        return {int(k): v for k, v in json.load(open(path)).items()}
    from ultralytics import YOLO
    from court_dataset import COURT_XY
    model = YOLO(AUTO_MODEL)
    points = {}
    cap = cv2.VideoCapture(meta["video"])
    for idx, frame in sampled_frames(cap, meta["start"], meta["end"], meta["fps"]):
        if idx not in frames:
            continue
        r = model(frame, imgsz=960, verbose=False, conf=0.25)[0]
        found = []
        if r.keypoints is not None and len(r.keypoints.conf):
            best = int(r.boxes.conf.argmax())
            xy, conf = r.keypoints.xy[best].cpu().numpy(), r.keypoints.conf[best].cpu().numpy()
            found = [[i, float(xy[i][0]), float(xy[i][1]), float(conf[i])] for i in range(len(COURT_XY)) if conf[i] >= AUTO_POINT_CONF]
        points[idx] = found
    cap.release()
    json.dump({str(k): v for k, v in points.items()}, open(path, "w"))
    return points


def auto_homographies(points, to_shared, shot, size, source_fps):
    """Per frame: court points from this frame and frames within +-AUTO_POOL_SECONDS (moved here with the camera
    motion), fitted robustly. Frames with too few agreeing points borrow the nearest good frame's mapping."""
    from court_dataset import COURT_XY
    frames = sorted(points)
    window = AUTO_POOL_SECONDS * source_fps
    solved = {}
    w, h = size
    for idx in frames:
        img, crt = [], []
        for k in frames:
            if abs(k - idx) > window or shot.get(k) != shot.get(idx):
                continue
            move = move_matrix(to_shared, idx, k)
            for i, x, y, _ in points[k]:
                p = move @ np.array([x, y, 1.0])
                img.append((p[0] / p[2], p[1] / p[2]))
                crt.append(COURT_XY[i])
        if len(img) < AUTO_MIN_INLIERS:
            continue
        H, mask = cv2.findHomography(np.float32(img), np.float32(crt), cv2.RANSAC, AUTO_RANSAC_METRES)
        if H is None or mask.sum() < AUTO_MIN_INLIERS:
            continue
        cx, cy = to_court(H, w / 2, h * 0.85)
        if -6 < cx < COURT_LENGTH + 6 and -4 < cy < COURT_WIDTH + 6:
            solved[idx] = H
    out = dict(solved)
    good = sorted(solved)
    for idx in frames:
        if idx in out or not good:
            continue
        k = min(good, key=lambda g: abs(g - idx))
        if abs(k - idx) / source_fps <= MAX_CARRY_SECONDS and shot.get(k) == shot.get(idx):
            out[idx] = solved[k] @ move_matrix(to_shared, k, idx)
    return out, len(solved)


def mapping_error(auto, truth, size, frames):
    """Metres between where the automatic and the reference mapping put the same visible court points."""
    w, h = size
    grid = [(x, y) for x in np.arange(0, COURT_LENGTH + 0.1, 2) for y in np.arange(0, COURT_WIDTH + 0.1, 1.5)]
    errs = []
    for idx in frames:
        if idx not in auto or idx not in truth:
            continue
        inv = np.linalg.inv(truth[idx])
        for x, y in grid:
            px, py = to_court(inv, x, y)
            if 0 <= px < w and 0.15 * h <= py < h:
                ax, ay = to_court(auto[idx], px, py)
                errs.append(np.hypot(ax - x, ay - y))
    return (float(np.median(errs)), float(np.percentile(errs, 90))) if errs else (None, None)


def cmd_auto(args):
    """Automatic court mapping with our court model; checks it against clicked keyframes when they exist."""
    out_dir = run_dir(args.name)
    meta = json.load(open(os.path.join(out_dir, "meta.json")))
    _, frames = load_tracks(out_dir)
    cap = cv2.VideoCapture(meta["video"])
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    to_shared, shot = load_camera(out_dir, meta, frames)
    points = detect_court_points(meta, frames, out_dir)
    homographies, n_solved = auto_homographies(points, to_shared, shot, size, meta["source_fps"])
    json.dump({"court_metres": [COURT_LENGTH, COURT_WIDTH], "source": "auto",
               "homography_image_to_court": {str(k): v.tolist() for k, v in homographies.items()}},
              open(os.path.join(out_dir, "court_auto.json"), "w"))
    mapped = len(homographies) / max(1, len(frames))
    print(f"Court points per frame (median): {np.median([len(v) for v in points.values()]):.0f} | "
          f"frames fitted directly: {100 * n_solved / max(1, len(frames)):.0f}% | frames mapped: {100 * mapped:.0f}%")
    kf_path = os.path.join(out_dir, "court_keyframes.json")
    if os.path.exists(kf_path):
        keyframes = {int(k): v for k, v in json.load(open(kf_path)).items()}
        truth = frame_homographies(keyframes, to_shared, shot, size, meta["source_fps"])
        for label, sel in (("all frames", frames), (f"held-out (from {args.valid_from:.0f} s)",
                           [f for f in frames if f / meta["source_fps"] - meta["start"] >= args.valid_from])):
            med, p90 = mapping_error(homographies, truth, size, sel)
            if med is not None:
                print(f"Error vs clicked calibration, {label}: median {med:.2f} m, 90th percentile {p90:.2f} m")
    # Mapping many frames says nothing about whether they are right: only the error check above shows that
    if mapped < AUTO_MIN_MAPPED:
        print("Too few frames mapped: calibrate this video by hand (python court_mapping.py calibrate).")
    print("Check court_auto.json against clicked keyframes (error above) before using --source auto.")


def cmd_apply(args):
    out_dir = run_dir(args.name)
    meta = json.load(open(os.path.join(out_dir, "meta.json")))
    by_track, frames = load_tracks(out_dir)
    cap = cv2.VideoCapture(meta["video"])
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    keyframes = {}
    shot = None
    if args.source == "auto":
        auto = os.path.join(out_dir, "court_auto.json")
        if not os.path.exists(auto):
            raise SystemExit("No automatic mapping yet. Run: python court_mapping.py auto --name " + args.name)
        homographies = {int(k): np.array(v) for k, v in json.load(open(auto))["homography_image_to_court"].items()}
    else:
        path = os.path.join(out_dir, "court_keyframes.json")
        if not os.path.exists(path):
            raise SystemExit("No calibration yet. Run: python court_mapping.py calibrate --name " + args.name)
        keyframes = {int(k): v for k, v in json.load(open(path)).items()}
        to_shared, shot = load_camera(out_dir, meta, frames)
        keyframes = recover_court_views(meta["video"], keyframes, shot, size, meta["source_fps"], args.carry)
        homographies = frame_homographies(keyframes, to_shared, shot, size, meta["source_fps"], args.carry)
    json.dump({"court_metres": [COURT_LENGTH, COURT_WIDTH],
               "homography_image_to_court": {str(k): v.tolist() for k, v in homographies.items()}},
              open(os.path.join(out_dir, "court.json"), "w"))

    rows = player_positions(by_track, homographies, meta["source_fps"], meta["fps"], shot)
    with open(os.path.join(out_dir, "positions.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "seconds", "track", "team", "number", "x_m", "y_m", "speed_kmh", "distance_m", "off_court"])
        for r in sorted(rows, key=lambda r: (r["frame"], r["track"])):
            writer.writerow([r["frame"], round(r["frame"] / meta["source_fps"], 3), r["track"], r["team"], r["number"],
                             round(r["x"], 2), round(r["y"], 2), "" if r["speed"] is None else round(r["speed"], 1),
                             round(r["distance"], 1), int(r["off_court"])])
    rows_by_frame = defaultdict(list)
    for r in rows:
        rows_by_frame[r["frame"]].append(r)
    video = render_check(meta, homographies, rows_by_frame, out_dir, size)

    team_rows = [r for r in rows if r["team"] in (0, 1) and not r["off_court"]]
    speeds = [r["speed"] for r in team_rows if r["speed"] is not None]
    calibrated = sum(1 for v in keyframes.values() if isinstance(v, list))
    print(f"Keyframes: {calibrated} calibrated, {sum(1 for v in keyframes.values() if v == 'skip')} skipped | "
          f"frames mapped: {100 * len(homographies) / max(1, len(frames)):.0f}%")
    print(f"Player speeds (on-court): median {np.median(speeds) if speeds else 0:.1f} km/h, "
          f"95th percentile {np.percentile(speeds, 95) if speeds else 0:.1f} km/h")
    print(f"Tracks flagged off-court (bench/staff): {len({r['track'] for r in rows if r['off_court']})}")
    print(f"Saved {video}, court.json and positions.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["auto", "calibrate", "apply", "prepare"])
    parser.add_argument("--out", help="prepare: folder for the keyframe pictures and calibration.json")
    parser.add_argument("--name", required=True)
    parser.add_argument("--source", choices=["keyframes", "auto"], default="keyframes",
                        help="apply: use clicked keyframes or the automatic court model")
    parser.add_argument("--valid-from", type=float, default=85.0, help="auto: held-out part of the clip for the accuracy check")
    parser.add_argument("--video", help="calibrate: a new video to calibrate (no player detection needed)")
    parser.add_argument("--start", type=float, default=0, help="calibrate --video: first second to use")
    parser.add_argument("--end", type=float, default=None, help="calibrate --video: last second to use")
    parser.add_argument("--add-at", type=float, nargs="*", default=[], help="calibrate: also show pictures at these seconds")
    parser.add_argument("--max-keyframes", type=int, default=0, help="calibrate: at most this many keyframes (0 = no limit)")
    parser.add_argument("--carry", type=float, default=MAX_CARRY_SECONDS,
                        help="apply: seconds a keyframe's mapping is carried by the camera motion")
    args = parser.parse_args()
    {"auto": cmd_auto, "calibrate": cmd_calibrate, "apply": cmd_apply, "prepare": cmd_prepare}[args.command](args)


if __name__ == "__main__":
    main()
