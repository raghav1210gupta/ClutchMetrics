"""
Build a court-landmark training set from a run that was calibrated with court_mapping.py.

    python court_dataset.py --names aus_ind_v2 --out datasets/court_kp

Every mapped frame already has an exact image->court mapping (clicked keyframes + camera motion), so the
position of every court landmark in that frame is known. We save each frame with those positions as
YOLO-pose labels. Each landmark has a fixed identity (left half / centre / right half of a FIBA court);
horizontal flips during training swap left and right correctly through `flip_idx`, which is what the
public court model got wrong.

Frames from the last part of each clip (from --valid-from seconds) are kept for validation only.
"""

import argparse
import json
import os
import shutil

import cv2
import numpy as np

from basketball_pipeline import run_dir, sampled_frames

COURT_LENGTH, COURT_WIDTH = 28.0, 15.0
_HALF = [  # left half (x from the left endline, y from the far sideline), metres, FIBA
    ("corner far", 0, 0), ("3pt corner far", 0, 0.9), ("paint base far", 0, 5.05), ("paint base near", 0, 9.95),
    ("3pt corner near", 0, 14.1), ("corner near", 0, 15), ("3pt straight end far", 2.99, 0.9),
    ("3pt straight end near", 2.99, 14.1), ("free-throw far", 5.8, 5.05), ("free-throw near", 5.8, 9.95),
    ("free-throw middle", 5.8, 7.5), ("free-throw circle top", 7.6, 7.5), ("3pt arc top", 8.325, 7.5)]
_CENTRE = [("centre far", 14, 0), ("centre near", 14, 15), ("centre spot", 14, 7.5), ("centre circle far", 14, 5.7),
           ("centre circle near", 14, 9.3), ("centre circle left", 12.2, 7.5), ("centre circle right", 15.8, 7.5)]
KEYPOINTS = ([(f"left {n}", x, y) for n, x, y in _HALF] + _CENTRE
             + [(f"right {n}", COURT_LENGTH - x, y) for n, x, y in _HALF])
COURT_XY = np.array([(x, y) for _, x, y in KEYPOINTS], float)


def flip_index():
    """For each landmark, the landmark it becomes when the image is mirrored left-right (court x -> 28 - x)."""
    out = []
    for x, y in COURT_XY:
        d = np.hypot(COURT_XY[:, 0] - (COURT_LENGTH - x), COURT_XY[:, 1] - y)
        out.append(int(np.argmin(d)))
    return out


def project(H_img_to_court, size):
    """Image positions of all landmarks and whether each is inside the picture."""
    inv = np.linalg.inv(np.array(H_img_to_court))
    pts = np.c_[COURT_XY, np.ones(len(COURT_XY))] @ inv.T
    xy = pts[:, :2] / pts[:, 2:3]
    w, h = size
    inside = (pts[:, 2] > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    return xy, inside


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--names", nargs="+", required=True, help="Calibrated run names")
    parser.add_argument("--out", default="datasets/court_kp")
    parser.add_argument("--every", type=int, default=2, help="Use every Nth mapped frame")
    parser.add_argument("--valid-from", type=float, default=85.0, help="Seconds into the clip where validation starts")
    parser.add_argument("--width", type=int, default=1280, help="Saved image width")
    args = parser.parse_args()

    shutil.rmtree(args.out, ignore_errors=True)
    for split in ("train", "valid"):
        os.makedirs(os.path.join(args.out, split, "images"))
        os.makedirs(os.path.join(args.out, split, "labels"))
    counts = {"train": 0, "valid": 0}
    seen = np.zeros(len(KEYPOINTS), int)
    for name in args.names:
        out_dir = run_dir(name)
        meta = json.load(open(os.path.join(out_dir, "meta.json")))
        H = {int(k): v for k, v in json.load(open(os.path.join(out_dir, "court.json")))["homography_image_to_court"].items()}
        keep = set(sorted(H)[::args.every])
        cap = cv2.VideoCapture(meta["video"])
        for idx, frame in sampled_frames(cap, meta["start"], meta["end"], meta["fps"]):
            if idx not in keep:
                continue
            h, w = frame.shape[:2]
            xy, inside = project(H[idx], (w, h))
            if inside.sum() < 4:
                continue
            split = "valid" if idx / meta["source_fps"] - meta["start"] >= args.valid_from else "train"
            scale = args.width / w
            img = cv2.resize(frame, (args.width, int(round(h * scale))))
            ih, iw = img.shape[:2]
            vis = xy[inside] * scale
            x1, y1 = vis.min(0)
            x2, y2 = vis.max(0)
            box = [(x1 + x2) / 2 / iw, (y1 + y2) / 2 / ih, max(x2 - x1, 8) / iw, max(y2 - y1, 8) / ih]
            kp = []
            for (px, py), ok in zip(xy * scale, inside):
                kp += [px / iw, py / ih, 2] if ok else [0, 0, 0]
            stem = f"{name}_{idx:06d}"
            cv2.imwrite(os.path.join(args.out, split, "images", stem + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(os.path.join(args.out, split, "labels", stem + ".txt"), "w") as f:
                f.write("0 " + " ".join(f"{v:.6f}" for v in box + kp) + "\n")
            counts[split] += 1
            seen += inside
        cap.release()

    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(args.out)}\ntrain: train/images\nval: valid/images\n"
                f"kpt_shape: [{len(KEYPOINTS)}, 3]\nflip_idx: {flip_index()}\nnames:\n  0: court\n")
    json.dump({"keypoints": [{"name": n, "x_m": x, "y_m": y} for n, x, y in KEYPOINTS]},
              open(os.path.join(args.out, "keypoints.json"), "w"), indent=1)
    print(f"Images: train {counts['train']}, valid {counts['valid']} | landmark visible in "
          f"{', '.join(f'{n}:{c}' for (n, _, _), c in zip(KEYPOINTS, seen) if c)}")
    print(f"Never visible: {[n for (n, _, _), c in zip(KEYPOINTS, seen) if not c]}")


if __name__ == "__main__":
    main()
