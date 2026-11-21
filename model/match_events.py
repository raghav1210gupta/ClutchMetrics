"""
Turn tracks, jersey numbers, court positions and the ball path into match events and statistics.

Run after court_mapping.py apply on the same run name:
    python match_events.py --name aus_ind_v2 --teams AUS IND

What is measured (all from the video, nothing typed in):
  Ball holder   - the on-court player whose body box contains the ball, closest to it; must hold for 0.2 s
  Ball control  - share of time each team had the last confirmed holder
  Pass          - ball goes from one player to a different player of the same team within 3 s
  Interception  - ball goes from a player of one team to a player of the other team within 3 s, with no shot
                  in between (includes steals; rebounds after a shot are not counted)
  Shot attempt  - ball in flight (no holder) reaches the area just above a rim found by the 5-class model;
                  shooter = last holder. Made = ball then seen just below the rim, inside its width, within 1.5 s.
  Speed / distance per player come from positions.csv (court mapping).
Every event stores its frame and players. Players without a confirmed number are shown as "AUS ?".

Outputs in runs/pipeline/<name>/: rims.json, possession.csv, events.json, report.json
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import cv2
import numpy as np

from basketball_pipeline import run_dir, sampled_frames

FIVE_CLASS = "weights/rfdetr_5class/checkpoint_best_total.pth"
RIM_CLASS, RIM_CONF = 4, 0.4
REFEREE = 2                 # team value of referees in tracks.csv
REF_HOLD_FRAMES = 5         # a referee holding the ball this many frames in a row (0.33 s) = dead ball; shorter = ball flying past
HOLD_FRAMES = 3            # processed frames (0.2 s at 15 fps) the same player must hold the ball
SOLID_HOLD_SECONDS = 0.4    # shooter / intercepting player must have held the ball at least this long
PASS_MAX_SECONDS = 3.0
MIN_PASS_METRES = 2.0       # passer and receiver closer than this = the holder flickered between neighbours
SAME_PLAYER_METRES = 1.5    # two track pieces this close at hand-over are the same player, not a pass
SHOT_ZONE_WIDTHS = 1.6      # ball within this many rim-widths sideways of the rim...
SHOT_ZONE_ABOVE = 4.0       # ...and up to this many rim-widths above it counts as reaching the rim
COURT_LENGTH, COURT_WIDTH = 28.0, 15.0       # FIBA court (Art. 2)
BASKET_FROM_ENDLINE = 1.575                  # point on the floor under the basket centre
THREE_POINT_RADIUS = 6.75                    # 3-point arc radius
CORNER_FROM_SIDELINE = 0.90                  # parallel 3-point lines
CORNER_STRAIGHT_LENGTH = 2.99                # where the arc meets the parallel lines (from the endline)
NEAR_LINE_METRES = 0.5                       # court positions are accurate to about this: closer = flagged
MADE_SECONDS = 1.5
SHOT_COOLDOWN_SECONDS = 2.0
RIM_CARRY_SECONDS = 2.0     # a rim missed by the detector is carried from a nearby frame by the camera motion
INTERCEPT_KEEP_SECONDS = 2.0  # the other team must keep the ball this long for an interception to count
SCORE_LAG_SECONDS = 8.0     # the TV scoreboard updates a few seconds after the basket
READER_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
SCORE_PROMPT = ("This is a basketball TV scoreboard. Reply with only the two team scores in the order shown, "
                "like 15-7. Reply none if there is no scoreboard.")


def load_rows(out_dir):
    tracks = defaultdict(dict)                  # frame -> track -> info
    with open(os.path.join(out_dir, "tracks_named.csv"), newline="") as f:
        for r in csv.DictReader(f):
            tracks[int(r["frame"])][int(r["track"])] = {
                "team": int(r["team"]), "number": r["number"], "box": [float(r[k]) for k in ("x1", "y1", "x2", "y2")]}
    # Court positions exist only when the court was mapped; without them, speed, distance and shot values are left out
    positions = os.path.join(out_dir, "positions.csv")
    if os.path.exists(positions):
        with open(positions, newline="") as f:
            for r in csv.DictReader(f):
                info = tracks[int(r["frame"])].get(int(r["track"]))
                if info is not None:
                    info.update(x=float(r["x_m"]), y=float(r["y_m"]), off_court=r["off_court"] == "1",
                                speed=float(r["speed_kmh"]) if r["speed_kmh"] else None, distance=float(r["distance_m"]))
    ball = {}
    with open(os.path.join(out_dir, "ball.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r["status"] in ("detected", "filled"):
                ball[int(r["frame"])] = (float(r["x"]), float(r["y"]), r["status"])
    return tracks, ball


def find_rims(meta, frames, out_dir):
    path = os.path.join(out_dir, "rims.json")
    if os.path.exists(path):
        return {int(k): v for k, v in json.load(open(path)).items()}
    from PIL import Image
    from rfdetr import RFDETRNano
    model = RFDETRNano(pretrain_weights=FIVE_CLASS, num_classes=5)
    rims = {}
    cap = cv2.VideoCapture(meta["video"])
    for n, (idx, frame) in enumerate(sampled_frames(cap, meta["start"], meta["end"], meta["fps"])):
        if idx not in frames:
            continue
        d = model.predict(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), threshold=RIM_CONF)
        rims[idx] = [[float(v) for v in b] for b, c in zip(d.xyxy, d.class_id) if c == RIM_CLASS]
        if n % 300 == 0:
            print(f"  rims: frame {n}")
    cap.release()
    json.dump({str(k): v for k, v in rims.items()}, open(path, "w"))
    return rims


def carry_rims(rims, out_dir, sfps):
    """Fill frames where the rim detector missed the rim, by moving a nearby frame's rim with the camera motion."""
    path = os.path.join(out_dir, "camera.json")
    if not os.path.exists(path):
        return rims
    raw = json.load(open(path))
    to_shared = {int(k): np.array(v) for k, v in raw["to_shared"].items()}
    shot = {int(k): v for k, v in raw["shot"].items()}
    seen = sorted(k for k, v in rims.items() if v)
    filled = dict(rims)
    for idx in rims:
        if rims[idx] or idx not in to_shared or not seen:
            continue
        k = min(seen, key=lambda s: abs(s - idx))
        if abs(k - idx) / sfps > RIM_CARRY_SECONDS or shot.get(k) != shot.get(idx):
            continue
        move = np.linalg.inv(to_shared[idx]) @ to_shared[k]
        boxes = []
        for x1, y1, x2, y2 in rims[k]:
            pts = [move @ np.array([x, y, 1.0]) for x, y in ((x1, y1), (x2, y2))]
            (a, b), (c, d) = [(p[0] / p[2], p[1] / p[2]) for p in pts]
            boxes.append([a, b, c, d])
        filled[idx] = boxes
    return filled


def read_scoreboard(meta, box, frames, out_dir):
    """Read the TV scoreboard once per second. Returns [(frame, score_a, score_b)] for readable frames."""
    path = os.path.join(out_dir, "scoreboard.json")
    if os.path.exists(path):
        return [tuple(r) for r in json.load(open(path))]
    import re
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(READER_MODEL)
    model = AutoModelForImageTextToText.from_pretrained(READER_MODEL, dtype=torch.bfloat16).to("cuda").eval()
    text = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": SCORE_PROMPT}]}],
                                    add_generation_prompt=True)
    wanted = set(frames[::max(1, int(meta["fps"]))])
    x1, y1, x2, y2 = box
    reads = []
    cap = cv2.VideoCapture(meta["video"])
    for idx, frame in sampled_frames(cap, meta["start"], meta["end"], meta["fps"]):
        if idx not in wanted:
            continue
        crop = Image.fromarray(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
        inputs = proc(text=[text], images=[crop], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        answer = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        m = re.fullmatch(r"\s*(\d{1,3})\s*\D\s*(\d{1,3})\s*", answer)
        if m:
            reads.append((idx, int(m.group(1)), int(m.group(2))))
    cap.release()
    json.dump(reads, open(path, "w"))
    return reads


def score_changes(reads):
    """Baskets from the scoreboard: a reading counts only if the next reading agrees, and scores never go down."""
    stable = [reads[i] for i in range(len(reads) - 1) if reads[i][1:] == reads[i + 1][1:]]
    changes, current = [], None
    for frame, a, b in stable:
        if current is None:
            current = (a, b)
            continue
        da, db = a - current[0], b - current[1]
        if da < 0 or db < 0 or da + db > 3 or da + db == 0:
            continue
        for team, points in ((0, da), (1, db)):
            if points:
                changes.append({"frame": frame, "team": team, "points": points})
        current = (a, b)
    first = stable[0][0] if stable else None
    last = stable[-1][0] if stable else None
    return changes, first, last, (stable[0][1:] if stable else None), (current if stable else None)


def shot_value(x, y):
    """FIBA Official Basketball Rules 2026, Art. 2.5.4 and 16.2.1: the value depends on where the ball was released.
    3-point area = beyond an arc of radius 6.75 m around the point on the floor under the basket (1.575 m from the
    endline), and beyond the parallel lines 0.90 m from the sidelines; the 3-point line itself is NOT part of the
    3-point area (feet on the line = 2 points). Returns (points, distance to basket, metres from the line)."""
    bx = BASKET_FROM_ENDLINE if x < COURT_LENGTH / 2 else COURT_LENGTH - BASKET_FROM_ENDLINE
    d = float(np.hypot(x - bx, y - COURT_WIDTH / 2))
    from_endline = x if x < COURT_LENGTH / 2 else COURT_LENGTH - x
    if from_endline <= CORNER_STRAIGHT_LENGTH:          # corner: straight lines 0.90 m from the sidelines
        margin = abs(y - COURT_WIDTH / 2) - (COURT_WIDTH / 2 - CORNER_FROM_SIDELINE)
    else:
        margin = d - THREE_POINT_RADIUS
    return (3 if margin > 0 else 2), d, margin


def release_position(tracks, frames, track, frame, sfps):
    """Shooter's court position at release: the last frame they held the ball, or the nearest mapped frame."""
    if "x" in tracks[frame].get(track, {}):
        return tracks[frame][track]["x"], tracks[frame][track]["y"]
    near = [f for f in frames if abs(f - frame) / sfps <= 0.5 and "x" in tracks[f].get(track, {})]
    if not near:
        return None
    f = min(near, key=lambda f: abs(f - frame))
    return tracks[f][track]["x"], tracks[f][track]["y"]


def label(team_names, info):
    return f"{team_names[info['team']]} #{info['number']}" if info["number"] else f"{team_names[info['team']]} ?"


def holder_candidates(players, ball_xy):
    """On-court players whose box (slightly widened) contains the ball, closest to the ball first."""
    bx, by = ball_xy
    found = []
    for track, p in players.items():
        # Referees count (they handle the ball at stoppages) even when standing off the court
        if p["team"] not in (0, 1, REFEREE) or (p["team"] != REFEREE and p.get("off_court")):
            continue
        x1, y1, x2, y2 = p["box"]
        w, h = x2 - x1, y2 - y1
        if x1 - 0.25 * w <= bx <= x2 + 0.25 * w and y1 - 0.1 * h <= by <= y2:
            found.append((np.hypot(bx - (x1 + x2) / 2, (by - (y1 + 0.4 * h)) * 0.6) / h, track))
    return [t for _, t in sorted(found)]


def same_player(a, b, frame_a, frame_b, tracks):
    """Two holders are the same person if they are the same track, or share team+number, or the second track
    starts where the first one was (a broken track)."""
    pa, pb = tracks[frame_a][a], tracks[frame_b][b]
    if a == b or (pa["number"] and pa["number"] == pb["number"] and pa["team"] == pb["team"]):
        return True
    if "x" in pa and "x" in pb and pa["team"] == pb["team"] and not (pa["number"] and pb["number"]):
        return np.hypot(pa["x"] - pb["x"], pa["y"] - pb["y"]) < SAME_PLAYER_METRES
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--teams", nargs=2, default=["Team A", "Team B"])
    parser.add_argument("--scoreboard", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"),
                        help="Pixel box of the TV scoreboard's score row (left team = first --teams name)")
    parser.add_argument("--scoreboard-swap", action="store_true",
                        help="The scoreboard shows the second --teams name on the left")
    args = parser.parse_args()
    out_dir = run_dir(args.name)
    meta = json.load(open(os.path.join(out_dir, "meta.json")))
    fps, sfps = meta["fps"], meta["source_fps"]
    tracks, ball = load_rows(out_dir)
    frames = sorted(tracks)
    rims = carry_rims(find_rims(meta, set(frames), out_dir), out_dir, sfps)

    # 1. Ball holder per frame, confirmed after HOLD_FRAMES in a row
    holder, run_track, run_len, current = {}, None, 0, None
    dead_ball, ref_run = set(), 0
    for idx in frames:
        cands = holder_candidates(tracks[idx], ball[idx][:2]) if idx in ball else []
        # A referee holding the ball (2+ frames in a row) means a dead ball: no pass or interception
        # can link the players before and after it
        if cands and tracks[idx][cands[0]]["team"] == REFEREE:
            ref_run += 1
            if ref_run >= REF_HOLD_FRAMES:
                dead_ball.add(idx)
                current, run_track, run_len = None, None, 0
            holder[idx] = None
            continue
        ref_run = 0
        cands = [c for c in cands if tracks[idx][c]["team"] in (0, 1)]
        # Keep the current holder while the ball is still within their reach: two team-mates standing
        # together would otherwise swap the ball back and forth and count as passes
        h = current if current in cands else (cands[0] if cands else None)
        run_track, run_len = (h, run_len + 1) if h is not None and h == run_track else (h, 1 if h is not None else 0)
        if h is not None and run_len >= HOLD_FRAMES:
            current = h
        elif h is None or h != current:
            current = None if h is None else current
        holder[idx] = current if current is not None and current in tracks[idx] else None

    # 2. Shot attempts: ball in flight reaching the zone above a rim
    shots, last_shot = [], -1e9
    holds = [(idx, holder[idx]) for idx in frames if holder[idx] is not None]
    # Length of each continuous hold: a defender's hand brushing the ball (a short hold) is not a real holder
    hold_seconds, i0 = {}, 0
    for i in range(1, len(holds) + 1):
        if i == len(holds) or holds[i][1] != holds[i0][1] or (holds[i][0] - holds[i - 1][0]) / sfps > 0.5:
            length = (holds[i - 1][0] - holds[i0][0]) / sfps + 1 / fps
            for f, _ in holds[i0:i]:
                hold_seconds[f] = length
            i0 = i
    solid = [(f, t) for f, t in holds if hold_seconds[f] >= SOLID_HOLD_SECONDS]
    for i, idx in enumerate(frames):
        if idx not in ball or ball[idx][2] != "detected" or holder[idx] is not None or not rims.get(idx):
            continue
        bx, by, _ = ball[idx]
        for rx1, ry1, rx2, ry2 in rims[idx]:
            rw, rcx, rcy = rx2 - rx1, (rx1 + rx2) / 2, (ry1 + ry2) / 2
            if abs(bx - rcx) <= SHOT_ZONE_WIDTHS * rw and rcy - SHOT_ZONE_ABOVE * rw <= by <= rcy + 0.3 * rw \
                    and (idx - last_shot) / sfps > SHOT_COOLDOWN_SECONDS:
                # Shooter = last player who really held the ball, not a defender who touched it on release
                before = [(f, t) for f, t in solid if f < idx and (idx - f) / sfps <= 3.0] or \
                         [(f, t) for f, t in holds if f < idx and (idx - f) / sfps <= 3.0]
                if not before:
                    continue
                shooter_frame, shooter = before[-1]
                made = False
                for k in frames[i + 1:]:
                    if (k - idx) / sfps > MADE_SECONDS:
                        break
                    if k in ball and ball[k][2] == "detected" and rims.get(k):
                        kx, ky, _ = ball[k]
                        for qx1, qy1, qx2, qy2 in rims[k]:
                            if qx1 - 0.1 * (qx2 - qx1) <= kx <= qx2 + 0.1 * (qx2 - qx1) and qy2 <= ky <= qy2 + 2.5 * (qx2 - qx1):
                                made = True
                info = tracks[shooter_frame][shooter]
                where = release_position(tracks, frames, shooter, shooter_frame, sfps)
                value = shot_value(*where) if where else None
                shots.append({"frame": idx, "seconds": round(idx / sfps, 2), "type": "shot", "kind": "field_goal",
                              "team": info["team"], "player": label(args.teams, info), "track": shooter, "made": made,
                              "points": None, "source": "ball",
                              "value": value[0] if value else None,
                              "distance_m": round(value[1], 2) if value else None,
                              "near_line": bool(value and abs(value[2]) < NEAR_LINE_METRES),
                              "release_xy": [round(where[0], 2), round(where[1], 2)] if where else None})
                last_shot = idx
                break

    # 2b. The TV scoreboard decides made/missed and points wherever it is readable
    score_info = None
    if args.scoreboard:
        readings = read_scoreboard(meta, args.scoreboard, frames, out_dir)
        if args.scoreboard_swap:
            readings = [(f, b, a) for f, a, b in readings]
        changes, first, last, start_score, end_score = score_changes(readings)
        score_info = {"first_frame": first, "last_frame": last, "start": start_score, "end": end_score, "baskets": changes}
        if first is not None:
            for s in shots:
                if first <= s["frame"] <= last:
                    s["made"], s["source"] = False, "scoreboard"
            for c in changes:
                if c["points"] == 1:
                    # +1 on the scoreboard = a free throw (FIBA Art. 16.2.1)
                    shots.append({"frame": c["frame"], "seconds": round(c["frame"] / sfps, 2), "type": "shot",
                                  "kind": "free_throw", "team": c["team"], "player": f"{args.teams[c['team']]} free throw",
                                  "track": None, "made": True, "points": 1, "source": "scoreboard", "value": 1})
                    continue
                window = [s for s in shots if s["team"] == c["team"] and not s["made"] and s.get("kind") == "field_goal"
                          and c["frame"] - SCORE_LAG_SECONDS * sfps <= s["frame"] <= c["frame"]]
                # Prefer the attempt whose release position gives the same value as the scoreboard
                same_value = [s for s in window if s.get("value") == c["points"]]
                if window:
                    (same_value or window)[-1].update(made=True, points=c["points"], source="scoreboard")
                else:
                    # Basket seen on the scoreboard but the shot itself was not tracked (e.g. a close-up)
                    shots.append({"frame": c["frame"], "seconds": round(c["frame"] / sfps, 2), "type": "shot",
                                  "kind": "field_goal", "team": c["team"],
                                  "player": f"{args.teams[c['team']]} (shooter not seen)", "track": None, "made": True,
                                  "points": c["points"], "source": "scoreboard", "value": None})
            shots.sort(key=lambda s: s["frame"])
    # Made shots the scoreboard could not confirm: points from the FIBA rule on the release position
    for s in shots:
        if s["made"] and s["points"] is None and s.get("value"):
            s["points"] = s["value"]
        s["value_agrees_with_scoreboard"] = (s["value"] == s["points"]) if s["source"] == "scoreboard" and s["made"] \
            and s.get("value") and s.get("kind") == "field_goal" else None

    # 3. Passes and interceptions from changes of holder
    events = list(shots)
    changes, prev = [], None
    for idx in frames:
        h = holder[idx]
        if idx in dead_ball:
            prev = None
            continue
        if h is None:
            continue
        if prev is not None and h != prev[1] and not same_player(prev[1], h, prev[0], idx, tracks):
            changes.append((prev, (idx, h)))
        prev = (idx, h) if prev is None or h != prev[1] or True else prev
    def keeps_ball(frame, team):
        """The new team keeps the ball: no player of the other team holds it within the next few seconds."""
        for idx in frames:
            if frame < idx and (idx - frame) / sfps <= INTERCEPT_KEEP_SECONDS and holder[idx] is not None \
                    and tracks[idx][holder[idx]]["team"] != team:
                return False
        return True

    for (fa, a), (fb, b) in changes:
        gap = (fb - fa) / sfps
        if gap > PASS_MAX_SECONDS:
            continue
        shot_between = any(fa <= s["frame"] <= fb for s in shots)
        pa, pb = tracks[fa][a], tracks[fb][b]
        # The ball must really travel: passer (at release) and receiver (at catch) at least 2 m apart
        if "x" in pa and "x" in pb and np.hypot(pa["x"] - pb["x"], pa["y"] - pb["y"]) < MIN_PASS_METRES:
            continue
        if pa["team"] == pb["team"] and not shot_between:
            events.append({"frame": fb, "seconds": round(fb / sfps, 2), "type": "pass", "team": pa["team"],
                           "from": label(args.teams, pa), "to": label(args.teams, pb), "from_track": a, "to_track": b})
        elif pa["team"] != pb["team"] and not shot_between and keeps_ball(fb, pb["team"]) \
                and hold_seconds.get(fb, 0) >= SOLID_HOLD_SECONDS:
            events.append({"frame": fb, "seconds": round(fb / sfps, 2), "type": "interception", "team": pb["team"],
                           "player": label(args.teams, pb), "from": label(args.teams, pa), "track": b})
    events.sort(key=lambda e: e["frame"])

    # 4. Team ball control: team of the last confirmed holder
    control, team_now = {}, None
    for idx in frames:
        if holder[idx] is not None:
            team_now = tracks[idx][holder[idx]]["team"]
        control[idx] = team_now
    held = Counter(t for t in control.values() if t is not None)
    with open(os.path.join(out_dir, "possession.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "seconds", "holder_track", "holder", "team_in_control"])
        for idx in frames:
            h = holder[idx]
            writer.writerow([idx, round(idx / sfps, 3), h if h is not None else "",
                             label(args.teams, tracks[idx][h]) if h is not None else "",
                             "" if control[idx] is None else args.teams[control[idx]]])

    # 5. Player statistics, grouped by team + jersey number (tracks without a number stay separate)
    players = {}
    has_court = os.path.exists(os.path.join(out_dir, "positions.csv"))
    for idx in frames:
        for track, p in tracks[idx].items():
            # With a court map, only players seen on the court count; without one, every team player counts
            if p["team"] not in (0, 1) or p.get("off_court") or (has_court and "x" not in p):
                continue
            key = label(args.teams, p) if p["number"] else f"{label(args.teams, p)} (track {track})"
            s = players.setdefault(key, {"team": args.teams[p["team"]], "number": p["number"] or None, "frames": 0,
                                         "speeds": [], "track_distance": {}, "holding_frames": 0,
                                         "passes": 0, "passes_received": 0, "interceptions": 0,
                                         "shots": 0, "shots_made": 0, "three_point_attempts": 0,
                                         "three_point_made": 0, "points": 0})
            s["frames"] += 1
            if p.get("speed") is not None:
                s["speeds"].append(p["speed"])
            if "distance" in p:
                s["track_distance"][track] = p["distance"]
            if holder[idx] == track:
                s["holding_frames"] += 1
    def player_key(team, track, frame):
        p = tracks[frame][track]
        return label(args.teams, p) if p["number"] else f"{label(args.teams, p)} (track {track})"
    for e in events:
        if e["type"] == "pass":
            for k, field in ((player_key(e["team"], e["from_track"], e["frame"]) if e["from_track"] in tracks[e["frame"]] else None, "passes"),
                             (player_key(e["team"], e["to_track"], e["frame"]), "passes_received")):
                if k in players:
                    players[k][field] += 1
        elif e["type"] == "interception" and player_key(e["team"], e["track"], e["frame"]) in players:
            players[player_key(e["team"], e["track"], e["frame"])]["interceptions"] += 1
    for s in shots:
        if s["track"] is None:
            continue
        for f in frames:
            if s["track"] in tracks[f]:
                k = player_key(s["team"], s["track"], f)
                if k in players:
                    players[k]["shots"] += 1
                    players[k]["shots_made"] += int(s["made"])
                    three = s.get("points") == 3 or (s.get("points") is None and s.get("value") == 3)
                    players[k]["three_point_attempts"] += int(three)
                    players[k]["three_point_made"] += int(three and s["made"])
                    players[k]["points"] += (s.get("points") or 0) if s["made"] else 0
                break
    player_report = {}
    for key, s in players.items():
        if s["frames"] < fps:            # seen for under a second: not worth reporting
            continue
        sp = s.pop("speeds")
        dist = sum(s.pop("track_distance").values())
        player_report[key] = {**s, "seconds_on_screen": round(s.pop("frames") / fps, 1),
                              "distance_m": round(dist, 1) if has_court else None,
                              "avg_speed_kmh": round(float(np.mean(sp)), 1) if sp else None,
                              "max_speed_kmh": round(float(np.percentile(sp, 98)), 1) if len(sp) >= 5 else None,
                              "time_with_ball_s": round(s.pop("holding_frames") / fps, 1)}

    team_stats = {}
    for t, name in enumerate(args.teams):
        team_stats[name] = {
            "ball_control_pct": round(100 * held[t] / max(1, sum(held.values())), 1),
            "passes": sum(e["type"] == "pass" and e["team"] == t for e in events),
            "interceptions": sum(e["type"] == "interception" and e["team"] == t for e in events),
            "field_goal_attempts": sum(s["team"] == t and s["kind"] == "field_goal" for s in shots),
            "field_goals_made": sum(s["team"] == t and s["kind"] == "field_goal" and s["made"] for s in shots),
            "three_point_attempts": sum(s["team"] == t and s["kind"] == "field_goal" and
                                        (s["points"] or s.get("value")) == 3 for s in shots),
            "three_point_made": sum(s["team"] == t and s["made"] and s["points"] == 3 for s in shots),
            "free_throws_made": sum(s["team"] == t and s["kind"] == "free_throw" for s in shots),
            "points_scored_in_clip": sum(s["points"] or 0 for s in shots if s["team"] == t and s["made"]),
            "points_confirmed_by_scoreboard": sum(s["points"] or 0 for s in shots
                                                  if s["team"] == t and s["made"] and s["source"] == "scoreboard"),
            "distance_m": round(sum(p["distance_m"] for p in player_report.values() if p["team"] == name), 1) if has_court else None}
    holder_frames = sum(1 for v in holder.values() if v is not None)
    report = {
        "video": meta["video"], "seconds_analysed": round(len(frames) / fps, 1), "teams": args.teams,
        "team_stats": team_stats, "players": player_report,
        "scoreboard": None if not score_info or score_info["first_frame"] is None else {
            "readable_from_s": round(score_info["first_frame"] / sfps, 1), "readable_to_s": round(score_info["last_frame"] / sfps, 1),
            "score_at_start": score_info["start"], "score_at_end": score_info["end"], "baskets": score_info["baskets"],
            "swapped": bool(args.scoreboard_swap)},
        "court_mapped": has_court,
        "quality": {"frames_with_ball_holder_pct": round(100 * holder_frames / len(frames), 1),
                    "frames_with_team_in_control_pct": round(100 * sum(held.values()) / len(frames), 1),
                    "notes": ["Points follow FIBA Official Basketball Rules 2026 Art. 16.2.1: free throw 1, released from the 2-point area 2, "
                              "from the 3-point area 3 (beyond the 6.75 m arc / 0.90 m corner lines; the line itself counts as 2).",
                              "Made shots and points are confirmed by the TV scoreboard where it is visible; elsewhere the value comes "
                              "from the shooter's court position at release and is marked unconfirmed.",
                              "Shot made/missed without a scoreboard is decided from the ball seen below the rim; fast or hidden balls can be missed.",
                              "Interceptions include steals; possession changes after a shot (rebounds) are not counted.",
                              "Players without a confirmed jersey number are listed per track."]}}
    json.dump(events, open(os.path.join(out_dir, "events.json"), "w"), indent=1)
    json.dump(report, open(os.path.join(out_dir, "report.json"), "w"), indent=1)

    print(f"Ball holder known in {report['quality']['frames_with_ball_holder_pct']}% of frames; "
          f"team in control known in {report['quality']['frames_with_team_in_control_pct']}%")
    for name, s in team_stats.items():
        print(f"{name}: control {s['ball_control_pct']}% | passes {s['passes']} | interceptions {s['interceptions']} | "
              f"FG {s['field_goals_made']}/{s['field_goal_attempts']} (3PT {s['three_point_made']}/{s['three_point_attempts']}) | "
              f"FT {s['free_throws_made']} | points {s['points_scored_in_clip']} "
              f"({s['points_confirmed_by_scoreboard']} confirmed by scoreboard) | distance {s['distance_m']} m")
    print(f"Events: {len(events)} | players reported: {len(player_report)}")
    print(f"Saved events.json, report.json, possession.csv")


if __name__ == "__main__":
    main()
