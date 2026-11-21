"""
Render the final broadcast-style analysis video.

Run last, after match_events.py:
    python render_broadcast.py --name aus_ind_v2

Shows, with live (running) numbers:
  - top bar: each team's ball control %, passes, interceptions, shots made/attempted and points in the clip
  - players: team-coloured ring at the feet, jersey-number badge, speed (km/h) under the player
  - ball holder: highlighted ring and a name tag with speed and distance covered
  - ball: orange marker with a short trail
  - mini-map: court with every on-court player (numbered dots) and the ball holder
  - event feed: the latest passes, interceptions and shots, plus a banner when a shot is taken
Output: runs/pipeline/<name>/analysis.mp4
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict, deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from basketball_pipeline import run_dir, sampled_frames
from court_mapping import COURT_LENGTH, COURT_WIDTH, court_lines

# Display colours (BGR): Team A = lighter jersey, Team B = darker jersey
TEAM_COLORS = [(60, 190, 250), (235, 110, 95)]
REF_COLOR = (170, 170, 170)
BALL_COLOR = (0, 140, 255)
HOLDER_COLOR = (0, 230, 255)
PANEL = (28, 24, 22)
MAP_SCALE, MAP_PAD = 15, 14
FEED_SECONDS = 6.0
BANNER_SECONDS = 2.0
TRAIL = 6


def font(size, bold=False):
    # Windows fonts on the laptop; DejaVu on Linux servers (Modal)
    windows = [os.path.join("C:/Windows/Fonts", n) for n in
               ((("seguisb.ttf", "segoeuib.ttf") if bold else ("segoeui.ttf",)) + ("arial.ttf",))]
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    linux = ["/usr/share/fonts/truetype/dejavu/" + name]
    try:    # cloud machines without system fonts: matplotlib (installed with ultralytics) ships DejaVu
        import matplotlib
        linux.append(os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", name))
    except ImportError:
        pass
    for path in windows + linux:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


F_SMALL, F_MED, F_BIG, F_BOLD, F_FEED = font(18), font(22), font(30, True), font(22, True), font(16, True)


def blend_rect(img, x1, y1, x2, y2, color, alpha):
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    cv2.addWeighted(np.full_like(roi, color), alpha, roi, 1 - alpha, 0, roi)


def rgb(bgr):
    return tuple(int(c) for c in bgr[::-1])


def load(out_dir):
    tracks = defaultdict(list)
    with open(os.path.join(out_dir, "tracks_named.csv"), newline="") as f:
        for r in csv.DictReader(f):
            tracks[int(r["frame"])].append((int(r["track"]), int(r["team"]), [float(r[k]) for k in ("x1", "y1", "x2", "y2")], r["number"]))
    positions = {}    # empty when the court was not mapped: no speed tags and no court map are drawn
    path = os.path.join(out_dir, "positions.csv")
    if os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                positions[(int(r["frame"]), int(r["track"]))] = {
                    "x": float(r["x_m"]), "y": float(r["y_m"]), "off": r["off_court"] == "1",
                    "speed": float(r["speed_kmh"]) if r["speed_kmh"] else None, "dist": float(r["distance_m"])}
    ball = {}
    with open(os.path.join(out_dir, "ball.csv"), newline="") as f:
        for r in csv.DictReader(f):
            ball[int(r["frame"])] = (r["status"], float(r["x"]) if r["x"] else None, float(r["y"]) if r["y"] else None)
    possession = {}
    with open(os.path.join(out_dir, "possession.csv"), newline="") as f:
        for r in csv.DictReader(f):
            possession[int(r["frame"])] = (int(r["holder_track"]) if r["holder_track"] else None, r["holder"], r["team_in_control"])
    events = json.load(open(os.path.join(out_dir, "events.json")))
    path = os.path.join(out_dir, "scoreboard.json")
    reads = [tuple(r) for r in json.load(open(path))] if os.path.exists(path) else []
    # Only readings confirmed by the next one (the scoreboard animates when it changes)
    scoreboard = [reads[i] for i in range(len(reads) - 1) if reads[i][1:] == reads[i + 1][1:]]
    return tracks, positions, ball, possession, events, scoreboard


def minimap_base():
    w, h = int(COURT_LENGTH * MAP_SCALE) + 2 * MAP_PAD, int(COURT_WIDTH * MAP_SCALE) + 2 * MAP_PAD
    img = np.full((h, w, 3), (70, 120, 175), np.uint8)
    cv2.rectangle(img, (MAP_PAD, MAP_PAD), (w - MAP_PAD, h - MAP_PAD), (90, 145, 205), -1)
    for line in court_lines():
        pts = np.array([(MAP_PAD + x * MAP_SCALE, MAP_PAD + y * MAP_SCALE) for x, y in line], np.int32)
        cv2.polylines(img, [pts], False, (245, 245, 245), 2, cv2.LINE_AA)
    return img


def event_text(e):
    if e["type"] == "pass":
        return f"PASS  {e['from']} → {e['to']}"
    if e["type"] == "interception":
        return f"INTERCEPTION  {e['player']} (from {e['from']})"
    if e.get("kind") == "free_throw":
        return f"FREE THROW +1  {e['player'].replace(' free throw', '')}"
    result = "MADE" if e["made"] else "MISSED"
    pts = f" +{e['points']}" if e.get("points") else ""
    sure = " (unconfirmed)" if e["made"] and e.get("source") == "ball" else ""
    return f"SHOT {result}{pts}{sure}  {e['player']}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    out_dir = run_dir(args.name)
    meta = json.load(open(os.path.join(out_dir, "meta.json")))
    teams = json.load(open(os.path.join(out_dir, "players.json")))["teams"]
    tracks, positions, ball, possession, events, scoreboard = load(out_dir)
    sfps = meta["source_fps"]
    # A frame where most mapped players land off the court has a wrong mapping (e.g. carried too far from its
    # picture): forget its positions so the players stay visible (rings, numbers) and nothing wrong is drawn on the map
    bad = 0
    for idx, rows in tracks.items():
        mapped = [positions[(idx, t)] for t, team, _, _ in rows if team in (0, 1) and (idx, t) in positions]
        if len(mapped) >= 3 and sum(p["off"] for p in mapped) > len(mapped) / 2:
            bad += 1
            for t, _, _, _ in rows:
                positions.pop((idx, t), None)
    if bad:
        print(f"Frames with an unreliable court mapping (positions ignored): {bad}")

    cap = cv2.VideoCapture(meta["video"])
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, round(sfps / meta["fps"]))
    path = os.path.join(out_dir, "analysis.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), sfps / step, (W, H))
    base_map = minimap_base()
    stats = [Counter(), Counter()]
    control = Counter()
    trail = deque(maxlen=TRAIL)
    done_events = 0

    for n, (idx, frame) in enumerate(sampled_frames(cap, meta["start"], meta["end"], meta["fps"])):
        if idx not in possession:
            continue
        # Running statistics up to this frame
        while done_events < len(events) and events[done_events]["frame"] <= idx:
            e = events[done_events]
            s = stats[e["team"]]
            if e["type"] == "pass":
                s["passes"] += 1
            elif e["type"] == "interception":
                s["interceptions"] += 1
            elif e.get("kind") == "free_throw":
                s["ft"] += 1
                s["points"] += 1
            else:
                s["shots"] += 1
                s["made"] += int(e["made"])
                three = (e.get("points") or e.get("value")) == 3
                s["3a"] += int(three)
                s["3m"] += int(three and e["made"])
                s["points"] += (e.get("points") or 0) if e["made"] else 0
            done_events += 1
        holder_track, holder_name, team_ctrl = possession[idx]
        if team_ctrl:
            control[teams.index(team_ctrl)] += 1

        texts = []          # (x, y, text, font, rgb, anchor)
        # Players
        for track, team, (x1, y1, x2, y2), number in tracks.get(idx, []):
            pos = positions.get((idx, track))
            if pos and pos["off"]:
                continue
            cx, w = int((x1 + x2) / 2), x2 - x1
            if team not in (0, 1):
                cv2.ellipse(frame, (cx, int(y2)), (int(w * 0.5), int(w * 0.16)), 0, -20, 200, REF_COLOR, 2, cv2.LINE_AA)
                continue
            color = TEAM_COLORS[team]
            is_holder = track == holder_track
            if is_holder:
                cv2.ellipse(frame, (cx, int(y2)), (int(w * 0.8), int(w * 0.26)), 0, 0, 360, HOLDER_COLOR, 4, cv2.LINE_AA)
            cv2.ellipse(frame, (cx, int(y2)), (int(w * 0.6), int(w * 0.19)), 0, -20, 200, color, 3, cv2.LINE_AA)
            # Number badge above the head
            by = int(y1) - 20
            cv2.circle(frame, (cx, by), 16, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, by), 16, (255, 255, 255), 2, cv2.LINE_AA)
            texts.append((cx, by, number or "?", F_BOLD if number else F_MED, (25, 25, 25) if team == 0 else (255, 255, 255), "mm"))
            speed = pos["speed"] if pos else None
            if is_holder:
                tag = holder_name + (f"  ·  {speed:.0f} km/h" if speed is not None else "") + (f"  ·  {pos['dist']:.0f} m" if pos else "")
                texts.append((cx, int(y2) + 30, tag, F_BOLD, (20, 20, 20), "pill", HOLDER_COLOR))
            elif speed is not None:
                texts.append((cx, int(y2) + 22, f"{speed:.0f} km/h", F_SMALL, (255, 255, 255), "pill", (30, 30, 30)))

        # Ball and trail
        status, bx, by_ = ball.get(idx, ("unknown", None, None))
        if status in ("detected", "filled") and bx is not None:
            trail.append((int(bx), int(by_)))
            for i in range(1, len(trail)):
                cv2.line(frame, trail[i - 1], trail[i], BALL_COLOR, 1 + i, cv2.LINE_AA)
            cv2.circle(frame, (int(bx), int(by_)), 15, BALL_COLOR, 3, cv2.LINE_AA)
        else:
            trail.clear()

        # Top bar: two team cards with running stats
        total_ctrl = max(1, sum(control.values()))
        compact = W < 1600                      # 720p and smaller: cards share the width, no centre title
        card_w = min(560, (W - 60) // 2) if compact else 560
        for t in (0, 1):
            x0 = 20 if t == 0 else W - 20 - card_w
            blend_rect(frame, x0, 16, x0 + card_w, 112, PANEL, 0.78)
            cv2.rectangle(frame, (x0, 16), (x0 + 8, 112), TEAM_COLORS[t], -1)
            pct = 100 * control[t] / total_ctrl
            cv2.rectangle(frame, (x0 + 24, 92), (x0 + card_w - 24, 100), (70, 70, 70), -1)
            cv2.rectangle(frame, (x0 + 24, 92), (x0 + 24 + int((card_w - 48) * pct / 100), 100), TEAM_COLORS[t], -1)
            s = stats[t]
            texts.append((x0 + 24, 24, teams[t], F_BIG, rgb(TEAM_COLORS[t]), "lt"))
            if compact:
                texts.append((x0 + 110, 32, f"Ball {pct:.0f}%", F_MED, (235, 235, 235), "lt"))
                texts.append((x0 + 24, 62, f"Pass {s['passes']}  Int {s['interceptions']}  "
                                           f"FG {s['made']}/{s['shots']}  3PT {s['3m']}/{s['3a']}", F_MED, (235, 235, 235), "lt"))
            else:
                texts.append((x0 + 130, 32, f"Ball control {pct:.0f}%", F_MED, (235, 235, 235), "lt"))
                texts.append((x0 + 24, 62, f"Passes {s['passes']}   Interceptions {s['interceptions']}   "
                                           f"FG {s['made']}/{s['shots']}   3PT {s['3m']}/{s['3a']}", F_MED, (235, 235, 235), "lt"))
            texts.append((x0 + card_w - 24, 26, f"{s['points']} PTS", F_BIG, (255, 255, 255), "rt"))
        if not compact:
            texts.append((W // 2, 40, "AI MATCH ANALYSIS", F_BOLD, (255, 255, 255), "mm"))
        tv = [r for r in scoreboard if r[0] <= idx]
        if tv and not compact:
            texts.append((W // 2, 72, f"Game score (TV)  {tv[-1][1]} – {tv[-1][2]}", F_MED, (200, 200, 200), "mm"))
        if not compact:
            texts.append((W // 2, 100, f"{(idx / sfps) - meta['start']:.0f}s  ·  points in clip, FIBA rules", F_SMALL, (170, 170, 170), "mm"))

        # Mini-map (bottom-left), only when the court was mapped
        mm = base_map.copy()
        map_has_players = False
        for track, team, box, number in (tracks.get(idx, []) if positions else []):
            pos = positions.get((idx, track))
            if team not in (0, 1) or not pos or pos["off"]:
                continue
            px, py = int(MAP_PAD + pos["x"] * MAP_SCALE), int(MAP_PAD + pos["y"] * MAP_SCALE)
            if not (0 <= px < mm.shape[1] and 0 <= py < mm.shape[0]):
                continue
            map_has_players = True
            if track == holder_track:
                cv2.circle(mm, (px, py), 13, BALL_COLOR, 3, cv2.LINE_AA)
            cv2.circle(mm, (px, py), 9, TEAM_COLORS[team], -1, cv2.LINE_AA)
            cv2.circle(mm, (px, py), 9, (255, 255, 255), 1, cv2.LINE_AA)
        mh, mw = mm.shape[:2]
        mx, my = 20, H - mh - 20
        if map_has_players:
            blend_rect(frame, mx - 6, my - 34, mx + mw + 6, my + mh + 6, PANEL, 0.75)
            frame[my:my + mh, mx:mx + mw] = cv2.addWeighted(mm, 0.92, frame[my:my + mh, mx:mx + mw], 0.08, 0)
            texts.append((mx + 4, my - 30, "COURT MAP", F_SMALL, (220, 220, 220), "lt"))
        for track, team, box, number in (tracks.get(idx, []) if positions else []):
            pos = positions.get((idx, track))
            if team in (0, 1) and pos and not pos["off"] and number and 0 <= pos["x"] <= COURT_LENGTH and 0 <= pos["y"] <= COURT_WIDTH:
                texts.append((int(mx + MAP_PAD + pos["x"] * MAP_SCALE), int(my + MAP_PAD + pos["y"] * MAP_SCALE),
                              number, font(12, True), (20, 20, 20) if team == 0 else (255, 255, 255), "mm"))

        # Event feed (bottom-right) and shot banner
        recent = [e for e in events[:done_events] if (idx - e["frame"]) / sfps <= FEED_SECONDS][-3:]
        if recent:
            # Right of the TV scoreboard, above the bottom edge
            fx, fy = W - 455, H - 30 - 30 * len(recent)
            blend_rect(frame, fx - 10, fy - 8, W - 12, H - 22, PANEL, 0.75)
            for i, e in enumerate(recent):
                texts.append((fx, fy + 30 * i, f"{(e['frame'] / sfps) - meta['start']:.0f}s  {event_text(e)}", F_FEED,
                              rgb(TEAM_COLORS[e["team"]]), "lt"))
        shot = [e for e in events[:done_events] if e["type"] == "shot" and (idx - e["frame"]) / sfps <= BANNER_SECONDS]
        if shot:
            e = shot[-1]
            unconfirmed = "  (unconfirmed)" if e["made"] and e.get("source") == "ball" else ""
            if e.get("kind") == "free_throw":
                label = "FREE THROW +1  " + e["player"].replace(" free throw", "")
            elif e["made"]:
                label = ("3-POINTER! +3  " if e.get("points") == 3 else "2 POINTS! +2  ") + e["player"] + unconfirmed
            else:
                label = ("3PT MISSED  " if e.get("value") == 3 else "SHOT MISSED  ") + e["player"]
            blend_rect(frame, W // 2 - 330, 130, W // 2 + 330, 186, TEAM_COLORS[e["team"]] if e["made"] else PANEL, 0.85)
            texts.append((W // 2, 158, label, F_BIG, (20, 20, 20) if e["made"] else (255, 255, 255), "mm"))

        # All text in one PIL pass (smooth fonts)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        for item in texts:
            x, y, text, fnt, color, anchor = item[:6]
            if anchor == "pill":
                l, t, r, b = draw.textbbox((x, y), text, font=fnt, anchor="mm")
                draw.rounded_rectangle((l - 10, t - 5, r + 10, b + 5), radius=10, fill=rgb(item[6]))
                draw.text((x, y), text, font=fnt, fill=color, anchor="mm")
            else:
                draw.text((x, y), text, font=fnt, fill=color, anchor=anchor)
        writer.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        if n % 300 == 0:
            print(f"  rendered frame {n}")
    writer.release()
    cap.release()
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
