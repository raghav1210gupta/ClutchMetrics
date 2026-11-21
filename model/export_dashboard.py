"""
Export one analysed match for the ClutchMetrics website.

Run last, after render_broadcast.py:
    python export_dashboard.py --name aus_ind_v2 --site ../website ^
        --team-names Australia India --title "Australia vs India" ^
        --competition "FIBA U16 Asian Championship 2023" --date 2023-09-17 --venue "Doha, Qatar"

Writes into the website:
  data/match.json    every number the coach dashboard shows (format in data/MATCH_FORMAT.md)
  data/analysis.mp4  the analysis video as 720p H.264, which every browser plays
Times in match.json are seconds of the analysis video (0 = its first frame).
"""

import argparse
import csv
import json
import os
import subprocess
from collections import Counter, defaultdict

import numpy as np

from basketball_pipeline import run_dir

COURT_LENGTH, COURT_WIDTH = 28.0, 15.0
PAINT_HALF_WIDTH, PAINT_LENGTH = 2.45, 5.8
CORNER_STRAIGHT_LENGTH = 2.99
SPEED_STEP_S = 0.5


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def shot_place(shot):
    """Release point on a half court seen from above the basket: x across (-7.5..7.5 m), y out from the endline.
    NOTE: the website draws x mirrored (dashboard.js toCourt, player.js); keep this sign as it is, or change both."""
    if not shot.get("release_xy"):
        return None
    x, y = shot["release_xy"]
    if x < COURT_LENGTH / 2:
        return round(y - COURT_WIDTH / 2, 2), round(x, 2)
    return round(COURT_WIDTH / 2 - y, 2), round(COURT_LENGTH - x, 2)


def zone_of(shot, place):
    if shot.get("kind") == "free_throw":
        return "Free throw"
    value = shot["points"] if shot["made"] and shot.get("points") else shot.get("value")
    if place is None:
        return "Three-pointer" if value == 3 else "Two-pointer" if value == 2 else "Unknown"
    across, out = place
    if value == 3:
        return "Corner 3" if out <= CORNER_STRAIGHT_LENGTH else "Above-break 3"
    if abs(across) <= PAINT_HALF_WIDTH and out <= PAINT_LENGTH:
        return "Paint"
    return "Mid-range"


def convert_video(src, dst):
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", src, "-vf", "scale=1280:-2", "-c:v", "libx264",
                    "-preset", "medium", "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst],
                   check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--site", required=True, help="Folder of the ClutchMetrics website")
    parser.add_argument("--team-names", nargs=2, default=None, help="Full team names (codes come from the run)")
    parser.add_argument("--title", default="")
    parser.add_argument("--competition", default="")
    parser.add_argument("--date", default="")
    parser.add_argument("--venue", default="")
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()

    out_dir = run_dir(args.name)
    meta = json.load(open(os.path.join(out_dir, "meta.json")))
    report = json.load(open(os.path.join(out_dir, "report.json")))
    events = json.load(open(os.path.join(out_dir, "events.json")))
    codes = report["teams"]
    names = args.team_names or codes
    sfps, fps, start = meta["source_fps"], meta["fps"], meta["start"]
    t_of = lambda frame: round(frame / sfps - start, 2)

    # Who is who in every frame: team and jersey number (after all fixes), joined with court positions
    named = {(int(r["frame"]), int(r["track"])): (int(r["team"]), r["number"]) for r in read_csv(os.path.join(out_dir, "tracks_named.csv"))}
    positions_path = os.path.join(out_dir, "positions.csv")
    court_mapped = os.path.exists(positions_path)
    rows = []
    for r in (read_csv(positions_path) if court_mapped else []):
        key = (int(r["frame"]), int(r["track"]))
        team, number = named.get(key, (int(r["team"]), r["number"]))
        if team not in (0, 1) or r["off_court"] == "1":
            continue
        rows.append({"frame": key[0], "track": key[1], "team": team, "no": number or None,
                     "x": float(r["x_m"]), "y": float(r["y_m"]),
                     "speed": float(r["speed_kmh"]) if r["speed_kmh"] else None})
    frames = sorted({r["frame"] for r in rows})

    # Heatmaps: seconds spent in each 1 m x 1 m cell, per team and per player
    def empty():
        return np.zeros((int(COURT_WIDTH), int(COURT_LENGTH)))
    heat_team, heat_player = [empty(), empty()], defaultdict(empty)
    for r in rows:
        cx = min(max(int(r["x"]), 0), int(COURT_LENGTH) - 1)
        cy = min(max(int(r["y"]), 0), int(COURT_WIDTH) - 1)
        if -1 <= r["x"] <= COURT_LENGTH + 1 and -1 <= r["y"] <= COURT_WIDTH + 1:
            heat_team[r["team"]][cy, cx] += 1 / fps
            if r["no"]:
                heat_player[f"{r['team']}-{r['no']}"][cy, cx] += 1 / fps

    # Speed over time per player (average over half-second steps)
    speed_bins = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["no"] and r["speed"] is not None:
            speed_bins[f"{r['team']}-{r['no']}"][int(t_of(r["frame"]) / SPEED_STEP_S)].append(r["speed"])
    speed = {k: [[round(b * SPEED_STEP_S, 1), round(float(np.mean(v)), 1)] for b, v in sorted(bins.items())]
             for k, bins in speed_bins.items()}

    # Players: numbered players from the report; the rest grouped as "unidentified" per team
    players, unidentified = [], [defaultdict(float), defaultdict(float)]
    for key, p in report["players"].items():
        team = codes.index(p["team"])
        entry = {"team": team, "no": p["number"], "seconds": p["seconds_on_screen"], "pts": p["points"],
                 "fgm": p["shots_made"], "fga": p["shots"], "tpm": p["three_point_made"], "tpa": p["three_point_attempts"],
                 "passes": p["passes"], "received": p["passes_received"], "int": p["interceptions"],
                 "distance_m": p["distance_m"], "avg_kmh": p["avg_speed_kmh"], "top_kmh": p["max_speed_kmh"],
                 "ball_s": p["time_with_ball_s"]}
        if p["number"]:
            players.append(entry)
        else:
            u = unidentified[team]
            for k in ("seconds", "pts", "fgm", "fga", "tpm", "tpa", "passes", "received", "int", "distance_m", "ball_s"):
                if entry[k] is None:
                    continue
                u[k] += entry[k] or 0
    players.sort(key=lambda p: (p["team"], int(p["no"])))

    # Shots
    shots = []
    for s in (e for e in events if e["type"] == "shot"):
        place = shot_place(s)
        number = None
        if s.get("track") is not None and s["player"].split("#")[-1].strip().isdigit():
            number = s["player"].split("#")[-1].strip()
        shots.append({"t": t_of(s["frame"]), "team": s["team"], "no": number, "player": s["player"],
                      "kind": s.get("kind", "field_goal"), "made": s["made"], "points": s.get("points"),
                      "value": s.get("value"), "confirmed": s["source"] == "scoreboard",
                      "near_line": bool(s.get("near_line")), "distance_m": s.get("distance_m"),
                      "x": place[0] if place else None, "y": place[1] if place else None, "zone": zone_of(s, place)})

    # Passes and interceptions
    plays = []
    for e in events:
        if e["type"] == "pass":
            plays.append({"t": t_of(e["frame"]), "type": "pass", "team": e["team"], "from": e["from"], "to": e["to"]})
        elif e["type"] == "interception":
            plays.append({"t": t_of(e["frame"]), "type": "interception", "team": e["team"], "player": e["player"], "from": e["from"]})

    # Pass network per team (numbered players only)
    network = []
    for team in (0, 1):
        edges = Counter()
        for p in plays:
            if p["type"] == "pass" and p["team"] == team and "#" in p["from"] and "#" in p["to"]:
                edges[(p["from"].split("#")[1], p["to"].split("#")[1])] += 1
        nodes = sorted({n for pair in edges for n in pair}, key=int)
        network.append({"nodes": nodes, "edges": [{"from": a, "to": b, "count": c} for (a, b), c in edges.most_common()]})

    # Possession: who was in control, as time segments
    poss = read_csv(os.path.join(out_dir, "possession.csv"))
    segments = []
    for r in poss:
        team = codes.index(r["team_in_control"]) if r["team_in_control"] in codes else None
        t = t_of(int(r["frame"]))
        if segments and segments[-1]["team"] == team:
            segments[-1]["t1"] = round(t + 1 / fps, 2)
        else:
            segments.append({"t0": t, "t1": round(t + 1 / fps, 2), "team": team})

    # TV scoreboard readings (only those confirmed by the next reading)
    sb_path = os.path.join(out_dir, "scoreboard.json")
    reads = [tuple(r) for r in json.load(open(sb_path))] if os.path.exists(sb_path) else []
    if (report.get("scoreboard") or {}).get("swapped"):      # the board shows the second team on the left
        reads = [(r[0], r[2], r[1]) for r in reads]
    stable = [[t_of(r[0]), r[1], r[2]] for i, r in enumerate(reads[:-1]) if r[1:] == reads[i + 1][1:]]

    ts = report["team_stats"]
    teams = []
    for i, code in enumerate(codes):
        s = ts[code]
        teams.append({"id": i, "code": code, "name": names[i], "points": s["points_scored_in_clip"],
                      "points_confirmed": s["points_confirmed_by_scoreboard"],
                      "fgm": s["field_goals_made"], "fga": s["field_goal_attempts"],
                      "tpm": s["three_point_made"], "tpa": s["three_point_attempts"], "ftm": s["free_throws_made"],
                      "ball_control_pct": s["ball_control_pct"], "passes": s["passes"], "interceptions": s["interceptions"],
                      "distance_m": s["distance_m"],
                      "unidentified": {k: round(v, 1) for k, v in unidentified[i].items()}})

    if court_mapped:
        on_court, numbered = len(rows), sum(1 for r in rows if r["no"])
    else:   # no court map: share of team-player boxes with a number, from the tracks
        team_rows = [v for v in named.values() if v[0] in (0, 1)]
        on_court, numbered = len(team_rows), sum(1 for v in team_rows if v[1])
    match = {
        "format": 1, "sample": False, "court_mapped": court_mapped,
        "title": args.title or f"{names[0]} vs {names[1]}", "competition": args.competition,
        "date": args.date, "venue": args.venue,
        "clip": {"source_start_s": start, "source_end_s": meta["end"], "duration_s": round(len(poss) / fps, 1), "fps": fps},
        "video": "data/analysis.mp4",
        "teams": teams, "players": players, "shots": shots, "plays": plays, "possession": segments,
        "pass_network": network,
        "tv_score": {"readings": stable},
        "heat": {"cell_m": 1, "cols": int(COURT_LENGTH), "rows": int(COURT_WIDTH),
                 "teams": [np.round(h, 2).tolist() for h in heat_team],
                 "players": {k: np.round(h, 2).tolist() for k, h in heat_player.items()}},
        "speed": speed,
        "quality": {"ball_holder_known_pct": report["quality"]["frames_with_ball_holder_pct"],
                    "players_with_number_pct": round(100 * numbered / max(1, on_court), 1),
                    "notes": report["quality"]["notes"]},
    }
    data_dir = os.path.join(args.site, "data")
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "match.json"), "w", encoding="utf-8") as f:
        json.dump(match, f, separators=(",", ":"))
    size = os.path.getsize(os.path.join(data_dir, "match.json")) / 1024
    print(f"Saved {data_dir}/match.json ({size:.0f} KB): {len(players)} players, {len(shots)} shots, {len(plays)} passes/interceptions")
    if not args.skip_video:
        dst = os.path.join(data_dir, "analysis.mp4")
        convert_video(os.path.join(out_dir, "analysis.mp4"), dst)
        print(f"Saved {dst} ({os.path.getsize(dst) / 1e6:.1f} MB, H.264 720p)")


if __name__ == "__main__":
    main()
