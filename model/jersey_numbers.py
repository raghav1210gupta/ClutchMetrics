"""
Give each tracked player a jersey number, so tracks become real players ("IND #7").

Run after `basketball_pipeline.py track` on the same run name:
    python jersey_numbers.py --name aus_ind_v2 --teams AUS IND

How it works:
  1. The 5-class RF-DETR model finds jersey-number boxes; each box is matched to the one player box that contains it.
  2. Qwen3-VL-2B reads the digits on each number crop (runs locally, no API key).
  3. Voting: a track only gets a number when several reads agree. Numbers that a team never shows
     repeatedly across the whole video are treated as misreads (e.g. a single "38" when the player is "36").
  4. Two tracks of the same team on screen at the same time cannot share a number; the weaker one is dropped.
Tracks without a confirmed number stay "unknown" - we never guess.

Outputs in runs/pipeline/<name>/: players.json, tracks_named.csv, annotated_numbers.mp4
"""

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict

import cv2
import numpy as np

from basketball_pipeline import REFEREE, run_dir, sampled_frames

NUMBER_MODEL = "weights/rfdetr_5class/checkpoint_best_total.pth"
NUMBER_CLASS = 1
NUMBER_CONF = 0.3
# Reader settings (environment variables, so the server and the experiments can switch without code changes):
#   CM_NUMBER_READER   Hugging Face model id (default Qwen3-VL-2B; Qwen/Qwen3-VL-8B-Instruct needs ~18 GB of GPU memory)
#   CM_NUMBER_SIZE     height in pixels the number crop is scaled to before reading (default 224)
#   CM_NUMBER_MIN_CONF reads the reader itself is less sure of than this are ignored (0 = keep all, the old behaviour)
READER_MODEL = os.environ.get("CM_NUMBER_READER", "Qwen/Qwen3-VL-2B-Instruct")
READ_SIZE = int(os.environ.get("CM_NUMBER_SIZE", "224"))   # 224 tested better than 160: 90% vs 87% correct, more numbers found
MIN_READ_CONF = float(os.environ.get("CM_NUMBER_MIN_CONF", "0"))
READER_PROMPT = "What is the jersey number in this image? Answer with only the digits, or 'none' if it is not readable."
MIN_NUMBER_HEIGHT = 13      # px; smaller numbers are too blurry to read
MAX_READS_PER_TRACK = 25    # stop reading a track once it has this many digit reads
BATCH = 16
SPLIT_MIN_READS = 3         # a number must be read this many times in a row to count as a track swap
MIN_VOTES = 2              # agreeing reads needed to confirm a track's number
MIN_SHARE = 0.4             # share of all the track's digit reads that must agree
MARGIN = 1.5                # the winning number needs this many times the votes of the runner-up
ROSTER_MIN_VOTES = 4        # a number is real for a team if it wins on some track with this many reads...
ROSTER_MIN_SHARE = 0.55     # ...and this share of that track's reads,
ROSTER_MIN_TRACKS = 2       # ...on at least this many separate tracks
# Off: tested on AUS-IND, most lineup guesses were wrong because of tracker swaps and team-colour errors
USE_LINEUP_RULE = False
LINEUP_SECONDS = 10.0       # numbers a team used within this time count as its current lineup
LINEUP_MIN_FRAMES = 8       # ignore very short tracks for the lineup rule
LINK_SECONDS = 3.0          # court linking: a broken track may continue up to this long after...
LINK_BASE_METRES = 1.0      # ...within this distance plus...
LINK_SPEED = 4.0            # ...this many metres per second of gap (realistic running speed)
CARRY_SECONDS = 1.0       # a broken track piece keeps the number if it continues within this time...
CARRY_DISTANCE = 1.0        # ...and within this many player-heights of where the numbered piece was
OVERLAP_FRAMES = 5         # same-team tracks sharing a number may not overlap for more than this


def load_tracks(out_dir):
    by_frame, team_of = defaultdict(list), {}
    with open(os.path.join(out_dir, "tracks.csv"), newline="") as f:
        for r in csv.DictReader(f):
            track, team = int(r["track"]), int(r["team"])
            team_of[track] = team
            box = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
            by_frame[int(r["frame"])].append((track, team, box))
    return by_frame, team_of


def owner_of(number_box, players):
    """The single team player whose upper body contains this number box, or None if unclear."""
    nx1, ny1, nx2, ny2 = number_box
    cx, cy = (nx1 + nx2) / 2, (ny1 + ny2) / 2
    area = max(1.0, (nx2 - nx1) * (ny2 - ny1))
    owners = []
    for track, team, (x1, y1, x2, y2) in players:
        if team not in (0, 1):
            continue
        inside = max(0, min(nx2, x2) - max(nx1, x1)) * max(0, min(ny2, y2) - max(ny1, y1)) / area
        if inside >= 0.8 and x1 <= cx <= x2 and y1 <= cy <= y1 + 0.65 * (y2 - y1):
            owners.append(track)
    return owners[0] if len(owners) == 1 else None


class NumberReader:
    def __init__(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(READER_MODEL)
        self.proc.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(READER_MODEL, dtype=torch.bfloat16).to("cuda").eval()
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": READER_PROMPT}]}]
        self.text = self.proc.apply_chat_template(messages, add_generation_prompt=True)

    def __call__(self, crops):
        from PIL import Image
        images = []
        for crop in crops:
            scale = READ_SIZE / crop.shape[0]
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            images.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        inputs = self.proc(text=[self.text] * len(images), images=images, return_tensors="pt", padding=True).to("cuda")
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=4, do_sample=False, output_scores=True, return_dict_in_generate=True)
        new = out.sequences[:, inputs["input_ids"].shape[1]:]
        answers = self.proc.batch_decode(new, skip_special_tokens=True)
        # how sure the reader is: probability of the answer it gave (product over its tokens, until the end token)
        probs = self.torch.stack([sc.float().softmax(-1) for sc in out.scores], 1)      # batch x steps x vocab
        picked = probs.gather(2, new[:, :probs.shape[1], None]).squeeze(2)
        special = set(self.proc.tokenizer.all_special_ids)
        result = []
        for i, a in enumerate(answers):
            conf = 1.0
            for step, tok in enumerate(new[i].tolist()[:probs.shape[1]]):
                if tok in special:
                    break
                conf *= float(picked[i, step])
            a = a.strip()
            result.append((a if re.fullmatch(r"\d{1,2}", a) else None, round(conf, 3)))
        return result


def collect_reads(meta, by_frame, every):
    from PIL import Image
    from rfdetr import RFDETRNano
    detector = RFDETRNano(pretrain_weights=NUMBER_MODEL, num_classes=5)
    reader = NumberReader()
    reads = defaultdict(list)          # track -> [(frame, number or None, reader confidence)]
    pending = []                       # (track, frame, crop)

    def flush():
        for (track, frame, _), (number, conf) in zip(pending, reader([p[2] for p in pending])):
            reads[track].append((frame, number, conf))
        pending.clear()

    cap = cv2.VideoCapture(meta["video"])
    frames = [i for i in sorted(by_frame)]
    keep = set(frames[::every])
    for n, (idx, frame) in enumerate(sampled_frames(cap, meta["start"], meta["end"], meta["fps"])):
        if idx not in keep:
            continue
        players = by_frame[idx]
        wanted = [t for t, team, _ in players if team in (0, 1)
                  and sum(r[1] is not None and r[2] >= MIN_READ_CONF for r in reads[t]) < MAX_READS_PER_TRACK]
        if not wanted:
            continue
        found = detector.predict(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), threshold=NUMBER_CONF)
        for box, cls in zip(found.xyxy, found.class_id):
            if cls != NUMBER_CLASS or box[3] - box[1] < MIN_NUMBER_HEIGHT:
                continue
            track = owner_of(box, players)
            if track not in wanted:
                continue
            x1, y1, x2, y2 = box
            px, py = (x2 - x1) * 0.25, (y2 - y1) * 0.2
            crop = frame[int(max(0, y1 - py)):int(y2 + py), int(max(0, x1 - px)):int(x2 + px)]
            if crop.size:
                pending.append((track, idx, crop))
        if len(pending) >= BATCH:
            flush()
        if n % 100 == 0:
            print(f"  frame {n}/{len(frames)} | number crops read: {sum(len(v) for v in reads.values())}")
    if pending:
        flush()
    cap.release()
    return reads


def decide_numbers(reads, team_of, by_frame):
    # Numbers each team really wears = numbers that clearly win on at least one track.
    # A common misread (36 read as 35) never wins on its own, so it is left out.
    # It must win on at least two separate tracks: one track alone can be a bench player,
    # a player put in the wrong team, or the same misread repeated.
    wins = Counter()
    for track, rs in reads.items():
        votes = Counter(n for _, n in rs if n)
        if votes:
            number, count = votes.most_common(1)[0]
            if count >= ROSTER_MIN_VOTES and count / sum(votes.values()) >= ROSTER_MIN_SHARE:
                wins[(team_of[track], number)] += 1
    roster = {0: set(), 1: set()}
    for (team, number), n_tracks in wins.items():
        if n_tracks >= ROSTER_MIN_TRACKS:
            roster[team].add(number)

    n_split = split_swapped_tracks(reads, team_of, by_frame, roster)

    def winner(votes, total):
        top = votes.most_common(2)
        if not top:
            return None
        number, count = top[0]
        runner_up = top[1][1] if len(top) > 1 else 0
        if count >= MIN_VOTES and count >= MARGIN * runner_up and count / total >= MIN_SHARE:
            return number, count
        return None

    players, flipped = {}, 0
    for track, rs in reads.items():
        team = team_of[track]
        digits = [n for _, n in rs if n]
        entry = {"team": team, "reads": len(rs), "digit_reads": len(digits), "votes": dict(Counter(digits)),
                 "number": None, "status": "unknown"}
        own = winner(Counter(n for n in digits if n in roster[team]), len(digits) or 1)
        # Jersey colour put the player in the wrong team: the reads keep showing a number only the other team wears
        other = winner(Counter(n for n in digits if n in roster[1 - team] and n not in roster[team]), len(digits) or 1)
        if own:
            entry.update(number=own[0], status="confirmed", agree=own[1])
        elif other:
            entry.update(number=other[0], status="confirmed", agree=other[1], team=1 - team, team_fixed=True)
            team_of[track] = 1 - team
            flipped += 1
        players[track] = entry
    if flipped:
        for idx, rows in by_frame.items():
            for i, (t, team, box) in enumerate(rows):
                if t in players and team in (0, 1):
                    rows[i] = (t, players[t]["team"], box)

    # Same team + same number on screen together: keep the track with more agreeing reads
    frames_of = defaultdict(set)
    for idx, rows in by_frame.items():
        for track, _, _ in rows:
            frames_of[track].add(idx)
    groups = defaultdict(list)
    for track, p in players.items():
        if p["number"]:
            groups[(p["team"], p["number"])].append(track)
    dropped = 0
    for tracks in groups.values():
        for a in sorted(tracks, key=lambda t: -players[t]["agree"]):
            if players[a]["number"] is None:
                continue
            for b in tracks:
                if b != a and players[b]["number"] and len(frames_of[a] & frames_of[b]) > OVERLAP_FRAMES \
                        and players[b]["agree"] <= players[a]["agree"]:
                    players[b].update(number=None, status="conflict")
                    dropped += 1
    carry_numbers(players, team_of, by_frame, frames_of)
    for _ in range(3 if USE_LINEUP_RULE else 0):
        if not assign_by_lineup(players, team_of, frames_of):
            break
        carry_numbers(players, team_of, by_frame, frames_of)
    return players, {t: sorted(r, key=int) for t, r in roster.items()}, dropped, n_split, flipped


def assign_by_lineup(players, team_of, frames_of):
    """Process of elimination: of the numbers this track was read as, only one is in the current lineup and not already on screen."""
    window = round(LINEUP_SECONDS * 60)
    span = {t: (min(f), max(f)) for t, f in frames_of.items() if team_of.get(t) in (0, 1)}
    numbered = {t: p for t, p in players.items() if p["number"] and t in span}
    changes = {}
    for t, (start, end) in span.items():
        p = players.get(t)
        # Only tracks whose own reads already point at a number; junk tracks (referees, the basket stand) have none
        if not p or p["number"] or len(frames_of[t]) < LINEUP_MIN_FRAMES or p["digit_reads"] < 2:
            continue
        team = team_of[t]
        lineup = {q["number"] for o, q in numbered.items() if team_of[o] == team and q["status"] != "lineup"
                  and span[o][1] >= start - window and span[o][0] <= end + window}
        on_screen = {q["number"] for o, q in numbered.items() if team_of[o] == team
                     and len(frames_of[o] & frames_of[t]) > OVERLAP_FRAMES}
        free = (lineup - on_screen) & set(p["votes"])
        if len(lineup) <= 6 and len(free) == 1:
            changes[t] = free.pop()
    # Two unknown tracks on screen together can't both take the same number
    for t, number in list(changes.items()):
        if any(o != t and changes.get(o) == number and len(frames_of[o] & frames_of[t]) > OVERLAP_FRAMES for o in changes):
            changes.pop(t)
    for t, number in changes.items():
        players.setdefault(t, {"team": team_of[t], "reads": 0, "digit_reads": 0, "votes": {}})
        players[t].update(number=number, status="lineup", agree=0)
    return bool(changes)


def split_swapped_tracks(reads, team_of, by_frame, roster):
    """When two players cross, the tracker can swap them: a track reads 7,7,7 then 36,36,36.
    Cut such a track where the number changes, so each piece gets its own number."""
    known = roster[0] | roster[1]
    next_id = max(team_of) + 1000
    splits = {}
    for track in list(reads):
        seq = [(f, n) for f, n in sorted(reads[track], key=lambda r: r[0]) if n in known]
        # Ignore single misreads that differ from both neighbours
        seq = [seq[i] for i in range(len(seq))
               if not (0 < i < len(seq) - 1 and seq[i - 1][1] == seq[i + 1][1] != seq[i][1])]
        runs = []
        for f, n in seq:
            if runs and runs[-1][0] == n:
                runs[-1][2], runs[-1][3] = f, runs[-1][3] + 1
            else:
                runs.append([n, f, f, 1])
        merged = []
        for r in (r for r in runs if r[3] >= SPLIT_MIN_READS):
            if merged and merged[-1][0] == r[0]:
                merged[-1][2], merged[-1][3] = r[2], merged[-1][3] + r[3]
            else:
                merged.append(list(r))
        if len(merged) < 2:
            continue
        bounds = [(merged[i][2] + merged[i + 1][1]) // 2 for i in range(len(merged) - 1)]
        ids = [track] + list(range(next_id, next_id + len(bounds)))
        next_id += len(bounds)
        splits[track] = (bounds, ids)

    for idx, rows in by_frame.items():
        for i, (t, team, box) in enumerate(rows):
            if t in splits:
                bounds, ids = splits[t]
                rows[i] = (ids[sum(idx > b for b in bounds)], team, box)
    for t, (bounds, ids) in splits.items():
        rs = reads.pop(t)
        for k, new_id in enumerate(ids):
            lo = bounds[k - 1] if k else -1
            hi = bounds[k] if k < len(bounds) else float("inf")
            team_of[new_id] = team_of[t]
            reads[new_id] = [r for r in rs if lo < r[0] <= hi]
    return len(splits)


def carry_numbers(players, team_of, by_frame, frames_of):
    """When a track breaks, the piece that continues at the same spot within a second keeps the number."""
    box_at = {(t, idx): box for idx, rows in by_frame.items() for t, _, box in rows}
    span = {t: (min(f), max(f)) for t, f in frames_of.items() if team_of.get(t) in (0, 1)}
    max_gap = round(CARRY_SECONDS * 60)      # source frames (60 fps video)

    def center(box):
        return (box[0] + box[2]) / 2, box[3], box[3] - box[1]

    def number_on_screen(team, number, frames):
        return any(p["number"] == number and p["team"] == team and len(frames_of[t] & frames) > OVERLAP_FRAMES
                   for t, p in players.items())

    changed = True
    while changed:
        changed = False
        for t in span:
            p = players.setdefault(t, {"team": team_of[t], "reads": 0, "digit_reads": 0, "votes": {},
                                       "number": None, "status": "unknown"})
            if p["number"]:
                continue
            start, end = span[t]
            candidates = []
            for o, (o_start, o_end) in span.items():
                q = players.get(o)
                if o == t or not q or not q["number"] or team_of[o] != team_of[t]:
                    continue
                if 0 < start - o_end <= max_gap:        # o ended just before t started
                    a, b = box_at[(o, o_end)], box_at[(t, start)]
                elif 0 < o_start - end <= max_gap:      # o starts just after t ended
                    a, b = box_at[(o, o_start)], box_at[(t, end)]
                else:
                    continue
                (ax, ay, ah), (bx, by, bh) = center(a), center(b)
                if np.hypot(ax - bx, ay - by) <= CARRY_DISTANCE * max(ah, bh):
                    candidates.append(o)
            numbers = {players[o]["number"] for o in candidates}
            if len(numbers) == 1:
                number = numbers.pop()
                if not number_on_screen(team_of[t], number, frames_of[t]):
                    p.update(number=number, status="carried", agree=0)
                    changed = True


def draw_badge(frame, center, text, fill, text_color, radius=17):
    cx, cy = center
    cv2.circle(frame, (cx, cy + 2), radius, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), radius, fill, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 2, cv2.LINE_AA)
    scale = 0.62 if len(text) < 2 else 0.55
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 1)
    cv2.putText(frame, text, (cx - w // 2, cy + h // 2), cv2.FONT_HERSHEY_DUPLEX, scale, text_color, 1, cv2.LINE_AA)


def render(meta, by_frame, players, team_colors, team_names, out_dir):
    cap = cv2.VideoCapture(meta["video"])
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, round(meta["source_fps"] / meta["fps"]))
    path = os.path.join(out_dir, "annotated_numbers.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), meta["source_fps"] / step, (width, height))
    for idx, frame in sampled_frames(cap, meta["start"], meta["end"], meta["fps"]):
        if idx not in by_frame:
            continue
        for track, team, (x1, y1, x2, y2) in by_frame[idx]:
            if team not in (0, 1):
                continue
            fill = tuple(int(c) for c in team_colors[team])
            text_color = (20, 20, 20) if sum(fill) > 380 else (255, 255, 255)
            cx = int((x1 + x2) / 2)
            # Team-coloured ellipse at the feet, badge above the head
            cv2.ellipse(frame, (cx, int(y2)), (int((x2 - x1) * 0.55), int((x2 - x1) * 0.18)), 0, -30, 210, fill, 2, cv2.LINE_AA)
            p = players.get(track)
            number = p["number"] if p else None
            draw_badge(frame, (cx, int(y1) - 22), number or "?", fill if number else (90, 90, 90),
                       text_color if number else (230, 230, 230))
        cv2.rectangle(frame, (20, 20), (330, 62), (25, 25, 25), -1)
        cv2.putText(frame, f"{team_names[0]} vs {team_names[1]}  |  jersey IDs", (32, 48),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    cap.release()
    return path


def court_link(out_dir, teams):
    """Court positions (metres) are not affected by camera motion, so a broken track can be followed on the court:
    a '?' piece that starts where a numbered team-mate's piece ended (reachable at sprint speed within 3 s), or ends
    where one starts, gets that number - only if exactly one number fits and that number is not on screen already."""
    meta = json.load(open(os.path.join(out_dir, "meta.json")))
    sfps = meta["source_fps"]
    rows = list(csv.DictReader(open(os.path.join(out_dir, "tracks_named.csv"), newline="")))
    pos = {}
    with open(os.path.join(out_dir, "positions.csv"), newline="") as f:
        for r in csv.DictReader(f):
            pos[(int(r["frame"]), int(r["track"]))] = (float(r["x_m"]), float(r["y_m"]), r["off_court"] == "1")
    frames_of, team_of, number_of = defaultdict(set), {}, {}
    for r in rows:
        t = int(r["track"])
        frames_of[t].add(int(r["frame"]))
        team_of[t] = int(r["team"])
        if r["number"]:
            number_of[t] = r["number"]
    tracks = [t for t in frames_of if team_of[t] in (0, 1)]
    ends = {}
    for t in tracks:
        mapped = sorted(f for f in frames_of[t] if (f, t) in pos and not pos[(f, t)][2])
        if mapped:
            ends[t] = (mapped[0], pos[(mapped[0], t)][:2], mapped[-1], pos[(mapped[-1], t)][:2])

    def on_screen(team, number, frames):
        return any(number_of.get(o) == number and team_of[o] == team and len(frames_of[o] & frames) > OVERLAP_FRAMES
                   for o in tracks if o in number_of)

    data = json.load(open(os.path.join(out_dir, "players.json")))
    roster = [set(data["roster_seen"][teams[0]]), set(data["roster_seen"][teams[1]])]
    votes = {int(k): Counter(v.get("votes") or {}) for k, v in data["tracks"].items()}
    team_fixed = set()
    window = LINEUP_SECONDS * sfps
    linked = by_team_fix = by_lineup = 0
    changed = True
    while changed:
        changed = False
        for t in tracks:
            if t in number_of or t not in ends:
                continue
            on_court_frames = sum(1 for f in frames_of[t] if (f, t) in pos and not pos[(f, t)][2])
            # Rule 1: the reads point to a number only the other team wears, and that player is not on screen
            top = votes.get(t, Counter()).most_common(1)
            if top and top[0][1] >= MIN_VOTES:
                n, other = top[0][0], 1 - team_of[t]
                if n in roster[other] and not on_screen(other, n, frames_of[t]) and \
                        (n not in roster[team_of[t]] or on_screen(team_of[t], n, frames_of[t])):
                    team_of[t], number_of[t] = other, n
                    team_fixed.add(t)
                    by_team_fix += 1
                    changed = True
                    continue
            # Rule 2: process of elimination among the team's current lineup
            if on_court_frames >= LINEUP_MIN_FRAMES * 2:
                s, e = min(frames_of[t]), max(frames_of[t])
                lineup = {number_of[o] for o in tracks if o in number_of and team_of[o] == team_of[t]
                          and max(frames_of[o]) >= s - window and min(frames_of[o]) <= e + window}
                visible = {number_of[o] for o in tracks if o in number_of and team_of[o] == team_of[t]
                           and len(frames_of[o] & frames_of[t]) > OVERLAP_FRAMES}
                free = lineup - visible
                # The track's own reads must back the answer: referees and wrong-team tracks have no such reads
                own = votes.get(t, Counter())
                backed = {n for n in free if own[n] >= 1 and own[n] == max(own.values())}
                if len(lineup) <= 5 and len(free) == 1 and backed == free:
                    number_of[t] = free.pop()
                    by_lineup += 1
                    changed = True
                    continue
            start, p_start, end, p_end = ends[t]
            fits = defaultdict(list)
            for o in tracks:
                if o not in number_of or o not in ends or team_of[o] != team_of[t]:
                    continue
                o_start, o_p_start, o_end, o_p_end = ends[o]
                if 0 < start - o_end <= LINK_SECONDS * sfps:          # o ended, t continues
                    gap, d = (start - o_end) / sfps, np.hypot(p_start[0] - o_p_end[0], p_start[1] - o_p_end[1])
                elif 0 < o_start - end <= LINK_SECONDS * sfps:        # t ended, o continues
                    gap, d = (o_start - end) / sfps, np.hypot(p_end[0] - o_p_start[0], p_end[1] - o_p_start[1])
                else:
                    continue
                if d <= LINK_BASE_METRES + LINK_SPEED * gap:
                    fits[number_of[o]].append(d)
            if len(fits) == 1:
                number = next(iter(fits))
                # Refuse when the track's own reads clearly say another number (a swapped track)
                own = votes.get(t, Counter()).most_common(1)
                if own and own[0][1] >= MIN_VOTES and own[0][0] != number:
                    continue
                if not on_screen(team_of[t], number, frames_of[t]):
                    number_of[t] = number
                    linked += 1
                    changed = True

    players_path = os.path.join(out_dir, "players.json")
    for t, number in number_of.items():
        entry = data["tracks"].setdefault(str(t), {"team": team_of[t], "reads": 0, "digit_reads": 0, "votes": {}})
        if not entry.get("number"):
            entry.update(number=number, team=team_of[t], status="team_fixed" if t in team_fixed else "court_linked")
    json.dump(data, open(players_path, "w"), indent=1)
    for r in rows:
        t = int(r["track"])
        if not r["number"] and t in number_of and team_of[t] in (0, 1):
            r["team"] = str(team_of[t])
            r["number"], r["player"] = number_of[t], f"{teams[team_of[t]]} #{number_of[t]}"
    with open(os.path.join(out_dir, "tracks_named.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    on_court = [(int(r["frame"]), int(r["track"])) for r in rows if r["team"] in ("0", "1")
                and (int(r["frame"]), int(r["track"])) in pos and not pos[(int(r["frame"]), int(r["track"]))][2]]
    named = sum(1 for f, t in on_court if t in number_of)
    print(f"Linked on the court: {linked} | team fixed by number: {by_team_fix} | by lineup elimination: {by_lineup}")
    print(f"On-court player boxes with a number: "
          f"{100 * named / max(1, len(on_court)):.0f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="Run name used for detect/track")
    parser.add_argument("--teams", nargs=2, default=["Team A", "Team B"], help="Names for Team A (lighter jersey) and Team B")
    parser.add_argument("--every", type=int, default=2, help="Look for numbers on every Nth processed frame")
    parser.add_argument("--reuse", action="store_true", help="Reuse saved reads, only redo voting and video")
    parser.add_argument("--no-video", action="store_true", help="Skip annotated_numbers.mp4 (faster, for experiments)")
    parser.add_argument("--court-link", action="store_true",
                        help="After court_mapping.py apply: give '?' track pieces the number of the player they continue on the court")
    args = parser.parse_args()

    out_dir = run_dir(args.name)
    if args.court_link:
        court_link(out_dir, args.teams)
        return
    meta = json.load(open(os.path.join(out_dir, "meta.json")))
    team_colors = json.load(open(os.path.join(out_dir, "team_colors.json")))
    by_frame, team_of = load_tracks(out_dir)

    reads_path = os.path.join(out_dir, "number_reads.json")
    if args.reuse and os.path.exists(reads_path):
        reads = {int(k): [tuple(r) for r in v] for k, v in json.load(open(reads_path)).items()}
    else:
        reads = collect_reads(meta, by_frame, args.every)
        json.dump({str(k): v for k, v in reads.items()}, open(reads_path, "w"))
    # unsure reads count as "no number"; voting below sees (frame, number) pairs as before
    reads = {t: [(r[0], r[1] if len(r) < 3 or r[2] >= MIN_READ_CONF else None) for r in rs] for t, rs in reads.items()}

    players, roster, dropped, n_split, flipped = decide_numbers(reads, team_of, by_frame)
    with open(os.path.join(out_dir, "players.json"), "w") as f:
        json.dump({"teams": args.teams, "roster_seen": {args.teams[t]: r for t, r in roster.items()},
                   "tracks": {str(k): v for k, v in players.items()}}, f, indent=1)
    with open(os.path.join(out_dir, "tracks.csv"), newline="") as f, \
            open(os.path.join(out_dir, "tracks_named.csv"), "w", newline="") as g:
        # Track ids include the pieces of split tracks, and teams include fixes made from the numbers
        writer = csv.writer(g)
        writer.writerow(["frame", "seconds", "track", "team", "x1", "y1", "x2", "y2", "number", "player"])
        for idx in sorted(by_frame):
            for track, team, box in by_frame[idx]:
                number = (players.get(track) or {}).get("number")
                writer.writerow([idx, round(idx / meta["source_fps"], 3), track, team, *[round(v, 1) for v in box],
                                 number or "", f"{args.teams[team]} #{number}" if number and team in (0, 1) else ""])
    video = "no video (--no-video)" if args.no_video else render(meta, by_frame, players, team_colors, args.teams, out_dir)

    # Summary: how much of the on-court time has a confirmed number
    team_rows = [(t, idx) for idx, rows in by_frame.items() for t, team, _ in rows if team in (0, 1)]
    named = sum(1 for t, _ in team_rows if (players.get(t) or {}).get("number"))
    status = Counter(p["status"] for p in players.values())
    team_tracks = len({t for rows in by_frame.values() for t, team, _ in rows if team in (0, 1)})
    print(f"Numbers each team wears (seen repeatedly): {args.teams[0]} {roster[0]} | {args.teams[1]} {roster[1]}")
    print(f"Team tracks: {team_tracks} | read: {status['confirmed']} | carried: {status['carried']} | "
          f"by lineup: {status['lineup']} | unknown: {team_tracks - status['confirmed'] - status['carried'] - status['lineup']}"
          f" | dropped as duplicates: {dropped}")
    print(f"Tracks split at a player swap: {n_split} | tracks moved to the other team by their number: {flipped}")
    print(f"Player boxes with a confirmed number: {100 * named / max(1, len(team_rows)):.0f}%")
    print(f"Saved {video}, players.json and tracks_named.csv")


if __name__ == "__main__":
    main()
