"""
ClutchMetrics analysis server: the website uploads a basketball video, this server converts it and runs the
whole model on this computer's GPU, then serves the results to the dashboards.

Start it (from the ClutchMetrics_Final/model folder):
    python server.py
or double-click start_server.bat, which also opens the public ngrok address.

API (all under /api):
  GET  /health                 server is up, how many videos are waiting
  POST /jobs                   upload: form fields file, team_a, team_b, title, competition, date, venue, role
  GET  /jobs                   all analyses, newest first
  GET  /jobs/{id}              status and progress of one analysis
  GET  /jobs/{id}/match.json   results for the dashboards (same format as the website's data/match.json)
  GET  /jobs/{id}/analysis.mp4 the analysis video (supports seeking)
  POST /jobs/{id}/me           a player says which player they are: {"team": 0/1, "no": "7"}
  POST /jobs/{id}/cancel       stop a waiting or running analysis (deletes its videos; it leaves the list)
  GET  /jobs/{id}/court        court mapping: the chosen pictures, court landmarks and camera data
  GET  /jobs/{id}/court/{n}.jpg one picture to map
  POST /jobs/{id}/court        the user's mapping {"keyframes": {frame: 3x3 image->court matrix | "skip"}} or {"skip": true}

Court mapping: right after converting, a few wide court views are picked (court_mapping.py prepare). The user maps
them in the website while the GPU keeps working; the analysis waits for the mapping only when it reaches the court
step (at most COURT_WAIT_MINUTES), then continues without court stats if nobody mapped it.

Videos are analysed one at a time (one GPU). A 2-minute clip takes about 20-30 minutes.
"""

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.environ.get("CM_JOBS_DIR", os.path.join(HERE, "server_jobs"))
MAX_JOBS_PER_DAY = int(os.environ.get("MAX_JOBS_PER_DAY", 0))
# Firebase project ID (from the website's firebase-config.js). When set, only signed-in website users can upload,
# and each user sees only their own analyses. Empty = open server (as before).
FIREBASE_PROJECT = os.environ.get("CM_FIREBASE_PROJECT", "")    # 0 = no limit (laptop); set on Modal to protect credits
PORT = int(os.environ.get("PORT", 8000))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", 4096))
MAX_SECONDS = int(os.environ.get("MAX_SECONDS", 0))      # 0 = analyse the whole video; a number cuts longer videos
VIDEO_TYPES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mpg", ".mpeg", ".wmv", ".3gp"}
# Automatic court mapping is OFF until the court model passes a test on a game it never saw (court_mapping.py auto).
# Without it, uploads get no speed, distance, heatmaps or shot positions (the dashboard says so).
AUTO_COURT = os.environ.get("AUTO_COURT") == "1"
COURT_PICTURES = int(os.environ.get("COURT_PICTURES", 12))           # pictures to map for a short clip (3 min)
COURT_PICTURES_MAX = int(os.environ.get("COURT_PICTURES_MAX", 20))   # longer videos get more, up to this
COURT_WAIT_MINUTES = float(os.environ.get("COURT_WAIT_MINUTES", 15))  # wait this long for the mapping at the court step
CLEANUP = os.environ.get("CM_CLEANUP") == "1"   # small cloud disk: keep only the results of finished analyses
ACTIVITY_FILE = os.environ.get("CM_ACTIVITY_FILE", "")   # cloud machine: time of the last website request
ALLOWED_ORIGINS = ["https://praveenregar30.github.io"]
ALLOWED_ORIGIN_REGEX = r"http://(localhost|127\.0\.0\.1)(:\d+)?"

# (step id, text shown on the website, share of the total time)
STEPS = [
    ("convert", "Converting the video", 4),
    ("prepare", "Choosing pictures for court mapping", 2),
    ("detect", "Finding players and the ball", 14),
    ("track", "Tracking players and teams", 9),
    ("ball", "Following the ball", 6),
    ("numbers", "Reading jersey numbers", 24),
    ("court", "Mapping the court", 3),
    ("link", "Joining player tracks on the court", 2),
    ("events", "Finding passes, shots and possession", 8),
    ("render", "Drawing the analysis video", 15),
    ("export", "Preparing the dashboard", 8),
]
TOTAL_WEIGHT = sum(w for _, _, w in STEPS)

app = FastAPI(title="ClutchMetrics analysis server")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_origin_regex=ALLOWED_ORIGIN_REGEX,
                   allow_methods=["GET", "POST"], allow_headers=["*"])


@app.middleware("http")
async def private_network(request: Request, call_next):
    # Chrome asks before a public website talks to a server on this computer (localhost)
    response = await call_next(request)
    if request.headers.get("access-control-request-private-network"):
        response.headers["Access-Control-Allow-Private-Network"] = "true"
    note_activity()
    return response


_last_note = [0.0]


def note_activity():
    """Cloud machine: record when the website last used this server (cloud/idle_pause.py pauses it after a quiet hour)."""
    if ACTIVITY_FILE and time.time() - _last_note[0] > 30:
        _last_note[0] = time.time()
        try:
            with open(ACTIVITY_FILE, "w") as f:
                f.write(str(_last_note[0]))
        except OSError:
            pass


work = queue.Queue()
lock = threading.Lock()

# Hooks so the same server runs on the laptop (in-process queue) and on Modal (GPU function + shared volume)
hooks = {
    "submit": lambda job_id: work.put(job_id),   # start an analysis
    "refresh": lambda: None,                       # see the newest files (Modal: volume.reload)
    "saved": lambda: None,                         # publish written files (Modal: volume.commit)
    # Small messages between the website and a running analysis. On the laptop they share one disk, so files do;
    # on Modal the GPU container cannot see new files while it has its own open, so a shared store is used.
    "message_put": lambda key, value: None,
    "message_get": lambda key: None,
    # Stop an analysis that runs somewhere else (Modal: cancel the GPU call). True when it was stopped that way,
    # so the website side marks it cancelled itself (the stopped analysis cannot report back).
    "stop": lambda job_id: False,
}


# ---------------------------------------------------------------- job records
def job_dir(job_id):
    if not re.fullmatch(r"[a-f0-9]{12}", job_id):
        raise HTTPException(404, "No such analysis")
    return os.path.join(JOBS_DIR, job_id)


def read_job(job_id, refresh=True):
    if refresh:
        hooks["refresh"]()
    path = os.path.join(job_dir(job_id), "job.json")
    if not os.path.exists(path):
        raise HTTPException(404, "No such analysis")
    with open(path, encoding="utf-8") as f:
        job = json.load(f)
    if job.get("removed"):
        raise HTTPException(404, "No such analysis")
    return job


def write_job(job):
    with lock:
        path = os.path.join(JOBS_DIR, job["id"], "job.json")
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(job, f, indent=1)
        os.replace(path + ".tmp", path)
    hooks["saved"]()


def public(job):
    keys = ("id", "status", "step", "step_text", "progress", "message", "title", "teams", "role",
            "created", "finished", "duration_s", "queue_position", "court_mapped", "court_pictures")
    out = {k: job.get(k) for k in keys}
    out["court"] = court_state(job)
    out["me"] = read_me(job["id"])
    out["court_can_map"] = out["court"] == "to_map" and can_map(job)
    waiting = list(work.queue)
    out["queue_position"] = waiting.index(job["id"]) + 1 if job["status"] == "queued" and job["id"] in waiting else None
    return out


def clean(text, limit):
    return re.sub(r"[^\w .,'&()-]", "", str(text or ""), flags=re.UNICODE).strip()[:limit]


def team_code(name):
    letters = re.sub(r"[^A-Za-z]", "", name).upper()
    return (letters[:3] or "TEAM")


def player_me(team, number):
    """A player's own team (0 = team A, 1 = team B; also 'a'/'b') and jersey number, or None when not valid."""
    team = {"a": 0, "b": 1, "0": 0, "1": 1}.get(str(team).strip().lower())
    number = str(number).strip().lstrip("#")
    if team is None or not re.fullmatch(r"\d{1,2}", number):
        return None
    return {"team": team, "no": str(int(number))}


def read_me(job_id):
    # kept in its own file: the analysis worker rewrites job.json while it runs
    path = os.path.join(job_dir(job_id), "me.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_me(job_id, me):
    path = os.path.join(job_dir(job_id), "me.json")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(me, f)
    os.replace(path + ".tmp", path)
    hooks["saved"]()


def court_answer(job_id):
    path = os.path.join(job_dir(job_id), "court", "answer.json")
    if not os.path.exists(path):
        return hooks["message_get"](f"court:{job_id}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def court_state(job):
    """none (no pictures: too early, or no wide court views), to_map, mapped, skipped."""
    answer = court_answer(job["id"])
    if answer:
        return "skipped" if answer.get("skip") else "mapped"
    return "to_map" if job.get("court_pictures") else "none"


def can_map(job):
    """The mapping is still useful: the analysis has not passed the court step."""
    early = [s for s, _, _ in STEPS[:STEPS.index(next(x for x in STEPS if x[0] == "court"))]]
    return job["status"] in ("queued", "running") and job.get("step") in (*early, "queued", "court_wait")


class Cancelled(Exception):
    """The user cancelled the analysis."""


running = {}    # job id -> the pipeline command running for it now (so a cancel can stop it at once)


def cancel_requested(job_id):
    return os.path.exists(os.path.join(JOBS_DIR, job_id, "cancel")) or bool(hooks["message_get"](f"cancel:{job_id}"))


def check_cancel(job_id):
    if cancel_requested(job_id):
        raise Cancelled()


def remove_files(job_id):
    """A cancelled analysis: delete the videos and the model's working files (keep job.json and the log)."""
    folder = os.path.join(JOBS_DIR, job_id)
    for item in os.listdir(folder):
        if item.startswith("upload") or item in ("input.mp4", "court", "data"):
            path = os.path.join(folder, item)
            shutil.rmtree(path, ignore_errors=True) if os.path.isdir(path) else os.remove(path)
    shutil.rmtree(os.path.join(HERE, "runs", "pipeline", f"upload_{job_id}"), ignore_errors=True)


def remove_working_files(job_id):
    """A finished analysis on a small disk (CM_CLEANUP=1): keep only what the website shows (data/ with match.json and
    analysis.mp4, the court mapping, job.json, me.json, the log); delete both video copies and the model's working files
    (about 0.9 GB per 3-minute video, of which the results are about 30 MB)."""
    folder = os.path.join(JOBS_DIR, job_id)
    for item in os.listdir(folder):
        if item.startswith("upload") or item == "input.mp4":
            os.remove(os.path.join(folder, item))
    shutil.rmtree(os.path.join(HERE, "runs", "pipeline", f"upload_{job_id}"), ignore_errors=True)


def wait_for_court(job):
    """Pause before the court step until the user has mapped (or skipped) the pictures, at most COURT_WAIT_MINUTES."""
    if court_answer(job["id"]) or not job.get("court_pictures"):
        return court_answer(job["id"])
    job.update(step="court_wait", step_text="Waiting for you to map the court")
    write_job(job)
    deadline = time.time() + 60 * COURT_WAIT_MINUTES
    while time.time() < deadline:
        time.sleep(3)
        hooks["refresh"]()
        check_cancel(job["id"])
        answer = court_answer(job["id"])
        if answer:
            return answer
    return None


def court_steps(job, name, codes, log):
    py = sys.executable
    link = ("link", [py, "jersey_numbers.py", "--name", name, "--teams", *codes, "--court-link"])
    if AUTO_COURT:
        return [("court", [py, "court_mapping.py", "auto", "--name", name]),
                ("court", [py, "court_mapping.py", "apply", "--name", name, "--source", "auto"]), link]
    answer = wait_for_court(job)
    if not answer or answer.get("skip"):
        log.write("\nNo court mapping: continuing without court stats.\n")
        return []
    with open(os.path.join(HERE, "runs", "pipeline", name, "court_keyframes.json"), "w", encoding="utf-8") as f:
        json.dump(answer["keyframes"], f)
    return [("court", [py, "court_mapping.py", "apply", "--name", name, "--source", "keyframes", "--carry", "16"]), link]


# ---------------------------------------------------------------- the analysis
def run_step(job, step_index, cmd, log):
    """Run one pipeline command; update progress from 'n/total' counters in its output."""
    step_id, text, weight = STEPS[step_index]
    done_before = sum(w for _, _, w in STEPS[:step_index])
    check_cancel(job["id"])
    job.update(status="running", step=step_id, step_text=text, progress=round(100 * done_before / TOTAL_WEIGHT))
    write_job(job)
    log.write(f"\n$ {' '.join(cmd)}\n")
    log.flush()
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", env=env)
    running[job["id"]] = proc
    last = checked = time.time()
    try:
        for line in proc.stdout:
            log.write(line)
            if time.time() - checked > 3:     # (the cancel endpoint also stops it directly on the laptop)
                checked = time.time()
                if cancel_requested(job["id"]):
                    proc.kill()
                    break
            m = re.search(r"(\d+)\s*/\s*(\d+)", line)
            if m and int(m.group(2)) > 0 and time.time() - last > 2:
                frac = min(int(m.group(1)) / int(m.group(2)), 1.0)
                job["progress"] = round(100 * (done_before + weight * frac) / TOTAL_WEIGHT)
                write_job(job)
                last = time.time()
        code = proc.wait()
    finally:
        running.pop(job["id"], None)
    check_cancel(job["id"])
    if code != 0:
        log.flush()
        raise RuntimeError(f"Step '{text}' failed")


def convert(src, dst, log):
    """Built-in converter: any common video -> H.264 MP4, at most 1080p and 60 fps, constant frame rate, no audio,
    cut to MAX_SECONDS. This is the format every step of the model reads reliably."""
    import imageio_ffmpeg
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    fps = min(fps if 5 < fps < 240 else 30, 60)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", src, *(["-t", str(MAX_SECONDS)] if MAX_SECONDS else []),
           "-vf", "scale='min(1920,iw)':-2", "-r", f"{fps:.3f}", "-fps_mode", "cfr", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", dst]
    log.write(f"$ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True)
    log.write(result.stderr)
    if result.returncode != 0 or not os.path.exists(dst):
        raise RuntimeError("This file could not be read as a video. Try an MP4 or MOV file.")
    cap = cv2.VideoCapture(dst)
    frames, fps_out = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    duration = frames / fps_out if fps_out else 0
    if duration < 5:
        raise RuntimeError("The video is shorter than 5 seconds.")
    return duration


def analyse(job_id):
    job = read_job(job_id)
    folder = job_dir(job_id)
    name = f"upload_{job_id}"
    py = sys.executable
    a, b = job["teams"]
    codes = [team_code(a), team_code(b)]
    if codes[0] == codes[1]:
        codes[1] = codes[1][:2] + "2"
    with open(os.path.join(folder, "log.txt"), "a", encoding="utf-8") as log:
        try:
            check_cancel(job_id)
            job.update(status="running", step="convert", step_text=STEPS[0][1], progress=0, started=time.time())
            write_job(job)
            video = os.path.join(folder, "input.mp4")
            duration = convert(os.path.join(folder, job["upload"]), video, log)
            check_cancel(job_id)
            job["duration_s"] = round(duration, 1)
            end = f"{duration:.2f}"
            # Pictures for court mapping in the website (a failure here only means no court stats)
            if not AUTO_COURT:
                try:
                    run_step(job, 1, [py, "court_mapping.py", "prepare", "--name", name, "--video", video, "--end", end,
                                      "--max-keyframes", str(min(COURT_PICTURES_MAX, COURT_PICTURES + int(max(0.0, duration - 180) // 180))), "--out", os.path.join(folder, "court")], log)
                    with open(os.path.join(folder, "court", "calibration.json"), encoding="utf-8") as f:
                        job["court_pictures"] = len(json.load(f)["keyframes"])
                except Cancelled:
                    raise
                except Exception as err:  # noqa: BLE001
                    log.write(f"\nCourt mapping pictures not available: {err}\n")
                    job["court_pictures"] = 0
                write_job(job)
            steps = [
                ("detect", [py, "basketball_pipeline.py", "detect", "--video", video, "--name", name, "--start", "0", "--end", end, "--fps", "15"]),
                ("track", [py, "basketball_pipeline.py", "track", "--video", video, "--name", name]),
                ("ball", [py, "basketball_pipeline.py", "ball", "--video", video, "--name", name]),
                ("numbers", [py, "jersey_numbers.py", "--name", name, "--teams", *codes]),
                ("court?", None),     # the user's court mapping (or the automatic one)
                ("events", [py, "match_events.py", "--name", name, "--teams", *codes]),
                ("render", [py, "render_broadcast.py", "--name", name]),
                ("export", [py, "export_dashboard.py", "--name", name, "--site", folder, "--team-names", a, b,
                            "--title", job["title"], "--competition", job.get("competition", ""),
                            "--date", job.get("date", ""), "--venue", job.get("venue", "")]),
            ]
            index = {s: i for i, (s, _, _) in enumerate(STEPS)}
            for step_id, cmd in steps:
                if step_id == "court?":
                    court = court_steps(job, name, codes, log)
                    for sid, c in court:
                        run_step(job, index[sid], c, log)
                    job["court_mapped"] = bool(court)
                    continue
                run_step(job, index[step_id], cmd, log)
            job.update(status="done", step="done", step_text="Analysis ready", progress=100, finished=time.time(),
                       message=f"Analysed {duration:.0f} s of video in {(time.time() - job['started']) / 60:.0f} minutes.")
        except Cancelled:
            log.write("\nCANCELLED by the user\n")
            job.update(status="cancelled", message="Cancelled", finished=time.time())
        except Exception as err:  # noqa: BLE001 - any failure is reported to the website
            if cancel_requested(job_id):
                job.update(status="cancelled", message="Cancelled", finished=time.time())
            else:
                log.write(f"\nFAILED: {err}\n")
                job.update(status="failed", message=str(err), finished=time.time())
        write_job(job)
    if job["status"] == "cancelled":
        remove_files(job_id)
    elif job["status"] == "done" and CLEANUP:
        remove_working_files(job_id)


def worker():
    while True:
        job_id = work.get()
        try:
            analyse(job_id)
        finally:
            work.task_done()


# ---------------------------------------------------------------- API
def signed_in_user(request: Request):
    """The caller's Firebase user ID, checked against Google's public keys. None when accounts are switched off."""
    if not FIREBASE_PROJECT:
        return None
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        raise HTTPException(401, "Sign in on the website to use the analysis server.")
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token
    try:
        # cloud machines' clocks can be minutes off (it changes with every resume), which makes fresh logins look
        # "used too early": allow for this machine's measured clock error plus 2 minutes
        claims = id_token.verify_firebase_token(header[7:], google_requests.Request(), audience=FIREBASE_PROJECT,
                                                clock_skew_in_seconds=int(abs(clock_error())) + 120)
    except Exception as err:  # noqa: BLE001 - expired, forged or wrong project: all the same to the caller
        print(f"login check failed: {type(err).__name__}: {err}", file=sys.stderr, flush=True)   # the real reason, in the log
        raise HTTPException(401, "Your sign-in has expired. Sign in again.") from None
    return claims.get("user_id") or claims["sub"]


_clock = {"at": 0.0, "error": 0.0}


def clock_error():
    """Seconds this machine's clock is behind Google's (which issues the logins), measured at most every 10 minutes."""
    if time.time() - _clock["at"] > 600:
        _clock["at"] = time.time()
        try:
            import email.utils
            import requests
            date = requests.head("https://www.googleapis.com/", timeout=5).headers["Date"]
            _clock["error"] = email.utils.parsedate_to_datetime(date).timestamp() - time.time()
            if abs(_clock["error"]) > 30:
                print(f"this machine's clock is {_clock['error']:+.0f} s off Google's; logins allow for it",
                      file=sys.stderr, flush=True)
        except Exception:  # noqa: BLE001 - keep the last measurement
            pass
    return _clock["error"]


def check_owner(job, uid):
    # someone else's analysis looks the same as one that does not exist
    if uid and job.get("owner") != uid:
        raise HTTPException(404, "No such analysis")


def all_jobs():
    hooks["refresh"]()
    jobs = []
    if os.path.isdir(JOBS_DIR):
        for job_id in os.listdir(JOBS_DIR):
            try:
                jobs.append(read_job(job_id, refresh=False))
            except HTTPException:
                continue
    return jobs


@app.get("/api/health")
def health():
    return {"ok": True, "queue": work.qsize(), "max_upload_mb": MAX_UPLOAD_MB, "max_seconds": MAX_SECONDS,
            "formats": sorted(VIDEO_TYPES)}


PARTS_DIR = os.path.join(os.path.dirname(os.path.abspath(JOBS_DIR)), "upload_parts")


def parts_path(token):
    if not re.fullmatch(r"[a-f0-9]{24}", token or ""):
        raise HTTPException(400, "Bad upload token.")
    return os.path.join(PARTS_DIR, token)


@app.post("/api/upload-part")
async def upload_part(request: Request, token: str, offset: int):
    """One piece of a big video, appended in order (offset = bytes already received; a repeated piece is ignored)."""
    uid = signed_in_user(request)
    os.makedirs(PARTS_DIR, exist_ok=True)
    path = parts_path(token)
    owner_file = path + ".owner"
    if offset == 0 and not os.path.exists(path):
        open(owner_file, "w").write(uid or "")
        open(path, "wb").close()
    if not os.path.exists(path) or open(owner_file).read() != (uid or ""):
        raise HTTPException(404, "Upload not found. Start the upload again.")
    have = os.path.getsize(path)
    body = await request.body()
    if offset + len(body) <= have:                      # this piece arrived already (a retry)
        return {"received": have}
    if offset != have:
        raise HTTPException(409, f"Expected the piece at byte {have}.")
    if have + len(body) > MAX_UPLOAD_MB * 1024 * 1024:
        os.remove(path)
        raise HTTPException(413, f"The file is larger than {MAX_UPLOAD_MB} MB.")
    with open(path, "ab") as out:
        out.write(body)
    return {"received": have + len(body)}


@app.post("/api/jobs")
async def create_job(request: Request, file: UploadFile = File(None), team_a: str = Form(...), team_b: str = Form(...),
                     title: str = Form(""), competition: str = Form(""), date: str = Form(""), venue: str = Form(""),
                     role: str = Form("coach"), my_team: str = Form(""), my_number: str = Form(""),
                     parts: str = Form(""), filename: str = Form("")):
    uid = signed_in_user(request)
    if file is None and not parts:
        raise HTTPException(400, "No video was sent.")
    ext = os.path.splitext((file.filename if file is not None else filename) or "")[1].lower()
    if ext not in VIDEO_TYPES:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Use one of: {', '.join(sorted(VIDEO_TYPES))}")
    a, b = clean(team_a, 30), clean(team_b, 30)
    if not a or not b:
        raise HTTPException(400, "Enter both team names.")
    if MAX_JOBS_PER_DAY:
        today = [j for j in all_jobs() if (j.get("created") or 0) > time.time() - 86400]
        if len(today) >= MAX_JOBS_PER_DAY:
            raise HTTPException(429, f"The daily limit of {MAX_JOBS_PER_DAY} analyses has been reached. Try again tomorrow.")
    job_id = uuid.uuid4().hex[:12]
    folder = os.path.join(JOBS_DIR, job_id)
    os.makedirs(folder)
    dest = os.path.join(folder, "upload" + ext)
    size, limit = 0, MAX_UPLOAD_MB * 1024 * 1024
    if file is None:                                    # the video arrived in pieces: move the joined file in
        path = parts_path(parts)
        if not os.path.exists(path) or open(path + ".owner").read() != (uid or ""):
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(404, "The uploaded video was not found. Upload it again.")
        shutil.move(path, dest)
        os.remove(path + ".owner")
        size = os.path.getsize(dest)
    else:
        with open(dest, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    out.close()
                    shutil.rmtree(folder, ignore_errors=True)
                    raise HTTPException(413, f"The file is larger than {MAX_UPLOAD_MB} MB.")
                out.write(chunk)
    job = {"id": job_id, "status": "queued", "step": "queued", "step_text": "Waiting for the GPU", "progress": 0,
           "message": "", "upload": os.path.basename(dest), "size_mb": round(size / 1e6, 1),
           "teams": [a, b], "title": clean(title, 60) or f"{a} vs {b}", "competition": clean(competition, 60),
           "date": clean(date, 10) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date or "") else "",
           "venue": clean(venue, 60), "role": "player" if role == "player" else "coach", "created": time.time(),
           "owner": uid}
    write_job(job)
    me = player_me(my_team, my_number)
    if me and job["role"] == "player":
        save_me(job_id, me)
    hooks["submit"](job_id)
    return public(job)


@app.get("/api/jobs")
def list_jobs(request: Request):
    uid = signed_in_user(request)
    jobs = [public(j) for j in all_jobs() if (not uid or j.get("owner") == uid) and j["status"] != "cancelled"]
    return sorted(jobs, key=lambda j: j["created"] or 0, reverse=True)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, request: Request):
    job = read_job(job_id)
    check_owner(job, signed_in_user(request))
    return public(job)


@app.get("/api/jobs/{job_id}/match.json")
def get_match(job_id: str, request: Request):
    job = read_job(job_id)
    check_owner(job, signed_in_user(request))
    path = os.path.join(job_dir(job_id), "data", "match.json")
    if job["status"] != "done" or not os.path.exists(path):
        raise HTTPException(409, "This analysis is not finished yet.")
    with open(path, encoding="utf-8") as f:
        match = json.load(f)
    match["video"] = f"/api/jobs/{job_id}/analysis.mp4"     # the website puts the server address in front
    match["job_id"] = job_id
    match["me"] = read_me(job_id)
    return JSONResponse(match)


@app.get("/api/jobs/{job_id}/analysis.mp4")
def get_video(job_id: str, request: Request):
    check_owner(read_job(job_id), signed_in_user(request))
    path = os.path.join(job_dir(job_id), "data", "analysis.mp4")
    if not os.path.exists(path):
        raise HTTPException(404, "No video yet")
    return FileResponse(path, media_type="video/mp4")


def owned_job(job_id, request):
    job = read_job(job_id)
    check_owner(job, signed_in_user(request))
    return job


@app.post("/api/jobs/{job_id}/me")
async def set_me(job_id: str, request: Request):
    """A player says which player in this video they are: {"team": 0 or 1, "no": "7"}."""
    owned_job(job_id, request)
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "Expected JSON") from None
    me = player_me(str(body.get("team", "")), str(body.get("no", "")))
    if not me:
        raise HTTPException(400, "Pick your team and a jersey number (0-99).")
    save_me(job_id, me)
    return {"ok": True, "me": me}


@app.post("/api/jobs/{job_id}/remove")
def remove_analysis(job_id: str, request: Request):
    """Delete a finished (or failed) analysis permanently: video, results, court mapping, record and working files."""
    if not re.fullmatch(r"[a-f0-9]{12}", job_id):
        raise HTTPException(404, "No such analysis")
    uid = signed_in_user(request)
    job = read_job(job_id)
    check_owner(job, uid)
    if job["status"] not in ("done", "failed"):
        raise HTTPException(409, "Cancel the active analysis first.")
    shutil.rmtree(os.path.join(HERE, "runs", "pipeline", f"upload_{job_id}"), ignore_errors=True)
    folder = job_dir(job_id)
    # everything except the record first: if a file is locked, the record stays and Remove can simply be retried
    for entry in os.scandir(folder):
        if entry.name != "job.json":
            shutil.rmtree(entry.path) if entry.is_dir(follow_symlinks=False) else os.unlink(entry.path)
    shutil.rmtree(folder)
    hooks["saved"]()
    return {"ok": True}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request):
    """Stop a waiting or running analysis; its videos and working files are deleted and it leaves the list."""
    job = owned_job(job_id, request)
    if job["status"] not in ("queued", "running"):
        raise HTTPException(409, "This analysis has already finished.")
    with open(os.path.join(job_dir(job_id), "cancel"), "w") as f:
        f.write(str(time.time()))
    hooks["saved"]()
    hooks["message_put"](f"cancel:{job_id}", True)
    proc = running.get(job_id)
    if proc:
        proc.kill()                     # the worker notices, marks it cancelled and deletes the files
    elif hooks["stop"](job_id) or job["status"] == "queued":
        # stopped elsewhere (Modal GPU) or not started yet: nobody else will mark it, so do it here
        job.update(status="cancelled", message="Cancelled", finished=time.time())
        write_job(job)                  # the worker skips it when its turn comes
        try:
            remove_files(job_id)
        except OSError:
            pass
    return {"ok": True}


def calibration_path(job_id):
    return os.path.join(job_dir(job_id), "court", "calibration.json")


@app.get("/api/jobs/{job_id}/court")
def get_court(job_id: str, request: Request):
    job = owned_job(job_id, request)
    if not job.get("court_pictures") or not os.path.exists(calibration_path(job_id)):
        raise HTTPException(409, "There are no court pictures to map for this video (yet).")
    with open(calibration_path(job_id), encoding="utf-8") as f:
        data = json.load(f)
    data.update(state=court_state(job), can_map=can_map(job), title=job.get("title"), status=job["status"])
    return data


@app.get("/api/jobs/{job_id}/court/{frame}.jpg")
def get_court_picture(job_id: str, frame: int, request: Request):
    owned_job(job_id, request)
    path = os.path.join(job_dir(job_id), "court", "keyframes", f"{int(frame)}.jpg")
    if not os.path.exists(path):
        raise HTTPException(404, "No such picture")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/jobs/{job_id}/court")
async def save_court(job_id: str, request: Request):
    job = owned_job(job_id, request)
    if not job.get("court_pictures") or not os.path.exists(calibration_path(job_id)):
        raise HTTPException(409, "There are no court pictures to map for this video.")
    if court_answer(job_id):
        raise HTTPException(409, "The court of this video has already been mapped.")
    if not can_map(job):
        raise HTTPException(409, "Too late: the analysis has already passed the court step.")
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "Expected JSON") from None
    if body.get("skip"):
        answer = {"skip": True, "saved": time.time()}
    else:
        with open(calibration_path(job_id), encoding="utf-8") as f:
            frames = {k["frame"] for k in json.load(f)["keyframes"]}
        keyframes = {}
        for key, value in (body.get("keyframes") or {}).items():
            if not str(key).isdigit() or int(key) not in frames:
                raise HTTPException(400, f"Unknown picture {key}")
            if value == "skip":
                keyframes[str(int(key))] = "skip"
                continue
            try:
                matrix = [[float(v) for v in row] for row in value]
            except (TypeError, ValueError):
                raise HTTPException(400, "Each mapping must be a 3x3 matrix") from None
            flat = [v for row in matrix for v in row]
            if len(matrix) != 3 or len(flat) != 9 or not all(abs(v) < 1e9 for v in flat):
                raise HTTPException(400, "Each mapping must be a 3x3 matrix")
            keyframes[str(int(key))] = matrix
        if not any(isinstance(v, list) for v in keyframes.values()):
            raise HTTPException(400, "Map at least one picture, or skip court mapping.")
        answer = {"skip": False, "keyframes": keyframes, "saved": time.time()}
    tmp = os.path.join(job_dir(job_id), "court", "answer.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(answer, f)
    os.replace(tmp, os.path.join(job_dir(job_id), "court", "answer.json"))
    hooks["saved"]()
    hooks["message_put"](f"court:{job_id}", answer)
    return {"ok": True, "court": court_state(job)}


def main():
    import uvicorn
    os.makedirs(JOBS_DIR, exist_ok=True)
    # Resume analyses that were waiting or running when the server stopped
    for job_id in sorted(os.listdir(JOBS_DIR)):
        try:
            job = read_job(job_id)
        except HTTPException:
            continue
        if job["status"] in ("queued", "running"):
            job.update(status="queued", step="queued", step_text="Waiting for the GPU", progress=0)
            write_job(job)
            work.put(job_id)
    threading.Thread(target=worker, daemon=True).start()
    length = f"first {MAX_SECONDS} s analysed" if MAX_SECONDS else "whole video analysed"
    print(f"ClutchMetrics analysis server on http://localhost:{PORT}  (videos up to {MAX_UPLOAD_MB} MB, {length})")
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
