"""
Pack everything the cloud GPU machine needs into one zip: ClutchMetrics_Final/clutchmetrics_cloud.zip
(code, the two trained models, the cloud scripts, and the finished analyses that belong to an account).
Upload it to the machine's /home/clutchmetrics and unzip it there.

    ..\\env\\Scripts\\python.exe cloud\\pack.py      (from the model folder)
"""
import json
import os
import zipfile

MODEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(MODEL), "clutchmetrics_cloud.zip")
CODE = ["server.py", "basketball_pipeline.py", "compare_detectors.py", "jersey_numbers.py", "court_mapping.py",
        "court_dataset.py", "match_events.py", "render_broadcast.py", "export_dashboard.py", "README.md"]
WEIGHTS = ["weights/rfdetr_basketball_v2/best", "weights/rfdetr_5class/checkpoint_best_total.pth"]


def add(z, local, arc, text=False):
    if text:   # shell scripts need Linux line endings
        z.writestr(arc, open(local, "rb").read().replace(b"\r\n", b"\n"))
    else:
        z.write(local, arc)


with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for f in CODE:
        add(z, os.path.join(MODEL, f), f"model/{f}")
    for f in os.listdir(os.path.join(MODEL, "cloud")):
        if f.endswith((".sh", ".txt", ".py")):
            add(z, os.path.join(MODEL, "cloud", f), f"model/cloud/{f}", text=f.endswith(".sh"))
    for w in WEIGHTS:
        path = os.path.join(MODEL, w)
        if os.path.isdir(path):
            for folder, _, files in os.walk(path):
                for name in files:
                    local = os.path.join(folder, name)
                    add(z, local, "model/" + os.path.relpath(local, MODEL).replace("\\", "/"))
        else:
            add(z, path, f"model/{w}")
    jobs = os.path.join(MODEL, "server_jobs")
    n = 0
    for job_id in sorted(os.listdir(jobs)):
        path = os.path.join(jobs, job_id, "job.json")
        if not os.path.exists(path):
            continue
        job = json.load(open(path, encoding="utf-8"))
        if job.get("status") != "done" or not job.get("owner"):
            continue
        for folder, _, files in os.walk(os.path.join(jobs, job_id)):
            for name in files:
                if name.startswith("upload") or name in ("input.mp4", "log.txt"):
                    continue
                local = os.path.join(folder, name)
                add(z, local, "model/" + os.path.relpath(local, MODEL).replace("\\", "/"))
        n += 1
print(f"{OUT}: {os.path.getsize(OUT) / 1e6:.0f} MB ({n} analyses included)")
