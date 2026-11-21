#!/usr/bin/env bash
# Pack every finished analysis (the results the website shows) into one zip to download to the laptop.
#   bash /home/clutchmetrics/model/cloud/backup.sh
# Then in the Jupyter file browser, right-click /home/clutchmetrics/backup_<date>.zip -> Download,
# and unzip it into ClutchMetrics_Final\model\server_jobs on the laptop (replace existing files).
APP=/home/clutchmetrics
OUT="$APP/backup_$(date +%Y-%m-%d_%H%M).zip"
cd "$APP/model/server_jobs"
python3 - "$OUT" <<'PY'
import json, os, sys, zipfile
out = sys.argv[1]
n = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for job_id in sorted(os.listdir(".")):
        path = os.path.join(job_id, "job.json")
        if not os.path.exists(path) or json.load(open(path)).get("status") != "done":
            continue
        for folder, _, files in os.walk(job_id):
            for name in files:
                if name.startswith("upload") or name == "input.mp4":
                    continue
                z.write(os.path.join(folder, name))
        n += 1
print(f"{n} finished analyses packed into {out} ({os.path.getsize(out) / 1e6:.0f} MB)")
PY
