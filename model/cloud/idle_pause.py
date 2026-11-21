"""
Pause this Jarvislabs machine automatically when nobody uses it, so the prepaid credit is not spent overnight.

It pauses when ALL of these hold for IDLE_MINUTES (default 60):
  - no analysis is waiting or running,
  - the website has not used the server (the server writes the time of each request to /home/clutchmetrics/last_activity),
  - the file /home/clutchmetrics/keep_awake does not exist (create it for the open house or recording sessions:
        touch /home/clutchmetrics/keep_awake        and remove it again with    rm /home/clutchmetrics/keep_awake).
It needs a Jarvislabs API key saved with cloud/save_key.sh. start.sh starts this watcher in the background.
"""
import glob
import json
import os
import time

APP = "/home/clutchmetrics"
KEY_FILE = f"{APP}/jl_api_key"
ACTIVITY = f"{APP}/last_activity"
KEEP_AWAKE = f"{APP}/keep_awake"
JOBS = f"{APP}/model/server_jobs"
NAME = os.environ.get("JL_INSTANCE_NAME", "clutchmetrics")
IDLE_MINUTES = float(os.environ.get("IDLE_MINUTES", 60))
started = time.time()


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def busy():
    for path in glob.glob(f"{JOBS}/*/job.json"):
        try:
            if json.load(open(path)).get("status") in ("queued", "running"):
                return True
        except (OSError, ValueError):
            continue
    return False


def last_use():
    try:
        return max(float(open(ACTIVITY).read().strip() or 0), started)
    except (OSError, ValueError):
        return started


def pause_myself():
    from jarvislabs import Client
    client = Client(api_key=open(KEY_FILE).read().strip())
    mine = [i for i in client.instances.list() if i.name == NAME and str(i.status).lower() == "running"]
    if len(mine) != 1:
        log(f"not pausing: found {len(mine)} running instances named {NAME!r}")
        return False
    log(f"pausing instance {mine[0].machine_id} ({NAME}) after {IDLE_MINUTES:.0f} quiet minutes")
    client.instances.pause(mine[0].machine_id)
    return True


def main():
    if not os.path.exists(KEY_FILE):
        log("no API key saved (run cloud/save_key.sh): auto-pause is off")
        return
    log(f"auto-pause on: after {IDLE_MINUTES:.0f} quiet minutes, unless {KEEP_AWAKE} exists")
    while True:
        time.sleep(60)
        quiet = (time.time() - last_use()) / 60
        if os.path.exists(KEEP_AWAKE) or busy() or quiet < IDLE_MINUTES:
            continue
        try:
            if pause_myself():
                return
        except Exception as err:  # noqa: BLE001 - try again in a while
            log(f"pause failed: {type(err).__name__}: {err}")
        time.sleep(600)


if __name__ == "__main__":
    main()
