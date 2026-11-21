"""
ClutchMetrics analysis server on Modal (cloud GPU), so the website works without the laptop.

Deploy (from the ClutchMetrics_Final/model folder, after `python -m modal setup`):
    python -m modal deploy modal_app.py
Modal prints the web address (https://<workspace>--clutchmetrics-web.modal.run); the website's config.js points at it.

How it runs:
  web      a small CPU web server (the same FastAPI app as server.py): uploads, status, results, video
  analyse  the whole model on an L4 GPU; started for each upload and stopped when it finishes (billed per second)
  storage  a Modal Volume "clutchmetrics-jobs" holds uploads and results, shared by both
  state    a Modal Dict "clutchmetrics-state" carries small messages to a running analysis: the user's court mapping
           and cancel requests (the GPU container cannot see new Volume files while it has its own files open)
At most one GPU runs at a time and at most MAX_JOBS_PER_DAY uploads are accepted per day, to protect the credits.
Cancel stops the GPU call itself, so a cancelled analysis stops costing at once.

Existing analyses from the laptop: copy each finished job folder into the Volume (see README, "Moving to Modal").
"""

import os

import modal

APP_NAME = "clutchmetrics"
CODE = "/root/final_dp"
JOBS = "/jobs"
GPU = "L4"
MAX_JOBS_PER_DAY = "15"
HERE = os.path.dirname(os.path.abspath(__file__))

CODE_FILES = ["server.py", "basketball_pipeline.py", "compare_detectors.py", "jersey_numbers.py", "court_mapping.py",
              "court_dataset.py", "match_events.py", "render_broadcast.py", "export_dashboard.py"]
WEIGHTS = ["weights/rfdetr_basketball_v2/best", "weights/rfdetr_5class"]

app = modal.App(APP_NAME)
jobs = modal.Volume.from_name("clutchmetrics-jobs", create_if_missing=True)
state = modal.Dict.from_name("clutchmetrics-state", create_if_missing=True)


def warm_models():
    """Runs once while the image is built: downloads and caches every model, so analyses start quickly."""
    import sys
    sys.path.insert(0, CODE)
    os.chdir(CODE)
    from huggingface_hub import snapshot_download
    snapshot_download("Qwen/Qwen3-VL-2B-Instruct")
    from rfdetr import RFDETRNano
    RFDETRNano(pretrain_weights=f"{CODE}/weights/rfdetr_5class/checkpoint_best_total.pth", num_classes=5)
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    AutoImageProcessor.from_pretrained(f"{CODE}/weights/rfdetr_basketball_v2/best")
    AutoModelForObjectDetection.from_pretrained(f"{CODE}/weights/rfdetr_basketball_v2/best")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0", "fonts-dejavu-core")
    .pip_install(
        "torch==2.11.0", "torchvision==0.26.0", "transformers==5.19.0", "rfdetr==1.12.0", "supervision==0.31.5",
        "ultralytics", "opencv-python-headless", "numpy", "scipy", "pillow", "imageio-ffmpeg", "safetensors",
        "huggingface_hub", "fastapi==0.162.2", "python-multipart", "google-auth", "requests",
    )
    .env({"CM_JOBS_DIR": JOBS, "MAX_JOBS_PER_DAY": MAX_JOBS_PER_DAY, "PYTHONUNBUFFERED": "1",
          "CM_FIREBASE_PROJECT": "basketball-analyzer-f610b",
          # the GPU waits at most this long for the court mapping (the user maps while the earlier steps run)
          "COURT_WAIT_MINUTES": "8"})
)
for w in WEIGHTS:
    image = image.add_local_dir(os.path.join(HERE, w), f"{CODE}/{w}", copy=True)
image = image.run_function(warm_models)
for f in CODE_FILES:
    image = image.add_local_file(os.path.join(HERE, f), f"{CODE}/{f}")


def _quiet(fn):
    """Volume reload/commit can refuse while a file is open (e.g. a video being streamed); retry later instead."""
    def run():
        try:
            fn()
        except Exception:  # noqa: BLE001
            pass
    return run


def _server():
    import sys
    sys.path.insert(0, CODE)
    os.chdir(CODE)
    import server
    server.hooks["refresh"] = _quiet(jobs.reload)
    server.hooks["saved"] = _quiet(jobs.commit)

    def put(key, value):
        try:
            state[key] = value
        except Exception:  # noqa: BLE001 - a lost message only means the file-based path is used
            pass

    def get(key):
        try:
            return state.get(key)
        except Exception:  # noqa: BLE001
            return None

    server.hooks["message_put"] = put
    server.hooks["message_get"] = get
    os.makedirs(JOBS, exist_ok=True)
    return server


@app.function(image=image, gpu=GPU, volumes={JOBS: jobs}, timeout=3 * 3600, max_containers=1)
def analyse(job_id: str):
    server = _server()
    server.analyse(job_id)
    jobs.commit()


@app.function(image=image, volumes={JOBS: jobs}, timeout=1800, min_containers=0)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    server = _server()

    def submit(job_id):
        call = analyse.spawn(job_id)
        state[f"call:{job_id}"] = call.object_id      # so Cancel can stop this exact GPU call

    def stop(job_id):
        try:
            call_id = state.get(f"call:{job_id}")
            if not call_id:
                return False
            modal.FunctionCall.from_id(call_id).cancel()
            return True
        except Exception:  # noqa: BLE001 - the cancel flag in the Dict still stops it at the next step
            return False

    server.hooks["submit"] = submit
    server.hooks["stop"] = stop
    return server.app
