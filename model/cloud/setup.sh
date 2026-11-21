#!/usr/bin/env bash
# One-time setup of the ClutchMetrics analysis server on a Jarvislabs GPU machine.
# Everything goes under /home/clutchmetrics: Jarvislabs keeps only /home when the machine is paused.
#   bash /home/clutchmetrics/model/cloud/setup.sh
set -euo pipefail
APP=/home/clutchmetrics
cd "$APP"
export HF_HOME="$APP/hf"            # model downloads (Qwen) stay under /home

echo "== 1/5 Python environment (in $APP/venv)"
if [ ! -d venv ]; then
  # reuse the machine's own PyTorch (built for its GPU driver) instead of downloading another 3 GB
  python3 -m venv --system-site-packages venv
fi
. venv/bin/activate
python -m pip install --quiet --upgrade pip

echo "== 2/5 PyTorch with the GPU"
if python -c "import torch; v = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]); assert torch.cuda.is_available() and v >= (2, 6)" 2>/dev/null; then
  python -c "import torch; print('  using the machine PyTorch', torch.__version__, 'on', torch.cuda.get_device_name(0))"
else
  echo "  installing PyTorch 2.11 (a few minutes)"
  pip install --quiet torch==2.11.0 torchvision==0.26.0
fi

echo "== 3/5 Model packages"
pip install --quiet -r model/cloud/requirements.txt
# no screen on a server: replace OpenCV by its headless build (the normal one needs desktop libraries)
pip uninstall --quiet -y opencv-python opencv-contrib-python 2>/dev/null || true
pip install --quiet --force-reinstall --no-deps opencv-python-headless

echo "== 4/5 Download Qwen3-VL and test the models on the GPU"
cd "$APP/model"
python - <<'PY'
import numpy as np
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-VL-2B-Instruct")
import torch
from basketball_pipeline import FINETUNED_MODEL, FinetunedDetector
detector = FinetunedDetector(FINETUNED_MODEL)
detector(np.zeros((720, 1280, 3), np.uint8))
print("  detector loads and runs on", torch.cuda.get_device_name(0))
PY

echo "== 5/5 ngrok (public address for the website)"
mkdir -p "$APP/bin"
if [ ! -x "$APP/bin/ngrok" ]; then
  curl -sSL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz | tar xz -C "$APP/bin"
fi
"$APP/bin/ngrok" version

echo
echo "Setup done. Disk used:"; du -sh "$APP" 2>/dev/null
echo
echo "Next, once only: connect ngrok to your account (type your own token, never share it):"
echo "  /home/clutchmetrics/bin/ngrok config add-authtoken <YOUR_TOKEN> --config /home/clutchmetrics/ngrok.yml"
echo "Then start the server:  bash /home/clutchmetrics/model/cloud/start.sh"
