#!/bin/bash
# Jersey-number reader test on the cloud GPU: Qwen3-VL-2B vs 8B on the AUS-IND clip, scored against the
# 150 hand-labelled player boxes. Does not touch the server or any uploaded game.
#   bash /home/clutchmetrics/model/cloud/num_test.sh 2>&1 | tee /home/clutchmetrics/num_test.log
set -e
APP=/home/clutchmetrics
cd "$APP/model"
. "$APP/venv/bin/activate"
export HF_HOME="$APP/hf"
made_awake=0
if [ ! -e "$APP/keep_awake" ]; then touch "$APP/keep_awake"; made_awake=1; fi   # no auto-pause during the test
trap '[ $made_awake = 1 ] && rm -f "$APP/keep_awake"' EXIT

run() {   # run <name> [reuse]   (reader settings come from CM_NUMBER_* variables)
  R=runs/pipeline; mkdir -p "$R/$1"; cp -r "$R/num_base/." "$R/$1/"
  start=$(date +%s)
  if [ "$2" = reuse ]; then cp "$R/$3/number_reads.json" "$R/$1/"; python jersey_numbers.py --name "$1" --teams AUS IND --reuse --no-video > /dev/null
  else python jersey_numbers.py --name "$1" --teams AUS IND --no-video 2>&1 | grep -E "Player boxes" ; fi
  python jersey_numbers.py --name "$1" --teams AUS IND --court-link > /dev/null
  echo "== $1 (CM_NUMBER_READER=${CM_NUMBER_READER:-2B} SIZE=${CM_NUMBER_SIZE:-160} MIN_CONF=${CM_NUMBER_MIN_CONF:-0}) $(( $(date +%s) - start )) s"
  python label_numbers.py score --name "$1" | sed -n 2,3p
}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
CM_NUMBER_SIZE=224 run t_2b
for c in 0.5 0.7 0.85; do CM_NUMBER_SIZE=224 CM_NUMBER_MIN_CONF=$c run t_2b_c$c reuse t_2b; done
echo "Downloading and running Qwen3-VL-8B (about 17 GB, first time only)..."
CM_NUMBER_SIZE=224 CM_NUMBER_READER=Qwen/Qwen3-VL-8B-Instruct run t_8b
for c in 0.5 0.7 0.85; do CM_NUMBER_SIZE=224 CM_NUMBER_MIN_CONF=$c run t_8b_c$c reuse t_8b; done
echo "TEST DONE"
