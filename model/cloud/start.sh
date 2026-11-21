#!/usr/bin/env bash
# Start (or restart) the ClutchMetrics analysis server and its public ngrok address on the Jarvislabs machine.
# Both keep running after you close the browser tab. Run this again after every resume of the machine.
#   bash /home/clutchmetrics/model/cloud/start.sh
APP=/home/clutchmetrics
URL=https://lizard-makeshift-clock.ngrok-free.dev

if [ ! -f "$APP/ngrok.yml" ]; then
  echo "ngrok is not connected yet. Run once (with your own token):"
  echo "  $APP/bin/ngrok config add-authtoken <YOUR_TOKEN> --config $APP/ngrok.yml"
  exit 1
fi

pkill -f "python server.py" 2>/dev/null
pkill -f "ngrok http" 2>/dev/null
pkill -f "cloud/idle_pause.py" 2>/dev/null
sleep 1

cd "$APP/model"
. "$APP/venv/bin/activate"
export HF_HOME="$APP/hf"
export CM_FIREBASE_PROJECT=basketball-analyzer-f610b   # only signed-in website users; each sees only their own games
export CM_CLEANUP=1                                    # small disk: keep only the results of finished analyses
export CM_ACTIVITY_FILE="$APP/last_activity"           # when the website last used the server (for auto-pause)
export PORT=6006                                       # Jarvislabs serves port 6006 at its public "API" address
export HOST=0.0.0.0                                     # accept that address (logins are still checked)
nohup python server.py >> "$APP/server.log" 2>&1 &

for i in $(seq 1 60); do
  curl -s -m 2 localhost:6006/api/health >/dev/null && break
  sleep 1
done
if ! curl -s -m 2 localhost:6006/api/health >/dev/null; then
  echo "The server did not start. Last lines of $APP/server.log:"; tail -20 "$APP/server.log"; exit 1
fi
echo "Server running on this machine."

nohup "$APP/bin/ngrok" http 6006 --url="$URL" --config "$APP/ngrok.yml" --log stdout >> "$APP/ngrok.log" 2>&1 &
sleep 6
if curl -s -m 10 -H "ngrok-skip-browser-warning: 1" "$URL/api/health" | grep -q '"ok":true'; then
  echo "Online at $URL  (the website now uses this machine)"
else
  echo "ngrok did not come up. Is the laptop's start_server.bat still running? Only one machine can use the address."
  echo "Last lines of $APP/ngrok.log:"; tail -5 "$APP/ngrok.log"
fi

# auto-pause after a quiet hour (only when an API key was saved with save_key.sh)
if [ -f "$APP/jl_api_key" ]; then
  nohup python "$APP/model/cloud/idle_pause.py" >> "$APP/idle_pause.log" 2>&1 &
  if [ -f "$APP/keep_awake" ]; then
    echo "Auto-pause: OFF for now (keep_awake file present; rm $APP/keep_awake to turn it on)"
  else
    echo "Auto-pause: on, after 60 quiet minutes (touch $APP/keep_awake to keep it running)"
  fi
else
  echo "Auto-pause: not set up (bash $APP/model/cloud/save_key.sh)"
fi
