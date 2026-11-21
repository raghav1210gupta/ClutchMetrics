#!/usr/bin/env bash
# Save your Jarvislabs API key on this machine (for automatic pausing). The key is not shown while you paste it.
#   bash /home/clutchmetrics/model/cloud/save_key.sh
APP=/home/clutchmetrics
read -r -s -p "Paste your Jarvislabs API key and press Enter: " KEY
echo
# drop hidden paste markers (ESC[200~ ... ESC[201~) and any spaces or line breaks around the key
KEY=$(printf '%s' "$KEY" | sed 's/\x1b\[20[01]~//g' | tr -d '[:space:]')
echo "Key length: ${#KEY} characters"
if [ -z "$KEY" ]; then echo "Nothing saved."; exit 1; fi
umask 077
printf '%s' "$KEY" > "$APP/jl_api_key"
"$APP/venv/bin/pip" install --quiet jarvislabs
if "$APP/venv/bin/python" -c "
from jarvislabs import Client
names = [i.name for i in Client(api_key=open('$APP/jl_api_key').read().strip()).instances.list()]
print('Key works. Your instances:', names)
"; then
  echo "Saved. Auto-pause starts with the next: bash $APP/model/cloud/start.sh"
else
  rm -f "$APP/jl_api_key"; echo "The key did not work; nothing saved. Check it and run this again."
fi
