#!/bin/bash
# Jarvislabs startup script: start the ClutchMetrics server, ngrok and auto-pause whenever the machine starts.
bash /home/clutchmetrics/model/cloud/start.sh >> /home/clutchmetrics/startup.log 2>&1
