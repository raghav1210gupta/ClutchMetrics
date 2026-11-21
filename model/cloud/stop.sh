#!/usr/bin/env bash
# Stop the server and ngrok on the Jarvislabs machine (before pausing it, or to switch back to the laptop).
pkill -f "python server.py" 2>/dev/null && echo "server stopped" || echo "server was not running"
pkill -f "ngrok http" 2>/dev/null && echo "ngrok stopped" || echo "ngrok was not running"
