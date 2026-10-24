#!/bin/bash
# stop.sh [PIDFILE] — SIGTERM the server (python front end), wait up to 2 min for the engine child to exit, then SIGKILL.
PIDFILE=${1:-./run/server.pid}; P=$(cat $PIDFILE 2>/dev/null); [ -z "$P" ] && exit 0
kill $P 2>/dev/null
for i in $(seq 60); do kill -0 $P 2>/dev/null || break; sleep 2; done
kill -0 $P 2>/dev/null && { kill -9 $P; echo forced; }
nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' '; echo stopped
