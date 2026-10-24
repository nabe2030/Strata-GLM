#!/bin/bash
# memwatch.sh [PIDFILE] [MAX_GPU_C] [MIN_AVAIL_MIB]
# While the server in PIDFILE is alive, poll every 5 s; kill it if the hottest GPU reaches MAX_GPU_C (default 85)
# or MemAvailable stays below MIN_AVAIL_MIB (default 2500) for 3 polls in a row.
PIDFILE=${1:-./run/server.pid}; MAXT=${2:-85}; MINA=${3:-2500}; LOG=${MEMWATCH_LOG:-./memwatch.log}
n=0
while kill -0 $(cat $PIDFILE 2>/dev/null) 2>/dev/null; do
  a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | sort -n | tail -1)
  [ "$a" -lt "$MINA" ] && n=$((n+1)) || n=0
  if [ $n -ge 3 ] || [ "$t" -ge "$MAXT" ]; then echo "$(date +%F_%T) STOP avail=$a temp=$t" >> $LOG; kill $(cat $PIDFILE); break; fi
  sleep 5
done
