#!/bin/bash
# start.sh CFG [TAG] — skeleton: start the Strata-GLM server in the background.
# Set STRATA_DIR (checkout with the build and venv) and RUN_DIR (pid file / logs). The config is the JSON passed to serve/server.py.
CFG=${1:?usage: start.sh CFG [TAG]}; TAG=${2:-main}
STRATA_DIR=${STRATA_DIR:-$HOME/Strata-GLM}; RUN_DIR=${RUN_DIR:-$PWD}; mkdir -p $RUN_DIR/run $RUN_DIR/logs
cd $STRATA_DIR
export CUDA_DEVICE_ORDER=PCI_BUS_ID
setsid nohup ${VENV_PYTHON:-python3} serve/server.py --engine strata --config $CFG > $RUN_DIR/logs/$TAG.server.log 2>&1 < /dev/null &
echo $! > $RUN_DIR/run/server.pid
echo started pid $(cat $RUN_DIR/run/server.pid)
