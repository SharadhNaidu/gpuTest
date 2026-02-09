#!/bin/bash
# AI Desktop Assessment - One Command Runner
# Usage: ./run.sh or bash run.sh
exec python3 "$(dirname "$0")/run.py" "$@"
