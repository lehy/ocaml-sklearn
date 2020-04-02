#!/bin/bash
set -e
set -x
ROOT=$(realpath $(dirname "$0"))/..
VENV="$ROOT/.venv"
echo "ROOT from $0 is $ROOT"
echo "creating virtualenv in $VENV"
python3 -mvenv "$VENV"
source "$VENV"/bin/activate
pip install scikit-learn==0.22.2 pytest
python3 "$ROOT"/lib/skdoc.py "$@"
