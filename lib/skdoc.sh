#!/bin/bash
set -e
set -x

ROOT=$(realpath $(dirname "$0"))/..
VENV="$ROOT/.venv-$@"
echo "ROOT from $0 is $ROOT"
source "$ROOT/lib/version.sh"
if ! test -e "$VENV/bin/activate"; then
   echo "creating virtualenv in $VENV"
   python3 -mvenv "$VENV"
fi
source "$VENV"/bin/activate
pip install scikit-learn=="$SKLEARN_FULL_VERSION" pytest regex cffi
python3 "$ROOT"/lib/skdoc.py "$@"
