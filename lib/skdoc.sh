#!/bin/bash
set -e
set -x

ROOT=$(realpath $(dirname "$0"))/..
# Name the venvs differently to avoid concurrency problems when
# running several tasks in parallel.
VENV="$ROOT/.venv-$1-$2"
echo "ROOT from $0 is $ROOT"
source "$ROOT/lib/version.sh"
if ! test -e "$VENV/bin/activate"; then
   echo "creating virtualenv in $VENV"
   python3 -mvenv "$VENV"
fi
source "$VENV"/bin/activate
pip install scikit-learn=="$SKLEARN_FULL_VERSION" numpy=="$NUMPY_FULL_VERSION" scipy=="$SCIPY_FULL_VERSION" pytest regex cffi
python3 "$ROOT"/lib/skdoc.py "$@"
