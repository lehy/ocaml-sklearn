#!/bin/bash

set -e
# set -x

IN="$1"
OUT="$2"

ROOT=$(realpath $(dirname "$0"))/..
echo "ROOT from $0 is $ROOT"

source "$ROOT/lib/version.sh"

EXPR="s/# scikit-learn for OCaml/scikit-learn for OCaml, version $OCAML_SKLEARN_FULL_VERSION/g"
for var in SKLEARN_BASIC_VERSION SKLEARN_FULL_VERSION OCAML_SKLEARN_VERSION OCAML_SKLEARN_FULL_VERSION \
           NUMPY_BASIC_VERSION NUMPY_FULL_VERSION OCAML_NUMPY_VERSION OCAML_NUMPY_FULL_VERSION \
           SCIPY_BASIC_VERSION SCIPY_FULL_VERSION OCAML_SCIPY_VERSION OCAML_SCIPY_FULL_VERSION; do
    val="${!var}"
    EXPR="$EXPR;s/%%$var%%/$val/g"
done

sed "$EXPR" "$IN" > "$OUT"
