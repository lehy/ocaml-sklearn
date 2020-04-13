# Bindings to Python's scikit-learn for OCaml

This OCaml library allows using Python's scikit-learn/sklearn machine learning
library.

## Example

```ocaml
let n_samples, n_features = 10, 5 in
Random.init 0;
let y = Sklearn.Ndarray.of_bigarray @@ Owl.Arr.uniform [|n_samples|] in
let x = Sklearn.Ndarray.of_bigarray @@ Matrix.uniform n_samples n_features in
let open Sklearn.Svm in
let clf = SVR.create ~c:1.0 ~epsilon:0.2 () in
print SVR.pp @@ SVR.fit clf ~x:(`Ndarray x) ~y;
```

This outputs:
```
SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='scale',
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

There are more examples in `examples/auto`, for instance
`examples/auto/svm.ml`.

## Installation

(TODO)

## API

We attempt to bind all of scikit-learn's APIs. However, not all of the
APIs are currently tested, and some are probably hard to use or
unusable at the moment.

Each Python module or class gets its own OCaml module. For instance
Python class `sklearn.svm.SVC` can be found in OCaml module
`Sklearn.Svm.SVC`. This module has a `create` function to construct an
`SVC` and functions corresponding to the Python methods and
attributes.

Most data is passed in and out of sklearn through a binding to Numpy's
`Ndarray`. This is a tensor type that can contain integers, floats, or
strings (as exposed in the current OCaml API). The way to get data
into or out of an `Ndarray.t` is to go through OCaml arrays or
bigarrays. A way to create bigarrays is to use `Owl`'s facilities. At
the moment, the `Ndarray` functions exposed here are extremely minimal
(no `np.zeros` or `np.ones`).

Bunches (as returned from the sklearn.datasets APIs) are exposed as
objects.

Arguments taking string values are converted (in most cases) to
polymorphic variants.

Each module has a conversion function to `Py.Object.t`, so that you
can always escape and use `pyml` directly if the API provided here is
incomplete.

## Development notes

ocaml-sklearn's sources are generated using a Python program (see
`lib/skdoc.py`) that loads up sklearn and uses introspection to generate
bindings based on `pyml`.


