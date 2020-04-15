# scikit-learn for OCaml, version %%OCAML_SKLEARN_FULL_VERSION%%

[Read the online documentation here.](https://lehy.github.io/ocaml-sklearn/)

ocaml-sklearn allows using Python's
[scikit-learn](https://scikit-learn.org/) machine learning library
from OCaml.

**The current API is not complete. It covers most parts of
scikit-learn and what is accessible should work, but some
functionalities may be hard to use or inaccessible yet.**

## Example

```ocaml
let n_samples, n_features = 10, 5 in
Random.init 0;
let y = Sklearn.Ndarray.of_bigarray @@ Owl.Arr.uniform [|n_samples|] in
let x = Sklearn.Ndarray.of_bigarray @@ Matrix.uniform n_samples n_features in
let open Sklearn.Svm in
let clf = SVR.create ~c:1.0 ~epsilon:0.2 () in
Format.printf "%a" SVR.pp @@ SVR.fit clf ~x:(`Ndarray x) ~y;
```

This outputs:
```
SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='scale',
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

There are more examples in
[`examples/auto`](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/),
for instance
[`examples/auto/svm.ml`](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/svm.ml).

## Installation

(TODO, need to publish on opam)

## Finding Python's scikit-learn at runtime

At runtime, ocaml-sklearn expects to load the right version of
Python's scikit-learn. One way to do that is to create a virtualenv,
install scikit-learn version %%SKLEARN_FULL_VERSION%% inside, and run
your OCaml program in the activated virtualenv.

Do this once to create the virtualenv in `.venv` and install
scikit-learn inside:

```sh
python3 -mvenv .venv
source .venv/bin/activate
pip install scikit-learn==%%SKLEARN_FULL_VERSION%%
```

Then run your compiled OCaml program inside the virtualenv:

```sh
source .venv/bin/activate
./my_ocaml_program.exe
```

A version of ocaml-sklearn is tied to a version of Python's
sklearn. For instance, a version of ocaml-sklearn for Python's
scikit-learn 0.22.2 will refuse to initialize (by throwing an
exception) if scikit-learn's version is not 0.22 (it can be 0.22.1,
0.22.2 or 0.22.2.post1).

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

### Python requirements

The requirements for developing (not using) the bindings are in file
`requirements-dev.txt`. Install it using:

~~~sh
python3 -mvenv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
~~~

### Running tests

~~~sh
dune runtest
~~~

### Generating documentation

~~~sh
dune build @mkdocs
~~~

Documentation can then be found in `_build/default/html_doc/`. Serve
it locally with something like:

~~~sh
python3 -mhttp.server --directory _build/default/html_doc
xdg-open http://localhost:8000
~~~

## License

BSD-3. See file [LICENSE](LICENSE).
