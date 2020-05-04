# scikit-learn for OCaml

ocaml-sklearn allows using Python's
[scikit-learn](https://scikit-learn.org/) machine learning library
from OCaml.

[Read the online scikit-learn OCaml API documentation
here.](https://lehy.github.io/ocaml-sklearn/)

If you are not familiar with scikit-learn, consult its Python [getting
started documentation](https://scikit-learn.org/stable/getting_started.html)
and [user guide](https://scikit-learn.org/stable/user_guide.html).

**This is a preview. The current OCaml API is not complete. Some functions may be hard or
impossible to use. Also, the existing API is not stable, it may change
to accomodate more functionality or make things easier to use.**

## Example : support vector regression with RBF kernel

```ocaml
let n_samples, n_features = 10, 5 in
Random.init 0;
let y = Sklearn.Arr.of_bigarray @@ Owl.Arr.uniform [|n_samples|] in
let x = Sklearn.Arr.of_bigarray @@ Owl.Dense.Matrix.D.uniform n_samples n_features in
let open Sklearn.Svm in
let clf = SVR.create ~c:1.0 ~epsilon:0.2 () in
Format.printf "%a\n" SVR.pp @@ SVR.fit clf ~x ~y;
Format.printf "%a\n" Sklearn.Arr.pp @@ SVR.support_vectors_ clf;;
```

This outputs:
```
SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='scale',
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
[[0.14509922 0.16277752 0.99033894 0.84013554 0.96508279]
 [0.8865312  0.80655193 0.07459775 0.36058768 0.22130337]
 [0.21844203 0.09612442 0.49908686 0.1154579  0.98202969]
 [0.07306658 0.97225754 0.20558949 0.16423512 0.57400651]
 [0.08153976 0.41462111 0.66190418 0.70208221 0.3600998 ]
 [0.20502873 0.04244781 0.21800856 0.28184598 0.4282653 ]
 [0.89211037 0.51466381 0.23432621 0.29850877 0.13323457]]
```

There are more examples in
[`examples/auto`](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/),
for instance
[`examples/auto/svm.ml`](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/svm.ml).

## Installation

~~~sh
opam install sklearn
~~~

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
pip install scikit-learn==%%SKLEARN_FULL_VERSION%% pytest
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

Most data is passed in and out of sklearn through module `Arr`. An
`Arr.t` can contain any `Numpy` or `Scipy` array-like, including
`ndarray` and `csr_matrix`.

You should generally build a dense array using the constructors in `Arr`:

~~~ocaml
let x = Arr.Float.matrix [|[| 1; 2 |]; [| 3; 4 |]|]
~~~

`Arr` currently covers a subset of `Numpy` functionality (added as
needed depending on tests and time).

One way to build an `Arr.t` is to use `Owl`'s functions to construct a
bigarray and then use `Arr.of_bigarray`. Data is shared between the
bigarray and the `Arr.t`.

To get data out of an `Arr.t`, use `to_int_array`, `to_float_array` or
`to_bigarray`.

Attributes are exposed read-only, each with two getters: one that
raises Not_found if the attribute is None, and the other that returns
an option.

Bunches (as returned from the sklearn.datasets APIs) are exposed as
objects.

Arguments taking string values are converted (in most cases) to
polymorphic variants.

Each module has a conversion function to `Py.Object.t`, so that you
can always escape and use `pyml` directly if the API provided here is
incomplete.

No attempt is made to expose features marked as deprecated.

## Development notes

ocaml-sklearn's sources are generated using a Python program (see
`lib/skdoc.py`) that loads up sklearn and uses introspection to
generate bindings based on `pyml`. To determine types, it parses
scikit-learn's documentation.

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

The tests are in `examples/auto`. They are based on examples extracted
from the Python documentation. A good way to develop is to pick one of
the files and start porting examples. One can refer to
`examples/auto/svm.ml` or `examples/auto/pipeline.ml`, whose examples
have already been ported (almost) completely.

The following examples have been ported completely:
- [ensemble](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/ensemble.ml)
- [metrics](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/metrics.ml)
- [neighbors](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/neighbors.ml)
- [pipeline](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/pipeline.ml)
- [preprocessing](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/preprocessing.ml)
- [svm](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/svm.ml)

The following examples still need to be ported:
- [calibration](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/calibration.ml)
- [cluster](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/cluster.ml)
- [compose](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/compose.ml)
- [covariance](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/covariance.ml)
- [cross_decomposition](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/cross_decomposition.ml)
- [datasets](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/datasets.ml)
- [decomposition](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/decomposition.ml)
- [discriminant_analysis](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/discriminant_analysis.ml)
- [feature_extraction](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/feature_extraction.ml)
- [feature_selection](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/feature_selection.ml)
- [gaussian_process](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/gaussian_process.ml)
- [impute](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/impute.ml)
- [inspection](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/inspection.ml)
- [isotonic](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/isotonic.ml)
- [kernel_approximation](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/kernel_approximation.ml)
- [linear_model](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/linear_model.ml)
- [manifold](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/manifold.ml)
- [model_selection](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/model_selection.ml)
- [multiclass](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/multiclass.ml)
- [multioutput](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/multioutput.ml)
- [naive_bayes](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/naive_bayes.ml)
- [neural_network](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/neural_network.ml)
- [random_projection](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/random_projection.ml)
- [semi_supervised](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/semi_supervised.ml)
- [tree](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/tree.ml)
- [utils](https://github.com/lehy/ocaml-sklearn/blob/master/examples/auto/utils.ml)

### Generating documentation

~~~sh
lib/build-doc
~~~

Documentation can then be found in `html_doc/`. Serve
it locally with something like:

~~~sh
python3 -mhttp.server --directory html_doc
xdg-open http://localhost:8000
~~~

## License

BSD-3. See file [LICENSE](LICENSE).
