module Bunch : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?kwargs:(string * Py.Object.t) list -> unit -> t
(**
Container object for datasets

Dictionary-like object that exposes its keys as attributes.

>>> b = Bunch(a=1, b=2)
>>> b['b']
2
>>> b.b
2
>>> b.a = 3
>>> b['a']
3
>>> b.c = 6
>>> b['c']
6
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module FeatureUnion : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_jobs:[`Int of int | `None] -> ?transformer_weights:Py.Object.t -> ?verbose:bool -> transformer_list:(string * Py.Object.t) list -> unit -> t
(**
Concatenates results of multiple transformer objects.

This estimator applies a list of transformer objects in parallel to the
input data, then concatenates the results. This is useful to combine
several feature extraction mechanisms into a single transformer.

Parameters of the transformers may be set using its name and the parameter
name separated by a '__'. A transformer may be replaced entirely by
setting the parameter with its name to another transformer,
or removed by setting to 'drop'.

Read more in the :ref:`User Guide <feature_union>`.

.. versionadded:: 0.13

Parameters
----------
transformer_list : list of (string, transformer) tuples
    List of transformer objects to be applied to the data. The first
    half of each tuple is the name of the transformer.

    .. versionchanged:: 0.22
       Deprecated `None` as a transformer in favor of 'drop'.

n_jobs : int or None, optional (default=None)
    Number of jobs to run in parallel.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

transformer_weights : dict, optional
    Multiplicative weights for features per transformer.
    Keys are transformer names, values the weights.

verbose : boolean, optional(default=False)
    If True, the time elapsed while fitting each transformer will be
    printed as it is completed.

See Also
--------
sklearn.pipeline.make_union : Convenience function for simplified
    feature union construction.

Examples
--------
>>> from sklearn.pipeline import FeatureUnion
>>> from sklearn.decomposition import PCA, TruncatedSVD
>>> union = FeatureUnion([("pca", PCA(n_components=1)),
...                       ("svd", TruncatedSVD(n_components=2))])
>>> X = [[0., 1., 3], [2., 2., 5]]
>>> union.fit_transform(X)
array([[ 1.5       ,  3.0...,  0.8...],
       [-1.5       ,  5.7..., -0.4...]])
*)

val fit : ?y:Ndarray.t -> ?fit_params:(string * Py.Object.t) list -> x:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> t -> t
(**
Fit all transformers using X.

Parameters
----------
X : iterable or array-like, depending on transformers
    Input data, used to fit transformers.

y : array-like, shape (n_samples, ...), optional
    Targets for supervised learning.

Returns
-------
self : FeatureUnion
    This estimator
*)

val fit_transform : ?y:Ndarray.t -> ?fit_params:(string * Py.Object.t) list -> x:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> t -> Ndarray.t
(**
Fit all transformers, transform the data and concatenate results.

Parameters
----------
X : iterable or array-like, depending on transformers
    Input data to be transformed.

y : array-like, shape (n_samples, ...), optional
    Targets for supervised learning.

Returns
-------
X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
    hstack of results of transformers. sum_n_components is the
    sum of n_components (output dimension) over transformers.
*)

val get_feature_names : t -> string list
(**
Get feature names from all transformers.

Returns
-------
feature_names : list of strings
    Names of the features produced by transform.
*)

val get_params : ?deep:bool -> t -> Py.Object.t
(**
Get parameters for this estimator.

Parameters
----------
deep : boolean, optional
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

Returns
-------
params : mapping of string to any
    Parameter names mapped to their values.
*)

val set_params : ?kwargs:(string * Py.Object.t) list -> t -> t
(**
Set the parameters of this estimator.

Valid parameter keys can be listed with ``get_params()``.

Returns
-------
self
*)

val transform : x:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> t -> Ndarray.t
(**
Transform X separately by each transformer, concatenate results.

Parameters
----------
X : iterable or array-like, depending on transformers
    Input data to be transformed.

Returns
-------
X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
    hstack of results of transformers. sum_n_components is the
    sum of n_components (output dimension) over transformers.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Parallel : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_jobs:int -> ?backend:Py.Object.t -> ?verbose:int -> ?timeout:float -> ?pre_dispatch:[`All | `Int of int | `PyObject of Py.Object.t] -> ?batch_size:[`Int of int | `Auto] -> ?temp_folder:string -> ?max_nbytes:Py.Object.t -> ?mmap_mode:[`R_ | `R | `W_ | `C | `None] -> ?prefer:[`Processes | `Threads | `None] -> ?require:[`Sharedmem | `None] -> unit -> t
(**
Helper class for readable parallel mapping.

Read more in the :ref:`User Guide <parallel>`.

Parameters
-----------
n_jobs: int, default: None
    The maximum number of concurrently running jobs, such as the number
    of Python worker processes when backend="multiprocessing"
    or the size of the thread-pool when backend="threading".
    If -1 all CPUs are used. If 1 is given, no parallel computing code
    is used at all, which is useful for debugging. For n_jobs below -1,
    (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all
    CPUs but one are used.
    None is a marker for 'unset' that will be interpreted as n_jobs=1
    (sequential execution) unless the call is performed under a
    parallel_backend context manager that sets another value for
    n_jobs.
backend: str, ParallelBackendBase instance or None, default: 'loky'
    Specify the parallelization backend implementation.
    Supported backends are:

    - "loky" used by default, can induce some
      communication and memory overhead when exchanging input and
      output data with the worker Python processes.
    - "multiprocessing" previous process-based backend based on
      `multiprocessing.Pool`. Less robust than `loky`.
    - "threading" is a very low-overhead backend but it suffers
      from the Python Global Interpreter Lock if the called function
      relies a lot on Python objects. "threading" is mostly useful
      when the execution bottleneck is a compiled extension that
      explicitly releases the GIL (for instance a Cython loop wrapped
      in a "with nogil" block or an expensive call to a library such
      as NumPy).
    - finally, you can register backends by calling
      register_parallel_backend. This will allow you to implement
      a backend of your liking.

    It is not recommended to hard-code the backend name in a call to
    Parallel in a library. Instead it is recommended to set soft hints
    (prefer) or hard constraints (require) so as to make it possible
    for library users to change the backend from the outside using the
    parallel_backend context manager.
prefer: str in {'processes', 'threads'} or None, default: None
    Soft hint to choose the default backend if no specific backend
    was selected with the parallel_backend context manager. The
    default process-based backend is 'loky' and the default
    thread-based backend is 'threading'. Ignored if the ``backend``
    parameter is specified.
require: 'sharedmem' or None, default None
    Hard constraint to select the backend. If set to 'sharedmem',
    the selected backend will be single-host and thread-based even
    if the user asked for a non-thread based backend with
    parallel_backend.
verbose: int, optional
    The verbosity level: if non zero, progress messages are
    printed. Above 50, the output is sent to stdout.
    The frequency of the messages increases with the verbosity level.
    If it more than 10, all iterations are reported.
timeout: float, optional
    Timeout limit for each task to complete.  If any task takes longer
    a TimeOutError will be raised. Only applied when n_jobs != 1
pre_dispatch: {'all', integer, or expression, as in '3*n_jobs'}
    The number of batches (of tasks) to be pre-dispatched.
    Default is '2*n_jobs'. When batch_size="auto" this is reasonable
    default and the workers should never starve.
batch_size: int or 'auto', default: 'auto'
    The number of atomic tasks to dispatch at once to each
    worker. When individual evaluations are very fast, dispatching
    calls to workers can be slower than sequential computation because
    of the overhead. Batching fast computations together can mitigate
    this.
    The ``'auto'`` strategy keeps track of the time it takes for a batch
    to complete, and dynamically adjusts the batch size to keep the time
    on the order of half a second, using a heuristic. The initial batch
    size is 1.
    ``batch_size="auto"`` with ``backend="threading"`` will dispatch
    batches of a single task at a time as the threading backend has
    very little overhead and using larger batch size has not proved to
    bring any gain in that case.
temp_folder: str, optional
    Folder to be used by the pool for memmapping large arrays
    for sharing memory with worker processes. If None, this will try in
    order:

    - a folder pointed by the JOBLIB_TEMP_FOLDER environment
      variable,
    - /dev/shm if the folder exists and is writable: this is a
      RAM disk filesystem available by default on modern Linux
      distributions,
    - the default system temporary folder that can be
      overridden with TMP, TMPDIR or TEMP environment
      variables, typically /tmp under Unix operating systems.

    Only active when backend="loky" or "multiprocessing".
max_nbytes int, str, or None, optional, 1M by default
    Threshold on the size of arrays passed to the workers that
    triggers automated memory mapping in temp_folder. Can be an int
    in Bytes, or a human-readable string, e.g., '1M' for 1 megabyte.
    Use None to disable memmapping of large arrays.
    Only active when backend="loky" or "multiprocessing".
mmap_mode: {None, 'r+', 'r', 'w+', 'c'}
    Memmapping mode for numpy arrays passed to workers.
    See 'max_nbytes' parameter documentation for more details.

Notes
-----

This object uses workers to compute in parallel the application of a
function to many different arguments. The main functionality it brings
in addition to using the raw multiprocessing or concurrent.futures API
are (see examples for details):

* More readable code, in particular since it avoids
  constructing list of arguments.

* Easier debugging:
    - informative tracebacks even when the error happens on
      the client side
    - using 'n_jobs=1' enables to turn off parallel computing
      for debugging without changing the codepath
    - early capture of pickling errors

* An optional progress meter.

* Interruption of multiprocesses jobs with 'Ctrl-C'

* Flexible pickling control for the communication to and from
  the worker processes.

* Ability to use shared memory efficiently with worker
  processes for large numpy-based datastructures.

Examples
--------

A simple example:

>>> from math import sqrt
>>> from joblib import Parallel, delayed
>>> Parallel(n_jobs=1)(delayed(sqrt)(i**2) for i in range(10))
[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

Reshaping the output when the function has several return
values:

>>> from math import modf
>>> from joblib import Parallel, delayed
>>> r = Parallel(n_jobs=1)(delayed(modf)(i/2.) for i in range(10))
>>> res, i = zip( *r)
>>> res
(0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5)
>>> i
(0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0)

The progress meter: the higher the value of `verbose`, the more
messages:

>>> from time import sleep
>>> from joblib import Parallel, delayed
>>> r = Parallel(n_jobs=2, verbose=10)(delayed(sleep)(.2) for _ in range(10)) #doctest: +SKIP
[Parallel(n_jobs=2)]: Done   1 tasks      | elapsed:    0.6s
[Parallel(n_jobs=2)]: Done   4 tasks      | elapsed:    0.8s
[Parallel(n_jobs=2)]: Done  10 out of  10 | elapsed:    1.4s finished

Traceback example, note how the line of the error is indicated
as well as the values of the parameter passed to the function that
triggered the exception, even though the traceback happens in the
child process:

>>> from heapq import nlargest
>>> from joblib import Parallel, delayed
>>> Parallel(n_jobs=2)(delayed(nlargest)(2, n) for n in (range(4), 'abcde', 3)) #doctest: +SKIP
#...
---------------------------------------------------------------------------
Sub-process traceback:
---------------------------------------------------------------------------
TypeError                                          Mon Nov 12 11:37:46 2012
PID: 12934                                    Python 2.7.3: /usr/bin/python
...........................................................................
/usr/lib/python2.7/heapq.pyc in nlargest(n=2, iterable=3, key=None)
    419         if n >= size:
    420             return sorted(iterable, key=key, reverse=True)[:n]
    421
    422     # When key is none, use simpler decoration
    423     if key is None:
--> 424         it = izip(iterable, count(0,-1))                    # decorate
    425         result = _nlargest(n, it)
    426         return map(itemgetter(0), result)                   # undecorate
    427
    428     # General case, slowest method
 TypeError: izip argument #1 must support iteration
___________________________________________________________________________


Using pre_dispatch in a producer/consumer situation, where the
data is generated on the fly. Note how the producer is first
called 3 times before the parallel loop is initiated, and then
called to generate new data on the fly:

>>> from math import sqrt
>>> from joblib import Parallel, delayed
>>> def producer():
...     for i in range(6):
...         print('Produced %s' % i)
...         yield i
>>> out = Parallel(n_jobs=2, verbose=100, pre_dispatch='1.5*n_jobs')(
...                delayed(sqrt)(i) for i in producer()) #doctest: +SKIP
Produced 0
Produced 1
Produced 2
[Parallel(n_jobs=2)]: Done 1 jobs     | elapsed:  0.0s
Produced 3
[Parallel(n_jobs=2)]: Done 2 jobs     | elapsed:  0.0s
Produced 4
[Parallel(n_jobs=2)]: Done 3 jobs     | elapsed:  0.0s
Produced 5
[Parallel(n_jobs=2)]: Done 4 jobs     | elapsed:  0.0s
[Parallel(n_jobs=2)]: Done 6 out of 6 | elapsed:  0.0s remaining: 0.0s
[Parallel(n_jobs=2)]: Done 6 out of 6 | elapsed:  0.0s finished
*)

val debug : msg:Py.Object.t -> t -> Py.Object.t
(**
None
*)

val dispatch_next : t -> Py.Object.t
(**
Dispatch more data for parallel processing

This method is meant to be called concurrently by the multiprocessing
callback. We rely on the thread-safety of dispatch_one_batch to protect
against concurrent consumption of the unprotected iterator.
*)

val dispatch_one_batch : iterator:Py.Object.t -> t -> Py.Object.t
(**
Prefetch the tasks for the next batch and dispatch them.

The effective size of the batch is computed here.
If there are no more jobs to dispatch, return False, else return True.

The iterator consumption and dispatching is protected by the same
lock so calling this function should be thread safe.
*)

val format : ?indent:Py.Object.t -> obj:Py.Object.t -> t -> Py.Object.t
(**
Return the formatted representation of the object.
*)

val print_progress : t -> Py.Object.t
(**
Display the process of the parallel execution only a fraction
of time, controlled by self.verbose.
*)

val retrieve : t -> Py.Object.t
(**
None
*)

val warn : msg:Py.Object.t -> t -> Py.Object.t
(**
None
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Pipeline : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?memory:[`None | `String of string | `JoblibMemory of Py.Object.t] -> ?verbose:bool -> steps:(string * Py.Object.t) list -> unit -> t
(**
Pipeline of transforms with a final estimator.

Sequentially apply a list of transforms and a final estimator.
Intermediate steps of the pipeline must be 'transforms', that is, they
must implement fit and transform methods.
The final estimator only needs to implement fit.
The transformers in the pipeline can be cached using ``memory`` argument.

The purpose of the pipeline is to assemble several steps that can be
cross-validated together while setting different parameters.
For this, it enables setting parameters of the various steps using their
names and the parameter name separated by a '__', as in the example below.
A step's estimator may be replaced entirely by setting the parameter
with its name to another estimator, or a transformer removed by setting
it to 'passthrough' or ``None``.

Read more in the :ref:`User Guide <pipeline>`.

.. versionadded:: 0.5

Parameters
----------
steps : list
    List of (name, transform) tuples (implementing fit/transform) that are
    chained, in the order in which they are chained, with the last object
    an estimator.

memory : None, str or object with the joblib.Memory interface, optional
    Used to cache the fitted transformers of the pipeline. By default,
    no caching is performed. If a string is given, it is the path to
    the caching directory. Enabling caching triggers a clone of
    the transformers before fitting. Therefore, the transformer
    instance given to the pipeline cannot be inspected
    directly. Use the attribute ``named_steps`` or ``steps`` to
    inspect estimators within the pipeline. Caching the
    transformers is advantageous when fitting is time consuming.

verbose : bool, default=False
    If True, the time elapsed while fitting each step will be printed as it
    is completed.

Attributes
----------
named_steps : bunch object, a dictionary with attribute access
    Read-only attribute to access any step parameter by user given name.
    Keys are step names and values are steps parameters.

See Also
--------
sklearn.pipeline.make_pipeline : Convenience function for simplified
    pipeline construction.

Examples
--------
>>> from sklearn import svm
>>> from sklearn.datasets import make_classification
>>> from sklearn.feature_selection import SelectKBest
>>> from sklearn.feature_selection import f_regression
>>> from sklearn.pipeline import Pipeline
>>> # generate some data to play with
>>> X, y = make_classification(
...     n_informative=5, n_redundant=0, random_state=42)
>>> # ANOVA SVM-C
>>> anova_filter = SelectKBest(f_regression, k=5)
>>> clf = svm.SVC(kernel='linear')
>>> anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])
>>> # You can set the parameters using the names issued
>>> # For instance, fit using a k of 10 in the SelectKBest
>>> # and a parameter 'C' of the svm
>>> anova_svm.set_params(anova__k=10, svc__C=.1).fit(X, y)
Pipeline(steps=[('anova', SelectKBest(...)), ('svc', SVC(...))])
>>> prediction = anova_svm.predict(X)
>>> anova_svm.score(X, y)
0.83
>>> # getting the selected features chosen by anova_filter
>>> anova_svm['anova'].get_support()
array([False, False,  True,  True, False, False,  True,  True, False,
       True, False,  True,  True, False,  True, False,  True,  True,
       False, False])
>>> # Another way to get selected features chosen by anova_filter
>>> anova_svm.named_steps.anova.get_support()
array([False, False,  True,  True, False, False,  True,  True, False,
       True, False,  True,  True, False,  True, False,  True,  True,
       False, False])
>>> # Indexing can also be used to extract a sub-pipeline.
>>> sub_pipeline = anova_svm[:1]
>>> sub_pipeline
Pipeline(steps=[('anova', SelectKBest(...))])
>>> coef = anova_svm[-1].coef_
>>> anova_svm['svc'] is anova_svm[-1]
True
>>> coef.shape
(1, 10)
>>> sub_pipeline.inverse_transform(coef).shape
(1, 20)
*)

val get_item : ind:[`Int of int | `String of string | `Slice of ([`None | `Int of int]) * ([`None | `Int of int]) * ([`None | `Int of int])] -> t -> Py.Object.t
(**
Returns a sub-pipeline or a single esimtator in the pipeline

Indexing with an integer will return an estimator; using a slice
returns another Pipeline instance which copies a slice of this
Pipeline. This copy is shallow: modifying (or fitting) estimators in
the sub-pipeline will affect the larger pipeline and vice-versa.
However, replacing a value in `step` will not affect a copy.
*)

val decision_function : x:Ndarray.t -> t -> Ndarray.t
(**
Apply transforms, and decision_function of the final estimator

Parameters
----------
X : iterable
    Data to predict on. Must fulfill input requirements of first step
    of the pipeline.

Returns
-------
y_score : array-like of shape (n_samples, n_classes)
*)

val fit : ?y:Ndarray.t -> ?fit_params:(string * Py.Object.t) list -> x:Ndarray.t -> t -> t
(**
Fit the model

Fit all the transforms one after the other and transform the
data, then fit the transformed data using the final estimator.

Parameters
----------
X : iterable
    Training data. Must fulfill input requirements of first step of the
    pipeline.

y : iterable, default=None
    Training targets. Must fulfill label requirements for all steps of
    the pipeline.

**fit_params : dict of string -> object
    Parameters passed to the ``fit`` method of each step, where
    each parameter name is prefixed such that parameter ``p`` for step
    ``s`` has key ``s__p``.

Returns
-------
self : Pipeline
    This estimator
*)

val fit_predict : ?y:Ndarray.t -> ?fit_params:(string * Py.Object.t) list -> x:Ndarray.t -> t -> Ndarray.t
(**
Applies fit_predict of last step in pipeline after transforms.

Applies fit_transforms of a pipeline to the data, followed by the
fit_predict method of the final estimator in the pipeline. Valid
only if the final estimator implements fit_predict.

Parameters
----------
X : iterable
    Training data. Must fulfill input requirements of first step of
    the pipeline.

y : iterable, default=None
    Training targets. Must fulfill label requirements for all steps
    of the pipeline.

**fit_params : dict of string -> object
    Parameters passed to the ``fit`` method of each step, where
    each parameter name is prefixed such that parameter ``p`` for step
    ``s`` has key ``s__p``.

Returns
-------
y_pred : array-like
*)

val fit_transform : ?y:Ndarray.t -> ?fit_params:(string * Py.Object.t) list -> x:Ndarray.t -> t -> Ndarray.t
(**
Fit the model and transform with the final estimator

Fits all the transforms one after the other and transforms the
data, then uses fit_transform on transformed data with the final
estimator.

Parameters
----------
X : iterable
    Training data. Must fulfill input requirements of first step of the
    pipeline.

y : iterable, default=None
    Training targets. Must fulfill label requirements for all steps of
    the pipeline.

**fit_params : dict of string -> object
    Parameters passed to the ``fit`` method of each step, where
    each parameter name is prefixed such that parameter ``p`` for step
    ``s`` has key ``s__p``.

Returns
-------
Xt : array-like of shape  (n_samples, n_transformed_features)
    Transformed samples
*)

val get_params : ?deep:bool -> t -> Py.Object.t
(**
Get parameters for this estimator.

Parameters
----------
deep : boolean, optional
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

Returns
-------
params : mapping of string to any
    Parameter names mapped to their values.
*)

val inverse_transform : ?x:Ndarray.t -> ?y:Ndarray.t -> t -> Ndarray.t
(**
Apply inverse transformations in reverse order

All estimators in the pipeline must support ``inverse_transform``.

Parameters
----------
Xt : array-like of shape  (n_samples, n_transformed_features)
    Data samples, where ``n_samples`` is the number of samples and
    ``n_features`` is the number of features. Must fulfill
    input requirements of last step of pipeline's
    ``inverse_transform`` method.

Returns
-------
Xt : array-like of shape (n_samples, n_features)
*)

val predict : ?predict_params:(string * Py.Object.t) list -> x:Ndarray.t -> t -> Ndarray.t
(**
Apply transforms to the data, and predict with the final estimator

Parameters
----------
X : iterable
    Data to predict on. Must fulfill input requirements of first step
    of the pipeline.

**predict_params : dict of string -> object
    Parameters to the ``predict`` called at the end of all
    transformations in the pipeline. Note that while this may be
    used to return uncertainties from some models with return_std
    or return_cov, uncertainties that are generated by the
    transformations in the pipeline are not propagated to the
    final estimator.

Returns
-------
y_pred : array-like
*)

val predict_log_proba : x:Ndarray.t -> t -> Ndarray.t
(**
Apply transforms, and predict_log_proba of the final estimator

Parameters
----------
X : iterable
    Data to predict on. Must fulfill input requirements of first step
    of the pipeline.

Returns
-------
y_score : array-like of shape (n_samples, n_classes)
*)

val predict_proba : x:Ndarray.t -> t -> Ndarray.t
(**
Apply transforms, and predict_proba of the final estimator

Parameters
----------
X : iterable
    Data to predict on. Must fulfill input requirements of first step
    of the pipeline.

Returns
-------
y_proba : array-like of shape (n_samples, n_classes)
*)

val score : ?y:Ndarray.t -> ?sample_weight:Ndarray.t -> x:Ndarray.t -> t -> float
(**
Apply transforms, and score with the final estimator

Parameters
----------
X : iterable
    Data to predict on. Must fulfill input requirements of first step
    of the pipeline.

y : iterable, default=None
    Targets used for scoring. Must fulfill label requirements for all
    steps of the pipeline.

sample_weight : array-like, default=None
    If not None, this argument is passed as ``sample_weight`` keyword
    argument to the ``score`` method of the final estimator.

Returns
-------
score : float
*)

val score_samples : x:Ndarray.t -> t -> Ndarray.t
(**
Apply transforms, and score_samples of the final estimator.

Parameters
----------
X : iterable
    Data to predict on. Must fulfill input requirements of first step
    of the pipeline.

Returns
-------
y_score : ndarray, shape (n_samples,)
*)

val set_params : ?kwargs:(string * Py.Object.t) list -> t -> t
(**
Set the parameters of this estimator.

Valid parameter keys can be listed with ``get_params()``.

Returns
-------
self
*)


(** Attribute named_steps: see constructor for documentation *)
val named_steps : t -> Py.Object.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module TransformerMixin : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : unit -> t
(**
Mixin class for all transformers in scikit-learn.
*)

val fit_transform : ?y:Ndarray.t -> ?fit_params:(string * Py.Object.t) list -> x:Ndarray.t -> t -> Ndarray.t
(**
Fit to data, then transform it.

Fits transformer to X and y with optional parameters fit_params
and returns a transformed version of X.

Parameters
----------
X : numpy array of shape [n_samples, n_features]
    Training set.

y : numpy array of shape [n_samples]
    Target values.

**fit_params : dict
    Additional fit parameters.

Returns
-------
X_new : numpy array of shape [n_samples, n_features_new]
    Transformed array.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val check_memory : memory:[`String of string | `JoblibMemory of Py.Object.t | `None] -> unit -> Py.Object.t
(**
Check that ``memory`` is joblib.Memory-like.

joblib.Memory-like means that ``memory`` can be converted into a
joblib.Memory instance (typically a str denoting the ``location``)
or has the same interface (has a ``cache`` method).

Parameters
----------
memory : None, str or object with the joblib.Memory interface

Returns
-------
memory : object with the joblib.Memory interface

Raises
------
ValueError
    If ``memory`` is not joblib.Memory-like.
*)

val clone : ?safe:bool -> estimator:[`Estimator of Py.Object.t | `ArrayLike of Py.Object.t | `PyObject of Py.Object.t] -> unit -> Py.Object.t
(**
Constructs a new estimator with the same parameters.

Clone does a deep copy of the model in an estimator
without actually copying attached data. It yields a new estimator
with the same parameters that has not been fit on any data.

Parameters
----------
estimator : estimator object, or list, tuple or set of objects
    The estimator or group of estimators to be cloned

safe : boolean, optional
    If safe is false, clone will fall back to a deep copy on objects
    that are not estimators.
*)

val delayed : ?check_pickle:Py.Object.t -> function_:Py.Object.t -> unit -> Py.Object.t
(**
Decorator used to capture the arguments of a function.
*)

val if_delegate_has_method : delegate:[`String of string | `StringList of string list | `PyObject of Py.Object.t] -> unit -> Py.Object.t
(**
Create a decorator for methods that are delegated to a sub-estimator

This enables ducktyping by hasattr returning True according to the
sub-estimator.

Parameters
----------
delegate : string, list of strings or tuple of strings
    Name of the sub-estimator that can be accessed as an attribute of the
    base object. If a list or a tuple of names are provided, the first
    sub-estimator that is an attribute of the base object will be used.
*)

val make_pipeline : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> Pipeline.t
(**
Construct a Pipeline from the given estimators.

This is a shorthand for the Pipeline constructor; it does not require, and
does not permit, naming the estimators. Instead, their names will be set
to the lowercase of their types automatically.

Parameters
----------
*steps : list of estimators.

memory : None, str or object with the joblib.Memory interface, optional
    Used to cache the fitted transformers of the pipeline. By default,
    no caching is performed. If a string is given, it is the path to
    the caching directory. Enabling caching triggers a clone of
    the transformers before fitting. Therefore, the transformer
    instance given to the pipeline cannot be inspected
    directly. Use the attribute ``named_steps`` or ``steps`` to
    inspect estimators within the pipeline. Caching the
    transformers is advantageous when fitting is time consuming.

verbose : boolean, default=False
    If True, the time elapsed while fitting each step will be printed as it
    is completed.

See Also
--------
sklearn.pipeline.Pipeline : Class for creating a pipeline of
    transforms with a final estimator.

Examples
--------
>>> from sklearn.naive_bayes import GaussianNB
>>> from sklearn.preprocessing import StandardScaler
>>> make_pipeline(StandardScaler(), GaussianNB(priors=None))
Pipeline(steps=[('standardscaler', StandardScaler()),
                ('gaussiannb', GaussianNB())])

Returns
-------
p : Pipeline
*)

val make_union : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> FeatureUnion.t
(**
Construct a FeatureUnion from the given transformers.

This is a shorthand for the FeatureUnion constructor; it does not require,
and does not permit, naming the transformers. Instead, they will be given
names automatically based on their types. It also does not allow weighting.

Parameters
----------
*transformers : list of estimators

n_jobs : int or None, optional (default=None)
    Number of jobs to run in parallel.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

verbose : boolean, optional(default=False)
    If True, the time elapsed while fitting each transformer will be
    printed as it is completed.

Returns
-------
f : FeatureUnion

See Also
--------
sklearn.pipeline.FeatureUnion : Class for concatenating the results
    of multiple transformer objects.

Examples
--------
>>> from sklearn.decomposition import PCA, TruncatedSVD
>>> from sklearn.pipeline import make_union
>>> make_union(PCA(), TruncatedSVD())
 FeatureUnion(transformer_list=[('pca', PCA()),
                               ('truncatedsvd', TruncatedSVD())])
*)

