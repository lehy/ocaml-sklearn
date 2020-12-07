(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module Bunch : sig
type tag = [`Bunch]
type t = [`Bunch | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?kwargs:(string * Py.Object.t) list -> unit -> t
(**
Container object exposing keys as attributes

Bunch objects are sometimes used as an output for functions and methods.
They extend dictionaries by enabling values to be accessed by key,
`bunch['value_key']`, or by an attribute, `bunch.value_key`.

Examples
--------
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
type tag = [`FeatureUnion]
type t = [`BaseEstimator | `FeatureUnion | `Object | `TransformerMixin] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_transformer : t -> [`TransformerMixin] Obj.t
val as_estimator : t -> [`BaseEstimator] Obj.t
val create : ?n_jobs:int -> ?transformer_weights:Dict.t -> ?verbose:int -> transformer_list:(string * [>`TransformerMixin] Np.Obj.t) list -> unit -> t
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

n_jobs : int, default=None
    Number of jobs to run in parallel.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

    .. versionchanged:: v0.20
       `n_jobs` default changed from 1 to None

transformer_weights : dict, default=None
    Multiplicative weights for features per transformer.
    Keys are transformer names, values the weights.

verbose : bool, default=False
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
>>> union = FeatureUnion([('pca', PCA(n_components=1)),
...                       ('svd', TruncatedSVD(n_components=2))])
>>> X = [[0., 1., 3], [2., 2., 5]]
>>> union.fit_transform(X)
array([[ 1.5       ,  3.0...,  0.8...],
       [-1.5       ,  5.7..., -0.4...]])
*)

val fit : ?y:[>`ArrayLike] Np.Obj.t -> ?fit_params:(string * Py.Object.t) list -> x:Py.Object.t -> [> tag] Obj.t -> t
(**
Fit all transformers using X.

Parameters
----------
X : iterable or array-like, depending on transformers
    Input data, used to fit transformers.

y : array-like of shape (n_samples, n_outputs), default=None
    Targets for supervised learning.

Returns
-------
self : FeatureUnion
    This estimator
*)

val fit_transform : ?y:[>`ArrayLike] Np.Obj.t -> ?fit_params:(string * Py.Object.t) list -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Fit all transformers, transform the data and concatenate results.

Parameters
----------
X : iterable or array-like, depending on transformers
    Input data to be transformed.

y : array-like of shape (n_samples, n_outputs), default=None
    Targets for supervised learning.

Returns
-------
X_t : array-like or sparse matrix of                 shape (n_samples, sum_n_components)
    hstack of results of transformers. sum_n_components is the
    sum of n_components (output dimension) over transformers.
*)

val get_feature_names : [> tag] Obj.t -> string list
(**
Get feature names from all transformers.

Returns
-------
feature_names : list of strings
    Names of the features produced by transform.
*)

val get_params : ?deep:bool -> [> tag] Obj.t -> Dict.t
(**
Get parameters for this estimator.

Parameters
----------
deep : bool, default=True
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

Returns
-------
params : mapping of string to any
    Parameter names mapped to their values.
*)

val set_params : ?kwargs:(string * Py.Object.t) list -> [> tag] Obj.t -> t
(**
Set the parameters of this estimator.

Valid parameter keys can be listed with ``get_params()``.

Returns
-------
self
*)

val transform : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Transform X separately by each transformer, concatenate results.

Parameters
----------
X : iterable or array-like, depending on transformers
    Input data to be transformed.

Returns
-------
X_t : array-like or sparse matrix of                 shape (n_samples, sum_n_components)
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

module Pipeline : sig
type tag = [`Pipeline]
type t = [`BaseEstimator | `Object | `Pipeline] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_estimator : t -> [`BaseEstimator] Obj.t
val create : ?memory:[`S of string | `Joblib_Memory of Py.Object.t] -> ?verbose:bool -> steps:(string * [>`BaseEstimator] Np.Obj.t) list -> unit -> t
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

memory : str or object with the joblib.Memory interface, default=None
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
named_steps : :class:`~sklearn.utils.Bunch`
    Dictionary-like object, with the following attributes.
    Read-only attribute to access any step parameter by user given name.
    Keys are step names and values are steps parameters.

See Also
--------
sklearn.pipeline.make_pipeline : Convenience function for simplified
    pipeline construction.

Examples
--------
>>> from sklearn.svm import SVC
>>> from sklearn.preprocessing import StandardScaler
>>> from sklearn.datasets import make_classification
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.pipeline import Pipeline
>>> X, y = make_classification(random_state=0)
>>> X_train, X_test, y_train, y_test = train_test_split(X, y,
...                                                     random_state=0)
>>> pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
>>> # The pipeline can be used as any other estimator
>>> # and avoids leaking the test set into the train set
>>> pipe.fit(X_train, y_train)
Pipeline(steps=[('scaler', StandardScaler()), ('svc', SVC())])
>>> pipe.score(X_test, y_test)
0.88
*)

val get_item : ind:[`I of int | `S of string | `Slice of Np.Wrap_utils.Slice.t] -> [> tag] Obj.t -> Py.Object.t
(**
Returns a sub-pipeline or a single esimtator in the pipeline

Indexing with an integer will return an estimator; using a slice
returns another Pipeline instance which copies a slice of this
Pipeline. This copy is shallow: modifying (or fitting) estimators in
the sub-pipeline will affect the larger pipeline and vice-versa.
However, replacing a value in `step` will not affect a copy.
*)

val decision_function : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
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

val fit : ?y:[>`ArrayLike] Np.Obj.t -> ?fit_params:(string * Py.Object.t) list -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
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

val fit_predict : ?y:[>`ArrayLike] Np.Obj.t -> ?fit_params:(string * Py.Object.t) list -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
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

val fit_transform : ?y:[>`ArrayLike] Np.Obj.t -> ?fit_params:(string * Py.Object.t) list -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
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

val get_params : ?deep:bool -> [> tag] Obj.t -> Dict.t
(**
Get parameters for this estimator.

Parameters
----------
deep : bool, default=True
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

Returns
-------
params : mapping of string to any
    Parameter names mapped to their values.
*)

val inverse_transform : ?x:[>`ArrayLike] Np.Obj.t -> ?y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
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

val predict : ?predict_params:(string * Py.Object.t) list -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
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

    .. versionadded:: 0.20

Returns
-------
y_pred : array-like
*)

val predict_log_proba : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
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

val predict_proba : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
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

val score : ?y:[>`ArrayLike] Np.Obj.t -> ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> float
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

val score_samples : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Apply transforms, and score_samples of the final estimator.

Parameters
----------
X : iterable
    Data to predict on. Must fulfill input requirements of first step
    of the pipeline.

Returns
-------
y_score : ndarray of shape (n_samples,)
*)

val set_params : ?kwargs:(string * Py.Object.t) list -> [> tag] Obj.t -> t
(**
Set the parameters of this estimator.

Valid parameter keys can be listed with ``get_params()``.

Returns
-------
self
*)


(** Attribute named_steps: get value or raise Not_found if None.*)
val named_steps : t -> Dict.t

(** Attribute named_steps: get value as an option. *)
val named_steps_opt : t -> (Dict.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Islice : sig
type tag = [`Islice]
type t = [`Islice | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : iterable:Py.Object.t -> stop:Py.Object.t -> unit -> t
(**
islice(iterable, stop) --> islice object
islice(iterable, start, stop[, step]) --> islice object

Return an iterator whose next() method returns selected values from an
iterable.  If start is specified, will skip all preceding elements;
otherwise, start defaults to zero.  Step defaults to one.  If
specified as another value, step determines how many values are
skipped between successive calls.  Works like a slice() on a list
but returns an iterator.
*)

val iter : [> tag] Obj.t -> Dict.t Seq.t
(**
Implement iter(self).
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val check_memory : [`S of string | `Object_with_the_joblib_Memory_interface of Py.Object.t | `None] -> Py.Object.t
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

val clone : ?safe:bool -> estimator:[>`BaseEstimator] Np.Obj.t -> unit -> Py.Object.t
(**
Constructs a new estimator with the same parameters.

Clone does a deep copy of the model in an estimator
without actually copying attached data. It yields a new estimator
with the same parameters that has not been fit on any data.

Parameters
----------
estimator : {list, tuple, set} of estimator objects or estimator object
    The estimator or group of estimators to be cloned.

safe : bool, default=True
    If safe is false, clone will fall back to a deep copy on objects
    that are not estimators.
*)

val delayed : ?check_pickle:Py.Object.t -> function_:Py.Object.t -> unit -> Py.Object.t
(**
Decorator used to capture the arguments of a function.
*)

val if_delegate_has_method : [`S of string | `StringList of string list] -> Py.Object.t
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

val make_pipeline : ?kwargs:(string * Py.Object.t) list -> [>`BaseEstimator] Np.Obj.t list -> Pipeline.t
(**
Construct a Pipeline from the given estimators.

This is a shorthand for the Pipeline constructor; it does not require, and
does not permit, naming the estimators. Instead, their names will be set
to the lowercase of their types automatically.

Parameters
----------
*steps : list of estimators.

memory : str or object with the joblib.Memory interface, default=None
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

val make_union : ?kwargs:(string * Py.Object.t) list -> [>`BaseEstimator] Np.Obj.t list -> FeatureUnion.t
(**
Construct a FeatureUnion from the given transformers.

This is a shorthand for the FeatureUnion constructor; it does not require,
and does not permit, naming the transformers. Instead, they will be given
names automatically based on their types. It also does not allow weighting.

Parameters
----------
*transformers : list of estimators

n_jobs : int, default=None
    Number of jobs to run in parallel.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

    .. versionchanged:: v0.20
       `n_jobs` default changed from 1 to None

verbose : bool, default=False
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

