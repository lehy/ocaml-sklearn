(** Get an attribute of this module as a Py.Object.t. This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module BaseEstimator : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : unit -> t
(**
Base class for all estimators in scikit-learn

Notes
-----
All estimators should specify all the parameters that can be set
at the class level in their ``__init__`` as explicit keyword
arguments (no ``*args`` or ``**kwargs``).
*)

val get_params : ?deep:bool -> t -> Dict.t
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

val set_params : ?params:(string * Py.Object.t) list -> t -> t
(**
Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The latter have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

Parameters
----------
**params : dict
    Estimator parameters.

Returns
-------
self : object
    Estimator instance.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module ClassifierMixin : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : unit -> t
(**
Mixin class for all classifiers in scikit-learn.
*)

val score : ?sample_weight:Arr.t -> x:Arr.t -> y:Arr.t -> t -> float
(**
Return the mean accuracy on the given test data and labels.

In multi-label classification, this is the subset accuracy
which is a harsh metric since you require for each sample that
each label set be correctly predicted.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Test samples.

y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    True labels for X.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

Returns
-------
score : float
    Mean accuracy of self.predict(X) wrt. y.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module DummyClassifier : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?strategy:string -> ?random_state:int -> ?constant:[`I of int | `S of string | `Arr of Arr.t] -> unit -> t
(**
DummyClassifier is a classifier that makes predictions using simple rules.

This classifier is useful as a simple baseline to compare with other
(real) classifiers. Do not use it for real problems.

Read more in the :ref:`User Guide <dummy_estimators>`.

.. versionadded:: 0.13

Parameters
----------
strategy : str, default="stratified"
    Strategy to use to generate predictions.

    * "stratified": generates predictions by respecting the training
      set's class distribution.
    * "most_frequent": always predicts the most frequent label in the
      training set.
    * "prior": always predicts the class that maximizes the class prior
      (like "most_frequent") and ``predict_proba`` returns the class prior.
    * "uniform": generates predictions uniformly at random.
    * "constant": always predicts a constant label that is provided by
      the user. This is useful for metrics that evaluate a non-majority
      class

      .. versionchanged:: 0.22
         The default value of `strategy` will change to "prior" in version
         0.24. Starting from version 0.22, a warning will be raised if
         `strategy` is not explicitly set.

      .. versionadded:: 0.17
         Dummy Classifier now supports prior fitting strategy using
         parameter *prior*.

random_state : int, RandomState instance or None, optional, default=None
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

constant : int or str or array-like of shape (n_outputs,)
    The explicit constant as predicted by the "constant" strategy. This
    parameter is useful only for the "constant" strategy.

Attributes
----------
classes_ : array or list of array of shape (n_classes,)
    Class labels for each output.

n_classes_ : array or list of array of shape (n_classes,)
    Number of label for each output.

class_prior_ : array or list of array of shape (n_classes,)
    Probability of each class for each output.

n_outputs_ : int,
    Number of outputs.

sparse_output_ : bool,
    True if the array returned from predict is to be in sparse CSC format.
    Is automatically set to True if the input y is passed in sparse format.

Examples
--------
>>> import numpy as np
>>> from sklearn.dummy import DummyClassifier
>>> X = np.array([-1, 1, 1, 1])
>>> y = np.array([0, 1, 1, 1])
>>> dummy_clf = DummyClassifier(strategy="most_frequent")
>>> dummy_clf.fit(X, y)
DummyClassifier(strategy='most_frequent')
>>> dummy_clf.predict(X)
array([1, 1, 1, 1])
>>> dummy_clf.score(X, y)
0.75
*)

val fit : ?sample_weight:Arr.t -> x:[`Arr of Arr.t | `Object_with_finite_shape of Py.Object.t] -> y:Arr.t -> t -> t
(**
Fit the random classifier.

Parameters
----------
X : {array-like, object with finite length or shape}
    Training data, requires length = n_samples

y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    Target values.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

Returns
-------
self : object
*)

val get_params : ?deep:bool -> t -> Dict.t
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

val predict : x:Arr.t -> t -> Arr.t
(**
Perform classification on test vectors X.

Parameters
----------
X : {array-like, object with finite length or shape}
    Training data, requires length = n_samples

Returns
-------
y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    Predicted target values for X.
*)

val predict_log_proba : x:[`Arr of Arr.t | `Object_with_finite_shape of Py.Object.t] -> t -> Py.Object.t
(**
Return log probability estimates for the test vectors X.

Parameters
----------
X : {array-like, object with finite length or shape}
    Training data, requires length = n_samples

Returns
-------
P : array-like or list of array-like of shape (n_samples, n_classes)
    Returns the log probability of the sample for each class in
    the model, where classes are ordered arithmetically for each
    output.
*)

val predict_proba : x:Arr.t -> t -> Arr.t
(**
Return probability estimates for the test vectors X.

Parameters
----------
X : {array-like, object with finite length or shape}
    Training data, requires length = n_samples

Returns
-------
P : array-like or list of array-lke of shape (n_samples, n_classes)
    Returns the probability of the sample for each class in
    the model, where classes are ordered arithmetically, for each
    output.
*)

val score : ?sample_weight:Arr.t -> x:[`Arr of Arr.t | `None] -> y:Arr.t -> t -> float
(**
Returns the mean accuracy on the given test data and labels.

In multi-label classification, this is the subset accuracy
which is a harsh metric since you require for each sample that
each label set be correctly predicted.

Parameters
----------
X : {array-like, None}
    Test samples with shape = (n_samples, n_features) or
    None. Passing None as test samples gives the same result
    as passing real test samples, since DummyClassifier
    operates independently of the sampled observations.

y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    True labels for X.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

Returns
-------
score : float
    Mean accuracy of self.predict(X) wrt. y.
*)

val set_params : ?params:(string * Py.Object.t) list -> t -> t
(**
Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The latter have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

Parameters
----------
**params : dict
    Estimator parameters.

Returns
-------
self : object
    Estimator instance.
*)


(** Attribute classes_: get value or raise Not_found if None.*)
val classes_ : t -> Arr.t

(** Attribute classes_: get value as an option. *)
val classes_opt : t -> (Arr.t) option


(** Attribute n_classes_: get value or raise Not_found if None.*)
val n_classes_ : t -> Py.Object.t

(** Attribute n_classes_: get value as an option. *)
val n_classes_opt : t -> (Py.Object.t) option


(** Attribute class_prior_: get value or raise Not_found if None.*)
val class_prior_ : t -> Py.Object.t

(** Attribute class_prior_: get value as an option. *)
val class_prior_opt : t -> (Py.Object.t) option


(** Attribute n_outputs_: get value or raise Not_found if None.*)
val n_outputs_ : t -> int

(** Attribute n_outputs_: get value as an option. *)
val n_outputs_opt : t -> (int) option


(** Attribute sparse_output_: get value or raise Not_found if None.*)
val sparse_output_ : t -> bool

(** Attribute sparse_output_: get value as an option. *)
val sparse_output_opt : t -> (bool) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module DummyRegressor : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?strategy:string -> ?constant:[`I of int | `F of float | `Arr of Arr.t] -> ?quantile:Py.Object.t -> unit -> t
(**
DummyRegressor is a regressor that makes predictions using
simple rules.

This regressor is useful as a simple baseline to compare with other
(real) regressors. Do not use it for real problems.

Read more in the :ref:`User Guide <dummy_estimators>`.

.. versionadded:: 0.13

Parameters
----------
strategy : str
    Strategy to use to generate predictions.

    * "mean": always predicts the mean of the training set
    * "median": always predicts the median of the training set
    * "quantile": always predicts a specified quantile of the training set,
      provided with the quantile parameter.
    * "constant": always predicts a constant value that is provided by
      the user.

constant : int or float or array-like of shape (n_outputs,)
    The explicit constant as predicted by the "constant" strategy. This
    parameter is useful only for the "constant" strategy.

quantile : float in [0.0, 1.0]
    The quantile to predict using the "quantile" strategy. A quantile of
    0.5 corresponds to the median, while 0.0 to the minimum and 1.0 to the
    maximum.

Attributes
----------
constant_ : array, shape (1, n_outputs)
    Mean or median or quantile of the training targets or constant value
    given by the user.

n_outputs_ : int,
    Number of outputs.

Examples
--------
>>> import numpy as np
>>> from sklearn.dummy import DummyRegressor
>>> X = np.array([1.0, 2.0, 3.0, 4.0])
>>> y = np.array([2.0, 3.0, 5.0, 10.0])
>>> dummy_regr = DummyRegressor(strategy="mean")
>>> dummy_regr.fit(X, y)
DummyRegressor()
>>> dummy_regr.predict(X)
array([5., 5., 5., 5.])
>>> dummy_regr.score(X, y)
0.0
*)

val fit : ?sample_weight:Arr.t -> x:[`Arr of Arr.t | `Object_with_finite_shape of Py.Object.t] -> y:Arr.t -> t -> t
(**
Fit the random regressor.

Parameters
----------
X : {array-like, object with finite length or shape}
    Training data, requires length = n_samples

y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    Target values.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

Returns
-------
self : object
*)

val get_params : ?deep:bool -> t -> Dict.t
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

val predict : ?return_std:bool -> x:Arr.t -> t -> Arr.t
(**
Perform classification on test vectors X.

Parameters
----------
X : {array-like, object with finite length or shape}
    Training data, requires length = n_samples

return_std : boolean, optional
    Whether to return the standard deviation of posterior prediction.
    All zeros in this case.

Returns
-------
y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    Predicted target values for X.

y_std : array-like of shape (n_samples,) or (n_samples, n_outputs)
    Standard deviation of predictive distribution of query points.
*)

val score : ?sample_weight:Arr.t -> x:[`Arr of Arr.t | `None] -> y:Arr.t -> t -> float
(**
Returns the coefficient of determination R^2 of the prediction.

The coefficient R^2 is defined as (1 - u/v), where u is the residual
sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
sum of squares ((y_true - y_true.mean()) ** 2).sum().
The best possible score is 1.0 and it can be negative (because the
model can be arbitrarily worse). A constant model that always
predicts the expected value of y, disregarding the input features,
would get a R^2 score of 0.0.

Parameters
----------
X : {array-like, None}
    Test samples with shape = (n_samples, n_features) or None.
    For some estimators this may be a
    precomputed kernel matrix instead, shape = (n_samples,
    n_samples_fitted], where n_samples_fitted is the number of
    samples used in the fitting for the estimator.
    Passing None as test samples gives the same result
    as passing real test samples, since DummyRegressor
    operates independently of the sampled observations.

y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    True values for X.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

Returns
-------
score : float
    R^2 of self.predict(X) wrt. y.
*)

val set_params : ?params:(string * Py.Object.t) list -> t -> t
(**
Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The latter have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

Parameters
----------
**params : dict
    Estimator parameters.

Returns
-------
self : object
    Estimator instance.
*)


(** Attribute constant_: get value or raise Not_found if None.*)
val constant_ : t -> Arr.t

(** Attribute constant_: get value as an option. *)
val constant_opt : t -> (Arr.t) option


(** Attribute n_outputs_: get value or raise Not_found if None.*)
val n_outputs_ : t -> int

(** Attribute n_outputs_: get value as an option. *)
val n_outputs_opt : t -> (int) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MultiOutputMixin : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : unit -> t
(**
Mixin to mark estimators that support multioutput.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module RegressorMixin : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : unit -> t
(**
Mixin class for all regression estimators in scikit-learn.
*)

val score : ?sample_weight:Arr.t -> x:Arr.t -> y:Arr.t -> t -> float
(**
Return the coefficient of determination R^2 of the prediction.

The coefficient R^2 is defined as (1 - u/v), where u is the residual
sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
sum of squares ((y_true - y_true.mean()) ** 2).sum().
The best possible score is 1.0 and it can be negative (because the
model can be arbitrarily worse). A constant model that always
predicts the expected value of y, disregarding the input features,
would get a R^2 score of 0.0.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Test samples. For some estimators this may be a
    precomputed kernel matrix or a list of generic objects instead,
    shape = (n_samples, n_samples_fitted),
    where n_samples_fitted is the number of
    samples used in the fitting for the estimator.

y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    True values for X.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

Returns
-------
score : float
    R^2 of self.predict(X) wrt. y.

Notes
-----
The R2 score used when calling ``score`` on a regressor will use
``multioutput='uniform_average'`` from version 0.23 to keep consistent
with :func:`~sklearn.metrics.r2_score`. This will influence the
``score`` method of all the multioutput regressors (except for
:class:`~sklearn.multioutput.MultiOutputRegressor`). To specify the
default value manually and avoid the warning, please either call
:func:`~sklearn.metrics.r2_score` directly or make a custom scorer with
:func:`~sklearn.metrics.make_scorer` (the built-in scorer ``'r2'`` uses
``multioutput='uniform_average'``).
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val check_array : ?accept_sparse:[`S of string | `Bool of bool | `StringList of string list] -> ?accept_large_sparse:bool -> ?dtype:[`S of string | `Dtype of Py.Object.t | `TypeList of Py.Object.t | `None] -> ?order:[`F | `C] -> ?copy:bool -> ?force_all_finite:[`Bool of bool | `Allow_nan] -> ?ensure_2d:bool -> ?allow_nd:bool -> ?ensure_min_samples:int -> ?ensure_min_features:int -> ?warn_on_dtype:bool -> ?estimator:[`S of string | `Estimator of Py.Object.t] -> array:Py.Object.t -> unit -> Py.Object.t
(**
Input validation on an array, list, sparse matrix or similar.

By default, the input is checked to be a non-empty 2D array containing
only finite values. If the dtype of the array is object, attempt
converting to float, raising on failure.

Parameters
----------
array : object
    Input object to check / convert.

accept_sparse : string, boolean or list/tuple of strings (default=False)
    String[s] representing allowed sparse matrix formats, such as 'csc',
    'csr', etc. If the input is sparse but not in the allowed format,
    it will be converted to the first listed format. True allows the input
    to be any format. False means that a sparse matrix input will
    raise an error.

accept_large_sparse : bool (default=True)
    If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
    accept_sparse, accept_large_sparse=False will cause it to be accepted
    only if its indices are stored with a 32-bit dtype.

    .. versionadded:: 0.20

dtype : string, type, list of types or None (default="numeric")
    Data type of result. If None, the dtype of the input is preserved.
    If "numeric", dtype is preserved unless array.dtype is object.
    If dtype is a list of types, conversion on the first type is only
    performed if the dtype of the input is not in the list.

order : 'F', 'C' or None (default=None)
    Whether an array will be forced to be fortran or c-style.
    When order is None (default), then if copy=False, nothing is ensured
    about the memory layout of the output array; otherwise (copy=True)
    the memory layout of the returned array is kept as close as possible
    to the original array.

copy : boolean (default=False)
    Whether a forced copy will be triggered. If copy=False, a copy might
    be triggered by a conversion.

force_all_finite : boolean or 'allow-nan', (default=True)
    Whether to raise an error on np.inf and np.nan in array. The
    possibilities are:

    - True: Force all values of array to be finite.
    - False: accept both np.inf and np.nan in array.
    - 'allow-nan': accept only np.nan values in array. Values cannot
      be infinite.

    For object dtyped data, only np.nan is checked and not np.inf.

    .. versionadded:: 0.20
       ``force_all_finite`` accepts the string ``'allow-nan'``.

ensure_2d : boolean (default=True)
    Whether to raise a value error if array is not 2D.

allow_nd : boolean (default=False)
    Whether to allow array.ndim > 2.

ensure_min_samples : int (default=1)
    Make sure that the array has a minimum number of samples in its first
    axis (rows for a 2D array). Setting to 0 disables this check.

ensure_min_features : int (default=1)
    Make sure that the 2D array has some minimum number of features
    (columns). The default value of 1 rejects empty datasets.
    This check is only enforced when the input data has effectively 2
    dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
    disables this check.

warn_on_dtype : boolean or None, optional (default=None)
    Raise DataConversionWarning if the dtype of the input data structure
    does not match the requested dtype, causing a memory copy.

    .. deprecated:: 0.21
        ``warn_on_dtype`` is deprecated in version 0.21 and will be
        removed in 0.23.

estimator : str or estimator instance (default=None)
    If passed, include the name of the estimator in warning messages.

Returns
-------
array_converted : object
    The converted and validated array.
*)

val check_consistent_length : Py.Object.t list -> Py.Object.t
(**
Check that all arrays have consistent first dimensions.

Checks whether all objects in arrays have the same shape or length.

Parameters
----------
*arrays : list or tuple of input objects.
    Objects that will be checked for consistent length.
*)

val check_is_fitted : ?attributes:[`S of string | `Arr of Arr.t | `StringList of string list] -> ?msg:string -> ?all_or_any:[`Callable of Py.Object.t | `PyObject of Py.Object.t] -> estimator:Py.Object.t -> unit -> Py.Object.t
(**
Perform is_fitted validation for estimator.

Checks if the estimator is fitted by verifying the presence of
fitted attributes (ending with a trailing underscore) and otherwise
raises a NotFittedError with the given message.

This utility is meant to be used internally by estimators themselves,
typically in their own predict / transform methods.

Parameters
----------
estimator : estimator instance.
    estimator instance for which the check is performed.

attributes : str, list or tuple of str, default=None
    Attribute name(s) given as string or a list/tuple of strings
    Eg.: ``["coef_", "estimator_", ...], "coef_"``

    If `None`, `estimator` is considered fitted if there exist an
    attribute that ends with a underscore and does not start with double
    underscore.

msg : string
    The default error message is, "This %(name)s instance is not fitted
    yet. Call 'fit' with appropriate arguments before using this
    estimator."

    For custom messages if "%(name)s" is present in the message string,
    it is substituted for the estimator name.

    Eg. : "Estimator, %(name)s, must be fitted before sparsifying".

all_or_any : callable, {all, any}, default all
    Specify whether all or any of the given attributes must exist.

Returns
-------
None

Raises
------
NotFittedError
    If the attributes are not found.
*)

val check_random_state : seed:[`I of int | `RandomState of Py.Object.t | `None] -> unit -> Py.Object.t
(**
Turn seed into a np.random.RandomState instance

Parameters
----------
seed : None | int | instance of RandomState
    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
*)

val class_distribution : ?sample_weight:Arr.t -> y:[`Arr of Arr.t | `PyObject of Py.Object.t] -> unit -> (Py.Object.t * Py.Object.t * Py.Object.t)
(**
Compute class priors from multioutput-multiclass target data

Parameters
----------
y : array like or sparse matrix of size (n_samples, n_outputs)
    The labels for each example.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

Returns
-------
classes : list of size n_outputs of arrays of size (n_classes,)
    List of classes for each column.

n_classes : list of integers of size n_outputs
    Number of classes in each column

class_prior : list of size n_outputs of arrays of size (n_classes,)
    Class distribution of each column.
*)

module Deprecated : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?extra:string -> unit -> t
(**
Decorator to mark a function or class as deprecated.

Issue a warning when the function is called/the class is instantiated and
adds a warning to the docstring.

The optional extra argument will be appended to the deprecation message
and the docstring. Note: to use this with the default value for extra, put
in an empty of parentheses:

>>> from sklearn.utils import deprecated
>>> deprecated()
<sklearn.utils.deprecation.deprecated object at ...>

>>> @deprecated()
... def some_function(): pass

Parameters
----------
extra : string
      to be added to the deprecation messages
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

