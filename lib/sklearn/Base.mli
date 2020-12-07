(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module BaseEstimator : sig
type tag = [`BaseEstimator]
type t = [`BaseEstimator | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : unit -> t
(**
Base class for all estimators in scikit-learn

Notes
-----
All estimators should specify all the parameters that can be set
at the class level in their ``__init__`` as explicit keyword
arguments (no ``*args`` or ``**kwargs``).
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

val set_params : ?params:(string * Py.Object.t) list -> [> tag] Obj.t -> t
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

module BiclusterMixin : sig
type tag = [`BiclusterMixin]
type t = [`BiclusterMixin | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : unit -> t
(**
Mixin class for all bicluster estimators in scikit-learn
*)

val get_indices : i:int -> [> tag] Obj.t -> (Py.Object.t * Py.Object.t)
(**
Row and column indices of the i'th bicluster.

Only works if ``rows_`` and ``columns_`` attributes exist.

Parameters
----------
i : int
    The index of the cluster.

Returns
-------
row_ind : ndarray, dtype=np.intp
    Indices of rows in the dataset that belong to the bicluster.
col_ind : ndarray, dtype=np.intp
    Indices of columns in the dataset that belong to the bicluster.
*)

val get_shape : i:int -> [> tag] Obj.t -> Py.Object.t
(**
Shape of the i'th bicluster.

Parameters
----------
i : int
    The index of the cluster.

Returns
-------
shape : tuple (int, int)
    Number of rows and columns (resp.) in the bicluster.
*)

val get_submatrix : i:int -> data:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Return the submatrix corresponding to bicluster `i`.

Parameters
----------
i : int
    The index of the cluster.
data : array-like
    The data.

Returns
-------
submatrix : ndarray
    The submatrix corresponding to bicluster i.

Notes
-----
Works with sparse matrices. Only works if ``rows_`` and
``columns_`` attributes exist.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module ClassifierMixin : sig
type tag = [`ClassifierMixin]
type t = [`ClassifierMixin | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : unit -> t
(**
Mixin class for all classifiers in scikit-learn.
*)

val score : ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> float
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

module ClusterMixin : sig
type tag = [`ClusterMixin]
type t = [`ClusterMixin | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : unit -> t
(**
Mixin class for all cluster estimators in scikit-learn.
*)

val fit_predict : ?y:Py.Object.t -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Perform clustering on X and returns cluster labels.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Input data.

y : Ignored
    Not used, present for API consistency by convention.

Returns
-------
labels : ndarray of shape (n_samples,)
    Cluster labels.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module DensityMixin : sig
type tag = [`DensityMixin]
type t = [`DensityMixin | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : unit -> t
(**
Mixin class for all density estimators in scikit-learn.
*)

val score : ?y:Py.Object.t -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> float
(**
Return the score of the model on the data X

Parameters
----------
X : array-like of shape (n_samples, n_features)

y : Ignored
    Not used, present for API consistency by convention.

Returns
-------
score : float
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MetaEstimatorMixin : sig
type tag = [`MetaEstimatorMixin]
type t = [`MetaEstimatorMixin | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : unit -> t
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

module MultiOutputMixin : sig
type tag = [`MultiOutputMixin]
type t = [`MultiOutputMixin | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

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

module OutlierMixin : sig
type tag = [`OutlierMixin]
type t = [`Object | `OutlierMixin] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : unit -> t
(**
Mixin class for all outlier detection estimators in scikit-learn.
*)

val fit_predict : ?y:Py.Object.t -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Perform fit on X and returns labels for X.

Returns -1 for outliers and 1 for inliers.

Parameters
----------
X : {array-like, sparse matrix, dataframe} of shape             (n_samples, n_features)

y : Ignored
    Not used, present for API consistency by convention.

Returns
-------
y : ndarray of shape (n_samples,)
    1 for inliers, -1 for outliers.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module RegressorMixin : sig
type tag = [`RegressorMixin]
type t = [`Object | `RegressorMixin] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : unit -> t
(**
Mixin class for all regression estimators in scikit-learn.
*)

val score : ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> float
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
The R2 score used when calling ``score`` on a regressor uses
``multioutput='uniform_average'`` from version 0.23 to keep consistent
with default value of :func:`~sklearn.metrics.r2_score`.
This influences the ``score`` method of all the multioutput
regressors (except for
:class:`~sklearn.multioutput.MultiOutputRegressor`).
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module TransformerMixin : sig
type tag = [`TransformerMixin]
type t = [`Object | `TransformerMixin] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : unit -> t
(**
Mixin class for all transformers in scikit-learn.
*)

val fit_transform : ?y:[>`ArrayLike] Np.Obj.t -> ?fit_params:(string * Py.Object.t) list -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Fit to data, then transform it.

Fits transformer to X and y with optional parameters fit_params
and returns a transformed version of X.

Parameters
----------
X : {array-like, sparse matrix, dataframe} of shape                 (n_samples, n_features)

y : ndarray of shape (n_samples,), default=None
    Target values.

**fit_params : dict
    Additional fit parameters.

Returns
-------
X_new : ndarray array of shape (n_samples, n_features_new)
    Transformed array.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Defaultdict : sig
type tag = [`Defaultdict]
type t = [`Defaultdict | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val get_item : y:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
x.__getitem__(y) <==> x[y]
*)

val iter : [> tag] Obj.t -> Dict.t Seq.t
(**
Implement iter(self).
*)

val __setitem__ : key:Py.Object.t -> value:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Set self[key] to value.
*)

val fromkeys : ?value:Py.Object.t -> iterable:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Create a new dictionary with keys from iterable and values set to value.
*)

val get : ?default:Py.Object.t -> key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return the value for key if key is in the dictionary, else default.
*)

val pop : ?d:Py.Object.t -> k:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
D.pop(k[,d]) -> v, remove specified key and return the corresponding value.
If key is not found, d is returned if given, otherwise KeyError is raised
*)

val popitem : [> tag] Obj.t -> Py.Object.t
(**
Remove and return a (key, value) pair as a 2-tuple.

Pairs are returned in LIFO (last-in, first-out) order.
Raises KeyError if the dict is empty.
*)

val setdefault : ?default:Py.Object.t -> key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Insert key with a value of default if key is not in the dictionary.

Return the value for key if key is in the dictionary, else default.
*)

val update : ?e:Py.Object.t -> ?f:(string * Py.Object.t) list -> [> tag] Obj.t -> Py.Object.t
(**
D.update([E, ]**F) -> None.  Update D from dict/iterable E and F.
If E is present and has a .keys() method, then does:  for k in E: D[k] = E[k]
If E is present and lacks a .keys() method, then does:  for k, v in E: D[k] = v
In either case, this is followed by: for k in F:  D[k] = F[k]
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val check_X_y : ?accept_sparse:[`S of string | `StringList of string list | `Bool of bool] -> ?accept_large_sparse:bool -> ?dtype:[`S of string | `Dtype of Np.Dtype.t | `Dtypes of Np.Dtype.t list | `None] -> ?order:[`C | `F] -> ?copy:bool -> ?force_all_finite:[`Allow_nan | `Bool of bool] -> ?ensure_2d:bool -> ?allow_nd:bool -> ?multi_output:bool -> ?ensure_min_samples:int -> ?ensure_min_features:int -> ?y_numeric:bool -> ?estimator:[>`BaseEstimator] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> unit -> (Py.Object.t * Py.Object.t)
(**
Input validation for standard estimators.

Checks X and y for consistent length, enforces X to be 2D and y 1D. By
default, X is checked to be non-empty and containing only finite values.
Standard input checks are also applied to y, such as checking that y
does not have np.nan or np.inf targets. For multi-label y, set
multi_output=True to allow 2D and sparse y. If the dtype of X is
object, attempt converting to float, raising on failure.

Parameters
----------
X : nd-array, list or sparse matrix
    Input data.

y : nd-array, list or sparse matrix
    Labels.

accept_sparse : string, boolean or list of string (default=False)
    String[s] representing allowed sparse matrix formats, such as 'csc',
    'csr', etc. If the input is sparse but not in the allowed format,
    it will be converted to the first listed format. True allows the input
    to be any format. False means that a sparse matrix input will
    raise an error.

accept_large_sparse : bool (default=True)
    If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
    accept_sparse, accept_large_sparse will cause it to be accepted only
    if its indices are stored with a 32-bit dtype.

    .. versionadded:: 0.20

dtype : string, type, list of types or None (default='numeric')
    Data type of result. If None, the dtype of the input is preserved.
    If 'numeric', dtype is preserved unless array.dtype is object.
    If dtype is a list of types, conversion on the first type is only
    performed if the dtype of the input is not in the list.

order : 'F', 'C' or None (default=None)
    Whether an array will be forced to be fortran or c-style.

copy : boolean (default=False)
    Whether a forced copy will be triggered. If copy=False, a copy might
    be triggered by a conversion.

force_all_finite : boolean or 'allow-nan', (default=True)
    Whether to raise an error on np.inf, np.nan, pd.NA in X. This parameter
    does not influence whether y can have np.inf, np.nan, pd.NA values.
    The possibilities are:

    - True: Force all values of X to be finite.
    - False: accepts np.inf, np.nan, pd.NA in X.
    - 'allow-nan': accepts only np.nan or pd.NA values in X. Values cannot
      be infinite.

    .. versionadded:: 0.20
       ``force_all_finite`` accepts the string ``'allow-nan'``.

    .. versionchanged:: 0.23
       Accepts `pd.NA` and converts it into `np.nan`

ensure_2d : boolean (default=True)
    Whether to raise a value error if X is not 2D.

allow_nd : boolean (default=False)
    Whether to allow X.ndim > 2.

multi_output : boolean (default=False)
    Whether to allow 2D y (array or sparse matrix). If false, y will be
    validated as a vector. y cannot have np.nan or np.inf values if
    multi_output=True.

ensure_min_samples : int (default=1)
    Make sure that X has a minimum number of samples in its first
    axis (rows for a 2D array).

ensure_min_features : int (default=1)
    Make sure that the 2D array has some minimum number of features
    (columns). The default value of 1 rejects empty datasets.
    This check is only enforced when X has effectively 2 dimensions or
    is originally 1D and ``ensure_2d`` is True. Setting to 0 disables
    this check.

y_numeric : boolean (default=False)
    Whether to ensure that y has a numeric type. If dtype of y is object,
    it is converted to float64. Should only be used for regression
    algorithms.

estimator : str or estimator instance (default=None)
    If passed, include the name of the estimator in warning messages.

Returns
-------
X_converted : object
    The converted and validated X.

y_converted : object
    The converted and validated y.
*)

val check_array : ?accept_sparse:[`S of string | `StringList of string list | `Bool of bool] -> ?accept_large_sparse:bool -> ?dtype:[`S of string | `Dtype of Np.Dtype.t | `Dtypes of Np.Dtype.t list | `None] -> ?order:[`C | `F] -> ?copy:bool -> ?force_all_finite:[`Allow_nan | `Bool of bool] -> ?ensure_2d:bool -> ?allow_nd:bool -> ?ensure_min_samples:int -> ?ensure_min_features:int -> ?estimator:[>`BaseEstimator] Np.Obj.t -> array:Py.Object.t -> unit -> Py.Object.t
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

dtype : string, type, list of types or None (default='numeric')
    Data type of result. If None, the dtype of the input is preserved.
    If 'numeric', dtype is preserved unless array.dtype is object.
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
    Whether to raise an error on np.inf, np.nan, pd.NA in array. The
    possibilities are:

    - True: Force all values of array to be finite.
    - False: accepts np.inf, np.nan, pd.NA in array.
    - 'allow-nan': accepts only np.nan and pd.NA values in array. Values
      cannot be infinite.

    .. versionadded:: 0.20
       ``force_all_finite`` accepts the string ``'allow-nan'``.

    .. versionchanged:: 0.23
       Accepts `pd.NA` and converts it into `np.nan`

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

estimator : str or estimator instance (default=None)
    If passed, include the name of the estimator in warning messages.

Returns
-------
array_converted : object
    The converted and validated array.
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

val estimator_html_repr : [>`BaseEstimator] Np.Obj.t -> string
(**
Build a HTML representation of an estimator.

Read more in the :ref:`User Guide <visualizing_composite_estimators>`.

Parameters
----------
estimator : estimator object
    The estimator to visualize.

Returns
-------
html: str
    HTML representation of estimator.
*)

val get_config : unit -> Dict.t
(**
Retrieve current values for configuration set by :func:`set_config`

Returns
-------
config : dict
    Keys are parameter names that can be passed to :func:`set_config`.

See Also
--------
config_context: Context manager for global scikit-learn configuration
set_config: Set global scikit-learn configuration
*)

val is_classifier : [>`BaseEstimator] Np.Obj.t -> bool
(**
Return True if the given estimator is (probably) a classifier.

Parameters
----------
estimator : object
    Estimator object to test.

Returns
-------
out : bool
    True if estimator is a classifier and False otherwise.
*)

val is_outlier_detector : [>`BaseEstimator] Np.Obj.t -> bool
(**
Return True if the given estimator is (probably) an outlier detector.

Parameters
----------
estimator : object
    Estimator object to test.

Returns
-------
out : bool
    True if estimator is an outlier detector and False otherwise.
*)

val is_regressor : [>`BaseEstimator] Np.Obj.t -> bool
(**
Return True if the given estimator is (probably) a regressor.

Parameters
----------
estimator : object
    Estimator object to test.

Returns
-------
out : bool
    True if estimator is a regressor and False otherwise.
*)

