module KNNImputer : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?missing_values:[`String of string | `None | `PyObject of Py.Object.t] -> ?n_neighbors:int -> ?weights:[`Uniform | `Distance | `Callable of Py.Object.t] -> ?metric:[`Nan_euclidean | `Callable of Py.Object.t] -> ?copy:bool -> ?add_indicator:bool -> unit -> t
(**
Imputation for completing missing values using k-Nearest Neighbors.

Each sample's missing values are imputed using the mean value from
`n_neighbors` nearest neighbors found in the training set. Two samples are
close if the features that neither is missing are close.

Read more in the :ref:`User Guide <knnimpute>`.

.. versionadded:: 0.22

Parameters
----------
missing_values : number, string, np.nan or None, default=`np.nan`
    The placeholder for the missing values. All occurrences of
    `missing_values` will be imputed.

n_neighbors : int, default=5
    Number of neighboring samples to use for imputation.

weights : {'uniform', 'distance'} or callable, default='uniform'
    Weight function used in prediction.  Possible values:

    - 'uniform' : uniform weights. All points in each neighborhood are
      weighted equally.
    - 'distance' : weight points by the inverse of their distance.
      in this case, closer neighbors of a query point will have a
      greater influence than neighbors which are further away.
    - callable : a user-defined function which accepts an
      array of distances, and returns an array of the same shape
      containing the weights.

metric : {'nan_euclidean'} or callable, default='nan_euclidean'
    Distance metric for searching neighbors. Possible values:

    - 'nan_euclidean'
    - callable : a user-defined function which conforms to the definition
      of ``_pairwise_callable(X, Y, metric, **kwds)``. The function
      accepts two arrays, X and Y, and a `missing_values` keyword in
      `kwds` and returns a scalar distance value.

copy : bool, default=True
    If True, a copy of X will be created. If False, imputation will
    be done in-place whenever possible.

add_indicator : bool, default=False
    If True, a :class:`MissingIndicator` transform will stack onto the
    output of the imputer's transform. This allows a predictive estimator
    to account for missingness despite imputation. If a feature has no
    missing values at fit/train time, the feature won't appear on the
    missing indicator even if there are missing values at transform/test
    time.

Attributes
----------
indicator_ : :class:`sklearn.impute.MissingIndicator`
    Indicator used to add binary indicators for missing values.
    ``None`` if add_indicator is False.

References
----------
* Olga Troyanskaya, Michael Cantor, Gavin Sherlock, Pat Brown, Trevor
  Hastie, Robert Tibshirani, David Botstein and Russ B. Altman, Missing
  value estimation methods for DNA microarrays, BIOINFORMATICS Vol. 17
  no. 6, 2001 Pages 520-525.

Examples
--------
>>> import numpy as np
>>> from sklearn.impute import KNNImputer
>>> X = [[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]]
>>> imputer = KNNImputer(n_neighbors=2)
>>> imputer.fit_transform(X)
array([[1. , 2. , 4. ],
       [3. , 4. , 3. ],
       [5.5, 6. , 5. ],
       [8. , 8. , 7. ]])
*)

val fit : ?y:Py.Object.t -> x:Py.Object.t -> t -> t
(**
Fit the imputer on X.

Parameters
----------
X : array-like shape of (n_samples, n_features)
    Input data, where `n_samples` is the number of samples and
    `n_features` is the number of features.

Returns
-------
self : object
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

val get_params : ?deep:bool -> t -> Py.Object.t
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

val transform : x:Ndarray.t -> t -> Ndarray.t
(**
Impute all missing values in X.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The input data to complete.

Returns
-------
X : array-like of shape (n_samples, n_output_features)
    The imputed dataset. `n_output_features` is the number of features
    that is not always missing during `fit`.
*)


(** Attribute indicator_: see constructor for documentation *)
val indicator_ : t -> Py.Object.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MissingIndicator : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?missing_values:[`String of string | `PyObject of Py.Object.t] -> ?features:string -> ?sparse:[`Bool of bool | `Auto] -> ?error_on_new:bool -> unit -> t
(**
Binary indicators for missing values.

Note that this component typically should not be used in a vanilla
:class:`Pipeline` consisting of transformers and a classifier, but rather
could be added using a :class:`FeatureUnion` or :class:`ColumnTransformer`.

Read more in the :ref:`User Guide <impute>`.

Parameters
----------
missing_values : number, string, np.nan (default) or None
    The placeholder for the missing values. All occurrences of
    `missing_values` will be indicated (True in the output array), the
    other values will be marked as False.

features : str, default=None
    Whether the imputer mask should represent all or a subset of
    features.

    - If "missing-only" (default), the imputer mask will only represent
      features containing missing values during fit time.
    - If "all", the imputer mask will represent all features.

sparse : boolean or "auto", default=None
    Whether the imputer mask format should be sparse or dense.

    - If "auto" (default), the imputer mask will be of same type as
      input.
    - If True, the imputer mask will be a sparse matrix.
    - If False, the imputer mask will be a numpy array.

error_on_new : boolean, default=None
    If True (default), transform will raise an error when there are
    features with missing values in transform that have no missing values
    in fit. This is applicable only when ``features="missing-only"``.

Attributes
----------
features_ : ndarray, shape (n_missing_features,) or (n_features,)
    The features indices which will be returned when calling ``transform``.
    They are computed during ``fit``. For ``features='all'``, it is
    to ``range(n_features)``.

Examples
--------
>>> import numpy as np
>>> from sklearn.impute import MissingIndicator
>>> X1 = np.array([[np.nan, 1, 3],
...                [4, 0, np.nan],
...                [8, 1, 0]])
>>> X2 = np.array([[5, 1, np.nan],
...                [np.nan, 2, 3],
...                [2, 4, 0]])
>>> indicator = MissingIndicator()
>>> indicator.fit(X1)
MissingIndicator()
>>> X2_tr = indicator.transform(X2)
>>> X2_tr
array([[False,  True],
       [ True, False],
       [False, False]])
*)

val fit : ?y:Py.Object.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> t
(**
Fit the transformer on X.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Input data, where ``n_samples`` is the number of samples and
    ``n_features`` is the number of features.

Returns
-------
self : object
    Returns self.
*)

val fit_transform : ?y:Py.Object.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Generate missing values indicator for X.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    The input data to complete.

Returns
-------
Xt : {ndarray or sparse matrix}, shape (n_samples, n_features)         or (n_samples, n_features_with_missing)
    The missing indicator for input data. The data type of ``Xt``
    will be boolean.
*)

val get_params : ?deep:bool -> t -> Py.Object.t
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

val transform : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Generate missing values indicator for X.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    The input data to complete.

Returns
-------
Xt : {ndarray or sparse matrix}, shape (n_samples, n_features)         or (n_samples, n_features_with_missing)
    The missing indicator for input data. The data type of ``Xt``
    will be boolean.
*)


(** Attribute features_: see constructor for documentation *)
val features_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module SimpleImputer : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?missing_values:[`String of string | `PyObject of Py.Object.t] -> ?strategy:string -> ?fill_value:[`String of string | `PyObject of Py.Object.t] -> ?verbose:int -> ?copy:bool -> ?add_indicator:bool -> unit -> t
(**
Imputation transformer for completing missing values.

Read more in the :ref:`User Guide <impute>`.

Parameters
----------
missing_values : number, string, np.nan (default) or None
    The placeholder for the missing values. All occurrences of
    `missing_values` will be imputed.

strategy : string, default='mean'
    The imputation strategy.

    - If "mean", then replace missing values using the mean along
      each column. Can only be used with numeric data.
    - If "median", then replace missing values using the median along
      each column. Can only be used with numeric data.
    - If "most_frequent", then replace missing using the most frequent
      value along each column. Can be used with strings or numeric data.
    - If "constant", then replace missing values with fill_value. Can be
      used with strings or numeric data.

    .. versionadded:: 0.20
       strategy="constant" for fixed value imputation.

fill_value : string or numerical value, default=None
    When strategy == "constant", fill_value is used to replace all
    occurrences of missing_values.
    If left to the default, fill_value will be 0 when imputing numerical
    data and "missing_value" for strings or object data types.

verbose : integer, default=0
    Controls the verbosity of the imputer.

copy : boolean, default=True
    If True, a copy of X will be created. If False, imputation will
    be done in-place whenever possible. Note that, in the following cases,
    a new copy will always be made, even if `copy=False`:

    - If X is not an array of floating values;
    - If X is encoded as a CSR matrix;
    - If add_indicator=True.

add_indicator : boolean, default=False
    If True, a :class:`MissingIndicator` transform will stack onto output
    of the imputer's transform. This allows a predictive estimator
    to account for missingness despite imputation. If a feature has no
    missing values at fit/train time, the feature won't appear on
    the missing indicator even if there are missing values at
    transform/test time.

Attributes
----------
statistics_ : array of shape (n_features,)
    The imputation fill value for each feature.
    Computing statistics can result in `np.nan` values.
    During :meth:`transform`, features corresponding to `np.nan`
    statistics will be discarded.

indicator_ : :class:`sklearn.impute.MissingIndicator`
    Indicator used to add binary indicators for missing values.
    ``None`` if add_indicator is False.

See also
--------
IterativeImputer : Multivariate imputation of missing values.

Examples
--------
>>> import numpy as np
>>> from sklearn.impute import SimpleImputer
>>> imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
>>> imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])
SimpleImputer()
>>> X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
>>> print(imp_mean.transform(X))
[[ 7.   2.   3. ]
 [ 4.   3.5  6. ]
 [10.   3.5  9. ]]

Notes
-----
Columns which only contained missing values at :meth:`fit` are discarded
upon :meth:`transform` if strategy is not "constant".
*)

val fit : ?y:Py.Object.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> t
(**
Fit the imputer on X.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Input data, where ``n_samples`` is the number of samples and
    ``n_features`` is the number of features.

Returns
-------
self : SimpleImputer
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

val get_params : ?deep:bool -> t -> Py.Object.t
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

val transform : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Impute all missing values in X.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    The input data to complete.
*)


(** Attribute statistics_: see constructor for documentation *)
val statistics_ : t -> Ndarray.t

(** Attribute indicator_: see constructor for documentation *)
val indicator_ : t -> Py.Object.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

