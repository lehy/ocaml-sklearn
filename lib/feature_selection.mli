module GenericUnivariateSelect : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?score_func:Py.Object.t -> ?mode:[`Percentile | `K_best | `Fpr | `Fdr | `Fwe] -> ?param:[`Float of float | `PyObject of Py.Object.t] -> unit -> t
(**
Univariate feature selector with configurable strategy.

Read more in the :ref:`User Guide <univariate_feature_selection>`.

Parameters
----------
score_func : callable
    Function taking two arrays X and y, and returning a pair of arrays
    (scores, pvalues). For modes 'percentile' or 'kbest' it can return
    a single array scores.

mode : {'percentile', 'k_best', 'fpr', 'fdr', 'fwe'}
    Feature selection mode.

param : float or int depending on the feature selection mode
    Parameter of the corresponding mode.

Attributes
----------
scores_ : array-like of shape (n_features,)
    Scores of features.

pvalues_ : array-like of shape (n_features,)
    p-values of feature scores, None if `score_func` returned scores only.

Examples
--------
>>> from sklearn.datasets import load_breast_cancer
>>> from sklearn.feature_selection import GenericUnivariateSelect, chi2
>>> X, y = load_breast_cancer(return_X_y=True)
>>> X.shape
(569, 30)
>>> transformer = GenericUnivariateSelect(chi2, 'k_best', param=20)
>>> X_new = transformer.fit_transform(X, y)
>>> X_new.shape
(569, 20)

See also
--------
f_classif: ANOVA F-value between label/feature for classification tasks.
mutual_info_classif: Mutual information for a discrete target.
chi2: Chi-squared stats of non-negative features for classification tasks.
f_regression: F-value between label/feature for regression tasks.
mutual_info_regression: Mutual information for a continuous target.
SelectPercentile: Select features based on percentile of the highest scores.
SelectKBest: Select features based on the k highest scores.
SelectFpr: Select features based on a false positive rate test.
SelectFdr: Select features based on an estimated false discovery rate.
SelectFwe: Select features based on family-wise error rate.
*)

val fit : x:Ndarray.t -> y:Ndarray.t -> t -> t
(**
Run score function on (X, y) and get the appropriate features.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The training input samples.

y : array-like of shape (n_samples,)
    The target values (class labels in classification, real numbers in
    regression).

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

val get_support : ?indices:bool -> t -> Ndarray.t
(**
Get a mask, or integer index, of the features selected

Parameters
----------
indices : boolean (default False)
    If True, the return value will be an array of integers, rather
    than a boolean mask.

Returns
-------
support : array
    An index that selects the retained features from a feature vector.
    If `indices` is False, this is a boolean array of shape
    [# input features], in which an element is True iff its
    corresponding feature is selected for retention. If `indices` is
    True, this is an integer array of shape [# output features] whose
    values are indices into the input feature vector.
*)

val inverse_transform : x:Ndarray.t -> t -> Ndarray.t
(**
Reverse the transformation operation

Parameters
----------
X : array of shape [n_samples, n_selected_features]
    The input samples.

Returns
-------
X_r : array of shape [n_samples, n_original_features]
    `X` with columns of zeros inserted where features would have
    been removed by :meth:`transform`.
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
Reduce X to the selected features.

Parameters
----------
X : array of shape [n_samples, n_features]
    The input samples.

Returns
-------
X_r : array of shape [n_samples, n_selected_features]
    The input samples with only the selected features.
*)


(** Attribute scores_: see constructor for documentation *)
val scores_ : t -> Ndarray.t

(** Attribute pvalues_: see constructor for documentation *)
val pvalues_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module RFE : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_features_to_select:[`Int of int | `None] -> ?step:[`Int of int | `Float of float] -> ?verbose:int -> estimator:Py.Object.t -> unit -> t
(**
Feature ranking with recursive feature elimination.

Given an external estimator that assigns weights to features (e.g., the
coefficients of a linear model), the goal of recursive feature elimination
(RFE) is to select features by recursively considering smaller and smaller
sets of features. First, the estimator is trained on the initial set of
features and the importance of each feature is obtained either through a
``coef_`` attribute or through a ``feature_importances_`` attribute.
Then, the least important features are pruned from current set of features.
That procedure is recursively repeated on the pruned set until the desired
number of features to select is eventually reached.

Read more in the :ref:`User Guide <rfe>`.

Parameters
----------
estimator : object
    A supervised learning estimator with a ``fit`` method that provides
    information about feature importance either through a ``coef_``
    attribute or through a ``feature_importances_`` attribute.

n_features_to_select : int or None (default=None)
    The number of features to select. If `None`, half of the features
    are selected.

step : int or float, optional (default=1)
    If greater than or equal to 1, then ``step`` corresponds to the
    (integer) number of features to remove at each iteration.
    If within (0.0, 1.0), then ``step`` corresponds to the percentage
    (rounded down) of features to remove at each iteration.

verbose : int, (default=0)
    Controls verbosity of output.

Attributes
----------
n_features_ : int
    The number of selected features.

support_ : array of shape [n_features]
    The mask of selected features.

ranking_ : array of shape [n_features]
    The feature ranking, such that ``ranking_[i]`` corresponds to the
    ranking position of the i-th feature. Selected (i.e., estimated
    best) features are assigned rank 1.

estimator_ : object
    The external estimator fit on the reduced dataset.

Examples
--------
The following example shows how to retrieve the 5 most informative
features in the Friedman #1 dataset.

>>> from sklearn.datasets import make_friedman1
>>> from sklearn.feature_selection import RFE
>>> from sklearn.svm import SVR
>>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
>>> estimator = SVR(kernel="linear")
>>> selector = RFE(estimator, 5, step=1)
>>> selector = selector.fit(X, y)
>>> selector.support_
array([ True,  True,  True,  True,  True, False, False, False, False,
       False])
>>> selector.ranking_
array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])

Notes
-----
Allows NaN/Inf in the input if the underlying estimator does as well.

See also
--------
RFECV : Recursive feature elimination with built-in cross-validated
    selection of the best number of features

References
----------

.. [1] Guyon, I., Weston, J., Barnhill, S., & Vapnik, V., "Gene selection
       for cancer classification using support vector machines",
       Mach. Learn., 46(1-3), 389--422, 2002.
*)

val decision_function : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Compute the decision function of ``X``.

Parameters
----------
X : {array-like or sparse matrix} of shape (n_samples, n_features)
    The input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csr_matrix``.

Returns
-------
score : array, shape = [n_samples, n_classes] or [n_samples]
    The decision function of the input samples. The order of the
    classes corresponds to that in the attribute :term:`classes_`.
    Regression and binary classification produce an array of shape
    [n_samples].
*)

val fit : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:Ndarray.t -> t -> t
(**
Fit the RFE model and then the underlying estimator on the selected
   features.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The training input samples.

y : array-like of shape (n_samples,)
    The target values.
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

val get_support : ?indices:bool -> t -> Ndarray.t
(**
Get a mask, or integer index, of the features selected

Parameters
----------
indices : boolean (default False)
    If True, the return value will be an array of integers, rather
    than a boolean mask.

Returns
-------
support : array
    An index that selects the retained features from a feature vector.
    If `indices` is False, this is a boolean array of shape
    [# input features], in which an element is True iff its
    corresponding feature is selected for retention. If `indices` is
    True, this is an integer array of shape [# output features] whose
    values are indices into the input feature vector.
*)

val inverse_transform : x:Ndarray.t -> t -> Ndarray.t
(**
Reverse the transformation operation

Parameters
----------
X : array of shape [n_samples, n_selected_features]
    The input samples.

Returns
-------
X_r : array of shape [n_samples, n_original_features]
    `X` with columns of zeros inserted where features would have
    been removed by :meth:`transform`.
*)

val predict : x:Ndarray.t -> t -> Ndarray.t
(**
Reduce X to the selected features and then predict using the
   underlying estimator.

Parameters
----------
X : array of shape [n_samples, n_features]
    The input samples.

Returns
-------
y : array of shape [n_samples]
    The predicted target values.
*)

val predict_log_proba : x:Ndarray.t -> t -> Ndarray.t
(**
Predict class log-probabilities for X.

Parameters
----------
X : array of shape [n_samples, n_features]
    The input samples.

Returns
-------
p : array of shape (n_samples, n_classes)
    The class log-probabilities of the input samples. The order of the
    classes corresponds to that in the attribute :term:`classes_`.
*)

val predict_proba : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Predict class probabilities for X.

Parameters
----------
X : {array-like or sparse matrix} of shape (n_samples, n_features)
    The input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csr_matrix``.

Returns
-------
p : array of shape (n_samples, n_classes)
    The class probabilities of the input samples. The order of the
    classes corresponds to that in the attribute :term:`classes_`.
*)

val score : x:Ndarray.t -> y:Ndarray.t -> t -> Py.Object.t
(**
Reduce X to the selected features and then return the score of the
   underlying estimator.

Parameters
----------
X : array of shape [n_samples, n_features]
    The input samples.

y : array of shape [n_samples]
    The target values.
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
Reduce X to the selected features.

Parameters
----------
X : array of shape [n_samples, n_features]
    The input samples.

Returns
-------
X_r : array of shape [n_samples, n_selected_features]
    The input samples with only the selected features.
*)


(** Attribute n_features_: see constructor for documentation *)
val n_features_ : t -> int

(** Attribute support_: see constructor for documentation *)
val support_ : t -> Ndarray.t

(** Attribute ranking_: see constructor for documentation *)
val ranking_ : t -> Ndarray.t

(** Attribute estimator_: see constructor for documentation *)
val estimator_ : t -> Py.Object.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module RFECV : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?step:[`Int of int | `Float of float] -> ?min_features_to_select:int -> ?cv:[`Int of int | `CrossValGenerator of Py.Object.t | `Ndarray of Ndarray.t] -> ?scoring:[`String of string | `Callable of Py.Object.t | `None] -> ?verbose:int -> ?n_jobs:[`Int of int | `None] -> estimator:Py.Object.t -> unit -> t
(**
Feature ranking with recursive feature elimination and cross-validated
selection of the best number of features.

See glossary entry for :term:`cross-validation estimator`.

Read more in the :ref:`User Guide <rfe>`.

Parameters
----------
estimator : object
    A supervised learning estimator with a ``fit`` method that provides
    information about feature importance either through a ``coef_``
    attribute or through a ``feature_importances_`` attribute.

step : int or float, optional (default=1)
    If greater than or equal to 1, then ``step`` corresponds to the
    (integer) number of features to remove at each iteration.
    If within (0.0, 1.0), then ``step`` corresponds to the percentage
    (rounded down) of features to remove at each iteration.
    Note that the last iteration may remove fewer than ``step`` features in
    order to reach ``min_features_to_select``.

min_features_to_select : int, (default=1)
    The minimum number of features to be selected. This number of features
    will always be scored, even if the difference between the original
    feature count and ``min_features_to_select`` isn't divisible by
    ``step``.

cv : int, cross-validation generator or an iterable, optional
    Determines the cross-validation splitting strategy.
    Possible inputs for cv are:

    - None, to use the default 5-fold cross-validation,
    - integer, to specify the number of folds.
    - :term:`CV splitter`,
    - An iterable yielding (train, test) splits as arrays of indices.

    For integer/None inputs, if ``y`` is binary or multiclass,
    :class:`sklearn.model_selection.StratifiedKFold` is used. If the
    estimator is a classifier or if ``y`` is neither binary nor multiclass,
    :class:`sklearn.model_selection.KFold` is used.

    Refer :ref:`User Guide <cross_validation>` for the various
    cross-validation strategies that can be used here.

    .. versionchanged:: 0.22
        ``cv`` default value of None changed from 3-fold to 5-fold.

scoring : string, callable or None, optional, (default=None)
    A string (see model evaluation documentation) or
    a scorer callable object / function with signature
    ``scorer(estimator, X, y)``.

verbose : int, (default=0)
    Controls verbosity of output.

n_jobs : int or None, optional (default=None)
    Number of cores to run in parallel while fitting across folds.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

Attributes
----------
n_features_ : int
    The number of selected features with cross-validation.

support_ : array of shape [n_features]
    The mask of selected features.

ranking_ : array of shape [n_features]
    The feature ranking, such that `ranking_[i]`
    corresponds to the ranking
    position of the i-th feature.
    Selected (i.e., estimated best)
    features are assigned rank 1.

grid_scores_ : array of shape [n_subsets_of_features]
    The cross-validation scores such that
    ``grid_scores_[i]`` corresponds to
    the CV score of the i-th subset of features.

estimator_ : object
    The external estimator fit on the reduced dataset.

Notes
-----
The size of ``grid_scores_`` is equal to
``ceil((n_features - min_features_to_select) / step) + 1``,
where step is the number of features removed at each iteration.

Allows NaN/Inf in the input if the underlying estimator does as well.

Examples
--------
The following example shows how to retrieve the a-priori not known 5
informative features in the Friedman #1 dataset.

>>> from sklearn.datasets import make_friedman1
>>> from sklearn.feature_selection import RFECV
>>> from sklearn.svm import SVR
>>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
>>> estimator = SVR(kernel="linear")
>>> selector = RFECV(estimator, step=1, cv=5)
>>> selector = selector.fit(X, y)
>>> selector.support_
array([ True,  True,  True,  True,  True, False, False, False, False,
       False])
>>> selector.ranking_
array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])

See also
--------
RFE : Recursive feature elimination

References
----------

.. [1] Guyon, I., Weston, J., Barnhill, S., & Vapnik, V., "Gene selection
       for cancer classification using support vector machines",
       Mach. Learn., 46(1-3), 389--422, 2002.
*)

val decision_function : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Compute the decision function of ``X``.

Parameters
----------
X : {array-like or sparse matrix} of shape (n_samples, n_features)
    The input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csr_matrix``.

Returns
-------
score : array, shape = [n_samples, n_classes] or [n_samples]
    The decision function of the input samples. The order of the
    classes corresponds to that in the attribute :term:`classes_`.
    Regression and binary classification produce an array of shape
    [n_samples].
*)

val fit : ?groups:[`Ndarray of Ndarray.t | `None] -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:Ndarray.t -> t -> t
(**
Fit the RFE model and automatically tune the number of selected
   features.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Training vector, where `n_samples` is the number of samples and
    `n_features` is the total number of features.

y : array-like of shape (n_samples,)
    Target values (integers for classification, real numbers for
    regression).

groups : array-like of shape (n_samples,) or None
    Group labels for the samples used while splitting the dataset into
    train/test set. Only used in conjunction with a "Group" :term:`cv`
    instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).
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

val get_support : ?indices:bool -> t -> Ndarray.t
(**
Get a mask, or integer index, of the features selected

Parameters
----------
indices : boolean (default False)
    If True, the return value will be an array of integers, rather
    than a boolean mask.

Returns
-------
support : array
    An index that selects the retained features from a feature vector.
    If `indices` is False, this is a boolean array of shape
    [# input features], in which an element is True iff its
    corresponding feature is selected for retention. If `indices` is
    True, this is an integer array of shape [# output features] whose
    values are indices into the input feature vector.
*)

val inverse_transform : x:Ndarray.t -> t -> Ndarray.t
(**
Reverse the transformation operation

Parameters
----------
X : array of shape [n_samples, n_selected_features]
    The input samples.

Returns
-------
X_r : array of shape [n_samples, n_original_features]
    `X` with columns of zeros inserted where features would have
    been removed by :meth:`transform`.
*)

val predict : x:Ndarray.t -> t -> Ndarray.t
(**
Reduce X to the selected features and then predict using the
   underlying estimator.

Parameters
----------
X : array of shape [n_samples, n_features]
    The input samples.

Returns
-------
y : array of shape [n_samples]
    The predicted target values.
*)

val predict_log_proba : x:Ndarray.t -> t -> Ndarray.t
(**
Predict class log-probabilities for X.

Parameters
----------
X : array of shape [n_samples, n_features]
    The input samples.

Returns
-------
p : array of shape (n_samples, n_classes)
    The class log-probabilities of the input samples. The order of the
    classes corresponds to that in the attribute :term:`classes_`.
*)

val predict_proba : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Predict class probabilities for X.

Parameters
----------
X : {array-like or sparse matrix} of shape (n_samples, n_features)
    The input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csr_matrix``.

Returns
-------
p : array of shape (n_samples, n_classes)
    The class probabilities of the input samples. The order of the
    classes corresponds to that in the attribute :term:`classes_`.
*)

val score : x:Ndarray.t -> y:Ndarray.t -> t -> Py.Object.t
(**
Reduce X to the selected features and then return the score of the
   underlying estimator.

Parameters
----------
X : array of shape [n_samples, n_features]
    The input samples.

y : array of shape [n_samples]
    The target values.
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
Reduce X to the selected features.

Parameters
----------
X : array of shape [n_samples, n_features]
    The input samples.

Returns
-------
X_r : array of shape [n_samples, n_selected_features]
    The input samples with only the selected features.
*)


(** Attribute n_features_: see constructor for documentation *)
val n_features_ : t -> int

(** Attribute support_: see constructor for documentation *)
val support_ : t -> Ndarray.t

(** Attribute ranking_: see constructor for documentation *)
val ranking_ : t -> Ndarray.t

(** Attribute grid_scores_: see constructor for documentation *)
val grid_scores_ : t -> Ndarray.t

(** Attribute estimator_: see constructor for documentation *)
val estimator_ : t -> Py.Object.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module SelectFdr : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?score_func:Py.Object.t -> ?alpha:float -> unit -> t
(**
Filter: Select the p-values for an estimated false discovery rate

This uses the Benjamini-Hochberg procedure. ``alpha`` is an upper bound
on the expected false discovery rate.

Read more in the :ref:`User Guide <univariate_feature_selection>`.

Parameters
----------
score_func : callable
    Function taking two arrays X and y, and returning a pair of arrays
    (scores, pvalues).
    Default is f_classif (see below "See also"). The default function only
    works with classification tasks.

alpha : float, optional
    The highest uncorrected p-value for features to keep.

Examples
--------
>>> from sklearn.datasets import load_breast_cancer
>>> from sklearn.feature_selection import SelectFdr, chi2
>>> X, y = load_breast_cancer(return_X_y=True)
>>> X.shape
(569, 30)
>>> X_new = SelectFdr(chi2, alpha=0.01).fit_transform(X, y)
>>> X_new.shape
(569, 16)

Attributes
----------
scores_ : array-like of shape (n_features,)
    Scores of features.

pvalues_ : array-like of shape (n_features,)
    p-values of feature scores.

References
----------
https://en.wikipedia.org/wiki/False_discovery_rate

See also
--------
f_classif: ANOVA F-value between label/feature for classification tasks.
mutual_info_classif: Mutual information for a discrete target.
chi2: Chi-squared stats of non-negative features for classification tasks.
f_regression: F-value between label/feature for regression tasks.
mutual_info_regression: Mutual information for a contnuous target.
SelectPercentile: Select features based on percentile of the highest scores.
SelectKBest: Select features based on the k highest scores.
SelectFpr: Select features based on a false positive rate test.
SelectFwe: Select features based on family-wise error rate.
GenericUnivariateSelect: Univariate feature selector with configurable mode.
*)

val fit : x:Ndarray.t -> y:Ndarray.t -> t -> t
(**
Run score function on (X, y) and get the appropriate features.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The training input samples.

y : array-like of shape (n_samples,)
    The target values (class labels in classification, real numbers in
    regression).

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

val get_support : ?indices:bool -> t -> Ndarray.t
(**
Get a mask, or integer index, of the features selected

Parameters
----------
indices : boolean (default False)
    If True, the return value will be an array of integers, rather
    than a boolean mask.

Returns
-------
support : array
    An index that selects the retained features from a feature vector.
    If `indices` is False, this is a boolean array of shape
    [# input features], in which an element is True iff its
    corresponding feature is selected for retention. If `indices` is
    True, this is an integer array of shape [# output features] whose
    values are indices into the input feature vector.
*)

val inverse_transform : x:Ndarray.t -> t -> Ndarray.t
(**
Reverse the transformation operation

Parameters
----------
X : array of shape [n_samples, n_selected_features]
    The input samples.

Returns
-------
X_r : array of shape [n_samples, n_original_features]
    `X` with columns of zeros inserted where features would have
    been removed by :meth:`transform`.
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
Reduce X to the selected features.

Parameters
----------
X : array of shape [n_samples, n_features]
    The input samples.

Returns
-------
X_r : array of shape [n_samples, n_selected_features]
    The input samples with only the selected features.
*)


(** Attribute scores_: see constructor for documentation *)
val scores_ : t -> Ndarray.t

(** Attribute pvalues_: see constructor for documentation *)
val pvalues_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module SelectFpr : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?score_func:Py.Object.t -> ?alpha:float -> unit -> t
(**
Filter: Select the pvalues below alpha based on a FPR test.

FPR test stands for False Positive Rate test. It controls the total
amount of false detections.

Read more in the :ref:`User Guide <univariate_feature_selection>`.

Parameters
----------
score_func : callable
    Function taking two arrays X and y, and returning a pair of arrays
    (scores, pvalues).
    Default is f_classif (see below "See also"). The default function only
    works with classification tasks.

alpha : float, optional
    The highest p-value for features to be kept.

Attributes
----------
scores_ : array-like of shape (n_features,)
    Scores of features.

pvalues_ : array-like of shape (n_features,)
    p-values of feature scores.

Examples
--------
>>> from sklearn.datasets import load_breast_cancer
>>> from sklearn.feature_selection import SelectFpr, chi2
>>> X, y = load_breast_cancer(return_X_y=True)
>>> X.shape
(569, 30)
>>> X_new = SelectFpr(chi2, alpha=0.01).fit_transform(X, y)
>>> X_new.shape
(569, 16)

See also
--------
f_classif: ANOVA F-value between label/feature for classification tasks.
chi2: Chi-squared stats of non-negative features for classification tasks.
mutual_info_classif:
f_regression: F-value between label/feature for regression tasks.
mutual_info_regression: Mutual information between features and the target.
SelectPercentile: Select features based on percentile of the highest scores.
SelectKBest: Select features based on the k highest scores.
SelectFdr: Select features based on an estimated false discovery rate.
SelectFwe: Select features based on family-wise error rate.
GenericUnivariateSelect: Univariate feature selector with configurable mode.
*)

val fit : x:Ndarray.t -> y:Ndarray.t -> t -> t
(**
Run score function on (X, y) and get the appropriate features.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The training input samples.

y : array-like of shape (n_samples,)
    The target values (class labels in classification, real numbers in
    regression).

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

val get_support : ?indices:bool -> t -> Ndarray.t
(**
Get a mask, or integer index, of the features selected

Parameters
----------
indices : boolean (default False)
    If True, the return value will be an array of integers, rather
    than a boolean mask.

Returns
-------
support : array
    An index that selects the retained features from a feature vector.
    If `indices` is False, this is a boolean array of shape
    [# input features], in which an element is True iff its
    corresponding feature is selected for retention. If `indices` is
    True, this is an integer array of shape [# output features] whose
    values are indices into the input feature vector.
*)

val inverse_transform : x:Ndarray.t -> t -> Ndarray.t
(**
Reverse the transformation operation

Parameters
----------
X : array of shape [n_samples, n_selected_features]
    The input samples.

Returns
-------
X_r : array of shape [n_samples, n_original_features]
    `X` with columns of zeros inserted where features would have
    been removed by :meth:`transform`.
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
Reduce X to the selected features.

Parameters
----------
X : array of shape [n_samples, n_features]
    The input samples.

Returns
-------
X_r : array of shape [n_samples, n_selected_features]
    The input samples with only the selected features.
*)


(** Attribute scores_: see constructor for documentation *)
val scores_ : t -> Ndarray.t

(** Attribute pvalues_: see constructor for documentation *)
val pvalues_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module SelectFromModel : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?threshold:[`String of string | `Float of float] -> ?prefit:bool -> ?norm_order:Py.Object.t -> ?max_features:[`Int of int | `None] -> estimator:Py.Object.t -> unit -> t
(**
Meta-transformer for selecting features based on importance weights.

.. versionadded:: 0.17

Parameters
----------
estimator : object
    The base estimator from which the transformer is built.
    This can be both a fitted (if ``prefit`` is set to True)
    or a non-fitted estimator. The estimator must have either a
    ``feature_importances_`` or ``coef_`` attribute after fitting.

threshold : string, float, optional default None
    The threshold value to use for feature selection. Features whose
    importance is greater or equal are kept while the others are
    discarded. If "median" (resp. "mean"), then the ``threshold`` value is
    the median (resp. the mean) of the feature importances. A scaling
    factor (e.g., "1.25*mean") may also be used. If None and if the
    estimator has a parameter penalty set to l1, either explicitly
    or implicitly (e.g, Lasso), the threshold used is 1e-5.
    Otherwise, "mean" is used by default.

prefit : bool, default False
    Whether a prefit model is expected to be passed into the constructor
    directly or not. If True, ``transform`` must be called directly
    and SelectFromModel cannot be used with ``cross_val_score``,
    ``GridSearchCV`` and similar utilities that clone the estimator.
    Otherwise train the model using ``fit`` and then ``transform`` to do
    feature selection.

norm_order : non-zero int, inf, -inf, default 1
    Order of the norm used to filter the vectors of coefficients below
    ``threshold`` in the case where the ``coef_`` attribute of the
    estimator is of dimension 2.

max_features : int or None, optional
    The maximum number of features selected scoring above ``threshold``.
    To disable ``threshold`` and only select based on ``max_features``,
    set ``threshold=-np.inf``.

    .. versionadded:: 0.20

Attributes
----------
estimator_ : an estimator
    The base estimator from which the transformer is built.
    This is stored only when a non-fitted estimator is passed to the
    ``SelectFromModel``, i.e when prefit is False.

threshold_ : float
    The threshold value used for feature selection.

Notes
-----
Allows NaN/Inf in the input if the underlying estimator does as well.

Examples
--------
>>> from sklearn.feature_selection import SelectFromModel
>>> from sklearn.linear_model import LogisticRegression
>>> X = [[ 0.87, -1.34,  0.31 ],
...      [-2.79, -0.02, -0.85 ],
...      [-1.34, -0.48, -2.55 ],
...      [ 1.92,  1.48,  0.65 ]]
>>> y = [0, 1, 0, 1]
>>> selector = SelectFromModel(estimator=LogisticRegression()).fit(X, y)
>>> selector.estimator_.coef_
array([[-0.3252302 ,  0.83462377,  0.49750423]])
>>> selector.threshold_
0.55245...
>>> selector.get_support()
array([False,  True, False])
>>> selector.transform(X)
array([[-1.34],
       [-0.02],
       [-0.48],
       [ 1.48]])
*)

val fit : ?y:Ndarray.t -> ?fit_params:(string * Py.Object.t) list -> x:Ndarray.t -> t -> t
(**
Fit the SelectFromModel meta-transformer.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The training input samples.

y : array-like, shape (n_samples,)
    The target values (integers that correspond to classes in
    classification, real numbers in regression).

**fit_params : Other estimator specific parameters

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

val get_support : ?indices:bool -> t -> Ndarray.t
(**
Get a mask, or integer index, of the features selected

Parameters
----------
indices : boolean (default False)
    If True, the return value will be an array of integers, rather
    than a boolean mask.

Returns
-------
support : array
    An index that selects the retained features from a feature vector.
    If `indices` is False, this is a boolean array of shape
    [# input features], in which an element is True iff its
    corresponding feature is selected for retention. If `indices` is
    True, this is an integer array of shape [# output features] whose
    values are indices into the input feature vector.
*)

val inverse_transform : x:Ndarray.t -> t -> Ndarray.t
(**
Reverse the transformation operation

Parameters
----------
X : array of shape [n_samples, n_selected_features]
    The input samples.

Returns
-------
X_r : array of shape [n_samples, n_original_features]
    `X` with columns of zeros inserted where features would have
    been removed by :meth:`transform`.
*)

val partial_fit : ?y:Ndarray.t -> ?fit_params:(string * Py.Object.t) list -> x:Ndarray.t -> t -> t
(**
Fit the SelectFromModel meta-transformer only once.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The training input samples.

y : array-like, shape (n_samples,)
    The target values (integers that correspond to classes in
    classification, real numbers in regression).

**fit_params : Other estimator specific parameters

Returns
-------
self : object
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
Reduce X to the selected features.

Parameters
----------
X : array of shape [n_samples, n_features]
    The input samples.

Returns
-------
X_r : array of shape [n_samples, n_selected_features]
    The input samples with only the selected features.
*)


(** Attribute estimator_: see constructor for documentation *)
val estimator_ : t -> Py.Object.t

(** Attribute threshold_: see constructor for documentation *)
val threshold_ : t -> float

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module SelectFwe : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?score_func:Py.Object.t -> ?alpha:float -> unit -> t
(**
Filter: Select the p-values corresponding to Family-wise error rate

Read more in the :ref:`User Guide <univariate_feature_selection>`.

Parameters
----------
score_func : callable
    Function taking two arrays X and y, and returning a pair of arrays
    (scores, pvalues).
    Default is f_classif (see below "See also"). The default function only
    works with classification tasks.

alpha : float, optional
    The highest uncorrected p-value for features to keep.

Examples
--------
>>> from sklearn.datasets import load_breast_cancer
>>> from sklearn.feature_selection import SelectFwe, chi2
>>> X, y = load_breast_cancer(return_X_y=True)
>>> X.shape
(569, 30)
>>> X_new = SelectFwe(chi2, alpha=0.01).fit_transform(X, y)
>>> X_new.shape
(569, 15)

Attributes
----------
scores_ : array-like of shape (n_features,)
    Scores of features.

pvalues_ : array-like of shape (n_features,)
    p-values of feature scores.

See also
--------
f_classif: ANOVA F-value between label/feature for classification tasks.
chi2: Chi-squared stats of non-negative features for classification tasks.
f_regression: F-value between label/feature for regression tasks.
SelectPercentile: Select features based on percentile of the highest scores.
SelectKBest: Select features based on the k highest scores.
SelectFpr: Select features based on a false positive rate test.
SelectFdr: Select features based on an estimated false discovery rate.
GenericUnivariateSelect: Univariate feature selector with configurable mode.
*)

val fit : x:Ndarray.t -> y:Ndarray.t -> t -> t
(**
Run score function on (X, y) and get the appropriate features.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The training input samples.

y : array-like of shape (n_samples,)
    The target values (class labels in classification, real numbers in
    regression).

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

val get_support : ?indices:bool -> t -> Ndarray.t
(**
Get a mask, or integer index, of the features selected

Parameters
----------
indices : boolean (default False)
    If True, the return value will be an array of integers, rather
    than a boolean mask.

Returns
-------
support : array
    An index that selects the retained features from a feature vector.
    If `indices` is False, this is a boolean array of shape
    [# input features], in which an element is True iff its
    corresponding feature is selected for retention. If `indices` is
    True, this is an integer array of shape [# output features] whose
    values are indices into the input feature vector.
*)

val inverse_transform : x:Ndarray.t -> t -> Ndarray.t
(**
Reverse the transformation operation

Parameters
----------
X : array of shape [n_samples, n_selected_features]
    The input samples.

Returns
-------
X_r : array of shape [n_samples, n_original_features]
    `X` with columns of zeros inserted where features would have
    been removed by :meth:`transform`.
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
Reduce X to the selected features.

Parameters
----------
X : array of shape [n_samples, n_features]
    The input samples.

Returns
-------
X_r : array of shape [n_samples, n_selected_features]
    The input samples with only the selected features.
*)


(** Attribute scores_: see constructor for documentation *)
val scores_ : t -> Ndarray.t

(** Attribute pvalues_: see constructor for documentation *)
val pvalues_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module SelectKBest : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?score_func:Py.Object.t -> ?k:[`Int of int | `All] -> unit -> t
(**
Select features according to the k highest scores.

Read more in the :ref:`User Guide <univariate_feature_selection>`.

Parameters
----------
score_func : callable
    Function taking two arrays X and y, and returning a pair of arrays
    (scores, pvalues) or a single array with scores.
    Default is f_classif (see below "See also"). The default function only
    works with classification tasks.

k : int or "all", optional, default=10
    Number of top features to select.
    The "all" option bypasses selection, for use in a parameter search.

Attributes
----------
scores_ : array-like of shape (n_features,)
    Scores of features.

pvalues_ : array-like of shape (n_features,)
    p-values of feature scores, None if `score_func` returned only scores.

Examples
--------
>>> from sklearn.datasets import load_digits
>>> from sklearn.feature_selection import SelectKBest, chi2
>>> X, y = load_digits(return_X_y=True)
>>> X.shape
(1797, 64)
>>> X_new = SelectKBest(chi2, k=20).fit_transform(X, y)
>>> X_new.shape
(1797, 20)

Notes
-----
Ties between features with equal scores will be broken in an unspecified
way.

See also
--------
f_classif: ANOVA F-value between label/feature for classification tasks.
mutual_info_classif: Mutual information for a discrete target.
chi2: Chi-squared stats of non-negative features for classification tasks.
f_regression: F-value between label/feature for regression tasks.
mutual_info_regression: Mutual information for a continuous target.
SelectPercentile: Select features based on percentile of the highest scores.
SelectFpr: Select features based on a false positive rate test.
SelectFdr: Select features based on an estimated false discovery rate.
SelectFwe: Select features based on family-wise error rate.
GenericUnivariateSelect: Univariate feature selector with configurable mode.
*)

val fit : x:Ndarray.t -> y:Ndarray.t -> t -> t
(**
Run score function on (X, y) and get the appropriate features.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The training input samples.

y : array-like of shape (n_samples,)
    The target values (class labels in classification, real numbers in
    regression).

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

val get_support : ?indices:bool -> t -> Ndarray.t
(**
Get a mask, or integer index, of the features selected

Parameters
----------
indices : boolean (default False)
    If True, the return value will be an array of integers, rather
    than a boolean mask.

Returns
-------
support : array
    An index that selects the retained features from a feature vector.
    If `indices` is False, this is a boolean array of shape
    [# input features], in which an element is True iff its
    corresponding feature is selected for retention. If `indices` is
    True, this is an integer array of shape [# output features] whose
    values are indices into the input feature vector.
*)

val inverse_transform : x:Ndarray.t -> t -> Ndarray.t
(**
Reverse the transformation operation

Parameters
----------
X : array of shape [n_samples, n_selected_features]
    The input samples.

Returns
-------
X_r : array of shape [n_samples, n_original_features]
    `X` with columns of zeros inserted where features would have
    been removed by :meth:`transform`.
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
Reduce X to the selected features.

Parameters
----------
X : array of shape [n_samples, n_features]
    The input samples.

Returns
-------
X_r : array of shape [n_samples, n_selected_features]
    The input samples with only the selected features.
*)


(** Attribute scores_: see constructor for documentation *)
val scores_ : t -> Ndarray.t

(** Attribute pvalues_: see constructor for documentation *)
val pvalues_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module SelectPercentile : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?score_func:Py.Object.t -> ?percentile:int -> unit -> t
(**
Select features according to a percentile of the highest scores.

Read more in the :ref:`User Guide <univariate_feature_selection>`.

Parameters
----------
score_func : callable
    Function taking two arrays X and y, and returning a pair of arrays
    (scores, pvalues) or a single array with scores.
    Default is f_classif (see below "See also"). The default function only
    works with classification tasks.

percentile : int, optional, default=10
    Percent of features to keep.

Attributes
----------
scores_ : array-like of shape (n_features,)
    Scores of features.

pvalues_ : array-like of shape (n_features,)
    p-values of feature scores, None if `score_func` returned only scores.

Examples
--------
>>> from sklearn.datasets import load_digits
>>> from sklearn.feature_selection import SelectPercentile, chi2
>>> X, y = load_digits(return_X_y=True)
>>> X.shape
(1797, 64)
>>> X_new = SelectPercentile(chi2, percentile=10).fit_transform(X, y)
>>> X_new.shape
(1797, 7)

Notes
-----
Ties between features with equal scores will be broken in an unspecified
way.

See also
--------
f_classif: ANOVA F-value between label/feature for classification tasks.
mutual_info_classif: Mutual information for a discrete target.
chi2: Chi-squared stats of non-negative features for classification tasks.
f_regression: F-value between label/feature for regression tasks.
mutual_info_regression: Mutual information for a continuous target.
SelectKBest: Select features based on the k highest scores.
SelectFpr: Select features based on a false positive rate test.
SelectFdr: Select features based on an estimated false discovery rate.
SelectFwe: Select features based on family-wise error rate.
GenericUnivariateSelect: Univariate feature selector with configurable mode.
*)

val fit : x:Ndarray.t -> y:Ndarray.t -> t -> t
(**
Run score function on (X, y) and get the appropriate features.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The training input samples.

y : array-like of shape (n_samples,)
    The target values (class labels in classification, real numbers in
    regression).

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

val get_support : ?indices:bool -> t -> Ndarray.t
(**
Get a mask, or integer index, of the features selected

Parameters
----------
indices : boolean (default False)
    If True, the return value will be an array of integers, rather
    than a boolean mask.

Returns
-------
support : array
    An index that selects the retained features from a feature vector.
    If `indices` is False, this is a boolean array of shape
    [# input features], in which an element is True iff its
    corresponding feature is selected for retention. If `indices` is
    True, this is an integer array of shape [# output features] whose
    values are indices into the input feature vector.
*)

val inverse_transform : x:Ndarray.t -> t -> Ndarray.t
(**
Reverse the transformation operation

Parameters
----------
X : array of shape [n_samples, n_selected_features]
    The input samples.

Returns
-------
X_r : array of shape [n_samples, n_original_features]
    `X` with columns of zeros inserted where features would have
    been removed by :meth:`transform`.
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
Reduce X to the selected features.

Parameters
----------
X : array of shape [n_samples, n_features]
    The input samples.

Returns
-------
X_r : array of shape [n_samples, n_selected_features]
    The input samples with only the selected features.
*)


(** Attribute scores_: see constructor for documentation *)
val scores_ : t -> Ndarray.t

(** Attribute pvalues_: see constructor for documentation *)
val pvalues_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module VarianceThreshold : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?threshold:float -> unit -> t
(**
Feature selector that removes all low-variance features.

This feature selection algorithm looks only at the features (X), not the
desired outputs (y), and can thus be used for unsupervised learning.

Read more in the :ref:`User Guide <variance_threshold>`.

Parameters
----------
threshold : float, optional
    Features with a training-set variance lower than this threshold will
    be removed. The default is to keep all features with non-zero variance,
    i.e. remove the features that have the same value in all samples.

Attributes
----------
variances_ : array, shape (n_features,)
    Variances of individual features.

Notes
-----
Allows NaN in the input.

Examples
--------
The following dataset has integer features, two of which are the same
in every sample. These are removed with the default setting for threshold::

    >>> X = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
    >>> selector = VarianceThreshold()
    >>> selector.fit_transform(X)
    array([[2, 0],
           [1, 4],
           [1, 1]])
*)

val fit : ?y:Py.Object.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> t
(**
Learn empirical variances from X.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Sample vectors from which to compute variances.

y : any
    Ignored. This parameter exists only for compatibility with
    sklearn.pipeline.Pipeline.

Returns
-------
self
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

val get_support : ?indices:bool -> t -> Ndarray.t
(**
Get a mask, or integer index, of the features selected

Parameters
----------
indices : boolean (default False)
    If True, the return value will be an array of integers, rather
    than a boolean mask.

Returns
-------
support : array
    An index that selects the retained features from a feature vector.
    If `indices` is False, this is a boolean array of shape
    [# input features], in which an element is True iff its
    corresponding feature is selected for retention. If `indices` is
    True, this is an integer array of shape [# output features] whose
    values are indices into the input feature vector.
*)

val inverse_transform : x:Ndarray.t -> t -> Ndarray.t
(**
Reverse the transformation operation

Parameters
----------
X : array of shape [n_samples, n_selected_features]
    The input samples.

Returns
-------
X_r : array of shape [n_samples, n_original_features]
    `X` with columns of zeros inserted where features would have
    been removed by :meth:`transform`.
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
Reduce X to the selected features.

Parameters
----------
X : array of shape [n_samples, n_features]
    The input samples.

Returns
-------
X_r : array of shape [n_samples, n_selected_features]
    The input samples with only the selected features.
*)


(** Attribute variances_: see constructor for documentation *)
val variances_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val chi2 : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:Ndarray.t -> unit -> Ndarray.t
(**
Compute chi-squared stats between each non-negative feature and class.

This score can be used to select the n_features features with the
highest values for the test chi-squared statistic from X, which must
contain only non-negative features such as booleans or frequencies
(e.g., term counts in document classification), relative to the classes.

Recall that the chi-square test measures dependence between stochastic
variables, so using this function "weeds out" the features that are the
most likely to be independent of class and therefore irrelevant for
classification.

Read more in the :ref:`User Guide <univariate_feature_selection>`.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Sample vectors.

y : array-like of shape (n_samples,)
    Target vector (class labels).

Returns
-------
chi2 : array, shape = (n_features,)
    chi2 statistics of each feature.
pval : array, shape = (n_features,)
    p-values of each feature.

Notes
-----
Complexity of this algorithm is O(n_classes * n_features).

See also
--------
f_classif: ANOVA F-value between label/feature for classification tasks.
f_regression: F-value between label/feature for regression tasks.
*)

val f_classif : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:Ndarray.t -> unit -> (Ndarray.t * Ndarray.t)
(**
Compute the ANOVA F-value for the provided sample.

Read more in the :ref:`User Guide <univariate_feature_selection>`.

Parameters
----------
X : {array-like, sparse matrix} shape = [n_samples, n_features]
    The set of regressors that will be tested sequentially.

y : array of shape(n_samples)
    The data matrix.

Returns
-------
F : array, shape = [n_features,]
    The set of F values.

pval : array, shape = [n_features,]
    The set of p-values.

See also
--------
chi2: Chi-squared stats of non-negative features for classification tasks.
f_regression: F-value between label/feature for regression tasks.
*)

val f_oneway : Py.Object.t list -> Py.Object.t
(**
Performs a 1-way ANOVA.

The one-way ANOVA tests the null hypothesis that 2 or more groups have
the same population mean. The test is applied to samples from two or
more groups, possibly with differing sizes.

Read more in the :ref:`User Guide <univariate_feature_selection>`.

Parameters
----------
*args : array_like, sparse matrices
    sample1, sample2... The sample measurements should be given as
    arguments.

Returns
-------
F-value : float
    The computed F-value of the test.
p-value : float
    The associated p-value from the F-distribution.

Notes
-----
The ANOVA test has important assumptions that must be satisfied in order
for the associated p-value to be valid.

1. The samples are independent
2. Each sample is from a normally distributed population
3. The population standard deviations of the groups are all equal. This
   property is known as homoscedasticity.

If these assumptions are not true for a given set of data, it may still be
possible to use the Kruskal-Wallis H-test (`scipy.stats.kruskal`_) although
with some loss of power.

The algorithm is from Heiman[2], pp.394-7.

See ``scipy.stats.f_oneway`` that should give the same results while
being less efficient.

References
----------

.. [1] Lowry, Richard.  "Concepts and Applications of Inferential
       Statistics". Chapter 14.
       http://faculty.vassar.edu/lowry/ch14pt1.html

.. [2] Heiman, G.W.  Research Methods in Statistics. 2002.
*)

val f_regression : ?center:[`True | `Bool of bool] -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:Ndarray.t -> unit -> (Ndarray.t * Ndarray.t)
(**
Univariate linear regression tests.

Linear model for testing the individual effect of each of many regressors.
This is a scoring function to be used in a feature selection procedure, not
a free standing feature selection procedure.

This is done in 2 steps:

1. The correlation between each regressor and the target is computed,
   that is, ((X[:, i] - mean(X[:, i])) * (y - mean_y)) / (std(X[:, i]) *
   std(y)).
2. It is converted to an F score then to a p-value.

For more on usage see the :ref:`User Guide <univariate_feature_selection>`.

Parameters
----------
X : {array-like, sparse matrix}  shape = (n_samples, n_features)
    The set of regressors that will be tested sequentially.

y : array of shape(n_samples).
    The data matrix

center : True, bool,
    If true, X and y will be centered.

Returns
-------
F : array, shape=(n_features,)
    F values of features.

pval : array, shape=(n_features,)
    p-values of F-scores.


See also
--------
mutual_info_regression: Mutual information for a continuous target.
f_classif: ANOVA F-value between label/feature for classification tasks.
chi2: Chi-squared stats of non-negative features for classification tasks.
SelectKBest: Select features based on the k highest scores.
SelectFpr: Select features based on a false positive rate test.
SelectFdr: Select features based on an estimated false discovery rate.
SelectFwe: Select features based on family-wise error rate.
SelectPercentile: Select features based on percentile of the highest
    scores.
*)

val mutual_info_classif : ?discrete_features:[`Auto | `Bool of bool | `Ndarray of Ndarray.t] -> ?n_neighbors:int -> ?copy:bool -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:Ndarray.t -> unit -> Ndarray.t
(**
Estimate mutual information for a discrete target variable.

Mutual information (MI) [1]_ between two random variables is a non-negative
value, which measures the dependency between the variables. It is equal
to zero if and only if two random variables are independent, and higher
values mean higher dependency.

The function relies on nonparametric methods based on entropy estimation
from k-nearest neighbors distances as described in [2]_ and [3]_. Both
methods are based on the idea originally proposed in [4]_.

It can be used for univariate features selection, read more in the
:ref:`User Guide <univariate_feature_selection>`.

Parameters
----------
X : array_like or sparse matrix, shape (n_samples, n_features)
    Feature matrix.

y : array_like, shape (n_samples,)
    Target vector.

discrete_features : {'auto', bool, array_like}, default 'auto'
    If bool, then determines whether to consider all features discrete
    or continuous. If array, then it should be either a boolean mask
    with shape (n_features,) or array with indices of discrete features.
    If 'auto', it is assigned to False for dense `X` and to True for
    sparse `X`.

n_neighbors : int, default 3
    Number of neighbors to use for MI estimation for continuous variables,
    see [2]_ and [3]_. Higher values reduce variance of the estimation, but
    could introduce a bias.

copy : bool, default True
    Whether to make a copy of the given data. If set to False, the initial
    data will be overwritten.

random_state : int, RandomState instance or None, optional, default None
    The seed of the pseudo random number generator for adding small noise
    to continuous variables in order to remove repeated values.  If int,
    random_state is the seed used by the random number generator; If
    RandomState instance, random_state is the random number generator; If
    None, the random number generator is the RandomState instance used by
    `np.random`.

Returns
-------
mi : ndarray, shape (n_features,)
    Estimated mutual information between each feature and the target.

Notes
-----
1. The term "discrete features" is used instead of naming them
   "categorical", because it describes the essence more accurately.
   For example, pixel intensities of an image are discrete features
   (but hardly categorical) and you will get better results if mark them
   as such. Also note, that treating a continuous variable as discrete and
   vice versa will usually give incorrect results, so be attentive about that.
2. True mutual information can't be negative. If its estimate turns out
   to be negative, it is replaced by zero.

References
----------
.. [1] `Mutual Information <https://en.wikipedia.org/wiki/Mutual_information>`_
       on Wikipedia.
.. [2] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
       information". Phys. Rev. E 69, 2004.
.. [3] B. C. Ross "Mutual Information between Discrete and Continuous
       Data Sets". PLoS ONE 9(2), 2014.
.. [4] L. F. Kozachenko, N. N. Leonenko, "Sample Estimate of the Entropy
       of a Random Vector:, Probl. Peredachi Inf., 23:2 (1987), 9-16"
*)

val mutual_info_regression : ?discrete_features:[`Auto | `Bool of bool | `Ndarray of Ndarray.t] -> ?n_neighbors:int -> ?copy:bool -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:Ndarray.t -> unit -> Ndarray.t
(**
Estimate mutual information for a continuous target variable.

Mutual information (MI) [1]_ between two random variables is a non-negative
value, which measures the dependency between the variables. It is equal
to zero if and only if two random variables are independent, and higher
values mean higher dependency.

The function relies on nonparametric methods based on entropy estimation
from k-nearest neighbors distances as described in [2]_ and [3]_. Both
methods are based on the idea originally proposed in [4]_.

It can be used for univariate features selection, read more in the
:ref:`User Guide <univariate_feature_selection>`.

Parameters
----------
X : array_like or sparse matrix, shape (n_samples, n_features)
    Feature matrix.

y : array_like, shape (n_samples,)
    Target vector.

discrete_features : {'auto', bool, array_like}, default 'auto'
    If bool, then determines whether to consider all features discrete
    or continuous. If array, then it should be either a boolean mask
    with shape (n_features,) or array with indices of discrete features.
    If 'auto', it is assigned to False for dense `X` and to True for
    sparse `X`.

n_neighbors : int, default 3
    Number of neighbors to use for MI estimation for continuous variables,
    see [2]_ and [3]_. Higher values reduce variance of the estimation, but
    could introduce a bias.

copy : bool, default True
    Whether to make a copy of the given data. If set to False, the initial
    data will be overwritten.

random_state : int, RandomState instance or None, optional, default None
    The seed of the pseudo random number generator for adding small noise
    to continuous variables in order to remove repeated values.
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

Returns
-------
mi : ndarray, shape (n_features,)
    Estimated mutual information between each feature and the target.

Notes
-----
1. The term "discrete features" is used instead of naming them
   "categorical", because it describes the essence more accurately.
   For example, pixel intensities of an image are discrete features
   (but hardly categorical) and you will get better results if mark them
   as such. Also note, that treating a continuous variable as discrete and
   vice versa will usually give incorrect results, so be attentive about that.
2. True mutual information can't be negative. If its estimate turns out
   to be negative, it is replaced by zero.

References
----------
.. [1] `Mutual Information <https://en.wikipedia.org/wiki/Mutual_information>`_
       on Wikipedia.
.. [2] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
       information". Phys. Rev. E 69, 2004.
.. [3] B. C. Ross "Mutual Information between Discrete and Continuous
       Data Sets". PLoS ONE 9(2), 2014.
.. [4] L. F. Kozachenko, N. N. Leonenko, "Sample Estimate of the Entropy
       of a Random Vector", Probl. Peredachi Inf., 23:2 (1987), 9-16
*)

