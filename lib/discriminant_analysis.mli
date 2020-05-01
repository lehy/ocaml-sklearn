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

module LinearClassifierMixin : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : unit -> t
(**
Mixin for linear classifiers.

Handles prediction for sparse and dense X.
*)

val decision_function : x:Arr.t -> t -> Arr.t
(**
Predict confidence scores for samples.

The confidence score for a sample is the signed distance of that
sample to the hyperplane.

Parameters
----------
X : array_like or sparse matrix, shape (n_samples, n_features)
    Samples.

Returns
-------
array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
    Confidence scores per (sample, class) combination. In the binary
    case, confidence score for self.classes_[1] where >0 means this
    class would be predicted.
*)

val predict : x:Arr.t -> t -> Arr.t
(**
Predict class labels for samples in X.

Parameters
----------
X : array_like or sparse matrix, shape (n_samples, n_features)
    Samples.

Returns
-------
C : array, shape [n_samples]
    Predicted class label per sample.
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

module LinearDiscriminantAnalysis : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?solver:string -> ?shrinkage:[`S of string | `F of float] -> ?priors:Arr.t -> ?n_components:int -> ?store_covariance:bool -> ?tol:float -> unit -> t
(**
Linear Discriminant Analysis

A classifier with a linear decision boundary, generated by fitting class
conditional densities to the data and using Bayes' rule.

The model fits a Gaussian density to each class, assuming that all classes
share the same covariance matrix.

The fitted model can also be used to reduce the dimensionality of the input
by projecting it to the most discriminative directions.

.. versionadded:: 0.17
   *LinearDiscriminantAnalysis*.

Read more in the :ref:`User Guide <lda_qda>`.

Parameters
----------
solver : string, optional
    Solver to use, possible values:
      - 'svd': Singular value decomposition (default).
        Does not compute the covariance matrix, therefore this solver is
        recommended for data with a large number of features.
      - 'lsqr': Least squares solution, can be combined with shrinkage.
      - 'eigen': Eigenvalue decomposition, can be combined with shrinkage.

shrinkage : string or float, optional
    Shrinkage parameter, possible values:
      - None: no shrinkage (default).
      - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
      - float between 0 and 1: fixed shrinkage parameter.

    Note that shrinkage works only with 'lsqr' and 'eigen' solvers.

priors : array, optional, shape (n_classes,)
    Class priors.

n_components : int, optional (default=None)
    Number of components (<= min(n_classes - 1, n_features)) for
    dimensionality reduction. If None, will be set to
    min(n_classes - 1, n_features).

store_covariance : bool, optional
    Additionally compute class covariance matrix (default False), used
    only in 'svd' solver.

    .. versionadded:: 0.17

tol : float, optional, (default 1.0e-4)
    Threshold used for rank estimation in SVD solver.

    .. versionadded:: 0.17

Attributes
----------
coef_ : array, shape (n_features,) or (n_classes, n_features)
    Weight vector(s).

intercept_ : array, shape (n_classes,)
    Intercept term.

covariance_ : array-like, shape (n_features, n_features)
    Covariance matrix (shared by all classes).

explained_variance_ratio_ : array, shape (n_components,)
    Percentage of variance explained by each of the selected components.
    If ``n_components`` is not set then all components are stored and the
    sum of explained variances is equal to 1.0. Only available when eigen
    or svd solver is used.

means_ : array-like, shape (n_classes, n_features)
    Class means.

priors_ : array-like, shape (n_classes,)
    Class priors (sum to 1).

scalings_ : array-like, shape (rank, n_classes - 1)
    Scaling of the features in the space spanned by the class centroids.

xbar_ : array-like, shape (n_features,)
    Overall mean.

classes_ : array-like, shape (n_classes,)
    Unique class labels.

See also
--------
sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis: Quadratic
    Discriminant Analysis

Notes
-----
The default solver is 'svd'. It can perform both classification and
transform, and it does not rely on the calculation of the covariance
matrix. This can be an advantage in situations where the number of features
is large. However, the 'svd' solver cannot be used with shrinkage.

The 'lsqr' solver is an efficient algorithm that only works for
classification. It supports shrinkage.

The 'eigen' solver is based on the optimization of the between class
scatter to within class scatter ratio. It can be used for both
classification and transform, and it supports shrinkage. However, the
'eigen' solver needs to compute the covariance matrix, so it might not be
suitable for situations with a high number of features.

Examples
--------
>>> import numpy as np
>>> from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
>>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
>>> y = np.array([1, 1, 1, 2, 2, 2])
>>> clf = LinearDiscriminantAnalysis()
>>> clf.fit(X, y)
LinearDiscriminantAnalysis()
>>> print(clf.predict([[-0.8, -1]]))
[1]
*)

val decision_function : x:Arr.t -> t -> Arr.t
(**
Predict confidence scores for samples.

The confidence score for a sample is the signed distance of that
sample to the hyperplane.

Parameters
----------
X : array_like or sparse matrix, shape (n_samples, n_features)
    Samples.

Returns
-------
array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
    Confidence scores per (sample, class) combination. In the binary
    case, confidence score for self.classes_[1] where >0 means this
    class would be predicted.
*)

val fit : x:Arr.t -> y:Arr.t -> t -> t
(**
Fit LinearDiscriminantAnalysis model according to the given
   training data and parameters.

   .. versionchanged:: 0.19
      *store_covariance* has been moved to main constructor.

   .. versionchanged:: 0.19
      *tol* has been moved to main constructor.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training data.

y : array, shape (n_samples,)
    Target values.
*)

val fit_transform : ?y:Arr.t -> ?fit_params:(string * Py.Object.t) list -> x:Arr.t -> t -> Arr.t
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
Predict class labels for samples in X.

Parameters
----------
X : array_like or sparse matrix, shape (n_samples, n_features)
    Samples.

Returns
-------
C : array, shape [n_samples]
    Predicted class label per sample.
*)

val predict_log_proba : x:Arr.t -> t -> Arr.t
(**
Estimate log probability.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Input data.

Returns
-------
C : array, shape (n_samples, n_classes)
    Estimated log probabilities.
*)

val predict_proba : x:Arr.t -> t -> Arr.t
(**
Estimate probability.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Input data.

Returns
-------
C : array, shape (n_samples, n_classes)
    Estimated probabilities.
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

val transform : x:Arr.t -> t -> Arr.t
(**
Project data to maximize class separation.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Input data.

Returns
-------
X_new : array, shape (n_samples, n_components)
    Transformed data.
*)


(** Attribute coef_: get value or raise Not_found if None.*)
val coef_ : t -> Arr.t

(** Attribute coef_: get value as an option. *)
val coef_opt : t -> (Arr.t) option


(** Attribute intercept_: get value or raise Not_found if None.*)
val intercept_ : t -> Arr.t

(** Attribute intercept_: get value as an option. *)
val intercept_opt : t -> (Arr.t) option


(** Attribute covariance_: get value or raise Not_found if None.*)
val covariance_ : t -> Arr.t

(** Attribute covariance_: get value as an option. *)
val covariance_opt : t -> (Arr.t) option


(** Attribute explained_variance_ratio_: get value or raise Not_found if None.*)
val explained_variance_ratio_ : t -> Arr.t

(** Attribute explained_variance_ratio_: get value as an option. *)
val explained_variance_ratio_opt : t -> (Arr.t) option


(** Attribute means_: get value or raise Not_found if None.*)
val means_ : t -> Arr.t

(** Attribute means_: get value as an option. *)
val means_opt : t -> (Arr.t) option


(** Attribute priors_: get value or raise Not_found if None.*)
val priors_ : t -> Arr.t

(** Attribute priors_: get value as an option. *)
val priors_opt : t -> (Arr.t) option


(** Attribute scalings_: get value or raise Not_found if None.*)
val scalings_ : t -> Arr.t

(** Attribute scalings_: get value as an option. *)
val scalings_opt : t -> (Arr.t) option


(** Attribute xbar_: get value or raise Not_found if None.*)
val xbar_ : t -> Arr.t

(** Attribute xbar_: get value as an option. *)
val xbar_opt : t -> (Arr.t) option


(** Attribute classes_: get value or raise Not_found if None.*)
val classes_ : t -> Arr.t

(** Attribute classes_: get value as an option. *)
val classes_opt : t -> (Arr.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module QuadraticDiscriminantAnalysis : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?priors:Arr.t -> ?reg_param:float -> ?store_covariance:bool -> ?tol:float -> unit -> t
(**
Quadratic Discriminant Analysis

A classifier with a quadratic decision boundary, generated
by fitting class conditional densities to the data
and using Bayes' rule.

The model fits a Gaussian density to each class.

.. versionadded:: 0.17
   *QuadraticDiscriminantAnalysis*

Read more in the :ref:`User Guide <lda_qda>`.

Parameters
----------
priors : array, optional, shape = [n_classes]
    Priors on classes

reg_param : float, optional
    Regularizes the covariance estimate as
    ``(1-reg_param)*Sigma + reg_param*np.eye(n_features)``

store_covariance : boolean
    If True the covariance matrices are computed and stored in the
    `self.covariance_` attribute.

    .. versionadded:: 0.17

tol : float, optional, default 1.0e-4
    Threshold used for rank estimation.

    .. versionadded:: 0.17

Attributes
----------
covariance_ : list of array-like of shape (n_features, n_features)
    Covariance matrices of each class.

means_ : array-like of shape (n_classes, n_features)
    Class means.

priors_ : array-like of shape (n_classes)
    Class priors (sum to 1).

rotations_ : list of arrays
    For each class k an array of shape [n_features, n_k], with
    ``n_k = min(n_features, number of elements in class k)``
    It is the rotation of the Gaussian distribution, i.e. its
    principal axis.

scalings_ : list of arrays
    For each class k an array of shape [n_k]. It contains the scaling
    of the Gaussian distributions along its principal axes, i.e. the
    variance in the rotated coordinate system.

classes_ : array-like, shape (n_classes,)
    Unique class labels.

Examples
--------
>>> from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
>>> import numpy as np
>>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
>>> y = np.array([1, 1, 1, 2, 2, 2])
>>> clf = QuadraticDiscriminantAnalysis()
>>> clf.fit(X, y)
QuadraticDiscriminantAnalysis()
>>> print(clf.predict([[-0.8, -1]]))
[1]

See also
--------
sklearn.discriminant_analysis.LinearDiscriminantAnalysis: Linear
    Discriminant Analysis
*)

val decision_function : x:Arr.t -> t -> Arr.t
(**
Apply decision function to an array of samples.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Array of samples (test vectors).

Returns
-------
C : ndarray of shape (n_samples,) or (n_samples, n_classes)
    Decision function values related to each class, per sample.
    In the two-class case, the shape is [n_samples,], giving the
    log likelihood ratio of the positive class.
*)

val fit : x:Arr.t -> y:Arr.t -> t -> t
(**
Fit the model according to the given training data and parameters.

    .. versionchanged:: 0.19
       ``store_covariances`` has been moved to main constructor as
       ``store_covariance``

    .. versionchanged:: 0.19
       ``tol`` has been moved to main constructor.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training vector, where n_samples is the number of samples and
    n_features is the number of features.

y : array, shape = [n_samples]
    Target values (integers)
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
Perform classification on an array of test vectors X.

The predicted class C for each sample in X is returned.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
C : ndarray of shape (n_samples,)
*)

val predict_log_proba : x:Arr.t -> t -> Arr.t
(**
Return posterior probabilities of classification.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Array of samples/test vectors.

Returns
-------
C : ndarray of shape (n_samples, n_classes)
    Posterior log-probabilities of classification per class.
*)

val predict_proba : x:Arr.t -> t -> Arr.t
(**
Return posterior probabilities of classification.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Array of samples/test vectors.

Returns
-------
C : ndarray of shape (n_samples, n_classes)
    Posterior probabilities of classification per class.
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


(** Attribute covariance_: get value or raise Not_found if None.*)
val covariance_ : t -> Py.Object.t

(** Attribute covariance_: get value as an option. *)
val covariance_opt : t -> (Py.Object.t) option


(** Attribute means_: get value or raise Not_found if None.*)
val means_ : t -> Arr.t

(** Attribute means_: get value as an option. *)
val means_opt : t -> (Arr.t) option


(** Attribute priors_: get value or raise Not_found if None.*)
val priors_ : t -> Arr.t

(** Attribute priors_: get value as an option. *)
val priors_opt : t -> (Arr.t) option


(** Attribute rotations_: get value or raise Not_found if None.*)
val rotations_ : t -> Arr.List.t

(** Attribute rotations_: get value as an option. *)
val rotations_opt : t -> (Arr.List.t) option


(** Attribute scalings_: get value or raise Not_found if None.*)
val scalings_ : t -> Arr.List.t

(** Attribute scalings_: get value as an option. *)
val scalings_opt : t -> (Arr.List.t) option


(** Attribute classes_: get value or raise Not_found if None.*)
val classes_ : t -> Arr.t

(** Attribute classes_: get value as an option. *)
val classes_opt : t -> (Arr.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module StandardScaler : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?copy:bool -> ?with_mean:bool -> ?with_std:bool -> unit -> t
(**
Standardize features by removing the mean and scaling to unit variance

The standard score of a sample `x` is calculated as:

    z = (x - u) / s

where `u` is the mean of the training samples or zero if `with_mean=False`,
and `s` is the standard deviation of the training samples or one if
`with_std=False`.

Centering and scaling happen independently on each feature by computing
the relevant statistics on the samples in the training set. Mean and
standard deviation are then stored to be used on later data using
:meth:`transform`.

Standardization of a dataset is a common requirement for many
machine learning estimators: they might behave badly if the
individual features do not more or less look like standard normally
distributed data (e.g. Gaussian with 0 mean and unit variance).

For instance many elements used in the objective function of
a learning algorithm (such as the RBF kernel of Support Vector
Machines or the L1 and L2 regularizers of linear models) assume that
all features are centered around 0 and have variance in the same
order. If a feature has a variance that is orders of magnitude larger
that others, it might dominate the objective function and make the
estimator unable to learn from other features correctly as expected.

This scaler can also be applied to sparse CSR or CSC matrices by passing
`with_mean=False` to avoid breaking the sparsity structure of the data.

Read more in the :ref:`User Guide <preprocessing_scaler>`.

Parameters
----------
copy : boolean, optional, default True
    If False, try to avoid a copy and do inplace scaling instead.
    This is not guaranteed to always work inplace; e.g. if the data is
    not a NumPy array or scipy.sparse CSR matrix, a copy may still be
    returned.

with_mean : boolean, True by default
    If True, center the data before scaling.
    This does not work (and will raise an exception) when attempted on
    sparse matrices, because centering them entails building a dense
    matrix which in common use cases is likely to be too large to fit in
    memory.

with_std : boolean, True by default
    If True, scale the data to unit variance (or equivalently,
    unit standard deviation).

Attributes
----------
scale_ : ndarray or None, shape (n_features,)
    Per feature relative scaling of the data. This is calculated using
    `np.sqrt(var_)`. Equal to ``None`` when ``with_std=False``.

    .. versionadded:: 0.17
       *scale_*

mean_ : ndarray or None, shape (n_features,)
    The mean value for each feature in the training set.
    Equal to ``None`` when ``with_mean=False``.

var_ : ndarray or None, shape (n_features,)
    The variance for each feature in the training set. Used to compute
    `scale_`. Equal to ``None`` when ``with_std=False``.

n_samples_seen_ : int or array, shape (n_features,)
    The number of samples processed by the estimator for each feature.
    If there are not missing samples, the ``n_samples_seen`` will be an
    integer, otherwise it will be an array.
    Will be reset on new calls to fit, but increments across
    ``partial_fit`` calls.

Examples
--------
>>> from sklearn.preprocessing import StandardScaler
>>> data = [[0, 0], [0, 0], [1, 1], [1, 1]]
>>> scaler = StandardScaler()
>>> print(scaler.fit(data))
StandardScaler()
>>> print(scaler.mean_)
[0.5 0.5]
>>> print(scaler.transform(data))
[[-1. -1.]
 [-1. -1.]
 [ 1.  1.]
 [ 1.  1.]]
>>> print(scaler.transform([[2, 2]]))
[[3. 3.]]

See also
--------
scale: Equivalent function without the estimator API.

:class:`sklearn.decomposition.PCA`
    Further removes the linear correlation across features with 'whiten=True'.

Notes
-----
NaNs are treated as missing values: disregarded in fit, and maintained in
transform.

We use a biased estimator for the standard deviation, equivalent to
`numpy.std(x, ddof=0)`. Note that the choice of `ddof` is unlikely to
affect model performance.

For a comparison of the different scalers, transformers, and normalizers,
see :ref:`examples/preprocessing/plot_all_scaling.py
<sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
*)

val fit : ?y:Py.Object.t -> x:Arr.t -> t -> t
(**
Compute the mean and std to be used for later scaling.

Parameters
----------
X : {array-like, sparse matrix}, shape [n_samples, n_features]
    The data used to compute the mean and standard deviation
    used for later scaling along the features axis.

y
    Ignored
*)

val fit_transform : ?y:Arr.t -> ?fit_params:(string * Py.Object.t) list -> x:Arr.t -> t -> Arr.t
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

val inverse_transform : ?copy:Py.Object.t -> x:Arr.t -> t -> Arr.t
(**
Scale back the data to the original representation

Parameters
----------
X : array-like, shape [n_samples, n_features]
    The data used to scale along the features axis.
copy : bool, optional (default: None)
    Copy the input X or not.

Returns
-------
X_tr : array-like, shape [n_samples, n_features]
    Transformed array.
*)

val partial_fit : ?y:Py.Object.t -> x:Arr.t -> t -> t
(**
Online computation of mean and std on X for later scaling.

All of X is processed as a single batch. This is intended for cases
when :meth:`fit` is not feasible due to very large number of
`n_samples` or because X is read from a continuous stream.

The algorithm for incremental mean and std is given in Equation 1.5a,b
in Chan, Tony F., Gene H. Golub, and Randall J. LeVeque. "Algorithms
for computing the sample variance: Analysis and recommendations."
The American Statistician 37.3 (1983): 242-247:

Parameters
----------
X : {array-like, sparse matrix}, shape [n_samples, n_features]
    The data used to compute the mean and standard deviation
    used for later scaling along the features axis.

y : None
    Ignored.

Returns
-------
self : object
    Transformer instance.
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

val transform : ?copy:Py.Object.t -> x:Arr.t -> t -> Arr.t
(**
Perform standardization by centering and scaling

Parameters
----------
X : array-like, shape [n_samples, n_features]
    The data used to scale along the features axis.
copy : bool, optional (default: None)
    Copy the input X or not.
*)


(** Attribute scale_: get value or raise Not_found if None.*)
val scale_ : t -> Arr.t

(** Attribute scale_: get value as an option. *)
val scale_opt : t -> (Arr.t) option


(** Attribute mean_: get value or raise Not_found if None.*)
val mean_ : t -> Arr.t

(** Attribute mean_: get value as an option. *)
val mean_opt : t -> (Arr.t) option


(** Attribute var_: get value or raise Not_found if None.*)
val var_ : t -> Arr.t

(** Attribute var_: get value as an option. *)
val var_opt : t -> (Arr.t) option


(** Attribute n_samples_seen_: get value or raise Not_found if None.*)
val n_samples_seen_ : t -> [`I of int | `Arr of Arr.t]

(** Attribute n_samples_seen_: get value as an option. *)
val n_samples_seen_opt : t -> ([`I of int | `Arr of Arr.t]) option


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

val fit_transform : ?y:Arr.t -> ?fit_params:(string * Py.Object.t) list -> x:Arr.t -> t -> Arr.t
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

val check_X_y : ?accept_sparse:[`S of string | `Bool of bool | `StringList of string list] -> ?accept_large_sparse:bool -> ?dtype:[`S of string | `Dtype of Py.Object.t | `TypeList of Py.Object.t | `None] -> ?order:[`F | `C] -> ?copy:bool -> ?force_all_finite:[`Bool of bool | `Allow_nan] -> ?ensure_2d:bool -> ?allow_nd:bool -> ?multi_output:bool -> ?ensure_min_samples:int -> ?ensure_min_features:int -> ?y_numeric:bool -> ?warn_on_dtype:bool -> ?estimator:[`S of string | `Estimator of Py.Object.t] -> x:Arr.t -> y:Arr.t -> unit -> (Py.Object.t * Py.Object.t)
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

dtype : string, type, list of types or None (default="numeric")
    Data type of result. If None, the dtype of the input is preserved.
    If "numeric", dtype is preserved unless array.dtype is object.
    If dtype is a list of types, conversion on the first type is only
    performed if the dtype of the input is not in the list.

order : 'F', 'C' or None (default=None)
    Whether an array will be forced to be fortran or c-style.

copy : boolean (default=False)
    Whether a forced copy will be triggered. If copy=False, a copy might
    be triggered by a conversion.

force_all_finite : boolean or 'allow-nan', (default=True)
    Whether to raise an error on np.inf and np.nan in X. This parameter
    does not influence whether y can have np.inf or np.nan values.
    The possibilities are:

    - True: Force all values of X to be finite.
    - False: accept both np.inf and np.nan in X.
    - 'allow-nan': accept only np.nan values in X. Values cannot be
      infinite.

    .. versionadded:: 0.20
       ``force_all_finite`` accepts the string ``'allow-nan'``.

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
X_converted : object
    The converted and validated X.

y_converted : object
    The converted and validated y.
*)

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

val check_classification_targets : y:Arr.t -> unit -> Py.Object.t
(**
Ensure that target y is of a non-regression type.

Only the following target types (as defined in type_of_target) are allowed:
    'binary', 'multiclass', 'multiclass-multioutput',
    'multilabel-indicator', 'multilabel-sequences'

Parameters
----------
y : array-like
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

val empirical_covariance : ?assume_centered:bool -> x:Arr.t -> unit -> Py.Object.t
(**
Computes the Maximum likelihood covariance estimator


Parameters
----------
X : ndarray, shape (n_samples, n_features)
    Data from which to compute the covariance estimate

assume_centered : boolean
    If True, data will not be centered before computation.
    Useful when working with data whose mean is almost, but not exactly
    zero.
    If False, data will be centered before computation.

Returns
-------
covariance : 2D ndarray, shape (n_features, n_features)
    Empirical covariance (Maximum Likelihood Estimator).
*)

val ledoit_wolf : ?assume_centered:bool -> ?block_size:int -> x:Arr.t -> unit -> (Arr.t * float)
(**
Estimates the shrunk Ledoit-Wolf covariance matrix.

Read more in the :ref:`User Guide <shrunk_covariance>`.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Data from which to compute the covariance estimate

assume_centered : boolean, default=False
    If True, data will not be centered before computation.
    Useful to work with data whose mean is significantly equal to
    zero but is not exactly zero.
    If False, data will be centered before computation.

block_size : int, default=1000
    Size of the blocks into which the covariance matrix will be split.
    This is purely a memory optimization and does not affect results.

Returns
-------
shrunk_cov : array-like, shape (n_features, n_features)
    Shrunk covariance.

shrinkage : float
    Coefficient in the convex combination used for the computation
    of the shrunk estimate.

Notes
-----
The regularized (shrunk) covariance is:

(1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

where mu = trace(cov) / n_features
*)

val shrunk_covariance : ?shrinkage:[`F of float | `T0_shrinkage_1 of Py.Object.t] -> emp_cov:Arr.t -> unit -> Arr.t
(**
Calculates a covariance matrix shrunk on the diagonal

Read more in the :ref:`User Guide <shrunk_covariance>`.

Parameters
----------
emp_cov : array-like, shape (n_features, n_features)
    Covariance matrix to be shrunk

shrinkage : float, 0 <= shrinkage <= 1
    Coefficient in the convex combination used for the computation
    of the shrunk estimate.

Returns
-------
shrunk_cov : array-like
    Shrunk covariance.

Notes
-----
The regularized (shrunk) covariance is given by:

(1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

where mu = trace(cov) / n_features
*)

val softmax : ?copy:bool -> x:Py.Object.t -> unit -> Arr.t
(**
Calculate the softmax function.

The softmax function is calculated by
np.exp(X) / np.sum(np.exp(X), axis=1)

This will cause overflow when large values are exponentiated.
Hence the largest value in each row is subtracted from each data
point to prevent this.

Parameters
----------
X : array-like of floats, shape (M, N)
    Argument to the logistic function

copy : bool, optional
    Copy X or not.

Returns
-------
out : array, shape (M, N)
    Softmax function evaluated at every point in x
*)

val unique_labels : Py.Object.t list -> Arr.t
(**
Extract an ordered array of unique labels

We don't allow:
    - mix of multilabel and multiclass (single label) targets
    - mix of label indicator matrix and anything else,
      because there are no explicit labels)
    - mix of label indicator matrices of different sizes
    - mix of string and integer labels

At the moment, we also don't allow "multiclass-multioutput" input type.

Parameters
----------
*ys : array-likes

Returns
-------
out : numpy array of shape [n_unique_labels]
    An ordered array of unique labels.

Examples
--------
>>> from sklearn.utils.multiclass import unique_labels
>>> unique_labels([3, 5, 5, 5, 7, 7])
array([3, 5, 7])
>>> unique_labels([1, 2, 3, 4], [2, 2, 3, 4])
array([1, 2, 3, 4])
>>> unique_labels([1, 2, 10], [5, 11])
array([ 1,  2,  5, 10, 11])
*)

