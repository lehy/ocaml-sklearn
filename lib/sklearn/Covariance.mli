(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module EllipticEnvelope : sig
type tag = [`EllipticEnvelope]
type t = [`BaseEstimator | `EllipticEnvelope | `Object | `OutlierMixin] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_estimator : t -> [`BaseEstimator] Obj.t
val as_outlier : t -> [`OutlierMixin] Obj.t
val create : ?store_precision:bool -> ?assume_centered:bool -> ?support_fraction:float -> ?contamination:float -> ?random_state:int -> unit -> t
(**
An object for detecting outliers in a Gaussian distributed dataset.

Read more in the :ref:`User Guide <outlier_detection>`.

Parameters
----------
store_precision : bool, default=True
    Specify if the estimated precision is stored.

assume_centered : bool, default=False
    If True, the support of robust location and covariance estimates
    is computed, and a covariance estimate is recomputed from it,
    without centering the data.
    Useful to work with data whose mean is significantly equal to
    zero but is not exactly zero.
    If False, the robust location and covariance are directly computed
    with the FastMCD algorithm without additional treatment.

support_fraction : float, default=None
    The proportion of points to be included in the support of the raw
    MCD estimate. If None, the minimum value of support_fraction will
    be used within the algorithm: `[n_sample + n_features + 1] / 2`.
    Range is (0, 1).

contamination : float, default=0.1
    The amount of contamination of the data set, i.e. the proportion
    of outliers in the data set. Range is (0, 0.5).

random_state : int or RandomState instance, default=None
    Determines the pseudo random number generator for shuffling
    the data. Pass an int for reproducible results across multiple function
    calls. See :term: `Glossary <random_state>`.

Attributes
----------
location_ : ndarray of shape (n_features,)
    Estimated robust location

covariance_ : ndarray of shape (n_features, n_features)
    Estimated robust covariance matrix

precision_ : ndarray of shape (n_features, n_features)
    Estimated pseudo inverse matrix.
    (stored only if store_precision is True)

support_ : ndarray of shape (n_samples,)
    A mask of the observations that have been used to compute the
    robust estimates of location and shape.

offset_ : float
    Offset used to define the decision function from the raw scores.
    We have the relation: ``decision_function = score_samples - offset_``.
    The offset depends on the contamination parameter and is defined in
    such a way we obtain the expected number of outliers (samples with
    decision function < 0) in training.

    .. versionadded:: 0.20

raw_location_ : ndarray of shape (n_features,)
    The raw robust estimated location before correction and re-weighting.

raw_covariance_ : ndarray of shape (n_features, n_features)
    The raw robust estimated covariance before correction and re-weighting.

raw_support_ : ndarray of shape (n_samples,)
    A mask of the observations that have been used to compute
    the raw robust estimates of location and shape, before correction
    and re-weighting.

dist_ : ndarray of shape (n_samples,)
    Mahalanobis distances of the training set (on which :meth:`fit` is
    called) observations.

Examples
--------
>>> import numpy as np
>>> from sklearn.covariance import EllipticEnvelope
>>> true_cov = np.array([[.8, .3],
...                      [.3, .4]])
>>> X = np.random.RandomState(0).multivariate_normal(mean=[0, 0],
...                                                  cov=true_cov,
...                                                  size=500)
>>> cov = EllipticEnvelope(random_state=0).fit(X)
>>> # predict returns 1 for an inlier and -1 for an outlier
>>> cov.predict([[0, 0],
...              [3, 3]])
array([ 1, -1])
>>> cov.covariance_
array([[0.7411..., 0.2535...],
       [0.2535..., 0.3053...]])
>>> cov.location_
array([0.0813... , 0.0427...])

See Also
--------
EmpiricalCovariance, MinCovDet

Notes
-----
Outlier detection from covariance estimation may break or not
perform well in high-dimensional settings. In particular, one will
always take care to work with ``n_samples > n_features ** 2``.

References
----------
.. [1] Rousseeuw, P.J., Van Driessen, K. 'A fast algorithm for the
   minimum covariance determinant estimator' Technometrics 41(3), 212
   (1999)
*)

val correct_covariance : data:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Apply a correction to raw Minimum Covariance Determinant estimates.

Correction using the empirical correction factor suggested
by Rousseeuw and Van Driessen in [RVD]_.

Parameters
----------
data : array-like of shape (n_samples, n_features)
    The data matrix, with p features and n samples.
    The data set must be the one which was used to compute
    the raw estimates.

Returns
-------
covariance_corrected : ndarray of shape (n_features, n_features)
    Corrected robust covariance estimate.

References
----------

.. [RVD] A Fast Algorithm for the Minimum Covariance
    Determinant Estimator, 1999, American Statistical Association
    and the American Society for Quality, TECHNOMETRICS
*)

val decision_function : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Compute the decision function of the given observations.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The data matrix.

Returns
-------
decision : ndarray of shape (n_samples, )
    Decision function of the samples.
    It is equal to the shifted Mahalanobis distances.
    The threshold for being an outlier is 0, which ensures a
    compatibility with other outlier detection algorithms.
*)

val error_norm : ?norm:[`Frobenius | `Spectral] -> ?scaling:bool -> ?squared:bool -> comp_cov:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> float
(**
Computes the Mean Squared Error between two covariance estimators.
(In the sense of the Frobenius norm).

Parameters
----------
comp_cov : array-like of shape (n_features, n_features)
    The covariance to compare with.

norm : {'frobenius', 'spectral'}, default='frobenius'
    The type of norm used to compute the error. Available error types:
    - 'frobenius' (default): sqrt(tr(A^t.A))
    - 'spectral': sqrt(max(eigenvalues(A^t.A))
    where A is the error ``(comp_cov - self.covariance_)``.

scaling : bool, default=True
    If True (default), the squared error norm is divided by n_features.
    If False, the squared error norm is not rescaled.

squared : bool, default=True
    Whether to compute the squared error norm or the error norm.
    If True (default), the squared error norm is returned.
    If False, the error norm is returned.

Returns
-------
result : float
    The Mean Squared Error (in the sense of the Frobenius norm) between
    `self` and `comp_cov` covariance estimators.
*)

val fit : ?y:Py.Object.t -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fit the EllipticEnvelope model.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Training data.

y : Ignored
    Not used, present for API consistency by convention.
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

val get_precision : [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Getter for the precision matrix.

Returns
-------
precision_ : array-like of shape (n_features, n_features)
    The precision matrix associated to the current covariance object.
*)

val mahalanobis : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Computes the squared Mahalanobis distances of given observations.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The observations, the Mahalanobis distances of the which we
    compute. Observations are assumed to be drawn from the same
    distribution than the data used in fit.

Returns
-------
dist : ndarray of shape (n_samples,)
    Squared Mahalanobis distances of the observations.
*)

val predict : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Predict the labels (1 inlier, -1 outlier) of X according to the
fitted model.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The data matrix.

Returns
-------
is_inlier : ndarray of shape (n_samples,)
    Returns -1 for anomalies/outliers and +1 for inliers.
*)

val reweight_covariance : data:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> ([>`ArrayLike] Np.Obj.t * [>`ArrayLike] Np.Obj.t * Py.Object.t)
(**
Re-weight raw Minimum Covariance Determinant estimates.

Re-weight observations using Rousseeuw's method (equivalent to
deleting outlying observations from the data set before
computing location and covariance estimates) described
in [RVDriessen]_.

Parameters
----------
data : array-like of shape (n_samples, n_features)
    The data matrix, with p features and n samples.
    The data set must be the one which was used to compute
    the raw estimates.

Returns
-------
location_reweighted : ndarray of shape (n_features,)
    Re-weighted robust location estimate.

covariance_reweighted : ndarray of shape (n_features, n_features)
    Re-weighted robust covariance estimate.

support_reweighted : ndarray of shape (n_samples,), dtype=bool
    A mask of the observations that have been used to compute
    the re-weighted robust location and covariance estimates.

References
----------

.. [RVDriessen] A Fast Algorithm for the Minimum Covariance
    Determinant Estimator, 1999, American Statistical Association
    and the American Society for Quality, TECHNOMETRICS
*)

val score : ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> float
(**
Returns the mean accuracy on the given test data and labels.

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
    Mean accuracy of self.predict(X) w.r.t. y.
*)

val score_samples : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Compute the negative Mahalanobis distances.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The data matrix.

Returns
-------
negative_mahal_distances : array-like of shape (n_samples,)
    Opposite of the Mahalanobis distances.
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


(** Attribute location_: get value or raise Not_found if None.*)
val location_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute location_: get value as an option. *)
val location_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute covariance_: get value or raise Not_found if None.*)
val covariance_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute covariance_: get value as an option. *)
val covariance_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute precision_: get value or raise Not_found if None.*)
val precision_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute precision_: get value as an option. *)
val precision_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute support_: get value or raise Not_found if None.*)
val support_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute support_: get value as an option. *)
val support_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute offset_: get value or raise Not_found if None.*)
val offset_ : t -> float

(** Attribute offset_: get value as an option. *)
val offset_opt : t -> (float) option


(** Attribute raw_location_: get value or raise Not_found if None.*)
val raw_location_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute raw_location_: get value as an option. *)
val raw_location_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute raw_covariance_: get value or raise Not_found if None.*)
val raw_covariance_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute raw_covariance_: get value as an option. *)
val raw_covariance_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute raw_support_: get value or raise Not_found if None.*)
val raw_support_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute raw_support_: get value as an option. *)
val raw_support_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute dist_: get value or raise Not_found if None.*)
val dist_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute dist_: get value as an option. *)
val dist_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module EmpiricalCovariance : sig
type tag = [`EmpiricalCovariance]
type t = [`BaseEstimator | `EmpiricalCovariance | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_estimator : t -> [`BaseEstimator] Obj.t
val create : ?store_precision:bool -> ?assume_centered:bool -> unit -> t
(**
Maximum likelihood covariance estimator

Read more in the :ref:`User Guide <covariance>`.

Parameters
----------
store_precision : bool, default=True
    Specifies if the estimated precision is stored.

assume_centered : bool, default=False
    If True, data are not centered before computation.
    Useful when working with data whose mean is almost, but not exactly
    zero.
    If False (default), data are centered before computation.

Attributes
----------
location_ : ndarray of shape (n_features,)
    Estimated location, i.e. the estimated mean.

covariance_ : ndarray of shape (n_features, n_features)
    Estimated covariance matrix

precision_ : ndarray of shape (n_features, n_features)
    Estimated pseudo-inverse matrix.
    (stored only if store_precision is True)

Examples
--------
>>> import numpy as np
>>> from sklearn.covariance import EmpiricalCovariance
>>> from sklearn.datasets import make_gaussian_quantiles
>>> real_cov = np.array([[.8, .3],
...                      [.3, .4]])
>>> rng = np.random.RandomState(0)
>>> X = rng.multivariate_normal(mean=[0, 0],
...                             cov=real_cov,
...                             size=500)
>>> cov = EmpiricalCovariance().fit(X)
>>> cov.covariance_
array([[0.7569..., 0.2818...],
       [0.2818..., 0.3928...]])
>>> cov.location_
array([0.0622..., 0.0193...])
*)

val error_norm : ?norm:[`Frobenius | `Spectral] -> ?scaling:bool -> ?squared:bool -> comp_cov:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> float
(**
Computes the Mean Squared Error between two covariance estimators.
(In the sense of the Frobenius norm).

Parameters
----------
comp_cov : array-like of shape (n_features, n_features)
    The covariance to compare with.

norm : {'frobenius', 'spectral'}, default='frobenius'
    The type of norm used to compute the error. Available error types:
    - 'frobenius' (default): sqrt(tr(A^t.A))
    - 'spectral': sqrt(max(eigenvalues(A^t.A))
    where A is the error ``(comp_cov - self.covariance_)``.

scaling : bool, default=True
    If True (default), the squared error norm is divided by n_features.
    If False, the squared error norm is not rescaled.

squared : bool, default=True
    Whether to compute the squared error norm or the error norm.
    If True (default), the squared error norm is returned.
    If False, the error norm is returned.

Returns
-------
result : float
    The Mean Squared Error (in the sense of the Frobenius norm) between
    `self` and `comp_cov` covariance estimators.
*)

val fit : ?y:Py.Object.t -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fits the Maximum Likelihood Estimator covariance model
according to the given training data and parameters.

Parameters
----------
X : array-like of shape (n_samples, n_features)
  Training data, where n_samples is the number of samples and
  n_features is the number of features.

y : Ignored
    Not used, present for API consistence purpose.

Returns
-------
self : object
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

val get_precision : [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Getter for the precision matrix.

Returns
-------
precision_ : array-like of shape (n_features, n_features)
    The precision matrix associated to the current covariance object.
*)

val mahalanobis : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Computes the squared Mahalanobis distances of given observations.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The observations, the Mahalanobis distances of the which we
    compute. Observations are assumed to be drawn from the same
    distribution than the data used in fit.

Returns
-------
dist : ndarray of shape (n_samples,)
    Squared Mahalanobis distances of the observations.
*)

val score : ?y:Py.Object.t -> x_test:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> float
(**
Computes the log-likelihood of a Gaussian data set with
`self.covariance_` as an estimator of its covariance matrix.

Parameters
----------
X_test : array-like of shape (n_samples, n_features)
    Test data of which we compute the likelihood, where n_samples is
    the number of samples and n_features is the number of features.
    X_test is assumed to be drawn from the same distribution than
    the data used in fit (including centering).

y : Ignored
    Not used, present for API consistence purpose.

Returns
-------
res : float
    The likelihood of the data set with `self.covariance_` as an
    estimator of its covariance matrix.
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


(** Attribute location_: get value or raise Not_found if None.*)
val location_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute location_: get value as an option. *)
val location_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute covariance_: get value or raise Not_found if None.*)
val covariance_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute covariance_: get value as an option. *)
val covariance_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute precision_: get value or raise Not_found if None.*)
val precision_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute precision_: get value as an option. *)
val precision_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module GraphicalLasso : sig
type tag = [`GraphicalLasso]
type t = [`BaseEstimator | `GraphicalLasso | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_estimator : t -> [`BaseEstimator] Obj.t
val create : ?alpha:float -> ?mode:[`Cd | `Lars] -> ?tol:float -> ?enet_tol:float -> ?max_iter:int -> ?verbose:int -> ?assume_centered:bool -> unit -> t
(**
Sparse inverse covariance estimation with an l1-penalized estimator.

Read more in the :ref:`User Guide <sparse_inverse_covariance>`.

.. versionchanged:: v0.20
    GraphLasso has been renamed to GraphicalLasso

Parameters
----------
alpha : float, default=0.01
    The regularization parameter: the higher alpha, the more
    regularization, the sparser the inverse covariance.
    Range is (0, inf].

mode : {'cd', 'lars'}, default='cd'
    The Lasso solver to use: coordinate descent or LARS. Use LARS for
    very sparse underlying graphs, where p > n. Elsewhere prefer cd
    which is more numerically stable.

tol : float, default=1e-4
    The tolerance to declare convergence: if the dual gap goes below
    this value, iterations are stopped. Range is (0, inf].

enet_tol : float, default=1e-4
    The tolerance for the elastic net solver used to calculate the descent
    direction. This parameter controls the accuracy of the search direction
    for a given column update, not of the overall parameter estimate. Only
    used for mode='cd'. Range is (0, inf].

max_iter : int, default=100
    The maximum number of iterations.

verbose : bool, default=False
    If verbose is True, the objective function and dual gap are
    plotted at each iteration.

assume_centered : bool, default=False
    If True, data are not centered before computation.
    Useful when working with data whose mean is almost, but not exactly
    zero.
    If False, data are centered before computation.

Attributes
----------
location_ : ndarray of shape (n_features,)
    Estimated location, i.e. the estimated mean.

covariance_ : ndarray of shape (n_features, n_features)
    Estimated covariance matrix

precision_ : ndarray of shape (n_features, n_features)
    Estimated pseudo inverse matrix.

n_iter_ : int
    Number of iterations run.

Examples
--------
>>> import numpy as np
>>> from sklearn.covariance import GraphicalLasso
>>> true_cov = np.array([[0.8, 0.0, 0.2, 0.0],
...                      [0.0, 0.4, 0.0, 0.0],
...                      [0.2, 0.0, 0.3, 0.1],
...                      [0.0, 0.0, 0.1, 0.7]])
>>> np.random.seed(0)
>>> X = np.random.multivariate_normal(mean=[0, 0, 0, 0],
...                                   cov=true_cov,
...                                   size=200)
>>> cov = GraphicalLasso().fit(X)
>>> np.around(cov.covariance_, decimals=3)
array([[0.816, 0.049, 0.218, 0.019],
       [0.049, 0.364, 0.017, 0.034],
       [0.218, 0.017, 0.322, 0.093],
       [0.019, 0.034, 0.093, 0.69 ]])
>>> np.around(cov.location_, decimals=3)
array([0.073, 0.04 , 0.038, 0.143])

See Also
--------
graphical_lasso, GraphicalLassoCV
*)

val error_norm : ?norm:[`Frobenius | `Spectral] -> ?scaling:bool -> ?squared:bool -> comp_cov:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> float
(**
Computes the Mean Squared Error between two covariance estimators.
(In the sense of the Frobenius norm).

Parameters
----------
comp_cov : array-like of shape (n_features, n_features)
    The covariance to compare with.

norm : {'frobenius', 'spectral'}, default='frobenius'
    The type of norm used to compute the error. Available error types:
    - 'frobenius' (default): sqrt(tr(A^t.A))
    - 'spectral': sqrt(max(eigenvalues(A^t.A))
    where A is the error ``(comp_cov - self.covariance_)``.

scaling : bool, default=True
    If True (default), the squared error norm is divided by n_features.
    If False, the squared error norm is not rescaled.

squared : bool, default=True
    Whether to compute the squared error norm or the error norm.
    If True (default), the squared error norm is returned.
    If False, the error norm is returned.

Returns
-------
result : float
    The Mean Squared Error (in the sense of the Frobenius norm) between
    `self` and `comp_cov` covariance estimators.
*)

val fit : ?y:Py.Object.t -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fits the GraphicalLasso model to X.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Data from which to compute the covariance estimate

y : Ignored
    Not used, present for API consistence purpose.

Returns
-------
self : object
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

val get_precision : [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Getter for the precision matrix.

Returns
-------
precision_ : array-like of shape (n_features, n_features)
    The precision matrix associated to the current covariance object.
*)

val mahalanobis : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Computes the squared Mahalanobis distances of given observations.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The observations, the Mahalanobis distances of the which we
    compute. Observations are assumed to be drawn from the same
    distribution than the data used in fit.

Returns
-------
dist : ndarray of shape (n_samples,)
    Squared Mahalanobis distances of the observations.
*)

val score : ?y:Py.Object.t -> x_test:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> float
(**
Computes the log-likelihood of a Gaussian data set with
`self.covariance_` as an estimator of its covariance matrix.

Parameters
----------
X_test : array-like of shape (n_samples, n_features)
    Test data of which we compute the likelihood, where n_samples is
    the number of samples and n_features is the number of features.
    X_test is assumed to be drawn from the same distribution than
    the data used in fit (including centering).

y : Ignored
    Not used, present for API consistence purpose.

Returns
-------
res : float
    The likelihood of the data set with `self.covariance_` as an
    estimator of its covariance matrix.
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


(** Attribute location_: get value or raise Not_found if None.*)
val location_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute location_: get value as an option. *)
val location_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute covariance_: get value or raise Not_found if None.*)
val covariance_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute covariance_: get value as an option. *)
val covariance_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute precision_: get value or raise Not_found if None.*)
val precision_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute precision_: get value as an option. *)
val precision_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute n_iter_: get value or raise Not_found if None.*)
val n_iter_ : t -> int

(** Attribute n_iter_: get value as an option. *)
val n_iter_opt : t -> (int) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module GraphicalLassoCV : sig
type tag = [`GraphicalLassoCV]
type t = [`BaseEstimator | `GraphicalLassoCV | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_estimator : t -> [`BaseEstimator] Obj.t
val create : ?alphas:Py.Object.t -> ?n_refinements:int -> ?cv:[`BaseCrossValidator of [>`BaseCrossValidator] Np.Obj.t | `Arr of [>`ArrayLike] Np.Obj.t | `I of int] -> ?tol:float -> ?enet_tol:float -> ?max_iter:int -> ?mode:[`Cd | `Lars] -> ?n_jobs:int -> ?verbose:int -> ?assume_centered:bool -> unit -> t
(**
Sparse inverse covariance w/ cross-validated choice of the l1 penalty.

See glossary entry for :term:`cross-validation estimator`.

Read more in the :ref:`User Guide <sparse_inverse_covariance>`.

.. versionchanged:: v0.20
    GraphLassoCV has been renamed to GraphicalLassoCV

Parameters
----------
alphas : int or array-like of shape (n_alphas,), dtype=float, default=4
    If an integer is given, it fixes the number of points on the
    grids of alpha to be used. If a list is given, it gives the
    grid to be used. See the notes in the class docstring for
    more details. Range is (0, inf] when floats given.

n_refinements : int, default=4
    The number of times the grid is refined. Not used if explicit
    values of alphas are passed. Range is [1, inf).

cv : int, cross-validation generator or iterable, default=None
    Determines the cross-validation splitting strategy.
    Possible inputs for cv are:

    - None, to use the default 5-fold cross-validation,
    - integer, to specify the number of folds.
    - :term:`CV splitter`,
    - An iterable yielding (train, test) splits as arrays of indices.

    For integer/None inputs :class:`KFold` is used.

    Refer :ref:`User Guide <cross_validation>` for the various
    cross-validation strategies that can be used here.

    .. versionchanged:: 0.20
        ``cv`` default value if None changed from 3-fold to 5-fold.

tol : float, default=1e-4
    The tolerance to declare convergence: if the dual gap goes below
    this value, iterations are stopped. Range is (0, inf].

enet_tol : float, default=1e-4
    The tolerance for the elastic net solver used to calculate the descent
    direction. This parameter controls the accuracy of the search direction
    for a given column update, not of the overall parameter estimate. Only
    used for mode='cd'. Range is (0, inf].

max_iter : int, default=100
    Maximum number of iterations.

mode : {'cd', 'lars'}, default='cd'
    The Lasso solver to use: coordinate descent or LARS. Use LARS for
    very sparse underlying graphs, where number of features is greater
    than number of samples. Elsewhere prefer cd which is more numerically
    stable.

n_jobs : int, default=None
    number of jobs to run in parallel.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

    .. versionchanged:: v0.20
       `n_jobs` default changed from 1 to None

verbose : bool, default=False
    If verbose is True, the objective function and duality gap are
    printed at each iteration.

assume_centered : bool, default=False
    If True, data are not centered before computation.
    Useful when working with data whose mean is almost, but not exactly
    zero.
    If False, data are centered before computation.

Attributes
----------
location_ : ndarray of shape (n_features,)
    Estimated location, i.e. the estimated mean.

covariance_ : ndarray of shape (n_features, n_features)
    Estimated covariance matrix.

precision_ : ndarray of shape (n_features, n_features)
    Estimated precision matrix (inverse covariance).

alpha_ : float
    Penalization parameter selected.

cv_alphas_ : list of shape (n_alphas,), dtype=float
    All penalization parameters explored.

grid_scores_ : ndarray of shape (n_alphas, n_folds)
    Log-likelihood score on left-out data across folds.

n_iter_ : int
    Number of iterations run for the optimal alpha.

Examples
--------
>>> import numpy as np
>>> from sklearn.covariance import GraphicalLassoCV
>>> true_cov = np.array([[0.8, 0.0, 0.2, 0.0],
...                      [0.0, 0.4, 0.0, 0.0],
...                      [0.2, 0.0, 0.3, 0.1],
...                      [0.0, 0.0, 0.1, 0.7]])
>>> np.random.seed(0)
>>> X = np.random.multivariate_normal(mean=[0, 0, 0, 0],
...                                   cov=true_cov,
...                                   size=200)
>>> cov = GraphicalLassoCV().fit(X)
>>> np.around(cov.covariance_, decimals=3)
array([[0.816, 0.051, 0.22 , 0.017],
       [0.051, 0.364, 0.018, 0.036],
       [0.22 , 0.018, 0.322, 0.094],
       [0.017, 0.036, 0.094, 0.69 ]])
>>> np.around(cov.location_, decimals=3)
array([0.073, 0.04 , 0.038, 0.143])

See Also
--------
graphical_lasso, GraphicalLasso

Notes
-----
The search for the optimal penalization parameter (alpha) is done on an
iteratively refined grid: first the cross-validated scores on a grid are
computed, then a new refined grid is centered around the maximum, and so
on.

One of the challenges which is faced here is that the solvers can
fail to converge to a well-conditioned estimate. The corresponding
values of alpha then come out as missing values, but the optimum may
be close to these missing values.
*)

val error_norm : ?norm:[`Frobenius | `Spectral] -> ?scaling:bool -> ?squared:bool -> comp_cov:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> float
(**
Computes the Mean Squared Error between two covariance estimators.
(In the sense of the Frobenius norm).

Parameters
----------
comp_cov : array-like of shape (n_features, n_features)
    The covariance to compare with.

norm : {'frobenius', 'spectral'}, default='frobenius'
    The type of norm used to compute the error. Available error types:
    - 'frobenius' (default): sqrt(tr(A^t.A))
    - 'spectral': sqrt(max(eigenvalues(A^t.A))
    where A is the error ``(comp_cov - self.covariance_)``.

scaling : bool, default=True
    If True (default), the squared error norm is divided by n_features.
    If False, the squared error norm is not rescaled.

squared : bool, default=True
    Whether to compute the squared error norm or the error norm.
    If True (default), the squared error norm is returned.
    If False, the error norm is returned.

Returns
-------
result : float
    The Mean Squared Error (in the sense of the Frobenius norm) between
    `self` and `comp_cov` covariance estimators.
*)

val fit : ?y:Py.Object.t -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fits the GraphicalLasso covariance model to X.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Data from which to compute the covariance estimate

y : Ignored
    Not used, present for API consistence purpose.

Returns
-------
self : object
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

val get_precision : [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Getter for the precision matrix.

Returns
-------
precision_ : array-like of shape (n_features, n_features)
    The precision matrix associated to the current covariance object.
*)

val mahalanobis : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Computes the squared Mahalanobis distances of given observations.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The observations, the Mahalanobis distances of the which we
    compute. Observations are assumed to be drawn from the same
    distribution than the data used in fit.

Returns
-------
dist : ndarray of shape (n_samples,)
    Squared Mahalanobis distances of the observations.
*)

val score : ?y:Py.Object.t -> x_test:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> float
(**
Computes the log-likelihood of a Gaussian data set with
`self.covariance_` as an estimator of its covariance matrix.

Parameters
----------
X_test : array-like of shape (n_samples, n_features)
    Test data of which we compute the likelihood, where n_samples is
    the number of samples and n_features is the number of features.
    X_test is assumed to be drawn from the same distribution than
    the data used in fit (including centering).

y : Ignored
    Not used, present for API consistence purpose.

Returns
-------
res : float
    The likelihood of the data set with `self.covariance_` as an
    estimator of its covariance matrix.
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


(** Attribute location_: get value or raise Not_found if None.*)
val location_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute location_: get value as an option. *)
val location_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute covariance_: get value or raise Not_found if None.*)
val covariance_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute covariance_: get value as an option. *)
val covariance_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute precision_: get value or raise Not_found if None.*)
val precision_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute precision_: get value as an option. *)
val precision_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute alpha_: get value or raise Not_found if None.*)
val alpha_ : t -> float

(** Attribute alpha_: get value as an option. *)
val alpha_opt : t -> (float) option


(** Attribute cv_alphas_: get value or raise Not_found if None.*)
val cv_alphas_ : t -> Py.Object.t

(** Attribute cv_alphas_: get value as an option. *)
val cv_alphas_opt : t -> (Py.Object.t) option


(** Attribute grid_scores_: get value or raise Not_found if None.*)
val grid_scores_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute grid_scores_: get value as an option. *)
val grid_scores_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute n_iter_: get value or raise Not_found if None.*)
val n_iter_ : t -> int

(** Attribute n_iter_: get value as an option. *)
val n_iter_opt : t -> (int) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module LedoitWolf : sig
type tag = [`LedoitWolf]
type t = [`BaseEstimator | `LedoitWolf | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_estimator : t -> [`BaseEstimator] Obj.t
val create : ?store_precision:bool -> ?assume_centered:bool -> ?block_size:int -> unit -> t
(**
LedoitWolf Estimator

Ledoit-Wolf is a particular form of shrinkage, where the shrinkage
coefficient is computed using O. Ledoit and M. Wolf's formula as
described in 'A Well-Conditioned Estimator for Large-Dimensional
Covariance Matrices', Ledoit and Wolf, Journal of Multivariate
Analysis, Volume 88, Issue 2, February 2004, pages 365-411.

Read more in the :ref:`User Guide <shrunk_covariance>`.

Parameters
----------
store_precision : bool, default=True
    Specify if the estimated precision is stored.

assume_centered : bool, default=False
    If True, data will not be centered before computation.
    Useful when working with data whose mean is almost, but not exactly
    zero.
    If False (default), data will be centered before computation.

block_size : int, default=1000
    Size of the blocks into which the covariance matrix will be split
    during its Ledoit-Wolf estimation. This is purely a memory
    optimization and does not affect results.

Attributes
----------
covariance_ : ndarray of shape (n_features, n_features)
    Estimated covariance matrix.

location_ : ndarray of shape (n_features,)
    Estimated location, i.e. the estimated mean.

precision_ : ndarray of shape (n_features, n_features)
    Estimated pseudo inverse matrix.
    (stored only if store_precision is True)

shrinkage_ : float
    Coefficient in the convex combination used for the computation
    of the shrunk estimate. Range is [0, 1].

Examples
--------
>>> import numpy as np
>>> from sklearn.covariance import LedoitWolf
>>> real_cov = np.array([[.4, .2],
...                      [.2, .8]])
>>> np.random.seed(0)
>>> X = np.random.multivariate_normal(mean=[0, 0],
...                                   cov=real_cov,
...                                   size=50)
>>> cov = LedoitWolf().fit(X)
>>> cov.covariance_
array([[0.4406..., 0.1616...],
       [0.1616..., 0.8022...]])
>>> cov.location_
array([ 0.0595... , -0.0075...])

Notes
-----
The regularised covariance is:

(1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

where mu = trace(cov) / n_features
and shrinkage is given by the Ledoit and Wolf formula (see References)

References
----------
'A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices',
Ledoit and Wolf, Journal of Multivariate Analysis, Volume 88, Issue 2,
February 2004, pages 365-411.
*)

val error_norm : ?norm:[`Frobenius | `Spectral] -> ?scaling:bool -> ?squared:bool -> comp_cov:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> float
(**
Computes the Mean Squared Error between two covariance estimators.
(In the sense of the Frobenius norm).

Parameters
----------
comp_cov : array-like of shape (n_features, n_features)
    The covariance to compare with.

norm : {'frobenius', 'spectral'}, default='frobenius'
    The type of norm used to compute the error. Available error types:
    - 'frobenius' (default): sqrt(tr(A^t.A))
    - 'spectral': sqrt(max(eigenvalues(A^t.A))
    where A is the error ``(comp_cov - self.covariance_)``.

scaling : bool, default=True
    If True (default), the squared error norm is divided by n_features.
    If False, the squared error norm is not rescaled.

squared : bool, default=True
    Whether to compute the squared error norm or the error norm.
    If True (default), the squared error norm is returned.
    If False, the error norm is returned.

Returns
-------
result : float
    The Mean Squared Error (in the sense of the Frobenius norm) between
    `self` and `comp_cov` covariance estimators.
*)

val fit : ?y:Py.Object.t -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fit the Ledoit-Wolf shrunk covariance model according to the given
training data and parameters.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training data, where `n_samples` is the number of samples
    and `n_features` is the number of features.
y : Ignored
    not used, present for API consistence purpose.

Returns
-------
self : object
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

val get_precision : [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Getter for the precision matrix.

Returns
-------
precision_ : array-like of shape (n_features, n_features)
    The precision matrix associated to the current covariance object.
*)

val mahalanobis : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Computes the squared Mahalanobis distances of given observations.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The observations, the Mahalanobis distances of the which we
    compute. Observations are assumed to be drawn from the same
    distribution than the data used in fit.

Returns
-------
dist : ndarray of shape (n_samples,)
    Squared Mahalanobis distances of the observations.
*)

val score : ?y:Py.Object.t -> x_test:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> float
(**
Computes the log-likelihood of a Gaussian data set with
`self.covariance_` as an estimator of its covariance matrix.

Parameters
----------
X_test : array-like of shape (n_samples, n_features)
    Test data of which we compute the likelihood, where n_samples is
    the number of samples and n_features is the number of features.
    X_test is assumed to be drawn from the same distribution than
    the data used in fit (including centering).

y : Ignored
    Not used, present for API consistence purpose.

Returns
-------
res : float
    The likelihood of the data set with `self.covariance_` as an
    estimator of its covariance matrix.
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


(** Attribute covariance_: get value or raise Not_found if None.*)
val covariance_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute covariance_: get value as an option. *)
val covariance_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute location_: get value or raise Not_found if None.*)
val location_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute location_: get value as an option. *)
val location_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute precision_: get value or raise Not_found if None.*)
val precision_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute precision_: get value as an option. *)
val precision_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute shrinkage_: get value or raise Not_found if None.*)
val shrinkage_ : t -> float

(** Attribute shrinkage_: get value as an option. *)
val shrinkage_opt : t -> (float) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MinCovDet : sig
type tag = [`MinCovDet]
type t = [`BaseEstimator | `MinCovDet | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_estimator : t -> [`BaseEstimator] Obj.t
val create : ?store_precision:bool -> ?assume_centered:bool -> ?support_fraction:float -> ?random_state:int -> unit -> t
(**
Minimum Covariance Determinant (MCD): robust estimator of covariance.

The Minimum Covariance Determinant covariance estimator is to be applied
on Gaussian-distributed data, but could still be relevant on data
drawn from a unimodal, symmetric distribution. It is not meant to be used
with multi-modal data (the algorithm used to fit a MinCovDet object is
likely to fail in such a case).
One should consider projection pursuit methods to deal with multi-modal
datasets.

Read more in the :ref:`User Guide <robust_covariance>`.

Parameters
----------
store_precision : bool, default=True
    Specify if the estimated precision is stored.

assume_centered : bool, default=False
    If True, the support of the robust location and the covariance
    estimates is computed, and a covariance estimate is recomputed from
    it, without centering the data.
    Useful to work with data whose mean is significantly equal to
    zero but is not exactly zero.
    If False, the robust location and covariance are directly computed
    with the FastMCD algorithm without additional treatment.

support_fraction : float, default=None
    The proportion of points to be included in the support of the raw
    MCD estimate. Default is None, which implies that the minimum
    value of support_fraction will be used within the algorithm:
    `(n_sample + n_features + 1) / 2`. The parameter must be in the range
    (0, 1).

random_state : int or RandomState instance, default=None
    Determines the pseudo random number generator for shuffling the data.
    Pass an int for reproducible results across multiple function calls.
    See :term: `Glossary <random_state>`.

Attributes
----------
raw_location_ : ndarray of shape (n_features,)
    The raw robust estimated location before correction and re-weighting.

raw_covariance_ : ndarray of shape (n_features, n_features)
    The raw robust estimated covariance before correction and re-weighting.

raw_support_ : ndarray of shape (n_samples,)
    A mask of the observations that have been used to compute
    the raw robust estimates of location and shape, before correction
    and re-weighting.

location_ : ndarray of shape (n_features,)
    Estimated robust location.

covariance_ : ndarray of shape (n_features, n_features)
    Estimated robust covariance matrix.

precision_ : ndarray of shape (n_features, n_features)
    Estimated pseudo inverse matrix.
    (stored only if store_precision is True)

support_ : ndarray of shape (n_samples,)
    A mask of the observations that have been used to compute
    the robust estimates of location and shape.

dist_ : ndarray of shape (n_samples,)
    Mahalanobis distances of the training set (on which :meth:`fit` is
    called) observations.

Examples
--------
>>> import numpy as np
>>> from sklearn.covariance import MinCovDet
>>> from sklearn.datasets import make_gaussian_quantiles
>>> real_cov = np.array([[.8, .3],
...                      [.3, .4]])
>>> rng = np.random.RandomState(0)
>>> X = rng.multivariate_normal(mean=[0, 0],
...                                   cov=real_cov,
...                                   size=500)
>>> cov = MinCovDet(random_state=0).fit(X)
>>> cov.covariance_
array([[0.7411..., 0.2535...],
       [0.2535..., 0.3053...]])
>>> cov.location_
array([0.0813... , 0.0427...])

References
----------

.. [Rouseeuw1984] P. J. Rousseeuw. Least median of squares regression.
    J. Am Stat Ass, 79:871, 1984.
.. [Rousseeuw] A Fast Algorithm for the Minimum Covariance Determinant
    Estimator, 1999, American Statistical Association and the American
    Society for Quality, TECHNOMETRICS
.. [ButlerDavies] R. W. Butler, P. L. Davies and M. Jhun,
    Asymptotics For The Minimum Covariance Determinant Estimator,
    The Annals of Statistics, 1993, Vol. 21, No. 3, 1385-1400
*)

val correct_covariance : data:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Apply a correction to raw Minimum Covariance Determinant estimates.

Correction using the empirical correction factor suggested
by Rousseeuw and Van Driessen in [RVD]_.

Parameters
----------
data : array-like of shape (n_samples, n_features)
    The data matrix, with p features and n samples.
    The data set must be the one which was used to compute
    the raw estimates.

Returns
-------
covariance_corrected : ndarray of shape (n_features, n_features)
    Corrected robust covariance estimate.

References
----------

.. [RVD] A Fast Algorithm for the Minimum Covariance
    Determinant Estimator, 1999, American Statistical Association
    and the American Society for Quality, TECHNOMETRICS
*)

val error_norm : ?norm:[`Frobenius | `Spectral] -> ?scaling:bool -> ?squared:bool -> comp_cov:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> float
(**
Computes the Mean Squared Error between two covariance estimators.
(In the sense of the Frobenius norm).

Parameters
----------
comp_cov : array-like of shape (n_features, n_features)
    The covariance to compare with.

norm : {'frobenius', 'spectral'}, default='frobenius'
    The type of norm used to compute the error. Available error types:
    - 'frobenius' (default): sqrt(tr(A^t.A))
    - 'spectral': sqrt(max(eigenvalues(A^t.A))
    where A is the error ``(comp_cov - self.covariance_)``.

scaling : bool, default=True
    If True (default), the squared error norm is divided by n_features.
    If False, the squared error norm is not rescaled.

squared : bool, default=True
    Whether to compute the squared error norm or the error norm.
    If True (default), the squared error norm is returned.
    If False, the error norm is returned.

Returns
-------
result : float
    The Mean Squared Error (in the sense of the Frobenius norm) between
    `self` and `comp_cov` covariance estimators.
*)

val fit : ?y:Py.Object.t -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fits a Minimum Covariance Determinant with the FastMCD algorithm.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training data, where `n_samples` is the number of samples
    and `n_features` is the number of features.

y: Ignored
    Not used, present for API consistence purpose.

Returns
-------
self : object
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

val get_precision : [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Getter for the precision matrix.

Returns
-------
precision_ : array-like of shape (n_features, n_features)
    The precision matrix associated to the current covariance object.
*)

val mahalanobis : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Computes the squared Mahalanobis distances of given observations.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The observations, the Mahalanobis distances of the which we
    compute. Observations are assumed to be drawn from the same
    distribution than the data used in fit.

Returns
-------
dist : ndarray of shape (n_samples,)
    Squared Mahalanobis distances of the observations.
*)

val reweight_covariance : data:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> ([>`ArrayLike] Np.Obj.t * [>`ArrayLike] Np.Obj.t * Py.Object.t)
(**
Re-weight raw Minimum Covariance Determinant estimates.

Re-weight observations using Rousseeuw's method (equivalent to
deleting outlying observations from the data set before
computing location and covariance estimates) described
in [RVDriessen]_.

Parameters
----------
data : array-like of shape (n_samples, n_features)
    The data matrix, with p features and n samples.
    The data set must be the one which was used to compute
    the raw estimates.

Returns
-------
location_reweighted : ndarray of shape (n_features,)
    Re-weighted robust location estimate.

covariance_reweighted : ndarray of shape (n_features, n_features)
    Re-weighted robust covariance estimate.

support_reweighted : ndarray of shape (n_samples,), dtype=bool
    A mask of the observations that have been used to compute
    the re-weighted robust location and covariance estimates.

References
----------

.. [RVDriessen] A Fast Algorithm for the Minimum Covariance
    Determinant Estimator, 1999, American Statistical Association
    and the American Society for Quality, TECHNOMETRICS
*)

val score : ?y:Py.Object.t -> x_test:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> float
(**
Computes the log-likelihood of a Gaussian data set with
`self.covariance_` as an estimator of its covariance matrix.

Parameters
----------
X_test : array-like of shape (n_samples, n_features)
    Test data of which we compute the likelihood, where n_samples is
    the number of samples and n_features is the number of features.
    X_test is assumed to be drawn from the same distribution than
    the data used in fit (including centering).

y : Ignored
    Not used, present for API consistence purpose.

Returns
-------
res : float
    The likelihood of the data set with `self.covariance_` as an
    estimator of its covariance matrix.
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


(** Attribute raw_location_: get value or raise Not_found if None.*)
val raw_location_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute raw_location_: get value as an option. *)
val raw_location_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute raw_covariance_: get value or raise Not_found if None.*)
val raw_covariance_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute raw_covariance_: get value as an option. *)
val raw_covariance_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute raw_support_: get value or raise Not_found if None.*)
val raw_support_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute raw_support_: get value as an option. *)
val raw_support_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute location_: get value or raise Not_found if None.*)
val location_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute location_: get value as an option. *)
val location_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute covariance_: get value or raise Not_found if None.*)
val covariance_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute covariance_: get value as an option. *)
val covariance_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute precision_: get value or raise Not_found if None.*)
val precision_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute precision_: get value as an option. *)
val precision_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute support_: get value or raise Not_found if None.*)
val support_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute support_: get value as an option. *)
val support_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute dist_: get value or raise Not_found if None.*)
val dist_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute dist_: get value as an option. *)
val dist_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module OAS : sig
type tag = [`OAS]
type t = [`BaseEstimator | `OAS | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_estimator : t -> [`BaseEstimator] Obj.t
val create : ?store_precision:bool -> ?assume_centered:bool -> unit -> t
(**
Oracle Approximating Shrinkage Estimator

Read more in the :ref:`User Guide <shrunk_covariance>`.

OAS is a particular form of shrinkage described in
'Shrinkage Algorithms for MMSE Covariance Estimation'
Chen et al., IEEE Trans. on Sign. Proc., Volume 58, Issue 10, October 2010.

The formula used here does not correspond to the one given in the
article. In the original article, formula (23) states that 2/p is
multiplied by Trace(cov*cov) in both the numerator and denominator, but
this operation is omitted because for a large p, the value of 2/p is
so small that it doesn't affect the value of the estimator.

Parameters
----------
store_precision : bool, default=True
    Specify if the estimated precision is stored.

assume_centered : bool, default=False
    If True, data will not be centered before computation.
    Useful when working with data whose mean is almost, but not exactly
    zero.
    If False (default), data will be centered before computation.

Attributes
----------
covariance_ : ndarray of shape (n_features, n_features)
    Estimated covariance matrix.

location_ : ndarray of shape (n_features,)
    Estimated location, i.e. the estimated mean.

precision_ : ndarray of shape (n_features, n_features)
    Estimated pseudo inverse matrix.
    (stored only if store_precision is True)

shrinkage_ : float
  coefficient in the convex combination used for the computation
  of the shrunk estimate. Range is [0, 1].

Examples
--------
>>> import numpy as np
>>> from sklearn.covariance import OAS
>>> from sklearn.datasets import make_gaussian_quantiles
>>> real_cov = np.array([[.8, .3],
...                      [.3, .4]])
>>> rng = np.random.RandomState(0)
>>> X = rng.multivariate_normal(mean=[0, 0],
...                             cov=real_cov,
...                             size=500)
>>> oas = OAS().fit(X)
>>> oas.covariance_
array([[0.7533..., 0.2763...],
       [0.2763..., 0.3964...]])
>>> oas.precision_
array([[ 1.7833..., -1.2431... ],
       [-1.2431...,  3.3889...]])
>>> oas.shrinkage_
0.0195...

Notes
-----
The regularised covariance is:

(1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

where mu = trace(cov) / n_features
and shrinkage is given by the OAS formula (see References)

References
----------
'Shrinkage Algorithms for MMSE Covariance Estimation'
Chen et al., IEEE Trans. on Sign. Proc., Volume 58, Issue 10, October 2010.
*)

val error_norm : ?norm:[`Frobenius | `Spectral] -> ?scaling:bool -> ?squared:bool -> comp_cov:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> float
(**
Computes the Mean Squared Error between two covariance estimators.
(In the sense of the Frobenius norm).

Parameters
----------
comp_cov : array-like of shape (n_features, n_features)
    The covariance to compare with.

norm : {'frobenius', 'spectral'}, default='frobenius'
    The type of norm used to compute the error. Available error types:
    - 'frobenius' (default): sqrt(tr(A^t.A))
    - 'spectral': sqrt(max(eigenvalues(A^t.A))
    where A is the error ``(comp_cov - self.covariance_)``.

scaling : bool, default=True
    If True (default), the squared error norm is divided by n_features.
    If False, the squared error norm is not rescaled.

squared : bool, default=True
    Whether to compute the squared error norm or the error norm.
    If True (default), the squared error norm is returned.
    If False, the error norm is returned.

Returns
-------
result : float
    The Mean Squared Error (in the sense of the Frobenius norm) between
    `self` and `comp_cov` covariance estimators.
*)

val fit : ?y:Py.Object.t -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fit the Oracle Approximating Shrinkage covariance model
according to the given training data and parameters.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training data, where `n_samples` is the number of samples
    and `n_features` is the number of features.
y : Ignored
    not used, present for API consistence purpose.

Returns
-------
self : object
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

val get_precision : [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Getter for the precision matrix.

Returns
-------
precision_ : array-like of shape (n_features, n_features)
    The precision matrix associated to the current covariance object.
*)

val mahalanobis : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Computes the squared Mahalanobis distances of given observations.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The observations, the Mahalanobis distances of the which we
    compute. Observations are assumed to be drawn from the same
    distribution than the data used in fit.

Returns
-------
dist : ndarray of shape (n_samples,)
    Squared Mahalanobis distances of the observations.
*)

val score : ?y:Py.Object.t -> x_test:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> float
(**
Computes the log-likelihood of a Gaussian data set with
`self.covariance_` as an estimator of its covariance matrix.

Parameters
----------
X_test : array-like of shape (n_samples, n_features)
    Test data of which we compute the likelihood, where n_samples is
    the number of samples and n_features is the number of features.
    X_test is assumed to be drawn from the same distribution than
    the data used in fit (including centering).

y : Ignored
    Not used, present for API consistence purpose.

Returns
-------
res : float
    The likelihood of the data set with `self.covariance_` as an
    estimator of its covariance matrix.
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


(** Attribute covariance_: get value or raise Not_found if None.*)
val covariance_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute covariance_: get value as an option. *)
val covariance_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute location_: get value or raise Not_found if None.*)
val location_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute location_: get value as an option. *)
val location_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute precision_: get value or raise Not_found if None.*)
val precision_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute precision_: get value as an option. *)
val precision_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute shrinkage_: get value or raise Not_found if None.*)
val shrinkage_ : t -> float

(** Attribute shrinkage_: get value as an option. *)
val shrinkage_opt : t -> (float) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module ShrunkCovariance : sig
type tag = [`ShrunkCovariance]
type t = [`BaseEstimator | `Object | `ShrunkCovariance] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_estimator : t -> [`BaseEstimator] Obj.t
val create : ?store_precision:bool -> ?assume_centered:bool -> ?shrinkage:float -> unit -> t
(**
Covariance estimator with shrinkage

Read more in the :ref:`User Guide <shrunk_covariance>`.

Parameters
----------
store_precision : bool, default=True
    Specify if the estimated precision is stored

assume_centered : bool, default=False
    If True, data will not be centered before computation.
    Useful when working with data whose mean is almost, but not exactly
    zero.
    If False, data will be centered before computation.

shrinkage : float, default=0.1
    Coefficient in the convex combination used for the computation
    of the shrunk estimate. Range is [0, 1].

Attributes
----------
covariance_ : ndarray of shape (n_features, n_features)
    Estimated covariance matrix

location_ : ndarray of shape (n_features,)
    Estimated location, i.e. the estimated mean.

precision_ : ndarray of shape (n_features, n_features)
    Estimated pseudo inverse matrix.
    (stored only if store_precision is True)

Examples
--------
>>> import numpy as np
>>> from sklearn.covariance import ShrunkCovariance
>>> from sklearn.datasets import make_gaussian_quantiles
>>> real_cov = np.array([[.8, .3],
...                      [.3, .4]])
>>> rng = np.random.RandomState(0)
>>> X = rng.multivariate_normal(mean=[0, 0],
...                                   cov=real_cov,
...                                   size=500)
>>> cov = ShrunkCovariance().fit(X)
>>> cov.covariance_
array([[0.7387..., 0.2536...],
       [0.2536..., 0.4110...]])
>>> cov.location_
array([0.0622..., 0.0193...])

Notes
-----
The regularized covariance is given by:

(1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

where mu = trace(cov) / n_features
*)

val error_norm : ?norm:[`Frobenius | `Spectral] -> ?scaling:bool -> ?squared:bool -> comp_cov:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> float
(**
Computes the Mean Squared Error between two covariance estimators.
(In the sense of the Frobenius norm).

Parameters
----------
comp_cov : array-like of shape (n_features, n_features)
    The covariance to compare with.

norm : {'frobenius', 'spectral'}, default='frobenius'
    The type of norm used to compute the error. Available error types:
    - 'frobenius' (default): sqrt(tr(A^t.A))
    - 'spectral': sqrt(max(eigenvalues(A^t.A))
    where A is the error ``(comp_cov - self.covariance_)``.

scaling : bool, default=True
    If True (default), the squared error norm is divided by n_features.
    If False, the squared error norm is not rescaled.

squared : bool, default=True
    Whether to compute the squared error norm or the error norm.
    If True (default), the squared error norm is returned.
    If False, the error norm is returned.

Returns
-------
result : float
    The Mean Squared Error (in the sense of the Frobenius norm) between
    `self` and `comp_cov` covariance estimators.
*)

val fit : ?y:Py.Object.t -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fit the shrunk covariance model according to the given training data
and parameters.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training data, where n_samples is the number of samples
    and n_features is the number of features.

y: Ignored
    not used, present for API consistence purpose.

Returns
-------
self : object
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

val get_precision : [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Getter for the precision matrix.

Returns
-------
precision_ : array-like of shape (n_features, n_features)
    The precision matrix associated to the current covariance object.
*)

val mahalanobis : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Computes the squared Mahalanobis distances of given observations.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The observations, the Mahalanobis distances of the which we
    compute. Observations are assumed to be drawn from the same
    distribution than the data used in fit.

Returns
-------
dist : ndarray of shape (n_samples,)
    Squared Mahalanobis distances of the observations.
*)

val score : ?y:Py.Object.t -> x_test:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> float
(**
Computes the log-likelihood of a Gaussian data set with
`self.covariance_` as an estimator of its covariance matrix.

Parameters
----------
X_test : array-like of shape (n_samples, n_features)
    Test data of which we compute the likelihood, where n_samples is
    the number of samples and n_features is the number of features.
    X_test is assumed to be drawn from the same distribution than
    the data used in fit (including centering).

y : Ignored
    Not used, present for API consistence purpose.

Returns
-------
res : float
    The likelihood of the data set with `self.covariance_` as an
    estimator of its covariance matrix.
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


(** Attribute covariance_: get value or raise Not_found if None.*)
val covariance_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute covariance_: get value as an option. *)
val covariance_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute location_: get value or raise Not_found if None.*)
val location_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute location_: get value as an option. *)
val location_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute precision_: get value or raise Not_found if None.*)
val precision_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute precision_: get value as an option. *)
val precision_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val empirical_covariance : ?assume_centered:bool -> x:[>`ArrayLike] Np.Obj.t -> unit -> [>`ArrayLike] Np.Obj.t
(**
Computes the Maximum likelihood covariance estimator


Parameters
----------
X : ndarray of shape (n_samples, n_features)
    Data from which to compute the covariance estimate

assume_centered : bool, default=False
    If True, data will not be centered before computation.
    Useful when working with data whose mean is almost, but not exactly
    zero.
    If False, data will be centered before computation.

Returns
-------
covariance : ndarray of shape (n_features, n_features)
    Empirical covariance (Maximum Likelihood Estimator).

Examples
--------
>>> from sklearn.covariance import empirical_covariance
>>> X = [[1,1,1],[1,1,1],[1,1,1],
...      [0,0,0],[0,0,0],[0,0,0]]
>>> empirical_covariance(X)
array([[0.25, 0.25, 0.25],
       [0.25, 0.25, 0.25],
       [0.25, 0.25, 0.25]])
*)

val fast_mcd : ?support_fraction:float -> ?cov_computation_method:Py.Object.t -> ?random_state:int -> x:[>`ArrayLike] Np.Obj.t -> unit -> ([>`ArrayLike] Np.Obj.t * [>`ArrayLike] Np.Obj.t * Py.Object.t)
(**
Estimates the Minimum Covariance Determinant matrix.

Read more in the :ref:`User Guide <robust_covariance>`.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The data matrix, with p features and n samples.

support_fraction : float, default=None
    The proportion of points to be included in the support of the raw
    MCD estimate. Default is `None`, which implies that the minimum
    value of `support_fraction` will be used within the algorithm:
    `(n_sample + n_features + 1) / 2`. This parameter must be in the
    range (0, 1).

cov_computation_method : callable,             default=:func:`sklearn.covariance.empirical_covariance`
    The function which will be used to compute the covariance.
    Must return an array of shape (n_features, n_features).

random_state : int or RandomState instance, default=None
    Determines the pseudo random number generator for shuffling the data.
    Pass an int for reproducible results across multiple function calls.
    See :term: `Glossary <random_state>`.

Returns
-------
location : ndarray of shape (n_features,)
    Robust location of the data.

covariance : ndarray of shape (n_features, n_features)
    Robust covariance of the features.

support : ndarray of shape (n_samples,), dtype=bool
    A mask of the observations that have been used to compute
    the robust location and covariance estimates of the data set.

Notes
-----
The FastMCD algorithm has been introduced by Rousseuw and Van Driessen
in 'A Fast Algorithm for the Minimum Covariance Determinant Estimator,
1999, American Statistical Association and the American Society
for Quality, TECHNOMETRICS'.
The principle is to compute robust estimates and random subsets before
pooling them into a larger subsets, and finally into the full data set.
Depending on the size of the initial sample, we have one, two or three
such computation levels.

Note that only raw estimates are returned. If one is interested in
the correction and reweighting steps described in [RouseeuwVan]_,
see the MinCovDet object.

References
----------

.. [RouseeuwVan] A Fast Algorithm for the Minimum Covariance
    Determinant Estimator, 1999, American Statistical Association
    and the American Society for Quality, TECHNOMETRICS

.. [Butler1993] R. W. Butler, P. L. Davies and M. Jhun,
    Asymptotics For The Minimum Covariance Determinant Estimator,
    The Annals of Statistics, 1993, Vol. 21, No. 3, 1385-1400
*)

val graphical_lasso : ?cov_init:[>`ArrayLike] Np.Obj.t -> ?mode:[`Cd | `Lars] -> ?tol:float -> ?enet_tol:float -> ?max_iter:int -> ?verbose:int -> ?return_costs:bool -> ?eps:float -> ?return_n_iter:bool -> emp_cov:[>`ArrayLike] Np.Obj.t -> alpha:float -> unit -> ([>`ArrayLike] Np.Obj.t * [>`ArrayLike] Np.Obj.t * Py.Object.t * int)
(**
l1-penalized covariance estimator

Read more in the :ref:`User Guide <sparse_inverse_covariance>`.

.. versionchanged:: v0.20
    graph_lasso has been renamed to graphical_lasso

Parameters
----------
emp_cov : ndarray of shape (n_features, n_features)
    Empirical covariance from which to compute the covariance estimate.

alpha : float
    The regularization parameter: the higher alpha, the more
    regularization, the sparser the inverse covariance.
    Range is (0, inf].

cov_init : array of shape (n_features, n_features), default=None
    The initial guess for the covariance.

mode : {'cd', 'lars'}, default='cd'
    The Lasso solver to use: coordinate descent or LARS. Use LARS for
    very sparse underlying graphs, where p > n. Elsewhere prefer cd
    which is more numerically stable.

tol : float, default=1e-4
    The tolerance to declare convergence: if the dual gap goes below
    this value, iterations are stopped. Range is (0, inf].

enet_tol : float, default=1e-4
    The tolerance for the elastic net solver used to calculate the descent
    direction. This parameter controls the accuracy of the search direction
    for a given column update, not of the overall parameter estimate. Only
    used for mode='cd'. Range is (0, inf].

max_iter : int, default=100
    The maximum number of iterations.

verbose : bool, default=False
    If verbose is True, the objective function and dual gap are
    printed at each iteration.

return_costs : bool, default=Flase
    If return_costs is True, the objective function and dual gap
    at each iteration are returned.

eps : float, default=eps
    The machine-precision regularization in the computation of the
    Cholesky diagonal factors. Increase this for very ill-conditioned
    systems. Default is `np.finfo(np.float64).eps`.

return_n_iter : bool, default=False
    Whether or not to return the number of iterations.

Returns
-------
covariance : ndarray of shape (n_features, n_features)
    The estimated covariance matrix.

precision : ndarray of shape (n_features, n_features)
    The estimated (sparse) precision matrix.

costs : list of (objective, dual_gap) pairs
    The list of values of the objective function and the dual gap at
    each iteration. Returned only if return_costs is True.

n_iter : int
    Number of iterations. Returned only if `return_n_iter` is set to True.

See Also
--------
GraphicalLasso, GraphicalLassoCV

Notes
-----
The algorithm employed to solve this problem is the GLasso algorithm,
from the Friedman 2008 Biostatistics paper. It is the same algorithm
as in the R `glasso` package.

One possible difference with the `glasso` R package is that the
diagonal coefficients are not penalized.
*)

val ledoit_wolf : ?assume_centered:bool -> ?block_size:int -> x:[>`ArrayLike] Np.Obj.t -> unit -> ([>`ArrayLike] Np.Obj.t * float)
(**
Estimates the shrunk Ledoit-Wolf covariance matrix.

Read more in the :ref:`User Guide <shrunk_covariance>`.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Data from which to compute the covariance estimate

assume_centered : bool, default=False
    If True, data will not be centered before computation.
    Useful to work with data whose mean is significantly equal to
    zero but is not exactly zero.
    If False, data will be centered before computation.

block_size : int, default=1000
    Size of the blocks into which the covariance matrix will be split.
    This is purely a memory optimization and does not affect results.

Returns
-------
shrunk_cov : ndarray of shape (n_features, n_features)
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

val ledoit_wolf_shrinkage : ?assume_centered:bool -> ?block_size:int -> x:[>`ArrayLike] Np.Obj.t -> unit -> float
(**
Estimates the shrunk Ledoit-Wolf covariance matrix.

Read more in the :ref:`User Guide <shrunk_covariance>`.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Data from which to compute the Ledoit-Wolf shrunk covariance shrinkage.

assume_centered : bool, default=False
    If True, data will not be centered before computation.
    Useful to work with data whose mean is significantly equal to
    zero but is not exactly zero.
    If False, data will be centered before computation.

block_size : int, default=1000
    Size of the blocks into which the covariance matrix will be split.

Returns
-------
shrinkage : float
    Coefficient in the convex combination used for the computation
    of the shrunk estimate.

Notes
-----
The regularized (shrunk) covariance is:

(1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

where mu = trace(cov) / n_features
*)

val log_likelihood : emp_cov:[>`ArrayLike] Np.Obj.t -> precision:[>`ArrayLike] Np.Obj.t -> unit -> float
(**
Computes the sample mean of the log_likelihood under a covariance model

computes the empirical expected log-likelihood (accounting for the
normalization terms and scaling), allowing for universal comparison (beyond
this software package)

Parameters
----------
emp_cov : ndarray of shape (n_features, n_features)
    Maximum Likelihood Estimator of covariance.

precision : ndarray of shape (n_features, n_features)
    The precision matrix of the covariance model to be tested.

Returns
-------
log_likelihood_ : float
    Sample mean of the log-likelihood.
*)

val oas : ?assume_centered:bool -> x:[>`ArrayLike] Np.Obj.t -> unit -> ([>`ArrayLike] Np.Obj.t * float)
(**
Estimate covariance with the Oracle Approximating Shrinkage algorithm.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Data from which to compute the covariance estimate.

assume_centered : bool, default=False
  If True, data will not be centered before computation.
  Useful to work with data whose mean is significantly equal to
  zero but is not exactly zero.
  If False, data will be centered before computation.

Returns
-------
shrunk_cov : array-like of shape (n_features, n_features)
    Shrunk covariance.

shrinkage : float
    Coefficient in the convex combination used for the computation
    of the shrunk estimate.

Notes
-----
The regularised (shrunk) covariance is:

(1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

where mu = trace(cov) / n_features

The formula we used to implement the OAS is slightly modified compared
to the one given in the article. See :class:`OAS` for more details.
*)

val shrunk_covariance : ?shrinkage:float -> emp_cov:[>`ArrayLike] Np.Obj.t -> unit -> [>`ArrayLike] Np.Obj.t
(**
Calculates a covariance matrix shrunk on the diagonal

Read more in the :ref:`User Guide <shrunk_covariance>`.

Parameters
----------
emp_cov : array-like of shape (n_features, n_features)
    Covariance matrix to be shrunk

shrinkage : float, default=0.1
    Coefficient in the convex combination used for the computation
    of the shrunk estimate. Range is [0, 1].

Returns
-------
shrunk_cov : ndarray of shape (n_features, n_features)
    Shrunk covariance.

Notes
-----
The regularized (shrunk) covariance is given by:

(1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

where mu = trace(cov) / n_features
*)

