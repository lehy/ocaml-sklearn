module EllipticEnvelope : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?store_precision:bool -> ?assume_centered:bool -> ?support_fraction:float -> ?contamination:float -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> unit -> t
(**
An object for detecting outliers in a Gaussian distributed dataset.

Read more in the :ref:`User Guide <outlier_detection>`.

Parameters
----------
store_precision : boolean, optional (default=True)
    Specify if the estimated precision is stored.

assume_centered : boolean, optional (default=False)
    If True, the support of robust location and covariance estimates
    is computed, and a covariance estimate is recomputed from it,
    without centering the data.
    Useful to work with data whose mean is significantly equal to
    zero but is not exactly zero.
    If False, the robust location and covariance are directly computed
    with the FastMCD algorithm without additional treatment.

support_fraction : float in (0., 1.), optional (default=None)
    The proportion of points to be included in the support of the raw
    MCD estimate. If None, the minimum value of support_fraction will
    be used within the algorithm: `[n_sample + n_features + 1] / 2`.

contamination : float in (0., 0.5), optional (default=0.1)
    The amount of contamination of the data set, i.e. the proportion
    of outliers in the data set.

random_state : int, RandomState instance or None, optional (default=None)
    The seed of the pseudo random number generator to use when shuffling
    the data.  If int, random_state is the seed used by the random number
    generator; If RandomState instance, random_state is the random number
    generator; If None, the random number generator is the RandomState
    instance used by `np.random`.

Attributes
----------
location_ : array-like, shape (n_features,)
    Estimated robust location

covariance_ : array-like, shape (n_features, n_features)
    Estimated robust covariance matrix

precision_ : array-like, shape (n_features, n_features)
    Estimated pseudo inverse matrix.
    (stored only if store_precision is True)

support_ : array-like, shape (n_samples,)
    A mask of the observations that have been used to compute the
    robust estimates of location and shape.

offset_ : float
    Offset used to define the decision function from the raw scores.
    We have the relation: ``decision_function = score_samples - offset_``.
    The offset depends on the contamination parameter and is defined in
    such a way we obtain the expected number of outliers (samples with
    decision function < 0) in training.

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
.. [1] Rousseeuw, P.J., Van Driessen, K. "A fast algorithm for the
   minimum covariance determinant estimator" Technometrics 41(3), 212
   (1999)
*)

val correct_covariance : data:Ndarray.t -> t -> Ndarray.t
(**
Apply a correction to raw Minimum Covariance Determinant estimates.

Correction using the empirical correction factor suggested
by Rousseeuw and Van Driessen in [RVD]_.

Parameters
----------
data : array-like, shape (n_samples, n_features)
    The data matrix, with p features and n samples.
    The data set must be the one which was used to compute
    the raw estimates.

References
----------

.. [RVD] A Fast Algorithm for the Minimum Covariance
    Determinant Estimator, 1999, American Statistical Association
    and the American Society for Quality, TECHNOMETRICS

Returns
-------
covariance_corrected : array-like, shape (n_features, n_features)
    Corrected robust covariance estimate.
*)

val decision_function : x:Ndarray.t -> t -> Ndarray.t
(**
Compute the decision function of the given observations.

Parameters
----------
X : array-like, shape (n_samples, n_features)

Returns
-------

decision : array-like, shape (n_samples, )
    Decision function of the samples.
    It is equal to the shifted Mahalanobis distances.
    The threshold for being an outlier is 0, which ensures a
    compatibility with other outlier detection algorithms.
*)

val error_norm : ?norm:string -> ?scaling:bool -> ?squared:bool -> comp_cov:Ndarray.t -> t -> Py.Object.t
(**
Computes the Mean Squared Error between two covariance estimators.
(In the sense of the Frobenius norm).

Parameters
----------
comp_cov : array-like of shape (n_features, n_features)
    The covariance to compare with.

norm : str
    The type of norm used to compute the error. Available error types:
    - 'frobenius' (default): sqrt(tr(A^t.A))
    - 'spectral': sqrt(max(eigenvalues(A^t.A))
    where A is the error ``(comp_cov - self.covariance_)``.

scaling : bool
    If True (default), the squared error norm is divided by n_features.
    If False, the squared error norm is not rescaled.

squared : bool
    Whether to compute the squared error norm or the error norm.
    If True (default), the squared error norm is returned.
    If False, the error norm is returned.

Returns
-------
The Mean Squared Error (in the sense of the Frobenius norm) between
`self` and `comp_cov` covariance estimators.
*)

val fit : ?y:Py.Object.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> t
(**
Fit the EllipticEnvelope model.

Parameters
----------
X : numpy array or sparse matrix, shape (n_samples, n_features).
    Training data

y : Ignored
    not used, present for API consistency by convention.
*)

val fit_predict : ?y:Py.Object.t -> x:Ndarray.t -> t -> Ndarray.t
(**
Perform fit on X and returns labels for X.

Returns -1 for outliers and 1 for inliers.

Parameters
----------
X : ndarray, shape (n_samples, n_features)
    Input data.

y : Ignored
    Not used, present for API consistency by convention.

Returns
-------
y : ndarray, shape (n_samples,)
    1 for inliers, -1 for outliers.
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

val get_precision : t -> Ndarray.t
(**
Getter for the precision matrix.

Returns
-------
precision_ : array-like
    The precision matrix associated to the current covariance object.
*)

val mahalanobis : x:Ndarray.t -> t -> Ndarray.t
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
dist : array, shape = [n_samples,]
    Squared Mahalanobis distances of the observations.
*)

val predict : x:Ndarray.t -> t -> Ndarray.t
(**
Predict the labels (1 inlier, -1 outlier) of X according to the
fitted model.

Parameters
----------
X : array-like, shape (n_samples, n_features)

Returns
-------
is_inlier : array, shape (n_samples,)
    Returns -1 for anomalies/outliers and +1 for inliers.
*)

val reweight_covariance : data:Ndarray.t -> t -> (Ndarray.t * Ndarray.t * Py.Object.t)
(**
Re-weight raw Minimum Covariance Determinant estimates.

Re-weight observations using Rousseeuw's method (equivalent to
deleting outlying observations from the data set before
computing location and covariance estimates) described
in [RVDriessen]_.

Parameters
----------
data : array-like, shape (n_samples, n_features)
    The data matrix, with p features and n samples.
    The data set must be the one which was used to compute
    the raw estimates.

References
----------

.. [RVDriessen] A Fast Algorithm for the Minimum Covariance
    Determinant Estimator, 1999, American Statistical Association
    and the American Society for Quality, TECHNOMETRICS

Returns
-------
location_reweighted : array-like, shape (n_features, )
    Re-weighted robust location estimate.

covariance_reweighted : array-like, shape (n_features, n_features)
    Re-weighted robust covariance estimate.

support_reweighted : array-like, type boolean, shape (n_samples,)
    A mask of the observations that have been used to compute
    the re-weighted robust location and covariance estimates.
*)

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
(**
Returns the mean accuracy on the given test data and labels.

In multi-label classification, this is the subset accuracy
which is a harsh metric since you require for each sample that
each label set be correctly predicted.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Test samples.

y : array-like, shape (n_samples,) or (n_samples, n_outputs)
    True labels for X.

sample_weight : array-like, shape (n_samples,), optional
    Sample weights.

Returns
-------
score : float
    Mean accuracy of self.predict(X) wrt. y.
*)

val score_samples : x:Ndarray.t -> t -> Ndarray.t
(**
Compute the negative Mahalanobis distances.

Parameters
----------
X : array-like, shape (n_samples, n_features)

Returns
-------
negative_mahal_distances : array-like, shape (n_samples, )
    Opposite of the Mahalanobis distances.
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


(** Attribute location_: see constructor for documentation *)
val location_ : t -> Ndarray.t

(** Attribute covariance_: see constructor for documentation *)
val covariance_ : t -> Ndarray.t

(** Attribute precision_: see constructor for documentation *)
val precision_ : t -> Ndarray.t

(** Attribute support_: see constructor for documentation *)
val support_ : t -> Ndarray.t

(** Attribute offset_: see constructor for documentation *)
val offset_ : t -> float

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module EmpiricalCovariance : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?store_precision:bool -> ?assume_centered:bool -> unit -> t
(**
Maximum likelihood covariance estimator

Read more in the :ref:`User Guide <covariance>`.

Parameters
----------
store_precision : bool
    Specifies if the estimated precision is stored.

assume_centered : bool
    If True, data are not centered before computation.
    Useful when working with data whose mean is almost, but not exactly
    zero.
    If False (default), data are centered before computation.

Attributes
----------
location_ : array-like, shape (n_features,)
    Estimated location, i.e. the estimated mean.

covariance_ : 2D ndarray, shape (n_features, n_features)
    Estimated covariance matrix

precision_ : 2D ndarray, shape (n_features, n_features)
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

val error_norm : ?norm:string -> ?scaling:bool -> ?squared:bool -> comp_cov:Ndarray.t -> t -> Py.Object.t
(**
Computes the Mean Squared Error between two covariance estimators.
(In the sense of the Frobenius norm).

Parameters
----------
comp_cov : array-like of shape (n_features, n_features)
    The covariance to compare with.

norm : str
    The type of norm used to compute the error. Available error types:
    - 'frobenius' (default): sqrt(tr(A^t.A))
    - 'spectral': sqrt(max(eigenvalues(A^t.A))
    where A is the error ``(comp_cov - self.covariance_)``.

scaling : bool
    If True (default), the squared error norm is divided by n_features.
    If False, the squared error norm is not rescaled.

squared : bool
    Whether to compute the squared error norm or the error norm.
    If True (default), the squared error norm is returned.
    If False, the error norm is returned.

Returns
-------
The Mean Squared Error (in the sense of the Frobenius norm) between
`self` and `comp_cov` covariance estimators.
*)

val fit : ?y:Py.Object.t -> x:Ndarray.t -> t -> t
(**
Fits the Maximum Likelihood Estimator covariance model
according to the given training data and parameters.

Parameters
----------
X : array-like of shape (n_samples, n_features)
  Training data, where n_samples is the number of samples and
  n_features is the number of features.

y
    not used, present for API consistence purpose.

Returns
-------
self : object
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

val get_precision : t -> Ndarray.t
(**
Getter for the precision matrix.

Returns
-------
precision_ : array-like
    The precision matrix associated to the current covariance object.
*)

val mahalanobis : x:Ndarray.t -> t -> Ndarray.t
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
dist : array, shape = [n_samples,]
    Squared Mahalanobis distances of the observations.
*)

val score : ?y:Py.Object.t -> x_test:Ndarray.t -> t -> float
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

y
    not used, present for API consistence purpose.

Returns
-------
res : float
    The likelihood of the data set with `self.covariance_` as an
    estimator of its covariance matrix.
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


(** Attribute location_: see constructor for documentation *)
val location_ : t -> Ndarray.t

(** Attribute covariance_: see constructor for documentation *)
val covariance_ : t -> Py.Object.t

(** Attribute precision_: see constructor for documentation *)
val precision_ : t -> Py.Object.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module GraphicalLasso : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?alpha:float -> ?mode:[`Cd | `Lars] -> ?tol:float -> ?enet_tol:float -> ?max_iter:int -> ?verbose:bool -> ?assume_centered:bool -> unit -> t
(**
Sparse inverse covariance estimation with an l1-penalized estimator.

Read more in the :ref:`User Guide <sparse_inverse_covariance>`.

Parameters
----------
alpha : positive float, default 0.01
    The regularization parameter: the higher alpha, the more
    regularization, the sparser the inverse covariance.

mode : {'cd', 'lars'}, default 'cd'
    The Lasso solver to use: coordinate descent or LARS. Use LARS for
    very sparse underlying graphs, where p > n. Elsewhere prefer cd
    which is more numerically stable.

tol : positive float, default 1e-4
    The tolerance to declare convergence: if the dual gap goes below
    this value, iterations are stopped.

enet_tol : positive float, optional
    The tolerance for the elastic net solver used to calculate the descent
    direction. This parameter controls the accuracy of the search direction
    for a given column update, not of the overall parameter estimate. Only
    used for mode='cd'.

max_iter : integer, default 100
    The maximum number of iterations.

verbose : boolean, default False
    If verbose is True, the objective function and dual gap are
    plotted at each iteration.

assume_centered : boolean, default False
    If True, data are not centered before computation.
    Useful when working with data whose mean is almost, but not exactly
    zero.
    If False, data are centered before computation.

Attributes
----------
location_ : array-like, shape (n_features,)
    Estimated location, i.e. the estimated mean.

covariance_ : array-like, shape (n_features, n_features)
    Estimated covariance matrix

precision_ : array-like, shape (n_features, n_features)
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

val error_norm : ?norm:string -> ?scaling:bool -> ?squared:bool -> comp_cov:Ndarray.t -> t -> Py.Object.t
(**
Computes the Mean Squared Error between two covariance estimators.
(In the sense of the Frobenius norm).

Parameters
----------
comp_cov : array-like of shape (n_features, n_features)
    The covariance to compare with.

norm : str
    The type of norm used to compute the error. Available error types:
    - 'frobenius' (default): sqrt(tr(A^t.A))
    - 'spectral': sqrt(max(eigenvalues(A^t.A))
    where A is the error ``(comp_cov - self.covariance_)``.

scaling : bool
    If True (default), the squared error norm is divided by n_features.
    If False, the squared error norm is not rescaled.

squared : bool
    Whether to compute the squared error norm or the error norm.
    If True (default), the squared error norm is returned.
    If False, the error norm is returned.

Returns
-------
The Mean Squared Error (in the sense of the Frobenius norm) between
`self` and `comp_cov` covariance estimators.
*)

val fit : ?y:Py.Object.t -> x:Ndarray.t -> t -> t
(**
Fits the GraphicalLasso model to X.

Parameters
----------
X : ndarray, shape (n_samples, n_features)
    Data from which to compute the covariance estimate
y : (ignored)
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

val get_precision : t -> Ndarray.t
(**
Getter for the precision matrix.

Returns
-------
precision_ : array-like
    The precision matrix associated to the current covariance object.
*)

val mahalanobis : x:Ndarray.t -> t -> Ndarray.t
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
dist : array, shape = [n_samples,]
    Squared Mahalanobis distances of the observations.
*)

val score : ?y:Py.Object.t -> x_test:Ndarray.t -> t -> float
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

y
    not used, present for API consistence purpose.

Returns
-------
res : float
    The likelihood of the data set with `self.covariance_` as an
    estimator of its covariance matrix.
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


(** Attribute location_: see constructor for documentation *)
val location_ : t -> Ndarray.t

(** Attribute covariance_: see constructor for documentation *)
val covariance_ : t -> Ndarray.t

(** Attribute precision_: see constructor for documentation *)
val precision_ : t -> Ndarray.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> int

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module GraphicalLassoCV : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?alphas:[`Int of int | `PyObject of Py.Object.t] -> ?n_refinements:Py.Object.t -> ?cv:[`Int of int | `CrossValGenerator of Py.Object.t | `Ndarray of Ndarray.t] -> ?tol:float -> ?enet_tol:float -> ?max_iter:int -> ?mode:[`Cd | `Lars] -> ?n_jobs:[`Int of int | `None] -> ?verbose:bool -> ?assume_centered:bool -> unit -> t
(**
Sparse inverse covariance w/ cross-validated choice of the l1 penalty.

See glossary entry for :term:`cross-validation estimator`.

Read more in the :ref:`User Guide <sparse_inverse_covariance>`.

Parameters
----------
alphas : integer, or list positive float, optional
    If an integer is given, it fixes the number of points on the
    grids of alpha to be used. If a list is given, it gives the
    grid to be used. See the notes in the class docstring for
    more details.

n_refinements : strictly positive integer
    The number of times the grid is refined. Not used if explicit
    values of alphas are passed.

cv : int, cross-validation generator or an iterable, optional
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

tol : positive float, optional
    The tolerance to declare convergence: if the dual gap goes below
    this value, iterations are stopped.

enet_tol : positive float, optional
    The tolerance for the elastic net solver used to calculate the descent
    direction. This parameter controls the accuracy of the search direction
    for a given column update, not of the overall parameter estimate. Only
    used for mode='cd'.

max_iter : integer, optional
    Maximum number of iterations.

mode : {'cd', 'lars'}
    The Lasso solver to use: coordinate descent or LARS. Use LARS for
    very sparse underlying graphs, where number of features is greater
    than number of samples. Elsewhere prefer cd which is more numerically
    stable.

n_jobs : int or None, optional (default=None)
    number of jobs to run in parallel.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

verbose : boolean, optional
    If verbose is True, the objective function and duality gap are
    printed at each iteration.

assume_centered : boolean
    If True, data are not centered before computation.
    Useful when working with data whose mean is almost, but not exactly
    zero.
    If False, data are centered before computation.

Attributes
----------
location_ : array-like, shape (n_features,)
    Estimated location, i.e. the estimated mean.

covariance_ : numpy.ndarray, shape (n_features, n_features)
    Estimated covariance matrix.

precision_ : numpy.ndarray, shape (n_features, n_features)
    Estimated precision matrix (inverse covariance).

alpha_ : float
    Penalization parameter selected.

cv_alphas_ : list of float
    All penalization parameters explored.

grid_scores_ : 2D numpy.ndarray (n_alphas, n_folds)
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

val error_norm : ?norm:string -> ?scaling:bool -> ?squared:bool -> comp_cov:Ndarray.t -> t -> Py.Object.t
(**
Computes the Mean Squared Error between two covariance estimators.
(In the sense of the Frobenius norm).

Parameters
----------
comp_cov : array-like of shape (n_features, n_features)
    The covariance to compare with.

norm : str
    The type of norm used to compute the error. Available error types:
    - 'frobenius' (default): sqrt(tr(A^t.A))
    - 'spectral': sqrt(max(eigenvalues(A^t.A))
    where A is the error ``(comp_cov - self.covariance_)``.

scaling : bool
    If True (default), the squared error norm is divided by n_features.
    If False, the squared error norm is not rescaled.

squared : bool
    Whether to compute the squared error norm or the error norm.
    If True (default), the squared error norm is returned.
    If False, the error norm is returned.

Returns
-------
The Mean Squared Error (in the sense of the Frobenius norm) between
`self` and `comp_cov` covariance estimators.
*)

val fit : ?y:Py.Object.t -> x:Ndarray.t -> t -> t
(**
Fits the GraphicalLasso covariance model to X.

Parameters
----------
X : ndarray, shape (n_samples, n_features)
    Data from which to compute the covariance estimate
y : (ignored)
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

val get_precision : t -> Ndarray.t
(**
Getter for the precision matrix.

Returns
-------
precision_ : array-like
    The precision matrix associated to the current covariance object.
*)

val mahalanobis : x:Ndarray.t -> t -> Ndarray.t
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
dist : array, shape = [n_samples,]
    Squared Mahalanobis distances of the observations.
*)

val score : ?y:Py.Object.t -> x_test:Ndarray.t -> t -> float
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

y
    not used, present for API consistence purpose.

Returns
-------
res : float
    The likelihood of the data set with `self.covariance_` as an
    estimator of its covariance matrix.
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


(** Attribute location_: see constructor for documentation *)
val location_ : t -> Ndarray.t

(** Attribute covariance_: see constructor for documentation *)
val covariance_ : t -> Py.Object.t

(** Attribute precision_: see constructor for documentation *)
val precision_ : t -> Py.Object.t

(** Attribute alpha_: see constructor for documentation *)
val alpha_ : t -> float

(** Attribute cv_alphas_: see constructor for documentation *)
val cv_alphas_ : t -> Py.Object.t

(** Attribute grid_scores_: see constructor for documentation *)
val grid_scores_ : t -> Py.Object.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> int

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module LedoitWolf : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?store_precision:bool -> ?assume_centered:bool -> ?block_size:int -> unit -> t
(**
LedoitWolf Estimator

Ledoit-Wolf is a particular form of shrinkage, where the shrinkage
coefficient is computed using O. Ledoit and M. Wolf's formula as
described in "A Well-Conditioned Estimator for Large-Dimensional
Covariance Matrices", Ledoit and Wolf, Journal of Multivariate
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
location_ : array-like, shape (n_features,)
    Estimated location, i.e. the estimated mean.

covariance_ : array-like, shape (n_features, n_features)
    Estimated covariance matrix

precision_ : array-like, shape (n_features, n_features)
    Estimated pseudo inverse matrix.
    (stored only if store_precision is True)

shrinkage_ : float, 0 <= shrinkage <= 1
    Coefficient in the convex combination used for the computation
    of the shrunk estimate.

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
"A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices",
Ledoit and Wolf, Journal of Multivariate Analysis, Volume 88, Issue 2,
February 2004, pages 365-411.
*)

val error_norm : ?norm:string -> ?scaling:bool -> ?squared:bool -> comp_cov:Ndarray.t -> t -> Py.Object.t
(**
Computes the Mean Squared Error between two covariance estimators.
(In the sense of the Frobenius norm).

Parameters
----------
comp_cov : array-like of shape (n_features, n_features)
    The covariance to compare with.

norm : str
    The type of norm used to compute the error. Available error types:
    - 'frobenius' (default): sqrt(tr(A^t.A))
    - 'spectral': sqrt(max(eigenvalues(A^t.A))
    where A is the error ``(comp_cov - self.covariance_)``.

scaling : bool
    If True (default), the squared error norm is divided by n_features.
    If False, the squared error norm is not rescaled.

squared : bool
    Whether to compute the squared error norm or the error norm.
    If True (default), the squared error norm is returned.
    If False, the error norm is returned.

Returns
-------
The Mean Squared Error (in the sense of the Frobenius norm) between
`self` and `comp_cov` covariance estimators.
*)

val fit : ?y:Py.Object.t -> x:Ndarray.t -> t -> t
(**
Fits the Ledoit-Wolf shrunk covariance model
according to the given training data and parameters.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training data, where n_samples is the number of samples
    and n_features is the number of features.
y
    not used, present for API consistence purpose.

Returns
-------
self : object
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

val get_precision : t -> Ndarray.t
(**
Getter for the precision matrix.

Returns
-------
precision_ : array-like
    The precision matrix associated to the current covariance object.
*)

val mahalanobis : x:Ndarray.t -> t -> Ndarray.t
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
dist : array, shape = [n_samples,]
    Squared Mahalanobis distances of the observations.
*)

val score : ?y:Py.Object.t -> x_test:Ndarray.t -> t -> float
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

y
    not used, present for API consistence purpose.

Returns
-------
res : float
    The likelihood of the data set with `self.covariance_` as an
    estimator of its covariance matrix.
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


(** Attribute location_: see constructor for documentation *)
val location_ : t -> Ndarray.t

(** Attribute covariance_: see constructor for documentation *)
val covariance_ : t -> Ndarray.t

(** Attribute precision_: see constructor for documentation *)
val precision_ : t -> Ndarray.t

(** Attribute shrinkage_: see constructor for documentation *)
val shrinkage_ : t -> Py.Object.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MinCovDet : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?store_precision:bool -> ?assume_centered:bool -> ?support_fraction:[`Float of float | `PyObject of Py.Object.t] -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> unit -> t
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
store_precision : bool
    Specify if the estimated precision is stored.

assume_centered : bool
    If True, the support of the robust location and the covariance
    estimates is computed, and a covariance estimate is recomputed from
    it, without centering the data.
    Useful to work with data whose mean is significantly equal to
    zero but is not exactly zero.
    If False, the robust location and covariance are directly computed
    with the FastMCD algorithm without additional treatment.

support_fraction : float, 0 < support_fraction < 1
    The proportion of points to be included in the support of the raw
    MCD estimate. Default is None, which implies that the minimum
    value of support_fraction will be used within the algorithm:
    [n_sample + n_features + 1] / 2

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

Attributes
----------
raw_location_ : array-like, shape (n_features,)
    The raw robust estimated location before correction and re-weighting.

raw_covariance_ : array-like, shape (n_features, n_features)
    The raw robust estimated covariance before correction and re-weighting.

raw_support_ : array-like, shape (n_samples,)
    A mask of the observations that have been used to compute
    the raw robust estimates of location and shape, before correction
    and re-weighting.

location_ : array-like, shape (n_features,)
    Estimated robust location

covariance_ : array-like, shape (n_features, n_features)
    Estimated robust covariance matrix

precision_ : array-like, shape (n_features, n_features)
    Estimated pseudo inverse matrix.
    (stored only if store_precision is True)

support_ : array-like, shape (n_samples,)
    A mask of the observations that have been used to compute
    the robust estimates of location and shape.

dist_ : array-like, shape (n_samples,)
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

val correct_covariance : data:Ndarray.t -> t -> Ndarray.t
(**
Apply a correction to raw Minimum Covariance Determinant estimates.

Correction using the empirical correction factor suggested
by Rousseeuw and Van Driessen in [RVD]_.

Parameters
----------
data : array-like, shape (n_samples, n_features)
    The data matrix, with p features and n samples.
    The data set must be the one which was used to compute
    the raw estimates.

References
----------

.. [RVD] A Fast Algorithm for the Minimum Covariance
    Determinant Estimator, 1999, American Statistical Association
    and the American Society for Quality, TECHNOMETRICS

Returns
-------
covariance_corrected : array-like, shape (n_features, n_features)
    Corrected robust covariance estimate.
*)

val error_norm : ?norm:string -> ?scaling:bool -> ?squared:bool -> comp_cov:Ndarray.t -> t -> Py.Object.t
(**
Computes the Mean Squared Error between two covariance estimators.
(In the sense of the Frobenius norm).

Parameters
----------
comp_cov : array-like of shape (n_features, n_features)
    The covariance to compare with.

norm : str
    The type of norm used to compute the error. Available error types:
    - 'frobenius' (default): sqrt(tr(A^t.A))
    - 'spectral': sqrt(max(eigenvalues(A^t.A))
    where A is the error ``(comp_cov - self.covariance_)``.

scaling : bool
    If True (default), the squared error norm is divided by n_features.
    If False, the squared error norm is not rescaled.

squared : bool
    Whether to compute the squared error norm or the error norm.
    If True (default), the squared error norm is returned.
    If False, the error norm is returned.

Returns
-------
The Mean Squared Error (in the sense of the Frobenius norm) between
`self` and `comp_cov` covariance estimators.
*)

val fit : ?y:Py.Object.t -> x:Ndarray.t -> t -> t
(**
Fits a Minimum Covariance Determinant with the FastMCD algorithm.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training data, where n_samples is the number of samples
    and n_features is the number of features.

y
    not used, present for API consistence purpose.

Returns
-------
self : object
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

val get_precision : t -> Ndarray.t
(**
Getter for the precision matrix.

Returns
-------
precision_ : array-like
    The precision matrix associated to the current covariance object.
*)

val mahalanobis : x:Ndarray.t -> t -> Ndarray.t
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
dist : array, shape = [n_samples,]
    Squared Mahalanobis distances of the observations.
*)

val reweight_covariance : data:Ndarray.t -> t -> (Ndarray.t * Ndarray.t * Py.Object.t)
(**
Re-weight raw Minimum Covariance Determinant estimates.

Re-weight observations using Rousseeuw's method (equivalent to
deleting outlying observations from the data set before
computing location and covariance estimates) described
in [RVDriessen]_.

Parameters
----------
data : array-like, shape (n_samples, n_features)
    The data matrix, with p features and n samples.
    The data set must be the one which was used to compute
    the raw estimates.

References
----------

.. [RVDriessen] A Fast Algorithm for the Minimum Covariance
    Determinant Estimator, 1999, American Statistical Association
    and the American Society for Quality, TECHNOMETRICS

Returns
-------
location_reweighted : array-like, shape (n_features, )
    Re-weighted robust location estimate.

covariance_reweighted : array-like, shape (n_features, n_features)
    Re-weighted robust covariance estimate.

support_reweighted : array-like, type boolean, shape (n_samples,)
    A mask of the observations that have been used to compute
    the re-weighted robust location and covariance estimates.
*)

val score : ?y:Py.Object.t -> x_test:Ndarray.t -> t -> float
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

y
    not used, present for API consistence purpose.

Returns
-------
res : float
    The likelihood of the data set with `self.covariance_` as an
    estimator of its covariance matrix.
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


(** Attribute raw_location_: see constructor for documentation *)
val raw_location_ : t -> Ndarray.t

(** Attribute raw_covariance_: see constructor for documentation *)
val raw_covariance_ : t -> Ndarray.t

(** Attribute raw_support_: see constructor for documentation *)
val raw_support_ : t -> Ndarray.t

(** Attribute location_: see constructor for documentation *)
val location_ : t -> Ndarray.t

(** Attribute covariance_: see constructor for documentation *)
val covariance_ : t -> Ndarray.t

(** Attribute precision_: see constructor for documentation *)
val precision_ : t -> Ndarray.t

(** Attribute support_: see constructor for documentation *)
val support_ : t -> Ndarray.t

(** Attribute dist_: see constructor for documentation *)
val dist_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module OAS : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?store_precision:bool -> ?assume_centered:bool -> unit -> t
(**
Oracle Approximating Shrinkage Estimator

Read more in the :ref:`User Guide <shrunk_covariance>`.

OAS is a particular form of shrinkage described in
"Shrinkage Algorithms for MMSE Covariance Estimation"
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
covariance_ : array-like, shape (n_features, n_features)
    Estimated covariance matrix.

precision_ : array-like, shape (n_features, n_features)
    Estimated pseudo inverse matrix.
    (stored only if store_precision is True)

shrinkage_ : float, 0 <= shrinkage <= 1
  coefficient in the convex combination used for the computation
  of the shrunk estimate.

Notes
-----
The regularised covariance is:

(1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

where mu = trace(cov) / n_features
and shrinkage is given by the OAS formula (see References)

References
----------
"Shrinkage Algorithms for MMSE Covariance Estimation"
Chen et al., IEEE Trans. on Sign. Proc., Volume 58, Issue 10, October 2010.
*)

val error_norm : ?norm:string -> ?scaling:bool -> ?squared:bool -> comp_cov:Ndarray.t -> t -> Py.Object.t
(**
Computes the Mean Squared Error between two covariance estimators.
(In the sense of the Frobenius norm).

Parameters
----------
comp_cov : array-like of shape (n_features, n_features)
    The covariance to compare with.

norm : str
    The type of norm used to compute the error. Available error types:
    - 'frobenius' (default): sqrt(tr(A^t.A))
    - 'spectral': sqrt(max(eigenvalues(A^t.A))
    where A is the error ``(comp_cov - self.covariance_)``.

scaling : bool
    If True (default), the squared error norm is divided by n_features.
    If False, the squared error norm is not rescaled.

squared : bool
    Whether to compute the squared error norm or the error norm.
    If True (default), the squared error norm is returned.
    If False, the error norm is returned.

Returns
-------
The Mean Squared Error (in the sense of the Frobenius norm) between
`self` and `comp_cov` covariance estimators.
*)

val fit : ?y:Py.Object.t -> x:Ndarray.t -> t -> t
(**
Fits the Oracle Approximating Shrinkage covariance model
according to the given training data and parameters.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training data, where n_samples is the number of samples
    and n_features is the number of features.
y
    not used, present for API consistence purpose.

Returns
-------
self : object
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

val get_precision : t -> Ndarray.t
(**
Getter for the precision matrix.

Returns
-------
precision_ : array-like
    The precision matrix associated to the current covariance object.
*)

val mahalanobis : x:Ndarray.t -> t -> Ndarray.t
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
dist : array, shape = [n_samples,]
    Squared Mahalanobis distances of the observations.
*)

val score : ?y:Py.Object.t -> x_test:Ndarray.t -> t -> float
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

y
    not used, present for API consistence purpose.

Returns
-------
res : float
    The likelihood of the data set with `self.covariance_` as an
    estimator of its covariance matrix.
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


(** Attribute covariance_: see constructor for documentation *)
val covariance_ : t -> Ndarray.t

(** Attribute precision_: see constructor for documentation *)
val precision_ : t -> Ndarray.t

(** Attribute shrinkage_: see constructor for documentation *)
val shrinkage_ : t -> Py.Object.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module ShrunkCovariance : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?store_precision:bool -> ?assume_centered:bool -> ?shrinkage:[`Float of float | `PyObject of Py.Object.t] -> unit -> t
(**
Covariance estimator with shrinkage

Read more in the :ref:`User Guide <shrunk_covariance>`.

Parameters
----------
store_precision : boolean, default True
    Specify if the estimated precision is stored

assume_centered : boolean, default False
    If True, data will not be centered before computation.
    Useful when working with data whose mean is almost, but not exactly
    zero.
    If False, data will be centered before computation.

shrinkage : float, 0 <= shrinkage <= 1, default 0.1
    Coefficient in the convex combination used for the computation
    of the shrunk estimate.

Attributes
----------
location_ : array-like, shape (n_features,)
    Estimated location, i.e. the estimated mean.

covariance_ : array-like, shape (n_features, n_features)
    Estimated covariance matrix

precision_ : array-like, shape (n_features, n_features)
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

val error_norm : ?norm:string -> ?scaling:bool -> ?squared:bool -> comp_cov:Ndarray.t -> t -> Py.Object.t
(**
Computes the Mean Squared Error between two covariance estimators.
(In the sense of the Frobenius norm).

Parameters
----------
comp_cov : array-like of shape (n_features, n_features)
    The covariance to compare with.

norm : str
    The type of norm used to compute the error. Available error types:
    - 'frobenius' (default): sqrt(tr(A^t.A))
    - 'spectral': sqrt(max(eigenvalues(A^t.A))
    where A is the error ``(comp_cov - self.covariance_)``.

scaling : bool
    If True (default), the squared error norm is divided by n_features.
    If False, the squared error norm is not rescaled.

squared : bool
    Whether to compute the squared error norm or the error norm.
    If True (default), the squared error norm is returned.
    If False, the error norm is returned.

Returns
-------
The Mean Squared Error (in the sense of the Frobenius norm) between
`self` and `comp_cov` covariance estimators.
*)

val fit : ?y:Py.Object.t -> x:Ndarray.t -> t -> t
(**
Fits the shrunk covariance model
according to the given training data and parameters.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training data, where n_samples is the number of samples
    and n_features is the number of features.

y
    not used, present for API consistence purpose.

Returns
-------
self : object
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

val get_precision : t -> Ndarray.t
(**
Getter for the precision matrix.

Returns
-------
precision_ : array-like
    The precision matrix associated to the current covariance object.
*)

val mahalanobis : x:Ndarray.t -> t -> Ndarray.t
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
dist : array, shape = [n_samples,]
    Squared Mahalanobis distances of the observations.
*)

val score : ?y:Py.Object.t -> x_test:Ndarray.t -> t -> float
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

y
    not used, present for API consistence purpose.

Returns
-------
res : float
    The likelihood of the data set with `self.covariance_` as an
    estimator of its covariance matrix.
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


(** Attribute location_: see constructor for documentation *)
val location_ : t -> Ndarray.t

(** Attribute covariance_: see constructor for documentation *)
val covariance_ : t -> Ndarray.t

(** Attribute precision_: see constructor for documentation *)
val precision_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val empirical_covariance : ?assume_centered:bool -> x:Ndarray.t -> unit -> Py.Object.t
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

val fast_mcd : ?support_fraction:[`Float of float | `PyObject of Py.Object.t] -> ?cov_computation_method:Py.Object.t -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> x:Ndarray.t -> unit -> (Ndarray.t * Ndarray.t * Py.Object.t)
(**
Estimates the Minimum Covariance Determinant matrix.

Read more in the :ref:`User Guide <robust_covariance>`.

Parameters
----------
X : array-like, shape (n_samples, n_features)
  The data matrix, with p features and n samples.

support_fraction : float, 0 < support_fraction < 1
      The proportion of points to be included in the support of the raw
      MCD estimate. Default is None, which implies that the minimum
      value of support_fraction will be used within the algorithm:
      `[n_sample + n_features + 1] / 2`.

cov_computation_method : callable, default empirical_covariance
    The function which will be used to compute the covariance.
    Must return shape (n_features, n_features)

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

Notes
-----
The FastMCD algorithm has been introduced by Rousseuw and Van Driessen
in "A Fast Algorithm for the Minimum Covariance Determinant Estimator,
1999, American Statistical Association and the American Society
for Quality, TECHNOMETRICS".
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

Returns
-------
location : array-like, shape (n_features,)
    Robust location of the data.

covariance : array-like, shape (n_features, n_features)
    Robust covariance of the features.

support : array-like, type boolean, shape (n_samples,)
    A mask of the observations that have been used to compute
    the robust location and covariance estimates of the data set.
*)

val graphical_lasso : ?cov_init:Py.Object.t -> ?mode:[`Cd | `Lars] -> ?tol:float -> ?enet_tol:float -> ?max_iter:int -> ?verbose:bool -> ?return_costs:bool -> ?eps:float -> ?return_n_iter:bool -> emp_cov:Py.Object.t -> alpha:float -> unit -> (Py.Object.t * Py.Object.t * Py.Object.t * int)
(**
l1-penalized covariance estimator

Read more in the :ref:`User Guide <sparse_inverse_covariance>`.

Parameters
----------
emp_cov : 2D ndarray, shape (n_features, n_features)
    Empirical covariance from which to compute the covariance estimate.

alpha : positive float
    The regularization parameter: the higher alpha, the more
    regularization, the sparser the inverse covariance.

cov_init : 2D array (n_features, n_features), optional
    The initial guess for the covariance.

mode : {'cd', 'lars'}
    The Lasso solver to use: coordinate descent or LARS. Use LARS for
    very sparse underlying graphs, where p > n. Elsewhere prefer cd
    which is more numerically stable.

tol : positive float, optional
    The tolerance to declare convergence: if the dual gap goes below
    this value, iterations are stopped.

enet_tol : positive float, optional
    The tolerance for the elastic net solver used to calculate the descent
    direction. This parameter controls the accuracy of the search direction
    for a given column update, not of the overall parameter estimate. Only
    used for mode='cd'.

max_iter : integer, optional
    The maximum number of iterations.

verbose : boolean, optional
    If verbose is True, the objective function and dual gap are
    printed at each iteration.

return_costs : boolean, optional
    If return_costs is True, the objective function and dual gap
    at each iteration are returned.

eps : float, optional
    The machine-precision regularization in the computation of the
    Cholesky diagonal factors. Increase this for very ill-conditioned
    systems.

return_n_iter : bool, optional
    Whether or not to return the number of iterations.

Returns
-------
covariance : 2D ndarray, shape (n_features, n_features)
    The estimated covariance matrix.

precision : 2D ndarray, shape (n_features, n_features)
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

val ledoit_wolf : ?assume_centered:bool -> ?block_size:int -> x:Ndarray.t -> unit -> (Ndarray.t * float)
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

val ledoit_wolf_shrinkage : ?assume_centered:bool -> ?block_size:int -> x:Ndarray.t -> unit -> float
(**
Estimates the shrunk Ledoit-Wolf covariance matrix.

Read more in the :ref:`User Guide <shrunk_covariance>`.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Data from which to compute the Ledoit-Wolf shrunk covariance shrinkage.

assume_centered : bool
    If True, data will not be centered before computation.
    Useful to work with data whose mean is significantly equal to
    zero but is not exactly zero.
    If False, data will be centered before computation.

block_size : int
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

val log_likelihood : emp_cov:Py.Object.t -> precision:Py.Object.t -> unit -> Py.Object.t
(**
Computes the sample mean of the log_likelihood under a covariance model

computes the empirical expected log-likelihood (accounting for the
normalization terms and scaling), allowing for universal comparison (beyond
this software package)

Parameters
----------
emp_cov : 2D ndarray (n_features, n_features)
    Maximum Likelihood Estimator of covariance

precision : 2D ndarray (n_features, n_features)
    The precision matrix of the covariance model to be tested

Returns
-------
sample mean of the log-likelihood
*)

val oas : ?assume_centered:bool -> x:Ndarray.t -> unit -> (Ndarray.t * float)
(**
Estimate covariance with the Oracle Approximating Shrinkage algorithm.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Data from which to compute the covariance estimate.

assume_centered : boolean
  If True, data will not be centered before computation.
  Useful to work with data whose mean is significantly equal to
  zero but is not exactly zero.
  If False, data will be centered before computation.

Returns
-------
shrunk_cov : array-like, shape (n_features, n_features)
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

val shrunk_covariance : ?shrinkage:[`Float of float | `PyObject of Py.Object.t] -> emp_cov:Ndarray.t -> unit -> Ndarray.t
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

