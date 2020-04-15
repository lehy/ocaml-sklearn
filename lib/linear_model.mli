module ARDRegression : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_iter:int -> ?tol:float -> ?alpha_1:float -> ?alpha_2:float -> ?lambda_1:float -> ?lambda_2:float -> ?compute_score:bool -> ?threshold_lambda:float -> ?fit_intercept:bool -> ?normalize:bool -> ?copy_X:bool -> ?verbose:bool -> unit -> t
(**
Bayesian ARD regression.

Fit the weights of a regression model, using an ARD prior. The weights of
the regression model are assumed to be in Gaussian distributions.
Also estimate the parameters lambda (precisions of the distributions of the
weights) and alpha (precision of the distribution of the noise).
The estimation is done by an iterative procedures (Evidence Maximization)

Read more in the :ref:`User Guide <bayesian_regression>`.

Parameters
----------
n_iter : int, default=300
    Maximum number of iterations.

tol : float, default=1e-3
    Stop the algorithm if w has converged.

alpha_1 : float, default=1e-6
    Hyper-parameter : shape parameter for the Gamma distribution prior
    over the alpha parameter.

alpha_2 : float, default=1e-6
    Hyper-parameter : inverse scale parameter (rate parameter) for the
    Gamma distribution prior over the alpha parameter.

lambda_1 : float, default=1e-6
    Hyper-parameter : shape parameter for the Gamma distribution prior
    over the lambda parameter.

lambda_2 : float, default=1e-6
    Hyper-parameter : inverse scale parameter (rate parameter) for the
    Gamma distribution prior over the lambda parameter.

compute_score : bool, default=False
    If True, compute the objective function at each step of the model.

threshold_lambda : float, default=10 000
    threshold for removing (pruning) weights with high precision from
    the computation.

fit_intercept : bool, default=True
    whether to calculate the intercept for this model. If set
    to false, no intercept will be used in calculations
    (i.e. data is expected to be centered).

normalize : bool, default=False
    This parameter is ignored when ``fit_intercept`` is set to False.
    If True, the regressors X will be normalized before regression by
    subtracting the mean and dividing by the l2-norm.
    If you wish to standardize, please use
    :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
    on an estimator with ``normalize=False``.

copy_X : bool, default=True
    If True, X will be copied; else, it may be overwritten.

verbose : bool, default=False
    Verbose mode when fitting the model.

Attributes
----------
coef_ : array-like of shape (n_features,)
    Coefficients of the regression model (mean of distribution)

alpha_ : float
   estimated precision of the noise.

lambda_ : array-like of shape (n_features,)
   estimated precisions of the weights.

sigma_ : array-like of shape (n_features, n_features)
    estimated variance-covariance matrix of the weights

scores_ : float
    if computed, value of the objective function (to be maximized)

intercept_ : float
    Independent term in decision function. Set to 0.0 if
    ``fit_intercept = False``.

Examples
--------
>>> from sklearn import linear_model
>>> clf = linear_model.ARDRegression()
>>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
ARDRegression()
>>> clf.predict([[1, 1]])
array([1.])

Notes
-----
For an example, see :ref:`examples/linear_model/plot_ard.py
<sphx_glr_auto_examples_linear_model_plot_ard.py>`.

References
----------
D. J. C. MacKay, Bayesian nonlinear modeling for the prediction
competition, ASHRAE Transactions, 1994.

R. Salakhutdinov, Lecture notes on Statistical Machine Learning,
http://www.utstat.toronto.edu/~rsalakhu/sta4273/notes/Lecture2.pdf#page=15
Their beta is our ``self.alpha_``
Their alpha is our ``self.lambda_``
ARD is a little different than the slide: only dimensions/features for
which ``self.lambda_ < self.threshold_lambda`` are kept and the rest are
discarded.
*)

val fit : x:Ndarray.t -> y:Py.Object.t -> t -> t
(**
Fit the ARDRegression model according to the given training data
and parameters.

Iterative procedure to maximize the evidence

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training vector, where n_samples in the number of samples and
    n_features is the number of features.
y : array-like of shape (n_samples,)
    Target values (integers). Will be cast to X's dtype if necessary

Returns
-------
self : returns an instance of self.
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

val predict : ?return_std:bool -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Predict using the linear model.

In addition to the mean of the predictive distribution, also its
standard deviation can be returned.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Samples.

return_std : bool, default=False
    Whether to return the standard deviation of posterior prediction.

Returns
-------
y_mean : array-like of shape (n_samples,)
    Mean of predictive distribution of query points.

y_std : array-like of shape (n_samples,)
    Standard deviation of predictive distribution of query points.
*)

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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


(** Attribute coef_: see constructor for documentation *)
val coef_ : t -> Ndarray.t

(** Attribute alpha_: see constructor for documentation *)
val alpha_ : t -> float

(** Attribute lambda_: see constructor for documentation *)
val lambda_ : t -> Ndarray.t

(** Attribute sigma_: see constructor for documentation *)
val sigma_ : t -> Ndarray.t

(** Attribute scores_: see constructor for documentation *)
val scores_ : t -> float

(** Attribute intercept_: see constructor for documentation *)
val intercept_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module BayesianRidge : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_iter:int -> ?tol:float -> ?alpha_1:float -> ?alpha_2:float -> ?lambda_1:float -> ?lambda_2:float -> ?alpha_init:float -> ?lambda_init:float -> ?compute_score:bool -> ?fit_intercept:bool -> ?normalize:bool -> ?copy_X:bool -> ?verbose:bool -> unit -> t
(**
Bayesian ridge regression.

Fit a Bayesian ridge model. See the Notes section for details on this
implementation and the optimization of the regularization parameters
lambda (precision of the weights) and alpha (precision of the noise).

Read more in the :ref:`User Guide <bayesian_regression>`.

Parameters
----------
n_iter : int, default=300
    Maximum number of iterations. Should be greater than or equal to 1.

tol : float, default=1e-3
    Stop the algorithm if w has converged.

alpha_1 : float, default=1e-6
    Hyper-parameter : shape parameter for the Gamma distribution prior
    over the alpha parameter.

alpha_2 : float, default=1e-6
    Hyper-parameter : inverse scale parameter (rate parameter) for the
    Gamma distribution prior over the alpha parameter.

lambda_1 : float, default=1e-6
    Hyper-parameter : shape parameter for the Gamma distribution prior
    over the lambda parameter.

lambda_2 : float, default=1e-6
    Hyper-parameter : inverse scale parameter (rate parameter) for the
    Gamma distribution prior over the lambda parameter.

alpha_init : float, default=None
    Initial value for alpha (precision of the noise).
    If not set, alpha_init is 1/Var(y).

        .. versionadded:: 0.22

lambda_init : float, default=None
    Initial value for lambda (precision of the weights).
    If not set, lambda_init is 1.

        .. versionadded:: 0.22

compute_score : bool, default=False
    If True, compute the log marginal likelihood at each iteration of the
    optimization.

fit_intercept : bool, default=True
    Whether to calculate the intercept for this model.
    The intercept is not treated as a probabilistic parameter
    and thus has no associated variance. If set
    to False, no intercept will be used in calculations
    (i.e. data is expected to be centered).

normalize : bool, default=False
    This parameter is ignored when ``fit_intercept`` is set to False.
    If True, the regressors X will be normalized before regression by
    subtracting the mean and dividing by the l2-norm.
    If you wish to standardize, please use
    :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
    on an estimator with ``normalize=False``.

copy_X : bool, default=True
    If True, X will be copied; else, it may be overwritten.

verbose : bool, default=False
    Verbose mode when fitting the model.


Attributes
----------
coef_ : array-like of shape (n_features,)
    Coefficients of the regression model (mean of distribution)

intercept_ : float
    Independent term in decision function. Set to 0.0 if
    ``fit_intercept = False``.

alpha_ : float
   Estimated precision of the noise.

lambda_ : float
   Estimated precision of the weights.

sigma_ : array-like of shape (n_features, n_features)
    Estimated variance-covariance matrix of the weights

scores_ : array-like of shape (n_iter_+1,)
    If computed_score is True, value of the log marginal likelihood (to be
    maximized) at each iteration of the optimization. The array starts
    with the value of the log marginal likelihood obtained for the initial
    values of alpha and lambda and ends with the value obtained for the
    estimated alpha and lambda.

n_iter_ : int
    The actual number of iterations to reach the stopping criterion.

Examples
--------
>>> from sklearn import linear_model
>>> clf = linear_model.BayesianRidge()
>>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
BayesianRidge()
>>> clf.predict([[1, 1]])
array([1.])

Notes
-----
There exist several strategies to perform Bayesian ridge regression. This
implementation is based on the algorithm described in Appendix A of
(Tipping, 2001) where updates of the regularization parameters are done as
suggested in (MacKay, 1992). Note that according to A New
View of Automatic Relevance Determination (Wipf and Nagarajan, 2008) these
update rules do not guarantee that the marginal likelihood is increasing
between two consecutive iterations of the optimization.

References
----------
D. J. C. MacKay, Bayesian Interpolation, Computation and Neural Systems,
Vol. 4, No. 3, 1992.

M. E. Tipping, Sparse Bayesian Learning and the Relevance Vector Machine,
Journal of Machine Learning Research, Vol. 1, 2001.
*)

val fit : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Py.Object.t -> t -> t
(**
Fit the model

Parameters
----------
X : ndarray of shape (n_samples, n_features)
    Training data
y : ndarray of shape (n_samples,)
    Target values. Will be cast to X's dtype if necessary

sample_weight : ndarray of shape (n_samples,), default=None
    Individual weights for each sample

    .. versionadded:: 0.20
       parameter *sample_weight* support to BayesianRidge.

Returns
-------
self : returns an instance of self.
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

val predict : ?return_std:bool -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Predict using the linear model.

In addition to the mean of the predictive distribution, also its
standard deviation can be returned.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Samples.

return_std : bool, default=False
    Whether to return the standard deviation of posterior prediction.

Returns
-------
y_mean : array-like of shape (n_samples,)
    Mean of predictive distribution of query points.

y_std : array-like of shape (n_samples,)
    Standard deviation of predictive distribution of query points.
*)

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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


(** Attribute coef_: see constructor for documentation *)
val coef_ : t -> Ndarray.t

(** Attribute intercept_: see constructor for documentation *)
val intercept_ : t -> Ndarray.t

(** Attribute alpha_: see constructor for documentation *)
val alpha_ : t -> float

(** Attribute lambda_: see constructor for documentation *)
val lambda_ : t -> float

(** Attribute sigma_: see constructor for documentation *)
val sigma_ : t -> Ndarray.t

(** Attribute scores_: see constructor for documentation *)
val scores_ : t -> Ndarray.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> int

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module ElasticNet : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?alpha:float -> ?l1_ratio:float -> ?fit_intercept:bool -> ?normalize:bool -> ?precompute:[`Bool of bool | `Ndarray of Ndarray.t] -> ?max_iter:int -> ?copy_X:bool -> ?tol:float -> ?warm_start:bool -> ?positive:bool -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> ?selection:string -> unit -> t
(**
Linear regression with combined L1 and L2 priors as regularizer.

Minimizes the objective function::

        1 / (2 * n_samples) * ||y - Xw||^2_2
        + alpha * l1_ratio * ||w||_1
        + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

If you are interested in controlling the L1 and L2 penalty
separately, keep in mind that this is equivalent to::

        a * L1 + b * L2

where::

        alpha = a + b and l1_ratio = a / (a + b)

The parameter l1_ratio corresponds to alpha in the glmnet R package while
alpha corresponds to the lambda parameter in glmnet. Specifically, l1_ratio
= 1 is the lasso penalty. Currently, l1_ratio <= 0.01 is not reliable,
unless you supply your own sequence of alpha.

Read more in the :ref:`User Guide <elastic_net>`.

Parameters
----------
alpha : float, optional
    Constant that multiplies the penalty terms. Defaults to 1.0.
    See the notes for the exact mathematical meaning of this
    parameter. ``alpha = 0`` is equivalent to an ordinary least square,
    solved by the :class:`LinearRegression` object. For numerical
    reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised.
    Given this, you should use the :class:`LinearRegression` object.

l1_ratio : float
    The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
    ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
    is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
    combination of L1 and L2.

fit_intercept : bool
    Whether the intercept should be estimated or not. If ``False``, the
    data is assumed to be already centered.

normalize : boolean, optional, default False
    This parameter is ignored when ``fit_intercept`` is set to False.
    If True, the regressors X will be normalized before regression by
    subtracting the mean and dividing by the l2-norm.
    If you wish to standardize, please use
    :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
    on an estimator with ``normalize=False``.

precompute : True | False | array-like
    Whether to use a precomputed Gram matrix to speed up
    calculations. The Gram matrix can also be passed as argument.
    For sparse input this option is always ``True`` to preserve sparsity.

max_iter : int, optional
    The maximum number of iterations

copy_X : boolean, optional, default True
    If ``True``, X will be copied; else, it may be overwritten.

tol : float, optional
    The tolerance for the optimization: if the updates are
    smaller than ``tol``, the optimization code checks the
    dual gap for optimality and continues until it is smaller
    than ``tol``.

warm_start : bool, optional
    When set to ``True``, reuse the solution of the previous call to fit as
    initialization, otherwise, just erase the previous solution.
    See :term:`the Glossary <warm_start>`.

positive : bool, optional
    When set to ``True``, forces the coefficients to be positive.

random_state : int, RandomState instance or None, optional, default None
    The seed of the pseudo random number generator that selects a random
    feature to update.  If int, random_state is the seed used by the random
    number generator; If RandomState instance, random_state is the random
    number generator; If None, the random number generator is the
    RandomState instance used by `np.random`. Used when ``selection`` ==
    'random'.

selection : str, default 'cyclic'
    If set to 'random', a random coefficient is updated every iteration
    rather than looping over features sequentially by default. This
    (setting to 'random') often leads to significantly faster convergence
    especially when tol is higher than 1e-4.

Attributes
----------
coef_ : array, shape (n_features,) | (n_targets, n_features)
    parameter vector (w in the cost function formula)

sparse_coef_ : scipy.sparse matrix, shape (n_features, 1) |             (n_targets, n_features)
    ``sparse_coef_`` is a readonly property derived from ``coef_``

intercept_ : float | array, shape (n_targets,)
    independent term in decision function.

n_iter_ : array-like, shape (n_targets,)
    number of iterations run by the coordinate descent solver to reach
    the specified tolerance.

Examples
--------
>>> from sklearn.linear_model import ElasticNet
>>> from sklearn.datasets import make_regression

>>> X, y = make_regression(n_features=2, random_state=0)
>>> regr = ElasticNet(random_state=0)
>>> regr.fit(X, y)
ElasticNet(random_state=0)
>>> print(regr.coef_)
[18.83816048 64.55968825]
>>> print(regr.intercept_)
1.451...
>>> print(regr.predict([[0, 0]]))
[1.451...]


Notes
-----
To avoid unnecessary memory duplication the X argument of the fit method
should be directly passed as a Fortran-contiguous numpy array.

See also
--------
ElasticNetCV : Elastic net model with best model selection by
    cross-validation.
SGDRegressor: implements elastic net regression with incremental training.
SGDClassifier: implements logistic regression with elastic net penalty
    (``SGDClassifier(loss="log", penalty="elasticnet")``).
*)

val fit : ?check_input:bool -> x:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> y:Ndarray.t -> t -> t
(**
Fit model with coordinate descent.

Parameters
----------
X : ndarray or scipy.sparse matrix, (n_samples, n_features)
    Data

y : ndarray, shape (n_samples,) or (n_samples, n_targets)
    Target. Will be cast to X's dtype if necessary

check_input : boolean, (default=True)
    Allow to bypass several input checking.
    Don't use this parameter unless you know what you do.

Notes
-----

Coordinate descent is an algorithm that considers each column of
data at a time hence it will automatically convert the X input
as a Fortran-contiguous numpy array if necessary.

To avoid memory re-allocation it is advised to allocate the
initial data in memory directly using that format.
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

val predict : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Predict using the linear model.

Parameters
----------
X : array_like or sparse matrix, shape (n_samples, n_features)
    Samples.

Returns
-------
C : array, shape (n_samples,)
    Returns predicted values.
*)

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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


(** Attribute coef_: see constructor for documentation *)
val coef_ : t -> Ndarray.t

(** Attribute sparse_coef_: see constructor for documentation *)
val sparse_coef_ : t -> Py.Object.t

(** Attribute intercept_: see constructor for documentation *)
val intercept_ : t -> Ndarray.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module ElasticNetCV : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?l1_ratio:[`Float of float | `Ndarray of Ndarray.t] -> ?eps:float -> ?n_alphas:int -> ?alphas:Ndarray.t -> ?fit_intercept:bool -> ?normalize:bool -> ?precompute:[`Bool of bool | `Auto | `Ndarray of Ndarray.t] -> ?max_iter:int -> ?tol:float -> ?cv:[`Int of int | `CrossValGenerator of Py.Object.t | `Ndarray of Ndarray.t] -> ?copy_X:bool -> ?verbose:[`Bool of bool | `Int of int] -> ?n_jobs:[`Int of int | `None] -> ?positive:bool -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> ?selection:string -> unit -> t
(**
Elastic Net model with iterative fitting along a regularization path.

See glossary entry for :term:`cross-validation estimator`.

Read more in the :ref:`User Guide <elastic_net>`.

Parameters
----------
l1_ratio : float or array of floats, optional
    float between 0 and 1 passed to ElasticNet (scaling between
    l1 and l2 penalties). For ``l1_ratio = 0``
    the penalty is an L2 penalty. For ``l1_ratio = 1`` it is an L1 penalty.
    For ``0 < l1_ratio < 1``, the penalty is a combination of L1 and L2
    This parameter can be a list, in which case the different
    values are tested by cross-validation and the one giving the best
    prediction score is used. Note that a good choice of list of
    values for l1_ratio is often to put more values close to 1
    (i.e. Lasso) and less close to 0 (i.e. Ridge), as in ``[.1, .5, .7,
    .9, .95, .99, 1]``

eps : float, optional
    Length of the path. ``eps=1e-3`` means that
    ``alpha_min / alpha_max = 1e-3``.

n_alphas : int, optional
    Number of alphas along the regularization path, used for each l1_ratio.

alphas : numpy array, optional
    List of alphas where to compute the models.
    If None alphas are set automatically

fit_intercept : boolean
    whether to calculate the intercept for this model. If set
    to false, no intercept will be used in calculations
    (i.e. data is expected to be centered).

normalize : boolean, optional, default False
    This parameter is ignored when ``fit_intercept`` is set to False.
    If True, the regressors X will be normalized before regression by
    subtracting the mean and dividing by the l2-norm.
    If you wish to standardize, please use
    :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
    on an estimator with ``normalize=False``.

precompute : True | False | 'auto' | array-like
    Whether to use a precomputed Gram matrix to speed up
    calculations. If set to ``'auto'`` let us decide. The Gram
    matrix can also be passed as argument.

max_iter : int, optional
    The maximum number of iterations

tol : float, optional
    The tolerance for the optimization: if the updates are
    smaller than ``tol``, the optimization code checks the
    dual gap for optimality and continues until it is smaller
    than ``tol``.

cv : int, cross-validation generator or an iterable, optional
    Determines the cross-validation splitting strategy.
    Possible inputs for cv are:

    - None, to use the default 5-fold cross-validation,
    - integer, to specify the number of folds.
    - :term:`CV splitter`,
    - An iterable yielding (train, test) splits as arrays of indices.

    For integer/None inputs, :class:`KFold` is used.

    Refer :ref:`User Guide <cross_validation>` for the various
    cross-validation strategies that can be used here.

    .. versionchanged:: 0.22
        ``cv`` default value if None changed from 3-fold to 5-fold.

copy_X : boolean, optional, default True
    If ``True``, X will be copied; else, it may be overwritten.

verbose : bool or integer
    Amount of verbosity.

n_jobs : int or None, optional (default=None)
    Number of CPUs to use during the cross validation.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

positive : bool, optional
    When set to ``True``, forces the coefficients to be positive.

random_state : int, RandomState instance or None, optional, default None
    The seed of the pseudo random number generator that selects a random
    feature to update.  If int, random_state is the seed used by the random
    number generator; If RandomState instance, random_state is the random
    number generator; If None, the random number generator is the
    RandomState instance used by `np.random`. Used when ``selection`` ==
    'random'.

selection : str, default 'cyclic'
    If set to 'random', a random coefficient is updated every iteration
    rather than looping over features sequentially by default. This
    (setting to 'random') often leads to significantly faster convergence
    especially when tol is higher than 1e-4.

Attributes
----------
alpha_ : float
    The amount of penalization chosen by cross validation

l1_ratio_ : float
    The compromise between l1 and l2 penalization chosen by
    cross validation

coef_ : array, shape (n_features,) | (n_targets, n_features)
    Parameter vector (w in the cost function formula),

intercept_ : float | array, shape (n_targets, n_features)
    Independent term in the decision function.

mse_path_ : array, shape (n_l1_ratio, n_alpha, n_folds)
    Mean square error for the test set on each fold, varying l1_ratio and
    alpha.

alphas_ : numpy array, shape (n_alphas,) or (n_l1_ratio, n_alphas)
    The grid of alphas used for fitting, for each l1_ratio.

n_iter_ : int
    number of iterations run by the coordinate descent solver to reach
    the specified tolerance for the optimal alpha.

Examples
--------
>>> from sklearn.linear_model import ElasticNetCV
>>> from sklearn.datasets import make_regression

>>> X, y = make_regression(n_features=2, random_state=0)
>>> regr = ElasticNetCV(cv=5, random_state=0)
>>> regr.fit(X, y)
ElasticNetCV(cv=5, random_state=0)
>>> print(regr.alpha_)
0.199...
>>> print(regr.intercept_)
0.398...
>>> print(regr.predict([[0, 0]]))
[0.398...]


Notes
-----
For an example, see
:ref:`examples/linear_model/plot_lasso_model_selection.py
<sphx_glr_auto_examples_linear_model_plot_lasso_model_selection.py>`.

To avoid unnecessary memory duplication the X argument of the fit method
should be directly passed as a Fortran-contiguous numpy array.

The parameter l1_ratio corresponds to alpha in the glmnet R package
while alpha corresponds to the lambda parameter in glmnet.
More specifically, the optimization objective is::

    1 / (2 * n_samples) * ||y - Xw||^2_2
    + alpha * l1_ratio * ||w||_1
    + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

If you are interested in controlling the L1 and L2 penalty
separately, keep in mind that this is equivalent to::

    a * L1 + b * L2

for::

    alpha = a + b and l1_ratio = a / (a + b).

See also
--------
enet_path
ElasticNet
*)

val fit : x:Ndarray.t -> y:Ndarray.t -> t -> t
(**
Fit linear model with coordinate descent

Fit is on grid of alphas and best alpha estimated by cross-validation.

Parameters
----------
X : {array-like}, shape (n_samples, n_features)
    Training data. Pass directly as Fortran-contiguous data
    to avoid unnecessary memory duplication. If y is mono-output,
    X can be sparse.

y : array-like, shape (n_samples,) or (n_samples, n_targets)
    Target values
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

val predict : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Predict using the linear model.

Parameters
----------
X : array_like or sparse matrix, shape (n_samples, n_features)
    Samples.

Returns
-------
C : array, shape (n_samples,)
    Returns predicted values.
*)

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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


(** Attribute alpha_: see constructor for documentation *)
val alpha_ : t -> float

(** Attribute l1_ratio_: see constructor for documentation *)
val l1_ratio_ : t -> float

(** Attribute coef_: see constructor for documentation *)
val coef_ : t -> Ndarray.t

(** Attribute intercept_: see constructor for documentation *)
val intercept_ : t -> Ndarray.t

(** Attribute mse_path_: see constructor for documentation *)
val mse_path_ : t -> Ndarray.t

(** Attribute alphas_: see constructor for documentation *)
val alphas_ : t -> Ndarray.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> int

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module HuberRegressor : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?epsilon:float -> ?max_iter:int -> ?alpha:float -> ?warm_start:bool -> ?fit_intercept:bool -> ?tol:float -> unit -> t
(**
Linear regression model that is robust to outliers.

The Huber Regressor optimizes the squared loss for the samples where
``|(y - X'w) / sigma| < epsilon`` and the absolute loss for the samples
where ``|(y - X'w) / sigma| > epsilon``, where w and sigma are parameters
to be optimized. The parameter sigma makes sure that if y is scaled up
or down by a certain factor, one does not need to rescale epsilon to
achieve the same robustness. Note that this does not take into account
the fact that the different features of X may be of different scales.

This makes sure that the loss function is not heavily influenced by the
outliers while not completely ignoring their effect.

Read more in the :ref:`User Guide <huber_regression>`

.. versionadded:: 0.18

Parameters
----------
epsilon : float, greater than 1.0, default 1.35
    The parameter epsilon controls the number of samples that should be
    classified as outliers. The smaller the epsilon, the more robust it is
    to outliers.

max_iter : int, default 100
    Maximum number of iterations that
    ``scipy.optimize.minimize(method="L-BFGS-B")`` should run for.

alpha : float, default 0.0001
    Regularization parameter.

warm_start : bool, default False
    This is useful if the stored attributes of a previously used model
    has to be reused. If set to False, then the coefficients will
    be rewritten for every call to fit.
    See :term:`the Glossary <warm_start>`.

fit_intercept : bool, default True
    Whether or not to fit the intercept. This can be set to False
    if the data is already centered around the origin.

tol : float, default 1e-5
    The iteration will stop when
    ``max{ |proj g_i | i = 1, ..., n}`` <= ``tol``
    where pg_i is the i-th component of the projected gradient.

Attributes
----------
coef_ : array, shape (n_features,)
    Features got by optimizing the Huber loss.

intercept_ : float
    Bias.

scale_ : float
    The value by which ``|y - X'w - c|`` is scaled down.

n_iter_ : int
    Number of iterations that
    ``scipy.optimize.minimize(method="L-BFGS-B")`` has run for.

    .. versionchanged:: 0.20

        In SciPy <= 1.0.0 the number of lbfgs iterations may exceed
        ``max_iter``. ``n_iter_`` will now report at most ``max_iter``.

outliers_ : array, shape (n_samples,)
    A boolean mask which is set to True where the samples are identified
    as outliers.

Examples
--------
>>> import numpy as np
>>> from sklearn.linear_model import HuberRegressor, LinearRegression
>>> from sklearn.datasets import make_regression
>>> rng = np.random.RandomState(0)
>>> X, y, coef = make_regression(
...     n_samples=200, n_features=2, noise=4.0, coef=True, random_state=0)
>>> X[:4] = rng.uniform(10, 20, (4, 2))
>>> y[:4] = rng.uniform(10, 20, 4)
>>> huber = HuberRegressor().fit(X, y)
>>> huber.score(X, y)
-7.284608623514573
>>> huber.predict(X[:1,])
array([806.7200...])
>>> linear = LinearRegression().fit(X, y)
>>> print("True coefficients:", coef)
True coefficients: [20.4923...  34.1698...]
>>> print("Huber coefficients:", huber.coef_)
Huber coefficients: [17.7906... 31.0106...]
>>> print("Linear Regression coefficients:", linear.coef_)
Linear Regression coefficients: [-1.9221...  7.0226...]

References
----------
.. [1] Peter J. Huber, Elvezio M. Ronchetti, Robust Statistics
       Concomitant scale estimates, pg 172
.. [2] Art B. Owen (2006), A robust hybrid of lasso and ridge regression.
       https://statweb.stanford.edu/~owen/reports/hhu.pdf
*)

val fit : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> t
(**
Fit the model according to the given training data.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training vector, where n_samples in the number of samples and
    n_features is the number of features.

y : array-like, shape (n_samples,)
    Target vector relative to X.

sample_weight : array-like, shape (n_samples,)
    Weight given to each sample.

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

val predict : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Predict using the linear model.

Parameters
----------
X : array_like or sparse matrix, shape (n_samples, n_features)
    Samples.

Returns
-------
C : array, shape (n_samples,)
    Returns predicted values.
*)

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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


(** Attribute coef_: see constructor for documentation *)
val coef_ : t -> Ndarray.t

(** Attribute intercept_: see constructor for documentation *)
val intercept_ : t -> Ndarray.t

(** Attribute scale_: see constructor for documentation *)
val scale_ : t -> float

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> int

(** Attribute outliers_: see constructor for documentation *)
val outliers_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Lars : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?fit_intercept:bool -> ?verbose:[`Bool of bool | `Int of int] -> ?normalize:bool -> ?precompute:[`Bool of bool | `Auto | `Ndarray of Ndarray.t] -> ?n_nonzero_coefs:int -> ?eps:float -> ?copy_X:bool -> ?fit_path:bool -> unit -> t
(**
Least Angle Regression model a.k.a. LAR

Read more in the :ref:`User Guide <least_angle_regression>`.

Parameters
----------
fit_intercept : bool, default=True
    Whether to calculate the intercept for this model. If set
    to false, no intercept will be used in calculations
    (i.e. data is expected to be centered).

verbose : bool or int, default=False
    Sets the verbosity amount

normalize : bool, default=True
    This parameter is ignored when ``fit_intercept`` is set to False.
    If True, the regressors X will be normalized before regression by
    subtracting the mean and dividing by the l2-norm.
    If you wish to standardize, please use
    :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
    on an estimator with ``normalize=False``.

precompute : bool, 'auto' or array-like , default='auto'
    Whether to use a precomputed Gram matrix to speed up
    calculations. If set to ``'auto'`` let us decide. The Gram
    matrix can also be passed as argument.

n_nonzero_coefs : int, default=500
    Target number of non-zero coefficients. Use ``np.inf`` for no limit.

eps : float, optional
    The machine-precision regularization in the computation of the
    Cholesky diagonal factors. Increase this for very ill-conditioned
    systems. Unlike the ``tol`` parameter in some iterative
    optimization-based algorithms, this parameter does not control
    the tolerance of the optimization.
    By default, ``np.finfo(np.float).eps`` is used.

copy_X : bool, default=True
    If ``True``, X will be copied; else, it may be overwritten.

fit_path : bool, default=True
    If True the full path is stored in the ``coef_path_`` attribute.
    If you compute the solution for a large problem or many targets,
    setting ``fit_path`` to ``False`` will lead to a speedup, especially
    with a small alpha.

Attributes
----------
alphas_ : array-like of shape (n_alphas + 1,) | list of n_targets such             arrays
    Maximum of covariances (in absolute value) at each iteration.         ``n_alphas`` is either ``n_nonzero_coefs`` or ``n_features``,         whichever is smaller.

active_ : list, length = n_alphas | list of n_targets such lists
    Indices of active variables at the end of the path.

coef_path_ : array-like of shape (n_features, n_alphas + 1)         | list of n_targets such arrays
    The varying values of the coefficients along the path. It is not
    present if the ``fit_path`` parameter is ``False``.

coef_ : array-like of shape (n_features,) or (n_targets, n_features)
    Parameter vector (w in the formulation formula).

intercept_ : float or array-like of shape (n_targets,)
    Independent term in decision function.

n_iter_ : array-like or int
    The number of iterations taken by lars_path to find the
    grid of alphas for each target.

Examples
--------
>>> from sklearn import linear_model
>>> reg = linear_model.Lars(n_nonzero_coefs=1)
>>> reg.fit([[-1, 1], [0, 0], [1, 1]], [-1.1111, 0, -1.1111])
Lars(n_nonzero_coefs=1)
>>> print(reg.coef_)
[ 0. -1.11...]

See also
--------
lars_path, LarsCV
sklearn.decomposition.sparse_encode
*)

val fit : ?xy:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> t
(**
Fit the model using X, y as training data.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training data.

y : array-like of shape (n_samples,) or (n_samples, n_targets)
    Target values.

Xy : array-like of shape (n_samples,) or (n_samples, n_targets),                 default=None
    Xy = np.dot(X.T, y) that can be precomputed. It is useful
    only when the Gram matrix is precomputed.

Returns
-------
self : object
    returns an instance of self.
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

val predict : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Predict using the linear model.

Parameters
----------
X : array_like or sparse matrix, shape (n_samples, n_features)
    Samples.

Returns
-------
C : array, shape (n_samples,)
    Returns predicted values.
*)

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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


(** Attribute alphas_: see constructor for documentation *)
val alphas_ : t -> Py.Object.t

(** Attribute active_: see constructor for documentation *)
val active_ : t -> Py.Object.t

(** Attribute coef_path_: see constructor for documentation *)
val coef_path_ : t -> Py.Object.t

(** Attribute coef_: see constructor for documentation *)
val coef_ : t -> Ndarray.t

(** Attribute intercept_: see constructor for documentation *)
val intercept_ : t -> Ndarray.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> Py.Object.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module LarsCV : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?fit_intercept:bool -> ?verbose:[`Bool of bool | `Int of int] -> ?max_iter:int -> ?normalize:bool -> ?precompute:[`Bool of bool | `Auto | `Ndarray of Ndarray.t] -> ?cv:[`Int of int | `CrossValGenerator of Py.Object.t | `Ndarray of Ndarray.t] -> ?max_n_alphas:int -> ?n_jobs:[`Int of int | `None] -> ?eps:float -> ?copy_X:bool -> unit -> t
(**
Cross-validated Least Angle Regression model.

See glossary entry for :term:`cross-validation estimator`.

Read more in the :ref:`User Guide <least_angle_regression>`.

Parameters
----------
fit_intercept : bool, default=True
    whether to calculate the intercept for this model. If set
    to false, no intercept will be used in calculations
    (i.e. data is expected to be centered).

verbose : bool or int, default=False
    Sets the verbosity amount

max_iter : int, default=500
    Maximum number of iterations to perform.

normalize : bool, default=True
    This parameter is ignored when ``fit_intercept`` is set to False.
    If True, the regressors X will be normalized before regression by
    subtracting the mean and dividing by the l2-norm.
    If you wish to standardize, please use
    :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
    on an estimator with ``normalize=False``.

precompute : bool, 'auto' or array-like , default='auto'
    Whether to use a precomputed Gram matrix to speed up
    calculations. If set to ``'auto'`` let us decide. The Gram matrix
    cannot be passed as argument since we will use only subsets of X.

cv : int, cross-validation generator or an iterable, default=None
    Determines the cross-validation splitting strategy.
    Possible inputs for cv are:

    - None, to use the default 5-fold cross-validation,
    - integer, to specify the number of folds.
    - :term:`CV splitter`,
    - An iterable yielding (train, test) splits as arrays of indices.

    For integer/None inputs, :class:`KFold` is used.

    Refer :ref:`User Guide <cross_validation>` for the various
    cross-validation strategies that can be used here.

    .. versionchanged:: 0.22
        ``cv`` default value if None changed from 3-fold to 5-fold.

max_n_alphas : int, default=1000
    The maximum number of points on the path used to compute the
    residuals in the cross-validation

n_jobs : int or None, default=None
    Number of CPUs to use during the cross validation.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

eps : float, optional
    The machine-precision regularization in the computation of the
    Cholesky diagonal factors. Increase this for very ill-conditioned
    systems. By default, ``np.finfo(np.float).eps`` is used.

copy_X : bool, default=True
    If ``True``, X will be copied; else, it may be overwritten.

Attributes
----------
coef_ : array-like of shape (n_features,)
    parameter vector (w in the formulation formula)

intercept_ : float
    independent term in decision function

coef_path_ : array-like of shape (n_features, n_alphas)
    the varying values of the coefficients along the path

alpha_ : float
    the estimated regularization parameter alpha

alphas_ : array-like of shape (n_alphas,)
    the different values of alpha along the path

cv_alphas_ : array-like of shape (n_cv_alphas,)
    all the values of alpha along the path for the different folds

mse_path_ : array-like of shape (n_folds, n_cv_alphas)
    the mean square error on left-out for each fold along the path
    (alpha values given by ``cv_alphas``)

n_iter_ : array-like or int
    the number of iterations run by Lars with the optimal alpha.

Examples
--------
>>> from sklearn.linear_model import LarsCV
>>> from sklearn.datasets import make_regression
>>> X, y = make_regression(n_samples=200, noise=4.0, random_state=0)
>>> reg = LarsCV(cv=5).fit(X, y)
>>> reg.score(X, y)
0.9996...
>>> reg.alpha_
0.0254...
>>> reg.predict(X[:1,])
array([154.0842...])

See also
--------
lars_path, LassoLars, LassoLarsCV
*)

val fit : x:Ndarray.t -> y:Ndarray.t -> t -> t
(**
Fit the model using X, y as training data.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training data.

y : array-like of shape (n_samples,)
    Target values.

Returns
-------
self : object
    returns an instance of self.
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

val predict : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Predict using the linear model.

Parameters
----------
X : array_like or sparse matrix, shape (n_samples, n_features)
    Samples.

Returns
-------
C : array, shape (n_samples,)
    Returns predicted values.
*)

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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


(** Attribute coef_: see constructor for documentation *)
val coef_ : t -> Ndarray.t

(** Attribute intercept_: see constructor for documentation *)
val intercept_ : t -> Ndarray.t

(** Attribute coef_path_: see constructor for documentation *)
val coef_path_ : t -> Ndarray.t

(** Attribute alpha_: see constructor for documentation *)
val alpha_ : t -> float

(** Attribute alphas_: see constructor for documentation *)
val alphas_ : t -> Ndarray.t

(** Attribute cv_alphas_: see constructor for documentation *)
val cv_alphas_ : t -> Ndarray.t

(** Attribute mse_path_: see constructor for documentation *)
val mse_path_ : t -> Ndarray.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> Py.Object.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Lasso : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?alpha:float -> ?fit_intercept:bool -> ?normalize:bool -> ?precompute:[`Bool of bool | `Ndarray of Ndarray.t] -> ?copy_X:bool -> ?max_iter:int -> ?tol:float -> ?warm_start:bool -> ?positive:bool -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> ?selection:string -> unit -> t
(**
Linear Model trained with L1 prior as regularizer (aka the Lasso)

The optimization objective for Lasso is::

    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

Technically the Lasso model is optimizing the same objective function as
the Elastic Net with ``l1_ratio=1.0`` (no L2 penalty).

Read more in the :ref:`User Guide <lasso>`.

Parameters
----------
alpha : float, optional
    Constant that multiplies the L1 term. Defaults to 1.0.
    ``alpha = 0`` is equivalent to an ordinary least square, solved
    by the :class:`LinearRegression` object. For numerical
    reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised.
    Given this, you should use the :class:`LinearRegression` object.

fit_intercept : boolean, optional, default True
    Whether to calculate the intercept for this model. If set
    to False, no intercept will be used in calculations
    (i.e. data is expected to be centered).

normalize : boolean, optional, default False
    This parameter is ignored when ``fit_intercept`` is set to False.
    If True, the regressors X will be normalized before regression by
    subtracting the mean and dividing by the l2-norm.
    If you wish to standardize, please use
    :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
    on an estimator with ``normalize=False``.

precompute : True | False | array-like, default=False
    Whether to use a precomputed Gram matrix to speed up
    calculations. If set to ``'auto'`` let us decide. The Gram
    matrix can also be passed as argument. For sparse input
    this option is always ``True`` to preserve sparsity.

copy_X : boolean, optional, default True
    If ``True``, X will be copied; else, it may be overwritten.

max_iter : int, optional
    The maximum number of iterations

tol : float, optional
    The tolerance for the optimization: if the updates are
    smaller than ``tol``, the optimization code checks the
    dual gap for optimality and continues until it is smaller
    than ``tol``.

warm_start : bool, optional
    When set to True, reuse the solution of the previous call to fit as
    initialization, otherwise, just erase the previous solution.
    See :term:`the Glossary <warm_start>`.

positive : bool, optional
    When set to ``True``, forces the coefficients to be positive.

random_state : int, RandomState instance or None, optional, default None
    The seed of the pseudo random number generator that selects a random
    feature to update.  If int, random_state is the seed used by the random
    number generator; If RandomState instance, random_state is the random
    number generator; If None, the random number generator is the
    RandomState instance used by `np.random`. Used when ``selection`` ==
    'random'.

selection : str, default 'cyclic'
    If set to 'random', a random coefficient is updated every iteration
    rather than looping over features sequentially by default. This
    (setting to 'random') often leads to significantly faster convergence
    especially when tol is higher than 1e-4.

Attributes
----------
coef_ : array, shape (n_features,) | (n_targets, n_features)
    parameter vector (w in the cost function formula)

sparse_coef_ : scipy.sparse matrix, shape (n_features, 1) |             (n_targets, n_features)
    ``sparse_coef_`` is a readonly property derived from ``coef_``

intercept_ : float | array, shape (n_targets,)
    independent term in decision function.

n_iter_ : int | array-like, shape (n_targets,)
    number of iterations run by the coordinate descent solver to reach
    the specified tolerance.

Examples
--------
>>> from sklearn import linear_model
>>> clf = linear_model.Lasso(alpha=0.1)
>>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
Lasso(alpha=0.1)
>>> print(clf.coef_)
[0.85 0.  ]
>>> print(clf.intercept_)
0.15...

See also
--------
lars_path
lasso_path
LassoLars
LassoCV
LassoLarsCV
sklearn.decomposition.sparse_encode

Notes
-----
The algorithm used to fit the model is coordinate descent.

To avoid unnecessary memory duplication the X argument of the fit method
should be directly passed as a Fortran-contiguous numpy array.
*)

val fit : ?check_input:bool -> x:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> y:Ndarray.t -> t -> t
(**
Fit model with coordinate descent.

Parameters
----------
X : ndarray or scipy.sparse matrix, (n_samples, n_features)
    Data

y : ndarray, shape (n_samples,) or (n_samples, n_targets)
    Target. Will be cast to X's dtype if necessary

check_input : boolean, (default=True)
    Allow to bypass several input checking.
    Don't use this parameter unless you know what you do.

Notes
-----

Coordinate descent is an algorithm that considers each column of
data at a time hence it will automatically convert the X input
as a Fortran-contiguous numpy array if necessary.

To avoid memory re-allocation it is advised to allocate the
initial data in memory directly using that format.
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

val predict : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Predict using the linear model.

Parameters
----------
X : array_like or sparse matrix, shape (n_samples, n_features)
    Samples.

Returns
-------
C : array, shape (n_samples,)
    Returns predicted values.
*)

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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


(** Attribute coef_: see constructor for documentation *)
val coef_ : t -> Ndarray.t

(** Attribute sparse_coef_: see constructor for documentation *)
val sparse_coef_ : t -> Py.Object.t

(** Attribute intercept_: see constructor for documentation *)
val intercept_ : t -> Ndarray.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> Py.Object.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module LassoCV : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?eps:float -> ?n_alphas:int -> ?alphas:Ndarray.t -> ?fit_intercept:bool -> ?normalize:bool -> ?precompute:[`Bool of bool | `Auto | `Ndarray of Ndarray.t] -> ?max_iter:int -> ?tol:float -> ?copy_X:bool -> ?cv:[`Int of int | `CrossValGenerator of Py.Object.t | `Ndarray of Ndarray.t] -> ?verbose:[`Bool of bool | `Int of int] -> ?n_jobs:[`Int of int | `None] -> ?positive:bool -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> ?selection:string -> unit -> t
(**
Lasso linear model with iterative fitting along a regularization path.

See glossary entry for :term:`cross-validation estimator`.

The best model is selected by cross-validation.

The optimization objective for Lasso is::

    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

Read more in the :ref:`User Guide <lasso>`.

Parameters
----------
eps : float, optional
    Length of the path. ``eps=1e-3`` means that
    ``alpha_min / alpha_max = 1e-3``.

n_alphas : int, optional
    Number of alphas along the regularization path

alphas : numpy array, optional
    List of alphas where to compute the models.
    If ``None`` alphas are set automatically

fit_intercept : boolean, default True
    whether to calculate the intercept for this model. If set
    to false, no intercept will be used in calculations
    (i.e. data is expected to be centered).

normalize : boolean, optional, default False
    This parameter is ignored when ``fit_intercept`` is set to False.
    If True, the regressors X will be normalized before regression by
    subtracting the mean and dividing by the l2-norm.
    If you wish to standardize, please use
    :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
    on an estimator with ``normalize=False``.

precompute : True | False | 'auto' | array-like
    Whether to use a precomputed Gram matrix to speed up
    calculations. If set to ``'auto'`` let us decide. The Gram
    matrix can also be passed as argument.

max_iter : int, optional
    The maximum number of iterations

tol : float, optional
    The tolerance for the optimization: if the updates are
    smaller than ``tol``, the optimization code checks the
    dual gap for optimality and continues until it is smaller
    than ``tol``.

copy_X : boolean, optional, default True
    If ``True``, X will be copied; else, it may be overwritten.

cv : int, cross-validation generator or an iterable, optional
    Determines the cross-validation splitting strategy.
    Possible inputs for cv are:

    - None, to use the default 5-fold cross-validation,
    - integer, to specify the number of folds.
    - :term:`CV splitter`,
    - An iterable yielding (train, test) splits as arrays of indices.

    For integer/None inputs, :class:`KFold` is used.

    Refer :ref:`User Guide <cross_validation>` for the various
    cross-validation strategies that can be used here.

    .. versionchanged:: 0.22
        ``cv`` default value if None changed from 3-fold to 5-fold.

verbose : bool or integer
    Amount of verbosity.

n_jobs : int or None, optional (default=None)
    Number of CPUs to use during the cross validation.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

positive : bool, optional
    If positive, restrict regression coefficients to be positive

random_state : int, RandomState instance or None, optional, default None
    The seed of the pseudo random number generator that selects a random
    feature to update.  If int, random_state is the seed used by the random
    number generator; If RandomState instance, random_state is the random
    number generator; If None, the random number generator is the
    RandomState instance used by `np.random`. Used when ``selection`` ==
    'random'.

selection : str, default 'cyclic'
    If set to 'random', a random coefficient is updated every iteration
    rather than looping over features sequentially by default. This
    (setting to 'random') often leads to significantly faster convergence
    especially when tol is higher than 1e-4.

Attributes
----------
alpha_ : float
    The amount of penalization chosen by cross validation

coef_ : array, shape (n_features,) | (n_targets, n_features)
    parameter vector (w in the cost function formula)

intercept_ : float | array, shape (n_targets,)
    independent term in decision function.

mse_path_ : array, shape (n_alphas, n_folds)
    mean square error for the test set on each fold, varying alpha

alphas_ : numpy array, shape (n_alphas,)
    The grid of alphas used for fitting

dual_gap_ : ndarray, shape ()
    The dual gap at the end of the optimization for the optimal alpha
    (``alpha_``).

n_iter_ : int
    number of iterations run by the coordinate descent solver to reach
    the specified tolerance for the optimal alpha.

Examples
--------
>>> from sklearn.linear_model import LassoCV
>>> from sklearn.datasets import make_regression
>>> X, y = make_regression(noise=4, random_state=0)
>>> reg = LassoCV(cv=5, random_state=0).fit(X, y)
>>> reg.score(X, y)
0.9993...
>>> reg.predict(X[:1,])
array([-78.4951...])

Notes
-----
For an example, see
:ref:`examples/linear_model/plot_lasso_model_selection.py
<sphx_glr_auto_examples_linear_model_plot_lasso_model_selection.py>`.

To avoid unnecessary memory duplication the X argument of the fit method
should be directly passed as a Fortran-contiguous numpy array.

See also
--------
lars_path
lasso_path
LassoLars
Lasso
LassoLarsCV
*)

val fit : x:Ndarray.t -> y:Ndarray.t -> t -> t
(**
Fit linear model with coordinate descent

Fit is on grid of alphas and best alpha estimated by cross-validation.

Parameters
----------
X : {array-like}, shape (n_samples, n_features)
    Training data. Pass directly as Fortran-contiguous data
    to avoid unnecessary memory duplication. If y is mono-output,
    X can be sparse.

y : array-like, shape (n_samples,) or (n_samples, n_targets)
    Target values
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

val predict : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Predict using the linear model.

Parameters
----------
X : array_like or sparse matrix, shape (n_samples, n_features)
    Samples.

Returns
-------
C : array, shape (n_samples,)
    Returns predicted values.
*)

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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


(** Attribute alpha_: see constructor for documentation *)
val alpha_ : t -> float

(** Attribute coef_: see constructor for documentation *)
val coef_ : t -> Ndarray.t

(** Attribute intercept_: see constructor for documentation *)
val intercept_ : t -> Ndarray.t

(** Attribute mse_path_: see constructor for documentation *)
val mse_path_ : t -> Ndarray.t

(** Attribute alphas_: see constructor for documentation *)
val alphas_ : t -> Ndarray.t

(** Attribute dual_gap_: see constructor for documentation *)
val dual_gap_ : t -> Ndarray.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> int

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module LassoLars : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?alpha:float -> ?fit_intercept:bool -> ?verbose:[`Bool of bool | `Int of int] -> ?normalize:bool -> ?precompute:[`Bool of bool | `Auto | `Ndarray of Ndarray.t] -> ?max_iter:int -> ?eps:float -> ?copy_X:bool -> ?fit_path:bool -> ?positive:bool -> unit -> t
(**
Lasso model fit with Least Angle Regression a.k.a. Lars

It is a Linear Model trained with an L1 prior as regularizer.

The optimization objective for Lasso is::

(1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

Read more in the :ref:`User Guide <least_angle_regression>`.

Parameters
----------
alpha : float, default=1.0
    Constant that multiplies the penalty term. Defaults to 1.0.
    ``alpha = 0`` is equivalent to an ordinary least square, solved
    by :class:`LinearRegression`. For numerical reasons, using
    ``alpha = 0`` with the LassoLars object is not advised and you
    should prefer the LinearRegression object.

fit_intercept : bool, default=True
    whether to calculate the intercept for this model. If set
    to false, no intercept will be used in calculations
    (i.e. data is expected to be centered).

verbose : bool or int, default=False
    Sets the verbosity amount

normalize : bool, default=True
    This parameter is ignored when ``fit_intercept`` is set to False.
    If True, the regressors X will be normalized before regression by
    subtracting the mean and dividing by the l2-norm.
    If you wish to standardize, please use
    :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
    on an estimator with ``normalize=False``.

precompute : bool, 'auto' or array-like, default='auto'
    Whether to use a precomputed Gram matrix to speed up
    calculations. If set to ``'auto'`` let us decide. The Gram
    matrix can also be passed as argument.

max_iter : int, default=500
    Maximum number of iterations to perform.

eps : float, optional
    The machine-precision regularization in the computation of the
    Cholesky diagonal factors. Increase this for very ill-conditioned
    systems. Unlike the ``tol`` parameter in some iterative
    optimization-based algorithms, this parameter does not control
    the tolerance of the optimization.
    By default, ``np.finfo(np.float).eps`` is used.

copy_X : bool, default=True
    If True, X will be copied; else, it may be overwritten.

fit_path : bool, default=True
    If ``True`` the full path is stored in the ``coef_path_`` attribute.
    If you compute the solution for a large problem or many targets,
    setting ``fit_path`` to ``False`` will lead to a speedup, especially
    with a small alpha.

positive : bool, default=False
    Restrict coefficients to be >= 0. Be aware that you might want to
    remove fit_intercept which is set True by default.
    Under the positive restriction the model coefficients will not converge
    to the ordinary-least-squares solution for small values of alpha.
    Only coefficients up to the smallest alpha value (``alphas_[alphas_ >
    0.].min()`` when fit_path=True) reached by the stepwise Lars-Lasso
    algorithm are typically in congruence with the solution of the
    coordinate descent Lasso estimator.

Attributes
----------
alphas_ : array-like of shape (n_alphas + 1,) | list of n_targets such             arrays
    Maximum of covariances (in absolute value) at each iteration.         ``n_alphas`` is either ``max_iter``, ``n_features``, or the number of         nodes in the path with correlation greater than ``alpha``, whichever         is smaller.

active_ : list, length = n_alphas | list of n_targets such lists
    Indices of active variables at the end of the path.

coef_path_ : array-like of shape (n_features, n_alphas + 1) or list
    If a list is passed it's expected to be one of n_targets such arrays.
    The varying values of the coefficients along the path. It is not
    present if the ``fit_path`` parameter is ``False``.

coef_ : array-like of shape (n_features,) or (n_targets, n_features)
    Parameter vector (w in the formulation formula).

intercept_ : float or array-like of shape (n_targets,)
    Independent term in decision function.

n_iter_ : array-like or int.
    The number of iterations taken by lars_path to find the
    grid of alphas for each target.

Examples
--------
>>> from sklearn import linear_model
>>> reg = linear_model.LassoLars(alpha=0.01)
>>> reg.fit([[-1, 1], [0, 0], [1, 1]], [-1, 0, -1])
LassoLars(alpha=0.01)
>>> print(reg.coef_)
[ 0.         -0.963257...]

See also
--------
lars_path
lasso_path
Lasso
LassoCV
LassoLarsCV
LassoLarsIC
sklearn.decomposition.sparse_encode
*)

val fit : ?xy:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> t
(**
Fit the model using X, y as training data.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training data.

y : array-like of shape (n_samples,) or (n_samples, n_targets)
    Target values.

Xy : array-like of shape (n_samples,) or (n_samples, n_targets),                 default=None
    Xy = np.dot(X.T, y) that can be precomputed. It is useful
    only when the Gram matrix is precomputed.

Returns
-------
self : object
    returns an instance of self.
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

val predict : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Predict using the linear model.

Parameters
----------
X : array_like or sparse matrix, shape (n_samples, n_features)
    Samples.

Returns
-------
C : array, shape (n_samples,)
    Returns predicted values.
*)

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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


(** Attribute alphas_: see constructor for documentation *)
val alphas_ : t -> Py.Object.t

(** Attribute active_: see constructor for documentation *)
val active_ : t -> Py.Object.t

(** Attribute coef_path_: see constructor for documentation *)
val coef_path_ : t -> Py.Object.t

(** Attribute coef_: see constructor for documentation *)
val coef_ : t -> Ndarray.t

(** Attribute intercept_: see constructor for documentation *)
val intercept_ : t -> Ndarray.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> Py.Object.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module LassoLarsCV : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?fit_intercept:bool -> ?verbose:[`Bool of bool | `Int of int] -> ?max_iter:int -> ?normalize:bool -> ?precompute:[`Bool of bool | `Auto] -> ?cv:[`Int of int | `CrossValGenerator of Py.Object.t | `Ndarray of Ndarray.t] -> ?max_n_alphas:int -> ?n_jobs:[`Int of int | `None] -> ?eps:float -> ?copy_X:bool -> ?positive:bool -> unit -> t
(**
Cross-validated Lasso, using the LARS algorithm.

See glossary entry for :term:`cross-validation estimator`.

The optimization objective for Lasso is::

(1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

Read more in the :ref:`User Guide <least_angle_regression>`.

Parameters
----------
fit_intercept : bool, default=True
    whether to calculate the intercept for this model. If set
    to false, no intercept will be used in calculations
    (i.e. data is expected to be centered).

verbose : bool or int, default=False
    Sets the verbosity amount

max_iter : int, default=500
    Maximum number of iterations to perform.

normalize : bool, default=True
    This parameter is ignored when ``fit_intercept`` is set to False.
    If True, the regressors X will be normalized before regression by
    subtracting the mean and dividing by the l2-norm.
    If you wish to standardize, please use
    :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
    on an estimator with ``normalize=False``.

precompute : bool or 'auto' , default='auto'
    Whether to use a precomputed Gram matrix to speed up
    calculations. If set to ``'auto'`` let us decide. The Gram matrix
    cannot be passed as argument since we will use only subsets of X.

cv : int, cross-validation generator or an iterable, default=None
    Determines the cross-validation splitting strategy.
    Possible inputs for cv are:

    - None, to use the default 5-fold cross-validation,
    - integer, to specify the number of folds.
    - :term:`CV splitter`,
    - An iterable yielding (train, test) splits as arrays of indices.

    For integer/None inputs, :class:`KFold` is used.

    Refer :ref:`User Guide <cross_validation>` for the various
    cross-validation strategies that can be used here.

    .. versionchanged:: 0.22
        ``cv`` default value if None changed from 3-fold to 5-fold.

max_n_alphas : int, default=1000
    The maximum number of points on the path used to compute the
    residuals in the cross-validation

n_jobs : int or None, default=None
    Number of CPUs to use during the cross validation.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

eps : float, optional
    The machine-precision regularization in the computation of the
    Cholesky diagonal factors. Increase this for very ill-conditioned
    systems. By default, ``np.finfo(np.float).eps`` is used.

copy_X : bool, default=True
    If True, X will be copied; else, it may be overwritten.

positive : bool, default=False
    Restrict coefficients to be >= 0. Be aware that you might want to
    remove fit_intercept which is set True by default.
    Under the positive restriction the model coefficients do not converge
    to the ordinary-least-squares solution for small values of alpha.
    Only coefficients up to the smallest alpha value (``alphas_[alphas_ >
    0.].min()`` when fit_path=True) reached by the stepwise Lars-Lasso
    algorithm are typically in congruence with the solution of the
    coordinate descent Lasso estimator.
    As a consequence using LassoLarsCV only makes sense for problems where
    a sparse solution is expected and/or reached.

Attributes
----------
coef_ : array-like of shape (n_features,)
    parameter vector (w in the formulation formula)

intercept_ : float
    independent term in decision function.

coef_path_ : array-like of shape (n_features, n_alphas)
    the varying values of the coefficients along the path

alpha_ : float
    the estimated regularization parameter alpha

alphas_ : array-like of shape (n_alphas,)
    the different values of alpha along the path

cv_alphas_ : array-like of shape (n_cv_alphas,)
    all the values of alpha along the path for the different folds

mse_path_ : array-like of shape (n_folds, n_cv_alphas)
    the mean square error on left-out for each fold along the path
    (alpha values given by ``cv_alphas``)

n_iter_ : array-like or int
    the number of iterations run by Lars with the optimal alpha.

Examples
--------
>>> from sklearn.linear_model import LassoLarsCV
>>> from sklearn.datasets import make_regression
>>> X, y = make_regression(noise=4.0, random_state=0)
>>> reg = LassoLarsCV(cv=5).fit(X, y)
>>> reg.score(X, y)
0.9992...
>>> reg.alpha_
0.0484...
>>> reg.predict(X[:1,])
array([-77.8723...])

Notes
-----

The object solves the same problem as the LassoCV object. However,
unlike the LassoCV, it find the relevant alphas values by itself.
In general, because of this property, it will be more stable.
However, it is more fragile to heavily multicollinear datasets.

It is more efficient than the LassoCV if only a small number of
features are selected compared to the total number, for instance if
there are very few samples compared to the number of features.

See also
--------
lars_path, LassoLars, LarsCV, LassoCV
*)

val fit : x:Ndarray.t -> y:Ndarray.t -> t -> t
(**
Fit the model using X, y as training data.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training data.

y : array-like of shape (n_samples,)
    Target values.

Returns
-------
self : object
    returns an instance of self.
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

val predict : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Predict using the linear model.

Parameters
----------
X : array_like or sparse matrix, shape (n_samples, n_features)
    Samples.

Returns
-------
C : array, shape (n_samples,)
    Returns predicted values.
*)

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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


(** Attribute coef_: see constructor for documentation *)
val coef_ : t -> Ndarray.t

(** Attribute intercept_: see constructor for documentation *)
val intercept_ : t -> Ndarray.t

(** Attribute coef_path_: see constructor for documentation *)
val coef_path_ : t -> Ndarray.t

(** Attribute alpha_: see constructor for documentation *)
val alpha_ : t -> float

(** Attribute alphas_: see constructor for documentation *)
val alphas_ : t -> Ndarray.t

(** Attribute cv_alphas_: see constructor for documentation *)
val cv_alphas_ : t -> Ndarray.t

(** Attribute mse_path_: see constructor for documentation *)
val mse_path_ : t -> Ndarray.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> Py.Object.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module LassoLarsIC : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?criterion:[`Bic | `Aic] -> ?fit_intercept:bool -> ?verbose:[`Bool of bool | `Int of int] -> ?normalize:bool -> ?precompute:[`Bool of bool | `Auto | `Ndarray of Ndarray.t] -> ?max_iter:int -> ?eps:float -> ?copy_X:bool -> ?positive:bool -> unit -> t
(**
Lasso model fit with Lars using BIC or AIC for model selection

The optimization objective for Lasso is::

(1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

AIC is the Akaike information criterion and BIC is the Bayes
Information criterion. Such criteria are useful to select the value
of the regularization parameter by making a trade-off between the
goodness of fit and the complexity of the model. A good model should
explain well the data while being simple.

Read more in the :ref:`User Guide <least_angle_regression>`.

Parameters
----------
criterion : {'bic' , 'aic'}, default='aic'
    The type of criterion to use.

fit_intercept : bool, default=True
    whether to calculate the intercept for this model. If set
    to false, no intercept will be used in calculations
    (i.e. data is expected to be centered).

verbose : bool or int, default=False
    Sets the verbosity amount

normalize : bool, default=True
    This parameter is ignored when ``fit_intercept`` is set to False.
    If True, the regressors X will be normalized before regression by
    subtracting the mean and dividing by the l2-norm.
    If you wish to standardize, please use
    :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
    on an estimator with ``normalize=False``.

precompute : bool, 'auto' or array-like, default='auto'
    Whether to use a precomputed Gram matrix to speed up
    calculations. If set to ``'auto'`` let us decide. The Gram
    matrix can also be passed as argument.

max_iter : int, default=500
    Maximum number of iterations to perform. Can be used for
    early stopping.

eps : float, optional
    The machine-precision regularization in the computation of the
    Cholesky diagonal factors. Increase this for very ill-conditioned
    systems. Unlike the ``tol`` parameter in some iterative
    optimization-based algorithms, this parameter does not control
    the tolerance of the optimization.
    By default, ``np.finfo(np.float).eps`` is used

copy_X : bool, default=True
    If True, X will be copied; else, it may be overwritten.

positive : bool, default=False
    Restrict coefficients to be >= 0. Be aware that you might want to
    remove fit_intercept which is set True by default.
    Under the positive restriction the model coefficients do not converge
    to the ordinary-least-squares solution for small values of alpha.
    Only coefficients up to the smallest alpha value (``alphas_[alphas_ >
    0.].min()`` when fit_path=True) reached by the stepwise Lars-Lasso
    algorithm are typically in congruence with the solution of the
    coordinate descent Lasso estimator.
    As a consequence using LassoLarsIC only makes sense for problems where
    a sparse solution is expected and/or reached.

Attributes
----------
coef_ : array-like of shape (n_features,)
    parameter vector (w in the formulation formula)

intercept_ : float
    independent term in decision function.

alpha_ : float
    the alpha parameter chosen by the information criterion

n_iter_ : int
    number of iterations run by lars_path to find the grid of
    alphas.

criterion_ : array-like of shape (n_alphas,)
    The value of the information criteria ('aic', 'bic') across all
    alphas. The alpha which has the smallest information criterion is
    chosen. This value is larger by a factor of ``n_samples`` compared to
    Eqns. 2.15 and 2.16 in (Zou et al, 2007).


Examples
--------
>>> from sklearn import linear_model
>>> reg = linear_model.LassoLarsIC(criterion='bic')
>>> reg.fit([[-1, 1], [0, 0], [1, 1]], [-1.1111, 0, -1.1111])
LassoLarsIC(criterion='bic')
>>> print(reg.coef_)
[ 0.  -1.11...]

Notes
-----
The estimation of the number of degrees of freedom is given by:

"On the degrees of freedom of the lasso"
Hui Zou, Trevor Hastie, and Robert Tibshirani
Ann. Statist. Volume 35, Number 5 (2007), 2173-2192.

https://en.wikipedia.org/wiki/Akaike_information_criterion
https://en.wikipedia.org/wiki/Bayesian_information_criterion

See also
--------
lars_path, LassoLars, LassoLarsCV
*)

val fit : ?copy_X:bool -> x:Ndarray.t -> y:Ndarray.t -> t -> t
(**
Fit the model using X, y as training data.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    training data.

y : array-like of shape (n_samples,)
    target values. Will be cast to X's dtype if necessary

copy_X : bool, default=None
    If provided, this parameter will override the choice
    of copy_X made at instance creation.
    If ``True``, X will be copied; else, it may be overwritten.

Returns
-------
self : object
    returns an instance of self.
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

val predict : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Predict using the linear model.

Parameters
----------
X : array_like or sparse matrix, shape (n_samples, n_features)
    Samples.

Returns
-------
C : array, shape (n_samples,)
    Returns predicted values.
*)

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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


(** Attribute coef_: see constructor for documentation *)
val coef_ : t -> Ndarray.t

(** Attribute intercept_: see constructor for documentation *)
val intercept_ : t -> Ndarray.t

(** Attribute alpha_: see constructor for documentation *)
val alpha_ : t -> float

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> int

(** Attribute criterion_: see constructor for documentation *)
val criterion_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module LinearRegression : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?fit_intercept:bool -> ?normalize:bool -> ?copy_X:bool -> ?n_jobs:[`Int of int | `None] -> unit -> t
(**
Ordinary least squares Linear Regression.

LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
to minimize the residual sum of squares between the observed targets in
the dataset, and the targets predicted by the linear approximation.

Parameters
----------
fit_intercept : bool, optional, default True
    Whether to calculate the intercept for this model. If set
    to False, no intercept will be used in calculations
    (i.e. data is expected to be centered).

normalize : bool, optional, default False
    This parameter is ignored when ``fit_intercept`` is set to False.
    If True, the regressors X will be normalized before regression by
    subtracting the mean and dividing by the l2-norm.
    If you wish to standardize, please use
    :class:`sklearn.preprocessing.StandardScaler` before calling ``fit`` on
    an estimator with ``normalize=False``.

copy_X : bool, optional, default True
    If True, X will be copied; else, it may be overwritten.

n_jobs : int or None, optional (default=None)
    The number of jobs to use for the computation. This will only provide
    speedup for n_targets > 1 and sufficient large problems.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

Attributes
----------
coef_ : array of shape (n_features, ) or (n_targets, n_features)
    Estimated coefficients for the linear regression problem.
    If multiple targets are passed during the fit (y 2D), this
    is a 2D array of shape (n_targets, n_features), while if only
    one target is passed, this is a 1D array of length n_features.

rank_ : int
    Rank of matrix `X`. Only available when `X` is dense.

singular_ : array of shape (min(X, y),)
    Singular values of `X`. Only available when `X` is dense.

intercept_ : float or array of shape of (n_targets,)
    Independent term in the linear model. Set to 0.0 if
    `fit_intercept = False`.

See Also
--------
sklearn.linear_model.Ridge : Ridge regression addresses some of the
    problems of Ordinary Least Squares by imposing a penalty on the
    size of the coefficients with l2 regularization.
sklearn.linear_model.Lasso : The Lasso is a linear model that estimates
    sparse coefficients with l1 regularization.
sklearn.linear_model.ElasticNet : Elastic-Net is a linear regression
    model trained with both l1 and l2 -norm regularization of the
    coefficients.

Notes
-----
From the implementation point of view, this is just plain Ordinary
Least Squares (scipy.linalg.lstsq) wrapped as a predictor object.

Examples
--------
>>> import numpy as np
>>> from sklearn.linear_model import LinearRegression
>>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
>>> # y = 1 * x_0 + 2 * x_1 + 3
>>> y = np.dot(X, np.array([1, 2])) + 3
>>> reg = LinearRegression().fit(X, y)
>>> reg.score(X, y)
1.0
>>> reg.coef_
array([1., 2.])
>>> reg.intercept_
3.0000...
>>> reg.predict(np.array([[3, 5]]))
array([16.])
*)

val fit : ?sample_weight:Ndarray.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:Ndarray.t -> t -> t
(**
Fit linear model.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Training data

y : array-like of shape (n_samples,) or (n_samples, n_targets)
    Target values. Will be cast to X's dtype if necessary

sample_weight : array-like of shape (n_samples,), default=None
    Individual weights for each sample

    .. versionadded:: 0.17
       parameter *sample_weight* support to LinearRegression.

Returns
-------
self : returns an instance of self.
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

val predict : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Predict using the linear model.

Parameters
----------
X : array_like or sparse matrix, shape (n_samples, n_features)
    Samples.

Returns
-------
C : array, shape (n_samples,)
    Returns predicted values.
*)

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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


(** Attribute coef_: see constructor for documentation *)
val coef_ : t -> Ndarray.t

(** Attribute rank_: see constructor for documentation *)
val rank_ : t -> int

(** Attribute singular_: see constructor for documentation *)
val singular_ : t -> Py.Object.t

(** Attribute intercept_: see constructor for documentation *)
val intercept_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module LogisticRegression : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?penalty:[`L1 | `L2 | `Elasticnet | `None] -> ?dual:bool -> ?tol:float -> ?c:float -> ?fit_intercept:bool -> ?intercept_scaling:float -> ?class_weight:[`DictIntToFloat of (int * float) list | `Balanced] -> ?random_state:[`Int of int | `RandomState of Py.Object.t] -> ?solver:[`Newton_cg | `Lbfgs | `Liblinear | `Sag | `Saga] -> ?max_iter:int -> ?multi_class:[`Auto | `Ovr | `Multinomial] -> ?verbose:int -> ?warm_start:bool -> ?n_jobs:int -> ?l1_ratio:float -> unit -> t
(**
Logistic Regression (aka logit, MaxEnt) classifier.

In the multiclass case, the training algorithm uses the one-vs-rest (OvR)
scheme if the 'multi_class' option is set to 'ovr', and uses the
cross-entropy loss if the 'multi_class' option is set to 'multinomial'.
(Currently the 'multinomial' option is supported only by the 'lbfgs',
'sag', 'saga' and 'newton-cg' solvers.)

This class implements regularized logistic regression using the
'liblinear' library, 'newton-cg', 'sag', 'saga' and 'lbfgs' solvers. **Note
that regularization is applied by default**. It can handle both dense
and sparse input. Use C-ordered arrays or CSR matrices containing 64-bit
floats for optimal performance; any other input format will be converted
(and copied).

The 'newton-cg', 'sag', and 'lbfgs' solvers support only L2 regularization
with primal formulation, or no regularization. The 'liblinear' solver
supports both L1 and L2 regularization, with a dual formulation only for
the L2 penalty. The Elastic-Net regularization is only supported by the
'saga' solver.

Read more in the :ref:`User Guide <logistic_regression>`.

Parameters
----------
penalty : {'l1', 'l2', 'elasticnet', 'none'}, default='l2'
    Used to specify the norm used in the penalization. The 'newton-cg',
    'sag' and 'lbfgs' solvers support only l2 penalties. 'elasticnet' is
    only supported by the 'saga' solver. If 'none' (not supported by the
    liblinear solver), no regularization is applied.

    .. versionadded:: 0.19
       l1 penalty with SAGA solver (allowing 'multinomial' + L1)

dual : bool, default=False
    Dual or primal formulation. Dual formulation is only implemented for
    l2 penalty with liblinear solver. Prefer dual=False when
    n_samples > n_features.

tol : float, default=1e-4
    Tolerance for stopping criteria.

C : float, default=1.0
    Inverse of regularization strength; must be a positive float.
    Like in support vector machines, smaller values specify stronger
    regularization.

fit_intercept : bool, default=True
    Specifies if a constant (a.k.a. bias or intercept) should be
    added to the decision function.

intercept_scaling : float, default=1
    Useful only when the solver 'liblinear' is used
    and self.fit_intercept is set to True. In this case, x becomes
    [x, self.intercept_scaling],
    i.e. a "synthetic" feature with constant value equal to
    intercept_scaling is appended to the instance vector.
    The intercept becomes ``intercept_scaling * synthetic_feature_weight``.

    Note! the synthetic feature weight is subject to l1/l2 regularization
    as all other features.
    To lessen the effect of regularization on synthetic feature weight
    (and therefore on the intercept) intercept_scaling has to be increased.

class_weight : dict or 'balanced', default=None
    Weights associated with classes in the form ``{class_label: weight}``.
    If not given, all classes are supposed to have weight one.

    The "balanced" mode uses the values of y to automatically adjust
    weights inversely proportional to class frequencies in the input data
    as ``n_samples / (n_classes * np.bincount(y))``.

    Note that these weights will be multiplied with sample_weight (passed
    through the fit method) if sample_weight is specified.

    .. versionadded:: 0.17
       *class_weight='balanced'*

random_state : int, RandomState instance, default=None
    The seed of the pseudo random number generator to use when shuffling
    the data.  If int, random_state is the seed used by the random number
    generator; If RandomState instance, random_state is the random number
    generator; If None, the random number generator is the RandomState
    instance used by `np.random`. Used when ``solver`` == 'sag' or
    'liblinear'.

solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'},             default='lbfgs'

    Algorithm to use in the optimization problem.

    - For small datasets, 'liblinear' is a good choice, whereas 'sag' and
      'saga' are faster for large ones.
    - For multiclass problems, only 'newton-cg', 'sag', 'saga' and 'lbfgs'
      handle multinomial loss; 'liblinear' is limited to one-versus-rest
      schemes.
    - 'newton-cg', 'lbfgs', 'sag' and 'saga' handle L2 or no penalty
    - 'liblinear' and 'saga' also handle L1 penalty
    - 'saga' also supports 'elasticnet' penalty
    - 'liblinear' does not support setting ``penalty='none'``

    Note that 'sag' and 'saga' fast convergence is only guaranteed on
    features with approximately the same scale. You can
    preprocess the data with a scaler from sklearn.preprocessing.

    .. versionadded:: 0.17
       Stochastic Average Gradient descent solver.
    .. versionadded:: 0.19
       SAGA solver.
    .. versionchanged:: 0.22
        The default solver changed from 'liblinear' to 'lbfgs' in 0.22.

max_iter : int, default=100
    Maximum number of iterations taken for the solvers to converge.

multi_class : {'auto', 'ovr', 'multinomial'}, default='auto'
    If the option chosen is 'ovr', then a binary problem is fit for each
    label. For 'multinomial' the loss minimised is the multinomial loss fit
    across the entire probability distribution, *even when the data is
    binary*. 'multinomial' is unavailable when solver='liblinear'.
    'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
    and otherwise selects 'multinomial'.

    .. versionadded:: 0.18
       Stochastic Average Gradient descent solver for 'multinomial' case.
    .. versionchanged:: 0.22
        Default changed from 'ovr' to 'auto' in 0.22.

verbose : int, default=0
    For the liblinear and lbfgs solvers set verbose to any positive
    number for verbosity.

warm_start : bool, default=False
    When set to True, reuse the solution of the previous call to fit as
    initialization, otherwise, just erase the previous solution.
    Useless for liblinear solver. See :term:`the Glossary <warm_start>`.

    .. versionadded:: 0.17
       *warm_start* to support *lbfgs*, *newton-cg*, *sag*, *saga* solvers.

n_jobs : int, default=None
    Number of CPU cores used when parallelizing over classes if
    multi_class='ovr'". This parameter is ignored when the ``solver`` is
    set to 'liblinear' regardless of whether 'multi_class' is specified or
    not. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
    context. ``-1`` means using all processors.
    See :term:`Glossary <n_jobs>` for more details.

l1_ratio : float, default=None
    The Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``. Only
    used if ``penalty='elasticnet'`. Setting ``l1_ratio=0`` is equivalent
    to using ``penalty='l2'``, while setting ``l1_ratio=1`` is equivalent
    to using ``penalty='l1'``. For ``0 < l1_ratio <1``, the penalty is a
    combination of L1 and L2.

Attributes
----------

classes_ : ndarray of shape (n_classes, )
    A list of class labels known to the classifier.

coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
    Coefficient of the features in the decision function.

    `coef_` is of shape (1, n_features) when the given problem is binary.
    In particular, when `multi_class='multinomial'`, `coef_` corresponds
    to outcome 1 (True) and `-coef_` corresponds to outcome 0 (False).

intercept_ : ndarray of shape (1,) or (n_classes,)
    Intercept (a.k.a. bias) added to the decision function.

    If `fit_intercept` is set to False, the intercept is set to zero.
    `intercept_` is of shape (1,) when the given problem is binary.
    In particular, when `multi_class='multinomial'`, `intercept_`
    corresponds to outcome 1 (True) and `-intercept_` corresponds to
    outcome 0 (False).

n_iter_ : ndarray of shape (n_classes,) or (1, )
    Actual number of iterations for all classes. If binary or multinomial,
    it returns only 1 element. For liblinear solver, only the maximum
    number of iteration across all classes is given.

    .. versionchanged:: 0.20

        In SciPy <= 1.0.0 the number of lbfgs iterations may exceed
        ``max_iter``. ``n_iter_`` will now report at most ``max_iter``.

See Also
--------
SGDClassifier : Incrementally trained logistic regression (when given
    the parameter ``loss="log"``).
LogisticRegressionCV : Logistic regression with built-in cross validation.

Notes
-----
The underlying C implementation uses a random number generator to
select features when fitting the model. It is thus not uncommon,
to have slightly different results for the same input data. If
that happens, try with a smaller tol parameter.

Predict output may not match that of standalone liblinear in certain
cases. See :ref:`differences from liblinear <liblinear_differences>`
in the narrative documentation.

References
----------

L-BFGS-B -- Software for Large-scale Bound-constrained Optimization
    Ciyou Zhu, Richard Byrd, Jorge Nocedal and Jose Luis Morales.
    http://users.iems.northwestern.edu/~nocedal/lbfgsb.html

LIBLINEAR -- A Library for Large Linear Classification
    https://www.csie.ntu.edu.tw/~cjlin/liblinear/

SAG -- Mark Schmidt, Nicolas Le Roux, and Francis Bach
    Minimizing Finite Sums with the Stochastic Average Gradient
    https://hal.inria.fr/hal-00860051/document

SAGA -- Defazio, A., Bach F. & Lacoste-Julien S. (2014).
    SAGA: A Fast Incremental Gradient Method With Support
    for Non-Strongly Convex Composite Objectives
    https://arxiv.org/abs/1407.0202

Hsiang-Fu Yu, Fang-Lan Huang, Chih-Jen Lin (2011). Dual coordinate descent
    methods for logistic regression and maximum entropy models.
    Machine Learning 85(1-2):41-75.
    https://www.csie.ntu.edu.tw/~cjlin/papers/maxent_dual.pdf

Examples
--------
>>> from sklearn.datasets import load_iris
>>> from sklearn.linear_model import LogisticRegression
>>> X, y = load_iris(return_X_y=True)
>>> clf = LogisticRegression(random_state=0).fit(X, y)
>>> clf.predict(X[:2, :])
array([0, 0])
>>> clf.predict_proba(X[:2, :])
array([[9.8...e-01, 1.8...e-02, 1.4...e-08],
       [9.7...e-01, 2.8...e-02, ...e-08]])
>>> clf.score(X, y)
0.97..."
*)

val decision_function : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
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

val densify : t -> t
(**
Convert coefficient matrix to dense array format.

Converts the ``coef_`` member (back) to a numpy.ndarray. This is the
default format of ``coef_`` and is required for fitting, so calling
this method is only required on models that have previously been
sparsified; otherwise, it is a no-op.

Returns
-------
self
    Fitted estimator.
*)

val fit : ?sample_weight:Ndarray.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:Ndarray.t -> t -> t
(**
Fit the model according to the given training data.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Training vector, where n_samples is the number of samples and
    n_features is the number of features.

y : array-like of shape (n_samples,)
    Target vector relative to X.

sample_weight : array-like of shape (n_samples,) default=None
    Array of weights that are assigned to individual samples.
    If not provided, then each sample is given unit weight.

    .. versionadded:: 0.17
       *sample_weight* support to LogisticRegression.

Returns
-------
self
    Fitted estimator.

Notes
-----
The SAGA solver supports both float64 and float32 bit arrays.
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

val predict : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
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

val predict_log_proba : x:Ndarray.t -> t -> Ndarray.t
(**
Predict logarithm of probability estimates.

The returned estimates for all classes are ordered by the
label of classes.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Vector to be scored, where `n_samples` is the number of samples and
    `n_features` is the number of features.

Returns
-------
T : array-like of shape (n_samples, n_classes)
    Returns the log-probability of the sample for each class in the
    model, where classes are ordered as they are in ``self.classes_``.
*)

val predict_proba : x:Ndarray.t -> t -> Ndarray.t
(**
Probability estimates.

The returned estimates for all classes are ordered by the
label of classes.

For a multi_class problem, if multi_class is set to be "multinomial"
the softmax function is used to find the predicted probability of
each class.
Else use a one-vs-rest approach, i.e calculate the probability
of each class assuming it to be positive using the logistic function.
and normalize these values across all the classes.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Vector to be scored, where `n_samples` is the number of samples and
    `n_features` is the number of features.

Returns
-------
T : array-like of shape (n_samples, n_classes)
    Returns the probability of the sample for each class in the model,
    where classes are ordered as they are in ``self.classes_``.
*)

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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

val sparsify : t -> t
(**
Convert coefficient matrix to sparse format.

Converts the ``coef_`` member to a scipy.sparse matrix, which for
L1-regularized models can be much more memory- and storage-efficient
than the usual numpy.ndarray representation.

The ``intercept_`` member is not converted.

Returns
-------
self
    Fitted estimator.

Notes
-----
For non-sparse models, i.e. when there are not many zeros in ``coef_``,
this may actually *increase* memory usage, so use this method with
care. A rule of thumb is that the number of zero elements, which can
be computed with ``(coef_ == 0).sum()``, must be more than 50% for this
to provide significant benefits.

After calling this method, further fitting with the partial_fit
method (if any) will not work until you call densify.
*)


(** Attribute classes_: see constructor for documentation *)
val classes_ : t -> Ndarray.t

(** Attribute coef_: see constructor for documentation *)
val coef_ : t -> Ndarray.t

(** Attribute intercept_: see constructor for documentation *)
val intercept_ : t -> Ndarray.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module LogisticRegressionCV : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?cs:[`Int of int | `FloatList of float list] -> ?fit_intercept:bool -> ?cv:[`Int of int | `CrossValGenerator of Py.Object.t] -> ?dual:bool -> ?penalty:[`L1 | `L2 | `Elasticnet] -> ?scoring:[`String of string | `Callable of Py.Object.t] -> ?solver:[`Newton_cg | `Lbfgs | `Liblinear | `Sag | `Saga] -> ?tol:float -> ?max_iter:int -> ?class_weight:[`DictIntToFloat of (int * float) list | `Balanced] -> ?n_jobs:int -> ?verbose:int -> ?refit:bool -> ?intercept_scaling:float -> ?multi_class:[`Ovr | `Multinomial | `PyObject of Py.Object.t] -> ?random_state:[`Int of int | `RandomState of Py.Object.t] -> ?l1_ratios:Py.Object.t -> unit -> t
(**
Logistic Regression CV (aka logit, MaxEnt) classifier.

See glossary entry for :term:`cross-validation estimator`.

This class implements logistic regression using liblinear, newton-cg, sag
of lbfgs optimizer. The newton-cg, sag and lbfgs solvers support only L2
regularization with primal formulation. The liblinear solver supports both
L1 and L2 regularization, with a dual formulation only for the L2 penalty.
Elastic-Net penalty is only supported by the saga solver.

For the grid of `Cs` values and `l1_ratios` values, the best hyperparameter
is selected by the cross-validator
:class:`~sklearn.model_selection.StratifiedKFold`, but it can be changed
using the :term:`cv` parameter. The 'newton-cg', 'sag', 'saga' and 'lbfgs'
solvers can warm-start the coefficients (see :term:`Glossary<warm_start>`).

Read more in the :ref:`User Guide <logistic_regression>`.

Parameters
----------
Cs : int or list of floats, default=10
    Each of the values in Cs describes the inverse of regularization
    strength. If Cs is as an int, then a grid of Cs values are chosen
    in a logarithmic scale between 1e-4 and 1e4.
    Like in support vector machines, smaller values specify stronger
    regularization.

fit_intercept : bool, default=True
    Specifies if a constant (a.k.a. bias or intercept) should be
    added to the decision function.

cv : int or cross-validation generator, default=None
    The default cross-validation generator used is Stratified K-Folds.
    If an integer is provided, then it is the number of folds used.
    See the module :mod:`sklearn.model_selection` module for the
    list of possible cross-validation objects.

    .. versionchanged:: 0.22
        ``cv`` default value if None changed from 3-fold to 5-fold.

dual : bool, default=False
    Dual or primal formulation. Dual formulation is only implemented for
    l2 penalty with liblinear solver. Prefer dual=False when
    n_samples > n_features.

penalty : {'l1', 'l2', 'elasticnet'}, default='l2'
    Used to specify the norm used in the penalization. The 'newton-cg',
    'sag' and 'lbfgs' solvers support only l2 penalties. 'elasticnet' is
    only supported by the 'saga' solver.

scoring : str or callable, default=None
    A string (see model evaluation documentation) or
    a scorer callable object / function with signature
    ``scorer(estimator, X, y)``. For a list of scoring functions
    that can be used, look at :mod:`sklearn.metrics`. The
    default scoring option used is 'accuracy'.

solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'},             default='lbfgs'

    Algorithm to use in the optimization problem.

    - For small datasets, 'liblinear' is a good choice, whereas 'sag' and
      'saga' are faster for large ones.
    - For multiclass problems, only 'newton-cg', 'sag', 'saga' and 'lbfgs'
      handle multinomial loss; 'liblinear' is limited to one-versus-rest
      schemes.
    - 'newton-cg', 'lbfgs' and 'sag' only handle L2 penalty, whereas
      'liblinear' and 'saga' handle L1 penalty.
    - 'liblinear' might be slower in LogisticRegressionCV because it does
      not handle warm-starting.

    Note that 'sag' and 'saga' fast convergence is only guaranteed on
    features with approximately the same scale. You can preprocess the data
    with a scaler from sklearn.preprocessing.

    .. versionadded:: 0.17
       Stochastic Average Gradient descent solver.
    .. versionadded:: 0.19
       SAGA solver.

tol : float, default=1e-4
    Tolerance for stopping criteria.

max_iter : int, default=100
    Maximum number of iterations of the optimization algorithm.

class_weight : dict or 'balanced', default=None
    Weights associated with classes in the form ``{class_label: weight}``.
    If not given, all classes are supposed to have weight one.

    The "balanced" mode uses the values of y to automatically adjust
    weights inversely proportional to class frequencies in the input data
    as ``n_samples / (n_classes * np.bincount(y))``.

    Note that these weights will be multiplied with sample_weight (passed
    through the fit method) if sample_weight is specified.

    .. versionadded:: 0.17
       class_weight == 'balanced'

n_jobs : int, default=None
    Number of CPU cores used during the cross-validation loop.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

verbose : int, default=0
    For the 'liblinear', 'sag' and 'lbfgs' solvers set verbose to any
    positive number for verbosity.

refit : bool, default=True
    If set to True, the scores are averaged across all folds, and the
    coefs and the C that corresponds to the best score is taken, and a
    final refit is done using these parameters.
    Otherwise the coefs, intercepts and C that correspond to the
    best scores across folds are averaged.

intercept_scaling : float, default=1
    Useful only when the solver 'liblinear' is used
    and self.fit_intercept is set to True. In this case, x becomes
    [x, self.intercept_scaling],
    i.e. a "synthetic" feature with constant value equal to
    intercept_scaling is appended to the instance vector.
    The intercept becomes ``intercept_scaling * synthetic_feature_weight``.

    Note! the synthetic feature weight is subject to l1/l2 regularization
    as all other features.
    To lessen the effect of regularization on synthetic feature weight
    (and therefore on the intercept) intercept_scaling has to be increased.

multi_class : {'auto, 'ovr', 'multinomial'}, default='auto'
    If the option chosen is 'ovr', then a binary problem is fit for each
    label. For 'multinomial' the loss minimised is the multinomial loss fit
    across the entire probability distribution, *even when the data is
    binary*. 'multinomial' is unavailable when solver='liblinear'.
    'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
    and otherwise selects 'multinomial'.

    .. versionadded:: 0.18
       Stochastic Average Gradient descent solver for 'multinomial' case.
    .. versionchanged:: 0.22
        Default changed from 'ovr' to 'auto' in 0.22.

random_state : int, RandomState instance, default=None
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`. Used when `solver='sag'` or `solver='liblinear'`.
    Note that this only applies to the solver and not the cross-validation
    generator.

l1_ratios : list of float, default=None
    The list of Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``.
    Only used if ``penalty='elasticnet'``. A value of 0 is equivalent to
    using ``penalty='l2'``, while 1 is equivalent to using
    ``penalty='l1'``. For ``0 < l1_ratio <1``, the penalty is a combination
    of L1 and L2.

Attributes
----------
classes_ : ndarray of shape (n_classes, )
    A list of class labels known to the classifier.

coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
    Coefficient of the features in the decision function.

    `coef_` is of shape (1, n_features) when the given problem
    is binary.

intercept_ : ndarray of shape (1,) or (n_classes,)
    Intercept (a.k.a. bias) added to the decision function.

    If `fit_intercept` is set to False, the intercept is set to zero.
    `intercept_` is of shape(1,) when the problem is binary.

Cs_ : ndarray of shape (n_cs)
    Array of C i.e. inverse of regularization parameter values used
    for cross-validation.

l1_ratios_ : ndarray of shape (n_l1_ratios)
    Array of l1_ratios used for cross-validation. If no l1_ratio is used
    (i.e. penalty is not 'elasticnet'), this is set to ``[None]``

coefs_paths_ : ndarray of shape (n_folds, n_cs, n_features) or                    (n_folds, n_cs, n_features + 1)
    dict with classes as the keys, and the path of coefficients obtained
    during cross-validating across each fold and then across each Cs
    after doing an OvR for the corresponding class as values.
    If the 'multi_class' option is set to 'multinomial', then
    the coefs_paths are the coefficients corresponding to each class.
    Each dict value has shape ``(n_folds, n_cs, n_features)`` or
    ``(n_folds, n_cs, n_features + 1)`` depending on whether the
    intercept is fit or not. If ``penalty='elasticnet'``, the shape is
    ``(n_folds, n_cs, n_l1_ratios_, n_features)`` or
    ``(n_folds, n_cs, n_l1_ratios_, n_features + 1)``.

scores_ : dict
    dict with classes as the keys, and the values as the
    grid of scores obtained during cross-validating each fold, after doing
    an OvR for the corresponding class. If the 'multi_class' option
    given is 'multinomial' then the same scores are repeated across
    all classes, since this is the multinomial class. Each dict value
    has shape ``(n_folds, n_cs`` or ``(n_folds, n_cs, n_l1_ratios)`` if
    ``penalty='elasticnet'``.

C_ : ndarray of shape (n_classes,) or (n_classes - 1,)
    Array of C that maps to the best scores across every class. If refit is
    set to False, then for each class, the best C is the average of the
    C's that correspond to the best scores for each fold.
    `C_` is of shape(n_classes,) when the problem is binary.

l1_ratio_ : ndarray of shape (n_classes,) or (n_classes - 1,)
    Array of l1_ratio that maps to the best scores across every class. If
    refit is set to False, then for each class, the best l1_ratio is the
    average of the l1_ratio's that correspond to the best scores for each
    fold.  `l1_ratio_` is of shape(n_classes,) when the problem is binary.

n_iter_ : ndarray of shape (n_classes, n_folds, n_cs) or (1, n_folds, n_cs)
    Actual number of iterations for all classes, folds and Cs.
    In the binary or multinomial cases, the first dimension is equal to 1.
    If ``penalty='elasticnet'``, the shape is ``(n_classes, n_folds,
    n_cs, n_l1_ratios)`` or ``(1, n_folds, n_cs, n_l1_ratios)``.


Examples
--------
>>> from sklearn.datasets import load_iris
>>> from sklearn.linear_model import LogisticRegressionCV
>>> X, y = load_iris(return_X_y=True)
>>> clf = LogisticRegressionCV(cv=5, random_state=0).fit(X, y)
>>> clf.predict(X[:2, :])
array([0, 0])
>>> clf.predict_proba(X[:2, :]).shape
(2, 3)
>>> clf.score(X, y)
0.98...

See also
--------
LogisticRegression
*)

val decision_function : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
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

val densify : t -> t
(**
Convert coefficient matrix to dense array format.

Converts the ``coef_`` member (back) to a numpy.ndarray. This is the
default format of ``coef_`` and is required for fitting, so calling
this method is only required on models that have previously been
sparsified; otherwise, it is a no-op.

Returns
-------
self
    Fitted estimator.
*)

val fit : ?sample_weight:Ndarray.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:Ndarray.t -> t -> t
(**
Fit the model according to the given training data.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Training vector, where n_samples is the number of samples and
    n_features is the number of features.

y : array-like of shape (n_samples,)
    Target vector relative to X.

sample_weight : array-like of shape (n_samples,) default=None
    Array of weights that are assigned to individual samples.
    If not provided, then each sample is given unit weight.

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

val predict : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
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

val predict_log_proba : x:Ndarray.t -> t -> Ndarray.t
(**
Predict logarithm of probability estimates.

The returned estimates for all classes are ordered by the
label of classes.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Vector to be scored, where `n_samples` is the number of samples and
    `n_features` is the number of features.

Returns
-------
T : array-like of shape (n_samples, n_classes)
    Returns the log-probability of the sample for each class in the
    model, where classes are ordered as they are in ``self.classes_``.
*)

val predict_proba : x:Ndarray.t -> t -> Ndarray.t
(**
Probability estimates.

The returned estimates for all classes are ordered by the
label of classes.

For a multi_class problem, if multi_class is set to be "multinomial"
the softmax function is used to find the predicted probability of
each class.
Else use a one-vs-rest approach, i.e calculate the probability
of each class assuming it to be positive using the logistic function.
and normalize these values across all the classes.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Vector to be scored, where `n_samples` is the number of samples and
    `n_features` is the number of features.

Returns
-------
T : array-like of shape (n_samples, n_classes)
    Returns the probability of the sample for each class in the model,
    where classes are ordered as they are in ``self.classes_``.
*)

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
(**
Returns the score using the `scoring` option on the given
test data and labels.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Test samples.

y : array-like of shape (n_samples,)
    True labels for X.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

Returns
-------
score : float
    Score of self.predict(X) wrt. y.
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

val sparsify : t -> t
(**
Convert coefficient matrix to sparse format.

Converts the ``coef_`` member to a scipy.sparse matrix, which for
L1-regularized models can be much more memory- and storage-efficient
than the usual numpy.ndarray representation.

The ``intercept_`` member is not converted.

Returns
-------
self
    Fitted estimator.

Notes
-----
For non-sparse models, i.e. when there are not many zeros in ``coef_``,
this may actually *increase* memory usage, so use this method with
care. A rule of thumb is that the number of zero elements, which can
be computed with ``(coef_ == 0).sum()``, must be more than 50% for this
to provide significant benefits.

After calling this method, further fitting with the partial_fit
method (if any) will not work until you call densify.
*)


(** Attribute classes_: see constructor for documentation *)
val classes_ : t -> Ndarray.t

(** Attribute coef_: see constructor for documentation *)
val coef_ : t -> Ndarray.t

(** Attribute intercept_: see constructor for documentation *)
val intercept_ : t -> Ndarray.t

(** Attribute Cs_: see constructor for documentation *)
val cs_ : t -> Ndarray.t

(** Attribute l1_ratios_: see constructor for documentation *)
val l1_ratios_ : t -> Ndarray.t

(** Attribute coefs_paths_: see constructor for documentation *)
val coefs_paths_ : t -> Py.Object.t

(** Attribute scores_: see constructor for documentation *)
val scores_ : t -> Py.Object.t

(** Attribute C_: see constructor for documentation *)
val c_ : t -> Py.Object.t

(** Attribute l1_ratio_: see constructor for documentation *)
val l1_ratio_ : t -> Py.Object.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MultiTaskElasticNet : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?alpha:float -> ?l1_ratio:float -> ?fit_intercept:bool -> ?normalize:bool -> ?copy_X:bool -> ?max_iter:int -> ?tol:float -> ?warm_start:bool -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> ?selection:string -> unit -> t
(**
Multi-task ElasticNet model trained with L1/L2 mixed-norm as regularizer

The optimization objective for MultiTaskElasticNet is::

    (1 / (2 * n_samples)) * ||Y - XW||_Fro^2
    + alpha * l1_ratio * ||W||_21
    + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2

Where::

    ||W||_21 = sum_i sqrt(sum_j w_ij ^ 2)

i.e. the sum of norm of each row.

Read more in the :ref:`User Guide <multi_task_elastic_net>`.

Parameters
----------
alpha : float, optional
    Constant that multiplies the L1/L2 term. Defaults to 1.0

l1_ratio : float
    The ElasticNet mixing parameter, with 0 < l1_ratio <= 1.
    For l1_ratio = 1 the penalty is an L1/L2 penalty. For l1_ratio = 0 it
    is an L2 penalty.
    For ``0 < l1_ratio < 1``, the penalty is a combination of L1/L2 and L2.

fit_intercept : boolean
    whether to calculate the intercept for this model. If set
    to false, no intercept will be used in calculations
    (i.e. data is expected to be centered).

normalize : boolean, optional, default False
    This parameter is ignored when ``fit_intercept`` is set to False.
    If True, the regressors X will be normalized before regression by
    subtracting the mean and dividing by the l2-norm.
    If you wish to standardize, please use
    :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
    on an estimator with ``normalize=False``.

copy_X : boolean, optional, default True
    If ``True``, X will be copied; else, it may be overwritten.

max_iter : int, optional
    The maximum number of iterations

tol : float, optional
    The tolerance for the optimization: if the updates are
    smaller than ``tol``, the optimization code checks the
    dual gap for optimality and continues until it is smaller
    than ``tol``.

warm_start : bool, optional
    When set to ``True``, reuse the solution of the previous call to fit as
    initialization, otherwise, just erase the previous solution.
    See :term:`the Glossary <warm_start>`.

random_state : int, RandomState instance or None, optional, default None
    The seed of the pseudo random number generator that selects a random
    feature to update.  If int, random_state is the seed used by the random
    number generator; If RandomState instance, random_state is the random
    number generator; If None, the random number generator is the
    RandomState instance used by `np.random`. Used when ``selection`` ==
    'random'.

selection : str, default 'cyclic'
    If set to 'random', a random coefficient is updated every iteration
    rather than looping over features sequentially by default. This
    (setting to 'random') often leads to significantly faster convergence
    especially when tol is higher than 1e-4.

Attributes
----------
intercept_ : array, shape (n_tasks,)
    Independent term in decision function.

coef_ : array, shape (n_tasks, n_features)
    Parameter vector (W in the cost function formula). If a 1D y is
    passed in at fit (non multi-task usage), ``coef_`` is then a 1D array.
    Note that ``coef_`` stores the transpose of ``W``, ``W.T``.

n_iter_ : int
    number of iterations run by the coordinate descent solver to reach
    the specified tolerance.

Examples
--------
>>> from sklearn import linear_model
>>> clf = linear_model.MultiTaskElasticNet(alpha=0.1)
>>> clf.fit([[0,0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]])
MultiTaskElasticNet(alpha=0.1)
>>> print(clf.coef_)
[[0.45663524 0.45612256]
 [0.45663524 0.45612256]]
>>> print(clf.intercept_)
[0.0872422 0.0872422]

See also
--------
MultiTaskElasticNet : Multi-task L1/L2 ElasticNet with built-in
    cross-validation.
ElasticNet
MultiTaskLasso

Notes
-----
The algorithm used to fit the model is coordinate descent.

To avoid unnecessary memory duplication the X argument of the fit method
should be directly passed as a Fortran-contiguous numpy array.
*)

val fit : x:Ndarray.t -> y:Py.Object.t -> t -> t
(**
Fit MultiTaskElasticNet model with coordinate descent

Parameters
----------
X : ndarray, shape (n_samples, n_features)
    Data
y : ndarray, shape (n_samples, n_tasks)
    Target. Will be cast to X's dtype if necessary

Notes
-----

Coordinate descent is an algorithm that considers each column of
data at a time hence it will automatically convert the X input
as a Fortran-contiguous numpy array if necessary.

To avoid memory re-allocation it is advised to allocate the
initial data in memory directly using that format.
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

val predict : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Predict using the linear model.

Parameters
----------
X : array_like or sparse matrix, shape (n_samples, n_features)
    Samples.

Returns
-------
C : array, shape (n_samples,)
    Returns predicted values.
*)

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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


(** Attribute intercept_: see constructor for documentation *)
val intercept_ : t -> Ndarray.t

(** Attribute coef_: see constructor for documentation *)
val coef_ : t -> Ndarray.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> int

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MultiTaskElasticNetCV : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?l1_ratio:[`Float of float | `Ndarray of Ndarray.t] -> ?eps:float -> ?n_alphas:int -> ?alphas:Ndarray.t -> ?fit_intercept:bool -> ?normalize:bool -> ?max_iter:int -> ?tol:float -> ?cv:[`Int of int | `CrossValGenerator of Py.Object.t | `Ndarray of Ndarray.t] -> ?copy_X:bool -> ?verbose:[`Bool of bool | `Int of int] -> ?n_jobs:[`Int of int | `None] -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> ?selection:string -> unit -> t
(**
Multi-task L1/L2 ElasticNet with built-in cross-validation.

See glossary entry for :term:`cross-validation estimator`.

The optimization objective for MultiTaskElasticNet is::

    (1 / (2 * n_samples)) * ||Y - XW||^Fro_2
    + alpha * l1_ratio * ||W||_21
    + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2

Where::

    ||W||_21 = \sum_i \sqrt{\sum_j w_{ij}^2}

i.e. the sum of norm of each row.

Read more in the :ref:`User Guide <multi_task_elastic_net>`.

.. versionadded:: 0.15

Parameters
----------
l1_ratio : float or array of floats
    The ElasticNet mixing parameter, with 0 < l1_ratio <= 1.
    For l1_ratio = 1 the penalty is an L1/L2 penalty. For l1_ratio = 0 it
    is an L2 penalty.
    For ``0 < l1_ratio < 1``, the penalty is a combination of L1/L2 and L2.
    This parameter can be a list, in which case the different
    values are tested by cross-validation and the one giving the best
    prediction score is used. Note that a good choice of list of
    values for l1_ratio is often to put more values close to 1
    (i.e. Lasso) and less close to 0 (i.e. Ridge), as in ``[.1, .5, .7,
    .9, .95, .99, 1]``

eps : float, optional
    Length of the path. ``eps=1e-3`` means that
    ``alpha_min / alpha_max = 1e-3``.

n_alphas : int, optional
    Number of alphas along the regularization path

alphas : array-like, optional
    List of alphas where to compute the models.
    If not provided, set automatically.

fit_intercept : boolean
    whether to calculate the intercept for this model. If set
    to false, no intercept will be used in calculations
    (i.e. data is expected to be centered).

normalize : boolean, optional, default False
    This parameter is ignored when ``fit_intercept`` is set to False.
    If True, the regressors X will be normalized before regression by
    subtracting the mean and dividing by the l2-norm.
    If you wish to standardize, please use
    :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
    on an estimator with ``normalize=False``.

max_iter : int, optional
    The maximum number of iterations

tol : float, optional
    The tolerance for the optimization: if the updates are
    smaller than ``tol``, the optimization code checks the
    dual gap for optimality and continues until it is smaller
    than ``tol``.

cv : int, cross-validation generator or an iterable, optional
    Determines the cross-validation splitting strategy.
    Possible inputs for cv are:

    - None, to use the default 5-fold cross-validation,
    - integer, to specify the number of folds.
    - :term:`CV splitter`,
    - An iterable yielding (train, test) splits as arrays of indices.

    For integer/None inputs, :class:`KFold` is used.

    Refer :ref:`User Guide <cross_validation>` for the various
    cross-validation strategies that can be used here.

    .. versionchanged:: 0.22
        ``cv`` default value if None changed from 3-fold to 5-fold.

copy_X : boolean, optional, default True
    If ``True``, X will be copied; else, it may be overwritten.

verbose : bool or integer
    Amount of verbosity.

n_jobs : int or None, optional (default=None)
    Number of CPUs to use during the cross validation. Note that this is
    used only if multiple values for l1_ratio are given.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

random_state : int, RandomState instance or None, optional, default None
    The seed of the pseudo random number generator that selects a random
    feature to update.  If int, random_state is the seed used by the random
    number generator; If RandomState instance, random_state is the random
    number generator; If None, the random number generator is the
    RandomState instance used by `np.random`. Used when ``selection`` ==
    'random'.

selection : str, default 'cyclic'
    If set to 'random', a random coefficient is updated every iteration
    rather than looping over features sequentially by default. This
    (setting to 'random') often leads to significantly faster convergence
    especially when tol is higher than 1e-4.

Attributes
----------
intercept_ : array, shape (n_tasks,)
    Independent term in decision function.

coef_ : array, shape (n_tasks, n_features)
    Parameter vector (W in the cost function formula).
    Note that ``coef_`` stores the transpose of ``W``, ``W.T``.

alpha_ : float
    The amount of penalization chosen by cross validation

mse_path_ : array, shape (n_alphas, n_folds) or                 (n_l1_ratio, n_alphas, n_folds)
    mean square error for the test set on each fold, varying alpha

alphas_ : numpy array, shape (n_alphas,) or (n_l1_ratio, n_alphas)
    The grid of alphas used for fitting, for each l1_ratio

l1_ratio_ : float
    best l1_ratio obtained by cross-validation.

n_iter_ : int
    number of iterations run by the coordinate descent solver to reach
    the specified tolerance for the optimal alpha.

Examples
--------
>>> from sklearn import linear_model
>>> clf = linear_model.MultiTaskElasticNetCV(cv=3)
>>> clf.fit([[0,0], [1, 1], [2, 2]],
...         [[0, 0], [1, 1], [2, 2]])
MultiTaskElasticNetCV(cv=3)
>>> print(clf.coef_)
[[0.52875032 0.46958558]
 [0.52875032 0.46958558]]
>>> print(clf.intercept_)
[0.00166409 0.00166409]

See also
--------
MultiTaskElasticNet
ElasticNetCV
MultiTaskLassoCV

Notes
-----
The algorithm used to fit the model is coordinate descent.

To avoid unnecessary memory duplication the X argument of the fit method
should be directly passed as a Fortran-contiguous numpy array.
*)

val fit : x:Ndarray.t -> y:Ndarray.t -> t -> t
(**
Fit linear model with coordinate descent

Fit is on grid of alphas and best alpha estimated by cross-validation.

Parameters
----------
X : {array-like}, shape (n_samples, n_features)
    Training data. Pass directly as Fortran-contiguous data
    to avoid unnecessary memory duplication. If y is mono-output,
    X can be sparse.

y : array-like, shape (n_samples,) or (n_samples, n_targets)
    Target values
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

val predict : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Predict using the linear model.

Parameters
----------
X : array_like or sparse matrix, shape (n_samples, n_features)
    Samples.

Returns
-------
C : array, shape (n_samples,)
    Returns predicted values.
*)

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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


(** Attribute intercept_: see constructor for documentation *)
val intercept_ : t -> Ndarray.t

(** Attribute coef_: see constructor for documentation *)
val coef_ : t -> Ndarray.t

(** Attribute alpha_: see constructor for documentation *)
val alpha_ : t -> float

(** Attribute mse_path_: see constructor for documentation *)
val mse_path_ : t -> Ndarray.t

(** Attribute alphas_: see constructor for documentation *)
val alphas_ : t -> Ndarray.t

(** Attribute l1_ratio_: see constructor for documentation *)
val l1_ratio_ : t -> float

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> int

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MultiTaskLasso : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?alpha:float -> ?fit_intercept:bool -> ?normalize:bool -> ?copy_X:bool -> ?max_iter:int -> ?tol:float -> ?warm_start:bool -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> ?selection:string -> unit -> t
(**
Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer.

The optimization objective for Lasso is::

    (1 / (2 * n_samples)) * ||Y - XW||^2_Fro + alpha * ||W||_21

Where::

    ||W||_21 = \sum_i \sqrt{\sum_j w_{ij}^2}

i.e. the sum of norm of each row.

Read more in the :ref:`User Guide <multi_task_lasso>`.

Parameters
----------
alpha : float, optional
    Constant that multiplies the L1/L2 term. Defaults to 1.0

fit_intercept : boolean
    whether to calculate the intercept for this model. If set
    to false, no intercept will be used in calculations
    (i.e. data is expected to be centered).

normalize : boolean, optional, default False
    This parameter is ignored when ``fit_intercept`` is set to False.
    If True, the regressors X will be normalized before regression by
    subtracting the mean and dividing by the l2-norm.
    If you wish to standardize, please use
    :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
    on an estimator with ``normalize=False``.

copy_X : boolean, optional, default True
    If ``True``, X will be copied; else, it may be overwritten.

max_iter : int, optional
    The maximum number of iterations

tol : float, optional
    The tolerance for the optimization: if the updates are
    smaller than ``tol``, the optimization code checks the
    dual gap for optimality and continues until it is smaller
    than ``tol``.

warm_start : bool, optional
    When set to ``True``, reuse the solution of the previous call to fit as
    initialization, otherwise, just erase the previous solution.
    See :term:`the Glossary <warm_start>`.

random_state : int, RandomState instance or None, optional, default None
    The seed of the pseudo random number generator that selects a random
    feature to update.  If int, random_state is the seed used by the random
    number generator; If RandomState instance, random_state is the random
    number generator; If None, the random number generator is the
    RandomState instance used by `np.random`. Used when ``selection`` ==
    'random'.

selection : str, default 'cyclic'
    If set to 'random', a random coefficient is updated every iteration
    rather than looping over features sequentially by default. This
    (setting to 'random') often leads to significantly faster convergence
    especially when tol is higher than 1e-4

Attributes
----------
coef_ : array, shape (n_tasks, n_features)
    Parameter vector (W in the cost function formula).
    Note that ``coef_`` stores the transpose of ``W``, ``W.T``.

intercept_ : array, shape (n_tasks,)
    independent term in decision function.

n_iter_ : int
    number of iterations run by the coordinate descent solver to reach
    the specified tolerance.

Examples
--------
>>> from sklearn import linear_model
>>> clf = linear_model.MultiTaskLasso(alpha=0.1)
>>> clf.fit([[0,0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]])
MultiTaskLasso(alpha=0.1)
>>> print(clf.coef_)
[[0.89393398 0.        ]
 [0.89393398 0.        ]]
>>> print(clf.intercept_)
[0.10606602 0.10606602]

See also
--------
MultiTaskLasso : Multi-task L1/L2 Lasso with built-in cross-validation
Lasso
MultiTaskElasticNet

Notes
-----
The algorithm used to fit the model is coordinate descent.

To avoid unnecessary memory duplication the X argument of the fit method
should be directly passed as a Fortran-contiguous numpy array.
*)

val fit : x:Ndarray.t -> y:Py.Object.t -> t -> t
(**
Fit MultiTaskElasticNet model with coordinate descent

Parameters
----------
X : ndarray, shape (n_samples, n_features)
    Data
y : ndarray, shape (n_samples, n_tasks)
    Target. Will be cast to X's dtype if necessary

Notes
-----

Coordinate descent is an algorithm that considers each column of
data at a time hence it will automatically convert the X input
as a Fortran-contiguous numpy array if necessary.

To avoid memory re-allocation it is advised to allocate the
initial data in memory directly using that format.
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

val predict : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Predict using the linear model.

Parameters
----------
X : array_like or sparse matrix, shape (n_samples, n_features)
    Samples.

Returns
-------
C : array, shape (n_samples,)
    Returns predicted values.
*)

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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


(** Attribute coef_: see constructor for documentation *)
val coef_ : t -> Ndarray.t

(** Attribute intercept_: see constructor for documentation *)
val intercept_ : t -> Ndarray.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> int

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MultiTaskLassoCV : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?eps:float -> ?n_alphas:int -> ?alphas:Ndarray.t -> ?fit_intercept:bool -> ?normalize:bool -> ?max_iter:int -> ?tol:float -> ?copy_X:bool -> ?cv:[`Int of int | `CrossValGenerator of Py.Object.t | `Ndarray of Ndarray.t] -> ?verbose:[`Bool of bool | `Int of int] -> ?n_jobs:[`Int of int | `None] -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> ?selection:string -> unit -> t
(**
Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer.

See glossary entry for :term:`cross-validation estimator`.

The optimization objective for MultiTaskLasso is::

    (1 / (2 * n_samples)) * ||Y - XW||^Fro_2 + alpha * ||W||_21

Where::

    ||W||_21 = \sum_i \sqrt{\sum_j w_{ij}^2}

i.e. the sum of norm of each row.

Read more in the :ref:`User Guide <multi_task_lasso>`.

.. versionadded:: 0.15

Parameters
----------
eps : float, optional
    Length of the path. ``eps=1e-3`` means that
    ``alpha_min / alpha_max = 1e-3``.

n_alphas : int, optional
    Number of alphas along the regularization path

alphas : array-like, optional
    List of alphas where to compute the models.
    If not provided, set automatically.

fit_intercept : boolean
    whether to calculate the intercept for this model. If set
    to false, no intercept will be used in calculations
    (i.e. data is expected to be centered).

normalize : boolean, optional, default False
    This parameter is ignored when ``fit_intercept`` is set to False.
    If True, the regressors X will be normalized before regression by
    subtracting the mean and dividing by the l2-norm.
    If you wish to standardize, please use
    :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
    on an estimator with ``normalize=False``.

max_iter : int, optional
    The maximum number of iterations.

tol : float, optional
    The tolerance for the optimization: if the updates are
    smaller than ``tol``, the optimization code checks the
    dual gap for optimality and continues until it is smaller
    than ``tol``.

copy_X : boolean, optional, default True
    If ``True``, X will be copied; else, it may be overwritten.

cv : int, cross-validation generator or an iterable, optional
    Determines the cross-validation splitting strategy.
    Possible inputs for cv are:

    - None, to use the default 5-fold cross-validation,
    - integer, to specify the number of folds.
    - :term:`CV splitter`,
    - An iterable yielding (train, test) splits as arrays of indices.

    For integer/None inputs, :class:`KFold` is used.

    Refer :ref:`User Guide <cross_validation>` for the various
    cross-validation strategies that can be used here.

    .. versionchanged:: 0.22
        ``cv`` default value if None changed from 3-fold to 5-fold.

verbose : bool or integer
    Amount of verbosity.

n_jobs : int or None, optional (default=None)
    Number of CPUs to use during the cross validation. Note that this is
    used only if multiple values for l1_ratio are given.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

random_state : int, RandomState instance or None, optional, default None
    The seed of the pseudo random number generator that selects a random
    feature to update.  If int, random_state is the seed used by the random
    number generator; If RandomState instance, random_state is the random
    number generator; If None, the random number generator is the
    RandomState instance used by `np.random`. Used when ``selection`` ==
    'random'

selection : str, default 'cyclic'
    If set to 'random', a random coefficient is updated every iteration
    rather than looping over features sequentially by default. This
    (setting to 'random') often leads to significantly faster convergence
    especially when tol is higher than 1e-4.

Attributes
----------
intercept_ : array, shape (n_tasks,)
    Independent term in decision function.

coef_ : array, shape (n_tasks, n_features)
    Parameter vector (W in the cost function formula).
    Note that ``coef_`` stores the transpose of ``W``, ``W.T``.

alpha_ : float
    The amount of penalization chosen by cross validation

mse_path_ : array, shape (n_alphas, n_folds)
    mean square error for the test set on each fold, varying alpha

alphas_ : numpy array, shape (n_alphas,)
    The grid of alphas used for fitting.

n_iter_ : int
    number of iterations run by the coordinate descent solver to reach
    the specified tolerance for the optimal alpha.

Examples
--------
>>> from sklearn.linear_model import MultiTaskLassoCV
>>> from sklearn.datasets import make_regression
>>> from sklearn.metrics import r2_score
>>> X, y = make_regression(n_targets=2, noise=4, random_state=0)
>>> reg = MultiTaskLassoCV(cv=5, random_state=0).fit(X, y)
>>> r2_score(y, reg.predict(X))
0.9994...
>>> reg.alpha_
0.5713...
>>> reg.predict(X[:1,])
array([[153.7971...,  94.9015...]])

See also
--------
MultiTaskElasticNet
ElasticNetCV
MultiTaskElasticNetCV

Notes
-----
The algorithm used to fit the model is coordinate descent.

To avoid unnecessary memory duplication the X argument of the fit method
should be directly passed as a Fortran-contiguous numpy array.
*)

val fit : x:Ndarray.t -> y:Ndarray.t -> t -> t
(**
Fit linear model with coordinate descent

Fit is on grid of alphas and best alpha estimated by cross-validation.

Parameters
----------
X : {array-like}, shape (n_samples, n_features)
    Training data. Pass directly as Fortran-contiguous data
    to avoid unnecessary memory duplication. If y is mono-output,
    X can be sparse.

y : array-like, shape (n_samples,) or (n_samples, n_targets)
    Target values
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

val predict : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Predict using the linear model.

Parameters
----------
X : array_like or sparse matrix, shape (n_samples, n_features)
    Samples.

Returns
-------
C : array, shape (n_samples,)
    Returns predicted values.
*)

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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


(** Attribute intercept_: see constructor for documentation *)
val intercept_ : t -> Ndarray.t

(** Attribute coef_: see constructor for documentation *)
val coef_ : t -> Ndarray.t

(** Attribute alpha_: see constructor for documentation *)
val alpha_ : t -> float

(** Attribute mse_path_: see constructor for documentation *)
val mse_path_ : t -> Ndarray.t

(** Attribute alphas_: see constructor for documentation *)
val alphas_ : t -> Ndarray.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> int

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module OrthogonalMatchingPursuit : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_nonzero_coefs:int -> ?tol:float -> ?fit_intercept:bool -> ?normalize:bool -> ?precompute:[`Bool of bool | `Auto] -> unit -> t
(**
Orthogonal Matching Pursuit model (OMP)

Read more in the :ref:`User Guide <omp>`.

Parameters
----------
n_nonzero_coefs : int, optional
    Desired number of non-zero entries in the solution. If None (by
    default) this value is set to 10% of n_features.

tol : float, optional
    Maximum norm of the residual. If not None, overrides n_nonzero_coefs.

fit_intercept : boolean, optional
    whether to calculate the intercept for this model. If set
    to false, no intercept will be used in calculations
    (i.e. data is expected to be centered).

normalize : boolean, optional, default True
    This parameter is ignored when ``fit_intercept`` is set to False.
    If True, the regressors X will be normalized before regression by
    subtracting the mean and dividing by the l2-norm.
    If you wish to standardize, please use
    :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
    on an estimator with ``normalize=False``.

precompute : {True, False, 'auto'}, default 'auto'
    Whether to use a precomputed Gram and Xy matrix to speed up
    calculations. Improves performance when :term:`n_targets` or
    :term:`n_samples` is very large. Note that if you already have such
    matrices, you can pass them directly to the fit method.

Attributes
----------
coef_ : array, shape (n_features,) or (n_targets, n_features)
    parameter vector (w in the formula)

intercept_ : float or array, shape (n_targets,)
    independent term in decision function.

n_iter_ : int or array-like
    Number of active features across every target.

Examples
--------
>>> from sklearn.linear_model import OrthogonalMatchingPursuit
>>> from sklearn.datasets import make_regression
>>> X, y = make_regression(noise=4, random_state=0)
>>> reg = OrthogonalMatchingPursuit().fit(X, y)
>>> reg.score(X, y)
0.9991...
>>> reg.predict(X[:1,])
array([-78.3854...])

Notes
-----
Orthogonal matching pursuit was introduced in G. Mallat, Z. Zhang,
Matching pursuits with time-frequency dictionaries, IEEE Transactions on
Signal Processing, Vol. 41, No. 12. (December 1993), pp. 3397-3415.
(http://blanche.polytechnique.fr/~mallat/papiers/MallatPursuit93.pdf)

This implementation is based on Rubinstein, R., Zibulevsky, M. and Elad,
M., Efficient Implementation of the K-SVD Algorithm using Batch Orthogonal
Matching Pursuit Technical Report - CS Technion, April 2008.
https://www.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf

See also
--------
orthogonal_mp
orthogonal_mp_gram
lars_path
Lars
LassoLars
decomposition.sparse_encode
OrthogonalMatchingPursuitCV
*)

val fit : x:Ndarray.t -> y:Ndarray.t -> t -> t
(**
Fit the model using X, y as training data.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training data.

y : array-like, shape (n_samples,) or (n_samples, n_targets)
    Target values. Will be cast to X's dtype if necessary


Returns
-------
self : object
    returns an instance of self.
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

val predict : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Predict using the linear model.

Parameters
----------
X : array_like or sparse matrix, shape (n_samples, n_features)
    Samples.

Returns
-------
C : array, shape (n_samples,)
    Returns predicted values.
*)

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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


(** Attribute coef_: see constructor for documentation *)
val coef_ : t -> Ndarray.t

(** Attribute intercept_: see constructor for documentation *)
val intercept_ : t -> Ndarray.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> Py.Object.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module OrthogonalMatchingPursuitCV : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?copy:bool -> ?fit_intercept:bool -> ?normalize:bool -> ?max_iter:int -> ?cv:[`Int of int | `CrossValGenerator of Py.Object.t | `Ndarray of Ndarray.t] -> ?n_jobs:[`Int of int | `None] -> ?verbose:[`Bool of bool | `Int of int] -> unit -> t
(**
Cross-validated Orthogonal Matching Pursuit model (OMP).

See glossary entry for :term:`cross-validation estimator`.

Read more in the :ref:`User Guide <omp>`.

Parameters
----------
copy : bool, optional
    Whether the design matrix X must be copied by the algorithm. A false
    value is only helpful if X is already Fortran-ordered, otherwise a
    copy is made anyway.

fit_intercept : boolean, optional
    whether to calculate the intercept for this model. If set
    to false, no intercept will be used in calculations
    (i.e. data is expected to be centered).

normalize : boolean, optional, default True
    This parameter is ignored when ``fit_intercept`` is set to False.
    If True, the regressors X will be normalized before regression by
    subtracting the mean and dividing by the l2-norm.
    If you wish to standardize, please use
    :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
    on an estimator with ``normalize=False``.

max_iter : integer, optional
    Maximum numbers of iterations to perform, therefore maximum features
    to include. 10% of ``n_features`` but at least 5 if available.

cv : int, cross-validation generator or an iterable, optional
    Determines the cross-validation splitting strategy.
    Possible inputs for cv are:

    - None, to use the default 5-fold cross-validation,
    - integer, to specify the number of folds.
    - :term:`CV splitter`,
    - An iterable yielding (train, test) splits as arrays of indices.

    For integer/None inputs, :class:`KFold` is used.

    Refer :ref:`User Guide <cross_validation>` for the various
    cross-validation strategies that can be used here.

    .. versionchanged:: 0.22
        ``cv`` default value if None changed from 3-fold to 5-fold.

n_jobs : int or None, optional (default=None)
    Number of CPUs to use during the cross validation.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

verbose : boolean or integer, optional
    Sets the verbosity amount

Attributes
----------
intercept_ : float or array, shape (n_targets,)
    Independent term in decision function.

coef_ : array, shape (n_features,) or (n_targets, n_features)
    Parameter vector (w in the problem formulation).

n_nonzero_coefs_ : int
    Estimated number of non-zero coefficients giving the best mean squared
    error over the cross-validation folds.

n_iter_ : int or array-like
    Number of active features across every target for the model refit with
    the best hyperparameters got by cross-validating across all folds.

Examples
--------
>>> from sklearn.linear_model import OrthogonalMatchingPursuitCV
>>> from sklearn.datasets import make_regression
>>> X, y = make_regression(n_features=100, n_informative=10,
...                        noise=4, random_state=0)
>>> reg = OrthogonalMatchingPursuitCV(cv=5).fit(X, y)
>>> reg.score(X, y)
0.9991...
>>> reg.n_nonzero_coefs_
10
>>> reg.predict(X[:1,])
array([-78.3854...])

See also
--------
orthogonal_mp
orthogonal_mp_gram
lars_path
Lars
LassoLars
OrthogonalMatchingPursuit
LarsCV
LassoLarsCV
decomposition.sparse_encode
*)

val fit : x:Ndarray.t -> y:Ndarray.t -> t -> t
(**
Fit the model using X, y as training data.

Parameters
----------
X : array-like, shape [n_samples, n_features]
    Training data.

y : array-like, shape [n_samples]
    Target values. Will be cast to X's dtype if necessary

Returns
-------
self : object
    returns an instance of self.
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

val predict : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Predict using the linear model.

Parameters
----------
X : array_like or sparse matrix, shape (n_samples, n_features)
    Samples.

Returns
-------
C : array, shape (n_samples,)
    Returns predicted values.
*)

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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


(** Attribute intercept_: see constructor for documentation *)
val intercept_ : t -> Ndarray.t

(** Attribute coef_: see constructor for documentation *)
val coef_ : t -> Ndarray.t

(** Attribute n_nonzero_coefs_: see constructor for documentation *)
val n_nonzero_coefs_ : t -> int

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> Py.Object.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module PassiveAggressiveClassifier : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?c:float -> ?fit_intercept:bool -> ?max_iter:int -> ?tol:[`Float of float | `None] -> ?early_stopping:bool -> ?validation_fraction:float -> ?n_iter_no_change:int -> ?shuffle:bool -> ?verbose:int -> ?loss:string -> ?n_jobs:[`Int of int | `None] -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> ?warm_start:bool -> ?class_weight:[`DictIntToFloat of (int * float) list | `Balanced | `None | `PyObject of Py.Object.t] -> ?average:[`Bool of bool | `Int of int] -> unit -> t
(**
Passive Aggressive Classifier

Read more in the :ref:`User Guide <passive_aggressive>`.

Parameters
----------

C : float
    Maximum step size (regularization). Defaults to 1.0.

fit_intercept : bool, default=False
    Whether the intercept should be estimated or not. If False, the
    data is assumed to be already centered.

max_iter : int, optional (default=1000)
    The maximum number of passes over the training data (aka epochs).
    It only impacts the behavior in the ``fit`` method, and not the
    :meth:`partial_fit` method.

    .. versionadded:: 0.19

tol : float or None, optional (default=1e-3)
    The stopping criterion. If it is not None, the iterations will stop
    when (loss > previous_loss - tol).

    .. versionadded:: 0.19

early_stopping : bool, default=False
    Whether to use early stopping to terminate training when validation.
    score is not improving. If set to True, it will automatically set aside
    a stratified fraction of training data as validation and terminate
    training when validation score is not improving by at least tol for
    n_iter_no_change consecutive epochs.

    .. versionadded:: 0.20

validation_fraction : float, default=0.1
    The proportion of training data to set aside as validation set for
    early stopping. Must be between 0 and 1.
    Only used if early_stopping is True.

    .. versionadded:: 0.20

n_iter_no_change : int, default=5
    Number of iterations with no improvement to wait before early stopping.

    .. versionadded:: 0.20

shuffle : bool, default=True
    Whether or not the training data should be shuffled after each epoch.

verbose : integer, optional
    The verbosity level

loss : string, optional
    The loss function to be used:
    hinge: equivalent to PA-I in the reference paper.
    squared_hinge: equivalent to PA-II in the reference paper.

n_jobs : int or None, optional (default=None)
    The number of CPUs to use to do the OVA (One Versus All, for
    multi-class problems) computation.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

random_state : int, RandomState instance or None, optional, default=None
    The seed of the pseudo random number generator to use when shuffling
    the data.  If int, random_state is the seed used by the random number
    generator; If RandomState instance, random_state is the random number
    generator; If None, the random number generator is the RandomState
    instance used by `np.random`.

warm_start : bool, optional
    When set to True, reuse the solution of the previous call to fit as
    initialization, otherwise, just erase the previous solution.
    See :term:`the Glossary <warm_start>`.

    Repeatedly calling fit or partial_fit when warm_start is True can
    result in a different solution than when calling fit a single time
    because of the way the data is shuffled.

class_weight : dict, {class_label: weight} or "balanced" or None, optional
    Preset for the class_weight fit parameter.

    Weights associated with classes. If not given, all classes
    are supposed to have weight one.

    The "balanced" mode uses the values of y to automatically adjust
    weights inversely proportional to class frequencies in the input data
    as ``n_samples / (n_classes * np.bincount(y))``

    .. versionadded:: 0.17
       parameter *class_weight* to automatically weight samples.

average : bool or int, optional
    When set to True, computes the averaged SGD weights and stores the
    result in the ``coef_`` attribute. If set to an int greater than 1,
    averaging will begin once the total number of samples seen reaches
    average. So average=10 will begin averaging after seeing 10 samples.

    .. versionadded:: 0.19
       parameter *average* to use weights averaging in SGD

Attributes
----------
coef_ : array, shape = [1, n_features] if n_classes == 2 else [n_classes,            n_features]
    Weights assigned to the features.

intercept_ : array, shape = [1] if n_classes == 2 else [n_classes]
    Constants in decision function.

n_iter_ : int
    The actual number of iterations to reach the stopping criterion.
    For multiclass fits, it is the maximum over every binary fit.

classes_ : array of shape (n_classes,)
    The unique classes labels.

t_ : int
    Number of weight updates performed during training.
    Same as ``(n_iter_ * n_samples)``.

Examples
--------
>>> from sklearn.linear_model import PassiveAggressiveClassifier
>>> from sklearn.datasets import make_classification

>>> X, y = make_classification(n_features=4, random_state=0)
>>> clf = PassiveAggressiveClassifier(max_iter=1000, random_state=0,
... tol=1e-3)
>>> clf.fit(X, y)
PassiveAggressiveClassifier(random_state=0)
>>> print(clf.coef_)
[[0.26642044 0.45070924 0.67251877 0.64185414]]
>>> print(clf.intercept_)
[1.84127814]
>>> print(clf.predict([[0, 0, 0, 0]]))
[1]

See also
--------

SGDClassifier
Perceptron

References
----------
Online Passive-Aggressive Algorithms
<http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf>
K. Crammer, O. Dekel, J. Keshat, S. Shalev-Shwartz, Y. Singer - JMLR (2006)
*)

val decision_function : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
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

val densify : t -> t
(**
Convert coefficient matrix to dense array format.

Converts the ``coef_`` member (back) to a numpy.ndarray. This is the
default format of ``coef_`` and is required for fitting, so calling
this method is only required on models that have previously been
sparsified; otherwise, it is a no-op.

Returns
-------
self
    Fitted estimator.
*)

val fit : ?coef_init:Ndarray.t -> ?intercept_init:Ndarray.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:Ndarray.t -> t -> t
(**
Fit linear model with Passive Aggressive algorithm.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Training data

y : numpy array of shape [n_samples]
    Target values

coef_init : array, shape = [n_classes,n_features]
    The initial coefficients to warm-start the optimization.

intercept_init : array, shape = [n_classes]
    The initial intercept to warm-start the optimization.

Returns
-------
self : returns an instance of self.
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

val partial_fit : ?classes:Ndarray.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:Ndarray.t -> t -> t
(**
Fit linear model with Passive Aggressive algorithm.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Subset of the training data

y : numpy array of shape [n_samples]
    Subset of the target values

classes : array, shape = [n_classes]
    Classes across all calls to partial_fit.
    Can be obtained by via `np.unique(y_all)`, where y_all is the
    target vector of the entire dataset.
    This argument is required for the first call to partial_fit
    and can be omitted in the subsequent calls.
    Note that y doesn't need to contain all labels in `classes`.

Returns
-------
self : returns an instance of self.
*)

val predict : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
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

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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

val set_params : ?kwargs:(string * Py.Object.t) list -> t -> t
(**
Set and validate the parameters of estimator.

Parameters
----------
**kwargs : dict
    Estimator parameters.

Returns
-------
self : object
    Estimator instance.
*)

val sparsify : t -> t
(**
Convert coefficient matrix to sparse format.

Converts the ``coef_`` member to a scipy.sparse matrix, which for
L1-regularized models can be much more memory- and storage-efficient
than the usual numpy.ndarray representation.

The ``intercept_`` member is not converted.

Returns
-------
self
    Fitted estimator.

Notes
-----
For non-sparse models, i.e. when there are not many zeros in ``coef_``,
this may actually *increase* memory usage, so use this method with
care. A rule of thumb is that the number of zero elements, which can
be computed with ``(coef_ == 0).sum()``, must be more than 50% for this
to provide significant benefits.

After calling this method, further fitting with the partial_fit
method (if any) will not work until you call densify.
*)


(** Attribute coef_: see constructor for documentation *)
val coef_ : t -> Ndarray.t

(** Attribute intercept_: see constructor for documentation *)
val intercept_ : t -> Ndarray.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> int

(** Attribute classes_: see constructor for documentation *)
val classes_ : t -> Ndarray.t

(** Attribute t_: see constructor for documentation *)
val t_ : t -> int

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module PassiveAggressiveRegressor : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?c:float -> ?fit_intercept:bool -> ?max_iter:int -> ?tol:[`Float of float | `None] -> ?early_stopping:bool -> ?validation_fraction:float -> ?n_iter_no_change:int -> ?shuffle:bool -> ?verbose:int -> ?loss:string -> ?epsilon:float -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> ?warm_start:bool -> ?average:[`Bool of bool | `Int of int] -> unit -> t
(**
Passive Aggressive Regressor

Read more in the :ref:`User Guide <passive_aggressive>`.

Parameters
----------

C : float
    Maximum step size (regularization). Defaults to 1.0.

fit_intercept : bool
    Whether the intercept should be estimated or not. If False, the
    data is assumed to be already centered. Defaults to True.

max_iter : int, optional (default=1000)
    The maximum number of passes over the training data (aka epochs).
    It only impacts the behavior in the ``fit`` method, and not the
    :meth:`partial_fit` method.

    .. versionadded:: 0.19

tol : float or None, optional (default=1e-3)
    The stopping criterion. If it is not None, the iterations will stop
    when (loss > previous_loss - tol).

    .. versionadded:: 0.19

early_stopping : bool, default=False
    Whether to use early stopping to terminate training when validation.
    score is not improving. If set to True, it will automatically set aside
    a fraction of training data as validation and terminate
    training when validation score is not improving by at least tol for
    n_iter_no_change consecutive epochs.

    .. versionadded:: 0.20

validation_fraction : float, default=0.1
    The proportion of training data to set aside as validation set for
    early stopping. Must be between 0 and 1.
    Only used if early_stopping is True.

    .. versionadded:: 0.20

n_iter_no_change : int, default=5
    Number of iterations with no improvement to wait before early stopping.

    .. versionadded:: 0.20

shuffle : bool, default=True
    Whether or not the training data should be shuffled after each epoch.

verbose : integer, optional
    The verbosity level

loss : string, optional
    The loss function to be used:
    epsilon_insensitive: equivalent to PA-I in the reference paper.
    squared_epsilon_insensitive: equivalent to PA-II in the reference
    paper.

epsilon : float
    If the difference between the current prediction and the correct label
    is below this threshold, the model is not updated.

random_state : int, RandomState instance or None, optional, default=None
    The seed of the pseudo random number generator to use when shuffling
    the data.  If int, random_state is the seed used by the random number
    generator; If RandomState instance, random_state is the random number
    generator; If None, the random number generator is the RandomState
    instance used by `np.random`.

warm_start : bool, optional
    When set to True, reuse the solution of the previous call to fit as
    initialization, otherwise, just erase the previous solution.
    See :term:`the Glossary <warm_start>`.

    Repeatedly calling fit or partial_fit when warm_start is True can
    result in a different solution than when calling fit a single time
    because of the way the data is shuffled.

average : bool or int, optional
    When set to True, computes the averaged SGD weights and stores the
    result in the ``coef_`` attribute. If set to an int greater than 1,
    averaging will begin once the total number of samples seen reaches
    average. So average=10 will begin averaging after seeing 10 samples.

    .. versionadded:: 0.19
       parameter *average* to use weights averaging in SGD

Attributes
----------
coef_ : array, shape = [1, n_features] if n_classes == 2 else [n_classes,            n_features]
    Weights assigned to the features.

intercept_ : array, shape = [1] if n_classes == 2 else [n_classes]
    Constants in decision function.

n_iter_ : int
    The actual number of iterations to reach the stopping criterion.

t_ : int
    Number of weight updates performed during training.
    Same as ``(n_iter_ * n_samples)``.

Examples
--------
>>> from sklearn.linear_model import PassiveAggressiveRegressor
>>> from sklearn.datasets import make_regression

>>> X, y = make_regression(n_features=4, random_state=0)
>>> regr = PassiveAggressiveRegressor(max_iter=100, random_state=0,
... tol=1e-3)
>>> regr.fit(X, y)
PassiveAggressiveRegressor(max_iter=100, random_state=0)
>>> print(regr.coef_)
[20.48736655 34.18818427 67.59122734 87.94731329]
>>> print(regr.intercept_)
[-0.02306214]
>>> print(regr.predict([[0, 0, 0, 0]]))
[-0.02306214]

See also
--------

SGDRegressor

References
----------
Online Passive-Aggressive Algorithms
<http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf>
K. Crammer, O. Dekel, J. Keshat, S. Shalev-Shwartz, Y. Singer - JMLR (2006)
*)

val densify : t -> t
(**
Convert coefficient matrix to dense array format.

Converts the ``coef_`` member (back) to a numpy.ndarray. This is the
default format of ``coef_`` and is required for fitting, so calling
this method is only required on models that have previously been
sparsified; otherwise, it is a no-op.

Returns
-------
self
    Fitted estimator.
*)

val fit : ?coef_init:Ndarray.t -> ?intercept_init:Ndarray.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:Ndarray.t -> t -> t
(**
Fit linear model with Passive Aggressive algorithm.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Training data

y : numpy array of shape [n_samples]
    Target values

coef_init : array, shape = [n_features]
    The initial coefficients to warm-start the optimization.

intercept_init : array, shape = [1]
    The initial intercept to warm-start the optimization.

Returns
-------
self : returns an instance of self.
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

val partial_fit : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:Ndarray.t -> t -> t
(**
Fit linear model with Passive Aggressive algorithm.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Subset of training data

y : numpy array of shape [n_samples]
    Subset of target values

Returns
-------
self : returns an instance of self.
*)

val predict : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Predict using the linear model

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)

Returns
-------
ndarray of shape (n_samples,)
   Predicted target values per element in X.
*)

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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

val set_params : ?kwargs:(string * Py.Object.t) list -> t -> t
(**
Set and validate the parameters of estimator.

Parameters
----------
**kwargs : dict
    Estimator parameters.

Returns
-------
self : object
    Estimator instance.
*)

val sparsify : t -> t
(**
Convert coefficient matrix to sparse format.

Converts the ``coef_`` member to a scipy.sparse matrix, which for
L1-regularized models can be much more memory- and storage-efficient
than the usual numpy.ndarray representation.

The ``intercept_`` member is not converted.

Returns
-------
self
    Fitted estimator.

Notes
-----
For non-sparse models, i.e. when there are not many zeros in ``coef_``,
this may actually *increase* memory usage, so use this method with
care. A rule of thumb is that the number of zero elements, which can
be computed with ``(coef_ == 0).sum()``, must be more than 50% for this
to provide significant benefits.

After calling this method, further fitting with the partial_fit
method (if any) will not work until you call densify.
*)


(** Attribute coef_: see constructor for documentation *)
val coef_ : t -> Ndarray.t

(** Attribute intercept_: see constructor for documentation *)
val intercept_ : t -> Ndarray.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> int

(** Attribute t_: see constructor for documentation *)
val t_ : t -> int

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Perceptron : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?penalty:[`L2 | `L1 | `Elasticnet] -> ?alpha:float -> ?fit_intercept:bool -> ?max_iter:int -> ?tol:float -> ?shuffle:bool -> ?verbose:int -> ?eta0:float -> ?n_jobs:int -> ?random_state:[`Int of int | `RandomState of Py.Object.t] -> ?early_stopping:bool -> ?validation_fraction:float -> ?n_iter_no_change:int -> ?class_weight:[`DictIntToFloat of (int * float) list | `Balanced | `PyObject of Py.Object.t] -> ?warm_start:bool -> unit -> t
(**
Perceptron

Read more in the :ref:`User Guide <perceptron>`.

Parameters
----------

penalty : {'l2','l1','elasticnet'}, default=None
    The penalty (aka regularization term) to be used.

alpha : float, default=0.0001
    Constant that multiplies the regularization term if regularization is
    used.

fit_intercept : bool, default=True
    Whether the intercept should be estimated or not. If False, the
    data is assumed to be already centered.

max_iter : int, default=1000
    The maximum number of passes over the training data (aka epochs).
    It only impacts the behavior in the ``fit`` method, and not the
    :meth:`partial_fit` method.

    .. versionadded:: 0.19

tol : float, default=1e-3
    The stopping criterion. If it is not None, the iterations will stop
    when (loss > previous_loss - tol).

    .. versionadded:: 0.19

shuffle : bool, default=True
    Whether or not the training data should be shuffled after each epoch.

verbose : int, default=0
    The verbosity level

eta0 : double, default=1
    Constant by which the updates are multiplied.

n_jobs : int, default=None
    The number of CPUs to use to do the OVA (One Versus All, for
    multi-class problems) computation.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

random_state : int, RandomState instance, default=None
    The seed of the pseudo random number generator to use when shuffling
    the data.  If int, random_state is the seed used by the random number
    generator; If RandomState instance, random_state is the random number
    generator; If None, the random number generator is the RandomState
    instance used by `np.random`.

early_stopping : bool, default=False
    Whether to use early stopping to terminate training when validation.
    score is not improving. If set to True, it will automatically set aside
    a stratified fraction of training data as validation and terminate
    training when validation score is not improving by at least tol for
    n_iter_no_change consecutive epochs.

    .. versionadded:: 0.20

validation_fraction : float, default=0.1
    The proportion of training data to set aside as validation set for
    early stopping. Must be between 0 and 1.
    Only used if early_stopping is True.

    .. versionadded:: 0.20

n_iter_no_change : int, default=5
    Number of iterations with no improvement to wait before early stopping.

    .. versionadded:: 0.20

class_weight : dict, {class_label: weight} or "balanced", default=None
    Preset for the class_weight fit parameter.

    Weights associated with classes. If not given, all classes
    are supposed to have weight one.

    The "balanced" mode uses the values of y to automatically adjust
    weights inversely proportional to class frequencies in the input data
    as ``n_samples / (n_classes * np.bincount(y))``

warm_start : bool, default=False
    When set to True, reuse the solution of the previous call to fit as
    initialization, otherwise, just erase the previous solution. See
    :term:`the Glossary <warm_start>`.

Attributes
----------
coef_ : ndarray of shape = [1, n_features] if n_classes == 2 else         [n_classes, n_features]
    Weights assigned to the features.

intercept_ : ndarray of shape = [1] if n_classes == 2 else [n_classes]
    Constants in decision function.

n_iter_ : int
    The actual number of iterations to reach the stopping criterion.
    For multiclass fits, it is the maximum over every binary fit.

classes_ : ndarray of shape (n_classes,)
    The unique classes labels.

t_ : int
    Number of weight updates performed during training.
    Same as ``(n_iter_ * n_samples)``.

Notes
-----

``Perceptron`` is a classification algorithm which shares the same
underlying implementation with ``SGDClassifier``. In fact,
``Perceptron()`` is equivalent to `SGDClassifier(loss="perceptron",
eta0=1, learning_rate="constant", penalty=None)`.

Examples
--------
>>> from sklearn.datasets import load_digits
>>> from sklearn.linear_model import Perceptron
>>> X, y = load_digits(return_X_y=True)
>>> clf = Perceptron(tol=1e-3, random_state=0)
>>> clf.fit(X, y)
Perceptron()
>>> clf.score(X, y)
0.939...

See also
--------

SGDClassifier

References
----------

https://en.wikipedia.org/wiki/Perceptron and references therein.
*)

val decision_function : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
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

val densify : t -> t
(**
Convert coefficient matrix to dense array format.

Converts the ``coef_`` member (back) to a numpy.ndarray. This is the
default format of ``coef_`` and is required for fitting, so calling
this method is only required on models that have previously been
sparsified; otherwise, it is a no-op.

Returns
-------
self
    Fitted estimator.
*)

val fit : ?coef_init:Ndarray.t -> ?intercept_init:Ndarray.t -> ?sample_weight:Ndarray.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:Ndarray.t -> t -> t
(**
Fit linear model with Stochastic Gradient Descent.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Training data.

y : ndarray of shape (n_samples,)
    Target values.

coef_init : ndarray of shape (n_classes, n_features), default=None
    The initial coefficients to warm-start the optimization.

intercept_init : ndarray of shape (n_classes,), default=None
    The initial intercept to warm-start the optimization.

sample_weight : array-like, shape (n_samples,), default=None
    Weights applied to individual samples.
    If not provided, uniform weights are assumed. These weights will
    be multiplied with class_weight (passed through the
    constructor) if class_weight is specified.

Returns
-------
self :
    Returns an instance of self.
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

val partial_fit : ?classes:Ndarray.t -> ?sample_weight:Ndarray.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:Ndarray.t -> t -> t
(**
Perform one epoch of stochastic gradient descent on given samples.

Internally, this method uses ``max_iter = 1``. Therefore, it is not
guaranteed that a minimum of the cost function is reached after calling
it once. Matters such as objective convergence and early stopping
should be handled by the user.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Subset of the training data.

y : ndarray of shape (n_samples,)
    Subset of the target values.

classes : ndarray of shape (n_classes,), default=None
    Classes across all calls to partial_fit.
    Can be obtained by via `np.unique(y_all)`, where y_all is the
    target vector of the entire dataset.
    This argument is required for the first call to partial_fit
    and can be omitted in the subsequent calls.
    Note that y doesn't need to contain all labels in `classes`.

sample_weight : array-like, shape (n_samples,), default=None
    Weights applied to individual samples.
    If not provided, uniform weights are assumed.

Returns
-------
self :
    Returns an instance of self.
*)

val predict : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
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

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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

val set_params : ?kwargs:(string * Py.Object.t) list -> t -> t
(**
Set and validate the parameters of estimator.

Parameters
----------
**kwargs : dict
    Estimator parameters.

Returns
-------
self : object
    Estimator instance.
*)

val sparsify : t -> t
(**
Convert coefficient matrix to sparse format.

Converts the ``coef_`` member to a scipy.sparse matrix, which for
L1-regularized models can be much more memory- and storage-efficient
than the usual numpy.ndarray representation.

The ``intercept_`` member is not converted.

Returns
-------
self
    Fitted estimator.

Notes
-----
For non-sparse models, i.e. when there are not many zeros in ``coef_``,
this may actually *increase* memory usage, so use this method with
care. A rule of thumb is that the number of zero elements, which can
be computed with ``(coef_ == 0).sum()``, must be more than 50% for this
to provide significant benefits.

After calling this method, further fitting with the partial_fit
method (if any) will not work until you call densify.
*)


(** Attribute coef_: see constructor for documentation *)
val coef_ : t -> Ndarray.t

(** Attribute intercept_: see constructor for documentation *)
val intercept_ : t -> Ndarray.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> int

(** Attribute classes_: see constructor for documentation *)
val classes_ : t -> Ndarray.t

(** Attribute t_: see constructor for documentation *)
val t_ : t -> int

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module RANSACRegressor : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?base_estimator:Py.Object.t -> ?min_samples:[`Int of int | `Float of float] -> ?residual_threshold:float -> ?is_data_valid:Py.Object.t -> ?is_model_valid:Py.Object.t -> ?max_trials:int -> ?max_skips:int -> ?stop_n_inliers:int -> ?stop_score:float -> ?stop_probability:Py.Object.t -> ?loss:[`String of string | `Callable of Py.Object.t] -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> unit -> t
(**
RANSAC (RANdom SAmple Consensus) algorithm.

RANSAC is an iterative algorithm for the robust estimation of parameters
from a subset of inliers from the complete data set.

Read more in the :ref:`User Guide <ransac_regression>`.

Parameters
----------
base_estimator : object, optional
    Base estimator object which implements the following methods:

     * `fit(X, y)`: Fit model to given training data and target values.
     * `score(X, y)`: Returns the mean accuracy on the given test data,
       which is used for the stop criterion defined by `stop_score`.
       Additionally, the score is used to decide which of two equally
       large consensus sets is chosen as the better one.
     * `predict(X)`: Returns predicted values using the linear model,
       which is used to compute residual error using loss function.

    If `base_estimator` is None, then
    ``base_estimator=sklearn.linear_model.LinearRegression()`` is used for
    target values of dtype float.

    Note that the current implementation only supports regression
    estimators.

min_samples : int (>= 1) or float ([0, 1]), optional
    Minimum number of samples chosen randomly from original data. Treated
    as an absolute number of samples for `min_samples >= 1`, treated as a
    relative number `ceil(min_samples * X.shape[0]`) for
    `min_samples < 1`. This is typically chosen as the minimal number of
    samples necessary to estimate the given `base_estimator`. By default a
    ``sklearn.linear_model.LinearRegression()`` estimator is assumed and
    `min_samples` is chosen as ``X.shape[1] + 1``.

residual_threshold : float, optional
    Maximum residual for a data sample to be classified as an inlier.
    By default the threshold is chosen as the MAD (median absolute
    deviation) of the target values `y`.

is_data_valid : callable, optional
    This function is called with the randomly selected data before the
    model is fitted to it: `is_data_valid(X, y)`. If its return value is
    False the current randomly chosen sub-sample is skipped.

is_model_valid : callable, optional
    This function is called with the estimated model and the randomly
    selected data: `is_model_valid(model, X, y)`. If its return value is
    False the current randomly chosen sub-sample is skipped.
    Rejecting samples with this function is computationally costlier than
    with `is_data_valid`. `is_model_valid` should therefore only be used if
    the estimated model is needed for making the rejection decision.

max_trials : int, optional
    Maximum number of iterations for random sample selection.

max_skips : int, optional
    Maximum number of iterations that can be skipped due to finding zero
    inliers or invalid data defined by ``is_data_valid`` or invalid models
    defined by ``is_model_valid``.

    .. versionadded:: 0.19

stop_n_inliers : int, optional
    Stop iteration if at least this number of inliers are found.

stop_score : float, optional
    Stop iteration if score is greater equal than this threshold.

stop_probability : float in range [0, 1], optional
    RANSAC iteration stops if at least one outlier-free set of the training
    data is sampled in RANSAC. This requires to generate at least N
    samples (iterations)::

        N >= log(1 - probability) / log(1 - e**m)

    where the probability (confidence) is typically set to high value such
    as 0.99 (the default) and e is the current fraction of inliers w.r.t.
    the total number of samples.

loss : string, callable, optional, default "absolute_loss"
    String inputs, "absolute_loss" and "squared_loss" are supported which
    find the absolute loss and squared loss per sample
    respectively.

    If ``loss`` is a callable, then it should be a function that takes
    two arrays as inputs, the true and predicted value and returns a 1-D
    array with the i-th value of the array corresponding to the loss
    on ``X[i]``.

    If the loss on a sample is greater than the ``residual_threshold``,
    then this sample is classified as an outlier.

random_state : int, RandomState instance or None, optional, default None
    The generator used to initialize the centers.  If int, random_state is
    the seed used by the random number generator; If RandomState instance,
    random_state is the random number generator; If None, the random number
    generator is the RandomState instance used by `np.random`.

Attributes
----------
estimator_ : object
    Best fitted model (copy of the `base_estimator` object).

n_trials_ : int
    Number of random selection trials until one of the stop criteria is
    met. It is always ``<= max_trials``.

inlier_mask_ : bool array of shape [n_samples]
    Boolean mask of inliers classified as ``True``.

n_skips_no_inliers_ : int
    Number of iterations skipped due to finding zero inliers.

    .. versionadded:: 0.19

n_skips_invalid_data_ : int
    Number of iterations skipped due to invalid data defined by
    ``is_data_valid``.

    .. versionadded:: 0.19

n_skips_invalid_model_ : int
    Number of iterations skipped due to an invalid model defined by
    ``is_model_valid``.

    .. versionadded:: 0.19

Examples
--------
>>> from sklearn.linear_model import RANSACRegressor
>>> from sklearn.datasets import make_regression
>>> X, y = make_regression(
...     n_samples=200, n_features=2, noise=4.0, random_state=0)
>>> reg = RANSACRegressor(random_state=0).fit(X, y)
>>> reg.score(X, y)
0.9885...
>>> reg.predict(X[:1,])
array([-31.9417...])

References
----------
.. [1] https://en.wikipedia.org/wiki/RANSAC
.. [2] https://www.sri.com/sites/default/files/publications/ransac-publication.pdf
.. [3] http://www.bmva.org/bmvc/2009/Papers/Paper355/Paper355.pdf
*)

val fit : ?sample_weight:Ndarray.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:Ndarray.t -> t -> t
(**
Fit estimator using RANSAC algorithm.

Parameters
----------
X : array-like or sparse matrix, shape [n_samples, n_features]
    Training data.

y : array-like of shape (n_samples,) or (n_samples, n_targets)
    Target values.

sample_weight : array-like of shape (n_samples,), default=None
    Individual weights for each sample
    raises error if sample_weight is passed and base_estimator
    fit method does not support it.

Raises
------
ValueError
    If no valid consensus set could be found. This occurs if
    `is_data_valid` and `is_model_valid` return False for all
    `max_trials` randomly chosen sub-samples.
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

val predict : x:Ndarray.t -> t -> Ndarray.t
(**
Predict using the estimated model.

This is a wrapper for `estimator_.predict(X)`.

Parameters
----------
X : numpy array of shape [n_samples, n_features]

Returns
-------
y : array, shape = [n_samples] or [n_samples, n_targets]
    Returns predicted values.
*)

val score : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:Ndarray.t -> t -> float
(**
Returns the score of the prediction.

This is a wrapper for `estimator_.score(X, y)`.

Parameters
----------
X : numpy array or sparse matrix of shape [n_samples, n_features]
    Training data.

y : array, shape = [n_samples] or [n_samples, n_targets]
    Target values.

Returns
-------
z : float
    Score of the prediction.
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


(** Attribute estimator_: see constructor for documentation *)
val estimator_ : t -> Py.Object.t

(** Attribute n_trials_: see constructor for documentation *)
val n_trials_ : t -> int

(** Attribute inlier_mask_: see constructor for documentation *)
val inlier_mask_ : t -> Py.Object.t

(** Attribute n_skips_no_inliers_: see constructor for documentation *)
val n_skips_no_inliers_ : t -> int

(** Attribute n_skips_invalid_data_: see constructor for documentation *)
val n_skips_invalid_data_ : t -> int

(** Attribute n_skips_invalid_model_: see constructor for documentation *)
val n_skips_invalid_model_ : t -> int

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Ridge : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?alpha:[`Float of float | `Ndarray of Ndarray.t] -> ?fit_intercept:bool -> ?normalize:bool -> ?copy_X:bool -> ?max_iter:int -> ?tol:float -> ?solver:[`Auto | `Svd | `Cholesky | `Lsqr | `Sparse_cg | `Sag | `Saga] -> ?random_state:[`Int of int | `RandomState of Py.Object.t] -> unit -> t
(**
Linear least squares with l2 regularization.

Minimizes the objective function::

||y - Xw||^2_2 + alpha * ||w||^2_2

This model solves a regression model where the loss function is
the linear least squares function and regularization is given by
the l2-norm. Also known as Ridge Regression or Tikhonov regularization.
This estimator has built-in support for multi-variate regression
(i.e., when y is a 2d-array of shape (n_samples, n_targets)).

Read more in the :ref:`User Guide <ridge_regression>`.

Parameters
----------
alpha : {float, ndarray of shape (n_targets,)}, default=1.0
    Regularization strength; must be a positive float. Regularization
    improves the conditioning of the problem and reduces the variance of
    the estimates. Larger values specify stronger regularization.
    Alpha corresponds to ``C^-1`` in other linear models such as
    LogisticRegression or LinearSVC. If an array is passed, penalties are
    assumed to be specific to the targets. Hence they must correspond in
    number.

fit_intercept : bool, default=True
    Whether to calculate the intercept for this model. If set
    to false, no intercept will be used in calculations
    (i.e. data is expected to be centered).

normalize : bool, default=False
    This parameter is ignored when ``fit_intercept`` is set to False.
    If True, the regressors X will be normalized before regression by
    subtracting the mean and dividing by the l2-norm.
    If you wish to standardize, please use
    :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
    on an estimator with ``normalize=False``.

copy_X : bool, default=True
    If True, X will be copied; else, it may be overwritten.

max_iter : int, default=None
    Maximum number of iterations for conjugate gradient solver.
    For 'sparse_cg' and 'lsqr' solvers, the default value is determined
    by scipy.sparse.linalg. For 'sag' solver, the default value is 1000.

tol : float, default=1e-3
    Precision of the solution.

solver : {'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'},         default='auto'
    Solver to use in the computational routines:

    - 'auto' chooses the solver automatically based on the type of data.

    - 'svd' uses a Singular Value Decomposition of X to compute the Ridge
      coefficients. More stable for singular matrices than
      'cholesky'.

    - 'cholesky' uses the standard scipy.linalg.solve function to
      obtain a closed-form solution.

    - 'sparse_cg' uses the conjugate gradient solver as found in
      scipy.sparse.linalg.cg. As an iterative algorithm, this solver is
      more appropriate than 'cholesky' for large-scale data
      (possibility to set `tol` and `max_iter`).

    - 'lsqr' uses the dedicated regularized least-squares routine
      scipy.sparse.linalg.lsqr. It is the fastest and uses an iterative
      procedure.

    - 'sag' uses a Stochastic Average Gradient descent, and 'saga' uses
      its improved, unbiased version named SAGA. Both methods also use an
      iterative procedure, and are often faster than other solvers when
      both n_samples and n_features are large. Note that 'sag' and
      'saga' fast convergence is only guaranteed on features with
      approximately the same scale. You can preprocess the data with a
      scaler from sklearn.preprocessing.

    All last five solvers support both dense and sparse data. However, only
    'sparse_cg' supports sparse input when `fit_intercept` is True.

    .. versionadded:: 0.17
       Stochastic Average Gradient descent solver.
    .. versionadded:: 0.19
       SAGA solver.

random_state : int, RandomState instance, default=None
    The seed of the pseudo random number generator to use when shuffling
    the data.  If int, random_state is the seed used by the random number
    generator; If RandomState instance, random_state is the random number
    generator; If None, the random number generator is the RandomState
    instance used by `np.random`. Used when ``solver`` == 'sag'.

    .. versionadded:: 0.17
       *random_state* to support Stochastic Average Gradient.

Attributes
----------
coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
    Weight vector(s).

intercept_ : float or ndarray of shape (n_targets,)
    Independent term in decision function. Set to 0.0 if
    ``fit_intercept = False``.

n_iter_ : None or ndarray of shape (n_targets,)
    Actual number of iterations for each target. Available only for
    sag and lsqr solvers. Other solvers will return None.

    .. versionadded:: 0.17

See also
--------
RidgeClassifier : Ridge classifier
RidgeCV : Ridge regression with built-in cross validation
:class:`sklearn.kernel_ridge.KernelRidge` : Kernel ridge regression
    combines ridge regression with the kernel trick

Examples
--------
>>> from sklearn.linear_model import Ridge
>>> import numpy as np
>>> n_samples, n_features = 10, 5
>>> rng = np.random.RandomState(0)
>>> y = rng.randn(n_samples)
>>> X = rng.randn(n_samples, n_features)
>>> clf = Ridge(alpha=1.0)
>>> clf.fit(X, y)
Ridge()
*)

val fit : ?sample_weight:[`Float of float | `Ndarray of Ndarray.t] -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:Ndarray.t -> t -> t
(**
Fit Ridge regression model.

Parameters
----------
X : {ndarray, sparse matrix} of shape (n_samples, n_features)
    Training data

y : ndarray of shape (n_samples,) or (n_samples, n_targets)
    Target values

sample_weight : float or ndarray of shape (n_samples,), default=None
    Individual weights for each sample. If given a float, every sample
    will have the same weight.

Returns
-------
self : returns an instance of self.
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

val predict : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Predict using the linear model.

Parameters
----------
X : array_like or sparse matrix, shape (n_samples, n_features)
    Samples.

Returns
-------
C : array, shape (n_samples,)
    Returns predicted values.
*)

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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


(** Attribute coef_: see constructor for documentation *)
val coef_ : t -> Ndarray.t

(** Attribute intercept_: see constructor for documentation *)
val intercept_ : t -> Ndarray.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> Py.Object.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module RidgeCV : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?alphas:Ndarray.t -> ?fit_intercept:bool -> ?normalize:bool -> ?scoring:[`String of string | `Callable of Py.Object.t] -> ?cv:[`Int of int | `CrossValGenerator of Py.Object.t | `Ndarray of Ndarray.t] -> ?gcv_mode:[`Auto | `Svd | `Eigen] -> ?store_cv_values:bool -> unit -> t
(**
Ridge regression with built-in cross-validation.

See glossary entry for :term:`cross-validation estimator`.

By default, it performs Generalized Cross-Validation, which is a form of
efficient Leave-One-Out cross-validation.

Read more in the :ref:`User Guide <ridge_regression>`.

Parameters
----------
alphas : ndarray of shape (n_alphas,), default=(0.1, 1.0, 10.0)
    Array of alpha values to try.
    Regularization strength; must be a positive float. Regularization
    improves the conditioning of the problem and reduces the variance of
    the estimates. Larger values specify stronger regularization.
    Alpha corresponds to ``C^-1`` in other linear models such as
    LogisticRegression or LinearSVC.
    If using generalized cross-validation, alphas must be positive.

fit_intercept : bool, default=True
    Whether to calculate the intercept for this model. If set
    to false, no intercept will be used in calculations
    (i.e. data is expected to be centered).

normalize : bool, default=False
    This parameter is ignored when ``fit_intercept`` is set to False.
    If True, the regressors X will be normalized before regression by
    subtracting the mean and dividing by the l2-norm.
    If you wish to standardize, please use
    :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
    on an estimator with ``normalize=False``.

scoring : string, callable, default=None
    A string (see model evaluation documentation) or
    a scorer callable object / function with signature
    ``scorer(estimator, X, y)``.
    If None, the negative mean squared error if cv is 'auto' or None
    (i.e. when using generalized cross-validation), and r2 score otherwise.

cv : int, cross-validation generator or an iterable, default=None
    Determines the cross-validation splitting strategy.
    Possible inputs for cv are:

    - None, to use the efficient Leave-One-Out cross-validation
      (also known as Generalized Cross-Validation).
    - integer, to specify the number of folds.
    - :term:`CV splitter`,
    - An iterable yielding (train, test) splits as arrays of indices.

    For integer/None inputs, if ``y`` is binary or multiclass,
    :class:`sklearn.model_selection.StratifiedKFold` is used, else,
    :class:`sklearn.model_selection.KFold` is used.

    Refer :ref:`User Guide <cross_validation>` for the various
    cross-validation strategies that can be used here.

gcv_mode : {'auto', 'svd', eigen'}, default='auto'
    Flag indicating which strategy to use when performing
    Generalized Cross-Validation. Options are::

        'auto' : use 'svd' if n_samples > n_features, otherwise use 'eigen'
        'svd' : force use of singular value decomposition of X when X is
            dense, eigenvalue decomposition of X^T.X when X is sparse.
        'eigen' : force computation via eigendecomposition of X.X^T

    The 'auto' mode is the default and is intended to pick the cheaper
    option of the two depending on the shape of the training data.

store_cv_values : bool, default=False
    Flag indicating if the cross-validation values corresponding to
    each alpha should be stored in the ``cv_values_`` attribute (see
    below). This flag is only compatible with ``cv=None`` (i.e. using
    Generalized Cross-Validation).

Attributes
----------
cv_values_ : ndarray of shape (n_samples, n_alphas) or         shape (n_samples, n_targets, n_alphas), optional
    Cross-validation values for each alpha (if ``store_cv_values=True``        and ``cv=None``). After ``fit()`` has been called, this attribute         will contain the mean squared errors (by default) or the values         of the ``{loss,score}_func`` function (if provided in the constructor).

coef_ : ndarray of shape (n_features) or (n_targets, n_features)
    Weight vector(s).

intercept_ : float or ndarray of shape (n_targets,)
    Independent term in decision function. Set to 0.0 if
    ``fit_intercept = False``.

alpha_ : float
    Estimated regularization parameter.

Examples
--------
>>> from sklearn.datasets import load_diabetes
>>> from sklearn.linear_model import RidgeCV
>>> X, y = load_diabetes(return_X_y=True)
>>> clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X, y)
>>> clf.score(X, y)
0.5166...

See also
--------
Ridge : Ridge regression
RidgeClassifier : Ridge classifier
RidgeClassifierCV : Ridge classifier with built-in cross validation
*)

val fit : ?sample_weight:[`Float of float | `Ndarray of Ndarray.t] -> x:Ndarray.t -> y:Ndarray.t -> t -> t
(**
Fit Ridge regression model with cv.

Parameters
----------
X : ndarray of shape (n_samples, n_features)
    Training data. If using GCV, will be cast to float64
    if necessary.

y : ndarray of shape (n_samples,) or (n_samples, n_targets)
    Target values. Will be cast to X's dtype if necessary.

sample_weight : float or ndarray of shape (n_samples,), default=None
    Individual weights for each sample. If given a float, every sample
    will have the same weight.

Returns
-------
self : object

Notes
-----
When sample_weight is provided, the selected hyperparameter may depend
on whether we use generalized cross-validation (cv=None or cv='auto')
or another form of cross-validation, because only generalized
cross-validation takes the sample weights into account when computing
the validation score.
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

val predict : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Predict using the linear model.

Parameters
----------
X : array_like or sparse matrix, shape (n_samples, n_features)
    Samples.

Returns
-------
C : array, shape (n_samples,)
    Returns predicted values.
*)

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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


(** Attribute cv_values_: see constructor for documentation *)
val cv_values_ : t -> Py.Object.t

(** Attribute coef_: see constructor for documentation *)
val coef_ : t -> Ndarray.t

(** Attribute intercept_: see constructor for documentation *)
val intercept_ : t -> Ndarray.t

(** Attribute alpha_: see constructor for documentation *)
val alpha_ : t -> float

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module RidgeClassifier : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?alpha:float -> ?fit_intercept:bool -> ?normalize:bool -> ?copy_X:bool -> ?max_iter:int -> ?tol:float -> ?class_weight:[`DictIntToFloat of (int * float) list | `Balanced] -> ?solver:[`Auto | `Svd | `Cholesky | `Lsqr | `Sparse_cg | `Sag | `Saga] -> ?random_state:[`Int of int | `RandomState of Py.Object.t] -> unit -> t
(**
Classifier using Ridge regression.

This classifier first converts the target values into ``{-1, 1}`` and
then treats the problem as a regression task (multi-output regression in
the multiclass case).

Read more in the :ref:`User Guide <ridge_regression>`.

Parameters
----------
alpha : float, default=1.0
    Regularization strength; must be a positive float. Regularization
    improves the conditioning of the problem and reduces the variance of
    the estimates. Larger values specify stronger regularization.
    Alpha corresponds to ``C^-1`` in other linear models such as
    LogisticRegression or LinearSVC.

fit_intercept : bool, default=True
    Whether to calculate the intercept for this model. If set to false, no
    intercept will be used in calculations (e.g. data is expected to be
    already centered).

normalize : bool, default=False
    This parameter is ignored when ``fit_intercept`` is set to False.
    If True, the regressors X will be normalized before regression by
    subtracting the mean and dividing by the l2-norm.
    If you wish to standardize, please use
    :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
    on an estimator with ``normalize=False``.

copy_X : bool, default=True
    If True, X will be copied; else, it may be overwritten.

max_iter : int, default=None
    Maximum number of iterations for conjugate gradient solver.
    The default value is determined by scipy.sparse.linalg.

tol : float, default=1e-3
    Precision of the solution.

class_weight : dict or 'balanced', default=None
    Weights associated with classes in the form ``{class_label: weight}``.
    If not given, all classes are supposed to have weight one.

    The "balanced" mode uses the values of y to automatically adjust
    weights inversely proportional to class frequencies in the input data
    as ``n_samples / (n_classes * np.bincount(y))``.

solver : {'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'},         default='auto'
    Solver to use in the computational routines:

    - 'auto' chooses the solver automatically based on the type of data.

    - 'svd' uses a Singular Value Decomposition of X to compute the Ridge
      coefficients. More stable for singular matrices than
      'cholesky'.

    - 'cholesky' uses the standard scipy.linalg.solve function to
      obtain a closed-form solution.

    - 'sparse_cg' uses the conjugate gradient solver as found in
      scipy.sparse.linalg.cg. As an iterative algorithm, this solver is
      more appropriate than 'cholesky' for large-scale data
      (possibility to set `tol` and `max_iter`).

    - 'lsqr' uses the dedicated regularized least-squares routine
      scipy.sparse.linalg.lsqr. It is the fastest and uses an iterative
      procedure.

    - 'sag' uses a Stochastic Average Gradient descent, and 'saga' uses
      its unbiased and more flexible version named SAGA. Both methods
      use an iterative procedure, and are often faster than other solvers
      when both n_samples and n_features are large. Note that 'sag' and
      'saga' fast convergence is only guaranteed on features with
      approximately the same scale. You can preprocess the data with a
      scaler from sklearn.preprocessing.

      .. versionadded:: 0.17
         Stochastic Average Gradient descent solver.
      .. versionadded:: 0.19
       SAGA solver.

random_state : int, RandomState instance, default=None
    The seed of the pseudo random number generator to use when shuffling
    the data.  If int, random_state is the seed used by the random number
    generator; If RandomState instance, random_state is the random number
    generator; If None, the random number generator is the RandomState
    instance used by `np.random`. Used when ``solver`` == 'sag'.

Attributes
----------
coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
    Coefficient of the features in the decision function.

    ``coef_`` is of shape (1, n_features) when the given problem is binary.

intercept_ : float or ndarray of shape (n_targets,)
    Independent term in decision function. Set to 0.0 if
    ``fit_intercept = False``.

n_iter_ : None or ndarray of shape (n_targets,)
    Actual number of iterations for each target. Available only for
    sag and lsqr solvers. Other solvers will return None.

classes_ : ndarray of shape (n_classes,)
    The classes labels.

See Also
--------
Ridge : Ridge regression.
RidgeClassifierCV :  Ridge classifier with built-in cross validation.

Notes
-----
For multi-class classification, n_class classifiers are trained in
a one-versus-all approach. Concretely, this is implemented by taking
advantage of the multi-variate response support in Ridge.

Examples
--------
>>> from sklearn.datasets import load_breast_cancer
>>> from sklearn.linear_model import RidgeClassifier
>>> X, y = load_breast_cancer(return_X_y=True)
>>> clf = RidgeClassifier().fit(X, y)
>>> clf.score(X, y)
0.9595...
*)

val decision_function : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
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

val fit : ?sample_weight:[`Float of float | `Ndarray of Ndarray.t] -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:Ndarray.t -> t -> t
(**
Fit Ridge classifier model.

Parameters
----------
X : {ndarray, sparse matrix} of shape (n_samples, n_features)
    Training data.

y : ndarray of shape (n_samples,)
    Target values.

sample_weight : float or ndarray of shape (n_samples,), default=None
    Individual weights for each sample. If given a float, every sample
    will have the same weight.

    .. versionadded:: 0.17
       *sample_weight* support to Classifier.

Returns
-------
self : object
    Instance of the estimator.
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

val predict : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
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

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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


(** Attribute coef_: see constructor for documentation *)
val coef_ : t -> Ndarray.t

(** Attribute intercept_: see constructor for documentation *)
val intercept_ : t -> Ndarray.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> Py.Object.t

(** Attribute classes_: see constructor for documentation *)
val classes_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module RidgeClassifierCV : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?alphas:Ndarray.t -> ?fit_intercept:bool -> ?normalize:bool -> ?scoring:[`String of string | `Callable of Py.Object.t] -> ?cv:[`Int of int | `CrossValGenerator of Py.Object.t | `Ndarray of Ndarray.t] -> ?class_weight:[`DictIntToFloat of (int * float) list | `Balanced] -> ?store_cv_values:bool -> unit -> t
(**
Ridge classifier with built-in cross-validation.

See glossary entry for :term:`cross-validation estimator`.

By default, it performs Generalized Cross-Validation, which is a form of
efficient Leave-One-Out cross-validation. Currently, only the n_features >
n_samples case is handled efficiently.

Read more in the :ref:`User Guide <ridge_regression>`.

Parameters
----------
alphas : ndarray of shape (n_alphas,), default=(0.1, 1.0, 10.0)
    Array of alpha values to try.
    Regularization strength; must be a positive float. Regularization
    improves the conditioning of the problem and reduces the variance of
    the estimates. Larger values specify stronger regularization.
    Alpha corresponds to ``C^-1`` in other linear models such as
    LogisticRegression or LinearSVC.

fit_intercept : bool, default=True
    Whether to calculate the intercept for this model. If set
    to false, no intercept will be used in calculations
    (i.e. data is expected to be centered).

normalize : bool, default=False
    This parameter is ignored when ``fit_intercept`` is set to False.
    If True, the regressors X will be normalized before regression by
    subtracting the mean and dividing by the l2-norm.
    If you wish to standardize, please use
    :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
    on an estimator with ``normalize=False``.

scoring : string, callable, default=None
    A string (see model evaluation documentation) or
    a scorer callable object / function with signature
    ``scorer(estimator, X, y)``.

cv : int, cross-validation generator or an iterable, default=None
    Determines the cross-validation splitting strategy.
    Possible inputs for cv are:

    - None, to use the efficient Leave-One-Out cross-validation
    - integer, to specify the number of folds.
    - :term:`CV splitter`,
    - An iterable yielding (train, test) splits as arrays of indices.

    Refer :ref:`User Guide <cross_validation>` for the various
    cross-validation strategies that can be used here.

class_weight : dict or 'balanced', default=None
    Weights associated with classes in the form ``{class_label: weight}``.
    If not given, all classes are supposed to have weight one.

    The "balanced" mode uses the values of y to automatically adjust
    weights inversely proportional to class frequencies in the input data
    as ``n_samples / (n_classes * np.bincount(y))``

store_cv_values : bool, default=False
    Flag indicating if the cross-validation values corresponding to
    each alpha should be stored in the ``cv_values_`` attribute (see
    below). This flag is only compatible with ``cv=None`` (i.e. using
    Generalized Cross-Validation).

Attributes
----------
cv_values_ : ndarray of shape (n_samples, n_targets, n_alphas), optional
    Cross-validation values for each alpha (if ``store_cv_values=True`` and
    ``cv=None``). After ``fit()`` has been called, this attribute will
    contain the mean squared errors (by default) or the values of the
    ``{loss,score}_func`` function (if provided in the constructor). This
    attribute exists only when ``store_cv_values`` is True.

coef_ : ndarray of shape (1, n_features) or (n_targets, n_features)
    Coefficient of the features in the decision function.

    ``coef_`` is of shape (1, n_features) when the given problem is binary.

intercept_ : float or ndarray of shape (n_targets,)
    Independent term in decision function. Set to 0.0 if
    ``fit_intercept = False``.

alpha_ : float
    Estimated regularization parameter

classes_ : ndarray of shape (n_classes,)
    The classes labels.

Examples
--------
>>> from sklearn.datasets import load_breast_cancer
>>> from sklearn.linear_model import RidgeClassifierCV
>>> X, y = load_breast_cancer(return_X_y=True)
>>> clf = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X, y)
>>> clf.score(X, y)
0.9630...

See also
--------
Ridge : Ridge regression
RidgeClassifier : Ridge classifier
RidgeCV : Ridge regression with built-in cross validation

Notes
-----
For multi-class classification, n_class classifiers are trained in
a one-versus-all approach. Concretely, this is implemented by taking
advantage of the multi-variate response support in Ridge.
*)

val decision_function : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
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

val fit : ?sample_weight:[`Float of float | `Ndarray of Ndarray.t] -> x:Ndarray.t -> y:Ndarray.t -> t -> t
(**
Fit Ridge classifier with cv.

Parameters
----------
X : ndarray of shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples
    and n_features is the number of features. When using GCV,
    will be cast to float64 if necessary.

y : ndarray of shape (n_samples,)
    Target values. Will be cast to X's dtype if necessary.

sample_weight : float or ndarray of shape (n_samples,), default=None
    Individual weights for each sample. If given a float, every sample
    will have the same weight.

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

val predict : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
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

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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


(** Attribute cv_values_: see constructor for documentation *)
val cv_values_ : t -> Ndarray.t

(** Attribute coef_: see constructor for documentation *)
val coef_ : t -> Ndarray.t

(** Attribute intercept_: see constructor for documentation *)
val intercept_ : t -> Ndarray.t

(** Attribute alpha_: see constructor for documentation *)
val alpha_ : t -> float

(** Attribute classes_: see constructor for documentation *)
val classes_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module SGDClassifier : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?loss:string -> ?penalty:[`L2 | `L1 | `Elasticnet] -> ?alpha:float -> ?l1_ratio:float -> ?fit_intercept:bool -> ?max_iter:int -> ?tol:float -> ?shuffle:bool -> ?verbose:int -> ?epsilon:float -> ?n_jobs:int -> ?random_state:[`Int of int | `RandomState of Py.Object.t] -> ?learning_rate:string -> ?eta0:float -> ?power_t:float -> ?early_stopping:bool -> ?validation_fraction:float -> ?n_iter_no_change:int -> ?class_weight:[`DictIntToFloat of (int * float) list | `Balanced | `PyObject of Py.Object.t] -> ?warm_start:bool -> ?average:[`Bool of bool | `Int of int] -> unit -> t
(**
Linear classifiers (SVM, logistic regression, a.o.) with SGD training.

This estimator implements regularized linear models with stochastic
gradient descent (SGD) learning: the gradient of the loss is estimated
each sample at a time and the model is updated along the way with a
decreasing strength schedule (aka learning rate). SGD allows minibatch
(online/out-of-core) learning, see the partial_fit method.
For best results using the default learning rate schedule, the data should
have zero mean and unit variance.

This implementation works with data represented as dense or sparse arrays
of floating point values for the features. The model it fits can be
controlled with the loss parameter; by default, it fits a linear support
vector machine (SVM).

The regularizer is a penalty added to the loss function that shrinks model
parameters towards the zero vector using either the squared euclidean norm
L2 or the absolute norm L1 or a combination of both (Elastic Net). If the
parameter update crosses the 0.0 value because of the regularizer, the
update is truncated to 0.0 to allow for learning sparse models and achieve
online feature selection.

Read more in the :ref:`User Guide <sgd>`.

Parameters
----------
loss : str, default='hinge'
    The loss function to be used. Defaults to 'hinge', which gives a
    linear SVM.

    The possible options are 'hinge', 'log', 'modified_huber',
    'squared_hinge', 'perceptron', or a regression loss: 'squared_loss',
    'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'.

    The 'log' loss gives logistic regression, a probabilistic classifier.
    'modified_huber' is another smooth loss that brings tolerance to
    outliers as well as probability estimates.
    'squared_hinge' is like hinge but is quadratically penalized.
    'perceptron' is the linear loss used by the perceptron algorithm.
    The other losses are designed for regression but can be useful in
    classification as well; see SGDRegressor for a description.

penalty : {'l2', 'l1', 'elasticnet'}, default='l2'
    The penalty (aka regularization term) to be used. Defaults to 'l2'
    which is the standard regularizer for linear SVM models. 'l1' and
    'elasticnet' might bring sparsity to the model (feature selection)
    not achievable with 'l2'.

alpha : float, default=0.0001
    Constant that multiplies the regularization term. Defaults to 0.0001.
    Also used to compute learning_rate when set to 'optimal'.

l1_ratio : float, default=0.15
    The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.
    l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.
    Defaults to 0.15.

fit_intercept : bool, default=True
    Whether the intercept should be estimated or not. If False, the
    data is assumed to be already centered. Defaults to True.

max_iter : int, default=1000
    The maximum number of passes over the training data (aka epochs).
    It only impacts the behavior in the ``fit`` method, and not the
    :meth:`partial_fit` method.

    .. versionadded:: 0.19

tol : float, default=1e-3
    The stopping criterion. If it is not None, the iterations will stop
    when (loss > best_loss - tol) for ``n_iter_no_change`` consecutive
    epochs.

    .. versionadded:: 0.19

shuffle : bool, default=True
    Whether or not the training data should be shuffled after each epoch.

verbose : int, default=0
    The verbosity level.

epsilon : float, default=0.1
    Epsilon in the epsilon-insensitive loss functions; only if `loss` is
    'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'.
    For 'huber', determines the threshold at which it becomes less
    important to get the prediction exactly right.
    For epsilon-insensitive, any differences between the current prediction
    and the correct label are ignored if they are less than this threshold.

n_jobs : int, default=None
    The number of CPUs to use to do the OVA (One Versus All, for
    multi-class problems) computation.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

random_state : int, RandomState instance, default=None
    The seed of the pseudo random number generator to use when shuffling
    the data.  If int, random_state is the seed used by the random number
    generator; If RandomState instance, random_state is the random number
    generator; If None, the random number generator is the RandomState
    instance used by `np.random`.

learning_rate : str, default='optimal'
    The learning rate schedule:

    'constant':
        eta = eta0
    'optimal': [default]
        eta = 1.0 / (alpha * (t + t0))
        where t0 is chosen by a heuristic proposed by Leon Bottou.
    'invscaling':
        eta = eta0 / pow(t, power_t)
    'adaptive':
        eta = eta0, as long as the training keeps decreasing.
        Each time n_iter_no_change consecutive epochs fail to decrease the
        training loss by tol or fail to increase validation score by tol if
        early_stopping is True, the current learning rate is divided by 5.

eta0 : double, default=0.0
    The initial learning rate for the 'constant', 'invscaling' or
    'adaptive' schedules. The default value is 0.0 as eta0 is not used by
    the default schedule 'optimal'.

power_t : double, default=0.5
    The exponent for inverse scaling learning rate [default 0.5].

early_stopping : bool, default=False
    Whether to use early stopping to terminate training when validation
    score is not improving. If set to True, it will automatically set aside
    a stratified fraction of training data as validation and terminate
    training when validation score is not improving by at least tol for
    n_iter_no_change consecutive epochs.

    .. versionadded:: 0.20

validation_fraction : float, default=0.1
    The proportion of training data to set aside as validation set for
    early stopping. Must be between 0 and 1.
    Only used if early_stopping is True.

    .. versionadded:: 0.20

n_iter_no_change : int, default=5
    Number of iterations with no improvement to wait before early stopping.

    .. versionadded:: 0.20

class_weight : dict, {class_label: weight} or "balanced", default=None
    Preset for the class_weight fit parameter.

    Weights associated with classes. If not given, all classes
    are supposed to have weight one.

    The "balanced" mode uses the values of y to automatically adjust
    weights inversely proportional to class frequencies in the input data
    as ``n_samples / (n_classes * np.bincount(y))``.

warm_start : bool, default=False
    When set to True, reuse the solution of the previous call to fit as
    initialization, otherwise, just erase the previous solution.
    See :term:`the Glossary <warm_start>`.

    Repeatedly calling fit or partial_fit when warm_start is True can
    result in a different solution than when calling fit a single time
    because of the way the data is shuffled.
    If a dynamic learning rate is used, the learning rate is adapted
    depending on the number of samples already seen. Calling ``fit`` resets
    this counter, while ``partial_fit`` will result in increasing the
    existing counter.

average : bool or int, default=False
    When set to True, computes the averaged SGD weights and stores the
    result in the ``coef_`` attribute. If set to an int greater than 1,
    averaging will begin once the total number of samples seen reaches
    average. So ``average=10`` will begin averaging after seeing 10
    samples.

Attributes
----------
coef_ : ndarray of shape (1, n_features) if n_classes == 2 else             (n_classes, n_features)
    Weights assigned to the features.

intercept_ : ndarray of shape (1,) if n_classes == 2 else (n_classes,)
    Constants in decision function.

n_iter_ : int
    The actual number of iterations to reach the stopping criterion.
    For multiclass fits, it is the maximum over every binary fit.

loss_function_ : concrete ``LossFunction``

classes_ : array of shape (n_classes,)

t_ : int
    Number of weight updates performed during training.
    Same as ``(n_iter_ * n_samples)``.

See Also
--------
sklearn.svm.LinearSVC: Linear support vector classification.
LogisticRegression: Logistic regression.
Perceptron: Inherits from SGDClassifier. ``Perceptron()`` is equivalent to
    ``SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant",
    penalty=None)``.

Examples
--------
>>> import numpy as np
>>> from sklearn import linear_model
>>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
>>> Y = np.array([1, 1, 2, 2])
>>> clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
>>> clf.fit(X, Y)
SGDClassifier()

>>> print(clf.predict([[-0.8, -1]]))
[1]
*)

val decision_function : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
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

val densify : t -> t
(**
Convert coefficient matrix to dense array format.

Converts the ``coef_`` member (back) to a numpy.ndarray. This is the
default format of ``coef_`` and is required for fitting, so calling
this method is only required on models that have previously been
sparsified; otherwise, it is a no-op.

Returns
-------
self
    Fitted estimator.
*)

val fit : ?coef_init:Ndarray.t -> ?intercept_init:Ndarray.t -> ?sample_weight:Ndarray.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:Ndarray.t -> t -> t
(**
Fit linear model with Stochastic Gradient Descent.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Training data.

y : ndarray of shape (n_samples,)
    Target values.

coef_init : ndarray of shape (n_classes, n_features), default=None
    The initial coefficients to warm-start the optimization.

intercept_init : ndarray of shape (n_classes,), default=None
    The initial intercept to warm-start the optimization.

sample_weight : array-like, shape (n_samples,), default=None
    Weights applied to individual samples.
    If not provided, uniform weights are assumed. These weights will
    be multiplied with class_weight (passed through the
    constructor) if class_weight is specified.

Returns
-------
self :
    Returns an instance of self.
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

val partial_fit : ?classes:Ndarray.t -> ?sample_weight:Ndarray.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:Ndarray.t -> t -> t
(**
Perform one epoch of stochastic gradient descent on given samples.

Internally, this method uses ``max_iter = 1``. Therefore, it is not
guaranteed that a minimum of the cost function is reached after calling
it once. Matters such as objective convergence and early stopping
should be handled by the user.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Subset of the training data.

y : ndarray of shape (n_samples,)
    Subset of the target values.

classes : ndarray of shape (n_classes,), default=None
    Classes across all calls to partial_fit.
    Can be obtained by via `np.unique(y_all)`, where y_all is the
    target vector of the entire dataset.
    This argument is required for the first call to partial_fit
    and can be omitted in the subsequent calls.
    Note that y doesn't need to contain all labels in `classes`.

sample_weight : array-like, shape (n_samples,), default=None
    Weights applied to individual samples.
    If not provided, uniform weights are assumed.

Returns
-------
self :
    Returns an instance of self.
*)

val predict : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
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

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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

val set_params : ?kwargs:(string * Py.Object.t) list -> t -> t
(**
Set and validate the parameters of estimator.

Parameters
----------
**kwargs : dict
    Estimator parameters.

Returns
-------
self : object
    Estimator instance.
*)

val sparsify : t -> t
(**
Convert coefficient matrix to sparse format.

Converts the ``coef_`` member to a scipy.sparse matrix, which for
L1-regularized models can be much more memory- and storage-efficient
than the usual numpy.ndarray representation.

The ``intercept_`` member is not converted.

Returns
-------
self
    Fitted estimator.

Notes
-----
For non-sparse models, i.e. when there are not many zeros in ``coef_``,
this may actually *increase* memory usage, so use this method with
care. A rule of thumb is that the number of zero elements, which can
be computed with ``(coef_ == 0).sum()``, must be more than 50% for this
to provide significant benefits.

After calling this method, further fitting with the partial_fit
method (if any) will not work until you call densify.
*)


(** Attribute coef_: see constructor for documentation *)
val coef_ : t -> Ndarray.t

(** Attribute intercept_: see constructor for documentation *)
val intercept_ : t -> Ndarray.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> int

(** Attribute loss_function_: see constructor for documentation *)
val loss_function_ : t -> Py.Object.t

(** Attribute classes_: see constructor for documentation *)
val classes_ : t -> Ndarray.t

(** Attribute t_: see constructor for documentation *)
val t_ : t -> int

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module SGDRegressor : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?loss:string -> ?penalty:[`L2 | `L1 | `Elasticnet] -> ?alpha:float -> ?l1_ratio:float -> ?fit_intercept:bool -> ?max_iter:int -> ?tol:float -> ?shuffle:bool -> ?verbose:int -> ?epsilon:float -> ?random_state:[`Int of int | `RandomState of Py.Object.t] -> ?learning_rate:string -> ?eta0:float -> ?power_t:float -> ?early_stopping:bool -> ?validation_fraction:float -> ?n_iter_no_change:int -> ?warm_start:bool -> ?average:[`Bool of bool | `Int of int] -> unit -> t
(**
Linear model fitted by minimizing a regularized empirical loss with SGD

SGD stands for Stochastic Gradient Descent: the gradient of the loss is
estimated each sample at a time and the model is updated along the way with
a decreasing strength schedule (aka learning rate).

The regularizer is a penalty added to the loss function that shrinks model
parameters towards the zero vector using either the squared euclidean norm
L2 or the absolute norm L1 or a combination of both (Elastic Net). If the
parameter update crosses the 0.0 value because of the regularizer, the
update is truncated to 0.0 to allow for learning sparse models and achieve
online feature selection.

This implementation works with data represented as dense numpy arrays of
floating point values for the features.

Read more in the :ref:`User Guide <sgd>`.

Parameters
----------
loss : str, default='squared_loss'
    The loss function to be used. The possible values are 'squared_loss',
    'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'

    The 'squared_loss' refers to the ordinary least squares fit.
    'huber' modifies 'squared_loss' to focus less on getting outliers
    correct by switching from squared to linear loss past a distance of
    epsilon. 'epsilon_insensitive' ignores errors less than epsilon and is
    linear past that; this is the loss function used in SVR.
    'squared_epsilon_insensitive' is the same but becomes squared loss past
    a tolerance of epsilon.

penalty : {'l2', 'l1', 'elasticnet'}, default='l2'
    The penalty (aka regularization term) to be used. Defaults to 'l2'
    which is the standard regularizer for linear SVM models. 'l1' and
    'elasticnet' might bring sparsity to the model (feature selection)
    not achievable with 'l2'.

alpha : float, default=0.0001
    Constant that multiplies the regularization term.
    Also used to compute learning_rate when set to 'optimal'.

l1_ratio : float, default=0.15
    The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.
    l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.

fit_intercept : bool, default=True
    Whether the intercept should be estimated or not. If False, the
    data is assumed to be already centered.

max_iter : int, default=1000
    The maximum number of passes over the training data (aka epochs).
    It only impacts the behavior in the ``fit`` method, and not the
    :meth:`partial_fit` method.

    .. versionadded:: 0.19

tol : float, default=1e-3
    The stopping criterion. If it is not None, the iterations will stop
    when (loss > best_loss - tol) for ``n_iter_no_change`` consecutive
    epochs.

    .. versionadded:: 0.19

shuffle : bool, default=True
    Whether or not the training data should be shuffled after each epoch.

verbose : int, default=0
    The verbosity level.

epsilon : float, default=0.1
    Epsilon in the epsilon-insensitive loss functions; only if `loss` is
    'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'.
    For 'huber', determines the threshold at which it becomes less
    important to get the prediction exactly right.
    For epsilon-insensitive, any differences between the current prediction
    and the correct label are ignored if they are less than this threshold.

random_state : int, RandomState instance, default=None
    The seed of the pseudo random number generator to use when shuffling
    the data.  If int, random_state is the seed used by the random number
    generator; If RandomState instance, random_state is the random number
    generator; If None, the random number generator is the RandomState
    instance used by `np.random`.

learning_rate : string, default='invscaling'
    The learning rate schedule:

    'constant':
        eta = eta0
    'optimal':
        eta = 1.0 / (alpha * (t + t0))
        where t0 is chosen by a heuristic proposed by Leon Bottou.
    'invscaling': [default]
        eta = eta0 / pow(t, power_t)
    'adaptive':
        eta = eta0, as long as the training keeps decreasing.
        Each time n_iter_no_change consecutive epochs fail to decrease the
        training loss by tol or fail to increase validation score by tol if
        early_stopping is True, the current learning rate is divided by 5.

eta0 : double, default=0.01
    The initial learning rate for the 'constant', 'invscaling' or
    'adaptive' schedules. The default value is 0.01.

power_t : double, default=0.25
    The exponent for inverse scaling learning rate.

early_stopping : bool, default=False
    Whether to use early stopping to terminate training when validation
    score is not improving. If set to True, it will automatically set aside
    a fraction of training data as validation and terminate
    training when validation score is not improving by at least tol for
    n_iter_no_change consecutive epochs.

    .. versionadded:: 0.20

validation_fraction : float, default=0.1
    The proportion of training data to set aside as validation set for
    early stopping. Must be between 0 and 1.
    Only used if early_stopping is True.

    .. versionadded:: 0.20

n_iter_no_change : int, default=5
    Number of iterations with no improvement to wait before early stopping.

    .. versionadded:: 0.20

warm_start : bool, default=False
    When set to True, reuse the solution of the previous call to fit as
    initialization, otherwise, just erase the previous solution.
    See :term:`the Glossary <warm_start>`.

    Repeatedly calling fit or partial_fit when warm_start is True can
    result in a different solution than when calling fit a single time
    because of the way the data is shuffled.
    If a dynamic learning rate is used, the learning rate is adapted
    depending on the number of samples already seen. Calling ``fit`` resets
    this counter, while ``partial_fit``  will result in increasing the
    existing counter.

average : bool or int, default=False
    When set to True, computes the averaged SGD weights and stores the
    result in the ``coef_`` attribute. If set to an int greater than 1,
    averaging will begin once the total number of samples seen reaches
    average. So ``average=10`` will begin averaging after seeing 10
    samples.

Attributes
----------
coef_ : ndarray of shape (n_features,)
    Weights assigned to the features.

intercept_ : ndarray of shape (1,)
    The intercept term.

average_coef_ : ndarray of shape (n_features,)
    Averaged weights assigned to the features.

average_intercept_ : ndarray of shape (1,)
    The averaged intercept term.

n_iter_ : int
    The actual number of iterations to reach the stopping criterion.

t_ : int
    Number of weight updates performed during training.
    Same as ``(n_iter_ * n_samples)``.

Examples
--------
>>> import numpy as np
>>> from sklearn import linear_model
>>> n_samples, n_features = 10, 5
>>> rng = np.random.RandomState(0)
>>> y = rng.randn(n_samples)
>>> X = rng.randn(n_samples, n_features)
>>> clf = linear_model.SGDRegressor(max_iter=1000, tol=1e-3)
>>> clf.fit(X, y)
SGDRegressor()

See also
--------
Ridge, ElasticNet, Lasso, sklearn.svm.SVR
*)

val densify : t -> t
(**
Convert coefficient matrix to dense array format.

Converts the ``coef_`` member (back) to a numpy.ndarray. This is the
default format of ``coef_`` and is required for fitting, so calling
this method is only required on models that have previously been
sparsified; otherwise, it is a no-op.

Returns
-------
self
    Fitted estimator.
*)

val fit : ?coef_init:Ndarray.t -> ?intercept_init:Ndarray.t -> ?sample_weight:Ndarray.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:Ndarray.t -> t -> t
(**
Fit linear model with Stochastic Gradient Descent.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Training data

y : ndarray of shape (n_samples,)
    Target values

coef_init : ndarray of shape (n_features,), default=None
    The initial coefficients to warm-start the optimization.

intercept_init : ndarray of shape (1,), default=None
    The initial intercept to warm-start the optimization.

sample_weight : array-like, shape (n_samples,), default=None
    Weights applied to individual samples (1. for unweighted).

Returns
-------
self : returns an instance of self.
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

val partial_fit : ?sample_weight:Ndarray.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:Ndarray.t -> t -> t
(**
Perform one epoch of stochastic gradient descent on given samples.

Internally, this method uses ``max_iter = 1``. Therefore, it is not
guaranteed that a minimum of the cost function is reached after calling
it once. Matters such as objective convergence and early stopping
should be handled by the user.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Subset of training data

y : numpy array of shape (n_samples,)
    Subset of target values

sample_weight : array-like, shape (n_samples,), default=None
    Weights applied to individual samples.
    If not provided, uniform weights are assumed.

Returns
-------
self : returns an instance of self.
*)

val predict : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Predict using the linear model

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)

Returns
-------
ndarray of shape (n_samples,)
   Predicted target values per element in X.
*)

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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

val set_params : ?kwargs:(string * Py.Object.t) list -> t -> t
(**
Set and validate the parameters of estimator.

Parameters
----------
**kwargs : dict
    Estimator parameters.

Returns
-------
self : object
    Estimator instance.
*)

val sparsify : t -> t
(**
Convert coefficient matrix to sparse format.

Converts the ``coef_`` member to a scipy.sparse matrix, which for
L1-regularized models can be much more memory- and storage-efficient
than the usual numpy.ndarray representation.

The ``intercept_`` member is not converted.

Returns
-------
self
    Fitted estimator.

Notes
-----
For non-sparse models, i.e. when there are not many zeros in ``coef_``,
this may actually *increase* memory usage, so use this method with
care. A rule of thumb is that the number of zero elements, which can
be computed with ``(coef_ == 0).sum()``, must be more than 50% for this
to provide significant benefits.

After calling this method, further fitting with the partial_fit
method (if any) will not work until you call densify.
*)


(** Attribute coef_: see constructor for documentation *)
val coef_ : t -> Ndarray.t

(** Attribute intercept_: see constructor for documentation *)
val intercept_ : t -> Ndarray.t

(** Attribute average_coef_: see constructor for documentation *)
val average_coef_ : t -> Ndarray.t

(** Attribute average_intercept_: see constructor for documentation *)
val average_intercept_ : t -> Ndarray.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> int

(** Attribute t_: see constructor for documentation *)
val t_ : t -> int

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module TheilSenRegressor : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?fit_intercept:bool -> ?copy_X:bool -> ?max_subpopulation:int -> ?n_subsamples:int -> ?max_iter:int -> ?tol:float -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> ?n_jobs:[`Int of int | `None] -> ?verbose:bool -> unit -> t
(**
Theil-Sen Estimator: robust multivariate regression model.

The algorithm calculates least square solutions on subsets with size
n_subsamples of the samples in X. Any value of n_subsamples between the
number of features and samples leads to an estimator with a compromise
between robustness and efficiency. Since the number of least square
solutions is "n_samples choose n_subsamples", it can be extremely large
and can therefore be limited with max_subpopulation. If this limit is
reached, the subsets are chosen randomly. In a final step, the spatial
median (or L1 median) is calculated of all least square solutions.

Read more in the :ref:`User Guide <theil_sen_regression>`.

Parameters
----------
fit_intercept : boolean, optional, default True
    Whether to calculate the intercept for this model. If set
    to false, no intercept will be used in calculations.

copy_X : boolean, optional, default True
    If True, X will be copied; else, it may be overwritten.

max_subpopulation : int, optional, default 1e4
    Instead of computing with a set of cardinality 'n choose k', where n is
    the number of samples and k is the number of subsamples (at least
    number of features), consider only a stochastic subpopulation of a
    given maximal size if 'n choose k' is larger than max_subpopulation.
    For other than small problem sizes this parameter will determine
    memory usage and runtime if n_subsamples is not changed.

n_subsamples : int, optional, default None
    Number of samples to calculate the parameters. This is at least the
    number of features (plus 1 if fit_intercept=True) and the number of
    samples as a maximum. A lower number leads to a higher breakdown
    point and a low efficiency while a high number leads to a low
    breakdown point and a high efficiency. If None, take the
    minimum number of subsamples leading to maximal robustness.
    If n_subsamples is set to n_samples, Theil-Sen is identical to least
    squares.

max_iter : int, optional, default 300
    Maximum number of iterations for the calculation of spatial median.

tol : float, optional, default 1.e-3
    Tolerance when calculating spatial median.

random_state : int, RandomState instance or None, optional, default None
    A random number generator instance to define the state of the random
    permutations generator.  If int, random_state is the seed used by the
    random number generator; If RandomState instance, random_state is the
    random number generator; If None, the random number generator is the
    RandomState instance used by `np.random`.

n_jobs : int or None, optional (default=None)
    Number of CPUs to use during the cross validation.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

verbose : boolean, optional, default False
    Verbose mode when fitting the model.

Attributes
----------
coef_ : array, shape = (n_features)
    Coefficients of the regression model (median of distribution).

intercept_ : float
    Estimated intercept of regression model.

breakdown_ : float
    Approximated breakdown point.

n_iter_ : int
    Number of iterations needed for the spatial median.

n_subpopulation_ : int
    Number of combinations taken into account from 'n choose k', where n is
    the number of samples and k is the number of subsamples.

Examples
--------
>>> from sklearn.linear_model import TheilSenRegressor
>>> from sklearn.datasets import make_regression
>>> X, y = make_regression(
...     n_samples=200, n_features=2, noise=4.0, random_state=0)
>>> reg = TheilSenRegressor(random_state=0).fit(X, y)
>>> reg.score(X, y)
0.9884...
>>> reg.predict(X[:1,])
array([-31.5871...])

References
----------
- Theil-Sen Estimators in a Multiple Linear Regression Model, 2009
  Xin Dang, Hanxiang Peng, Xueqin Wang and Heping Zhang
  http://home.olemiss.edu/~xdang/papers/MTSE.pdf
*)

val fit : x:Ndarray.t -> y:Py.Object.t -> t -> t
(**
Fit linear model.

Parameters
----------
X : numpy array of shape [n_samples, n_features]
    Training data
y : numpy array of shape [n_samples]
    Target values

Returns
-------
self : returns an instance of self.
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

val predict : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Predict using the linear model.

Parameters
----------
X : array_like or sparse matrix, shape (n_samples, n_features)
    Samples.

Returns
-------
C : array, shape (n_samples,)
    Returns predicted values.
*)

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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


(** Attribute coef_: see constructor for documentation *)
val coef_ : t -> Ndarray.t

(** Attribute intercept_: see constructor for documentation *)
val intercept_ : t -> Ndarray.t

(** Attribute breakdown_: see constructor for documentation *)
val breakdown_ : t -> float

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> int

(** Attribute n_subpopulation_: see constructor for documentation *)
val n_subpopulation_ : t -> int

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val enet_path : ?l1_ratio:float -> ?eps:float -> ?n_alphas:int -> ?alphas:Ndarray.t -> ?precompute:[`Bool of bool | `Auto | `Ndarray of Ndarray.t] -> ?xy:Ndarray.t -> ?copy_X:bool -> ?coef_init:[`Ndarray of Ndarray.t | `None] -> ?verbose:[`Bool of bool | `Int of int] -> ?return_n_iter:bool -> ?positive:bool -> ?check_input:bool -> ?params:(string * Py.Object.t) list -> x:Ndarray.t -> y:Ndarray.t -> unit -> (Ndarray.t * Ndarray.t * Ndarray.t * Ndarray.t)
(**
Compute elastic net path with coordinate descent.

The elastic net optimization function varies for mono and multi-outputs.

For mono-output tasks it is::

    1 / (2 * n_samples) * ||y - Xw||^2_2
    + alpha * l1_ratio * ||w||_1
    + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

For multi-output tasks it is::

    (1 / (2 * n_samples)) * ||Y - XW||^Fro_2
    + alpha * l1_ratio * ||W||_21
    + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2

Where::

    ||W||_21 = \sum_i \sqrt{\sum_j w_{ij}^2}

i.e. the sum of norm of each row.

Read more in the :ref:`User Guide <elastic_net>`.

Parameters
----------
X : {array-like}, shape (n_samples, n_features)
    Training data. Pass directly as Fortran-contiguous data to avoid
    unnecessary memory duplication. If ``y`` is mono-output then ``X``
    can be sparse.

y : ndarray, shape (n_samples,) or (n_samples, n_outputs)
    Target values.

l1_ratio : float, optional
    Number between 0 and 1 passed to elastic net (scaling between
    l1 and l2 penalties). ``l1_ratio=1`` corresponds to the Lasso.

eps : float
    Length of the path. ``eps=1e-3`` means that
    ``alpha_min / alpha_max = 1e-3``.

n_alphas : int, optional
    Number of alphas along the regularization path.

alphas : ndarray, optional
    List of alphas where to compute the models.
    If None alphas are set automatically.

precompute : True | False | 'auto' | array-like
    Whether to use a precomputed Gram matrix to speed up
    calculations. If set to ``'auto'`` let us decide. The Gram
    matrix can also be passed as argument.

Xy : array-like, optional
    Xy = np.dot(X.T, y) that can be precomputed. It is useful
    only when the Gram matrix is precomputed.

copy_X : bool, optional, default True
    If ``True``, X will be copied; else, it may be overwritten.

coef_init : array, shape (n_features, ) | None
    The initial values of the coefficients.

verbose : bool or int
    Amount of verbosity.

return_n_iter : bool
    Whether to return the number of iterations or not.

positive : bool, default False
    If set to True, forces coefficients to be positive.
    (Only allowed when ``y.ndim == 1``).

check_input : bool, default True
    Skip input validation checks, including the Gram matrix when provided
    assuming there are handled by the caller when check_input=False.

**params : kwargs
    Keyword arguments passed to the coordinate descent solver.

Returns
-------
alphas : array, shape (n_alphas,)
    The alphas along the path where models are computed.

coefs : array, shape (n_features, n_alphas) or             (n_outputs, n_features, n_alphas)
    Coefficients along the path.

dual_gaps : array, shape (n_alphas,)
    The dual gaps at the end of the optimization for each alpha.

n_iters : array-like, shape (n_alphas,)
    The number of iterations taken by the coordinate descent optimizer to
    reach the specified tolerance for each alpha.
    (Is returned when ``return_n_iter`` is set to True).

See Also
--------
MultiTaskElasticNet
MultiTaskElasticNetCV
ElasticNet
ElasticNetCV

Notes
-----
For an example, see
:ref:`examples/linear_model/plot_lasso_coordinate_descent_path.py
<sphx_glr_auto_examples_linear_model_plot_lasso_coordinate_descent_path.py>`.
*)

val lars_path : ?xy:Ndarray.t -> ?gram:[`Auto | `Ndarray of Ndarray.t | `None] -> ?max_iter:int -> ?alpha_min:float -> ?method_:[`Lar | `Lasso] -> ?copy_X:bool -> ?eps:float -> ?copy_Gram:bool -> ?verbose:int -> ?return_path:bool -> ?return_n_iter:bool -> ?positive:bool -> x:[`Ndarray of Ndarray.t | `None] -> y:[`Ndarray of Ndarray.t | `None] -> unit -> (Ndarray.t * Ndarray.t * Ndarray.t * int)
(**
Compute Least Angle Regression or Lasso path using LARS algorithm [1]

The optimization objective for the case method='lasso' is::

(1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

in the case of method='lars', the objective function is only known in
the form of an implicit equation (see discussion in [1])

Read more in the :ref:`User Guide <least_angle_regression>`.

Parameters
----------
X : None or array-like of shape (n_samples, n_features)
    Input data. Note that if X is None then the Gram matrix must be
    specified, i.e., cannot be None or False.

    .. deprecated:: 0.21

       The use of ``X`` is ``None`` in combination with ``Gram`` is not
       ``None`` will be removed in v0.23. Use :func:`lars_path_gram`
       instead.

y : None or array-like of shape (n_samples,)
    Input targets.

Xy : array-like of shape (n_samples,) or (n_samples, n_targets),             default=None
    Xy = np.dot(X.T, y) that can be precomputed. It is useful
    only when the Gram matrix is precomputed.

Gram : None, 'auto', array-like of shape (n_features, n_features),             default=None
    Precomputed Gram matrix (X' * X), if ``'auto'``, the Gram
    matrix is precomputed from the given X, if there are more samples
    than features.

    .. deprecated:: 0.21

       The use of ``X`` is ``None`` in combination with ``Gram`` is not
       None will be removed in v0.23. Use :func:`lars_path_gram` instead.

max_iter : int, default=500
    Maximum number of iterations to perform, set to infinity for no limit.

alpha_min : float, default=0
    Minimum correlation along the path. It corresponds to the
    regularization parameter alpha parameter in the Lasso.

method : {'lar', 'lasso'}, default='lar'
    Specifies the returned model. Select ``'lar'`` for Least Angle
    Regression, ``'lasso'`` for the Lasso.

copy_X : bool, default=True
    If ``False``, ``X`` is overwritten.

eps : float, optional
    The machine-precision regularization in the computation of the
    Cholesky diagonal factors. Increase this for very ill-conditioned
    systems. By default, ``np.finfo(np.float).eps`` is used.

copy_Gram : bool, default=True
    If ``False``, ``Gram`` is overwritten.

verbose : int, default=0
    Controls output verbosity.

return_path : bool, default=True
    If ``return_path==True`` returns the entire path, else returns only the
    last point of the path.

return_n_iter : bool, default=False
    Whether to return the number of iterations.

positive : bool, default=False
    Restrict coefficients to be >= 0.
    This option is only allowed with method 'lasso'. Note that the model
    coefficients will not converge to the ordinary-least-squares solution
    for small values of alpha. Only coefficients up to the smallest alpha
    value (``alphas_[alphas_ > 0.].min()`` when fit_path=True) reached by
    the stepwise Lars-Lasso algorithm are typically in congruence with the
    solution of the coordinate descent lasso_path function.

Returns
-------
alphas : array-like of shape (n_alphas + 1,)
    Maximum of covariances (in absolute value) at each iteration.
    ``n_alphas`` is either ``max_iter``, ``n_features`` or the
    number of nodes in the path with ``alpha >= alpha_min``, whichever
    is smaller.

active : array-like of shape (n_alphas,)
    Indices of active variables at the end of the path.

coefs : array-like of shape (n_features, n_alphas + 1)
    Coefficients along the path

n_iter : int
    Number of iterations run. Returned only if return_n_iter is set
    to True.

See also
--------
lars_path_gram
lasso_path
lasso_path_gram
LassoLars
Lars
LassoLarsCV
LarsCV
sklearn.decomposition.sparse_encode

References
----------
.. [1] "Least Angle Regression", Efron et al.
       http://statweb.stanford.edu/~tibs/ftp/lars.pdf

.. [2] `Wikipedia entry on the Least-angle regression
       <https://en.wikipedia.org/wiki/Least-angle_regression>`_

.. [3] `Wikipedia entry on the Lasso
       <https://en.wikipedia.org/wiki/Lasso_(statistics)>`_
*)

val lars_path_gram : ?max_iter:int -> ?alpha_min:float -> ?method_:[`Lar | `Lasso] -> ?copy_X:bool -> ?eps:float -> ?copy_Gram:bool -> ?verbose:int -> ?return_path:bool -> ?return_n_iter:bool -> ?positive:bool -> xy:Ndarray.t -> gram:Ndarray.t -> n_samples:[`Int of int | `Float of float] -> unit -> (Ndarray.t * Ndarray.t * Ndarray.t * int)
(**
lars_path in the sufficient stats mode [1]

The optimization objective for the case method='lasso' is::

(1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

in the case of method='lars', the objective function is only known in
the form of an implicit equation (see discussion in [1])

Read more in the :ref:`User Guide <least_angle_regression>`.

Parameters
----------
Xy : array-like of shape (n_samples,) or (n_samples, n_targets)
    Xy = np.dot(X.T, y).

Gram : array-like of shape (n_features, n_features)
    Gram = np.dot(X.T * X).

n_samples : int or float
    Equivalent size of sample.

max_iter : int, default=500
    Maximum number of iterations to perform, set to infinity for no limit.

alpha_min : float, default=0
    Minimum correlation along the path. It corresponds to the
    regularization parameter alpha parameter in the Lasso.

method : {'lar', 'lasso'}, default='lar'
    Specifies the returned model. Select ``'lar'`` for Least Angle
    Regression, ``'lasso'`` for the Lasso.

copy_X : bool, default=True
    If ``False``, ``X`` is overwritten.

eps : float, optional
    The machine-precision regularization in the computation of the
    Cholesky diagonal factors. Increase this for very ill-conditioned
    systems. By default, ``np.finfo(np.float).eps`` is used.

copy_Gram : bool, default=True
    If ``False``, ``Gram`` is overwritten.

verbose : int, default=0
    Controls output verbosity.

return_path : bool, default=True
    If ``return_path==True`` returns the entire path, else returns only the
    last point of the path.

return_n_iter : bool, default=False
    Whether to return the number of iterations.

positive : bool, default=False
    Restrict coefficients to be >= 0.
    This option is only allowed with method 'lasso'. Note that the model
    coefficients will not converge to the ordinary-least-squares solution
    for small values of alpha. Only coefficients up to the smallest alpha
    value (``alphas_[alphas_ > 0.].min()`` when fit_path=True) reached by
    the stepwise Lars-Lasso algorithm are typically in congruence with the
    solution of the coordinate descent lasso_path function.

Returns
-------
alphas : array-like of shape (n_alphas + 1,)
    Maximum of covariances (in absolute value) at each iteration.
    ``n_alphas`` is either ``max_iter``, ``n_features`` or the
    number of nodes in the path with ``alpha >= alpha_min``, whichever
    is smaller.

active : array-like of shape (n_alphas,)
    Indices of active variables at the end of the path.

coefs : array-like of shape (n_features, n_alphas + 1)
    Coefficients along the path

n_iter : int
    Number of iterations run. Returned only if return_n_iter is set
    to True.

See also
--------
lars_path
lasso_path
lasso_path_gram
LassoLars
Lars
LassoLarsCV
LarsCV
sklearn.decomposition.sparse_encode

References
----------
.. [1] "Least Angle Regression", Efron et al.
       http://statweb.stanford.edu/~tibs/ftp/lars.pdf

.. [2] `Wikipedia entry on the Least-angle regression
       <https://en.wikipedia.org/wiki/Least-angle_regression>`_

.. [3] `Wikipedia entry on the Lasso
       <https://en.wikipedia.org/wiki/Lasso_(statistics)>`_
*)

val lasso_path : ?eps:float -> ?n_alphas:int -> ?alphas:Ndarray.t -> ?precompute:[`Bool of bool | `Auto | `Ndarray of Ndarray.t] -> ?xy:Ndarray.t -> ?copy_X:bool -> ?coef_init:[`Ndarray of Ndarray.t | `None] -> ?verbose:[`Bool of bool | `Int of int] -> ?return_n_iter:bool -> ?positive:bool -> ?params:(string * Py.Object.t) list -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:Ndarray.t -> unit -> (Ndarray.t * Ndarray.t * Ndarray.t * Ndarray.t)
(**
Compute Lasso path with coordinate descent

The Lasso optimization function varies for mono and multi-outputs.

For mono-output tasks it is::

    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

For multi-output tasks it is::

    (1 / (2 * n_samples)) * ||Y - XW||^2_Fro + alpha * ||W||_21

Where::

    ||W||_21 = \sum_i \sqrt{\sum_j w_{ij}^2}

i.e. the sum of norm of each row.

Read more in the :ref:`User Guide <lasso>`.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Training data. Pass directly as Fortran-contiguous data to avoid
    unnecessary memory duplication. If ``y`` is mono-output then ``X``
    can be sparse.

y : ndarray, shape (n_samples,), or (n_samples, n_outputs)
    Target values

eps : float, optional
    Length of the path. ``eps=1e-3`` means that
    ``alpha_min / alpha_max = 1e-3``

n_alphas : int, optional
    Number of alphas along the regularization path

alphas : ndarray, optional
    List of alphas where to compute the models.
    If ``None`` alphas are set automatically

precompute : True | False | 'auto' | array-like
    Whether to use a precomputed Gram matrix to speed up
    calculations. If set to ``'auto'`` let us decide. The Gram
    matrix can also be passed as argument.

Xy : array-like, optional
    Xy = np.dot(X.T, y) that can be precomputed. It is useful
    only when the Gram matrix is precomputed.

copy_X : boolean, optional, default True
    If ``True``, X will be copied; else, it may be overwritten.

coef_init : array, shape (n_features, ) | None
    The initial values of the coefficients.

verbose : bool or integer
    Amount of verbosity.

return_n_iter : bool
    whether to return the number of iterations or not.

positive : bool, default False
    If set to True, forces coefficients to be positive.
    (Only allowed when ``y.ndim == 1``).

**params : kwargs
    keyword arguments passed to the coordinate descent solver.

Returns
-------
alphas : array, shape (n_alphas,)
    The alphas along the path where models are computed.

coefs : array, shape (n_features, n_alphas) or             (n_outputs, n_features, n_alphas)
    Coefficients along the path.

dual_gaps : array, shape (n_alphas,)
    The dual gaps at the end of the optimization for each alpha.

n_iters : array-like, shape (n_alphas,)
    The number of iterations taken by the coordinate descent optimizer to
    reach the specified tolerance for each alpha.

Notes
-----
For an example, see
:ref:`examples/linear_model/plot_lasso_coordinate_descent_path.py
<sphx_glr_auto_examples_linear_model_plot_lasso_coordinate_descent_path.py>`.

To avoid unnecessary memory duplication the X argument of the fit method
should be directly passed as a Fortran-contiguous numpy array.

Note that in certain cases, the Lars solver may be significantly
faster to implement this functionality. In particular, linear
interpolation can be used to retrieve model coefficients between the
values output by lars_path

Examples
--------

Comparing lasso_path and lars_path with interpolation:

>>> X = np.array([[1, 2, 3.1], [2.3, 5.4, 4.3]]).T
>>> y = np.array([1, 2, 3.1])
>>> # Use lasso_path to compute a coefficient path
>>> _, coef_path, _ = lasso_path(X, y, alphas=[5., 1., .5])
>>> print(coef_path)
[[0.         0.         0.46874778]
 [0.2159048  0.4425765  0.23689075]]

>>> # Now use lars_path and 1D linear interpolation to compute the
>>> # same path
>>> from sklearn.linear_model import lars_path
>>> alphas, active, coef_path_lars = lars_path(X, y, method='lasso')
>>> from scipy import interpolate
>>> coef_path_continuous = interpolate.interp1d(alphas[::-1],
...                                             coef_path_lars[:, ::-1])
>>> print(coef_path_continuous([5., 1., .5]))
[[0.         0.         0.46915237]
 [0.2159048  0.4425765  0.23668876]]


See also
--------
lars_path
Lasso
LassoLars
LassoCV
LassoLarsCV
sklearn.decomposition.sparse_encode
*)

val logistic_regression_path : ?pos_class:[`Int of int | `None] -> ?cs:[`Int of int | `Ndarray of Ndarray.t] -> ?fit_intercept:bool -> ?max_iter:int -> ?tol:float -> ?verbose:int -> ?solver:[`Lbfgs | `Newton_cg | `Liblinear | `Sag | `Saga] -> ?coef:Ndarray.t -> ?class_weight:[`DictIntToFloat of (int * float) list | `Balanced] -> ?dual:bool -> ?penalty:[`L1 | `L2 | `Elasticnet] -> ?intercept_scaling:float -> ?multi_class:[`Ovr | `Multinomial | `Auto] -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> ?check_input:bool -> ?max_squared_sum:float -> ?sample_weight:Ndarray.t -> ?l1_ratio:[`Float of float | `None] -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:Ndarray.t -> unit -> (Py.Object.t * Ndarray.t * Ndarray.t)
(**
DEPRECATED: logistic_regression_path was deprecated in version 0.21 and will be removed in version 0.23.0

Compute a Logistic Regression model for a list of regularization
    parameters.

    This is an implementation that uses the result of the previous model
    to speed up computations along the set of solutions, making it faster
    than sequentially calling LogisticRegression for the different parameters.
    Note that there will be no speedup with liblinear solver, since it does
    not handle warm-starting.

    .. deprecated:: 0.21
        ``logistic_regression_path`` was deprecated in version 0.21 and will
        be removed in 0.23.

    Read more in the :ref:`User Guide <logistic_regression>`.

    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        Input data.

    y : array-like, shape (n_samples,) or (n_samples, n_targets)
        Input data, target values.

    pos_class : int, None
        The class with respect to which we perform a one-vs-all fit.
        If None, then it is assumed that the given problem is binary.

    Cs : int | array-like, shape (n_cs,)
        List of values for the regularization parameter or integer specifying
        the number of regularization parameters that should be used. In this
        case, the parameters will be chosen in a logarithmic scale between
        1e-4 and 1e4.

    fit_intercept : bool
        Whether to fit an intercept for the model. In this case the shape of
        the returned array is (n_cs, n_features + 1).

    max_iter : int
        Maximum number of iterations for the solver.

    tol : float
        Stopping criterion. For the newton-cg and lbfgs solvers, the iteration
        will stop when ``max{ |g_i | i = 1, ..., n} <= tol``
        where ``g_i`` is the i-th component of the gradient.

    verbose : int
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    solver : {'lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'}
        Numerical solver to use.

    coef : array-like, shape (n_features,), default None
        Initialization value for coefficients of logistic regression.
        Useless for liblinear solver.

    class_weight : dict or 'balanced', optional
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    dual : bool
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    penalty : str, 'l1', 'l2', or 'elasticnet'
        Used to specify the norm used in the penalization. The 'newton-cg',
        'sag' and 'lbfgs' solvers support only l2 penalties. 'elasticnet' is
        only supported by the 'saga' solver.

    intercept_scaling : float, default 1.
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equal to
        intercept_scaling is appended to the instance vector.
        The intercept becomes ``intercept_scaling * synthetic_feature_weight``.

        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    multi_class : {'ovr', 'multinomial', 'auto'}, default='auto'
        If the option chosen is 'ovr', then a binary problem is fit for each
        label. For 'multinomial' the loss minimised is the multinomial loss fit
        across the entire probability distribution, *even when the data is
        binary*. 'multinomial' is unavailable when solver='liblinear'.
        'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
        and otherwise selects 'multinomial'.

        .. versionadded:: 0.18
           Stochastic Average Gradient descent solver for 'multinomial' case.
        .. versionchanged:: 0.22
            Default changed from 'ovr' to 'auto' in 0.22.

    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`. Used when ``solver`` == 'sag' or
        'liblinear'.

    check_input : bool, default True
        If False, the input arrays X and y will not be checked.

    max_squared_sum : float, default None
        Maximum squared sum of X over samples. Used only in SAG solver.
        If None, it will be computed, going through all the samples.
        The value should be precomputed to speed up cross validation.

    sample_weight : array-like, shape(n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.

    l1_ratio : float or None, optional (default=None)
        The Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``. Only
        used if ``penalty='elasticnet'``. Setting ``l1_ratio=0`` is equivalent
        to using ``penalty='l2'``, while setting ``l1_ratio=1`` is equivalent
        to using ``penalty='l1'``. For ``0 < l1_ratio <1``, the penalty is a
        combination of L1 and L2.

    Returns
    -------
    coefs : ndarray, shape (n_cs, n_features) or (n_cs, n_features + 1)
        List of coefficients for the Logistic Regression model. If
        fit_intercept is set to True then the second dimension will be
        n_features + 1, where the last item represents the intercept. For
        ``multiclass='multinomial'``, the shape is (n_classes, n_cs,
        n_features) or (n_classes, n_cs, n_features + 1).

    Cs : ndarray
        Grid of Cs used for cross-validation.

    n_iter : array, shape (n_cs,)
        Actual number of iteration for each Cs.

    Notes
    -----
    You might get slightly different results with the solver liblinear than
    with the others since this uses LIBLINEAR which penalizes the intercept.

    .. versionchanged:: 0.19
        The "copy" parameter was removed.
    
*)

val orthogonal_mp : ?n_nonzero_coefs:int -> ?tol:float -> ?precompute:[`Bool of bool | `Auto] -> ?copy_X:bool -> ?return_path:bool -> ?return_n_iter:bool -> x:Ndarray.t -> y:Ndarray.t -> unit -> (Ndarray.t * Py.Object.t)
(**
Orthogonal Matching Pursuit (OMP)

Solves n_targets Orthogonal Matching Pursuit problems.
An instance of the problem has the form:

When parametrized by the number of non-zero coefficients using
`n_nonzero_coefs`:
argmin ||y - X\gamma||^2 subject to ||\gamma||_0 <= n_{nonzero coefs}

When parametrized by error using the parameter `tol`:
argmin ||\gamma||_0 subject to ||y - X\gamma||^2 <= tol

Read more in the :ref:`User Guide <omp>`.

Parameters
----------
X : array, shape (n_samples, n_features)
    Input data. Columns are assumed to have unit norm.

y : array, shape (n_samples,) or (n_samples, n_targets)
    Input targets

n_nonzero_coefs : int
    Desired number of non-zero entries in the solution. If None (by
    default) this value is set to 10% of n_features.

tol : float
    Maximum norm of the residual. If not None, overrides n_nonzero_coefs.

precompute : {True, False, 'auto'},
    Whether to perform precomputations. Improves performance when n_targets
    or n_samples is very large.

copy_X : bool, optional
    Whether the design matrix X must be copied by the algorithm. A false
    value is only helpful if X is already Fortran-ordered, otherwise a
    copy is made anyway.

return_path : bool, optional. Default: False
    Whether to return every value of the nonzero coefficients along the
    forward path. Useful for cross-validation.

return_n_iter : bool, optional default False
    Whether or not to return the number of iterations.

Returns
-------
coef : array, shape (n_features,) or (n_features, n_targets)
    Coefficients of the OMP solution. If `return_path=True`, this contains
    the whole coefficient path. In this case its shape is
    (n_features, n_features) or (n_features, n_targets, n_features) and
    iterating over the last axis yields coefficients in increasing order
    of active features.

n_iters : array-like or int
    Number of active features across every target. Returned only if
    `return_n_iter` is set to True.

See also
--------
OrthogonalMatchingPursuit
orthogonal_mp_gram
lars_path
decomposition.sparse_encode

Notes
-----
Orthogonal matching pursuit was introduced in S. Mallat, Z. Zhang,
Matching pursuits with time-frequency dictionaries, IEEE Transactions on
Signal Processing, Vol. 41, No. 12. (December 1993), pp. 3397-3415.
(http://blanche.polytechnique.fr/~mallat/papiers/MallatPursuit93.pdf)

This implementation is based on Rubinstein, R., Zibulevsky, M. and Elad,
M., Efficient Implementation of the K-SVD Algorithm using Batch Orthogonal
Matching Pursuit Technical Report - CS Technion, April 2008.
https://www.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf
*)

val orthogonal_mp_gram : ?n_nonzero_coefs:int -> ?tol:float -> ?norms_squared:Ndarray.t -> ?copy_Gram:bool -> ?copy_Xy:bool -> ?return_path:bool -> ?return_n_iter:bool -> gram:Ndarray.t -> xy:Ndarray.t -> unit -> (Ndarray.t * Py.Object.t)
(**
Gram Orthogonal Matching Pursuit (OMP)

Solves n_targets Orthogonal Matching Pursuit problems using only
the Gram matrix X.T * X and the product X.T * y.

Read more in the :ref:`User Guide <omp>`.

Parameters
----------
Gram : array, shape (n_features, n_features)
    Gram matrix of the input data: X.T * X

Xy : array, shape (n_features,) or (n_features, n_targets)
    Input targets multiplied by X: X.T * y

n_nonzero_coefs : int
    Desired number of non-zero entries in the solution. If None (by
    default) this value is set to 10% of n_features.

tol : float
    Maximum norm of the residual. If not None, overrides n_nonzero_coefs.

norms_squared : array-like, shape (n_targets,)
    Squared L2 norms of the lines of y. Required if tol is not None.

copy_Gram : bool, optional
    Whether the gram matrix must be copied by the algorithm. A false
    value is only helpful if it is already Fortran-ordered, otherwise a
    copy is made anyway.

copy_Xy : bool, optional
    Whether the covariance vector Xy must be copied by the algorithm.
    If False, it may be overwritten.

return_path : bool, optional. Default: False
    Whether to return every value of the nonzero coefficients along the
    forward path. Useful for cross-validation.

return_n_iter : bool, optional default False
    Whether or not to return the number of iterations.

Returns
-------
coef : array, shape (n_features,) or (n_features, n_targets)
    Coefficients of the OMP solution. If `return_path=True`, this contains
    the whole coefficient path. In this case its shape is
    (n_features, n_features) or (n_features, n_targets, n_features) and
    iterating over the last axis yields coefficients in increasing order
    of active features.

n_iters : array-like or int
    Number of active features across every target. Returned only if
    `return_n_iter` is set to True.

See also
--------
OrthogonalMatchingPursuit
orthogonal_mp
lars_path
decomposition.sparse_encode

Notes
-----
Orthogonal matching pursuit was introduced in G. Mallat, Z. Zhang,
Matching pursuits with time-frequency dictionaries, IEEE Transactions on
Signal Processing, Vol. 41, No. 12. (December 1993), pp. 3397-3415.
(http://blanche.polytechnique.fr/~mallat/papiers/MallatPursuit93.pdf)

This implementation is based on Rubinstein, R., Zibulevsky, M. and Elad,
M., Efficient Implementation of the K-SVD Algorithm using Batch Orthogonal
Matching Pursuit Technical Report - CS Technion, April 2008.
https://www.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf
*)

val ridge_regression : ?sample_weight:[`Float of float | `Ndarray of Ndarray.t] -> ?solver:[`Auto | `Svd | `Cholesky | `Lsqr | `Sparse_cg | `Sag | `Saga] -> ?max_iter:int -> ?tol:float -> ?verbose:int -> ?random_state:[`Int of int | `RandomState of Py.Object.t] -> ?return_n_iter:bool -> ?return_intercept:bool -> ?check_input:bool -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t | `LinearOperator of Py.Object.t] -> y:Ndarray.t -> alpha:[`Float of float | `Ndarray of Ndarray.t] -> unit -> (Ndarray.t * int * Py.Object.t)
(**
Solve the ridge equation by the method of normal equations.

Read more in the :ref:`User Guide <ridge_regression>`.

Parameters
----------
X : {ndarray, sparse matrix, LinearOperator} of shape         (n_samples, n_features)
    Training data

y : ndarray of shape (n_samples,) or (n_samples, n_targets)
    Target values

alpha : float or array-like of shape (n_targets,)
    Regularization strength; must be a positive float. Regularization
    improves the conditioning of the problem and reduces the variance of
    the estimates. Larger values specify stronger regularization.
    Alpha corresponds to ``C^-1`` in other linear models such as
    LogisticRegression or LinearSVC. If an array is passed, penalties are
    assumed to be specific to the targets. Hence they must correspond in
    number.

sample_weight : float or array-like of shape (n_samples,), default=None
    Individual weights for each sample. If given a float, every sample
    will have the same weight. If sample_weight is not None and
    solver='auto', the solver will be set to 'cholesky'.

    .. versionadded:: 0.17

solver : {'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'},         default='auto'
    Solver to use in the computational routines:

    - 'auto' chooses the solver automatically based on the type of data.

    - 'svd' uses a Singular Value Decomposition of X to compute the Ridge
      coefficients. More stable for singular matrices than
      'cholesky'.

    - 'cholesky' uses the standard scipy.linalg.solve function to
      obtain a closed-form solution via a Cholesky decomposition of
      dot(X.T, X)

    - 'sparse_cg' uses the conjugate gradient solver as found in
      scipy.sparse.linalg.cg. As an iterative algorithm, this solver is
      more appropriate than 'cholesky' for large-scale data
      (possibility to set `tol` and `max_iter`).

    - 'lsqr' uses the dedicated regularized least-squares routine
      scipy.sparse.linalg.lsqr. It is the fastest and uses an iterative
      procedure.

    - 'sag' uses a Stochastic Average Gradient descent, and 'saga' uses
      its improved, unbiased version named SAGA. Both methods also use an
      iterative procedure, and are often faster than other solvers when
      both n_samples and n_features are large. Note that 'sag' and
      'saga' fast convergence is only guaranteed on features with
      approximately the same scale. You can preprocess the data with a
      scaler from sklearn.preprocessing.


    All last five solvers support both dense and sparse data. However, only
    'sag' and 'sparse_cg' supports sparse input when`fit_intercept` is
    True.

    .. versionadded:: 0.17
       Stochastic Average Gradient descent solver.
    .. versionadded:: 0.19
       SAGA solver.

max_iter : int, default=None
    Maximum number of iterations for conjugate gradient solver.
    For the 'sparse_cg' and 'lsqr' solvers, the default value is determined
    by scipy.sparse.linalg. For 'sag' and saga solver, the default value is
    1000.

tol : float, default=1e-3
    Precision of the solution.

verbose : int, default=0
    Verbosity level. Setting verbose > 0 will display additional
    information depending on the solver used.

random_state : int, RandomState instance, default=None
    The seed of the pseudo random number generator to use when shuffling
    the data.  If int, random_state is the seed used by the random number
    generator; If RandomState instance, random_state is the random number
    generator; If None, the random number generator is the RandomState
    instance used by `np.random`. Used when ``solver`` == 'sag'.

return_n_iter : bool, default=False
    If True, the method also returns `n_iter`, the actual number of
    iteration performed by the solver.

    .. versionadded:: 0.17

return_intercept : bool, default=False
    If True and if X is sparse, the method also returns the intercept,
    and the solver is automatically changed to 'sag'. This is only a
    temporary fix for fitting the intercept with sparse data. For dense
    data, use sklearn.linear_model._preprocess_data before your regression.

    .. versionadded:: 0.17

check_input : bool, default=True
    If False, the input arrays X and y will not be checked.

    .. versionadded:: 0.21

Returns
-------
coef : ndarray of shape (n_features,) or (n_targets, n_features)
    Weight vector(s).

n_iter : int, optional
    The actual number of iteration performed by the solver.
    Only returned if `return_n_iter` is True.

intercept : float or ndarray of shape (n_targets,)
    The intercept of the model. Only returned if `return_intercept`
    is True and if X is a scipy sparse array.

Notes
-----
This function won't compute the intercept.
*)

