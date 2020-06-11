(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module GaussianProcessClassifier : sig
type tag = [`GaussianProcessClassifier]
type t = [`BaseEstimator | `ClassifierMixin | `GaussianProcessClassifier | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_classifier : t -> [`ClassifierMixin] Obj.t
val as_estimator : t -> [`BaseEstimator] Obj.t
val create : ?kernel:Py.Object.t -> ?optimizer:[`Callable of Py.Object.t | `S of string] -> ?n_restarts_optimizer:int -> ?max_iter_predict:int -> ?warm_start:bool -> ?copy_X_train:bool -> ?random_state:int -> ?multi_class:string -> ?n_jobs:int -> unit -> t
(**
Gaussian process classification (GPC) based on Laplace approximation.

The implementation is based on Algorithm 3.1, 3.2, and 5.1 of
Gaussian Processes for Machine Learning (GPML) by Rasmussen and
Williams.

Internally, the Laplace approximation is used for approximating the
non-Gaussian posterior by a Gaussian.

Currently, the implementation is restricted to using the logistic link
function. For multi-class classification, several binary one-versus rest
classifiers are fitted. Note that this class thus does not implement
a true multi-class Laplace approximation.

Parameters
----------
kernel : kernel object
    The kernel specifying the covariance function of the GP. If None is
    passed, the kernel '1.0 * RBF(1.0)' is used as default. Note that
    the kernel's hyperparameters are optimized during fitting.

optimizer : string or callable, optional (default: 'fmin_l_bfgs_b')
    Can either be one of the internally supported optimizers for optimizing
    the kernel's parameters, specified by a string, or an externally
    defined optimizer passed as a callable. If a callable is passed, it
    must have the  signature::

        def optimizer(obj_func, initial_theta, bounds):
            # * 'obj_func' is the objective function to be maximized, which
            #   takes the hyperparameters theta as parameter and an
            #   optional flag eval_gradient, which determines if the
            #   gradient is returned additionally to the function value
            # * 'initial_theta': the initial value for theta, which can be
            #   used by local optimizers
            # * 'bounds': the bounds on the values of theta
            ....
            # Returned are the best found hyperparameters theta and
            # the corresponding value of the target function.
            return theta_opt, func_min

    Per default, the 'L-BFGS-B' algorithm from scipy.optimize.minimize
    is used. If None is passed, the kernel's parameters are kept fixed.
    Available internal optimizers are::

        'fmin_l_bfgs_b'

n_restarts_optimizer : int, optional (default: 0)
    The number of restarts of the optimizer for finding the kernel's
    parameters which maximize the log-marginal likelihood. The first run
    of the optimizer is performed from the kernel's initial parameters,
    the remaining ones (if any) from thetas sampled log-uniform randomly
    from the space of allowed theta-values. If greater than 0, all bounds
    must be finite. Note that n_restarts_optimizer=0 implies that one
    run is performed.

max_iter_predict : int, optional (default: 100)
    The maximum number of iterations in Newton's method for approximating
    the posterior during predict. Smaller values will reduce computation
    time at the cost of worse results.

warm_start : bool, optional (default: False)
    If warm-starts are enabled, the solution of the last Newton iteration
    on the Laplace approximation of the posterior mode is used as
    initialization for the next call of _posterior_mode(). This can speed
    up convergence when _posterior_mode is called several times on similar
    problems as in hyperparameter optimization. See :term:`the Glossary
    <warm_start>`.

copy_X_train : bool, optional (default: True)
    If True, a persistent copy of the training data is stored in the
    object. Otherwise, just a reference to the training data is stored,
    which might cause predictions to change if the data is modified
    externally.

random_state : int, RandomState instance or None, optional (default: None)
    The generator used to initialize the centers.
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

multi_class : string, default : 'one_vs_rest'
    Specifies how multi-class classification problems are handled.
    Supported are 'one_vs_rest' and 'one_vs_one'. In 'one_vs_rest',
    one binary Gaussian process classifier is fitted for each class, which
    is trained to separate this class from the rest. In 'one_vs_one', one
    binary Gaussian process classifier is fitted for each pair of classes,
    which is trained to separate these two classes. The predictions of
    these binary predictors are combined into multi-class predictions.
    Note that 'one_vs_one' does not support predicting probability
    estimates.

n_jobs : int or None, optional (default=None)
    The number of jobs to use for the computation.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

Attributes
----------
kernel_ : kernel object
    The kernel used for prediction. In case of binary classification,
    the structure of the kernel is the same as the one passed as parameter
    but with optimized hyperparameters. In case of multi-class
    classification, a CompoundKernel is returned which consists of the
    different kernels used in the one-versus-rest classifiers.

log_marginal_likelihood_value_ : float
    The log-marginal-likelihood of ``self.kernel_.theta``

classes_ : array-like of shape (n_classes,)
    Unique class labels.

n_classes_ : int
    The number of classes in the training data

Examples
--------
>>> from sklearn.datasets import load_iris
>>> from sklearn.gaussian_process import GaussianProcessClassifier
>>> from sklearn.gaussian_process.kernels import RBF
>>> X, y = load_iris(return_X_y=True)
>>> kernel = 1.0 * RBF(1.0)
>>> gpc = GaussianProcessClassifier(kernel=kernel,
...         random_state=0).fit(X, y)
>>> gpc.score(X, y)
0.9866...
>>> gpc.predict_proba(X[:2,:])
array([[0.83548752, 0.03228706, 0.13222543],
       [0.79064206, 0.06525643, 0.14410151]])

.. versionadded:: 0.18
*)

val fit : x:Py.Object.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fit Gaussian process classification model

Parameters
----------
X : sequence of length n_samples
    Feature vectors or other representations of training data.
    Could either be array-like with shape = (n_samples, n_features)
    or a list of objects.

y : array-like of shape (n_samples,)
    Target values, must be binary

Returns
-------
self : returns an instance of self.
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

val log_marginal_likelihood : ?theta:[>`ArrayLike] Np.Obj.t -> ?eval_gradient:bool -> ?clone_kernel:bool -> [> tag] Obj.t -> (float * [>`ArrayLike] Np.Obj.t)
(**
Returns log-marginal likelihood of theta for training data.

In the case of multi-class classification, the mean log-marginal
likelihood of the one-versus-rest classifiers are returned.

Parameters
----------
theta : array-like of shape (n_kernel_params,) or None
    Kernel hyperparameters for which the log-marginal likelihood is
    evaluated. In the case of multi-class classification, theta may
    be the  hyperparameters of the compound kernel or of an individual
    kernel. In the latter case, all individual kernel get assigned the
    same theta values. If None, the precomputed log_marginal_likelihood
    of ``self.kernel_.theta`` is returned.

eval_gradient : bool, default: False
    If True, the gradient of the log-marginal likelihood with respect
    to the kernel hyperparameters at position theta is returned
    additionally. Note that gradient computation is not supported
    for non-binary classification. If True, theta must not be None.

clone_kernel : bool, default=True
    If True, the kernel attribute is copied. If False, the kernel
    attribute is modified, but may result in a performance improvement.

Returns
-------
log_likelihood : float
    Log-marginal likelihood of theta for training data.

log_likelihood_gradient : array, shape = (n_kernel_params,), optional
    Gradient of the log-marginal likelihood with respect to the kernel
    hyperparameters at position theta.
    Only returned when eval_gradient is True.
*)

val predict : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Perform classification on an array of test vectors X.

Parameters
----------
X : sequence of length n_samples
    Query points where the GP is evaluated for classification.
    Could either be array-like with shape = (n_samples, n_features)
    or a list of objects.

Returns
-------
C : ndarray of shape (n_samples,)
    Predicted target values for X, values are from ``classes_``
*)

val predict_proba : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Return probability estimates for the test vector X.

Parameters
----------
X : sequence of length n_samples
    Query points where the GP is evaluated for classification.
    Could either be array-like with shape = (n_samples, n_features)
    or a list of objects.

Returns
-------
C : array-like of shape (n_samples, n_classes)
    Returns the probability of the samples for each class in
    the model. The columns correspond to the classes in sorted
    order, as they appear in the attribute :term:`classes_`.
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


(** Attribute kernel_: get value or raise Not_found if None.*)
val kernel_ : t -> Py.Object.t

(** Attribute kernel_: get value as an option. *)
val kernel_opt : t -> (Py.Object.t) option


(** Attribute log_marginal_likelihood_value_: get value or raise Not_found if None.*)
val log_marginal_likelihood_value_ : t -> float

(** Attribute log_marginal_likelihood_value_: get value as an option. *)
val log_marginal_likelihood_value_opt : t -> (float) option


(** Attribute classes_: get value or raise Not_found if None.*)
val classes_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute classes_: get value as an option. *)
val classes_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute n_classes_: get value or raise Not_found if None.*)
val n_classes_ : t -> int

(** Attribute n_classes_: get value as an option. *)
val n_classes_opt : t -> (int) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module GaussianProcessRegressor : sig
type tag = [`GaussianProcessRegressor]
type t = [`BaseEstimator | `GaussianProcessRegressor | `MultiOutputMixin | `Object | `RegressorMixin] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_regressor : t -> [`RegressorMixin] Obj.t
val as_multi_output : t -> [`MultiOutputMixin] Obj.t
val as_estimator : t -> [`BaseEstimator] Obj.t
val create : ?kernel:Py.Object.t -> ?alpha:[>`ArrayLike] Np.Obj.t -> ?optimizer:[`Callable of Py.Object.t | `S of string] -> ?n_restarts_optimizer:int -> ?normalize_y:bool -> ?copy_X_train:bool -> ?random_state:int -> unit -> t
(**
Gaussian process regression (GPR).

The implementation is based on Algorithm 2.1 of Gaussian Processes
for Machine Learning (GPML) by Rasmussen and Williams.

In addition to standard scikit-learn estimator API,
GaussianProcessRegressor:

   * allows prediction without prior fitting (based on the GP prior)
   * provides an additional method sample_y(X), which evaluates samples
     drawn from the GPR (prior or posterior) at given inputs
   * exposes a method log_marginal_likelihood(theta), which can be used
     externally for other ways of selecting hyperparameters, e.g., via
     Markov chain Monte Carlo.

Read more in the :ref:`User Guide <gaussian_process>`.

.. versionadded:: 0.18

Parameters
----------
kernel : kernel object
    The kernel specifying the covariance function of the GP. If None is
    passed, the kernel '1.0 * RBF(1.0)' is used as default. Note that
    the kernel's hyperparameters are optimized during fitting.

alpha : float or array-like, optional (default: 1e-10)
    Value added to the diagonal of the kernel matrix during fitting.
    Larger values correspond to increased noise level in the observations.
    This can also prevent a potential numerical issue during fitting, by
    ensuring that the calculated values form a positive definite matrix.
    If an array is passed, it must have the same number of entries as the
    data used for fitting and is used as datapoint-dependent noise level.
    Note that this is equivalent to adding a WhiteKernel with c=alpha.
    Allowing to specify the noise level directly as a parameter is mainly
    for convenience and for consistency with Ridge.

optimizer : string or callable, optional (default: 'fmin_l_bfgs_b')
    Can either be one of the internally supported optimizers for optimizing
    the kernel's parameters, specified by a string, or an externally
    defined optimizer passed as a callable. If a callable is passed, it
    must have the signature::

        def optimizer(obj_func, initial_theta, bounds):
            # * 'obj_func' is the objective function to be minimized, which
            #   takes the hyperparameters theta as parameter and an
            #   optional flag eval_gradient, which determines if the
            #   gradient is returned additionally to the function value
            # * 'initial_theta': the initial value for theta, which can be
            #   used by local optimizers
            # * 'bounds': the bounds on the values of theta
            ....
            # Returned are the best found hyperparameters theta and
            # the corresponding value of the target function.
            return theta_opt, func_min

    Per default, the 'L-BGFS-B' algorithm from scipy.optimize.minimize
    is used. If None is passed, the kernel's parameters are kept fixed.
    Available internal optimizers are::

        'fmin_l_bfgs_b'

n_restarts_optimizer : int, optional (default: 0)
    The number of restarts of the optimizer for finding the kernel's
    parameters which maximize the log-marginal likelihood. The first run
    of the optimizer is performed from the kernel's initial parameters,
    the remaining ones (if any) from thetas sampled log-uniform randomly
    from the space of allowed theta-values. If greater than 0, all bounds
    must be finite. Note that n_restarts_optimizer == 0 implies that one
    run is performed.

normalize_y : boolean, optional (default: False)
    Whether the target values y are normalized, i.e., the mean of the
    observed target values become zero. This parameter should be set to
    True if the target values' mean is expected to differ considerable from
    zero. When enabled, the normalization effectively modifies the GP's
    prior based on the data, which contradicts the likelihood principle;
    normalization is thus disabled per default.

copy_X_train : bool, optional (default: True)
    If True, a persistent copy of the training data is stored in the
    object. Otherwise, just a reference to the training data is stored,
    which might cause predictions to change if the data is modified
    externally.

random_state : int, RandomState instance or None, optional (default: None)
    The generator used to initialize the centers. If int, random_state is
    the seed used by the random number generator; If RandomState instance,
    random_state is the random number generator; If None, the random number
    generator is the RandomState instance used by `np.random`.

Attributes
----------
X_train_ : sequence of length n_samples
    Feature vectors or other representations of training data (also
    required for prediction). Could either be array-like with shape =
    (n_samples, n_features) or a list of objects.

y_train_ : array-like of shape (n_samples,) or (n_samples, n_targets)
    Target values in training data (also required for prediction)

kernel_ : kernel object
    The kernel used for prediction. The structure of the kernel is the
    same as the one passed as parameter but with optimized hyperparameters

L_ : array-like of shape (n_samples, n_samples)
    Lower-triangular Cholesky decomposition of the kernel in ``X_train_``

alpha_ : array-like of shape (n_samples,)
    Dual coefficients of training data points in kernel space

log_marginal_likelihood_value_ : float
    The log-marginal-likelihood of ``self.kernel_.theta``

Examples
--------
>>> from sklearn.datasets import make_friedman2
>>> from sklearn.gaussian_process import GaussianProcessRegressor
>>> from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
>>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
>>> kernel = DotProduct() + WhiteKernel()
>>> gpr = GaussianProcessRegressor(kernel=kernel,
...         random_state=0).fit(X, y)
>>> gpr.score(X, y)
0.3680...
>>> gpr.predict(X[:2,:], return_std=True)
(array([653.0..., 592.1...]), array([316.6..., 316.6...]))
*)

val fit : x:Py.Object.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fit Gaussian process regression model.

Parameters
----------
X : sequence of length n_samples
    Feature vectors or other representations of training data.
    Could either be array-like with shape = (n_samples, n_features)
    or a list of objects.

y : array-like of shape (n_samples,) or (n_samples, n_targets)
    Target values

Returns
-------
self : returns an instance of self.
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

val log_marginal_likelihood : ?theta:[>`ArrayLike] Np.Obj.t -> ?eval_gradient:bool -> ?clone_kernel:bool -> [> tag] Obj.t -> (float * [>`ArrayLike] Np.Obj.t)
(**
Returns log-marginal likelihood of theta for training data.

Parameters
----------
theta : array-like of shape (n_kernel_params,) or None
    Kernel hyperparameters for which the log-marginal likelihood is
    evaluated. If None, the precomputed log_marginal_likelihood
    of ``self.kernel_.theta`` is returned.

eval_gradient : bool, default: False
    If True, the gradient of the log-marginal likelihood with respect
    to the kernel hyperparameters at position theta is returned
    additionally. If True, theta must not be None.

clone_kernel : bool, default=True
    If True, the kernel attribute is copied. If False, the kernel
    attribute is modified, but may result in a performance improvement.

Returns
-------
log_likelihood : float
    Log-marginal likelihood of theta for training data.

log_likelihood_gradient : array, shape = (n_kernel_params,), optional
    Gradient of the log-marginal likelihood with respect to the kernel
    hyperparameters at position theta.
    Only returned when eval_gradient is True.
*)

val predict : ?return_std:bool -> ?return_cov:bool -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Predict using the Gaussian process regression model

We can also predict based on an unfitted model by using the GP prior.
In addition to the mean of the predictive distribution, also its
standard deviation (return_std=True) or covariance (return_cov=True).
Note that at most one of the two can be requested.

Parameters
----------
X : sequence of length n_samples
    Query points where the GP is evaluated.
    Could either be array-like with shape = (n_samples, n_features)
    or a list of objects.

return_std : bool, default: False
    If True, the standard-deviation of the predictive distribution at
    the query points is returned along with the mean.

return_cov : bool, default: False
    If True, the covariance of the joint predictive distribution at
    the query points is returned along with the mean

Returns
-------
y_mean : array, shape = (n_samples, [n_output_dims])
    Mean of predictive distribution a query points

y_std : array, shape = (n_samples,), optional
    Standard deviation of predictive distribution at query points.
    Only returned when return_std is True.

y_cov : array, shape = (n_samples, n_samples), optional
    Covariance of joint predictive distribution a query points.
    Only returned when return_cov is True.
*)

val sample_y : ?n_samples:int -> ?random_state:int -> x:Py.Object.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Draw samples from Gaussian process and evaluate at X.

Parameters
----------
X : sequence of length n_samples
    Query points where the GP is evaluated.
    Could either be array-like with shape = (n_samples, n_features)
    or a list of objects.

n_samples : int, default: 1
    The number of samples drawn from the Gaussian process

random_state : int, RandomState instance or None, optional (default=0)
    If int, random_state is the seed used by the random number
    generator; If RandomState instance, random_state is the
    random number generator; If None, the random number
    generator is the RandomState instance used by `np.random`.

Returns
-------
y_samples : array, shape = (n_samples_X, [n_output_dims], n_samples)
    Values of n_samples samples drawn from Gaussian process and
    evaluated at query points.
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


(** Attribute X_train_: get value or raise Not_found if None.*)
val x_train_ : t -> Py.Object.t

(** Attribute X_train_: get value as an option. *)
val x_train_opt : t -> (Py.Object.t) option


(** Attribute y_train_: get value or raise Not_found if None.*)
val y_train_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute y_train_: get value as an option. *)
val y_train_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute kernel_: get value or raise Not_found if None.*)
val kernel_ : t -> Py.Object.t

(** Attribute kernel_: get value as an option. *)
val kernel_opt : t -> (Py.Object.t) option


(** Attribute L_: get value or raise Not_found if None.*)
val l_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute L_: get value as an option. *)
val l_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute alpha_: get value or raise Not_found if None.*)
val alpha_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute alpha_: get value as an option. *)
val alpha_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute log_marginal_likelihood_value_: get value or raise Not_found if None.*)
val log_marginal_likelihood_value_ : t -> float

(** Attribute log_marginal_likelihood_value_: get value as an option. *)
val log_marginal_likelihood_value_opt : t -> (float) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Kernels : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module CompoundKernel : sig
type tag = [`CompoundKernel]
type t = [`CompoundKernel | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : Py.Object.t -> t
(**
Kernel which is composed of a set of other kernels.

.. versionadded:: 0.18

Parameters
----------
kernels : list of Kernel objects
    The other kernels
*)

val clone_with_theta : theta:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a clone of self with given hyperparameters theta.

Parameters
----------
theta : array, shape (n_dims,)
    The hyperparameters
*)

val diag : x:Py.Object.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Returns the diagonal of the kernel k(X, X).

The result of this method is identical to np.diag(self(X)); however,
it can be evaluated more efficiently since only the diagonal is
evaluated.

Parameters
----------
X : sequence of length n_samples_X
    Argument to the kernel. Could either be array-like with
    shape = (n_samples_X, n_features) or a list of objects.

Returns
-------
K_diag : array, shape (n_samples_X, n_kernels)
    Diagonal of kernel k(X, X)
*)

val get_params : ?deep:bool -> [> tag] Obj.t -> Dict.t
(**
Get parameters of this kernel.

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

val is_stationary : [> tag] Obj.t -> Py.Object.t
(**
Returns whether the kernel is stationary. 
*)

val set_params : ?params:(string * Py.Object.t) list -> [> tag] Obj.t -> t
(**
Set the parameters of this kernel.

The method works on simple kernels as well as on nested kernels.
The latter have parameters of the form ``<component>__<parameter>``
so that it's possible to update each component of a nested object.

Returns
-------
self
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module ConstantKernel : sig
type tag = [`ConstantKernel]
type t = [`ConstantKernel | `GenericKernelMixin | `Object | `StationaryKernelMixin] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_stationary_kernel : t -> [`StationaryKernelMixin] Obj.t
val as_generic_kernel : t -> [`GenericKernelMixin] Obj.t
val create : ?constant_value:float -> ?constant_value_bounds:(float * float) -> unit -> t
(**
Constant kernel.

Can be used as part of a product-kernel where it scales the magnitude of
the other factor (kernel) or as part of a sum-kernel, where it modifies
the mean of the Gaussian process.

k(x_1, x_2) = constant_value for all x_1, x_2

.. versionadded:: 0.18

Parameters
----------
constant_value : float, default: 1.0
    The constant value which defines the covariance:
    k(x_1, x_2) = constant_value

constant_value_bounds : pair of floats >= 0, default: (1e-5, 1e5)
    The lower and upper bound on constant_value
*)

val clone_with_theta : theta:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a clone of self with given hyperparameters theta.

Parameters
----------
theta : array, shape (n_dims,)
    The hyperparameters
*)

val diag : x:Py.Object.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Returns the diagonal of the kernel k(X, X).

The result of this method is identical to np.diag(self(X)); however,
it can be evaluated more efficiently since only the diagonal is
evaluated.

Parameters
----------
X : sequence of length n_samples_X
    Argument to the kernel. Could either be array-like with
    shape = (n_samples_X, n_features) or a list of objects.

Returns
-------
K_diag : array, shape (n_samples_X,)
    Diagonal of kernel k(X, X)
*)

val get_params : ?deep:bool -> [> tag] Obj.t -> Dict.t
(**
Get parameters of this kernel.

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

val is_stationary : [> tag] Obj.t -> Py.Object.t
(**
Returns whether the kernel is stationary. 
*)

val set_params : ?params:(string * Py.Object.t) list -> [> tag] Obj.t -> t
(**
Set the parameters of this kernel.

The method works on simple kernels as well as on nested kernels.
The latter have parameters of the form ``<component>__<parameter>``
so that it's possible to update each component of a nested object.

Returns
-------
self
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module DotProduct : sig
type tag = [`DotProduct]
type t = [`DotProduct | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?sigma_0:float -> ?sigma_0_bounds:(float * float) -> unit -> t
(**
Dot-Product kernel.

The DotProduct kernel is non-stationary and can be obtained from linear
regression by putting N(0, 1) priors on the coefficients of x_d (d = 1, . .
. , D) and a prior of N(0, \sigma_0^2) on the bias. The DotProduct kernel
is invariant to a rotation of the coordinates about the origin, but not
translations. It is parameterized by a parameter sigma_0^2. For
sigma_0^2 =0, the kernel is called the homogeneous linear kernel, otherwise
it is inhomogeneous. The kernel is given by

k(x_i, x_j) = sigma_0 ^ 2 + x_i \cdot x_j

The DotProduct kernel is commonly combined with exponentiation.

.. versionadded:: 0.18

Parameters
----------
sigma_0 : float >= 0, default: 1.0
    Parameter controlling the inhomogenity of the kernel. If sigma_0=0,
    the kernel is homogenous.

sigma_0_bounds : pair of floats >= 0, default: (1e-5, 1e5)
    The lower and upper bound on l
*)

val clone_with_theta : theta:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a clone of self with given hyperparameters theta.

Parameters
----------
theta : array, shape (n_dims,)
    The hyperparameters
*)

val diag : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Returns the diagonal of the kernel k(X, X).

The result of this method is identical to np.diag(self(X)); however,
it can be evaluated more efficiently since only the diagonal is
evaluated.

Parameters
----------
X : array, shape (n_samples_X, n_features)
    Left argument of the returned kernel k(X, Y)

Returns
-------
K_diag : array, shape (n_samples_X,)
    Diagonal of kernel k(X, X)
*)

val get_params : ?deep:bool -> [> tag] Obj.t -> Dict.t
(**
Get parameters of this kernel.

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

val is_stationary : [> tag] Obj.t -> Py.Object.t
(**
Returns whether the kernel is stationary. 
*)

val set_params : ?params:(string * Py.Object.t) list -> [> tag] Obj.t -> t
(**
Set the parameters of this kernel.

The method works on simple kernels as well as on nested kernels.
The latter have parameters of the form ``<component>__<parameter>``
so that it's possible to update each component of a nested object.

Returns
-------
self
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module ExpSineSquared : sig
type tag = [`ExpSineSquared]
type t = [`ExpSineSquared | `NormalizedKernelMixin | `Object | `StationaryKernelMixin] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_stationary_kernel : t -> [`StationaryKernelMixin] Obj.t
val as_normalized_kernel : t -> [`NormalizedKernelMixin] Obj.t
val create : ?length_scale:float -> ?periodicity:float -> ?length_scale_bounds:(float * float) -> ?periodicity_bounds:(float * float) -> unit -> t
(**
Exp-Sine-Squared kernel.

The ExpSineSquared kernel allows modeling periodic functions. It is
parameterized by a length-scale parameter length_scale>0 and a periodicity
parameter periodicity>0. Only the isotropic variant where l is a scalar is
supported at the moment. The kernel given by:

k(x_i, x_j) =
exp(-2 (sin(\pi / periodicity * d(x_i, x_j)) / length_scale) ^ 2)

.. versionadded:: 0.18

Parameters
----------
length_scale : float > 0, default: 1.0
    The length scale of the kernel.

periodicity : float > 0, default: 1.0
    The periodicity of the kernel.

length_scale_bounds : pair of floats >= 0, default: (1e-5, 1e5)
    The lower and upper bound on length_scale

periodicity_bounds : pair of floats >= 0, default: (1e-5, 1e5)
    The lower and upper bound on periodicity
*)

val clone_with_theta : theta:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a clone of self with given hyperparameters theta.

Parameters
----------
theta : array, shape (n_dims,)
    The hyperparameters
*)

val diag : x:Py.Object.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Returns the diagonal of the kernel k(X, X).

The result of this method is identical to np.diag(self(X)); however,
it can be evaluated more efficiently since only the diagonal is
evaluated.

Parameters
----------
X : sequence of length n_samples
    Left argument of the returned kernel k(X, Y)

Returns
-------
K_diag : array, shape (n_samples_X,)
    Diagonal of kernel k(X, X)
*)

val get_params : ?deep:bool -> [> tag] Obj.t -> Dict.t
(**
Get parameters of this kernel.

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

val is_stationary : [> tag] Obj.t -> Py.Object.t
(**
Returns whether the kernel is stationary. 
*)

val set_params : ?params:(string * Py.Object.t) list -> [> tag] Obj.t -> t
(**
Set the parameters of this kernel.

The method works on simple kernels as well as on nested kernels.
The latter have parameters of the form ``<component>__<parameter>``
so that it's possible to update each component of a nested object.

Returns
-------
self
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Exponentiation : sig
type tag = [`Exponentiation]
type t = [`Exponentiation | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : kernel:Py.Object.t -> exponent:float -> unit -> t
(**
Exponentiate kernel by given exponent.

The resulting kernel is defined as
k_exp(X, Y) = k(X, Y) ** exponent

.. versionadded:: 0.18

Parameters
----------
kernel : Kernel object
    The base kernel

exponent : float
    The exponent for the base kernel
*)

val clone_with_theta : theta:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a clone of self with given hyperparameters theta.

Parameters
----------
theta : array, shape (n_dims,)
    The hyperparameters
*)

val diag : x:Py.Object.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Returns the diagonal of the kernel k(X, X).

The result of this method is identical to np.diag(self(X)); however,
it can be evaluated more efficiently since only the diagonal is
evaluated.

Parameters
----------
X : sequence of length n_samples_X
    Argument to the kernel. Could either be array-like with
    shape = (n_samples_X, n_features) or a list of objects.

Returns
-------
K_diag : array, shape (n_samples_X,)
    Diagonal of kernel k(X, X)
*)

val get_params : ?deep:bool -> [> tag] Obj.t -> Dict.t
(**
Get parameters of this kernel.

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

val is_stationary : [> tag] Obj.t -> Py.Object.t
(**
Returns whether the kernel is stationary. 
*)

val set_params : ?params:(string * Py.Object.t) list -> [> tag] Obj.t -> t
(**
Set the parameters of this kernel.

The method works on simple kernels as well as on nested kernels.
The latter have parameters of the form ``<component>__<parameter>``
so that it's possible to update each component of a nested object.

Returns
-------
self
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module GenericKernelMixin : sig
type tag = [`GenericKernelMixin]
type t = [`GenericKernelMixin | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : unit -> t
(**
Mixin for kernels which operate on generic objects such as variable-
length sequences, trees, and graphs.

.. versionadded:: 0.22
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Hyperparameter : sig
type tag = [`Hyperparameter]
type t = [`Hyperparameter | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?n_elements:Py.Object.t -> ?fixed:Py.Object.t -> name:Py.Object.t -> value_type:Py.Object.t -> bounds:Py.Object.t -> unit -> t
(**
A kernel hyperparameter's specification in form of a namedtuple.

.. versionadded:: 0.18

Attributes
----------
name : string
    The name of the hyperparameter. Note that a kernel using a
    hyperparameter with name 'x' must have the attributes self.x and
    self.x_bounds

value_type : string
    The type of the hyperparameter. Currently, only 'numeric'
    hyperparameters are supported.

bounds : pair of floats >= 0 or 'fixed'
    The lower and upper bound on the parameter. If n_elements>1, a pair
    of 1d array with n_elements each may be given alternatively. If
    the string 'fixed' is passed as bounds, the hyperparameter's value
    cannot be changed.

n_elements : int, default=1
    The number of elements of the hyperparameter value. Defaults to 1,
    which corresponds to a scalar hyperparameter. n_elements > 1
    corresponds to a hyperparameter which is vector-valued,
    such as, e.g., anisotropic length-scales.

fixed : bool, default: None
    Whether the value of this hyperparameter is fixed, i.e., cannot be
    changed during hyperparameter tuning. If None is passed, the 'fixed' is
    derived based on the given bounds.
*)

val get_item : key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return self[key].
*)

val iter : [> tag] Obj.t -> Dict.t Seq.t
(**
Implement iter(self).
*)

val count : value:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return number of occurrences of value.
*)

val index : ?start:Py.Object.t -> ?stop:Py.Object.t -> value:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return first index of value.

Raises ValueError if the value is not present.
*)


(** Attribute name: get value or raise Not_found if None.*)
val name : t -> string

(** Attribute name: get value as an option. *)
val name_opt : t -> (string) option


(** Attribute value_type: get value or raise Not_found if None.*)
val value_type : t -> string

(** Attribute value_type: get value as an option. *)
val value_type_opt : t -> (string) option


(** Attribute bounds: get value or raise Not_found if None.*)
val bounds : t -> Py.Object.t

(** Attribute bounds: get value as an option. *)
val bounds_opt : t -> (Py.Object.t) option


(** Attribute n_elements: get value or raise Not_found if None.*)
val n_elements : t -> int

(** Attribute n_elements: get value as an option. *)
val n_elements_opt : t -> (int) option


(** Attribute fixed: get value or raise Not_found if None.*)
val fixed : t -> bool

(** Attribute fixed: get value as an option. *)
val fixed_opt : t -> (bool) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Kernel : sig
type tag = [`Kernel]
type t = [`Kernel | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val clone_with_theta : theta:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a clone of self with given hyperparameters theta.

Parameters
----------
theta : array, shape (n_dims,)
    The hyperparameters
*)

val diag : x:Py.Object.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Returns the diagonal of the kernel k(X, X).

The result of this method is identical to np.diag(self(X)); however,
it can be evaluated more efficiently since only the diagonal is
evaluated.

Parameters
----------
X : sequence of length n_samples
    Left argument of the returned kernel k(X, Y)

Returns
-------
K_diag : array, shape (n_samples_X,)
    Diagonal of kernel k(X, X)
*)

val get_params : ?deep:bool -> [> tag] Obj.t -> Dict.t
(**
Get parameters of this kernel.

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

val is_stationary : [> tag] Obj.t -> Py.Object.t
(**
Returns whether the kernel is stationary. 
*)

val set_params : ?params:(string * Py.Object.t) list -> [> tag] Obj.t -> t
(**
Set the parameters of this kernel.

The method works on simple kernels as well as on nested kernels.
The latter have parameters of the form ``<component>__<parameter>``
so that it's possible to update each component of a nested object.

Returns
-------
self
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module KernelOperator : sig
type tag = [`KernelOperator]
type t = [`KernelOperator | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val clone_with_theta : theta:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a clone of self with given hyperparameters theta.

Parameters
----------
theta : array, shape (n_dims,)
    The hyperparameters
*)

val diag : x:Py.Object.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Returns the diagonal of the kernel k(X, X).

The result of this method is identical to np.diag(self(X)); however,
it can be evaluated more efficiently since only the diagonal is
evaluated.

Parameters
----------
X : sequence of length n_samples
    Left argument of the returned kernel k(X, Y)

Returns
-------
K_diag : array, shape (n_samples_X,)
    Diagonal of kernel k(X, X)
*)

val get_params : ?deep:bool -> [> tag] Obj.t -> Dict.t
(**
Get parameters of this kernel.

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

val is_stationary : [> tag] Obj.t -> Py.Object.t
(**
Returns whether the kernel is stationary. 
*)

val set_params : ?params:(string * Py.Object.t) list -> [> tag] Obj.t -> t
(**
Set the parameters of this kernel.

The method works on simple kernels as well as on nested kernels.
The latter have parameters of the form ``<component>__<parameter>``
so that it's possible to update each component of a nested object.

Returns
-------
self
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Matern : sig
type tag = [`Matern]
type t = [`Matern | `NormalizedKernelMixin | `Object | `StationaryKernelMixin] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_stationary_kernel : t -> [`StationaryKernelMixin] Obj.t
val as_normalized_kernel : t -> [`NormalizedKernelMixin] Obj.t
val create : ?length_scale:[>`ArrayLike] Np.Obj.t -> ?length_scale_bounds:(float * float) -> ?nu:float -> unit -> t
(**
Matern kernel.

The class of Matern kernels is a generalization of the RBF and the
absolute exponential kernel parameterized by an additional parameter
nu. The smaller nu, the less smooth the approximated function is.
For nu=inf, the kernel becomes equivalent to the RBF kernel and for nu=0.5
to the absolute exponential kernel. Important intermediate values are
nu=1.5 (once differentiable functions) and nu=2.5 (twice differentiable
functions).

See Rasmussen and Williams 2006, pp84 for details regarding the
different variants of the Matern kernel.

.. versionadded:: 0.18

Parameters
----------
length_scale : float or array with shape (n_features,), default: 1.0
    The length scale of the kernel. If a float, an isotropic kernel is
    used. If an array, an anisotropic kernel is used where each dimension
    of l defines the length-scale of the respective feature dimension.

length_scale_bounds : pair of floats >= 0, default: (1e-5, 1e5)
    The lower and upper bound on length_scale

nu : float, default: 1.5
    The parameter nu controlling the smoothness of the learned function.
    The smaller nu, the less smooth the approximated function is.
    For nu=inf, the kernel becomes equivalent to the RBF kernel and for
    nu=0.5 to the absolute exponential kernel. Important intermediate
    values are nu=1.5 (once differentiable functions) and nu=2.5
    (twice differentiable functions). Note that values of nu not in
    [0.5, 1.5, 2.5, inf] incur a considerably higher computational cost
    (appr. 10 times higher) since they require to evaluate the modified
    Bessel function. Furthermore, in contrast to l, nu is kept fixed to
    its initial value and not optimized.
*)

val clone_with_theta : theta:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a clone of self with given hyperparameters theta.

Parameters
----------
theta : array, shape (n_dims,)
    The hyperparameters
*)

val diag : x:Py.Object.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Returns the diagonal of the kernel k(X, X).

The result of this method is identical to np.diag(self(X)); however,
it can be evaluated more efficiently since only the diagonal is
evaluated.

Parameters
----------
X : sequence of length n_samples
    Left argument of the returned kernel k(X, Y)

Returns
-------
K_diag : array, shape (n_samples_X,)
    Diagonal of kernel k(X, X)
*)

val get_params : ?deep:bool -> [> tag] Obj.t -> Dict.t
(**
Get parameters of this kernel.

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

val is_stationary : [> tag] Obj.t -> Py.Object.t
(**
Returns whether the kernel is stationary. 
*)

val set_params : ?params:(string * Py.Object.t) list -> [> tag] Obj.t -> t
(**
Set the parameters of this kernel.

The method works on simple kernels as well as on nested kernels.
The latter have parameters of the form ``<component>__<parameter>``
so that it's possible to update each component of a nested object.

Returns
-------
self
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module NormalizedKernelMixin : sig
type tag = [`NormalizedKernelMixin]
type t = [`NormalizedKernelMixin | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : unit -> t
(**
Mixin for kernels which are normalized: k(X, X)=1.

.. versionadded:: 0.18
*)

val diag : x:Py.Object.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Returns the diagonal of the kernel k(X, X).

The result of this method is identical to np.diag(self(X)); however,
it can be evaluated more efficiently since only the diagonal is
evaluated.

Parameters
----------
X : sequence of length n_samples
    Left argument of the returned kernel k(X, Y)

Returns
-------
K_diag : array, shape (n_samples_X,)
    Diagonal of kernel k(X, X)
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module PairwiseKernel : sig
type tag = [`PairwiseKernel]
type t = [`Object | `PairwiseKernel] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?gamma:float -> ?gamma_bounds:(float * float) -> ?metric:[`Callable of Py.Object.t | `S of string] -> ?pairwise_kernels_kwargs:Dict.t -> unit -> t
(**
Wrapper for kernels in sklearn.metrics.pairwise.

A thin wrapper around the functionality of the kernels in
sklearn.metrics.pairwise.

Note: Evaluation of eval_gradient is not analytic but numeric and all
      kernels support only isotropic distances. The parameter gamma is
      considered to be a hyperparameter and may be optimized. The other
      kernel parameters are set directly at initialization and are kept
      fixed.

.. versionadded:: 0.18

Parameters
----------
gamma : float >= 0, default: 1.0
    Parameter gamma of the pairwise kernel specified by metric

gamma_bounds : pair of floats >= 0, default: (1e-5, 1e5)
    The lower and upper bound on gamma

metric : string, or callable, default: 'linear'
    The metric to use when calculating kernel between instances in a
    feature array. If metric is a string, it must be one of the metrics
    in pairwise.PAIRWISE_KERNEL_FUNCTIONS.
    If metric is 'precomputed', X is assumed to be a kernel matrix.
    Alternatively, if metric is a callable function, it is called on each
    pair of instances (rows) and the resulting value recorded. The callable
    should take two arrays from X as input and return a value indicating
    the distance between them.

pairwise_kernels_kwargs : dict, default: None
    All entries of this dict (if any) are passed as keyword arguments to
    the pairwise kernel function.
*)

val clone_with_theta : theta:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a clone of self with given hyperparameters theta.

Parameters
----------
theta : array, shape (n_dims,)
    The hyperparameters
*)

val diag : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Returns the diagonal of the kernel k(X, X).

The result of this method is identical to np.diag(self(X)); however,
it can be evaluated more efficiently since only the diagonal is
evaluated.

Parameters
----------
X : array, shape (n_samples_X, n_features)
    Left argument of the returned kernel k(X, Y)

Returns
-------
K_diag : array, shape (n_samples_X,)
    Diagonal of kernel k(X, X)
*)

val get_params : ?deep:bool -> [> tag] Obj.t -> Dict.t
(**
Get parameters of this kernel.

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

val is_stationary : [> tag] Obj.t -> Py.Object.t
(**
Returns whether the kernel is stationary. 
*)

val set_params : ?params:(string * Py.Object.t) list -> [> tag] Obj.t -> t
(**
Set the parameters of this kernel.

The method works on simple kernels as well as on nested kernels.
The latter have parameters of the form ``<component>__<parameter>``
so that it's possible to update each component of a nested object.

Returns
-------
self
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Product : sig
type tag = [`Product]
type t = [`Object | `Product] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : k1:Py.Object.t -> k2:Py.Object.t -> unit -> t
(**
Product-kernel k1 * k2 of two kernels k1 and k2.

The resulting kernel is defined as
k_prod(X, Y) = k1(X, Y) * k2(X, Y)

.. versionadded:: 0.18

Parameters
----------
k1 : Kernel object
    The first base-kernel of the product-kernel

k2 : Kernel object
    The second base-kernel of the product-kernel
*)

val clone_with_theta : theta:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a clone of self with given hyperparameters theta.

Parameters
----------
theta : array, shape (n_dims,)
    The hyperparameters
*)

val diag : x:Py.Object.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Returns the diagonal of the kernel k(X, X).

The result of this method is identical to np.diag(self(X)); however,
it can be evaluated more efficiently since only the diagonal is
evaluated.

Parameters
----------
X : sequence of length n_samples_X
    Argument to the kernel. Could either be array-like with
    shape = (n_samples_X, n_features) or a list of objects.

Returns
-------
K_diag : array, shape (n_samples_X,)
    Diagonal of kernel k(X, X)
*)

val get_params : ?deep:bool -> [> tag] Obj.t -> Dict.t
(**
Get parameters of this kernel.

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

val is_stationary : [> tag] Obj.t -> Py.Object.t
(**
Returns whether the kernel is stationary. 
*)

val set_params : ?params:(string * Py.Object.t) list -> [> tag] Obj.t -> t
(**
Set the parameters of this kernel.

The method works on simple kernels as well as on nested kernels.
The latter have parameters of the form ``<component>__<parameter>``
so that it's possible to update each component of a nested object.

Returns
-------
self
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module RBF : sig
type tag = [`RBF]
type t = [`NormalizedKernelMixin | `Object | `RBF | `StationaryKernelMixin] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_stationary_kernel : t -> [`StationaryKernelMixin] Obj.t
val as_normalized_kernel : t -> [`NormalizedKernelMixin] Obj.t
val create : ?length_scale:[>`ArrayLike] Np.Obj.t -> ?length_scale_bounds:(float * float) -> unit -> t
(**
Radial-basis function kernel (aka squared-exponential kernel).

The RBF kernel is a stationary kernel. It is also known as the
'squared exponential' kernel. It is parameterized by a length-scale
parameter length_scale>0, which can either be a scalar (isotropic variant
of the kernel) or a vector with the same number of dimensions as the inputs
X (anisotropic variant of the kernel). The kernel is given by:

k(x_i, x_j) = exp(-1 / 2 d(x_i / length_scale, x_j / length_scale)^2)

This kernel is infinitely differentiable, which implies that GPs with this
kernel as covariance function have mean square derivatives of all orders,
and are thus very smooth.

.. versionadded:: 0.18

Parameters
----------
length_scale : float or array with shape (n_features,), default: 1.0
    The length scale of the kernel. If a float, an isotropic kernel is
    used. If an array, an anisotropic kernel is used where each dimension
    of l defines the length-scale of the respective feature dimension.

length_scale_bounds : pair of floats >= 0, default: (1e-5, 1e5)
    The lower and upper bound on length_scale
*)

val clone_with_theta : theta:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a clone of self with given hyperparameters theta.

Parameters
----------
theta : array, shape (n_dims,)
    The hyperparameters
*)

val diag : x:Py.Object.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Returns the diagonal of the kernel k(X, X).

The result of this method is identical to np.diag(self(X)); however,
it can be evaluated more efficiently since only the diagonal is
evaluated.

Parameters
----------
X : sequence of length n_samples
    Left argument of the returned kernel k(X, Y)

Returns
-------
K_diag : array, shape (n_samples_X,)
    Diagonal of kernel k(X, X)
*)

val get_params : ?deep:bool -> [> tag] Obj.t -> Dict.t
(**
Get parameters of this kernel.

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

val is_stationary : [> tag] Obj.t -> Py.Object.t
(**
Returns whether the kernel is stationary. 
*)

val set_params : ?params:(string * Py.Object.t) list -> [> tag] Obj.t -> t
(**
Set the parameters of this kernel.

The method works on simple kernels as well as on nested kernels.
The latter have parameters of the form ``<component>__<parameter>``
so that it's possible to update each component of a nested object.

Returns
-------
self
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module RationalQuadratic : sig
type tag = [`RationalQuadratic]
type t = [`NormalizedKernelMixin | `Object | `RationalQuadratic | `StationaryKernelMixin] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_stationary_kernel : t -> [`StationaryKernelMixin] Obj.t
val as_normalized_kernel : t -> [`NormalizedKernelMixin] Obj.t
val create : ?length_scale:float -> ?alpha:float -> ?length_scale_bounds:(float * float) -> ?alpha_bounds:(float * float) -> unit -> t
(**
Rational Quadratic kernel.

The RationalQuadratic kernel can be seen as a scale mixture (an infinite
sum) of RBF kernels with different characteristic length-scales. It is
parameterized by a length-scale parameter length_scale>0 and a scale
mixture parameter alpha>0. Only the isotropic variant where length_scale is
a scalar is supported at the moment. The kernel given by:

k(x_i, x_j) = (1 + d(x_i, x_j)^2 / (2*alpha * length_scale^2))^-alpha

.. versionadded:: 0.18

Parameters
----------
length_scale : float > 0, default: 1.0
    The length scale of the kernel.

alpha : float > 0, default: 1.0
    Scale mixture parameter

length_scale_bounds : pair of floats >= 0, default: (1e-5, 1e5)
    The lower and upper bound on length_scale

alpha_bounds : pair of floats >= 0, default: (1e-5, 1e5)
    The lower and upper bound on alpha
*)

val clone_with_theta : theta:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a clone of self with given hyperparameters theta.

Parameters
----------
theta : array, shape (n_dims,)
    The hyperparameters
*)

val diag : x:Py.Object.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Returns the diagonal of the kernel k(X, X).

The result of this method is identical to np.diag(self(X)); however,
it can be evaluated more efficiently since only the diagonal is
evaluated.

Parameters
----------
X : sequence of length n_samples
    Left argument of the returned kernel k(X, Y)

Returns
-------
K_diag : array, shape (n_samples_X,)
    Diagonal of kernel k(X, X)
*)

val get_params : ?deep:bool -> [> tag] Obj.t -> Dict.t
(**
Get parameters of this kernel.

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

val is_stationary : [> tag] Obj.t -> Py.Object.t
(**
Returns whether the kernel is stationary. 
*)

val set_params : ?params:(string * Py.Object.t) list -> [> tag] Obj.t -> t
(**
Set the parameters of this kernel.

The method works on simple kernels as well as on nested kernels.
The latter have parameters of the form ``<component>__<parameter>``
so that it's possible to update each component of a nested object.

Returns
-------
self
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module StationaryKernelMixin : sig
type tag = [`StationaryKernelMixin]
type t = [`Object | `StationaryKernelMixin] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : unit -> t
(**
Mixin for kernels which are stationary: k(X, Y)= f(X-Y).

.. versionadded:: 0.18
*)

val is_stationary : [> tag] Obj.t -> Py.Object.t
(**
Returns whether the kernel is stationary. 
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Sum : sig
type tag = [`Sum]
type t = [`Object | `Sum] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : k1:Py.Object.t -> k2:Py.Object.t -> unit -> t
(**
Sum-kernel k1 + k2 of two kernels k1 and k2.

The resulting kernel is defined as
k_sum(X, Y) = k1(X, Y) + k2(X, Y)

.. versionadded:: 0.18

Parameters
----------
k1 : Kernel object
    The first base-kernel of the sum-kernel

k2 : Kernel object
    The second base-kernel of the sum-kernel
*)

val clone_with_theta : theta:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a clone of self with given hyperparameters theta.

Parameters
----------
theta : array, shape (n_dims,)
    The hyperparameters
*)

val diag : x:Py.Object.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Returns the diagonal of the kernel k(X, X).

The result of this method is identical to np.diag(self(X)); however,
it can be evaluated more efficiently since only the diagonal is
evaluated.

Parameters
----------
X : sequence of length n_samples_X
    Argument to the kernel. Could either be array-like with
    shape = (n_samples_X, n_features) or a list of objects.

Returns
-------
K_diag : array, shape (n_samples_X,)
    Diagonal of kernel k(X, X)
*)

val get_params : ?deep:bool -> [> tag] Obj.t -> Dict.t
(**
Get parameters of this kernel.

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

val is_stationary : [> tag] Obj.t -> Py.Object.t
(**
Returns whether the kernel is stationary. 
*)

val set_params : ?params:(string * Py.Object.t) list -> [> tag] Obj.t -> t
(**
Set the parameters of this kernel.

The method works on simple kernels as well as on nested kernels.
The latter have parameters of the form ``<component>__<parameter>``
so that it's possible to update each component of a nested object.

Returns
-------
self
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module WhiteKernel : sig
type tag = [`WhiteKernel]
type t = [`GenericKernelMixin | `Object | `StationaryKernelMixin | `WhiteKernel] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_stationary_kernel : t -> [`StationaryKernelMixin] Obj.t
val as_generic_kernel : t -> [`GenericKernelMixin] Obj.t
val create : ?noise_level:float -> ?noise_level_bounds:(float * float) -> unit -> t
(**
White kernel.

The main use-case of this kernel is as part of a sum-kernel where it
explains the noise of the signal as independently and identically
normally-distributed. The parameter noise_level equals the variance of this
noise.

k(x_1, x_2) = noise_level if x_1 == x_2 else 0

.. versionadded:: 0.18

Parameters
----------
noise_level : float, default: 1.0
    Parameter controlling the noise level (variance)

noise_level_bounds : pair of floats >= 0, default: (1e-5, 1e5)
    The lower and upper bound on noise_level
*)

val clone_with_theta : theta:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a clone of self with given hyperparameters theta.

Parameters
----------
theta : array, shape (n_dims,)
    The hyperparameters
*)

val diag : x:Py.Object.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Returns the diagonal of the kernel k(X, X).

The result of this method is identical to np.diag(self(X)); however,
it can be evaluated more efficiently since only the diagonal is
evaluated.

Parameters
----------
X : sequence of length n_samples_X
    Argument to the kernel. Could either be array-like with
    shape = (n_samples_X, n_features) or a list of objects.

Returns
-------
K_diag : array, shape (n_samples_X,)
    Diagonal of kernel k(X, X)
*)

val get_params : ?deep:bool -> [> tag] Obj.t -> Dict.t
(**
Get parameters of this kernel.

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

val is_stationary : [> tag] Obj.t -> Py.Object.t
(**
Returns whether the kernel is stationary. 
*)

val set_params : ?params:(string * Py.Object.t) list -> [> tag] Obj.t -> t
(**
Set the parameters of this kernel.

The method works on simple kernels as well as on nested kernels.
The latter have parameters of the form ``<component>__<parameter>``
so that it's possible to update each component of a nested object.

Returns
-------
self
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val abstractmethod : Py.Object.t -> Py.Object.t
(**
A decorator indicating abstract methods.

Requires that the metaclass is ABCMeta or derived from it.  A
class that has a metaclass derived from ABCMeta cannot be
instantiated unless all of its abstract methods are overridden.
The abstract methods can be called using any of the normal
'super' call mechanisms.  abstractmethod() may be used to declare
abstract methods for properties and descriptors.

Usage:

    class C(metaclass=ABCMeta):
        @abstractmethod
        def my_abstract_method(self, ...):
            ...
*)

val cdist : ?metric:[`Callable of Py.Object.t | `S of string] -> ?kwargs:(string * Py.Object.t) list -> xa:[>`ArrayLike] Np.Obj.t -> xb:[>`ArrayLike] Np.Obj.t -> Py.Object.t list -> [>`ArrayLike] Np.Obj.t
(**
Compute distance between each pair of the two collections of inputs.

See Notes for common calling conventions.

Parameters
----------
XA : ndarray
    An :math:`m_A` by :math:`n` array of :math:`m_A`
    original observations in an :math:`n`-dimensional space.
    Inputs are converted to float type.
XB : ndarray
    An :math:`m_B` by :math:`n` array of :math:`m_B`
    original observations in an :math:`n`-dimensional space.
    Inputs are converted to float type.
metric : str or callable, optional
    The distance metric to use.  If a string, the distance function can be
    'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
    'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
    'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
    'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
    'wminkowski', 'yule'.
*args : tuple. Deprecated.
    Additional arguments should be passed as keyword arguments
**kwargs : dict, optional
    Extra arguments to `metric`: refer to each metric documentation for a
    list of all possible arguments.

    Some possible arguments:

    p : scalar
    The p-norm to apply for Minkowski, weighted and unweighted.
    Default: 2.

    w : ndarray
    The weight vector for metrics that support weights (e.g., Minkowski).

    V : ndarray
    The variance vector for standardized Euclidean.
    Default: var(vstack([XA, XB]), axis=0, ddof=1)

    VI : ndarray
    The inverse of the covariance matrix for Mahalanobis.
    Default: inv(cov(vstack([XA, XB].T))).T

    out : ndarray
    The output array
    If not None, the distance matrix Y is stored in this array.
    Note: metric independent, it will become a regular keyword arg in a
    future scipy version

Returns
-------
Y : ndarray
    A :math:`m_A` by :math:`m_B` distance matrix is returned.
    For each :math:`i` and :math:`j`, the metric
    ``dist(u=XA[i], v=XB[j])`` is computed and stored in the
    :math:`ij` th entry.

Raises
------
ValueError
    An exception is thrown if `XA` and `XB` do not have
    the same number of columns.

Notes
-----
The following are common calling conventions:

1. ``Y = cdist(XA, XB, 'euclidean')``

   Computes the distance between :math:`m` points using
   Euclidean distance (2-norm) as the distance metric between the
   points. The points are arranged as :math:`m`
   :math:`n`-dimensional row vectors in the matrix X.

2. ``Y = cdist(XA, XB, 'minkowski', p=2.)``

   Computes the distances using the Minkowski distance
   :math:`||u-v||_p` (:math:`p`-norm) where :math:`p \geq 1`.

3. ``Y = cdist(XA, XB, 'cityblock')``

   Computes the city block or Manhattan distance between the
   points.

4. ``Y = cdist(XA, XB, 'seuclidean', V=None)``

   Computes the standardized Euclidean distance. The standardized
   Euclidean distance between two n-vectors ``u`` and ``v`` is

   .. math::

      \sqrt{\sum {(u_i-v_i)^2 / V[x_i]}}.

   V is the variance vector; V[i] is the variance computed over all
   the i'th components of the points. If not passed, it is
   automatically computed.

5. ``Y = cdist(XA, XB, 'sqeuclidean')``

   Computes the squared Euclidean distance :math:`||u-v||_2^2` between
   the vectors.

6. ``Y = cdist(XA, XB, 'cosine')``

   Computes the cosine distance between vectors u and v,

   .. math::

      1 - \frac{u \cdot v}
               {{ ||u|| }_2 { ||v|| }_2}

   where :math:`||*||_2` is the 2-norm of its argument ``*``, and
   :math:`u \cdot v` is the dot product of :math:`u` and :math:`v`.

7. ``Y = cdist(XA, XB, 'correlation')``

   Computes the correlation distance between vectors u and v. This is

   .. math::

      1 - \frac{(u - \bar{u}) \cdot (v - \bar{v})}
               {{ ||(u - \bar{u})|| }_2 { ||(v - \bar{v})|| }_2}

   where :math:`\bar{v}` is the mean of the elements of vector v,
   and :math:`x \cdot y` is the dot product of :math:`x` and :math:`y`.


8. ``Y = cdist(XA, XB, 'hamming')``

   Computes the normalized Hamming distance, or the proportion of
   those vector elements between two n-vectors ``u`` and ``v``
   which disagree. To save memory, the matrix ``X`` can be of type
   boolean.

9. ``Y = cdist(XA, XB, 'jaccard')``

   Computes the Jaccard distance between the points. Given two
   vectors, ``u`` and ``v``, the Jaccard distance is the
   proportion of those elements ``u[i]`` and ``v[i]`` that
   disagree where at least one of them is non-zero.

10. ``Y = cdist(XA, XB, 'chebyshev')``

   Computes the Chebyshev distance between the points. The
   Chebyshev distance between two n-vectors ``u`` and ``v`` is the
   maximum norm-1 distance between their respective elements. More
   precisely, the distance is given by

   .. math::

      d(u,v) = \max_i { |u_i-v_i| }.

11. ``Y = cdist(XA, XB, 'canberra')``

   Computes the Canberra distance between the points. The
   Canberra distance between two points ``u`` and ``v`` is

   .. math::

     d(u,v) = \sum_i \frac{ |u_i-v_i| }
                          { |u_i|+|v_i| }.

12. ``Y = cdist(XA, XB, 'braycurtis')``

   Computes the Bray-Curtis distance between the points. The
   Bray-Curtis distance between two points ``u`` and ``v`` is


   .. math::

        d(u,v) = \frac{\sum_i (|u_i-v_i|)}
                      {\sum_i (|u_i+v_i|)}

13. ``Y = cdist(XA, XB, 'mahalanobis', VI=None)``

   Computes the Mahalanobis distance between the points. The
   Mahalanobis distance between two points ``u`` and ``v`` is
   :math:`\sqrt{(u-v)(1/V)(u-v)^T}` where :math:`(1/V)` (the ``VI``
   variable) is the inverse covariance. If ``VI`` is not None,
   ``VI`` will be used as the inverse covariance matrix.

14. ``Y = cdist(XA, XB, 'yule')``

   Computes the Yule distance between the boolean
   vectors. (see `yule` function documentation)

15. ``Y = cdist(XA, XB, 'matching')``

   Synonym for 'hamming'.

16. ``Y = cdist(XA, XB, 'dice')``

   Computes the Dice distance between the boolean vectors. (see
   `dice` function documentation)

17. ``Y = cdist(XA, XB, 'kulsinski')``

   Computes the Kulsinski distance between the boolean
   vectors. (see `kulsinski` function documentation)

18. ``Y = cdist(XA, XB, 'rogerstanimoto')``

   Computes the Rogers-Tanimoto distance between the boolean
   vectors. (see `rogerstanimoto` function documentation)

19. ``Y = cdist(XA, XB, 'russellrao')``

   Computes the Russell-Rao distance between the boolean
   vectors. (see `russellrao` function documentation)

20. ``Y = cdist(XA, XB, 'sokalmichener')``

   Computes the Sokal-Michener distance between the boolean
   vectors. (see `sokalmichener` function documentation)

21. ``Y = cdist(XA, XB, 'sokalsneath')``

   Computes the Sokal-Sneath distance between the vectors. (see
   `sokalsneath` function documentation)


22. ``Y = cdist(XA, XB, 'wminkowski', p=2., w=w)``

   Computes the weighted Minkowski distance between the
   vectors. (see `wminkowski` function documentation)

23. ``Y = cdist(XA, XB, f)``

   Computes the distance between all pairs of vectors in X
   using the user supplied 2-arity function f. For example,
   Euclidean distance between the vectors could be computed
   as follows::

     dm = cdist(XA, XB, lambda u, v: np.sqrt(((u-v)**2).sum()))

   Note that you should avoid passing a reference to one of
   the distance functions defined in this library. For example,::

     dm = cdist(XA, XB, sokalsneath)

   would calculate the pair-wise distances between the vectors in
   X using the Python function `sokalsneath`. This would result in
   sokalsneath being called :math:`{n \choose 2}` times, which
   is inefficient. Instead, the optimized C version is more
   efficient, and we call it using the following syntax::

     dm = cdist(XA, XB, 'sokalsneath')

Examples
--------
Find the Euclidean distances between four 2-D coordinates:

>>> from scipy.spatial import distance
>>> coords = [(35.0456, -85.2672),
...           (35.1174, -89.9711),
...           (35.9728, -83.9422),
...           (36.1667, -86.7833)]
>>> distance.cdist(coords, coords, 'euclidean')
array([[ 0.    ,  4.7044,  1.6172,  1.8856],
       [ 4.7044,  0.    ,  6.0893,  3.3561],
       [ 1.6172,  6.0893,  0.    ,  2.8477],
       [ 1.8856,  3.3561,  2.8477,  0.    ]])


Find the Manhattan distance from a 3-D point to the corners of the unit
cube:

>>> a = np.array([[0, 0, 0],
...               [0, 0, 1],
...               [0, 1, 0],
...               [0, 1, 1],
...               [1, 0, 0],
...               [1, 0, 1],
...               [1, 1, 0],
...               [1, 1, 1]])
>>> b = np.array([[ 0.1,  0.2,  0.4]])
>>> distance.cdist(a, b, 'cityblock')
array([[ 0.7],
       [ 0.9],
       [ 1.3],
       [ 1.5],
       [ 1.5],
       [ 1.7],
       [ 2.1],
       [ 2.3]])
*)

val clone : ?safe:bool -> estimator:[>`BaseEstimator] Np.Obj.t -> unit -> Py.Object.t
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

val gamma : ?out:Py.Object.t -> ?where:Py.Object.t -> x:Py.Object.t -> unit -> Py.Object.t
(**
gamma(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

gamma(z)

Gamma function.

The Gamma function is defined as

.. math::

   \Gamma(z) = \int_0^\infty t^{z-1} e^{-t} dt

for :math:`\Re(z) > 0` and is extended to the rest of the complex
plane by analytic continuation. See [dlmf]_ for more details.

Parameters
----------
z : array_like
    Real or complex valued argument

Returns
-------
scalar or ndarray
    Values of the Gamma function

Notes
-----
The Gamma function is often referred to as the generalized
factorial since :math:`\Gamma(n + 1) = n!` for natural numbers
:math:`n`. More generally it satisfies the recurrence relation
:math:`\Gamma(z + 1) = z \cdot \Gamma(z)` for complex :math:`z`,
which, combined with the fact that :math:`\Gamma(1) = 1`, implies
the above identity for :math:`z = n`.

References
----------
.. [dlmf] NIST Digital Library of Mathematical Functions
          https://dlmf.nist.gov/5.2#E1

Examples
--------
>>> from scipy.special import gamma, factorial

>>> gamma([0, 0.5, 1, 5])
array([         inf,   1.77245385,   1.        ,  24.        ])

>>> z = 2.5 + 1j
>>> gamma(z)
(0.77476210455108352+0.70763120437959293j)
>>> gamma(z+1), z*gamma(z)  # Recurrence property
((1.2292740569981171+2.5438401155000685j),
 (1.2292740569981158+2.5438401155000658j))

>>> gamma(0.5)**2  # gamma(0.5) = sqrt(pi)
3.1415926535897927

Plot gamma(x) for real x

>>> x = np.linspace(-3.5, 5.5, 2251)
>>> y = gamma(x)

>>> import matplotlib.pyplot as plt
>>> plt.plot(x, y, 'b', alpha=0.6, label='gamma(x)')
>>> k = np.arange(1, 7)
>>> plt.plot(k, factorial(k-1), 'k*', alpha=0.6,
...          label='(x-1)!, x = 1, 2, ...')
>>> plt.xlim(-3.5, 5.5)
>>> plt.ylim(-10, 25)
>>> plt.grid()
>>> plt.xlabel('x')
>>> plt.legend(loc='lower right')
>>> plt.show()
*)

val kv : ?out:Py.Object.t -> ?where:Py.Object.t -> x:Py.Object.t -> unit -> [>`ArrayLike] Np.Obj.t
(**
kv(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

kv(v, z)

Modified Bessel function of the second kind of real order `v`

Returns the modified Bessel function of the second kind for real order
`v` at complex `z`.

These are also sometimes called functions of the third kind, Basset
functions, or Macdonald functions.  They are defined as those solutions
of the modified Bessel equation for which,

.. math::
    K_v(x) \sim \sqrt{\pi/(2x)} \exp(-x)

as :math:`x \to \infty` [3]_.

Parameters
----------
v : array_like of float
    Order of Bessel functions
z : array_like of complex
    Argument at which to evaluate the Bessel functions

Returns
-------
out : ndarray
    The results. Note that input must be of complex type to get complex
    output, e.g. ``kv(3, -2+0j)`` instead of ``kv(3, -2)``.

Notes
-----
Wrapper for AMOS [1]_ routine `zbesk`.  For a discussion of the
algorithm used, see [2]_ and the references therein.

See Also
--------
kve : This function with leading exponential behavior stripped off.
kvp : Derivative of this function

References
----------
.. [1] Donald E. Amos, 'AMOS, A Portable Package for Bessel Functions
       of a Complex Argument and Nonnegative Order',
       http://netlib.org/amos/
.. [2] Donald E. Amos, 'Algorithm 644: A portable package for Bessel
       functions of a complex argument and nonnegative order', ACM
       TOMS Vol. 12 Issue 3, Sept. 1986, p. 265
.. [3] NIST Digital Library of Mathematical Functions,
       Eq. 10.25.E3. https://dlmf.nist.gov/10.25.E3

Examples
--------
Plot the function of several orders for real input:

>>> from scipy.special import kv
>>> import matplotlib.pyplot as plt
>>> x = np.linspace(0, 5, 1000)
>>> for N in np.linspace(0, 6, 5):
...     plt.plot(x, kv(N, x), label='$K_{{{}}}(x)$'.format(N))
>>> plt.ylim(0, 10)
>>> plt.legend()
>>> plt.title(r'Modified Bessel function of the second kind $K_\nu(x)$')
>>> plt.show()

Calculate for a single value at multiple orders:

>>> kv([4, 4.5, 5], 1+2j)
array([ 0.1992+2.3892j,  2.3493+3.6j   ,  7.2827+3.8104j])
*)

val namedtuple : ?rename:Py.Object.t -> ?defaults:Py.Object.t -> ?module_:Py.Object.t -> typename:Py.Object.t -> field_names:Py.Object.t -> unit -> Py.Object.t
(**
Returns a new subclass of tuple with named fields.

>>> Point = namedtuple('Point', ['x', 'y'])
>>> Point.__doc__                   # docstring for the new class
'Point(x, y)'
>>> p = Point(11, y=22)             # instantiate with positional args or keywords
>>> p[0] + p[1]                     # indexable like a plain tuple
33
>>> x, y = p                        # unpack like a regular tuple
>>> x, y
(11, 22)
>>> p.x + p.y                       # fields also accessible by name
33
>>> d = p._asdict()                 # convert to a dictionary
>>> d['x']
11
>>> Point( **d)                      # convert from a dictionary
Point(x=11, y=22)
>>> p._replace(x=100)               # _replace() is like str.replace() but targets named fields
Point(x=100, y=22)
*)

val pairwise_kernels : ?y:[>`ArrayLike] Np.Obj.t -> ?metric:[`Callable of Py.Object.t | `S of string] -> ?filter_params:bool -> ?n_jobs:int -> ?kwds:(string * Py.Object.t) list -> x:[`Otherwise of Py.Object.t | `Arr of [>`ArrayLike] Np.Obj.t] -> unit -> [>`ArrayLike] Np.Obj.t
(**
Compute the kernel between arrays X and optional array Y.

This method takes either a vector array or a kernel matrix, and returns
a kernel matrix. If the input is a vector array, the kernels are
computed. If the input is a kernel matrix, it is returned instead.

This method provides a safe way to take a kernel matrix as input, while
preserving compatibility with many other algorithms that take a vector
array.

If Y is given (default is None), then the returned matrix is the pairwise
kernel between the arrays from both X and Y.

Valid values for metric are:
    ['additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf',
    'laplacian', 'sigmoid', 'cosine']

Read more in the :ref:`User Guide <metrics>`.

Parameters
----------
X : array [n_samples_a, n_samples_a] if metric == 'precomputed', or,              [n_samples_a, n_features] otherwise
    Array of pairwise kernels between samples, or a feature array.

Y : array [n_samples_b, n_features]
    A second feature array only if X has shape [n_samples_a, n_features].

metric : string, or callable
    The metric to use when calculating kernel between instances in a
    feature array. If metric is a string, it must be one of the metrics
    in pairwise.PAIRWISE_KERNEL_FUNCTIONS.
    If metric is 'precomputed', X is assumed to be a kernel matrix.
    Alternatively, if metric is a callable function, it is called on each
    pair of instances (rows) and the resulting value recorded. The callable
    should take two rows from X as input and return the corresponding
    kernel value as a single number. This means that callables from
    :mod:`sklearn.metrics.pairwise` are not allowed, as they operate on
    matrices, not single samples. Use the string identifying the kernel
    instead.

filter_params : boolean
    Whether to filter invalid parameters or not.

n_jobs : int or None, optional (default=None)
    The number of jobs to use for the computation. This works by breaking
    down the pairwise matrix into n_jobs even slices and computing them in
    parallel.

    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

**kwds : optional keyword parameters
    Any further parameters are passed directly to the kernel function.

Returns
-------
K : array [n_samples_a, n_samples_a] or [n_samples_a, n_samples_b]
    A kernel matrix K such that K_{i, j} is the kernel between the
    ith and jth vectors of the given matrix X, if Y is None.
    If Y is not None, then K_{i, j} is the kernel between the ith array
    from X and the jth array from Y.

Notes
-----
If metric is 'precomputed', Y is ignored and X is returned.
*)

val pdist : ?metric:[`Callable of Py.Object.t | `S of string] -> ?kwargs:(string * Py.Object.t) list -> x:[>`ArrayLike] Np.Obj.t -> Py.Object.t list -> [>`ArrayLike] Np.Obj.t
(**
Pairwise distances between observations in n-dimensional space.

See Notes for common calling conventions.

Parameters
----------
X : ndarray
    An m by n array of m original observations in an
    n-dimensional space.
metric : str or function, optional
    The distance metric to use. The distance function can
    be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
    'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
    'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching',
    'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
    'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'.
*args : tuple. Deprecated.
    Additional arguments should be passed as keyword arguments
**kwargs : dict, optional
    Extra arguments to `metric`: refer to each metric documentation for a
    list of all possible arguments.

    Some possible arguments:

    p : scalar
    The p-norm to apply for Minkowski, weighted and unweighted.
    Default: 2.

    w : ndarray
    The weight vector for metrics that support weights (e.g., Minkowski).

    V : ndarray
    The variance vector for standardized Euclidean.
    Default: var(X, axis=0, ddof=1)

    VI : ndarray
    The inverse of the covariance matrix for Mahalanobis.
    Default: inv(cov(X.T)).T

    out : ndarray.
    The output array
    If not None, condensed distance matrix Y is stored in this array.
    Note: metric independent, it will become a regular keyword arg in a
    future scipy version

Returns
-------
Y : ndarray
    Returns a condensed distance matrix Y.  For
    each :math:`i` and :math:`j` (where :math:`i<j<m`),where m is the number
    of original observations. The metric ``dist(u=X[i], v=X[j])``
    is computed and stored in entry ``ij``.

See Also
--------
squareform : converts between condensed distance matrices and
             square distance matrices.

Notes
-----
See ``squareform`` for information on how to calculate the index of
this entry or to convert the condensed distance matrix to a
redundant square matrix.

The following are common calling conventions.

1. ``Y = pdist(X, 'euclidean')``

   Computes the distance between m points using Euclidean distance
   (2-norm) as the distance metric between the points. The points
   are arranged as m n-dimensional row vectors in the matrix X.

2. ``Y = pdist(X, 'minkowski', p=2.)``

   Computes the distances using the Minkowski distance
   :math:`||u-v||_p` (p-norm) where :math:`p \geq 1`.

3. ``Y = pdist(X, 'cityblock')``

   Computes the city block or Manhattan distance between the
   points.

4. ``Y = pdist(X, 'seuclidean', V=None)``

   Computes the standardized Euclidean distance. The standardized
   Euclidean distance between two n-vectors ``u`` and ``v`` is

   .. math::

      \sqrt{\sum {(u_i-v_i)^2 / V[x_i]}}


   V is the variance vector; V[i] is the variance computed over all
   the i'th components of the points.  If not passed, it is
   automatically computed.

5. ``Y = pdist(X, 'sqeuclidean')``

   Computes the squared Euclidean distance :math:`||u-v||_2^2` between
   the vectors.

6. ``Y = pdist(X, 'cosine')``

   Computes the cosine distance between vectors u and v,

   .. math::

      1 - \frac{u \cdot v}
               {{ ||u|| }_2 { ||v|| }_2}

   where :math:`||*||_2` is the 2-norm of its argument ``*``, and
   :math:`u \cdot v` is the dot product of ``u`` and ``v``.

7. ``Y = pdist(X, 'correlation')``

   Computes the correlation distance between vectors u and v. This is

   .. math::

      1 - \frac{(u - \bar{u}) \cdot (v - \bar{v})}
               {{ ||(u - \bar{u})|| }_2 { ||(v - \bar{v})|| }_2}

   where :math:`\bar{v}` is the mean of the elements of vector v,
   and :math:`x \cdot y` is the dot product of :math:`x` and :math:`y`.

8. ``Y = pdist(X, 'hamming')``

   Computes the normalized Hamming distance, or the proportion of
   those vector elements between two n-vectors ``u`` and ``v``
   which disagree. To save memory, the matrix ``X`` can be of type
   boolean.

9. ``Y = pdist(X, 'jaccard')``

   Computes the Jaccard distance between the points. Given two
   vectors, ``u`` and ``v``, the Jaccard distance is the
   proportion of those elements ``u[i]`` and ``v[i]`` that
   disagree.

10. ``Y = pdist(X, 'chebyshev')``

   Computes the Chebyshev distance between the points. The
   Chebyshev distance between two n-vectors ``u`` and ``v`` is the
   maximum norm-1 distance between their respective elements. More
   precisely, the distance is given by

   .. math::

      d(u,v) = \max_i { |u_i-v_i| }

11. ``Y = pdist(X, 'canberra')``

   Computes the Canberra distance between the points. The
   Canberra distance between two points ``u`` and ``v`` is

   .. math::

     d(u,v) = \sum_i \frac{ |u_i-v_i| }
                          { |u_i|+|v_i| }


12. ``Y = pdist(X, 'braycurtis')``

   Computes the Bray-Curtis distance between the points. The
   Bray-Curtis distance between two points ``u`` and ``v`` is


   .. math::

        d(u,v) = \frac{\sum_i { |u_i-v_i| }}
                       {\sum_i { |u_i+v_i| }}

13. ``Y = pdist(X, 'mahalanobis', VI=None)``

   Computes the Mahalanobis distance between the points. The
   Mahalanobis distance between two points ``u`` and ``v`` is
   :math:`\sqrt{(u-v)(1/V)(u-v)^T}` where :math:`(1/V)` (the ``VI``
   variable) is the inverse covariance. If ``VI`` is not None,
   ``VI`` will be used as the inverse covariance matrix.

14. ``Y = pdist(X, 'yule')``

   Computes the Yule distance between each pair of boolean
   vectors. (see yule function documentation)

15. ``Y = pdist(X, 'matching')``

   Synonym for 'hamming'.

16. ``Y = pdist(X, 'dice')``

   Computes the Dice distance between each pair of boolean
   vectors. (see dice function documentation)

17. ``Y = pdist(X, 'kulsinski')``

   Computes the Kulsinski distance between each pair of
   boolean vectors. (see kulsinski function documentation)

18. ``Y = pdist(X, 'rogerstanimoto')``

   Computes the Rogers-Tanimoto distance between each pair of
   boolean vectors. (see rogerstanimoto function documentation)

19. ``Y = pdist(X, 'russellrao')``

   Computes the Russell-Rao distance between each pair of
   boolean vectors. (see russellrao function documentation)

20. ``Y = pdist(X, 'sokalmichener')``

   Computes the Sokal-Michener distance between each pair of
   boolean vectors. (see sokalmichener function documentation)

21. ``Y = pdist(X, 'sokalsneath')``

   Computes the Sokal-Sneath distance between each pair of
   boolean vectors. (see sokalsneath function documentation)

22. ``Y = pdist(X, 'wminkowski', p=2, w=w)``

   Computes the weighted Minkowski distance between each pair of
   vectors. (see wminkowski function documentation)

23. ``Y = pdist(X, f)``

   Computes the distance between all pairs of vectors in X
   using the user supplied 2-arity function f. For example,
   Euclidean distance between the vectors could be computed
   as follows::

     dm = pdist(X, lambda u, v: np.sqrt(((u-v)**2).sum()))

   Note that you should avoid passing a reference to one of
   the distance functions defined in this library. For example,::

     dm = pdist(X, sokalsneath)

   would calculate the pair-wise distances between the vectors in
   X using the Python function sokalsneath. This would result in
   sokalsneath being called :math:`{n \choose 2}` times, which
   is inefficient. Instead, the optimized C version is more
   efficient, and we call it using the following syntax.::

     dm = pdist(X, 'sokalsneath')
*)

val signature : ?follow_wrapped:Py.Object.t -> obj:Py.Object.t -> unit -> Py.Object.t
(**
Get a signature object for the passed callable.
*)

val squareform : ?force:string -> ?checks:bool -> x:[>`ArrayLike] Np.Obj.t -> unit -> [>`ArrayLike] Np.Obj.t
(**
Convert a vector-form distance vector to a square-form distance
matrix, and vice-versa.

Parameters
----------
X : ndarray
    Either a condensed or redundant distance matrix.
force : str, optional
    As with MATLAB(TM), if force is equal to ``'tovector'`` or
    ``'tomatrix'``, the input will be treated as a distance matrix or
    distance vector respectively.
checks : bool, optional
    If set to False, no checks will be made for matrix
    symmetry nor zero diagonals. This is useful if it is known that
    ``X - X.T1`` is small and ``diag(X)`` is close to zero.
    These values are ignored any way so they do not disrupt the
    squareform transformation.

Returns
-------
Y : ndarray
    If a condensed distance matrix is passed, a redundant one is
    returned, or if a redundant one is passed, a condensed distance
    matrix is returned.

Notes
-----
1. v = squareform(X)

   Given a square d-by-d symmetric distance matrix X,
   ``v = squareform(X)`` returns a ``d * (d-1) / 2`` (or
   :math:`{n \choose 2}`) sized vector v.

  :math:`v[{n \choose 2}-{n-i \choose 2} + (j-i-1)]` is the distance
  between points i and j. If X is non-square or asymmetric, an error
  is returned.

2. X = squareform(v)

  Given a ``d*(d-1)/2`` sized v for some integer ``d >= 2`` encoding
  distances as described, ``X = squareform(v)`` returns a d by d distance
  matrix X.  The ``X[i, j]`` and ``X[j, i]`` values are set to
  :math:`v[{n \choose 2}-{n-i \choose 2} + (j-i-1)]` and all
  diagonal elements are zero.

In SciPy 0.19.0, ``squareform`` stopped casting all input types to
float64, and started returning arrays of the same dtype as the input.
*)


end

