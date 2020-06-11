(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module BayesianGaussianMixture : sig
type tag = [`BayesianGaussianMixture]
type t = [`BaseEstimator | `BaseMixture | `BayesianGaussianMixture | `DensityMixin | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_density : t -> [`DensityMixin] Obj.t
val as_estimator : t -> [`BaseEstimator] Obj.t
val as_mixture : t -> [`BaseMixture] Obj.t
val create : ?n_components:int -> ?covariance_type:[`Full | `Tied | `Diag | `Spherical] -> ?tol:float -> ?reg_covar:float -> ?max_iter:int -> ?n_init:int -> ?init_params:[`Kmeans | `Random] -> ?weight_concentration_prior_type:string -> ?weight_concentration_prior:float -> ?mean_precision_prior:float -> ?mean_prior:[>`ArrayLike] Np.Obj.t -> ?degrees_of_freedom_prior:float -> ?covariance_prior:[>`ArrayLike] Np.Obj.t -> ?random_state:int -> ?warm_start:bool -> ?verbose:int -> ?verbose_interval:int -> unit -> t
(**
Variational Bayesian estimation of a Gaussian mixture.

This class allows to infer an approximate posterior distribution over the
parameters of a Gaussian mixture distribution. The effective number of
components can be inferred from the data.

This class implements two types of prior for the weights distribution: a
finite mixture model with Dirichlet distribution and an infinite mixture
model with the Dirichlet Process. In practice Dirichlet Process inference
algorithm is approximated and uses a truncated distribution with a fixed
maximum number of components (called the Stick-breaking representation).
The number of components actually used almost always depends on the data.

.. versionadded:: 0.18

Read more in the :ref:`User Guide <bgmm>`.

Parameters
----------
n_components : int, defaults to 1.
    The number of mixture components. Depending on the data and the value
    of the `weight_concentration_prior` the model can decide to not use
    all the components by setting some component `weights_` to values very
    close to zero. The number of effective components is therefore smaller
    than n_components.

covariance_type : {'full', 'tied', 'diag', 'spherical'}, defaults to 'full'
    String describing the type of covariance parameters to use.
    Must be one of::

        'full' (each component has its own general covariance matrix),
        'tied' (all components share the same general covariance matrix),
        'diag' (each component has its own diagonal covariance matrix),
        'spherical' (each component has its own single variance).

tol : float, defaults to 1e-3.
    The convergence threshold. EM iterations will stop when the
    lower bound average gain on the likelihood (of the training data with
    respect to the model) is below this threshold.

reg_covar : float, defaults to 1e-6.
    Non-negative regularization added to the diagonal of covariance.
    Allows to assure that the covariance matrices are all positive.

max_iter : int, defaults to 100.
    The number of EM iterations to perform.

n_init : int, defaults to 1.
    The number of initializations to perform. The result with the highest
    lower bound value on the likelihood is kept.

init_params : {'kmeans', 'random'}, defaults to 'kmeans'.
    The method used to initialize the weights, the means and the
    covariances.
    Must be one of::

        'kmeans' : responsibilities are initialized using kmeans.
        'random' : responsibilities are initialized randomly.

weight_concentration_prior_type : str, defaults to 'dirichlet_process'.
    String describing the type of the weight concentration prior.
    Must be one of::

        'dirichlet_process' (using the Stick-breaking representation),
        'dirichlet_distribution' (can favor more uniform weights).

weight_concentration_prior : float | None, optional.
    The dirichlet concentration of each component on the weight
    distribution (Dirichlet). This is commonly called gamma in the
    literature. The higher concentration puts more mass in
    the center and will lead to more components being active, while a lower
    concentration parameter will lead to more mass at the edge of the
    mixture weights simplex. The value of the parameter must be greater
    than 0. If it is None, it's set to ``1. / n_components``.

mean_precision_prior : float | None, optional.
    The precision prior on the mean distribution (Gaussian).
    Controls the extent of where means can be placed. Larger
    values concentrate the cluster means around `mean_prior`.
    The value of the parameter must be greater than 0.
    If it is None, it is set to 1.

mean_prior : array-like, shape (n_features,), optional
    The prior on the mean distribution (Gaussian).
    If it is None, it is set to the mean of X.

degrees_of_freedom_prior : float | None, optional.
    The prior of the number of degrees of freedom on the covariance
    distributions (Wishart). If it is None, it's set to `n_features`.

covariance_prior : float or array-like, optional
    The prior on the covariance distribution (Wishart).
    If it is None, the emiprical covariance prior is initialized using the
    covariance of X. The shape depends on `covariance_type`::

            (n_features, n_features) if 'full',
            (n_features, n_features) if 'tied',
            (n_features)             if 'diag',
            float                    if 'spherical'

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

warm_start : bool, default to False.
    If 'warm_start' is True, the solution of the last fitting is used as
    initialization for the next call of fit(). This can speed up
    convergence when fit is called several times on similar problems.
    See :term:`the Glossary <warm_start>`.

verbose : int, default to 0.
    Enable verbose output. If 1 then it prints the current
    initialization and each iteration step. If greater than 1 then
    it prints also the log probability and the time needed
    for each step.

verbose_interval : int, default to 10.
    Number of iteration done before the next print.

Attributes
----------
weights_ : array-like, shape (n_components,)
    The weights of each mixture components.

means_ : array-like, shape (n_components, n_features)
    The mean of each mixture component.

covariances_ : array-like
    The covariance of each mixture component.
    The shape depends on `covariance_type`::

        (n_components,)                        if 'spherical',
        (n_features, n_features)               if 'tied',
        (n_components, n_features)             if 'diag',
        (n_components, n_features, n_features) if 'full'

precisions_ : array-like
    The precision matrices for each component in the mixture. A precision
    matrix is the inverse of a covariance matrix. A covariance matrix is
    symmetric positive definite so the mixture of Gaussian can be
    equivalently parameterized by the precision matrices. Storing the
    precision matrices instead of the covariance matrices makes it more
    efficient to compute the log-likelihood of new samples at test time.
    The shape depends on ``covariance_type``::

        (n_components,)                        if 'spherical',
        (n_features, n_features)               if 'tied',
        (n_components, n_features)             if 'diag',
        (n_components, n_features, n_features) if 'full'

precisions_cholesky_ : array-like
    The cholesky decomposition of the precision matrices of each mixture
    component. A precision matrix is the inverse of a covariance matrix.
    A covariance matrix is symmetric positive definite so the mixture of
    Gaussian can be equivalently parameterized by the precision matrices.
    Storing the precision matrices instead of the covariance matrices makes
    it more efficient to compute the log-likelihood of new samples at test
    time. The shape depends on ``covariance_type``::

        (n_components,)                        if 'spherical',
        (n_features, n_features)               if 'tied',
        (n_components, n_features)             if 'diag',
        (n_components, n_features, n_features) if 'full'

converged_ : bool
    True when convergence was reached in fit(), False otherwise.

n_iter_ : int
    Number of step used by the best fit of inference to reach the
    convergence.

lower_bound_ : float
    Lower bound value on the likelihood (of the training data with
    respect to the model) of the best fit of inference.

weight_concentration_prior_ : tuple or float
    The dirichlet concentration of each component on the weight
    distribution (Dirichlet). The type depends on
    ``weight_concentration_prior_type``::

        (float, float) if 'dirichlet_process' (Beta parameters),
        float          if 'dirichlet_distribution' (Dirichlet parameters).

    The higher concentration puts more mass in
    the center and will lead to more components being active, while a lower
    concentration parameter will lead to more mass at the edge of the
    simplex.

weight_concentration_ : array-like, shape (n_components,)
    The dirichlet concentration of each component on the weight
    distribution (Dirichlet).

mean_precision_prior_ : float
    The precision prior on the mean distribution (Gaussian).
    Controls the extent of where means can be placed.
    Larger values concentrate the cluster means around `mean_prior`.
    If mean_precision_prior is set to None, `mean_precision_prior_` is set
    to 1.

mean_precision_ : array-like, shape (n_components,)
    The precision of each components on the mean distribution (Gaussian).

mean_prior_ : array-like, shape (n_features,)
    The prior on the mean distribution (Gaussian).

degrees_of_freedom_prior_ : float
    The prior of the number of degrees of freedom on the covariance
    distributions (Wishart).

degrees_of_freedom_ : array-like, shape (n_components,)
    The number of degrees of freedom of each components in the model.

covariance_prior_ : float or array-like
    The prior on the covariance distribution (Wishart).
    The shape depends on `covariance_type`::

        (n_features, n_features) if 'full',
        (n_features, n_features) if 'tied',
        (n_features)             if 'diag',
        float                    if 'spherical'

See Also
--------
GaussianMixture : Finite Gaussian mixture fit with EM.

References
----------

.. [1] `Bishop, Christopher M. (2006). 'Pattern recognition and machine
   learning'. Vol. 4 No. 4. New York: Springer.
   <https://www.springer.com/kr/book/9780387310732>`_

.. [2] `Hagai Attias. (2000). 'A Variational Bayesian Framework for
   Graphical Models'. In Advances in Neural Information Processing
   Systems 12.
   <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.36.2841&rep=rep1&type=pdf>`_

.. [3] `Blei, David M. and Michael I. Jordan. (2006). 'Variational
   inference for Dirichlet process mixtures'. Bayesian analysis 1.1
   <https://www.cs.princeton.edu/courses/archive/fall11/cos597C/reading/BleiJordan2005.pdf>`_
*)

val fit : ?y:Py.Object.t -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Estimate model parameters with the EM algorithm.

The method fits the model ``n_init`` times and sets the parameters with
which the model has the largest likelihood or lower bound. Within each
trial, the method iterates between E-step and M-step for ``max_iter``
times until the change of likelihood or lower bound is less than
``tol``, otherwise, a ``ConvergenceWarning`` is raised.
If ``warm_start`` is ``True``, then ``n_init`` is ignored and a single
initialization is performed upon the first call. Upon consecutive
calls, training starts where it left off.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    List of n_features-dimensional data points. Each row
    corresponds to a single data point.

Returns
-------
self
*)

val fit_predict : ?y:Py.Object.t -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Estimate model parameters using X and predict the labels for X.

The method fits the model n_init times and sets the parameters with
which the model has the largest likelihood or lower bound. Within each
trial, the method iterates between E-step and M-step for `max_iter`
times until the change of likelihood or lower bound is less than
`tol`, otherwise, a :class:`~sklearn.exceptions.ConvergenceWarning` is
raised. After fitting, it predicts the most probable label for the
input data points.

.. versionadded:: 0.20

Parameters
----------
X : array-like, shape (n_samples, n_features)
    List of n_features-dimensional data points. Each row
    corresponds to a single data point.

Returns
-------
labels : array, shape (n_samples,)
    Component labels.
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

val predict : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Predict the labels for the data samples in X using trained model.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    List of n_features-dimensional data points. Each row
    corresponds to a single data point.

Returns
-------
labels : array, shape (n_samples,)
    Component labels.
*)

val predict_proba : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Predict posterior probability of each component given the data.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    List of n_features-dimensional data points. Each row
    corresponds to a single data point.

Returns
-------
resp : array, shape (n_samples, n_components)
    Returns the probability each Gaussian (state) in
    the model given each sample.
*)

val sample : ?n_samples:int -> [> tag] Obj.t -> ([>`ArrayLike] Np.Obj.t * [>`ArrayLike] Np.Obj.t)
(**
Generate random samples from the fitted Gaussian distribution.

Parameters
----------
n_samples : int, optional
    Number of samples to generate. Defaults to 1.

Returns
-------
X : array, shape (n_samples, n_features)
    Randomly generated sample

y : array, shape (nsamples,)
    Component labels
*)

val score : ?y:Py.Object.t -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> float
(**
Compute the per-sample average log-likelihood of the given data X.

Parameters
----------
X : array-like, shape (n_samples, n_dimensions)
    List of n_features-dimensional data points. Each row
    corresponds to a single data point.

Returns
-------
log_likelihood : float
    Log likelihood of the Gaussian mixture given X.
*)

val score_samples : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Compute the weighted log probabilities for each sample.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    List of n_features-dimensional data points. Each row
    corresponds to a single data point.

Returns
-------
log_prob : array, shape (n_samples,)
    Log probabilities of each data point in X.
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


(** Attribute weights_: get value or raise Not_found if None.*)
val weights_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute weights_: get value as an option. *)
val weights_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute means_: get value or raise Not_found if None.*)
val means_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute means_: get value as an option. *)
val means_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute covariances_: get value or raise Not_found if None.*)
val covariances_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute covariances_: get value as an option. *)
val covariances_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute precisions_: get value or raise Not_found if None.*)
val precisions_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute precisions_: get value as an option. *)
val precisions_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute precisions_cholesky_: get value or raise Not_found if None.*)
val precisions_cholesky_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute precisions_cholesky_: get value as an option. *)
val precisions_cholesky_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute converged_: get value or raise Not_found if None.*)
val converged_ : t -> bool

(** Attribute converged_: get value as an option. *)
val converged_opt : t -> (bool) option


(** Attribute n_iter_: get value or raise Not_found if None.*)
val n_iter_ : t -> int

(** Attribute n_iter_: get value as an option. *)
val n_iter_opt : t -> (int) option


(** Attribute lower_bound_: get value or raise Not_found if None.*)
val lower_bound_ : t -> float

(** Attribute lower_bound_: get value as an option. *)
val lower_bound_opt : t -> (float) option


(** Attribute weight_concentration_prior_: get value or raise Not_found if None.*)
val weight_concentration_prior_ : t -> Py.Object.t

(** Attribute weight_concentration_prior_: get value as an option. *)
val weight_concentration_prior_opt : t -> (Py.Object.t) option


(** Attribute weight_concentration_: get value or raise Not_found if None.*)
val weight_concentration_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute weight_concentration_: get value as an option. *)
val weight_concentration_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute mean_precision_prior_: get value or raise Not_found if None.*)
val mean_precision_prior_ : t -> float

(** Attribute mean_precision_prior_: get value as an option. *)
val mean_precision_prior_opt : t -> (float) option


(** Attribute mean_precision_: get value or raise Not_found if None.*)
val mean_precision_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute mean_precision_: get value as an option. *)
val mean_precision_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute mean_prior_: get value or raise Not_found if None.*)
val mean_prior_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute mean_prior_: get value as an option. *)
val mean_prior_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute degrees_of_freedom_prior_: get value or raise Not_found if None.*)
val degrees_of_freedom_prior_ : t -> float

(** Attribute degrees_of_freedom_prior_: get value as an option. *)
val degrees_of_freedom_prior_opt : t -> (float) option


(** Attribute degrees_of_freedom_: get value or raise Not_found if None.*)
val degrees_of_freedom_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute degrees_of_freedom_: get value as an option. *)
val degrees_of_freedom_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute covariance_prior_: get value or raise Not_found if None.*)
val covariance_prior_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute covariance_prior_: get value as an option. *)
val covariance_prior_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module GaussianMixture : sig
type tag = [`GaussianMixture]
type t = [`BaseEstimator | `BaseMixture | `DensityMixin | `GaussianMixture | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_density : t -> [`DensityMixin] Obj.t
val as_estimator : t -> [`BaseEstimator] Obj.t
val as_mixture : t -> [`BaseMixture] Obj.t
val create : ?n_components:int -> ?covariance_type:[`Full | `Tied | `Diag | `Spherical] -> ?tol:float -> ?reg_covar:float -> ?max_iter:int -> ?n_init:int -> ?init_params:[`Kmeans | `Random] -> ?weights_init:[>`ArrayLike] Np.Obj.t -> ?means_init:[>`ArrayLike] Np.Obj.t -> ?precisions_init:[>`ArrayLike] Np.Obj.t -> ?random_state:int -> ?warm_start:bool -> ?verbose:int -> ?verbose_interval:int -> unit -> t
(**
Gaussian Mixture.

Representation of a Gaussian mixture model probability distribution.
This class allows to estimate the parameters of a Gaussian mixture
distribution.

Read more in the :ref:`User Guide <gmm>`.

.. versionadded:: 0.18

Parameters
----------
n_components : int, defaults to 1.
    The number of mixture components.

covariance_type : {'full' (default), 'tied', 'diag', 'spherical'}
    String describing the type of covariance parameters to use.
    Must be one of:

    'full'
        each component has its own general covariance matrix
    'tied'
        all components share the same general covariance matrix
    'diag'
        each component has its own diagonal covariance matrix
    'spherical'
        each component has its own single variance

tol : float, defaults to 1e-3.
    The convergence threshold. EM iterations will stop when the
    lower bound average gain is below this threshold.

reg_covar : float, defaults to 1e-6.
    Non-negative regularization added to the diagonal of covariance.
    Allows to assure that the covariance matrices are all positive.

max_iter : int, defaults to 100.
    The number of EM iterations to perform.

n_init : int, defaults to 1.
    The number of initializations to perform. The best results are kept.

init_params : {'kmeans', 'random'}, defaults to 'kmeans'.
    The method used to initialize the weights, the means and the
    precisions.
    Must be one of::

        'kmeans' : responsibilities are initialized using kmeans.
        'random' : responsibilities are initialized randomly.

weights_init : array-like, shape (n_components, ), optional
    The user-provided initial weights, defaults to None.
    If it None, weights are initialized using the `init_params` method.

means_init : array-like, shape (n_components, n_features), optional
    The user-provided initial means, defaults to None,
    If it None, means are initialized using the `init_params` method.

precisions_init : array-like, optional.
    The user-provided initial precisions (inverse of the covariance
    matrices), defaults to None.
    If it None, precisions are initialized using the 'init_params' method.
    The shape depends on 'covariance_type'::

        (n_components,)                        if 'spherical',
        (n_features, n_features)               if 'tied',
        (n_components, n_features)             if 'diag',
        (n_components, n_features, n_features) if 'full'

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

warm_start : bool, default to False.
    If 'warm_start' is True, the solution of the last fitting is used as
    initialization for the next call of fit(). This can speed up
    convergence when fit is called several times on similar problems.
    In that case, 'n_init' is ignored and only a single initialization
    occurs upon the first call.
    See :term:`the Glossary <warm_start>`.

verbose : int, default to 0.
    Enable verbose output. If 1 then it prints the current
    initialization and each iteration step. If greater than 1 then
    it prints also the log probability and the time needed
    for each step.

verbose_interval : int, default to 10.
    Number of iteration done before the next print.

Attributes
----------
weights_ : array-like, shape (n_components,)
    The weights of each mixture components.

means_ : array-like, shape (n_components, n_features)
    The mean of each mixture component.

covariances_ : array-like
    The covariance of each mixture component.
    The shape depends on `covariance_type`::

        (n_components,)                        if 'spherical',
        (n_features, n_features)               if 'tied',
        (n_components, n_features)             if 'diag',
        (n_components, n_features, n_features) if 'full'

precisions_ : array-like
    The precision matrices for each component in the mixture. A precision
    matrix is the inverse of a covariance matrix. A covariance matrix is
    symmetric positive definite so the mixture of Gaussian can be
    equivalently parameterized by the precision matrices. Storing the
    precision matrices instead of the covariance matrices makes it more
    efficient to compute the log-likelihood of new samples at test time.
    The shape depends on `covariance_type`::

        (n_components,)                        if 'spherical',
        (n_features, n_features)               if 'tied',
        (n_components, n_features)             if 'diag',
        (n_components, n_features, n_features) if 'full'

precisions_cholesky_ : array-like
    The cholesky decomposition of the precision matrices of each mixture
    component. A precision matrix is the inverse of a covariance matrix.
    A covariance matrix is symmetric positive definite so the mixture of
    Gaussian can be equivalently parameterized by the precision matrices.
    Storing the precision matrices instead of the covariance matrices makes
    it more efficient to compute the log-likelihood of new samples at test
    time. The shape depends on `covariance_type`::

        (n_components,)                        if 'spherical',
        (n_features, n_features)               if 'tied',
        (n_components, n_features)             if 'diag',
        (n_components, n_features, n_features) if 'full'

converged_ : bool
    True when convergence was reached in fit(), False otherwise.

n_iter_ : int
    Number of step used by the best fit of EM to reach the convergence.

lower_bound_ : float
    Lower bound value on the log-likelihood (of the training data with
    respect to the model) of the best fit of EM.

See Also
--------
BayesianGaussianMixture : Gaussian mixture model fit with a variational
    inference.
*)

val aic : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> float
(**
Akaike information criterion for the current model on the input X.

Parameters
----------
X : array of shape (n_samples, n_dimensions)

Returns
-------
aic : float
    The lower the better.
*)

val bic : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> float
(**
Bayesian information criterion for the current model on the input X.

Parameters
----------
X : array of shape (n_samples, n_dimensions)

Returns
-------
bic : float
    The lower the better.
*)

val fit : ?y:Py.Object.t -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Estimate model parameters with the EM algorithm.

The method fits the model ``n_init`` times and sets the parameters with
which the model has the largest likelihood or lower bound. Within each
trial, the method iterates between E-step and M-step for ``max_iter``
times until the change of likelihood or lower bound is less than
``tol``, otherwise, a ``ConvergenceWarning`` is raised.
If ``warm_start`` is ``True``, then ``n_init`` is ignored and a single
initialization is performed upon the first call. Upon consecutive
calls, training starts where it left off.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    List of n_features-dimensional data points. Each row
    corresponds to a single data point.

Returns
-------
self
*)

val fit_predict : ?y:Py.Object.t -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Estimate model parameters using X and predict the labels for X.

The method fits the model n_init times and sets the parameters with
which the model has the largest likelihood or lower bound. Within each
trial, the method iterates between E-step and M-step for `max_iter`
times until the change of likelihood or lower bound is less than
`tol`, otherwise, a :class:`~sklearn.exceptions.ConvergenceWarning` is
raised. After fitting, it predicts the most probable label for the
input data points.

.. versionadded:: 0.20

Parameters
----------
X : array-like, shape (n_samples, n_features)
    List of n_features-dimensional data points. Each row
    corresponds to a single data point.

Returns
-------
labels : array, shape (n_samples,)
    Component labels.
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

val predict : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Predict the labels for the data samples in X using trained model.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    List of n_features-dimensional data points. Each row
    corresponds to a single data point.

Returns
-------
labels : array, shape (n_samples,)
    Component labels.
*)

val predict_proba : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Predict posterior probability of each component given the data.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    List of n_features-dimensional data points. Each row
    corresponds to a single data point.

Returns
-------
resp : array, shape (n_samples, n_components)
    Returns the probability each Gaussian (state) in
    the model given each sample.
*)

val sample : ?n_samples:int -> [> tag] Obj.t -> ([>`ArrayLike] Np.Obj.t * [>`ArrayLike] Np.Obj.t)
(**
Generate random samples from the fitted Gaussian distribution.

Parameters
----------
n_samples : int, optional
    Number of samples to generate. Defaults to 1.

Returns
-------
X : array, shape (n_samples, n_features)
    Randomly generated sample

y : array, shape (nsamples,)
    Component labels
*)

val score : ?y:Py.Object.t -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> float
(**
Compute the per-sample average log-likelihood of the given data X.

Parameters
----------
X : array-like, shape (n_samples, n_dimensions)
    List of n_features-dimensional data points. Each row
    corresponds to a single data point.

Returns
-------
log_likelihood : float
    Log likelihood of the Gaussian mixture given X.
*)

val score_samples : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Compute the weighted log probabilities for each sample.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    List of n_features-dimensional data points. Each row
    corresponds to a single data point.

Returns
-------
log_prob : array, shape (n_samples,)
    Log probabilities of each data point in X.
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


(** Attribute weights_: get value or raise Not_found if None.*)
val weights_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute weights_: get value as an option. *)
val weights_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute means_: get value or raise Not_found if None.*)
val means_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute means_: get value as an option. *)
val means_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute covariances_: get value or raise Not_found if None.*)
val covariances_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute covariances_: get value as an option. *)
val covariances_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute precisions_: get value or raise Not_found if None.*)
val precisions_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute precisions_: get value as an option. *)
val precisions_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute precisions_cholesky_: get value or raise Not_found if None.*)
val precisions_cholesky_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute precisions_cholesky_: get value as an option. *)
val precisions_cholesky_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute converged_: get value or raise Not_found if None.*)
val converged_ : t -> bool

(** Attribute converged_: get value as an option. *)
val converged_opt : t -> (bool) option


(** Attribute n_iter_: get value or raise Not_found if None.*)
val n_iter_ : t -> int

(** Attribute n_iter_: get value as an option. *)
val n_iter_opt : t -> (int) option


(** Attribute lower_bound_: get value or raise Not_found if None.*)
val lower_bound_ : t -> float

(** Attribute lower_bound_: get value as an option. *)
val lower_bound_opt : t -> (float) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

