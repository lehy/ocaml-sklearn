module DictionaryLearning : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_components:int -> ?alpha:float -> ?max_iter:int -> ?tol:float -> ?fit_algorithm:[`Lars | `Cd] -> ?transform_algorithm:[`Lasso_lars | `Lasso_cd | `Lars | `Omp | `Threshold] -> ?transform_n_nonzero_coefs:int -> ?transform_alpha:float -> ?n_jobs:[`Int of int | `None] -> ?code_init:Ndarray.t -> ?dict_init:Ndarray.t -> ?verbose:bool -> ?split_sign:bool -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> ?positive_code:bool -> ?positive_dict:bool -> ?transform_max_iter:int -> unit -> t
(**
Dictionary learning

Finds a dictionary (a set of atoms) that can best be used to represent data
using a sparse code.

Solves the optimization problem::

    (U^*,V^* ) = argmin 0.5 || Y - U V ||_2^2 + alpha * || U ||_1
                (U,V)
                with || V_k ||_2 = 1 for all  0 <= k < n_components

Read more in the :ref:`User Guide <DictionaryLearning>`.

Parameters
----------
n_components : int, default=n_features
    number of dictionary elements to extract

alpha : float, default=1.0
    sparsity controlling parameter

max_iter : int, default=1000
    maximum number of iterations to perform

tol : float, default=1e-8
    tolerance for numerical error

fit_algorithm : {'lars', 'cd'}, default='lars'
    lars: uses the least angle regression method to solve the lasso problem
    (linear_model.lars_path)
    cd: uses the coordinate descent method to compute the
    Lasso solution (linear_model.Lasso). Lars will be faster if
    the estimated components are sparse.

    .. versionadded:: 0.17
       *cd* coordinate descent method to improve speed.

transform_algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp',     'threshold'}, default='omp'
    Algorithm used to transform the data
    lars: uses the least angle regression method (linear_model.lars_path)
    lasso_lars: uses Lars to compute the Lasso solution
    lasso_cd: uses the coordinate descent method to compute the
    Lasso solution (linear_model.Lasso). lasso_lars will be faster if
    the estimated components are sparse.
    omp: uses orthogonal matching pursuit to estimate the sparse solution
    threshold: squashes to zero all coefficients less than alpha from
    the projection ``dictionary * X'``

    .. versionadded:: 0.17
       *lasso_cd* coordinate descent method to improve speed.

transform_n_nonzero_coefs : int, default=0.1*n_features
    Number of nonzero coefficients to target in each column of the
    solution. This is only used by `algorithm='lars'` and `algorithm='omp'`
    and is overridden by `alpha` in the `omp` case.

transform_alpha : float, default=1.0
    If `algorithm='lasso_lars'` or `algorithm='lasso_cd'`, `alpha` is the
    penalty applied to the L1 norm.
    If `algorithm='threshold'`, `alpha` is the absolute value of the
    threshold below which coefficients will be squashed to zero.
    If `algorithm='omp'`, `alpha` is the tolerance parameter: the value of
    the reconstruction error targeted. In this case, it overrides
    `n_nonzero_coefs`.

n_jobs : int or None, default=None
    Number of parallel jobs to run.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

code_init : array of shape (n_samples, n_components), default=None
    initial value for the code, for warm restart

dict_init : array of shape (n_components, n_features), default=None
    initial values for the dictionary, for warm restart

verbose : bool, default=False
    To control the verbosity of the procedure.

split_sign : bool, default=False
    Whether to split the sparse feature vector into the concatenation of
    its negative part and its positive part. This can improve the
    performance of downstream classifiers.

random_state : int, RandomState instance or None, default=None
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

positive_code : bool, default=False
    Whether to enforce positivity when finding the code.

    .. versionadded:: 0.20

positive_dict : bool, default=False
    Whether to enforce positivity when finding the dictionary

    .. versionadded:: 0.20

transform_max_iter : int, default=1000
    Maximum number of iterations to perform if `algorithm='lasso_cd'` or
    `lasso_lars`.

    .. versionadded:: 0.22

Attributes
----------
components_ : array, [n_components, n_features]
    dictionary atoms extracted from the data

error_ : array
    vector of errors at each iteration

n_iter_ : int
    Number of iterations run.

Notes
-----
**References:**

J. Mairal, F. Bach, J. Ponce, G. Sapiro, 2009: Online dictionary learning
for sparse coding (https://www.di.ens.fr/sierra/pdfs/icml09.pdf)

See also
--------
SparseCoder
MiniBatchDictionaryLearning
SparsePCA
MiniBatchSparsePCA
*)

val fit : ?y:Py.Object.t -> x:Ndarray.t -> t -> t
(**
Fit the model from data in X.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training vector, where n_samples in the number of samples
    and n_features is the number of features.

y : Ignored

Returns
-------
self : object
    Returns the object itself
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
Encode the data as a sparse combination of the dictionary atoms.

Coding method is determined by the object parameter
`transform_algorithm`.

Parameters
----------
X : array of shape (n_samples, n_features)
    Test data to be transformed, must have the same number of
    features as the data used to train the model.

Returns
-------
X_new : array, shape (n_samples, n_components)
    Transformed data
*)


(** Attribute components_: see constructor for documentation *)
val components_ : t -> Ndarray.t

(** Attribute error_: see constructor for documentation *)
val error_ : t -> Ndarray.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> int

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module FactorAnalysis : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_components:[`Int of int | `None] -> ?tol:float -> ?copy:bool -> ?max_iter:int -> ?noise_variance_init:[`Ndarray of Ndarray.t | `None] -> ?svd_method:[`Lapack | `Randomized] -> ?iterated_power:int -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> unit -> t
(**
Factor Analysis (FA)

A simple linear generative model with Gaussian latent variables.

The observations are assumed to be caused by a linear transformation of
lower dimensional latent factors and added Gaussian noise.
Without loss of generality the factors are distributed according to a
Gaussian with zero mean and unit covariance. The noise is also zero mean
and has an arbitrary diagonal covariance matrix.

If we would restrict the model further, by assuming that the Gaussian
noise is even isotropic (all diagonal entries are the same) we would obtain
:class:`PPCA`.

FactorAnalysis performs a maximum likelihood estimate of the so-called
`loading` matrix, the transformation of the latent variables to the
observed ones, using SVD based approach.

Read more in the :ref:`User Guide <FA>`.

.. versionadded:: 0.13

Parameters
----------
n_components : int | None
    Dimensionality of latent space, the number of components
    of ``X`` that are obtained after ``transform``.
    If None, n_components is set to the number of features.

tol : float
    Stopping tolerance for log-likelihood increase.

copy : bool
    Whether to make a copy of X. If ``False``, the input X gets overwritten
    during fitting.

max_iter : int
    Maximum number of iterations.

noise_variance_init : None | array, shape=(n_features,)
    The initial guess of the noise variance for each feature.
    If None, it defaults to np.ones(n_features)

svd_method : {'lapack', 'randomized'}
    Which SVD method to use. If 'lapack' use standard SVD from
    scipy.linalg, if 'randomized' use fast ``randomized_svd`` function.
    Defaults to 'randomized'. For most applications 'randomized' will
    be sufficiently precise while providing significant speed gains.
    Accuracy can also be improved by setting higher values for
    `iterated_power`. If this is not sufficient, for maximum precision
    you should choose 'lapack'.

iterated_power : int, optional
    Number of iterations for the power method. 3 by default. Only used
    if ``svd_method`` equals 'randomized'

random_state : int, RandomState instance or None, optional (default=0)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`. Only used when ``svd_method`` equals 'randomized'.

Attributes
----------
components_ : array, [n_components, n_features]
    Components with maximum variance.

loglike_ : list, [n_iterations]
    The log likelihood at each iteration.

noise_variance_ : array, shape=(n_features,)
    The estimated noise variance for each feature.

n_iter_ : int
    Number of iterations run.

mean_ : array, shape (n_features,)
    Per-feature empirical mean, estimated from the training set.

Examples
--------
>>> from sklearn.datasets import load_digits
>>> from sklearn.decomposition import FactorAnalysis
>>> X, _ = load_digits(return_X_y=True)
>>> transformer = FactorAnalysis(n_components=7, random_state=0)
>>> X_transformed = transformer.fit_transform(X)
>>> X_transformed.shape
(1797, 7)

References
----------
.. David Barber, Bayesian Reasoning and Machine Learning,
    Algorithm 21.1

.. Christopher M. Bishop: Pattern Recognition and Machine Learning,
    Chapter 12.2.4

See also
--------
PCA: Principal component analysis is also a latent linear variable model
    which however assumes equal noise variance for each feature.
    This extra assumption makes probabilistic PCA faster as it can be
    computed in closed form.
FastICA: Independent component analysis, a latent variable model with
    non-Gaussian latent variables.
*)

val fit : ?y:Py.Object.t -> x:Ndarray.t -> t -> t
(**
Fit the FactorAnalysis model to X using SVD based approach

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training data.

y : Ignored

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

val get_covariance : t -> Ndarray.t
(**
Compute data covariance with the FactorAnalysis model.

``cov = components_.T * components_ + diag(noise_variance)``

Returns
-------
cov : array, shape (n_features, n_features)
    Estimated covariance of data.
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
Compute data precision matrix with the FactorAnalysis model.

Returns
-------
precision : array, shape (n_features, n_features)
    Estimated precision of data.
*)

val score : ?y:Py.Object.t -> x:Ndarray.t -> t -> float
(**
Compute the average log-likelihood of the samples

Parameters
----------
X : array, shape (n_samples, n_features)
    The data

y : Ignored

Returns
-------
ll : float
    Average log-likelihood of the samples under the current model
*)

val score_samples : x:Ndarray.t -> t -> Ndarray.t
(**
Compute the log-likelihood of each sample

Parameters
----------
X : array, shape (n_samples, n_features)
    The data

Returns
-------
ll : array, shape (n_samples,)
    Log-likelihood of each sample under the current model
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
Apply dimensionality reduction to X using the model.

Compute the expected mean of the latent variables.
See Barber, 21.2.33 (or Bishop, 12.66).

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training data.

Returns
-------
X_new : array-like, shape (n_samples, n_components)
    The latent variables of X.
*)


(** Attribute components_: see constructor for documentation *)
val components_ : t -> Ndarray.t

(** Attribute loglike_: see constructor for documentation *)
val loglike_ : t -> Py.Object.t

(** Attribute noise_variance_: see constructor for documentation *)
val noise_variance_ : t -> Ndarray.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> int

(** Attribute mean_: see constructor for documentation *)
val mean_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module FastICA : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_components:int -> ?algorithm:[`Parallel | `Deflation] -> ?whiten:bool -> ?fun_:[`String of string | `Callable of Py.Object.t] -> ?fun_args:Py.Object.t -> ?max_iter:int -> ?tol:float -> ?w_init:Py.Object.t -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> unit -> t
(**
FastICA: a fast algorithm for Independent Component Analysis.

Read more in the :ref:`User Guide <ICA>`.

Parameters
----------
n_components : int, optional
    Number of components to use. If none is passed, all are used.

algorithm : {'parallel', 'deflation'}
    Apply parallel or deflational algorithm for FastICA.

whiten : boolean, optional
    If whiten is false, the data is already considered to be
    whitened, and no whitening is performed.

fun : string or function, optional. Default: 'logcosh'
    The functional form of the G function used in the
    approximation to neg-entropy. Could be either 'logcosh', 'exp',
    or 'cube'.
    You can also provide your own function. It should return a tuple
    containing the value of the function, and of its derivative, in the
    point. Example:

    def my_g(x):
        return x ** 3, (3 * x ** 2).mean(axis=-1)

fun_args : dictionary, optional
    Arguments to send to the functional form.
    If empty and if fun='logcosh', fun_args will take value
    {'alpha' : 1.0}.

max_iter : int, optional
    Maximum number of iterations during fit.

tol : float, optional
    Tolerance on update at each iteration.

w_init : None of an (n_components, n_components) ndarray
    The mixing matrix to be used to initialize the algorithm.

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

Attributes
----------
components_ : 2D array, shape (n_components, n_features)
    The linear operator to apply to the data to get the independent
    sources. This is equal to the unmixing matrix when ``whiten`` is
    False, and equal to ``np.dot(unmixing_matrix, self.whitening_)`` when
    ``whiten`` is True.

mixing_ : array, shape (n_features, n_components)
    The pseudo-inverse of ``components_``. It is the linear operator
    that maps independent sources to the data.

mean_ : array, shape(n_features)
    The mean over features. Only set if `self.whiten` is True.

n_iter_ : int
    If the algorithm is "deflation", n_iter is the
    maximum number of iterations run across all components. Else
    they are just the number of iterations taken to converge.

whitening_ : array, shape (n_components, n_features)
    Only set if whiten is 'True'. This is the pre-whitening matrix
    that projects data onto the first `n_components` principal components.

Examples
--------
>>> from sklearn.datasets import load_digits
>>> from sklearn.decomposition import FastICA
>>> X, _ = load_digits(return_X_y=True)
>>> transformer = FastICA(n_components=7,
...         random_state=0)
>>> X_transformed = transformer.fit_transform(X)
>>> X_transformed.shape
(1797, 7)

Notes
-----
Implementation based on
*A. Hyvarinen and E. Oja, Independent Component Analysis:
Algorithms and Applications, Neural Networks, 13(4-5), 2000,
pp. 411-430*
*)

val fit : ?y:Py.Object.t -> x:Ndarray.t -> t -> t
(**
Fit the model to X.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training data, where n_samples is the number of samples
    and n_features is the number of features.

y : Ignored

Returns
-------
self
*)

val fit_transform : ?y:Py.Object.t -> x:Ndarray.t -> t -> Ndarray.t
(**
Fit the model and recover the sources from X.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training data, where n_samples is the number of samples
    and n_features is the number of features.

y : Ignored

Returns
-------
X_new : array-like, shape (n_samples, n_components)
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

val inverse_transform : ?copy:Py.Object.t -> x:Ndarray.t -> t -> Ndarray.t
(**
Transform the sources back to the mixed data (apply mixing matrix).

Parameters
----------
X : array-like, shape (n_samples, n_components)
    Sources, where n_samples is the number of samples
    and n_components is the number of components.
copy : bool (optional)
    If False, data passed to fit are overwritten. Defaults to True.

Returns
-------
X_new : array-like, shape (n_samples, n_features)
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

val transform : ?copy:bool -> x:Ndarray.t -> t -> Ndarray.t
(**
Recover the sources from X (apply the unmixing matrix).

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Data to transform, where n_samples is the number of samples
    and n_features is the number of features.

copy : bool (optional)
    If False, data passed to fit are overwritten. Defaults to True.

Returns
-------
X_new : array-like, shape (n_samples, n_components)
*)


(** Attribute components_: see constructor for documentation *)
val components_ : t -> Py.Object.t

(** Attribute mixing_: see constructor for documentation *)
val mixing_ : t -> Ndarray.t

(** Attribute mean_: see constructor for documentation *)
val mean_ : t -> Ndarray.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> int

(** Attribute whitening_: see constructor for documentation *)
val whitening_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module IncrementalPCA : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_components:[`Int of int | `None] -> ?whiten:bool -> ?copy:bool -> ?batch_size:[`Int of int | `None] -> unit -> t
(**
Incremental principal components analysis (IPCA).

Linear dimensionality reduction using Singular Value Decomposition of
the data, keeping only the most significant singular vectors to
project the data to a lower dimensional space. The input data is centered
but not scaled for each feature before applying the SVD.

Depending on the size of the input data, this algorithm can be much more
memory efficient than a PCA, and allows sparse input.

This algorithm has constant memory complexity, on the order
of ``batch_size * n_features``, enabling use of np.memmap files without
loading the entire file into memory. For sparse matrices, the input
is converted to dense in batches (in order to be able to subtract the
mean) which avoids storing the entire dense matrix at any one time.

The computational overhead of each SVD is
``O(batch_size * n_features ** 2)``, but only 2 * batch_size samples
remain in memory at a time. There will be ``n_samples / batch_size`` SVD
computations to get the principal components, versus 1 large SVD of
complexity ``O(n_samples * n_features ** 2)`` for PCA.

Read more in the :ref:`User Guide <IncrementalPCA>`.

.. versionadded:: 0.16

Parameters
----------
n_components : int or None, (default=None)
    Number of components to keep. If ``n_components `` is ``None``,
    then ``n_components`` is set to ``min(n_samples, n_features)``.

whiten : bool, optional
    When True (False by default) the ``components_`` vectors are divided
    by ``n_samples`` times ``components_`` to ensure uncorrelated outputs
    with unit component-wise variances.

    Whitening will remove some information from the transformed signal
    (the relative variance scales of the components) but can sometimes
    improve the predictive accuracy of the downstream estimators by
    making data respect some hard-wired assumptions.

copy : bool, (default=True)
    If False, X will be overwritten. ``copy=False`` can be used to
    save memory but is unsafe for general use.

batch_size : int or None, (default=None)
    The number of samples to use for each batch. Only used when calling
    ``fit``. If ``batch_size`` is ``None``, then ``batch_size``
    is inferred from the data and set to ``5 * n_features``, to provide a
    balance between approximation accuracy and memory consumption.

Attributes
----------
components_ : array, shape (n_components, n_features)
    Components with maximum variance.

explained_variance_ : array, shape (n_components,)
    Variance explained by each of the selected components.

explained_variance_ratio_ : array, shape (n_components,)
    Percentage of variance explained by each of the selected components.
    If all components are stored, the sum of explained variances is equal
    to 1.0.

singular_values_ : array, shape (n_components,)
    The singular values corresponding to each of the selected components.
    The singular values are equal to the 2-norms of the ``n_components``
    variables in the lower-dimensional space.

mean_ : array, shape (n_features,)
    Per-feature empirical mean, aggregate over calls to ``partial_fit``.

var_ : array, shape (n_features,)
    Per-feature empirical variance, aggregate over calls to
    ``partial_fit``.

noise_variance_ : float
    The estimated noise covariance following the Probabilistic PCA model
    from Tipping and Bishop 1999. See "Pattern Recognition and
    Machine Learning" by C. Bishop, 12.2.1 p. 574 or
    http://www.miketipping.com/papers/met-mppca.pdf.

n_components_ : int
    The estimated number of components. Relevant when
    ``n_components=None``.

n_samples_seen_ : int
    The number of samples processed by the estimator. Will be reset on
    new calls to fit, but increments across ``partial_fit`` calls.

Examples
--------
>>> from sklearn.datasets import load_digits
>>> from sklearn.decomposition import IncrementalPCA
>>> from scipy import sparse
>>> X, _ = load_digits(return_X_y=True)
>>> transformer = IncrementalPCA(n_components=7, batch_size=200)
>>> # either partially fit on smaller batches of data
>>> transformer.partial_fit(X[:100, :])
IncrementalPCA(batch_size=200, n_components=7)
>>> # or let the fit function itself divide the data into batches
>>> X_sparse = sparse.csr_matrix(X)
>>> X_transformed = transformer.fit_transform(X_sparse)
>>> X_transformed.shape
(1797, 7)

Notes
-----
Implements the incremental PCA model from:
*D. Ross, J. Lim, R. Lin, M. Yang, Incremental Learning for Robust Visual
Tracking, International Journal of Computer Vision, Volume 77, Issue 1-3,
pp. 125-141, May 2008.*
See https://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf

This model is an extension of the Sequential Karhunen-Loeve Transform from:
*A. Levy and M. Lindenbaum, Sequential Karhunen-Loeve Basis Extraction and
its Application to Images, IEEE Transactions on Image Processing, Volume 9,
Number 8, pp. 1371-1374, August 2000.*
See https://www.cs.technion.ac.il/~mic/doc/skl-ip.pdf

We have specifically abstained from an optimization used by authors of both
papers, a QR decomposition used in specific situations to reduce the
algorithmic complexity of the SVD. The source for this technique is
*Matrix Computations, Third Edition, G. Holub and C. Van Loan, Chapter 5,
section 5.4.4, pp 252-253.*. This technique has been omitted because it is
advantageous only when decomposing a matrix with ``n_samples`` (rows)
>= 5/3 * ``n_features`` (columns), and hurts the readability of the
implemented algorithm. This would be a good opportunity for future
optimization, if it is deemed necessary.

References
----------
D. Ross, J. Lim, R. Lin, M. Yang. Incremental Learning for Robust Visual
Tracking, International Journal of Computer Vision, Volume 77,
Issue 1-3, pp. 125-141, May 2008.

G. Golub and C. Van Loan. Matrix Computations, Third Edition, Chapter 5,
Section 5.4.4, pp. 252-253.

See also
--------
PCA
KernelPCA
SparsePCA
TruncatedSVD
*)

val fit : ?y:Py.Object.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> t
(**
Fit the model with X, using minibatches of size batch_size.

Parameters
----------
X : array-like or sparse matrix, shape (n_samples, n_features)
    Training data, where n_samples is the number of samples and
    n_features is the number of features.

y : Ignored

Returns
-------
self : object
    Returns the instance itself.
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

val get_covariance : t -> Ndarray.t
(**
Compute data covariance with the generative model.

``cov = components_.T * S**2 * components_ + sigma2 * eye(n_features)``
where S**2 contains the explained variances, and sigma2 contains the
noise variances.

Returns
-------
cov : array, shape=(n_features, n_features)
    Estimated covariance of data.
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
Compute data precision matrix with the generative model.

Equals the inverse of the covariance but computed with
the matrix inversion lemma for efficiency.

Returns
-------
precision : array, shape=(n_features, n_features)
    Estimated precision of data.
*)

val inverse_transform : x:Ndarray.t -> t -> Py.Object.t
(**
Transform data back to its original space.

In other words, return an input X_original whose transform would be X.

Parameters
----------
X : array-like, shape (n_samples, n_components)
    New data, where n_samples is the number of samples
    and n_components is the number of components.

Returns
-------
X_original array-like, shape (n_samples, n_features)

Notes
-----
If whitening is enabled, inverse_transform will compute the
exact inverse operation, which includes reversing whitening.
*)

val partial_fit : ?y:Py.Object.t -> ?check_input:Py.Object.t -> x:Ndarray.t -> t -> t
(**
Incremental fit with X. All of X is processed as a single batch.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training data, where n_samples is the number of samples and
    n_features is the number of features.
check_input : bool
    Run check_array on X.

y : Ignored

Returns
-------
self : object
    Returns the instance itself.
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
Apply dimensionality reduction to X.

X is projected on the first principal components previously extracted
from a training set, using minibatches of size batch_size if X is
sparse.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    New data, where n_samples is the number of samples
    and n_features is the number of features.

Returns
-------
X_new : array-like, shape (n_samples, n_components)

Examples
--------

>>> import numpy as np
>>> from sklearn.decomposition import IncrementalPCA
>>> X = np.array([[-1, -1], [-2, -1], [-3, -2],
...               [1, 1], [2, 1], [3, 2]])
>>> ipca = IncrementalPCA(n_components=2, batch_size=3)
>>> ipca.fit(X)
IncrementalPCA(batch_size=3, n_components=2)
>>> ipca.transform(X) # doctest: +SKIP
*)


(** Attribute components_: see constructor for documentation *)
val components_ : t -> Ndarray.t

(** Attribute explained_variance_: see constructor for documentation *)
val explained_variance_ : t -> Ndarray.t

(** Attribute explained_variance_ratio_: see constructor for documentation *)
val explained_variance_ratio_ : t -> Ndarray.t

(** Attribute singular_values_: see constructor for documentation *)
val singular_values_ : t -> Ndarray.t

(** Attribute mean_: see constructor for documentation *)
val mean_ : t -> Ndarray.t

(** Attribute var_: see constructor for documentation *)
val var_ : t -> Ndarray.t

(** Attribute noise_variance_: see constructor for documentation *)
val noise_variance_ : t -> float

(** Attribute n_components_: see constructor for documentation *)
val n_components_ : t -> int

(** Attribute n_samples_seen_: see constructor for documentation *)
val n_samples_seen_ : t -> int

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module KernelPCA : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_components:int -> ?kernel:[`Linear | `Poly | `Rbf | `Sigmoid | `Cosine | `Precomputed] -> ?gamma:float -> ?degree:int -> ?coef0:float -> ?kernel_params:Py.Object.t -> ?alpha:int -> ?fit_inverse_transform:bool -> ?eigen_solver:[`Auto | `Dense | `Arpack] -> ?tol:float -> ?max_iter:int -> ?remove_zero_eig:bool -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> ?copy_X:bool -> ?n_jobs:[`Int of int | `None] -> unit -> t
(**
Kernel Principal component analysis (KPCA)

Non-linear dimensionality reduction through the use of kernels (see
:ref:`metrics`).

Read more in the :ref:`User Guide <kernel_PCA>`.

Parameters
----------
n_components : int, default=None
    Number of components. If None, all non-zero components are kept.

kernel : "linear" | "poly" | "rbf" | "sigmoid" | "cosine" | "precomputed"
    Kernel. Default="linear".

gamma : float, default=1/n_features
    Kernel coefficient for rbf, poly and sigmoid kernels. Ignored by other
    kernels.

degree : int, default=3
    Degree for poly kernels. Ignored by other kernels.

coef0 : float, default=1
    Independent term in poly and sigmoid kernels.
    Ignored by other kernels.

kernel_params : mapping of string to any, default=None
    Parameters (keyword arguments) and values for kernel passed as
    callable object. Ignored by other kernels.

alpha : int, default=1.0
    Hyperparameter of the ridge regression that learns the
    inverse transform (when fit_inverse_transform=True).

fit_inverse_transform : bool, default=False
    Learn the inverse transform for non-precomputed kernels.
    (i.e. learn to find the pre-image of a point)

eigen_solver : string ['auto'|'dense'|'arpack'], default='auto'
    Select eigensolver to use. If n_components is much less than
    the number of training samples, arpack may be more efficient
    than the dense eigensolver.

tol : float, default=0
    Convergence tolerance for arpack.
    If 0, optimal value will be chosen by arpack.

max_iter : int, default=None
    Maximum number of iterations for arpack.
    If None, optimal value will be chosen by arpack.

remove_zero_eig : boolean, default=False
    If True, then all components with zero eigenvalues are removed, so
    that the number of components in the output may be < n_components
    (and sometimes even zero due to numerical instability).
    When n_components is None, this parameter is ignored and components
    with zero eigenvalues are removed regardless.

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`. Used when ``eigen_solver`` == 'arpack'.

    .. versionadded:: 0.18

copy_X : boolean, default=True
    If True, input X is copied and stored by the model in the `X_fit_`
    attribute. If no further changes will be done to X, setting
    `copy_X=False` saves memory by storing a reference.

    .. versionadded:: 0.18

n_jobs : int or None, optional (default=None)
    The number of parallel jobs to run.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

    .. versionadded:: 0.18

Attributes
----------
lambdas_ : array, (n_components,)
    Eigenvalues of the centered kernel matrix in decreasing order.
    If `n_components` and `remove_zero_eig` are not set,
    then all values are stored.

alphas_ : array, (n_samples, n_components)
    Eigenvectors of the centered kernel matrix. If `n_components` and
    `remove_zero_eig` are not set, then all components are stored.

dual_coef_ : array, (n_samples, n_features)
    Inverse transform matrix. Only available when
    ``fit_inverse_transform`` is True.

X_transformed_fit_ : array, (n_samples, n_components)
    Projection of the fitted data on the kernel principal components.
    Only available when ``fit_inverse_transform`` is True.

X_fit_ : (n_samples, n_features)
    The data used to fit the model. If `copy_X=False`, then `X_fit_` is
    a reference. This attribute is used for the calls to transform.

Examples
--------
>>> from sklearn.datasets import load_digits
>>> from sklearn.decomposition import KernelPCA
>>> X, _ = load_digits(return_X_y=True)
>>> transformer = KernelPCA(n_components=7, kernel='linear')
>>> X_transformed = transformer.fit_transform(X)
>>> X_transformed.shape
(1797, 7)

References
----------
Kernel PCA was introduced in:
    Bernhard Schoelkopf, Alexander J. Smola,
    and Klaus-Robert Mueller. 1999. Kernel principal
    component analysis. In Advances in kernel methods,
    MIT Press, Cambridge, MA, USA 327-352.
*)

val fit : ?y:Py.Object.t -> x:Ndarray.t -> t -> t
(**
Fit the model from data in X.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training vector, where n_samples in the number of samples
    and n_features is the number of features.

Returns
-------
self : object
    Returns the instance itself.
*)

val fit_transform : ?y:Py.Object.t -> ?params:(string * Py.Object.t) list -> x:Ndarray.t -> t -> Ndarray.t
(**
Fit the model from data in X and transform X.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training vector, where n_samples in the number of samples
    and n_features is the number of features.

Returns
-------
X_new : array-like, shape (n_samples, n_components)
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

val inverse_transform : x:Ndarray.t -> t -> Ndarray.t
(**
Transform X back to original space.

Parameters
----------
X : array-like, shape (n_samples, n_components)

Returns
-------
X_new : array-like, shape (n_samples, n_features)

References
----------
"Learning to Find Pre-Images", G BakIr et al, 2004.
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
Transform X.

Parameters
----------
X : array-like, shape (n_samples, n_features)

Returns
-------
X_new : array-like, shape (n_samples, n_components)
*)


(** Attribute lambdas_: see constructor for documentation *)
val lambdas_ : t -> Py.Object.t

(** Attribute alphas_: see constructor for documentation *)
val alphas_ : t -> Py.Object.t

(** Attribute dual_coef_: see constructor for documentation *)
val dual_coef_ : t -> Py.Object.t

(** Attribute X_transformed_fit_: see constructor for documentation *)
val x_transformed_fit_ : t -> Py.Object.t

(** Attribute X_fit_: see constructor for documentation *)
val x_fit_ : t -> Py.Object.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module LatentDirichletAllocation : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_components:int -> ?doc_topic_prior:float -> ?topic_word_prior:float -> ?learning_method:[`Batch | `Online] -> ?learning_decay:float -> ?learning_offset:float -> ?max_iter:int -> ?batch_size:int -> ?evaluate_every:int -> ?total_samples:int -> ?perp_tol:float -> ?mean_change_tol:float -> ?max_doc_update_iter:int -> ?n_jobs:[`Int of int | `None] -> ?verbose:int -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> unit -> t
(**
Latent Dirichlet Allocation with online variational Bayes algorithm

.. versionadded:: 0.17

Read more in the :ref:`User Guide <LatentDirichletAllocation>`.

Parameters
----------
n_components : int, optional (default=10)
    Number of topics.

doc_topic_prior : float, optional (default=None)
    Prior of document topic distribution `theta`. If the value is None,
    defaults to `1 / n_components`.
    In [1]_, this is called `alpha`.

topic_word_prior : float, optional (default=None)
    Prior of topic word distribution `beta`. If the value is None, defaults
    to `1 / n_components`.
    In [1]_, this is called `eta`.

learning_method : 'batch' | 'online', default='batch'
    Method used to update `_component`. Only used in :meth:`fit` method.
    In general, if the data size is large, the online update will be much
    faster than the batch update.

    Valid options::

        'batch': Batch variational Bayes method. Use all training data in
            each EM update.
            Old `components_` will be overwritten in each iteration.
        'online': Online variational Bayes method. In each EM update, use
            mini-batch of training data to update the ``components_``
            variable incrementally. The learning rate is controlled by the
            ``learning_decay`` and the ``learning_offset`` parameters.

    .. versionchanged:: 0.20
        The default learning method is now ``"batch"``.

learning_decay : float, optional (default=0.7)
    It is a parameter that control learning rate in the online learning
    method. The value should be set between (0.5, 1.0] to guarantee
    asymptotic convergence. When the value is 0.0 and batch_size is
    ``n_samples``, the update method is same as batch learning. In the
    literature, this is called kappa.

learning_offset : float, optional (default=10.)
    A (positive) parameter that downweights early iterations in online
    learning.  It should be greater than 1.0. In the literature, this is
    called tau_0.

max_iter : integer, optional (default=10)
    The maximum number of iterations.

batch_size : int, optional (default=128)
    Number of documents to use in each EM iteration. Only used in online
    learning.

evaluate_every : int, optional (default=0)
    How often to evaluate perplexity. Only used in `fit` method.
    set it to 0 or negative number to not evaluate perplexity in
    training at all. Evaluating perplexity can help you check convergence
    in training process, but it will also increase total training time.
    Evaluating perplexity in every iteration might increase training time
    up to two-fold.

total_samples : int, optional (default=1e6)
    Total number of documents. Only used in the :meth:`partial_fit` method.

perp_tol : float, optional (default=1e-1)
    Perplexity tolerance in batch learning. Only used when
    ``evaluate_every`` is greater than 0.

mean_change_tol : float, optional (default=1e-3)
    Stopping tolerance for updating document topic distribution in E-step.

max_doc_update_iter : int (default=100)
    Max number of iterations for updating document topic distribution in
    the E-step.

n_jobs : int or None, optional (default=None)
    The number of jobs to use in the E-step.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

verbose : int, optional (default=0)
    Verbosity level.

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

Attributes
----------
components_ : array, [n_components, n_features]
    Variational parameters for topic word distribution. Since the complete
    conditional for topic word distribution is a Dirichlet,
    ``components_[i, j]`` can be viewed as pseudocount that represents the
    number of times word `j` was assigned to topic `i`.
    It can also be viewed as distribution over the words for each topic
    after normalization:
    ``model.components_ / model.components_.sum(axis=1)[:, np.newaxis]``.

n_batch_iter_ : int
    Number of iterations of the EM step.

n_iter_ : int
    Number of passes over the dataset.

bound_ : float
    Final perplexity score on training set.

doc_topic_prior_ : float
    Prior of document topic distribution `theta`. If the value is None,
    it is `1 / n_components`.

topic_word_prior_ : float
    Prior of topic word distribution `beta`. If the value is None, it is
    `1 / n_components`.

Examples
--------
>>> from sklearn.decomposition import LatentDirichletAllocation
>>> from sklearn.datasets import make_multilabel_classification
>>> # This produces a feature matrix of token counts, similar to what
>>> # CountVectorizer would produce on text.
>>> X, _ = make_multilabel_classification(random_state=0)
>>> lda = LatentDirichletAllocation(n_components=5,
...     random_state=0)
>>> lda.fit(X)
LatentDirichletAllocation(...)
>>> # get topics for some given samples:
>>> lda.transform(X[-2:])
array([[0.00360392, 0.25499205, 0.0036211 , 0.64236448, 0.09541846],
       [0.15297572, 0.00362644, 0.44412786, 0.39568399, 0.003586  ]])

References
----------
.. [1] "Online Learning for Latent Dirichlet Allocation", Matthew D.
    Hoffman, David M. Blei, Francis Bach, 2010

[2] "Stochastic Variational Inference", Matthew D. Hoffman, David M. Blei,
    Chong Wang, John Paisley, 2013

[3] Matthew D. Hoffman's onlineldavb code. Link:
    https://github.com/blei-lab/onlineldavb
*)

val fit : ?y:Py.Object.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> t
(**
Learn model for the data X with variational Bayes method.

When `learning_method` is 'online', use mini-batch update.
Otherwise, use batch update.

Parameters
----------
X : array-like or sparse matrix, shape=(n_samples, n_features)
    Document word matrix.

y : Ignored

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

val partial_fit : ?y:Py.Object.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Py.Object.t
(**
Online VB with Mini-Batch update.

Parameters
----------
X : array-like or sparse matrix, shape=(n_samples, n_features)
    Document word matrix.

y : Ignored

Returns
-------
self
*)

val perplexity : ?sub_sampling:bool -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> float
(**
Calculate approximate perplexity for data X.

Perplexity is defined as exp(-1. * log-likelihood per word)

.. versionchanged:: 0.19
   *doc_topic_distr* argument has been deprecated and is ignored
   because user no longer has access to unnormalized distribution

Parameters
----------
X : array-like or sparse matrix, [n_samples, n_features]
    Document word matrix.

sub_sampling : bool
    Do sub-sampling or not.

Returns
-------
score : float
    Perplexity score.
*)

val score : ?y:Py.Object.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> float
(**
Calculate approximate log-likelihood as score.

Parameters
----------
X : array-like or sparse matrix, shape=(n_samples, n_features)
    Document word matrix.

y : Ignored

Returns
-------
score : float
    Use approximate bound as score.
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
Transform data X according to the fitted model.

   .. versionchanged:: 0.18
      *doc_topic_distr* is now normalized

Parameters
----------
X : array-like or sparse matrix, shape=(n_samples, n_features)
    Document word matrix.

Returns
-------
doc_topic_distr : shape=(n_samples, n_components)
    Document topic distribution for X.
*)


(** Attribute components_: see constructor for documentation *)
val components_ : t -> Ndarray.t

(** Attribute n_batch_iter_: see constructor for documentation *)
val n_batch_iter_ : t -> int

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> int

(** Attribute bound_: see constructor for documentation *)
val bound_ : t -> float

(** Attribute doc_topic_prior_: see constructor for documentation *)
val doc_topic_prior_ : t -> float

(** Attribute topic_word_prior_: see constructor for documentation *)
val topic_word_prior_ : t -> float

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MiniBatchDictionaryLearning : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_components:int -> ?alpha:float -> ?n_iter:int -> ?fit_algorithm:[`Lars | `Cd] -> ?n_jobs:[`Int of int | `None] -> ?batch_size:int -> ?shuffle:bool -> ?dict_init:Ndarray.t -> ?transform_algorithm:[`Lasso_lars | `Lasso_cd | `Lars | `Omp | `Threshold] -> ?transform_n_nonzero_coefs:[`Int of int | `PyObject of Py.Object.t] -> ?transform_alpha:float -> ?verbose:bool -> ?split_sign:bool -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> ?positive_code:bool -> ?positive_dict:bool -> ?transform_max_iter:int -> unit -> t
(**
Mini-batch dictionary learning

Finds a dictionary (a set of atoms) that can best be used to represent data
using a sparse code.

Solves the optimization problem::

   (U^*,V^* ) = argmin 0.5 || Y - U V ||_2^2 + alpha * || U ||_1
                (U,V)
                with || V_k ||_2 = 1 for all  0 <= k < n_components

Read more in the :ref:`User Guide <DictionaryLearning>`.

Parameters
----------
n_components : int,
    number of dictionary elements to extract

alpha : float,
    sparsity controlling parameter

n_iter : int,
    total number of iterations to perform

fit_algorithm : {'lars', 'cd'}
    lars: uses the least angle regression method to solve the lasso problem
    (linear_model.lars_path)
    cd: uses the coordinate descent method to compute the
    Lasso solution (linear_model.Lasso). Lars will be faster if
    the estimated components are sparse.

n_jobs : int or None, optional (default=None)
    Number of parallel jobs to run.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

batch_size : int,
    number of samples in each mini-batch

shuffle : bool,
    whether to shuffle the samples before forming batches

dict_init : array of shape (n_components, n_features),
    initial value of the dictionary for warm restart scenarios

transform_algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp',     'threshold'}
    Algorithm used to transform the data.
    lars: uses the least angle regression method (linear_model.lars_path)
    lasso_lars: uses Lars to compute the Lasso solution
    lasso_cd: uses the coordinate descent method to compute the
    Lasso solution (linear_model.Lasso). lasso_lars will be faster if
    the estimated components are sparse.
    omp: uses orthogonal matching pursuit to estimate the sparse solution
    threshold: squashes to zero all coefficients less than alpha from
    the projection dictionary * X'

transform_n_nonzero_coefs : int, ``0.1 * n_features`` by default
    Number of nonzero coefficients to target in each column of the
    solution. This is only used by `algorithm='lars'` and `algorithm='omp'`
    and is overridden by `alpha` in the `omp` case.

transform_alpha : float, 1. by default
    If `algorithm='lasso_lars'` or `algorithm='lasso_cd'`, `alpha` is the
    penalty applied to the L1 norm.
    If `algorithm='threshold'`, `alpha` is the absolute value of the
    threshold below which coefficients will be squashed to zero.
    If `algorithm='omp'`, `alpha` is the tolerance parameter: the value of
    the reconstruction error targeted. In this case, it overrides
    `n_nonzero_coefs`.

verbose : bool, optional (default: False)
    To control the verbosity of the procedure.

split_sign : bool, False by default
    Whether to split the sparse feature vector into the concatenation of
    its negative part and its positive part. This can improve the
    performance of downstream classifiers.

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

positive_code : bool
    Whether to enforce positivity when finding the code.

    .. versionadded:: 0.20

positive_dict : bool
    Whether to enforce positivity when finding the dictionary.

    .. versionadded:: 0.20

transform_max_iter : int, optional (default=1000)
    Maximum number of iterations to perform if `algorithm='lasso_cd'` or
    `lasso_lars`.

    .. versionadded:: 0.22

Attributes
----------
components_ : array, [n_components, n_features]
    components extracted from the data

inner_stats_ : tuple of (A, B) ndarrays
    Internal sufficient statistics that are kept by the algorithm.
    Keeping them is useful in online settings, to avoid losing the
    history of the evolution, but they shouldn't have any use for the
    end user.
    A (n_components, n_components) is the dictionary covariance matrix.
    B (n_features, n_components) is the data approximation matrix

n_iter_ : int
    Number of iterations run.

iter_offset_ : int
    The number of iteration on data batches that has been
    performed before.

random_state_ : RandomState
    RandomState instance that is generated either from a seed, the random
    number generattor or by `np.random`.

Notes
-----
**References:**

J. Mairal, F. Bach, J. Ponce, G. Sapiro, 2009: Online dictionary learning
for sparse coding (https://www.di.ens.fr/sierra/pdfs/icml09.pdf)

See also
--------
SparseCoder
DictionaryLearning
SparsePCA
MiniBatchSparsePCA
*)

val fit : ?y:Py.Object.t -> x:Ndarray.t -> t -> t
(**
Fit the model from data in X.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training vector, where n_samples in the number of samples
    and n_features is the number of features.

y : Ignored

Returns
-------
self : object
    Returns the instance itself.
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

val partial_fit : ?y:Py.Object.t -> ?iter_offset:int -> x:Ndarray.t -> t -> t
(**
Updates the model using the data in X as a mini-batch.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training vector, where n_samples in the number of samples
    and n_features is the number of features.

y : Ignored

iter_offset : integer, optional
    The number of iteration on data batches that has been
    performed before this call to partial_fit. This is optional:
    if no number is passed, the memory of the object is
    used.

Returns
-------
self : object
    Returns the instance itself.
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
Encode the data as a sparse combination of the dictionary atoms.

Coding method is determined by the object parameter
`transform_algorithm`.

Parameters
----------
X : array of shape (n_samples, n_features)
    Test data to be transformed, must have the same number of
    features as the data used to train the model.

Returns
-------
X_new : array, shape (n_samples, n_components)
    Transformed data
*)


(** Attribute components_: see constructor for documentation *)
val components_ : t -> Ndarray.t

(** Attribute inner_stats_: see constructor for documentation *)
val inner_stats_ : t -> Py.Object.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> int

(** Attribute iter_offset_: see constructor for documentation *)
val iter_offset_ : t -> int

(** Attribute random_state_: see constructor for documentation *)
val random_state_ : t -> Py.Object.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MiniBatchSparsePCA : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_components:int -> ?alpha:int -> ?ridge_alpha:float -> ?n_iter:int -> ?callback:[`Callable of Py.Object.t | `None] -> ?batch_size:int -> ?verbose:int -> ?shuffle:bool -> ?n_jobs:[`Int of int | `None] -> ?method_:[`Lars | `Cd] -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> ?normalize_components:string -> unit -> t
(**
Mini-batch Sparse Principal Components Analysis

Finds the set of sparse components that can optimally reconstruct
the data.  The amount of sparseness is controllable by the coefficient
of the L1 penalty, given by the parameter alpha.

Read more in the :ref:`User Guide <SparsePCA>`.

Parameters
----------
n_components : int,
    number of sparse atoms to extract

alpha : int,
    Sparsity controlling parameter. Higher values lead to sparser
    components.

ridge_alpha : float,
    Amount of ridge shrinkage to apply in order to improve
    conditioning when calling the transform method.

n_iter : int,
    number of iterations to perform for each mini batch

callback : callable or None, optional (default: None)
    callable that gets invoked every five iterations

batch_size : int,
    the number of features to take in each mini batch

verbose : int
    Controls the verbosity; the higher, the more messages. Defaults to 0.

shuffle : boolean,
    whether to shuffle the data before splitting it in batches

n_jobs : int or None, optional (default=None)
    Number of parallel jobs to run.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

method : {'lars', 'cd'}
    lars: uses the least angle regression method to solve the lasso problem
    (linear_model.lars_path)
    cd: uses the coordinate descent method to compute the
    Lasso solution (linear_model.Lasso). Lars will be faster if
    the estimated components are sparse.

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

normalize_components : 'deprecated'
    This parameter does not have any effect. The components are always
    normalized.

    .. versionadded:: 0.20

    .. deprecated:: 0.22
       ``normalize_components`` is deprecated in 0.22 and will be removed
       in 0.24.

Attributes
----------
components_ : array, [n_components, n_features]
    Sparse components extracted from the data.

n_iter_ : int
    Number of iterations run.

mean_ : array, shape (n_features,)
    Per-feature empirical mean, estimated from the training set.
    Equal to ``X.mean(axis=0)``.

Examples
--------
>>> import numpy as np
>>> from sklearn.datasets import make_friedman1
>>> from sklearn.decomposition import MiniBatchSparsePCA
>>> X, _ = make_friedman1(n_samples=200, n_features=30, random_state=0)
>>> transformer = MiniBatchSparsePCA(n_components=5, batch_size=50,
...                                  random_state=0)
>>> transformer.fit(X)
MiniBatchSparsePCA(...)
>>> X_transformed = transformer.transform(X)
>>> X_transformed.shape
(200, 5)
>>> # most values in the components_ are zero (sparsity)
>>> np.mean(transformer.components_ == 0)
0.94

See also
--------
PCA
SparsePCA
DictionaryLearning
*)

val fit : ?y:Py.Object.t -> x:Ndarray.t -> t -> t
(**
Fit the model from data in X.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training vector, where n_samples in the number of samples
    and n_features is the number of features.

y : Ignored

Returns
-------
self : object
    Returns the instance itself.
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
Least Squares projection of the data onto the sparse components.

To avoid instability issues in case the system is under-determined,
regularization can be applied (Ridge regression) via the
`ridge_alpha` parameter.

Note that Sparse PCA components orthogonality is not enforced as in PCA
hence one cannot use a simple linear projection.

Parameters
----------
X : array of shape (n_samples, n_features)
    Test data to be transformed, must have the same number of
    features as the data used to train the model.

Returns
-------
X_new array, shape (n_samples, n_components)
    Transformed data.
*)


(** Attribute components_: see constructor for documentation *)
val components_ : t -> Ndarray.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> int

(** Attribute mean_: see constructor for documentation *)
val mean_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module NMF : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_components:[`Int of int | `None] -> ?init:[`Random | `Nndsvd | `Nndsvda | `Nndsvdar | `Custom | `None] -> ?solver:[`Cd | `Mu] -> ?beta_loss:[`Float of float | `String of string] -> ?tol:float -> ?max_iter:int -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> ?alpha:float -> ?l1_ratio:float -> ?verbose:bool -> ?shuffle:bool -> unit -> t
(**
Non-Negative Matrix Factorization (NMF)

Find two non-negative matrices (W, H) whose product approximates the non-
negative matrix X. This factorization can be used for example for
dimensionality reduction, source separation or topic extraction.

The objective function is::

    0.5 * ||X - WH||_Fro^2
    + alpha * l1_ratio * ||vec(W)||_1
    + alpha * l1_ratio * ||vec(H)||_1
    + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2
    + 0.5 * alpha * (1 - l1_ratio) * ||H||_Fro^2

Where::

    ||A||_Fro^2 = \sum_{i,j} A_{ij}^2 (Frobenius norm)
    ||vec(A)||_1 = \sum_{i,j} abs(A_{ij}) (Elementwise L1 norm)

For multiplicative-update ('mu') solver, the Frobenius norm
(0.5 * ||X - WH||_Fro^2) can be changed into another beta-divergence loss,
by changing the beta_loss parameter.

The objective function is minimized with an alternating minimization of W
and H.

Read more in the :ref:`User Guide <NMF>`.

Parameters
----------
n_components : int or None
    Number of components, if n_components is not set all features
    are kept.

init : None | 'random' | 'nndsvd' |  'nndsvda' | 'nndsvdar' | 'custom'
    Method used to initialize the procedure.
    Default: None.
    Valid options:

    - None: 'nndsvd' if n_components <= min(n_samples, n_features),
        otherwise random.

    - 'random': non-negative random matrices, scaled with:
        sqrt(X.mean() / n_components)

    - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
        initialization (better for sparseness)

    - 'nndsvda': NNDSVD with zeros filled with the average of X
        (better when sparsity is not desired)

    - 'nndsvdar': NNDSVD with zeros filled with small random values
        (generally faster, less accurate alternative to NNDSVDa
        for when sparsity is not desired)

    - 'custom': use custom matrices W and H

solver : 'cd' | 'mu'
    Numerical solver to use:
    'cd' is a Coordinate Descent solver.
    'mu' is a Multiplicative Update solver.

    .. versionadded:: 0.17
       Coordinate Descent solver.

    .. versionadded:: 0.19
       Multiplicative Update solver.

beta_loss : float or string, default 'frobenius'
    String must be in {'frobenius', 'kullback-leibler', 'itakura-saito'}.
    Beta divergence to be minimized, measuring the distance between X
    and the dot product WH. Note that values different from 'frobenius'
    (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
    fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
    matrix X cannot contain zeros. Used only in 'mu' solver.

    .. versionadded:: 0.19

tol : float, default: 1e-4
    Tolerance of the stopping condition.

max_iter : integer, default: 200
    Maximum number of iterations before timing out.

random_state : int, RandomState instance or None, optional, default: None
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

alpha : double, default: 0.
    Constant that multiplies the regularization terms. Set it to zero to
    have no regularization.

    .. versionadded:: 0.17
       *alpha* used in the Coordinate Descent solver.

l1_ratio : double, default: 0.
    The regularization mixing parameter, with 0 <= l1_ratio <= 1.
    For l1_ratio = 0 the penalty is an elementwise L2 penalty
    (aka Frobenius Norm).
    For l1_ratio = 1 it is an elementwise L1 penalty.
    For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

    .. versionadded:: 0.17
       Regularization parameter *l1_ratio* used in the Coordinate Descent
       solver.

verbose : bool, default=False
    Whether to be verbose.

shuffle : boolean, default: False
    If true, randomize the order of coordinates in the CD solver.

    .. versionadded:: 0.17
       *shuffle* parameter used in the Coordinate Descent solver.

Attributes
----------
components_ : array, [n_components, n_features]
    Factorization matrix, sometimes called 'dictionary'.

n_components_ : integer
    The number of components. It is same as the `n_components` parameter
    if it was given. Otherwise, it will be same as the number of
    features.

reconstruction_err_ : number
    Frobenius norm of the matrix difference, or beta-divergence, between
    the training data ``X`` and the reconstructed data ``WH`` from
    the fitted model.

n_iter_ : int
    Actual number of iterations.

Examples
--------
>>> import numpy as np
>>> X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
>>> from sklearn.decomposition import NMF
>>> model = NMF(n_components=2, init='random', random_state=0)
>>> W = model.fit_transform(X)
>>> H = model.components_

References
----------
Cichocki, Andrzej, and P. H. A. N. Anh-Huy. "Fast local algorithms for
large scale nonnegative matrix and tensor factorizations."
IEICE transactions on fundamentals of electronics, communications and
computer sciences 92.3: 708-721, 2009.

Fevotte, C., & Idier, J. (2011). Algorithms for nonnegative matrix
factorization with the beta-divergence. Neural Computation, 23(9).
*)

val fit : ?y:Py.Object.t -> ?params:(string * Py.Object.t) list -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> t
(**
Learn a NMF model for the data X.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Data matrix to be decomposed

y : Ignored

Returns
-------
self
*)

val fit_transform : ?y:Py.Object.t -> ?w:Ndarray.t -> ?h:Ndarray.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Learn a NMF model for the data X and returns the transformed data.

This is more efficient than calling fit followed by transform.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Data matrix to be decomposed

y : Ignored

W : array-like, shape (n_samples, n_components)
    If init='custom', it is used as initial guess for the solution.

H : array-like, shape (n_components, n_features)
    If init='custom', it is used as initial guess for the solution.

Returns
-------
W : array, shape (n_samples, n_components)
    Transformed data.
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

val inverse_transform : w:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Py.Object.t
(**
Transform data back to its original space.

Parameters
----------
W : {array-like, sparse matrix}, shape (n_samples, n_components)
    Transformed data matrix

Returns
-------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Data matrix of original shape

.. versionadded:: 0.18
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
Transform the data X according to the fitted NMF model

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Data matrix to be transformed by the model

Returns
-------
W : array, shape (n_samples, n_components)
    Transformed data
*)


(** Attribute components_: see constructor for documentation *)
val components_ : t -> Ndarray.t

(** Attribute n_components_: see constructor for documentation *)
val n_components_ : t -> int

(** Attribute reconstruction_err_: see constructor for documentation *)
val reconstruction_err_ : t -> Py.Object.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> int

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module PCA : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_components:[`Int of int | `Float of float | `String of string | `None] -> ?copy:bool -> ?whiten:bool -> ?svd_solver:[`Auto | `Full | `Arpack | `Randomized] -> ?tol:float -> ?iterated_power:[`Int of int | `Auto] -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> unit -> t
(**
Principal component analysis (PCA).

Linear dimensionality reduction using Singular Value Decomposition of the
data to project it to a lower dimensional space. The input data is centered
but not scaled for each feature before applying the SVD.

It uses the LAPACK implementation of the full SVD or a randomized truncated
SVD by the method of Halko et al. 2009, depending on the shape of the input
data and the number of components to extract.

It can also use the scipy.sparse.linalg ARPACK implementation of the
truncated SVD.

Notice that this class does not support sparse input. See
:class:`TruncatedSVD` for an alternative with sparse data.

Read more in the :ref:`User Guide <PCA>`.

Parameters
----------
n_components : int, float, None or str
    Number of components to keep.
    if n_components is not set all components are kept::

        n_components == min(n_samples, n_features)

    If ``n_components == 'mle'`` and ``svd_solver == 'full'``, Minka's
    MLE is used to guess the dimension. Use of ``n_components == 'mle'``
    will interpret ``svd_solver == 'auto'`` as ``svd_solver == 'full'``.

    If ``0 < n_components < 1`` and ``svd_solver == 'full'``, select the
    number of components such that the amount of variance that needs to be
    explained is greater than the percentage specified by n_components.

    If ``svd_solver == 'arpack'``, the number of components must be
    strictly less than the minimum of n_features and n_samples.

    Hence, the None case results in::

        n_components == min(n_samples, n_features) - 1

copy : bool, default=True
    If False, data passed to fit are overwritten and running
    fit(X).transform(X) will not yield the expected results,
    use fit_transform(X) instead.

whiten : bool, optional (default False)
    When True (False by default) the `components_` vectors are multiplied
    by the square root of n_samples and then divided by the singular values
    to ensure uncorrelated outputs with unit component-wise variances.

    Whitening will remove some information from the transformed signal
    (the relative variance scales of the components) but can sometime
    improve the predictive accuracy of the downstream estimators by
    making their data respect some hard-wired assumptions.

svd_solver : str {'auto', 'full', 'arpack', 'randomized'}
    If auto :
        The solver is selected by a default policy based on `X.shape` and
        `n_components`: if the input data is larger than 500x500 and the
        number of components to extract is lower than 80% of the smallest
        dimension of the data, then the more efficient 'randomized'
        method is enabled. Otherwise the exact full SVD is computed and
        optionally truncated afterwards.
    If full :
        run exact full SVD calling the standard LAPACK solver via
        `scipy.linalg.svd` and select the components by postprocessing
    If arpack :
        run SVD truncated to n_components calling ARPACK solver via
        `scipy.sparse.linalg.svds`. It requires strictly
        0 < n_components < min(X.shape)
    If randomized :
        run randomized SVD by the method of Halko et al.

    .. versionadded:: 0.18.0

tol : float >= 0, optional (default .0)
    Tolerance for singular values computed by svd_solver == 'arpack'.

    .. versionadded:: 0.18.0

iterated_power : int >= 0, or 'auto', (default 'auto')
    Number of iterations for the power method computed by
    svd_solver == 'randomized'.

    .. versionadded:: 0.18.0

random_state : int, RandomState instance or None, optional (default None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`. Used when ``svd_solver`` == 'arpack' or 'randomized'.

    .. versionadded:: 0.18.0

Attributes
----------
components_ : array, shape (n_components, n_features)
    Principal axes in feature space, representing the directions of
    maximum variance in the data. The components are sorted by
    ``explained_variance_``.

explained_variance_ : array, shape (n_components,)
    The amount of variance explained by each of the selected components.

    Equal to n_components largest eigenvalues
    of the covariance matrix of X.

    .. versionadded:: 0.18

explained_variance_ratio_ : array, shape (n_components,)
    Percentage of variance explained by each of the selected components.

    If ``n_components`` is not set then all components are stored and the
    sum of the ratios is equal to 1.0.

singular_values_ : array, shape (n_components,)
    The singular values corresponding to each of the selected components.
    The singular values are equal to the 2-norms of the ``n_components``
    variables in the lower-dimensional space.

    .. versionadded:: 0.19

mean_ : array, shape (n_features,)
    Per-feature empirical mean, estimated from the training set.

    Equal to `X.mean(axis=0)`.

n_components_ : int
    The estimated number of components. When n_components is set
    to 'mle' or a number between 0 and 1 (with svd_solver == 'full') this
    number is estimated from input data. Otherwise it equals the parameter
    n_components, or the lesser value of n_features and n_samples
    if n_components is None.

n_features_ : int
    Number of features in the training data.

n_samples_ : int
    Number of samples in the training data.

noise_variance_ : float
    The estimated noise covariance following the Probabilistic PCA model
    from Tipping and Bishop 1999. See "Pattern Recognition and
    Machine Learning" by C. Bishop, 12.2.1 p. 574 or
    http://www.miketipping.com/papers/met-mppca.pdf. It is required to
    compute the estimated data covariance and score samples.

    Equal to the average of (min(n_features, n_samples) - n_components)
    smallest eigenvalues of the covariance matrix of X.

See Also
--------
KernelPCA : Kernel Principal Component Analysis.
SparsePCA : Sparse Principal Component Analysis.
TruncatedSVD : Dimensionality reduction using truncated SVD.
IncrementalPCA : Incremental Principal Component Analysis.

References
----------
For n_components == 'mle', this class uses the method of *Minka, T. P.
"Automatic choice of dimensionality for PCA". In NIPS, pp. 598-604*

Implements the probabilistic PCA model from:
Tipping, M. E., and Bishop, C. M. (1999). "Probabilistic principal
component analysis". Journal of the Royal Statistical Society:
Series B (Statistical Methodology), 61(3), 611-622.
via the score and score_samples methods.
See http://www.miketipping.com/papers/met-mppca.pdf

For svd_solver == 'arpack', refer to `scipy.sparse.linalg.svds`.

For svd_solver == 'randomized', see:
*Halko, N., Martinsson, P. G., and Tropp, J. A. (2011).
"Finding structure with randomness: Probabilistic algorithms for
constructing approximate matrix decompositions".
SIAM review, 53(2), 217-288.* and also
*Martinsson, P. G., Rokhlin, V., and Tygert, M. (2011).
"A randomized algorithm for the decomposition of matrices".
Applied and Computational Harmonic Analysis, 30(1), 47-68.*

Examples
--------
>>> import numpy as np
>>> from sklearn.decomposition import PCA
>>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
>>> pca = PCA(n_components=2)
>>> pca.fit(X)
PCA(n_components=2)
>>> print(pca.explained_variance_ratio_)
[0.9924... 0.0075...]
>>> print(pca.singular_values_)
[6.30061... 0.54980...]

>>> pca = PCA(n_components=2, svd_solver='full')
>>> pca.fit(X)
PCA(n_components=2, svd_solver='full')
>>> print(pca.explained_variance_ratio_)
[0.9924... 0.00755...]
>>> print(pca.singular_values_)
[6.30061... 0.54980...]

>>> pca = PCA(n_components=1, svd_solver='arpack')
>>> pca.fit(X)
PCA(n_components=1, svd_solver='arpack')
>>> print(pca.explained_variance_ratio_)
[0.99244...]
>>> print(pca.singular_values_)
[6.30061...]
*)

val fit : ?y:Py.Object.t -> x:Ndarray.t -> t -> t
(**
Fit the model with X.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training data, where n_samples is the number of samples
    and n_features is the number of features.

y : None
    Ignored variable.

Returns
-------
self : object
    Returns the instance itself.
*)

val fit_transform : ?y:Py.Object.t -> x:Ndarray.t -> t -> Ndarray.t
(**
Fit the model with X and apply the dimensionality reduction on X.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training data, where n_samples is the number of samples
    and n_features is the number of features.

y : None
    Ignored variable.

Returns
-------
X_new : array-like, shape (n_samples, n_components)
    Transformed values.

Notes
-----
This method returns a Fortran-ordered array. To convert it to a
C-ordered array, use 'np.ascontiguousarray'.
*)

val get_covariance : t -> Ndarray.t
(**
Compute data covariance with the generative model.

``cov = components_.T * S**2 * components_ + sigma2 * eye(n_features)``
where S**2 contains the explained variances, and sigma2 contains the
noise variances.

Returns
-------
cov : array, shape=(n_features, n_features)
    Estimated covariance of data.
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
Compute data precision matrix with the generative model.

Equals the inverse of the covariance but computed with
the matrix inversion lemma for efficiency.

Returns
-------
precision : array, shape=(n_features, n_features)
    Estimated precision of data.
*)

val inverse_transform : x:Ndarray.t -> t -> Py.Object.t
(**
Transform data back to its original space.

In other words, return an input X_original whose transform would be X.

Parameters
----------
X : array-like, shape (n_samples, n_components)
    New data, where n_samples is the number of samples
    and n_components is the number of components.

Returns
-------
X_original array-like, shape (n_samples, n_features)

Notes
-----
If whitening is enabled, inverse_transform will compute the
exact inverse operation, which includes reversing whitening.
*)

val score : ?y:Py.Object.t -> x:Ndarray.t -> t -> float
(**
Return the average log-likelihood of all samples.

See. "Pattern Recognition and Machine Learning"
by C. Bishop, 12.2.1 p. 574
or http://www.miketipping.com/papers/met-mppca.pdf

Parameters
----------
X : array, shape(n_samples, n_features)
    The data.

y : None
    Ignored variable.

Returns
-------
ll : float
    Average log-likelihood of the samples under the current model.
*)

val score_samples : x:Ndarray.t -> t -> Ndarray.t
(**
Return the log-likelihood of each sample.

See. "Pattern Recognition and Machine Learning"
by C. Bishop, 12.2.1 p. 574
or http://www.miketipping.com/papers/met-mppca.pdf

Parameters
----------
X : array, shape(n_samples, n_features)
    The data.

Returns
-------
ll : array, shape (n_samples,)
    Log-likelihood of each sample under the current model.
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
Apply dimensionality reduction to X.

X is projected on the first principal components previously extracted
from a training set.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    New data, where n_samples is the number of samples
    and n_features is the number of features.

Returns
-------
X_new : array-like, shape (n_samples, n_components)

Examples
--------

>>> import numpy as np
>>> from sklearn.decomposition import IncrementalPCA
>>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
>>> ipca = IncrementalPCA(n_components=2, batch_size=3)
>>> ipca.fit(X)
IncrementalPCA(batch_size=3, n_components=2)
>>> ipca.transform(X) # doctest: +SKIP
*)


(** Attribute components_: see constructor for documentation *)
val components_ : t -> Ndarray.t

(** Attribute explained_variance_: see constructor for documentation *)
val explained_variance_ : t -> Ndarray.t

(** Attribute explained_variance_ratio_: see constructor for documentation *)
val explained_variance_ratio_ : t -> Ndarray.t

(** Attribute singular_values_: see constructor for documentation *)
val singular_values_ : t -> Ndarray.t

(** Attribute mean_: see constructor for documentation *)
val mean_ : t -> Ndarray.t

(** Attribute n_components_: see constructor for documentation *)
val n_components_ : t -> int

(** Attribute n_features_: see constructor for documentation *)
val n_features_ : t -> int

(** Attribute n_samples_: see constructor for documentation *)
val n_samples_ : t -> int

(** Attribute noise_variance_: see constructor for documentation *)
val noise_variance_ : t -> float

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module SparseCoder : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?transform_algorithm:[`Lasso_lars | `Lasso_cd | `Lars | `Omp | `Threshold] -> ?transform_n_nonzero_coefs:int -> ?transform_alpha:float -> ?split_sign:bool -> ?n_jobs:[`Int of int | `None] -> ?positive_code:bool -> ?transform_max_iter:int -> dictionary:Ndarray.t -> unit -> t
(**
Sparse coding

Finds a sparse representation of data against a fixed, precomputed
dictionary.

Each row of the result is the solution to a sparse coding problem.
The goal is to find a sparse array `code` such that::

    X ~= code * dictionary

Read more in the :ref:`User Guide <SparseCoder>`.

Parameters
----------
dictionary : array, [n_components, n_features]
    The dictionary atoms used for sparse coding. Lines are assumed to be
    normalized to unit norm.

transform_algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp',     'threshold'}, default='omp'
    Algorithm used to transform the data:
    lars: uses the least angle regression method (linear_model.lars_path)
    lasso_lars: uses Lars to compute the Lasso solution
    lasso_cd: uses the coordinate descent method to compute the
    Lasso solution (linear_model.Lasso). lasso_lars will be faster if
    the estimated components are sparse.
    omp: uses orthogonal matching pursuit to estimate the sparse solution
    threshold: squashes to zero all coefficients less than alpha from
    the projection ``dictionary * X'``

transform_n_nonzero_coefs : int, default=0.1*n_features
    Number of nonzero coefficients to target in each column of the
    solution. This is only used by `algorithm='lars'` and `algorithm='omp'`
    and is overridden by `alpha` in the `omp` case.

transform_alpha : float, default=1.
    If `algorithm='lasso_lars'` or `algorithm='lasso_cd'`, `alpha` is the
    penalty applied to the L1 norm.
    If `algorithm='threshold'`, `alpha` is the absolute value of the
    threshold below which coefficients will be squashed to zero.
    If `algorithm='omp'`, `alpha` is the tolerance parameter: the value of
    the reconstruction error targeted. In this case, it overrides
    `n_nonzero_coefs`.

split_sign : bool, default=False
    Whether to split the sparse feature vector into the concatenation of
    its negative part and its positive part. This can improve the
    performance of downstream classifiers.

n_jobs : int or None, default=None
    Number of parallel jobs to run.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

positive_code : bool, default=False
    Whether to enforce positivity when finding the code.

    .. versionadded:: 0.20

transform_max_iter : int, default=1000
    Maximum number of iterations to perform if `algorithm='lasso_cd'` or
    `lasso_lars`.

    .. versionadded:: 0.22

Attributes
----------
components_ : array, [n_components, n_features]
    The unchanged dictionary atoms

See also
--------
DictionaryLearning
MiniBatchDictionaryLearning
SparsePCA
MiniBatchSparsePCA
sparse_encode
*)

val fit : ?y:Py.Object.t -> x:Py.Object.t -> t -> t
(**
Do nothing and return the estimator unchanged

This method is just there to implement the usual API and hence
work in pipelines.

Parameters
----------
X : Ignored

y : Ignored

Returns
-------
self : object
    Returns the object itself
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
Encode the data as a sparse combination of the dictionary atoms.

Coding method is determined by the object parameter
`transform_algorithm`.

Parameters
----------
X : array of shape (n_samples, n_features)
    Test data to be transformed, must have the same number of
    features as the data used to train the model.

Returns
-------
X_new : array, shape (n_samples, n_components)
    Transformed data
*)


(** Attribute components_: see constructor for documentation *)
val components_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module SparsePCA : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_components:int -> ?alpha:float -> ?ridge_alpha:float -> ?max_iter:int -> ?tol:float -> ?method_:[`Lars | `Cd] -> ?n_jobs:[`Int of int | `None] -> ?u_init:Ndarray.t -> ?v_init:Ndarray.t -> ?verbose:int -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> ?normalize_components:string -> unit -> t
(**
Sparse Principal Components Analysis (SparsePCA)

Finds the set of sparse components that can optimally reconstruct
the data.  The amount of sparseness is controllable by the coefficient
of the L1 penalty, given by the parameter alpha.

Read more in the :ref:`User Guide <SparsePCA>`.

Parameters
----------
n_components : int,
    Number of sparse atoms to extract.

alpha : float,
    Sparsity controlling parameter. Higher values lead to sparser
    components.

ridge_alpha : float,
    Amount of ridge shrinkage to apply in order to improve
    conditioning when calling the transform method.

max_iter : int,
    Maximum number of iterations to perform.

tol : float,
    Tolerance for the stopping condition.

method : {'lars', 'cd'}
    lars: uses the least angle regression method to solve the lasso problem
    (linear_model.lars_path)
    cd: uses the coordinate descent method to compute the
    Lasso solution (linear_model.Lasso). Lars will be faster if
    the estimated components are sparse.

n_jobs : int or None, optional (default=None)
    Number of parallel jobs to run.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

U_init : array of shape (n_samples, n_components),
    Initial values for the loadings for warm restart scenarios.

V_init : array of shape (n_components, n_features),
    Initial values for the components for warm restart scenarios.

verbose : int
    Controls the verbosity; the higher, the more messages. Defaults to 0.

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

normalize_components : 'deprecated'
    This parameter does not have any effect. The components are always
    normalized.

    .. versionadded:: 0.20

    .. deprecated:: 0.22
       ``normalize_components`` is deprecated in 0.22 and will be removed
       in 0.24.

Attributes
----------
components_ : array, [n_components, n_features]
    Sparse components extracted from the data.

error_ : array
    Vector of errors at each iteration.

n_iter_ : int
    Number of iterations run.

mean_ : array, shape (n_features,)
    Per-feature empirical mean, estimated from the training set.
    Equal to ``X.mean(axis=0)``.

Examples
--------
>>> import numpy as np
>>> from sklearn.datasets import make_friedman1
>>> from sklearn.decomposition import SparsePCA
>>> X, _ = make_friedman1(n_samples=200, n_features=30, random_state=0)
>>> transformer = SparsePCA(n_components=5, random_state=0)
>>> transformer.fit(X)
SparsePCA(...)
>>> X_transformed = transformer.transform(X)
>>> X_transformed.shape
(200, 5)
>>> # most values in the components_ are zero (sparsity)
>>> np.mean(transformer.components_ == 0)
0.9666...

See also
--------
PCA
MiniBatchSparsePCA
DictionaryLearning
*)

val fit : ?y:Py.Object.t -> x:Ndarray.t -> t -> t
(**
Fit the model from data in X.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training vector, where n_samples in the number of samples
    and n_features is the number of features.

y : Ignored

Returns
-------
self : object
    Returns the instance itself.
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
Least Squares projection of the data onto the sparse components.

To avoid instability issues in case the system is under-determined,
regularization can be applied (Ridge regression) via the
`ridge_alpha` parameter.

Note that Sparse PCA components orthogonality is not enforced as in PCA
hence one cannot use a simple linear projection.

Parameters
----------
X : array of shape (n_samples, n_features)
    Test data to be transformed, must have the same number of
    features as the data used to train the model.

Returns
-------
X_new array, shape (n_samples, n_components)
    Transformed data.
*)


(** Attribute components_: see constructor for documentation *)
val components_ : t -> Ndarray.t

(** Attribute error_: see constructor for documentation *)
val error_ : t -> Ndarray.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> int

(** Attribute mean_: see constructor for documentation *)
val mean_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module TruncatedSVD : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_components:int -> ?algorithm:string -> ?n_iter:int -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> ?tol:float -> unit -> t
(**
Dimensionality reduction using truncated SVD (aka LSA).

This transformer performs linear dimensionality reduction by means of
truncated singular value decomposition (SVD). Contrary to PCA, this
estimator does not center the data before computing the singular value
decomposition. This means it can work with scipy.sparse matrices
efficiently.

In particular, truncated SVD works on term count/tf-idf matrices as
returned by the vectorizers in sklearn.feature_extraction.text. In that
context, it is known as latent semantic analysis (LSA).

This estimator supports two algorithms: a fast randomized SVD solver, and
a "naive" algorithm that uses ARPACK as an eigensolver on (X * X.T) or
(X.T * X), whichever is more efficient.

Read more in the :ref:`User Guide <LSA>`.

Parameters
----------
n_components : int, default = 2
    Desired dimensionality of output data.
    Must be strictly less than the number of features.
    The default value is useful for visualisation. For LSA, a value of
    100 is recommended.

algorithm : string, default = "randomized"
    SVD solver to use. Either "arpack" for the ARPACK wrapper in SciPy
    (scipy.sparse.linalg.svds), or "randomized" for the randomized
    algorithm due to Halko (2009).

n_iter : int, optional (default 5)
    Number of iterations for randomized SVD solver. Not used by ARPACK. The
    default is larger than the default in
    `~sklearn.utils.extmath.randomized_svd` to handle sparse matrices that
    may have large slowly decaying spectrum.

random_state : int, RandomState instance or None, optional, default = None
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

tol : float, optional
    Tolerance for ARPACK. 0 means machine precision. Ignored by randomized
    SVD solver.

Attributes
----------
components_ : array, shape (n_components, n_features)

explained_variance_ : array, shape (n_components,)
    The variance of the training samples transformed by a projection to
    each component.

explained_variance_ratio_ : array, shape (n_components,)
    Percentage of variance explained by each of the selected components.

singular_values_ : array, shape (n_components,)
    The singular values corresponding to each of the selected components.
    The singular values are equal to the 2-norms of the ``n_components``
    variables in the lower-dimensional space.

Examples
--------
>>> from sklearn.decomposition import TruncatedSVD
>>> from scipy.sparse import random as sparse_random
>>> from sklearn.random_projection import sparse_random_matrix
>>> X = sparse_random(100, 100, density=0.01, format='csr',
...                   random_state=42)
>>> svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
>>> svd.fit(X)
TruncatedSVD(n_components=5, n_iter=7, random_state=42)
>>> print(svd.explained_variance_ratio_)
[0.0646... 0.0633... 0.0639... 0.0535... 0.0406...]
>>> print(svd.explained_variance_ratio_.sum())
0.286...
>>> print(svd.singular_values_)
[1.553... 1.512...  1.510... 1.370... 1.199...]

See also
--------
PCA

References
----------
Finding structure with randomness: Stochastic algorithms for constructing
approximate matrix decompositions
Halko, et al., 2009 (arXiv:909) https://arxiv.org/pdf/0909.4061.pdf

Notes
-----
SVD suffers from a problem called "sign indeterminacy", which means the
sign of the ``components_`` and the output from transform depend on the
algorithm and random state. To work around this, fit instances of this
class to data once, then keep the instance around to do transformations.
*)

val fit : ?y:Py.Object.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> t
(**
Fit LSI model on training data X.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Training data.

y : Ignored

Returns
-------
self : object
    Returns the transformer object.
*)

val fit_transform : ?y:Py.Object.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Fit LSI model to X and perform dimensionality reduction on X.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Training data.

y : Ignored

Returns
-------
X_new : array, shape (n_samples, n_components)
    Reduced version of X. This will always be a dense array.
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

val inverse_transform : x:Ndarray.t -> t -> Ndarray.t
(**
Transform X back to its original space.

Returns an array X_original whose transform would be X.

Parameters
----------
X : array-like, shape (n_samples, n_components)
    New data.

Returns
-------
X_original : array, shape (n_samples, n_features)
    Note that this is always a dense array.
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
Perform dimensionality reduction on X.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    New data.

Returns
-------
X_new : array, shape (n_samples, n_components)
    Reduced version of X. This will always be a dense array.
*)


(** Attribute components_: see constructor for documentation *)
val components_ : t -> Ndarray.t

(** Attribute explained_variance_: see constructor for documentation *)
val explained_variance_ : t -> Ndarray.t

(** Attribute explained_variance_ratio_: see constructor for documentation *)
val explained_variance_ratio_ : t -> Ndarray.t

(** Attribute singular_values_: see constructor for documentation *)
val singular_values_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val dict_learning : ?max_iter:int -> ?tol:float -> ?method_:[`Lars | `Cd] -> ?n_jobs:[`Int of int | `None] -> ?dict_init:Ndarray.t -> ?code_init:Ndarray.t -> ?callback:[`Callable of Py.Object.t | `None] -> ?verbose:bool -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> ?return_n_iter:bool -> ?positive_dict:bool -> ?positive_code:bool -> ?method_max_iter:int -> x:Ndarray.t -> n_components:int -> alpha:int -> unit -> (Ndarray.t * Ndarray.t * Ndarray.t * int)
(**
Solves a dictionary learning matrix factorization problem.

Finds the best dictionary and the corresponding sparse code for
approximating the data matrix X by solving::

    (U^*, V^* ) = argmin 0.5 || X - U V ||_2^2 + alpha * || U ||_1
                 (U,V)
                with || V_k ||_2 = 1 for all  0 <= k < n_components

where V is the dictionary and U is the sparse code.

Read more in the :ref:`User Guide <DictionaryLearning>`.

Parameters
----------
X : array of shape (n_samples, n_features)
    Data matrix.

n_components : int,
    Number of dictionary atoms to extract.

alpha : int,
    Sparsity controlling parameter.

max_iter : int,
    Maximum number of iterations to perform.

tol : float,
    Tolerance for the stopping condition.

method : {'lars', 'cd'}
    lars: uses the least angle regression method to solve the lasso problem
    (linear_model.lars_path)
    cd: uses the coordinate descent method to compute the
    Lasso solution (linear_model.Lasso). Lars will be faster if
    the estimated components are sparse.

n_jobs : int or None, optional (default=None)
    Number of parallel jobs to run.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

dict_init : array of shape (n_components, n_features),
    Initial value for the dictionary for warm restart scenarios.

code_init : array of shape (n_samples, n_components),
    Initial value for the sparse code for warm restart scenarios.

callback : callable or None, optional (default: None)
    Callable that gets invoked every five iterations

verbose : bool, optional (default: False)
    To control the verbosity of the procedure.

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

return_n_iter : bool
    Whether or not to return the number of iterations.

positive_dict : bool
    Whether to enforce positivity when finding the dictionary.

    .. versionadded:: 0.20

positive_code : bool
    Whether to enforce positivity when finding the code.

    .. versionadded:: 0.20

method_max_iter : int, optional (default=1000)
    Maximum number of iterations to perform.

    .. versionadded:: 0.22

Returns
-------
code : array of shape (n_samples, n_components)
    The sparse code factor in the matrix factorization.

dictionary : array of shape (n_components, n_features),
    The dictionary factor in the matrix factorization.

errors : array
    Vector of errors at each iteration.

n_iter : int
    Number of iterations run. Returned only if `return_n_iter` is
    set to True.

See also
--------
dict_learning_online
DictionaryLearning
MiniBatchDictionaryLearning
SparsePCA
MiniBatchSparsePCA
*)

val dict_learning_online : ?n_components:int -> ?alpha:float -> ?n_iter:int -> ?return_code:bool -> ?dict_init:Ndarray.t -> ?callback:[`Callable of Py.Object.t | `None] -> ?batch_size:int -> ?verbose:bool -> ?shuffle:bool -> ?n_jobs:[`Int of int | `None] -> ?method_:[`Lars | `Cd] -> ?iter_offset:int -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> ?return_inner_stats:bool -> ?inner_stats:Py.Object.t -> ?return_n_iter:bool -> ?positive_dict:bool -> ?positive_code:bool -> ?method_max_iter:int -> x:Ndarray.t -> unit -> (Ndarray.t * Ndarray.t * int)
(**
Solves a dictionary learning matrix factorization problem online.

Finds the best dictionary and the corresponding sparse code for
approximating the data matrix X by solving::

    (U^*, V^* ) = argmin 0.5 || X - U V ||_2^2 + alpha * || U ||_1
                 (U,V)
                 with || V_k ||_2 = 1 for all  0 <= k < n_components

where V is the dictionary and U is the sparse code. This is
accomplished by repeatedly iterating over mini-batches by slicing
the input data.

Read more in the :ref:`User Guide <DictionaryLearning>`.

Parameters
----------
X : array of shape (n_samples, n_features)
    Data matrix.

n_components : int,
    Number of dictionary atoms to extract.

alpha : float,
    Sparsity controlling parameter.

n_iter : int,
    Number of mini-batch iterations to perform.

return_code : boolean,
    Whether to also return the code U or just the dictionary V.

dict_init : array of shape (n_components, n_features),
    Initial value for the dictionary for warm restart scenarios.

callback : callable or None, optional (default: None)
    callable that gets invoked every five iterations

batch_size : int,
    The number of samples to take in each batch.

verbose : bool, optional (default: False)
    To control the verbosity of the procedure.

shuffle : boolean,
    Whether to shuffle the data before splitting it in batches.

n_jobs : int or None, optional (default=None)
    Number of parallel jobs to run.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

method : {'lars', 'cd'}
    lars: uses the least angle regression method to solve the lasso problem
    (linear_model.lars_path)
    cd: uses the coordinate descent method to compute the
    Lasso solution (linear_model.Lasso). Lars will be faster if
    the estimated components are sparse.

iter_offset : int, default 0
    Number of previous iterations completed on the dictionary used for
    initialization.

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

return_inner_stats : boolean, optional
    Return the inner statistics A (dictionary covariance) and B
    (data approximation). Useful to restart the algorithm in an
    online setting. If return_inner_stats is True, return_code is
    ignored

inner_stats : tuple of (A, B) ndarrays
    Inner sufficient statistics that are kept by the algorithm.
    Passing them at initialization is useful in online settings, to
    avoid losing the history of the evolution.
    A (n_components, n_components) is the dictionary covariance matrix.
    B (n_features, n_components) is the data approximation matrix

return_n_iter : bool
    Whether or not to return the number of iterations.

positive_dict : bool
    Whether to enforce positivity when finding the dictionary.

    .. versionadded:: 0.20

positive_code : bool
    Whether to enforce positivity when finding the code.

    .. versionadded:: 0.20

method_max_iter : int, optional (default=1000)
    Maximum number of iterations to perform when solving the lasso problem.

    .. versionadded:: 0.22

Returns
-------
code : array of shape (n_samples, n_components),
    the sparse code (only returned if `return_code=True`)

dictionary : array of shape (n_components, n_features),
    the solutions to the dictionary learning problem

n_iter : int
    Number of iterations run. Returned only if `return_n_iter` is
    set to `True`.

See also
--------
dict_learning
DictionaryLearning
MiniBatchDictionaryLearning
SparsePCA
MiniBatchSparsePCA
*)

val fastica : ?n_components:int -> ?algorithm:[`Parallel | `Deflation] -> ?whiten:bool -> ?fun_:[`String of string | `Callable of Py.Object.t] -> ?fun_args:Py.Object.t -> ?max_iter:int -> ?tol:float -> ?w_init:Py.Object.t -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> ?return_X_mean:bool -> ?compute_sources:bool -> ?return_n_iter:bool -> x:Ndarray.t -> unit -> (Py.Object.t * Ndarray.t * Py.Object.t * Ndarray.t * int)
(**
Perform Fast Independent Component Analysis.

Read more in the :ref:`User Guide <ICA>`.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training vector, where n_samples is the number of samples and
    n_features is the number of features.

n_components : int, optional
    Number of components to extract. If None no dimension reduction
    is performed.

algorithm : {'parallel', 'deflation'}, optional
    Apply a parallel or deflational FASTICA algorithm.

whiten : boolean, optional
    If True perform an initial whitening of the data.
    If False, the data is assumed to have already been
    preprocessed: it should be centered, normed and white.
    Otherwise you will get incorrect results.
    In this case the parameter n_components will be ignored.

fun : string or function, optional. Default: 'logcosh'
    The functional form of the G function used in the
    approximation to neg-entropy. Could be either 'logcosh', 'exp',
    or 'cube'.
    You can also provide your own function. It should return a tuple
    containing the value of the function, and of its derivative, in the
    point. The derivative should be averaged along its last dimension.
    Example:

    def my_g(x):
        return x ** 3, np.mean(3 * x ** 2, axis=-1)

fun_args : dictionary, optional
    Arguments to send to the functional form.
    If empty or None and if fun='logcosh', fun_args will take value
    {'alpha' : 1.0}

max_iter : int, optional
    Maximum number of iterations to perform.

tol : float, optional
    A positive scalar giving the tolerance at which the
    un-mixing matrix is considered to have converged.

w_init : (n_components, n_components) array, optional
    Initial un-mixing array of dimension (n.comp,n.comp).
    If None (default) then an array of normal r.v.'s is used.

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

return_X_mean : bool, optional
    If True, X_mean is returned too.

compute_sources : bool, optional
    If False, sources are not computed, but only the rotation matrix.
    This can save memory when working with big data. Defaults to True.

return_n_iter : bool, optional
    Whether or not to return the number of iterations.

Returns
-------
K : array, shape (n_components, n_features) | None.
    If whiten is 'True', K is the pre-whitening matrix that projects data
    onto the first n_components principal components. If whiten is 'False',
    K is 'None'.

W : array, shape (n_components, n_components)
    The square matrix that unmixes the data after whitening.
    The mixing matrix is the pseudo-inverse of matrix ``W K``
    if K is not None, else it is the inverse of W.

S : array, shape (n_samples, n_components) | None
    Estimated source matrix

X_mean : array, shape (n_features, )
    The mean over features. Returned only if return_X_mean is True.

n_iter : int
    If the algorithm is "deflation", n_iter is the
    maximum number of iterations run across all components. Else
    they are just the number of iterations taken to converge. This is
    returned only when return_n_iter is set to `True`.

Notes
-----

The data matrix X is considered to be a linear combination of
non-Gaussian (independent) components i.e. X = AS where columns of S
contain the independent components and A is a linear mixing
matrix. In short ICA attempts to `un-mix' the data by estimating an
un-mixing matrix W where ``S = W K X.``
While FastICA was proposed to estimate as many sources
as features, it is possible to estimate less by setting
n_components < n_features. It this case K is not a square matrix
and the estimated A is the pseudo-inverse of ``W K``.

This implementation was originally made for data of shape
[n_features, n_samples]. Now the input is transposed
before the algorithm is applied. This makes it slightly
faster for Fortran-ordered input.

Implemented using FastICA:
*A. Hyvarinen and E. Oja, Independent Component Analysis:
Algorithms and Applications, Neural Networks, 13(4-5), 2000,
pp. 411-430*
*)

val non_negative_factorization : ?w:Ndarray.t -> ?h:Ndarray.t -> ?n_components:int -> ?init:[`Random | `Nndsvd | `Nndsvda | `Nndsvdar | `Custom | `None] -> ?update_H:bool -> ?solver:[`Cd | `Mu] -> ?beta_loss:[`Float of float | `String of string] -> ?tol:float -> ?max_iter:int -> ?alpha:float -> ?l1_ratio:float -> ?regularization:[`Both | `Components | `Transformation | `None] -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> ?verbose:int -> ?shuffle:bool -> x:Ndarray.t -> unit -> (Ndarray.t * Ndarray.t * int)
(**
Compute Non-negative Matrix Factorization (NMF)

Find two non-negative matrices (W, H) whose product approximates the non-
negative matrix X. This factorization can be used for example for
dimensionality reduction, source separation or topic extraction.

The objective function is::

    0.5 * ||X - WH||_Fro^2
    + alpha * l1_ratio * ||vec(W)||_1
    + alpha * l1_ratio * ||vec(H)||_1
    + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2
    + 0.5 * alpha * (1 - l1_ratio) * ||H||_Fro^2

Where::

    ||A||_Fro^2 = \sum_{i,j} A_{ij}^2 (Frobenius norm)
    ||vec(A)||_1 = \sum_{i,j} abs(A_{ij}) (Elementwise L1 norm)

For multiplicative-update ('mu') solver, the Frobenius norm
(0.5 * ||X - WH||_Fro^2) can be changed into another beta-divergence loss,
by changing the beta_loss parameter.

The objective function is minimized with an alternating minimization of W
and H. If H is given and update_H=False, it solves for W only.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Constant matrix.

W : array-like, shape (n_samples, n_components)
    If init='custom', it is used as initial guess for the solution.

H : array-like, shape (n_components, n_features)
    If init='custom', it is used as initial guess for the solution.
    If update_H=False, it is used as a constant, to solve for W only.

n_components : integer
    Number of components, if n_components is not set all features
    are kept.

init : None | 'random' | 'nndsvd' | 'nndsvda' | 'nndsvdar' | 'custom'
    Method used to initialize the procedure.
    Default: 'random'.

    The default value will change from 'random' to None in version 0.23
    to make it consistent with decomposition.NMF.

    Valid options:

    - None: 'nndsvd' if n_components < n_features, otherwise 'random'.

    - 'random': non-negative random matrices, scaled with:
        sqrt(X.mean() / n_components)

    - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
        initialization (better for sparseness)

    - 'nndsvda': NNDSVD with zeros filled with the average of X
        (better when sparsity is not desired)

    - 'nndsvdar': NNDSVD with zeros filled with small random values
        (generally faster, less accurate alternative to NNDSVDa
        for when sparsity is not desired)

    - 'custom': use custom matrices W and H

update_H : boolean, default: True
    Set to True, both W and H will be estimated from initial guesses.
    Set to False, only W will be estimated.

solver : 'cd' | 'mu'
    Numerical solver to use:

    - 'cd' is a Coordinate Descent solver that uses Fast Hierarchical
        Alternating Least Squares (Fast HALS).

    - 'mu' is a Multiplicative Update solver.

    .. versionadded:: 0.17
       Coordinate Descent solver.

    .. versionadded:: 0.19
       Multiplicative Update solver.

beta_loss : float or string, default 'frobenius'
    String must be in {'frobenius', 'kullback-leibler', 'itakura-saito'}.
    Beta divergence to be minimized, measuring the distance between X
    and the dot product WH. Note that values different from 'frobenius'
    (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
    fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
    matrix X cannot contain zeros. Used only in 'mu' solver.

    .. versionadded:: 0.19

tol : float, default: 1e-4
    Tolerance of the stopping condition.

max_iter : integer, default: 200
    Maximum number of iterations before timing out.

alpha : double, default: 0.
    Constant that multiplies the regularization terms.

l1_ratio : double, default: 0.
    The regularization mixing parameter, with 0 <= l1_ratio <= 1.
    For l1_ratio = 0 the penalty is an elementwise L2 penalty
    (aka Frobenius Norm).
    For l1_ratio = 1 it is an elementwise L1 penalty.
    For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

regularization : 'both' | 'components' | 'transformation' | None
    Select whether the regularization affects the components (H), the
    transformation (W), both or none of them.

random_state : int, RandomState instance or None, optional, default: None
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

verbose : integer, default: 0
    The verbosity level.

shuffle : boolean, default: False
    If true, randomize the order of coordinates in the CD solver.

Returns
-------
W : array-like, shape (n_samples, n_components)
    Solution to the non-negative least squares problem.

H : array-like, shape (n_components, n_features)
    Solution to the non-negative least squares problem.

n_iter : int
    Actual number of iterations.

Examples
--------
>>> import numpy as np
>>> X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
>>> from sklearn.decomposition import non_negative_factorization
>>> W, H, n_iter = non_negative_factorization(X, n_components=2,
... init='random', random_state=0)

References
----------
Cichocki, Andrzej, and P. H. A. N. Anh-Huy. "Fast local algorithms for
large scale nonnegative matrix and tensor factorizations."
IEICE transactions on fundamentals of electronics, communications and
computer sciences 92.3: 708-721, 2009.

Fevotte, C., & Idier, J. (2011). Algorithms for nonnegative matrix
factorization with the beta-divergence. Neural Computation, 23(9).
*)

val randomized_svd : ?n_oversamples:Py.Object.t -> ?n_iter:[`Int of int | `PyObject of Py.Object.t] -> ?power_iteration_normalizer:string -> ?transpose:[`Bool of bool | `Auto] -> ?flip_sign:[`Bool of bool | `PyObject of Py.Object.t] -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> m:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> n_components:int -> unit -> Py.Object.t
(**
Computes a truncated randomized SVD

Parameters
----------
M : ndarray or sparse matrix
    Matrix to decompose

n_components : int
    Number of singular values and vectors to extract.

n_oversamples : int (default is 10)
    Additional number of random vectors to sample the range of M so as
    to ensure proper conditioning. The total number of random vectors
    used to find the range of M is n_components + n_oversamples. Smaller
    number can improve speed but can negatively impact the quality of
    approximation of singular vectors and singular values.

n_iter : int or 'auto' (default is 'auto')
    Number of power iterations. It can be used to deal with very noisy
    problems. When 'auto', it is set to 4, unless `n_components` is small
    (< .1 * min(X.shape)) `n_iter` in which case is set to 7.
    This improves precision with few components.

    .. versionchanged:: 0.18

power_iteration_normalizer : 'auto' (default), 'QR', 'LU', 'none'
    Whether the power iterations are normalized with step-by-step
    QR factorization (the slowest but most accurate), 'none'
    (the fastest but numerically unstable when `n_iter` is large, e.g.
    typically 5 or larger), or 'LU' factorization (numerically stable
    but can lose slightly in accuracy). The 'auto' mode applies no
    normalization if `n_iter` <= 2 and switches to LU otherwise.

    .. versionadded:: 0.18

transpose : True, False or 'auto' (default)
    Whether the algorithm should be applied to M.T instead of M. The
    result should approximately be the same. The 'auto' mode will
    trigger the transposition if M.shape[1] > M.shape[0] since this
    implementation of randomized SVD tend to be a little faster in that
    case.

    .. versionchanged:: 0.18

flip_sign : boolean, (True by default)
    The output of a singular value decomposition is only unique up to a
    permutation of the signs of the singular vectors. If `flip_sign` is
    set to `True`, the sign ambiguity is resolved by making the largest
    loadings for each component in the left singular vectors positive.

random_state : int, RandomState instance or None, optional (default=None)
    The seed of the pseudo random number generator to use when shuffling
    the data.  If int, random_state is the seed used by the random number
    generator; If RandomState instance, random_state is the random number
    generator; If None, the random number generator is the RandomState
    instance used by `np.random`.

Notes
-----
This algorithm finds a (usually very good) approximate truncated
singular value decomposition using randomization to speed up the
computations. It is particularly fast on large matrices on which
you wish to extract only a small number of components. In order to
obtain further speed up, `n_iter` can be set <=2 (at the cost of
loss of precision).

References
----------
* Finding structure with randomness: Stochastic algorithms for constructing
  approximate matrix decompositions
  Halko, et al., 2009 https://arxiv.org/abs/0909.4061

* A randomized algorithm for the decomposition of matrices
  Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert

* An implementation of a randomized algorithm for principal component
  analysis
  A. Szlam et al. 2014
*)

val sparse_encode : ?gram:Ndarray.t -> ?cov:Ndarray.t -> ?algorithm:[`Lasso_lars | `Lasso_cd | `Lars | `Omp | `Threshold] -> ?n_nonzero_coefs:[`Int of int | `PyObject of Py.Object.t] -> ?alpha:float -> ?copy_cov:bool -> ?init:Ndarray.t -> ?max_iter:int -> ?n_jobs:[`Int of int | `None] -> ?check_input:bool -> ?verbose:int -> ?positive:bool -> x:Ndarray.t -> dictionary:Ndarray.t -> unit -> Ndarray.t
(**
Sparse coding

Each row of the result is the solution to a sparse coding problem.
The goal is to find a sparse array `code` such that::

    X ~= code * dictionary

Read more in the :ref:`User Guide <SparseCoder>`.

Parameters
----------
X : array of shape (n_samples, n_features)
    Data matrix

dictionary : array of shape (n_components, n_features)
    The dictionary matrix against which to solve the sparse coding of
    the data. Some of the algorithms assume normalized rows for meaningful
    output.

gram : array, shape=(n_components, n_components)
    Precomputed Gram matrix, dictionary * dictionary'

cov : array, shape=(n_components, n_samples)
    Precomputed covariance, dictionary' * X

algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'}
    lars: uses the least angle regression method (linear_model.lars_path)
    lasso_lars: uses Lars to compute the Lasso solution
    lasso_cd: uses the coordinate descent method to compute the
    Lasso solution (linear_model.Lasso). lasso_lars will be faster if
    the estimated components are sparse.
    omp: uses orthogonal matching pursuit to estimate the sparse solution
    threshold: squashes to zero all coefficients less than alpha from
    the projection dictionary * X'

n_nonzero_coefs : int, 0.1 * n_features by default
    Number of nonzero coefficients to target in each column of the
    solution. This is only used by `algorithm='lars'` and `algorithm='omp'`
    and is overridden by `alpha` in the `omp` case.

alpha : float, 1. by default
    If `algorithm='lasso_lars'` or `algorithm='lasso_cd'`, `alpha` is the
    penalty applied to the L1 norm.
    If `algorithm='threshold'`, `alpha` is the absolute value of the
    threshold below which coefficients will be squashed to zero.
    If `algorithm='omp'`, `alpha` is the tolerance parameter: the value of
    the reconstruction error targeted. In this case, it overrides
    `n_nonzero_coefs`.

copy_cov : boolean, optional
    Whether to copy the precomputed covariance matrix; if False, it may be
    overwritten.

init : array of shape (n_samples, n_components)
    Initialization value of the sparse codes. Only used if
    `algorithm='lasso_cd'`.

max_iter : int, 1000 by default
    Maximum number of iterations to perform if `algorithm='lasso_cd'` or
    `lasso_lars`.

n_jobs : int or None, optional (default=None)
    Number of parallel jobs to run.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

check_input : boolean, optional
    If False, the input arrays X and dictionary will not be checked.

verbose : int, optional
    Controls the verbosity; the higher, the more messages. Defaults to 0.

positive : boolean, optional
    Whether to enforce positivity when finding the encoding.

    .. versionadded:: 0.20

Returns
-------
code : array of shape (n_samples, n_components)
    The sparse codes

See also
--------
sklearn.linear_model.lars_path
sklearn.linear_model.orthogonal_mp
sklearn.linear_model.Lasso
SparseCoder
*)

