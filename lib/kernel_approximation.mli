(** Get an attribute of this module as a Py.Object.t. This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module AdditiveChi2Sampler : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?sample_steps:int -> ?sample_interval:Py.Object.t -> unit -> t
(**
Approximate feature map for additive chi2 kernel.

Uses sampling the fourier transform of the kernel characteristic
at regular intervals.

Since the kernel that is to be approximated is additive, the components of
the input vectors can be treated separately.  Each entry in the original
space is transformed into 2*sample_steps+1 features, where sample_steps is
a parameter of the method. Typical values of sample_steps include 1, 2 and
3.

Optimal choices for the sampling interval for certain data ranges can be
computed (see the reference). The default values should be reasonable.

Read more in the :ref:`User Guide <additive_chi_kernel_approx>`.

Parameters
----------
sample_steps : int, optional
    Gives the number of (complex) sampling points.
sample_interval : float, optional
    Sampling interval. Must be specified when sample_steps not in {1,2,3}.

Attributes
----------
sample_interval_ : float
    Stored sampling interval. Specified as a parameter if sample_steps not
    in {1,2,3}.

Examples
--------
>>> from sklearn.datasets import load_digits
>>> from sklearn.linear_model import SGDClassifier
>>> from sklearn.kernel_approximation import AdditiveChi2Sampler
>>> X, y = load_digits(return_X_y=True)
>>> chi2sampler = AdditiveChi2Sampler(sample_steps=2)
>>> X_transformed = chi2sampler.fit_transform(X, y)
>>> clf = SGDClassifier(max_iter=5, random_state=0, tol=1e-3)
>>> clf.fit(X_transformed, y)
SGDClassifier(max_iter=5, random_state=0)
>>> clf.score(X_transformed, y)
0.9499...

Notes
-----
This estimator approximates a slightly different version of the additive
chi squared kernel then ``metric.additive_chi2`` computes.

See also
--------
SkewedChi2Sampler : A Fourier-approximation to a non-additive variant of
    the chi squared kernel.

sklearn.metrics.pairwise.chi2_kernel : The exact chi squared kernel.

sklearn.metrics.pairwise.additive_chi2_kernel : The exact additive chi
    squared kernel.

References
----------
See `"Efficient additive kernels via explicit feature maps"
<http://www.robots.ox.ac.uk/~vedaldi/assets/pubs/vedaldi11efficient.pdf>`_
A. Vedaldi and A. Zisserman, Pattern Analysis and Machine Intelligence,
2011
*)

val fit : ?y:Py.Object.t -> x:Arr.t -> t -> t
(**
Set the parameters

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training data, where n_samples in the number of samples
    and n_features is the number of features.

Returns
-------
self : object
    Returns the transformer.
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
Apply approximate feature map to X.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)

Returns
-------
X_new : {array, sparse matrix},                shape = (n_samples, n_features * (2*sample_steps + 1))
    Whether the return value is an array of sparse matrix depends on
    the type of the input X.
*)


(** Attribute sample_interval_: get value or raise Not_found if None.*)
val sample_interval_ : t -> float

(** Attribute sample_interval_: get value as an option. *)
val sample_interval_opt : t -> (float) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

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

module Nystroem : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?kernel:[`S of string | `Callable of Py.Object.t] -> ?gamma:float -> ?coef0:float -> ?degree:float -> ?kernel_params:Dict.t -> ?n_components:int -> ?random_state:int -> unit -> t
(**
Approximate a kernel map using a subset of the training data.

Constructs an approximate feature map for an arbitrary kernel
using a subset of the data as basis.

Read more in the :ref:`User Guide <nystroem_kernel_approx>`.

.. versionadded:: 0.13

Parameters
----------
kernel : string or callable, default="rbf"
    Kernel map to be approximated. A callable should accept two arguments
    and the keyword arguments passed to this object as kernel_params, and
    should return a floating point number.

gamma : float, default=None
    Gamma parameter for the RBF, laplacian, polynomial, exponential chi2
    and sigmoid kernels. Interpretation of the default value is left to
    the kernel; see the documentation for sklearn.metrics.pairwise.
    Ignored by other kernels.

coef0 : float, default=None
    Zero coefficient for polynomial and sigmoid kernels.
    Ignored by other kernels.

degree : float, default=None
    Degree of the polynomial kernel. Ignored by other kernels.

kernel_params : mapping of string to any, optional
    Additional parameters (keyword arguments) for kernel function passed
    as callable object.

n_components : int
    Number of features to construct.
    How many data points will be used to construct the mapping.

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

Attributes
----------
components_ : array, shape (n_components, n_features)
    Subset of training points used to construct the feature map.

component_indices_ : array, shape (n_components)
    Indices of ``components_`` in the training set.

normalization_ : array, shape (n_components, n_components)
    Normalization matrix needed for embedding.
    Square root of the kernel matrix on ``components_``.

Examples
--------
>>> from sklearn import datasets, svm
>>> from sklearn.kernel_approximation import Nystroem
>>> X, y = datasets.load_digits(n_class=9, return_X_y=True)
>>> data = X / 16.
>>> clf = svm.LinearSVC()
>>> feature_map_nystroem = Nystroem(gamma=.2,
...                                 random_state=1,
...                                 n_components=300)
>>> data_transformed = feature_map_nystroem.fit_transform(data)
>>> clf.fit(data_transformed, y)
LinearSVC()
>>> clf.score(data_transformed, y)
0.9987...

References
----------
* Williams, C.K.I. and Seeger, M.
  "Using the Nystroem method to speed up kernel machines",
  Advances in neural information processing systems 2001

* T. Yang, Y. Li, M. Mahdavi, R. Jin and Z. Zhou
  "Nystroem Method vs Random Fourier Features: A Theoretical and Empirical
  Comparison",
  Advances in Neural Information Processing Systems 2012


See also
--------
RBFSampler : An approximation to the RBF kernel using random Fourier
             features.

sklearn.metrics.pairwise.kernel_metrics : List of built-in kernels.
*)

val fit : ?y:Py.Object.t -> x:Arr.t -> t -> t
(**
Fit estimator to data.

Samples a subset of training points, computes kernel
on these and computes normalization matrix.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training data.
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
Apply feature map to X.

Computes an approximate feature map using the kernel
between some training points and X.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Data to transform.

Returns
-------
X_transformed : array, shape=(n_samples, n_components)
    Transformed data.
*)


(** Attribute components_: get value or raise Not_found if None.*)
val components_ : t -> Arr.t

(** Attribute components_: get value as an option. *)
val components_opt : t -> (Arr.t) option


(** Attribute component_indices_: get value or raise Not_found if None.*)
val component_indices_ : t -> Arr.t

(** Attribute component_indices_: get value as an option. *)
val component_indices_opt : t -> (Arr.t) option


(** Attribute normalization_: get value or raise Not_found if None.*)
val normalization_ : t -> Arr.t

(** Attribute normalization_: get value as an option. *)
val normalization_opt : t -> (Arr.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module RBFSampler : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?gamma:float -> ?n_components:int -> ?random_state:int -> unit -> t
(**
Approximates feature map of an RBF kernel by Monte Carlo approximation
of its Fourier transform.

It implements a variant of Random Kitchen Sinks.[1]

Read more in the :ref:`User Guide <rbf_kernel_approx>`.

Parameters
----------
gamma : float
    Parameter of RBF kernel: exp(-gamma * x^2)

n_components : int
    Number of Monte Carlo samples per original feature.
    Equals the dimensionality of the computed feature space.

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

Examples
--------
>>> from sklearn.kernel_approximation import RBFSampler
>>> from sklearn.linear_model import SGDClassifier
>>> X = [[0, 0], [1, 1], [1, 0], [0, 1]]
>>> y = [0, 0, 1, 1]
>>> rbf_feature = RBFSampler(gamma=1, random_state=1)
>>> X_features = rbf_feature.fit_transform(X)
>>> clf = SGDClassifier(max_iter=5, tol=1e-3)
>>> clf.fit(X_features, y)
SGDClassifier(max_iter=5)
>>> clf.score(X_features, y)
1.0

Notes
-----
See "Random Features for Large-Scale Kernel Machines" by A. Rahimi and
Benjamin Recht.

[1] "Weighted Sums of Random Kitchen Sinks: Replacing
minimization with randomization in learning" by A. Rahimi and
Benjamin Recht.
(https://people.eecs.berkeley.edu/~brecht/papers/08.rah.rec.nips.pdf)
*)

val fit : ?y:Py.Object.t -> x:Arr.t -> t -> t
(**
Fit the model with X.

Samples random projection according to n_features.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Training data, where n_samples in the number of samples
    and n_features is the number of features.

Returns
-------
self : object
    Returns the transformer.
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
Apply the approximate feature map to X.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    New data, where n_samples in the number of samples
    and n_features is the number of features.

Returns
-------
X_new : array-like, shape (n_samples, n_components)
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module SkewedChi2Sampler : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?skewedness:float -> ?n_components:int -> ?random_state:int -> unit -> t
(**
Approximates feature map of the "skewed chi-squared" kernel by Monte
Carlo approximation of its Fourier transform.

Read more in the :ref:`User Guide <skewed_chi_kernel_approx>`.

Parameters
----------
skewedness : float
    "skewedness" parameter of the kernel. Needs to be cross-validated.

n_components : int
    number of Monte Carlo samples per original feature.
    Equals the dimensionality of the computed feature space.

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

Examples
--------
>>> from sklearn.kernel_approximation import SkewedChi2Sampler
>>> from sklearn.linear_model import SGDClassifier
>>> X = [[0, 0], [1, 1], [1, 0], [0, 1]]
>>> y = [0, 0, 1, 1]
>>> chi2_feature = SkewedChi2Sampler(skewedness=.01,
...                                  n_components=10,
...                                  random_state=0)
>>> X_features = chi2_feature.fit_transform(X, y)
>>> clf = SGDClassifier(max_iter=10, tol=1e-3)
>>> clf.fit(X_features, y)
SGDClassifier(max_iter=10)
>>> clf.score(X_features, y)
1.0

References
----------
See "Random Fourier Approximations for Skewed Multiplicative Histogram
Kernels" by Fuxin Li, Catalin Ionescu and Cristian Sminchisescu.

See also
--------
AdditiveChi2Sampler : A different approach for approximating an additive
    variant of the chi squared kernel.

sklearn.metrics.pairwise.chi2_kernel : The exact chi squared kernel.
*)

val fit : ?y:Py.Object.t -> x:Arr.t -> t -> t
(**
Fit the model with X.

Samples random projection according to n_features.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training data, where n_samples in the number of samples
    and n_features is the number of features.

Returns
-------
self : object
    Returns the transformer.
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
Apply the approximate feature map to X.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    New data, where n_samples in the number of samples
    and n_features is the number of features. All values of X must be
    strictly greater than "-skewedness".

Returns
-------
X_new : array-like, shape (n_samples, n_components)
*)


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

val as_float_array : ?copy:bool -> ?force_all_finite:[`Bool of bool | `Allow_nan] -> x:Arr.t -> unit -> Arr.t
(**
Converts an array-like to an array of floats.

The new dtype will be np.float32 or np.float64, depending on the original
type. The function can create a copy or modify the argument depending
on the argument copy.

Parameters
----------
X : {array-like, sparse matrix}

copy : bool, optional
    If True, a copy of X will be created. If False, a copy may still be
    returned if X's dtype is not a floating point type.

force_all_finite : boolean or 'allow-nan', (default=True)
    Whether to raise an error on np.inf and np.nan in X. The possibilities
    are:

    - True: Force all values of X to be finite.
    - False: accept both np.inf and np.nan in X.
    - 'allow-nan': accept only np.nan values in X. Values cannot be
      infinite.

    .. versionadded:: 0.20
       ``force_all_finite`` accepts the string ``'allow-nan'``.

Returns
-------
XT : {array, sparse matrix}
    An array of type np.float
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

val check_random_state : seed:[`I of int | `RandomState of Py.Object.t | `None] -> unit -> Py.Object.t
(**
Turn seed into a np.random.RandomState instance

Parameters
----------
seed : None | int | instance of RandomState
    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
*)

val pairwise_kernels : ?y:Arr.t -> ?metric:[`S of string | `Callable of Py.Object.t] -> ?filter_params:bool -> ?n_jobs:int -> ?kwds:(string * Py.Object.t) list -> x:[`Arr of Arr.t | `Otherwise of Py.Object.t] -> unit -> Py.Object.t
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
X : array [n_samples_a, n_samples_a] if metric == "precomputed", or,              [n_samples_a, n_features] otherwise
    Array of pairwise kernels between samples, or a feature array.

Y : array [n_samples_b, n_features]
    A second feature array only if X has shape [n_samples_a, n_features].

metric : string, or callable
    The metric to use when calculating kernel between instances in a
    feature array. If metric is a string, it must be one of the metrics
    in pairwise.PAIRWISE_KERNEL_FUNCTIONS.
    If metric is "precomputed", X is assumed to be a kernel matrix.
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

val safe_sparse_dot : ?dense_output:Py.Object.t -> a:Arr.t -> b:Py.Object.t -> unit -> Arr.t
(**
Dot product that handle the sparse matrix case correctly

Parameters
----------
a : array or sparse matrix
b : array or sparse matrix
dense_output : boolean, (default=False)
    When False, ``a`` and ``b`` both being sparse will yield sparse output.
    When True, output will always be a dense array.

Returns
-------
dot_product : array or sparse matrix
    sparse if ``a`` and ``b`` are sparse and ``dense_output=False``.
*)

val svd : ?full_matrices:Py.Object.t -> ?compute_uv:Py.Object.t -> ?overwrite_a:Py.Object.t -> ?check_finite:Py.Object.t -> ?lapack_driver:Py.Object.t -> a:Py.Object.t -> unit -> Arr.t
(**
Singular Value Decomposition.

Factorizes the matrix `a` into two unitary matrices ``U`` and ``Vh``, and
a 1-D array ``s`` of singular values (real, non-negative) such that
``a == U @ S @ Vh``, where ``S`` is a suitably shaped matrix of zeros with
main diagonal ``s``.

Parameters
----------
a : (M, N) array_like
    Matrix to decompose.
full_matrices : bool, optional
    If True (default), `U` and `Vh` are of shape ``(M, M)``, ``(N, N)``.
    If False, the shapes are ``(M, K)`` and ``(K, N)``, where
    ``K = min(M, N)``.
compute_uv : bool, optional
    Whether to compute also ``U`` and ``Vh`` in addition to ``s``.
    Default is True.
overwrite_a : bool, optional
    Whether to overwrite `a`; may improve performance.
    Default is False.
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.
lapack_driver : {'gesdd', 'gesvd'}, optional
    Whether to use the more efficient divide-and-conquer approach
    (``'gesdd'``) or general rectangular approach (``'gesvd'``)
    to compute the SVD. MATLAB and Octave use the ``'gesvd'`` approach.
    Default is ``'gesdd'``.

    .. versionadded:: 0.18

Returns
-------
U : ndarray
    Unitary matrix having left singular vectors as columns.
    Of shape ``(M, M)`` or ``(M, K)``, depending on `full_matrices`.
s : ndarray
    The singular values, sorted in non-increasing order.
    Of shape (K,), with ``K = min(M, N)``.
Vh : ndarray
    Unitary matrix having right singular vectors as rows.
    Of shape ``(N, N)`` or ``(K, N)`` depending on `full_matrices`.

For ``compute_uv=False``, only ``s`` is returned.

Raises
------
LinAlgError
    If SVD computation does not converge.

See also
--------
svdvals : Compute singular values of a matrix.
diagsvd : Construct the Sigma matrix, given the vector s.

Examples
--------
>>> from scipy import linalg
>>> m, n = 9, 6
>>> a = np.random.randn(m, n) + 1.j*np.random.randn(m, n)
>>> U, s, Vh = linalg.svd(a)
>>> U.shape,  s.shape, Vh.shape
((9, 9), (6,), (6, 6))

Reconstruct the original matrix from the decomposition:

>>> sigma = np.zeros((m, n))
>>> for i in range(min(m, n)):
...     sigma[i, i] = s[i]
>>> a1 = np.dot(U, np.dot(sigma, Vh))
>>> np.allclose(a, a1)
True

Alternatively, use ``full_matrices=False`` (notice that the shape of
``U`` is then ``(m, n)`` instead of ``(m, m)``):

>>> U, s, Vh = linalg.svd(a, full_matrices=False)
>>> U.shape, s.shape, Vh.shape
((9, 6), (6,), (6, 6))
>>> S = np.diag(s)
>>> np.allclose(a, np.dot(U, np.dot(S, Vh)))
True

>>> s2 = linalg.svd(a, compute_uv=False)
>>> np.allclose(s, s2)
True
*)

