module ABCMeta : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?kwargs:(string * Py.Object.t) list -> name:Py.Object.t -> bases:Py.Object.t -> namespace:Py.Object.t -> unit -> t
(**
Metaclass for defining Abstract Base Classes (ABCs).

Use this metaclass to create an ABC.  An ABC can be subclassed
directly, and then acts as a mix-in class.  You can also register
unrelated concrete classes (even built-in classes) and unrelated
ABCs as 'virtual subclasses' -- these and their descendants will
be considered subclasses of the registering ABC by the built-in
issubclass() function, but the registering ABC won't show up in
their MRO (Method Resolution Order) nor will method
implementations defined by the registering ABC be callable (not
even via super()).
*)

val mro : t -> Py.Object.t
(**
Return a type's method resolution order.
*)


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


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module GaussianRandomProjection : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_components:[`Int of int | `Auto] -> ?eps:float -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> unit -> t
(**
Reduce dimensionality through Gaussian random projection

The components of the random matrix are drawn from N(0, 1 / n_components).

Read more in the :ref:`User Guide <gaussian_random_matrix>`.

.. versionadded:: 0.13

Parameters
----------
n_components : int or 'auto', optional (default = 'auto')
    Dimensionality of the target projection space.

    n_components can be automatically adjusted according to the
    number of samples in the dataset and the bound given by the
    Johnson-Lindenstrauss lemma. In that case the quality of the
    embedding is controlled by the ``eps`` parameter.

    It should be noted that Johnson-Lindenstrauss lemma can yield
    very conservative estimated of the required number of components
    as it makes no assumption on the structure of the dataset.

eps : strictly positive float, optional (default=0.1)
    Parameter to control the quality of the embedding according to
    the Johnson-Lindenstrauss lemma when n_components is set to
    'auto'.

    Smaller values lead to better embedding and higher number of
    dimensions (n_components) in the target projection space.

random_state : int, RandomState instance or None, optional (default=None)
    Control the pseudo random number generator used to generate the matrix
    at fit time.  If int, random_state is the seed used by the random
    number generator; If RandomState instance, random_state is the random
    number generator; If None, the random number generator is the
    RandomState instance used by `np.random`.

Attributes
----------
n_components_ : int
    Concrete number of components computed when n_components="auto".

components_ : numpy array of shape [n_components, n_features]
    Random matrix used for the projection.

Examples
--------
>>> import numpy as np
>>> from sklearn.random_projection import GaussianRandomProjection
>>> rng = np.random.RandomState(42)
>>> X = rng.rand(100, 10000)
>>> transformer = GaussianRandomProjection(random_state=rng)
>>> X_new = transformer.fit_transform(X)
>>> X_new.shape
(100, 3947)

See Also
--------
SparseRandomProjection
*)

val fit : ?y:Py.Object.t -> x:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> t -> t
(**
Generate a sparse random projection matrix

Parameters
----------
X : numpy array or scipy.sparse of shape [n_samples, n_features]
    Training set: only the shape is used to find optimal random
    matrix dimensions based on the theory referenced in the
    afore mentioned papers.

y
    Ignored

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

val transform : x:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> t -> Ndarray.t
(**
Project the data by using matrix product with the random matrix

Parameters
----------
X : numpy array or scipy.sparse of shape [n_samples, n_features]
    The input data to project into a smaller dimensional space.

Returns
-------
X_new : numpy array or scipy sparse of shape [n_samples, n_components]
    Projected array.
*)


(** Attribute n_components_: see constructor for documentation *)
val n_components_ : t -> int

(** Attribute components_: see constructor for documentation *)
val components_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module SparseRandomProjection : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_components:[`Int of int | `Auto] -> ?density:float -> ?eps:float -> ?dense_output:bool -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> unit -> t
(**
Reduce dimensionality through sparse random projection

Sparse random matrix is an alternative to dense random
projection matrix that guarantees similar embedding quality while being
much more memory efficient and allowing faster computation of the
projected data.

If we note `s = 1 / density` the components of the random matrix are
drawn from:

  - -sqrt(s) / sqrt(n_components)   with probability 1 / 2s
  -  0                              with probability 1 - 1 / s
  - +sqrt(s) / sqrt(n_components)   with probability 1 / 2s

Read more in the :ref:`User Guide <sparse_random_matrix>`.

.. versionadded:: 0.13

Parameters
----------
n_components : int or 'auto', optional (default = 'auto')
    Dimensionality of the target projection space.

    n_components can be automatically adjusted according to the
    number of samples in the dataset and the bound given by the
    Johnson-Lindenstrauss lemma. In that case the quality of the
    embedding is controlled by the ``eps`` parameter.

    It should be noted that Johnson-Lindenstrauss lemma can yield
    very conservative estimated of the required number of components
    as it makes no assumption on the structure of the dataset.

density : float in range ]0, 1], optional (default='auto')
    Ratio of non-zero component in the random projection matrix.

    If density = 'auto', the value is set to the minimum density
    as recommended by Ping Li et al.: 1 / sqrt(n_features).

    Use density = 1 / 3.0 if you want to reproduce the results from
    Achlioptas, 2001.

eps : strictly positive float, optional, (default=0.1)
    Parameter to control the quality of the embedding according to
    the Johnson-Lindenstrauss lemma when n_components is set to
    'auto'.

    Smaller values lead to better embedding and higher number of
    dimensions (n_components) in the target projection space.

dense_output : boolean, optional (default=False)
    If True, ensure that the output of the random projection is a
    dense numpy array even if the input and random projection matrix
    are both sparse. In practice, if the number of components is
    small the number of zero components in the projected data will
    be very small and it will be more CPU and memory efficient to
    use a dense representation.

    If False, the projected data uses a sparse representation if
    the input is sparse.

random_state : int, RandomState instance or None, optional (default=None)
    Control the pseudo random number generator used to generate the matrix
    at fit time.  If int, random_state is the seed used by the random
    number generator; If RandomState instance, random_state is the random
    number generator; If None, the random number generator is the
    RandomState instance used by `np.random`.

Attributes
----------
n_components_ : int
    Concrete number of components computed when n_components="auto".

components_ : CSR matrix with shape [n_components, n_features]
    Random matrix used for the projection.

density_ : float in range 0.0 - 1.0
    Concrete density computed from when density = "auto".

Examples
--------
>>> import numpy as np
>>> from sklearn.random_projection import SparseRandomProjection
>>> rng = np.random.RandomState(42)
>>> X = rng.rand(100, 10000)
>>> transformer = SparseRandomProjection(random_state=rng)
>>> X_new = transformer.fit_transform(X)
>>> X_new.shape
(100, 3947)
>>> # very few components are non-zero
>>> np.mean(transformer.components_ != 0)
0.0100...

See Also
--------
GaussianRandomProjection

References
----------

.. [1] Ping Li, T. Hastie and K. W. Church, 2006,
       "Very Sparse Random Projections".
       https://web.stanford.edu/~hastie/Papers/Ping/KDD06_rp.pdf

.. [2] D. Achlioptas, 2001, "Database-friendly random projections",
       https://users.soe.ucsc.edu/~optas/papers/jl.pdf
*)

val fit : ?y:Py.Object.t -> x:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> t -> t
(**
Generate a sparse random projection matrix

Parameters
----------
X : numpy array or scipy.sparse of shape [n_samples, n_features]
    Training set: only the shape is used to find optimal random
    matrix dimensions based on the theory referenced in the
    afore mentioned papers.

y
    Ignored

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

val transform : x:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> t -> Ndarray.t
(**
Project the data by using matrix product with the random matrix

Parameters
----------
X : numpy array or scipy.sparse of shape [n_samples, n_features]
    The input data to project into a smaller dimensional space.

Returns
-------
X_new : numpy array or scipy sparse of shape [n_samples, n_components]
    Projected array.
*)


(** Attribute n_components_: see constructor for documentation *)
val n_components_ : t -> int

(** Attribute components_: see constructor for documentation *)
val components_ : t -> Py.Object.t

(** Attribute density_: see constructor for documentation *)
val density_ : t -> Py.Object.t

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


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val abstractmethod : funcobj:Py.Object.t -> unit -> Py.Object.t
(**
A decorator indicating abstract methods.

Requires that the metaclass is ABCMeta or derived from it.  A
class that has a metaclass derived from ABCMeta cannot be
instantiated unless all of its abstract methods are overridden.
The abstract methods can be called using any of the normal
'super' call mechanisms.

Usage:

    class C(metaclass=ABCMeta):
        @abstractmethod
        def my_abstract_method(self, ...):
            ...
*)

val check_array : ?accept_sparse:[`String of string | `Bool of bool | `StringList of string list] -> ?accept_large_sparse:bool -> ?dtype:[`String of string | `Dtype of Py.Object.t | `TypeList of Py.Object.t | `None] -> ?order:[`F | `C | `None] -> ?copy:bool -> ?force_all_finite:[`Bool of bool | `Allow_nan] -> ?ensure_2d:bool -> ?allow_nd:bool -> ?ensure_min_samples:int -> ?ensure_min_features:int -> ?warn_on_dtype:[`Bool of bool | `None] -> ?estimator:[`String of string | `Estimator of Py.Object.t] -> array:Py.Object.t -> unit -> Py.Object.t
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

val check_is_fitted : ?attributes:[`String of string | `ArrayLike of Py.Object.t | `StringList of string list] -> ?msg:string -> ?all_or_any:[`Callable of Py.Object.t | `PyObject of Py.Object.t] -> estimator:Py.Object.t -> unit -> Py.Object.t
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

val check_random_state : seed:[`Int of int | `RandomState of Py.Object.t | `None] -> unit -> Py.Object.t
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

module Deprecated : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?extra:string -> unit -> t
(**
Decorator to mark a function or class as deprecated.

Issue a warning when the function is called/the class is instantiated and
adds a warning to the docstring.

The optional extra argument will be appended to the deprecation message
and the docstring. Note: to use this with the default value for extra, put
in an empty of parentheses:

>>> from sklearn.utils import deprecated
>>> deprecated()
<sklearn.utils.deprecation.deprecated object at ...>

>>> @deprecated()
... def some_function(): pass

Parameters
----------
extra : string
      to be added to the deprecation messages
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val gaussian_random_matrix : ?random_state:Py.Object.t -> n_components:Py.Object.t -> n_features:Py.Object.t -> unit -> Py.Object.t
(**
DEPRECATED: gaussian_random_matrix is deprecated in 0.22 and will be removed in version 0.24.
*)

val johnson_lindenstrauss_min_dim : ?eps:[`Float of float | `PyObject of Py.Object.t] -> n_samples:[`Int of int | `PyObject of Py.Object.t] -> unit -> Py.Object.t
(**
Find a 'safe' number of components to randomly project to

The distortion introduced by a random projection `p` only changes the
distance between two points by a factor (1 +- eps) in an euclidean space
with good probability. The projection `p` is an eps-embedding as defined
by:

  (1 - eps) ||u - v||^2 < ||p(u) - p(v)||^2 < (1 + eps) ||u - v||^2

Where u and v are any rows taken from a dataset of shape [n_samples,
n_features], eps is in ]0, 1[ and p is a projection by a random Gaussian
N(0, 1) matrix with shape [n_components, n_features] (or a sparse
Achlioptas matrix).

The minimum number of components to guarantee the eps-embedding is
given by:

  n_components >= 4 log(n_samples) / (eps^2 / 2 - eps^3 / 3)

Note that the number of dimensions is independent of the original
number of features but instead depends on the size of the dataset:
the larger the dataset, the higher is the minimal dimensionality of
an eps-embedding.

Read more in the :ref:`User Guide <johnson_lindenstrauss>`.

Parameters
----------
n_samples : int or numpy array of int greater than 0,
    Number of samples. If an array is given, it will compute
    a safe number of components array-wise.

eps : float or numpy array of float in ]0,1[, optional (default=0.1)
    Maximum distortion rate as defined by the Johnson-Lindenstrauss lemma.
    If an array is given, it will compute a safe number of components
    array-wise.

Returns
-------
n_components : int or numpy array of int,
    The minimal number of components to guarantee with good probability
    an eps-embedding with n_samples.

Examples
--------

>>> johnson_lindenstrauss_min_dim(1e6, eps=0.5)
663

>>> johnson_lindenstrauss_min_dim(1e6, eps=[0.5, 0.1, 0.01])
array([    663,   11841, 1112658])

>>> johnson_lindenstrauss_min_dim([1e4, 1e5, 1e6], eps=0.1)
array([ 7894,  9868, 11841])

References
----------

.. [1] https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma

.. [2] Sanjoy Dasgupta and Anupam Gupta, 1999,
       "An elementary proof of the Johnson-Lindenstrauss Lemma."
       http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.45.3654
*)

val safe_sparse_dot : ?dense_output:Py.Object.t -> a:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> b:Py.Object.t -> unit -> Py.Object.t
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

val sparse_random_matrix : ?density:Py.Object.t -> ?random_state:Py.Object.t -> n_components:Py.Object.t -> n_features:Py.Object.t -> unit -> Py.Object.t
(**
DEPRECATED: gaussian_random_matrix is deprecated in 0.22 and will be removed in version 0.24.
*)

