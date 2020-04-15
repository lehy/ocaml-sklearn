module Binarizer : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?threshold:[`Float of float | `PyObject of Py.Object.t] -> ?copy:bool -> unit -> t
(**
Binarize data (set feature values to 0 or 1) according to a threshold

Values greater than the threshold map to 1, while values less than
or equal to the threshold map to 0. With the default threshold of 0,
only positive values map to 1.

Binarization is a common operation on text count data where the
analyst can decide to only consider the presence or absence of a
feature rather than a quantified number of occurrences for instance.

It can also be used as a pre-processing step for estimators that
consider boolean random variables (e.g. modelled using the Bernoulli
distribution in a Bayesian setting).

Read more in the :ref:`User Guide <preprocessing_binarization>`.

Parameters
----------
threshold : float, optional (0.0 by default)
    Feature values below or equal to this are replaced by 0, above it by 1.
    Threshold may not be less than 0 for operations on sparse matrices.

copy : boolean, optional, default True
    set to False to perform inplace binarization and avoid a copy (if
    the input is already a numpy array or a scipy.sparse CSR matrix).

Examples
--------
>>> from sklearn.preprocessing import Binarizer
>>> X = [[ 1., -1.,  2.],
...      [ 2.,  0.,  0.],
...      [ 0.,  1., -1.]]
>>> transformer = Binarizer().fit(X)  # fit does nothing.
>>> transformer
Binarizer()
>>> transformer.transform(X)
array([[1., 0., 1.],
       [1., 0., 0.],
       [0., 1., 0.]])

Notes
-----
If the input is a sparse matrix, only the non-zero values are subject
to update by the Binarizer class.

This estimator is stateless (besides constructor parameters), the
fit method does nothing but is useful when used in a pipeline.

See also
--------
binarize: Equivalent function without the estimator API.
*)

val fit : ?y:Py.Object.t -> x:Ndarray.t -> t -> t
(**
Do nothing and return the estimator unchanged

This method is just there to implement the usual API and hence
work in pipelines.

Parameters
----------
X : array-like
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

val transform : ?copy:bool -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Binarize each element of X

Parameters
----------
X : {array-like, sparse matrix}, shape [n_samples, n_features]
    The data to binarize, element by element.
    scipy.sparse matrices should be in CSR format to avoid an
    un-necessary copy.

copy : bool
    Copy the input X or not.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

module FunctionTransformer : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?func:Py.Object.t -> ?inverse_func:Py.Object.t -> ?validate:bool -> ?accept_sparse:bool -> ?check_inverse:bool -> ?kw_args:Py.Object.t -> ?inv_kw_args:Py.Object.t -> unit -> t
(**
Constructs a transformer from an arbitrary callable.

A FunctionTransformer forwards its X (and optionally y) arguments to a
user-defined function or function object and returns the result of this
function. This is useful for stateless transformations such as taking the
log of frequencies, doing custom scaling, etc.

Note: If a lambda is used as the function, then the resulting
transformer will not be pickleable.

.. versionadded:: 0.17

Read more in the :ref:`User Guide <function_transformer>`.

Parameters
----------
func : callable, optional default=None
    The callable to use for the transformation. This will be passed
    the same arguments as transform, with args and kwargs forwarded.
    If func is None, then func will be the identity function.

inverse_func : callable, optional default=None
    The callable to use for the inverse transformation. This will be
    passed the same arguments as inverse transform, with args and
    kwargs forwarded. If inverse_func is None, then inverse_func
    will be the identity function.

validate : bool, optional default=False
    Indicate that the input X array should be checked before calling
    ``func``. The possibilities are:

    - If False, there is no input validation.
    - If True, then X will be converted to a 2-dimensional NumPy array or
      sparse matrix. If the conversion is not possible an exception is
      raised.

    .. versionchanged:: 0.22
       The default of ``validate`` changed from True to False.

accept_sparse : boolean, optional
    Indicate that func accepts a sparse matrix as input. If validate is
    False, this has no effect. Otherwise, if accept_sparse is false,
    sparse matrix inputs will cause an exception to be raised.

check_inverse : bool, default=True
   Whether to check that or ``func`` followed by ``inverse_func`` leads to
   the original inputs. It can be used for a sanity check, raising a
   warning when the condition is not fulfilled.

   .. versionadded:: 0.20

kw_args : dict, optional
    Dictionary of additional keyword arguments to pass to func.

inv_kw_args : dict, optional
    Dictionary of additional keyword arguments to pass to inverse_func.

Examples
--------
>>> import numpy as np
>>> from sklearn.preprocessing import FunctionTransformer
>>> transformer = FunctionTransformer(np.log1p)
>>> X = np.array([[0, 1], [2, 3]])
>>> transformer.transform(X)
array([[0.       , 0.6931...],
       [1.0986..., 1.3862...]])
*)

val fit : ?y:Py.Object.t -> x:Ndarray.t -> t -> t
(**
Fit transformer by checking X.

If ``validate`` is ``True``, ``X`` will be checked.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Input array.

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

val inverse_transform : x:Ndarray.t -> t -> Ndarray.t
(**
Transform X using the inverse function.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Input array.

Returns
-------
X_out : array-like, shape (n_samples, n_features)
    Transformed input.
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
Transform X using the forward function.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Input array.

Returns
-------
X_out : array-like, shape (n_samples, n_features)
    Transformed input.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

module KBinsDiscretizer : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_bins:[`Int of int | `Ndarray of Ndarray.t] -> ?encode:[`Onehot | `Onehot_dense | `Ordinal] -> ?strategy:[`Uniform | `Quantile | `Kmeans] -> unit -> t
(**
Bin continuous data into intervals.

Read more in the :ref:`User Guide <preprocessing_discretization>`.

Parameters
----------
n_bins : int or array-like, shape (n_features,) (default=5)
    The number of bins to produce. Raises ValueError if ``n_bins < 2``.

encode : {'onehot', 'onehot-dense', 'ordinal'}, (default='onehot')
    Method used to encode the transformed result.

    onehot
        Encode the transformed result with one-hot encoding
        and return a sparse matrix. Ignored features are always
        stacked to the right.
    onehot-dense
        Encode the transformed result with one-hot encoding
        and return a dense array. Ignored features are always
        stacked to the right.
    ordinal
        Return the bin identifier encoded as an integer value.

strategy : {'uniform', 'quantile', 'kmeans'}, (default='quantile')
    Strategy used to define the widths of the bins.

    uniform
        All bins in each feature have identical widths.
    quantile
        All bins in each feature have the same number of points.
    kmeans
        Values in each bin have the same nearest center of a 1D k-means
        cluster.

Attributes
----------
n_bins_ : int array, shape (n_features,)
    Number of bins per feature. Bins whose width are too small
    (i.e., <= 1e-8) are removed with a warning.

bin_edges_ : array of arrays, shape (n_features, )
    The edges of each bin. Contain arrays of varying shapes ``(n_bins_, )``
    Ignored features will have empty arrays.

See Also
--------
 sklearn.preprocessing.Binarizer : Class used to bin values as ``0`` or
    ``1`` based on a parameter ``threshold``.

Notes
-----
In bin edges for feature ``i``, the first and last values are used only for
``inverse_transform``. During transform, bin edges are extended to::

  np.concatenate([-np.inf, bin_edges_[i][1:-1], np.inf])

You can combine ``KBinsDiscretizer`` with
:class:`sklearn.compose.ColumnTransformer` if you only want to preprocess
part of the features.

``KBinsDiscretizer`` might produce constant features (e.g., when
``encode = 'onehot'`` and certain bins do not contain any data).
These features can be removed with feature selection algorithms
(e.g., :class:`sklearn.feature_selection.VarianceThreshold`).

Examples
--------
>>> X = [[-2, 1, -4,   -1],
...      [-1, 2, -3, -0.5],
...      [ 0, 3, -2,  0.5],
...      [ 1, 4, -1,    2]]
>>> est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
>>> est.fit(X)
KBinsDiscretizer(...)
>>> Xt = est.transform(X)
>>> Xt  # doctest: +SKIP
array([[ 0., 0., 0., 0.],
       [ 1., 1., 1., 0.],
       [ 2., 2., 2., 1.],
       [ 2., 2., 2., 2.]])

Sometimes it may be useful to convert the data back into the original
feature space. The ``inverse_transform`` function converts the binned
data into the original feature space. Each value will be equal to the mean
of the two bin edges.

>>> est.bin_edges_[0]
array([-2., -1.,  0.,  1.])
>>> est.inverse_transform(Xt)
array([[-1.5,  1.5, -3.5, -0.5],
       [-0.5,  2.5, -2.5, -0.5],
       [ 0.5,  3.5, -1.5,  0.5],
       [ 0.5,  3.5, -1.5,  1.5]])
*)

val fit : ?y:Py.Object.t -> x:Ndarray.t -> t -> t
(**
Fit the estimator.

Parameters
----------
X : numeric array-like, shape (n_samples, n_features)
    Data to be discretized.

y : None
    Ignored. This parameter exists only for compatibility with
    :class:`sklearn.pipeline.Pipeline`.

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

val inverse_transform : xt:Ndarray.t -> t -> Ndarray.t
(**
Transform discretized data back to original feature space.

Note that this function does not regenerate the original data
due to discretization rounding.

Parameters
----------
Xt : numeric array-like, shape (n_sample, n_features)
    Transformed data in the binned space.

Returns
-------
Xinv : numeric array-like
    Data in the original feature space.
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
Discretize the data.

Parameters
----------
X : numeric array-like, shape (n_samples, n_features)
    Data to be discretized.

Returns
-------
Xt : numeric array-like or sparse matrix
    Data in the binned space.
*)


(** Attribute n_bins_: see constructor for documentation *)
val n_bins_ : t -> Py.Object.t

(** Attribute bin_edges_: see constructor for documentation *)
val bin_edges_ : t -> Py.Object.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

module KernelCenterer : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : unit -> t
(**
Center a kernel matrix

Let K(x, z) be a kernel defined by phi(x)^T phi(z), where phi is a
function mapping x to a Hilbert space. KernelCenterer centers (i.e.,
normalize to have zero mean) the data without explicitly computing phi(x).
It is equivalent to centering phi(x) with
sklearn.preprocessing.StandardScaler(with_std=False).

Read more in the :ref:`User Guide <kernel_centering>`.

Attributes
----------
K_fit_rows_ : array, shape (n_samples,)
    Average of each column of kernel matrix

K_fit_all_ : float
    Average of kernel matrix

Examples
--------
>>> from sklearn.preprocessing import KernelCenterer
>>> from sklearn.metrics.pairwise import pairwise_kernels
>>> X = [[ 1., -2.,  2.],
...      [ -2.,  1.,  3.],
...      [ 4.,  1., -2.]]
>>> K = pairwise_kernels(X, metric='linear')
>>> K
array([[  9.,   2.,  -2.],
       [  2.,  14., -13.],
       [ -2., -13.,  21.]])
>>> transformer = KernelCenterer().fit(K)
>>> transformer
KernelCenterer()
>>> transformer.transform(K)
array([[  5.,   0.,  -5.],
       [  0.,  14., -14.],
       [ -5., -14.,  19.]])
*)

val fit : ?y:Py.Object.t -> k:Ndarray.t -> t -> t
(**
Fit KernelCenterer

Parameters
----------
K : numpy array of shape [n_samples, n_samples]
    Kernel matrix.

Returns
-------
self : returns an instance of self.
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

val transform : ?copy:bool -> k:Ndarray.t -> t -> Ndarray.t
(**
Center kernel matrix.

Parameters
----------
K : numpy array of shape [n_samples1, n_samples2]
    Kernel matrix.

copy : boolean, optional, default True
    Set to False to perform inplace computation.

Returns
-------
K_new : numpy array of shape [n_samples1, n_samples2]
*)


(** Attribute K_fit_rows_: see constructor for documentation *)
val k_fit_rows_ : t -> Ndarray.t

(** Attribute K_fit_all_: see constructor for documentation *)
val k_fit_all_ : t -> float

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

module LabelBinarizer : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?neg_label:int -> ?pos_label:int -> ?sparse_output:bool -> unit -> t
(**
Binarize labels in a one-vs-all fashion

Several regression and binary classification algorithms are
available in scikit-learn. A simple way to extend these algorithms
to the multi-class classification case is to use the so-called
one-vs-all scheme.

At learning time, this simply consists in learning one regressor
or binary classifier per class. In doing so, one needs to convert
multi-class labels to binary labels (belong or does not belong
to the class). LabelBinarizer makes this process easy with the
transform method.

At prediction time, one assigns the class for which the corresponding
model gave the greatest confidence. LabelBinarizer makes this easy
with the inverse_transform method.

Read more in the :ref:`User Guide <preprocessing_targets>`.

Parameters
----------

neg_label : int (default: 0)
    Value with which negative labels must be encoded.

pos_label : int (default: 1)
    Value with which positive labels must be encoded.

sparse_output : boolean (default: False)
    True if the returned array from transform is desired to be in sparse
    CSR format.

Attributes
----------

classes_ : array of shape [n_class]
    Holds the label for each class.

y_type_ : str,
    Represents the type of the target data as evaluated by
    utils.multiclass.type_of_target. Possible type are 'continuous',
    'continuous-multioutput', 'binary', 'multiclass',
    'multiclass-multioutput', 'multilabel-indicator', and 'unknown'.

sparse_input_ : boolean,
    True if the input data to transform is given as a sparse matrix, False
    otherwise.

Examples
--------
>>> from sklearn import preprocessing
>>> lb = preprocessing.LabelBinarizer()
>>> lb.fit([1, 2, 6, 4, 2])
LabelBinarizer()
>>> lb.classes_
array([1, 2, 4, 6])
>>> lb.transform([1, 6])
array([[1, 0, 0, 0],
       [0, 0, 0, 1]])

Binary targets transform to a column vector

>>> lb = preprocessing.LabelBinarizer()
>>> lb.fit_transform(['yes', 'no', 'no', 'yes'])
array([[1],
       [0],
       [0],
       [1]])

Passing a 2D matrix for multilabel classification

>>> import numpy as np
>>> lb.fit(np.array([[0, 1, 1], [1, 0, 0]]))
LabelBinarizer()
>>> lb.classes_
array([0, 1, 2])
>>> lb.transform([0, 1, 2, 1])
array([[1, 0, 0],
       [0, 1, 0],
       [0, 0, 1],
       [0, 1, 0]])

See also
--------
label_binarize : function to perform the transform operation of
    LabelBinarizer with fixed classes.
sklearn.preprocessing.OneHotEncoder : encode categorical features
    using a one-hot aka one-of-K scheme.
*)

val fit : y:Ndarray.t -> t -> t
(**
Fit label binarizer

Parameters
----------
y : array of shape [n_samples,] or [n_samples, n_classes]
    Target values. The 2-d matrix should only contain 0 and 1,
    represents multilabel classification.

Returns
-------
self : returns an instance of self.
*)

val fit_transform : y:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Fit label binarizer and transform multi-class labels to binary
labels.

The output of transform is sometimes referred to as
the 1-of-K coding scheme.

Parameters
----------
y : array or sparse matrix of shape [n_samples,] or             [n_samples, n_classes]
    Target values. The 2-d matrix should only contain 0 and 1,
    represents multilabel classification. Sparse matrix can be
    CSR, CSC, COO, DOK, or LIL.

Returns
-------
Y : array or CSR matrix of shape [n_samples, n_classes]
    Shape will be [n_samples, 1] for binary problems.
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

val inverse_transform : ?threshold:[`Float of float | `None] -> y:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> t -> Py.Object.t
(**
Transform binary labels back to multi-class labels

Parameters
----------
Y : numpy array or sparse matrix with shape [n_samples, n_classes]
    Target values. All sparse matrices are converted to CSR before
    inverse transformation.

threshold : float or None
    Threshold used in the binary and multi-label cases.

    Use 0 when ``Y`` contains the output of decision_function
    (classifier).
    Use 0.5 when ``Y`` contains the output of predict_proba.

    If None, the threshold is assumed to be half way between
    neg_label and pos_label.

Returns
-------
y : numpy array or CSR matrix of shape [n_samples] Target values.

Notes
-----
In the case when the binary labels are fractional
(probabilistic), inverse_transform chooses the class with the
greatest value. Typically, this allows to use the output of a
linear model's decision_function method directly as the input
of inverse_transform.
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

val transform : y:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Transform multi-class labels to binary labels

The output of transform is sometimes referred to by some authors as
the 1-of-K coding scheme.

Parameters
----------
y : array or sparse matrix of shape [n_samples,] or             [n_samples, n_classes]
    Target values. The 2-d matrix should only contain 0 and 1,
    represents multilabel classification. Sparse matrix can be
    CSR, CSC, COO, DOK, or LIL.

Returns
-------
Y : numpy array or CSR matrix of shape [n_samples, n_classes]
    Shape will be [n_samples, 1] for binary problems.
*)


(** Attribute classes_: see constructor for documentation *)
val classes_ : t -> Ndarray.t

(** Attribute y_type_: see constructor for documentation *)
val y_type_ : t -> string

(** Attribute sparse_input_: see constructor for documentation *)
val sparse_input_ : t -> bool

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

module LabelEncoder : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : unit -> t
(**
Encode target labels with value between 0 and n_classes-1.

This transformer should be used to encode target values, *i.e.* `y`, and
not the input `X`.

Read more in the :ref:`User Guide <preprocessing_targets>`.

.. versionadded:: 0.12

Attributes
----------
classes_ : array of shape (n_class,)
    Holds the label for each class.

Examples
--------
`LabelEncoder` can be used to normalize labels.

>>> from sklearn import preprocessing
>>> le = preprocessing.LabelEncoder()
>>> le.fit([1, 2, 2, 6])
LabelEncoder()
>>> le.classes_
array([1, 2, 6])
>>> le.transform([1, 1, 2, 6])
array([0, 0, 1, 2]...)
>>> le.inverse_transform([0, 0, 1, 2])
array([1, 1, 2, 6])

It can also be used to transform non-numerical labels (as long as they are
hashable and comparable) to numerical labels.

>>> le = preprocessing.LabelEncoder()
>>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
LabelEncoder()
>>> list(le.classes_)
['amsterdam', 'paris', 'tokyo']
>>> le.transform(["tokyo", "tokyo", "paris"])
array([2, 2, 1]...)
>>> list(le.inverse_transform([2, 2, 1]))
['tokyo', 'tokyo', 'paris']

See also
--------
sklearn.preprocessing.OrdinalEncoder : Encode categorical features
    using an ordinal encoding scheme.

sklearn.preprocessing.OneHotEncoder : Encode categorical features
    as a one-hot numeric array.
*)

val fit : y:Ndarray.t -> t -> t
(**
Fit label encoder

Parameters
----------
y : array-like of shape (n_samples,)
    Target values.

Returns
-------
self : returns an instance of self.
*)

val fit_transform : y:Ndarray.t -> t -> Ndarray.t
(**
Fit label encoder and return encoded labels

Parameters
----------
y : array-like of shape [n_samples]
    Target values.

Returns
-------
y : array-like of shape [n_samples]
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

val inverse_transform : y:Ndarray.t -> t -> Ndarray.t
(**
Transform labels back to original encoding.

Parameters
----------
y : numpy array of shape [n_samples]
    Target values.

Returns
-------
y : numpy array of shape [n_samples]
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

val transform : y:Ndarray.t -> t -> Ndarray.t
(**
Transform labels to normalized encoding.

Parameters
----------
y : array-like of shape [n_samples]
    Target values.

Returns
-------
y : array-like of shape [n_samples]
*)


(** Attribute classes_: see constructor for documentation *)
val classes_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

module MaxAbsScaler : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?copy:bool -> unit -> t
(**
Scale each feature by its maximum absolute value.

This estimator scales and translates each feature individually such
that the maximal absolute value of each feature in the
training set will be 1.0. It does not shift/center the data, and
thus does not destroy any sparsity.

This scaler can also be applied to sparse CSR or CSC matrices.

.. versionadded:: 0.17

Parameters
----------
copy : boolean, optional, default is True
    Set to False to perform inplace scaling and avoid a copy (if the input
    is already a numpy array).

Attributes
----------
scale_ : ndarray, shape (n_features,)
    Per feature relative scaling of the data.

    .. versionadded:: 0.17
       *scale_* attribute.

max_abs_ : ndarray, shape (n_features,)
    Per feature maximum absolute value.

n_samples_seen_ : int
    The number of samples processed by the estimator. Will be reset on
    new calls to fit, but increments across ``partial_fit`` calls.

Examples
--------
>>> from sklearn.preprocessing import MaxAbsScaler
>>> X = [[ 1., -1.,  2.],
...      [ 2.,  0.,  0.],
...      [ 0.,  1., -1.]]
>>> transformer = MaxAbsScaler().fit(X)
>>> transformer
MaxAbsScaler()
>>> transformer.transform(X)
array([[ 0.5, -1. ,  1. ],
       [ 1. ,  0. ,  0. ],
       [ 0. ,  1. , -0.5]])

See also
--------
maxabs_scale: Equivalent function without the estimator API.

Notes
-----
NaNs are treated as missing values: disregarded in fit, and maintained in
transform.

For a comparison of the different scalers, transformers, and normalizers,
see :ref:`examples/preprocessing/plot_all_scaling.py
<sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
*)

val fit : ?y:Py.Object.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> t
(**
Compute the maximum absolute value to be used for later scaling.

Parameters
----------
X : {array-like, sparse matrix}, shape [n_samples, n_features]
    The data used to compute the per-feature minimum and maximum
    used for later scaling along the features axis.
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

val inverse_transform : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Py.Object.t
(**
Scale back the data to the original representation

Parameters
----------
X : {array-like, sparse matrix}
    The data that should be transformed back.
*)

val partial_fit : ?y:Py.Object.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> t
(**
Online computation of max absolute value of X for later scaling.

All of X is processed as a single batch. This is intended for cases
when :meth:`fit` is not feasible due to very large number of
`n_samples` or because X is read from a continuous stream.

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

val transform : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Scale the data

Parameters
----------
X : {array-like, sparse matrix}
    The data that should be scaled.
*)


(** Attribute scale_: see constructor for documentation *)
val scale_ : t -> Ndarray.t

(** Attribute max_abs_: see constructor for documentation *)
val max_abs_ : t -> Ndarray.t

(** Attribute n_samples_seen_: see constructor for documentation *)
val n_samples_seen_ : t -> int

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

module MinMaxScaler : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?feature_range:Py.Object.t -> ?copy:bool -> unit -> t
(**
Transform features by scaling each feature to a given range.

This estimator scales and translates each feature individually such
that it is in the given range on the training set, e.g. between
zero and one.

The transformation is given by::

    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = X_std * (max - min) + min

where min, max = feature_range.

The transformation is calculated as::

    X_scaled = scale * X + min - X.min(axis=0) * scale
    where scale = (max - min) / (X.max(axis=0) - X.min(axis=0))

This transformation is often used as an alternative to zero mean,
unit variance scaling.

Read more in the :ref:`User Guide <preprocessing_scaler>`.

Parameters
----------
feature_range : tuple (min, max), default=(0, 1)
    Desired range of transformed data.

copy : bool, default=True
    Set to False to perform inplace row normalization and avoid a
    copy (if the input is already a numpy array).

Attributes
----------
min_ : ndarray of shape (n_features,)
    Per feature adjustment for minimum. Equivalent to
    ``min - X.min(axis=0) * self.scale_``

scale_ : ndarray of shape (n_features,)
    Per feature relative scaling of the data. Equivalent to
    ``(max - min) / (X.max(axis=0) - X.min(axis=0))``

    .. versionadded:: 0.17
       *scale_* attribute.

data_min_ : ndarray of shape (n_features,)
    Per feature minimum seen in the data

    .. versionadded:: 0.17
       *data_min_*

data_max_ : ndarray of shape (n_features,)
    Per feature maximum seen in the data

    .. versionadded:: 0.17
       *data_max_*

data_range_ : ndarray of shape (n_features,)
    Per feature range ``(data_max_ - data_min_)`` seen in the data

    .. versionadded:: 0.17
       *data_range_*

n_samples_seen_ : int
    The number of samples processed by the estimator.
    It will be reset on new calls to fit, but increments across
    ``partial_fit`` calls.

Examples
--------
>>> from sklearn.preprocessing import MinMaxScaler
>>> data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
>>> scaler = MinMaxScaler()
>>> print(scaler.fit(data))
MinMaxScaler()
>>> print(scaler.data_max_)
[ 1. 18.]
>>> print(scaler.transform(data))
[[0.   0.  ]
 [0.25 0.25]
 [0.5  0.5 ]
 [1.   1.  ]]
>>> print(scaler.transform([[2, 2]]))
[[1.5 0. ]]

See also
--------
minmax_scale: Equivalent function without the estimator API.

Notes
-----
NaNs are treated as missing values: disregarded in fit, and maintained in
transform.

For a comparison of the different scalers, transformers, and normalizers,
see :ref:`examples/preprocessing/plot_all_scaling.py
<sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
*)

val fit : ?y:Py.Object.t -> x:Ndarray.t -> t -> t
(**
Compute the minimum and maximum to be used for later scaling.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The data used to compute the per-feature minimum and maximum
    used for later scaling along the features axis.

y : None
    Ignored.

Returns
-------
self : object
    Fitted scaler.
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

val inverse_transform : x:Ndarray.t -> t -> Ndarray.t
(**
Undo the scaling of X according to feature_range.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Input data that will be transformed. It cannot be sparse.

Returns
-------
Xt : array-like of shape (n_samples, n_features)
    Transformed data.
*)

val partial_fit : ?y:Py.Object.t -> x:Ndarray.t -> t -> t
(**
Online computation of min and max on X for later scaling.

All of X is processed as a single batch. This is intended for cases
when :meth:`fit` is not feasible due to very large number of
`n_samples` or because X is read from a continuous stream.

Parameters
----------
X : array-like of shape (n_samples, n_features)
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

val transform : x:Ndarray.t -> t -> Ndarray.t
(**
Scale features of X according to feature_range.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Input data that will be transformed.

Returns
-------
Xt : array-like of shape (n_samples, n_features)
    Transformed data.
*)


(** Attribute min_: see constructor for documentation *)
val min_ : t -> Ndarray.t

(** Attribute scale_: see constructor for documentation *)
val scale_ : t -> Ndarray.t

(** Attribute data_min_: see constructor for documentation *)
val data_min_ : t -> Ndarray.t

(** Attribute data_max_: see constructor for documentation *)
val data_max_ : t -> Ndarray.t

(** Attribute data_range_: see constructor for documentation *)
val data_range_ : t -> Ndarray.t

(** Attribute n_samples_seen_: see constructor for documentation *)
val n_samples_seen_ : t -> int

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

module MultiLabelBinarizer : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?classes:Ndarray.t -> ?sparse_output:bool -> unit -> t
(**
Transform between iterable of iterables and a multilabel format

Although a list of sets or tuples is a very intuitive format for multilabel
data, it is unwieldy to process. This transformer converts between this
intuitive format and the supported multilabel format: a (samples x classes)
binary matrix indicating the presence of a class label.

Parameters
----------
classes : array-like of shape [n_classes] (optional)
    Indicates an ordering for the class labels.
    All entries should be unique (cannot contain duplicate classes).

sparse_output : boolean (default: False),
    Set to true if output binary array is desired in CSR sparse format

Attributes
----------
classes_ : array of labels
    A copy of the `classes` parameter where provided,
    or otherwise, the sorted set of classes found when fitting.

Examples
--------
>>> from sklearn.preprocessing import MultiLabelBinarizer
>>> mlb = MultiLabelBinarizer()
>>> mlb.fit_transform([(1, 2), (3,)])
array([[1, 1, 0],
       [0, 0, 1]])
>>> mlb.classes_
array([1, 2, 3])

>>> mlb.fit_transform([{'sci-fi', 'thriller'}, {'comedy'}])
array([[0, 1, 1],
       [1, 0, 0]])
>>> list(mlb.classes_)
['comedy', 'sci-fi', 'thriller']

A common mistake is to pass in a list, which leads to the following issue:

>>> mlb = MultiLabelBinarizer()
>>> mlb.fit(['sci-fi', 'thriller', 'comedy'])
MultiLabelBinarizer()
>>> mlb.classes_
array(['-', 'c', 'd', 'e', 'f', 'h', 'i', 'l', 'm', 'o', 'r', 's', 't',
    'y'], dtype=object)

To correct this, the list of labels should be passed in as:

>>> mlb = MultiLabelBinarizer()
>>> mlb.fit([['sci-fi', 'thriller', 'comedy']])
MultiLabelBinarizer()
>>> mlb.classes_
array(['comedy', 'sci-fi', 'thriller'], dtype=object)

See also
--------
sklearn.preprocessing.OneHotEncoder : encode categorical features
    using a one-hot aka one-of-K scheme.
*)

val fit : y:Ndarray.List.t -> t -> t
(**
Fit the label sets binarizer, storing :term:`classes_`

Parameters
----------
y : iterable of iterables
    A set of labels (any orderable and hashable object) for each
    sample. If the `classes` parameter is set, `y` will not be
    iterated.

Returns
-------
self : returns this MultiLabelBinarizer instance
*)

val fit_transform : y:Ndarray.List.t -> t -> Ndarray.t
(**
Fit the label sets binarizer and transform the given label sets

Parameters
----------
y : iterable of iterables
    A set of labels (any orderable and hashable object) for each
    sample. If the `classes` parameter is set, `y` will not be
    iterated.

Returns
-------
y_indicator : array or CSR matrix, shape (n_samples, n_classes)
    A matrix such that `y_indicator[i, j] = 1` iff `classes_[j]` is in
    `y[i]`, and 0 otherwise.
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

val inverse_transform : yt:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Py.Object.t
(**
Transform the given indicator matrix into label sets

Parameters
----------
yt : array or sparse matrix of shape (n_samples, n_classes)
    A matrix containing only 1s ands 0s.

Returns
-------
y : list of tuples
    The set of labels for each sample such that `y[i]` consists of
    `classes_[j]` for each `yt[i, j] == 1`.
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

val transform : y:Py.Object.t -> t -> Ndarray.t
(**
Transform the given label sets

Parameters
----------
y : iterable of iterables
    A set of labels (any orderable and hashable object) for each
    sample. If the `classes` parameter is set, `y` will not be
    iterated.

Returns
-------
y_indicator : array or CSR matrix, shape (n_samples, n_classes)
    A matrix such that `y_indicator[i, j] = 1` iff `classes_[j]` is in
    `y[i]`, and 0 otherwise.
*)


(** Attribute classes_: see constructor for documentation *)
val classes_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

module Normalizer : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?norm:[`L1 | `L2 | `Max | `PyObject of Py.Object.t] -> ?copy:bool -> unit -> t
(**
Normalize samples individually to unit norm.

Each sample (i.e. each row of the data matrix) with at least one
non zero component is rescaled independently of other samples so
that its norm (l1 or l2) equals one.

This transformer is able to work both with dense numpy arrays and
scipy.sparse matrix (use CSR format if you want to avoid the burden of
a copy / conversion).

Scaling inputs to unit norms is a common operation for text
classification or clustering for instance. For instance the dot
product of two l2-normalized TF-IDF vectors is the cosine similarity
of the vectors and is the base similarity metric for the Vector
Space Model commonly used by the Information Retrieval community.

Read more in the :ref:`User Guide <preprocessing_normalization>`.

Parameters
----------
norm : 'l1', 'l2', or 'max', optional ('l2' by default)
    The norm to use to normalize each non zero sample.

copy : boolean, optional, default True
    set to False to perform inplace row normalization and avoid a
    copy (if the input is already a numpy array or a scipy.sparse
    CSR matrix).

Examples
--------
>>> from sklearn.preprocessing import Normalizer
>>> X = [[4, 1, 2, 2],
...      [1, 3, 9, 3],
...      [5, 7, 5, 1]]
>>> transformer = Normalizer().fit(X)  # fit does nothing.
>>> transformer
Normalizer()
>>> transformer.transform(X)
array([[0.8, 0.2, 0.4, 0.4],
       [0.1, 0.3, 0.9, 0.3],
       [0.5, 0.7, 0.5, 0.1]])

Notes
-----
This estimator is stateless (besides constructor parameters), the
fit method does nothing but is useful when used in a pipeline.

For a comparison of the different scalers, transformers, and normalizers,
see :ref:`examples/preprocessing/plot_all_scaling.py
<sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.


See also
--------
normalize: Equivalent function without the estimator API.
*)

val fit : ?y:Py.Object.t -> x:Ndarray.t -> t -> t
(**
Do nothing and return the estimator unchanged

This method is just there to implement the usual API and hence
work in pipelines.

Parameters
----------
X : array-like
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

val transform : ?copy:Py.Object.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Scale each non zero row of X to unit norm

Parameters
----------
X : {array-like, sparse matrix}, shape [n_samples, n_features]
    The data to normalize, row by row. scipy.sparse matrices should be
    in CSR format to avoid an un-necessary copy.
copy : bool, optional (default: None)
    Copy the input X or not.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

module OneHotEncoder : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?categories:[`Auto | `PyObject of Py.Object.t] -> ?drop:[`First | `PyObject of Py.Object.t] -> ?sparse:bool -> ?dtype:Py.Object.t -> ?handle_unknown:[`Error | `Ignore] -> unit -> t
(**
Encode categorical features as a one-hot numeric array.

The input to this transformer should be an array-like of integers or
strings, denoting the values taken on by categorical (discrete) features.
The features are encoded using a one-hot (aka 'one-of-K' or 'dummy')
encoding scheme. This creates a binary column for each category and
returns a sparse matrix or dense array (depending on the ``sparse``
parameter)

By default, the encoder derives the categories based on the unique values
in each feature. Alternatively, you can also specify the `categories`
manually.

This encoding is needed for feeding categorical data to many scikit-learn
estimators, notably linear models and SVMs with the standard kernels.

Note: a one-hot encoding of y labels should use a LabelBinarizer
instead.

Read more in the :ref:`User Guide <preprocessing_categorical_features>`.

.. versionchanged:: 0.20

Parameters
----------
categories : 'auto' or a list of array-like, default='auto'
    Categories (unique values) per feature:

    - 'auto' : Determine categories automatically from the training data.
    - list : ``categories[i]`` holds the categories expected in the ith
      column. The passed categories should not mix strings and numeric
      values within a single feature, and should be sorted in case of
      numeric values.

    The used categories can be found in the ``categories_`` attribute.

drop : 'first' or a array-like of shape (n_features,), default=None
    Specifies a methodology to use to drop one of the categories per
    feature. This is useful in situations where perfectly collinear
    features cause problems, such as when feeding the resulting data
    into a neural network or an unregularized regression.

    - None : retain all features (the default).
    - 'first' : drop the first category in each feature. If only one
      category is present, the feature will be dropped entirely.
    - array : ``drop[i]`` is the category in feature ``X[:, i]`` that
      should be dropped.

sparse : bool, default=True
    Will return sparse matrix if set True else will return an array.

dtype : number type, default=np.float
    Desired dtype of output.

handle_unknown : {'error', 'ignore'}, default='error'
    Whether to raise an error or ignore if an unknown categorical feature
    is present during transform (default is to raise). When this parameter
    is set to 'ignore' and an unknown category is encountered during
    transform, the resulting one-hot encoded columns for this feature
    will be all zeros. In the inverse transform, an unknown category
    will be denoted as None.

Attributes
----------
categories_ : list of arrays
    The categories of each feature determined during fitting
    (in order of the features in X and corresponding with the output
    of ``transform``). This includes the category specified in ``drop``
    (if any).

drop_idx_ : array of shape (n_features,)
    ``drop_idx_[i]`` is the index in ``categories_[i]`` of the category to
    be dropped for each feature. None if all the transformed features will
    be retained.

See Also
--------
sklearn.preprocessing.OrdinalEncoder : Performs an ordinal (integer)
  encoding of the categorical features.
sklearn.feature_extraction.DictVectorizer : Performs a one-hot encoding of
  dictionary items (also handles string-valued features).
sklearn.feature_extraction.FeatureHasher : Performs an approximate one-hot
  encoding of dictionary items or strings.
sklearn.preprocessing.LabelBinarizer : Binarizes labels in a one-vs-all
  fashion.
sklearn.preprocessing.MultiLabelBinarizer : Transforms between iterable of
  iterables and a multilabel format, e.g. a (samples x classes) binary
  matrix indicating the presence of a class label.

Examples
--------
Given a dataset with two features, we let the encoder find the unique
values per feature and transform the data to a binary one-hot encoding.

>>> from sklearn.preprocessing import OneHotEncoder
>>> enc = OneHotEncoder(handle_unknown='ignore')
>>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
>>> enc.fit(X)
OneHotEncoder(handle_unknown='ignore')
>>> enc.categories_
[array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
>>> enc.transform([['Female', 1], ['Male', 4]]).toarray()
array([[1., 0., 1., 0., 0.],
       [0., 1., 0., 0., 0.]])
>>> enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])
array([['Male', 1],
       [None, 2]], dtype=object)
>>> enc.get_feature_names(['gender', 'group'])
array(['gender_Female', 'gender_Male', 'group_1', 'group_2', 'group_3'],
  dtype=object)
>>> drop_enc = OneHotEncoder(drop='first').fit(X)
>>> drop_enc.categories_
[array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
>>> drop_enc.transform([['Female', 1], ['Male', 2]]).toarray()
array([[0., 0., 0.],
       [1., 1., 0.]])
*)

val fit : ?y:Py.Object.t -> x:Ndarray.t -> t -> t
(**
Fit OneHotEncoder to X.

Parameters
----------
X : array-like, shape [n_samples, n_features]
    The data to determine the categories of each feature.

y : None
    Ignored. This parameter exists only for compatibility with
    :class:`sklearn.pipeline.Pipeline`.

Returns
-------
self
*)

val fit_transform : ?y:Py.Object.t -> x:Ndarray.t -> t -> Ndarray.t
(**
Fit OneHotEncoder to X, then transform X.

Equivalent to fit(X).transform(X) but more convenient.

Parameters
----------
X : array-like, shape [n_samples, n_features]
    The data to encode.

y : None
    Ignored. This parameter exists only for compatibility with
    :class:`sklearn.pipeline.Pipeline`.

Returns
-------
X_out : sparse matrix if sparse=True else a 2-d array
    Transformed input.
*)

val get_feature_names : ?input_features:string list -> t -> Ndarray.t
(**
Return feature names for output features.

Parameters
----------
input_features : list of str of shape (n_features,)
    String names for input features if available. By default,
    "x0", "x1", ... "xn_features" is used.

Returns
-------
output_feature_names : ndarray of shape (n_output_features,)
    Array of feature names.
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

val inverse_transform : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Convert the data back to the original representation.

In case unknown categories are encountered (all zeros in the
one-hot encoding), ``None`` is used to represent this category.

Parameters
----------
X : array-like or sparse matrix, shape [n_samples, n_encoded_features]
    The transformed data.

Returns
-------
X_tr : array-like, shape [n_samples, n_features]
    Inverse transformed array.
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
Transform X using one-hot encoding.

Parameters
----------
X : array-like, shape [n_samples, n_features]
    The data to encode.

Returns
-------
X_out : sparse matrix if sparse=True else a 2-d array
    Transformed input.
*)


(** Attribute categories_: see constructor for documentation *)
val categories_ : t -> Py.Object.t

(** Attribute drop_idx_: see constructor for documentation *)
val drop_idx_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

module OrdinalEncoder : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?categories:[`Auto | `PyObject of Py.Object.t] -> ?dtype:Py.Object.t -> unit -> t
(**
Encode categorical features as an integer array.

The input to this transformer should be an array-like of integers or
strings, denoting the values taken on by categorical (discrete) features.
The features are converted to ordinal integers. This results in
a single column of integers (0 to n_categories - 1) per feature.

Read more in the :ref:`User Guide <preprocessing_categorical_features>`.

.. versionchanged:: 0.20.1

Parameters
----------
categories : 'auto' or a list of array-like, default='auto'
    Categories (unique values) per feature:

    - 'auto' : Determine categories automatically from the training data.
    - list : ``categories[i]`` holds the categories expected in the ith
      column. The passed categories should not mix strings and numeric
      values, and should be sorted in case of numeric values.

    The used categories can be found in the ``categories_`` attribute.

dtype : number type, default np.float64
    Desired dtype of output.

Attributes
----------
categories_ : list of arrays
    The categories of each feature determined during fitting
    (in order of the features in X and corresponding with the output
    of ``transform``).

See Also
--------
sklearn.preprocessing.OneHotEncoder : Performs a one-hot encoding of
  categorical features.
sklearn.preprocessing.LabelEncoder : Encodes target labels with values
  between 0 and n_classes-1.

Examples
--------
Given a dataset with two features, we let the encoder find the unique
values per feature and transform the data to an ordinal encoding.

>>> from sklearn.preprocessing import OrdinalEncoder
>>> enc = OrdinalEncoder()
>>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
>>> enc.fit(X)
OrdinalEncoder()
>>> enc.categories_
[array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
>>> enc.transform([['Female', 3], ['Male', 1]])
array([[0., 2.],
       [1., 0.]])

>>> enc.inverse_transform([[1, 0], [0, 1]])
array([['Male', 1],
       ['Female', 2]], dtype=object)
*)

val fit : ?y:Py.Object.t -> x:Ndarray.t -> t -> t
(**
Fit the OrdinalEncoder to X.

Parameters
----------
X : array-like, shape [n_samples, n_features]
    The data to determine the categories of each feature.

y : None
    Ignored. This parameter exists only for compatibility with
    :class:`sklearn.pipeline.Pipeline`.

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

val inverse_transform : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Ndarray.t
(**
Convert the data back to the original representation.

Parameters
----------
X : array-like or sparse matrix, shape [n_samples, n_encoded_features]
    The transformed data.

Returns
-------
X_tr : array-like, shape [n_samples, n_features]
    Inverse transformed array.
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
Transform X to ordinal codes.

Parameters
----------
X : array-like, shape [n_samples, n_features]
    The data to encode.

Returns
-------
X_out : sparse matrix or a 2-d array
    Transformed input.
*)


(** Attribute categories_: see constructor for documentation *)
val categories_ : t -> Py.Object.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

module PolynomialFeatures : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?degree:int -> ?interaction_only:bool -> ?include_bias:bool -> ?order:[`C | `F] -> unit -> t
(**
Generate polynomial and interaction features.

Generate a new feature matrix consisting of all polynomial combinations
of the features with degree less than or equal to the specified degree.
For example, if an input sample is two dimensional and of the form
[a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].

Parameters
----------
degree : integer
    The degree of the polynomial features. Default = 2.

interaction_only : boolean, default = False
    If true, only interaction features are produced: features that are
    products of at most ``degree`` *distinct* input features (so not
    ``x[1] ** 2``, ``x[0] * x[2] ** 3``, etc.).

include_bias : boolean
    If True (default), then include a bias column, the feature in which
    all polynomial powers are zero (i.e. a column of ones - acts as an
    intercept term in a linear model).

order : str in {'C', 'F'}, default 'C'
    Order of output array in the dense case. 'F' order is faster to
    compute, but may slow down subsequent estimators.

    .. versionadded:: 0.21

Examples
--------
>>> import numpy as np
>>> from sklearn.preprocessing import PolynomialFeatures
>>> X = np.arange(6).reshape(3, 2)
>>> X
array([[0, 1],
       [2, 3],
       [4, 5]])
>>> poly = PolynomialFeatures(2)
>>> poly.fit_transform(X)
array([[ 1.,  0.,  1.,  0.,  0.,  1.],
       [ 1.,  2.,  3.,  4.,  6.,  9.],
       [ 1.,  4.,  5., 16., 20., 25.]])
>>> poly = PolynomialFeatures(interaction_only=True)
>>> poly.fit_transform(X)
array([[ 1.,  0.,  1.,  0.],
       [ 1.,  2.,  3.,  6.],
       [ 1.,  4.,  5., 20.]])

Attributes
----------
powers_ : array, shape (n_output_features, n_input_features)
    powers_[i, j] is the exponent of the jth input in the ith output.

n_input_features_ : int
    The total number of input features.

n_output_features_ : int
    The total number of polynomial output features. The number of output
    features is computed by iterating over all suitably sized combinations
    of input features.

Notes
-----
Be aware that the number of features in the output array scales
polynomially in the number of features of the input array, and
exponentially in the degree. High degrees can cause overfitting.

See :ref:`examples/linear_model/plot_polynomial_interpolation.py
<sphx_glr_auto_examples_linear_model_plot_polynomial_interpolation.py>`
*)

val fit : ?y:Py.Object.t -> x:Ndarray.t -> t -> t
(**
Compute number of output features.


Parameters
----------
X : array-like, shape (n_samples, n_features)
    The data.

Returns
-------
self : instance
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

val get_feature_names : ?input_features:string list -> t -> string list
(**
Return feature names for output features

Parameters
----------
input_features : list of string, length n_features, optional
    String names for input features if available. By default,
    "x0", "x1", ... "xn_features" is used.

Returns
-------
output_feature_names : list of string, length n_output_features
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
Transform data to polynomial features

Parameters
----------
X : array-like or CSR/CSC sparse matrix, shape [n_samples, n_features]
    The data to transform, row by row.

    Prefer CSR over CSC for sparse input (for speed), but CSC is
    required if the degree is 4 or higher. If the degree is less than
    4 and the input format is CSC, it will be converted to CSR, have
    its polynomial features generated, then converted back to CSC.

    If the degree is 2 or 3, the method described in "Leveraging
    Sparsity to Speed Up Polynomial Feature Expansions of CSR Matrices
    Using K-Simplex Numbers" by Andrew Nystrom and John Hughes is
    used, which is much faster than the method used on CSC input. For
    this reason, a CSC input will be converted to CSR, and the output
    will be converted back to CSC prior to being returned, hence the
    preference of CSR.

Returns
-------
XP : np.ndarray or CSR/CSC sparse matrix, shape [n_samples, NP]
    The matrix of features, where NP is the number of polynomial
    features generated from the combination of inputs.
*)


(** Attribute powers_: see constructor for documentation *)
val powers_ : t -> Ndarray.t

(** Attribute n_input_features_: see constructor for documentation *)
val n_input_features_ : t -> int

(** Attribute n_output_features_: see constructor for documentation *)
val n_output_features_ : t -> int

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

module PowerTransformer : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?method_:string -> ?standardize:bool -> ?copy:bool -> unit -> t
(**
Apply a power transform featurewise to make data more Gaussian-like.

Power transforms are a family of parametric, monotonic transformations
that are applied to make data more Gaussian-like. This is useful for
modeling issues related to heteroscedasticity (non-constant variance),
or other situations where normality is desired.

Currently, PowerTransformer supports the Box-Cox transform and the
Yeo-Johnson transform. The optimal parameter for stabilizing variance and
minimizing skewness is estimated through maximum likelihood.

Box-Cox requires input data to be strictly positive, while Yeo-Johnson
supports both positive or negative data.

By default, zero-mean, unit-variance normalization is applied to the
transformed data.

Read more in the :ref:`User Guide <preprocessing_transformer>`.

.. versionadded:: 0.20

Parameters
----------
method : str, (default='yeo-johnson')
    The power transform method. Available methods are:

    - 'yeo-johnson' [1]_, works with positive and negative values
    - 'box-cox' [2]_, only works with strictly positive values

standardize : boolean, default=True
    Set to True to apply zero-mean, unit-variance normalization to the
    transformed output.

copy : boolean, optional, default=True
    Set to False to perform inplace computation during transformation.

Attributes
----------
lambdas_ : array of float, shape (n_features,)
    The parameters of the power transformation for the selected features.

Examples
--------
>>> import numpy as np
>>> from sklearn.preprocessing import PowerTransformer
>>> pt = PowerTransformer()
>>> data = [[1, 2], [3, 2], [4, 5]]
>>> print(pt.fit(data))
PowerTransformer()
>>> print(pt.lambdas_)
[ 1.386... -3.100...]
>>> print(pt.transform(data))
[[-1.316... -0.707...]
 [ 0.209... -0.707...]
 [ 1.106...  1.414...]]

See also
--------
power_transform : Equivalent function without the estimator API.

QuantileTransformer : Maps data to a standard normal distribution with
    the parameter `output_distribution='normal'`.

Notes
-----
NaNs are treated as missing values: disregarded in ``fit``, and maintained
in ``transform``.

For a comparison of the different scalers, transformers, and normalizers,
see :ref:`examples/preprocessing/plot_all_scaling.py
<sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.

References
----------

.. [1] I.K. Yeo and R.A. Johnson, "A new family of power transformations to
       improve normality or symmetry." Biometrika, 87(4), pp.954-959,
       (2000).

.. [2] G.E.P. Box and D.R. Cox, "An Analysis of Transformations", Journal
       of the Royal Statistical Society B, 26, 211-252 (1964).
*)

val fit : ?y:Py.Object.t -> x:Ndarray.t -> t -> t
(**
Estimate the optimal parameter lambda for each feature.

The optimal lambda parameter for minimizing skewness is estimated on
each feature independently using maximum likelihood.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    The data used to estimate the optimal transformation parameters.

y : Ignored

Returns
-------
self : object
*)

val fit_transform : ?y:Py.Object.t -> x:Py.Object.t -> t -> Ndarray.t
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

val inverse_transform : x:Ndarray.t -> t -> Ndarray.t
(**
Apply the inverse power transformation using the fitted lambdas.

The inverse of the Box-Cox transformation is given by::

    if lambda_ == 0:
        X = exp(X_trans)
    else:
        X = (X_trans * lambda_ + 1) ** (1 / lambda_)

The inverse of the Yeo-Johnson transformation is given by::

    if X >= 0 and lambda_ == 0:
        X = exp(X_trans) - 1
    elif X >= 0 and lambda_ != 0:
        X = (X_trans * lambda_ + 1) ** (1 / lambda_) - 1
    elif X < 0 and lambda_ != 2:
        X = 1 - (-(2 - lambda_) * X_trans + 1) ** (1 / (2 - lambda_))
    elif X < 0 and lambda_ == 2:
        X = 1 - exp(-X_trans)

Parameters
----------
X : array-like, shape (n_samples, n_features)
    The transformed data.

Returns
-------
X : array-like, shape (n_samples, n_features)
    The original data
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
Apply the power transform to each feature using the fitted lambdas.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    The data to be transformed using a power transformation.

Returns
-------
X_trans : array-like, shape (n_samples, n_features)
    The transformed data.
*)


(** Attribute lambdas_: see constructor for documentation *)
val lambdas_ : t -> Py.Object.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

module QuantileTransformer : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_quantiles:int -> ?output_distribution:string -> ?ignore_implicit_zeros:bool -> ?subsample:int -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> ?copy:bool -> unit -> t
(**
Transform features using quantiles information.

This method transforms the features to follow a uniform or a normal
distribution. Therefore, for a given feature, this transformation tends
to spread out the most frequent values. It also reduces the impact of
(marginal) outliers: this is therefore a robust preprocessing scheme.

The transformation is applied on each feature independently. First an
estimate of the cumulative distribution function of a feature is
used to map the original values to a uniform distribution. The obtained
values are then mapped to the desired output distribution using the
associated quantile function. Features values of new/unseen data that fall
below or above the fitted range will be mapped to the bounds of the output
distribution. Note that this transform is non-linear. It may distort linear
correlations between variables measured at the same scale but renders
variables measured at different scales more directly comparable.

Read more in the :ref:`User Guide <preprocessing_transformer>`.

.. versionadded:: 0.19

Parameters
----------
n_quantiles : int, optional (default=1000 or n_samples)
    Number of quantiles to be computed. It corresponds to the number
    of landmarks used to discretize the cumulative distribution function.
    If n_quantiles is larger than the number of samples, n_quantiles is set
    to the number of samples as a larger number of quantiles does not give
    a better approximation of the cumulative distribution function
    estimator.

output_distribution : str, optional (default='uniform')
    Marginal distribution for the transformed data. The choices are
    'uniform' (default) or 'normal'.

ignore_implicit_zeros : bool, optional (default=False)
    Only applies to sparse matrices. If True, the sparse entries of the
    matrix are discarded to compute the quantile statistics. If False,
    these entries are treated as zeros.

subsample : int, optional (default=1e5)
    Maximum number of samples used to estimate the quantiles for
    computational efficiency. Note that the subsampling procedure may
    differ for value-identical sparse and dense matrices.

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by np.random. Note that this is used by subsampling and smoothing
    noise.

copy : boolean, optional, (default=True)
    Set to False to perform inplace transformation and avoid a copy (if the
    input is already a numpy array).

Attributes
----------
n_quantiles_ : integer
    The actual number of quantiles used to discretize the cumulative
    distribution function.

quantiles_ : ndarray, shape (n_quantiles, n_features)
    The values corresponding the quantiles of reference.

references_ : ndarray, shape(n_quantiles, )
    Quantiles of references.

Examples
--------
>>> import numpy as np
>>> from sklearn.preprocessing import QuantileTransformer
>>> rng = np.random.RandomState(0)
>>> X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)
>>> qt = QuantileTransformer(n_quantiles=10, random_state=0)
>>> qt.fit_transform(X)
array([...])

See also
--------
quantile_transform : Equivalent function without the estimator API.
PowerTransformer : Perform mapping to a normal distribution using a power
    transform.
StandardScaler : Perform standardization that is faster, but less robust
    to outliers.
RobustScaler : Perform robust standardization that removes the influence
    of outliers but does not put outliers and inliers on the same scale.

Notes
-----
NaNs are treated as missing values: disregarded in fit, and maintained in
transform.

For a comparison of the different scalers, transformers, and normalizers,
see :ref:`examples/preprocessing/plot_all_scaling.py
<sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
*)

val fit : ?y:Py.Object.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> t
(**
Compute the quantiles used for transforming.

Parameters
----------
X : ndarray or sparse matrix, shape (n_samples, n_features)
    The data used to scale along the features axis. If a sparse
    matrix is provided, it will be converted into a sparse
    ``csc_matrix``. Additionally, the sparse matrix needs to be
    nonnegative if `ignore_implicit_zeros` is False.

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

val inverse_transform : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Py.Object.t
(**
Back-projection to the original space.

Parameters
----------
X : ndarray or sparse matrix, shape (n_samples, n_features)
    The data used to scale along the features axis. If a sparse
    matrix is provided, it will be converted into a sparse
    ``csc_matrix``. Additionally, the sparse matrix needs to be
    nonnegative if `ignore_implicit_zeros` is False.

Returns
-------
Xt : ndarray or sparse matrix, shape (n_samples, n_features)
    The projected data.
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
Feature-wise transformation of the data.

Parameters
----------
X : ndarray or sparse matrix, shape (n_samples, n_features)
    The data used to scale along the features axis. If a sparse
    matrix is provided, it will be converted into a sparse
    ``csc_matrix``. Additionally, the sparse matrix needs to be
    nonnegative if `ignore_implicit_zeros` is False.

Returns
-------
Xt : ndarray or sparse matrix, shape (n_samples, n_features)
    The projected data.
*)


(** Attribute n_quantiles_: see constructor for documentation *)
val n_quantiles_ : t -> int

(** Attribute quantiles_: see constructor for documentation *)
val quantiles_ : t -> Ndarray.t

(** Attribute references_: see constructor for documentation *)
val references_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

module RobustScaler : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?with_centering:bool -> ?with_scaling:bool -> ?quantile_range:Py.Object.t -> ?copy:bool -> unit -> t
(**
Scale features using statistics that are robust to outliers.

This Scaler removes the median and scales the data according to
the quantile range (defaults to IQR: Interquartile Range).
The IQR is the range between the 1st quartile (25th quantile)
and the 3rd quartile (75th quantile).

Centering and scaling happen independently on each feature by
computing the relevant statistics on the samples in the training
set. Median and interquartile range are then stored to be used on
later data using the ``transform`` method.

Standardization of a dataset is a common requirement for many
machine learning estimators. Typically this is done by removing the mean
and scaling to unit variance. However, outliers can often influence the
sample mean / variance in a negative way. In such cases, the median and
the interquartile range often give better results.

.. versionadded:: 0.17

Read more in the :ref:`User Guide <preprocessing_scaler>`.

Parameters
----------
with_centering : boolean, True by default
    If True, center the data before scaling.
    This will cause ``transform`` to raise an exception when attempted on
    sparse matrices, because centering them entails building a dense
    matrix which in common use cases is likely to be too large to fit in
    memory.

with_scaling : boolean, True by default
    If True, scale the data to interquartile range.

quantile_range : tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0
    Default: (25.0, 75.0) = (1st quantile, 3rd quantile) = IQR
    Quantile range used to calculate ``scale_``.

    .. versionadded:: 0.18

copy : boolean, optional, default is True
    If False, try to avoid a copy and do inplace scaling instead.
    This is not guaranteed to always work inplace; e.g. if the data is
    not a NumPy array or scipy.sparse CSR matrix, a copy may still be
    returned.

Attributes
----------
center_ : array of floats
    The median value for each feature in the training set.

scale_ : array of floats
    The (scaled) interquartile range for each feature in the training set.

    .. versionadded:: 0.17
       *scale_* attribute.

Examples
--------
>>> from sklearn.preprocessing import RobustScaler
>>> X = [[ 1., -2.,  2.],
...      [ -2.,  1.,  3.],
...      [ 4.,  1., -2.]]
>>> transformer = RobustScaler().fit(X)
>>> transformer
RobustScaler()
>>> transformer.transform(X)
array([[ 0. , -2. ,  0. ],
       [-1. ,  0. ,  0.4],
       [ 1. ,  0. , -1.6]])

See also
--------
robust_scale: Equivalent function without the estimator API.

:class:`sklearn.decomposition.PCA`
    Further removes the linear correlation across features with
    'whiten=True'.

Notes
-----
For a comparison of the different scalers, transformers, and normalizers,
see :ref:`examples/preprocessing/plot_all_scaling.py
<sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.

https://en.wikipedia.org/wiki/Median
https://en.wikipedia.org/wiki/Interquartile_range
*)

val fit : ?y:Py.Object.t -> x:Ndarray.t -> t -> t
(**
Compute the median and quantiles to be used for scaling.

Parameters
----------
X : array-like, shape [n_samples, n_features]
    The data used to compute the median and quantiles
    used for later scaling along the features axis.
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

val inverse_transform : x:Ndarray.t -> t -> Py.Object.t
(**
Scale back the data to the original representation

Parameters
----------
X : array-like
    The data used to scale along the specified axis.
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
Center and scale the data.

Parameters
----------
X : {array-like, sparse matrix}
    The data used to scale along the specified axis.
*)


(** Attribute center_: see constructor for documentation *)
val center_ : t -> Ndarray.t

(** Attribute scale_: see constructor for documentation *)
val scale_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


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

val fit : ?y:Py.Object.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> t
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

val inverse_transform : ?copy:Py.Object.t -> x:Ndarray.t -> t -> Ndarray.t
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

val partial_fit : ?y:Py.Object.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> t
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

val transform : ?copy:Py.Object.t -> x:Ndarray.t -> t -> Ndarray.t
(**
Perform standardization by centering and scaling

Parameters
----------
X : array-like, shape [n_samples, n_features]
    The data used to scale along the features axis.
copy : bool, optional (default: None)
    Copy the input X or not.
*)


(** Attribute scale_: see constructor for documentation *)
val scale_ : t -> Py.Object.t

(** Attribute mean_: see constructor for documentation *)
val mean_ : t -> Py.Object.t

(** Attribute var_: see constructor for documentation *)
val var_ : t -> Py.Object.t

(** Attribute n_samples_seen_: see constructor for documentation *)
val n_samples_seen_ : t -> Py.Object.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

val add_dummy_feature : ?value:float -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> unit -> Py.Object.t
(**
Augment dataset with an additional dummy feature.

This is useful for fitting an intercept term with implementations which
cannot otherwise fit it directly.

Parameters
----------
X : {array-like, sparse matrix}, shape [n_samples, n_features]
    Data.

value : float
    Value to use for the dummy feature.

Returns
-------

X : {array, sparse matrix}, shape [n_samples, n_features + 1]
    Same data with dummy feature added as first column.

Examples
--------

>>> from sklearn.preprocessing import add_dummy_feature
>>> add_dummy_feature([[0, 1], [1, 0]])
array([[1., 0., 1.],
       [1., 1., 0.]])
*)

val binarize : ?threshold:[`Float of float | `PyObject of Py.Object.t] -> ?copy:bool -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> unit -> Py.Object.t
(**
Boolean thresholding of array-like or scipy.sparse matrix

Read more in the :ref:`User Guide <preprocessing_binarization>`.

Parameters
----------
X : {array-like, sparse matrix}, shape [n_samples, n_features]
    The data to binarize, element by element.
    scipy.sparse matrices should be in CSR or CSC format to avoid an
    un-necessary copy.

threshold : float, optional (0.0 by default)
    Feature values below or equal to this are replaced by 0, above it by 1.
    Threshold may not be less than 0 for operations on sparse matrices.

copy : boolean, optional, default True
    set to False to perform inplace binarization and avoid a copy
    (if the input is already a numpy array or a scipy.sparse CSR / CSC
    matrix and if axis is 1).

See also
--------
Binarizer: Performs binarization using the ``Transformer`` API
    (e.g. as part of a preprocessing :class:`sklearn.pipeline.Pipeline`).
*)

val label_binarize : ?neg_label:int -> ?pos_label:int -> ?sparse_output:bool -> y:Ndarray.t -> classes:Ndarray.t -> unit -> Py.Object.t
(**
Binarize labels in a one-vs-all fashion

Several regression and binary classification algorithms are
available in scikit-learn. A simple way to extend these algorithms
to the multi-class classification case is to use the so-called
one-vs-all scheme.

This function makes it possible to compute this transformation for a
fixed set of class labels known ahead of time.

Parameters
----------
y : array-like
    Sequence of integer labels or multilabel data to encode.

classes : array-like of shape [n_classes]
    Uniquely holds the label for each class.

neg_label : int (default: 0)
    Value with which negative labels must be encoded.

pos_label : int (default: 1)
    Value with which positive labels must be encoded.

sparse_output : boolean (default: False),
    Set to true if output binary array is desired in CSR sparse format

Returns
-------
Y : numpy array or CSR matrix of shape [n_samples, n_classes]
    Shape will be [n_samples, 1] for binary problems.

Examples
--------
>>> from sklearn.preprocessing import label_binarize
>>> label_binarize([1, 6], classes=[1, 2, 4, 6])
array([[1, 0, 0, 0],
       [0, 0, 0, 1]])

The class ordering is preserved:

>>> label_binarize([1, 6], classes=[1, 6, 4, 2])
array([[1, 0, 0, 0],
       [0, 1, 0, 0]])

Binary targets transform to a column vector

>>> label_binarize(['yes', 'no', 'no', 'yes'], classes=['no', 'yes'])
array([[1],
       [0],
       [0],
       [1]])

See also
--------
LabelBinarizer : class used to wrap the functionality of label_binarize and
    allow for fitting to classes independently of the transform operation
*)

val maxabs_scale : ?axis:Py.Object.t -> ?copy:bool -> x:Ndarray.t -> unit -> Py.Object.t
(**
Scale each feature to the [-1, 1] range without breaking the sparsity.

This estimator scales each feature individually such
that the maximal absolute value of each feature in the
training set will be 1.0.

This scaler can also be applied to sparse CSR or CSC matrices.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    The data.

axis : int (0 by default)
    axis used to scale along. If 0, independently scale each feature,
    otherwise (if 1) scale each sample.

copy : boolean, optional, default is True
    Set to False to perform inplace scaling and avoid a copy (if the input
    is already a numpy array).

See also
--------
MaxAbsScaler: Performs scaling to the [-1, 1] range using the``Transformer`` API
    (e.g. as part of a preprocessing :class:`sklearn.pipeline.Pipeline`).

Notes
-----
NaNs are treated as missing values: disregarded to compute the statistics,
and maintained during the data transformation.

For a comparison of the different scalers, transformers, and normalizers,
see :ref:`examples/preprocessing/plot_all_scaling.py
<sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
*)

val minmax_scale : ?feature_range:Py.Object.t -> ?axis:int -> ?copy:bool -> x:Ndarray.t -> unit -> Py.Object.t
(**
Transform features by scaling each feature to a given range.

This estimator scales and translates each feature individually such
that it is in the given range on the training set, i.e. between
zero and one.

The transformation is given by (when ``axis=0``)::

    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = X_std * (max - min) + min

where min, max = feature_range.

The transformation is calculated as (when ``axis=0``)::

   X_scaled = scale * X + min - X.min(axis=0) * scale
   where scale = (max - min) / (X.max(axis=0) - X.min(axis=0))

This transformation is often used as an alternative to zero mean,
unit variance scaling.

Read more in the :ref:`User Guide <preprocessing_scaler>`.

.. versionadded:: 0.17
   *minmax_scale* function interface
   to :class:`sklearn.preprocessing.MinMaxScaler`.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The data.

feature_range : tuple (min, max), default=(0, 1)
    Desired range of transformed data.

axis : int, default=0
    Axis used to scale along. If 0, independently scale each feature,
    otherwise (if 1) scale each sample.

copy : bool, default=True
    Set to False to perform inplace scaling and avoid a copy (if the input
    is already a numpy array).

See also
--------
MinMaxScaler: Performs scaling to a given range using the``Transformer`` API
    (e.g. as part of a preprocessing :class:`sklearn.pipeline.Pipeline`).

Notes
-----
For a comparison of the different scalers, transformers, and normalizers,
see :ref:`examples/preprocessing/plot_all_scaling.py
<sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
*)

val normalize : ?norm:[`L1 | `L2 | `Max | `PyObject of Py.Object.t] -> ?axis:Py.Object.t -> ?copy:bool -> ?return_norm:bool -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> unit -> (Py.Object.t * Py.Object.t)
(**
Scale input vectors individually to unit norm (vector length).

Read more in the :ref:`User Guide <preprocessing_normalization>`.

Parameters
----------
X : {array-like, sparse matrix}, shape [n_samples, n_features]
    The data to normalize, element by element.
    scipy.sparse matrices should be in CSR format to avoid an
    un-necessary copy.

norm : 'l1', 'l2', or 'max', optional ('l2' by default)
    The norm to use to normalize each non zero sample (or each non-zero
    feature if axis is 0).

axis : 0 or 1, optional (1 by default)
    axis used to normalize the data along. If 1, independently normalize
    each sample, otherwise (if 0) normalize each feature.

copy : boolean, optional, default True
    set to False to perform inplace row normalization and avoid a
    copy (if the input is already a numpy array or a scipy.sparse
    CSR matrix and if axis is 1).

return_norm : boolean, default False
    whether to return the computed norms

Returns
-------
X : {array-like, sparse matrix}, shape [n_samples, n_features]
    Normalized input X.

norms : array, shape [n_samples] if axis=1 else [n_features]
    An array of norms along given axis for X.
    When X is sparse, a NotImplementedError will be raised
    for norm 'l1' or 'l2'.

See also
--------
Normalizer: Performs normalization using the ``Transformer`` API
    (e.g. as part of a preprocessing :class:`sklearn.pipeline.Pipeline`).

Notes
-----
For a comparison of the different scalers, transformers, and normalizers,
see :ref:`examples/preprocessing/plot_all_scaling.py
<sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
*)

val power_transform : ?method_:string -> ?standardize:bool -> ?copy:bool -> x:Ndarray.t -> unit -> Ndarray.t
(**
Power transforms are a family of parametric, monotonic transformations
that are applied to make data more Gaussian-like. This is useful for
modeling issues related to heteroscedasticity (non-constant variance),
or other situations where normality is desired.

Currently, power_transform supports the Box-Cox transform and the
Yeo-Johnson transform. The optimal parameter for stabilizing variance and
minimizing skewness is estimated through maximum likelihood.

Box-Cox requires input data to be strictly positive, while Yeo-Johnson
supports both positive or negative data.

By default, zero-mean, unit-variance normalization is applied to the
transformed data.

Read more in the :ref:`User Guide <preprocessing_transformer>`.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    The data to be transformed using a power transformation.

method : str
    The power transform method. Available methods are:

    - 'yeo-johnson' [1]_, works with positive and negative values
    - 'box-cox' [2]_, only works with strictly positive values

    The default method will be changed from 'box-cox' to 'yeo-johnson'
    in version 0.23. To suppress the FutureWarning, explicitly set the
    parameter.

standardize : boolean, default=True
    Set to True to apply zero-mean, unit-variance normalization to the
    transformed output.

copy : boolean, optional, default=True
    Set to False to perform inplace computation during transformation.

Returns
-------
X_trans : array-like, shape (n_samples, n_features)
    The transformed data.

Examples
--------
>>> import numpy as np
>>> from sklearn.preprocessing import power_transform
>>> data = [[1, 2], [3, 2], [4, 5]]
>>> print(power_transform(data, method='box-cox'))
[[-1.332... -0.707...]
 [ 0.256... -0.707...]
 [ 1.076...  1.414...]]

See also
--------
PowerTransformer : Equivalent transformation with the
    ``Transformer`` API (e.g. as part of a preprocessing
    :class:`sklearn.pipeline.Pipeline`).

quantile_transform : Maps data to a standard normal distribution with
    the parameter `output_distribution='normal'`.

Notes
-----
NaNs are treated as missing values: disregarded in ``fit``, and maintained
in ``transform``.

For a comparison of the different scalers, transformers, and normalizers,
see :ref:`examples/preprocessing/plot_all_scaling.py
<sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.

References
----------

.. [1] I.K. Yeo and R.A. Johnson, "A new family of power transformations to
       improve normality or symmetry." Biometrika, 87(4), pp.954-959,
       (2000).

.. [2] G.E.P. Box and D.R. Cox, "An Analysis of Transformations", Journal
       of the Royal Statistical Society B, 26, 211-252 (1964).
*)

val quantile_transform : ?axis:int -> ?n_quantiles:int -> ?output_distribution:string -> ?ignore_implicit_zeros:bool -> ?subsample:int -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> ?copy:bool -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> unit -> Py.Object.t
(**
Transform features using quantiles information.

This method transforms the features to follow a uniform or a normal
distribution. Therefore, for a given feature, this transformation tends
to spread out the most frequent values. It also reduces the impact of
(marginal) outliers: this is therefore a robust preprocessing scheme.

The transformation is applied on each feature independently. First an
estimate of the cumulative distribution function of a feature is
used to map the original values to a uniform distribution. The obtained
values are then mapped to the desired output distribution using the
associated quantile function. Features values of new/unseen data that fall
below or above the fitted range will be mapped to the bounds of the output
distribution. Note that this transform is non-linear. It may distort linear
correlations between variables measured at the same scale but renders
variables measured at different scales more directly comparable.

Read more in the :ref:`User Guide <preprocessing_transformer>`.

Parameters
----------
X : array-like, sparse matrix
    The data to transform.

axis : int, (default=0)
    Axis used to compute the means and standard deviations along. If 0,
    transform each feature, otherwise (if 1) transform each sample.

n_quantiles : int, optional (default=1000 or n_samples)
    Number of quantiles to be computed. It corresponds to the number
    of landmarks used to discretize the cumulative distribution function.
    If n_quantiles is larger than the number of samples, n_quantiles is set
    to the number of samples as a larger number of quantiles does not give
    a better approximation of the cumulative distribution function
    estimator.

output_distribution : str, optional (default='uniform')
    Marginal distribution for the transformed data. The choices are
    'uniform' (default) or 'normal'.

ignore_implicit_zeros : bool, optional (default=False)
    Only applies to sparse matrices. If True, the sparse entries of the
    matrix are discarded to compute the quantile statistics. If False,
    these entries are treated as zeros.

subsample : int, optional (default=1e5)
    Maximum number of samples used to estimate the quantiles for
    computational efficiency. Note that the subsampling procedure may
    differ for value-identical sparse and dense matrices.

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by np.random. Note that this is used by subsampling and smoothing
    noise.

copy : boolean, optional, (default="warn")
    Set to False to perform inplace transformation and avoid a copy (if the
    input is already a numpy array). If True, a copy of `X` is transformed,
    leaving the original `X` unchanged

    .. deprecated:: 0.21
        The default value of parameter `copy` will be changed from False
        to True in 0.23. The current default of False is being changed to
        make it more consistent with the default `copy` values of other
        functions in :mod:`sklearn.preprocessing`. Furthermore, the
        current default of False may have unexpected side effects by
        modifying the value of `X` inplace

Returns
-------
Xt : ndarray or sparse matrix, shape (n_samples, n_features)
    The transformed data.

Examples
--------
>>> import numpy as np
>>> from sklearn.preprocessing import quantile_transform
>>> rng = np.random.RandomState(0)
>>> X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)
>>> quantile_transform(X, n_quantiles=10, random_state=0, copy=True)
array([...])

See also
--------
QuantileTransformer : Performs quantile-based scaling using the
    ``Transformer`` API (e.g. as part of a preprocessing
    :class:`sklearn.pipeline.Pipeline`).
power_transform : Maps data to a normal distribution using a
    power transformation.
scale : Performs standardization that is faster, but less robust
    to outliers.
robust_scale : Performs robust standardization that removes the influence
    of outliers but does not put outliers and inliers on the same scale.

Notes
-----
NaNs are treated as missing values: disregarded in fit, and maintained in
transform.

For a comparison of the different scalers, transformers, and normalizers,
see :ref:`examples/preprocessing/plot_all_scaling.py
<sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
*)

val robust_scale : ?axis:Py.Object.t -> ?with_centering:bool -> ?with_scaling:bool -> ?quantile_range:Py.Object.t -> ?copy:bool -> x:Ndarray.t -> unit -> Py.Object.t
(**
Standardize a dataset along any axis

Center to the median and component wise scale
according to the interquartile range.

Read more in the :ref:`User Guide <preprocessing_scaler>`.

Parameters
----------
X : array-like
    The data to center and scale.

axis : int (0 by default)
    axis used to compute the medians and IQR along. If 0,
    independently scale each feature, otherwise (if 1) scale
    each sample.

with_centering : boolean, True by default
    If True, center the data before scaling.

with_scaling : boolean, True by default
    If True, scale the data to unit variance (or equivalently,
    unit standard deviation).

quantile_range : tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0
    Default: (25.0, 75.0) = (1st quantile, 3rd quantile) = IQR
    Quantile range used to calculate ``scale_``.

    .. versionadded:: 0.18

copy : boolean, optional, default is True
    set to False to perform inplace row normalization and avoid a
    copy (if the input is already a numpy array or a scipy.sparse
    CSR matrix and if axis is 1).

Notes
-----
This implementation will refuse to center scipy.sparse matrices
since it would make them non-sparse and would potentially crash the
program with memory exhaustion problems.

Instead the caller is expected to either set explicitly
`with_centering=False` (in that case, only variance scaling will be
performed on the features of the CSR matrix) or to call `X.toarray()`
if he/she expects the materialized dense array to fit in memory.

To avoid memory copy the caller should pass a CSR matrix.

For a comparison of the different scalers, transformers, and normalizers,
see :ref:`examples/preprocessing/plot_all_scaling.py
<sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.

See also
--------
RobustScaler: Performs centering and scaling using the ``Transformer`` API
    (e.g. as part of a preprocessing :class:`sklearn.pipeline.Pipeline`).
*)

val scale : ?axis:Py.Object.t -> ?with_mean:bool -> ?with_std:bool -> ?copy:bool -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> unit -> Py.Object.t
(**
Standardize a dataset along any axis

Center to the mean and component wise scale to unit variance.

Read more in the :ref:`User Guide <preprocessing_scaler>`.

Parameters
----------
X : {array-like, sparse matrix}
    The data to center and scale.

axis : int (0 by default)
    axis used to compute the means and standard deviations along. If 0,
    independently standardize each feature, otherwise (if 1) standardize
    each sample.

with_mean : boolean, True by default
    If True, center the data before scaling.

with_std : boolean, True by default
    If True, scale the data to unit variance (or equivalently,
    unit standard deviation).

copy : boolean, optional, default True
    set to False to perform inplace row normalization and avoid a
    copy (if the input is already a numpy array or a scipy.sparse
    CSC matrix and if axis is 1).

Notes
-----
This implementation will refuse to center scipy.sparse matrices
since it would make them non-sparse and would potentially crash the
program with memory exhaustion problems.

Instead the caller is expected to either set explicitly
`with_mean=False` (in that case, only variance scaling will be
performed on the features of the CSC matrix) or to call `X.toarray()`
if he/she expects the materialized dense array to fit in memory.

To avoid memory copy the caller should pass a CSC matrix.

NaNs are treated as missing values: disregarded to compute the statistics,
and maintained during the data transformation.

We use a biased estimator for the standard deviation, equivalent to
`numpy.std(x, ddof=0)`. Note that the choice of `ddof` is unlikely to
affect model performance.

For a comparison of the different scalers, transformers, and normalizers,
see :ref:`examples/preprocessing/plot_all_scaling.py
<sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.

See also
--------
StandardScaler: Performs scaling to unit variance using the``Transformer`` API
    (e.g. as part of a preprocessing :class:`sklearn.pipeline.Pipeline`).
*)

