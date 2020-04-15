module DictVectorizer : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?dtype:Py.Object.t -> ?separator:Py.Object.t -> ?sparse:Py.Object.t -> ?sort:Py.Object.t -> unit -> t
(**
Transforms lists of feature-value mappings to vectors.

This transformer turns lists of mappings (dict-like objects) of feature
names to feature values into Numpy arrays or scipy.sparse matrices for use
with scikit-learn estimators.

When feature values are strings, this transformer will do a binary one-hot
(aka one-of-K) coding: one boolean-valued feature is constructed for each
of the possible string values that the feature can take on. For instance,
a feature "f" that can take on the values "ham" and "spam" will become two
features in the output, one signifying "f=ham", the other "f=spam".

However, note that this transformer will only do a binary one-hot encoding
when feature values are of type string. If categorical features are
represented as numeric values such as int, the DictVectorizer can be
followed by :class:`sklearn.preprocessing.OneHotEncoder` to complete
binary one-hot encoding.

Features that do not occur in a sample (mapping) will have a zero value
in the resulting array/matrix.

Read more in the :ref:`User Guide <dict_feature_extraction>`.

Parameters
----------
dtype : callable, optional
    The type of feature values. Passed to Numpy array/scipy.sparse matrix
    constructors as the dtype argument.
separator : string, optional
    Separator string used when constructing new features for one-hot
    coding.
sparse : boolean, optional.
    Whether transform should produce scipy.sparse matrices.
    True by default.
sort : boolean, optional.
    Whether ``feature_names_`` and ``vocabulary_`` should be
    sorted when fitting. True by default.

Attributes
----------
vocabulary_ : dict
    A dictionary mapping feature names to feature indices.

feature_names_ : list
    A list of length n_features containing the feature names (e.g., "f=ham"
    and "f=spam").

Examples
--------
>>> from sklearn.feature_extraction import DictVectorizer
>>> v = DictVectorizer(sparse=False)
>>> D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
>>> X = v.fit_transform(D)
>>> X
array([[2., 0., 1.],
       [0., 1., 3.]])
>>> v.inverse_transform(X) ==         [{'bar': 2.0, 'foo': 1.0}, {'baz': 1.0, 'foo': 3.0}]
True
>>> v.transform({'foo': 4, 'unseen_feature': 3})
array([[0., 0., 4.]])

See also
--------
FeatureHasher : performs vectorization using only a hash function.
sklearn.preprocessing.OrdinalEncoder : handles nominal/categorical
  features encoded as columns of arbitrary data types.
*)

val fit : ?y:Py.Object.t -> x:Py.Object.t -> t -> t
(**
Learn a list of feature name -> indices mappings.

Parameters
----------
X : Mapping or iterable over Mappings
    Dict(s) or Mapping(s) from feature names (arbitrary Python
    objects) to feature values (strings or convertible to dtype).
y : (ignored)

Returns
-------
self
*)

val fit_transform : ?y:Py.Object.t -> x:Py.Object.t -> t -> Ndarray.t
(**
Learn a list of feature name -> indices mappings and transform X.

Like fit(X) followed by transform(X), but does not require
materializing X in memory.

Parameters
----------
X : Mapping or iterable over Mappings
    Dict(s) or Mapping(s) from feature names (arbitrary Python
    objects) to feature values (strings or convertible to dtype).
y : (ignored)

Returns
-------
Xa : {array, sparse matrix}
    Feature vectors; always 2-d.
*)

val get_feature_names : t -> Py.Object.t
(**
Returns a list of feature names, ordered by their indices.

If one-of-K coding is applied to categorical features, this will
include the constructed feature names but not the original ones.
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

val inverse_transform : ?dict_type:Py.Object.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Py.Object.t
(**
Transform array or sparse matrix X back to feature mappings.

X must have been produced by this DictVectorizer's transform or
fit_transform method; it may only have passed through transformers
that preserve the number of features and their order.

In the case of one-hot/one-of-K coding, the constructed feature
names and values are returned rather than the original ones.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Sample matrix.
dict_type : callable, optional
    Constructor for feature mappings. Must conform to the
    collections.Mapping API.

Returns
-------
D : list of dict_type objects, length = n_samples
    Feature mappings for the samples in X.
*)

val restrict : ?indices:Py.Object.t -> support:Ndarray.t -> t -> Py.Object.t
(**
Restrict the features to those in support using feature selection.

This function modifies the estimator in-place.

Parameters
----------
support : array-like
    Boolean mask or list of indices (as returned by the get_support
    member of feature selectors).
indices : boolean, optional
    Whether support is a list of indices.

Returns
-------
self

Examples
--------
>>> from sklearn.feature_extraction import DictVectorizer
>>> from sklearn.feature_selection import SelectKBest, chi2
>>> v = DictVectorizer()
>>> D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
>>> X = v.fit_transform(D)
>>> support = SelectKBest(chi2, k=2).fit(X, [0, 1])
>>> v.get_feature_names()
['bar', 'baz', 'foo']
>>> v.restrict(support.get_support())
DictVectorizer()
>>> v.get_feature_names()
['bar', 'foo']
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

val transform : x:Py.Object.t -> t -> Ndarray.t
(**
Transform feature->value dicts to array or sparse matrix.

Named features not encountered during fit or fit_transform will be
silently ignored.

Parameters
----------
X : Mapping or iterable over Mappings, length = n_samples
    Dict(s) or Mapping(s) from feature names (arbitrary Python
    objects) to feature values (strings or convertible to dtype).

Returns
-------
Xa : {array, sparse matrix}
    Feature vectors; always 2-d.
*)


(** Attribute vocabulary_: see constructor for documentation *)
val vocabulary_ : t -> Py.Object.t

(** Attribute feature_names_: see constructor for documentation *)
val feature_names_ : t -> Py.Object.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module FeatureHasher : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_features:int -> ?input_type:Py.Object.t -> ?dtype:Py.Object.t -> ?alternate_sign:Py.Object.t -> unit -> t
(**
Implements feature hashing, aka the hashing trick.

This class turns sequences of symbolic feature names (strings) into
scipy.sparse matrices, using a hash function to compute the matrix column
corresponding to a name. The hash function employed is the signed 32-bit
version of Murmurhash3.

Feature names of type byte string are used as-is. Unicode strings are
converted to UTF-8 first, but no Unicode normalization is done.
Feature values must be (finite) numbers.

This class is a low-memory alternative to DictVectorizer and
CountVectorizer, intended for large-scale (online) learning and situations
where memory is tight, e.g. when running prediction code on embedded
devices.

Read more in the :ref:`User Guide <feature_hashing>`.

.. versionadded:: 0.13

Parameters
----------
n_features : integer, optional
    The number of features (columns) in the output matrices. Small numbers
    of features are likely to cause hash collisions, but large numbers
    will cause larger coefficient dimensions in linear learners.
input_type : string, optional, default "dict"
    Either "dict" (the default) to accept dictionaries over
    (feature_name, value); "pair" to accept pairs of (feature_name, value);
    or "string" to accept single strings.
    feature_name should be a string, while value should be a number.
    In the case of "string", a value of 1 is implied.
    The feature_name is hashed to find the appropriate column for the
    feature. The value's sign might be flipped in the output (but see
    non_negative, below).
dtype : numpy type, optional, default np.float64
    The type of feature values. Passed to scipy.sparse matrix constructors
    as the dtype argument. Do not set this to bool, np.boolean or any
    unsigned integer type.
alternate_sign : boolean, optional, default True
    When True, an alternating sign is added to the features as to
    approximately conserve the inner product in the hashed space even for
    small n_features. This approach is similar to sparse random projection.

Examples
--------
>>> from sklearn.feature_extraction import FeatureHasher
>>> h = FeatureHasher(n_features=10)
>>> D = [{'dog': 1, 'cat':2, 'elephant':4},{'dog': 2, 'run': 5}]
>>> f = h.transform(D)
>>> f.toarray()
array([[ 0.,  0., -4., -1.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 0.,  0.,  0., -2., -5.,  0.,  0.,  0.,  0.,  0.]])

See also
--------
DictVectorizer : vectorizes string-valued features using a hash table.
sklearn.preprocessing.OneHotEncoder : handles nominal/categorical features.
*)

val fit : ?x:Ndarray.t -> ?y:Py.Object.t -> t -> t
(**
No-op.

This method doesn't do anything. It exists purely for compatibility
with the scikit-learn transformer API.

Parameters
----------
X : array-like

Returns
-------
self : FeatureHasher
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

val transform : raw_X:Py.Object.t -> t -> Ndarray.t
(**
Transform a sequence of instances to a scipy.sparse matrix.

Parameters
----------
raw_X : iterable over iterable over raw features, length = n_samples
    Samples. Each sample must be iterable an (e.g., a list or tuple)
    containing/generating feature names (and optionally values, see
    the input_type constructor argument) which will be hashed.
    raw_X need not support the len function, so it can be the result
    of a generator; n_samples is determined on the fly.

Returns
-------
X : sparse matrix of shape (n_samples, n_features)
    Feature matrix, for use with estimators or further transformers.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val grid_to_graph : ?n_z:Py.Object.t -> ?mask:Py.Object.t -> ?return_as:Py.Object.t -> ?dtype:Py.Object.t -> n_x:int -> n_y:Py.Object.t -> unit -> Py.Object.t
(**
Graph of the pixel-to-pixel connections

Edges exist if 2 voxels are connected.

Parameters
----------
n_x : int
    Dimension in x axis
n_y : int
    Dimension in y axis
n_z : int, optional, default 1
    Dimension in z axis
mask : ndarray of booleans, optional
    An optional mask of the image, to consider only part of the
    pixels.
return_as : np.ndarray or a sparse matrix class, optional
    The class to use to build the returned adjacency matrix.
dtype : dtype, optional, default int
    The data of the returned sparse matrix. By default it is int

Notes
-----
For scikit-learn versions 0.14.1 and prior, return_as=np.ndarray was
handled by returning a dense np.matrix instance.  Going forward, np.ndarray
returns an np.ndarray, as expected.

For compatibility, user code relying on this method should wrap its
calls in ``np.asarray`` to avoid type issues.
*)

module Image : sig
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

module PatchExtractor : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?patch_size:Py.Object.t -> ?max_patches:[`Int of int | `Float of float] -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> unit -> t
(**
Extracts patches from a collection of images

Read more in the :ref:`User Guide <image_feature_extraction>`.

.. versionadded:: 0.9

Parameters
----------
patch_size : tuple of ints (patch_height, patch_width)
    the dimensions of one patch

max_patches : integer or float, optional default is None
    The maximum number of patches per image to extract. If max_patches is a
    float in (0, 1), it is taken to mean a proportion of the total number
    of patches.

random_state : int, RandomState instance or None, optional (default=None)
    Determines the random number generator used for random sampling when
    `max_patches` is not None. Use an int to make the randomness
    deterministic.
    See :term:`Glossary <random_state>`.


Examples
--------
>>> from sklearn.datasets import load_sample_images
>>> from sklearn.feature_extraction import image
>>> # Use the array data from the second image in this dataset:
>>> X = load_sample_images().images[1]
>>> print('Image shape: {}'.format(X.shape))
Image shape: (427, 640, 3)
>>> pe = image.PatchExtractor(patch_size=(2, 2))
>>> pe_fit = pe.fit(X)
>>> pe_trans = pe.transform(X)
>>> print('Patches shape: {}'.format(pe_trans.shape))
Patches shape: (545706, 2, 2)
*)

val fit : ?y:Py.Object.t -> x:Ndarray.t -> t -> t
(**
Do nothing and return the estimator unchanged

This method is just there to implement the usual API and hence
work in pipelines.

Parameters
----------
X : array-like, shape [n_samples, n_features]
    Training data.
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
Transforms the image samples in X into a matrix of patch data.

Parameters
----------
X : array, shape = (n_samples, image_height, image_width) or
    (n_samples, image_height, image_width, n_channels)
    Array of images from which to extract patches. For color images,
    the last dimension specifies the channel: a RGB image would have
    `n_channels=3`.

Returns
-------
patches : array, shape = (n_patches, patch_height, patch_width) or
     (n_patches, patch_height, patch_width, n_channels)
     The collection of patches extracted from the images, where
     `n_patches` is either `n_samples * max_patches` or the total
     number of patches that can be extracted.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val as_strided : ?shape:int list -> ?strides:Py.Object.t -> ?subok:Py.Object.t -> ?writeable:bool -> x:Ndarray.t -> unit -> Ndarray.t
(**
Create a view into the array with the given shape and strides.

.. warning:: This function has to be used with extreme care, see notes.

Parameters
----------
x : ndarray
    Array to create a new.
shape : sequence of int, optional
    The shape of the new array. Defaults to ``x.shape``.
strides : sequence of int, optional
    The strides of the new array. Defaults to ``x.strides``.
subok : bool, optional
    .. versionadded:: 1.10

    If True, subclasses are preserved.
writeable : bool, optional
    .. versionadded:: 1.12

    If set to False, the returned array will always be readonly.
    Otherwise it will be writable if the original array was. It
    is advisable to set this to False if possible (see Notes).

Returns
-------
view : ndarray

See also
--------
broadcast_to: broadcast an array to a given shape.
reshape : reshape an array.

Notes
-----
``as_strided`` creates a view into the array given the exact strides
and shape. This means it manipulates the internal data structure of
ndarray and, if done incorrectly, the array elements can point to
invalid memory and can corrupt results or crash your program.
It is advisable to always use the original ``x.strides`` when
calculating new strides to avoid reliance on a contiguous memory
layout.

Furthermore, arrays created with this function often contain self
overlapping memory, so that two elements are identical.
Vectorized write operations on such arrays will typically be
unpredictable. They may even give different results for small, large,
or transposed arrays.
Since writing to these arrays has to be tested and done with great
care, you may want to use ``writeable=False`` to avoid accidental write
operations.

For these reasons it is advisable to avoid ``as_strided`` when
possible.
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

val extract_patches : ?patch_shape:[`Int of int | `PyObject of Py.Object.t] -> ?extraction_step:[`Int of int | `PyObject of Py.Object.t] -> arr:Ndarray.t -> unit -> Py.Object.t
(**
DEPRECATED: The function feature_extraction.image.extract_patches has been deprecated in 0.22 and will be removed in 0.24.

Extracts patches of any n-dimensional array in place using strides.

    Given an n-dimensional array it will return a 2n-dimensional array with
    the first n dimensions indexing patch position and the last n indexing
    the patch content. This operation is immediate (O(1)). A reshape
    performed on the first n dimensions will cause numpy to copy data, leading
    to a list of extracted patches.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    arr : ndarray
        n-dimensional array of which patches are to be extracted

    patch_shape : integer or tuple of length arr.ndim
        Indicates the shape of the patches to be extracted. If an
        integer is given, the shape will be a hypercube of
        sidelength given by its value.

    extraction_step : integer or tuple of length arr.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.


    Returns
    -------
    patches : strided ndarray
        2n-dimensional array indexing patches on first n dimensions and
        containing patches on the last n dimensions. These dimensions
        are fake, but this way no data is copied. A simple reshape invokes
        a copying operation to obtain a list of patches:
        result.reshape([-1] + list(patch_shape))
    
*)

val extract_patches_2d : ?max_patches:[`Int of int | `Float of float] -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> image:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> patch_size:Py.Object.t -> unit -> Py.Object.t
(**
Reshape a 2D image into a collection of patches

The resulting patches are allocated in a dedicated array.

Read more in the :ref:`User Guide <image_feature_extraction>`.

Parameters
----------
image : array, shape = (image_height, image_width) or
    (image_height, image_width, n_channels)
    The original image data. For color images, the last dimension specifies
    the channel: a RGB image would have `n_channels=3`.

patch_size : tuple of ints (patch_height, patch_width)
    the dimensions of one patch

max_patches : integer or float, optional default is None
    The maximum number of patches to extract. If max_patches is a float
    between 0 and 1, it is taken to be a proportion of the total number
    of patches.

random_state : int, RandomState instance or None, optional (default=None)
    Determines the random number generator used for random sampling when
    `max_patches` is not None. Use an int to make the randomness
    deterministic.
    See :term:`Glossary <random_state>`.

Returns
-------
patches : array, shape = (n_patches, patch_height, patch_width) or
    (n_patches, patch_height, patch_width, n_channels)
    The collection of patches extracted from the image, where `n_patches`
    is either `max_patches` or the total number of patches that can be
    extracted.

Examples
--------
>>> from sklearn.datasets import load_sample_image
>>> from sklearn.feature_extraction import image
>>> # Use the array data from the first image in this dataset:
>>> one_image = load_sample_image("china.jpg")
>>> print('Image shape: {}'.format(one_image.shape))
Image shape: (427, 640, 3)
>>> patches = image.extract_patches_2d(one_image, (2, 2))
>>> print('Patches shape: {}'.format(patches.shape))
Patches shape: (272214, 2, 2, 3)
>>> # Here are just two of these patches:
>>> print(patches[1])
[[[174 201 231]
  [174 201 231]]
 [[173 200 230]
  [173 200 230]]]
>>> print(patches[800])
[[[187 214 243]
  [188 215 244]]
 [[187 214 243]
  [188 215 244]]]
*)

val grid_to_graph : ?n_z:Py.Object.t -> ?mask:Py.Object.t -> ?return_as:Py.Object.t -> ?dtype:Py.Object.t -> n_x:int -> n_y:Py.Object.t -> unit -> Py.Object.t
(**
Graph of the pixel-to-pixel connections

Edges exist if 2 voxels are connected.

Parameters
----------
n_x : int
    Dimension in x axis
n_y : int
    Dimension in y axis
n_z : int, optional, default 1
    Dimension in z axis
mask : ndarray of booleans, optional
    An optional mask of the image, to consider only part of the
    pixels.
return_as : np.ndarray or a sparse matrix class, optional
    The class to use to build the returned adjacency matrix.
dtype : dtype, optional, default int
    The data of the returned sparse matrix. By default it is int

Notes
-----
For scikit-learn versions 0.14.1 and prior, return_as=np.ndarray was
handled by returning a dense np.matrix instance.  Going forward, np.ndarray
returns an np.ndarray, as expected.

For compatibility, user code relying on this method should wrap its
calls in ``np.asarray`` to avoid type issues.
*)

val img_to_graph : ?mask:Py.Object.t -> ?return_as:Py.Object.t -> ?dtype:Py.Object.t -> img:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> unit -> Py.Object.t
(**
Graph of the pixel-to-pixel gradient connections

Edges are weighted with the gradient values.

Read more in the :ref:`User Guide <image_feature_extraction>`.

Parameters
----------
img : ndarray, 2D or 3D
    2D or 3D image
mask : ndarray of booleans, optional
    An optional mask of the image, to consider only part of the
    pixels.
return_as : np.ndarray or a sparse matrix class, optional
    The class to use to build the returned adjacency matrix.
dtype : None or dtype, optional
    The data of the returned sparse matrix. By default it is the
    dtype of img

Notes
-----
For scikit-learn versions 0.14.1 and prior, return_as=np.ndarray was
handled by returning a dense np.matrix instance.  Going forward, np.ndarray
returns an np.ndarray, as expected.

For compatibility, user code relying on this method should wrap its
calls in ``np.asarray`` to avoid type issues.
*)

val reconstruct_from_patches_2d : patches:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> image_size:Py.Object.t -> unit -> Py.Object.t
(**
Reconstruct the image from all of its patches.

Patches are assumed to overlap and the image is constructed by filling in
the patches from left to right, top to bottom, averaging the overlapping
regions.

Read more in the :ref:`User Guide <image_feature_extraction>`.

Parameters
----------
patches : array, shape = (n_patches, patch_height, patch_width) or
    (n_patches, patch_height, patch_width, n_channels)
    The complete set of patches. If the patches contain colour information,
    channels are indexed along the last dimension: RGB patches would
    have `n_channels=3`.

image_size : tuple of ints (image_height, image_width) or
    (image_height, image_width, n_channels)
    the size of the image that will be reconstructed

Returns
-------
image : array, shape = image_size
    the reconstructed image
*)


end

val img_to_graph : ?mask:Py.Object.t -> ?return_as:Py.Object.t -> ?dtype:Py.Object.t -> img:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> unit -> Py.Object.t
(**
Graph of the pixel-to-pixel gradient connections

Edges are weighted with the gradient values.

Read more in the :ref:`User Guide <image_feature_extraction>`.

Parameters
----------
img : ndarray, 2D or 3D
    2D or 3D image
mask : ndarray of booleans, optional
    An optional mask of the image, to consider only part of the
    pixels.
return_as : np.ndarray or a sparse matrix class, optional
    The class to use to build the returned adjacency matrix.
dtype : None or dtype, optional
    The data of the returned sparse matrix. By default it is the
    dtype of img

Notes
-----
For scikit-learn versions 0.14.1 and prior, return_as=np.ndarray was
handled by returning a dense np.matrix instance.  Going forward, np.ndarray
returns an np.ndarray, as expected.

For compatibility, user code relying on this method should wrap its
calls in ``np.asarray`` to avoid type issues.
*)

module Text : sig
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

module CountVectorizer : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?input:[`Filename | `File | `Content] -> ?encoding:[`String of string | `PyObject of Py.Object.t] -> ?decode_error:[`Strict | `Ignore | `Replace] -> ?strip_accents:[`Ascii | `Unicode | `None] -> ?lowercase:bool -> ?preprocessor:[`Callable of Py.Object.t | `None] -> ?tokenizer:[`Callable of Py.Object.t | `None] -> ?stop_words:[`English | `ArrayLike of Py.Object.t | `None] -> ?token_pattern:string -> ?ngram_range:Py.Object.t -> ?analyzer:[`String of string | `Word | `Char | `Char_wb | `Callable of Py.Object.t] -> ?max_df:[`Int of int | `PyObject of Py.Object.t] -> ?min_df:[`Int of int | `PyObject of Py.Object.t] -> ?max_features:[`Int of int | `None] -> ?vocabulary:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> ?binary:bool -> ?dtype:Py.Object.t -> unit -> t
(**
Convert a collection of text documents to a matrix of token counts

This implementation produces a sparse representation of the counts using
scipy.sparse.csr_matrix.

If you do not provide an a-priori dictionary and you do not use an analyzer
that does some kind of feature selection then the number of features will
be equal to the vocabulary size found by analyzing the data.

Read more in the :ref:`User Guide <text_feature_extraction>`.

Parameters
----------
input : string {'filename', 'file', 'content'}
    If 'filename', the sequence passed as an argument to fit is
    expected to be a list of filenames that need reading to fetch
    the raw content to analyze.

    If 'file', the sequence items must have a 'read' method (file-like
    object) that is called to fetch the bytes in memory.

    Otherwise the input is expected to be a sequence of items that
    can be of type string or byte.

encoding : string, 'utf-8' by default.
    If bytes or files are given to analyze, this encoding is used to
    decode.

decode_error : {'strict', 'ignore', 'replace'}
    Instruction on what to do if a byte sequence is given to analyze that
    contains characters not of the given `encoding`. By default, it is
    'strict', meaning that a UnicodeDecodeError will be raised. Other
    values are 'ignore' and 'replace'.

strip_accents : {'ascii', 'unicode', None}
    Remove accents and perform other character normalization
    during the preprocessing step.
    'ascii' is a fast method that only works on characters that have
    an direct ASCII mapping.
    'unicode' is a slightly slower method that works on any characters.
    None (default) does nothing.

    Both 'ascii' and 'unicode' use NFKD normalization from
    :func:`unicodedata.normalize`.

lowercase : boolean, True by default
    Convert all characters to lowercase before tokenizing.

preprocessor : callable or None (default)
    Override the preprocessing (string transformation) stage while
    preserving the tokenizing and n-grams generation steps.
    Only applies if ``analyzer is not callable``.

tokenizer : callable or None (default)
    Override the string tokenization step while preserving the
    preprocessing and n-grams generation steps.
    Only applies if ``analyzer == 'word'``.

stop_words : string {'english'}, list, or None (default)
    If 'english', a built-in stop word list for English is used.
    There are several known issues with 'english' and you should
    consider an alternative (see :ref:`stop_words`).

    If a list, that list is assumed to contain stop words, all of which
    will be removed from the resulting tokens.
    Only applies if ``analyzer == 'word'``.

    If None, no stop words will be used. max_df can be set to a value
    in the range [0.7, 1.0) to automatically detect and filter stop
    words based on intra corpus document frequency of terms.

token_pattern : string
    Regular expression denoting what constitutes a "token", only used
    if ``analyzer == 'word'``. The default regexp select tokens of 2
    or more alphanumeric characters (punctuation is completely ignored
    and always treated as a token separator).

ngram_range : tuple (min_n, max_n), default=(1, 1)
    The lower and upper boundary of the range of n-values for different
    word n-grams or char n-grams to be extracted. All values of n such
    such that min_n <= n <= max_n will be used. For example an
    ``ngram_range`` of ``(1, 1)`` means only unigrams, ``(1, 2)`` means
    unigrams and bigrams, and ``(2, 2)`` means only bigrams.
    Only applies if ``analyzer is not callable``.

analyzer : string, {'word', 'char', 'char_wb'} or callable
    Whether the feature should be made of word n-gram or character
    n-grams.
    Option 'char_wb' creates character n-grams only from text inside
    word boundaries; n-grams at the edges of words are padded with space.

    If a callable is passed it is used to extract the sequence of features
    out of the raw, unprocessed input.

    .. versionchanged:: 0.21

    Since v0.21, if ``input`` is ``filename`` or ``file``, the data is
    first read from the file and then passed to the given callable
    analyzer.

max_df : float in range [0.0, 1.0] or int, default=1.0
    When building the vocabulary ignore terms that have a document
    frequency strictly higher than the given threshold (corpus-specific
    stop words).
    If float, the parameter represents a proportion of documents, integer
    absolute counts.
    This parameter is ignored if vocabulary is not None.

min_df : float in range [0.0, 1.0] or int, default=1
    When building the vocabulary ignore terms that have a document
    frequency strictly lower than the given threshold. This value is also
    called cut-off in the literature.
    If float, the parameter represents a proportion of documents, integer
    absolute counts.
    This parameter is ignored if vocabulary is not None.

max_features : int or None, default=None
    If not None, build a vocabulary that only consider the top
    max_features ordered by term frequency across the corpus.

    This parameter is ignored if vocabulary is not None.

vocabulary : Mapping or iterable, optional
    Either a Mapping (e.g., a dict) where keys are terms and values are
    indices in the feature matrix, or an iterable over terms. If not
    given, a vocabulary is determined from the input documents. Indices
    in the mapping should not be repeated and should not have any gap
    between 0 and the largest index.

binary : boolean, default=False
    If True, all non zero counts are set to 1. This is useful for discrete
    probabilistic models that model binary events rather than integer
    counts.

dtype : type, optional
    Type of the matrix returned by fit_transform() or transform().

Attributes
----------
vocabulary_ : dict
    A mapping of terms to feature indices.

fixed_vocabulary_: boolean
    True if a fixed vocabulary of term to indices mapping
    is provided by the user

stop_words_ : set
    Terms that were ignored because they either:

      - occurred in too many documents (`max_df`)
      - occurred in too few documents (`min_df`)
      - were cut off by feature selection (`max_features`).

    This is only available if no vocabulary was given.

Examples
--------
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> corpus = [
...     'This is the first document.',
...     'This document is the second document.',
...     'And this is the third one.',
...     'Is this the first document?',
... ]
>>> vectorizer = CountVectorizer()
>>> X = vectorizer.fit_transform(corpus)
>>> print(vectorizer.get_feature_names())
['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
>>> print(X.toarray())
[[0 1 1 1 0 0 1 0 1]
 [0 2 0 1 0 1 1 0 1]
 [1 0 0 1 1 0 1 1 1]
 [0 1 1 1 0 0 1 0 1]]
>>> vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
>>> X2 = vectorizer2.fit_transform(corpus)
>>> print(vectorizer2.get_feature_names())
['and this', 'document is', 'first document', 'is the', 'is this',
'second document', 'the first', 'the second', 'the third', 'third one',
 'this document', 'this is', 'this the']
 >>> print(X2.toarray())
 [[0 0 1 1 0 0 1 0 0 0 0 1 0]
 [0 1 0 1 0 1 0 1 0 0 1 0 0]
 [1 0 0 1 0 0 0 0 1 1 0 1 0]
 [0 0 1 0 1 0 1 0 0 0 0 0 1]]

See Also
--------
HashingVectorizer, TfidfVectorizer

Notes
-----
The ``stop_words_`` attribute can get large and increase the model size
when pickling. This attribute is provided only for introspection and can
be safely removed using delattr or set to None before pickling.
*)

val build_analyzer : t -> Py.Object.t
(**
Return a callable that handles preprocessing, tokenization
and n-grams generation.

Returns
-------
analyzer: callable
    A function to handle preprocessing, tokenization
    and n-grams generation.
*)

val build_preprocessor : t -> Py.Object.t
(**
Return a function to preprocess the text before tokenization.

Returns
-------
preprocessor: callable
      A function to preprocess the text before tokenization.
*)

val build_tokenizer : t -> Py.Object.t
(**
Return a function that splits a string into a sequence of tokens.

Returns
-------
tokenizer: callable
      A function to split a string into a sequence of tokens.
*)

val decode : doc:string -> t -> string
(**
Decode the input into a string of unicode symbols.

The decoding strategy depends on the vectorizer parameters.

Parameters
----------
doc : str
    The string to decode.

Returns
-------
doc: str
    A string of unicode symbols.
*)

val fit : ?y:Py.Object.t -> raw_documents:Ndarray.t -> t -> t
(**
Learn a vocabulary dictionary of all tokens in the raw documents.

Parameters
----------
raw_documents : iterable
    An iterable which yields either str, unicode or file objects.

Returns
-------
self
*)

val fit_transform : ?y:Py.Object.t -> raw_documents:Ndarray.t -> t -> Ndarray.t
(**
Learn the vocabulary dictionary and return term-document matrix.

This is equivalent to fit followed by transform, but more efficiently
implemented.

Parameters
----------
raw_documents : iterable
    An iterable which yields either str, unicode or file objects.

Returns
-------
X : array, [n_samples, n_features]
    Document-term matrix.
*)

val get_feature_names : t -> string list
(**
Array mapping from feature integer indices to feature name.

Returns
-------
feature_names : list
    A list of feature names.
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

val get_stop_words : t -> Py.Object.t
(**
Build or fetch the effective stop words list.

Returns
-------
stop_words: list or None
        A list of stop words.
*)

val inverse_transform : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Py.Object.t
(**
Return terms per document with nonzero entries in X.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Document-term matrix.

Returns
-------
X_inv : list of arrays, len = n_samples
    List of arrays of terms.
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

val transform : raw_documents:Ndarray.t -> t -> Ndarray.t
(**
Transform documents to document-term matrix.

Extract token counts out of raw text documents using the vocabulary
fitted with fit or the one provided to the constructor.

Parameters
----------
raw_documents : iterable
    An iterable which yields either str, unicode or file objects.

Returns
-------
X : sparse matrix, [n_samples, n_features]
    Document-term matrix.
*)


(** Attribute vocabulary_: see constructor for documentation *)
val vocabulary_ : t -> Py.Object.t

(** Attribute fixed_vocabulary_: see constructor for documentation *)
val fixed_vocabulary_ : t -> bool

(** Attribute stop_words_: see constructor for documentation *)
val stop_words_ : t -> Py.Object.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module FeatureHasher : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_features:int -> ?input_type:Py.Object.t -> ?dtype:Py.Object.t -> ?alternate_sign:Py.Object.t -> unit -> t
(**
Implements feature hashing, aka the hashing trick.

This class turns sequences of symbolic feature names (strings) into
scipy.sparse matrices, using a hash function to compute the matrix column
corresponding to a name. The hash function employed is the signed 32-bit
version of Murmurhash3.

Feature names of type byte string are used as-is. Unicode strings are
converted to UTF-8 first, but no Unicode normalization is done.
Feature values must be (finite) numbers.

This class is a low-memory alternative to DictVectorizer and
CountVectorizer, intended for large-scale (online) learning and situations
where memory is tight, e.g. when running prediction code on embedded
devices.

Read more in the :ref:`User Guide <feature_hashing>`.

.. versionadded:: 0.13

Parameters
----------
n_features : integer, optional
    The number of features (columns) in the output matrices. Small numbers
    of features are likely to cause hash collisions, but large numbers
    will cause larger coefficient dimensions in linear learners.
input_type : string, optional, default "dict"
    Either "dict" (the default) to accept dictionaries over
    (feature_name, value); "pair" to accept pairs of (feature_name, value);
    or "string" to accept single strings.
    feature_name should be a string, while value should be a number.
    In the case of "string", a value of 1 is implied.
    The feature_name is hashed to find the appropriate column for the
    feature. The value's sign might be flipped in the output (but see
    non_negative, below).
dtype : numpy type, optional, default np.float64
    The type of feature values. Passed to scipy.sparse matrix constructors
    as the dtype argument. Do not set this to bool, np.boolean or any
    unsigned integer type.
alternate_sign : boolean, optional, default True
    When True, an alternating sign is added to the features as to
    approximately conserve the inner product in the hashed space even for
    small n_features. This approach is similar to sparse random projection.

Examples
--------
>>> from sklearn.feature_extraction import FeatureHasher
>>> h = FeatureHasher(n_features=10)
>>> D = [{'dog': 1, 'cat':2, 'elephant':4},{'dog': 2, 'run': 5}]
>>> f = h.transform(D)
>>> f.toarray()
array([[ 0.,  0., -4., -1.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 0.,  0.,  0., -2., -5.,  0.,  0.,  0.,  0.,  0.]])

See also
--------
DictVectorizer : vectorizes string-valued features using a hash table.
sklearn.preprocessing.OneHotEncoder : handles nominal/categorical features.
*)

val fit : ?x:Ndarray.t -> ?y:Py.Object.t -> t -> t
(**
No-op.

This method doesn't do anything. It exists purely for compatibility
with the scikit-learn transformer API.

Parameters
----------
X : array-like

Returns
-------
self : FeatureHasher
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

val transform : raw_X:Py.Object.t -> t -> Ndarray.t
(**
Transform a sequence of instances to a scipy.sparse matrix.

Parameters
----------
raw_X : iterable over iterable over raw features, length = n_samples
    Samples. Each sample must be iterable an (e.g., a list or tuple)
    containing/generating feature names (and optionally values, see
    the input_type constructor argument) which will be hashed.
    raw_X need not support the len function, so it can be the result
    of a generator; n_samples is determined on the fly.

Returns
-------
X : sparse matrix of shape (n_samples, n_features)
    Feature matrix, for use with estimators or further transformers.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module HashingVectorizer : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?input:[`Filename | `File | `Content] -> ?encoding:string -> ?decode_error:[`Strict | `Ignore | `Replace] -> ?strip_accents:[`Ascii | `Unicode | `None] -> ?lowercase:bool -> ?preprocessor:[`Callable of Py.Object.t | `None] -> ?tokenizer:[`Callable of Py.Object.t | `None] -> ?stop_words:[`English | `ArrayLike of Py.Object.t | `None] -> ?token_pattern:string -> ?ngram_range:Py.Object.t -> ?analyzer:[`String of string | `Word | `Char | `Char_wb | `Callable of Py.Object.t] -> ?n_features:int -> ?binary:bool -> ?norm:[`L1 | `L2 | `None] -> ?alternate_sign:bool -> ?dtype:Py.Object.t -> unit -> t
(**
Convert a collection of text documents to a matrix of token occurrences

It turns a collection of text documents into a scipy.sparse matrix holding
token occurrence counts (or binary occurrence information), possibly
normalized as token frequencies if norm='l1' or projected on the euclidean
unit sphere if norm='l2'.

This text vectorizer implementation uses the hashing trick to find the
token string name to feature integer index mapping.

This strategy has several advantages:

- it is very low memory scalable to large datasets as there is no need to
  store a vocabulary dictionary in memory

- it is fast to pickle and un-pickle as it holds no state besides the
  constructor parameters

- it can be used in a streaming (partial fit) or parallel pipeline as there
  is no state computed during fit.

There are also a couple of cons (vs using a CountVectorizer with an
in-memory vocabulary):

- there is no way to compute the inverse transform (from feature indices to
  string feature names) which can be a problem when trying to introspect
  which features are most important to a model.

- there can be collisions: distinct tokens can be mapped to the same
  feature index. However in practice this is rarely an issue if n_features
  is large enough (e.g. 2 ** 18 for text classification problems).

- no IDF weighting as this would render the transformer stateful.

The hash function employed is the signed 32-bit version of Murmurhash3.

Read more in the :ref:`User Guide <text_feature_extraction>`.

Parameters
----------

input : string {'filename', 'file', 'content'}
    If 'filename', the sequence passed as an argument to fit is
    expected to be a list of filenames that need reading to fetch
    the raw content to analyze.

    If 'file', the sequence items must have a 'read' method (file-like
    object) that is called to fetch the bytes in memory.

    Otherwise the input is expected to be a sequence of items that
    can be of type string or byte.

encoding : string, default='utf-8'
    If bytes or files are given to analyze, this encoding is used to
    decode.

decode_error : {'strict', 'ignore', 'replace'}
    Instruction on what to do if a byte sequence is given to analyze that
    contains characters not of the given `encoding`. By default, it is
    'strict', meaning that a UnicodeDecodeError will be raised. Other
    values are 'ignore' and 'replace'.

strip_accents : {'ascii', 'unicode', None}
    Remove accents and perform other character normalization
    during the preprocessing step.
    'ascii' is a fast method that only works on characters that have
    an direct ASCII mapping.
    'unicode' is a slightly slower method that works on any characters.
    None (default) does nothing.

    Both 'ascii' and 'unicode' use NFKD normalization from
    :func:`unicodedata.normalize`.

lowercase : boolean, default=True
    Convert all characters to lowercase before tokenizing.

preprocessor : callable or None (default)
    Override the preprocessing (string transformation) stage while
    preserving the tokenizing and n-grams generation steps.
    Only applies if ``analyzer is not callable``.

tokenizer : callable or None (default)
    Override the string tokenization step while preserving the
    preprocessing and n-grams generation steps.
    Only applies if ``analyzer == 'word'``.

stop_words : string {'english'}, list, or None (default)
    If 'english', a built-in stop word list for English is used.
    There are several known issues with 'english' and you should
    consider an alternative (see :ref:`stop_words`).

    If a list, that list is assumed to contain stop words, all of which
    will be removed from the resulting tokens.
    Only applies if ``analyzer == 'word'``.

token_pattern : string
    Regular expression denoting what constitutes a "token", only used
    if ``analyzer == 'word'``. The default regexp selects tokens of 2
    or more alphanumeric characters (punctuation is completely ignored
    and always treated as a token separator).

ngram_range : tuple (min_n, max_n), default=(1, 1)
    The lower and upper boundary of the range of n-values for different
    n-grams to be extracted. All values of n such that min_n <= n <= max_n
    will be used. For example an ``ngram_range`` of ``(1, 1)`` means only
    unigrams, ``(1, 2)`` means unigrams and bigrams, and ``(2, 2)`` means
    only bigrams.
    Only applies if ``analyzer is not callable``.

analyzer : string, {'word', 'char', 'char_wb'} or callable
    Whether the feature should be made of word or character n-grams.
    Option 'char_wb' creates character n-grams only from text inside
    word boundaries; n-grams at the edges of words are padded with space.

    If a callable is passed it is used to extract the sequence of features
    out of the raw, unprocessed input.

    .. versionchanged:: 0.21

    Since v0.21, if ``input`` is ``filename`` or ``file``, the data is
    first read from the file and then passed to the given callable
    analyzer.

n_features : integer, default=(2 ** 20)
    The number of features (columns) in the output matrices. Small numbers
    of features are likely to cause hash collisions, but large numbers
    will cause larger coefficient dimensions in linear learners.

binary : boolean, default=False.
    If True, all non zero counts are set to 1. This is useful for discrete
    probabilistic models that model binary events rather than integer
    counts.

norm : 'l1', 'l2' or None, optional
    Norm used to normalize term vectors. None for no normalization.

alternate_sign : boolean, optional, default True
    When True, an alternating sign is added to the features as to
    approximately conserve the inner product in the hashed space even for
    small n_features. This approach is similar to sparse random projection.

    .. versionadded:: 0.19

dtype : type, optional
    Type of the matrix returned by fit_transform() or transform().

Examples
--------
>>> from sklearn.feature_extraction.text import HashingVectorizer
>>> corpus = [
...     'This is the first document.',
...     'This document is the second document.',
...     'And this is the third one.',
...     'Is this the first document?',
... ]
>>> vectorizer = HashingVectorizer(n_features=2**4)
>>> X = vectorizer.fit_transform(corpus)
>>> print(X.shape)
(4, 16)

See Also
--------
CountVectorizer, TfidfVectorizer
*)

val build_analyzer : t -> Py.Object.t
(**
Return a callable that handles preprocessing, tokenization
and n-grams generation.

Returns
-------
analyzer: callable
    A function to handle preprocessing, tokenization
    and n-grams generation.
*)

val build_preprocessor : t -> Py.Object.t
(**
Return a function to preprocess the text before tokenization.

Returns
-------
preprocessor: callable
      A function to preprocess the text before tokenization.
*)

val build_tokenizer : t -> Py.Object.t
(**
Return a function that splits a string into a sequence of tokens.

Returns
-------
tokenizer: callable
      A function to split a string into a sequence of tokens.
*)

val decode : doc:string -> t -> string
(**
Decode the input into a string of unicode symbols.

The decoding strategy depends on the vectorizer parameters.

Parameters
----------
doc : str
    The string to decode.

Returns
-------
doc: str
    A string of unicode symbols.
*)

val fit : ?y:Py.Object.t -> x:Ndarray.t -> t -> t
(**
Does nothing: this transformer is stateless.

Parameters
----------
X : array-like, shape [n_samples, n_features]
    Training data.
*)

val fit_transform : ?y:Py.Object.t -> x:Py.Object.t -> t -> Ndarray.t
(**
Transform a sequence of documents to a document-term matrix.

Parameters
----------
X : iterable over raw text documents, length = n_samples
    Samples. Each sample must be a text document (either bytes or
    unicode strings, file name or file object depending on the
    constructor argument) which will be tokenized and hashed.
y : any
    Ignored. This parameter exists only for compatibility with
    sklearn.pipeline.Pipeline.

Returns
-------
X : sparse matrix of shape (n_samples, n_features)
    Document-term matrix.
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

val get_stop_words : t -> Py.Object.t
(**
Build or fetch the effective stop words list.

Returns
-------
stop_words: list or None
        A list of stop words.
*)

val partial_fit : ?y:Py.Object.t -> x:Ndarray.t -> t -> Py.Object.t
(**
Does nothing: this transformer is stateless.

This method is just there to mark the fact that this transformer
can work in a streaming setup.

Parameters
----------
X : array-like, shape [n_samples, n_features]
    Training data.
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

val transform : x:Py.Object.t -> t -> Ndarray.t
(**
Transform a sequence of documents to a document-term matrix.

Parameters
----------
X : iterable over raw text documents, length = n_samples
    Samples. Each sample must be a text document (either bytes or
    unicode strings, file name or file object depending on the
    constructor argument) which will be tokenized and hashed.

Returns
-------
X : sparse matrix of shape (n_samples, n_features)
    Document-term matrix.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module TfidfTransformer : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?norm:[`L1 | `L2 | `None] -> ?use_idf:bool -> ?smooth_idf:bool -> ?sublinear_tf:bool -> unit -> t
(**
Transform a count matrix to a normalized tf or tf-idf representation

Tf means term-frequency while tf-idf means term-frequency times inverse
document-frequency. This is a common term weighting scheme in information
retrieval, that has also found good use in document classification.

The goal of using tf-idf instead of the raw frequencies of occurrence of a
token in a given document is to scale down the impact of tokens that occur
very frequently in a given corpus and that are hence empirically less
informative than features that occur in a small fraction of the training
corpus.

The formula that is used to compute the tf-idf for a term t of a document d
in a document set is tf-idf(t, d) = tf(t, d) * idf(t), and the idf is
computed as idf(t) = log [ n / df(t) ] + 1 (if ``smooth_idf=False``), where
n is the total number of documents in the document set and df(t) is the
document frequency of t; the document frequency is the number of documents
in the document set that contain the term t. The effect of adding "1" to
the idf in the equation above is that terms with zero idf, i.e., terms
that occur in all documents in a training set, will not be entirely
ignored.
(Note that the idf formula above differs from the standard textbook
notation that defines the idf as
idf(t) = log [ n / (df(t) + 1) ]).

If ``smooth_idf=True`` (the default), the constant "1" is added to the
numerator and denominator of the idf as if an extra document was seen
containing every term in the collection exactly once, which prevents
zero divisions: idf(d, t) = log [ (1 + n) / (1 + df(d, t)) ] + 1.

Furthermore, the formulas used to compute tf and idf depend
on parameter settings that correspond to the SMART notation used in IR
as follows:

Tf is "n" (natural) by default, "l" (logarithmic) when
``sublinear_tf=True``.
Idf is "t" when use_idf is given, "n" (none) otherwise.
Normalization is "c" (cosine) when ``norm='l2'``, "n" (none)
when ``norm=None``.

Read more in the :ref:`User Guide <text_feature_extraction>`.

Parameters
----------
norm : 'l1', 'l2' or None, optional (default='l2')
    Each output row will have unit norm, either:
    * 'l2': Sum of squares of vector elements is 1. The cosine
    similarity between two vectors is their dot product when l2 norm has
    been applied.
    * 'l1': Sum of absolute values of vector elements is 1.
    See :func:`preprocessing.normalize`

use_idf : boolean (default=True)
    Enable inverse-document-frequency reweighting.

smooth_idf : boolean (default=True)
    Smooth idf weights by adding one to document frequencies, as if an
    extra document was seen containing every term in the collection
    exactly once. Prevents zero divisions.

sublinear_tf : boolean (default=False)
    Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

Attributes
----------
idf_ : array, shape (n_features)
    The inverse document frequency (IDF) vector; only defined
    if  ``use_idf`` is True.

Examples
--------
>>> from sklearn.feature_extraction.text import TfidfTransformer
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> from sklearn.pipeline import Pipeline
>>> import numpy as np
>>> corpus = ['this is the first document',
...           'this document is the second document',
...           'and this is the third one',
...           'is this the first document']
>>> vocabulary = ['this', 'document', 'first', 'is', 'second', 'the',
...               'and', 'one']
>>> pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)),
...                  ('tfid', TfidfTransformer())]).fit(corpus)
>>> pipe['count'].transform(corpus).toarray()
array([[1, 1, 1, 1, 0, 1, 0, 0],
       [1, 2, 0, 1, 1, 1, 0, 0],
       [1, 0, 0, 1, 0, 1, 1, 1],
       [1, 1, 1, 1, 0, 1, 0, 0]])
>>> pipe['tfid'].idf_
array([1.        , 1.22314355, 1.51082562, 1.        , 1.91629073,
       1.        , 1.91629073, 1.91629073])
>>> pipe.transform(corpus).shape
(4, 8)

References
----------

.. [Yates2011] R. Baeza-Yates and B. Ribeiro-Neto (2011). Modern
               Information Retrieval. Addison Wesley, pp. 68-74.

.. [MRS2008] C.D. Manning, P. Raghavan and H. Schütze  (2008).
               Introduction to Information Retrieval. Cambridge University
               Press, pp. 118-120.
*)

val fit : ?y:Py.Object.t -> x:Csr_matrix.t -> t -> t
(**
Learn the idf vector (global term weights)

Parameters
----------
X : sparse matrix, [n_samples, n_features]
    a matrix of term/token counts
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

val transform : ?copy:bool -> x:Csr_matrix.t -> t -> Ndarray.t
(**
Transform a count matrix to a tf or tf-idf representation

Parameters
----------
X : sparse matrix, [n_samples, n_features]
    a matrix of term/token counts

copy : boolean, default True
    Whether to copy X and operate on the copy or perform in-place
    operations.

Returns
-------
vectors : sparse matrix, [n_samples, n_features]
*)


(** Attribute idf_: see constructor for documentation *)
val idf_ : t -> Ndarray.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module TfidfVectorizer : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?input:[`Filename | `File | `Content] -> ?encoding:string -> ?decode_error:[`Strict | `Ignore | `Replace] -> ?strip_accents:[`Ascii | `Unicode | `None] -> ?lowercase:bool -> ?preprocessor:[`Callable of Py.Object.t | `None] -> ?tokenizer:[`Callable of Py.Object.t | `None] -> ?analyzer:[`String of string | `Word | `Char | `Char_wb | `Callable of Py.Object.t] -> ?stop_words:[`English | `ArrayLike of Py.Object.t | `None] -> ?token_pattern:string -> ?ngram_range:Py.Object.t -> ?max_df:[`Int of int | `PyObject of Py.Object.t] -> ?min_df:[`Int of int | `PyObject of Py.Object.t] -> ?max_features:[`Int of int | `None] -> ?vocabulary:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> ?binary:bool -> ?dtype:Py.Object.t -> ?norm:[`L1 | `L2 | `None] -> ?use_idf:bool -> ?smooth_idf:bool -> ?sublinear_tf:bool -> unit -> t
(**
Convert a collection of raw documents to a matrix of TF-IDF features.

Equivalent to :class:`CountVectorizer` followed by
:class:`TfidfTransformer`.

Read more in the :ref:`User Guide <text_feature_extraction>`.

Parameters
----------
input : str {'filename', 'file', 'content'}
    If 'filename', the sequence passed as an argument to fit is
    expected to be a list of filenames that need reading to fetch
    the raw content to analyze.

    If 'file', the sequence items must have a 'read' method (file-like
    object) that is called to fetch the bytes in memory.

    Otherwise the input is expected to be a sequence of items that
    can be of type string or byte.

encoding : str, default='utf-8'
    If bytes or files are given to analyze, this encoding is used to
    decode.

decode_error : {'strict', 'ignore', 'replace'} (default='strict')
    Instruction on what to do if a byte sequence is given to analyze that
    contains characters not of the given `encoding`. By default, it is
    'strict', meaning that a UnicodeDecodeError will be raised. Other
    values are 'ignore' and 'replace'.

strip_accents : {'ascii', 'unicode', None} (default=None)
    Remove accents and perform other character normalization
    during the preprocessing step.
    'ascii' is a fast method that only works on characters that have
    an direct ASCII mapping.
    'unicode' is a slightly slower method that works on any characters.
    None (default) does nothing.

    Both 'ascii' and 'unicode' use NFKD normalization from
    :func:`unicodedata.normalize`.

lowercase : bool (default=True)
    Convert all characters to lowercase before tokenizing.

preprocessor : callable or None (default=None)
    Override the preprocessing (string transformation) stage while
    preserving the tokenizing and n-grams generation steps.
    Only applies if ``analyzer is not callable``.

tokenizer : callable or None (default=None)
    Override the string tokenization step while preserving the
    preprocessing and n-grams generation steps.
    Only applies if ``analyzer == 'word'``.

analyzer : str, {'word', 'char', 'char_wb'} or callable
    Whether the feature should be made of word or character n-grams.
    Option 'char_wb' creates character n-grams only from text inside
    word boundaries; n-grams at the edges of words are padded with space.

    If a callable is passed it is used to extract the sequence of features
    out of the raw, unprocessed input.

    .. versionchanged:: 0.21

    Since v0.21, if ``input`` is ``filename`` or ``file``, the data is
    first read from the file and then passed to the given callable
    analyzer.

stop_words : str {'english'}, list, or None (default=None)
    If a string, it is passed to _check_stop_list and the appropriate stop
    list is returned. 'english' is currently the only supported string
    value.
    There are several known issues with 'english' and you should
    consider an alternative (see :ref:`stop_words`).

    If a list, that list is assumed to contain stop words, all of which
    will be removed from the resulting tokens.
    Only applies if ``analyzer == 'word'``.

    If None, no stop words will be used. max_df can be set to a value
    in the range [0.7, 1.0) to automatically detect and filter stop
    words based on intra corpus document frequency of terms.

token_pattern : str
    Regular expression denoting what constitutes a "token", only used
    if ``analyzer == 'word'``. The default regexp selects tokens of 2
    or more alphanumeric characters (punctuation is completely ignored
    and always treated as a token separator).

ngram_range : tuple (min_n, max_n), default=(1, 1)
    The lower and upper boundary of the range of n-values for different
    n-grams to be extracted. All values of n such that min_n <= n <= max_n
    will be used. For example an ``ngram_range`` of ``(1, 1)`` means only
    unigrams, ``(1, 2)`` means unigrams and bigrams, and ``(2, 2)`` means
    only bigrams.
    Only applies if ``analyzer is not callable``.

max_df : float in range [0.0, 1.0] or int (default=1.0)
    When building the vocabulary ignore terms that have a document
    frequency strictly higher than the given threshold (corpus-specific
    stop words).
    If float, the parameter represents a proportion of documents, integer
    absolute counts.
    This parameter is ignored if vocabulary is not None.

min_df : float in range [0.0, 1.0] or int (default=1)
    When building the vocabulary ignore terms that have a document
    frequency strictly lower than the given threshold. This value is also
    called cut-off in the literature.
    If float, the parameter represents a proportion of documents, integer
    absolute counts.
    This parameter is ignored if vocabulary is not None.

max_features : int or None (default=None)
    If not None, build a vocabulary that only consider the top
    max_features ordered by term frequency across the corpus.

    This parameter is ignored if vocabulary is not None.

vocabulary : Mapping or iterable, optional (default=None)
    Either a Mapping (e.g., a dict) where keys are terms and values are
    indices in the feature matrix, or an iterable over terms. If not
    given, a vocabulary is determined from the input documents.

binary : bool (default=False)
    If True, all non-zero term counts are set to 1. This does not mean
    outputs will have only 0/1 values, only that the tf term in tf-idf
    is binary. (Set idf and normalization to False to get 0/1 outputs).

dtype : type, optional (default=float64)
    Type of the matrix returned by fit_transform() or transform().

norm : 'l1', 'l2' or None, optional (default='l2')
    Each output row will have unit norm, either:
    * 'l2': Sum of squares of vector elements is 1. The cosine
    similarity between two vectors is their dot product when l2 norm has
    been applied.
    * 'l1': Sum of absolute values of vector elements is 1.
    See :func:`preprocessing.normalize`.

use_idf : bool (default=True)
    Enable inverse-document-frequency reweighting.

smooth_idf : bool (default=True)
    Smooth idf weights by adding one to document frequencies, as if an
    extra document was seen containing every term in the collection
    exactly once. Prevents zero divisions.

sublinear_tf : bool (default=False)
    Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

Attributes
----------
vocabulary_ : dict
    A mapping of terms to feature indices.

fixed_vocabulary_: bool
    True if a fixed vocabulary of term to indices mapping
    is provided by the user

idf_ : array, shape (n_features)
    The inverse document frequency (IDF) vector; only defined
    if ``use_idf`` is True.

stop_words_ : set
    Terms that were ignored because they either:

      - occurred in too many documents (`max_df`)
      - occurred in too few documents (`min_df`)
      - were cut off by feature selection (`max_features`).

    This is only available if no vocabulary was given.

See Also
--------
CountVectorizer : Transforms text into a sparse matrix of n-gram counts.

TfidfTransformer : Performs the TF-IDF transformation from a provided
    matrix of counts.

Notes
-----
The ``stop_words_`` attribute can get large and increase the model size
when pickling. This attribute is provided only for introspection and can
be safely removed using delattr or set to None before pickling.

Examples
--------
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> corpus = [
...     'This is the first document.',
...     'This document is the second document.',
...     'And this is the third one.',
...     'Is this the first document?',
... ]
>>> vectorizer = TfidfVectorizer()
>>> X = vectorizer.fit_transform(corpus)
>>> print(vectorizer.get_feature_names())
['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
>>> print(X.shape)
(4, 9)
*)

val build_analyzer : t -> Py.Object.t
(**
Return a callable that handles preprocessing, tokenization
and n-grams generation.

Returns
-------
analyzer: callable
    A function to handle preprocessing, tokenization
    and n-grams generation.
*)

val build_preprocessor : t -> Py.Object.t
(**
Return a function to preprocess the text before tokenization.

Returns
-------
preprocessor: callable
      A function to preprocess the text before tokenization.
*)

val build_tokenizer : t -> Py.Object.t
(**
Return a function that splits a string into a sequence of tokens.

Returns
-------
tokenizer: callable
      A function to split a string into a sequence of tokens.
*)

val decode : doc:string -> t -> string
(**
Decode the input into a string of unicode symbols.

The decoding strategy depends on the vectorizer parameters.

Parameters
----------
doc : str
    The string to decode.

Returns
-------
doc: str
    A string of unicode symbols.
*)

val fit : ?y:Py.Object.t -> raw_documents:Ndarray.t -> t -> t
(**
Learn vocabulary and idf from training set.

Parameters
----------
raw_documents : iterable
    An iterable which yields either str, unicode or file objects.
y : None
    This parameter is not needed to compute tfidf.

Returns
-------
self : object
    Fitted vectorizer.
*)

val fit_transform : ?y:Py.Object.t -> raw_documents:Ndarray.t -> t -> Ndarray.t
(**
Learn vocabulary and idf, return term-document matrix.

This is equivalent to fit followed by transform, but more efficiently
implemented.

Parameters
----------
raw_documents : iterable
    An iterable which yields either str, unicode or file objects.
y : None
    This parameter is ignored.

Returns
-------
X : sparse matrix, [n_samples, n_features]
    Tf-idf-weighted document-term matrix.
*)

val get_feature_names : t -> string list
(**
Array mapping from feature integer indices to feature name.

Returns
-------
feature_names : list
    A list of feature names.
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

val get_stop_words : t -> Py.Object.t
(**
Build or fetch the effective stop words list.

Returns
-------
stop_words: list or None
        A list of stop words.
*)

val inverse_transform : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> t -> Py.Object.t
(**
Return terms per document with nonzero entries in X.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Document-term matrix.

Returns
-------
X_inv : list of arrays, len = n_samples
    List of arrays of terms.
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

val transform : ?copy:bool -> raw_documents:Ndarray.t -> t -> Ndarray.t
(**
Transform documents to document-term matrix.

Uses the vocabulary and document frequencies (df) learned by fit (or
fit_transform).

Parameters
----------
raw_documents : iterable
    An iterable which yields either str, unicode or file objects.

copy : bool, default True
    Whether to copy X and operate on the copy or perform in-place
    operations.

    .. deprecated:: 0.22
       The `copy` parameter is unused and was deprecated in version
       0.22 and will be removed in 0.24. This parameter will be
       ignored.

Returns
-------
X : sparse matrix, [n_samples, n_features]
    Tf-idf-weighted document-term matrix.
*)


(** Attribute vocabulary_: see constructor for documentation *)
val vocabulary_ : t -> Py.Object.t

(** Attribute fixed_vocabulary_: see constructor for documentation *)
val fixed_vocabulary_ : t -> bool

(** Attribute idf_: see constructor for documentation *)
val idf_ : t -> Ndarray.t

(** Attribute stop_words_: see constructor for documentation *)
val stop_words_ : t -> Py.Object.t

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

val strip_accents_ascii : s:string -> unit -> Py.Object.t
(**
Transform accentuated unicode symbols into ascii or nothing

Warning: this solution is only suited for languages that have a direct
transliteration to ASCII symbols.

Parameters
----------
s : string
    The string to strip

See Also
--------
strip_accents_unicode
    Remove accentuated char for any unicode symbol.
*)

val strip_accents_unicode : s:string -> unit -> Py.Object.t
(**
Transform accentuated unicode symbols into their simple counterpart

Warning: the python-level loop and join operations make this
implementation 20 times slower than the strip_accents_ascii basic
normalization.

Parameters
----------
s : string
    The string to strip

See Also
--------
strip_accents_ascii
    Remove accentuated char for any unicode symbol that has a direct
    ASCII equivalent.
*)

val strip_tags : s:string -> unit -> Py.Object.t
(**
Basic regexp based HTML / XML tag stripper function

For serious HTML/XML preprocessing you should rather use an external
library such as lxml or BeautifulSoup.

Parameters
----------
s : string
    The string to strip
*)


end

