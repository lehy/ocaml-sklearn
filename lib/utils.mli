module Bunch : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?kwargs:(string * Py.Object.t) list -> unit -> t
(**
Container object for datasets

Dictionary-like object that exposes its keys as attributes.

>>> b = Bunch(a=1, b=2)
>>> b['b']
2
>>> b.b
2
>>> b.a = 3
>>> b['a']
3
>>> b.c = 6
>>> b['c']
6
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

module Path : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> t
(**
PurePath subclass that can make system calls.

Path represents a filesystem path but unlike PurePath, also offers
methods to do system calls on path objects. Depending on your system,
instantiating a Path will return either a PosixPath or a WindowsPath
object. You can also instantiate a PosixPath or WindowsPath directly,
but cannot instantiate a WindowsPath on a POSIX system or vice versa.
*)

val absolute : t -> Py.Object.t
(**
Return an absolute version of this path.  This function works
even if the path doesn't point to anything.

No normalization is done, i.e. all '.' and '..' will be kept along.
Use resolve() to get the canonical path to a file.
*)

val as_posix : t -> Py.Object.t
(**
Return the string representation of the path with forward (/)
slashes.
*)

val as_uri : t -> Py.Object.t
(**
Return the path as a 'file' URI.
*)

val chmod : mode:Py.Object.t -> t -> Py.Object.t
(**
Change the permissions of the path, like os.chmod().
*)

val cwd : t -> Py.Object.t
(**
Return a new path pointing to the current working directory
(as returned by os.getcwd()).
*)

val exists : t -> Py.Object.t
(**
Whether this path exists.
*)

val expanduser : t -> Py.Object.t
(**
Return a new path with expanded ~ and ~user constructs
(as returned by os.path.expanduser)
*)

val glob : pattern:Py.Object.t -> t -> Py.Object.t
(**
Iterate over this subtree and yield all existing files (of any
kind, including directories) matching the given relative pattern.
*)

val group : t -> Py.Object.t
(**
Return the group name of the file gid.
*)

val home : t -> Py.Object.t
(**
Return a new path pointing to the user's home directory (as
returned by os.path.expanduser('~')).
*)

val is_absolute : t -> Py.Object.t
(**
True if the path is absolute (has both a root and, if applicable,
a drive).
*)

val is_block_device : t -> Py.Object.t
(**
Whether this path is a block device.
*)

val is_char_device : t -> Py.Object.t
(**
Whether this path is a character device.
*)

val is_dir : t -> Py.Object.t
(**
Whether this path is a directory.
*)

val is_fifo : t -> Py.Object.t
(**
Whether this path is a FIFO.
*)

val is_file : t -> Py.Object.t
(**
Whether this path is a regular file (also True for symlinks pointing
to regular files).
*)

val is_mount : t -> Py.Object.t
(**
Check if this path is a POSIX mount point
*)

val is_reserved : t -> Py.Object.t
(**
Return True if the path contains one of the special names reserved
by the system, if any.
*)

val is_socket : t -> Py.Object.t
(**
Whether this path is a socket.
*)

val is_symlink : t -> Py.Object.t
(**
Whether this path is a symbolic link.
*)

val iterdir : t -> Py.Object.t
(**
Iterate over the files in this directory.  Does not yield any
result for the special paths '.' and '..'.
*)

val joinpath : Py.Object.t list -> t -> Py.Object.t
(**
Combine this path with one or several arguments, and return a
new path representing either a subpath (if all arguments are relative
paths) or a totally different path (if one of the arguments is
anchored).
*)

val lchmod : mode:Py.Object.t -> t -> Py.Object.t
(**
Like chmod(), except if the path points to a symlink, the symlink's
permissions are changed, rather than its target's.
*)

val lstat : t -> Py.Object.t
(**
Like stat(), except if the path points to a symlink, the symlink's
status information is returned, rather than its target's.
*)

val match_ : path_pattern:Py.Object.t -> t -> Py.Object.t
(**
Return True if this path matches the given pattern.
*)

val mkdir : ?mode:Py.Object.t -> ?parents:Py.Object.t -> ?exist_ok:Py.Object.t -> t -> Py.Object.t
(**
Create a new directory at this given path.
*)

val open_ : ?mode:Py.Object.t -> ?buffering:Py.Object.t -> ?encoding:Py.Object.t -> ?errors:Py.Object.t -> ?newline:Py.Object.t -> t -> Py.Object.t
(**
Open the file pointed by this path and return a file object, as
the built-in open() function does.
*)

val owner : t -> Py.Object.t
(**
Return the login name of the file owner.
*)

val read_bytes : t -> Py.Object.t
(**
Open the file in bytes mode, read it, and close the file.
*)

val read_text : ?encoding:Py.Object.t -> ?errors:Py.Object.t -> t -> Py.Object.t
(**
Open the file in text mode, read it, and close the file.
*)

val relative_to : Py.Object.t list -> t -> Py.Object.t
(**
Return the relative path to another path identified by the passed
arguments.  If the operation is not possible (because this is not
a subpath of the other path), raise ValueError.
*)

val rename : target:Py.Object.t -> t -> Py.Object.t
(**
Rename this path to the given path.
*)

val replace : target:Py.Object.t -> t -> Py.Object.t
(**
Rename this path to the given path, clobbering the existing
destination if it exists.
*)

val resolve : ?strict:Py.Object.t -> t -> Py.Object.t
(**
Make the path absolute, resolving all symlinks on the way and also
normalizing it (for example turning slashes into backslashes under
Windows).
*)

val rglob : pattern:Py.Object.t -> t -> Py.Object.t
(**
Recursively yield all existing files (of any kind, including
directories) matching the given relative pattern, anywhere in
this subtree.
*)

val rmdir : t -> Py.Object.t
(**
Remove this directory.  The directory must be empty.
*)

val samefile : other_path:Py.Object.t -> t -> Py.Object.t
(**
Return whether other_path is the same or not as this file
(as returned by os.path.samefile()).
*)

val stat : t -> Py.Object.t
(**
Return the result of the stat() system call on this path, like
os.stat() does.
*)

val symlink_to : ?target_is_directory:Py.Object.t -> target:Py.Object.t -> t -> Py.Object.t
(**
Make this path a symlink pointing to the given path.
Note the order of arguments (self, target) is the reverse of os.symlink's.
*)

val touch : ?mode:Py.Object.t -> ?exist_ok:Py.Object.t -> t -> Py.Object.t
(**
Create this file with the given access mode, if it doesn't exist.
*)

val unlink : t -> Py.Object.t
(**
Remove this file or link.
If the path is a directory, use rmdir() instead.
*)

val with_name : name:Py.Object.t -> t -> Py.Object.t
(**
Return a new path with the file name changed.
*)

val with_suffix : suffix:Py.Object.t -> t -> Py.Object.t
(**
Return a new path with the file suffix changed.  If the path
has no suffix, add given suffix.  If the given suffix is an empty
string, remove the suffix from the path.
*)

val write_bytes : data:Py.Object.t -> t -> Py.Object.t
(**
Open the file in bytes mode, write to it, and close the file.
*)

val write_text : ?encoding:Py.Object.t -> ?errors:Py.Object.t -> data:Py.Object.t -> t -> Py.Object.t
(**
Open the file in text mode, write to it, and close the file.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

val all_estimators : ?include_meta_estimators:bool -> ?include_other:bool -> ?type_filter:[`String of string | `StringList of string list | `None] -> ?include_dont_test:bool -> unit -> Py.Object.t
(**
Get a list of all estimators from sklearn.

This function crawls the module and gets all classes that inherit
from BaseEstimator. Classes that are defined in test-modules are not
included.
By default meta_estimators such as GridSearchCV are also not included.

Parameters
----------
include_meta_estimators : boolean, default=False
    Deprecated, ignored.

    .. deprecated:: 0.21
       ``include_meta_estimators`` has been deprecated and has no effect in
       0.21 and will be removed in 0.23.

include_other : boolean, default=False
    Deprecated, ignored.

    .. deprecated:: 0.21
       ``include_other`` has been deprecated and has not effect in 0.21 and
       will be removed in 0.23.

type_filter : string, list of string,  or None, default=None
    Which kind of estimators should be returned. If None, no filter is
    applied and all estimators are returned.  Possible values are
    'classifier', 'regressor', 'cluster' and 'transformer' to get
    estimators only of these specific types, or a list of these to
    get the estimators that fit at least one of the types.

include_dont_test : boolean, default=False
    Deprecated, ignored.

    .. deprecated:: 0.21
       ``include_dont_test`` has been deprecated and has no effect in 0.21
       and will be removed in 0.23.

Returns
-------
estimators : list of tuples
    List of (name, class), where ``name`` is the class name as string
    and ``class`` is the actuall type of the class.
*)

module Arrayfuncs : sig
val cholesky_delete : l:Py.Object.t -> go_out:Py.Object.t -> unit -> Py.Object.t
(**
None
*)


end

val as_float_array : ?copy:bool -> ?force_all_finite:[`Bool of bool | `Allow_nan] -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> unit -> Py.Object.t
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

val assert_all_finite : ?allow_nan:bool -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> unit -> Py.Object.t
(**
Throw a ValueError if X contains NaN or infinity.

Parameters
----------
X : array or sparse matrix

allow_nan : bool
*)

val axis0_safe_slice : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> mask:Ndarray.t -> len_mask:int -> unit -> Py.Object.t
(**
This mask is safer than safe_mask since it returns an
empty array, when a sparse matrix is sliced with a boolean mask
with all False, instead of raising an unhelpful error in older
versions of SciPy.

See: https://github.com/scipy/scipy/issues/5361

Also note that we can avoid doing the dot product by checking if
the len_mask is not zero in _huber_loss_and_gradient but this
is not going to be the bottleneck, since the number of outliers
and non_outliers are typically non-zero and it makes the code
tougher to follow.

Parameters
----------
X : {array-like, sparse matrix}
    Data on which to apply mask.

mask : array
    Mask to be used on X.

len_mask : int
    The length of the mask.

Returns
-------
    mask
*)

val check_X_y : ?accept_sparse:[`String of string | `Bool of bool | `StringList of string list] -> ?accept_large_sparse:bool -> ?dtype:[`String of string | `Dtype of Py.Object.t | `TypeList of Py.Object.t | `None] -> ?order:[`F | `C | `None] -> ?copy:bool -> ?force_all_finite:[`Bool of bool | `Allow_nan] -> ?ensure_2d:bool -> ?allow_nd:bool -> ?multi_output:bool -> ?ensure_min_samples:int -> ?ensure_min_features:int -> ?y_numeric:bool -> ?warn_on_dtype:[`Bool of bool | `None] -> ?estimator:[`String of string | `Estimator of Py.Object.t] -> x:[`Ndarray of Ndarray.t | `ArrayLike of Py.Object.t | `SparseMatrix of Csr_matrix.t] -> y:[`Ndarray of Ndarray.t | `ArrayLike of Py.Object.t | `SparseMatrix of Csr_matrix.t] -> unit -> (Py.Object.t * Py.Object.t)
(**
Input validation for standard estimators.

Checks X and y for consistent length, enforces X to be 2D and y 1D. By
default, X is checked to be non-empty and containing only finite values.
Standard input checks are also applied to y, such as checking that y
does not have np.nan or np.inf targets. For multi-label y, set
multi_output=True to allow 2D and sparse y. If the dtype of X is
object, attempt converting to float, raising on failure.

Parameters
----------
X : nd-array, list or sparse matrix
    Input data.

y : nd-array, list or sparse matrix
    Labels.

accept_sparse : string, boolean or list of string (default=False)
    String[s] representing allowed sparse matrix formats, such as 'csc',
    'csr', etc. If the input is sparse but not in the allowed format,
    it will be converted to the first listed format. True allows the input
    to be any format. False means that a sparse matrix input will
    raise an error.

accept_large_sparse : bool (default=True)
    If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
    accept_sparse, accept_large_sparse will cause it to be accepted only
    if its indices are stored with a 32-bit dtype.

    .. versionadded:: 0.20

dtype : string, type, list of types or None (default="numeric")
    Data type of result. If None, the dtype of the input is preserved.
    If "numeric", dtype is preserved unless array.dtype is object.
    If dtype is a list of types, conversion on the first type is only
    performed if the dtype of the input is not in the list.

order : 'F', 'C' or None (default=None)
    Whether an array will be forced to be fortran or c-style.

copy : boolean (default=False)
    Whether a forced copy will be triggered. If copy=False, a copy might
    be triggered by a conversion.

force_all_finite : boolean or 'allow-nan', (default=True)
    Whether to raise an error on np.inf and np.nan in X. This parameter
    does not influence whether y can have np.inf or np.nan values.
    The possibilities are:

    - True: Force all values of X to be finite.
    - False: accept both np.inf and np.nan in X.
    - 'allow-nan': accept only np.nan values in X. Values cannot be
      infinite.

    .. versionadded:: 0.20
       ``force_all_finite`` accepts the string ``'allow-nan'``.

ensure_2d : boolean (default=True)
    Whether to raise a value error if X is not 2D.

allow_nd : boolean (default=False)
    Whether to allow X.ndim > 2.

multi_output : boolean (default=False)
    Whether to allow 2D y (array or sparse matrix). If false, y will be
    validated as a vector. y cannot have np.nan or np.inf values if
    multi_output=True.

ensure_min_samples : int (default=1)
    Make sure that X has a minimum number of samples in its first
    axis (rows for a 2D array).

ensure_min_features : int (default=1)
    Make sure that the 2D array has some minimum number of features
    (columns). The default value of 1 rejects empty datasets.
    This check is only enforced when X has effectively 2 dimensions or
    is originally 1D and ``ensure_2d`` is True. Setting to 0 disables
    this check.

y_numeric : boolean (default=False)
    Whether to ensure that y has a numeric type. If dtype of y is object,
    it is converted to float64. Should only be used for regression
    algorithms.

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
X_converted : object
    The converted and validated X.

y_converted : object
    The converted and validated y.
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

val check_consistent_length : Py.Object.t list -> Py.Object.t
(**
Check that all arrays have consistent first dimensions.

Checks whether all objects in arrays have the same shape or length.

Parameters
----------
*arrays : list or tuple of input objects.
    Objects that will be checked for consistent length.
*)

val check_matplotlib_support : caller_name:string -> unit -> Py.Object.t
(**
Raise ImportError with detailed error message if mpl is not installed.

Plot utilities like :func:`plot_partial_dependence` should lazily import
matplotlib and call this helper before any computation.

Parameters
----------
caller_name : str
    The name of the caller that requires matplotlib.
*)

val check_pandas_support : caller_name:string -> unit -> Py.Object.t
(**
Raise ImportError with detailed error message if pandsa is not
installed.

Plot utilities like :func:`fetch_openml` should lazily import
pandas and call this helper before any computation.

Parameters
----------
caller_name : str
    The name of the caller that requires pandas.
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

val check_scalar : ?min_val:[`Float of float | `Int of int] -> ?max_val:[`Float of float | `Int of int] -> x:Py.Object.t -> name:string -> target_type:[`Dtype of Py.Object.t | `PyObject of Py.Object.t] -> unit -> Py.Object.t
(**
Validate scalar parameters type and value.

Parameters
----------
x : object
    The scalar parameter to validate.

name : str
    The name of the parameter to be printed in error messages.

target_type : type or tuple
    Acceptable data types for the parameter.

min_val : float or int, optional (default=None)
    The minimum valid value the parameter can take. If None (default) it
    is implied that the parameter does not have a lower bound.

max_val : float or int, optional (default=None)
    The maximum valid value the parameter can take. If None (default) it
    is implied that the parameter does not have an upper bound.

Raises
-------
TypeError
    If the parameter's type does not match the desired type.

ValueError
    If the parameter's value violates the given bounds.
*)

val check_symmetric : ?tol:Py.Object.t -> ?raise_warning:Py.Object.t -> ?raise_exception:Py.Object.t -> array:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> unit -> Py.Object.t
(**
Make sure that array is 2D, square and symmetric.

If the array is not symmetric, then a symmetrized version is returned.
Optionally, a warning or exception is raised if the matrix is not
symmetric.

Parameters
----------
array : nd-array or sparse matrix
    Input object to check / convert. Must be two-dimensional and square,
    otherwise a ValueError will be raised.
tol : float
    Absolute tolerance for equivalence of arrays. Default = 1E-10.
raise_warning : boolean (default=True)
    If True then raise a warning if conversion is required.
raise_exception : boolean (default=False)
    If True then raise an exception if array is not symmetric.

Returns
-------
array_sym : ndarray or sparse matrix
    Symmetrized version of the input array, i.e. the average of array
    and array.transpose(). If sparse, then duplicate entries are first
    summed and zeros are eliminated.
*)

module Class_weight : sig
val compute_class_weight : class_weight:[`DictIntToFloat of (int * float) list | `Balanced | `None] -> classes:Ndarray.t -> y:Ndarray.t -> unit -> Ndarray.t
(**
Estimate class weights for unbalanced datasets.

Parameters
----------
class_weight : dict, 'balanced' or None
    If 'balanced', class weights will be given by
    ``n_samples / (n_classes * np.bincount(y))``.
    If a dictionary is given, keys are classes and values
    are corresponding class weights.
    If None is given, the class weights will be uniform.

classes : ndarray
    Array of the classes occurring in the data, as given by
    ``np.unique(y_org)`` with ``y_org`` the original class labels.

y : array-like, shape (n_samples,)
    Array of original class labels per sample;

Returns
-------
class_weight_vect : ndarray, shape (n_classes,)
    Array with class_weight_vect[i] the weight for i-th class

References
----------
The "balanced" heuristic is inspired by
Logistic Regression in Rare Events Data, King, Zen, 2001.
*)

val compute_sample_weight : ?indices:[`Ndarray of Ndarray.t | `None] -> class_weight:[`DictIntToFloat of (int * float) list | `Balanced | `None | `PyObject of Py.Object.t] -> y:Ndarray.t -> unit -> Ndarray.t
(**
Estimate sample weights by class for unbalanced datasets.

Parameters
----------
class_weight : dict, list of dicts, "balanced", or None, optional
    Weights associated with classes in the form ``{class_label: weight}``.
    If not given, all classes are supposed to have weight one. For
    multi-output problems, a list of dicts can be provided in the same
    order as the columns of y.

    Note that for multioutput (including multilabel) weights should be
    defined for each class of every column in its own dict. For example,
    for four-class multilabel classification weights should be
    [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
    [{1:1}, {2:5}, {3:1}, {4:1}].

    The "balanced" mode uses the values of y to automatically adjust
    weights inversely proportional to class frequencies in the input data:
    ``n_samples / (n_classes * np.bincount(y))``.

    For multi-output, the weights of each column of y will be multiplied.

y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    Array of original class labels per sample.

indices : array-like, shape (n_subsample,), or None
    Array of indices to be used in a subsample. Can be of length less than
    n_samples in the case of a subsample, or equal to n_samples in the
    case of a bootstrap subsample with repeated indices. If None, the
    sample weight will be calculated over the full sample. Only "balanced"
    is supported for class_weight if this is provided.

Returns
-------
sample_weight_vect : ndarray, shape (n_samples,)
    Array with sample weights as applied to the original y
*)


end

val column_or_1d : ?warn:bool -> y:Ndarray.t -> unit -> Ndarray.t
(**
Ravel column or 1d numpy array, else raises an error

Parameters
----------
y : array-like

warn : boolean, default False
   To control display of warnings.

Returns
-------
y : array
*)

val compute_class_weight : class_weight:[`DictIntToFloat of (int * float) list | `Balanced | `None] -> classes:Ndarray.t -> y:Ndarray.t -> unit -> Ndarray.t
(**
Estimate class weights for unbalanced datasets.

Parameters
----------
class_weight : dict, 'balanced' or None
    If 'balanced', class weights will be given by
    ``n_samples / (n_classes * np.bincount(y))``.
    If a dictionary is given, keys are classes and values
    are corresponding class weights.
    If None is given, the class weights will be uniform.

classes : ndarray
    Array of the classes occurring in the data, as given by
    ``np.unique(y_org)`` with ``y_org`` the original class labels.

y : array-like, shape (n_samples,)
    Array of original class labels per sample;

Returns
-------
class_weight_vect : ndarray, shape (n_classes,)
    Array with class_weight_vect[i] the weight for i-th class

References
----------
The "balanced" heuristic is inspired by
Logistic Regression in Rare Events Data, King, Zen, 2001.
*)

val compute_sample_weight : ?indices:[`Ndarray of Ndarray.t | `None] -> class_weight:[`DictIntToFloat of (int * float) list | `Balanced | `None | `PyObject of Py.Object.t] -> y:Ndarray.t -> unit -> Ndarray.t
(**
Estimate sample weights by class for unbalanced datasets.

Parameters
----------
class_weight : dict, list of dicts, "balanced", or None, optional
    Weights associated with classes in the form ``{class_label: weight}``.
    If not given, all classes are supposed to have weight one. For
    multi-output problems, a list of dicts can be provided in the same
    order as the columns of y.

    Note that for multioutput (including multilabel) weights should be
    defined for each class of every column in its own dict. For example,
    for four-class multilabel classification weights should be
    [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
    [{1:1}, {2:5}, {3:1}, {4:1}].

    The "balanced" mode uses the values of y to automatically adjust
    weights inversely proportional to class frequencies in the input data:
    ``n_samples / (n_classes * np.bincount(y))``.

    For multi-output, the weights of each column of y will be multiplied.

y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    Array of original class labels per sample.

indices : array-like, shape (n_subsample,), or None
    Array of indices to be used in a subsample. Can be of length less than
    n_samples in the case of a subsample, or equal to n_samples in the
    case of a bootstrap subsample with repeated indices. If None, the
    sample weight will be calculated over the full sample. Only "balanced"
    is supported for class_weight if this is provided.

Returns
-------
sample_weight_vect : ndarray, shape (n_samples,)
    Array with sample weights as applied to the original y
*)

val contextmanager : func:Py.Object.t -> unit -> Py.Object.t
(**
@contextmanager decorator.

Typical usage:

    @contextmanager
    def some_generator(<arguments>):
        <setup>
        try:
            yield <value>
        finally:
            <cleanup>

This makes this:

    with some_generator(<arguments>) as <variable>:
        <body>

equivalent to this:

    <setup>
    try:
        <variable> = <value>
        <body>
    finally:
        <cleanup>
*)

val cpu_count : unit -> Py.Object.t
(**
DEPRECATED: deprecated in version 0.20.1 to be removed in version 0.23. Please import this functionality directly from joblib, which can be installed with: pip install joblib.

Return the number of CPUs.
*)

val delayed : ?check_pickle:Py.Object.t -> function_:Py.Object.t -> unit -> Py.Object.t
(**
DEPRECATED: deprecated in version 0.20.1 to be removed in version 0.23. Please import this functionality directly from joblib, which can be installed with: pip install joblib.

Decorator used to capture the arguments of a function.
*)

val deprecate : obj:Py.Object.t -> unit -> Py.Object.t
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
val pp : Format.formatter -> t -> unit


end

module Deprecation : sig
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
val pp : Format.formatter -> t -> unit


end


end

val effective_n_jobs : ?n_jobs:Py.Object.t -> unit -> Py.Object.t
(**
DEPRECATED: deprecated in version 0.20.1 to be removed in version 0.23. Please import this functionality directly from joblib, which can be installed with: pip install joblib.

Determine the number of jobs that can actually run in parallel

    n_jobs is the number of workers requested by the callers. Passing n_jobs=-1
    means requesting all available workers for instance matching the number of
    CPU cores on the worker host(s).

    This method should return a guesstimate of the number of workers that can
    actually perform work concurrently with the currently enabled default
    backend. The primary use case is to make it possible for the caller to know
    in how many chunks to slice the work.

    In general working on larger data chunks is more efficient (less scheduling
    overhead and better use of CPU cache prefetching heuristics) as long as all
    the workers have enough work to do.

    Warning: this function is experimental and subject to change in a future
    version of joblib.

    .. versionadded:: 0.10

    
*)

module Extmath : sig
val cartesian : ?out:Py.Object.t -> arrays:Py.Object.t -> unit -> Ndarray.t
(**
Generate a cartesian product of input arrays.

Parameters
----------
arrays : list of array-like
    1-D arrays to form the cartesian product of.
out : ndarray
    Array to place the cartesian product in.

Returns
-------
out : ndarray
    2-D array of shape (M, len(arrays)) containing cartesian products
    formed of input arrays.

Examples
--------
>>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
array([[1, 4, 6],
       [1, 4, 7],
       [1, 5, 6],
       [1, 5, 7],
       [2, 4, 6],
       [2, 4, 7],
       [2, 5, 6],
       [2, 5, 7],
       [3, 4, 6],
       [3, 4, 7],
       [3, 5, 6],
       [3, 5, 7]])
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

val density : ?kwargs:(string * Py.Object.t) list -> w:Ndarray.t -> unit -> Py.Object.t
(**
Compute density of a sparse vector

Parameters
----------
w : array_like
    The sparse vector

Returns
-------
float
    The density of w, between 0 and 1
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
val pp : Format.formatter -> t -> unit


end

val fast_logdet : a:Ndarray.t -> unit -> Py.Object.t
(**
Compute log(det(A)) for A symmetric

Equivalent to : np.log(nl.det(A)) but more robust.
It returns -Inf if det(A) is non positive or is not defined.

Parameters
----------
A : array_like
    The matrix
*)

val log_logistic : ?out:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> x:Ndarray.t -> unit -> Ndarray.t
(**
Compute the log of the logistic function, ``log(1 / (1 + e ** -x))``.

This implementation is numerically stable because it splits positive and
negative values::

    -log(1 + exp(-x_i))     if x_i > 0
    x_i - log(1 + exp(x_i)) if x_i <= 0

For the ordinary logistic function, use ``scipy.special.expit``.

Parameters
----------
X : array-like, shape (M, N) or (M, )
    Argument to the logistic function

out : array-like, shape: (M, N) or (M, ), optional:
    Preallocated output array.

Returns
-------
out : array, shape (M, N) or (M, )
    Log of the logistic function evaluated at every point in x

Notes
-----
See the blog post describing this implementation:
http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
*)

val make_nonnegative : ?min_value:Py.Object.t -> x:Ndarray.t -> unit -> Py.Object.t
(**
Ensure `X.min()` >= `min_value`.

Parameters
----------
X : array_like
    The matrix to make non-negative
min_value : float
    The threshold value

Returns
-------
array_like
    The thresholded array

Raises
------
ValueError
    When X is sparse
*)

val randomized_range_finder : ?power_iteration_normalizer:string -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> a:Py.Object.t -> size:int -> n_iter:int -> unit -> Py.Object.t
(**
Computes an orthonormal matrix whose range approximates the range of A.

Parameters
----------
A : 2D array
    The input data matrix

size : integer
    Size of the return array

n_iter : integer
    Number of power iterations used to stabilize the result

power_iteration_normalizer : 'auto' (default), 'QR', 'LU', 'none'
    Whether the power iterations are normalized with step-by-step
    QR factorization (the slowest but most accurate), 'none'
    (the fastest but numerically unstable when `n_iter` is large, e.g.
    typically 5 or larger), or 'LU' factorization (numerically stable
    but can lose slightly in accuracy). The 'auto' mode applies no
    normalization if `n_iter` <= 2 and switches to LU otherwise.

    .. versionadded:: 0.18

random_state : int, RandomState instance or None, optional (default=None)
    The seed of the pseudo random number generator to use when shuffling
    the data.  If int, random_state is the seed used by the random number
    generator; If RandomState instance, random_state is the random number
    generator; If None, the random number generator is the RandomState
    instance used by `np.random`.

Returns
-------
Q : 2D array
    A (size x size) projection matrix, the range of which
    approximates well the range of the input matrix A.

Notes
-----

Follows Algorithm 4.3 of
Finding structure with randomness: Stochastic algorithms for constructing
approximate matrix decompositions
Halko, et al., 2009 (arXiv:909) https://arxiv.org/pdf/0909.4061.pdf

An implementation of a randomized algorithm for principal component
analysis
A. Szlam et al. 2014
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

val row_norms : ?squared:Py.Object.t -> x:Ndarray.t -> unit -> Py.Object.t
(**
Row-wise (squared) Euclidean norm of X.

Equivalent to np.sqrt((X * X).sum(axis=1)), but also supports sparse
matrices and does not create an X.shape-sized temporary.

Performs no input validation.

Parameters
----------
X : array_like
    The input array
squared : bool, optional (default = False)
    If True, return squared norms.

Returns
-------
array_like
    The row-wise (squared) Euclidean norm of X.
*)

val safe_min : x:Ndarray.t -> unit -> Py.Object.t
(**
DEPRECATED: safe_min is deprecated in version 0.22 and will be removed in version 0.24.

Returns the minimum value of a dense or a CSR/CSC matrix.

    Adapated from https://stackoverflow.com/q/13426580

    .. deprecated:: 0.22.0

    Parameters
    ----------
    X : array_like
        The input array or sparse matrix

    Returns
    -------
    Float
        The min value of X
    
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

val softmax : ?copy:bool -> x:Py.Object.t -> unit -> Ndarray.t
(**
Calculate the softmax function.

The softmax function is calculated by
np.exp(X) / np.sum(np.exp(X), axis=1)

This will cause overflow when large values are exponentiated.
Hence the largest value in each row is subtracted from each data
point to prevent this.

Parameters
----------
X : array-like of floats, shape (M, N)
    Argument to the logistic function

copy : bool, optional
    Copy X or not.

Returns
-------
out : array, shape (M, N)
    Softmax function evaluated at every point in x
*)

val squared_norm : x:Ndarray.t -> unit -> Py.Object.t
(**
Squared Euclidean or Frobenius norm of x.

Faster than norm(x) ** 2.

Parameters
----------
x : array_like

Returns
-------
float
    The Euclidean norm when x is a vector, the Frobenius norm when x
    is a matrix (2-d array).
*)

val stable_cumsum : ?axis:Py.Object.t -> ?rtol:Py.Object.t -> ?atol:Py.Object.t -> arr:Ndarray.t -> unit -> Py.Object.t
(**
Use high precision for cumsum and check that final value matches sum

Parameters
----------
arr : array-like
    To be cumulatively summed as flat
axis : int, optional
    Axis along which the cumulative sum is computed.
    The default (None) is to compute the cumsum over the flattened array.
rtol : float
    Relative tolerance, see ``np.allclose``
atol : float
    Absolute tolerance, see ``np.allclose``
*)

val svd_flip : ?u_based_decision:bool -> u:Ndarray.t -> v:Ndarray.t -> unit -> Py.Object.t
(**
Sign correction to ensure deterministic output from SVD.

Adjusts the columns of u and the rows of v such that the loadings in the
columns in u that are largest in absolute value are always positive.

Parameters
----------
u : ndarray
    u and v are the output of `linalg.svd` or
    :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
    dimensions so one can compute `np.dot(u * s, v)`.

v : ndarray
    u and v are the output of `linalg.svd` or
    :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
    dimensions so one can compute `np.dot(u * s, v)`.

u_based_decision : boolean, (default=True)
    If True, use the columns of u as the basis for sign flipping.
    Otherwise, use the rows of v. The choice of which variable to base the
    decision on is generally algorithm dependent.


Returns
-------
u_adjusted, v_adjusted : arrays with the same dimensions as the input.
*)

val weighted_mode : ?axis:Py.Object.t -> a:Ndarray.t -> w:Py.Object.t -> unit -> Ndarray.t
(**
Returns an array of the weighted modal (most common) value in a

If there is more than one such value, only the first is returned.
The bin-count for the modal bins is also returned.

This is an extension of the algorithm in scipy.stats.mode.

Parameters
----------
a : array_like
    n-dimensional array of which to find mode(s).
w : array_like
    n-dimensional array of weights for each value
axis : int, optional
    Axis along which to operate. Default is 0, i.e. the first axis.

Returns
-------
vals : ndarray
    Array of modal values.
score : ndarray
    Array of weighted counts for each mode.

Examples
--------
>>> from sklearn.utils.extmath import weighted_mode
>>> x = [4, 1, 4, 2, 4, 2]
>>> weights = [1, 1, 1, 1, 1, 1]
>>> weighted_mode(x, weights)
(array([4.]), array([3.]))

The value 4 appears three times: with uniform weights, the result is
simply the mode of the distribution.

>>> weights = [1, 3, 0.5, 1.5, 1, 2]  # deweight the 4's
>>> weighted_mode(x, weights)
(array([2.]), array([3.5]))

The value 2 has the highest score: it appears twice with weights of
1.5 and 2: the sum of these is 3.5.

See Also
--------
scipy.stats.mode
*)


end

module Fixes : sig
module LooseVersion : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?vstring:Py.Object.t -> unit -> t
(**
Version numbering for anarchists and software realists.
Implements the standard interface for version number classes as
described above.  A version number consists of a series of numbers,
separated by either periods or strings of letters.  When comparing
version numbers, the numeric components will be compared
numerically, and the alphabetic components lexically.  The following
are all valid version numbers, in no particular order:

    1.5.1
    1.5.2b2
    161
    3.10a
    8.02
    3.4j
    1996.07.12
    3.2.pl0
    3.1.1.6
    2g6
    11g
    0.960923
    2.2beta29
    1.13++
    5.5.kw
    2.0b1pl0

In fact, there is no such thing as an invalid version number under
this scheme; the rules for comparison are simple and predictable,
but may not always give the results you want (for some definition
of "want").
*)

val parse : vstring:Py.Object.t -> t -> Py.Object.t
(**
None
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

module MaskedArray : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?data:Ndarray.t -> ?mask:Py.Object.t -> ?dtype:Py.Object.t -> ?copy:Py.Object.t -> ?subok:Py.Object.t -> ?ndmin:Py.Object.t -> ?fill_value:Py.Object.t -> ?keep_mask:Py.Object.t -> ?hard_mask:Py.Object.t -> ?shrink:Py.Object.t -> ?order:Py.Object.t -> ?options:(string * Py.Object.t) list -> unit -> t
(**
An array class with possibly masked values.

Masked values of True exclude the corresponding element from any
computation.

Construction::

  x = MaskedArray(data, mask=nomask, dtype=None, copy=False, subok=True,
                  ndmin=0, fill_value=None, keep_mask=True, hard_mask=None,
                  shrink=True, order=None)

Parameters
----------
data : array_like
    Input data.
mask : sequence, optional
    Mask. Must be convertible to an array of booleans with the same
    shape as `data`. True indicates a masked (i.e. invalid) data.
dtype : dtype, optional
    Data type of the output.
    If `dtype` is None, the type of the data argument (``data.dtype``)
    is used. If `dtype` is not None and different from ``data.dtype``,
    a copy is performed.
copy : bool, optional
    Whether to copy the input data (True), or to use a reference instead.
    Default is False.
subok : bool, optional
    Whether to return a subclass of `MaskedArray` if possible (True) or a
    plain `MaskedArray`. Default is True.
ndmin : int, optional
    Minimum number of dimensions. Default is 0.
fill_value : scalar, optional
    Value used to fill in the masked values when necessary.
    If None, a default based on the data-type is used.
keep_mask : bool, optional
    Whether to combine `mask` with the mask of the input data, if any
    (True), or to use only `mask` for the output (False). Default is True.
hard_mask : bool, optional
    Whether to use a hard mask or not. With a hard mask, masked values
    cannot be unmasked. Default is False.
shrink : bool, optional
    Whether to force compression of an empty mask. Default is True.
order : {'C', 'F', 'A'}, optional
    Specify the order of the array.  If order is 'C', then the array
    will be in C-contiguous order (last-index varies the fastest).
    If order is 'F', then the returned array will be in
    Fortran-contiguous order (first-index varies the fastest).
    If order is 'A' (default), then the returned array may be
    in any order (either C-, Fortran-contiguous, or even discontiguous),
    unless a copy is required, in which case it will be C-contiguous.
*)

val get_item : indx:Py.Object.t -> t -> Py.Object.t
(**
x.__getitem__(y) <==> x[y]

Return the item described by i, as a masked array.
*)

val all : ?axis:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> t -> Py.Object.t
(**
Returns True if all elements evaluate to True.

The output array is masked where all the values along the given axis
are masked: if the output would have been a scalar and that all the
values are masked, then the output is `masked`.

Refer to `numpy.all` for full documentation.

See Also
--------
numpy.ndarray.all : corresponding function for ndarrays
numpy.all : equivalent function

Examples
--------
>>> np.ma.array([1,2,3]).all()
True
>>> a = np.ma.array([1,2,3], mask=True)
>>> (a.all() is np.ma.masked)
True
*)

val anom : ?axis:int -> ?dtype:Py.Object.t -> t -> Py.Object.t
(**
Compute the anomalies (deviations from the arithmetic mean)
along the given axis.

Returns an array of anomalies, with the same shape as the input and
where the arithmetic mean is computed along the given axis.

Parameters
----------
axis : int, optional
    Axis over which the anomalies are taken.
    The default is to use the mean of the flattened array as reference.
dtype : dtype, optional
    Type to use in computing the variance. For arrays of integer type
     the default is float32; for arrays of float types it is the same as
     the array type.

See Also
--------
mean : Compute the mean of the array.

Examples
--------
>>> a = np.ma.array([1,2,3])
>>> a.anom()
masked_array(data=[-1.,  0.,  1.],
             mask=False,
       fill_value=1e+20)
*)

val any : ?axis:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> t -> Py.Object.t
(**
Returns True if any of the elements of `a` evaluate to True.

Masked values are considered as False during computation.

Refer to `numpy.any` for full documentation.

See Also
--------
numpy.ndarray.any : corresponding function for ndarrays
numpy.any : equivalent function
*)

val argmax : ?axis:[`Int of int | `None] -> ?fill_value:Py.Object.t -> ?out:Py.Object.t -> t -> Py.Object.t
(**
Returns array of indices of the maximum values along the given axis.
Masked values are treated as if they had the value fill_value.

Parameters
----------
axis : {None, integer}
    If None, the index is into the flattened array, otherwise along
    the specified axis
fill_value : {var}, optional
    Value used to fill in the masked values.  If None, the output of
    maximum_fill_value(self._data) is used instead.
out : {None, array}, optional
    Array into which the result can be placed. Its type is preserved
    and it must be of the right shape to hold the output.

Returns
-------
index_array : {integer_array}

Examples
--------
>>> a = np.arange(6).reshape(2,3)
>>> a.argmax()
5
>>> a.argmax(0)
array([1, 1, 1])
>>> a.argmax(1)
array([2, 2])
*)

val argmin : ?axis:[`Int of int | `None] -> ?fill_value:Py.Object.t -> ?out:Py.Object.t -> t -> Py.Object.t
(**
Return array of indices to the minimum values along the given axis.

Parameters
----------
axis : {None, integer}
    If None, the index is into the flattened array, otherwise along
    the specified axis
fill_value : {var}, optional
    Value used to fill in the masked values.  If None, the output of
    minimum_fill_value(self._data) is used instead.
out : {None, array}, optional
    Array into which the result can be placed. Its type is preserved
    and it must be of the right shape to hold the output.

Returns
-------
ndarray or scalar
    If multi-dimension input, returns a new ndarray of indices to the
    minimum values along the given axis.  Otherwise, returns a scalar
    of index to the minimum values along the given axis.

Examples
--------
>>> x = np.ma.array(np.arange(4), mask=[1,1,0,0])
>>> x.shape = (2,2)
>>> x
masked_array(
  data=[[--, --],
        [2, 3]],
  mask=[[ True,  True],
        [False, False]],
  fill_value=999999)
>>> x.argmin(axis=0, fill_value=-1)
array([0, 0])
>>> x.argmin(axis=0, fill_value=9)
array([1, 1])
*)

val argpartition : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> t -> Py.Object.t
(**
a.argpartition(kth, axis=-1, kind='introselect', order=None)

Returns the indices that would partition this array.

Refer to `numpy.argpartition` for full documentation.

.. versionadded:: 1.8.0

See Also
--------
numpy.argpartition : equivalent function
*)

val argsort : ?axis:int -> ?kind:[`Quicksort | `Mergesort | `Heapsort | `Stable] -> ?order:Py.Object.t -> ?endwith:bool -> ?fill_value:Py.Object.t -> t -> Py.Object.t
(**
Return an ndarray of indices that sort the array along the
specified axis.  Masked values are filled beforehand to
`fill_value`.

Parameters
----------
axis : int, optional
    Axis along which to sort. If None, the default, the flattened array
    is used.

    ..  versionchanged:: 1.13.0
        Previously, the default was documented to be -1, but that was
        in error. At some future date, the default will change to -1, as
        originally intended.
        Until then, the axis should be given explicitly when
        ``arr.ndim > 1``, to avoid a FutureWarning.
kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
    The sorting algorithm used.
order : list, optional
    When `a` is an array with fields defined, this argument specifies
    which fields to compare first, second, etc.  Not all fields need be
    specified.
endwith : {True, False}, optional
    Whether missing values (if any) should be treated as the largest values
    (True) or the smallest values (False)
    When the array contains unmasked values at the same extremes of the
    datatype, the ordering of these values and the masked values is
    undefined.
fill_value : {var}, optional
    Value used internally for the masked values.
    If ``fill_value`` is not None, it supersedes ``endwith``.

Returns
-------
index_array : ndarray, int
    Array of indices that sort `a` along the specified axis.
    In other words, ``a[index_array]`` yields a sorted `a`.

See Also
--------
MaskedArray.sort : Describes sorting algorithms used.
lexsort : Indirect stable sort with multiple keys.
numpy.ndarray.sort : Inplace sort.

Notes
-----
See `sort` for notes on the different sorting algorithms.

Examples
--------
>>> a = np.ma.array([3,2,1], mask=[False, False, True])
>>> a
masked_array(data=[3, 2, --],
             mask=[False, False,  True],
       fill_value=999999)
>>> a.argsort()
array([1, 0, 2])
*)

val compress : ?axis:Py.Object.t -> ?out:Py.Object.t -> condition:Py.Object.t -> t -> Py.Object.t
(**
Return `a` where condition is ``True``.

If condition is a `MaskedArray`, missing values are considered
as ``False``.

Parameters
----------
condition : var
    Boolean 1-d array selecting which entries to return. If len(condition)
    is less than the size of a along the axis, then output is truncated
    to length of condition array.
axis : {None, int}, optional
    Axis along which the operation must be performed.
out : {None, ndarray}, optional
    Alternative output array in which to place the result. It must have
    the same shape as the expected output but the type will be cast if
    necessary.

Returns
-------
result : MaskedArray
    A :class:`MaskedArray` object.

Notes
-----
Please note the difference with :meth:`compressed` !
The output of :meth:`compress` has a mask, the output of
:meth:`compressed` does not.

Examples
--------
>>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
>>> x
masked_array(
  data=[[1, --, 3],
        [--, 5, --],
        [7, --, 9]],
  mask=[[False,  True, False],
        [ True, False,  True],
        [False,  True, False]],
  fill_value=999999)
>>> x.compress([1, 0, 1])
masked_array(data=[1, 3],
             mask=[False, False],
       fill_value=999999)

>>> x.compress([1, 0, 1], axis=1)
masked_array(
  data=[[1, 3],
        [--, --],
        [7, 9]],
  mask=[[False, False],
        [ True,  True],
        [False, False]],
  fill_value=999999)
*)

val compressed : t -> Ndarray.t
(**
Return all the non-masked data as a 1-D array.

Returns
-------
data : ndarray
    A new `ndarray` holding the non-masked data is returned.

Notes
-----
The result is **not** a MaskedArray!

Examples
--------
>>> x = np.ma.array(np.arange(5), mask=[0]*2 + [1]*3)
>>> x.compressed()
array([0, 1])
>>> type(x.compressed())
<class 'numpy.ndarray'>
*)

val copy : ?params:(string * Py.Object.t) list -> Py.Object.t list -> t -> Py.Object.t
(**
a.copy(order='C')

Return a copy of the array.

Parameters
----------
order : {'C', 'F', 'A', 'K'}, optional
    Controls the memory layout of the copy. 'C' means C-order,
    'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
    'C' otherwise. 'K' means match the layout of `a` as closely
    as possible. (Note that this function and :func:`numpy.copy` are very
    similar, but have different default values for their order=
    arguments.)

See also
--------
numpy.copy
numpy.copyto

Examples
--------
>>> x = np.array([[1,2,3],[4,5,6]], order='F')

>>> y = x.copy()

>>> x.fill(0)

>>> x
array([[0, 0, 0],
       [0, 0, 0]])

>>> y
array([[1, 2, 3],
       [4, 5, 6]])

>>> y.flags['C_CONTIGUOUS']
True
*)

val count : ?axis:[`Int of int | `None | `PyObject of Py.Object.t] -> ?keepdims:bool -> t -> Py.Object.t
(**
Count the non-masked elements of the array along the given axis.

Parameters
----------
axis : None or int or tuple of ints, optional
    Axis or axes along which the count is performed.
    The default, None, performs the count over all
    the dimensions of the input array. `axis` may be negative, in
    which case it counts from the last to the first axis.

    .. versionadded:: 1.10.0

    If this is a tuple of ints, the count is performed on multiple
    axes, instead of a single axis or all the axes as before.
keepdims : bool, optional
    If this is set to True, the axes which are reduced are left
    in the result as dimensions with size one. With this option,
    the result will broadcast correctly against the array.

Returns
-------
result : ndarray or scalar
    An array with the same shape as the input array, with the specified
    axis removed. If the array is a 0-d array, or if `axis` is None, a
    scalar is returned.

See Also
--------
count_masked : Count masked elements in array or along a given axis.

Examples
--------
>>> import numpy.ma as ma
>>> a = ma.arange(6).reshape((2, 3))
>>> a[1, :] = ma.masked
>>> a
masked_array(
  data=[[0, 1, 2],
        [--, --, --]],
  mask=[[False, False, False],
        [ True,  True,  True]],
  fill_value=999999)
>>> a.count()
3

When the `axis` keyword is specified an array of appropriate size is
returned.

>>> a.count(axis=0)
array([1, 1, 1])
>>> a.count(axis=1)
array([3, 0])
*)

val cumprod : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> t -> Py.Object.t
(**
Return the cumulative product of the array elements over the given axis.

Masked values are set to 1 internally during the computation.
However, their position is saved, and the result will be masked at
the same locations.

Refer to `numpy.cumprod` for full documentation.

Notes
-----
The mask is lost if `out` is not a valid MaskedArray !

Arithmetic is modular when using integer types, and no error is
raised on overflow.

See Also
--------
numpy.ndarray.cumprod : corresponding function for ndarrays
numpy.cumprod : equivalent function
*)

val cumsum : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> t -> Py.Object.t
(**
Return the cumulative sum of the array elements over the given axis.

Masked values are set to 0 internally during the computation.
However, their position is saved, and the result will be masked at
the same locations.

Refer to `numpy.cumsum` for full documentation.

Notes
-----
The mask is lost if `out` is not a valid :class:`MaskedArray` !

Arithmetic is modular when using integer types, and no error is
raised on overflow.

See Also
--------
numpy.ndarray.cumsum : corresponding function for ndarrays
numpy.cumsum : equivalent function

Examples
--------
>>> marr = np.ma.array(np.arange(10), mask=[0,0,0,1,1,1,0,0,0,0])
>>> marr.cumsum()
masked_array(data=[0, 1, 3, --, --, --, 9, 16, 24, 33],
             mask=[False, False, False,  True,  True,  True, False, False,
                   False, False],
       fill_value=999999)
*)

val diagonal : ?params:(string * Py.Object.t) list -> Py.Object.t list -> t -> Py.Object.t
(**
a.diagonal(offset=0, axis1=0, axis2=1)

Return specified diagonals. In NumPy 1.9 the returned array is a
read-only view instead of a copy as in previous NumPy versions.  In
a future version the read-only restriction will be removed.

Refer to :func:`numpy.diagonal` for full documentation.

See Also
--------
numpy.diagonal : equivalent function
*)

val dot : ?out:Py.Object.t -> ?strict:Py.Object.t -> b:Py.Object.t -> t -> Py.Object.t
(**
a.dot(b, out=None)

Masked dot product of two arrays. Note that `out` and `strict` are
located in different positions than in `ma.dot`. In order to
maintain compatibility with the functional version, it is
recommended that the optional arguments be treated as keyword only.
At some point that may be mandatory.

.. versionadded:: 1.10.0

Parameters
----------
b : masked_array_like
    Inputs array.
out : masked_array, optional
    Output argument. This must have the exact kind that would be
    returned if it was not used. In particular, it must have the
    right type, must be C-contiguous, and its dtype must be the
    dtype that would be returned for `ma.dot(a,b)`. This is a
    performance feature. Therefore, if these conditions are not
    met, an exception is raised, instead of attempting to be
    flexible.
strict : bool, optional
    Whether masked data are propagated (True) or set to 0 (False)
    for the computation. Default is False.  Propagating the mask
    means that if a masked value appears in a row or column, the
    whole row or column is considered masked.

    .. versionadded:: 1.10.2

See Also
--------
numpy.ma.dot : equivalent function
*)

val filled : ?fill_value:Ndarray.t -> t -> Ndarray.t
(**
Return a copy of self, with masked values filled with a given value.
**However**, if there are no masked values to fill, self will be
returned instead as an ndarray.

Parameters
----------
fill_value : array_like, optional
    The value to use for invalid entries. Can be scalar or non-scalar.
    If non-scalar, the resulting ndarray must be broadcastable over
    input array. Default is None, in which case, the `fill_value`
    attribute of the array is used instead.

Returns
-------
filled_array : ndarray
    A copy of ``self`` with invalid entries replaced by *fill_value*
    (be it the function argument or the attribute of ``self``), or
    ``self`` itself as an ndarray if there are no invalid entries to
    be replaced.

Notes
-----
The result is **not** a MaskedArray!

Examples
--------
>>> x = np.ma.array([1,2,3,4,5], mask=[0,0,1,0,1], fill_value=-999)
>>> x.filled()
array([   1,    2, -999,    4, -999])
>>> x.filled(fill_value=1000)
array([   1,    2, 1000,    4, 1000])
>>> type(x.filled())
<class 'numpy.ndarray'>

Subclassing is preserved. This means that if, e.g., the data part of
the masked array is a recarray, `filled` returns a recarray:

>>> x = np.array([(-1, 2), (-3, 4)], dtype='i8,i8').view(np.recarray)
>>> m = np.ma.array(x, mask=[(True, False), (False, True)])
>>> m.filled()
rec.array([(999999,      2), (    -3, 999999)],
          dtype=[('f0', '<i8'), ('f1', '<i8')])
*)

val flatten : ?params:(string * Py.Object.t) list -> Py.Object.t list -> t -> Ndarray.t
(**
a.flatten(order='C')

Return a copy of the array collapsed into one dimension.

Parameters
----------
order : {'C', 'F', 'A', 'K'}, optional
    'C' means to flatten in row-major (C-style) order.
    'F' means to flatten in column-major (Fortran-
    style) order. 'A' means to flatten in column-major
    order if `a` is Fortran *contiguous* in memory,
    row-major order otherwise. 'K' means to flatten
    `a` in the order the elements occur in memory.
    The default is 'C'.

Returns
-------
y : ndarray
    A copy of the input array, flattened to one dimension.

See Also
--------
ravel : Return a flattened array.
flat : A 1-D flat iterator over the array.

Examples
--------
>>> a = np.array([[1,2], [3,4]])
>>> a.flatten()
array([1, 2, 3, 4])
>>> a.flatten('F')
array([1, 3, 2, 4])
*)

val get_fill_value : t -> Py.Object.t
(**
The filling value of the masked array is a scalar. When setting, None
will set to a default based on the data type.

Examples
--------
>>> for dt in [np.int32, np.int64, np.float64, np.complex128]:
...     np.ma.array([0, 1], dtype=dt).get_fill_value()
...
999999
999999
1e+20
(1e+20+0j)

>>> x = np.ma.array([0, 1.], fill_value=-np.inf)
>>> x.fill_value
-inf
>>> x.fill_value = np.pi
>>> x.fill_value
3.1415926535897931 # may vary

Reset to default:

>>> x.fill_value = None
>>> x.fill_value
1e+20
*)

val get_imag : t -> Py.Object.t
(**
The imaginary part of the masked array.

This property is a view on the imaginary part of this `MaskedArray`.

See Also
--------
real

Examples
--------
>>> x = np.ma.array([1+1.j, -2j, 3.45+1.6j], mask=[False, True, False])
>>> x.imag
masked_array(data=[1.0, --, 1.6],
             mask=[False,  True, False],
       fill_value=1e+20)
*)

val get_real : t -> Py.Object.t
(**
The real part of the masked array.

This property is a view on the real part of this `MaskedArray`.

See Also
--------
imag

Examples
--------
>>> x = np.ma.array([1+1.j, -2j, 3.45+1.6j], mask=[False, True, False])
>>> x.real
masked_array(data=[1.0, --, 3.45],
             mask=[False,  True, False],
       fill_value=1e+20)
*)

val harden_mask : t -> Py.Object.t
(**
Force the mask to hard.

Whether the mask of a masked array is hard or soft is determined by
its `hardmask` property. `harden_mask` sets `hardmask` to True.

See Also
--------
hardmask
*)

val ids : t -> Py.Object.t
(**
Return the addresses of the data and mask areas.

Parameters
----------
None

Examples
--------
>>> x = np.ma.array([1, 2, 3], mask=[0, 1, 1])
>>> x.ids()
(166670640, 166659832) # may vary

If the array has no mask, the address of `nomask` is returned. This address
is typically not close to the data in memory:

>>> x = np.ma.array([1, 2, 3])
>>> x.ids()
(166691080, 3083169284L) # may vary
*)

val iscontiguous : t -> Py.Object.t
(**
Return a boolean indicating whether the data is contiguous.

Parameters
----------
None

Examples
--------
>>> x = np.ma.array([1, 2, 3])
>>> x.iscontiguous()
True

`iscontiguous` returns one of the flags of the masked array:

>>> x.flags
  C_CONTIGUOUS : True
  F_CONTIGUOUS : True
  OWNDATA : False
  WRITEABLE : True
  ALIGNED : True
  WRITEBACKIFCOPY : False
  UPDATEIFCOPY : False
*)

val max : ?axis:[`Int of int | `None] -> ?out:Py.Object.t -> ?fill_value:Py.Object.t -> ?keepdims:Py.Object.t -> t -> Ndarray.t
(**
Return the maximum along a given axis.

Parameters
----------
axis : {None, int}, optional
    Axis along which to operate.  By default, ``axis`` is None and the
    flattened input is used.
out : array_like, optional
    Alternative output array in which to place the result.  Must
    be of the same shape and buffer length as the expected output.
fill_value : {var}, optional
    Value used to fill in the masked values.
    If None, use the output of maximum_fill_value().
keepdims : bool, optional
    If this is set to True, the axes which are reduced are left
    in the result as dimensions with size one. With this option,
    the result will broadcast correctly against the array.

Returns
-------
amax : array_like
    New array holding the result.
    If ``out`` was specified, ``out`` is returned.

See Also
--------
maximum_fill_value
    Returns the maximum filling value for a given datatype.
*)

val mean : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> t -> Py.Object.t
(**
Returns the average of the array elements along given axis.

Masked entries are ignored, and result elements which are not
finite will be masked.

Refer to `numpy.mean` for full documentation.

See Also
--------
numpy.ndarray.mean : corresponding function for ndarrays
numpy.mean : Equivalent function
numpy.ma.average: Weighted average.

Examples
--------
>>> a = np.ma.array([1,2,3], mask=[False, False, True])
>>> a
masked_array(data=[1, 2, --],
             mask=[False, False,  True],
       fill_value=999999)
>>> a.mean()
1.5
*)

val min : ?axis:[`Int of int | `None] -> ?out:Py.Object.t -> ?fill_value:Py.Object.t -> ?keepdims:Py.Object.t -> t -> Ndarray.t
(**
Return the minimum along a given axis.

Parameters
----------
axis : {None, int}, optional
    Axis along which to operate.  By default, ``axis`` is None and the
    flattened input is used.
out : array_like, optional
    Alternative output array in which to place the result.  Must be of
    the same shape and buffer length as the expected output.
fill_value : {var}, optional
    Value used to fill in the masked values.
    If None, use the output of `minimum_fill_value`.
keepdims : bool, optional
    If this is set to True, the axes which are reduced are left
    in the result as dimensions with size one. With this option,
    the result will broadcast correctly against the array.

Returns
-------
amin : array_like
    New array holding the result.
    If ``out`` was specified, ``out`` is returned.

See Also
--------
minimum_fill_value
    Returns the minimum filling value for a given datatype.
*)

val mini : ?axis:int -> t -> Py.Object.t
(**
Return the array minimum along the specified axis.

.. deprecated:: 1.13.0
   This function is identical to both:

    * ``self.min(keepdims=True, axis=axis).squeeze(axis=axis)``
    * ``np.ma.minimum.reduce(self, axis=axis)``

   Typically though, ``self.min(axis=axis)`` is sufficient.

Parameters
----------
axis : int, optional
    The axis along which to find the minima. Default is None, in which case
    the minimum value in the whole array is returned.

Returns
-------
min : scalar or MaskedArray
    If `axis` is None, the result is a scalar. Otherwise, if `axis` is
    given and the array is at least 2-D, the result is a masked array with
    dimension one smaller than the array on which `mini` is called.

Examples
--------
>>> x = np.ma.array(np.arange(6), mask=[0 ,1, 0, 0, 0 ,1]).reshape(3, 2)
>>> x
masked_array(
  data=[[0, --],
        [2, 3],
        [4, --]],
  mask=[[False,  True],
        [False, False],
        [False,  True]],
  fill_value=999999)
>>> x.mini()
masked_array(data=0,
             mask=False,
       fill_value=999999)
>>> x.mini(axis=0)
masked_array(data=[0, 3],
             mask=[False, False],
       fill_value=999999)
>>> x.mini(axis=1)
masked_array(data=[0, 2, 4],
             mask=[False, False, False],
       fill_value=999999)

There is a small difference between `mini` and `min`:

>>> x[:,1].mini(axis=0)
masked_array(data=3,
             mask=False,
       fill_value=999999)
>>> x[:,1].min(axis=0)
3
*)

val nonzero : t -> Py.Object.t
(**
Return the indices of unmasked elements that are not zero.

Returns a tuple of arrays, one for each dimension, containing the
indices of the non-zero elements in that dimension. The corresponding
non-zero values can be obtained with::

    a[a.nonzero()]

To group the indices by element, rather than dimension, use
instead::

    np.transpose(a.nonzero())

The result of this is always a 2d array, with a row for each non-zero
element.

Parameters
----------
None

Returns
-------
tuple_of_arrays : tuple
    Indices of elements that are non-zero.

See Also
--------
numpy.nonzero :
    Function operating on ndarrays.
flatnonzero :
    Return indices that are non-zero in the flattened version of the input
    array.
numpy.ndarray.nonzero :
    Equivalent ndarray method.
count_nonzero :
    Counts the number of non-zero elements in the input array.

Examples
--------
>>> import numpy.ma as ma
>>> x = ma.array(np.eye(3))
>>> x
masked_array(
  data=[[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]],
  mask=False,
  fill_value=1e+20)
>>> x.nonzero()
(array([0, 1, 2]), array([0, 1, 2]))

Masked elements are ignored.

>>> x[1, 1] = ma.masked
>>> x
masked_array(
  data=[[1.0, 0.0, 0.0],
        [0.0, --, 0.0],
        [0.0, 0.0, 1.0]],
  mask=[[False, False, False],
        [False,  True, False],
        [False, False, False]],
  fill_value=1e+20)
>>> x.nonzero()
(array([0, 2]), array([0, 2]))

Indices can also be grouped by element.

>>> np.transpose(x.nonzero())
array([[0, 0],
       [2, 2]])

A common use for ``nonzero`` is to find the indices of an array, where
a condition is True.  Given an array `a`, the condition `a` > 3 is a
boolean array and since False is interpreted as 0, ma.nonzero(a > 3)
yields the indices of the `a` where the condition is true.

>>> a = ma.array([[1,2,3],[4,5,6],[7,8,9]])
>>> a > 3
masked_array(
  data=[[False, False, False],
        [ True,  True,  True],
        [ True,  True,  True]],
  mask=False,
  fill_value=True)
>>> ma.nonzero(a > 3)
(array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))

The ``nonzero`` method of the condition array can also be called.

>>> (a > 3).nonzero()
(array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))
*)

val partition : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> t -> Py.Object.t
(**
a.partition(kth, axis=-1, kind='introselect', order=None)

Rearranges the elements in the array in such a way that the value of the
element in kth position is in the position it would be in a sorted array.
All elements smaller than the kth element are moved before this element and
all equal or greater are moved behind it. The ordering of the elements in
the two partitions is undefined.

.. versionadded:: 1.8.0

Parameters
----------
kth : int or sequence of ints
    Element index to partition by. The kth element value will be in its
    final sorted position and all smaller elements will be moved before it
    and all equal or greater elements behind it.
    The order of all elements in the partitions is undefined.
    If provided with a sequence of kth it will partition all elements
    indexed by kth of them into their sorted position at once.
axis : int, optional
    Axis along which to sort. Default is -1, which means sort along the
    last axis.
kind : {'introselect'}, optional
    Selection algorithm. Default is 'introselect'.
order : str or list of str, optional
    When `a` is an array with fields defined, this argument specifies
    which fields to compare first, second, etc. A single field can
    be specified as a string, and not all fields need to be specified,
    but unspecified fields will still be used, in the order in which
    they come up in the dtype, to break ties.

See Also
--------
numpy.partition : Return a parititioned copy of an array.
argpartition : Indirect partition.
sort : Full sort.

Notes
-----
See ``np.partition`` for notes on the different algorithms.

Examples
--------
>>> a = np.array([3, 4, 2, 1])
>>> a.partition(3)
>>> a
array([2, 1, 3, 4])

>>> a.partition((1, 3))
>>> a
array([1, 2, 3, 4])
*)

val prod : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> t -> Py.Object.t
(**
Return the product of the array elements over the given axis.

Masked elements are set to 1 internally for computation.

Refer to `numpy.prod` for full documentation.

Notes
-----
Arithmetic is modular when using integer types, and no error is raised
on overflow.

See Also
--------
numpy.ndarray.prod : corresponding function for ndarrays
numpy.prod : equivalent function
*)

val product : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> t -> Py.Object.t
(**
Return the product of the array elements over the given axis.

Masked elements are set to 1 internally for computation.

Refer to `numpy.prod` for full documentation.

Notes
-----
Arithmetic is modular when using integer types, and no error is raised
on overflow.

See Also
--------
numpy.ndarray.prod : corresponding function for ndarrays
numpy.prod : equivalent function
*)

val ptp : ?axis:[`Int of int | `None] -> ?out:Py.Object.t -> ?fill_value:Py.Object.t -> ?keepdims:Py.Object.t -> t -> Ndarray.t
(**
Return (maximum - minimum) along the given dimension
(i.e. peak-to-peak value).

Parameters
----------
axis : {None, int}, optional
    Axis along which to find the peaks.  If None (default) the
    flattened array is used.
out : {None, array_like}, optional
    Alternative output array in which to place the result. It must
    have the same shape and buffer length as the expected output
    but the type will be cast if necessary.
fill_value : {var}, optional
    Value used to fill in the masked values.
keepdims : bool, optional
    If this is set to True, the axes which are reduced are left
    in the result as dimensions with size one. With this option,
    the result will broadcast correctly against the array.

Returns
-------
ptp : ndarray.
    A new array holding the result, unless ``out`` was
    specified, in which case a reference to ``out`` is returned.
*)

val put : ?mode:Py.Object.t -> indices:Py.Object.t -> values:Py.Object.t -> t -> Py.Object.t
(**
Set storage-indexed locations to corresponding values.

Sets self._data.flat[n] = values[n] for each n in indices.
If `values` is shorter than `indices` then it will repeat.
If `values` has some masked values, the initial mask is updated
in consequence, else the corresponding values are unmasked.

Parameters
----------
indices : 1-D array_like
    Target indices, interpreted as integers.
values : array_like
    Values to place in self._data copy at target indices.
mode : {'raise', 'wrap', 'clip'}, optional
    Specifies how out-of-bounds indices will behave.
    'raise' : raise an error.
    'wrap' : wrap around.
    'clip' : clip to the range.

Notes
-----
`values` can be a scalar or length 1 array.

Examples
--------
>>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
>>> x
masked_array(
  data=[[1, --, 3],
        [--, 5, --],
        [7, --, 9]],
  mask=[[False,  True, False],
        [ True, False,  True],
        [False,  True, False]],
  fill_value=999999)
>>> x.put([0,4,8],[10,20,30])
>>> x
masked_array(
  data=[[10, --, 3],
        [--, 20, --],
        [7, --, 30]],
  mask=[[False,  True, False],
        [ True, False,  True],
        [False,  True, False]],
  fill_value=999999)

>>> x.put(4,999)
>>> x
masked_array(
  data=[[10, --, 3],
        [--, 999, --],
        [7, --, 30]],
  mask=[[False,  True, False],
        [ True, False,  True],
        [False,  True, False]],
  fill_value=999999)
*)

val ravel : ?order:[`C | `F | `A | `K] -> t -> Py.Object.t
(**
Returns a 1D version of self, as a view.

Parameters
----------
order : {'C', 'F', 'A', 'K'}, optional
    The elements of `a` are read using this index order. 'C' means to
    index the elements in C-like order, with the last axis index
    changing fastest, back to the first axis index changing slowest.
    'F' means to index the elements in Fortran-like index order, with
    the first index changing fastest, and the last index changing
    slowest. Note that the 'C' and 'F' options take no account of the
    memory layout of the underlying array, and only refer to the order
    of axis indexing.  'A' means to read the elements in Fortran-like
    index order if `m` is Fortran *contiguous* in memory, C-like order
    otherwise.  'K' means to read the elements in the order they occur
    in memory, except for reversing the data when strides are negative.
    By default, 'C' index order is used.

Returns
-------
MaskedArray
    Output view is of shape ``(self.size,)`` (or
    ``(np.ma.product(self.shape),)``).

Examples
--------
>>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
>>> x
masked_array(
  data=[[1, --, 3],
        [--, 5, --],
        [7, --, 9]],
  mask=[[False,  True, False],
        [ True, False,  True],
        [False,  True, False]],
  fill_value=999999)
>>> x.ravel()
masked_array(data=[1, --, 3, --, 5, --, 7, --, 9],
             mask=[False,  True, False,  True, False,  True, False,  True,
                   False],
       fill_value=999999)
*)

val repeat : ?params:(string * Py.Object.t) list -> Py.Object.t list -> t -> Py.Object.t
(**
a.repeat(repeats, axis=None)

Repeat elements of an array.

Refer to `numpy.repeat` for full documentation.

See Also
--------
numpy.repeat : equivalent function
*)

val reshape : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> t -> Ndarray.t
(**
Give a new shape to the array without changing its data.

Returns a masked array containing the same data, but with a new shape.
The result is a view on the original array; if this is not possible, a
ValueError is raised.

Parameters
----------
shape : int or tuple of ints
    The new shape should be compatible with the original shape. If an
    integer is supplied, then the result will be a 1-D array of that
    length.
order : {'C', 'F'}, optional
    Determines whether the array data should be viewed as in C
    (row-major) or FORTRAN (column-major) order.

Returns
-------
reshaped_array : array
    A new view on the array.

See Also
--------
reshape : Equivalent function in the masked array module.
numpy.ndarray.reshape : Equivalent method on ndarray object.
numpy.reshape : Equivalent function in the NumPy module.

Notes
-----
The reshaping operation cannot guarantee that a copy will not be made,
to modify the shape in place, use ``a.shape = s``

Examples
--------
>>> x = np.ma.array([[1,2],[3,4]], mask=[1,0,0,1])
>>> x
masked_array(
  data=[[--, 2],
        [3, --]],
  mask=[[ True, False],
        [False,  True]],
  fill_value=999999)
>>> x = x.reshape((4,1))
>>> x
masked_array(
  data=[[--],
        [2],
        [3],
        [--]],
  mask=[[ True],
        [False],
        [False],
        [ True]],
  fill_value=999999)
*)

val resize : ?refcheck:Py.Object.t -> ?order:Py.Object.t -> newshape:Py.Object.t -> t -> Py.Object.t
(**
.. warning::

    This method does nothing, except raise a ValueError exception. A
    masked array does not own its data and therefore cannot safely be
    resized in place. Use the `numpy.ma.resize` function instead.

This method is difficult to implement safely and may be deprecated in
future releases of NumPy.
*)

val round : ?decimals:Py.Object.t -> ?out:Py.Object.t -> t -> Py.Object.t
(**
Return each element rounded to the given number of decimals.

Refer to `numpy.around` for full documentation.

See Also
--------
numpy.ndarray.around : corresponding function for ndarrays
numpy.around : equivalent function
*)

val set_fill_value : ?value:Py.Object.t -> t -> Py.Object.t
(**
The filling value of the masked array is a scalar. When setting, None
will set to a default based on the data type.

Examples
--------
>>> for dt in [np.int32, np.int64, np.float64, np.complex128]:
...     np.ma.array([0, 1], dtype=dt).get_fill_value()
...
999999
999999
1e+20
(1e+20+0j)

>>> x = np.ma.array([0, 1.], fill_value=-np.inf)
>>> x.fill_value
-inf
>>> x.fill_value = np.pi
>>> x.fill_value
3.1415926535897931 # may vary

Reset to default:

>>> x.fill_value = None
>>> x.fill_value
1e+20
*)

val shrink_mask : t -> Py.Object.t
(**
Reduce a mask to nomask when possible.

Parameters
----------
None

Returns
-------
None

Examples
--------
>>> x = np.ma.array([[1,2 ], [3, 4]], mask=[0]*4)
>>> x.mask
array([[False, False],
       [False, False]])
>>> x.shrink_mask()
masked_array(
  data=[[1, 2],
        [3, 4]],
  mask=False,
  fill_value=999999)
>>> x.mask
False
*)

val soften_mask : t -> Py.Object.t
(**
Force the mask to soft.

Whether the mask of a masked array is hard or soft is determined by
its `hardmask` property. `soften_mask` sets `hardmask` to False.

See Also
--------
hardmask
*)

val sort : ?axis:Py.Object.t -> ?kind:Py.Object.t -> ?order:Py.Object.t -> ?endwith:Py.Object.t -> ?fill_value:Py.Object.t -> t -> Ndarray.t
(**
Sort the array, in-place

Parameters
----------
a : array_like
    Array to be sorted.
axis : int, optional
    Axis along which to sort. If None, the array is flattened before
    sorting. The default is -1, which sorts along the last axis.
kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
    The sorting algorithm used.
order : list, optional
    When `a` is a structured array, this argument specifies which fields
    to compare first, second, and so on.  This list does not need to
    include all of the fields.
endwith : {True, False}, optional
    Whether missing values (if any) should be treated as the largest values
    (True) or the smallest values (False)
    When the array contains unmasked values sorting at the same extremes of the
    datatype, the ordering of these values and the masked values is
    undefined.
fill_value : {var}, optional
    Value used internally for the masked values.
    If ``fill_value`` is not None, it supersedes ``endwith``.

Returns
-------
sorted_array : ndarray
    Array of the same type and shape as `a`.

See Also
--------
numpy.ndarray.sort : Method to sort an array in-place.
argsort : Indirect sort.
lexsort : Indirect stable sort on multiple keys.
searchsorted : Find elements in a sorted array.

Notes
-----
See ``sort`` for notes on the different sorting algorithms.

Examples
--------
>>> a = np.ma.array([1, 2, 5, 4, 3],mask=[0, 1, 0, 1, 0])
>>> # Default
>>> a.sort()
>>> a
masked_array(data=[1, 3, 5, --, --],
             mask=[False, False, False,  True,  True],
       fill_value=999999)

>>> a = np.ma.array([1, 2, 5, 4, 3],mask=[0, 1, 0, 1, 0])
>>> # Put missing values in the front
>>> a.sort(endwith=False)
>>> a
masked_array(data=[--, --, 1, 3, 5],
             mask=[ True,  True, False, False, False],
       fill_value=999999)

>>> a = np.ma.array([1, 2, 5, 4, 3],mask=[0, 1, 0, 1, 0])
>>> # fill_value takes over endwith
>>> a.sort(endwith=False, fill_value=3)
>>> a
masked_array(data=[1, --, --, 3, 5],
             mask=[False,  True,  True, False, False],
       fill_value=999999)
*)

val squeeze : ?params:(string * Py.Object.t) list -> Py.Object.t list -> t -> Py.Object.t
(**
a.squeeze(axis=None)

Remove single-dimensional entries from the shape of `a`.

Refer to `numpy.squeeze` for full documentation.

See Also
--------
numpy.squeeze : equivalent function
*)

val std : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> ?ddof:Py.Object.t -> ?keepdims:Py.Object.t -> t -> Py.Object.t
(**
Returns the standard deviation of the array elements along given axis.

Masked entries are ignored.

Refer to `numpy.std` for full documentation.

See Also
--------
numpy.ndarray.std : corresponding function for ndarrays
numpy.std : Equivalent function
*)

val sum : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> t -> Py.Object.t
(**
Return the sum of the array elements over the given axis.

Masked elements are set to 0 internally.

Refer to `numpy.sum` for full documentation.

See Also
--------
numpy.ndarray.sum : corresponding function for ndarrays
numpy.sum : equivalent function

Examples
--------
>>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
>>> x
masked_array(
  data=[[1, --, 3],
        [--, 5, --],
        [7, --, 9]],
  mask=[[False,  True, False],
        [ True, False,  True],
        [False,  True, False]],
  fill_value=999999)
>>> x.sum()
25
>>> x.sum(axis=1)
masked_array(data=[4, 5, 16],
             mask=[False, False, False],
       fill_value=999999)
>>> x.sum(axis=0)
masked_array(data=[8, 5, 12],
             mask=[False, False, False],
       fill_value=999999)
>>> print(type(x.sum(axis=0, dtype=np.int64)[0]))
<class 'numpy.int64'>
*)

val swapaxes : ?params:(string * Py.Object.t) list -> Py.Object.t list -> t -> Py.Object.t
(**
a.swapaxes(axis1, axis2)

Return a view of the array with `axis1` and `axis2` interchanged.

Refer to `numpy.swapaxes` for full documentation.

See Also
--------
numpy.swapaxes : equivalent function
*)

val take : ?axis:Py.Object.t -> ?out:Py.Object.t -> ?mode:Py.Object.t -> indices:Py.Object.t -> t -> Py.Object.t
(**
        
*)

val tobytes : ?fill_value:Py.Object.t -> ?order:Py.Object.t -> t -> Py.Object.t
(**
Return the array data as a string containing the raw bytes in the array.

The array is filled with a fill value before the string conversion.

.. versionadded:: 1.9.0

Parameters
----------
fill_value : scalar, optional
    Value used to fill in the masked values. Default is None, in which
    case `MaskedArray.fill_value` is used.
order : {'C','F','A'}, optional
    Order of the data item in the copy. Default is 'C'.

    - 'C'   -- C order (row major).
    - 'F'   -- Fortran order (column major).
    - 'A'   -- Any, current order of array.
    - None  -- Same as 'A'.

See Also
--------
numpy.ndarray.tobytes
tolist, tofile

Notes
-----
As for `ndarray.tobytes`, information about the shape, dtype, etc.,
but also about `fill_value`, will be lost.

Examples
--------
>>> x = np.ma.array(np.array([[1, 2], [3, 4]]), mask=[[0, 1], [1, 0]])
>>> x.tobytes()
b'\x01\x00\x00\x00\x00\x00\x00\x00?B\x0f\x00\x00\x00\x00\x00?B\x0f\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00'
*)

val tofile : ?sep:Py.Object.t -> ?format:Py.Object.t -> fid:Py.Object.t -> t -> Py.Object.t
(**
Save a masked array to a file in binary format.

.. warning::
  This function is not implemented yet.

Raises
------
NotImplementedError
    When `tofile` is called.
*)

val toflex : t -> Ndarray.t
(**
Transforms a masked array into a flexible-type array.

The flexible type array that is returned will have two fields:

* the ``_data`` field stores the ``_data`` part of the array.
* the ``_mask`` field stores the ``_mask`` part of the array.

Parameters
----------
None

Returns
-------
record : ndarray
    A new flexible-type `ndarray` with two fields: the first element
    containing a value, the second element containing the corresponding
    mask boolean. The returned record shape matches self.shape.

Notes
-----
A side-effect of transforming a masked array into a flexible `ndarray` is
that meta information (``fill_value``, ...) will be lost.

Examples
--------
>>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
>>> x
masked_array(
  data=[[1, --, 3],
        [--, 5, --],
        [7, --, 9]],
  mask=[[False,  True, False],
        [ True, False,  True],
        [False,  True, False]],
  fill_value=999999)
>>> x.toflex()
array([[(1, False), (2,  True), (3, False)],
       [(4,  True), (5, False), (6,  True)],
       [(7, False), (8,  True), (9, False)]],
      dtype=[('_data', '<i8'), ('_mask', '?')])
*)

val tolist : ?fill_value:Py.Object.t -> t -> Py.Object.t
(**
Return the data portion of the masked array as a hierarchical Python list.

Data items are converted to the nearest compatible Python type.
Masked values are converted to `fill_value`. If `fill_value` is None,
the corresponding entries in the output list will be ``None``.

Parameters
----------
fill_value : scalar, optional
    The value to use for invalid entries. Default is None.

Returns
-------
result : list
    The Python list representation of the masked array.

Examples
--------
>>> x = np.ma.array([[1,2,3], [4,5,6], [7,8,9]], mask=[0] + [1,0]*4)
>>> x.tolist()
[[1, None, 3], [None, 5, None], [7, None, 9]]
>>> x.tolist(-999)
[[1, -999, 3], [-999, 5, -999], [7, -999, 9]]
*)

val torecords : t -> Ndarray.t
(**
Transforms a masked array into a flexible-type array.

The flexible type array that is returned will have two fields:

* the ``_data`` field stores the ``_data`` part of the array.
* the ``_mask`` field stores the ``_mask`` part of the array.

Parameters
----------
None

Returns
-------
record : ndarray
    A new flexible-type `ndarray` with two fields: the first element
    containing a value, the second element containing the corresponding
    mask boolean. The returned record shape matches self.shape.

Notes
-----
A side-effect of transforming a masked array into a flexible `ndarray` is
that meta information (``fill_value``, ...) will be lost.

Examples
--------
>>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
>>> x
masked_array(
  data=[[1, --, 3],
        [--, 5, --],
        [7, --, 9]],
  mask=[[False,  True, False],
        [ True, False,  True],
        [False,  True, False]],
  fill_value=999999)
>>> x.toflex()
array([[(1, False), (2,  True), (3, False)],
       [(4,  True), (5, False), (6,  True)],
       [(7, False), (8,  True), (9, False)]],
      dtype=[('_data', '<i8'), ('_mask', '?')])
*)

val tostring : ?fill_value:Py.Object.t -> ?order:Py.Object.t -> t -> Py.Object.t
(**
This function is a compatibility alias for tobytes. Despite its name it
returns bytes not strings.
*)

val trace : ?offset:Py.Object.t -> ?axis1:Py.Object.t -> ?axis2:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> t -> Py.Object.t
(**
a.trace(offset=0, axis1=0, axis2=1, dtype=None, out=None)

Return the sum along diagonals of the array.

Refer to `numpy.trace` for full documentation.

See Also
--------
numpy.trace : equivalent function
*)

val transpose : ?params:(string * Py.Object.t) list -> Py.Object.t list -> t -> Ndarray.t
(**
a.transpose( *axes)

Returns a view of the array with axes transposed.

For a 1-D array this has no effect, as a transposed vector is simply the
same vector. To convert a 1-D array into a 2D column vector, an additional
dimension must be added. `np.atleast2d(a).T` achieves this, as does
`a[:, np.newaxis]`.
For a 2-D array, this is a standard matrix transpose.
For an n-D array, if axes are given, their order indicates how the
axes are permuted (see Examples). If axes are not provided and
``a.shape = (i[0], i[1], ... i[n-2], i[n-1])``, then
``a.transpose().shape = (i[n-1], i[n-2], ... i[1], i[0])``.

Parameters
----------
axes : None, tuple of ints, or `n` ints

 * None or no argument: reverses the order of the axes.

 * tuple of ints: `i` in the `j`-th place in the tuple means `a`'s
   `i`-th axis becomes `a.transpose()`'s `j`-th axis.

 * `n` ints: same as an n-tuple of the same ints (this form is
   intended simply as a "convenience" alternative to the tuple form)

Returns
-------
out : ndarray
    View of `a`, with axes suitably permuted.

See Also
--------
ndarray.T : Array property returning the array transposed.
ndarray.reshape : Give a new shape to an array without changing its data.

Examples
--------
>>> a = np.array([[1, 2], [3, 4]])
>>> a
array([[1, 2],
       [3, 4]])
>>> a.transpose()
array([[1, 3],
       [2, 4]])
>>> a.transpose((1, 0))
array([[1, 3],
       [2, 4]])
>>> a.transpose(1, 0)
array([[1, 3],
       [2, 4]])
*)

val unshare_mask : t -> Py.Object.t
(**
Copy the mask and set the sharedmask flag to False.

Whether the mask is shared between masked arrays can be seen from
the `sharedmask` property. `unshare_mask` ensures the mask is not shared.
A copy of the mask is only made if it was shared.

See Also
--------
sharedmask
*)

val var : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Ndarray.t -> ?ddof:int -> ?keepdims:bool -> t -> Py.Object.t
(**
Compute the variance along the specified axis.

Returns the variance of the array elements, a measure of the spread of a
distribution.  The variance is computed for the flattened array by
default, otherwise over the specified axis.

Parameters
----------
a : array_like
    Array containing numbers whose variance is desired.  If `a` is not an
    array, a conversion is attempted.
axis : None or int or tuple of ints, optional
    Axis or axes along which the variance is computed.  The default is to
    compute the variance of the flattened array.

    .. versionadded:: 1.7.0

    If this is a tuple of ints, a variance is performed over multiple axes,
    instead of a single axis or all the axes as before.
dtype : data-type, optional
    Type to use in computing the variance.  For arrays of integer type
    the default is `float64`; for arrays of float types it is the same as
    the array type.
out : ndarray, optional
    Alternate output array in which to place the result.  It must have
    the same shape as the expected output, but the type is cast if
    necessary.
ddof : int, optional
    "Delta Degrees of Freedom": the divisor used in the calculation is
    ``N - ddof``, where ``N`` represents the number of elements. By
    default `ddof` is zero.
keepdims : bool, optional
    If this is set to True, the axes which are reduced are left
    in the result as dimensions with size one. With this option,
    the result will broadcast correctly against the input array.

    If the default value is passed, then `keepdims` will not be
    passed through to the `var` method of sub-classes of
    `ndarray`, however any non-default value will be.  If the
    sub-class' method does not implement `keepdims` any
    exceptions will be raised.

Returns
-------
variance : ndarray, see dtype parameter above
    If ``out=None``, returns a new array containing the variance;
    otherwise, a reference to the output array is returned.

See Also
--------
std, mean, nanmean, nanstd, nanvar
ufuncs-output-type

Notes
-----
The variance is the average of the squared deviations from the mean,
i.e.,  ``var = mean(abs(x - x.mean())**2)``.

The mean is normally calculated as ``x.sum() / N``, where ``N = len(x)``.
If, however, `ddof` is specified, the divisor ``N - ddof`` is used
instead.  In standard statistical practice, ``ddof=1`` provides an
unbiased estimator of the variance of a hypothetical infinite population.
``ddof=0`` provides a maximum likelihood estimate of the variance for
normally distributed variables.

Note that for complex numbers, the absolute value is taken before
squaring, so that the result is always real and nonnegative.

For floating-point input, the variance is computed using the same
precision the input has.  Depending on the input data, this can cause
the results to be inaccurate, especially for `float32` (see example
below).  Specifying a higher-accuracy accumulator using the ``dtype``
keyword can alleviate this issue.

Examples
--------
>>> a = np.array([[1, 2], [3, 4]])
>>> np.var(a)
1.25
>>> np.var(a, axis=0)
array([1.,  1.])
>>> np.var(a, axis=1)
array([0.25,  0.25])

In single precision, var() can be inaccurate:

>>> a = np.zeros((2, 512*512), dtype=np.float32)
>>> a[0, :] = 1.0
>>> a[1, :] = 0.1
>>> np.var(a)
0.20250003

Computing the variance in float64 is more accurate:

>>> np.var(a, dtype=np.float64)
0.20249999932944759 # may vary
>>> ((1-0.55)**2 + (0.1-0.55)**2)/2
0.2025
*)

val view : ?dtype:Py.Object.t -> ?type_:Py.Object.t -> ?fill_value:Py.Object.t -> t -> Py.Object.t
(**
Return a view of the MaskedArray data.

Parameters
----------
dtype : data-type or ndarray sub-class, optional
    Data-type descriptor of the returned view, e.g., float32 or int16.
    The default, None, results in the view having the same data-type
    as `a`. As with ``ndarray.view``, dtype can also be specified as
    an ndarray sub-class, which then specifies the type of the
    returned object (this is equivalent to setting the ``type``
    parameter).
type : Python type, optional
    Type of the returned view, either ndarray or a subclass.  The
    default None results in type preservation.
fill_value : scalar, optional
    The value to use for invalid entries (None by default).
    If None, then this argument is inferred from the passed `dtype`, or
    in its absence the original array, as discussed in the notes below.

See Also
--------
numpy.ndarray.view : Equivalent method on ndarray object.

Notes
-----

``a.view()`` is used two different ways:

``a.view(some_dtype)`` or ``a.view(dtype=some_dtype)`` constructs a view
of the array's memory with a different data-type.  This can cause a
reinterpretation of the bytes of memory.

``a.view(ndarray_subclass)`` or ``a.view(type=ndarray_subclass)`` just
returns an instance of `ndarray_subclass` that looks at the same array
(same shape, dtype, etc.)  This does not cause a reinterpretation of the
memory.

If `fill_value` is not specified, but `dtype` is specified (and is not
an ndarray sub-class), the `fill_value` of the MaskedArray will be
reset. If neither `fill_value` nor `dtype` are specified (or if
`dtype` is an ndarray sub-class), then the fill value is preserved.
Finally, if `fill_value` is specified, but `dtype` is not, the fill
value is set to the specified value.

For ``a.view(some_dtype)``, if ``some_dtype`` has a different number of
bytes per entry than the previous dtype (for example, converting a
regular array to a structured array), then the behavior of the view
cannot be predicted just from the superficial appearance of ``a`` (shown
by ``print(a)``). It also depends on exactly how ``a`` is stored in
memory. Therefore if ``a`` is C-ordered versus fortran-ordered, versus
defined as a slice or transpose, etc., the view may give different
results.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

val comb : ?exact:Py.Object.t -> ?repetition:Py.Object.t -> n:[`Int of int | `Ndarray of Ndarray.t] -> k:Py.Object.t -> unit -> Py.Object.t
(**
The number of combinations of N things taken k at a time.

This is often expressed as "N choose k".

Parameters
----------
N : int, ndarray
    Number of things.
k : int, ndarray
    Number of elements taken.
exact : bool, optional
    If `exact` is False, then floating point precision is used, otherwise
    exact long integer is computed.
repetition : bool, optional
    If `repetition` is True, then the number of combinations with
    repetition is computed.

Returns
-------
val : int, float, ndarray
    The total number of combinations.

See Also
--------
binom : Binomial coefficient ufunc

Notes
-----
- Array arguments accepted only for exact=False case.
- If N < 0, or k < 0, then 0 is returned.
- If k > N and repetition=False, then 0 is returned.

Examples
--------
>>> from scipy.special import comb
>>> k = np.array([3, 4])
>>> n = np.array([10, 10])
>>> comb(n, k, exact=False)
array([ 120.,  210.])
>>> comb(10, 3, exact=True)
120L
>>> comb(10, 3, exact=True, repetition=True)
220L
*)

val lobpcg : ?b:Py.Object.t -> ?m:Py.Object.t -> ?y:Py.Object.t -> ?tol:Py.Object.t -> ?maxiter:Py.Object.t -> ?largest:Py.Object.t -> ?verbosityLevel:Py.Object.t -> ?retLambdaHistory:Py.Object.t -> ?retResidualNormsHistory:Py.Object.t -> a:[`SparseMatrix of Csr_matrix.t | `LinearOperator of Py.Object.t | `PyObject of Py.Object.t] -> x:Py.Object.t -> unit -> Ndarray.t
(**
Locally Optimal Block Preconditioned Conjugate Gradient Method (LOBPCG)

LOBPCG is a preconditioned eigensolver for large symmetric positive
definite (SPD) generalized eigenproblems.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The symmetric linear operator of the problem, usually a
    sparse matrix.  Often called the "stiffness matrix".
X : ndarray, float32 or float64
    Initial approximation to the ``k`` eigenvectors (non-sparse). If `A`
    has ``shape=(n,n)`` then `X` should have shape ``shape=(n,k)``.
B : {dense matrix, sparse matrix, LinearOperator}, optional
    The right hand side operator in a generalized eigenproblem.
    By default, ``B = Identity``.  Often called the "mass matrix".
M : {dense matrix, sparse matrix, LinearOperator}, optional
    Preconditioner to `A`; by default ``M = Identity``.
    `M` should approximate the inverse of `A`.
Y : ndarray, float32 or float64, optional
    n-by-sizeY matrix of constraints (non-sparse), sizeY < n
    The iterations will be performed in the B-orthogonal complement
    of the column-space of Y. Y must be full rank.
tol : scalar, optional
    Solver tolerance (stopping criterion).
    The default is ``tol=n*sqrt(eps)``.
maxiter : int, optional
    Maximum number of iterations.  The default is ``maxiter = 20``.
largest : bool, optional
    When True, solve for the largest eigenvalues, otherwise the smallest.
verbosityLevel : int, optional
    Controls solver output.  The default is ``verbosityLevel=0``.
retLambdaHistory : bool, optional
    Whether to return eigenvalue history.  Default is False.
retResidualNormsHistory : bool, optional
    Whether to return history of residual norms.  Default is False.

Returns
-------
w : ndarray
    Array of ``k`` eigenvalues
v : ndarray
    An array of ``k`` eigenvectors.  `v` has the same shape as `X`.
lambdas : list of ndarray, optional
    The eigenvalue history, if `retLambdaHistory` is True.
rnorms : list of ndarray, optional
    The history of residual norms, if `retResidualNormsHistory` is True.

Notes
-----
If both ``retLambdaHistory`` and ``retResidualNormsHistory`` are True,
the return tuple has the following format
``(lambda, V, lambda history, residual norms history)``.

In the following ``n`` denotes the matrix size and ``m`` the number
of required eigenvalues (smallest or largest).

The LOBPCG code internally solves eigenproblems of the size ``3m`` on every
iteration by calling the "standard" dense eigensolver, so if ``m`` is not
small enough compared to ``n``, it does not make sense to call the LOBPCG
code, but rather one should use the "standard" eigensolver, e.g. numpy or
scipy function in this case.
If one calls the LOBPCG algorithm for ``5m > n``, it will most likely break
internally, so the code tries to call the standard function instead.

It is not that ``n`` should be large for the LOBPCG to work, but rather the
ratio ``n / m`` should be large. It you call LOBPCG with ``m=1``
and ``n=10``, it works though ``n`` is small. The method is intended
for extremely large ``n / m``, see e.g., reference [28] in
https://arxiv.org/abs/0705.2626

The convergence speed depends basically on two factors:

1. How well relatively separated the seeking eigenvalues are from the rest
   of the eigenvalues. One can try to vary ``m`` to make this better.

2. How well conditioned the problem is. This can be changed by using proper
   preconditioning. For example, a rod vibration test problem (under tests
   directory) is ill-conditioned for large ``n``, so convergence will be
   slow, unless efficient preconditioning is used. For this specific
   problem, a good simple preconditioner function would be a linear solve
   for `A`, which is easy to code since A is tridiagonal.

References
----------
.. [1] A. V. Knyazev (2001),
       Toward the Optimal Preconditioned Eigensolver: Locally Optimal
       Block Preconditioned Conjugate Gradient Method.
       SIAM Journal on Scientific Computing 23, no. 2,
       pp. 517-541. http://dx.doi.org/10.1137/S1064827500366124

.. [2] A. V. Knyazev, I. Lashuk, M. E. Argentati, and E. Ovchinnikov
       (2007), Block Locally Optimal Preconditioned Eigenvalue Xolvers
       (BLOPEX) in hypre and PETSc. https://arxiv.org/abs/0705.2626

.. [3] A. V. Knyazev's C and MATLAB implementations:
       https://bitbucket.org/joseroman/blopex

Examples
--------

Solve ``A x = lambda x`` with constraints and preconditioning.

>>> import numpy as np
>>> from scipy.sparse import spdiags, issparse
>>> from scipy.sparse.linalg import lobpcg, LinearOperator
>>> n = 100
>>> vals = np.arange(1, n + 1)
>>> A = spdiags(vals, 0, n, n)
>>> A.toarray()
array([[  1.,   0.,   0., ...,   0.,   0.,   0.],
       [  0.,   2.,   0., ...,   0.,   0.,   0.],
       [  0.,   0.,   3., ...,   0.,   0.,   0.],
       ...,
       [  0.,   0.,   0., ...,  98.,   0.,   0.],
       [  0.,   0.,   0., ...,   0.,  99.,   0.],
       [  0.,   0.,   0., ...,   0.,   0., 100.]])

Constraints:

>>> Y = np.eye(n, 3)

Initial guess for eigenvectors, should have linearly independent
columns. Column dimension = number of requested eigenvalues.

>>> X = np.random.rand(n, 3)

Preconditioner in the inverse of A in this example:

>>> invA = spdiags([1./vals], 0, n, n)

The preconditiner must be defined by a function:

>>> def precond( x ):
...     return invA @ x

The argument x of the preconditioner function is a matrix inside `lobpcg`,
thus the use of matrix-matrix product ``@``.

The preconditioner function is passed to lobpcg as a `LinearOperator`:

>>> M = LinearOperator(matvec=precond, matmat=precond,
...                    shape=(n, n), dtype=float)

Let us now solve the eigenvalue problem for the matrix A:

>>> eigenvalues, _ = lobpcg(A, X, Y=Y, M=M, largest=False)
>>> eigenvalues
array([4., 5., 6.])

Note that the vectors passed in Y are the eigenvectors of the 3 smallest
eigenvalues. The results returned are orthogonal to those.
*)

val logsumexp : ?axis:Py.Object.t -> ?b:Ndarray.t -> ?keepdims:bool -> ?return_sign:bool -> a:Ndarray.t -> unit -> Ndarray.t
(**
Compute the log of the sum of exponentials of input elements.

Parameters
----------
a : array_like
    Input array.
axis : None or int or tuple of ints, optional
    Axis or axes over which the sum is taken. By default `axis` is None,
    and all elements are summed.

    .. versionadded:: 0.11.0
keepdims : bool, optional
    If this is set to True, the axes which are reduced are left in the
    result as dimensions with size one. With this option, the result
    will broadcast correctly against the original array.

    .. versionadded:: 0.15.0
b : array-like, optional
    Scaling factor for exp(`a`) must be of the same shape as `a` or
    broadcastable to `a`. These values may be negative in order to
    implement subtraction.

    .. versionadded:: 0.12.0
return_sign : bool, optional
    If this is set to True, the result will be a pair containing sign
    information; if False, results that are negative will be returned
    as NaN. Default is False (no sign information).

    .. versionadded:: 0.16.0

Returns
-------
res : ndarray
    The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
    more stable way. If `b` is given then ``np.log(np.sum(b*np.exp(a)))``
    is returned.
sgn : ndarray
    If return_sign is True, this will be an array of floating-point
    numbers matching res and +1, 0, or -1 depending on the sign
    of the result. If False, only one result is returned.

See Also
--------
numpy.logaddexp, numpy.logaddexp2

Notes
-----
NumPy has a logaddexp function which is very similar to `logsumexp`, but
only handles two arguments. `logaddexp.reduce` is similar to this
function, but may be less stable.

Examples
--------
>>> from scipy.special import logsumexp
>>> a = np.arange(10)
>>> np.log(np.sum(np.exp(a)))
9.4586297444267107
>>> logsumexp(a)
9.4586297444267107

With weights

>>> a = np.arange(10)
>>> b = np.arange(10, 0, -1)
>>> logsumexp(a, b=b)
9.9170178533034665
>>> np.log(np.sum(b*np.exp(a)))
9.9170178533034647

Returning a sign flag

>>> logsumexp([1,2],b=[1,-1],return_sign=True)
(1.5413248546129181, -1.0)

Notice that `logsumexp` does not directly support masked arrays. To use it
on a masked array, convert the mask into zero weights:

>>> a = np.ma.array([np.log(2), 2, np.log(3)],
...                  mask=[False, True, False])
>>> b = (~a.mask).astype(int)
>>> logsumexp(a.data, b=b), np.log(5)
1.6094379124341005, 1.6094379124341005
*)

val loguniform : ?kwds:(string * Py.Object.t) list -> Py.Object.t list -> Py.Object.t
(**
A loguniform or reciprocal continuous random variable.

As an instance of the `rv_continuous` class, `Distribution` object inherits from it
a collection of generic methods (see below for the full list),
and completes them with details specific for this particular distribution.

Methods
-------
rvs(a, b, loc=0, scale=1, size=1, random_state=None)
    Random variates.
pdf(x, a, b, loc=0, scale=1)
    Probability density function.
logpdf(x, a, b, loc=0, scale=1)
    Log of the probability density function.
cdf(x, a, b, loc=0, scale=1)
    Cumulative distribution function.
logcdf(x, a, b, loc=0, scale=1)
    Log of the cumulative distribution function.
sf(x, a, b, loc=0, scale=1)
    Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
logsf(x, a, b, loc=0, scale=1)
    Log of the survival function.
ppf(q, a, b, loc=0, scale=1)
    Percent point function (inverse of ``cdf`` --- percentiles).
isf(q, a, b, loc=0, scale=1)
    Inverse survival function (inverse of ``sf``).
moment(n, a, b, loc=0, scale=1)
    Non-central moment of order n
stats(a, b, loc=0, scale=1, moments='mv')
    Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
entropy(a, b, loc=0, scale=1)
    (Differential) entropy of the RV.
fit(data, a, b, loc=0, scale=1)
    Parameter estimates for generic data.
expect(func, args=(a, b), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
    Expected value of a function (of one argument) with respect to the distribution.
median(a, b, loc=0, scale=1)
    Median of the distribution.
mean(a, b, loc=0, scale=1)
    Mean of the distribution.
var(a, b, loc=0, scale=1)
    Variance of the distribution.
std(a, b, loc=0, scale=1)
    Standard deviation of the distribution.
interval(alpha, a, b, loc=0, scale=1)
    Endpoints of the range that contains alpha percent of the distribution

Notes
-----
The probability density function for this class is:

.. math::

    f(x, a, b) = \frac{1}{x \log(b/a)}

for :math:`a \le x \le b`, :math:`b > a > 0`. This class takes
:math:`a` and :math:`b` as shape parameters. The probability density above is defined in the "standardized" form. To shift
and/or scale the distribution use the ``loc`` and ``scale`` parameters.
Specifically, ``Distribution.pdf(x, a, b, loc, scale)`` is identically
equivalent to ``Distribution.pdf(y, a, b) / scale`` with
``y = (x - loc) / scale``.

Examples
--------
>>> from scipy.stats import Distribution
>>> import matplotlib.pyplot as plt
>>> fig, ax = plt.subplots(1, 1)

Calculate a few first moments:

>>> a, b = 
>>> mean, var, skew, kurt = Distribution.stats(a, b, moments='mvsk')

Display the probability density function (``pdf``):

>>> x = np.linspace(Distribution.ppf(0.01, a, b),
...                 Distribution.ppf(0.99, a, b), 100)
>>> ax.plot(x, Distribution.pdf(x, a, b),
...        'r-', lw=5, alpha=0.6, label='Distribution pdf')

Alternatively, the distribution object can be called (as a function)
to fix the shape, location and scale parameters. This returns a "frozen"
RV object holding the given parameters fixed.

Freeze the distribution and display the frozen ``pdf``:

>>> rv = Distribution(a, b)
>>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

Check accuracy of ``cdf`` and ``ppf``:

>>> vals = Distribution.ppf([0.001, 0.5, 0.999], a, b)
>>> np.allclose([0.001, 0.5, 0.999], Distribution.cdf(vals, a, b))
True

Generate random numbers:

>>> r = Distribution.rvs(a, b, size=1000)

And compare the histogram:

>>> ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
>>> ax.legend(loc='best', frameon=False)
>>> plt.show()


This doesn't show the equal probability of ``0.01``, ``0.1`` and
``1``. This is best when the x-axis is log-scaled:

>>> import numpy as np
>>> fig, ax = plt.subplots(1, 1)
>>> ax.hist(np.log10(r))
>>> ax.set_ylabel("Frequency")
>>> ax.set_xlabel("Value of random variable")
>>> ax.xaxis.set_major_locator(plt.FixedLocator([-2, -1, 0]))
>>> ticks = ["$10^{{ {} }}$".format(i) for i in [-2, -1, 0]]
>>> ax.set_xticklabels(ticks)  # doctest: +SKIP
>>> plt.show()

This random variable will be log-uniform regardless of the base chosen for
``a`` and ``b``. Let's specify with base ``2`` instead:

>>> rvs = Distribution(2**-2, 2**0).rvs(size=1000)

Values of ``1/4``, ``1/2`` and ``1`` are equally likely with this random
variable.  Here's the histogram:

>>> fig, ax = plt.subplots(1, 1)
>>> ax.hist(np.log2(rvs))
>>> ax.set_ylabel("Frequency")
>>> ax.set_xlabel("Value of random variable")
>>> ax.xaxis.set_major_locator(plt.FixedLocator([-2, -1, 0]))
>>> ticks = ["$2^{{ {} }}$".format(i) for i in [-2, -1, 0]]
>>> ax.set_xticklabels(ticks)  # doctest: +SKIP
>>> plt.show()
*)

val pinvh : ?cond:Py.Object.t -> ?rcond:Py.Object.t -> ?lower:bool -> ?return_rank:bool -> ?check_finite:bool -> a:Py.Object.t -> unit -> Py.Object.t
(**
Compute the (Moore-Penrose) pseudo-inverse of a Hermitian matrix.

Copied in from scipy==1.2.2, in order to preserve the default choice of the
`cond` and `above_cutoff` values which determine which values of the matrix
inversion lie below threshold and are so set to zero. Changes in scipy 1.3
resulted in a smaller default threshold and thus slower convergence of
dependent algorithms in some cases (see Sklearn github issue #14055).

Calculate a generalized inverse of a Hermitian or real symmetric matrix
using its eigenvalue decomposition and including all eigenvalues with
'large' absolute value.

Parameters
----------
a : (N, N) array_like
    Real symmetric or complex hermetian matrix to be pseudo-inverted
cond, rcond : float or None
    Cutoff for 'small' eigenvalues.
    Singular values smaller than rcond * largest_eigenvalue are considered
    zero.

    If None or -1, suitable machine precision is used.
lower : bool, optional
    Whether the pertinent array data is taken from the lower or upper
    triangle of a. (Default: lower)
return_rank : bool, optional
    if True, return the effective rank of the matrix
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
B : (N, N) ndarray
    The pseudo-inverse of matrix `a`.
rank : int
    The effective rank of the matrix.  Returned if return_rank == True

Raises
------
LinAlgError
    If eigenvalue does not converge

Examples
--------
>>> from scipy.linalg import pinvh
>>> a = np.random.randn(9, 6)
>>> a = np.dot(a, a.T)
>>> B = pinvh(a)
>>> np.allclose(a, np.dot(a, np.dot(B, a)))
True
>>> np.allclose(B, np.dot(B, np.dot(a, B)))
True
*)

val sparse_lsqr : ?damp:Py.Object.t -> ?atol:Py.Object.t -> ?btol:Py.Object.t -> ?conlim:Py.Object.t -> ?iter_lim:Py.Object.t -> ?show:Py.Object.t -> ?calc_var:Py.Object.t -> ?x0:Py.Object.t -> a:[`SparseMatrix of Csr_matrix.t | `Ndarray of Ndarray.t | `LinearOperator of Py.Object.t] -> b:Py.Object.t -> unit -> Py.Object.t
(**
Find the least-squares solution to a large, sparse, linear system
of equations.

The function solves ``Ax = b``  or  ``min ||b - Ax||^2`` or
``min ||Ax - b||^2 + d^2 ||x||^2``.

The matrix A may be square or rectangular (over-determined or
under-determined), and may have any rank.

::

  1. Unsymmetric equations --    solve  A*x = b

  2. Linear least squares  --    solve  A*x = b
                                 in the least-squares sense

  3. Damped least squares  --    solve  (   A    )*x = ( b )
                                        ( damp*I )     ( 0 )
                                 in the least-squares sense

Parameters
----------
A : {sparse matrix, ndarray, LinearOperator}
    Representation of an m-by-n matrix.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` and ``A^T x`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : array_like, shape (m,)
    Right-hand side vector ``b``.
damp : float
    Damping coefficient.
atol, btol : float, optional
    Stopping tolerances. If both are 1.0e-9 (say), the final
    residual norm should be accurate to about 9 digits.  (The
    final x will usually have fewer correct digits, depending on
    cond(A) and the size of damp.)
conlim : float, optional
    Another stopping tolerance.  lsqr terminates if an estimate of
    ``cond(A)`` exceeds `conlim`.  For compatible systems ``Ax =
    b``, `conlim` could be as large as 1.0e+12 (say).  For
    least-squares problems, conlim should be less than 1.0e+8.
    Maximum precision can be obtained by setting ``atol = btol =
    conlim = zero``, but the number of iterations may then be
    excessive.
iter_lim : int, optional
    Explicit limitation on number of iterations (for safety).
show : bool, optional
    Display an iteration log.
calc_var : bool, optional
    Whether to estimate diagonals of ``(A'A + damp^2*I)^{-1}``.
x0 : array_like, shape (n,), optional
    Initial guess of x, if None zeros are used.

    .. versionadded:: 1.0.0

Returns
-------
x : ndarray of float
    The final solution.
istop : int
    Gives the reason for termination.
    1 means x is an approximate solution to Ax = b.
    2 means x approximately solves the least-squares problem.
itn : int
    Iteration number upon termination.
r1norm : float
    ``norm(r)``, where ``r = b - Ax``.
r2norm : float
    ``sqrt( norm(r)^2  +  damp^2 * norm(x)^2 )``.  Equal to `r1norm` if
    ``damp == 0``.
anorm : float
    Estimate of Frobenius norm of ``Abar = [[A]; [damp*I]]``.
acond : float
    Estimate of ``cond(Abar)``.
arnorm : float
    Estimate of ``norm(A'*r - damp^2*x)``.
xnorm : float
    ``norm(x)``
var : ndarray of float
    If ``calc_var`` is True, estimates all diagonals of
    ``(A'A)^{-1}`` (if ``damp == 0``) or more generally ``(A'A +
    damp^2*I)^{-1}``.  This is well defined if A has full column
    rank or ``damp > 0``.  (Not sure what var means if ``rank(A)
    < n`` and ``damp = 0.``)

Notes
-----
LSQR uses an iterative method to approximate the solution.  The
number of iterations required to reach a certain accuracy depends
strongly on the scaling of the problem.  Poor scaling of the rows
or columns of A should therefore be avoided where possible.

For example, in problem 1 the solution is unaltered by
row-scaling.  If a row of A is very small or large compared to
the other rows of A, the corresponding row of ( A  b ) should be
scaled up or down.

In problems 1 and 2, the solution x is easily recovered
following column-scaling.  Unless better information is known,
the nonzero columns of A should be scaled so that they all have
the same Euclidean norm (e.g., 1.0).

In problem 3, there is no freedom to re-scale if damp is
nonzero.  However, the value of damp should be assigned only
after attention has been paid to the scaling of A.

The parameter damp is intended to help regularize
ill-conditioned systems, by preventing the true solution from
being very large.  Another aid to regularization is provided by
the parameter acond, which may be used to terminate iterations
before the computed solution becomes very large.

If some initial estimate ``x0`` is known and if ``damp == 0``,
one could proceed as follows:

  1. Compute a residual vector ``r0 = b - A*x0``.
  2. Use LSQR to solve the system  ``A*dx = r0``.
  3. Add the correction dx to obtain a final solution ``x = x0 + dx``.

This requires that ``x0`` be available before and after the call
to LSQR.  To judge the benefits, suppose LSQR takes k1 iterations
to solve A*x = b and k2 iterations to solve A*dx = r0.
If x0 is "good", norm(r0) will be smaller than norm(b).
If the same stopping tolerances atol and btol are used for each
system, k1 and k2 will be similar, but the final solution x0 + dx
should be more accurate.  The only way to reduce the total work
is to use a larger stopping tolerance for the second system.
If some value btol is suitable for A*x = b, the larger value
btol*norm(b)/norm(r0)  should be suitable for A*dx = r0.

Preconditioning is another way to reduce the number of iterations.
If it is possible to solve a related system ``M*x = b``
efficiently, where M approximates A in some helpful way (e.g. M -
A has low rank or its elements are small relative to those of A),
LSQR may converge more rapidly on the system ``A*M(inverse)*z =
b``, after which x can be recovered by solving M*x = z.

If A is symmetric, LSQR should not be used!

Alternatives are the symmetric conjugate-gradient method (cg)
and/or SYMMLQ.  SYMMLQ is an implementation of symmetric cg that
applies to any symmetric A and will converge more rapidly than
LSQR.  If A is positive definite, there are other implementations
of symmetric cg that require slightly less work per iteration than
SYMMLQ (but will take the same number of iterations).

References
----------
.. [1] C. C. Paige and M. A. Saunders (1982a).
       "LSQR: An algorithm for sparse linear equations and
       sparse least squares", ACM TOMS 8(1), 43-71.
.. [2] C. C. Paige and M. A. Saunders (1982b).
       "Algorithm 583.  LSQR: Sparse linear equations and least
       squares problems", ACM TOMS 8(2), 195-209.
.. [3] M. A. Saunders (1995).  "Solution of sparse rectangular
       systems using LSQR and CRAIG", BIT 35, 588-604.

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import lsqr
>>> A = csc_matrix([[1., 0.], [1., 1.], [0., 1.]], dtype=float)

The first example has the trivial solution `[0, 0]`

>>> b = np.array([0., 0., 0.], dtype=float)
>>> x, istop, itn, normr = lsqr(A, b)[:4]
The exact solution is  x = 0
>>> istop
0
>>> x
array([ 0.,  0.])

The stopping code `istop=0` returned indicates that a vector of zeros was
found as a solution. The returned solution `x` indeed contains `[0., 0.]`.
The next example has a non-trivial solution:

>>> b = np.array([1., 0., -1.], dtype=float)
>>> x, istop, itn, r1norm = lsqr(A, b)[:4]
>>> istop
1
>>> x
array([ 1., -1.])
>>> itn
1
>>> r1norm
4.440892098500627e-16

As indicated by `istop=1`, `lsqr` found a solution obeying the tolerance
limits. The given solution `[1., -1.]` obviously solves the equation. The
remaining return values include information about the number of iterations
(`itn=1`) and the remaining difference of left and right side of the solved
equation.
The final example demonstrates the behavior in the case where there is no
solution for the equation:

>>> b = np.array([1., 0.01, -1.], dtype=float)
>>> x, istop, itn, r1norm = lsqr(A, b)[:4]
>>> istop
2
>>> x
array([ 1.00333333, -0.99666667])
>>> A.dot(x)-b
array([ 0.00333333, -0.00333333,  0.00333333])
>>> r1norm
0.005773502691896255

`istop` indicates that the system is inconsistent and thus `x` is rather an
approximate solution to the corresponding least-squares problem. `r1norm`
contains the norm of the minimal residual that was found.
*)


end

val gen_batches : ?min_batch_size:Py.Object.t -> n:int -> batch_size:Py.Object.t -> unit -> Py.Object.t
(**
Generator to create slices containing batch_size elements, from 0 to n.

The last slice may contain less than batch_size elements, when batch_size
does not divide n.

Parameters
----------
n : int
batch_size : int
    Number of element in each batch
min_batch_size : int, default=0
    Minimum batch size to produce.

Yields
------
slice of batch_size elements

Examples
--------
>>> from sklearn.utils import gen_batches
>>> list(gen_batches(7, 3))
[slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]
>>> list(gen_batches(6, 3))
[slice(0, 3, None), slice(3, 6, None)]
>>> list(gen_batches(2, 3))
[slice(0, 2, None)]
>>> list(gen_batches(7, 3, min_batch_size=0))
[slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]
>>> list(gen_batches(7, 3, min_batch_size=2))
[slice(0, 3, None), slice(3, 7, None)]
*)

val gen_even_slices : ?n_samples:Py.Object.t -> n:int -> n_packs:Py.Object.t -> unit -> Py.Object.t
(**
Generator to create n_packs slices going up to n.

Parameters
----------
n : int
n_packs : int
    Number of slices to generate.
n_samples : int or None (default = None)
    Number of samples. Pass n_samples when the slices are to be used for
    sparse matrix indexing; slicing off-the-end raises an exception, while
    it works for NumPy arrays.

Yields
------
slice

Examples
--------
>>> from sklearn.utils import gen_even_slices
>>> list(gen_even_slices(10, 1))
[slice(0, 10, None)]
>>> list(gen_even_slices(10, 10))
[slice(0, 1, None), slice(1, 2, None), ..., slice(9, 10, None)]
>>> list(gen_even_slices(10, 5))
[slice(0, 2, None), slice(2, 4, None), ..., slice(8, 10, None)]
>>> list(gen_even_slices(10, 3))
[slice(0, 4, None), slice(4, 7, None), slice(7, 10, None)]
*)

val get_chunk_n_rows : ?max_n_rows:Py.Object.t -> ?working_memory:Py.Object.t -> row_bytes:int -> unit -> Py.Object.t
(**
Calculates how many rows can be processed within working_memory

Parameters
----------
row_bytes : int
    The expected number of bytes of memory that will be consumed
    during the processing of each row.
max_n_rows : int, optional
    The maximum return value.
working_memory : int or float, optional
    The number of rows to fit inside this number of MiB will be returned.
    When None (default), the value of
    ``sklearn.get_config()['working_memory']`` is used.

Returns
-------
int or the value of n_samples

Warns
-----
Issues a UserWarning if ``row_bytes`` exceeds ``working_memory`` MiB.
*)

val get_config : unit -> Py.Object.t
(**
Retrieve current values for configuration set by :func:`set_config`

Returns
-------
config : dict
    Keys are parameter names that can be passed to :func:`set_config`.

See Also
--------
config_context: Context manager for global scikit-learn configuration
set_config: Set global scikit-learn configuration
*)

module Graph : sig
val single_source_shortest_path_length : ?cutoff:Py.Object.t -> graph:[`SparseMatrix of Csr_matrix.t | `PyObject of Py.Object.t] -> source:Py.Object.t -> unit -> Py.Object.t
(**
Return the shortest path length from source to all reachable nodes.

Returns a dictionary of shortest path lengths keyed by target.

Parameters
----------
graph : sparse matrix or 2D array (preferably LIL matrix)
    Adjacency matrix of the graph
source : integer
   Starting node for path
cutoff : integer, optional
    Depth to stop the search - only
    paths of length <= cutoff are returned.

Examples
--------
>>> from sklearn.utils.graph import single_source_shortest_path_length
>>> import numpy as np
>>> graph = np.array([[ 0, 1, 0, 0],
...                   [ 1, 0, 1, 0],
...                   [ 0, 1, 0, 1],
...                   [ 0, 0, 1, 0]])
>>> list(sorted(single_source_shortest_path_length(graph, 0).items()))
[(0, 0), (1, 1), (2, 2), (3, 3)]
>>> graph = np.ones((6, 6))
>>> list(sorted(single_source_shortest_path_length(graph, 2).items()))
[(0, 1), (1, 1), (2, 0), (3, 1), (4, 1), (5, 1)]
*)


end

module Graph_shortest_path : sig
module Float64 : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?x:Py.Object.t -> unit -> t
(**
Double-precision floating-point number type, compatible with Python `float`
and C ``double``.
Character code: ``'d'``.
Canonical name: ``np.double``.
Alias: ``np.float_``.
Alias *on this platform*: ``np.float64``: 64-bit precision floating-point number type: sign bit, 11 bits exponent, 52 bits mantissa.
*)

val get_item : key:Py.Object.t -> t -> Py.Object.t
(**
Return self[key].
*)

val fromhex : string:Py.Object.t -> t -> Py.Object.t
(**
Create a floating-point number from a hexadecimal string.

>>> float.fromhex('0x1.ffffp10')
2047.984375
>>> float.fromhex('-0x1p-1074')
-5e-324
*)

val hex : t -> Py.Object.t
(**
Return a hexadecimal representation of a floating-point number.

>>> (-0.1).hex()
'-0x1.999999999999ap-4'
>>> 3.14159.hex()
'0x1.921f9f01b866ep+1'
*)

val is_integer : t -> Py.Object.t
(**
Return True if the float is an integer.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

val isspmatrix : x:Py.Object.t -> unit -> Py.Object.t
(**
Is x of a sparse matrix type?

Parameters
----------
x
    object to check for being a sparse matrix

Returns
-------
bool
    True if x is a sparse matrix, False otherwise

Notes
-----
issparse and isspmatrix are aliases for the same function.

Examples
--------
>>> from scipy.sparse import csr_matrix, isspmatrix
>>> isspmatrix(csr_matrix([[5]]))
True

>>> from scipy.sparse import isspmatrix
>>> isspmatrix(5)
False
*)

val isspmatrix_csr : x:Py.Object.t -> unit -> Py.Object.t
(**
Is x of csr_matrix type?

Parameters
----------
x
    object to check for being a csr matrix

Returns
-------
bool
    True if x is a csr matrix, False otherwise

Examples
--------
>>> from scipy.sparse import csr_matrix, isspmatrix_csr
>>> isspmatrix_csr(csr_matrix([[5]]))
True

>>> from scipy.sparse import csc_matrix, csr_matrix, isspmatrix_csc
>>> isspmatrix_csr(csc_matrix([[5]]))
False
*)


end

val hash : ?hash_name:[`Md5 | `Sha1] -> ?coerce_mmap:Py.Object.t -> obj:Py.Object.t -> unit -> Py.Object.t
(**
DEPRECATED: deprecated in version 0.20.1 to be removed in version 0.23. Please import this functionality directly from joblib, which can be installed with: pip install joblib.

Quick calculation of a hash to identify uniquely Python objects
       containing numpy arrays.


       Parameters
       -----------
       hash_name: 'md5' or 'sha1'
           Hashing algorithm used. sha1 is supposedly safer, but md5 is
           faster.
       coerce_mmap: boolean
           Make no difference between np.memmap and np.ndarray
   
*)

val import_module : ?package:Py.Object.t -> name:Py.Object.t -> unit -> Py.Object.t
(**
Import a module.

The 'package' argument is required when performing a relative import. It
specifies the package to use as the anchor point from which to resolve the
relative import to an absolute import.
*)

val indexable : Py.Object.t list -> Py.Object.t
(**
Make arrays indexable for cross-validation.

Checks consistent length, passes through None, and ensures that everything
can be indexed by converting sparse matrices to csr and converting
non-interable objects to arrays.

Parameters
----------
*iterables : lists, dataframes, arrays, sparse matrices
    List of objects to ensure sliceability.
*)

val indices_to_mask : indices:Py.Object.t -> mask_length:Py.Object.t -> unit -> Py.Object.t
(**
Convert list of indices to boolean mask.

Parameters
----------
indices : list-like
    List of integers treated as indices.
mask_length : int
    Length of boolean mask to be generated.
    This parameter must be greater than max(indices)

Returns
-------
mask : 1d boolean nd-array
    Boolean array that is True where indices are present, else False.

Examples
--------
>>> from sklearn.utils import indices_to_mask
>>> indices = [1, 2 , 3, 4]
>>> indices_to_mask(indices, 5)
array([False,  True,  True,  True,  True])
*)

val is_scalar_nan : x:Py.Object.t -> unit -> Py.Object.t
(**
Tests if x is NaN

This function is meant to overcome the issue that np.isnan does not allow
non-numerical types as input, and that np.nan is not np.float('nan').

Parameters
----------
x : any type

Returns
-------
boolean

Examples
--------
>>> is_scalar_nan(np.nan)
True
>>> is_scalar_nan(float("nan"))
True
>>> is_scalar_nan(None)
False
>>> is_scalar_nan("")
False
>>> is_scalar_nan([np.nan])
False
*)

val issparse : x:Py.Object.t -> unit -> Py.Object.t
(**
Is x of a sparse matrix type?

Parameters
----------
x
    object to check for being a sparse matrix

Returns
-------
bool
    True if x is a sparse matrix, False otherwise

Notes
-----
issparse and isspmatrix are aliases for the same function.

Examples
--------
>>> from scipy.sparse import csr_matrix, isspmatrix
>>> isspmatrix(csr_matrix([[5]]))
True

>>> from scipy.sparse import isspmatrix
>>> isspmatrix(5)
False
*)

module Metaestimators : sig
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
val pp : Format.formatter -> t -> unit


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
val pp : Format.formatter -> t -> unit


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

val if_delegate_has_method : delegate:[`String of string | `StringList of string list | `PyObject of Py.Object.t] -> unit -> Py.Object.t
(**
Create a decorator for methods that are delegated to a sub-estimator

This enables ducktyping by hasattr returning True according to the
sub-estimator.

Parameters
----------
delegate : string, list of strings or tuple of strings
    Name of the sub-estimator that can be accessed as an attribute of the
    base object. If a list or a tuple of names are provided, the first
    sub-estimator that is an attribute of the base object will be used.
*)

val update_wrapper : ?assigned:Py.Object.t -> ?updated:Py.Object.t -> wrapper:Py.Object.t -> wrapped:Py.Object.t -> unit -> Py.Object.t
(**
Update a wrapper function to look like the wrapped function

wrapper is the function to be updated
wrapped is the original function
assigned is a tuple naming the attributes assigned directly
from the wrapped function to the wrapper function (defaults to
functools.WRAPPER_ASSIGNMENTS)
updated is a tuple naming the attributes of the wrapper that
are updated with the corresponding attribute from the wrapped
function (defaults to functools.WRAPPER_UPDATES)
*)


end

module Multiclass : sig
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

val check_classification_targets : y:Ndarray.t -> unit -> Py.Object.t
(**
Ensure that target y is of a non-regression type.

Only the following target types (as defined in type_of_target) are allowed:
    'binary', 'multiclass', 'multiclass-multioutput',
    'multilabel-indicator', 'multilabel-sequences'

Parameters
----------
y : array-like
*)

val class_distribution : ?sample_weight:Ndarray.t -> y:Py.Object.t -> unit -> (Py.Object.t * Py.Object.t * Py.Object.t)
(**
Compute class priors from multioutput-multiclass target data

Parameters
----------
y : array like or sparse matrix of size (n_samples, n_outputs)
    The labels for each example.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

Returns
-------
classes : list of size n_outputs of arrays of size (n_classes,)
    List of classes for each column.

n_classes : list of integers of size n_outputs
    Number of classes in each column

class_prior : list of size n_outputs of arrays of size (n_classes,)
    Class distribution of each column.
*)

module Dok_matrix : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?shape:int list -> ?dtype:Py.Object.t -> ?copy:Py.Object.t -> arg1:Py.Object.t -> unit -> t
(**
Dictionary Of Keys based sparse matrix.

This is an efficient structure for constructing sparse
matrices incrementally.

This can be instantiated in several ways:
    dok_matrix(D)
        with a dense matrix, D

    dok_matrix(S)
        with a sparse matrix, S

    dok_matrix((M,N), [dtype])
        create the matrix with initial shape (M,N)
        dtype is optional, defaulting to dtype='d'

Attributes
----------
dtype : dtype
    Data type of the matrix
shape : 2-tuple
    Shape of the matrix
ndim : int
    Number of dimensions (this is always 2)
nnz
    Number of nonzero elements

Notes
-----

Sparse matrices can be used in arithmetic operations: they support
addition, subtraction, multiplication, division, and matrix power.

Allows for efficient O(1) access of individual elements.
Duplicates are not allowed.
Can be efficiently converted to a coo_matrix once constructed.

Examples
--------
>>> import numpy as np
>>> from scipy.sparse import dok_matrix
>>> S = dok_matrix((5, 5), dtype=np.float32)
>>> for i in range(5):
...     for j in range(5):
...         S[i, j] = i + j    # Update element
*)

val get_item : key:Py.Object.t -> t -> Py.Object.t
(**
None
*)

val asformat : ?copy:Py.Object.t -> format:string -> t -> Py.Object.t
(**
Return this matrix in the passed format.

Parameters
----------
format : {str, None}
    The desired matrix format ("csr", "csc", "lil", "dok", "array", ...)
    or None for no conversion.
copy : bool, optional
    If True, the result is guaranteed to not share data with self.

Returns
-------
A : This matrix in the passed format.
*)

val asfptype : t -> Py.Object.t
(**
Upcast matrix to a floating point format (if necessary)
*)

val astype : ?casting:Py.Object.t -> ?copy:Py.Object.t -> dtype:[`String of string | `PyObject of Py.Object.t] -> t -> Py.Object.t
(**
Cast the matrix elements to a specified type.

Parameters
----------
dtype : string or numpy dtype
    Typecode or data-type to which to cast the data.
casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
    Controls what kind of data casting may occur.
    Defaults to 'unsafe' for backwards compatibility.
    'no' means the data types should not be cast at all.
    'equiv' means only byte-order changes are allowed.
    'safe' means only casts which can preserve values are allowed.
    'same_kind' means only safe casts or casts within a kind,
    like float64 to float32, are allowed.
    'unsafe' means any data conversions may be done.
copy : bool, optional
    If `copy` is `False`, the result might share some memory with this
    matrix. If `copy` is `True`, it is guaranteed that the result and
    this matrix do not share any memory.
*)

val conj : ?copy:bool -> t -> Py.Object.t
(**
Element-wise complex conjugation.

If the matrix is of non-complex data type and `copy` is False,
this method does nothing and the data is not copied.

Parameters
----------
copy : bool, optional
    If True, the result is guaranteed to not share data with self.

Returns
-------
A : The element-wise complex conjugate.
*)

val conjtransp : t -> Py.Object.t
(**
Return the conjugate transpose.
*)

val conjugate : ?copy:bool -> t -> Py.Object.t
(**
Element-wise complex conjugation.

If the matrix is of non-complex data type and `copy` is False,
this method does nothing and the data is not copied.

Parameters
----------
copy : bool, optional
    If True, the result is guaranteed to not share data with self.

Returns
-------
A : The element-wise complex conjugate.
*)

val copy : t -> Py.Object.t
(**
Returns a copy of this matrix.

No data/indices will be shared between the returned value and current
matrix.
*)

val count_nonzero : t -> Py.Object.t
(**
Number of non-zero entries, equivalent to

np.count_nonzero(a.toarray())

Unlike getnnz() and the nnz property, which return the number of stored
entries (the length of the data attribute), this method counts the
actual number of non-zero entries in data.
*)

val diagonal : ?k:int -> t -> Py.Object.t
(**
Returns the k-th diagonal of the matrix.

Parameters
----------
k : int, optional
    Which diagonal to get, corresponding to elements a[i, i+k].
    Default: 0 (the main diagonal).

    .. versionadded:: 1.0

See also
--------
numpy.diagonal : Equivalent numpy function.

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
>>> A.diagonal()
array([1, 0, 5])
>>> A.diagonal(k=1)
array([2, 3])
*)

val dot : other:Py.Object.t -> t -> Py.Object.t
(**
Ordinary dot product

Examples
--------
>>> import numpy as np
>>> from scipy.sparse import csr_matrix
>>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
>>> v = np.array([1, 0, -1])
>>> A.dot(v)
array([ 1, -3, -1], dtype=int64)
*)

val fromkeys : ?value:Py.Object.t -> iterable:Py.Object.t -> t -> Py.Object.t
(**
Create a new dictionary with keys from iterable and values set to value.
*)

val get : ?default:Py.Object.t -> key:Py.Object.t -> t -> Py.Object.t
(**
This overrides the dict.get method, providing type checking
but otherwise equivalent functionality.
*)

val getH : t -> Py.Object.t
(**
Return the Hermitian transpose of this matrix.

See Also
--------
numpy.matrix.getH : NumPy's implementation of `getH` for matrices
*)

val get_shape : t -> Py.Object.t
(**
Get shape of a matrix.
*)

val getcol : j:Py.Object.t -> t -> Py.Object.t
(**
Returns a copy of column j of the matrix, as an (m x 1) sparse
matrix (column vector).
*)

val getformat : t -> Py.Object.t
(**
Format of a matrix representation as a string.
*)

val getmaxprint : t -> Py.Object.t
(**
Maximum number of elements to display when printed.
*)

val getnnz : ?axis:[`None | `PyObject of Py.Object.t] -> t -> Py.Object.t
(**
Number of stored values, including explicit zeros.

Parameters
----------
axis : None, 0, or 1
    Select between the number of values across the whole matrix, in
    each column, or in each row.

See also
--------
count_nonzero : Number of non-zero entries
*)

val getrow : i:Py.Object.t -> t -> Py.Object.t
(**
Returns a copy of row i of the matrix, as a (1 x n) sparse
matrix (row vector).
*)

val maximum : other:Py.Object.t -> t -> Py.Object.t
(**
Element-wise maximum between this and another matrix.
*)

val mean : ?axis:[`None | `PyObject of Py.Object.t] -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> t -> Py.Object.t
(**
Compute the arithmetic mean along the specified axis.

Returns the average of the matrix elements. The average is taken
over all elements in the matrix by default, otherwise over the
specified axis. `float64` intermediate and return values are used
for integer inputs.

Parameters
----------
axis : {-2, -1, 0, 1, None} optional
    Axis along which the mean is computed. The default is to compute
    the mean of all elements in the matrix (i.e. `axis` = `None`).
dtype : data-type, optional
    Type to use in computing the mean. For integer inputs, the default
    is `float64`; for floating point inputs, it is the same as the
    input dtype.

    .. versionadded:: 0.18.0

out : np.matrix, optional
    Alternative output matrix in which to place the result. It must
    have the same shape as the expected output, but the type of the
    output values will be cast if necessary.

    .. versionadded:: 0.18.0

Returns
-------
m : np.matrix

See Also
--------
numpy.matrix.mean : NumPy's implementation of 'mean' for matrices
*)

val minimum : other:Py.Object.t -> t -> Py.Object.t
(**
Element-wise minimum between this and another matrix.
*)

val multiply : other:Py.Object.t -> t -> Py.Object.t
(**
Point-wise multiplication by another matrix
        
*)

val nonzero : t -> Py.Object.t
(**
nonzero indices

Returns a tuple of arrays (row,col) containing the indices
of the non-zero elements of the matrix.

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> A = csr_matrix([[1,2,0],[0,0,3],[4,0,5]])
>>> A.nonzero()
(array([0, 0, 1, 2, 2]), array([0, 1, 2, 0, 2]))
*)

val power : ?dtype:Py.Object.t -> n:Py.Object.t -> t -> Py.Object.t
(**
Element-wise power.
*)

val reshape : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> t -> Csr_matrix.t
(**
reshape(self, shape, order='C', copy=False)

Gives a new shape to a sparse matrix without changing its data.

Parameters
----------
shape : length-2 tuple of ints
    The new shape should be compatible with the original shape.
order : {'C', 'F'}, optional
    Read the elements using this index order. 'C' means to read and
    write the elements using C-like index order; e.g. read entire first
    row, then second row, etc. 'F' means to read and write the elements
    using Fortran-like index order; e.g. read entire first column, then
    second column, etc.
copy : bool, optional
    Indicates whether or not attributes of self should be copied
    whenever possible. The degree to which attributes are copied varies
    depending on the type of sparse matrix being used.

Returns
-------
reshaped_matrix : sparse matrix
    A sparse matrix with the given `shape`, not necessarily of the same
    format as the current object.

See Also
--------
numpy.matrix.reshape : NumPy's implementation of 'reshape' for
                       matrices
*)

val resize : int list -> t -> Py.Object.t
(**
Resize the matrix in-place to dimensions given by ``shape``

Any elements that lie within the new shape will remain at the same
indices, while non-zero elements lying outside the new shape are
removed.

Parameters
----------
shape : (int, int)
    number of rows and columns in the new matrix

Notes
-----
The semantics are not identical to `numpy.ndarray.resize` or
`numpy.resize`.  Here, the same data will be maintained at each index
before and after reshape, if that index is within the new bounds.  In
numpy, resizing maintains contiguity of the array, moving elements
around in the logical matrix but not within a flattened representation.

We give no guarantees about whether the underlying data attributes
(arrays, etc.) will be modified in place or replaced with new objects.
*)

val set_shape : shape:int list -> t -> Py.Object.t
(**
See `reshape`.
*)

val setdefault : ?default:Py.Object.t -> key:Py.Object.t -> t -> Py.Object.t
(**
Insert key with a value of default if key is not in the dictionary.

Return the value for key if key is in the dictionary, else default.
*)

val setdiag : ?k:int -> values:Ndarray.t -> t -> Py.Object.t
(**
Set diagonal or off-diagonal elements of the array.

Parameters
----------
values : array_like
    New values of the diagonal elements.

    Values may have any length.  If the diagonal is longer than values,
    then the remaining diagonal entries will not be set.  If values if
    longer than the diagonal, then the remaining values are ignored.

    If a scalar value is given, all of the diagonal is set to it.

k : int, optional
    Which off-diagonal to set, corresponding to elements a[i,i+k].
    Default: 0 (the main diagonal).
*)

val sum : ?axis:[`None | `PyObject of Py.Object.t] -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> t -> Py.Object.t
(**
Sum the matrix elements over a given axis.

Parameters
----------
axis : {-2, -1, 0, 1, None} optional
    Axis along which the sum is computed. The default is to
    compute the sum of all the matrix elements, returning a scalar
    (i.e. `axis` = `None`).
dtype : dtype, optional
    The type of the returned matrix and of the accumulator in which
    the elements are summed.  The dtype of `a` is used by default
    unless `a` has an integer dtype of less precision than the default
    platform integer.  In that case, if `a` is signed then the platform
    integer is used while if `a` is unsigned then an unsigned integer
    of the same precision as the platform integer is used.

    .. versionadded:: 0.18.0

out : np.matrix, optional
    Alternative output matrix in which to place the result. It must
    have the same shape as the expected output, but the type of the
    output values will be cast if necessary.

    .. versionadded:: 0.18.0

Returns
-------
sum_along_axis : np.matrix
    A matrix with the same shape as `self`, with the specified
    axis removed.

See Also
--------
numpy.matrix.sum : NumPy's implementation of 'sum' for matrices
*)

val toarray : ?order:[`C | `F] -> ?out:Ndarray.t -> t -> Ndarray.t
(**
Return a dense ndarray representation of this matrix.

Parameters
----------
order : {'C', 'F'}, optional
    Whether to store multi-dimensional data in C (row-major)
    or Fortran (column-major) order in memory. The default
    is 'None', indicating the NumPy default of C-ordered.
    Cannot be specified in conjunction with the `out`
    argument.

out : ndarray, 2-dimensional, optional
    If specified, uses this array as the output buffer
    instead of allocating a new array to return. The provided
    array must have the same shape and dtype as the sparse
    matrix on which you are calling the method. For most
    sparse types, `out` is required to be memory contiguous
    (either C or Fortran ordered).

Returns
-------
arr : ndarray, 2-dimensional
    An array with the same shape and containing the same
    data represented by the sparse matrix, with the requested
    memory order. If `out` was passed, the same object is
    returned after being modified in-place to contain the
    appropriate values.
*)

val tobsr : ?blocksize:Py.Object.t -> ?copy:Py.Object.t -> t -> Py.Object.t
(**
Convert this matrix to Block Sparse Row format.

With copy=False, the data/indices may be shared between this matrix and
the resultant bsr_matrix.

When blocksize=(R, C) is provided, it will be used for construction of
the bsr_matrix.
*)

val tocoo : ?copy:Py.Object.t -> t -> Py.Object.t
(**
Convert this matrix to COOrdinate format.

With copy=False, the data/indices may be shared between this matrix and
the resultant coo_matrix.
*)

val tocsc : ?copy:Py.Object.t -> t -> Py.Object.t
(**
Convert this matrix to Compressed Sparse Column format.

With copy=False, the data/indices may be shared between this matrix and
the resultant csc_matrix.
*)

val tocsr : ?copy:Py.Object.t -> t -> Py.Object.t
(**
Convert this matrix to Compressed Sparse Row format.

With copy=False, the data/indices may be shared between this matrix and
the resultant csr_matrix.
*)

val todense : ?order:[`C | `F] -> ?out:Ndarray.t -> t -> Py.Object.t
(**
Return a dense matrix representation of this matrix.

Parameters
----------
order : {'C', 'F'}, optional
    Whether to store multi-dimensional data in C (row-major)
    or Fortran (column-major) order in memory. The default
    is 'None', indicating the NumPy default of C-ordered.
    Cannot be specified in conjunction with the `out`
    argument.

out : ndarray, 2-dimensional, optional
    If specified, uses this array (or `numpy.matrix`) as the
    output buffer instead of allocating a new array to
    return. The provided array must have the same shape and
    dtype as the sparse matrix on which you are calling the
    method.

Returns
-------
arr : numpy.matrix, 2-dimensional
    A NumPy matrix object with the same shape and containing
    the same data represented by the sparse matrix, with the
    requested memory order. If `out` was passed and was an
    array (rather than a `numpy.matrix`), it will be filled
    with the appropriate values and returned wrapped in a
    `numpy.matrix` object that shares the same memory.
*)

val todia : ?copy:Py.Object.t -> t -> Py.Object.t
(**
Convert this matrix to sparse DIAgonal format.

With copy=False, the data/indices may be shared between this matrix and
the resultant dia_matrix.
*)

val todok : ?copy:Py.Object.t -> t -> Py.Object.t
(**
Convert this matrix to Dictionary Of Keys format.

With copy=False, the data/indices may be shared between this matrix and
the resultant dok_matrix.
*)

val tolil : ?copy:Py.Object.t -> t -> Py.Object.t
(**
Convert this matrix to List of Lists format.

With copy=False, the data/indices may be shared between this matrix and
the resultant lil_matrix.
*)

val transpose : ?axes:Py.Object.t -> ?copy:Py.Object.t -> t -> Py.Object.t
(**
Reverses the dimensions of the sparse matrix.

Parameters
----------
axes : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except
    for the default value.
copy : bool, optional
    Indicates whether or not attributes of `self` should be
    copied whenever possible. The degree to which attributes
    are copied varies depending on the type of sparse matrix
    being used.

Returns
-------
p : `self` with the dimensions reversed.

See Also
--------
numpy.matrix.transpose : NumPy's implementation of 'transpose'
                         for matrices
*)

val update : val_:Py.Object.t -> t -> Py.Object.t
(**
D.update([E, ]**F) -> None.  Update D from dict/iterable E and F.
If E is present and has a .keys() method, then does:  for k in E: D[k] = E[k]
If E is present and lacks a .keys() method, then does:  for k, v in E: D[k] = v
In either case, this is followed by: for k in F:  D[k] = F[k]
*)


(** Attribute dtype: see constructor for documentation *)
val dtype : t -> Py.Object.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

val is_multilabel : y:Ndarray.t -> unit -> bool
(**
Check if ``y`` is in a multilabel format.

Parameters
----------
y : numpy array of shape [n_samples]
    Target values.

Returns
-------
out : bool,
    Return ``True``, if ``y`` is in a multilabel format, else ```False``.

Examples
--------
>>> import numpy as np
>>> from sklearn.utils.multiclass import is_multilabel
>>> is_multilabel([0, 1, 0, 1])
False
>>> is_multilabel([[1], [0, 2], []])
False
>>> is_multilabel(np.array([[1, 0], [0, 0]]))
True
>>> is_multilabel(np.array([[1], [0], [0]]))
False
>>> is_multilabel(np.array([[1, 0, 0]]))
True
*)

val issparse : x:Py.Object.t -> unit -> Py.Object.t
(**
Is x of a sparse matrix type?

Parameters
----------
x
    object to check for being a sparse matrix

Returns
-------
bool
    True if x is a sparse matrix, False otherwise

Notes
-----
issparse and isspmatrix are aliases for the same function.

Examples
--------
>>> from scipy.sparse import csr_matrix, isspmatrix
>>> isspmatrix(csr_matrix([[5]]))
True

>>> from scipy.sparse import isspmatrix
>>> isspmatrix(5)
False
*)

module Lil_matrix : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?shape:int list -> ?dtype:Py.Object.t -> ?copy:Py.Object.t -> arg1:Py.Object.t -> unit -> t
(**
Row-based list of lists sparse matrix

This is a structure for constructing sparse matrices incrementally.
Note that inserting a single item can take linear time in the worst case;
to construct a matrix efficiently, make sure the items are pre-sorted by
index, per row.

This can be instantiated in several ways:
    lil_matrix(D)
        with a dense matrix or rank-2 ndarray D

    lil_matrix(S)
        with another sparse matrix S (equivalent to S.tolil())

    lil_matrix((M, N), [dtype])
        to construct an empty matrix with shape (M, N)
        dtype is optional, defaulting to dtype='d'.

Attributes
----------
dtype : dtype
    Data type of the matrix
shape : 2-tuple
    Shape of the matrix
ndim : int
    Number of dimensions (this is always 2)
nnz
    Number of stored values, including explicit zeros
data
    LIL format data array of the matrix
rows
    LIL format row index array of the matrix

Notes
-----

Sparse matrices can be used in arithmetic operations: they support
addition, subtraction, multiplication, division, and matrix power.

Advantages of the LIL format
    - supports flexible slicing
    - changes to the matrix sparsity structure are efficient

Disadvantages of the LIL format
    - arithmetic operations LIL + LIL are slow (consider CSR or CSC)
    - slow column slicing (consider CSC)
    - slow matrix vector products (consider CSR or CSC)

Intended Usage
    - LIL is a convenient format for constructing sparse matrices
    - once a matrix has been constructed, convert to CSR or
      CSC format for fast arithmetic and matrix vector operations
    - consider using the COO format when constructing large matrices

Data Structure
    - An array (``self.rows``) of rows, each of which is a sorted
      list of column indices of non-zero elements.
    - The corresponding nonzero values are stored in similar
      fashion in ``self.data``.
*)

val get_item : key:Py.Object.t -> t -> Py.Object.t
(**
None
*)

val asformat : ?copy:Py.Object.t -> format:string -> t -> Py.Object.t
(**
Return this matrix in the passed format.

Parameters
----------
format : {str, None}
    The desired matrix format ("csr", "csc", "lil", "dok", "array", ...)
    or None for no conversion.
copy : bool, optional
    If True, the result is guaranteed to not share data with self.

Returns
-------
A : This matrix in the passed format.
*)

val asfptype : t -> Py.Object.t
(**
Upcast matrix to a floating point format (if necessary)
*)

val astype : ?casting:Py.Object.t -> ?copy:Py.Object.t -> dtype:[`String of string | `PyObject of Py.Object.t] -> t -> Py.Object.t
(**
Cast the matrix elements to a specified type.

Parameters
----------
dtype : string or numpy dtype
    Typecode or data-type to which to cast the data.
casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
    Controls what kind of data casting may occur.
    Defaults to 'unsafe' for backwards compatibility.
    'no' means the data types should not be cast at all.
    'equiv' means only byte-order changes are allowed.
    'safe' means only casts which can preserve values are allowed.
    'same_kind' means only safe casts or casts within a kind,
    like float64 to float32, are allowed.
    'unsafe' means any data conversions may be done.
copy : bool, optional
    If `copy` is `False`, the result might share some memory with this
    matrix. If `copy` is `True`, it is guaranteed that the result and
    this matrix do not share any memory.
*)

val conj : ?copy:bool -> t -> Py.Object.t
(**
Element-wise complex conjugation.

If the matrix is of non-complex data type and `copy` is False,
this method does nothing and the data is not copied.

Parameters
----------
copy : bool, optional
    If True, the result is guaranteed to not share data with self.

Returns
-------
A : The element-wise complex conjugate.
*)

val conjugate : ?copy:bool -> t -> Py.Object.t
(**
Element-wise complex conjugation.

If the matrix is of non-complex data type and `copy` is False,
this method does nothing and the data is not copied.

Parameters
----------
copy : bool, optional
    If True, the result is guaranteed to not share data with self.

Returns
-------
A : The element-wise complex conjugate.
*)

val copy : t -> Py.Object.t
(**
Returns a copy of this matrix.

No data/indices will be shared between the returned value and current
matrix.
*)

val count_nonzero : t -> Py.Object.t
(**
Number of non-zero entries, equivalent to

np.count_nonzero(a.toarray())

Unlike getnnz() and the nnz property, which return the number of stored
entries (the length of the data attribute), this method counts the
actual number of non-zero entries in data.
*)

val diagonal : ?k:int -> t -> Py.Object.t
(**
Returns the k-th diagonal of the matrix.

Parameters
----------
k : int, optional
    Which diagonal to get, corresponding to elements a[i, i+k].
    Default: 0 (the main diagonal).

    .. versionadded:: 1.0

See also
--------
numpy.diagonal : Equivalent numpy function.

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
>>> A.diagonal()
array([1, 0, 5])
>>> A.diagonal(k=1)
array([2, 3])
*)

val dot : other:Py.Object.t -> t -> Py.Object.t
(**
Ordinary dot product

Examples
--------
>>> import numpy as np
>>> from scipy.sparse import csr_matrix
>>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
>>> v = np.array([1, 0, -1])
>>> A.dot(v)
array([ 1, -3, -1], dtype=int64)
*)

val getH : t -> Py.Object.t
(**
Return the Hermitian transpose of this matrix.

See Also
--------
numpy.matrix.getH : NumPy's implementation of `getH` for matrices
*)

val get_shape : t -> Py.Object.t
(**
Get shape of a matrix.
*)

val getcol : j:Py.Object.t -> t -> Py.Object.t
(**
Returns a copy of column j of the matrix, as an (m x 1) sparse
matrix (column vector).
*)

val getformat : t -> Py.Object.t
(**
Format of a matrix representation as a string.
*)

val getmaxprint : t -> Py.Object.t
(**
Maximum number of elements to display when printed.
*)

val getnnz : ?axis:[`None | `PyObject of Py.Object.t] -> t -> Py.Object.t
(**
Number of stored values, including explicit zeros.

Parameters
----------
axis : None, 0, or 1
    Select between the number of values across the whole matrix, in
    each column, or in each row.

See also
--------
count_nonzero : Number of non-zero entries
*)

val getrow : i:Py.Object.t -> t -> Py.Object.t
(**
Returns a copy of the 'i'th row.
        
*)

val getrowview : i:Py.Object.t -> t -> Py.Object.t
(**
Returns a view of the 'i'th row (without copying).
        
*)

val maximum : other:Py.Object.t -> t -> Py.Object.t
(**
Element-wise maximum between this and another matrix.
*)

val mean : ?axis:[`None | `PyObject of Py.Object.t] -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> t -> Py.Object.t
(**
Compute the arithmetic mean along the specified axis.

Returns the average of the matrix elements. The average is taken
over all elements in the matrix by default, otherwise over the
specified axis. `float64` intermediate and return values are used
for integer inputs.

Parameters
----------
axis : {-2, -1, 0, 1, None} optional
    Axis along which the mean is computed. The default is to compute
    the mean of all elements in the matrix (i.e. `axis` = `None`).
dtype : data-type, optional
    Type to use in computing the mean. For integer inputs, the default
    is `float64`; for floating point inputs, it is the same as the
    input dtype.

    .. versionadded:: 0.18.0

out : np.matrix, optional
    Alternative output matrix in which to place the result. It must
    have the same shape as the expected output, but the type of the
    output values will be cast if necessary.

    .. versionadded:: 0.18.0

Returns
-------
m : np.matrix

See Also
--------
numpy.matrix.mean : NumPy's implementation of 'mean' for matrices
*)

val minimum : other:Py.Object.t -> t -> Py.Object.t
(**
Element-wise minimum between this and another matrix.
*)

val multiply : other:Py.Object.t -> t -> Py.Object.t
(**
Point-wise multiplication by another matrix
        
*)

val nonzero : t -> Py.Object.t
(**
nonzero indices

Returns a tuple of arrays (row,col) containing the indices
of the non-zero elements of the matrix.

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> A = csr_matrix([[1,2,0],[0,0,3],[4,0,5]])
>>> A.nonzero()
(array([0, 0, 1, 2, 2]), array([0, 1, 2, 0, 2]))
*)

val power : ?dtype:Py.Object.t -> n:Py.Object.t -> t -> Py.Object.t
(**
Element-wise power.
*)

val reshape : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> t -> Csr_matrix.t
(**
reshape(self, shape, order='C', copy=False)

Gives a new shape to a sparse matrix without changing its data.

Parameters
----------
shape : length-2 tuple of ints
    The new shape should be compatible with the original shape.
order : {'C', 'F'}, optional
    Read the elements using this index order. 'C' means to read and
    write the elements using C-like index order; e.g. read entire first
    row, then second row, etc. 'F' means to read and write the elements
    using Fortran-like index order; e.g. read entire first column, then
    second column, etc.
copy : bool, optional
    Indicates whether or not attributes of self should be copied
    whenever possible. The degree to which attributes are copied varies
    depending on the type of sparse matrix being used.

Returns
-------
reshaped_matrix : sparse matrix
    A sparse matrix with the given `shape`, not necessarily of the same
    format as the current object.

See Also
--------
numpy.matrix.reshape : NumPy's implementation of 'reshape' for
                       matrices
*)

val resize : int list -> t -> Py.Object.t
(**
Resize the matrix in-place to dimensions given by ``shape``

Any elements that lie within the new shape will remain at the same
indices, while non-zero elements lying outside the new shape are
removed.

Parameters
----------
shape : (int, int)
    number of rows and columns in the new matrix

Notes
-----
The semantics are not identical to `numpy.ndarray.resize` or
`numpy.resize`.  Here, the same data will be maintained at each index
before and after reshape, if that index is within the new bounds.  In
numpy, resizing maintains contiguity of the array, moving elements
around in the logical matrix but not within a flattened representation.

We give no guarantees about whether the underlying data attributes
(arrays, etc.) will be modified in place or replaced with new objects.
*)

val set_shape : shape:int list -> t -> Py.Object.t
(**
See `reshape`.
*)

val setdiag : ?k:int -> values:Ndarray.t -> t -> Py.Object.t
(**
Set diagonal or off-diagonal elements of the array.

Parameters
----------
values : array_like
    New values of the diagonal elements.

    Values may have any length.  If the diagonal is longer than values,
    then the remaining diagonal entries will not be set.  If values if
    longer than the diagonal, then the remaining values are ignored.

    If a scalar value is given, all of the diagonal is set to it.

k : int, optional
    Which off-diagonal to set, corresponding to elements a[i,i+k].
    Default: 0 (the main diagonal).
*)

val sum : ?axis:[`None | `PyObject of Py.Object.t] -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> t -> Py.Object.t
(**
Sum the matrix elements over a given axis.

Parameters
----------
axis : {-2, -1, 0, 1, None} optional
    Axis along which the sum is computed. The default is to
    compute the sum of all the matrix elements, returning a scalar
    (i.e. `axis` = `None`).
dtype : dtype, optional
    The type of the returned matrix and of the accumulator in which
    the elements are summed.  The dtype of `a` is used by default
    unless `a` has an integer dtype of less precision than the default
    platform integer.  In that case, if `a` is signed then the platform
    integer is used while if `a` is unsigned then an unsigned integer
    of the same precision as the platform integer is used.

    .. versionadded:: 0.18.0

out : np.matrix, optional
    Alternative output matrix in which to place the result. It must
    have the same shape as the expected output, but the type of the
    output values will be cast if necessary.

    .. versionadded:: 0.18.0

Returns
-------
sum_along_axis : np.matrix
    A matrix with the same shape as `self`, with the specified
    axis removed.

See Also
--------
numpy.matrix.sum : NumPy's implementation of 'sum' for matrices
*)

val toarray : ?order:[`C | `F] -> ?out:Ndarray.t -> t -> Ndarray.t
(**
Return a dense ndarray representation of this matrix.

Parameters
----------
order : {'C', 'F'}, optional
    Whether to store multi-dimensional data in C (row-major)
    or Fortran (column-major) order in memory. The default
    is 'None', indicating the NumPy default of C-ordered.
    Cannot be specified in conjunction with the `out`
    argument.

out : ndarray, 2-dimensional, optional
    If specified, uses this array as the output buffer
    instead of allocating a new array to return. The provided
    array must have the same shape and dtype as the sparse
    matrix on which you are calling the method. For most
    sparse types, `out` is required to be memory contiguous
    (either C or Fortran ordered).

Returns
-------
arr : ndarray, 2-dimensional
    An array with the same shape and containing the same
    data represented by the sparse matrix, with the requested
    memory order. If `out` was passed, the same object is
    returned after being modified in-place to contain the
    appropriate values.
*)

val tobsr : ?blocksize:Py.Object.t -> ?copy:Py.Object.t -> t -> Py.Object.t
(**
Convert this matrix to Block Sparse Row format.

With copy=False, the data/indices may be shared between this matrix and
the resultant bsr_matrix.

When blocksize=(R, C) is provided, it will be used for construction of
the bsr_matrix.
*)

val tocoo : ?copy:Py.Object.t -> t -> Py.Object.t
(**
Convert this matrix to COOrdinate format.

With copy=False, the data/indices may be shared between this matrix and
the resultant coo_matrix.
*)

val tocsc : ?copy:Py.Object.t -> t -> Py.Object.t
(**
Convert this matrix to Compressed Sparse Column format.

With copy=False, the data/indices may be shared between this matrix and
the resultant csc_matrix.
*)

val tocsr : ?copy:Py.Object.t -> t -> Py.Object.t
(**
Convert this matrix to Compressed Sparse Row format.

With copy=False, the data/indices may be shared between this matrix and
the resultant csr_matrix.
*)

val todense : ?order:[`C | `F] -> ?out:Ndarray.t -> t -> Py.Object.t
(**
Return a dense matrix representation of this matrix.

Parameters
----------
order : {'C', 'F'}, optional
    Whether to store multi-dimensional data in C (row-major)
    or Fortran (column-major) order in memory. The default
    is 'None', indicating the NumPy default of C-ordered.
    Cannot be specified in conjunction with the `out`
    argument.

out : ndarray, 2-dimensional, optional
    If specified, uses this array (or `numpy.matrix`) as the
    output buffer instead of allocating a new array to
    return. The provided array must have the same shape and
    dtype as the sparse matrix on which you are calling the
    method.

Returns
-------
arr : numpy.matrix, 2-dimensional
    A NumPy matrix object with the same shape and containing
    the same data represented by the sparse matrix, with the
    requested memory order. If `out` was passed and was an
    array (rather than a `numpy.matrix`), it will be filled
    with the appropriate values and returned wrapped in a
    `numpy.matrix` object that shares the same memory.
*)

val todia : ?copy:Py.Object.t -> t -> Py.Object.t
(**
Convert this matrix to sparse DIAgonal format.

With copy=False, the data/indices may be shared between this matrix and
the resultant dia_matrix.
*)

val todok : ?copy:Py.Object.t -> t -> Py.Object.t
(**
Convert this matrix to Dictionary Of Keys format.

With copy=False, the data/indices may be shared between this matrix and
the resultant dok_matrix.
*)

val tolil : ?copy:Py.Object.t -> t -> Py.Object.t
(**
Convert this matrix to List of Lists format.

With copy=False, the data/indices may be shared between this matrix and
the resultant lil_matrix.
*)

val transpose : ?axes:Py.Object.t -> ?copy:Py.Object.t -> t -> Py.Object.t
(**
Reverses the dimensions of the sparse matrix.

Parameters
----------
axes : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except
    for the default value.
copy : bool, optional
    Indicates whether or not attributes of `self` should be
    copied whenever possible. The degree to which attributes
    are copied varies depending on the type of sparse matrix
    being used.

Returns
-------
p : `self` with the dimensions reversed.

See Also
--------
numpy.matrix.transpose : NumPy's implementation of 'transpose'
                         for matrices
*)


(** Attribute dtype: see constructor for documentation *)
val dtype : t -> Py.Object.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

module Spmatrix : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?maxprint:Py.Object.t -> unit -> t
(**
This class provides a base class for all sparse matrices.  It
cannot be instantiated.  Most of the work is provided by subclasses.
*)

val asformat : ?copy:Py.Object.t -> format:string -> t -> Py.Object.t
(**
Return this matrix in the passed format.

Parameters
----------
format : {str, None}
    The desired matrix format ("csr", "csc", "lil", "dok", "array", ...)
    or None for no conversion.
copy : bool, optional
    If True, the result is guaranteed to not share data with self.

Returns
-------
A : This matrix in the passed format.
*)

val asfptype : t -> Py.Object.t
(**
Upcast matrix to a floating point format (if necessary)
*)

val astype : ?casting:Py.Object.t -> ?copy:Py.Object.t -> dtype:[`String of string | `PyObject of Py.Object.t] -> t -> Py.Object.t
(**
Cast the matrix elements to a specified type.

Parameters
----------
dtype : string or numpy dtype
    Typecode or data-type to which to cast the data.
casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
    Controls what kind of data casting may occur.
    Defaults to 'unsafe' for backwards compatibility.
    'no' means the data types should not be cast at all.
    'equiv' means only byte-order changes are allowed.
    'safe' means only casts which can preserve values are allowed.
    'same_kind' means only safe casts or casts within a kind,
    like float64 to float32, are allowed.
    'unsafe' means any data conversions may be done.
copy : bool, optional
    If `copy` is `False`, the result might share some memory with this
    matrix. If `copy` is `True`, it is guaranteed that the result and
    this matrix do not share any memory.
*)

val conj : ?copy:bool -> t -> Py.Object.t
(**
Element-wise complex conjugation.

If the matrix is of non-complex data type and `copy` is False,
this method does nothing and the data is not copied.

Parameters
----------
copy : bool, optional
    If True, the result is guaranteed to not share data with self.

Returns
-------
A : The element-wise complex conjugate.
*)

val conjugate : ?copy:bool -> t -> Py.Object.t
(**
Element-wise complex conjugation.

If the matrix is of non-complex data type and `copy` is False,
this method does nothing and the data is not copied.

Parameters
----------
copy : bool, optional
    If True, the result is guaranteed to not share data with self.

Returns
-------
A : The element-wise complex conjugate.
*)

val copy : t -> Py.Object.t
(**
Returns a copy of this matrix.

No data/indices will be shared between the returned value and current
matrix.
*)

val count_nonzero : t -> Py.Object.t
(**
Number of non-zero entries, equivalent to

np.count_nonzero(a.toarray())

Unlike getnnz() and the nnz property, which return the number of stored
entries (the length of the data attribute), this method counts the
actual number of non-zero entries in data.
*)

val diagonal : ?k:int -> t -> Py.Object.t
(**
Returns the k-th diagonal of the matrix.

Parameters
----------
k : int, optional
    Which diagonal to get, corresponding to elements a[i, i+k].
    Default: 0 (the main diagonal).

    .. versionadded:: 1.0

See also
--------
numpy.diagonal : Equivalent numpy function.

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
>>> A.diagonal()
array([1, 0, 5])
>>> A.diagonal(k=1)
array([2, 3])
*)

val dot : other:Py.Object.t -> t -> Py.Object.t
(**
Ordinary dot product

Examples
--------
>>> import numpy as np
>>> from scipy.sparse import csr_matrix
>>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
>>> v = np.array([1, 0, -1])
>>> A.dot(v)
array([ 1, -3, -1], dtype=int64)
*)

val getH : t -> Py.Object.t
(**
Return the Hermitian transpose of this matrix.

See Also
--------
numpy.matrix.getH : NumPy's implementation of `getH` for matrices
*)

val get_shape : t -> Py.Object.t
(**
Get shape of a matrix.
*)

val getcol : j:Py.Object.t -> t -> Py.Object.t
(**
Returns a copy of column j of the matrix, as an (m x 1) sparse
matrix (column vector).
*)

val getformat : t -> Py.Object.t
(**
Format of a matrix representation as a string.
*)

val getmaxprint : t -> Py.Object.t
(**
Maximum number of elements to display when printed.
*)

val getnnz : ?axis:[`None | `PyObject of Py.Object.t] -> t -> Py.Object.t
(**
Number of stored values, including explicit zeros.

Parameters
----------
axis : None, 0, or 1
    Select between the number of values across the whole matrix, in
    each column, or in each row.

See also
--------
count_nonzero : Number of non-zero entries
*)

val getrow : i:Py.Object.t -> t -> Py.Object.t
(**
Returns a copy of row i of the matrix, as a (1 x n) sparse
matrix (row vector).
*)

val maximum : other:Py.Object.t -> t -> Py.Object.t
(**
Element-wise maximum between this and another matrix.
*)

val mean : ?axis:[`None | `PyObject of Py.Object.t] -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> t -> Py.Object.t
(**
Compute the arithmetic mean along the specified axis.

Returns the average of the matrix elements. The average is taken
over all elements in the matrix by default, otherwise over the
specified axis. `float64` intermediate and return values are used
for integer inputs.

Parameters
----------
axis : {-2, -1, 0, 1, None} optional
    Axis along which the mean is computed. The default is to compute
    the mean of all elements in the matrix (i.e. `axis` = `None`).
dtype : data-type, optional
    Type to use in computing the mean. For integer inputs, the default
    is `float64`; for floating point inputs, it is the same as the
    input dtype.

    .. versionadded:: 0.18.0

out : np.matrix, optional
    Alternative output matrix in which to place the result. It must
    have the same shape as the expected output, but the type of the
    output values will be cast if necessary.

    .. versionadded:: 0.18.0

Returns
-------
m : np.matrix

See Also
--------
numpy.matrix.mean : NumPy's implementation of 'mean' for matrices
*)

val minimum : other:Py.Object.t -> t -> Py.Object.t
(**
Element-wise minimum between this and another matrix.
*)

val multiply : other:Py.Object.t -> t -> Py.Object.t
(**
Point-wise multiplication by another matrix
        
*)

val nonzero : t -> Py.Object.t
(**
nonzero indices

Returns a tuple of arrays (row,col) containing the indices
of the non-zero elements of the matrix.

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> A = csr_matrix([[1,2,0],[0,0,3],[4,0,5]])
>>> A.nonzero()
(array([0, 0, 1, 2, 2]), array([0, 1, 2, 0, 2]))
*)

val power : ?dtype:Py.Object.t -> n:Py.Object.t -> t -> Py.Object.t
(**
Element-wise power.
*)

val reshape : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> t -> Csr_matrix.t
(**
reshape(self, shape, order='C', copy=False)

Gives a new shape to a sparse matrix without changing its data.

Parameters
----------
shape : length-2 tuple of ints
    The new shape should be compatible with the original shape.
order : {'C', 'F'}, optional
    Read the elements using this index order. 'C' means to read and
    write the elements using C-like index order; e.g. read entire first
    row, then second row, etc. 'F' means to read and write the elements
    using Fortran-like index order; e.g. read entire first column, then
    second column, etc.
copy : bool, optional
    Indicates whether or not attributes of self should be copied
    whenever possible. The degree to which attributes are copied varies
    depending on the type of sparse matrix being used.

Returns
-------
reshaped_matrix : sparse matrix
    A sparse matrix with the given `shape`, not necessarily of the same
    format as the current object.

See Also
--------
numpy.matrix.reshape : NumPy's implementation of 'reshape' for
                       matrices
*)

val resize : shape:int list -> t -> Py.Object.t
(**
Resize the matrix in-place to dimensions given by ``shape``

Any elements that lie within the new shape will remain at the same
indices, while non-zero elements lying outside the new shape are
removed.

Parameters
----------
shape : (int, int)
    number of rows and columns in the new matrix

Notes
-----
The semantics are not identical to `numpy.ndarray.resize` or
`numpy.resize`.  Here, the same data will be maintained at each index
before and after reshape, if that index is within the new bounds.  In
numpy, resizing maintains contiguity of the array, moving elements
around in the logical matrix but not within a flattened representation.

We give no guarantees about whether the underlying data attributes
(arrays, etc.) will be modified in place or replaced with new objects.
*)

val set_shape : shape:int list -> t -> Py.Object.t
(**
See `reshape`.
*)

val setdiag : ?k:int -> values:Ndarray.t -> t -> Py.Object.t
(**
Set diagonal or off-diagonal elements of the array.

Parameters
----------
values : array_like
    New values of the diagonal elements.

    Values may have any length.  If the diagonal is longer than values,
    then the remaining diagonal entries will not be set.  If values if
    longer than the diagonal, then the remaining values are ignored.

    If a scalar value is given, all of the diagonal is set to it.

k : int, optional
    Which off-diagonal to set, corresponding to elements a[i,i+k].
    Default: 0 (the main diagonal).
*)

val sum : ?axis:[`None | `PyObject of Py.Object.t] -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> t -> Py.Object.t
(**
Sum the matrix elements over a given axis.

Parameters
----------
axis : {-2, -1, 0, 1, None} optional
    Axis along which the sum is computed. The default is to
    compute the sum of all the matrix elements, returning a scalar
    (i.e. `axis` = `None`).
dtype : dtype, optional
    The type of the returned matrix and of the accumulator in which
    the elements are summed.  The dtype of `a` is used by default
    unless `a` has an integer dtype of less precision than the default
    platform integer.  In that case, if `a` is signed then the platform
    integer is used while if `a` is unsigned then an unsigned integer
    of the same precision as the platform integer is used.

    .. versionadded:: 0.18.0

out : np.matrix, optional
    Alternative output matrix in which to place the result. It must
    have the same shape as the expected output, but the type of the
    output values will be cast if necessary.

    .. versionadded:: 0.18.0

Returns
-------
sum_along_axis : np.matrix
    A matrix with the same shape as `self`, with the specified
    axis removed.

See Also
--------
numpy.matrix.sum : NumPy's implementation of 'sum' for matrices
*)

val toarray : ?order:[`C | `F] -> ?out:Ndarray.t -> t -> Ndarray.t
(**
Return a dense ndarray representation of this matrix.

Parameters
----------
order : {'C', 'F'}, optional
    Whether to store multi-dimensional data in C (row-major)
    or Fortran (column-major) order in memory. The default
    is 'None', indicating the NumPy default of C-ordered.
    Cannot be specified in conjunction with the `out`
    argument.

out : ndarray, 2-dimensional, optional
    If specified, uses this array as the output buffer
    instead of allocating a new array to return. The provided
    array must have the same shape and dtype as the sparse
    matrix on which you are calling the method. For most
    sparse types, `out` is required to be memory contiguous
    (either C or Fortran ordered).

Returns
-------
arr : ndarray, 2-dimensional
    An array with the same shape and containing the same
    data represented by the sparse matrix, with the requested
    memory order. If `out` was passed, the same object is
    returned after being modified in-place to contain the
    appropriate values.
*)

val tobsr : ?blocksize:Py.Object.t -> ?copy:Py.Object.t -> t -> Py.Object.t
(**
Convert this matrix to Block Sparse Row format.

With copy=False, the data/indices may be shared between this matrix and
the resultant bsr_matrix.

When blocksize=(R, C) is provided, it will be used for construction of
the bsr_matrix.
*)

val tocoo : ?copy:Py.Object.t -> t -> Py.Object.t
(**
Convert this matrix to COOrdinate format.

With copy=False, the data/indices may be shared between this matrix and
the resultant coo_matrix.
*)

val tocsc : ?copy:Py.Object.t -> t -> Py.Object.t
(**
Convert this matrix to Compressed Sparse Column format.

With copy=False, the data/indices may be shared between this matrix and
the resultant csc_matrix.
*)

val tocsr : ?copy:Py.Object.t -> t -> Py.Object.t
(**
Convert this matrix to Compressed Sparse Row format.

With copy=False, the data/indices may be shared between this matrix and
the resultant csr_matrix.
*)

val todense : ?order:[`C | `F] -> ?out:Ndarray.t -> t -> Py.Object.t
(**
Return a dense matrix representation of this matrix.

Parameters
----------
order : {'C', 'F'}, optional
    Whether to store multi-dimensional data in C (row-major)
    or Fortran (column-major) order in memory. The default
    is 'None', indicating the NumPy default of C-ordered.
    Cannot be specified in conjunction with the `out`
    argument.

out : ndarray, 2-dimensional, optional
    If specified, uses this array (or `numpy.matrix`) as the
    output buffer instead of allocating a new array to
    return. The provided array must have the same shape and
    dtype as the sparse matrix on which you are calling the
    method.

Returns
-------
arr : numpy.matrix, 2-dimensional
    A NumPy matrix object with the same shape and containing
    the same data represented by the sparse matrix, with the
    requested memory order. If `out` was passed and was an
    array (rather than a `numpy.matrix`), it will be filled
    with the appropriate values and returned wrapped in a
    `numpy.matrix` object that shares the same memory.
*)

val todia : ?copy:Py.Object.t -> t -> Py.Object.t
(**
Convert this matrix to sparse DIAgonal format.

With copy=False, the data/indices may be shared between this matrix and
the resultant dia_matrix.
*)

val todok : ?copy:Py.Object.t -> t -> Py.Object.t
(**
Convert this matrix to Dictionary Of Keys format.

With copy=False, the data/indices may be shared between this matrix and
the resultant dok_matrix.
*)

val tolil : ?copy:Py.Object.t -> t -> Py.Object.t
(**
Convert this matrix to List of Lists format.

With copy=False, the data/indices may be shared between this matrix and
the resultant lil_matrix.
*)

val transpose : ?axes:Py.Object.t -> ?copy:Py.Object.t -> t -> Py.Object.t
(**
Reverses the dimensions of the sparse matrix.

Parameters
----------
axes : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except
    for the default value.
copy : bool, optional
    Indicates whether or not attributes of `self` should be
    copied whenever possible. The degree to which attributes
    are copied varies depending on the type of sparse matrix
    being used.

Returns
-------
p : `self` with the dimensions reversed.

See Also
--------
numpy.matrix.transpose : NumPy's implementation of 'transpose'
                         for matrices
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

val type_of_target : y:Ndarray.t -> unit -> string
(**
Determine the type of data indicated by the target.

Note that this type is the most specific type that can be inferred.
For example:

    * ``binary`` is more specific but compatible with ``multiclass``.
    * ``multiclass`` of integers is more specific but compatible with
      ``continuous``.
    * ``multilabel-indicator`` is more specific but compatible with
      ``multiclass-multioutput``.

Parameters
----------
y : array-like

Returns
-------
target_type : string
    One of:

    * 'continuous': `y` is an array-like of floats that are not all
      integers, and is 1d or a column vector.
    * 'continuous-multioutput': `y` is a 2d array of floats that are
      not all integers, and both dimensions are of size > 1.
    * 'binary': `y` contains <= 2 discrete values and is 1d or a column
      vector.
    * 'multiclass': `y` contains more than two discrete values, is not a
      sequence of sequences, and is 1d or a column vector.
    * 'multiclass-multioutput': `y` is a 2d array that contains more
      than two discrete values, is not a sequence of sequences, and both
      dimensions are of size > 1.
    * 'multilabel-indicator': `y` is a label indicator matrix, an array
      of two dimensions with at least two columns, and at most 2 unique
      values.
    * 'unknown': `y` is array-like but none of the above, such as a 3d
      array, sequence of sequences, or an array of non-sequence objects.

Examples
--------
>>> import numpy as np
>>> type_of_target([0.1, 0.6])
'continuous'
>>> type_of_target([1, -1, -1, 1])
'binary'
>>> type_of_target(['a', 'b', 'a'])
'binary'
>>> type_of_target([1.0, 2.0])
'binary'
>>> type_of_target([1, 0, 2])
'multiclass'
>>> type_of_target([1.0, 0.0, 3.0])
'multiclass'
>>> type_of_target(['a', 'b', 'c'])
'multiclass'
>>> type_of_target(np.array([[1, 2], [3, 1]]))
'multiclass-multioutput'
>>> type_of_target([[1, 2]])
'multilabel-indicator'
>>> type_of_target(np.array([[1.5, 2.0], [3.0, 1.6]]))
'continuous-multioutput'
>>> type_of_target(np.array([[0, 1], [1, 1]]))
'multilabel-indicator'
*)

val unique_labels : Py.Object.t list -> Ndarray.t
(**
Extract an ordered array of unique labels

We don't allow:
    - mix of multilabel and multiclass (single label) targets
    - mix of label indicator matrix and anything else,
      because there are no explicit labels)
    - mix of label indicator matrices of different sizes
    - mix of string and integer labels

At the moment, we also don't allow "multiclass-multioutput" input type.

Parameters
----------
*ys : array-likes

Returns
-------
out : numpy array of shape [n_unique_labels]
    An ordered array of unique labels.

Examples
--------
>>> from sklearn.utils.multiclass import unique_labels
>>> unique_labels([3, 5, 5, 5, 7, 7])
array([3, 5, 7])
>>> unique_labels([1, 2, 3, 4], [2, 2, 3, 4])
array([1, 2, 3, 4])
>>> unique_labels([1, 2, 10], [5, 11])
array([ 1,  2,  5, 10, 11])
*)


end

module Murmurhash : sig

end

module Optimize : sig
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
val pp : Format.formatter -> t -> unit


end

val line_search_wolfe1 : ?gfk:Ndarray.t -> ?old_fval:Py.Object.t -> ?old_old_fval:Py.Object.t -> ?args:Py.Object.t -> ?c1:Py.Object.t -> ?c2:Py.Object.t -> ?amax:Py.Object.t -> ?amin:Py.Object.t -> ?xtol:Py.Object.t -> f:Py.Object.t -> fprime:Py.Object.t -> xk:Py.Object.t -> pk:Py.Object.t -> unit -> Py.Object.t
(**
As `scalar_search_wolfe1` but do a line search to direction `pk`

Parameters
----------
f : callable
    Function `f(x)`
fprime : callable
    Gradient of `f`
xk : array_like
    Current point
pk : array_like
    Search direction

gfk : array_like, optional
    Gradient of `f` at point `xk`
old_fval : float, optional
    Value of `f` at point `xk`
old_old_fval : float, optional
    Value of `f` at point preceding `xk`

The rest of the parameters are the same as for `scalar_search_wolfe1`.

Returns
-------
stp, f_count, g_count, fval, old_fval
    As in `line_search_wolfe1`
gval : array
    Gradient of `f` at the final point
*)

val line_search_wolfe2 : ?gfk:Py.Object.t -> ?old_fval:Py.Object.t -> ?old_old_fval:Py.Object.t -> ?args:Py.Object.t -> ?c1:Py.Object.t -> ?c2:Py.Object.t -> ?amax:Py.Object.t -> ?extra_condition:Py.Object.t -> ?maxiter:Py.Object.t -> f:Py.Object.t -> myfprime:Py.Object.t -> xk:Py.Object.t -> pk:Py.Object.t -> unit -> Py.Object.t
(**
Find alpha that satisfies strong Wolfe conditions.

Parameters
----------
f : callable f(x,*args)
    Objective function.
myfprime : callable f'(x,*args)
    Objective function gradient.
xk : ndarray
    Starting point.
pk : ndarray
    Search direction.
gfk : ndarray, optional
    Gradient value for x=xk (xk being the current parameter
    estimate). Will be recomputed if omitted.
old_fval : float, optional
    Function value for x=xk. Will be recomputed if omitted.
old_old_fval : float, optional
    Function value for the point preceding x=xk
args : tuple, optional
    Additional arguments passed to objective function.
c1 : float, optional
    Parameter for Armijo condition rule.
c2 : float, optional
    Parameter for curvature condition rule.
amax : float, optional
    Maximum step size
extra_condition : callable, optional
    A callable of the form ``extra_condition(alpha, x, f, g)``
    returning a boolean. Arguments are the proposed step ``alpha``
    and the corresponding ``x``, ``f`` and ``g`` values. The line search 
    accepts the value of ``alpha`` only if this 
    callable returns ``True``. If the callable returns ``False`` 
    for the step length, the algorithm will continue with 
    new iterates. The callable is only called for iterates 
    satisfying the strong Wolfe conditions.
maxiter : int, optional
    Maximum number of iterations to perform

Returns
-------
alpha : float or None
    Alpha for which ``x_new = x0 + alpha * pk``,
    or None if the line search algorithm did not converge.
fc : int
    Number of function evaluations made.
gc : int
    Number of gradient evaluations made.
new_fval : float or None
    New function value ``f(x_new)=f(x0+alpha*pk)``,
    or None if the line search algorithm did not converge.
old_fval : float
    Old function value ``f(x0)``.
new_slope : float or None
    The local slope along the search direction at the
    new value ``<myfprime(x_new), pk>``,
    or None if the line search algorithm did not converge.


Notes
-----
Uses the line search algorithm to enforce strong Wolfe
conditions.  See Wright and Nocedal, 'Numerical Optimization',
1999, pg. 59-60.

For the zoom phase it uses an algorithm by [...].
*)

val newton_cg : ?args:Py.Object.t -> ?tol:Py.Object.t -> ?maxiter:Py.Object.t -> ?maxinner:Py.Object.t -> ?line_search:Py.Object.t -> ?warn:Py.Object.t -> grad_hess:Py.Object.t -> func:Py.Object.t -> grad:Py.Object.t -> x0:Py.Object.t -> unit -> Py.Object.t
(**
DEPRECATED: newton_cg is deprecated in version 0.22 and will be removed in version 0.24.
*)


end

module Parallel_backend : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_jobs:Py.Object.t -> ?inner_max_num_threads:Py.Object.t -> ?backend_params:(string * Py.Object.t) list -> backend:Py.Object.t -> unit -> t
(**
Change the default backend used by Parallel inside a with block.

If ``backend`` is a string it must match a previously registered
implementation using the ``register_parallel_backend`` function.

By default the following backends are available:

- 'loky': single-host, process-based parallelism (used by default),
- 'threading': single-host, thread-based parallelism,
- 'multiprocessing': legacy single-host, process-based parallelism.

'loky' is recommended to run functions that manipulate Python objects.
'threading' is a low-overhead alternative that is most efficient for
functions that release the Global Interpreter Lock: e.g. I/O-bound code or
CPU-bound code in a few calls to native code that explicitly releases the
GIL.

In addition, if the `dask` and `distributed` Python packages are installed,
it is possible to use the 'dask' backend for better scheduling of nested
parallel calls without over-subscription and potentially distribute
parallel calls over a networked cluster of several hosts.

Alternatively the backend can be passed directly as an instance.

By default all available workers will be used (``n_jobs=-1``) unless the
caller passes an explicit value for the ``n_jobs`` parameter.

This is an alternative to passing a ``backend='backend_name'`` argument to
the ``Parallel`` class constructor. It is particularly useful when calling
into library code that uses joblib internally but does not expose the
backend argument in its own API.

>>> from operator import neg
>>> with parallel_backend('threading'):
...     print(Parallel()(delayed(neg)(i + 1) for i in range(5)))
...
[-1, -2, -3, -4, -5]

Warning: this function is experimental and subject to change in a future
version of joblib.

Joblib also tries to limit the oversubscription by limiting the number of
threads usable in some third-party library threadpools like OpenBLAS, MKL
or OpenMP. The default limit in each worker is set to
``max(cpu_count() // effective_n_jobs, 1)`` but this limit can be
overwritten with the ``inner_max_num_threads`` argument which will be used
to set this limit in the child processes.

.. versionadded:: 0.10
*)

val unregister : t -> Py.Object.t
(**
None
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

module Random : sig
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
val pp : Format.formatter -> t -> unit


end

val random_choice_csc : ?class_probability:Py.Object.t -> ?random_state:Py.Object.t -> n_samples:Py.Object.t -> classes:Py.Object.t -> unit -> Py.Object.t
(**
DEPRECATED: random_choice_csc is deprecated in version 0.22 and will be removed in version 0.24.
*)


end

val register_parallel_backend : ?make_default:Py.Object.t -> name:Py.Object.t -> factory:Py.Object.t -> unit -> Py.Object.t
(**
Register a new Parallel backend factory.

The new backend can then be selected by passing its name as the backend
argument to the Parallel class. Moreover, the default backend can be
overwritten globally by setting make_default=True.

The factory can be any callable that takes no argument and return an
instance of ``ParallelBackendBase``.

Warning: this function is experimental and subject to change in a future
version of joblib.

.. versionadded:: 0.10
*)

val resample : ?options:(string * Py.Object.t) list -> Py.Object.t list -> Py.Object.t
(**
Resample arrays or sparse matrices in a consistent way

The default strategy implements one step of the bootstrapping
procedure.

Parameters
----------
*arrays : sequence of indexable data-structures
    Indexable data-structures can be arrays, lists, dataframes or scipy
    sparse matrices with consistent first dimension.

Other Parameters
----------------
replace : boolean, True by default
    Implements resampling with replacement. If False, this will implement
    (sliced) random permutations.

n_samples : int, None by default
    Number of samples to generate. If left to None this is
    automatically set to the first dimension of the arrays.
    If replace is False it should not be larger than the length of
    arrays.

random_state : int, RandomState instance or None, optional (default=None)
    The seed of the pseudo random number generator to use when shuffling
    the data.  If int, random_state is the seed used by the random number
    generator; If RandomState instance, random_state is the random number
    generator; If None, the random number generator is the RandomState
    instance used by `np.random`.

stratify : array-like or None (default=None)
    If not None, data is split in a stratified fashion, using this as
    the class labels.

Returns
-------
resampled_arrays : sequence of indexable data-structures
    Sequence of resampled copies of the collections. The original arrays
    are not impacted.

Examples
--------
It is possible to mix sparse and dense arrays in the same run::

  >>> X = np.array([[1., 0.], [2., 1.], [0., 0.]])
  >>> y = np.array([0, 1, 2])

  >>> from scipy.sparse import coo_matrix
  >>> X_sparse = coo_matrix(X)

  >>> from sklearn.utils import resample
  >>> X, X_sparse, y = resample(X, X_sparse, y, random_state=0)
  >>> X
  array([[1., 0.],
         [2., 1.],
         [1., 0.]])

  >>> X_sparse
  <3x2 sparse matrix of type '<... 'numpy.float64'>'
      with 4 stored elements in Compressed Sparse Row format>

  >>> X_sparse.toarray()
  array([[1., 0.],
         [2., 1.],
         [1., 0.]])

  >>> y
  array([0, 1, 0])

  >>> resample(y, n_samples=2, random_state=0)
  array([0, 1])

Example using stratification::

  >>> y = [0, 0, 1, 1, 1, 1, 1, 1, 1]
  >>> resample(y, n_samples=5, replace=False, stratify=y,
  ...          random_state=0)
  [1, 1, 1, 0, 1]


See also
--------
:func:`sklearn.utils.shuffle`
*)

val safe_indexing : ?axis:int -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t | `ArrayLike of Py.Object.t | `PyObject of Py.Object.t] -> indices:[`Bool of bool | `Int of int | `String of string | `Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> unit -> Py.Object.t
(**
DEPRECATED: safe_indexing is deprecated in version 0.22 and will be removed in version 0.24.

Return rows, items or columns of X using indices.

    .. deprecated:: 0.22
        This function was deprecated in version 0.22 and will be removed in
        version 0.24.

    Parameters
    ----------
    X : array-like, sparse-matrix, list, pandas.DataFrame, pandas.Series
        Data from which to sample rows, items or columns. `list` are only
        supported when `axis=0`.

    indices : bool, int, str, slice, array-like

        - If `axis=0`, boolean and integer array-like, integer slice,
          and scalar integer are supported.
        - If `axis=1`:

            - to select a single column, `indices` can be of `int` type for
              all `X` types and `str` only for dataframe. The selected subset
              will be 1D, unless `X` is a sparse matrix in which case it will
              be 2D.
            - to select multiples columns, `indices` can be one of the
              following: `list`, `array`, `slice`. The type used in
              these containers can be one of the following: `int`, 'bool' and
              `str`. However, `str` is only supported when `X` is a dataframe.
              The selected subset will be 2D.

    axis : int, default=0
        The axis along which `X` will be subsampled. `axis=0` will select
        rows while `axis=1` will select columns.

    Returns
    -------
    subset
        Subset of X on axis 0 or 1.

    Notes
    -----
    CSR, CSC, and LIL sparse matrices are supported. COO sparse matrices are
    not supported.
    
*)

val safe_mask : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> mask:Ndarray.t -> unit -> Py.Object.t
(**
Return a mask which is safe to use on X.

Parameters
----------
X : {array-like, sparse matrix}
    Data on which to apply mask.

mask : array
    Mask to be used on X.

Returns
-------
    mask
*)

val safe_sqr : ?copy:bool -> x:[`SparseMatrix of Csr_matrix.t | `PyObject of Py.Object.t] -> unit -> Py.Object.t
(**
Element wise squaring of array-likes and sparse matrices.

Parameters
----------
X : array like, matrix, sparse matrix

copy : boolean, optional, default True
    Whether to create a copy of X and operate on it or to perform
    inplace computation (default behaviour).

Returns
-------
X ** 2 : element wise square
*)

val shuffle : ?options:(string * Py.Object.t) list -> Py.Object.t list -> Py.Object.t
(**
Shuffle arrays or sparse matrices in a consistent way

This is a convenience alias to ``resample( *arrays, replace=False)`` to do
random permutations of the collections.

Parameters
----------
*arrays : sequence of indexable data-structures
    Indexable data-structures can be arrays, lists, dataframes or scipy
    sparse matrices with consistent first dimension.

Other Parameters
----------------
random_state : int, RandomState instance or None, optional (default=None)
    The seed of the pseudo random number generator to use when shuffling
    the data.  If int, random_state is the seed used by the random number
    generator; If RandomState instance, random_state is the random number
    generator; If None, the random number generator is the RandomState
    instance used by `np.random`.

n_samples : int, None by default
    Number of samples to generate. If left to None this is
    automatically set to the first dimension of the arrays.

Returns
-------
shuffled_arrays : sequence of indexable data-structures
    Sequence of shuffled copies of the collections. The original arrays
    are not impacted.

Examples
--------
It is possible to mix sparse and dense arrays in the same run::

  >>> X = np.array([[1., 0.], [2., 1.], [0., 0.]])
  >>> y = np.array([0, 1, 2])

  >>> from scipy.sparse import coo_matrix
  >>> X_sparse = coo_matrix(X)

  >>> from sklearn.utils import shuffle
  >>> X, X_sparse, y = shuffle(X, X_sparse, y, random_state=0)
  >>> X
  array([[0., 0.],
         [2., 1.],
         [1., 0.]])

  >>> X_sparse
  <3x2 sparse matrix of type '<... 'numpy.float64'>'
      with 3 stored elements in Compressed Sparse Row format>

  >>> X_sparse.toarray()
  array([[0., 0.],
         [2., 1.],
         [1., 0.]])

  >>> y
  array([2, 1, 0])

  >>> shuffle(y, n_samples=2, random_state=0)
  array([0, 1])

See also
--------
:func:`sklearn.utils.resample`
*)

module Sparsefuncs : sig
val count_nonzero : ?axis:[`None | `PyObject of Py.Object.t] -> ?sample_weight:Ndarray.t -> x:Py.Object.t -> unit -> Py.Object.t
(**
A variant of X.getnnz() with extension to weighting on axis 0

Useful in efficiently calculating multilabel metrics.

Parameters
----------
X : CSR sparse matrix of shape (n_samples, n_labels)
    Input data.

axis : None, 0 or 1
    The axis on which the data is aggregated.

sample_weight : array-like of shape (n_samples,), default=None
    Weight for each row of X.
*)

val csc_median_axis_0 : x:Py.Object.t -> unit -> Ndarray.t
(**
Find the median across axis 0 of a CSC matrix.
It is equivalent to doing np.median(X, axis=0).

Parameters
----------
X : CSC sparse matrix, shape (n_samples, n_features)
    Input data.

Returns
-------
median : ndarray, shape (n_features,)
    Median.
*)

val incr_mean_variance_axis : x:Py.Object.t -> axis:Py.Object.t -> last_mean:Py.Object.t -> last_var:Py.Object.t -> last_n:Py.Object.t -> unit -> (Py.Object.t * Py.Object.t * Py.Object.t)
(**
Compute incremental mean and variance along an axix on a CSR or
CSC matrix.

last_mean, last_var are the statistics computed at the last step by this
function. Both must be initialized to 0-arrays of the proper size, i.e.
the number of features in X. last_n is the number of samples encountered
until now.

Parameters
----------
X : CSR or CSC sparse matrix, shape (n_samples, n_features)
    Input data.

axis : int (either 0 or 1)
    Axis along which the axis should be computed.

last_mean : float array with shape (n_features,)
    Array of feature-wise means to update with the new data X.

last_var : float array with shape (n_features,)
    Array of feature-wise var to update with the new data X.

last_n : int with shape (n_features,)
    Number of samples seen so far, excluded X.

Returns
-------

means : float array with shape (n_features,)
    Updated feature-wise means.

variances : float array with shape (n_features,)
    Updated feature-wise variances.

n : int with shape (n_features,)
    Updated number of seen samples.

Notes
-----
NaNs are ignored in the algorithm.
*)

val inplace_column_scale : x:Py.Object.t -> scale:Py.Object.t -> unit -> Py.Object.t
(**
Inplace column scaling of a CSC/CSR matrix.

Scale each feature of the data matrix by multiplying with specific scale
provided by the caller assuming a (n_samples, n_features) shape.

Parameters
----------
X : CSC or CSR matrix with shape (n_samples, n_features)
    Matrix to normalize using the variance of the features.

scale : float array with shape (n_features,)
    Array of precomputed feature-wise values to use for scaling.
*)

val inplace_csr_column_scale : x:Py.Object.t -> scale:Py.Object.t -> unit -> Py.Object.t
(**
Inplace column scaling of a CSR matrix.

Scale each feature of the data matrix by multiplying with specific scale
provided by the caller assuming a (n_samples, n_features) shape.

Parameters
----------
X : CSR matrix with shape (n_samples, n_features)
    Matrix to normalize using the variance of the features.

scale : float array with shape (n_features,)
    Array of precomputed feature-wise values to use for scaling.
*)

val inplace_csr_row_scale : x:Py.Object.t -> scale:Py.Object.t -> unit -> Py.Object.t
(**
Inplace row scaling of a CSR matrix.

Scale each sample of the data matrix by multiplying with specific scale
provided by the caller assuming a (n_samples, n_features) shape.

Parameters
----------
X : CSR sparse matrix, shape (n_samples, n_features)
    Matrix to be scaled.

scale : float array with shape (n_samples,)
    Array of precomputed sample-wise values to use for scaling.
*)

val inplace_row_scale : x:Py.Object.t -> scale:Py.Object.t -> unit -> Py.Object.t
(**
Inplace row scaling of a CSR or CSC matrix.

Scale each row of the data matrix by multiplying with specific scale
provided by the caller assuming a (n_samples, n_features) shape.

Parameters
----------
X : CSR or CSC sparse matrix, shape (n_samples, n_features)
    Matrix to be scaled.

scale : float array with shape (n_features,)
    Array of precomputed sample-wise values to use for scaling.
*)

val inplace_swap_column : x:Py.Object.t -> m:int -> n:int -> unit -> Py.Object.t
(**
Swaps two columns of a CSC/CSR matrix in-place.

Parameters
----------
X : CSR or CSC sparse matrix, shape=(n_samples, n_features)
    Matrix whose two columns are to be swapped.

m : int
    Index of the column of X to be swapped.

n : int
    Index of the column of X to be swapped.
*)

val inplace_swap_row : x:Py.Object.t -> m:int -> n:int -> unit -> Py.Object.t
(**
Swaps two rows of a CSC/CSR matrix in-place.

Parameters
----------
X : CSR or CSC sparse matrix, shape=(n_samples, n_features)
    Matrix whose two rows are to be swapped.

m : int
    Index of the row of X to be swapped.

n : int
    Index of the row of X to be swapped.
*)

val inplace_swap_row_csc : x:Py.Object.t -> m:int -> n:int -> unit -> Py.Object.t
(**
Swaps two rows of a CSC matrix in-place.

Parameters
----------
X : scipy.sparse.csc_matrix, shape=(n_samples, n_features)
    Matrix whose two rows are to be swapped.

m : int
    Index of the row of X to be swapped.

n : int
    Index of the row of X to be swapped.
*)

val inplace_swap_row_csr : x:Py.Object.t -> m:int -> n:int -> unit -> Py.Object.t
(**
Swaps two rows of a CSR matrix in-place.

Parameters
----------
X : scipy.sparse.csr_matrix, shape=(n_samples, n_features)
    Matrix whose two rows are to be swapped.

m : int
    Index of the row of X to be swapped.

n : int
    Index of the row of X to be swapped.
*)

val mean_variance_axis : x:Py.Object.t -> axis:Py.Object.t -> unit -> (Py.Object.t * Py.Object.t)
(**
Compute mean and variance along an axix on a CSR or CSC matrix

Parameters
----------
X : CSR or CSC sparse matrix, shape (n_samples, n_features)
    Input data.

axis : int (either 0 or 1)
    Axis along which the axis should be computed.

Returns
-------

means : float array with shape (n_features,)
    Feature-wise means

variances : float array with shape (n_features,)
    Feature-wise variances
*)

val min_max_axis : ?ignore_nan:bool -> x:Py.Object.t -> axis:Py.Object.t -> unit -> (Py.Object.t * Py.Object.t)
(**
Compute minimum and maximum along an axis on a CSR or CSC matrix and
optionally ignore NaN values.

Parameters
----------
X : CSR or CSC sparse matrix, shape (n_samples, n_features)
    Input data.

axis : int (either 0 or 1)
    Axis along which the axis should be computed.

ignore_nan : bool, default is False
    Ignore or passing through NaN values.

    .. versionadded:: 0.20

Returns
-------

mins : float array with shape (n_features,)
    Feature-wise minima

maxs : float array with shape (n_features,)
    Feature-wise maxima
*)


end

module Sparsefuncs_fast : sig
val assign_rows_csr : x:Py.Object.t -> x_rows:Py.Object.t -> out_rows:Py.Object.t -> out:Py.Object.t -> unit -> Py.Object.t
(**
Densify selected rows of a CSR matrix into a preallocated array.

Like out[out_rows] = X[X_rows].toarray() but without copying.
No-copy supported for both dtype=np.float32 and dtype=np.float64.

Parameters
----------
X : scipy.sparse.csr_matrix, shape=(n_samples, n_features)
X_rows : array, dtype=np.intp, shape=n_rows
out_rows : array, dtype=np.intp, shape=n_rows
out : array, shape=(arbitrary, n_features)
*)


end

module Stats : sig
val stable_cumsum : ?axis:Py.Object.t -> ?rtol:Py.Object.t -> ?atol:Py.Object.t -> arr:Ndarray.t -> unit -> Py.Object.t
(**
Use high precision for cumsum and check that final value matches sum

Parameters
----------
arr : array-like
    To be cumulatively summed as flat
axis : int, optional
    Axis along which the cumulative sum is computed.
    The default (None) is to compute the cumsum over the flattened array.
rtol : float
    Relative tolerance, see ``np.allclose``
atol : float
    Absolute tolerance, see ``np.allclose``
*)


end

val tosequence : x:Ndarray.t -> unit -> Py.Object.t
(**
Cast iterable x to a Sequence, avoiding a copy if possible.

Parameters
----------
x : iterable
*)

module Validation : sig
module LooseVersion : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?vstring:Py.Object.t -> unit -> t
(**
Version numbering for anarchists and software realists.
Implements the standard interface for version number classes as
described above.  A version number consists of a series of numbers,
separated by either periods or strings of letters.  When comparing
version numbers, the numeric components will be compared
numerically, and the alphabetic components lexically.  The following
are all valid version numbers, in no particular order:

    1.5.1
    1.5.2b2
    161
    3.10a
    8.02
    3.4j
    1996.07.12
    3.2.pl0
    3.1.1.6
    2g6
    11g
    0.960923
    2.2beta29
    1.13++
    5.5.kw
    2.0b1pl0

In fact, there is no such thing as an invalid version number under
this scheme; the rules for comparison are simple and predictable,
but may not always give the results you want (for some definition
of "want").
*)

val parse : vstring:Py.Object.t -> t -> Py.Object.t
(**
None
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

module Parameter : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : name:Py.Object.t -> kind:Py.Object.t -> default:Py.Object.t -> annotation:Py.Object.t -> unit -> t
(**
Represents a parameter in a function signature.

Has the following public attributes:

* name : str
    The name of the parameter as a string.
* default : object
    The default value for the parameter if specified.  If the
    parameter has no default value, this attribute is set to
    `Parameter.empty`.
* annotation
    The annotation for the parameter if specified.  If the
    parameter has no annotation, this attribute is set to
    `Parameter.empty`.
* kind : str
    Describes how argument values are bound to the parameter.
    Possible values: `Parameter.POSITIONAL_ONLY`,
    `Parameter.POSITIONAL_OR_KEYWORD`, `Parameter.VAR_POSITIONAL`,
    `Parameter.KEYWORD_ONLY`, `Parameter.VAR_KEYWORD`.
*)

val replace : ?name:Py.Object.t -> ?kind:Py.Object.t -> ?annotation:Py.Object.t -> ?default:Py.Object.t -> t -> Py.Object.t
(**
Creates a customized copy of the Parameter.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

val as_float_array : ?copy:bool -> ?force_all_finite:[`Bool of bool | `Allow_nan] -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> unit -> Py.Object.t
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

val assert_all_finite : ?allow_nan:bool -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> unit -> Py.Object.t
(**
Throw a ValueError if X contains NaN or infinity.

Parameters
----------
X : array or sparse matrix

allow_nan : bool
*)

val check_X_y : ?accept_sparse:[`String of string | `Bool of bool | `StringList of string list] -> ?accept_large_sparse:bool -> ?dtype:[`String of string | `Dtype of Py.Object.t | `TypeList of Py.Object.t | `None] -> ?order:[`F | `C | `None] -> ?copy:bool -> ?force_all_finite:[`Bool of bool | `Allow_nan] -> ?ensure_2d:bool -> ?allow_nd:bool -> ?multi_output:bool -> ?ensure_min_samples:int -> ?ensure_min_features:int -> ?y_numeric:bool -> ?warn_on_dtype:[`Bool of bool | `None] -> ?estimator:[`String of string | `Estimator of Py.Object.t] -> x:[`Ndarray of Ndarray.t | `ArrayLike of Py.Object.t | `SparseMatrix of Csr_matrix.t] -> y:[`Ndarray of Ndarray.t | `ArrayLike of Py.Object.t | `SparseMatrix of Csr_matrix.t] -> unit -> (Py.Object.t * Py.Object.t)
(**
Input validation for standard estimators.

Checks X and y for consistent length, enforces X to be 2D and y 1D. By
default, X is checked to be non-empty and containing only finite values.
Standard input checks are also applied to y, such as checking that y
does not have np.nan or np.inf targets. For multi-label y, set
multi_output=True to allow 2D and sparse y. If the dtype of X is
object, attempt converting to float, raising on failure.

Parameters
----------
X : nd-array, list or sparse matrix
    Input data.

y : nd-array, list or sparse matrix
    Labels.

accept_sparse : string, boolean or list of string (default=False)
    String[s] representing allowed sparse matrix formats, such as 'csc',
    'csr', etc. If the input is sparse but not in the allowed format,
    it will be converted to the first listed format. True allows the input
    to be any format. False means that a sparse matrix input will
    raise an error.

accept_large_sparse : bool (default=True)
    If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
    accept_sparse, accept_large_sparse will cause it to be accepted only
    if its indices are stored with a 32-bit dtype.

    .. versionadded:: 0.20

dtype : string, type, list of types or None (default="numeric")
    Data type of result. If None, the dtype of the input is preserved.
    If "numeric", dtype is preserved unless array.dtype is object.
    If dtype is a list of types, conversion on the first type is only
    performed if the dtype of the input is not in the list.

order : 'F', 'C' or None (default=None)
    Whether an array will be forced to be fortran or c-style.

copy : boolean (default=False)
    Whether a forced copy will be triggered. If copy=False, a copy might
    be triggered by a conversion.

force_all_finite : boolean or 'allow-nan', (default=True)
    Whether to raise an error on np.inf and np.nan in X. This parameter
    does not influence whether y can have np.inf or np.nan values.
    The possibilities are:

    - True: Force all values of X to be finite.
    - False: accept both np.inf and np.nan in X.
    - 'allow-nan': accept only np.nan values in X. Values cannot be
      infinite.

    .. versionadded:: 0.20
       ``force_all_finite`` accepts the string ``'allow-nan'``.

ensure_2d : boolean (default=True)
    Whether to raise a value error if X is not 2D.

allow_nd : boolean (default=False)
    Whether to allow X.ndim > 2.

multi_output : boolean (default=False)
    Whether to allow 2D y (array or sparse matrix). If false, y will be
    validated as a vector. y cannot have np.nan or np.inf values if
    multi_output=True.

ensure_min_samples : int (default=1)
    Make sure that X has a minimum number of samples in its first
    axis (rows for a 2D array).

ensure_min_features : int (default=1)
    Make sure that the 2D array has some minimum number of features
    (columns). The default value of 1 rejects empty datasets.
    This check is only enforced when X has effectively 2 dimensions or
    is originally 1D and ``ensure_2d`` is True. Setting to 0 disables
    this check.

y_numeric : boolean (default=False)
    Whether to ensure that y has a numeric type. If dtype of y is object,
    it is converted to float64. Should only be used for regression
    algorithms.

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
X_converted : object
    The converted and validated X.

y_converted : object
    The converted and validated y.
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

val check_consistent_length : Py.Object.t list -> Py.Object.t
(**
Check that all arrays have consistent first dimensions.

Checks whether all objects in arrays have the same shape or length.

Parameters
----------
*arrays : list or tuple of input objects.
    Objects that will be checked for consistent length.
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

val check_memory : memory:[`String of string | `JoblibMemory of Py.Object.t | `None] -> unit -> Py.Object.t
(**
Check that ``memory`` is joblib.Memory-like.

joblib.Memory-like means that ``memory`` can be converted into a
joblib.Memory instance (typically a str denoting the ``location``)
or has the same interface (has a ``cache`` method).

Parameters
----------
memory : None, str or object with the joblib.Memory interface

Returns
-------
memory : object with the joblib.Memory interface

Raises
------
ValueError
    If ``memory`` is not joblib.Memory-like.
*)

val check_non_negative : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> whom:string -> unit -> Py.Object.t
(**
Check if there is any negative value in an array.

Parameters
----------
X : array-like or sparse matrix
    Input data.

whom : string
    Who passed X to this function.
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

val check_scalar : ?min_val:[`Float of float | `Int of int] -> ?max_val:[`Float of float | `Int of int] -> x:Py.Object.t -> name:string -> target_type:[`Dtype of Py.Object.t | `PyObject of Py.Object.t] -> unit -> Py.Object.t
(**
Validate scalar parameters type and value.

Parameters
----------
x : object
    The scalar parameter to validate.

name : str
    The name of the parameter to be printed in error messages.

target_type : type or tuple
    Acceptable data types for the parameter.

min_val : float or int, optional (default=None)
    The minimum valid value the parameter can take. If None (default) it
    is implied that the parameter does not have a lower bound.

max_val : float or int, optional (default=None)
    The maximum valid value the parameter can take. If None (default) it
    is implied that the parameter does not have an upper bound.

Raises
-------
TypeError
    If the parameter's type does not match the desired type.

ValueError
    If the parameter's value violates the given bounds.
*)

val check_symmetric : ?tol:Py.Object.t -> ?raise_warning:Py.Object.t -> ?raise_exception:Py.Object.t -> array:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> unit -> Py.Object.t
(**
Make sure that array is 2D, square and symmetric.

If the array is not symmetric, then a symmetrized version is returned.
Optionally, a warning or exception is raised if the matrix is not
symmetric.

Parameters
----------
array : nd-array or sparse matrix
    Input object to check / convert. Must be two-dimensional and square,
    otherwise a ValueError will be raised.
tol : float
    Absolute tolerance for equivalence of arrays. Default = 1E-10.
raise_warning : boolean (default=True)
    If True then raise a warning if conversion is required.
raise_exception : boolean (default=False)
    If True then raise an exception if array is not symmetric.

Returns
-------
array_sym : ndarray or sparse matrix
    Symmetrized version of the input array, i.e. the average of array
    and array.transpose(). If sparse, then duplicate entries are first
    summed and zeros are eliminated.
*)

val column_or_1d : ?warn:bool -> y:Ndarray.t -> unit -> Ndarray.t
(**
Ravel column or 1d numpy array, else raises an error

Parameters
----------
y : array-like

warn : boolean, default False
   To control display of warnings.

Returns
-------
y : array
*)

val has_fit_parameter : estimator:Py.Object.t -> parameter:string -> unit -> bool
(**
Checks whether the estimator's fit method supports the given parameter.

Parameters
----------
estimator : object
    An estimator to inspect.

parameter : str
    The searched parameter.

Returns
-------
is_parameter: bool
    Whether the parameter was found to be a named parameter of the
    estimator's fit method.

Examples
--------
>>> from sklearn.svm import SVC
>>> has_fit_parameter(SVC(), "sample_weight")
True
*)

val indexable : Py.Object.t list -> Py.Object.t
(**
Make arrays indexable for cross-validation.

Checks consistent length, passes through None, and ensures that everything
can be indexed by converting sparse matrices to csr and converting
non-interable objects to arrays.

Parameters
----------
*iterables : lists, dataframes, arrays, sparse matrices
    List of objects to ensure sliceability.
*)

val isclass : object_:Py.Object.t -> unit -> Py.Object.t
(**
Return true if the object is a class.

Class objects provide these attributes:
    __doc__         documentation string
    __module__      name of module in which this class was defined
*)

val signature : ?follow_wrapped:Py.Object.t -> obj:Py.Object.t -> unit -> Py.Object.t
(**
Get a signature object for the passed callable.
*)

val wraps : ?assigned:Py.Object.t -> ?updated:Py.Object.t -> wrapped:Py.Object.t -> unit -> Py.Object.t
(**
Decorator factory to apply update_wrapper() to a wrapper function

Returns a decorator that invokes update_wrapper() with the decorated
function as the wrapper argument and the arguments to wraps() as the
remaining arguments. Default arguments are as for update_wrapper().
This is a convenience function to simplify applying partial() to
update_wrapper().
*)


end

