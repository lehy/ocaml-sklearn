(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module SparseEfficiencyWarning : sig
type tag = [`SparseEfficiencyWarning]
type t = [`BaseException | `Object | `SparseEfficiencyWarning] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_exception : t -> [`BaseException] Obj.t
val with_traceback : tb:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Exception.with_traceback(tb) --
set self.__traceback__ to tb and return self.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module SparseWarning : sig
type tag = [`SparseWarning]
type t = [`BaseException | `Object | `SparseWarning] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_exception : t -> [`BaseException] Obj.t
val with_traceback : tb:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Exception.with_traceback(tb) --
set self.__traceback__ to tb and return self.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Bsr_matrix : sig
type tag = [`Bsr_matrix]
type t = [`ArrayLike | `Bsr_matrix | `IndexMixin | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_index : t -> [`IndexMixin] Obj.t
val create : ?shape:Py.Object.t -> ?dtype:Py.Object.t -> ?copy:Py.Object.t -> ?blocksize:Py.Object.t -> arg1:Py.Object.t -> unit -> t
(**
Block Sparse Row matrix

This can be instantiated in several ways:
    bsr_matrix(D, [blocksize=(R,C)])
        where D is a dense matrix or 2-D ndarray.

    bsr_matrix(S, [blocksize=(R,C)])
        with another sparse matrix S (equivalent to S.tobsr())

    bsr_matrix((M, N), [blocksize=(R,C), dtype])
        to construct an empty matrix with shape (M, N)
        dtype is optional, defaulting to dtype='d'.

    bsr_matrix((data, ij), [blocksize=(R,C), shape=(M, N)])
        where ``data`` and ``ij`` satisfy ``a[ij[0, k], ij[1, k]] = data[k]``

    bsr_matrix((data, indices, indptr), [shape=(M, N)])
        is the standard BSR representation where the block column
        indices for row i are stored in ``indices[indptr[i]:indptr[i+1]]``
        and their corresponding block values are stored in
        ``data[ indptr[i]: indptr[i+1] ]``.  If the shape parameter is not
        supplied, the matrix dimensions are inferred from the index arrays.

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
    Data array of the matrix
indices
    BSR format index array
indptr
    BSR format index pointer array
blocksize
    Block size of the matrix
has_sorted_indices
    Whether indices are sorted

Notes
-----
Sparse matrices can be used in arithmetic operations: they support
addition, subtraction, multiplication, division, and matrix power.

**Summary of BSR format**

The Block Compressed Row (BSR) format is very similar to the Compressed
Sparse Row (CSR) format.  BSR is appropriate for sparse matrices with dense
sub matrices like the last example below.  Block matrices often arise in
vector-valued finite element discretizations.  In such cases, BSR is
considerably more efficient than CSR and CSC for many sparse arithmetic
operations.

**Blocksize**

The blocksize (R,C) must evenly divide the shape of the matrix (M,N).
That is, R and C must satisfy the relationship ``M % R = 0`` and
``N % C = 0``.

If no blocksize is specified, a simple heuristic is applied to determine
an appropriate blocksize.

Examples
--------
>>> from scipy.sparse import bsr_matrix
>>> bsr_matrix((3, 4), dtype=np.int8).toarray()
array([[0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]], dtype=int8)

>>> row = np.array([0, 0, 1, 2, 2, 2])
>>> col = np.array([0, 2, 2, 0, 1, 2])
>>> data = np.array([1, 2, 3 ,4, 5, 6])
>>> bsr_matrix((data, (row, col)), shape=(3, 3)).toarray()
array([[1, 0, 2],
       [0, 0, 3],
       [4, 5, 6]])

>>> indptr = np.array([0, 2, 3, 6])
>>> indices = np.array([0, 2, 2, 0, 1, 2])
>>> data = np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)
>>> bsr_matrix((data,indices,indptr), shape=(6, 6)).toarray()
array([[1, 1, 0, 0, 2, 2],
       [1, 1, 0, 0, 2, 2],
       [0, 0, 0, 0, 3, 3],
       [0, 0, 0, 0, 3, 3],
       [4, 4, 5, 5, 6, 6],
       [4, 4, 5, 5, 6, 6]])
*)

val __getitem__ : key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val __iter__ : [> tag] Obj.t -> Py.Object.t
(**
None
*)

val __setitem__ : key:Py.Object.t -> val_:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val arcsin : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arcsin.

See numpy.arcsin for more information.
*)

val arcsinh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arcsinh.

See numpy.arcsinh for more information.
*)

val arctan : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arctan.

See numpy.arctan for more information.
*)

val arctanh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arctanh.

See numpy.arctanh for more information.
*)

val argmax : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return indices of maximum elements along an axis.

Implicit zero elements are also taken into account. If there are
several maximum values, the index of the first occurrence is returned.

Parameters
----------
axis : {-2, -1, 0, 1, None}, optional
    Axis along which the argmax is computed. If None (default), index
    of the maximum element in the flatten data is returned.
out : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except for
    the default value, as this argument is not used.

Returns
-------
ind : numpy.matrix or int
    Indices of maximum elements. If matrix, its size along `axis` is 1.
*)

val argmin : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return indices of minimum elements along an axis.

Implicit zero elements are also taken into account. If there are
several minimum values, the index of the first occurrence is returned.

Parameters
----------
axis : {-2, -1, 0, 1, None}, optional
    Axis along which the argmin is computed. If None (default), index
    of the minimum element in the flatten data is returned.
out : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except for
    the default value, as this argument is not used.

Returns
-------
 ind : numpy.matrix or int
    Indices of minimum elements. If matrix, its size along `axis` is 1.
*)

val asformat : ?copy:bool -> format:[`S of string | `None] -> [> tag] Obj.t -> Py.Object.t
(**
Return this matrix in the passed format.

Parameters
----------
format : {str, None}
    The desired matrix format ('csr', 'csc', 'lil', 'dok', 'array', ...)
    or None for no conversion.
copy : bool, optional
    If True, the result is guaranteed to not share data with self.

Returns
-------
A : This matrix in the passed format.
*)

val asfptype : [> tag] Obj.t -> Py.Object.t
(**
Upcast matrix to a floating point format (if necessary)
*)

val astype : ?casting:[`No | `Equiv | `Safe | `Same_kind | `Unsafe] -> ?copy:bool -> dtype:[`S of string | `Dtype of Np.Dtype.t] -> [> tag] Obj.t -> Py.Object.t
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

val ceil : [> tag] Obj.t -> Py.Object.t
(**
Element-wise ceil.

See numpy.ceil for more information.
*)

val check_format : ?full_check:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
check whether the matrix format is valid

*Parameters*:
    full_check:
        True  - rigorous check, O(N) operations : default
        False - basic check, O(1) operations
*)

val conj : ?copy:bool -> [> tag] Obj.t -> Py.Object.t
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

val conjugate : ?copy:bool -> [> tag] Obj.t -> Py.Object.t
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

val copy : [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of this matrix.

No data/indices will be shared between the returned value and current
matrix.
*)

val count_nonzero : [> tag] Obj.t -> Py.Object.t
(**
Number of non-zero entries, equivalent to

np.count_nonzero(a.toarray())

Unlike getnnz() and the nnz property, which return the number of stored
entries (the length of the data attribute), this method counts the
actual number of non-zero entries in data.
*)

val deg2rad : [> tag] Obj.t -> Py.Object.t
(**
Element-wise deg2rad.

See numpy.deg2rad for more information.
*)

val diagonal : ?k:int -> [> tag] Obj.t -> Py.Object.t
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

val dot : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
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

val eliminate_zeros : [> tag] Obj.t -> Py.Object.t
(**
Remove zero elements in-place.
*)

val expm1 : [> tag] Obj.t -> Py.Object.t
(**
Element-wise expm1.

See numpy.expm1 for more information.
*)

val floor : [> tag] Obj.t -> Py.Object.t
(**
Element-wise floor.

See numpy.floor for more information.
*)

val getH : [> tag] Obj.t -> Py.Object.t
(**
Return the Hermitian transpose of this matrix.

See Also
--------
numpy.matrix.getH : NumPy's implementation of `getH` for matrices
*)

val get_shape : [> tag] Obj.t -> Py.Object.t
(**
Get shape of a matrix.
*)

val getcol : j:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of column j of the matrix, as an (m x 1) sparse
matrix (column vector).
*)

val getformat : [> tag] Obj.t -> Py.Object.t
(**
Format of a matrix representation as a string.
*)

val getmaxprint : [> tag] Obj.t -> Py.Object.t
(**
Maximum number of elements to display when printed.
*)

val getnnz : ?axis:[`Zero | `One] -> [> tag] Obj.t -> Py.Object.t
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

val getrow : i:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of row i of the matrix, as a (1 x n) sparse
matrix (row vector).
*)

val log1p : [> tag] Obj.t -> Py.Object.t
(**
Element-wise log1p.

See numpy.log1p for more information.
*)

val max : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return the maximum of the matrix or maximum along an axis.
This takes all elements into account, not just the non-zero ones.

Parameters
----------
axis : {-2, -1, 0, 1, None} optional
    Axis along which the sum is computed. The default is to
    compute the maximum over all the matrix elements, returning
    a scalar (i.e. `axis` = `None`).

out : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except
    for the default value, as this argument is not used.

Returns
-------
amax : coo_matrix or scalar
    Maximum of `a`. If `axis` is None, the result is a scalar value.
    If `axis` is given, the result is a sparse.coo_matrix of dimension
    ``a.ndim - 1``.

See Also
--------
min : The minimum value of a sparse matrix along a given axis.
numpy.matrix.max : NumPy's implementation of 'max' for matrices
*)

val maximum : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Element-wise maximum between this and another matrix.
*)

val mean : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?dtype:Np.Dtype.t -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val min : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return the minimum of the matrix or maximum along an axis.
This takes all elements into account, not just the non-zero ones.

Parameters
----------
axis : {-2, -1, 0, 1, None} optional
    Axis along which the sum is computed. The default is to
    compute the minimum over all the matrix elements, returning
    a scalar (i.e. `axis` = `None`).

out : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except for
    the default value, as this argument is not used.

Returns
-------
amin : coo_matrix or scalar
    Minimum of `a`. If `axis` is None, the result is a scalar value.
    If `axis` is given, the result is a sparse.coo_matrix of dimension
    ``a.ndim - 1``.

See Also
--------
max : The maximum value of a sparse matrix along a given axis.
numpy.matrix.min : NumPy's implementation of 'min' for matrices
*)

val minimum : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Element-wise minimum between this and another matrix.
*)

val multiply : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Point-wise multiplication by another matrix, vector, or
scalar.
*)

val nonzero : [> tag] Obj.t -> Py.Object.t
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

val power : ?dtype:Py.Object.t -> n:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
This function performs element-wise power.

Parameters
----------
n : n is a scalar

dtype : If dtype is not specified, the current dtype will be preserved.
*)

val prune : [> tag] Obj.t -> Py.Object.t
(**
Remove empty space after all non-zero elements.
        
*)

val rad2deg : [> tag] Obj.t -> Py.Object.t
(**
Element-wise rad2deg.

See numpy.rad2deg for more information.
*)

val reshape : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> [> tag] Obj.t -> [`ArrayLike|`Object|`Spmatrix] Np.Obj.t
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

val resize : Py.Object.t list -> [> tag] Obj.t -> Py.Object.t
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

val rint : [> tag] Obj.t -> Py.Object.t
(**
Element-wise rint.

See numpy.rint for more information.
*)

val set_shape : shape:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
See `reshape`.
*)

val setdiag : ?k:int -> values:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
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

val sign : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sign.

See numpy.sign for more information.
*)

val sin : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sin.

See numpy.sin for more information.
*)

val sinh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sinh.

See numpy.sinh for more information.
*)

val sort_indices : [> tag] Obj.t -> Py.Object.t
(**
Sort the indices of this matrix *in place*
        
*)

val sorted_indices : [> tag] Obj.t -> Py.Object.t
(**
Return a copy of this matrix with sorted indices
        
*)

val sqrt : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sqrt.

See numpy.sqrt for more information.
*)

val sum : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?dtype:Np.Dtype.t -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val sum_duplicates : [> tag] Obj.t -> Py.Object.t
(**
Eliminate duplicate matrix entries by adding them together

The is an *in place* operation
*)

val tan : [> tag] Obj.t -> Py.Object.t
(**
Element-wise tan.

See numpy.tan for more information.
*)

val tanh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise tanh.

See numpy.tanh for more information.
*)

val toarray : ?order:[`F | `C] -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val tobsr : ?blocksize:Py.Object.t -> ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix into Block Sparse Row Format.

With copy=False, the data/indices may be shared between this
matrix and the resultant bsr_matrix.

If blocksize=(R, C) is provided, it will be used for determining
block size of the bsr_matrix.
*)

val tocoo : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to COOrdinate format.

When copy=False the data array will be shared between
this matrix and the resultant coo_matrix.
*)

val tocsc : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Compressed Sparse Column format.

With copy=False, the data/indices may be shared between this matrix and
the resultant csc_matrix.
*)

val tocsr : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Compressed Sparse Row format.

With copy=False, the data/indices may be shared between this matrix and
the resultant csr_matrix.
*)

val todense : ?order:[`F | `C] -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val todia : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to sparse DIAgonal format.

With copy=False, the data/indices may be shared between this matrix and
the resultant dia_matrix.
*)

val todok : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Dictionary Of Keys format.

With copy=False, the data/indices may be shared between this matrix and
the resultant dok_matrix.
*)

val tolil : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to List of Lists format.

With copy=False, the data/indices may be shared between this matrix and
the resultant lil_matrix.
*)

val transpose : ?axes:Py.Object.t -> ?copy:bool -> [> tag] Obj.t -> Py.Object.t
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

val trunc : [> tag] Obj.t -> Py.Object.t
(**
Element-wise trunc.

See numpy.trunc for more information.
*)


(** Attribute dtype: get value or raise Not_found if None.*)
val dtype : t -> Np.Dtype.t

(** Attribute dtype: get value as an option. *)
val dtype_opt : t -> (Np.Dtype.t) option


(** Attribute shape: get value or raise Not_found if None.*)
val shape : t -> Py.Object.t

(** Attribute shape: get value as an option. *)
val shape_opt : t -> (Py.Object.t) option


(** Attribute ndim: get value or raise Not_found if None.*)
val ndim : t -> int

(** Attribute ndim: get value as an option. *)
val ndim_opt : t -> (int) option


(** Attribute nnz: get value or raise Not_found if None.*)
val nnz : t -> Py.Object.t

(** Attribute nnz: get value as an option. *)
val nnz_opt : t -> (Py.Object.t) option


(** Attribute data: get value or raise Not_found if None.*)
val data : t -> Py.Object.t

(** Attribute data: get value as an option. *)
val data_opt : t -> (Py.Object.t) option


(** Attribute indices: get value or raise Not_found if None.*)
val indices : t -> Py.Object.t

(** Attribute indices: get value as an option. *)
val indices_opt : t -> (Py.Object.t) option


(** Attribute indptr: get value or raise Not_found if None.*)
val indptr : t -> Py.Object.t

(** Attribute indptr: get value as an option. *)
val indptr_opt : t -> (Py.Object.t) option


(** Attribute blocksize: get value or raise Not_found if None.*)
val blocksize : t -> Py.Object.t

(** Attribute blocksize: get value as an option. *)
val blocksize_opt : t -> (Py.Object.t) option


(** Attribute has_sorted_indices: get value or raise Not_found if None.*)
val has_sorted_indices : t -> Py.Object.t

(** Attribute has_sorted_indices: get value as an option. *)
val has_sorted_indices_opt : t -> (Py.Object.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Coo_matrix : sig
type tag = [`Coo_matrix]
type t = [`ArrayLike | `Coo_matrix | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?shape:Py.Object.t -> ?dtype:Py.Object.t -> ?copy:Py.Object.t -> arg1:Py.Object.t -> unit -> t
(**
A sparse matrix in COOrdinate format.

Also known as the 'ijv' or 'triplet' format.

This can be instantiated in several ways:
    coo_matrix(D)
        with a dense matrix D

    coo_matrix(S)
        with another sparse matrix S (equivalent to S.tocoo())

    coo_matrix((M, N), [dtype])
        to construct an empty matrix with shape (M, N)
        dtype is optional, defaulting to dtype='d'.

    coo_matrix((data, (i, j)), [shape=(M, N)])
        to construct from three arrays:
            1. data[:]   the entries of the matrix, in any order
            2. i[:]      the row indices of the matrix entries
            3. j[:]      the column indices of the matrix entries

        Where ``A[i[k], j[k]] = data[k]``.  When shape is not
        specified, it is inferred from the index arrays

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
    COO format data array of the matrix
row
    COO format row index array of the matrix
col
    COO format column index array of the matrix

Notes
-----

Sparse matrices can be used in arithmetic operations: they support
addition, subtraction, multiplication, division, and matrix power.

Advantages of the COO format
    - facilitates fast conversion among sparse formats
    - permits duplicate entries (see example)
    - very fast conversion to and from CSR/CSC formats

Disadvantages of the COO format
    - does not directly support:
        + arithmetic operations
        + slicing

Intended Usage
    - COO is a fast format for constructing sparse matrices
    - Once a matrix has been constructed, convert to CSR or
      CSC format for fast arithmetic and matrix vector operations
    - By default when converting to CSR or CSC format, duplicate (i,j)
      entries will be summed together.  This facilitates efficient
      construction of finite element matrices and the like. (see example)

Examples
--------

>>> # Constructing an empty matrix
>>> from scipy.sparse import coo_matrix
>>> coo_matrix((3, 4), dtype=np.int8).toarray()
array([[0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]], dtype=int8)

>>> # Constructing a matrix using ijv format
>>> row  = np.array([0, 3, 1, 0])
>>> col  = np.array([0, 3, 1, 2])
>>> data = np.array([4, 5, 7, 9])
>>> coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
array([[4, 0, 9, 0],
       [0, 7, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 5]])

>>> # Constructing a matrix with duplicate indices
>>> row  = np.array([0, 0, 1, 3, 1, 0, 0])
>>> col  = np.array([0, 2, 1, 3, 1, 0, 0])
>>> data = np.array([1, 1, 1, 1, 1, 1, 1])
>>> coo = coo_matrix((data, (row, col)), shape=(4, 4))
>>> # Duplicate indices are maintained until implicitly or explicitly summed
>>> np.max(coo.data)
1
>>> coo.toarray()
array([[3, 0, 1, 0],
       [0, 2, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 1]])
*)

val __iter__ : [> tag] Obj.t -> Py.Object.t
(**
None
*)

val arcsin : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arcsin.

See numpy.arcsin for more information.
*)

val arcsinh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arcsinh.

See numpy.arcsinh for more information.
*)

val arctan : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arctan.

See numpy.arctan for more information.
*)

val arctanh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arctanh.

See numpy.arctanh for more information.
*)

val argmax : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return indices of maximum elements along an axis.

Implicit zero elements are also taken into account. If there are
several maximum values, the index of the first occurrence is returned.

Parameters
----------
axis : {-2, -1, 0, 1, None}, optional
    Axis along which the argmax is computed. If None (default), index
    of the maximum element in the flatten data is returned.
out : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except for
    the default value, as this argument is not used.

Returns
-------
ind : numpy.matrix or int
    Indices of maximum elements. If matrix, its size along `axis` is 1.
*)

val argmin : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return indices of minimum elements along an axis.

Implicit zero elements are also taken into account. If there are
several minimum values, the index of the first occurrence is returned.

Parameters
----------
axis : {-2, -1, 0, 1, None}, optional
    Axis along which the argmin is computed. If None (default), index
    of the minimum element in the flatten data is returned.
out : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except for
    the default value, as this argument is not used.

Returns
-------
 ind : numpy.matrix or int
    Indices of minimum elements. If matrix, its size along `axis` is 1.
*)

val asformat : ?copy:bool -> format:[`S of string | `None] -> [> tag] Obj.t -> Py.Object.t
(**
Return this matrix in the passed format.

Parameters
----------
format : {str, None}
    The desired matrix format ('csr', 'csc', 'lil', 'dok', 'array', ...)
    or None for no conversion.
copy : bool, optional
    If True, the result is guaranteed to not share data with self.

Returns
-------
A : This matrix in the passed format.
*)

val asfptype : [> tag] Obj.t -> Py.Object.t
(**
Upcast matrix to a floating point format (if necessary)
*)

val astype : ?casting:[`No | `Equiv | `Safe | `Same_kind | `Unsafe] -> ?copy:bool -> dtype:[`S of string | `Dtype of Np.Dtype.t] -> [> tag] Obj.t -> Py.Object.t
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

val ceil : [> tag] Obj.t -> Py.Object.t
(**
Element-wise ceil.

See numpy.ceil for more information.
*)

val conj : ?copy:bool -> [> tag] Obj.t -> Py.Object.t
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

val conjugate : ?copy:bool -> [> tag] Obj.t -> Py.Object.t
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

val copy : [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of this matrix.

No data/indices will be shared between the returned value and current
matrix.
*)

val count_nonzero : [> tag] Obj.t -> Py.Object.t
(**
Number of non-zero entries, equivalent to

np.count_nonzero(a.toarray())

Unlike getnnz() and the nnz property, which return the number of stored
entries (the length of the data attribute), this method counts the
actual number of non-zero entries in data.
*)

val deg2rad : [> tag] Obj.t -> Py.Object.t
(**
Element-wise deg2rad.

See numpy.deg2rad for more information.
*)

val diagonal : ?k:int -> [> tag] Obj.t -> Py.Object.t
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

val dot : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
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

val eliminate_zeros : [> tag] Obj.t -> Py.Object.t
(**
Remove zero entries from the matrix

This is an *in place* operation
*)

val expm1 : [> tag] Obj.t -> Py.Object.t
(**
Element-wise expm1.

See numpy.expm1 for more information.
*)

val floor : [> tag] Obj.t -> Py.Object.t
(**
Element-wise floor.

See numpy.floor for more information.
*)

val getH : [> tag] Obj.t -> Py.Object.t
(**
Return the Hermitian transpose of this matrix.

See Also
--------
numpy.matrix.getH : NumPy's implementation of `getH` for matrices
*)

val get_shape : [> tag] Obj.t -> Py.Object.t
(**
Get shape of a matrix.
*)

val getcol : j:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of column j of the matrix, as an (m x 1) sparse
matrix (column vector).
*)

val getformat : [> tag] Obj.t -> Py.Object.t
(**
Format of a matrix representation as a string.
*)

val getmaxprint : [> tag] Obj.t -> Py.Object.t
(**
Maximum number of elements to display when printed.
*)

val getnnz : ?axis:[`Zero | `One] -> [> tag] Obj.t -> Py.Object.t
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

val getrow : i:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of row i of the matrix, as a (1 x n) sparse
matrix (row vector).
*)

val log1p : [> tag] Obj.t -> Py.Object.t
(**
Element-wise log1p.

See numpy.log1p for more information.
*)

val max : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return the maximum of the matrix or maximum along an axis.
This takes all elements into account, not just the non-zero ones.

Parameters
----------
axis : {-2, -1, 0, 1, None} optional
    Axis along which the sum is computed. The default is to
    compute the maximum over all the matrix elements, returning
    a scalar (i.e. `axis` = `None`).

out : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except
    for the default value, as this argument is not used.

Returns
-------
amax : coo_matrix or scalar
    Maximum of `a`. If `axis` is None, the result is a scalar value.
    If `axis` is given, the result is a sparse.coo_matrix of dimension
    ``a.ndim - 1``.

See Also
--------
min : The minimum value of a sparse matrix along a given axis.
numpy.matrix.max : NumPy's implementation of 'max' for matrices
*)

val maximum : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Element-wise maximum between this and another matrix.
*)

val mean : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?dtype:Np.Dtype.t -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val min : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return the minimum of the matrix or maximum along an axis.
This takes all elements into account, not just the non-zero ones.

Parameters
----------
axis : {-2, -1, 0, 1, None} optional
    Axis along which the sum is computed. The default is to
    compute the minimum over all the matrix elements, returning
    a scalar (i.e. `axis` = `None`).

out : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except for
    the default value, as this argument is not used.

Returns
-------
amin : coo_matrix or scalar
    Minimum of `a`. If `axis` is None, the result is a scalar value.
    If `axis` is given, the result is a sparse.coo_matrix of dimension
    ``a.ndim - 1``.

See Also
--------
max : The maximum value of a sparse matrix along a given axis.
numpy.matrix.min : NumPy's implementation of 'min' for matrices
*)

val minimum : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Element-wise minimum between this and another matrix.
*)

val multiply : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Point-wise multiplication by another matrix
        
*)

val nonzero : [> tag] Obj.t -> Py.Object.t
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

val power : ?dtype:Py.Object.t -> n:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
This function performs element-wise power.

Parameters
----------
n : n is a scalar

dtype : If dtype is not specified, the current dtype will be preserved.
*)

val rad2deg : [> tag] Obj.t -> Py.Object.t
(**
Element-wise rad2deg.

See numpy.rad2deg for more information.
*)

val reshape : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> [> tag] Obj.t -> [`ArrayLike|`Object|`Spmatrix] Np.Obj.t
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

val resize : Py.Object.t list -> [> tag] Obj.t -> Py.Object.t
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

val rint : [> tag] Obj.t -> Py.Object.t
(**
Element-wise rint.

See numpy.rint for more information.
*)

val set_shape : shape:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
See `reshape`.
*)

val setdiag : ?k:int -> values:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
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

val sign : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sign.

See numpy.sign for more information.
*)

val sin : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sin.

See numpy.sin for more information.
*)

val sinh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sinh.

See numpy.sinh for more information.
*)

val sqrt : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sqrt.

See numpy.sqrt for more information.
*)

val sum : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?dtype:Np.Dtype.t -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val sum_duplicates : [> tag] Obj.t -> Py.Object.t
(**
Eliminate duplicate matrix entries by adding them together

This is an *in place* operation
*)

val tan : [> tag] Obj.t -> Py.Object.t
(**
Element-wise tan.

See numpy.tan for more information.
*)

val tanh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise tanh.

See numpy.tanh for more information.
*)

val toarray : ?order:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
See the docstring for `spmatrix.toarray`.
*)

val tobsr : ?blocksize:Py.Object.t -> ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Block Sparse Row format.

With copy=False, the data/indices may be shared between this matrix and
the resultant bsr_matrix.

When blocksize=(R, C) is provided, it will be used for construction of
the bsr_matrix.
*)

val tocoo : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to COOrdinate format.

With copy=False, the data/indices may be shared between this matrix and
the resultant coo_matrix.
*)

val tocsc : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Compressed Sparse Column format

Duplicate entries will be summed together.

Examples
--------
>>> from numpy import array
>>> from scipy.sparse import coo_matrix
>>> row  = array([0, 0, 1, 3, 1, 0, 0])
>>> col  = array([0, 2, 1, 3, 1, 0, 0])
>>> data = array([1, 1, 1, 1, 1, 1, 1])
>>> A = coo_matrix((data, (row, col)), shape=(4, 4)).tocsc()
>>> A.toarray()
array([[3, 0, 1, 0],
       [0, 2, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 1]])
*)

val tocsr : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Compressed Sparse Row format

Duplicate entries will be summed together.

Examples
--------
>>> from numpy import array
>>> from scipy.sparse import coo_matrix
>>> row  = array([0, 0, 1, 3, 1, 0, 0])
>>> col  = array([0, 2, 1, 3, 1, 0, 0])
>>> data = array([1, 1, 1, 1, 1, 1, 1])
>>> A = coo_matrix((data, (row, col)), shape=(4, 4)).tocsr()
>>> A.toarray()
array([[3, 0, 1, 0],
       [0, 2, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 1]])
*)

val todense : ?order:[`F | `C] -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val todia : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to sparse DIAgonal format.

With copy=False, the data/indices may be shared between this matrix and
the resultant dia_matrix.
*)

val todok : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Dictionary Of Keys format.

With copy=False, the data/indices may be shared between this matrix and
the resultant dok_matrix.
*)

val tolil : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to List of Lists format.

With copy=False, the data/indices may be shared between this matrix and
the resultant lil_matrix.
*)

val transpose : ?axes:Py.Object.t -> ?copy:bool -> [> tag] Obj.t -> Py.Object.t
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

val trunc : [> tag] Obj.t -> Py.Object.t
(**
Element-wise trunc.

See numpy.trunc for more information.
*)


(** Attribute dtype: get value or raise Not_found if None.*)
val dtype : t -> Np.Dtype.t

(** Attribute dtype: get value as an option. *)
val dtype_opt : t -> (Np.Dtype.t) option


(** Attribute shape: get value or raise Not_found if None.*)
val shape : t -> Py.Object.t

(** Attribute shape: get value as an option. *)
val shape_opt : t -> (Py.Object.t) option


(** Attribute ndim: get value or raise Not_found if None.*)
val ndim : t -> int

(** Attribute ndim: get value as an option. *)
val ndim_opt : t -> (int) option


(** Attribute nnz: get value or raise Not_found if None.*)
val nnz : t -> Py.Object.t

(** Attribute nnz: get value as an option. *)
val nnz_opt : t -> (Py.Object.t) option


(** Attribute data: get value or raise Not_found if None.*)
val data : t -> Py.Object.t

(** Attribute data: get value as an option. *)
val data_opt : t -> (Py.Object.t) option


(** Attribute row: get value or raise Not_found if None.*)
val row : t -> Py.Object.t

(** Attribute row: get value as an option. *)
val row_opt : t -> (Py.Object.t) option


(** Attribute col: get value or raise Not_found if None.*)
val col : t -> Py.Object.t

(** Attribute col: get value as an option. *)
val col_opt : t -> (Py.Object.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Csc_matrix : sig
type tag = [`Csc_matrix]
type t = [`ArrayLike | `Csc_matrix | `IndexMixin | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_index : t -> [`IndexMixin] Obj.t
val create : ?shape:Py.Object.t -> ?dtype:Py.Object.t -> ?copy:Py.Object.t -> arg1:Py.Object.t -> unit -> t
(**
Compressed Sparse Column matrix

This can be instantiated in several ways:

    csc_matrix(D)
        with a dense matrix or rank-2 ndarray D

    csc_matrix(S)
        with another sparse matrix S (equivalent to S.tocsc())

    csc_matrix((M, N), [dtype])
        to construct an empty matrix with shape (M, N)
        dtype is optional, defaulting to dtype='d'.

    csc_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
        where ``data``, ``row_ind`` and ``col_ind`` satisfy the
        relationship ``a[row_ind[k], col_ind[k]] = data[k]``.

    csc_matrix((data, indices, indptr), [shape=(M, N)])
        is the standard CSC representation where the row indices for
        column i are stored in ``indices[indptr[i]:indptr[i+1]]``
        and their corresponding values are stored in
        ``data[indptr[i]:indptr[i+1]]``.  If the shape parameter is
        not supplied, the matrix dimensions are inferred from
        the index arrays.

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
    Data array of the matrix
indices
    CSC format index array
indptr
    CSC format index pointer array
has_sorted_indices
    Whether indices are sorted

Notes
-----

Sparse matrices can be used in arithmetic operations: they support
addition, subtraction, multiplication, division, and matrix power.

Advantages of the CSC format
    - efficient arithmetic operations CSC + CSC, CSC * CSC, etc.
    - efficient column slicing
    - fast matrix vector products (CSR, BSR may be faster)

Disadvantages of the CSC format
  - slow row slicing operations (consider CSR)
  - changes to the sparsity structure are expensive (consider LIL or DOK)


Examples
--------

>>> import numpy as np
>>> from scipy.sparse import csc_matrix
>>> csc_matrix((3, 4), dtype=np.int8).toarray()
array([[0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]], dtype=int8)

>>> row = np.array([0, 2, 2, 0, 1, 2])
>>> col = np.array([0, 0, 1, 2, 2, 2])
>>> data = np.array([1, 2, 3, 4, 5, 6])
>>> csc_matrix((data, (row, col)), shape=(3, 3)).toarray()
array([[1, 0, 4],
       [0, 0, 5],
       [2, 3, 6]])

>>> indptr = np.array([0, 2, 3, 6])
>>> indices = np.array([0, 2, 2, 0, 1, 2])
>>> data = np.array([1, 2, 3, 4, 5, 6])
>>> csc_matrix((data, indices, indptr), shape=(3, 3)).toarray()
array([[1, 0, 4],
       [0, 0, 5],
       [2, 3, 6]])
*)

val __getitem__ : key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val __iter__ : [> tag] Obj.t -> Py.Object.t
(**
None
*)

val __setitem__ : key:Py.Object.t -> x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val arcsin : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arcsin.

See numpy.arcsin for more information.
*)

val arcsinh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arcsinh.

See numpy.arcsinh for more information.
*)

val arctan : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arctan.

See numpy.arctan for more information.
*)

val arctanh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arctanh.

See numpy.arctanh for more information.
*)

val argmax : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return indices of maximum elements along an axis.

Implicit zero elements are also taken into account. If there are
several maximum values, the index of the first occurrence is returned.

Parameters
----------
axis : {-2, -1, 0, 1, None}, optional
    Axis along which the argmax is computed. If None (default), index
    of the maximum element in the flatten data is returned.
out : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except for
    the default value, as this argument is not used.

Returns
-------
ind : numpy.matrix or int
    Indices of maximum elements. If matrix, its size along `axis` is 1.
*)

val argmin : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return indices of minimum elements along an axis.

Implicit zero elements are also taken into account. If there are
several minimum values, the index of the first occurrence is returned.

Parameters
----------
axis : {-2, -1, 0, 1, None}, optional
    Axis along which the argmin is computed. If None (default), index
    of the minimum element in the flatten data is returned.
out : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except for
    the default value, as this argument is not used.

Returns
-------
 ind : numpy.matrix or int
    Indices of minimum elements. If matrix, its size along `axis` is 1.
*)

val asformat : ?copy:bool -> format:[`S of string | `None] -> [> tag] Obj.t -> Py.Object.t
(**
Return this matrix in the passed format.

Parameters
----------
format : {str, None}
    The desired matrix format ('csr', 'csc', 'lil', 'dok', 'array', ...)
    or None for no conversion.
copy : bool, optional
    If True, the result is guaranteed to not share data with self.

Returns
-------
A : This matrix in the passed format.
*)

val asfptype : [> tag] Obj.t -> Py.Object.t
(**
Upcast matrix to a floating point format (if necessary)
*)

val astype : ?casting:[`No | `Equiv | `Safe | `Same_kind | `Unsafe] -> ?copy:bool -> dtype:[`S of string | `Dtype of Np.Dtype.t] -> [> tag] Obj.t -> Py.Object.t
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

val ceil : [> tag] Obj.t -> Py.Object.t
(**
Element-wise ceil.

See numpy.ceil for more information.
*)

val check_format : ?full_check:bool -> [> tag] Obj.t -> Py.Object.t
(**
check whether the matrix format is valid

Parameters
----------
full_check : bool, optional
    If `True`, rigorous check, O(N) operations. Otherwise
    basic check, O(1) operations (default True).
*)

val conj : ?copy:bool -> [> tag] Obj.t -> Py.Object.t
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

val conjugate : ?copy:bool -> [> tag] Obj.t -> Py.Object.t
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

val copy : [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of this matrix.

No data/indices will be shared between the returned value and current
matrix.
*)

val count_nonzero : [> tag] Obj.t -> Py.Object.t
(**
Number of non-zero entries, equivalent to

np.count_nonzero(a.toarray())

Unlike getnnz() and the nnz property, which return the number of stored
entries (the length of the data attribute), this method counts the
actual number of non-zero entries in data.
*)

val deg2rad : [> tag] Obj.t -> Py.Object.t
(**
Element-wise deg2rad.

See numpy.deg2rad for more information.
*)

val diagonal : ?k:int -> [> tag] Obj.t -> Py.Object.t
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

val dot : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
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

val eliminate_zeros : [> tag] Obj.t -> Py.Object.t
(**
Remove zero entries from the matrix

This is an *in place* operation
*)

val expm1 : [> tag] Obj.t -> Py.Object.t
(**
Element-wise expm1.

See numpy.expm1 for more information.
*)

val floor : [> tag] Obj.t -> Py.Object.t
(**
Element-wise floor.

See numpy.floor for more information.
*)

val getH : [> tag] Obj.t -> Py.Object.t
(**
Return the Hermitian transpose of this matrix.

See Also
--------
numpy.matrix.getH : NumPy's implementation of `getH` for matrices
*)

val get_shape : [> tag] Obj.t -> Py.Object.t
(**
Get shape of a matrix.
*)

val getcol : i:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of column i of the matrix, as a (m x 1)
CSC matrix (column vector).
*)

val getformat : [> tag] Obj.t -> Py.Object.t
(**
Format of a matrix representation as a string.
*)

val getmaxprint : [> tag] Obj.t -> Py.Object.t
(**
Maximum number of elements to display when printed.
*)

val getnnz : ?axis:[`Zero | `One] -> [> tag] Obj.t -> Py.Object.t
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

val getrow : i:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of row i of the matrix, as a (1 x n)
CSR matrix (row vector).
*)

val log1p : [> tag] Obj.t -> Py.Object.t
(**
Element-wise log1p.

See numpy.log1p for more information.
*)

val max : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return the maximum of the matrix or maximum along an axis.
This takes all elements into account, not just the non-zero ones.

Parameters
----------
axis : {-2, -1, 0, 1, None} optional
    Axis along which the sum is computed. The default is to
    compute the maximum over all the matrix elements, returning
    a scalar (i.e. `axis` = `None`).

out : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except
    for the default value, as this argument is not used.

Returns
-------
amax : coo_matrix or scalar
    Maximum of `a`. If `axis` is None, the result is a scalar value.
    If `axis` is given, the result is a sparse.coo_matrix of dimension
    ``a.ndim - 1``.

See Also
--------
min : The minimum value of a sparse matrix along a given axis.
numpy.matrix.max : NumPy's implementation of 'max' for matrices
*)

val maximum : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Element-wise maximum between this and another matrix.
*)

val mean : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?dtype:Np.Dtype.t -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val min : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return the minimum of the matrix or maximum along an axis.
This takes all elements into account, not just the non-zero ones.

Parameters
----------
axis : {-2, -1, 0, 1, None} optional
    Axis along which the sum is computed. The default is to
    compute the minimum over all the matrix elements, returning
    a scalar (i.e. `axis` = `None`).

out : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except for
    the default value, as this argument is not used.

Returns
-------
amin : coo_matrix or scalar
    Minimum of `a`. If `axis` is None, the result is a scalar value.
    If `axis` is given, the result is a sparse.coo_matrix of dimension
    ``a.ndim - 1``.

See Also
--------
max : The maximum value of a sparse matrix along a given axis.
numpy.matrix.min : NumPy's implementation of 'min' for matrices
*)

val minimum : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Element-wise minimum between this and another matrix.
*)

val multiply : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Point-wise multiplication by another matrix, vector, or
scalar.
*)

val nonzero : [> tag] Obj.t -> Py.Object.t
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

val power : ?dtype:Py.Object.t -> n:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
This function performs element-wise power.

Parameters
----------
n : n is a scalar

dtype : If dtype is not specified, the current dtype will be preserved.
*)

val prune : [> tag] Obj.t -> Py.Object.t
(**
Remove empty space after all non-zero elements.
        
*)

val rad2deg : [> tag] Obj.t -> Py.Object.t
(**
Element-wise rad2deg.

See numpy.rad2deg for more information.
*)

val reshape : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> [> tag] Obj.t -> [`ArrayLike|`Object|`Spmatrix] Np.Obj.t
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

val resize : Py.Object.t list -> [> tag] Obj.t -> Py.Object.t
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

val rint : [> tag] Obj.t -> Py.Object.t
(**
Element-wise rint.

See numpy.rint for more information.
*)

val set_shape : shape:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
See `reshape`.
*)

val setdiag : ?k:int -> values:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
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

val sign : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sign.

See numpy.sign for more information.
*)

val sin : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sin.

See numpy.sin for more information.
*)

val sinh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sinh.

See numpy.sinh for more information.
*)

val sort_indices : [> tag] Obj.t -> Py.Object.t
(**
Sort the indices of this matrix *in place*
        
*)

val sorted_indices : [> tag] Obj.t -> Py.Object.t
(**
Return a copy of this matrix with sorted indices
        
*)

val sqrt : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sqrt.

See numpy.sqrt for more information.
*)

val sum : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?dtype:Np.Dtype.t -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val sum_duplicates : [> tag] Obj.t -> Py.Object.t
(**
Eliminate duplicate matrix entries by adding them together

The is an *in place* operation
*)

val tan : [> tag] Obj.t -> Py.Object.t
(**
Element-wise tan.

See numpy.tan for more information.
*)

val tanh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise tanh.

See numpy.tanh for more information.
*)

val toarray : ?order:[`F | `C] -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val tobsr : ?blocksize:Py.Object.t -> ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Block Sparse Row format.

With copy=False, the data/indices may be shared between this matrix and
the resultant bsr_matrix.

When blocksize=(R, C) is provided, it will be used for construction of
the bsr_matrix.
*)

val tocoo : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to COOrdinate format.

With copy=False, the data/indices may be shared between this matrix and
the resultant coo_matrix.
*)

val tocsc : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Compressed Sparse Column format.

With copy=False, the data/indices may be shared between this matrix and
the resultant csc_matrix.
*)

val tocsr : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Compressed Sparse Row format.

With copy=False, the data/indices may be shared between this matrix and
the resultant csr_matrix.
*)

val todense : ?order:[`F | `C] -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val todia : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to sparse DIAgonal format.

With copy=False, the data/indices may be shared between this matrix and
the resultant dia_matrix.
*)

val todok : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Dictionary Of Keys format.

With copy=False, the data/indices may be shared between this matrix and
the resultant dok_matrix.
*)

val tolil : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to List of Lists format.

With copy=False, the data/indices may be shared between this matrix and
the resultant lil_matrix.
*)

val transpose : ?axes:Py.Object.t -> ?copy:bool -> [> tag] Obj.t -> Py.Object.t
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

val trunc : [> tag] Obj.t -> Py.Object.t
(**
Element-wise trunc.

See numpy.trunc for more information.
*)


(** Attribute dtype: get value or raise Not_found if None.*)
val dtype : t -> Np.Dtype.t

(** Attribute dtype: get value as an option. *)
val dtype_opt : t -> (Np.Dtype.t) option


(** Attribute shape: get value or raise Not_found if None.*)
val shape : t -> Py.Object.t

(** Attribute shape: get value as an option. *)
val shape_opt : t -> (Py.Object.t) option


(** Attribute ndim: get value or raise Not_found if None.*)
val ndim : t -> int

(** Attribute ndim: get value as an option. *)
val ndim_opt : t -> (int) option


(** Attribute nnz: get value or raise Not_found if None.*)
val nnz : t -> Py.Object.t

(** Attribute nnz: get value as an option. *)
val nnz_opt : t -> (Py.Object.t) option


(** Attribute data: get value or raise Not_found if None.*)
val data : t -> Py.Object.t

(** Attribute data: get value as an option. *)
val data_opt : t -> (Py.Object.t) option


(** Attribute indices: get value or raise Not_found if None.*)
val indices : t -> Py.Object.t

(** Attribute indices: get value as an option. *)
val indices_opt : t -> (Py.Object.t) option


(** Attribute indptr: get value or raise Not_found if None.*)
val indptr : t -> Py.Object.t

(** Attribute indptr: get value as an option. *)
val indptr_opt : t -> (Py.Object.t) option


(** Attribute has_sorted_indices: get value or raise Not_found if None.*)
val has_sorted_indices : t -> Py.Object.t

(** Attribute has_sorted_indices: get value as an option. *)
val has_sorted_indices_opt : t -> (Py.Object.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Csr_matrix : sig
type tag = [`Csr_matrix]
type t = [`ArrayLike | `Csr_matrix | `IndexMixin | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_index : t -> [`IndexMixin] Obj.t
val create : ?shape:Py.Object.t -> ?dtype:Py.Object.t -> ?copy:Py.Object.t -> arg1:Py.Object.t -> unit -> t
(**
Compressed Sparse Row matrix

This can be instantiated in several ways:
    csr_matrix(D)
        with a dense matrix or rank-2 ndarray D

    csr_matrix(S)
        with another sparse matrix S (equivalent to S.tocsr())

    csr_matrix((M, N), [dtype])
        to construct an empty matrix with shape (M, N)
        dtype is optional, defaulting to dtype='d'.

    csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
        where ``data``, ``row_ind`` and ``col_ind`` satisfy the
        relationship ``a[row_ind[k], col_ind[k]] = data[k]``.

    csr_matrix((data, indices, indptr), [shape=(M, N)])
        is the standard CSR representation where the column indices for
        row i are stored in ``indices[indptr[i]:indptr[i+1]]`` and their
        corresponding values are stored in ``data[indptr[i]:indptr[i+1]]``.
        If the shape parameter is not supplied, the matrix dimensions
        are inferred from the index arrays.

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
    CSR format data array of the matrix
indices
    CSR format index array of the matrix
indptr
    CSR format index pointer array of the matrix
has_sorted_indices
    Whether indices are sorted

Notes
-----

Sparse matrices can be used in arithmetic operations: they support
addition, subtraction, multiplication, division, and matrix power.

Advantages of the CSR format
  - efficient arithmetic operations CSR + CSR, CSR * CSR, etc.
  - efficient row slicing
  - fast matrix vector products

Disadvantages of the CSR format
  - slow column slicing operations (consider CSC)
  - changes to the sparsity structure are expensive (consider LIL or DOK)

Examples
--------

>>> import numpy as np
>>> from scipy.sparse import csr_matrix
>>> csr_matrix((3, 4), dtype=np.int8).toarray()
array([[0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]], dtype=int8)

>>> row = np.array([0, 0, 1, 2, 2, 2])
>>> col = np.array([0, 2, 2, 0, 1, 2])
>>> data = np.array([1, 2, 3, 4, 5, 6])
>>> csr_matrix((data, (row, col)), shape=(3, 3)).toarray()
array([[1, 0, 2],
       [0, 0, 3],
       [4, 5, 6]])

>>> indptr = np.array([0, 2, 3, 6])
>>> indices = np.array([0, 2, 2, 0, 1, 2])
>>> data = np.array([1, 2, 3, 4, 5, 6])
>>> csr_matrix((data, indices, indptr), shape=(3, 3)).toarray()
array([[1, 0, 2],
       [0, 0, 3],
       [4, 5, 6]])

As an example of how to construct a CSR matrix incrementally,
the following snippet builds a term-document matrix from texts:

>>> docs = [['hello', 'world', 'hello'], ['goodbye', 'cruel', 'world']]
>>> indptr = [0]
>>> indices = []
>>> data = []
>>> vocabulary = {}
>>> for d in docs:
...     for term in d:
...         index = vocabulary.setdefault(term, len(vocabulary))
...         indices.append(index)
...         data.append(1)
...     indptr.append(len(indices))
...
>>> csr_matrix((data, indices, indptr), dtype=int).toarray()
array([[2, 1, 0, 0],
       [0, 1, 1, 1]])
*)

val __getitem__ : key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val __iter__ : [> tag] Obj.t -> Py.Object.t
(**
None
*)

val __setitem__ : key:Py.Object.t -> x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val arcsin : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arcsin.

See numpy.arcsin for more information.
*)

val arcsinh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arcsinh.

See numpy.arcsinh for more information.
*)

val arctan : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arctan.

See numpy.arctan for more information.
*)

val arctanh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arctanh.

See numpy.arctanh for more information.
*)

val argmax : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return indices of maximum elements along an axis.

Implicit zero elements are also taken into account. If there are
several maximum values, the index of the first occurrence is returned.

Parameters
----------
axis : {-2, -1, 0, 1, None}, optional
    Axis along which the argmax is computed. If None (default), index
    of the maximum element in the flatten data is returned.
out : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except for
    the default value, as this argument is not used.

Returns
-------
ind : numpy.matrix or int
    Indices of maximum elements. If matrix, its size along `axis` is 1.
*)

val argmin : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return indices of minimum elements along an axis.

Implicit zero elements are also taken into account. If there are
several minimum values, the index of the first occurrence is returned.

Parameters
----------
axis : {-2, -1, 0, 1, None}, optional
    Axis along which the argmin is computed. If None (default), index
    of the minimum element in the flatten data is returned.
out : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except for
    the default value, as this argument is not used.

Returns
-------
 ind : numpy.matrix or int
    Indices of minimum elements. If matrix, its size along `axis` is 1.
*)

val asformat : ?copy:bool -> format:[`S of string | `None] -> [> tag] Obj.t -> Py.Object.t
(**
Return this matrix in the passed format.

Parameters
----------
format : {str, None}
    The desired matrix format ('csr', 'csc', 'lil', 'dok', 'array', ...)
    or None for no conversion.
copy : bool, optional
    If True, the result is guaranteed to not share data with self.

Returns
-------
A : This matrix in the passed format.
*)

val asfptype : [> tag] Obj.t -> Py.Object.t
(**
Upcast matrix to a floating point format (if necessary)
*)

val astype : ?casting:[`No | `Equiv | `Safe | `Same_kind | `Unsafe] -> ?copy:bool -> dtype:[`S of string | `Dtype of Np.Dtype.t] -> [> tag] Obj.t -> Py.Object.t
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

val ceil : [> tag] Obj.t -> Py.Object.t
(**
Element-wise ceil.

See numpy.ceil for more information.
*)

val check_format : ?full_check:bool -> [> tag] Obj.t -> Py.Object.t
(**
check whether the matrix format is valid

Parameters
----------
full_check : bool, optional
    If `True`, rigorous check, O(N) operations. Otherwise
    basic check, O(1) operations (default True).
*)

val conj : ?copy:bool -> [> tag] Obj.t -> Py.Object.t
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

val conjugate : ?copy:bool -> [> tag] Obj.t -> Py.Object.t
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

val copy : [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of this matrix.

No data/indices will be shared between the returned value and current
matrix.
*)

val count_nonzero : [> tag] Obj.t -> Py.Object.t
(**
Number of non-zero entries, equivalent to

np.count_nonzero(a.toarray())

Unlike getnnz() and the nnz property, which return the number of stored
entries (the length of the data attribute), this method counts the
actual number of non-zero entries in data.
*)

val deg2rad : [> tag] Obj.t -> Py.Object.t
(**
Element-wise deg2rad.

See numpy.deg2rad for more information.
*)

val diagonal : ?k:int -> [> tag] Obj.t -> Py.Object.t
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

val dot : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
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

val eliminate_zeros : [> tag] Obj.t -> Py.Object.t
(**
Remove zero entries from the matrix

This is an *in place* operation
*)

val expm1 : [> tag] Obj.t -> Py.Object.t
(**
Element-wise expm1.

See numpy.expm1 for more information.
*)

val floor : [> tag] Obj.t -> Py.Object.t
(**
Element-wise floor.

See numpy.floor for more information.
*)

val getH : [> tag] Obj.t -> Py.Object.t
(**
Return the Hermitian transpose of this matrix.

See Also
--------
numpy.matrix.getH : NumPy's implementation of `getH` for matrices
*)

val get_shape : [> tag] Obj.t -> Py.Object.t
(**
Get shape of a matrix.
*)

val getcol : i:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of column i of the matrix, as a (m x 1)
CSR matrix (column vector).
*)

val getformat : [> tag] Obj.t -> Py.Object.t
(**
Format of a matrix representation as a string.
*)

val getmaxprint : [> tag] Obj.t -> Py.Object.t
(**
Maximum number of elements to display when printed.
*)

val getnnz : ?axis:[`Zero | `One] -> [> tag] Obj.t -> Py.Object.t
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

val getrow : i:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of row i of the matrix, as a (1 x n)
CSR matrix (row vector).
*)

val log1p : [> tag] Obj.t -> Py.Object.t
(**
Element-wise log1p.

See numpy.log1p for more information.
*)

val max : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return the maximum of the matrix or maximum along an axis.
This takes all elements into account, not just the non-zero ones.

Parameters
----------
axis : {-2, -1, 0, 1, None} optional
    Axis along which the sum is computed. The default is to
    compute the maximum over all the matrix elements, returning
    a scalar (i.e. `axis` = `None`).

out : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except
    for the default value, as this argument is not used.

Returns
-------
amax : coo_matrix or scalar
    Maximum of `a`. If `axis` is None, the result is a scalar value.
    If `axis` is given, the result is a sparse.coo_matrix of dimension
    ``a.ndim - 1``.

See Also
--------
min : The minimum value of a sparse matrix along a given axis.
numpy.matrix.max : NumPy's implementation of 'max' for matrices
*)

val maximum : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Element-wise maximum between this and another matrix.
*)

val mean : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?dtype:Np.Dtype.t -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val min : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return the minimum of the matrix or maximum along an axis.
This takes all elements into account, not just the non-zero ones.

Parameters
----------
axis : {-2, -1, 0, 1, None} optional
    Axis along which the sum is computed. The default is to
    compute the minimum over all the matrix elements, returning
    a scalar (i.e. `axis` = `None`).

out : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except for
    the default value, as this argument is not used.

Returns
-------
amin : coo_matrix or scalar
    Minimum of `a`. If `axis` is None, the result is a scalar value.
    If `axis` is given, the result is a sparse.coo_matrix of dimension
    ``a.ndim - 1``.

See Also
--------
max : The maximum value of a sparse matrix along a given axis.
numpy.matrix.min : NumPy's implementation of 'min' for matrices
*)

val minimum : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Element-wise minimum between this and another matrix.
*)

val multiply : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Point-wise multiplication by another matrix, vector, or
scalar.
*)

val nonzero : [> tag] Obj.t -> Py.Object.t
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

val power : ?dtype:Py.Object.t -> n:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
This function performs element-wise power.

Parameters
----------
n : n is a scalar

dtype : If dtype is not specified, the current dtype will be preserved.
*)

val prune : [> tag] Obj.t -> Py.Object.t
(**
Remove empty space after all non-zero elements.
        
*)

val rad2deg : [> tag] Obj.t -> Py.Object.t
(**
Element-wise rad2deg.

See numpy.rad2deg for more information.
*)

val reshape : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> [> tag] Obj.t -> [`ArrayLike|`Object|`Spmatrix] Np.Obj.t
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

val resize : Py.Object.t list -> [> tag] Obj.t -> Py.Object.t
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

val rint : [> tag] Obj.t -> Py.Object.t
(**
Element-wise rint.

See numpy.rint for more information.
*)

val set_shape : shape:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
See `reshape`.
*)

val setdiag : ?k:int -> values:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
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

val sign : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sign.

See numpy.sign for more information.
*)

val sin : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sin.

See numpy.sin for more information.
*)

val sinh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sinh.

See numpy.sinh for more information.
*)

val sort_indices : [> tag] Obj.t -> Py.Object.t
(**
Sort the indices of this matrix *in place*
        
*)

val sorted_indices : [> tag] Obj.t -> Py.Object.t
(**
Return a copy of this matrix with sorted indices
        
*)

val sqrt : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sqrt.

See numpy.sqrt for more information.
*)

val sum : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?dtype:Np.Dtype.t -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val sum_duplicates : [> tag] Obj.t -> Py.Object.t
(**
Eliminate duplicate matrix entries by adding them together

The is an *in place* operation
*)

val tan : [> tag] Obj.t -> Py.Object.t
(**
Element-wise tan.

See numpy.tan for more information.
*)

val tanh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise tanh.

See numpy.tanh for more information.
*)

val toarray : ?order:[`F | `C] -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val tobsr : ?blocksize:Py.Object.t -> ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Block Sparse Row format.

With copy=False, the data/indices may be shared between this matrix and
the resultant bsr_matrix.

When blocksize=(R, C) is provided, it will be used for construction of
the bsr_matrix.
*)

val tocoo : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to COOrdinate format.

With copy=False, the data/indices may be shared between this matrix and
the resultant coo_matrix.
*)

val tocsc : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Compressed Sparse Column format.

With copy=False, the data/indices may be shared between this matrix and
the resultant csc_matrix.
*)

val tocsr : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Compressed Sparse Row format.

With copy=False, the data/indices may be shared between this matrix and
the resultant csr_matrix.
*)

val todense : ?order:[`F | `C] -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val todia : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to sparse DIAgonal format.

With copy=False, the data/indices may be shared between this matrix and
the resultant dia_matrix.
*)

val todok : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Dictionary Of Keys format.

With copy=False, the data/indices may be shared between this matrix and
the resultant dok_matrix.
*)

val tolil : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to List of Lists format.

With copy=False, the data/indices may be shared between this matrix and
the resultant lil_matrix.
*)

val transpose : ?axes:Py.Object.t -> ?copy:bool -> [> tag] Obj.t -> Py.Object.t
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

val trunc : [> tag] Obj.t -> Py.Object.t
(**
Element-wise trunc.

See numpy.trunc for more information.
*)


(** Attribute dtype: get value or raise Not_found if None.*)
val dtype : t -> Np.Dtype.t

(** Attribute dtype: get value as an option. *)
val dtype_opt : t -> (Np.Dtype.t) option


(** Attribute shape: get value or raise Not_found if None.*)
val shape : t -> Py.Object.t

(** Attribute shape: get value as an option. *)
val shape_opt : t -> (Py.Object.t) option


(** Attribute ndim: get value or raise Not_found if None.*)
val ndim : t -> int

(** Attribute ndim: get value as an option. *)
val ndim_opt : t -> (int) option


(** Attribute nnz: get value or raise Not_found if None.*)
val nnz : t -> Py.Object.t

(** Attribute nnz: get value as an option. *)
val nnz_opt : t -> (Py.Object.t) option


(** Attribute data: get value or raise Not_found if None.*)
val data : t -> Py.Object.t

(** Attribute data: get value as an option. *)
val data_opt : t -> (Py.Object.t) option


(** Attribute indices: get value or raise Not_found if None.*)
val indices : t -> Py.Object.t

(** Attribute indices: get value as an option. *)
val indices_opt : t -> (Py.Object.t) option


(** Attribute indptr: get value or raise Not_found if None.*)
val indptr : t -> Py.Object.t

(** Attribute indptr: get value as an option. *)
val indptr_opt : t -> (Py.Object.t) option


(** Attribute has_sorted_indices: get value or raise Not_found if None.*)
val has_sorted_indices : t -> Py.Object.t

(** Attribute has_sorted_indices: get value as an option. *)
val has_sorted_indices_opt : t -> (Py.Object.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Dia_matrix : sig
type tag = [`Dia_matrix]
type t = [`ArrayLike | `Dia_matrix | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?shape:Py.Object.t -> ?dtype:Py.Object.t -> ?copy:Py.Object.t -> arg1:Py.Object.t -> unit -> t
(**
Sparse matrix with DIAgonal storage

This can be instantiated in several ways:
    dia_matrix(D)
        with a dense matrix

    dia_matrix(S)
        with another sparse matrix S (equivalent to S.todia())

    dia_matrix((M, N), [dtype])
        to construct an empty matrix with shape (M, N),
        dtype is optional, defaulting to dtype='d'.

    dia_matrix((data, offsets), shape=(M, N))
        where the ``data[k,:]`` stores the diagonal entries for
        diagonal ``offsets[k]`` (See example below)

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
    DIA format data array of the matrix
offsets
    DIA format offset array of the matrix

Notes
-----

Sparse matrices can be used in arithmetic operations: they support
addition, subtraction, multiplication, division, and matrix power.

Examples
--------

>>> import numpy as np
>>> from scipy.sparse import dia_matrix
>>> dia_matrix((3, 4), dtype=np.int8).toarray()
array([[0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]], dtype=int8)

>>> data = np.array([[1, 2, 3, 4]]).repeat(3, axis=0)
>>> offsets = np.array([0, -1, 2])
>>> dia_matrix((data, offsets), shape=(4, 4)).toarray()
array([[1, 0, 3, 0],
       [1, 2, 0, 4],
       [0, 2, 3, 0],
       [0, 0, 3, 4]])
*)

val __iter__ : [> tag] Obj.t -> Py.Object.t
(**
None
*)

val arcsin : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arcsin.

See numpy.arcsin for more information.
*)

val arcsinh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arcsinh.

See numpy.arcsinh for more information.
*)

val arctan : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arctan.

See numpy.arctan for more information.
*)

val arctanh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arctanh.

See numpy.arctanh for more information.
*)

val asformat : ?copy:bool -> format:[`S of string | `None] -> [> tag] Obj.t -> Py.Object.t
(**
Return this matrix in the passed format.

Parameters
----------
format : {str, None}
    The desired matrix format ('csr', 'csc', 'lil', 'dok', 'array', ...)
    or None for no conversion.
copy : bool, optional
    If True, the result is guaranteed to not share data with self.

Returns
-------
A : This matrix in the passed format.
*)

val asfptype : [> tag] Obj.t -> Py.Object.t
(**
Upcast matrix to a floating point format (if necessary)
*)

val astype : ?casting:[`No | `Equiv | `Safe | `Same_kind | `Unsafe] -> ?copy:bool -> dtype:[`S of string | `Dtype of Np.Dtype.t] -> [> tag] Obj.t -> Py.Object.t
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

val ceil : [> tag] Obj.t -> Py.Object.t
(**
Element-wise ceil.

See numpy.ceil for more information.
*)

val conj : ?copy:bool -> [> tag] Obj.t -> Py.Object.t
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

val conjugate : ?copy:bool -> [> tag] Obj.t -> Py.Object.t
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

val copy : [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of this matrix.

No data/indices will be shared between the returned value and current
matrix.
*)

val count_nonzero : [> tag] Obj.t -> Py.Object.t
(**
Number of non-zero entries, equivalent to

np.count_nonzero(a.toarray())

Unlike getnnz() and the nnz property, which return the number of stored
entries (the length of the data attribute), this method counts the
actual number of non-zero entries in data.
*)

val deg2rad : [> tag] Obj.t -> Py.Object.t
(**
Element-wise deg2rad.

See numpy.deg2rad for more information.
*)

val diagonal : ?k:int -> [> tag] Obj.t -> Py.Object.t
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

val dot : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
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

val expm1 : [> tag] Obj.t -> Py.Object.t
(**
Element-wise expm1.

See numpy.expm1 for more information.
*)

val floor : [> tag] Obj.t -> Py.Object.t
(**
Element-wise floor.

See numpy.floor for more information.
*)

val getH : [> tag] Obj.t -> Py.Object.t
(**
Return the Hermitian transpose of this matrix.

See Also
--------
numpy.matrix.getH : NumPy's implementation of `getH` for matrices
*)

val get_shape : [> tag] Obj.t -> Py.Object.t
(**
Get shape of a matrix.
*)

val getcol : j:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of column j of the matrix, as an (m x 1) sparse
matrix (column vector).
*)

val getformat : [> tag] Obj.t -> Py.Object.t
(**
Format of a matrix representation as a string.
*)

val getmaxprint : [> tag] Obj.t -> Py.Object.t
(**
Maximum number of elements to display when printed.
*)

val getnnz : ?axis:[`Zero | `One] -> [> tag] Obj.t -> Py.Object.t
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

val getrow : i:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of row i of the matrix, as a (1 x n) sparse
matrix (row vector).
*)

val log1p : [> tag] Obj.t -> Py.Object.t
(**
Element-wise log1p.

See numpy.log1p for more information.
*)

val maximum : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Element-wise maximum between this and another matrix.
*)

val mean : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?dtype:Np.Dtype.t -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val minimum : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Element-wise minimum between this and another matrix.
*)

val multiply : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Point-wise multiplication by another matrix
        
*)

val nonzero : [> tag] Obj.t -> Py.Object.t
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

val power : ?dtype:Py.Object.t -> n:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
This function performs element-wise power.

Parameters
----------
n : n is a scalar

dtype : If dtype is not specified, the current dtype will be preserved.
*)

val rad2deg : [> tag] Obj.t -> Py.Object.t
(**
Element-wise rad2deg.

See numpy.rad2deg for more information.
*)

val reshape : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> [> tag] Obj.t -> [`ArrayLike|`Object|`Spmatrix] Np.Obj.t
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

val resize : Py.Object.t list -> [> tag] Obj.t -> Py.Object.t
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

val rint : [> tag] Obj.t -> Py.Object.t
(**
Element-wise rint.

See numpy.rint for more information.
*)

val set_shape : shape:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
See `reshape`.
*)

val setdiag : ?k:int -> values:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
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

val sign : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sign.

See numpy.sign for more information.
*)

val sin : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sin.

See numpy.sin for more information.
*)

val sinh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sinh.

See numpy.sinh for more information.
*)

val sqrt : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sqrt.

See numpy.sqrt for more information.
*)

val sum : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?dtype:Np.Dtype.t -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val tan : [> tag] Obj.t -> Py.Object.t
(**
Element-wise tan.

See numpy.tan for more information.
*)

val tanh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise tanh.

See numpy.tanh for more information.
*)

val toarray : ?order:[`F | `C] -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val tobsr : ?blocksize:Py.Object.t -> ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Block Sparse Row format.

With copy=False, the data/indices may be shared between this matrix and
the resultant bsr_matrix.

When blocksize=(R, C) is provided, it will be used for construction of
the bsr_matrix.
*)

val tocoo : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to COOrdinate format.

With copy=False, the data/indices may be shared between this matrix and
the resultant coo_matrix.
*)

val tocsc : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Compressed Sparse Column format.

With copy=False, the data/indices may be shared between this matrix and
the resultant csc_matrix.
*)

val tocsr : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Compressed Sparse Row format.

With copy=False, the data/indices may be shared between this matrix and
the resultant csr_matrix.
*)

val todense : ?order:[`F | `C] -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val todia : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to sparse DIAgonal format.

With copy=False, the data/indices may be shared between this matrix and
the resultant dia_matrix.
*)

val todok : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Dictionary Of Keys format.

With copy=False, the data/indices may be shared between this matrix and
the resultant dok_matrix.
*)

val tolil : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to List of Lists format.

With copy=False, the data/indices may be shared between this matrix and
the resultant lil_matrix.
*)

val transpose : ?axes:Py.Object.t -> ?copy:bool -> [> tag] Obj.t -> Py.Object.t
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

val trunc : [> tag] Obj.t -> Py.Object.t
(**
Element-wise trunc.

See numpy.trunc for more information.
*)


(** Attribute dtype: get value or raise Not_found if None.*)
val dtype : t -> Np.Dtype.t

(** Attribute dtype: get value as an option. *)
val dtype_opt : t -> (Np.Dtype.t) option


(** Attribute shape: get value or raise Not_found if None.*)
val shape : t -> Py.Object.t

(** Attribute shape: get value as an option. *)
val shape_opt : t -> (Py.Object.t) option


(** Attribute ndim: get value or raise Not_found if None.*)
val ndim : t -> int

(** Attribute ndim: get value as an option. *)
val ndim_opt : t -> (int) option


(** Attribute nnz: get value or raise Not_found if None.*)
val nnz : t -> Py.Object.t

(** Attribute nnz: get value as an option. *)
val nnz_opt : t -> (Py.Object.t) option


(** Attribute data: get value or raise Not_found if None.*)
val data : t -> Py.Object.t

(** Attribute data: get value as an option. *)
val data_opt : t -> (Py.Object.t) option


(** Attribute offsets: get value or raise Not_found if None.*)
val offsets : t -> Py.Object.t

(** Attribute offsets: get value as an option. *)
val offsets_opt : t -> (Py.Object.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Dok_matrix : sig
type tag = [`Dok_matrix]
type t = [`ArrayLike | `Dok_matrix | `IndexMixin | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_index : t -> [`IndexMixin] Obj.t
val create : ?shape:Py.Object.t -> ?dtype:Py.Object.t -> ?copy:Py.Object.t -> arg1:Py.Object.t -> unit -> t
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

val __getitem__ : key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val __iter__ : [> tag] Obj.t -> Py.Object.t
(**
None
*)

val __setitem__ : key:Py.Object.t -> x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val asformat : ?copy:bool -> format:[`S of string | `None] -> [> tag] Obj.t -> Py.Object.t
(**
Return this matrix in the passed format.

Parameters
----------
format : {str, None}
    The desired matrix format ('csr', 'csc', 'lil', 'dok', 'array', ...)
    or None for no conversion.
copy : bool, optional
    If True, the result is guaranteed to not share data with self.

Returns
-------
A : This matrix in the passed format.
*)

val asfptype : [> tag] Obj.t -> Py.Object.t
(**
Upcast matrix to a floating point format (if necessary)
*)

val astype : ?casting:[`No | `Equiv | `Safe | `Same_kind | `Unsafe] -> ?copy:bool -> dtype:[`S of string | `Dtype of Np.Dtype.t] -> [> tag] Obj.t -> Py.Object.t
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

val clear : [> tag] Obj.t -> Py.Object.t
(**
D.clear() -> None.  Remove all items from D.
*)

val conj : ?copy:bool -> [> tag] Obj.t -> Py.Object.t
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

val conjtransp : [> tag] Obj.t -> Py.Object.t
(**
Return the conjugate transpose.
*)

val conjugate : ?copy:bool -> [> tag] Obj.t -> Py.Object.t
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

val copy : [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of this matrix.

No data/indices will be shared between the returned value and current
matrix.
*)

val count_nonzero : [> tag] Obj.t -> Py.Object.t
(**
Number of non-zero entries, equivalent to

np.count_nonzero(a.toarray())

Unlike getnnz() and the nnz property, which return the number of stored
entries (the length of the data attribute), this method counts the
actual number of non-zero entries in data.
*)

val diagonal : ?k:int -> [> tag] Obj.t -> Py.Object.t
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

val dot : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
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

val fromkeys : ?value:Py.Object.t -> iterable:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Create a new dictionary with keys from iterable and values set to value.
*)

val get : ?default:Py.Object.t -> key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
This overrides the dict.get method, providing type checking
but otherwise equivalent functionality.
*)

val getH : [> tag] Obj.t -> Py.Object.t
(**
Return the Hermitian transpose of this matrix.

See Also
--------
numpy.matrix.getH : NumPy's implementation of `getH` for matrices
*)

val get_shape : [> tag] Obj.t -> Py.Object.t
(**
Get shape of a matrix.
*)

val getcol : j:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of column j of the matrix, as an (m x 1) sparse
matrix (column vector).
*)

val getformat : [> tag] Obj.t -> Py.Object.t
(**
Format of a matrix representation as a string.
*)

val getmaxprint : [> tag] Obj.t -> Py.Object.t
(**
Maximum number of elements to display when printed.
*)

val getnnz : ?axis:[`Zero | `One] -> [> tag] Obj.t -> Py.Object.t
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

val getrow : i:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of row i of the matrix, as a (1 x n) sparse
matrix (row vector).
*)

val items : [> tag] Obj.t -> Py.Object.t
(**
D.items() -> a set-like object providing a view on D's items
*)

val keys : [> tag] Obj.t -> Py.Object.t
(**
D.keys() -> a set-like object providing a view on D's keys
*)

val maximum : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Element-wise maximum between this and another matrix.
*)

val mean : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?dtype:Np.Dtype.t -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val minimum : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Element-wise minimum between this and another matrix.
*)

val multiply : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Point-wise multiplication by another matrix
        
*)

val nonzero : [> tag] Obj.t -> Py.Object.t
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

val pop : ?d:Py.Object.t -> k:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
D.pop(k[,d]) -> v, remove specified key and return the corresponding value.
If key is not found, d is returned if given, otherwise KeyError is raised
*)

val popitem : [> tag] Obj.t -> Py.Object.t
(**
Remove and return a (key, value) pair as a 2-tuple.

Pairs are returned in LIFO (last-in, first-out) order.
Raises KeyError if the dict is empty.
*)

val power : ?dtype:Py.Object.t -> n:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Element-wise power.
*)

val reshape : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> [> tag] Obj.t -> [`ArrayLike|`Object|`Spmatrix] Np.Obj.t
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

val resize : Py.Object.t list -> [> tag] Obj.t -> Py.Object.t
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

val set_shape : shape:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
See `reshape`.
*)

val setdefault : ?default:Py.Object.t -> key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Insert key with a value of default if key is not in the dictionary.

Return the value for key if key is in the dictionary, else default.
*)

val setdiag : ?k:int -> values:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
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

val sum : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?dtype:Np.Dtype.t -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val toarray : ?order:[`F | `C] -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val tobsr : ?blocksize:Py.Object.t -> ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Block Sparse Row format.

With copy=False, the data/indices may be shared between this matrix and
the resultant bsr_matrix.

When blocksize=(R, C) is provided, it will be used for construction of
the bsr_matrix.
*)

val tocoo : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to COOrdinate format.

With copy=False, the data/indices may be shared between this matrix and
the resultant coo_matrix.
*)

val tocsc : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Compressed Sparse Column format.

With copy=False, the data/indices may be shared between this matrix and
the resultant csc_matrix.
*)

val tocsr : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Compressed Sparse Row format.

With copy=False, the data/indices may be shared between this matrix and
the resultant csr_matrix.
*)

val todense : ?order:[`F | `C] -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val todia : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to sparse DIAgonal format.

With copy=False, the data/indices may be shared between this matrix and
the resultant dia_matrix.
*)

val todok : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Dictionary Of Keys format.

With copy=False, the data/indices may be shared between this matrix and
the resultant dok_matrix.
*)

val tolil : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to List of Lists format.

With copy=False, the data/indices may be shared between this matrix and
the resultant lil_matrix.
*)

val transpose : ?axes:Py.Object.t -> ?copy:bool -> [> tag] Obj.t -> Py.Object.t
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

val update : val_:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
D.update([E, ]**F) -> None.  Update D from dict/iterable E and F.
If E is present and has a .keys() method, then does:  for k in E: D[k] = E[k]
If E is present and lacks a .keys() method, then does:  for k, v in E: D[k] = v
In either case, this is followed by: for k in F:  D[k] = F[k]
*)

val values : [> tag] Obj.t -> Py.Object.t
(**
D.values() -> an object providing a view on D's values
*)


(** Attribute dtype: get value or raise Not_found if None.*)
val dtype : t -> Np.Dtype.t

(** Attribute dtype: get value as an option. *)
val dtype_opt : t -> (Np.Dtype.t) option


(** Attribute shape: get value or raise Not_found if None.*)
val shape : t -> Py.Object.t

(** Attribute shape: get value as an option. *)
val shape_opt : t -> (Py.Object.t) option


(** Attribute ndim: get value or raise Not_found if None.*)
val ndim : t -> int

(** Attribute ndim: get value as an option. *)
val ndim_opt : t -> (int) option


(** Attribute nnz: get value or raise Not_found if None.*)
val nnz : t -> Py.Object.t

(** Attribute nnz: get value as an option. *)
val nnz_opt : t -> (Py.Object.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Lil_matrix : sig
type tag = [`Lil_matrix]
type t = [`ArrayLike | `IndexMixin | `Lil_matrix | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_index : t -> [`IndexMixin] Obj.t
val create : ?shape:Py.Object.t -> ?dtype:Py.Object.t -> ?copy:Py.Object.t -> arg1:Py.Object.t -> unit -> t
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

val __getitem__ : key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val __iter__ : [> tag] Obj.t -> Py.Object.t
(**
None
*)

val __setitem__ : key:Py.Object.t -> x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val asformat : ?copy:bool -> format:[`S of string | `None] -> [> tag] Obj.t -> Py.Object.t
(**
Return this matrix in the passed format.

Parameters
----------
format : {str, None}
    The desired matrix format ('csr', 'csc', 'lil', 'dok', 'array', ...)
    or None for no conversion.
copy : bool, optional
    If True, the result is guaranteed to not share data with self.

Returns
-------
A : This matrix in the passed format.
*)

val asfptype : [> tag] Obj.t -> Py.Object.t
(**
Upcast matrix to a floating point format (if necessary)
*)

val astype : ?casting:[`No | `Equiv | `Safe | `Same_kind | `Unsafe] -> ?copy:bool -> dtype:[`S of string | `Dtype of Np.Dtype.t] -> [> tag] Obj.t -> Py.Object.t
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

val conj : ?copy:bool -> [> tag] Obj.t -> Py.Object.t
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

val conjugate : ?copy:bool -> [> tag] Obj.t -> Py.Object.t
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

val copy : [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of this matrix.

No data/indices will be shared between the returned value and current
matrix.
*)

val count_nonzero : [> tag] Obj.t -> Py.Object.t
(**
Number of non-zero entries, equivalent to

np.count_nonzero(a.toarray())

Unlike getnnz() and the nnz property, which return the number of stored
entries (the length of the data attribute), this method counts the
actual number of non-zero entries in data.
*)

val diagonal : ?k:int -> [> tag] Obj.t -> Py.Object.t
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

val dot : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
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

val getH : [> tag] Obj.t -> Py.Object.t
(**
Return the Hermitian transpose of this matrix.

See Also
--------
numpy.matrix.getH : NumPy's implementation of `getH` for matrices
*)

val get_shape : [> tag] Obj.t -> Py.Object.t
(**
Get shape of a matrix.
*)

val getcol : j:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of column j of the matrix, as an (m x 1) sparse
matrix (column vector).
*)

val getformat : [> tag] Obj.t -> Py.Object.t
(**
Format of a matrix representation as a string.
*)

val getmaxprint : [> tag] Obj.t -> Py.Object.t
(**
Maximum number of elements to display when printed.
*)

val getnnz : ?axis:[`Zero | `One] -> [> tag] Obj.t -> Py.Object.t
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

val getrow : i:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of the 'i'th row.
        
*)

val getrowview : i:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a view of the 'i'th row (without copying).
        
*)

val maximum : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Element-wise maximum between this and another matrix.
*)

val mean : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?dtype:Np.Dtype.t -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val minimum : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Element-wise minimum between this and another matrix.
*)

val multiply : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Point-wise multiplication by another matrix
        
*)

val nonzero : [> tag] Obj.t -> Py.Object.t
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

val power : ?dtype:Py.Object.t -> n:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Element-wise power.
*)

val reshape : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> [> tag] Obj.t -> [`ArrayLike|`Object|`Spmatrix] Np.Obj.t
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

val resize : Py.Object.t list -> [> tag] Obj.t -> Py.Object.t
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

val set_shape : shape:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
See `reshape`.
*)

val setdiag : ?k:int -> values:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
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

val sum : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?dtype:Np.Dtype.t -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val toarray : ?order:[`F | `C] -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val tobsr : ?blocksize:Py.Object.t -> ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Block Sparse Row format.

With copy=False, the data/indices may be shared between this matrix and
the resultant bsr_matrix.

When blocksize=(R, C) is provided, it will be used for construction of
the bsr_matrix.
*)

val tocoo : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to COOrdinate format.

With copy=False, the data/indices may be shared between this matrix and
the resultant coo_matrix.
*)

val tocsc : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Compressed Sparse Column format.

With copy=False, the data/indices may be shared between this matrix and
the resultant csc_matrix.
*)

val tocsr : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Compressed Sparse Row format.

With copy=False, the data/indices may be shared between this matrix and
the resultant csr_matrix.
*)

val todense : ?order:[`F | `C] -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val todia : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to sparse DIAgonal format.

With copy=False, the data/indices may be shared between this matrix and
the resultant dia_matrix.
*)

val todok : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Dictionary Of Keys format.

With copy=False, the data/indices may be shared between this matrix and
the resultant dok_matrix.
*)

val tolil : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to List of Lists format.

With copy=False, the data/indices may be shared between this matrix and
the resultant lil_matrix.
*)

val transpose : ?axes:Py.Object.t -> ?copy:bool -> [> tag] Obj.t -> Py.Object.t
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


(** Attribute dtype: get value or raise Not_found if None.*)
val dtype : t -> Np.Dtype.t

(** Attribute dtype: get value as an option. *)
val dtype_opt : t -> (Np.Dtype.t) option


(** Attribute shape: get value or raise Not_found if None.*)
val shape : t -> Py.Object.t

(** Attribute shape: get value as an option. *)
val shape_opt : t -> (Py.Object.t) option


(** Attribute ndim: get value or raise Not_found if None.*)
val ndim : t -> int

(** Attribute ndim: get value as an option. *)
val ndim_opt : t -> (int) option


(** Attribute nnz: get value or raise Not_found if None.*)
val nnz : t -> Py.Object.t

(** Attribute nnz: get value as an option. *)
val nnz_opt : t -> (Py.Object.t) option


(** Attribute data: get value or raise Not_found if None.*)
val data : t -> Py.Object.t

(** Attribute data: get value as an option. *)
val data_opt : t -> (Py.Object.t) option


(** Attribute rows: get value or raise Not_found if None.*)
val rows : t -> Py.Object.t

(** Attribute rows: get value as an option. *)
val rows_opt : t -> (Py.Object.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Spmatrix : sig
type tag = [`Spmatrix]
type t = [`ArrayLike | `Object | `Spmatrix] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?maxprint:Py.Object.t -> unit -> t
(**
This class provides a base class for all sparse matrices.  It
cannot be instantiated.  Most of the work is provided by subclasses.
*)

val __iter__ : [> tag] Obj.t -> Py.Object.t
(**
None
*)

val asformat : ?copy:bool -> format:[`S of string | `None] -> [> tag] Obj.t -> Py.Object.t
(**
Return this matrix in the passed format.

Parameters
----------
format : {str, None}
    The desired matrix format ('csr', 'csc', 'lil', 'dok', 'array', ...)
    or None for no conversion.
copy : bool, optional
    If True, the result is guaranteed to not share data with self.

Returns
-------
A : This matrix in the passed format.
*)

val asfptype : [> tag] Obj.t -> Py.Object.t
(**
Upcast matrix to a floating point format (if necessary)
*)

val astype : ?casting:[`No | `Equiv | `Safe | `Same_kind | `Unsafe] -> ?copy:bool -> dtype:[`S of string | `Dtype of Np.Dtype.t] -> [> tag] Obj.t -> Py.Object.t
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

val conj : ?copy:bool -> [> tag] Obj.t -> Py.Object.t
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

val conjugate : ?copy:bool -> [> tag] Obj.t -> Py.Object.t
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

val copy : [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of this matrix.

No data/indices will be shared between the returned value and current
matrix.
*)

val count_nonzero : [> tag] Obj.t -> Py.Object.t
(**
Number of non-zero entries, equivalent to

np.count_nonzero(a.toarray())

Unlike getnnz() and the nnz property, which return the number of stored
entries (the length of the data attribute), this method counts the
actual number of non-zero entries in data.
*)

val diagonal : ?k:int -> [> tag] Obj.t -> Py.Object.t
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

val dot : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
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

val getH : [> tag] Obj.t -> Py.Object.t
(**
Return the Hermitian transpose of this matrix.

See Also
--------
numpy.matrix.getH : NumPy's implementation of `getH` for matrices
*)

val get_shape : [> tag] Obj.t -> Py.Object.t
(**
Get shape of a matrix.
*)

val getcol : j:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of column j of the matrix, as an (m x 1) sparse
matrix (column vector).
*)

val getformat : [> tag] Obj.t -> Py.Object.t
(**
Format of a matrix representation as a string.
*)

val getmaxprint : [> tag] Obj.t -> Py.Object.t
(**
Maximum number of elements to display when printed.
*)

val getnnz : ?axis:[`Zero | `One] -> [> tag] Obj.t -> Py.Object.t
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

val getrow : i:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of row i of the matrix, as a (1 x n) sparse
matrix (row vector).
*)

val maximum : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Element-wise maximum between this and another matrix.
*)

val mean : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?dtype:Np.Dtype.t -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val minimum : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Element-wise minimum between this and another matrix.
*)

val multiply : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Point-wise multiplication by another matrix
        
*)

val nonzero : [> tag] Obj.t -> Py.Object.t
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

val power : ?dtype:Py.Object.t -> n:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Element-wise power.
*)

val reshape : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> [> tag] Obj.t -> [`ArrayLike|`Object|`Spmatrix] Np.Obj.t
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

val resize : shape:(int * int) -> [> tag] Obj.t -> Py.Object.t
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

val set_shape : shape:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
See `reshape`.
*)

val setdiag : ?k:int -> values:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
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

val sum : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?dtype:Np.Dtype.t -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val toarray : ?order:[`F | `C] -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val tobsr : ?blocksize:Py.Object.t -> ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Block Sparse Row format.

With copy=False, the data/indices may be shared between this matrix and
the resultant bsr_matrix.

When blocksize=(R, C) is provided, it will be used for construction of
the bsr_matrix.
*)

val tocoo : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to COOrdinate format.

With copy=False, the data/indices may be shared between this matrix and
the resultant coo_matrix.
*)

val tocsc : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Compressed Sparse Column format.

With copy=False, the data/indices may be shared between this matrix and
the resultant csc_matrix.
*)

val tocsr : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Compressed Sparse Row format.

With copy=False, the data/indices may be shared between this matrix and
the resultant csr_matrix.
*)

val todense : ?order:[`F | `C] -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val todia : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to sparse DIAgonal format.

With copy=False, the data/indices may be shared between this matrix and
the resultant dia_matrix.
*)

val todok : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Dictionary Of Keys format.

With copy=False, the data/indices may be shared between this matrix and
the resultant dok_matrix.
*)

val tolil : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to List of Lists format.

With copy=False, the data/indices may be shared between this matrix and
the resultant lil_matrix.
*)

val transpose : ?axes:Py.Object.t -> ?copy:bool -> [> tag] Obj.t -> Py.Object.t
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
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Base : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module SparseFormatWarning : sig
type tag = [`SparseFormatWarning]
type t = [`BaseException | `Object | `SparseFormatWarning] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_exception : t -> [`BaseException] Obj.t
val with_traceback : tb:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Exception.with_traceback(tb) --
set self.__traceback__ to tb and return self.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val asmatrix : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> Py.Object.t
(**
None
*)

val broadcast_to : ?subok:bool -> array:[>`Ndarray] Np.Obj.t -> shape:Py.Object.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Broadcast an array to a new shape.

Parameters
----------
array : array_like
    The array to broadcast.
shape : tuple
    The shape of the desired array.
subok : bool, optional
    If True, then sub-classes will be passed-through, otherwise
    the returned array will be forced to be a base-class array (default).

Returns
-------
broadcast : array
    A readonly view on the original array with the given shape. It is
    typically not contiguous. Furthermore, more than one element of a
    broadcasted array may refer to a single memory location.

Raises
------
ValueError
    If the array is not compatible with the new shape according to NumPy's
    broadcasting rules.

Notes
-----
.. versionadded:: 1.10.0

Examples
--------
>>> x = np.array([1, 2, 3])
>>> np.broadcast_to(x, (3, 3))
array([[1, 2, 3],
       [1, 2, 3],
       [1, 2, 3]])
*)

val check_reshape_kwargs : Py.Object.t -> Py.Object.t
(**
Unpack keyword arguments for reshape function.

This is useful because keyword arguments after star arguments are not
allowed in Python 2, but star keyword arguments are. This function unpacks
'order' and 'copy' from the star keyword arguments (with defaults) and
throws an error for any remaining.
*)

val check_shape : ?current_shape:Py.Object.t -> args:Py.Object.t -> unit -> Py.Object.t
(**
Imitate numpy.matrix handling of shape arguments
*)

val get_sum_dtype : Py.Object.t -> Py.Object.t
(**
Mimic numpy's casting for np.sum
*)

val isdense : Py.Object.t -> Py.Object.t
(**
None
*)

val isintlike : Py.Object.t -> Py.Object.t
(**
Is x appropriate as an index into a sparse matrix? Returns True
if it can be cast safely to a machine int.
*)

val isscalarlike : Py.Object.t -> Py.Object.t
(**
Is x either a scalar, an array scalar, or a 0-dim array?
*)

val issparse : Py.Object.t -> Py.Object.t
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

val isspmatrix : Py.Object.t -> Py.Object.t
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

val validateaxis : Py.Object.t -> Py.Object.t
(**
None
*)


end

module Bsr : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val check_shape : ?current_shape:Py.Object.t -> args:Py.Object.t -> unit -> Py.Object.t
(**
Imitate numpy.matrix handling of shape arguments
*)

val get_index_dtype : ?arrays:Py.Object.t -> ?maxval:float -> ?check_contents:bool -> unit -> Np.Dtype.t
(**
Based on input (integer) arrays `a`, determine a suitable index data
type that can hold the data in the arrays.

Parameters
----------
arrays : tuple of array_like
    Input arrays whose types/contents to check
maxval : float, optional
    Maximum value needed
check_contents : bool, optional
    Whether to check the values in the arrays and not just their types.
    Default: False (check only the types)

Returns
-------
dtype : dtype
    Suitable index data type (int32 or int64)
*)

val getdtype : ?a:Py.Object.t -> ?default:Py.Object.t -> dtype:Py.Object.t -> unit -> Py.Object.t
(**
Function used to simplify argument processing.  If 'dtype' is not
specified (is None), returns a.dtype; otherwise returns a np.dtype
object created from the specified dtype argument.  If 'dtype' and 'a'
are both None, construct a data type out of the 'default' parameter.
Furthermore, 'dtype' must be in 'allowed' set.
*)

val isshape : ?nonneg:Py.Object.t -> x:Py.Object.t -> unit -> Py.Object.t
(**
Is x a valid 2-tuple of dimensions?

If nonneg, also checks that the dimensions are non-negative.
*)

val isspmatrix : Py.Object.t -> Py.Object.t
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

val isspmatrix_bsr : Py.Object.t -> Py.Object.t
(**
Is x of a bsr_matrix type?

Parameters
----------
x
    object to check for being a bsr matrix

Returns
-------
bool
    True if x is a bsr matrix, False otherwise

Examples
--------
>>> from scipy.sparse import bsr_matrix, isspmatrix_bsr
>>> isspmatrix_bsr(bsr_matrix([[5]]))
True

>>> from scipy.sparse import bsr_matrix, csr_matrix, isspmatrix_bsr
>>> isspmatrix_bsr(csr_matrix([[5]]))
False
*)

val to_native : Py.Object.t -> Py.Object.t
(**
None
*)

val upcast : Py.Object.t list -> Py.Object.t
(**
Returns the nearest supported sparse dtype for the
combination of one or more types.

upcast(t0, t1, ..., tn) -> T  where T is a supported dtype

Examples
--------

>>> upcast('int32')
<type 'numpy.int32'>
>>> upcast('bool')
<type 'numpy.bool_'>
>>> upcast('int32','float32')
<type 'numpy.float64'>
>>> upcast('bool',complex,float)
<type 'numpy.complex128'>
*)

val warn : ?category:Py.Object.t -> ?stacklevel:Py.Object.t -> ?source:Py.Object.t -> message:Py.Object.t -> unit -> Py.Object.t
(**
Issue a warning, or maybe ignore it or raise an exception.
*)


end

module Compressed : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module IndexMixin : sig
type tag = [`IndexMixin]
type t = [`IndexMixin | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : unit -> t
(**
This class provides common dispatching and validation logic for indexing.
*)

val __getitem__ : key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val __setitem__ : key:Py.Object.t -> x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val getcol : i:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return a copy of column i of the matrix, as a (m x 1) column vector.
        
*)

val getrow : i:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return a copy of row i of the matrix, as a (1 x n) row vector.
        
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val asmatrix : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> Py.Object.t
(**
None
*)

val check_shape : ?current_shape:Py.Object.t -> args:Py.Object.t -> unit -> Py.Object.t
(**
Imitate numpy.matrix handling of shape arguments
*)

val downcast_intp_index : Py.Object.t -> Py.Object.t
(**
Down-cast index array to np.intp dtype if it is of a larger dtype.

Raise an error if the array contains a value that is too large for
intp.
*)

val get_index_dtype : ?arrays:Py.Object.t -> ?maxval:float -> ?check_contents:bool -> unit -> Np.Dtype.t
(**
Based on input (integer) arrays `a`, determine a suitable index data
type that can hold the data in the arrays.

Parameters
----------
arrays : tuple of array_like
    Input arrays whose types/contents to check
maxval : float, optional
    Maximum value needed
check_contents : bool, optional
    Whether to check the values in the arrays and not just their types.
    Default: False (check only the types)

Returns
-------
dtype : dtype
    Suitable index data type (int32 or int64)
*)

val get_sum_dtype : Py.Object.t -> Py.Object.t
(**
Mimic numpy's casting for np.sum
*)

val getdtype : ?a:Py.Object.t -> ?default:Py.Object.t -> dtype:Py.Object.t -> unit -> Py.Object.t
(**
Function used to simplify argument processing.  If 'dtype' is not
specified (is None), returns a.dtype; otherwise returns a np.dtype
object created from the specified dtype argument.  If 'dtype' and 'a'
are both None, construct a data type out of the 'default' parameter.
Furthermore, 'dtype' must be in 'allowed' set.
*)

val isdense : Py.Object.t -> Py.Object.t
(**
None
*)

val isintlike : Py.Object.t -> Py.Object.t
(**
Is x appropriate as an index into a sparse matrix? Returns True
if it can be cast safely to a machine int.
*)

val isscalarlike : Py.Object.t -> Py.Object.t
(**
Is x either a scalar, an array scalar, or a 0-dim array?
*)

val isshape : ?nonneg:Py.Object.t -> x:Py.Object.t -> unit -> Py.Object.t
(**
Is x a valid 2-tuple of dimensions?

If nonneg, also checks that the dimensions are non-negative.
*)

val isspmatrix : Py.Object.t -> Py.Object.t
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

val matrix : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> Py.Object.t
(**
None
*)

val to_native : Py.Object.t -> Py.Object.t
(**
None
*)

val upcast : Py.Object.t list -> Py.Object.t
(**
Returns the nearest supported sparse dtype for the
combination of one or more types.

upcast(t0, t1, ..., tn) -> T  where T is a supported dtype

Examples
--------

>>> upcast('int32')
<type 'numpy.int32'>
>>> upcast('bool')
<type 'numpy.bool_'>
>>> upcast('int32','float32')
<type 'numpy.float64'>
>>> upcast('bool',complex,float)
<type 'numpy.complex128'>
*)

val upcast_char : Py.Object.t list -> Py.Object.t
(**
Same as `upcast` but taking dtype.char as input (faster).
*)

val warn : ?category:Py.Object.t -> ?stacklevel:Py.Object.t -> ?source:Py.Object.t -> message:Py.Object.t -> unit -> Py.Object.t
(**
Issue a warning, or maybe ignore it or raise an exception.
*)


end

module Construct : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val block_diag : ?format:string -> ?dtype:Py.Object.t -> mats:Py.Object.t -> unit -> [`ArrayLike|`Object|`Spmatrix] Np.Obj.t
(**
Build a block diagonal sparse matrix from provided matrices.

Parameters
----------
mats : sequence of matrices
    Input matrices.
format : str, optional
    The sparse format of the result (e.g. 'csr').  If not given, the matrix
    is returned in 'coo' format.
dtype : dtype specifier, optional
    The data-type of the output matrix.  If not given, the dtype is
    determined from that of `blocks`.

Returns
-------
res : sparse matrix

Notes
-----

.. versionadded:: 0.11.0

See Also
--------
bmat, diags

Examples
--------
>>> from scipy.sparse import coo_matrix, block_diag
>>> A = coo_matrix([[1, 2], [3, 4]])
>>> B = coo_matrix([[5], [6]])
>>> C = coo_matrix([[7]])
>>> block_diag((A, B, C)).toarray()
array([[1, 2, 0, 0],
       [3, 4, 0, 0],
       [0, 0, 5, 0],
       [0, 0, 6, 0],
       [0, 0, 0, 7]])
*)

val bmat : ?format:[`Lil | `Bsr | `Csr | `Csc | `Coo | `Dia | `Dok] -> ?dtype:Np.Dtype.t -> blocks:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Object|`Spmatrix] Np.Obj.t
(**
Build a sparse matrix from sparse sub-blocks

Parameters
----------
blocks : array_like
    Grid of sparse matrices with compatible shapes.
    An entry of None implies an all-zero matrix.
format : {'bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil'}, optional
    The sparse format of the result (e.g. 'csr').  By default an
    appropriate sparse matrix format is returned.
    This choice is subject to change.
dtype : dtype, optional
    The data-type of the output matrix.  If not given, the dtype is
    determined from that of `blocks`.

Returns
-------
bmat : sparse matrix

See Also
--------
block_diag, diags

Examples
--------
>>> from scipy.sparse import coo_matrix, bmat
>>> A = coo_matrix([[1, 2], [3, 4]])
>>> B = coo_matrix([[5], [6]])
>>> C = coo_matrix([[7]])
>>> bmat([[A, B], [None, C]]).toarray()
array([[1, 2, 5],
       [3, 4, 6],
       [0, 0, 7]])

>>> bmat([[A, None], [None, C]]).toarray()
array([[1, 2, 0],
       [3, 4, 0],
       [0, 0, 7]])
*)

val diags : ?offsets:Py.Object.t -> ?shape:Py.Object.t -> ?format:[`Lil | `Csr | `Csc | `Dia | `T of Py.Object.t] -> ?dtype:Np.Dtype.t -> diagonals:Py.Object.t -> unit -> Py.Object.t
(**
Construct a sparse matrix from diagonals.

Parameters
----------
diagonals : sequence of array_like
    Sequence of arrays containing the matrix diagonals,
    corresponding to `offsets`.
offsets : sequence of int or an int, optional
    Diagonals to set:
      - k = 0  the main diagonal (default)
      - k > 0  the k-th upper diagonal
      - k < 0  the k-th lower diagonal
shape : tuple of int, optional
    Shape of the result. If omitted, a square matrix large enough
    to contain the diagonals is returned.
format : {'dia', 'csr', 'csc', 'lil', ...}, optional
    Matrix format of the result.  By default (format=None) an
    appropriate sparse matrix format is returned.  This choice is
    subject to change.
dtype : dtype, optional
    Data type of the matrix.

See Also
--------
spdiags : construct matrix from diagonals

Notes
-----
This function differs from `spdiags` in the way it handles
off-diagonals.

The result from `diags` is the sparse equivalent of::

    np.diag(diagonals[0], offsets[0])
    + ...
    + np.diag(diagonals[k], offsets[k])

Repeated diagonal offsets are disallowed.

.. versionadded:: 0.11

Examples
--------
>>> from scipy.sparse import diags
>>> diagonals = [[1, 2, 3, 4], [1, 2, 3], [1, 2]]
>>> diags(diagonals, [0, -1, 2]).toarray()
array([[1, 0, 1, 0],
       [1, 2, 0, 2],
       [0, 2, 3, 0],
       [0, 0, 3, 4]])

Broadcasting of scalars is supported (but shape needs to be
specified):

>>> diags([1, -2, 1], [-1, 0, 1], shape=(4, 4)).toarray()
array([[-2.,  1.,  0.,  0.],
       [ 1., -2.,  1.,  0.],
       [ 0.,  1., -2.,  1.],
       [ 0.,  0.,  1., -2.]])


If only one diagonal is wanted (as in `numpy.diag`), the following
works as well:

>>> diags([1, 2, 3], 1).toarray()
array([[ 0.,  1.,  0.,  0.],
       [ 0.,  0.,  2.,  0.],
       [ 0.,  0.,  0.,  3.],
       [ 0.,  0.,  0.,  0.]])
*)

val eye : ?n:int -> ?k:int -> ?dtype:Np.Dtype.t -> ?format:string -> m:int -> unit -> Py.Object.t
(**
Sparse matrix with ones on diagonal

Returns a sparse (m x n) matrix where the k-th diagonal
is all ones and everything else is zeros.

Parameters
----------
m : int
    Number of rows in the matrix.
n : int, optional
    Number of columns. Default: `m`.
k : int, optional
    Diagonal to place ones on. Default: 0 (main diagonal).
dtype : dtype, optional
    Data type of the matrix.
format : str, optional
    Sparse format of the result, e.g. format='csr', etc.

Examples
--------
>>> from scipy import sparse
>>> sparse.eye(3).toarray()
array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]])
>>> sparse.eye(3, dtype=np.int8)
<3x3 sparse matrix of type '<class 'numpy.int8'>'
    with 3 stored elements (1 diagonals) in DIAgonal format>
*)

val get_index_dtype : ?arrays:Py.Object.t -> ?maxval:float -> ?check_contents:bool -> unit -> Np.Dtype.t
(**
Based on input (integer) arrays `a`, determine a suitable index data
type that can hold the data in the arrays.

Parameters
----------
arrays : tuple of array_like
    Input arrays whose types/contents to check
maxval : float, optional
    Maximum value needed
check_contents : bool, optional
    Whether to check the values in the arrays and not just their types.
    Default: False (check only the types)

Returns
-------
dtype : dtype
    Suitable index data type (int32 or int64)
*)

val get_randint : Py.Object.t -> Py.Object.t
(**
None
*)

val hstack : ?format:string -> ?dtype:Np.Dtype.t -> blocks:Py.Object.t -> unit -> Py.Object.t
(**
Stack sparse matrices horizontally (column wise)

Parameters
----------
blocks
    sequence of sparse matrices with compatible shapes
format : str
    sparse format of the result (e.g. 'csr')
    by default an appropriate sparse matrix format is returned.
    This choice is subject to change.
dtype : dtype, optional
    The data-type of the output matrix.  If not given, the dtype is
    determined from that of `blocks`.

See Also
--------
vstack : stack sparse matrices vertically (row wise)

Examples
--------
>>> from scipy.sparse import coo_matrix, hstack
>>> A = coo_matrix([[1, 2], [3, 4]])
>>> B = coo_matrix([[5], [6]])
>>> hstack([A,B]).toarray()
array([[1, 2, 5],
       [3, 4, 6]])
*)

val identity : ?dtype:Np.Dtype.t -> ?format:string -> n:int -> unit -> Py.Object.t
(**
Identity matrix in sparse format

Returns an identity matrix with shape (n,n) using a given
sparse format and dtype.

Parameters
----------
n : int
    Shape of the identity matrix.
dtype : dtype, optional
    Data type of the matrix
format : str, optional
    Sparse format of the result, e.g. format='csr', etc.

Examples
--------
>>> from scipy.sparse import identity
>>> identity(3).toarray()
array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]])
>>> identity(3, dtype='int8', format='dia')
<3x3 sparse matrix of type '<class 'numpy.int8'>'
        with 3 stored elements (1 diagonals) in DIAgonal format>
*)

val isscalarlike : Py.Object.t -> Py.Object.t
(**
Is x either a scalar, an array scalar, or a 0-dim array?
*)

val issparse : Py.Object.t -> Py.Object.t
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

val kron : ?format:string -> a:Py.Object.t -> b:Py.Object.t -> unit -> Py.Object.t
(**
kronecker product of sparse matrices A and B

Parameters
----------
A : sparse or dense matrix
    first matrix of the product
B : sparse or dense matrix
    second matrix of the product
format : str, optional
    format of the result (e.g. 'csr')

Returns
-------
kronecker product in a sparse matrix format


Examples
--------
>>> from scipy import sparse
>>> A = sparse.csr_matrix(np.array([[0, 2], [5, 0]]))
>>> B = sparse.csr_matrix(np.array([[1, 2], [3, 4]]))
>>> sparse.kron(A, B).toarray()
array([[ 0,  0,  2,  4],
       [ 0,  0,  6,  8],
       [ 5, 10,  0,  0],
       [15, 20,  0,  0]])

>>> sparse.kron(A, [[1, 2], [3, 4]]).toarray()
array([[ 0,  0,  2,  4],
       [ 0,  0,  6,  8],
       [ 5, 10,  0,  0],
       [15, 20,  0,  0]])
*)

val kronsum : ?format:string -> a:Py.Object.t -> b:Py.Object.t -> unit -> Py.Object.t
(**
kronecker sum of sparse matrices A and B

Kronecker sum of two sparse matrices is a sum of two Kronecker
products kron(I_n,A) + kron(B,I_m) where A has shape (m,m)
and B has shape (n,n) and I_m and I_n are identity matrices
of shape (m,m) and (n,n) respectively.

Parameters
----------
A
    square matrix
B
    square matrix
format : str
    format of the result (e.g. 'csr')

Returns
-------
kronecker sum in a sparse matrix format

Examples
--------
*)

val rand : ?density:Py.Object.t -> ?format:string -> ?dtype:Np.Dtype.t -> ?random_state:[`I of int | `Numpy_random_RandomState of Py.Object.t] -> m:Py.Object.t -> n:Py.Object.t -> unit -> [`ArrayLike|`Object|`Spmatrix] Np.Obj.t
(**
Generate a sparse matrix of the given shape and density with uniformly
distributed values.

Parameters
----------
m, n : int
    shape of the matrix
density : real, optional
    density of the generated matrix: density equal to one means a full
    matrix, density of 0 means a matrix with no non-zero items.
format : str, optional
    sparse matrix format.
dtype : dtype, optional
    type of the returned matrix values.
random_state : {numpy.random.RandomState, int}, optional
    Random number generator or random seed. If not given, the singleton
    numpy.random will be used.

Returns
-------
res : sparse matrix

Notes
-----
Only float types are supported for now.

See Also
--------
scipy.sparse.random : Similar function that allows a user-specified random
    data source.

Examples
--------
>>> from scipy.sparse import rand
>>> matrix = rand(3, 4, density=0.25, format='csr', random_state=42)
>>> matrix
<3x4 sparse matrix of type '<class 'numpy.float64'>'
   with 3 stored elements in Compressed Sparse Row format>
>>> matrix.todense()
matrix([[0.05641158, 0.        , 0.        , 0.65088847],
        [0.        , 0.        , 0.        , 0.14286682],
        [0.        , 0.        , 0.        , 0.        ]])
*)

val random : ?density:Py.Object.t -> ?format:string -> ?dtype:Np.Dtype.t -> ?random_state:[`I of int | `Numpy_random_RandomState of Py.Object.t] -> ?data_rvs:Py.Object.t -> m:Py.Object.t -> n:Py.Object.t -> unit -> [`ArrayLike|`Object|`Spmatrix] Np.Obj.t
(**
Generate a sparse matrix of the given shape and density with randomly
distributed values.

Parameters
----------
m, n : int
    shape of the matrix
density : real, optional
    density of the generated matrix: density equal to one means a full
    matrix, density of 0 means a matrix with no non-zero items.
format : str, optional
    sparse matrix format.
dtype : dtype, optional
    type of the returned matrix values.
random_state : {numpy.random.RandomState, int}, optional
    Random number generator or random seed. If not given, the singleton
    numpy.random will be used.  This random state will be used
    for sampling the sparsity structure, but not necessarily for sampling
    the values of the structurally nonzero entries of the matrix.
data_rvs : callable, optional
    Samples a requested number of random values.
    This function should take a single argument specifying the length
    of the ndarray that it will return.  The structurally nonzero entries
    of the sparse random matrix will be taken from the array sampled
    by this function.  By default, uniform [0, 1) random values will be
    sampled using the same random state as is used for sampling
    the sparsity structure.

Returns
-------
res : sparse matrix

Notes
-----
Only float types are supported for now.

Examples
--------
>>> from scipy.sparse import random
>>> from scipy import stats

>>> class CustomRandomState(np.random.RandomState):
...     def randint(self, k):
...         i = np.random.randint(k)
...         return i - i % 2
>>> np.random.seed(12345)
>>> rs = CustomRandomState()
>>> rvs = stats.poisson(25, loc=10).rvs
>>> S = random(3, 4, density=0.25, random_state=rs, data_rvs=rvs)
>>> S.A
array([[ 36.,   0.,  33.,   0.],   # random
       [  0.,   0.,   0.,   0.],
       [  0.,   0.,  36.,   0.]])

>>> from scipy.sparse import random
>>> from scipy.stats import rv_continuous
>>> class CustomDistribution(rv_continuous):
...     def _rvs(self, *args, **kwargs):
...         return self._random_state.randn( *self._size)
>>> X = CustomDistribution(seed=2906)
>>> Y = X()  # get a frozen version of the distribution
>>> S = random(3, 4, density=0.25, random_state=2906, data_rvs=Y.rvs)
>>> S.A
array([[ 0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.13569738,  1.9467163 , -0.81205367,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ]])
*)

val spdiags : ?format:string -> data:[>`Ndarray] Np.Obj.t -> diags:Py.Object.t -> m:Py.Object.t -> n:Py.Object.t -> unit -> Py.Object.t
(**
Return a sparse matrix from diagonals.

Parameters
----------
data : array_like
    matrix diagonals stored row-wise
diags : diagonals to set
    - k = 0  the main diagonal
    - k > 0  the k-th upper diagonal
    - k < 0  the k-th lower diagonal
m, n : int
    shape of the result
format : str, optional
    Format of the result. By default (format=None) an appropriate sparse
    matrix format is returned.  This choice is subject to change.

See Also
--------
diags : more convenient form of this function
dia_matrix : the sparse DIAgonal format.

Examples
--------
>>> from scipy.sparse import spdiags
>>> data = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
>>> diags = np.array([0, -1, 2])
>>> spdiags(data, diags, 4, 4).toarray()
array([[1, 0, 3, 0],
       [1, 2, 0, 4],
       [0, 2, 3, 0],
       [0, 0, 3, 4]])
*)

val upcast : Py.Object.t list -> Py.Object.t
(**
Returns the nearest supported sparse dtype for the
combination of one or more types.

upcast(t0, t1, ..., tn) -> T  where T is a supported dtype

Examples
--------

>>> upcast('int32')
<type 'numpy.int32'>
>>> upcast('bool')
<type 'numpy.bool_'>
>>> upcast('int32','float32')
<type 'numpy.float64'>
>>> upcast('bool',complex,float)
<type 'numpy.complex128'>
*)

val vstack : ?format:string -> ?dtype:Np.Dtype.t -> blocks:Py.Object.t -> unit -> Py.Object.t
(**
Stack sparse matrices vertically (row wise)

Parameters
----------
blocks
    sequence of sparse matrices with compatible shapes
format : str, optional
    sparse format of the result (e.g. 'csr')
    by default an appropriate sparse matrix format is returned.
    This choice is subject to change.
dtype : dtype, optional
    The data-type of the output matrix.  If not given, the dtype is
    determined from that of `blocks`.

See Also
--------
hstack : stack sparse matrices horizontally (column wise)

Examples
--------
>>> from scipy.sparse import coo_matrix, vstack
>>> A = coo_matrix([[1, 2], [3, 4]])
>>> B = coo_matrix([[5, 6]])
>>> vstack([A, B]).toarray()
array([[1, 2],
       [3, 4],
       [5, 6]])
*)


end

module Coo : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module Izip : sig
type tag = [`Zip]
type t = [`Object | `Zip] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : Py.Object.t list -> t
(**
zip( *iterables) --> zip object

Return a zip object whose .__next__() method returns a tuple where
the i-th element comes from the i-th iterable argument.  The .__next__()
method continues until the shortest iterable in the argument sequence
is exhausted and then it raises StopIteration.
*)

val __iter__ : [> tag] Obj.t -> Py.Object.t
(**
Implement iter(self).
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val check_reshape_kwargs : Py.Object.t -> Py.Object.t
(**
Unpack keyword arguments for reshape function.

This is useful because keyword arguments after star arguments are not
allowed in Python 2, but star keyword arguments are. This function unpacks
'order' and 'copy' from the star keyword arguments (with defaults) and
throws an error for any remaining.
*)

val check_shape : ?current_shape:Py.Object.t -> args:Py.Object.t -> unit -> Py.Object.t
(**
Imitate numpy.matrix handling of shape arguments
*)

val downcast_intp_index : Py.Object.t -> Py.Object.t
(**
Down-cast index array to np.intp dtype if it is of a larger dtype.

Raise an error if the array contains a value that is too large for
intp.
*)

val get_index_dtype : ?arrays:Py.Object.t -> ?maxval:float -> ?check_contents:bool -> unit -> Np.Dtype.t
(**
Based on input (integer) arrays `a`, determine a suitable index data
type that can hold the data in the arrays.

Parameters
----------
arrays : tuple of array_like
    Input arrays whose types/contents to check
maxval : float, optional
    Maximum value needed
check_contents : bool, optional
    Whether to check the values in the arrays and not just their types.
    Default: False (check only the types)

Returns
-------
dtype : dtype
    Suitable index data type (int32 or int64)
*)

val getdtype : ?a:Py.Object.t -> ?default:Py.Object.t -> dtype:Py.Object.t -> unit -> Py.Object.t
(**
Function used to simplify argument processing.  If 'dtype' is not
specified (is None), returns a.dtype; otherwise returns a np.dtype
object created from the specified dtype argument.  If 'dtype' and 'a'
are both None, construct a data type out of the 'default' parameter.
Furthermore, 'dtype' must be in 'allowed' set.
*)

val isshape : ?nonneg:Py.Object.t -> x:Py.Object.t -> unit -> Py.Object.t
(**
Is x a valid 2-tuple of dimensions?

If nonneg, also checks that the dimensions are non-negative.
*)

val isspmatrix : Py.Object.t -> Py.Object.t
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

val isspmatrix_coo : Py.Object.t -> Py.Object.t
(**
Is x of coo_matrix type?

Parameters
----------
x
    object to check for being a coo matrix

Returns
-------
bool
    True if x is a coo matrix, False otherwise

Examples
--------
>>> from scipy.sparse import coo_matrix, isspmatrix_coo
>>> isspmatrix_coo(coo_matrix([[5]]))
True

>>> from scipy.sparse import coo_matrix, csr_matrix, isspmatrix_coo
>>> isspmatrix_coo(csr_matrix([[5]]))
False
*)

val matrix : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> Py.Object.t
(**
None
*)

val to_native : Py.Object.t -> Py.Object.t
(**
None
*)

val upcast : Py.Object.t list -> Py.Object.t
(**
Returns the nearest supported sparse dtype for the
combination of one or more types.

upcast(t0, t1, ..., tn) -> T  where T is a supported dtype

Examples
--------

>>> upcast('int32')
<type 'numpy.int32'>
>>> upcast('bool')
<type 'numpy.bool_'>
>>> upcast('int32','float32')
<type 'numpy.float64'>
>>> upcast('bool',complex,float)
<type 'numpy.complex128'>
*)

val upcast_char : Py.Object.t list -> Py.Object.t
(**
Same as `upcast` but taking dtype.char as input (faster).
*)

val warn : ?category:Py.Object.t -> ?stacklevel:Py.Object.t -> ?source:Py.Object.t -> message:Py.Object.t -> unit -> Py.Object.t
(**
Issue a warning, or maybe ignore it or raise an exception.
*)


end

module Csc : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val get_index_dtype : ?arrays:Py.Object.t -> ?maxval:float -> ?check_contents:bool -> unit -> Np.Dtype.t
(**
Based on input (integer) arrays `a`, determine a suitable index data
type that can hold the data in the arrays.

Parameters
----------
arrays : tuple of array_like
    Input arrays whose types/contents to check
maxval : float, optional
    Maximum value needed
check_contents : bool, optional
    Whether to check the values in the arrays and not just their types.
    Default: False (check only the types)

Returns
-------
dtype : dtype
    Suitable index data type (int32 or int64)
*)

val isspmatrix_csc : Py.Object.t -> Py.Object.t
(**
Is x of csc_matrix type?

Parameters
----------
x
    object to check for being a csc matrix

Returns
-------
bool
    True if x is a csc matrix, False otherwise

Examples
--------
>>> from scipy.sparse import csc_matrix, isspmatrix_csc
>>> isspmatrix_csc(csc_matrix([[5]]))
True

>>> from scipy.sparse import csc_matrix, csr_matrix, isspmatrix_csc
>>> isspmatrix_csc(csr_matrix([[5]]))
False
*)

val upcast : Py.Object.t list -> Py.Object.t
(**
Returns the nearest supported sparse dtype for the
combination of one or more types.

upcast(t0, t1, ..., tn) -> T  where T is a supported dtype

Examples
--------

>>> upcast('int32')
<type 'numpy.int32'>
>>> upcast('bool')
<type 'numpy.bool_'>
>>> upcast('int32','float32')
<type 'numpy.float64'>
>>> upcast('bool',complex,float)
<type 'numpy.complex128'>
*)


end

module Csgraph : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module NegativeCycleError : sig
type tag = [`NegativeCycleError]
type t = [`BaseException | `NegativeCycleError | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_exception : t -> [`BaseException] Obj.t
val create : ?message:Py.Object.t -> unit -> t
(**
Common base class for all non-exit exceptions.
*)

val with_traceback : tb:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Exception.with_traceback(tb) --
set self.__traceback__ to tb and return self.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val bellman_ford : ?directed:bool -> ?indices:[`Ndarray of [>`Ndarray] Np.Obj.t | `I of int] -> ?return_predecessors:bool -> ?unweighted:bool -> csgraph:Py.Object.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
bellman_ford(csgraph, directed=True, indices=None, return_predecessors=False,
             unweighted=False)

Compute the shortest path lengths using the Bellman-Ford algorithm.

The Bellman-Ford algorithm can robustly deal with graphs with negative
weights.  If a negative cycle is detected, an error is raised.  For
graphs without negative edge weights, Dijkstra's algorithm may be faster.

.. versionadded:: 0.11.0

Parameters
----------
csgraph : array, matrix, or sparse matrix, 2 dimensions
    The N x N array of distances representing the input graph.
directed : bool, optional
    If True (default), then find the shortest path on a directed graph:
    only move from point i to point j along paths csgraph[i, j].
    If False, then find the shortest path on an undirected graph: the
    algorithm can progress from point i to j along csgraph[i, j] or
    csgraph[j, i]
indices : array_like or int, optional
    if specified, only compute the paths from the points at the given
    indices.
return_predecessors : bool, optional
    If True, return the size (N, N) predecesor matrix
unweighted : bool, optional
    If True, then find unweighted distances.  That is, rather than finding
    the path between each point such that the sum of weights is minimized,
    find the path such that the number of edges is minimized.

Returns
-------
dist_matrix : ndarray
    The N x N matrix of distances between graph nodes. dist_matrix[i,j]
    gives the shortest distance from point i to point j along the graph.

predecessors : ndarray
    Returned only if return_predecessors == True.
    The N x N matrix of predecessors, which can be used to reconstruct
    the shortest paths.  Row i of the predecessor matrix contains
    information on the shortest paths from point i: each entry
    predecessors[i, j] gives the index of the previous node in the
    path from point i to point j.  If no path exists between point
    i and j, then predecessors[i, j] = -9999

Raises
------
NegativeCycleError:
    if there are negative cycles in the graph

Notes
-----
This routine is specially designed for graphs with negative edge weights.
If all edge weights are positive, then Dijkstra's algorithm is a better
choice.

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> from scipy.sparse.csgraph import bellman_ford

>>> graph = [
... [0, 1 ,2, 0],
... [0, 0, 0, 1],
... [2, 0, 0, 3],
... [0, 0, 0, 0]
... ]
>>> graph = csr_matrix(graph)
>>> print(graph)
  (0, 1)    1
  (0, 2)    2
  (1, 3)    1
  (2, 0)    2
  (2, 3)    3

>>> dist_matrix, predecessors = bellman_ford(csgraph=graph, directed=False, indices=0, return_predecessors=True)
>>> dist_matrix
array([ 0.,  1.,  2.,  2.])
>>> predecessors
array([-9999,     0,     0,     1], dtype=int32)
*)

val breadth_first_order : ?directed:bool -> ?return_predecessors:bool -> csgraph:[>`ArrayLike] Np.Obj.t -> i_start:int -> unit -> (Py.Object.t * Py.Object.t)
(**
breadth_first_order(csgraph, i_start, directed=True, return_predecessors=True)

Return a breadth-first ordering starting with specified node.

Note that a breadth-first order is not unique, but the tree which it
generates is unique.

.. versionadded:: 0.11.0

Parameters
----------
csgraph : array_like or sparse matrix
    The N x N compressed sparse graph.  The input csgraph will be
    converted to csr format for the calculation.
i_start : int
    The index of starting node.
directed : bool, optional
    If True (default), then operate on a directed graph: only
    move from point i to point j along paths csgraph[i, j].
    If False, then find the shortest path on an undirected graph: the
    algorithm can progress from point i to j along csgraph[i, j] or
    csgraph[j, i].
return_predecessors : bool, optional
    If True (default), then return the predecesor array (see below).

Returns
-------
node_array : ndarray, one dimension
    The breadth-first list of nodes, starting with specified node.  The
    length of node_array is the number of nodes reachable from the
    specified node.
predecessors : ndarray, one dimension
    Returned only if return_predecessors is True.
    The length-N list of predecessors of each node in a breadth-first
    tree.  If node i is in the tree, then its parent is given by
    predecessors[i]. If node i is not in the tree (and for the parent
    node) then predecessors[i] = -9999.

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> from scipy.sparse.csgraph import breadth_first_order

>>> graph = [
... [0, 1 , 2, 0],
... [0, 0, 0, 1],
... [2, 0, 0, 3],
... [0, 0, 0, 0]
... ]
>>> graph = csr_matrix(graph)
>>> print(graph)
  (0, 1)    1
  (0, 2)    2
  (1, 3)    1
  (2, 0)    2
  (2, 3)    3

>>> breadth_first_order(graph,0)
(array([0, 1, 2, 3], dtype=int32), array([-9999,     0,     0,     1], dtype=int32))
*)

val breadth_first_tree : ?directed:bool -> csgraph:[>`ArrayLike] Np.Obj.t -> i_start:int -> unit -> Py.Object.t
(**
breadth_first_tree(csgraph, i_start, directed=True)

Return the tree generated by a breadth-first search

Note that a breadth-first tree from a specified node is unique.

.. versionadded:: 0.11.0

Parameters
----------
csgraph : array_like or sparse matrix
    The N x N matrix representing the compressed sparse graph.  The input
    csgraph will be converted to csr format for the calculation.
i_start : int
    The index of starting node.
directed : bool, optional
    If True (default), then operate on a directed graph: only
    move from point i to point j along paths csgraph[i, j].
    If False, then find the shortest path on an undirected graph: the
    algorithm can progress from point i to j along csgraph[i, j] or
    csgraph[j, i].

Returns
-------
cstree : csr matrix
    The N x N directed compressed-sparse representation of the breadth-
    first tree drawn from csgraph, starting at the specified node.

Examples
--------
The following example shows the computation of a depth-first tree
over a simple four-component graph, starting at node 0::

     input graph          breadth first tree from (0)

         (0)                         (0)
        /   \                       /   \
       3     8                     3     8
      /       \                   /       \
    (3)---5---(1)               (3)       (1)
      \       /                           /
       6     2                           2
        \   /                           /
         (2)                         (2)

In compressed sparse representation, the solution looks like this:

>>> from scipy.sparse import csr_matrix
>>> from scipy.sparse.csgraph import breadth_first_tree
>>> X = csr_matrix([[0, 8, 0, 3],
...                 [0, 0, 2, 5],
...                 [0, 0, 0, 6],
...                 [0, 0, 0, 0]])
>>> Tcsr = breadth_first_tree(X, 0, directed=False)
>>> Tcsr.toarray().astype(int)
array([[0, 8, 0, 3],
       [0, 0, 2, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]])

Note that the resulting graph is a Directed Acyclic Graph which spans
the graph.  A breadth-first tree from a given node is unique.
*)

val connected_components : ?directed:bool -> ?connection:string -> ?return_labels:bool -> csgraph:[>`ArrayLike] Np.Obj.t -> unit -> (int * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
connected_components(csgraph, directed=True, connection='weak',
                     return_labels=True)

Analyze the connected components of a sparse graph

.. versionadded:: 0.11.0

Parameters
----------
csgraph : array_like or sparse matrix
    The N x N matrix representing the compressed sparse graph.  The input
    csgraph will be converted to csr format for the calculation.
directed : bool, optional
    If True (default), then operate on a directed graph: only
    move from point i to point j along paths csgraph[i, j].
    If False, then find the shortest path on an undirected graph: the
    algorithm can progress from point i to j along csgraph[i, j] or
    csgraph[j, i].
connection : str, optional
    ['weak'|'strong'].  For directed graphs, the type of connection to
    use.  Nodes i and j are strongly connected if a path exists both
    from i to j and from j to i. A directed graph is weakly connected
    if replacing all of its directed edges with undirected edges produces
    a connected (undirected) graph. If directed == False, this keyword
    is not referenced.
return_labels : bool, optional
    If True (default), then return the labels for each of the connected
    components.

Returns
-------
n_components: int
    The number of connected components.
labels: ndarray
    The length-N array of labels of the connected components.

References
----------
.. [1] D. J. Pearce, 'An Improved Algorithm for Finding the Strongly
       Connected Components of a Directed Graph', Technical Report, 2005

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> from scipy.sparse.csgraph import connected_components

>>> graph = [
... [ 0, 1 , 1, 0 , 0 ],
... [ 0, 0 , 1 , 0 ,0 ],
... [ 0, 0, 0, 0, 0],
... [0, 0 , 0, 0, 1],
... [0, 0, 0, 0, 0]
... ]
>>> graph = csr_matrix(graph)
>>> print(graph)
  (0, 1)    1
  (0, 2)    1
  (1, 2)    1
  (3, 4)    1

>>> n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
>>> n_components
2
>>> labels
array([0, 0, 0, 1, 1], dtype=int32)
*)

val csgraph_from_dense : ?null_value:[`F of float | `None] -> ?nan_null:bool -> ?infinity_null:bool -> graph:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
csgraph_from_dense(graph, null_value=0, nan_null=True, infinity_null=True)

Construct a CSR-format sparse graph from a dense matrix.

.. versionadded:: 0.11.0

Parameters
----------
graph : array_like
    Input graph.  Shape should be (n_nodes, n_nodes).
null_value : float or None (optional)
    Value that denotes non-edges in the graph.  Default is zero.
infinity_null : bool
    If True (default), then infinite entries (both positive and negative)
    are treated as null edges.
nan_null : bool
    If True (default), then NaN entries are treated as non-edges

Returns
-------
csgraph : csr_matrix
    Compressed sparse representation of graph,

Examples
--------
>>> from scipy.sparse.csgraph import csgraph_from_dense

>>> graph = [
... [0, 1 , 2, 0],
... [0, 0, 0, 1],
... [0, 0, 0, 3],
... [0, 0, 0, 0]
... ]

>>> csgraph_from_dense(graph)
<4x4 sparse matrix of type '<class 'numpy.float64'>'
    with 4 stored elements in Compressed Sparse Row format>
*)

val csgraph_from_masked : Py.Object.t -> Py.Object.t
(**
csgraph_from_masked(graph)

Construct a CSR-format graph from a masked array.

.. versionadded:: 0.11.0

Parameters
----------
graph : MaskedArray
    Input graph.  Shape should be (n_nodes, n_nodes).

Returns
-------
csgraph : csr_matrix
    Compressed sparse representation of graph,

Examples
--------
>>> import numpy as np
>>> from scipy.sparse.csgraph import csgraph_from_masked

>>> graph_masked = np.ma.masked_array(data =[
... [0, 1 , 2, 0],
... [0, 0, 0, 1],
... [0, 0, 0, 3],
... [0, 0, 0, 0]
...  ],
... mask=[[ True, False, False , True],
... [ True,  True , True, False],
... [ True , True,  True ,False],
... [ True ,True , True , True]],
... fill_value = 0)

>>> csgraph_from_masked(graph_masked)
<4x4 sparse matrix of type '<class 'numpy.float64'>'
    with 4 stored elements in Compressed Sparse Row format>
*)

val csgraph_masked_from_dense : ?null_value:[`F of float | `None] -> ?nan_null:bool -> ?infinity_null:bool -> ?copy:Py.Object.t -> graph:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
csgraph_masked_from_dense(graph, null_value=0, nan_null=True,
                          infinity_null=True, copy=True)

Construct a masked array graph representation from a dense matrix.

.. versionadded:: 0.11.0

Parameters
----------
graph : array_like
    Input graph.  Shape should be (n_nodes, n_nodes).
null_value : float or None (optional)
    Value that denotes non-edges in the graph.  Default is zero.
infinity_null : bool
    If True (default), then infinite entries (both positive and negative)
    are treated as null edges.
nan_null : bool
    If True (default), then NaN entries are treated as non-edges

Returns
-------
csgraph : MaskedArray
    masked array representation of graph

Examples
--------
>>> from scipy.sparse.csgraph import csgraph_masked_from_dense

>>> graph = [
... [0, 1 , 2, 0],
... [0, 0, 0, 1],
... [0, 0, 0, 3],
... [0, 0, 0, 0]
... ]

>>> csgraph_masked_from_dense(graph)
masked_array(
  data=[[--, 1, 2, --],
        [--, --, --, 1],
        [--, --, --, 3],
        [--, --, --, --]],
  mask=[[ True, False, False,  True],
        [ True,  True,  True, False],
        [ True,  True,  True, False],
        [ True,  True,  True,  True]],
  fill_value=0)
*)

val csgraph_to_dense : ?null_value:float -> csgraph:Py.Object.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
csgraph_to_dense(csgraph, null_value=0)

Convert a sparse graph representation to a dense representation

.. versionadded:: 0.11.0

Parameters
----------
csgraph : csr_matrix, csc_matrix, or lil_matrix
    Sparse representation of a graph.
null_value : float, optional
    The value used to indicate null edges in the dense representation.
    Default is 0.

Returns
-------
graph : ndarray
    The dense representation of the sparse graph.

Notes
-----
For normal sparse graph representations, calling csgraph_to_dense with
null_value=0 produces an equivalent result to using dense format
conversions in the main sparse package.  When the sparse representations
have repeated values, however, the results will differ.  The tools in
scipy.sparse will add repeating values to obtain a final value.  This
function will select the minimum among repeating values to obtain a
final value.  For example, here we'll create a two-node directed sparse
graph with multiple edges from node 0 to node 1, of weights 2 and 3.
This illustrates the difference in behavior:

>>> from scipy.sparse import csr_matrix, csgraph
>>> data = np.array([2, 3])
>>> indices = np.array([1, 1])
>>> indptr = np.array([0, 2, 2])
>>> M = csr_matrix((data, indices, indptr), shape=(2, 2))
>>> M.toarray()
array([[0, 5],
       [0, 0]])
>>> csgraph.csgraph_to_dense(M)
array([[0., 2.],
       [0., 0.]])

The reason for this difference is to allow a compressed sparse graph to
represent multiple edges between any two nodes.  As most sparse graph
algorithms are concerned with the single lowest-cost edge between any
two nodes, the default scipy.sparse behavior of summming multiple weights
does not make sense in this context.

The other reason for using this routine is to allow for graphs with
zero-weight edges.  Let's look at the example of a two-node directed
graph, connected by an edge of weight zero:

>>> from scipy.sparse import csr_matrix, csgraph
>>> data = np.array([0.0])
>>> indices = np.array([1])
>>> indptr = np.array([0, 1, 1])
>>> M = csr_matrix((data, indices, indptr), shape=(2, 2))
>>> M.toarray()
array([[0, 0],
       [0, 0]])
>>> csgraph.csgraph_to_dense(M, np.inf)
array([[ inf,   0.],
       [ inf,  inf]])

In the first case, the zero-weight edge gets lost in the dense
representation.  In the second case, we can choose a different null value
and see the true form of the graph.

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> from scipy.sparse.csgraph import csgraph_to_dense

>>> graph = csr_matrix( [
... [0, 1 , 2, 0],
... [0, 0, 0, 1],
... [0, 0, 0, 3],
... [0, 0, 0, 0]
... ])
>>> graph
<4x4 sparse matrix of type '<class 'numpy.int64'>'
    with 4 stored elements in Compressed Sparse Row format>

>>> csgraph_to_dense(graph)
array([[ 0.,  1.,  2.,  0.],
       [ 0.,  0.,  0.,  1.],
       [ 0.,  0.,  0.,  3.],
       [ 0.,  0.,  0.,  0.]])
*)

val csgraph_to_masked : Py.Object.t -> Py.Object.t
(**
csgraph_to_masked(csgraph)

Convert a sparse graph representation to a masked array representation

.. versionadded:: 0.11.0

Parameters
----------
csgraph : csr_matrix, csc_matrix, or lil_matrix
    Sparse representation of a graph.

Returns
-------
graph : MaskedArray
    The masked dense representation of the sparse graph.

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> from scipy.sparse.csgraph import csgraph_to_masked

>>> graph = csr_matrix( [
... [0, 1 , 2, 0],
... [0, 0, 0, 1],
... [0, 0, 0, 3],
... [0, 0, 0, 0]
... ])
>>> graph
<4x4 sparse matrix of type '<class 'numpy.int64'>'
    with 4 stored elements in Compressed Sparse Row format>

>>> csgraph_to_masked(graph)
masked_array(
  data=[[--, 1.0, 2.0, --],
        [--, --, --, 1.0],
        [--, --, --, 3.0],
        [--, --, --, --]],
  mask=[[ True, False, False,  True],
        [ True,  True,  True, False],
        [ True,  True,  True, False],
        [ True,  True,  True,  True]],
  fill_value=1e+20)
*)

val depth_first_order : ?directed:bool -> ?return_predecessors:bool -> csgraph:[>`ArrayLike] Np.Obj.t -> i_start:int -> unit -> (Py.Object.t * Py.Object.t)
(**
depth_first_order(csgraph, i_start, directed=True, return_predecessors=True)

Return a depth-first ordering starting with specified node.

Note that a depth-first order is not unique.  Furthermore, for graphs
with cycles, the tree generated by a depth-first search is not
unique either.

.. versionadded:: 0.11.0

Parameters
----------
csgraph : array_like or sparse matrix
    The N x N compressed sparse graph.  The input csgraph will be
    converted to csr format for the calculation.
i_start : int
    The index of starting node.
directed : bool, optional
    If True (default), then operate on a directed graph: only
    move from point i to point j along paths csgraph[i, j].
    If False, then find the shortest path on an undirected graph: the
    algorithm can progress from point i to j along csgraph[i, j] or
    csgraph[j, i].
return_predecessors : bool, optional
    If True (default), then return the predecesor array (see below).

Returns
-------
node_array : ndarray, one dimension
    The depth-first list of nodes, starting with specified node.  The
    length of node_array is the number of nodes reachable from the
    specified node.
predecessors : ndarray, one dimension
    Returned only if return_predecessors is True.
    The length-N list of predecessors of each node in a depth-first
    tree.  If node i is in the tree, then its parent is given by
    predecessors[i]. If node i is not in the tree (and for the parent
    node) then predecessors[i] = -9999.

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> from scipy.sparse.csgraph import depth_first_order

>>> graph = [
... [0, 1 , 2, 0],
... [0, 0, 0, 1],
... [2, 0, 0, 3],
... [0, 0, 0, 0]
... ]
>>> graph = csr_matrix(graph)
>>> print(graph)
  (0, 1)    1
  (0, 2)    2
  (1, 3)    1
  (2, 0)    2
  (2, 3)    3

>>> depth_first_order(graph,0)
(array([0, 1, 3, 2], dtype=int32), array([-9999,     0,     0,     1], dtype=int32))
*)

val depth_first_tree : ?directed:bool -> csgraph:[>`ArrayLike] Np.Obj.t -> i_start:int -> unit -> Py.Object.t
(**
depth_first_tree(csgraph, i_start, directed=True)

Return a tree generated by a depth-first search.

Note that a tree generated by a depth-first search is not unique:
it depends on the order that the children of each node are searched.

.. versionadded:: 0.11.0

Parameters
----------
csgraph : array_like or sparse matrix
    The N x N matrix representing the compressed sparse graph.  The input
    csgraph will be converted to csr format for the calculation.
i_start : int
    The index of starting node.
directed : bool, optional
    If True (default), then operate on a directed graph: only
    move from point i to point j along paths csgraph[i, j].
    If False, then find the shortest path on an undirected graph: the
    algorithm can progress from point i to j along csgraph[i, j] or
    csgraph[j, i].

Returns
-------
cstree : csr matrix
    The N x N directed compressed-sparse representation of the depth-
    first tree drawn from csgraph, starting at the specified node.

Examples
--------
The following example shows the computation of a depth-first tree
over a simple four-component graph, starting at node 0::

     input graph           depth first tree from (0)

         (0)                         (0)
        /   \                           \
       3     8                           8
      /       \                           \
    (3)---5---(1)               (3)       (1)
      \       /                   \       /
       6     2                     6     2
        \   /                       \   /
         (2)                         (2)

In compressed sparse representation, the solution looks like this:

>>> from scipy.sparse import csr_matrix
>>> from scipy.sparse.csgraph import depth_first_tree
>>> X = csr_matrix([[0, 8, 0, 3],
...                 [0, 0, 2, 5],
...                 [0, 0, 0, 6],
...                 [0, 0, 0, 0]])
>>> Tcsr = depth_first_tree(X, 0, directed=False)
>>> Tcsr.toarray().astype(int)
array([[0, 8, 0, 0],
       [0, 0, 2, 0],
       [0, 0, 0, 6],
       [0, 0, 0, 0]])

Note that the resulting graph is a Directed Acyclic Graph which spans
the graph.  Unlike a breadth-first tree, a depth-first tree of a given
graph is not unique if the graph contains cycles.  If the above solution
had begun with the edge connecting nodes 0 and 3, the result would have
been different.
*)

val floyd_warshall : ?directed:bool -> ?return_predecessors:bool -> ?unweighted:bool -> ?overwrite:bool -> csgraph:Py.Object.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
floyd_warshall(csgraph, directed=True, return_predecessors=False,
               unweighted=False, overwrite=False)

Compute the shortest path lengths using the Floyd-Warshall algorithm

.. versionadded:: 0.11.0

Parameters
----------
csgraph : array, matrix, or sparse matrix, 2 dimensions
    The N x N array of distances representing the input graph.
directed : bool, optional
    If True (default), then find the shortest path on a directed graph:
    only move from point i to point j along paths csgraph[i, j].
    If False, then find the shortest path on an undirected graph: the
    algorithm can progress from point i to j along csgraph[i, j] or
    csgraph[j, i]
return_predecessors : bool, optional
    If True, return the size (N, N) predecesor matrix
unweighted : bool, optional
    If True, then find unweighted distances.  That is, rather than finding
    the path between each point such that the sum of weights is minimized,
    find the path such that the number of edges is minimized.
overwrite : bool, optional
    If True, overwrite csgraph with the result.  This applies only if
    csgraph is a dense, c-ordered array with dtype=float64.

Returns
-------
dist_matrix : ndarray
    The N x N matrix of distances between graph nodes. dist_matrix[i,j]
    gives the shortest distance from point i to point j along the graph.

predecessors : ndarray
    Returned only if return_predecessors == True.
    The N x N matrix of predecessors, which can be used to reconstruct
    the shortest paths.  Row i of the predecessor matrix contains
    information on the shortest paths from point i: each entry
    predecessors[i, j] gives the index of the previous node in the
    path from point i to point j.  If no path exists between point
    i and j, then predecessors[i, j] = -9999

Raises
------
NegativeCycleError:
    if there are negative cycles in the graph

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> from scipy.sparse.csgraph import floyd_warshall

>>> graph = [
... [0, 1 , 2, 0],
... [0, 0, 0, 1],
... [2, 0, 0, 3],
... [0, 0, 0, 0]
... ]
>>> graph = csr_matrix(graph)
>>> print(graph)
  (0, 1)    1
  (0, 2)    2
  (1, 3)    1
  (2, 0)    2
  (2, 3)    3


>>> dist_matrix, predecessors = floyd_warshall(csgraph=graph, directed=False, return_predecessors=True)
>>> dist_matrix
array([[ 0.,  1.,  2.,  2.],
       [ 1.,  0.,  3.,  1.],
       [ 2.,  3.,  0.,  3.],
       [ 2.,  1.,  3.,  0.]])
>>> predecessors
array([[-9999,     0,     0,     1],
       [    1, -9999,     0,     1],
       [    2,     0, -9999,     2],
       [    1,     3,     3, -9999]], dtype=int32)
*)

val johnson : ?directed:bool -> ?indices:[`Ndarray of [>`Ndarray] Np.Obj.t | `I of int] -> ?return_predecessors:bool -> ?unweighted:bool -> csgraph:Py.Object.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
johnson(csgraph, directed=True, indices=None, return_predecessors=False,
        unweighted=False)

Compute the shortest path lengths using Johnson's algorithm.

Johnson's algorithm combines the Bellman-Ford algorithm and Dijkstra's
algorithm to quickly find shortest paths in a way that is robust to
the presence of negative cycles.  If a negative cycle is detected,
an error is raised.  For graphs without negative edge weights,
dijkstra may be faster.

.. versionadded:: 0.11.0

Parameters
----------
csgraph : array, matrix, or sparse matrix, 2 dimensions
    The N x N array of distances representing the input graph.
directed : bool, optional
    If True (default), then find the shortest path on a directed graph:
    only move from point i to point j along paths csgraph[i, j].
    If False, then find the shortest path on an undirected graph: the
    algorithm can progress from point i to j along csgraph[i, j] or
    csgraph[j, i]
indices : array_like or int, optional
    if specified, only compute the paths from the points at the given
    indices.
return_predecessors : bool, optional
    If True, return the size (N, N) predecesor matrix
unweighted : bool, optional
    If True, then find unweighted distances.  That is, rather than finding
    the path between each point such that the sum of weights is minimized,
    find the path such that the number of edges is minimized.

Returns
-------
dist_matrix : ndarray
    The N x N matrix of distances between graph nodes. dist_matrix[i,j]
    gives the shortest distance from point i to point j along the graph.

predecessors : ndarray
    Returned only if return_predecessors == True.
    The N x N matrix of predecessors, which can be used to reconstruct
    the shortest paths.  Row i of the predecessor matrix contains
    information on the shortest paths from point i: each entry
    predecessors[i, j] gives the index of the previous node in the
    path from point i to point j.  If no path exists between point
    i and j, then predecessors[i, j] = -9999

Raises
------
NegativeCycleError:
    if there are negative cycles in the graph

Notes
-----
This routine is specially designed for graphs with negative edge weights.
If all edge weights are positive, then Dijkstra's algorithm is a better
choice.

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> from scipy.sparse.csgraph import johnson

>>> graph = [
... [0, 1, 2, 0],
... [0, 0, 0, 1],
... [2, 0, 0, 3],
... [0, 0, 0, 0]
... ]
>>> graph = csr_matrix(graph)
>>> print(graph)
  (0, 1)    1
  (0, 2)    2
  (1, 3)    1
  (2, 0)    2
  (2, 3)    3

>>> dist_matrix, predecessors = johnson(csgraph=graph, directed=False, indices=0, return_predecessors=True)
>>> dist_matrix
array([ 0.,  1.,  2.,  2.])
>>> predecessors
array([-9999,     0,     0,     1], dtype=int32)
*)

val laplacian : ?normed:bool -> ?return_diag:bool -> ?use_out_degree:bool -> csgraph:Py.Object.t -> unit -> ([>`ArrayLike] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Return the Laplacian matrix of a directed graph.

Parameters
----------
csgraph : array_like or sparse matrix, 2 dimensions
    compressed-sparse graph, with shape (N, N).
normed : bool, optional
    If True, then compute symmetric normalized Laplacian.
return_diag : bool, optional
    If True, then also return an array related to vertex degrees.
use_out_degree : bool, optional
    If True, then use out-degree instead of in-degree.
    This distinction matters only if the graph is asymmetric.
    Default: False.

Returns
-------
lap : ndarray or sparse matrix
    The N x N laplacian matrix of csgraph. It will be a numpy array (dense)
    if the input was dense, or a sparse matrix otherwise.
diag : ndarray, optional
    The length-N diagonal of the Laplacian matrix.
    For the normalized Laplacian, this is the array of square roots
    of vertex degrees or 1 if the degree is zero.

Notes
-----
The Laplacian matrix of a graph is sometimes referred to as the
'Kirchoff matrix' or the 'admittance matrix', and is useful in many
parts of spectral graph theory.  In particular, the eigen-decomposition
of the laplacian matrix can give insight into many properties of the graph.

Examples
--------
>>> from scipy.sparse import csgraph
>>> G = np.arange(5) * np.arange(5)[:, np.newaxis]
>>> G
array([[ 0,  0,  0,  0,  0],
       [ 0,  1,  2,  3,  4],
       [ 0,  2,  4,  6,  8],
       [ 0,  3,  6,  9, 12],
       [ 0,  4,  8, 12, 16]])
>>> csgraph.laplacian(G, normed=False)
array([[  0,   0,   0,   0,   0],
       [  0,   9,  -2,  -3,  -4],
       [  0,  -2,  16,  -6,  -8],
       [  0,  -3,  -6,  21, -12],
       [  0,  -4,  -8, -12,  24]])
*)

val maximum_bipartite_matching : ?perm_type:[`Row | `Column] -> graph:[>`Spmatrix] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
maximum_bipartite_matching(graph, perm_type='row')

Returns a matching of a bipartite graph whose cardinality is as least that
of any given matching of the graph.

Parameters
----------
graph : sparse matrix
    Input sparse in CSR format whose rows represent one partition of the
    graph and whose columns represent the other partition. An edge between
    two vertices is indicated by the corresponding entry in the matrix
    existing in its sparse representation.
perm_type : str, {'row', 'column'}
    Which partition to return the matching in terms of: If ``'row'``, the
    function produces an array whose length is the number of columns in the
    input, and whose :math:`j`'th element is the row matched to the
    :math:`j`'th column. Conversely, if ``perm_type`` is ``'column'``, this
    returns the columns matched to each row.

Returns
-------
perm : ndarray
    A matching of the vertices in one of the two partitions. Unmatched
    vertices are represented by a ``-1`` in the result.

Notes
-----
This function implements the Hopcroft--Karp algorithm [1]_. Its time
complexity is :math:`O(\lvert E \rvert \sqrt{\lvert V \rvert})`, and its
space complexity is linear in the number of rows. In practice, this
asymmetry between rows and columns means that it can be more efficient to
transpose the input if it contains more columns than rows.

By Konig's theorem, the cardinality of the matching is also the number of
vertices appearing in a minimum vertex cover of the graph.

Note that if the sparse representation contains explicit zeros, these are
still counted as edges.

The implementation was changed in SciPy 1.4.0 to allow matching of general
bipartite graphs, where previous versions would assume that a perfect
matching existed. As such, code written against 1.4.0 will not necessarily
work on older versions.

References
----------
.. [1] John E. Hopcroft and Richard M. Karp. 'An n^{5 / 2} Algorithm for
       Maximum Matchings in Bipartite Graphs' In: SIAM Journal of Computing
       2.4 (1973), pp. 225--231. <https://dx.doi.org/10.1137/0202019>.

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> from scipy.sparse.csgraph import maximum_bipartite_matching

As a simple example, consider a bipartite graph in which the partitions
contain 2 and 3 elements respectively. Suppose that one partition contains
vertices labelled 0 and 1, and that the other partition contains vertices
labelled A, B, and C. Suppose that there are edges connecting 0 and C,
1 and A, and 1 and B. This graph would then be represented by the following
sparse matrix:

>>> graph = csr_matrix([[0, 0, 1], [1, 1, 0]])

Here, the 1s could be anything, as long as they end up being stored as
elements in the sparse matrix. We can now calculate maximum matchings as
follows:

>>> print(maximum_bipartite_matching(graph, perm_type='column'))
[2 0]
>>> print(maximum_bipartite_matching(graph, perm_type='row'))
[ 1 -1  0]

The first output tells us that 1 and 2 are matched with C and A
respectively, and the second output tells us that A, B, and C are matched
with 1, nothing, and 0 respectively.

Note that explicit zeros are still converted to edges. This means that a
different way to represent the above graph is by using the CSR structure
directly as follows:

>>> data = [0, 0, 0]
>>> indices = [2, 0, 1]
>>> indptr = [0, 1, 3]
>>> graph = csr_matrix((data, indices, indptr))
>>> print(maximum_bipartite_matching(graph, perm_type='column'))
[2 0]
>>> print(maximum_bipartite_matching(graph, perm_type='row'))
[ 1 -1  0]

When one or both of the partitions are empty, the matching is empty as
well:

>>> graph = csr_matrix((2, 0))
>>> print(maximum_bipartite_matching(graph, perm_type='column'))
[-1 -1]
>>> print(maximum_bipartite_matching(graph, perm_type='row'))
[]

When the input matrix is square, and the graph is known to admit a perfect
matching, i.e. a matching with the property that every vertex in the graph
belongs to some edge in the matching, then one can view the output as the
permutation of rows (or columns) turning the input matrix into one with the
property that all diagonal elements are non-empty:

>>> a = [[0, 1, 2, 0], [1, 0, 0, 1], [2, 0, 0, 3], [0, 1, 3, 0]]
>>> graph = csr_matrix(a)
>>> perm = maximum_bipartite_matching(graph, perm_type='row')
>>> print(graph[perm].toarray())
[[1 0 0 1]
 [0 1 2 0]
 [0 1 3 0]
 [2 0 0 3]]
*)

val maximum_flow : csgraph:Py.Object.t -> source:int -> sink:int -> unit -> Py.Object.t
(**
maximum_flow(csgraph, source, sink)
              
Maximize the flow between two vertices in a graph.

.. versionadded:: 1.4.0

Parameters
----------
csgraph : csr_matrix
    The square matrix representing a directed graph whose (i, j)'th entry
    is an integer representing the capacity of the edge between
    vertices i and j.
source : int
    The source vertex from which the flow flows.
sink : int
    The sink vertex to which the flow flows.

Returns
-------
res : MaximumFlowResult
    A maximum flow represented by a ``MaximumFlowResult``
    which includes the value of the flow in ``flow_value``,
    and the residual graph in ``residual``.

Raises
------
TypeError:
    if the input graph is not in CSR format.

ValueError:
    if the capacity values are not integers, or the source or sink are out
    of bounds.

Notes
-----
This solves the maximum flow problem on a given directed weighted graph:
A flow associates to every edge a value, also called a flow, less than the
capacity of the edge, so that for every vertex (apart from the source and
the sink vertices), the total incoming flow is equal to the total outgoing
flow. The value of a flow is the sum of the flow of all edges leaving the
source vertex, and the maximum flow problem consists of finding a flow
whose value is maximal.

By the max-flow min-cut theorem, the maximal value of the flow is also the
total weight of the edges in a minimum cut.

To solve the problem, we use the Edmonds--Karp algorithm. [1]_ This
particular implementation strives to exploit sparsity. Its time complexity
is :math:`O(VE^2)` and its space complexity is :math:`O(E)`.

The maximum flow problem is usually defined with real valued capacities,
but we require that all capacities are integral to ensure convergence. When
dealing with rational capacities, or capacities belonging to
:math:`x\mathbb{Q}` for some fixed :math:`x \in \mathbb{R}`, it is possible
to reduce the problem to the integral case by scaling all capacities
accordingly.

References
----------
.. [1] Edmonds, J. and Karp, R. M.
       Theoretical improvements in algorithmic efficiency for network flow
       problems. 1972. Journal of the ACM. 19 (2): pp. 248-264
.. [2] Cormen, T. H. and Leiserson, C. E. and Rivest, R. L. and Stein C.
       Introduction to Algorithms. Second Edition. 2001. MIT Press.

Examples
--------
Perhaps the simplest flow problem is that of a graph of only two vertices
with an edge from source (0) to sink (1)::

    (0) --5--> (1)

Here, the maximum flow is simply the capacity of the edge:

>>> from scipy.sparse import csr_matrix
>>> from scipy.sparse.csgraph import maximum_flow
>>> graph = csr_matrix([[0, 5], [0, 0]])
>>> maximum_flow(graph, 0, 1).flow_value
5

If, on the other hand, there is a bottleneck between source and sink, that
can reduce the maximum flow::

    (0) --5--> (1) --3--> (2)

>>> graph = csr_matrix([[0, 5, 0], [0, 0, 3], [0, 0, 0]])
>>> maximum_flow(graph, 0, 2).flow_value
3

A less trivial example is given in [2]_, Chapter 26.1:

>>> graph = csr_matrix([[0, 16, 13,  0,  0,  0],
...                     [0, 10,  0, 12,  0,  0],
...                     [0,  4,  0,  0, 14,  0],
...                     [0,  0,  9,  0,  0, 20],
...                     [0,  0,  0,  7,  0,  4],
...                     [0,  0,  0,  0,  0,  0]])
>>> maximum_flow(graph, 0, 5).flow_value
23

It is possible to reduce the problem of finding a maximum matching in a
bipartite graph to a maximum flow problem: Let :math:`G = ((U, V), E)` be a
bipartite graph. Then, add to the graph a source vertex with edges to every
vertex in :math:`U` and a sink vertex with edges from every vertex in
:math:`V`. Finally, give every edge in the resulting graph a capacity of 1.
Then, a maximum flow in the new graph gives a maximum matching in the
original graph consisting of the edges in :math:`E` whose flow is positive.

Assume that the edges are represented by a
:math:`\lvert U \rvert \times \lvert V \rvert` matrix in CSR format whose
:math:`(i, j)`'th entry is 1 if there is an edge from :math:`i \in U` to
:math:`j \in V` and 0 otherwise; that is, the input is of the form required
by :func:`maximum_bipartite_matching`. Then the CSR representation of the
graph constructed above contains this matrix as a block. Here's an example:

>>> graph = csr_matrix([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0]])
>>> print(graph.toarray())
[[0 1 0 1]
 [1 0 1 0]
 [0 1 1 0]]
>>> i, j = graph.shape
>>> n = graph.nnz
>>> indptr = np.concatenate([[0],
...                          graph.indptr + i,
...                          np.arange(n + i + 1, n + i + j + 1),
...                          [n + i + j]])
>>> indices = np.concatenate([np.arange(1, i + 1),
...                           graph.indices + i + 1,
...                           np.repeat(i + j + 1, j)])
>>> data = np.ones(n + i + j, dtype=int)
>>>
>>> graph_flow = csr_matrix((data, indices, indptr))
>>> print(graph_flow.toarray())
[[0 1 1 1 0 0 0 0 0]
 [0 0 0 0 0 1 0 1 0]
 [0 0 0 0 1 0 1 0 0]
 [0 0 0 0 0 1 1 0 0]
 [0 0 0 0 0 0 0 0 1]
 [0 0 0 0 0 0 0 0 1]
 [0 0 0 0 0 0 0 0 1]
 [0 0 0 0 0 0 0 0 1]
 [0 0 0 0 0 0 0 0 0]]

At this point, we can find the maximum flow between the added sink and the
added source and the desired matching can be obtained by restricting the
residual graph to the block corresponding to the original graph:

>>> flow = maximum_flow(graph_flow, 0, i+j+1)
>>> matching = flow.residual[1:i+1, i+1:i+j+1]
>>> print(matching.toarray())
[[0 1 0 0]
 [1 0 0 0]
 [0 0 1 0]]

This tells us that the first, second, and third vertex in :math:`U` are
matched with the second, first, and third vertex in :math:`V` respectively.

While this solves the maximum bipartite matching problem in general, note
that algorithms specialized to that problem will perform better. In
particular, :func:`maximum_bipartite_matching` will be faster when its
preconditions are met.

This approach can also be used to solve various common generalizations of
the maximum bipartite matching problem. If, for instance, some vertices can
be matched with more than one other vertex, this may be handled by
modifying the capacities of the new graph appropriately.
*)

val minimum_spanning_tree : ?overwrite:bool -> csgraph:Py.Object.t -> unit -> Py.Object.t
(**
minimum_spanning_tree(csgraph, overwrite=False)

Return a minimum spanning tree of an undirected graph

A minimum spanning tree is a graph consisting of the subset of edges
which together connect all connected nodes, while minimizing the total
sum of weights on the edges.  This is computed using the Kruskal algorithm.

.. versionadded:: 0.11.0

Parameters
----------
csgraph : array_like or sparse matrix, 2 dimensions
    The N x N matrix representing an undirected graph over N nodes
    (see notes below).
overwrite : bool, optional
    if true, then parts of the input graph will be overwritten for
    efficiency.

Returns
-------
span_tree : csr matrix
    The N x N compressed-sparse representation of the undirected minimum
    spanning tree over the input (see notes below).

Notes
-----
This routine uses undirected graphs as input and output.  That is, if
graph[i, j] and graph[j, i] are both zero, then nodes i and j do not
have an edge connecting them.  If either is nonzero, then the two are
connected by the minimum nonzero value of the two.

Examples
--------
The following example shows the computation of a minimum spanning tree
over a simple four-component graph::

     input graph             minimum spanning tree

         (0)                         (0)
        /   \                       /
       3     8                     3
      /       \                   /
    (3)---5---(1)               (3)---5---(1)
      \       /                           /
       6     2                           2
        \   /                           /
         (2)                         (2)

It is easy to see from inspection that the minimum spanning tree involves
removing the edges with weights 8 and 6.  In compressed sparse
representation, the solution looks like this:

>>> from scipy.sparse import csr_matrix
>>> from scipy.sparse.csgraph import minimum_spanning_tree
>>> X = csr_matrix([[0, 8, 0, 3],
...                 [0, 0, 2, 5],
...                 [0, 0, 0, 6],
...                 [0, 0, 0, 0]])
>>> Tcsr = minimum_spanning_tree(X)
>>> Tcsr.toarray().astype(int)
array([[0, 0, 0, 3],
       [0, 0, 2, 5],
       [0, 0, 0, 0],
       [0, 0, 0, 0]])
*)

val reconstruct_path : ?directed:bool -> csgraph:[>`ArrayLike] Np.Obj.t -> predecessors:[`Ndarray of [>`Ndarray] Np.Obj.t | `One_dimension of Py.Object.t] -> unit -> Py.Object.t
(**
reconstruct_path(csgraph, predecessors, directed=True)

Construct a tree from a graph and a predecessor list.

.. versionadded:: 0.11.0

Parameters
----------
csgraph : array_like or sparse matrix
    The N x N matrix representing the directed or undirected graph
    from which the predecessors are drawn.
predecessors : array_like, one dimension
    The length-N array of indices of predecessors for the tree.  The
    index of the parent of node i is given by predecessors[i].
directed : bool, optional
    If True (default), then operate on a directed graph: only move from
    point i to point j along paths csgraph[i, j].
    If False, then operate on an undirected graph: the algorithm can
    progress from point i to j along csgraph[i, j] or csgraph[j, i].

Returns
-------
cstree : csr matrix
    The N x N directed compressed-sparse representation of the tree drawn
    from csgraph which is encoded by the predecessor list.

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> from scipy.sparse.csgraph import reconstruct_path

>>> graph = [
... [0, 1 , 2, 0],
... [0, 0, 0, 1],
... [0, 0, 0, 3],
... [0, 0, 0, 0]
... ]
>>> graph = csr_matrix(graph)
>>> print(graph)
  (0, 1)    1
  (0, 2)    2
  (1, 3)    1
  (2, 3)    3

>>> pred = np.array([-9999, 0, 0, 1], dtype=np.int32)

>>> cstree = reconstruct_path(csgraph=graph, predecessors=pred, directed=False)
>>> cstree.todense()
matrix([[ 0.,  1.,  2.,  0.],
        [ 0.,  0.,  0.,  1.],
        [ 0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.]])
*)

val reverse_cuthill_mckee : ?symmetric_mode:bool -> graph:[>`Spmatrix] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
reverse_cuthill_mckee(graph, symmetric_mode=False)

Returns the permutation array that orders a sparse CSR or CSC matrix
in Reverse-Cuthill McKee ordering.  

It is assumed by default, ``symmetric_mode=False``, that the input matrix 
is not symmetric and works on the matrix ``A+A.T``. If you are 
guaranteed that the matrix is symmetric in structure (values of matrix 
elements do not matter) then set ``symmetric_mode=True``.

Parameters
----------
graph : sparse matrix
    Input sparse in CSC or CSR sparse matrix format.
symmetric_mode : bool, optional
    Is input matrix guaranteed to be symmetric.

Returns
-------
perm : ndarray
    Array of permuted row and column indices.

Notes
-----
.. versionadded:: 0.15.0

References
----------
E. Cuthill and J. McKee, 'Reducing the Bandwidth of Sparse Symmetric Matrices',
ACM '69 Proceedings of the 1969 24th national conference, (1969).

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> from scipy.sparse.csgraph import reverse_cuthill_mckee

>>> graph = [
... [0, 1 , 2, 0],
... [0, 0, 0, 1],
... [2, 0, 0, 3],
... [0, 0, 0, 0]
... ]
>>> graph = csr_matrix(graph)
>>> print(graph)
  (0, 1)    1
  (0, 2)    2
  (1, 3)    1
  (2, 0)    2
  (2, 3)    3

>>> reverse_cuthill_mckee(graph)
array([3, 2, 1, 0], dtype=int32)
*)

val shortest_path : ?method_:[`Auto | `FW | `D] -> ?directed:bool -> ?return_predecessors:bool -> ?unweighted:bool -> ?overwrite:bool -> ?indices:[`Ndarray of [>`Ndarray] Np.Obj.t | `I of int] -> csgraph:Py.Object.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
shortest_path(csgraph, method='auto', directed=True, return_predecessors=False,
              unweighted=False, overwrite=False, indices=None)

Perform a shortest-path graph search on a positive directed or
undirected graph.

.. versionadded:: 0.11.0

Parameters
----------
csgraph : array, matrix, or sparse matrix, 2 dimensions
    The N x N array of distances representing the input graph.
method : string ['auto'|'FW'|'D'], optional
    Algorithm to use for shortest paths.  Options are:

       'auto' -- (default) select the best among 'FW', 'D', 'BF', or 'J'
                 based on the input data.

       'FW'   -- Floyd-Warshall algorithm.  Computational cost is
                 approximately ``O[N^3]``.  The input csgraph will be
                 converted to a dense representation.

       'D'    -- Dijkstra's algorithm with Fibonacci heaps.  Computational
                 cost is approximately ``O[N(N*k + N*log(N))]``, where
                 ``k`` is the average number of connected edges per node.
                 The input csgraph will be converted to a csr
                 representation.

       'BF'   -- Bellman-Ford algorithm.  This algorithm can be used when
                 weights are negative.  If a negative cycle is encountered,
                 an error will be raised.  Computational cost is
                 approximately ``O[N(N^2 k)]``, where ``k`` is the average
                 number of connected edges per node. The input csgraph will
                 be converted to a csr representation.

       'J'    -- Johnson's algorithm.  Like the Bellman-Ford algorithm,
                 Johnson's algorithm is designed for use when the weights
                 are negative.  It combines the Bellman-Ford algorithm
                 with Dijkstra's algorithm for faster computation.

directed : bool, optional
    If True (default), then find the shortest path on a directed graph:
    only move from point i to point j along paths csgraph[i, j].
    If False, then find the shortest path on an undirected graph: the
    algorithm can progress from point i to j along csgraph[i, j] or
    csgraph[j, i]
return_predecessors : bool, optional
    If True, return the size (N, N) predecesor matrix
unweighted : bool, optional
    If True, then find unweighted distances.  That is, rather than finding
    the path between each point such that the sum of weights is minimized,
    find the path such that the number of edges is minimized.
overwrite : bool, optional
    If True, overwrite csgraph with the result.  This applies only if
    method == 'FW' and csgraph is a dense, c-ordered array with
    dtype=float64.
indices : array_like or int, optional
    If specified, only compute the paths from the points at the given
    indices. Incompatible with method == 'FW'.

Returns
-------
dist_matrix : ndarray
    The N x N matrix of distances between graph nodes. dist_matrix[i,j]
    gives the shortest distance from point i to point j along the graph.
predecessors : ndarray
    Returned only if return_predecessors == True.
    The N x N matrix of predecessors, which can be used to reconstruct
    the shortest paths.  Row i of the predecessor matrix contains
    information on the shortest paths from point i: each entry
    predecessors[i, j] gives the index of the previous node in the
    path from point i to point j.  If no path exists between point
    i and j, then predecessors[i, j] = -9999

Raises
------
NegativeCycleError:
    if there are negative cycles in the graph

Notes
-----
As currently implemented, Dijkstra's algorithm and Johnson's algorithm
do not work for graphs with direction-dependent distances when
directed == False.  i.e., if csgraph[i,j] and csgraph[j,i] are non-equal
edges, method='D' may yield an incorrect result.

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> from scipy.sparse.csgraph import shortest_path

>>> graph = [
... [0, 1, 2, 0],
... [0, 0, 0, 1],
... [2, 0, 0, 3],
... [0, 0, 0, 0]
... ]
>>> graph = csr_matrix(graph)
>>> print(graph)
  (0, 1)    1
  (0, 2)    2
  (1, 3)    1
  (2, 0)    2
  (2, 3)    3

>>> dist_matrix, predecessors = shortest_path(csgraph=graph, directed=False, indices=0, return_predecessors=True)
>>> dist_matrix
array([ 0.,  1.,  2.,  2.])
>>> predecessors
array([-9999,     0,     0,     1], dtype=int32)
*)

val structural_rank : [>`Spmatrix] Np.Obj.t -> int
(**
structural_rank(graph)

Compute the structural rank of a graph (matrix) with a given 
sparsity pattern.

The structural rank of a matrix is the number of entries in the maximum 
transversal of the corresponding bipartite graph, and is an upper bound 
on the numerical rank of the matrix. A graph has full structural rank 
if it is possible to permute the elements to make the diagonal zero-free.

.. versionadded:: 0.19.0

Parameters
----------
graph : sparse matrix
    Input sparse matrix.

Returns
-------
rank : int
    The structural rank of the sparse graph.

References
----------
.. [1] I. S. Duff, 'Computing the Structural Index', SIAM J. Alg. Disc. 
        Meth., Vol. 7, 594 (1986).

.. [2] http://www.cise.ufl.edu/research/sparse/matrices/legend.html

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> from scipy.sparse.csgraph import structural_rank

>>> graph = [
... [0, 1, 2, 0],
... [1, 0, 0, 1],
... [2, 0, 0, 3],
... [0, 1, 3, 0]
... ]
>>> graph = csr_matrix(graph)
>>> print(graph)
  (0, 1)    1
  (0, 2)    2
  (1, 0)    1
  (1, 3)    1
  (2, 0)    2
  (2, 3)    3
  (3, 1)    1
  (3, 2)    3

>>> structural_rank(graph)
4
*)


end

module Csr : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module Xrange : sig
type tag = [`Range]
type t = [`Object | `Range] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : Py.Object.t -> t
(**
range(stop) -> range object
range(start, stop[, step]) -> range object

Return an object that produces a sequence of integers from start (inclusive)
to stop (exclusive) by step.  range(i, j) produces i, i+1, i+2, ..., j-1.
start defaults to 0, and stop is omitted!  range(4) produces 0, 1, 2, 3.
These are exactly the valid indices for a list of 4 elements.
When step is given, it specifies the increment (or decrement).
*)

val __getitem__ : key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return self[key].
*)

val __iter__ : [> tag] Obj.t -> Py.Object.t
(**
Implement iter(self).
*)

val count : value:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
rangeobject.count(value) -> integer -- return number of occurrences of value
*)

val index : value:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
rangeobject.index(value) -> integer -- return index of value.
Raise ValueError if the value is not present.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val get_index_dtype : ?arrays:Py.Object.t -> ?maxval:float -> ?check_contents:bool -> unit -> Np.Dtype.t
(**
Based on input (integer) arrays `a`, determine a suitable index data
type that can hold the data in the arrays.

Parameters
----------
arrays : tuple of array_like
    Input arrays whose types/contents to check
maxval : float, optional
    Maximum value needed
check_contents : bool, optional
    Whether to check the values in the arrays and not just their types.
    Default: False (check only the types)

Returns
-------
dtype : dtype
    Suitable index data type (int32 or int64)
*)

val isspmatrix_csr : Py.Object.t -> Py.Object.t
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

val upcast : Py.Object.t list -> Py.Object.t
(**
Returns the nearest supported sparse dtype for the
combination of one or more types.

upcast(t0, t1, ..., tn) -> T  where T is a supported dtype

Examples
--------

>>> upcast('int32')
<type 'numpy.int32'>
>>> upcast('bool')
<type 'numpy.bool_'>
>>> upcast('int32','float32')
<type 'numpy.float64'>
>>> upcast('bool',complex,float)
<type 'numpy.complex128'>
*)


end

module Data : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val isscalarlike : Py.Object.t -> Py.Object.t
(**
Is x either a scalar, an array scalar, or a 0-dim array?
*)

val matrix : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> Py.Object.t
(**
None
*)

val npfunc : ?out:[`Ndarray of [>`Ndarray] Np.Obj.t | `Tuple_of_ndarray_and_None of Py.Object.t] -> ?where:[>`Ndarray] Np.Obj.t -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
sign(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

Returns an element-wise indication of the sign of a number.

The `sign` function returns ``-1 if x < 0, 0 if x==0, 1 if x > 0``.  nan
is returned for nan inputs.

For complex inputs, the `sign` function returns
``sign(x.real) + 0j if x.real != 0 else sign(x.imag) + 0j``.

complex(nan, 0) is returned for complex nan inputs.

Parameters
----------
x : array_like
    Input values.
out : ndarray, None, or tuple of ndarray and None, optional
    A location into which the result is stored. If provided, it must have
    a shape that the inputs broadcast to. If not provided or None,
    a freshly-allocated array is returned. A tuple (possible only as a
    keyword argument) must have length equal to the number of outputs.
where : array_like, optional
    This condition is broadcast over the input. At locations where the
    condition is True, the `out` array will be set to the ufunc result.
    Elsewhere, the `out` array will retain its original value.
    Note that if an uninitialized `out` array is created via the default
    ``out=None``, locations within it where the condition is False will
    remain uninitialized.
**kwargs
    For other keyword-only arguments, see the
    :ref:`ufunc docs <ufuncs.kwargs>`.

Returns
-------
y : ndarray
    The sign of `x`.
    This is a scalar if `x` is a scalar.

Notes
-----
There is more than one definition of sign in common use for complex
numbers.  The definition used here is equivalent to :math:`x/\sqrt{x*x}`
which is different from a common alternative, :math:`x/|x|`.

Examples
--------
>>> np.sign([-5., 4.5])
array([-1.,  1.])
>>> np.sign(0)
0
>>> np.sign(5-2j)
(1+0j)
*)

val validateaxis : Py.Object.t -> Py.Object.t
(**
None
*)


end

module Dia : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val check_shape : ?current_shape:Py.Object.t -> args:Py.Object.t -> unit -> Py.Object.t
(**
Imitate numpy.matrix handling of shape arguments
*)

val get_index_dtype : ?arrays:Py.Object.t -> ?maxval:float -> ?check_contents:bool -> unit -> Np.Dtype.t
(**
Based on input (integer) arrays `a`, determine a suitable index data
type that can hold the data in the arrays.

Parameters
----------
arrays : tuple of array_like
    Input arrays whose types/contents to check
maxval : float, optional
    Maximum value needed
check_contents : bool, optional
    Whether to check the values in the arrays and not just their types.
    Default: False (check only the types)

Returns
-------
dtype : dtype
    Suitable index data type (int32 or int64)
*)

val get_sum_dtype : Py.Object.t -> Py.Object.t
(**
Mimic numpy's casting for np.sum
*)

val getdtype : ?a:Py.Object.t -> ?default:Py.Object.t -> dtype:Py.Object.t -> unit -> Py.Object.t
(**
Function used to simplify argument processing.  If 'dtype' is not
specified (is None), returns a.dtype; otherwise returns a np.dtype
object created from the specified dtype argument.  If 'dtype' and 'a'
are both None, construct a data type out of the 'default' parameter.
Furthermore, 'dtype' must be in 'allowed' set.
*)

val isshape : ?nonneg:Py.Object.t -> x:Py.Object.t -> unit -> Py.Object.t
(**
Is x a valid 2-tuple of dimensions?

If nonneg, also checks that the dimensions are non-negative.
*)

val isspmatrix : Py.Object.t -> Py.Object.t
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

val isspmatrix_dia : Py.Object.t -> Py.Object.t
(**
Is x of dia_matrix type?

Parameters
----------
x
    object to check for being a dia matrix

Returns
-------
bool
    True if x is a dia matrix, False otherwise

Examples
--------
>>> from scipy.sparse import dia_matrix, isspmatrix_dia
>>> isspmatrix_dia(dia_matrix([[5]]))
True

>>> from scipy.sparse import dia_matrix, csr_matrix, isspmatrix_dia
>>> isspmatrix_dia(csr_matrix([[5]]))
False
*)

val matrix : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> Py.Object.t
(**
None
*)

val upcast_char : Py.Object.t list -> Py.Object.t
(**
Same as `upcast` but taking dtype.char as input (faster).
*)

val validateaxis : Py.Object.t -> Py.Object.t
(**
None
*)


end

module Dok : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module IndexMixin : sig
type tag = [`IndexMixin]
type t = [`IndexMixin | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : unit -> t
(**
This class provides common dispatching and validation logic for indexing.
*)

val __getitem__ : key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val __setitem__ : key:Py.Object.t -> x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val getcol : i:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return a copy of column i of the matrix, as a (m x 1) column vector.
        
*)

val getrow : i:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return a copy of row i of the matrix, as a (1 x n) row vector.
        
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Izip : sig
type tag = [`Zip]
type t = [`Object | `Zip] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : Py.Object.t list -> t
(**
zip( *iterables) --> zip object

Return a zip object whose .__next__() method returns a tuple where
the i-th element comes from the i-th iterable argument.  The .__next__()
method continues until the shortest iterable in the argument sequence
is exhausted and then it raises StopIteration.
*)

val __iter__ : [> tag] Obj.t -> Py.Object.t
(**
Implement iter(self).
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Xrange : sig
type tag = [`Range]
type t = [`Object | `Range] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : Py.Object.t -> t
(**
range(stop) -> range object
range(start, stop[, step]) -> range object

Return an object that produces a sequence of integers from start (inclusive)
to stop (exclusive) by step.  range(i, j) produces i, i+1, i+2, ..., j-1.
start defaults to 0, and stop is omitted!  range(4) produces 0, 1, 2, 3.
These are exactly the valid indices for a list of 4 elements.
When step is given, it specifies the increment (or decrement).
*)

val __getitem__ : key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return self[key].
*)

val __iter__ : [> tag] Obj.t -> Py.Object.t
(**
Implement iter(self).
*)

val count : value:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
rangeobject.count(value) -> integer -- return number of occurrences of value
*)

val index : value:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
rangeobject.index(value) -> integer -- return index of value.
Raise ValueError if the value is not present.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val check_shape : ?current_shape:Py.Object.t -> args:Py.Object.t -> unit -> Py.Object.t
(**
Imitate numpy.matrix handling of shape arguments
*)

val get_index_dtype : ?arrays:Py.Object.t -> ?maxval:float -> ?check_contents:bool -> unit -> Np.Dtype.t
(**
Based on input (integer) arrays `a`, determine a suitable index data
type that can hold the data in the arrays.

Parameters
----------
arrays : tuple of array_like
    Input arrays whose types/contents to check
maxval : float, optional
    Maximum value needed
check_contents : bool, optional
    Whether to check the values in the arrays and not just their types.
    Default: False (check only the types)

Returns
-------
dtype : dtype
    Suitable index data type (int32 or int64)
*)

val getdtype : ?a:Py.Object.t -> ?default:Py.Object.t -> dtype:Py.Object.t -> unit -> Py.Object.t
(**
Function used to simplify argument processing.  If 'dtype' is not
specified (is None), returns a.dtype; otherwise returns a np.dtype
object created from the specified dtype argument.  If 'dtype' and 'a'
are both None, construct a data type out of the 'default' parameter.
Furthermore, 'dtype' must be in 'allowed' set.
*)

val isdense : Py.Object.t -> Py.Object.t
(**
None
*)

val isintlike : Py.Object.t -> Py.Object.t
(**
Is x appropriate as an index into a sparse matrix? Returns True
if it can be cast safely to a machine int.
*)

val isscalarlike : Py.Object.t -> Py.Object.t
(**
Is x either a scalar, an array scalar, or a 0-dim array?
*)

val isshape : ?nonneg:Py.Object.t -> x:Py.Object.t -> unit -> Py.Object.t
(**
Is x a valid 2-tuple of dimensions?

If nonneg, also checks that the dimensions are non-negative.
*)

val isspmatrix : Py.Object.t -> Py.Object.t
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

val isspmatrix_dok : Py.Object.t -> Py.Object.t
(**
Is x of dok_matrix type?

Parameters
----------
x
    object to check for being a dok matrix

Returns
-------
bool
    True if x is a dok matrix, False otherwise

Examples
--------
>>> from scipy.sparse import dok_matrix, isspmatrix_dok
>>> isspmatrix_dok(dok_matrix([[5]]))
True

>>> from scipy.sparse import dok_matrix, csr_matrix, isspmatrix_dok
>>> isspmatrix_dok(csr_matrix([[5]]))
False
*)

val iteritems : Py.Object.t -> Py.Object.t
(**
Return an iterator over the (key, value) pairs of a dictionary.
*)

val iterkeys : Py.Object.t -> Py.Object.t
(**
Return an iterator over the keys of a dictionary.
*)

val itervalues : Py.Object.t -> Py.Object.t
(**
Return an iterator over the values of a dictionary.
*)

val upcast : Py.Object.t list -> Py.Object.t
(**
Returns the nearest supported sparse dtype for the
combination of one or more types.

upcast(t0, t1, ..., tn) -> T  where T is a supported dtype

Examples
--------

>>> upcast('int32')
<type 'numpy.int32'>
>>> upcast('bool')
<type 'numpy.bool_'>
>>> upcast('int32','float32')
<type 'numpy.float64'>
>>> upcast('bool',complex,float)
<type 'numpy.complex128'>
*)

val upcast_scalar : dtype:Py.Object.t -> scalar:Py.Object.t -> unit -> Py.Object.t
(**
Determine data type for binary operation between an array of
type `dtype` and a scalar.
*)


end

module Extract : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val find : [`Dense of Py.Object.t | `Spmatrix of [>`Spmatrix] Np.Obj.t] -> Py.Object.t
(**
Return the indices and values of the nonzero elements of a matrix

Parameters
----------
A : dense or sparse matrix
    Matrix whose nonzero elements are desired.

Returns
-------
(I,J,V) : tuple of arrays
    I,J, and V contain the row indices, column indices, and values
    of the nonzero matrix entries.


Examples
--------
>>> from scipy.sparse import csr_matrix, find
>>> A = csr_matrix([[7.0, 8.0, 0],[0, 0, 9.0]])
>>> find(A)
(array([0, 0, 1], dtype=int32), array([0, 1, 2], dtype=int32), array([ 7.,  8.,  9.]))
*)

val tril : ?k:Py.Object.t -> ?format:string -> a:[`Dense of Py.Object.t | `Spmatrix of [>`Spmatrix] Np.Obj.t] -> unit -> [`ArrayLike|`Object|`Spmatrix] Np.Obj.t
(**
Return the lower triangular portion of a matrix in sparse format

Returns the elements on or below the k-th diagonal of the matrix A.
    - k = 0 corresponds to the main diagonal
    - k > 0 is above the main diagonal
    - k < 0 is below the main diagonal

Parameters
----------
A : dense or sparse matrix
    Matrix whose lower trianglar portion is desired.
k : integer : optional
    The top-most diagonal of the lower triangle.
format : string
    Sparse format of the result, e.g. format='csr', etc.

Returns
-------
L : sparse matrix
    Lower triangular portion of A in sparse format.

See Also
--------
triu : upper triangle in sparse format

Examples
--------
>>> from scipy.sparse import csr_matrix, tril
>>> A = csr_matrix([[1, 2, 0, 0, 3], [4, 5, 0, 6, 7], [0, 0, 8, 9, 0]],
...                dtype='int32')
>>> A.toarray()
array([[1, 2, 0, 0, 3],
       [4, 5, 0, 6, 7],
       [0, 0, 8, 9, 0]])
>>> tril(A).toarray()
array([[1, 0, 0, 0, 0],
       [4, 5, 0, 0, 0],
       [0, 0, 8, 0, 0]])
>>> tril(A).nnz
4
>>> tril(A, k=1).toarray()
array([[1, 2, 0, 0, 0],
       [4, 5, 0, 0, 0],
       [0, 0, 8, 9, 0]])
>>> tril(A, k=-1).toarray()
array([[0, 0, 0, 0, 0],
       [4, 0, 0, 0, 0],
       [0, 0, 0, 0, 0]])
>>> tril(A, format='csc')
<3x5 sparse matrix of type '<class 'numpy.int32'>'
        with 4 stored elements in Compressed Sparse Column format>
*)

val triu : ?k:Py.Object.t -> ?format:string -> a:[`Dense of Py.Object.t | `Spmatrix of [>`Spmatrix] Np.Obj.t] -> unit -> [`ArrayLike|`Object|`Spmatrix] Np.Obj.t
(**
Return the upper triangular portion of a matrix in sparse format

Returns the elements on or above the k-th diagonal of the matrix A.
    - k = 0 corresponds to the main diagonal
    - k > 0 is above the main diagonal
    - k < 0 is below the main diagonal

Parameters
----------
A : dense or sparse matrix
    Matrix whose upper trianglar portion is desired.
k : integer : optional
    The bottom-most diagonal of the upper triangle.
format : string
    Sparse format of the result, e.g. format='csr', etc.

Returns
-------
L : sparse matrix
    Upper triangular portion of A in sparse format.

See Also
--------
tril : lower triangle in sparse format

Examples
--------
>>> from scipy.sparse import csr_matrix, triu
>>> A = csr_matrix([[1, 2, 0, 0, 3], [4, 5, 0, 6, 7], [0, 0, 8, 9, 0]],
...                dtype='int32')
>>> A.toarray()
array([[1, 2, 0, 0, 3],
       [4, 5, 0, 6, 7],
       [0, 0, 8, 9, 0]])
>>> triu(A).toarray()
array([[1, 2, 0, 0, 3],
       [0, 5, 0, 6, 7],
       [0, 0, 8, 9, 0]])
>>> triu(A).nnz
8
>>> triu(A, k=1).toarray()
array([[0, 2, 0, 0, 3],
       [0, 0, 0, 6, 7],
       [0, 0, 0, 9, 0]])
>>> triu(A, k=-1).toarray()
array([[1, 2, 0, 0, 3],
       [4, 5, 0, 6, 7],
       [0, 0, 8, 9, 0]])
>>> triu(A, format='csc')
<3x5 sparse matrix of type '<class 'numpy.int32'>'
        with 8 stored elements in Compressed Sparse Column format>
*)


end

module Lil : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module IndexMixin : sig
type tag = [`IndexMixin]
type t = [`IndexMixin | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : unit -> t
(**
This class provides common dispatching and validation logic for indexing.
*)

val __getitem__ : key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val __setitem__ : key:Py.Object.t -> x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val getcol : i:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return a copy of column i of the matrix, as a (m x 1) column vector.
        
*)

val getrow : i:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return a copy of row i of the matrix, as a (1 x n) row vector.
        
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Xrange : sig
type tag = [`Range]
type t = [`Object | `Range] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : Py.Object.t -> t
(**
range(stop) -> range object
range(start, stop[, step]) -> range object

Return an object that produces a sequence of integers from start (inclusive)
to stop (exclusive) by step.  range(i, j) produces i, i+1, i+2, ..., j-1.
start defaults to 0, and stop is omitted!  range(4) produces 0, 1, 2, 3.
These are exactly the valid indices for a list of 4 elements.
When step is given, it specifies the increment (or decrement).
*)

val __getitem__ : key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return self[key].
*)

val __iter__ : [> tag] Obj.t -> Py.Object.t
(**
Implement iter(self).
*)

val count : value:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
rangeobject.count(value) -> integer -- return number of occurrences of value
*)

val index : value:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
rangeobject.index(value) -> integer -- return index of value.
Raise ValueError if the value is not present.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Zip : sig
type tag = [`Zip]
type t = [`Object | `Zip] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : Py.Object.t list -> t
(**
zip( *iterables) --> zip object

Return a zip object whose .__next__() method returns a tuple where
the i-th element comes from the i-th iterable argument.  The .__next__()
method continues until the shortest iterable in the argument sequence
is exhausted and then it raises StopIteration.
*)

val __iter__ : [> tag] Obj.t -> Py.Object.t
(**
Implement iter(self).
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val asmatrix : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> Py.Object.t
(**
None
*)

val check_reshape_kwargs : Py.Object.t -> Py.Object.t
(**
Unpack keyword arguments for reshape function.

This is useful because keyword arguments after star arguments are not
allowed in Python 2, but star keyword arguments are. This function unpacks
'order' and 'copy' from the star keyword arguments (with defaults) and
throws an error for any remaining.
*)

val check_shape : ?current_shape:Py.Object.t -> args:Py.Object.t -> unit -> Py.Object.t
(**
Imitate numpy.matrix handling of shape arguments
*)

val get_index_dtype : ?arrays:Py.Object.t -> ?maxval:float -> ?check_contents:bool -> unit -> Np.Dtype.t
(**
Based on input (integer) arrays `a`, determine a suitable index data
type that can hold the data in the arrays.

Parameters
----------
arrays : tuple of array_like
    Input arrays whose types/contents to check
maxval : float, optional
    Maximum value needed
check_contents : bool, optional
    Whether to check the values in the arrays and not just their types.
    Default: False (check only the types)

Returns
-------
dtype : dtype
    Suitable index data type (int32 or int64)
*)

val getdtype : ?a:Py.Object.t -> ?default:Py.Object.t -> dtype:Py.Object.t -> unit -> Py.Object.t
(**
Function used to simplify argument processing.  If 'dtype' is not
specified (is None), returns a.dtype; otherwise returns a np.dtype
object created from the specified dtype argument.  If 'dtype' and 'a'
are both None, construct a data type out of the 'default' parameter.
Furthermore, 'dtype' must be in 'allowed' set.
*)

val isscalarlike : Py.Object.t -> Py.Object.t
(**
Is x either a scalar, an array scalar, or a 0-dim array?
*)

val isshape : ?nonneg:Py.Object.t -> x:Py.Object.t -> unit -> Py.Object.t
(**
Is x a valid 2-tuple of dimensions?

If nonneg, also checks that the dimensions are non-negative.
*)

val isspmatrix : Py.Object.t -> Py.Object.t
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

val isspmatrix_lil : Py.Object.t -> Py.Object.t
(**
Is x of lil_matrix type?

Parameters
----------
x
    object to check for being a lil matrix

Returns
-------
bool
    True if x is a lil matrix, False otherwise

Examples
--------
>>> from scipy.sparse import lil_matrix, isspmatrix_lil
>>> isspmatrix_lil(lil_matrix([[5]]))
True

>>> from scipy.sparse import lil_matrix, csr_matrix, isspmatrix_lil
>>> isspmatrix_lil(csr_matrix([[5]]))
False
*)

val upcast_scalar : dtype:Py.Object.t -> scalar:Py.Object.t -> unit -> Py.Object.t
(**
Determine data type for binary operation between an array of
type `dtype` and a scalar.
*)


end

module Linalg : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module ArpackError : sig
type tag = [`ArpackError]
type t = [`ArpackError | `BaseException | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_exception : t -> [`BaseException] Obj.t
val create : ?infodict:Py.Object.t -> info:Py.Object.t -> unit -> t
(**
ARPACK error
*)

val with_traceback : tb:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Exception.with_traceback(tb) --
set self.__traceback__ to tb and return self.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module ArpackNoConvergence : sig
type tag = [`ArpackNoConvergence]
type t = [`ArpackNoConvergence | `BaseException | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_exception : t -> [`BaseException] Obj.t
val create : msg:Py.Object.t -> eigenvalues:Py.Object.t -> eigenvectors:Py.Object.t -> unit -> t
(**
ARPACK iteration did not converge

Attributes
----------
eigenvalues : ndarray
    Partial result. Converged eigenvalues.
eigenvectors : ndarray
    Partial result. Converged eigenvectors.
*)

val with_traceback : tb:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Exception.with_traceback(tb) --
set self.__traceback__ to tb and return self.
*)


(** Attribute eigenvalues: get value or raise Not_found if None.*)
val eigenvalues : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute eigenvalues: get value as an option. *)
val eigenvalues_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute eigenvectors: get value or raise Not_found if None.*)
val eigenvectors : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute eigenvectors: get value as an option. *)
val eigenvectors_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module LinearOperator : sig
type tag = [`LinearOperator]
type t = [`LinearOperator | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> t
(**
Common interface for performing matrix vector products

Many iterative methods (e.g. cg, gmres) do not need to know the
individual entries of a matrix to solve a linear system A*x=b.
Such solvers only require the computation of matrix vector
products, A*v where v is a dense vector.  This class serves as
an abstract interface between iterative solvers and matrix-like
objects.

To construct a concrete LinearOperator, either pass appropriate
callables to the constructor of this class, or subclass it.

A subclass must implement either one of the methods ``_matvec``
and ``_matmat``, and the attributes/properties ``shape`` (pair of
integers) and ``dtype`` (may be None). It may call the ``__init__``
on this class to have these attributes validated. Implementing
``_matvec`` automatically implements ``_matmat`` (using a naive
algorithm) and vice-versa.

Optionally, a subclass may implement ``_rmatvec`` or ``_adjoint``
to implement the Hermitian adjoint (conjugate transpose). As with
``_matvec`` and ``_matmat``, implementing either ``_rmatvec`` or
``_adjoint`` implements the other automatically. Implementing
``_adjoint`` is preferable; ``_rmatvec`` is mostly there for
backwards compatibility.

Parameters
----------
shape : tuple
    Matrix dimensions (M, N).
matvec : callable f(v)
    Returns returns A * v.
rmatvec : callable f(v)
    Returns A^H * v, where A^H is the conjugate transpose of A.
matmat : callable f(V)
    Returns A * V, where V is a dense matrix with dimensions (N, K).
dtype : dtype
    Data type of the matrix.
rmatmat : callable f(V)
    Returns A^H * V, where V is a dense matrix with dimensions (M, K).

Attributes
----------
args : tuple
    For linear operators describing products etc. of other linear
    operators, the operands of the binary operation.

See Also
--------
aslinearoperator : Construct LinearOperators

Notes
-----
The user-defined matvec() function must properly handle the case
where v has shape (N,) as well as the (N,1) case.  The shape of
the return type is handled internally by LinearOperator.

LinearOperator instances can also be multiplied, added with each
other and exponentiated, all lazily: the result of these operations
is always a new, composite LinearOperator, that defers linear
operations to the original operators and combines the results.

More details regarding how to subclass a LinearOperator and several
examples of concrete LinearOperator instances can be found in the
external project `PyLops <https://pylops.readthedocs.io>`_.


Examples
--------
>>> import numpy as np
>>> from scipy.sparse.linalg import LinearOperator
>>> def mv(v):
...     return np.array([2*v[0], 3*v[1]])
...
>>> A = LinearOperator((2,2), matvec=mv)
>>> A
<2x2 _CustomLinearOperator with dtype=float64>
>>> A.matvec(np.ones(2))
array([ 2.,  3.])
>>> A * np.ones(2)
array([ 2.,  3.])
*)

val adjoint : [> tag] Obj.t -> Py.Object.t
(**
Hermitian adjoint.

Returns the Hermitian adjoint of self, aka the Hermitian
conjugate or Hermitian transpose. For a complex matrix, the
Hermitian adjoint is equal to the conjugate transpose.

Can be abbreviated self.H instead of self.adjoint().

Returns
-------
A_H : LinearOperator
    Hermitian adjoint of self.
*)

val dot : x:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Matrix-matrix or matrix-vector multiplication.

Parameters
----------
x : array_like
    1-d or 2-d array, representing a vector or matrix.

Returns
-------
Ax : array
    1-d or 2-d array (depending on the shape of x) that represents
    the result of applying this linear operator on x.
*)

val matmat : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Matrix-matrix multiplication.

Performs the operation y=A*X where A is an MxN linear
operator and X dense N*K matrix or ndarray.

Parameters
----------
X : {matrix, ndarray}
    An array with shape (N,K).

Returns
-------
Y : {matrix, ndarray}
    A matrix or ndarray with shape (M,K) depending on
    the type of the X argument.

Notes
-----
This matmat wraps any user-specified matmat routine or overridden
_matmat method to ensure that y has the correct type.
*)

val matvec : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Matrix-vector multiplication.

Performs the operation y=A*x where A is an MxN linear
operator and x is a column vector or 1-d array.

Parameters
----------
x : {matrix, ndarray}
    An array with shape (N,) or (N,1).

Returns
-------
y : {matrix, ndarray}
    A matrix or ndarray with shape (M,) or (M,1) depending
    on the type and shape of the x argument.

Notes
-----
This matvec wraps the user-specified matvec routine or overridden
_matvec method to ensure that y has the correct shape and type.
*)

val rmatmat : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Adjoint matrix-matrix multiplication.

Performs the operation y = A^H * x where A is an MxN linear
operator and x is a column vector or 1-d array, or 2-d array.
The default implementation defers to the adjoint.

Parameters
----------
X : {matrix, ndarray}
    A matrix or 2D array.

Returns
-------
Y : {matrix, ndarray}
    A matrix or 2D array depending on the type of the input.

Notes
-----
This rmatmat wraps the user-specified rmatmat routine.
*)

val rmatvec : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Adjoint matrix-vector multiplication.

Performs the operation y = A^H * x where A is an MxN linear
operator and x is a column vector or 1-d array.

Parameters
----------
x : {matrix, ndarray}
    An array with shape (M,) or (M,1).

Returns
-------
y : {matrix, ndarray}
    A matrix or ndarray with shape (N,) or (N,1) depending
    on the type and shape of the x argument.

Notes
-----
This rmatvec wraps the user-specified rmatvec routine or overridden
_rmatvec method to ensure that y has the correct shape and type.
*)

val transpose : [> tag] Obj.t -> Py.Object.t
(**
Transpose this linear operator.

Returns a LinearOperator that represents the transpose of this one.
Can be abbreviated self.T instead of self.transpose().
*)


(** Attribute args: get value or raise Not_found if None.*)
val args : t -> Py.Object.t

(** Attribute args: get value as an option. *)
val args_opt : t -> (Py.Object.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MatrixRankWarning : sig
type tag = [`MatrixRankWarning]
type t = [`BaseException | `MatrixRankWarning | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_exception : t -> [`BaseException] Obj.t
val with_traceback : tb:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Exception.with_traceback(tb) --
set self.__traceback__ to tb and return self.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module SuperLU : sig
type tag = [`SuperLU]
type t = [`Object | `SuperLU] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : unit -> t
(**
LU factorization of a sparse matrix.

Factorization is represented as::

    Pr * A * Pc = L * U

To construct these `SuperLU` objects, call the `splu` and `spilu`
functions.

Attributes
----------
shape
nnz
perm_c
perm_r
L
U

Methods
-------
solve

Notes
-----

.. versionadded:: 0.14.0

Examples
--------
The LU decomposition can be used to solve matrix equations. Consider:

>>> import numpy as np
>>> from scipy.sparse import csc_matrix, linalg as sla
>>> A = csc_matrix([[1,2,0,4],[1,0,0,1],[1,0,2,1],[2,2,1,0.]])

This can be solved for a given right-hand side:

>>> lu = sla.splu(A)
>>> b = np.array([1, 2, 3, 4])
>>> x = lu.solve(b)
>>> A.dot(x)
array([ 1.,  2.,  3.,  4.])

The ``lu`` object also contains an explicit representation of the
decomposition. The permutations are represented as mappings of
indices:

>>> lu.perm_r
array([0, 2, 1, 3], dtype=int32)
>>> lu.perm_c
array([2, 0, 1, 3], dtype=int32)

The L and U factors are sparse matrices in CSC format:

>>> lu.L.A
array([[ 1. ,  0. ,  0. ,  0. ],
       [ 0. ,  1. ,  0. ,  0. ],
       [ 0. ,  0. ,  1. ,  0. ],
       [ 1. ,  0.5,  0.5,  1. ]])
>>> lu.U.A
array([[ 2.,  0.,  1.,  4.],
       [ 0.,  2.,  1.,  1.],
       [ 0.,  0.,  1.,  1.],
       [ 0.,  0.,  0., -5.]])

The permutation matrices can be constructed:

>>> Pr = csc_matrix((np.ones(4), (lu.perm_r, np.arange(4))))
>>> Pc = csc_matrix((np.ones(4), (np.arange(4), lu.perm_c)))

We can reassemble the original matrix:

>>> (Pr.T * (lu.L * lu.U) * Pc.T).A
array([[ 1.,  2.,  0.,  4.],
       [ 1.,  0.,  0.,  1.],
       [ 1.,  0.,  2.,  1.],
       [ 2.,  2.,  1.,  0.]])
*)


(** Attribute shape: get value or raise Not_found if None.*)
val shape : t -> Py.Object.t

(** Attribute shape: get value as an option. *)
val shape_opt : t -> (Py.Object.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Arpack : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module IterInv : sig
type tag = [`IterInv]
type t = [`IterInv | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> t
(**
IterInv:
   helper class to repeatedly solve M*x=b
   using an iterative method.
*)

val adjoint : [> tag] Obj.t -> Py.Object.t
(**
Hermitian adjoint.

Returns the Hermitian adjoint of self, aka the Hermitian
conjugate or Hermitian transpose. For a complex matrix, the
Hermitian adjoint is equal to the conjugate transpose.

Can be abbreviated self.H instead of self.adjoint().

Returns
-------
A_H : LinearOperator
    Hermitian adjoint of self.
*)

val dot : x:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Matrix-matrix or matrix-vector multiplication.

Parameters
----------
x : array_like
    1-d or 2-d array, representing a vector or matrix.

Returns
-------
Ax : array
    1-d or 2-d array (depending on the shape of x) that represents
    the result of applying this linear operator on x.
*)

val matmat : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Matrix-matrix multiplication.

Performs the operation y=A*X where A is an MxN linear
operator and X dense N*K matrix or ndarray.

Parameters
----------
X : {matrix, ndarray}
    An array with shape (N,K).

Returns
-------
Y : {matrix, ndarray}
    A matrix or ndarray with shape (M,K) depending on
    the type of the X argument.

Notes
-----
This matmat wraps any user-specified matmat routine or overridden
_matmat method to ensure that y has the correct type.
*)

val matvec : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Matrix-vector multiplication.

Performs the operation y=A*x where A is an MxN linear
operator and x is a column vector or 1-d array.

Parameters
----------
x : {matrix, ndarray}
    An array with shape (N,) or (N,1).

Returns
-------
y : {matrix, ndarray}
    A matrix or ndarray with shape (M,) or (M,1) depending
    on the type and shape of the x argument.

Notes
-----
This matvec wraps the user-specified matvec routine or overridden
_matvec method to ensure that y has the correct shape and type.
*)

val rmatmat : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Adjoint matrix-matrix multiplication.

Performs the operation y = A^H * x where A is an MxN linear
operator and x is a column vector or 1-d array, or 2-d array.
The default implementation defers to the adjoint.

Parameters
----------
X : {matrix, ndarray}
    A matrix or 2D array.

Returns
-------
Y : {matrix, ndarray}
    A matrix or 2D array depending on the type of the input.

Notes
-----
This rmatmat wraps the user-specified rmatmat routine.
*)

val rmatvec : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Adjoint matrix-vector multiplication.

Performs the operation y = A^H * x where A is an MxN linear
operator and x is a column vector or 1-d array.

Parameters
----------
x : {matrix, ndarray}
    An array with shape (M,) or (M,1).

Returns
-------
y : {matrix, ndarray}
    A matrix or ndarray with shape (N,) or (N,1) depending
    on the type and shape of the x argument.

Notes
-----
This rmatvec wraps the user-specified rmatvec routine or overridden
_rmatvec method to ensure that y has the correct shape and type.
*)

val transpose : [> tag] Obj.t -> Py.Object.t
(**
Transpose this linear operator.

Returns a LinearOperator that represents the transpose of this one.
Can be abbreviated self.T instead of self.transpose().
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module IterOpInv : sig
type tag = [`IterOpInv]
type t = [`IterOpInv | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> t
(**
IterOpInv:
   helper class to repeatedly solve [A-sigma*M]*x = b
   using an iterative method
*)

val adjoint : [> tag] Obj.t -> Py.Object.t
(**
Hermitian adjoint.

Returns the Hermitian adjoint of self, aka the Hermitian
conjugate or Hermitian transpose. For a complex matrix, the
Hermitian adjoint is equal to the conjugate transpose.

Can be abbreviated self.H instead of self.adjoint().

Returns
-------
A_H : LinearOperator
    Hermitian adjoint of self.
*)

val dot : x:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Matrix-matrix or matrix-vector multiplication.

Parameters
----------
x : array_like
    1-d or 2-d array, representing a vector or matrix.

Returns
-------
Ax : array
    1-d or 2-d array (depending on the shape of x) that represents
    the result of applying this linear operator on x.
*)

val matmat : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Matrix-matrix multiplication.

Performs the operation y=A*X where A is an MxN linear
operator and X dense N*K matrix or ndarray.

Parameters
----------
X : {matrix, ndarray}
    An array with shape (N,K).

Returns
-------
Y : {matrix, ndarray}
    A matrix or ndarray with shape (M,K) depending on
    the type of the X argument.

Notes
-----
This matmat wraps any user-specified matmat routine or overridden
_matmat method to ensure that y has the correct type.
*)

val matvec : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Matrix-vector multiplication.

Performs the operation y=A*x where A is an MxN linear
operator and x is a column vector or 1-d array.

Parameters
----------
x : {matrix, ndarray}
    An array with shape (N,) or (N,1).

Returns
-------
y : {matrix, ndarray}
    A matrix or ndarray with shape (M,) or (M,1) depending
    on the type and shape of the x argument.

Notes
-----
This matvec wraps the user-specified matvec routine or overridden
_matvec method to ensure that y has the correct shape and type.
*)

val rmatmat : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Adjoint matrix-matrix multiplication.

Performs the operation y = A^H * x where A is an MxN linear
operator and x is a column vector or 1-d array, or 2-d array.
The default implementation defers to the adjoint.

Parameters
----------
X : {matrix, ndarray}
    A matrix or 2D array.

Returns
-------
Y : {matrix, ndarray}
    A matrix or 2D array depending on the type of the input.

Notes
-----
This rmatmat wraps the user-specified rmatmat routine.
*)

val rmatvec : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Adjoint matrix-vector multiplication.

Performs the operation y = A^H * x where A is an MxN linear
operator and x is a column vector or 1-d array.

Parameters
----------
x : {matrix, ndarray}
    An array with shape (M,) or (M,1).

Returns
-------
y : {matrix, ndarray}
    A matrix or ndarray with shape (N,) or (N,1) depending
    on the type and shape of the x argument.

Notes
-----
This rmatvec wraps the user-specified rmatvec routine or overridden
_rmatvec method to ensure that y has the correct shape and type.
*)

val transpose : [> tag] Obj.t -> Py.Object.t
(**
Transpose this linear operator.

Returns a LinearOperator that represents the transpose of this one.
Can be abbreviated self.T instead of self.transpose().
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module LuInv : sig
type tag = [`LuInv]
type t = [`LuInv | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> t
(**
LuInv:
   helper class to repeatedly solve M*x=b
   using an LU-decomposition of M
*)

val adjoint : [> tag] Obj.t -> Py.Object.t
(**
Hermitian adjoint.

Returns the Hermitian adjoint of self, aka the Hermitian
conjugate or Hermitian transpose. For a complex matrix, the
Hermitian adjoint is equal to the conjugate transpose.

Can be abbreviated self.H instead of self.adjoint().

Returns
-------
A_H : LinearOperator
    Hermitian adjoint of self.
*)

val dot : x:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Matrix-matrix or matrix-vector multiplication.

Parameters
----------
x : array_like
    1-d or 2-d array, representing a vector or matrix.

Returns
-------
Ax : array
    1-d or 2-d array (depending on the shape of x) that represents
    the result of applying this linear operator on x.
*)

val matmat : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Matrix-matrix multiplication.

Performs the operation y=A*X where A is an MxN linear
operator and X dense N*K matrix or ndarray.

Parameters
----------
X : {matrix, ndarray}
    An array with shape (N,K).

Returns
-------
Y : {matrix, ndarray}
    A matrix or ndarray with shape (M,K) depending on
    the type of the X argument.

Notes
-----
This matmat wraps any user-specified matmat routine or overridden
_matmat method to ensure that y has the correct type.
*)

val matvec : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Matrix-vector multiplication.

Performs the operation y=A*x where A is an MxN linear
operator and x is a column vector or 1-d array.

Parameters
----------
x : {matrix, ndarray}
    An array with shape (N,) or (N,1).

Returns
-------
y : {matrix, ndarray}
    A matrix or ndarray with shape (M,) or (M,1) depending
    on the type and shape of the x argument.

Notes
-----
This matvec wraps the user-specified matvec routine or overridden
_matvec method to ensure that y has the correct shape and type.
*)

val rmatmat : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Adjoint matrix-matrix multiplication.

Performs the operation y = A^H * x where A is an MxN linear
operator and x is a column vector or 1-d array, or 2-d array.
The default implementation defers to the adjoint.

Parameters
----------
X : {matrix, ndarray}
    A matrix or 2D array.

Returns
-------
Y : {matrix, ndarray}
    A matrix or 2D array depending on the type of the input.

Notes
-----
This rmatmat wraps the user-specified rmatmat routine.
*)

val rmatvec : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Adjoint matrix-vector multiplication.

Performs the operation y = A^H * x where A is an MxN linear
operator and x is a column vector or 1-d array.

Parameters
----------
x : {matrix, ndarray}
    An array with shape (M,) or (M,1).

Returns
-------
y : {matrix, ndarray}
    A matrix or ndarray with shape (N,) or (N,1) depending
    on the type and shape of the x argument.

Notes
-----
This rmatvec wraps the user-specified rmatvec routine or overridden
_rmatvec method to ensure that y has the correct shape and type.
*)

val transpose : [> tag] Obj.t -> Py.Object.t
(**
Transpose this linear operator.

Returns a LinearOperator that represents the transpose of this one.
Can be abbreviated self.T instead of self.transpose().
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module ReentrancyLock : sig
type tag = [`ReentrancyLock]
type t = [`Object | `ReentrancyLock] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : Py.Object.t -> t
(**
Threading lock that raises an exception for reentrant calls.

Calls from different threads are serialized, and nested calls from the
same thread result to an error.

The object can be used as a context manager, or to decorate functions
via the decorate() method.
*)

val decorate : func:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module SpLuInv : sig
type tag = [`SpLuInv]
type t = [`Object | `SpLuInv] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> t
(**
SpLuInv:
   helper class to repeatedly solve M*x=b
   using a sparse LU-decopposition of M
*)

val adjoint : [> tag] Obj.t -> Py.Object.t
(**
Hermitian adjoint.

Returns the Hermitian adjoint of self, aka the Hermitian
conjugate or Hermitian transpose. For a complex matrix, the
Hermitian adjoint is equal to the conjugate transpose.

Can be abbreviated self.H instead of self.adjoint().

Returns
-------
A_H : LinearOperator
    Hermitian adjoint of self.
*)

val dot : x:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Matrix-matrix or matrix-vector multiplication.

Parameters
----------
x : array_like
    1-d or 2-d array, representing a vector or matrix.

Returns
-------
Ax : array
    1-d or 2-d array (depending on the shape of x) that represents
    the result of applying this linear operator on x.
*)

val matmat : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Matrix-matrix multiplication.

Performs the operation y=A*X where A is an MxN linear
operator and X dense N*K matrix or ndarray.

Parameters
----------
X : {matrix, ndarray}
    An array with shape (N,K).

Returns
-------
Y : {matrix, ndarray}
    A matrix or ndarray with shape (M,K) depending on
    the type of the X argument.

Notes
-----
This matmat wraps any user-specified matmat routine or overridden
_matmat method to ensure that y has the correct type.
*)

val matvec : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Matrix-vector multiplication.

Performs the operation y=A*x where A is an MxN linear
operator and x is a column vector or 1-d array.

Parameters
----------
x : {matrix, ndarray}
    An array with shape (N,) or (N,1).

Returns
-------
y : {matrix, ndarray}
    A matrix or ndarray with shape (M,) or (M,1) depending
    on the type and shape of the x argument.

Notes
-----
This matvec wraps the user-specified matvec routine or overridden
_matvec method to ensure that y has the correct shape and type.
*)

val rmatmat : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Adjoint matrix-matrix multiplication.

Performs the operation y = A^H * x where A is an MxN linear
operator and x is a column vector or 1-d array, or 2-d array.
The default implementation defers to the adjoint.

Parameters
----------
X : {matrix, ndarray}
    A matrix or 2D array.

Returns
-------
Y : {matrix, ndarray}
    A matrix or 2D array depending on the type of the input.

Notes
-----
This rmatmat wraps the user-specified rmatmat routine.
*)

val rmatvec : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Adjoint matrix-vector multiplication.

Performs the operation y = A^H * x where A is an MxN linear
operator and x is a column vector or 1-d array.

Parameters
----------
x : {matrix, ndarray}
    An array with shape (M,) or (M,1).

Returns
-------
y : {matrix, ndarray}
    A matrix or ndarray with shape (N,) or (N,1) depending
    on the type and shape of the x argument.

Notes
-----
This rmatvec wraps the user-specified rmatvec routine or overridden
_rmatvec method to ensure that y has the correct shape and type.
*)

val transpose : [> tag] Obj.t -> Py.Object.t
(**
Transpose this linear operator.

Returns a LinearOperator that represents the transpose of this one.
Can be abbreviated self.T instead of self.transpose().
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val aslinearoperator : Py.Object.t -> Py.Object.t
(**
Return A as a LinearOperator.

'A' may be any of the following types:
 - ndarray
 - matrix
 - sparse matrix (e.g. csr_matrix, lil_matrix, etc.)
 - LinearOperator
 - An object with .shape and .matvec attributes

See the LinearOperator documentation for additional information.

Notes
-----
If 'A' has no .dtype attribute, the data type is determined by calling
:func:`LinearOperator.matvec()` - set the .dtype attribute to prevent this
call upon the linear operator creation.

Examples
--------
>>> from scipy.sparse.linalg import aslinearoperator
>>> M = np.array([[1,2,3],[4,5,6]], dtype=np.int32)
>>> aslinearoperator(M)
<2x3 MatrixLinearOperator with dtype=int32>
*)

val choose_ncv : Py.Object.t -> Py.Object.t
(**
Choose number of lanczos vectors based on target number
of singular/eigen values and vectors to compute, k.
*)

val eig : ?b:[>`Ndarray] Np.Obj.t -> ?left:bool -> ?right:bool -> ?overwrite_a:bool -> ?overwrite_b:bool -> ?check_finite:bool -> ?homogeneous_eigvals:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> (Py.Object.t * Py.Object.t * Py.Object.t)
(**
Solve an ordinary or generalized eigenvalue problem of a square matrix.

Find eigenvalues w and right or left eigenvectors of a general matrix::

    a   vr[:,i] = w[i]        b   vr[:,i]
    a.H vl[:,i] = w[i].conj() b.H vl[:,i]

where ``.H`` is the Hermitian conjugation.

Parameters
----------
a : (M, M) array_like
    A complex or real matrix whose eigenvalues and eigenvectors
    will be computed.
b : (M, M) array_like, optional
    Right-hand side matrix in a generalized eigenvalue problem.
    Default is None, identity matrix is assumed.
left : bool, optional
    Whether to calculate and return left eigenvectors.  Default is False.
right : bool, optional
    Whether to calculate and return right eigenvectors.  Default is True.
overwrite_a : bool, optional
    Whether to overwrite `a`; may improve performance.  Default is False.
overwrite_b : bool, optional
    Whether to overwrite `b`; may improve performance.  Default is False.
check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.
homogeneous_eigvals : bool, optional
    If True, return the eigenvalues in homogeneous coordinates.
    In this case ``w`` is a (2, M) array so that::

        w[1,i] a vr[:,i] = w[0,i] b vr[:,i]

    Default is False.

Returns
-------
w : (M,) or (2, M) double or complex ndarray
    The eigenvalues, each repeated according to its
    multiplicity. The shape is (M,) unless
    ``homogeneous_eigvals=True``.
vl : (M, M) double or complex ndarray
    The normalized left eigenvector corresponding to the eigenvalue
    ``w[i]`` is the column vl[:,i]. Only returned if ``left=True``.
vr : (M, M) double or complex ndarray
    The normalized right eigenvector corresponding to the eigenvalue
    ``w[i]`` is the column ``vr[:,i]``.  Only returned if ``right=True``.

Raises
------
LinAlgError
    If eigenvalue computation does not converge.

See Also
--------
eigvals : eigenvalues of general arrays
eigh : Eigenvalues and right eigenvectors for symmetric/Hermitian arrays.
eig_banded : eigenvalues and right eigenvectors for symmetric/Hermitian
    band matrices
eigh_tridiagonal : eigenvalues and right eiegenvectors for
    symmetric/Hermitian tridiagonal matrices

Examples
--------
>>> from scipy import linalg
>>> a = np.array([[0., -1.], [1., 0.]])
>>> linalg.eigvals(a)
array([0.+1.j, 0.-1.j])

>>> b = np.array([[0., 1.], [1., 1.]])
>>> linalg.eigvals(a, b)
array([ 1.+0.j, -1.+0.j])

>>> a = np.array([[3., 0., 0.], [0., 8., 0.], [0., 0., 7.]])
>>> linalg.eigvals(a, homogeneous_eigvals=True)
array([[3.+0.j, 8.+0.j, 7.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j]])

>>> a = np.array([[0., -1.], [1., 0.]])
>>> linalg.eigvals(a) == linalg.eig(a)[0]
array([ True,  True])
>>> linalg.eig(a, left=True, right=False)[1] # normalized left eigenvector
array([[-0.70710678+0.j        , -0.70710678-0.j        ],
       [-0.        +0.70710678j, -0.        -0.70710678j]])
>>> linalg.eig(a, left=False, right=True)[1] # normalized right eigenvector
array([[0.70710678+0.j        , 0.70710678-0.j        ],
       [0.        -0.70710678j, 0.        +0.70710678j]])
*)

val eigh : ?b:[>`Ndarray] Np.Obj.t -> ?lower:bool -> ?eigvals_only:bool -> ?overwrite_a:bool -> ?overwrite_b:bool -> ?turbo:bool -> ?eigvals:Py.Object.t -> ?type_:int -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * Py.Object.t)
(**
Solve an ordinary or generalized eigenvalue problem for a complex
Hermitian or real symmetric matrix.

Find eigenvalues w and optionally eigenvectors v of matrix `a`, where
`b` is positive definite::

                  a v[:,i] = w[i] b v[:,i]
    v[i,:].conj() a v[:,i] = w[i]
    v[i,:].conj() b v[:,i] = 1

Parameters
----------
a : (M, M) array_like
    A complex Hermitian or real symmetric matrix whose eigenvalues and
    eigenvectors will be computed.
b : (M, M) array_like, optional
    A complex Hermitian or real symmetric definite positive matrix in.
    If omitted, identity matrix is assumed.
lower : bool, optional
    Whether the pertinent array data is taken from the lower or upper
    triangle of `a`. (Default: lower)
eigvals_only : bool, optional
    Whether to calculate only eigenvalues and no eigenvectors.
    (Default: both are calculated)
turbo : bool, optional
    Use divide and conquer algorithm (faster but expensive in memory,
    only for generalized eigenvalue problem and if eigvals=None)
eigvals : tuple (lo, hi), optional
    Indexes of the smallest and largest (in ascending order) eigenvalues
    and corresponding eigenvectors to be returned: 0 <= lo <= hi <= M-1.
    If omitted, all eigenvalues and eigenvectors are returned.
type : int, optional
    Specifies the problem type to be solved:

       type = 1: a   v[:,i] = w[i] b v[:,i]

       type = 2: a b v[:,i] = w[i]   v[:,i]

       type = 3: b a v[:,i] = w[i]   v[:,i]
overwrite_a : bool, optional
    Whether to overwrite data in `a` (may improve performance)
overwrite_b : bool, optional
    Whether to overwrite data in `b` (may improve performance)
check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
w : (N,) float ndarray
    The N (1<=N<=M) selected eigenvalues, in ascending order, each
    repeated according to its multiplicity.
v : (M, N) complex ndarray
    (if eigvals_only == False)

    The normalized selected eigenvector corresponding to the
    eigenvalue w[i] is the column v[:,i].

    Normalization:

        type 1 and 3: v.conj() a      v  = w

        type 2: inv(v).conj() a  inv(v) = w

        type = 1 or 2: v.conj() b      v  = I

        type = 3: v.conj() inv(b) v  = I

Raises
------
LinAlgError
    If eigenvalue computation does not converge,
    an error occurred, or b matrix is not definite positive. Note that
    if input matrices are not symmetric or hermitian, no error is reported
    but results will be wrong.

See Also
--------
eigvalsh : eigenvalues of symmetric or Hermitian arrays
eig : eigenvalues and right eigenvectors for non-symmetric arrays
eigh : eigenvalues and right eigenvectors for symmetric/Hermitian arrays
eigh_tridiagonal : eigenvalues and right eiegenvectors for
    symmetric/Hermitian tridiagonal matrices

Notes
-----
This function does not check the input array for being hermitian/symmetric
in order to allow for representing arrays with only their upper/lower
triangular parts.

Examples
--------
>>> from scipy.linalg import eigh
>>> A = np.array([[6, 3, 1, 5], [3, 0, 5, 1], [1, 5, 6, 2], [5, 1, 2, 2]])
>>> w, v = eigh(A)
>>> np.allclose(A @ v - v @ np.diag(w), np.zeros((4, 4)))
True
*)

val eigs : ?k:int -> ?m:[`Arr of [>`ArrayLike] Np.Obj.t | `LinearOperator of Py.Object.t] -> ?sigma:Py.Object.t -> ?which:[`LM | `SM | `LR | `SR | `LI | `SI] -> ?v0:[>`Ndarray] Np.Obj.t -> ?ncv:int -> ?maxiter:int -> ?tol:float -> ?return_eigenvectors:bool -> ?minv:[`Arr of [>`ArrayLike] Np.Obj.t | `LinearOperator of Py.Object.t] -> ?oPinv:[`Arr of [>`ArrayLike] Np.Obj.t | `LinearOperator of Py.Object.t] -> ?oPpart:Py.Object.t -> a:[`Arr of [>`ArrayLike] Np.Obj.t | `LinearOperator of Py.Object.t] -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Find k eigenvalues and eigenvectors of the square matrix A.

Solves ``A * x[i] = w[i] * x[i]``, the standard eigenvalue problem
for w[i] eigenvalues with corresponding eigenvectors x[i].

If M is specified, solves ``A * x[i] = w[i] * M * x[i]``, the
generalized eigenvalue problem for w[i] eigenvalues
with corresponding eigenvectors x[i]

Parameters
----------
A : ndarray, sparse matrix or LinearOperator
    An array, sparse matrix, or LinearOperator representing
    the operation ``A * x``, where A is a real or complex square matrix.
k : int, optional
    The number of eigenvalues and eigenvectors desired.
    `k` must be smaller than N-1. It is not possible to compute all
    eigenvectors of a matrix.
M : ndarray, sparse matrix or LinearOperator, optional
    An array, sparse matrix, or LinearOperator representing
    the operation M*x for the generalized eigenvalue problem

        A * x = w * M * x.

    M must represent a real, symmetric matrix if A is real, and must
    represent a complex, hermitian matrix if A is complex. For best
    results, the data type of M should be the same as that of A.
    Additionally:

        If `sigma` is None, M is positive definite

        If sigma is specified, M is positive semi-definite

    If sigma is None, eigs requires an operator to compute the solution
    of the linear equation ``M * x = b``.  This is done internally via a
    (sparse) LU decomposition for an explicit matrix M, or via an
    iterative solver for a general linear operator.  Alternatively,
    the user can supply the matrix or operator Minv, which gives
    ``x = Minv * b = M^-1 * b``.
sigma : real or complex, optional
    Find eigenvalues near sigma using shift-invert mode.  This requires
    an operator to compute the solution of the linear system
    ``[A - sigma * M] * x = b``, where M is the identity matrix if
    unspecified. This is computed internally via a (sparse) LU
    decomposition for explicit matrices A & M, or via an iterative
    solver if either A or M is a general linear operator.
    Alternatively, the user can supply the matrix or operator OPinv,
    which gives ``x = OPinv * b = [A - sigma * M]^-1 * b``.
    For a real matrix A, shift-invert can either be done in imaginary
    mode or real mode, specified by the parameter OPpart ('r' or 'i').
    Note that when sigma is specified, the keyword 'which' (below)
    refers to the shifted eigenvalues ``w'[i]`` where:

        If A is real and OPpart == 'r' (default),
          ``w'[i] = 1/2 * [1/(w[i]-sigma) + 1/(w[i]-conj(sigma))]``.

        If A is real and OPpart == 'i',
          ``w'[i] = 1/2i * [1/(w[i]-sigma) - 1/(w[i]-conj(sigma))]``.

        If A is complex, ``w'[i] = 1/(w[i]-sigma)``.

v0 : ndarray, optional
    Starting vector for iteration.
    Default: random
ncv : int, optional
    The number of Lanczos vectors generated
    `ncv` must be greater than `k`; it is recommended that ``ncv > 2*k``.
    Default: ``min(n, max(2*k + 1, 20))``
which : str, ['LM' | 'SM' | 'LR' | 'SR' | 'LI' | 'SI'], optional
    Which `k` eigenvectors and eigenvalues to find:

        'LM' : largest magnitude

        'SM' : smallest magnitude

        'LR' : largest real part

        'SR' : smallest real part

        'LI' : largest imaginary part

        'SI' : smallest imaginary part

    When sigma != None, 'which' refers to the shifted eigenvalues w'[i]
    (see discussion in 'sigma', above).  ARPACK is generally better
    at finding large values than small values.  If small eigenvalues are
    desired, consider using shift-invert mode for better performance.
maxiter : int, optional
    Maximum number of Arnoldi update iterations allowed
    Default: ``n*10``
tol : float, optional
    Relative accuracy for eigenvalues (stopping criterion)
    The default value of 0 implies machine precision.
return_eigenvectors : bool, optional
    Return eigenvectors (True) in addition to eigenvalues
Minv : ndarray, sparse matrix or LinearOperator, optional
    See notes in M, above.
OPinv : ndarray, sparse matrix or LinearOperator, optional
    See notes in sigma, above.
OPpart : {'r' or 'i'}, optional
    See notes in sigma, above

Returns
-------
w : ndarray
    Array of k eigenvalues.
v : ndarray
    An array of `k` eigenvectors.
    ``v[:, i]`` is the eigenvector corresponding to the eigenvalue w[i].

Raises
------
ArpackNoConvergence
    When the requested convergence is not obtained.
    The currently converged eigenvalues and eigenvectors can be found
    as ``eigenvalues`` and ``eigenvectors`` attributes of the exception
    object.

See Also
--------
eigsh : eigenvalues and eigenvectors for symmetric matrix A
svds : singular value decomposition for a matrix A

Notes
-----
This function is a wrapper to the ARPACK [1]_ SNEUPD, DNEUPD, CNEUPD,
ZNEUPD, functions which use the Implicitly Restarted Arnoldi Method to
find the eigenvalues and eigenvectors [2]_.

References
----------
.. [1] ARPACK Software, http://www.caam.rice.edu/software/ARPACK/
.. [2] R. B. Lehoucq, D. C. Sorensen, and C. Yang,  ARPACK USERS GUIDE:
   Solution of Large Scale Eigenvalue Problems by Implicitly Restarted
   Arnoldi Methods. SIAM, Philadelphia, PA, 1998.

Examples
--------
Find 6 eigenvectors of the identity matrix:

>>> from scipy.sparse.linalg import eigs
>>> id = np.eye(13)
>>> vals, vecs = eigs(id, k=6)
>>> vals
array([ 1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j])
>>> vecs.shape
(13, 6)
*)

val eigsh : ?k:int -> ?m:Py.Object.t -> ?sigma:Py.Object.t -> ?which:Py.Object.t -> ?v0:Py.Object.t -> ?ncv:Py.Object.t -> ?maxiter:Py.Object.t -> ?tol:Py.Object.t -> ?return_eigenvectors:Py.Object.t -> ?minv:Py.Object.t -> ?oPinv:Py.Object.t -> ?mode:Py.Object.t -> a:[`Arr of [>`ArrayLike] Np.Obj.t | `LinearOperator of Py.Object.t] -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Find k eigenvalues and eigenvectors of the real symmetric square matrix
or complex hermitian matrix A.

Solves ``A * x[i] = w[i] * x[i]``, the standard eigenvalue problem for
w[i] eigenvalues with corresponding eigenvectors x[i].

If M is specified, solves ``A * x[i] = w[i] * M * x[i]``, the
generalized eigenvalue problem for w[i] eigenvalues
with corresponding eigenvectors x[i].

Parameters
----------
A : ndarray, sparse matrix or LinearOperator
    A square operator representing the operation ``A * x``, where ``A`` is
    real symmetric or complex hermitian. For buckling mode (see below)
    ``A`` must additionally be positive-definite.
k : int, optional
    The number of eigenvalues and eigenvectors desired.
    `k` must be smaller than N. It is not possible to compute all
    eigenvectors of a matrix.

Returns
-------
w : array
    Array of k eigenvalues.
v : array
    An array representing the `k` eigenvectors.  The column ``v[:, i]`` is
    the eigenvector corresponding to the eigenvalue ``w[i]``.

Other Parameters
----------------
M : An N x N matrix, array, sparse matrix, or linear operator representing
    the operation ``M @ x`` for the generalized eigenvalue problem

        A @ x = w * M @ x.

    M must represent a real, symmetric matrix if A is real, and must
    represent a complex, hermitian matrix if A is complex. For best
    results, the data type of M should be the same as that of A.
    Additionally:

        If sigma is None, M is symmetric positive definite.

        If sigma is specified, M is symmetric positive semi-definite.

        In buckling mode, M is symmetric indefinite.

    If sigma is None, eigsh requires an operator to compute the solution
    of the linear equation ``M @ x = b``. This is done internally via a
    (sparse) LU decomposition for an explicit matrix M, or via an
    iterative solver for a general linear operator.  Alternatively,
    the user can supply the matrix or operator Minv, which gives
    ``x = Minv @ b = M^-1 @ b``.
sigma : real
    Find eigenvalues near sigma using shift-invert mode.  This requires
    an operator to compute the solution of the linear system
    ``[A - sigma * M] x = b``, where M is the identity matrix if
    unspecified.  This is computed internally via a (sparse) LU
    decomposition for explicit matrices A & M, or via an iterative
    solver if either A or M is a general linear operator.
    Alternatively, the user can supply the matrix or operator OPinv,
    which gives ``x = OPinv @ b = [A - sigma * M]^-1 @ b``.
    Note that when sigma is specified, the keyword 'which' refers to
    the shifted eigenvalues ``w'[i]`` where:

        if mode == 'normal', ``w'[i] = 1 / (w[i] - sigma)``.

        if mode == 'cayley', ``w'[i] = (w[i] + sigma) / (w[i] - sigma)``.

        if mode == 'buckling', ``w'[i] = w[i] / (w[i] - sigma)``.

    (see further discussion in 'mode' below)
v0 : ndarray, optional
    Starting vector for iteration.
    Default: random
ncv : int, optional
    The number of Lanczos vectors generated ncv must be greater than k and
    smaller than n; it is recommended that ``ncv > 2*k``.
    Default: ``min(n, max(2*k + 1, 20))``
which : str ['LM' | 'SM' | 'LA' | 'SA' | 'BE']
    If A is a complex hermitian matrix, 'BE' is invalid.
    Which `k` eigenvectors and eigenvalues to find:

        'LM' : Largest (in magnitude) eigenvalues.

        'SM' : Smallest (in magnitude) eigenvalues.

        'LA' : Largest (algebraic) eigenvalues.

        'SA' : Smallest (algebraic) eigenvalues.

        'BE' : Half (k/2) from each end of the spectrum.

    When k is odd, return one more (k/2+1) from the high end.
    When sigma != None, 'which' refers to the shifted eigenvalues ``w'[i]``
    (see discussion in 'sigma', above).  ARPACK is generally better
    at finding large values than small values.  If small eigenvalues are
    desired, consider using shift-invert mode for better performance.
maxiter : int, optional
    Maximum number of Arnoldi update iterations allowed.
    Default: ``n*10``
tol : float
    Relative accuracy for eigenvalues (stopping criterion).
    The default value of 0 implies machine precision.
Minv : N x N matrix, array, sparse matrix, or LinearOperator
    See notes in M, above.
OPinv : N x N matrix, array, sparse matrix, or LinearOperator
    See notes in sigma, above.
return_eigenvectors : bool
    Return eigenvectors (True) in addition to eigenvalues.
    This value determines the order in which eigenvalues are sorted.
    The sort order is also dependent on the `which` variable.

        For which = 'LM' or 'SA':
            If `return_eigenvectors` is True, eigenvalues are sorted by
            algebraic value.

            If `return_eigenvectors` is False, eigenvalues are sorted by
            absolute value.

        For which = 'BE' or 'LA':
            eigenvalues are always sorted by algebraic value.

        For which = 'SM':
            If `return_eigenvectors` is True, eigenvalues are sorted by
            algebraic value.

            If `return_eigenvectors` is False, eigenvalues are sorted by
            decreasing absolute value.

mode : string ['normal' | 'buckling' | 'cayley']
    Specify strategy to use for shift-invert mode.  This argument applies
    only for real-valued A and sigma != None.  For shift-invert mode,
    ARPACK internally solves the eigenvalue problem
    ``OP * x'[i] = w'[i] * B * x'[i]``
    and transforms the resulting Ritz vectors x'[i] and Ritz values w'[i]
    into the desired eigenvectors and eigenvalues of the problem
    ``A * x[i] = w[i] * M * x[i]``.
    The modes are as follows:

        'normal' :
            OP = [A - sigma * M]^-1 @ M,
            B = M,
            w'[i] = 1 / (w[i] - sigma)

        'buckling' :
            OP = [A - sigma * M]^-1 @ A,
            B = A,
            w'[i] = w[i] / (w[i] - sigma)

        'cayley' :
            OP = [A - sigma * M]^-1 @ [A + sigma * M],
            B = M,
            w'[i] = (w[i] + sigma) / (w[i] - sigma)

    The choice of mode will affect which eigenvalues are selected by
    the keyword 'which', and can also impact the stability of
    convergence (see [2] for a discussion).

Raises
------
ArpackNoConvergence
    When the requested convergence is not obtained.

    The currently converged eigenvalues and eigenvectors can be found
    as ``eigenvalues`` and ``eigenvectors`` attributes of the exception
    object.

See Also
--------
eigs : eigenvalues and eigenvectors for a general (nonsymmetric) matrix A
svds : singular value decomposition for a matrix A

Notes
-----
This function is a wrapper to the ARPACK [1]_ SSEUPD and DSEUPD
functions which use the Implicitly Restarted Lanczos Method to
find the eigenvalues and eigenvectors [2]_.

References
----------
.. [1] ARPACK Software, http://www.caam.rice.edu/software/ARPACK/
.. [2] R. B. Lehoucq, D. C. Sorensen, and C. Yang,  ARPACK USERS GUIDE:
   Solution of Large Scale Eigenvalue Problems by Implicitly Restarted
   Arnoldi Methods. SIAM, Philadelphia, PA, 1998.

Examples
--------
>>> from scipy.sparse.linalg import eigsh
>>> identity = np.eye(13)
>>> eigenvalues, eigenvectors = eigsh(identity, k=6)
>>> eigenvalues
array([1., 1., 1., 1., 1., 1.])
>>> eigenvectors.shape
(13, 6)
*)

val eye : ?n:int -> ?k:int -> ?dtype:Np.Dtype.t -> ?format:string -> m:int -> unit -> Py.Object.t
(**
Sparse matrix with ones on diagonal

Returns a sparse (m x n) matrix where the k-th diagonal
is all ones and everything else is zeros.

Parameters
----------
m : int
    Number of rows in the matrix.
n : int, optional
    Number of columns. Default: `m`.
k : int, optional
    Diagonal to place ones on. Default: 0 (main diagonal).
dtype : dtype, optional
    Data type of the matrix.
format : str, optional
    Sparse format of the result, e.g. format='csr', etc.

Examples
--------
>>> from scipy import sparse
>>> sparse.eye(3).toarray()
array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]])
>>> sparse.eye(3, dtype=np.int8)
<3x3 sparse matrix of type '<class 'numpy.int8'>'
    with 3 stored elements (1 diagonals) in DIAgonal format>
*)

val get_OPinv_matvec : ?hermitian:Py.Object.t -> ?tol:Py.Object.t -> a:Py.Object.t -> m:Py.Object.t -> sigma:Py.Object.t -> unit -> Py.Object.t
(**
None
*)

val get_inv_matvec : ?hermitian:Py.Object.t -> ?tol:Py.Object.t -> m:Py.Object.t -> unit -> Py.Object.t
(**
None
*)

val gmres : ?x0:Py.Object.t -> ?tol:Py.Object.t -> ?restart:Py.Object.t -> ?maxiter:Py.Object.t -> ?m:Py.Object.t -> ?callback:Py.Object.t -> ?restrt:Py.Object.t -> ?atol:Py.Object.t -> ?callback_type:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Use Generalized Minimal RESidual iteration to solve ``Ax = b``.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real or complex N-by-N matrix of the linear system.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).

Returns
-------
x : {array, matrix}
    The converged solution.
info : int
    Provides convergence information:
      * 0  : successful exit
      * >0 : convergence to tolerance not achieved, number of iterations
      * <0 : illegal input or breakdown

Other parameters
----------------
x0 : {array, matrix}
    Starting guess for the solution (a vector of zeros by default).
tol, atol : float, optional
    Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
    The default for ``atol`` is ``'legacy'``, which emulates
    a different legacy behavior.

    .. warning::

       The default value for `atol` will be changed in a future release.
       For future compatibility, specify `atol` explicitly.
restart : int, optional
    Number of iterations between restarts. Larger values increase
    iteration cost, but may be necessary for convergence.
    Default is 20.
maxiter : int, optional
    Maximum number of iterations (restart cycles).  Iteration will stop
    after maxiter steps even if the specified tolerance has not been
    achieved.
M : {sparse matrix, dense matrix, LinearOperator}
    Inverse of the preconditioner of A.  M should approximate the
    inverse of A and be easy to solve for (see Notes).  Effective
    preconditioning dramatically improves the rate of convergence,
    which implies that fewer iterations are needed to reach a given
    error tolerance.  By default, no preconditioner is used.
callback : function
    User-supplied function to call after each iteration.  It is called
    as `callback(args)`, where `args` are selected by `callback_type`.
callback_type : {'x', 'pr_norm', 'legacy'}, optional
    Callback function argument requested:
      - ``x``: current iterate (ndarray), called on every restart
      - ``pr_norm``: relative (preconditioned) residual norm (float),
        called on every inner iteration
      - ``legacy`` (default): same as ``pr_norm``, but also changes the
        meaning of 'maxiter' to count inner iterations instead of restart
        cycles.
restrt : int, optional
    DEPRECATED - use `restart` instead.

See Also
--------
LinearOperator

Notes
-----
A preconditioner, P, is chosen such that P is close to A but easy to solve
for. The preconditioner parameter required by this routine is
``M = P^-1``. The inverse should preferably not be calculated
explicitly.  Rather, use the following template to produce M::

  # Construct a linear operator that computes P^-1 * x.
  import scipy.sparse.linalg as spla
  M_x = lambda x: spla.spsolve(P, x)
  M = spla.LinearOperator((n, n), M_x)

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import gmres
>>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
>>> b = np.array([2, 4, -1], dtype=float)
>>> x, exitCode = gmres(A, b)
>>> print(exitCode)            # 0 indicates successful convergence
0
>>> np.allclose(A.dot(x), b)
True
*)

val gmres_loose : a:Py.Object.t -> b:Py.Object.t -> tol:Py.Object.t -> unit -> Py.Object.t
(**
gmres with looser termination condition.
*)

val is_pydata_spmatrix : Py.Object.t -> Py.Object.t
(**
Check whether object is pydata/sparse matrix, avoiding importing the module.
*)

val isdense : Py.Object.t -> Py.Object.t
(**
None
*)

val issparse : Py.Object.t -> Py.Object.t
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

val isspmatrix : Py.Object.t -> Py.Object.t
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

val isspmatrix_csr : Py.Object.t -> Py.Object.t
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

val lobpcg : ?b:[`PyObject of Py.Object.t | `Spmatrix of [>`Spmatrix] Np.Obj.t] -> ?m:[`PyObject of Py.Object.t | `Spmatrix of [>`Spmatrix] Np.Obj.t] -> ?y:[`Ndarray of [>`Ndarray] Np.Obj.t | `PyObject of Py.Object.t] -> ?tol:[`Bool of bool | `S of string | `I of int | `F of float] -> ?maxiter:int -> ?largest:bool -> ?verbosityLevel:int -> ?retLambdaHistory:bool -> ?retResidualNormsHistory:bool -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> x:[`Ndarray of [>`Ndarray] Np.Obj.t | `PyObject of Py.Object.t] -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t * Py.Object.t * Py.Object.t)
(**
Locally Optimal Block Preconditioned Conjugate Gradient Method (LOBPCG)

LOBPCG is a preconditioned eigensolver for large symmetric positive
definite (SPD) generalized eigenproblems.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The symmetric linear operator of the problem, usually a
    sparse matrix.  Often called the 'stiffness matrix'.
X : ndarray, float32 or float64
    Initial approximation to the ``k`` eigenvectors (non-sparse). If `A`
    has ``shape=(n,n)`` then `X` should have shape ``shape=(n,k)``.
B : {dense matrix, sparse matrix, LinearOperator}, optional
    The right hand side operator in a generalized eigenproblem.
    By default, ``B = Identity``.  Often called the 'mass matrix'.
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
iteration by calling the 'standard' dense eigensolver, so if ``m`` is not
small enough compared to ``n``, it does not make sense to call the LOBPCG
code, but rather one should use the 'standard' eigensolver, e.g. numpy or
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

val lu_factor : ?overwrite_a:bool -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Compute pivoted LU decomposition of a matrix.

The decomposition is::

    A = P L U

where P is a permutation matrix, L lower triangular with unit
diagonal elements, and U upper triangular.

Parameters
----------
a : (M, M) array_like
    Matrix to decompose
overwrite_a : bool, optional
    Whether to overwrite data in A (may increase performance)
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
lu : (N, N) ndarray
    Matrix containing U in its upper triangle, and L in its lower triangle.
    The unit diagonal elements of L are not stored.
piv : (N,) ndarray
    Pivot indices representing the permutation matrix P:
    row i of matrix was interchanged with row piv[i].

See also
--------
lu_solve : solve an equation system using the LU factorization of a matrix

Notes
-----
This is a wrapper to the ``*GETRF`` routines from LAPACK.

Examples
--------
>>> from scipy.linalg import lu_factor
>>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
>>> lu, piv = lu_factor(A)
>>> piv
array([2, 2, 3, 3], dtype=int32)

Convert LAPACK's ``piv`` array to NumPy index and test the permutation 

>>> piv_py = [2, 0, 3, 1]
>>> L, U = np.tril(lu, k=-1) + np.eye(4), np.triu(lu)
>>> np.allclose(A[piv_py] - L @ U, np.zeros((4, 4)))
True
*)

val lu_solve : ?trans:[`Zero | `One | `Two] -> ?overwrite_b:bool -> ?check_finite:bool -> lu_and_piv:Py.Object.t -> b:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Solve an equation system, a x = b, given the LU factorization of a

Parameters
----------
(lu, piv)
    Factorization of the coefficient matrix a, as given by lu_factor
b : array
    Right-hand side
trans : {0, 1, 2}, optional
    Type of system to solve:

    =====  =========
    trans  system
    =====  =========
    0      a x   = b
    1      a^T x = b
    2      a^H x = b
    =====  =========
overwrite_b : bool, optional
    Whether to overwrite data in b (may increase performance)
check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
x : array
    Solution to the system

See also
--------
lu_factor : LU factorize a matrix

Examples
--------
>>> from scipy.linalg import lu_factor, lu_solve
>>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
>>> b = np.array([1, 1, 1, 1])
>>> lu, piv = lu_factor(A)
>>> x = lu_solve((lu, piv), b)
>>> np.allclose(A @ x - b, np.zeros((4,)))
True
*)

val splu : ?permc_spec:string -> ?diag_pivot_thresh:float -> ?relax:int -> ?panel_size:int -> ?options:Py.Object.t -> a:[>`Spmatrix] Np.Obj.t -> unit -> Py.Object.t
(**
Compute the LU decomposition of a sparse, square matrix.

Parameters
----------
A : sparse matrix
    Sparse matrix to factorize. Should be in CSR or CSC format.
permc_spec : str, optional
    How to permute the columns of the matrix for sparsity preservation.
    (default: 'COLAMD')

    - ``NATURAL``: natural ordering.
    - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
    - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
    - ``COLAMD``: approximate minimum degree column ordering

diag_pivot_thresh : float, optional
    Threshold used for a diagonal entry to be an acceptable pivot.
    See SuperLU user's guide for details [1]_
relax : int, optional
    Expert option for customizing the degree of relaxing supernodes.
    See SuperLU user's guide for details [1]_
panel_size : int, optional
    Expert option for customizing the panel size.
    See SuperLU user's guide for details [1]_
options : dict, optional
    Dictionary containing additional expert options to SuperLU.
    See SuperLU user guide [1]_ (section 2.4 on the 'Options' argument)
    for more details. For example, you can specify
    ``options=dict(Equil=False, IterRefine='SINGLE'))``
    to turn equilibration off and perform a single iterative refinement.

Returns
-------
invA : scipy.sparse.linalg.SuperLU
    Object, which has a ``solve`` method.

See also
--------
spilu : incomplete LU decomposition

Notes
-----
This function uses the SuperLU library.

References
----------
.. [1] SuperLU http://crd.lbl.gov/~xiaoye/SuperLU/

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import splu
>>> A = csc_matrix([[1., 0., 0.], [5., 0., 2.], [0., -1., 0.]], dtype=float)
>>> B = splu(A)
>>> x = np.array([1., 2., 3.], dtype=float)
>>> B.solve(x)
array([ 1. , -3. , -1.5])
>>> A.dot(B.solve(x))
array([ 1.,  2.,  3.])
>>> B.solve(A.dot(x))
array([ 1.,  2.,  3.])
*)

val svds : ?k:int -> ?ncv:int -> ?tol:float -> ?which:[`LM | `SM] -> ?v0:[>`Ndarray] Np.Obj.t -> ?maxiter:int -> ?return_singular_vectors:[`Bool of bool | `S of string] -> ?solver:string -> a:[`LinearOperator of Py.Object.t | `Spmatrix of [>`Spmatrix] Np.Obj.t] -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Compute the largest or smallest k singular values/vectors for a sparse matrix. The order of the singular values is not guaranteed.

Parameters
----------
A : {sparse matrix, LinearOperator}
    Array to compute the SVD on, of shape (M, N)
k : int, optional
    Number of singular values and vectors to compute.
    Must be 1 <= k < min(A.shape).
ncv : int, optional
    The number of Lanczos vectors generated
    ncv must be greater than k+1 and smaller than n;
    it is recommended that ncv > 2*k
    Default: ``min(n, max(2*k + 1, 20))``
tol : float, optional
    Tolerance for singular values. Zero (default) means machine precision.
which : str, ['LM' | 'SM'], optional
    Which `k` singular values to find:

        - 'LM' : largest singular values
        - 'SM' : smallest singular values

    .. versionadded:: 0.12.0
v0 : ndarray, optional
    Starting vector for iteration, of length min(A.shape). Should be an
    (approximate) left singular vector if N > M and a right singular
    vector otherwise.
    Default: random

    .. versionadded:: 0.12.0
maxiter : int, optional
    Maximum number of iterations.

    .. versionadded:: 0.12.0
return_singular_vectors : bool or str, optional
    - True: return singular vectors (True) in addition to singular values.

    .. versionadded:: 0.12.0

    - 'u': only return the u matrix, without computing vh (if N > M).
    - 'vh': only return the vh matrix, without computing u (if N <= M).

    .. versionadded:: 0.16.0
solver : str, optional
        Eigenvalue solver to use. Should be 'arpack' or 'lobpcg'.
        Default: 'arpack'

Returns
-------
u : ndarray, shape=(M, k)
    Unitary matrix having left singular vectors as columns.
    If `return_singular_vectors` is 'vh', this variable is not computed,
    and None is returned instead.
s : ndarray, shape=(k,)
    The singular values.
vt : ndarray, shape=(k, N)
    Unitary matrix having right singular vectors as rows.
    If `return_singular_vectors` is 'u', this variable is not computed,
    and None is returned instead.


Notes
-----
This is a naive implementation using ARPACK or LOBPCG as an eigensolver
on A.H * A or A * A.H, depending on which one is more efficient.

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import svds, eigs
>>> A = csc_matrix([[1, 0, 0], [5, 0, 2], [0, -1, 0], [0, 0, 3]], dtype=float)
>>> u, s, vt = svds(A, k=2)
>>> s
array([ 2.75193379,  5.6059665 ])
>>> np.sqrt(eigs(A.dot(A.T), k=2)[0]).real
array([ 5.6059665 ,  2.75193379])
*)


end

module Dsolve : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module Linsolve : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val asarray : ?dtype:Np.Dtype.t -> ?order:[`F | `C] -> a:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Convert the input to an array.

Parameters
----------
a : array_like
    Input data, in any form that can be converted to an array.  This
    includes lists, lists of tuples, tuples, tuples of tuples, tuples
    of lists and ndarrays.
dtype : data-type, optional
    By default, the data-type is inferred from the input data.
order : {'C', 'F'}, optional
    Whether to use row-major (C-style) or
    column-major (Fortran-style) memory representation.
    Defaults to 'C'.

Returns
-------
out : ndarray
    Array interpretation of `a`.  No copy is performed if the input
    is already an ndarray with matching dtype and order.  If `a` is a
    subclass of ndarray, a base class ndarray is returned.

See Also
--------
asanyarray : Similar function which passes through subclasses.
ascontiguousarray : Convert input to a contiguous array.
asfarray : Convert input to a floating point ndarray.
asfortranarray : Convert input to an ndarray with column-major
                 memory order.
asarray_chkfinite : Similar function which checks input for NaNs and Infs.
fromiter : Create an array from an iterator.
fromfunction : Construct an array by executing a function on grid
               positions.

Examples
--------
Convert a list into an array:

>>> a = [1, 2]
>>> np.asarray(a)
array([1, 2])

Existing arrays are not copied:

>>> a = np.array([1, 2])
>>> np.asarray(a) is a
True

If `dtype` is set, array is copied only if dtype does not match:

>>> a = np.array([1, 2], dtype=np.float32)
>>> np.asarray(a, dtype=np.float32) is a
True
>>> np.asarray(a, dtype=np.float64) is a
False

Contrary to `asanyarray`, ndarray subclasses are not passed through:

>>> issubclass(np.recarray, np.ndarray)
True
>>> a = np.array([(1.0, 2), (3.0, 4)], dtype='f4,i4').view(np.recarray)
>>> np.asarray(a) is a
False
>>> np.asanyarray(a) is a
True
*)

val factorized : [>`Ndarray] Np.Obj.t -> Py.Object.t
(**
Return a function for solving a sparse linear system, with A pre-factorized.

Parameters
----------
A : (N, N) array_like
    Input.

Returns
-------
solve : callable
    To solve the linear system of equations given in `A`, the `solve`
    callable should be passed an ndarray of shape (N,).

Examples
--------
>>> from scipy.sparse.linalg import factorized
>>> A = np.array([[ 3. ,  2. , -1. ],
...               [ 2. , -2. ,  4. ],
...               [-1. ,  0.5, -1. ]])
>>> solve = factorized(A) # Makes LU decomposition.
>>> rhs1 = np.array([1, -2, 0])
>>> solve(rhs1) # Uses the LU factors.
array([ 1., -2., -2.])
*)

val is_pydata_spmatrix : Py.Object.t -> Py.Object.t
(**
Check whether object is pydata/sparse matrix, avoiding importing the module.
*)

val isspmatrix : Py.Object.t -> Py.Object.t
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

val isspmatrix_csc : Py.Object.t -> Py.Object.t
(**
Is x of csc_matrix type?

Parameters
----------
x
    object to check for being a csc matrix

Returns
-------
bool
    True if x is a csc matrix, False otherwise

Examples
--------
>>> from scipy.sparse import csc_matrix, isspmatrix_csc
>>> isspmatrix_csc(csc_matrix([[5]]))
True

>>> from scipy.sparse import csc_matrix, csr_matrix, isspmatrix_csc
>>> isspmatrix_csc(csr_matrix([[5]]))
False
*)

val isspmatrix_csr : Py.Object.t -> Py.Object.t
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

val spilu : ?drop_tol:float -> ?fill_factor:float -> ?drop_rule:string -> ?permc_spec:Py.Object.t -> ?diag_pivot_thresh:Py.Object.t -> ?relax:Py.Object.t -> ?panel_size:Py.Object.t -> ?options:Py.Object.t -> a:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Compute an incomplete LU decomposition for a sparse, square matrix.

The resulting object is an approximation to the inverse of `A`.

Parameters
----------
A : (N, N) array_like
    Sparse matrix to factorize
drop_tol : float, optional
    Drop tolerance (0 <= tol <= 1) for an incomplete LU decomposition.
    (default: 1e-4)
fill_factor : float, optional
    Specifies the fill ratio upper bound (>= 1.0) for ILU. (default: 10)
drop_rule : str, optional
    Comma-separated string of drop rules to use.
    Available rules: ``basic``, ``prows``, ``column``, ``area``,
    ``secondary``, ``dynamic``, ``interp``. (Default: ``basic,area``)

    See SuperLU documentation for details.

Remaining other options
    Same as for `splu`

Returns
-------
invA_approx : scipy.sparse.linalg.SuperLU
    Object, which has a ``solve`` method.

See also
--------
splu : complete LU decomposition

Notes
-----
To improve the better approximation to the inverse, you may need to
increase `fill_factor` AND decrease `drop_tol`.

This function uses the SuperLU library.

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import spilu
>>> A = csc_matrix([[1., 0., 0.], [5., 0., 2.], [0., -1., 0.]], dtype=float)
>>> B = spilu(A)
>>> x = np.array([1., 2., 3.], dtype=float)
>>> B.solve(x)
array([ 1. , -3. , -1.5])
>>> A.dot(B.solve(x))
array([ 1.,  2.,  3.])
>>> B.solve(A.dot(x))
array([ 1.,  2.,  3.])
*)

val splu : ?permc_spec:string -> ?diag_pivot_thresh:float -> ?relax:int -> ?panel_size:int -> ?options:Py.Object.t -> a:[>`Spmatrix] Np.Obj.t -> unit -> Py.Object.t
(**
Compute the LU decomposition of a sparse, square matrix.

Parameters
----------
A : sparse matrix
    Sparse matrix to factorize. Should be in CSR or CSC format.
permc_spec : str, optional
    How to permute the columns of the matrix for sparsity preservation.
    (default: 'COLAMD')

    - ``NATURAL``: natural ordering.
    - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
    - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
    - ``COLAMD``: approximate minimum degree column ordering

diag_pivot_thresh : float, optional
    Threshold used for a diagonal entry to be an acceptable pivot.
    See SuperLU user's guide for details [1]_
relax : int, optional
    Expert option for customizing the degree of relaxing supernodes.
    See SuperLU user's guide for details [1]_
panel_size : int, optional
    Expert option for customizing the panel size.
    See SuperLU user's guide for details [1]_
options : dict, optional
    Dictionary containing additional expert options to SuperLU.
    See SuperLU user guide [1]_ (section 2.4 on the 'Options' argument)
    for more details. For example, you can specify
    ``options=dict(Equil=False, IterRefine='SINGLE'))``
    to turn equilibration off and perform a single iterative refinement.

Returns
-------
invA : scipy.sparse.linalg.SuperLU
    Object, which has a ``solve`` method.

See also
--------
spilu : incomplete LU decomposition

Notes
-----
This function uses the SuperLU library.

References
----------
.. [1] SuperLU http://crd.lbl.gov/~xiaoye/SuperLU/

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import splu
>>> A = csc_matrix([[1., 0., 0.], [5., 0., 2.], [0., -1., 0.]], dtype=float)
>>> B = splu(A)
>>> x = np.array([1., 2., 3.], dtype=float)
>>> B.solve(x)
array([ 1. , -3. , -1.5])
>>> A.dot(B.solve(x))
array([ 1.,  2.,  3.])
>>> B.solve(A.dot(x))
array([ 1.,  2.,  3.])
*)

val spsolve : ?permc_spec:string -> ?use_umfpack:bool -> a:[>`ArrayLike] Np.Obj.t -> b:[>`ArrayLike] Np.Obj.t -> unit -> [>`ArrayLike] Np.Obj.t
(**
Solve the sparse linear system Ax=b, where b may be a vector or a matrix.

Parameters
----------
A : ndarray or sparse matrix
    The square matrix A will be converted into CSC or CSR form
b : ndarray or sparse matrix
    The matrix or vector representing the right hand side of the equation.
    If a vector, b.shape must be (n,) or (n, 1).
permc_spec : str, optional
    How to permute the columns of the matrix for sparsity preservation.
    (default: 'COLAMD')

    - ``NATURAL``: natural ordering.
    - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
    - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
    - ``COLAMD``: approximate minimum degree column ordering
use_umfpack : bool, optional
    if True (default) then use umfpack for the solution.  This is
    only referenced if b is a vector and ``scikit-umfpack`` is installed.

Returns
-------
x : ndarray or sparse matrix
    the solution of the sparse linear equation.
    If b is a vector, then x is a vector of size A.shape[1]
    If b is a matrix, then x is a matrix of size (A.shape[1], b.shape[1])

Notes
-----
For solving the matrix expression AX = B, this solver assumes the resulting
matrix X is sparse, as is often the case for very sparse inputs.  If the
resulting X is dense, the construction of this sparse result will be
relatively expensive.  In that case, consider converting A to a dense
matrix and using scipy.linalg.solve or its variants.

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import spsolve
>>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
>>> B = csc_matrix([[2, 0], [-1, 0], [2, 0]], dtype=float)
>>> x = spsolve(A, B)
>>> np.allclose(A.dot(x).todense(), B.todense())
True
*)

val spsolve_triangular : ?lower:bool -> ?overwrite_A:bool -> ?overwrite_b:bool -> ?unit_diagonal:bool -> a:[>`Spmatrix] Np.Obj.t -> b:Py.Object.t -> unit -> Py.Object.t
(**
Solve the equation `A x = b` for `x`, assuming A is a triangular matrix.

Parameters
----------
A : (M, M) sparse matrix
    A sparse square triangular matrix. Should be in CSR format.
b : (M,) or (M, N) array_like
    Right-hand side matrix in `A x = b`
lower : bool, optional
    Whether `A` is a lower or upper triangular matrix.
    Default is lower triangular matrix.
overwrite_A : bool, optional
    Allow changing `A`. The indices of `A` are going to be sorted and zero
    entries are going to be removed.
    Enabling gives a performance gain. Default is False.
overwrite_b : bool, optional
    Allow overwriting data in `b`.
    Enabling gives a performance gain. Default is False.
    If `overwrite_b` is True, it should be ensured that
    `b` has an appropriate dtype to be able to store the result.
unit_diagonal : bool, optional
    If True, diagonal elements of `a` are assumed to be 1 and will not be
    referenced.

    .. versionadded:: 1.4.0

Returns
-------
x : (M,) or (M, N) ndarray
    Solution to the system `A x = b`. Shape of return matches shape of `b`.

Raises
------
LinAlgError
    If `A` is singular or not triangular.
ValueError
    If shape of `A` or shape of `b` do not match the requirements.

Notes
-----
.. versionadded:: 0.19.0

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> from scipy.sparse.linalg import spsolve_triangular
>>> A = csr_matrix([[3, 0, 0], [1, -1, 0], [2, 0, 1]], dtype=float)
>>> B = np.array([[2, 0], [-1, 0], [2, 0]], dtype=float)
>>> x = spsolve_triangular(A, B)
>>> np.allclose(A.dot(x), B)
True
*)

val use_solver : ?kwargs:(string * Py.Object.t) list -> unit -> Py.Object.t
(**
Select default sparse direct solver to be used.

Parameters
----------
useUmfpack : bool, optional
    Use UMFPACK over SuperLU. Has effect only if scikits.umfpack is
    installed. Default: True
assumeSortedIndices : bool, optional
    Allow UMFPACK to skip the step of sorting indices for a CSR/CSC matrix.
    Has effect only if useUmfpack is True and scikits.umfpack is installed.
    Default: False

Notes
-----
The default sparse solver is umfpack when available
(scikits.umfpack is installed). This can be changed by passing
useUmfpack = False, which then causes the always present SuperLU
based solver to be used.

Umfpack requires a CSR/CSC matrix to have sorted column/row indices. If
sure that the matrix fulfills this, pass ``assumeSortedIndices=True``
to gain some speed.
*)

val warn : ?category:Py.Object.t -> ?stacklevel:Py.Object.t -> ?source:Py.Object.t -> message:Py.Object.t -> unit -> Py.Object.t
(**
Issue a warning, or maybe ignore it or raise an exception.
*)


end

val factorized : [>`Ndarray] Np.Obj.t -> Py.Object.t
(**
Return a function for solving a sparse linear system, with A pre-factorized.

Parameters
----------
A : (N, N) array_like
    Input.

Returns
-------
solve : callable
    To solve the linear system of equations given in `A`, the `solve`
    callable should be passed an ndarray of shape (N,).

Examples
--------
>>> from scipy.sparse.linalg import factorized
>>> A = np.array([[ 3. ,  2. , -1. ],
...               [ 2. , -2. ,  4. ],
...               [-1. ,  0.5, -1. ]])
>>> solve = factorized(A) # Makes LU decomposition.
>>> rhs1 = np.array([1, -2, 0])
>>> solve(rhs1) # Uses the LU factors.
array([ 1., -2., -2.])
*)

val spilu : ?drop_tol:float -> ?fill_factor:float -> ?drop_rule:string -> ?permc_spec:Py.Object.t -> ?diag_pivot_thresh:Py.Object.t -> ?relax:Py.Object.t -> ?panel_size:Py.Object.t -> ?options:Py.Object.t -> a:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Compute an incomplete LU decomposition for a sparse, square matrix.

The resulting object is an approximation to the inverse of `A`.

Parameters
----------
A : (N, N) array_like
    Sparse matrix to factorize
drop_tol : float, optional
    Drop tolerance (0 <= tol <= 1) for an incomplete LU decomposition.
    (default: 1e-4)
fill_factor : float, optional
    Specifies the fill ratio upper bound (>= 1.0) for ILU. (default: 10)
drop_rule : str, optional
    Comma-separated string of drop rules to use.
    Available rules: ``basic``, ``prows``, ``column``, ``area``,
    ``secondary``, ``dynamic``, ``interp``. (Default: ``basic,area``)

    See SuperLU documentation for details.

Remaining other options
    Same as for `splu`

Returns
-------
invA_approx : scipy.sparse.linalg.SuperLU
    Object, which has a ``solve`` method.

See also
--------
splu : complete LU decomposition

Notes
-----
To improve the better approximation to the inverse, you may need to
increase `fill_factor` AND decrease `drop_tol`.

This function uses the SuperLU library.

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import spilu
>>> A = csc_matrix([[1., 0., 0.], [5., 0., 2.], [0., -1., 0.]], dtype=float)
>>> B = spilu(A)
>>> x = np.array([1., 2., 3.], dtype=float)
>>> B.solve(x)
array([ 1. , -3. , -1.5])
>>> A.dot(B.solve(x))
array([ 1.,  2.,  3.])
>>> B.solve(A.dot(x))
array([ 1.,  2.,  3.])
*)

val splu : ?permc_spec:string -> ?diag_pivot_thresh:float -> ?relax:int -> ?panel_size:int -> ?options:Py.Object.t -> a:[>`Spmatrix] Np.Obj.t -> unit -> Py.Object.t
(**
Compute the LU decomposition of a sparse, square matrix.

Parameters
----------
A : sparse matrix
    Sparse matrix to factorize. Should be in CSR or CSC format.
permc_spec : str, optional
    How to permute the columns of the matrix for sparsity preservation.
    (default: 'COLAMD')

    - ``NATURAL``: natural ordering.
    - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
    - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
    - ``COLAMD``: approximate minimum degree column ordering

diag_pivot_thresh : float, optional
    Threshold used for a diagonal entry to be an acceptable pivot.
    See SuperLU user's guide for details [1]_
relax : int, optional
    Expert option for customizing the degree of relaxing supernodes.
    See SuperLU user's guide for details [1]_
panel_size : int, optional
    Expert option for customizing the panel size.
    See SuperLU user's guide for details [1]_
options : dict, optional
    Dictionary containing additional expert options to SuperLU.
    See SuperLU user guide [1]_ (section 2.4 on the 'Options' argument)
    for more details. For example, you can specify
    ``options=dict(Equil=False, IterRefine='SINGLE'))``
    to turn equilibration off and perform a single iterative refinement.

Returns
-------
invA : scipy.sparse.linalg.SuperLU
    Object, which has a ``solve`` method.

See also
--------
spilu : incomplete LU decomposition

Notes
-----
This function uses the SuperLU library.

References
----------
.. [1] SuperLU http://crd.lbl.gov/~xiaoye/SuperLU/

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import splu
>>> A = csc_matrix([[1., 0., 0.], [5., 0., 2.], [0., -1., 0.]], dtype=float)
>>> B = splu(A)
>>> x = np.array([1., 2., 3.], dtype=float)
>>> B.solve(x)
array([ 1. , -3. , -1.5])
>>> A.dot(B.solve(x))
array([ 1.,  2.,  3.])
>>> B.solve(A.dot(x))
array([ 1.,  2.,  3.])
*)

val spsolve : ?permc_spec:string -> ?use_umfpack:bool -> a:[>`ArrayLike] Np.Obj.t -> b:[>`ArrayLike] Np.Obj.t -> unit -> [>`ArrayLike] Np.Obj.t
(**
Solve the sparse linear system Ax=b, where b may be a vector or a matrix.

Parameters
----------
A : ndarray or sparse matrix
    The square matrix A will be converted into CSC or CSR form
b : ndarray or sparse matrix
    The matrix or vector representing the right hand side of the equation.
    If a vector, b.shape must be (n,) or (n, 1).
permc_spec : str, optional
    How to permute the columns of the matrix for sparsity preservation.
    (default: 'COLAMD')

    - ``NATURAL``: natural ordering.
    - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
    - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
    - ``COLAMD``: approximate minimum degree column ordering
use_umfpack : bool, optional
    if True (default) then use umfpack for the solution.  This is
    only referenced if b is a vector and ``scikit-umfpack`` is installed.

Returns
-------
x : ndarray or sparse matrix
    the solution of the sparse linear equation.
    If b is a vector, then x is a vector of size A.shape[1]
    If b is a matrix, then x is a matrix of size (A.shape[1], b.shape[1])

Notes
-----
For solving the matrix expression AX = B, this solver assumes the resulting
matrix X is sparse, as is often the case for very sparse inputs.  If the
resulting X is dense, the construction of this sparse result will be
relatively expensive.  In that case, consider converting A to a dense
matrix and using scipy.linalg.solve or its variants.

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import spsolve
>>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
>>> B = csc_matrix([[2, 0], [-1, 0], [2, 0]], dtype=float)
>>> x = spsolve(A, B)
>>> np.allclose(A.dot(x).todense(), B.todense())
True
*)

val spsolve_triangular : ?lower:bool -> ?overwrite_A:bool -> ?overwrite_b:bool -> ?unit_diagonal:bool -> a:[>`Spmatrix] Np.Obj.t -> b:Py.Object.t -> unit -> Py.Object.t
(**
Solve the equation `A x = b` for `x`, assuming A is a triangular matrix.

Parameters
----------
A : (M, M) sparse matrix
    A sparse square triangular matrix. Should be in CSR format.
b : (M,) or (M, N) array_like
    Right-hand side matrix in `A x = b`
lower : bool, optional
    Whether `A` is a lower or upper triangular matrix.
    Default is lower triangular matrix.
overwrite_A : bool, optional
    Allow changing `A`. The indices of `A` are going to be sorted and zero
    entries are going to be removed.
    Enabling gives a performance gain. Default is False.
overwrite_b : bool, optional
    Allow overwriting data in `b`.
    Enabling gives a performance gain. Default is False.
    If `overwrite_b` is True, it should be ensured that
    `b` has an appropriate dtype to be able to store the result.
unit_diagonal : bool, optional
    If True, diagonal elements of `a` are assumed to be 1 and will not be
    referenced.

    .. versionadded:: 1.4.0

Returns
-------
x : (M,) or (M, N) ndarray
    Solution to the system `A x = b`. Shape of return matches shape of `b`.

Raises
------
LinAlgError
    If `A` is singular or not triangular.
ValueError
    If shape of `A` or shape of `b` do not match the requirements.

Notes
-----
.. versionadded:: 0.19.0

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> from scipy.sparse.linalg import spsolve_triangular
>>> A = csr_matrix([[3, 0, 0], [1, -1, 0], [2, 0, 1]], dtype=float)
>>> B = np.array([[2, 0], [-1, 0], [2, 0]], dtype=float)
>>> x = spsolve_triangular(A, B)
>>> np.allclose(A.dot(x), B)
True
*)

val use_solver : ?kwargs:(string * Py.Object.t) list -> unit -> Py.Object.t
(**
Select default sparse direct solver to be used.

Parameters
----------
useUmfpack : bool, optional
    Use UMFPACK over SuperLU. Has effect only if scikits.umfpack is
    installed. Default: True
assumeSortedIndices : bool, optional
    Allow UMFPACK to skip the step of sorting indices for a CSR/CSC matrix.
    Has effect only if useUmfpack is True and scikits.umfpack is installed.
    Default: False

Notes
-----
The default sparse solver is umfpack when available
(scikits.umfpack is installed). This can be changed by passing
useUmfpack = False, which then causes the always present SuperLU
based solver to be used.

Umfpack requires a CSR/CSC matrix to have sorted column/row indices. If
sure that the matrix fulfills this, pass ``assumeSortedIndices=True``
to gain some speed.
*)


end

module Eigen : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module Arpack : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val aslinearoperator : Py.Object.t -> Py.Object.t
(**
Return A as a LinearOperator.

'A' may be any of the following types:
 - ndarray
 - matrix
 - sparse matrix (e.g. csr_matrix, lil_matrix, etc.)
 - LinearOperator
 - An object with .shape and .matvec attributes

See the LinearOperator documentation for additional information.

Notes
-----
If 'A' has no .dtype attribute, the data type is determined by calling
:func:`LinearOperator.matvec()` - set the .dtype attribute to prevent this
call upon the linear operator creation.

Examples
--------
>>> from scipy.sparse.linalg import aslinearoperator
>>> M = np.array([[1,2,3],[4,5,6]], dtype=np.int32)
>>> aslinearoperator(M)
<2x3 MatrixLinearOperator with dtype=int32>
*)

val choose_ncv : Py.Object.t -> Py.Object.t
(**
Choose number of lanczos vectors based on target number
of singular/eigen values and vectors to compute, k.
*)

val eig : ?b:[>`Ndarray] Np.Obj.t -> ?left:bool -> ?right:bool -> ?overwrite_a:bool -> ?overwrite_b:bool -> ?check_finite:bool -> ?homogeneous_eigvals:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> (Py.Object.t * Py.Object.t * Py.Object.t)
(**
Solve an ordinary or generalized eigenvalue problem of a square matrix.

Find eigenvalues w and right or left eigenvectors of a general matrix::

    a   vr[:,i] = w[i]        b   vr[:,i]
    a.H vl[:,i] = w[i].conj() b.H vl[:,i]

where ``.H`` is the Hermitian conjugation.

Parameters
----------
a : (M, M) array_like
    A complex or real matrix whose eigenvalues and eigenvectors
    will be computed.
b : (M, M) array_like, optional
    Right-hand side matrix in a generalized eigenvalue problem.
    Default is None, identity matrix is assumed.
left : bool, optional
    Whether to calculate and return left eigenvectors.  Default is False.
right : bool, optional
    Whether to calculate and return right eigenvectors.  Default is True.
overwrite_a : bool, optional
    Whether to overwrite `a`; may improve performance.  Default is False.
overwrite_b : bool, optional
    Whether to overwrite `b`; may improve performance.  Default is False.
check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.
homogeneous_eigvals : bool, optional
    If True, return the eigenvalues in homogeneous coordinates.
    In this case ``w`` is a (2, M) array so that::

        w[1,i] a vr[:,i] = w[0,i] b vr[:,i]

    Default is False.

Returns
-------
w : (M,) or (2, M) double or complex ndarray
    The eigenvalues, each repeated according to its
    multiplicity. The shape is (M,) unless
    ``homogeneous_eigvals=True``.
vl : (M, M) double or complex ndarray
    The normalized left eigenvector corresponding to the eigenvalue
    ``w[i]`` is the column vl[:,i]. Only returned if ``left=True``.
vr : (M, M) double or complex ndarray
    The normalized right eigenvector corresponding to the eigenvalue
    ``w[i]`` is the column ``vr[:,i]``.  Only returned if ``right=True``.

Raises
------
LinAlgError
    If eigenvalue computation does not converge.

See Also
--------
eigvals : eigenvalues of general arrays
eigh : Eigenvalues and right eigenvectors for symmetric/Hermitian arrays.
eig_banded : eigenvalues and right eigenvectors for symmetric/Hermitian
    band matrices
eigh_tridiagonal : eigenvalues and right eiegenvectors for
    symmetric/Hermitian tridiagonal matrices

Examples
--------
>>> from scipy import linalg
>>> a = np.array([[0., -1.], [1., 0.]])
>>> linalg.eigvals(a)
array([0.+1.j, 0.-1.j])

>>> b = np.array([[0., 1.], [1., 1.]])
>>> linalg.eigvals(a, b)
array([ 1.+0.j, -1.+0.j])

>>> a = np.array([[3., 0., 0.], [0., 8., 0.], [0., 0., 7.]])
>>> linalg.eigvals(a, homogeneous_eigvals=True)
array([[3.+0.j, 8.+0.j, 7.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j]])

>>> a = np.array([[0., -1.], [1., 0.]])
>>> linalg.eigvals(a) == linalg.eig(a)[0]
array([ True,  True])
>>> linalg.eig(a, left=True, right=False)[1] # normalized left eigenvector
array([[-0.70710678+0.j        , -0.70710678-0.j        ],
       [-0.        +0.70710678j, -0.        -0.70710678j]])
>>> linalg.eig(a, left=False, right=True)[1] # normalized right eigenvector
array([[0.70710678+0.j        , 0.70710678-0.j        ],
       [0.        -0.70710678j, 0.        +0.70710678j]])
*)

val eigh : ?b:[>`Ndarray] Np.Obj.t -> ?lower:bool -> ?eigvals_only:bool -> ?overwrite_a:bool -> ?overwrite_b:bool -> ?turbo:bool -> ?eigvals:Py.Object.t -> ?type_:int -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * Py.Object.t)
(**
Solve an ordinary or generalized eigenvalue problem for a complex
Hermitian or real symmetric matrix.

Find eigenvalues w and optionally eigenvectors v of matrix `a`, where
`b` is positive definite::

                  a v[:,i] = w[i] b v[:,i]
    v[i,:].conj() a v[:,i] = w[i]
    v[i,:].conj() b v[:,i] = 1

Parameters
----------
a : (M, M) array_like
    A complex Hermitian or real symmetric matrix whose eigenvalues and
    eigenvectors will be computed.
b : (M, M) array_like, optional
    A complex Hermitian or real symmetric definite positive matrix in.
    If omitted, identity matrix is assumed.
lower : bool, optional
    Whether the pertinent array data is taken from the lower or upper
    triangle of `a`. (Default: lower)
eigvals_only : bool, optional
    Whether to calculate only eigenvalues and no eigenvectors.
    (Default: both are calculated)
turbo : bool, optional
    Use divide and conquer algorithm (faster but expensive in memory,
    only for generalized eigenvalue problem and if eigvals=None)
eigvals : tuple (lo, hi), optional
    Indexes of the smallest and largest (in ascending order) eigenvalues
    and corresponding eigenvectors to be returned: 0 <= lo <= hi <= M-1.
    If omitted, all eigenvalues and eigenvectors are returned.
type : int, optional
    Specifies the problem type to be solved:

       type = 1: a   v[:,i] = w[i] b v[:,i]

       type = 2: a b v[:,i] = w[i]   v[:,i]

       type = 3: b a v[:,i] = w[i]   v[:,i]
overwrite_a : bool, optional
    Whether to overwrite data in `a` (may improve performance)
overwrite_b : bool, optional
    Whether to overwrite data in `b` (may improve performance)
check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
w : (N,) float ndarray
    The N (1<=N<=M) selected eigenvalues, in ascending order, each
    repeated according to its multiplicity.
v : (M, N) complex ndarray
    (if eigvals_only == False)

    The normalized selected eigenvector corresponding to the
    eigenvalue w[i] is the column v[:,i].

    Normalization:

        type 1 and 3: v.conj() a      v  = w

        type 2: inv(v).conj() a  inv(v) = w

        type = 1 or 2: v.conj() b      v  = I

        type = 3: v.conj() inv(b) v  = I

Raises
------
LinAlgError
    If eigenvalue computation does not converge,
    an error occurred, or b matrix is not definite positive. Note that
    if input matrices are not symmetric or hermitian, no error is reported
    but results will be wrong.

See Also
--------
eigvalsh : eigenvalues of symmetric or Hermitian arrays
eig : eigenvalues and right eigenvectors for non-symmetric arrays
eigh : eigenvalues and right eigenvectors for symmetric/Hermitian arrays
eigh_tridiagonal : eigenvalues and right eiegenvectors for
    symmetric/Hermitian tridiagonal matrices

Notes
-----
This function does not check the input array for being hermitian/symmetric
in order to allow for representing arrays with only their upper/lower
triangular parts.

Examples
--------
>>> from scipy.linalg import eigh
>>> A = np.array([[6, 3, 1, 5], [3, 0, 5, 1], [1, 5, 6, 2], [5, 1, 2, 2]])
>>> w, v = eigh(A)
>>> np.allclose(A @ v - v @ np.diag(w), np.zeros((4, 4)))
True
*)

val eigs : ?k:int -> ?m:[`Arr of [>`ArrayLike] Np.Obj.t | `LinearOperator of Py.Object.t] -> ?sigma:Py.Object.t -> ?which:[`LM | `SM | `LR | `SR | `LI | `SI] -> ?v0:[>`Ndarray] Np.Obj.t -> ?ncv:int -> ?maxiter:int -> ?tol:float -> ?return_eigenvectors:bool -> ?minv:[`Arr of [>`ArrayLike] Np.Obj.t | `LinearOperator of Py.Object.t] -> ?oPinv:[`Arr of [>`ArrayLike] Np.Obj.t | `LinearOperator of Py.Object.t] -> ?oPpart:Py.Object.t -> a:[`Arr of [>`ArrayLike] Np.Obj.t | `LinearOperator of Py.Object.t] -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Find k eigenvalues and eigenvectors of the square matrix A.

Solves ``A * x[i] = w[i] * x[i]``, the standard eigenvalue problem
for w[i] eigenvalues with corresponding eigenvectors x[i].

If M is specified, solves ``A * x[i] = w[i] * M * x[i]``, the
generalized eigenvalue problem for w[i] eigenvalues
with corresponding eigenvectors x[i]

Parameters
----------
A : ndarray, sparse matrix or LinearOperator
    An array, sparse matrix, or LinearOperator representing
    the operation ``A * x``, where A is a real or complex square matrix.
k : int, optional
    The number of eigenvalues and eigenvectors desired.
    `k` must be smaller than N-1. It is not possible to compute all
    eigenvectors of a matrix.
M : ndarray, sparse matrix or LinearOperator, optional
    An array, sparse matrix, or LinearOperator representing
    the operation M*x for the generalized eigenvalue problem

        A * x = w * M * x.

    M must represent a real, symmetric matrix if A is real, and must
    represent a complex, hermitian matrix if A is complex. For best
    results, the data type of M should be the same as that of A.
    Additionally:

        If `sigma` is None, M is positive definite

        If sigma is specified, M is positive semi-definite

    If sigma is None, eigs requires an operator to compute the solution
    of the linear equation ``M * x = b``.  This is done internally via a
    (sparse) LU decomposition for an explicit matrix M, or via an
    iterative solver for a general linear operator.  Alternatively,
    the user can supply the matrix or operator Minv, which gives
    ``x = Minv * b = M^-1 * b``.
sigma : real or complex, optional
    Find eigenvalues near sigma using shift-invert mode.  This requires
    an operator to compute the solution of the linear system
    ``[A - sigma * M] * x = b``, where M is the identity matrix if
    unspecified. This is computed internally via a (sparse) LU
    decomposition for explicit matrices A & M, or via an iterative
    solver if either A or M is a general linear operator.
    Alternatively, the user can supply the matrix or operator OPinv,
    which gives ``x = OPinv * b = [A - sigma * M]^-1 * b``.
    For a real matrix A, shift-invert can either be done in imaginary
    mode or real mode, specified by the parameter OPpart ('r' or 'i').
    Note that when sigma is specified, the keyword 'which' (below)
    refers to the shifted eigenvalues ``w'[i]`` where:

        If A is real and OPpart == 'r' (default),
          ``w'[i] = 1/2 * [1/(w[i]-sigma) + 1/(w[i]-conj(sigma))]``.

        If A is real and OPpart == 'i',
          ``w'[i] = 1/2i * [1/(w[i]-sigma) - 1/(w[i]-conj(sigma))]``.

        If A is complex, ``w'[i] = 1/(w[i]-sigma)``.

v0 : ndarray, optional
    Starting vector for iteration.
    Default: random
ncv : int, optional
    The number of Lanczos vectors generated
    `ncv` must be greater than `k`; it is recommended that ``ncv > 2*k``.
    Default: ``min(n, max(2*k + 1, 20))``
which : str, ['LM' | 'SM' | 'LR' | 'SR' | 'LI' | 'SI'], optional
    Which `k` eigenvectors and eigenvalues to find:

        'LM' : largest magnitude

        'SM' : smallest magnitude

        'LR' : largest real part

        'SR' : smallest real part

        'LI' : largest imaginary part

        'SI' : smallest imaginary part

    When sigma != None, 'which' refers to the shifted eigenvalues w'[i]
    (see discussion in 'sigma', above).  ARPACK is generally better
    at finding large values than small values.  If small eigenvalues are
    desired, consider using shift-invert mode for better performance.
maxiter : int, optional
    Maximum number of Arnoldi update iterations allowed
    Default: ``n*10``
tol : float, optional
    Relative accuracy for eigenvalues (stopping criterion)
    The default value of 0 implies machine precision.
return_eigenvectors : bool, optional
    Return eigenvectors (True) in addition to eigenvalues
Minv : ndarray, sparse matrix or LinearOperator, optional
    See notes in M, above.
OPinv : ndarray, sparse matrix or LinearOperator, optional
    See notes in sigma, above.
OPpart : {'r' or 'i'}, optional
    See notes in sigma, above

Returns
-------
w : ndarray
    Array of k eigenvalues.
v : ndarray
    An array of `k` eigenvectors.
    ``v[:, i]`` is the eigenvector corresponding to the eigenvalue w[i].

Raises
------
ArpackNoConvergence
    When the requested convergence is not obtained.
    The currently converged eigenvalues and eigenvectors can be found
    as ``eigenvalues`` and ``eigenvectors`` attributes of the exception
    object.

See Also
--------
eigsh : eigenvalues and eigenvectors for symmetric matrix A
svds : singular value decomposition for a matrix A

Notes
-----
This function is a wrapper to the ARPACK [1]_ SNEUPD, DNEUPD, CNEUPD,
ZNEUPD, functions which use the Implicitly Restarted Arnoldi Method to
find the eigenvalues and eigenvectors [2]_.

References
----------
.. [1] ARPACK Software, http://www.caam.rice.edu/software/ARPACK/
.. [2] R. B. Lehoucq, D. C. Sorensen, and C. Yang,  ARPACK USERS GUIDE:
   Solution of Large Scale Eigenvalue Problems by Implicitly Restarted
   Arnoldi Methods. SIAM, Philadelphia, PA, 1998.

Examples
--------
Find 6 eigenvectors of the identity matrix:

>>> from scipy.sparse.linalg import eigs
>>> id = np.eye(13)
>>> vals, vecs = eigs(id, k=6)
>>> vals
array([ 1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j])
>>> vecs.shape
(13, 6)
*)

val eigsh : ?k:int -> ?m:Py.Object.t -> ?sigma:Py.Object.t -> ?which:Py.Object.t -> ?v0:Py.Object.t -> ?ncv:Py.Object.t -> ?maxiter:Py.Object.t -> ?tol:Py.Object.t -> ?return_eigenvectors:Py.Object.t -> ?minv:Py.Object.t -> ?oPinv:Py.Object.t -> ?mode:Py.Object.t -> a:[`Arr of [>`ArrayLike] Np.Obj.t | `LinearOperator of Py.Object.t] -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Find k eigenvalues and eigenvectors of the real symmetric square matrix
or complex hermitian matrix A.

Solves ``A * x[i] = w[i] * x[i]``, the standard eigenvalue problem for
w[i] eigenvalues with corresponding eigenvectors x[i].

If M is specified, solves ``A * x[i] = w[i] * M * x[i]``, the
generalized eigenvalue problem for w[i] eigenvalues
with corresponding eigenvectors x[i].

Parameters
----------
A : ndarray, sparse matrix or LinearOperator
    A square operator representing the operation ``A * x``, where ``A`` is
    real symmetric or complex hermitian. For buckling mode (see below)
    ``A`` must additionally be positive-definite.
k : int, optional
    The number of eigenvalues and eigenvectors desired.
    `k` must be smaller than N. It is not possible to compute all
    eigenvectors of a matrix.

Returns
-------
w : array
    Array of k eigenvalues.
v : array
    An array representing the `k` eigenvectors.  The column ``v[:, i]`` is
    the eigenvector corresponding to the eigenvalue ``w[i]``.

Other Parameters
----------------
M : An N x N matrix, array, sparse matrix, or linear operator representing
    the operation ``M @ x`` for the generalized eigenvalue problem

        A @ x = w * M @ x.

    M must represent a real, symmetric matrix if A is real, and must
    represent a complex, hermitian matrix if A is complex. For best
    results, the data type of M should be the same as that of A.
    Additionally:

        If sigma is None, M is symmetric positive definite.

        If sigma is specified, M is symmetric positive semi-definite.

        In buckling mode, M is symmetric indefinite.

    If sigma is None, eigsh requires an operator to compute the solution
    of the linear equation ``M @ x = b``. This is done internally via a
    (sparse) LU decomposition for an explicit matrix M, or via an
    iterative solver for a general linear operator.  Alternatively,
    the user can supply the matrix or operator Minv, which gives
    ``x = Minv @ b = M^-1 @ b``.
sigma : real
    Find eigenvalues near sigma using shift-invert mode.  This requires
    an operator to compute the solution of the linear system
    ``[A - sigma * M] x = b``, where M is the identity matrix if
    unspecified.  This is computed internally via a (sparse) LU
    decomposition for explicit matrices A & M, or via an iterative
    solver if either A or M is a general linear operator.
    Alternatively, the user can supply the matrix or operator OPinv,
    which gives ``x = OPinv @ b = [A - sigma * M]^-1 @ b``.
    Note that when sigma is specified, the keyword 'which' refers to
    the shifted eigenvalues ``w'[i]`` where:

        if mode == 'normal', ``w'[i] = 1 / (w[i] - sigma)``.

        if mode == 'cayley', ``w'[i] = (w[i] + sigma) / (w[i] - sigma)``.

        if mode == 'buckling', ``w'[i] = w[i] / (w[i] - sigma)``.

    (see further discussion in 'mode' below)
v0 : ndarray, optional
    Starting vector for iteration.
    Default: random
ncv : int, optional
    The number of Lanczos vectors generated ncv must be greater than k and
    smaller than n; it is recommended that ``ncv > 2*k``.
    Default: ``min(n, max(2*k + 1, 20))``
which : str ['LM' | 'SM' | 'LA' | 'SA' | 'BE']
    If A is a complex hermitian matrix, 'BE' is invalid.
    Which `k` eigenvectors and eigenvalues to find:

        'LM' : Largest (in magnitude) eigenvalues.

        'SM' : Smallest (in magnitude) eigenvalues.

        'LA' : Largest (algebraic) eigenvalues.

        'SA' : Smallest (algebraic) eigenvalues.

        'BE' : Half (k/2) from each end of the spectrum.

    When k is odd, return one more (k/2+1) from the high end.
    When sigma != None, 'which' refers to the shifted eigenvalues ``w'[i]``
    (see discussion in 'sigma', above).  ARPACK is generally better
    at finding large values than small values.  If small eigenvalues are
    desired, consider using shift-invert mode for better performance.
maxiter : int, optional
    Maximum number of Arnoldi update iterations allowed.
    Default: ``n*10``
tol : float
    Relative accuracy for eigenvalues (stopping criterion).
    The default value of 0 implies machine precision.
Minv : N x N matrix, array, sparse matrix, or LinearOperator
    See notes in M, above.
OPinv : N x N matrix, array, sparse matrix, or LinearOperator
    See notes in sigma, above.
return_eigenvectors : bool
    Return eigenvectors (True) in addition to eigenvalues.
    This value determines the order in which eigenvalues are sorted.
    The sort order is also dependent on the `which` variable.

        For which = 'LM' or 'SA':
            If `return_eigenvectors` is True, eigenvalues are sorted by
            algebraic value.

            If `return_eigenvectors` is False, eigenvalues are sorted by
            absolute value.

        For which = 'BE' or 'LA':
            eigenvalues are always sorted by algebraic value.

        For which = 'SM':
            If `return_eigenvectors` is True, eigenvalues are sorted by
            algebraic value.

            If `return_eigenvectors` is False, eigenvalues are sorted by
            decreasing absolute value.

mode : string ['normal' | 'buckling' | 'cayley']
    Specify strategy to use for shift-invert mode.  This argument applies
    only for real-valued A and sigma != None.  For shift-invert mode,
    ARPACK internally solves the eigenvalue problem
    ``OP * x'[i] = w'[i] * B * x'[i]``
    and transforms the resulting Ritz vectors x'[i] and Ritz values w'[i]
    into the desired eigenvectors and eigenvalues of the problem
    ``A * x[i] = w[i] * M * x[i]``.
    The modes are as follows:

        'normal' :
            OP = [A - sigma * M]^-1 @ M,
            B = M,
            w'[i] = 1 / (w[i] - sigma)

        'buckling' :
            OP = [A - sigma * M]^-1 @ A,
            B = A,
            w'[i] = w[i] / (w[i] - sigma)

        'cayley' :
            OP = [A - sigma * M]^-1 @ [A + sigma * M],
            B = M,
            w'[i] = (w[i] + sigma) / (w[i] - sigma)

    The choice of mode will affect which eigenvalues are selected by
    the keyword 'which', and can also impact the stability of
    convergence (see [2] for a discussion).

Raises
------
ArpackNoConvergence
    When the requested convergence is not obtained.

    The currently converged eigenvalues and eigenvectors can be found
    as ``eigenvalues`` and ``eigenvectors`` attributes of the exception
    object.

See Also
--------
eigs : eigenvalues and eigenvectors for a general (nonsymmetric) matrix A
svds : singular value decomposition for a matrix A

Notes
-----
This function is a wrapper to the ARPACK [1]_ SSEUPD and DSEUPD
functions which use the Implicitly Restarted Lanczos Method to
find the eigenvalues and eigenvectors [2]_.

References
----------
.. [1] ARPACK Software, http://www.caam.rice.edu/software/ARPACK/
.. [2] R. B. Lehoucq, D. C. Sorensen, and C. Yang,  ARPACK USERS GUIDE:
   Solution of Large Scale Eigenvalue Problems by Implicitly Restarted
   Arnoldi Methods. SIAM, Philadelphia, PA, 1998.

Examples
--------
>>> from scipy.sparse.linalg import eigsh
>>> identity = np.eye(13)
>>> eigenvalues, eigenvectors = eigsh(identity, k=6)
>>> eigenvalues
array([1., 1., 1., 1., 1., 1.])
>>> eigenvectors.shape
(13, 6)
*)

val eye : ?n:int -> ?k:int -> ?dtype:Np.Dtype.t -> ?format:string -> m:int -> unit -> Py.Object.t
(**
Sparse matrix with ones on diagonal

Returns a sparse (m x n) matrix where the k-th diagonal
is all ones and everything else is zeros.

Parameters
----------
m : int
    Number of rows in the matrix.
n : int, optional
    Number of columns. Default: `m`.
k : int, optional
    Diagonal to place ones on. Default: 0 (main diagonal).
dtype : dtype, optional
    Data type of the matrix.
format : str, optional
    Sparse format of the result, e.g. format='csr', etc.

Examples
--------
>>> from scipy import sparse
>>> sparse.eye(3).toarray()
array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]])
>>> sparse.eye(3, dtype=np.int8)
<3x3 sparse matrix of type '<class 'numpy.int8'>'
    with 3 stored elements (1 diagonals) in DIAgonal format>
*)

val get_OPinv_matvec : ?hermitian:Py.Object.t -> ?tol:Py.Object.t -> a:Py.Object.t -> m:Py.Object.t -> sigma:Py.Object.t -> unit -> Py.Object.t
(**
None
*)

val get_inv_matvec : ?hermitian:Py.Object.t -> ?tol:Py.Object.t -> m:Py.Object.t -> unit -> Py.Object.t
(**
None
*)

val gmres : ?x0:Py.Object.t -> ?tol:Py.Object.t -> ?restart:Py.Object.t -> ?maxiter:Py.Object.t -> ?m:Py.Object.t -> ?callback:Py.Object.t -> ?restrt:Py.Object.t -> ?atol:Py.Object.t -> ?callback_type:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Use Generalized Minimal RESidual iteration to solve ``Ax = b``.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real or complex N-by-N matrix of the linear system.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).

Returns
-------
x : {array, matrix}
    The converged solution.
info : int
    Provides convergence information:
      * 0  : successful exit
      * >0 : convergence to tolerance not achieved, number of iterations
      * <0 : illegal input or breakdown

Other parameters
----------------
x0 : {array, matrix}
    Starting guess for the solution (a vector of zeros by default).
tol, atol : float, optional
    Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
    The default for ``atol`` is ``'legacy'``, which emulates
    a different legacy behavior.

    .. warning::

       The default value for `atol` will be changed in a future release.
       For future compatibility, specify `atol` explicitly.
restart : int, optional
    Number of iterations between restarts. Larger values increase
    iteration cost, but may be necessary for convergence.
    Default is 20.
maxiter : int, optional
    Maximum number of iterations (restart cycles).  Iteration will stop
    after maxiter steps even if the specified tolerance has not been
    achieved.
M : {sparse matrix, dense matrix, LinearOperator}
    Inverse of the preconditioner of A.  M should approximate the
    inverse of A and be easy to solve for (see Notes).  Effective
    preconditioning dramatically improves the rate of convergence,
    which implies that fewer iterations are needed to reach a given
    error tolerance.  By default, no preconditioner is used.
callback : function
    User-supplied function to call after each iteration.  It is called
    as `callback(args)`, where `args` are selected by `callback_type`.
callback_type : {'x', 'pr_norm', 'legacy'}, optional
    Callback function argument requested:
      - ``x``: current iterate (ndarray), called on every restart
      - ``pr_norm``: relative (preconditioned) residual norm (float),
        called on every inner iteration
      - ``legacy`` (default): same as ``pr_norm``, but also changes the
        meaning of 'maxiter' to count inner iterations instead of restart
        cycles.
restrt : int, optional
    DEPRECATED - use `restart` instead.

See Also
--------
LinearOperator

Notes
-----
A preconditioner, P, is chosen such that P is close to A but easy to solve
for. The preconditioner parameter required by this routine is
``M = P^-1``. The inverse should preferably not be calculated
explicitly.  Rather, use the following template to produce M::

  # Construct a linear operator that computes P^-1 * x.
  import scipy.sparse.linalg as spla
  M_x = lambda x: spla.spsolve(P, x)
  M = spla.LinearOperator((n, n), M_x)

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import gmres
>>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
>>> b = np.array([2, 4, -1], dtype=float)
>>> x, exitCode = gmres(A, b)
>>> print(exitCode)            # 0 indicates successful convergence
0
>>> np.allclose(A.dot(x), b)
True
*)

val gmres_loose : a:Py.Object.t -> b:Py.Object.t -> tol:Py.Object.t -> unit -> Py.Object.t
(**
gmres with looser termination condition.
*)

val is_pydata_spmatrix : Py.Object.t -> Py.Object.t
(**
Check whether object is pydata/sparse matrix, avoiding importing the module.
*)

val isdense : Py.Object.t -> Py.Object.t
(**
None
*)

val issparse : Py.Object.t -> Py.Object.t
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

val isspmatrix : Py.Object.t -> Py.Object.t
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

val isspmatrix_csr : Py.Object.t -> Py.Object.t
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

val lobpcg : ?b:[`PyObject of Py.Object.t | `Spmatrix of [>`Spmatrix] Np.Obj.t] -> ?m:[`PyObject of Py.Object.t | `Spmatrix of [>`Spmatrix] Np.Obj.t] -> ?y:[`Ndarray of [>`Ndarray] Np.Obj.t | `PyObject of Py.Object.t] -> ?tol:[`Bool of bool | `S of string | `I of int | `F of float] -> ?maxiter:int -> ?largest:bool -> ?verbosityLevel:int -> ?retLambdaHistory:bool -> ?retResidualNormsHistory:bool -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> x:[`Ndarray of [>`Ndarray] Np.Obj.t | `PyObject of Py.Object.t] -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t * Py.Object.t * Py.Object.t)
(**
Locally Optimal Block Preconditioned Conjugate Gradient Method (LOBPCG)

LOBPCG is a preconditioned eigensolver for large symmetric positive
definite (SPD) generalized eigenproblems.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The symmetric linear operator of the problem, usually a
    sparse matrix.  Often called the 'stiffness matrix'.
X : ndarray, float32 or float64
    Initial approximation to the ``k`` eigenvectors (non-sparse). If `A`
    has ``shape=(n,n)`` then `X` should have shape ``shape=(n,k)``.
B : {dense matrix, sparse matrix, LinearOperator}, optional
    The right hand side operator in a generalized eigenproblem.
    By default, ``B = Identity``.  Often called the 'mass matrix'.
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
iteration by calling the 'standard' dense eigensolver, so if ``m`` is not
small enough compared to ``n``, it does not make sense to call the LOBPCG
code, but rather one should use the 'standard' eigensolver, e.g. numpy or
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

val lu_factor : ?overwrite_a:bool -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Compute pivoted LU decomposition of a matrix.

The decomposition is::

    A = P L U

where P is a permutation matrix, L lower triangular with unit
diagonal elements, and U upper triangular.

Parameters
----------
a : (M, M) array_like
    Matrix to decompose
overwrite_a : bool, optional
    Whether to overwrite data in A (may increase performance)
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
lu : (N, N) ndarray
    Matrix containing U in its upper triangle, and L in its lower triangle.
    The unit diagonal elements of L are not stored.
piv : (N,) ndarray
    Pivot indices representing the permutation matrix P:
    row i of matrix was interchanged with row piv[i].

See also
--------
lu_solve : solve an equation system using the LU factorization of a matrix

Notes
-----
This is a wrapper to the ``*GETRF`` routines from LAPACK.

Examples
--------
>>> from scipy.linalg import lu_factor
>>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
>>> lu, piv = lu_factor(A)
>>> piv
array([2, 2, 3, 3], dtype=int32)

Convert LAPACK's ``piv`` array to NumPy index and test the permutation 

>>> piv_py = [2, 0, 3, 1]
>>> L, U = np.tril(lu, k=-1) + np.eye(4), np.triu(lu)
>>> np.allclose(A[piv_py] - L @ U, np.zeros((4, 4)))
True
*)

val lu_solve : ?trans:[`Zero | `One | `Two] -> ?overwrite_b:bool -> ?check_finite:bool -> lu_and_piv:Py.Object.t -> b:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Solve an equation system, a x = b, given the LU factorization of a

Parameters
----------
(lu, piv)
    Factorization of the coefficient matrix a, as given by lu_factor
b : array
    Right-hand side
trans : {0, 1, 2}, optional
    Type of system to solve:

    =====  =========
    trans  system
    =====  =========
    0      a x   = b
    1      a^T x = b
    2      a^H x = b
    =====  =========
overwrite_b : bool, optional
    Whether to overwrite data in b (may increase performance)
check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
x : array
    Solution to the system

See also
--------
lu_factor : LU factorize a matrix

Examples
--------
>>> from scipy.linalg import lu_factor, lu_solve
>>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
>>> b = np.array([1, 1, 1, 1])
>>> lu, piv = lu_factor(A)
>>> x = lu_solve((lu, piv), b)
>>> np.allclose(A @ x - b, np.zeros((4,)))
True
*)

val splu : ?permc_spec:string -> ?diag_pivot_thresh:float -> ?relax:int -> ?panel_size:int -> ?options:Py.Object.t -> a:[>`Spmatrix] Np.Obj.t -> unit -> Py.Object.t
(**
Compute the LU decomposition of a sparse, square matrix.

Parameters
----------
A : sparse matrix
    Sparse matrix to factorize. Should be in CSR or CSC format.
permc_spec : str, optional
    How to permute the columns of the matrix for sparsity preservation.
    (default: 'COLAMD')

    - ``NATURAL``: natural ordering.
    - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
    - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
    - ``COLAMD``: approximate minimum degree column ordering

diag_pivot_thresh : float, optional
    Threshold used for a diagonal entry to be an acceptable pivot.
    See SuperLU user's guide for details [1]_
relax : int, optional
    Expert option for customizing the degree of relaxing supernodes.
    See SuperLU user's guide for details [1]_
panel_size : int, optional
    Expert option for customizing the panel size.
    See SuperLU user's guide for details [1]_
options : dict, optional
    Dictionary containing additional expert options to SuperLU.
    See SuperLU user guide [1]_ (section 2.4 on the 'Options' argument)
    for more details. For example, you can specify
    ``options=dict(Equil=False, IterRefine='SINGLE'))``
    to turn equilibration off and perform a single iterative refinement.

Returns
-------
invA : scipy.sparse.linalg.SuperLU
    Object, which has a ``solve`` method.

See also
--------
spilu : incomplete LU decomposition

Notes
-----
This function uses the SuperLU library.

References
----------
.. [1] SuperLU http://crd.lbl.gov/~xiaoye/SuperLU/

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import splu
>>> A = csc_matrix([[1., 0., 0.], [5., 0., 2.], [0., -1., 0.]], dtype=float)
>>> B = splu(A)
>>> x = np.array([1., 2., 3.], dtype=float)
>>> B.solve(x)
array([ 1. , -3. , -1.5])
>>> A.dot(B.solve(x))
array([ 1.,  2.,  3.])
>>> B.solve(A.dot(x))
array([ 1.,  2.,  3.])
*)

val svds : ?k:int -> ?ncv:int -> ?tol:float -> ?which:[`LM | `SM] -> ?v0:[>`Ndarray] Np.Obj.t -> ?maxiter:int -> ?return_singular_vectors:[`Bool of bool | `S of string] -> ?solver:string -> a:[`LinearOperator of Py.Object.t | `Spmatrix of [>`Spmatrix] Np.Obj.t] -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Compute the largest or smallest k singular values/vectors for a sparse matrix. The order of the singular values is not guaranteed.

Parameters
----------
A : {sparse matrix, LinearOperator}
    Array to compute the SVD on, of shape (M, N)
k : int, optional
    Number of singular values and vectors to compute.
    Must be 1 <= k < min(A.shape).
ncv : int, optional
    The number of Lanczos vectors generated
    ncv must be greater than k+1 and smaller than n;
    it is recommended that ncv > 2*k
    Default: ``min(n, max(2*k + 1, 20))``
tol : float, optional
    Tolerance for singular values. Zero (default) means machine precision.
which : str, ['LM' | 'SM'], optional
    Which `k` singular values to find:

        - 'LM' : largest singular values
        - 'SM' : smallest singular values

    .. versionadded:: 0.12.0
v0 : ndarray, optional
    Starting vector for iteration, of length min(A.shape). Should be an
    (approximate) left singular vector if N > M and a right singular
    vector otherwise.
    Default: random

    .. versionadded:: 0.12.0
maxiter : int, optional
    Maximum number of iterations.

    .. versionadded:: 0.12.0
return_singular_vectors : bool or str, optional
    - True: return singular vectors (True) in addition to singular values.

    .. versionadded:: 0.12.0

    - 'u': only return the u matrix, without computing vh (if N > M).
    - 'vh': only return the vh matrix, without computing u (if N <= M).

    .. versionadded:: 0.16.0
solver : str, optional
        Eigenvalue solver to use. Should be 'arpack' or 'lobpcg'.
        Default: 'arpack'

Returns
-------
u : ndarray, shape=(M, k)
    Unitary matrix having left singular vectors as columns.
    If `return_singular_vectors` is 'vh', this variable is not computed,
    and None is returned instead.
s : ndarray, shape=(k,)
    The singular values.
vt : ndarray, shape=(k, N)
    Unitary matrix having right singular vectors as rows.
    If `return_singular_vectors` is 'u', this variable is not computed,
    and None is returned instead.


Notes
-----
This is a naive implementation using ARPACK or LOBPCG as an eigensolver
on A.H * A or A * A.H, depending on which one is more efficient.

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import svds, eigs
>>> A = csc_matrix([[1, 0, 0], [5, 0, 2], [0, -1, 0], [0, 0, 3]], dtype=float)
>>> u, s, vt = svds(A, k=2)
>>> s
array([ 2.75193379,  5.6059665 ])
>>> np.sqrt(eigs(A.dot(A.T), k=2)[0]).real
array([ 5.6059665 ,  2.75193379])
*)


end

val eigs : ?k:int -> ?m:[`Arr of [>`ArrayLike] Np.Obj.t | `LinearOperator of Py.Object.t] -> ?sigma:Py.Object.t -> ?which:[`LM | `SM | `LR | `SR | `LI | `SI] -> ?v0:[>`Ndarray] Np.Obj.t -> ?ncv:int -> ?maxiter:int -> ?tol:float -> ?return_eigenvectors:bool -> ?minv:[`Arr of [>`ArrayLike] Np.Obj.t | `LinearOperator of Py.Object.t] -> ?oPinv:[`Arr of [>`ArrayLike] Np.Obj.t | `LinearOperator of Py.Object.t] -> ?oPpart:Py.Object.t -> a:[`Arr of [>`ArrayLike] Np.Obj.t | `LinearOperator of Py.Object.t] -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Find k eigenvalues and eigenvectors of the square matrix A.

Solves ``A * x[i] = w[i] * x[i]``, the standard eigenvalue problem
for w[i] eigenvalues with corresponding eigenvectors x[i].

If M is specified, solves ``A * x[i] = w[i] * M * x[i]``, the
generalized eigenvalue problem for w[i] eigenvalues
with corresponding eigenvectors x[i]

Parameters
----------
A : ndarray, sparse matrix or LinearOperator
    An array, sparse matrix, or LinearOperator representing
    the operation ``A * x``, where A is a real or complex square matrix.
k : int, optional
    The number of eigenvalues and eigenvectors desired.
    `k` must be smaller than N-1. It is not possible to compute all
    eigenvectors of a matrix.
M : ndarray, sparse matrix or LinearOperator, optional
    An array, sparse matrix, or LinearOperator representing
    the operation M*x for the generalized eigenvalue problem

        A * x = w * M * x.

    M must represent a real, symmetric matrix if A is real, and must
    represent a complex, hermitian matrix if A is complex. For best
    results, the data type of M should be the same as that of A.
    Additionally:

        If `sigma` is None, M is positive definite

        If sigma is specified, M is positive semi-definite

    If sigma is None, eigs requires an operator to compute the solution
    of the linear equation ``M * x = b``.  This is done internally via a
    (sparse) LU decomposition for an explicit matrix M, or via an
    iterative solver for a general linear operator.  Alternatively,
    the user can supply the matrix or operator Minv, which gives
    ``x = Minv * b = M^-1 * b``.
sigma : real or complex, optional
    Find eigenvalues near sigma using shift-invert mode.  This requires
    an operator to compute the solution of the linear system
    ``[A - sigma * M] * x = b``, where M is the identity matrix if
    unspecified. This is computed internally via a (sparse) LU
    decomposition for explicit matrices A & M, or via an iterative
    solver if either A or M is a general linear operator.
    Alternatively, the user can supply the matrix or operator OPinv,
    which gives ``x = OPinv * b = [A - sigma * M]^-1 * b``.
    For a real matrix A, shift-invert can either be done in imaginary
    mode or real mode, specified by the parameter OPpart ('r' or 'i').
    Note that when sigma is specified, the keyword 'which' (below)
    refers to the shifted eigenvalues ``w'[i]`` where:

        If A is real and OPpart == 'r' (default),
          ``w'[i] = 1/2 * [1/(w[i]-sigma) + 1/(w[i]-conj(sigma))]``.

        If A is real and OPpart == 'i',
          ``w'[i] = 1/2i * [1/(w[i]-sigma) - 1/(w[i]-conj(sigma))]``.

        If A is complex, ``w'[i] = 1/(w[i]-sigma)``.

v0 : ndarray, optional
    Starting vector for iteration.
    Default: random
ncv : int, optional
    The number of Lanczos vectors generated
    `ncv` must be greater than `k`; it is recommended that ``ncv > 2*k``.
    Default: ``min(n, max(2*k + 1, 20))``
which : str, ['LM' | 'SM' | 'LR' | 'SR' | 'LI' | 'SI'], optional
    Which `k` eigenvectors and eigenvalues to find:

        'LM' : largest magnitude

        'SM' : smallest magnitude

        'LR' : largest real part

        'SR' : smallest real part

        'LI' : largest imaginary part

        'SI' : smallest imaginary part

    When sigma != None, 'which' refers to the shifted eigenvalues w'[i]
    (see discussion in 'sigma', above).  ARPACK is generally better
    at finding large values than small values.  If small eigenvalues are
    desired, consider using shift-invert mode for better performance.
maxiter : int, optional
    Maximum number of Arnoldi update iterations allowed
    Default: ``n*10``
tol : float, optional
    Relative accuracy for eigenvalues (stopping criterion)
    The default value of 0 implies machine precision.
return_eigenvectors : bool, optional
    Return eigenvectors (True) in addition to eigenvalues
Minv : ndarray, sparse matrix or LinearOperator, optional
    See notes in M, above.
OPinv : ndarray, sparse matrix or LinearOperator, optional
    See notes in sigma, above.
OPpart : {'r' or 'i'}, optional
    See notes in sigma, above

Returns
-------
w : ndarray
    Array of k eigenvalues.
v : ndarray
    An array of `k` eigenvectors.
    ``v[:, i]`` is the eigenvector corresponding to the eigenvalue w[i].

Raises
------
ArpackNoConvergence
    When the requested convergence is not obtained.
    The currently converged eigenvalues and eigenvectors can be found
    as ``eigenvalues`` and ``eigenvectors`` attributes of the exception
    object.

See Also
--------
eigsh : eigenvalues and eigenvectors for symmetric matrix A
svds : singular value decomposition for a matrix A

Notes
-----
This function is a wrapper to the ARPACK [1]_ SNEUPD, DNEUPD, CNEUPD,
ZNEUPD, functions which use the Implicitly Restarted Arnoldi Method to
find the eigenvalues and eigenvectors [2]_.

References
----------
.. [1] ARPACK Software, http://www.caam.rice.edu/software/ARPACK/
.. [2] R. B. Lehoucq, D. C. Sorensen, and C. Yang,  ARPACK USERS GUIDE:
   Solution of Large Scale Eigenvalue Problems by Implicitly Restarted
   Arnoldi Methods. SIAM, Philadelphia, PA, 1998.

Examples
--------
Find 6 eigenvectors of the identity matrix:

>>> from scipy.sparse.linalg import eigs
>>> id = np.eye(13)
>>> vals, vecs = eigs(id, k=6)
>>> vals
array([ 1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j])
>>> vecs.shape
(13, 6)
*)

val eigsh : ?k:int -> ?m:Py.Object.t -> ?sigma:Py.Object.t -> ?which:Py.Object.t -> ?v0:Py.Object.t -> ?ncv:Py.Object.t -> ?maxiter:Py.Object.t -> ?tol:Py.Object.t -> ?return_eigenvectors:Py.Object.t -> ?minv:Py.Object.t -> ?oPinv:Py.Object.t -> ?mode:Py.Object.t -> a:[`Arr of [>`ArrayLike] Np.Obj.t | `LinearOperator of Py.Object.t] -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Find k eigenvalues and eigenvectors of the real symmetric square matrix
or complex hermitian matrix A.

Solves ``A * x[i] = w[i] * x[i]``, the standard eigenvalue problem for
w[i] eigenvalues with corresponding eigenvectors x[i].

If M is specified, solves ``A * x[i] = w[i] * M * x[i]``, the
generalized eigenvalue problem for w[i] eigenvalues
with corresponding eigenvectors x[i].

Parameters
----------
A : ndarray, sparse matrix or LinearOperator
    A square operator representing the operation ``A * x``, where ``A`` is
    real symmetric or complex hermitian. For buckling mode (see below)
    ``A`` must additionally be positive-definite.
k : int, optional
    The number of eigenvalues and eigenvectors desired.
    `k` must be smaller than N. It is not possible to compute all
    eigenvectors of a matrix.

Returns
-------
w : array
    Array of k eigenvalues.
v : array
    An array representing the `k` eigenvectors.  The column ``v[:, i]`` is
    the eigenvector corresponding to the eigenvalue ``w[i]``.

Other Parameters
----------------
M : An N x N matrix, array, sparse matrix, or linear operator representing
    the operation ``M @ x`` for the generalized eigenvalue problem

        A @ x = w * M @ x.

    M must represent a real, symmetric matrix if A is real, and must
    represent a complex, hermitian matrix if A is complex. For best
    results, the data type of M should be the same as that of A.
    Additionally:

        If sigma is None, M is symmetric positive definite.

        If sigma is specified, M is symmetric positive semi-definite.

        In buckling mode, M is symmetric indefinite.

    If sigma is None, eigsh requires an operator to compute the solution
    of the linear equation ``M @ x = b``. This is done internally via a
    (sparse) LU decomposition for an explicit matrix M, or via an
    iterative solver for a general linear operator.  Alternatively,
    the user can supply the matrix or operator Minv, which gives
    ``x = Minv @ b = M^-1 @ b``.
sigma : real
    Find eigenvalues near sigma using shift-invert mode.  This requires
    an operator to compute the solution of the linear system
    ``[A - sigma * M] x = b``, where M is the identity matrix if
    unspecified.  This is computed internally via a (sparse) LU
    decomposition for explicit matrices A & M, or via an iterative
    solver if either A or M is a general linear operator.
    Alternatively, the user can supply the matrix or operator OPinv,
    which gives ``x = OPinv @ b = [A - sigma * M]^-1 @ b``.
    Note that when sigma is specified, the keyword 'which' refers to
    the shifted eigenvalues ``w'[i]`` where:

        if mode == 'normal', ``w'[i] = 1 / (w[i] - sigma)``.

        if mode == 'cayley', ``w'[i] = (w[i] + sigma) / (w[i] - sigma)``.

        if mode == 'buckling', ``w'[i] = w[i] / (w[i] - sigma)``.

    (see further discussion in 'mode' below)
v0 : ndarray, optional
    Starting vector for iteration.
    Default: random
ncv : int, optional
    The number of Lanczos vectors generated ncv must be greater than k and
    smaller than n; it is recommended that ``ncv > 2*k``.
    Default: ``min(n, max(2*k + 1, 20))``
which : str ['LM' | 'SM' | 'LA' | 'SA' | 'BE']
    If A is a complex hermitian matrix, 'BE' is invalid.
    Which `k` eigenvectors and eigenvalues to find:

        'LM' : Largest (in magnitude) eigenvalues.

        'SM' : Smallest (in magnitude) eigenvalues.

        'LA' : Largest (algebraic) eigenvalues.

        'SA' : Smallest (algebraic) eigenvalues.

        'BE' : Half (k/2) from each end of the spectrum.

    When k is odd, return one more (k/2+1) from the high end.
    When sigma != None, 'which' refers to the shifted eigenvalues ``w'[i]``
    (see discussion in 'sigma', above).  ARPACK is generally better
    at finding large values than small values.  If small eigenvalues are
    desired, consider using shift-invert mode for better performance.
maxiter : int, optional
    Maximum number of Arnoldi update iterations allowed.
    Default: ``n*10``
tol : float
    Relative accuracy for eigenvalues (stopping criterion).
    The default value of 0 implies machine precision.
Minv : N x N matrix, array, sparse matrix, or LinearOperator
    See notes in M, above.
OPinv : N x N matrix, array, sparse matrix, or LinearOperator
    See notes in sigma, above.
return_eigenvectors : bool
    Return eigenvectors (True) in addition to eigenvalues.
    This value determines the order in which eigenvalues are sorted.
    The sort order is also dependent on the `which` variable.

        For which = 'LM' or 'SA':
            If `return_eigenvectors` is True, eigenvalues are sorted by
            algebraic value.

            If `return_eigenvectors` is False, eigenvalues are sorted by
            absolute value.

        For which = 'BE' or 'LA':
            eigenvalues are always sorted by algebraic value.

        For which = 'SM':
            If `return_eigenvectors` is True, eigenvalues are sorted by
            algebraic value.

            If `return_eigenvectors` is False, eigenvalues are sorted by
            decreasing absolute value.

mode : string ['normal' | 'buckling' | 'cayley']
    Specify strategy to use for shift-invert mode.  This argument applies
    only for real-valued A and sigma != None.  For shift-invert mode,
    ARPACK internally solves the eigenvalue problem
    ``OP * x'[i] = w'[i] * B * x'[i]``
    and transforms the resulting Ritz vectors x'[i] and Ritz values w'[i]
    into the desired eigenvectors and eigenvalues of the problem
    ``A * x[i] = w[i] * M * x[i]``.
    The modes are as follows:

        'normal' :
            OP = [A - sigma * M]^-1 @ M,
            B = M,
            w'[i] = 1 / (w[i] - sigma)

        'buckling' :
            OP = [A - sigma * M]^-1 @ A,
            B = A,
            w'[i] = w[i] / (w[i] - sigma)

        'cayley' :
            OP = [A - sigma * M]^-1 @ [A + sigma * M],
            B = M,
            w'[i] = (w[i] + sigma) / (w[i] - sigma)

    The choice of mode will affect which eigenvalues are selected by
    the keyword 'which', and can also impact the stability of
    convergence (see [2] for a discussion).

Raises
------
ArpackNoConvergence
    When the requested convergence is not obtained.

    The currently converged eigenvalues and eigenvectors can be found
    as ``eigenvalues`` and ``eigenvectors`` attributes of the exception
    object.

See Also
--------
eigs : eigenvalues and eigenvectors for a general (nonsymmetric) matrix A
svds : singular value decomposition for a matrix A

Notes
-----
This function is a wrapper to the ARPACK [1]_ SSEUPD and DSEUPD
functions which use the Implicitly Restarted Lanczos Method to
find the eigenvalues and eigenvectors [2]_.

References
----------
.. [1] ARPACK Software, http://www.caam.rice.edu/software/ARPACK/
.. [2] R. B. Lehoucq, D. C. Sorensen, and C. Yang,  ARPACK USERS GUIDE:
   Solution of Large Scale Eigenvalue Problems by Implicitly Restarted
   Arnoldi Methods. SIAM, Philadelphia, PA, 1998.

Examples
--------
>>> from scipy.sparse.linalg import eigsh
>>> identity = np.eye(13)
>>> eigenvalues, eigenvectors = eigsh(identity, k=6)
>>> eigenvalues
array([1., 1., 1., 1., 1., 1.])
>>> eigenvectors.shape
(13, 6)
*)

val lobpcg : ?b:[`PyObject of Py.Object.t | `Spmatrix of [>`Spmatrix] Np.Obj.t] -> ?m:[`PyObject of Py.Object.t | `Spmatrix of [>`Spmatrix] Np.Obj.t] -> ?y:[`Ndarray of [>`Ndarray] Np.Obj.t | `PyObject of Py.Object.t] -> ?tol:[`Bool of bool | `S of string | `I of int | `F of float] -> ?maxiter:int -> ?largest:bool -> ?verbosityLevel:int -> ?retLambdaHistory:bool -> ?retResidualNormsHistory:bool -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> x:[`Ndarray of [>`Ndarray] Np.Obj.t | `PyObject of Py.Object.t] -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t * Py.Object.t * Py.Object.t)
(**
Locally Optimal Block Preconditioned Conjugate Gradient Method (LOBPCG)

LOBPCG is a preconditioned eigensolver for large symmetric positive
definite (SPD) generalized eigenproblems.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The symmetric linear operator of the problem, usually a
    sparse matrix.  Often called the 'stiffness matrix'.
X : ndarray, float32 or float64
    Initial approximation to the ``k`` eigenvectors (non-sparse). If `A`
    has ``shape=(n,n)`` then `X` should have shape ``shape=(n,k)``.
B : {dense matrix, sparse matrix, LinearOperator}, optional
    The right hand side operator in a generalized eigenproblem.
    By default, ``B = Identity``.  Often called the 'mass matrix'.
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
iteration by calling the 'standard' dense eigensolver, so if ``m`` is not
small enough compared to ``n``, it does not make sense to call the LOBPCG
code, but rather one should use the 'standard' eigensolver, e.g. numpy or
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

val svds : ?k:int -> ?ncv:int -> ?tol:float -> ?which:[`LM | `SM] -> ?v0:[>`Ndarray] Np.Obj.t -> ?maxiter:int -> ?return_singular_vectors:[`Bool of bool | `S of string] -> ?solver:string -> a:[`LinearOperator of Py.Object.t | `Spmatrix of [>`Spmatrix] Np.Obj.t] -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Compute the largest or smallest k singular values/vectors for a sparse matrix. The order of the singular values is not guaranteed.

Parameters
----------
A : {sparse matrix, LinearOperator}
    Array to compute the SVD on, of shape (M, N)
k : int, optional
    Number of singular values and vectors to compute.
    Must be 1 <= k < min(A.shape).
ncv : int, optional
    The number of Lanczos vectors generated
    ncv must be greater than k+1 and smaller than n;
    it is recommended that ncv > 2*k
    Default: ``min(n, max(2*k + 1, 20))``
tol : float, optional
    Tolerance for singular values. Zero (default) means machine precision.
which : str, ['LM' | 'SM'], optional
    Which `k` singular values to find:

        - 'LM' : largest singular values
        - 'SM' : smallest singular values

    .. versionadded:: 0.12.0
v0 : ndarray, optional
    Starting vector for iteration, of length min(A.shape). Should be an
    (approximate) left singular vector if N > M and a right singular
    vector otherwise.
    Default: random

    .. versionadded:: 0.12.0
maxiter : int, optional
    Maximum number of iterations.

    .. versionadded:: 0.12.0
return_singular_vectors : bool or str, optional
    - True: return singular vectors (True) in addition to singular values.

    .. versionadded:: 0.12.0

    - 'u': only return the u matrix, without computing vh (if N > M).
    - 'vh': only return the vh matrix, without computing u (if N <= M).

    .. versionadded:: 0.16.0
solver : str, optional
        Eigenvalue solver to use. Should be 'arpack' or 'lobpcg'.
        Default: 'arpack'

Returns
-------
u : ndarray, shape=(M, k)
    Unitary matrix having left singular vectors as columns.
    If `return_singular_vectors` is 'vh', this variable is not computed,
    and None is returned instead.
s : ndarray, shape=(k,)
    The singular values.
vt : ndarray, shape=(k, N)
    Unitary matrix having right singular vectors as rows.
    If `return_singular_vectors` is 'u', this variable is not computed,
    and None is returned instead.


Notes
-----
This is a naive implementation using ARPACK or LOBPCG as an eigensolver
on A.H * A or A * A.H, depending on which one is more efficient.

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import svds, eigs
>>> A = csc_matrix([[1, 0, 0], [5, 0, 2], [0, -1, 0], [0, 0, 3]], dtype=float)
>>> u, s, vt = svds(A, k=2)
>>> s
array([ 2.75193379,  5.6059665 ])
>>> np.sqrt(eigs(A.dot(A.T), k=2)[0]).real
array([ 5.6059665 ,  2.75193379])
*)


end

module Interface : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module IdentityOperator : sig
type tag = [`IdentityOperator]
type t = [`IdentityOperator | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> t
(**
Common interface for performing matrix vector products

Many iterative methods (e.g. cg, gmres) do not need to know the
individual entries of a matrix to solve a linear system A*x=b.
Such solvers only require the computation of matrix vector
products, A*v where v is a dense vector.  This class serves as
an abstract interface between iterative solvers and matrix-like
objects.

To construct a concrete LinearOperator, either pass appropriate
callables to the constructor of this class, or subclass it.

A subclass must implement either one of the methods ``_matvec``
and ``_matmat``, and the attributes/properties ``shape`` (pair of
integers) and ``dtype`` (may be None). It may call the ``__init__``
on this class to have these attributes validated. Implementing
``_matvec`` automatically implements ``_matmat`` (using a naive
algorithm) and vice-versa.

Optionally, a subclass may implement ``_rmatvec`` or ``_adjoint``
to implement the Hermitian adjoint (conjugate transpose). As with
``_matvec`` and ``_matmat``, implementing either ``_rmatvec`` or
``_adjoint`` implements the other automatically. Implementing
``_adjoint`` is preferable; ``_rmatvec`` is mostly there for
backwards compatibility.

Parameters
----------
shape : tuple
    Matrix dimensions (M, N).
matvec : callable f(v)
    Returns returns A * v.
rmatvec : callable f(v)
    Returns A^H * v, where A^H is the conjugate transpose of A.
matmat : callable f(V)
    Returns A * V, where V is a dense matrix with dimensions (N, K).
dtype : dtype
    Data type of the matrix.
rmatmat : callable f(V)
    Returns A^H * V, where V is a dense matrix with dimensions (M, K).

Attributes
----------
args : tuple
    For linear operators describing products etc. of other linear
    operators, the operands of the binary operation.

See Also
--------
aslinearoperator : Construct LinearOperators

Notes
-----
The user-defined matvec() function must properly handle the case
where v has shape (N,) as well as the (N,1) case.  The shape of
the return type is handled internally by LinearOperator.

LinearOperator instances can also be multiplied, added with each
other and exponentiated, all lazily: the result of these operations
is always a new, composite LinearOperator, that defers linear
operations to the original operators and combines the results.

More details regarding how to subclass a LinearOperator and several
examples of concrete LinearOperator instances can be found in the
external project `PyLops <https://pylops.readthedocs.io>`_.


Examples
--------
>>> import numpy as np
>>> from scipy.sparse.linalg import LinearOperator
>>> def mv(v):
...     return np.array([2*v[0], 3*v[1]])
...
>>> A = LinearOperator((2,2), matvec=mv)
>>> A
<2x2 _CustomLinearOperator with dtype=float64>
>>> A.matvec(np.ones(2))
array([ 2.,  3.])
>>> A * np.ones(2)
array([ 2.,  3.])
*)

val adjoint : [> tag] Obj.t -> Py.Object.t
(**
Hermitian adjoint.

Returns the Hermitian adjoint of self, aka the Hermitian
conjugate or Hermitian transpose. For a complex matrix, the
Hermitian adjoint is equal to the conjugate transpose.

Can be abbreviated self.H instead of self.adjoint().

Returns
-------
A_H : LinearOperator
    Hermitian adjoint of self.
*)

val dot : x:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Matrix-matrix or matrix-vector multiplication.

Parameters
----------
x : array_like
    1-d or 2-d array, representing a vector or matrix.

Returns
-------
Ax : array
    1-d or 2-d array (depending on the shape of x) that represents
    the result of applying this linear operator on x.
*)

val matmat : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Matrix-matrix multiplication.

Performs the operation y=A*X where A is an MxN linear
operator and X dense N*K matrix or ndarray.

Parameters
----------
X : {matrix, ndarray}
    An array with shape (N,K).

Returns
-------
Y : {matrix, ndarray}
    A matrix or ndarray with shape (M,K) depending on
    the type of the X argument.

Notes
-----
This matmat wraps any user-specified matmat routine or overridden
_matmat method to ensure that y has the correct type.
*)

val matvec : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Matrix-vector multiplication.

Performs the operation y=A*x where A is an MxN linear
operator and x is a column vector or 1-d array.

Parameters
----------
x : {matrix, ndarray}
    An array with shape (N,) or (N,1).

Returns
-------
y : {matrix, ndarray}
    A matrix or ndarray with shape (M,) or (M,1) depending
    on the type and shape of the x argument.

Notes
-----
This matvec wraps the user-specified matvec routine or overridden
_matvec method to ensure that y has the correct shape and type.
*)

val rmatmat : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Adjoint matrix-matrix multiplication.

Performs the operation y = A^H * x where A is an MxN linear
operator and x is a column vector or 1-d array, or 2-d array.
The default implementation defers to the adjoint.

Parameters
----------
X : {matrix, ndarray}
    A matrix or 2D array.

Returns
-------
Y : {matrix, ndarray}
    A matrix or 2D array depending on the type of the input.

Notes
-----
This rmatmat wraps the user-specified rmatmat routine.
*)

val rmatvec : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Adjoint matrix-vector multiplication.

Performs the operation y = A^H * x where A is an MxN linear
operator and x is a column vector or 1-d array.

Parameters
----------
x : {matrix, ndarray}
    An array with shape (M,) or (M,1).

Returns
-------
y : {matrix, ndarray}
    A matrix or ndarray with shape (N,) or (N,1) depending
    on the type and shape of the x argument.

Notes
-----
This rmatvec wraps the user-specified rmatvec routine or overridden
_rmatvec method to ensure that y has the correct shape and type.
*)

val transpose : [> tag] Obj.t -> Py.Object.t
(**
Transpose this linear operator.

Returns a LinearOperator that represents the transpose of this one.
Can be abbreviated self.T instead of self.transpose().
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MatrixLinearOperator : sig
type tag = [`MatrixLinearOperator]
type t = [`MatrixLinearOperator | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> t
(**
Common interface for performing matrix vector products

Many iterative methods (e.g. cg, gmres) do not need to know the
individual entries of a matrix to solve a linear system A*x=b.
Such solvers only require the computation of matrix vector
products, A*v where v is a dense vector.  This class serves as
an abstract interface between iterative solvers and matrix-like
objects.

To construct a concrete LinearOperator, either pass appropriate
callables to the constructor of this class, or subclass it.

A subclass must implement either one of the methods ``_matvec``
and ``_matmat``, and the attributes/properties ``shape`` (pair of
integers) and ``dtype`` (may be None). It may call the ``__init__``
on this class to have these attributes validated. Implementing
``_matvec`` automatically implements ``_matmat`` (using a naive
algorithm) and vice-versa.

Optionally, a subclass may implement ``_rmatvec`` or ``_adjoint``
to implement the Hermitian adjoint (conjugate transpose). As with
``_matvec`` and ``_matmat``, implementing either ``_rmatvec`` or
``_adjoint`` implements the other automatically. Implementing
``_adjoint`` is preferable; ``_rmatvec`` is mostly there for
backwards compatibility.

Parameters
----------
shape : tuple
    Matrix dimensions (M, N).
matvec : callable f(v)
    Returns returns A * v.
rmatvec : callable f(v)
    Returns A^H * v, where A^H is the conjugate transpose of A.
matmat : callable f(V)
    Returns A * V, where V is a dense matrix with dimensions (N, K).
dtype : dtype
    Data type of the matrix.
rmatmat : callable f(V)
    Returns A^H * V, where V is a dense matrix with dimensions (M, K).

Attributes
----------
args : tuple
    For linear operators describing products etc. of other linear
    operators, the operands of the binary operation.

See Also
--------
aslinearoperator : Construct LinearOperators

Notes
-----
The user-defined matvec() function must properly handle the case
where v has shape (N,) as well as the (N,1) case.  The shape of
the return type is handled internally by LinearOperator.

LinearOperator instances can also be multiplied, added with each
other and exponentiated, all lazily: the result of these operations
is always a new, composite LinearOperator, that defers linear
operations to the original operators and combines the results.

More details regarding how to subclass a LinearOperator and several
examples of concrete LinearOperator instances can be found in the
external project `PyLops <https://pylops.readthedocs.io>`_.


Examples
--------
>>> import numpy as np
>>> from scipy.sparse.linalg import LinearOperator
>>> def mv(v):
...     return np.array([2*v[0], 3*v[1]])
...
>>> A = LinearOperator((2,2), matvec=mv)
>>> A
<2x2 _CustomLinearOperator with dtype=float64>
>>> A.matvec(np.ones(2))
array([ 2.,  3.])
>>> A * np.ones(2)
array([ 2.,  3.])
*)

val adjoint : [> tag] Obj.t -> Py.Object.t
(**
Hermitian adjoint.

Returns the Hermitian adjoint of self, aka the Hermitian
conjugate or Hermitian transpose. For a complex matrix, the
Hermitian adjoint is equal to the conjugate transpose.

Can be abbreviated self.H instead of self.adjoint().

Returns
-------
A_H : LinearOperator
    Hermitian adjoint of self.
*)

val dot : x:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Matrix-matrix or matrix-vector multiplication.

Parameters
----------
x : array_like
    1-d or 2-d array, representing a vector or matrix.

Returns
-------
Ax : array
    1-d or 2-d array (depending on the shape of x) that represents
    the result of applying this linear operator on x.
*)

val matmat : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Matrix-matrix multiplication.

Performs the operation y=A*X where A is an MxN linear
operator and X dense N*K matrix or ndarray.

Parameters
----------
X : {matrix, ndarray}
    An array with shape (N,K).

Returns
-------
Y : {matrix, ndarray}
    A matrix or ndarray with shape (M,K) depending on
    the type of the X argument.

Notes
-----
This matmat wraps any user-specified matmat routine or overridden
_matmat method to ensure that y has the correct type.
*)

val matvec : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Matrix-vector multiplication.

Performs the operation y=A*x where A is an MxN linear
operator and x is a column vector or 1-d array.

Parameters
----------
x : {matrix, ndarray}
    An array with shape (N,) or (N,1).

Returns
-------
y : {matrix, ndarray}
    A matrix or ndarray with shape (M,) or (M,1) depending
    on the type and shape of the x argument.

Notes
-----
This matvec wraps the user-specified matvec routine or overridden
_matvec method to ensure that y has the correct shape and type.
*)

val rmatmat : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Adjoint matrix-matrix multiplication.

Performs the operation y = A^H * x where A is an MxN linear
operator and x is a column vector or 1-d array, or 2-d array.
The default implementation defers to the adjoint.

Parameters
----------
X : {matrix, ndarray}
    A matrix or 2D array.

Returns
-------
Y : {matrix, ndarray}
    A matrix or 2D array depending on the type of the input.

Notes
-----
This rmatmat wraps the user-specified rmatmat routine.
*)

val rmatvec : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Adjoint matrix-vector multiplication.

Performs the operation y = A^H * x where A is an MxN linear
operator and x is a column vector or 1-d array.

Parameters
----------
x : {matrix, ndarray}
    An array with shape (M,) or (M,1).

Returns
-------
y : {matrix, ndarray}
    A matrix or ndarray with shape (N,) or (N,1) depending
    on the type and shape of the x argument.

Notes
-----
This rmatvec wraps the user-specified rmatvec routine or overridden
_rmatvec method to ensure that y has the correct shape and type.
*)

val transpose : [> tag] Obj.t -> Py.Object.t
(**
Transpose this linear operator.

Returns a LinearOperator that represents the transpose of this one.
Can be abbreviated self.T instead of self.transpose().
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val aslinearoperator : Py.Object.t -> Py.Object.t
(**
Return A as a LinearOperator.

'A' may be any of the following types:
 - ndarray
 - matrix
 - sparse matrix (e.g. csr_matrix, lil_matrix, etc.)
 - LinearOperator
 - An object with .shape and .matvec attributes

See the LinearOperator documentation for additional information.

Notes
-----
If 'A' has no .dtype attribute, the data type is determined by calling
:func:`LinearOperator.matvec()` - set the .dtype attribute to prevent this
call upon the linear operator creation.

Examples
--------
>>> from scipy.sparse.linalg import aslinearoperator
>>> M = np.array([[1,2,3],[4,5,6]], dtype=np.int32)
>>> aslinearoperator(M)
<2x3 MatrixLinearOperator with dtype=int32>
*)

val asmatrix : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> Py.Object.t
(**
None
*)

val is_pydata_spmatrix : Py.Object.t -> Py.Object.t
(**
Check whether object is pydata/sparse matrix, avoiding importing the module.
*)

val isintlike : Py.Object.t -> Py.Object.t
(**
Is x appropriate as an index into a sparse matrix? Returns True
if it can be cast safely to a machine int.
*)

val isshape : ?nonneg:Py.Object.t -> x:Py.Object.t -> unit -> Py.Object.t
(**
Is x a valid 2-tuple of dimensions?

If nonneg, also checks that the dimensions are non-negative.
*)

val isspmatrix : Py.Object.t -> Py.Object.t
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


end

module Isolve : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module Iterative : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val bicg : ?x0:Py.Object.t -> ?tol:Py.Object.t -> ?maxiter:Py.Object.t -> ?m:Py.Object.t -> ?callback:Py.Object.t -> ?atol:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Use BIConjugate Gradient iteration to solve ``Ax = b``.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real or complex N-by-N matrix of the linear system.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` and ``A^T x`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).

Returns
-------
x : {array, matrix}
    The converged solution.
info : integer
    Provides convergence information:
        0  : successful exit
        >0 : convergence to tolerance not achieved, number of iterations
        <0 : illegal input or breakdown

Other Parameters
----------------
x0  : {array, matrix}
    Starting guess for the solution.
tol, atol : float, optional
    Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
    The default for ``atol`` is ``'legacy'``, which emulates
    a different legacy behavior.

    .. warning::

       The default value for `atol` will be changed in a future release.
       For future compatibility, specify `atol` explicitly.
maxiter : integer
    Maximum number of iterations.  Iteration will stop after maxiter
    steps even if the specified tolerance has not been achieved.
M : {sparse matrix, dense matrix, LinearOperator}
    Preconditioner for A.  The preconditioner should approximate the
    inverse of A.  Effective preconditioning dramatically improves the
    rate of convergence, which implies that fewer iterations are needed
    to reach a given error tolerance.
callback : function
    User-supplied function to call after each iteration.  It is called
    as callback(xk), where xk is the current solution vector.
*)

val bicgstab : ?x0:Py.Object.t -> ?tol:Py.Object.t -> ?maxiter:Py.Object.t -> ?m:Py.Object.t -> ?callback:Py.Object.t -> ?atol:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Use BIConjugate Gradient STABilized iteration to solve ``Ax = b``.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real or complex N-by-N matrix of the linear system.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).

Returns
-------
x : {array, matrix}
    The converged solution.
info : integer
    Provides convergence information:
        0  : successful exit
        >0 : convergence to tolerance not achieved, number of iterations
        <0 : illegal input or breakdown

Other Parameters
----------------
x0  : {array, matrix}
    Starting guess for the solution.
tol, atol : float, optional
    Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
    The default for ``atol`` is ``'legacy'``, which emulates
    a different legacy behavior.

    .. warning::

       The default value for `atol` will be changed in a future release.
       For future compatibility, specify `atol` explicitly.
maxiter : integer
    Maximum number of iterations.  Iteration will stop after maxiter
    steps even if the specified tolerance has not been achieved.
M : {sparse matrix, dense matrix, LinearOperator}
    Preconditioner for A.  The preconditioner should approximate the
    inverse of A.  Effective preconditioning dramatically improves the
    rate of convergence, which implies that fewer iterations are needed
    to reach a given error tolerance.
callback : function
    User-supplied function to call after each iteration.  It is called
    as callback(xk), where xk is the current solution vector.
*)

val cg : ?x0:Py.Object.t -> ?tol:Py.Object.t -> ?maxiter:Py.Object.t -> ?m:Py.Object.t -> ?callback:Py.Object.t -> ?atol:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Use Conjugate Gradient iteration to solve ``Ax = b``.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real or complex N-by-N matrix of the linear system.
    ``A`` must represent a hermitian, positive definite matrix.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).

Returns
-------
x : {array, matrix}
    The converged solution.
info : integer
    Provides convergence information:
        0  : successful exit
        >0 : convergence to tolerance not achieved, number of iterations
        <0 : illegal input or breakdown

Other Parameters
----------------
x0  : {array, matrix}
    Starting guess for the solution.
tol, atol : float, optional
    Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
    The default for ``atol`` is ``'legacy'``, which emulates
    a different legacy behavior.

    .. warning::

       The default value for `atol` will be changed in a future release.
       For future compatibility, specify `atol` explicitly.
maxiter : integer
    Maximum number of iterations.  Iteration will stop after maxiter
    steps even if the specified tolerance has not been achieved.
M : {sparse matrix, dense matrix, LinearOperator}
    Preconditioner for A.  The preconditioner should approximate the
    inverse of A.  Effective preconditioning dramatically improves the
    rate of convergence, which implies that fewer iterations are needed
    to reach a given error tolerance.
callback : function
    User-supplied function to call after each iteration.  It is called
    as callback(xk), where xk is the current solution vector.
*)

val cgs : ?x0:Py.Object.t -> ?tol:Py.Object.t -> ?maxiter:Py.Object.t -> ?m:Py.Object.t -> ?callback:Py.Object.t -> ?atol:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Use Conjugate Gradient Squared iteration to solve ``Ax = b``.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real-valued N-by-N matrix of the linear system.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).

Returns
-------
x : {array, matrix}
    The converged solution.
info : integer
    Provides convergence information:
        0  : successful exit
        >0 : convergence to tolerance not achieved, number of iterations
        <0 : illegal input or breakdown

Other Parameters
----------------
x0  : {array, matrix}
    Starting guess for the solution.
tol, atol : float, optional
    Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
    The default for ``atol`` is ``'legacy'``, which emulates
    a different legacy behavior.

    .. warning::

       The default value for `atol` will be changed in a future release.
       For future compatibility, specify `atol` explicitly.
maxiter : integer
    Maximum number of iterations.  Iteration will stop after maxiter
    steps even if the specified tolerance has not been achieved.
M : {sparse matrix, dense matrix, LinearOperator}
    Preconditioner for A.  The preconditioner should approximate the
    inverse of A.  Effective preconditioning dramatically improves the
    rate of convergence, which implies that fewer iterations are needed
    to reach a given error tolerance.
callback : function
    User-supplied function to call after each iteration.  It is called
    as callback(xk), where xk is the current solution vector.
*)

val gmres : ?x0:Py.Object.t -> ?tol:Py.Object.t -> ?restart:Py.Object.t -> ?maxiter:Py.Object.t -> ?m:Py.Object.t -> ?callback:Py.Object.t -> ?restrt:Py.Object.t -> ?atol:Py.Object.t -> ?callback_type:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Use Generalized Minimal RESidual iteration to solve ``Ax = b``.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real or complex N-by-N matrix of the linear system.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).

Returns
-------
x : {array, matrix}
    The converged solution.
info : int
    Provides convergence information:
      * 0  : successful exit
      * >0 : convergence to tolerance not achieved, number of iterations
      * <0 : illegal input or breakdown

Other parameters
----------------
x0 : {array, matrix}
    Starting guess for the solution (a vector of zeros by default).
tol, atol : float, optional
    Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
    The default for ``atol`` is ``'legacy'``, which emulates
    a different legacy behavior.

    .. warning::

       The default value for `atol` will be changed in a future release.
       For future compatibility, specify `atol` explicitly.
restart : int, optional
    Number of iterations between restarts. Larger values increase
    iteration cost, but may be necessary for convergence.
    Default is 20.
maxiter : int, optional
    Maximum number of iterations (restart cycles).  Iteration will stop
    after maxiter steps even if the specified tolerance has not been
    achieved.
M : {sparse matrix, dense matrix, LinearOperator}
    Inverse of the preconditioner of A.  M should approximate the
    inverse of A and be easy to solve for (see Notes).  Effective
    preconditioning dramatically improves the rate of convergence,
    which implies that fewer iterations are needed to reach a given
    error tolerance.  By default, no preconditioner is used.
callback : function
    User-supplied function to call after each iteration.  It is called
    as `callback(args)`, where `args` are selected by `callback_type`.
callback_type : {'x', 'pr_norm', 'legacy'}, optional
    Callback function argument requested:
      - ``x``: current iterate (ndarray), called on every restart
      - ``pr_norm``: relative (preconditioned) residual norm (float),
        called on every inner iteration
      - ``legacy`` (default): same as ``pr_norm``, but also changes the
        meaning of 'maxiter' to count inner iterations instead of restart
        cycles.
restrt : int, optional
    DEPRECATED - use `restart` instead.

See Also
--------
LinearOperator

Notes
-----
A preconditioner, P, is chosen such that P is close to A but easy to solve
for. The preconditioner parameter required by this routine is
``M = P^-1``. The inverse should preferably not be calculated
explicitly.  Rather, use the following template to produce M::

  # Construct a linear operator that computes P^-1 * x.
  import scipy.sparse.linalg as spla
  M_x = lambda x: spla.spsolve(P, x)
  M = spla.LinearOperator((n, n), M_x)

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import gmres
>>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
>>> b = np.array([2, 4, -1], dtype=float)
>>> x, exitCode = gmres(A, b)
>>> print(exitCode)            # 0 indicates successful convergence
0
>>> np.allclose(A.dot(x), b)
True
*)

val make_system : a:Py.Object.t -> m:Py.Object.t -> x0:[`Ndarray of [>`Ndarray] Np.Obj.t | `None] -> b:[>`Ndarray] Np.Obj.t -> unit -> (Py.Object.t * Py.Object.t * Py.Object.t * Py.Object.t)
(**
Make a linear system Ax=b

Parameters
----------
A : LinearOperator
    sparse or dense matrix (or any valid input to aslinearoperator)
M : {LinearOperator, Nones}
    preconditioner
    sparse or dense matrix (or any valid input to aslinearoperator)
x0 : {array_like, None}
    initial guess to iterative method
b : array_like
    right hand side

Returns
-------
(A, M, x, b, postprocess)
    A : LinearOperator
        matrix of the linear system
    M : LinearOperator
        preconditioner
    x : rank 1 ndarray
        initial guess
    b : rank 1 ndarray
        right hand side
    postprocess : function
        converts the solution vector to the appropriate
        type and dimensions (e.g. (N,1) matrix)
*)

val non_reentrant : ?err_msg:Py.Object.t -> unit -> Py.Object.t
(**
Decorate a function with a threading lock and prevent reentrant calls.
*)

val qmr : ?x0:Py.Object.t -> ?tol:Py.Object.t -> ?maxiter:Py.Object.t -> ?m1:Py.Object.t -> ?m2:Py.Object.t -> ?callback:Py.Object.t -> ?atol:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Use Quasi-Minimal Residual iteration to solve ``Ax = b``.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real-valued N-by-N matrix of the linear system.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` and ``A^T x`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).

Returns
-------
x : {array, matrix}
    The converged solution.
info : integer
    Provides convergence information:
        0  : successful exit
        >0 : convergence to tolerance not achieved, number of iterations
        <0 : illegal input or breakdown

Other Parameters
----------------
x0  : {array, matrix}
    Starting guess for the solution.
tol, atol : float, optional
    Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
    The default for ``atol`` is ``'legacy'``, which emulates
    a different legacy behavior.

    .. warning::

       The default value for `atol` will be changed in a future release.
       For future compatibility, specify `atol` explicitly.
maxiter : integer
    Maximum number of iterations.  Iteration will stop after maxiter
    steps even if the specified tolerance has not been achieved.
M1 : {sparse matrix, dense matrix, LinearOperator}
    Left preconditioner for A.
M2 : {sparse matrix, dense matrix, LinearOperator}
    Right preconditioner for A. Used together with the left
    preconditioner M1.  The matrix M1*A*M2 should have better
    conditioned than A alone.
callback : function
    User-supplied function to call after each iteration.  It is called
    as callback(xk), where xk is the current solution vector.

See Also
--------
LinearOperator

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import qmr
>>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
>>> b = np.array([2, 4, -1], dtype=float)
>>> x, exitCode = qmr(A, b)
>>> print(exitCode)            # 0 indicates successful convergence
0
>>> np.allclose(A.dot(x), b)
True
*)

val set_docstring : ?footer:Py.Object.t -> ?atol_default:Py.Object.t -> header:Py.Object.t -> ainfo:Py.Object.t -> unit -> Py.Object.t
(**
None
*)


end

module Utils : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module Matrix : sig
type tag = [`Matrix]
type t = [`ArrayLike | `Matrix | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?dtype:Np.Dtype.t -> ?copy:bool -> data:[`Ndarray of [>`Ndarray] Np.Obj.t | `S of string] -> unit -> t
(**
matrix(data, dtype=None, copy=True)

.. note:: It is no longer recommended to use this class, even for linear
          algebra. Instead use regular arrays. The class may be removed
          in the future.

Returns a matrix from an array-like object, or from a string of data.
A matrix is a specialized 2-D array that retains its 2-D nature
through operations.  It has certain special operators, such as ``*``
(matrix multiplication) and ``**`` (matrix power).

Parameters
----------
data : array_like or string
   If `data` is a string, it is interpreted as a matrix with commas
   or spaces separating columns, and semicolons separating rows.
dtype : data-type
   Data-type of the output matrix.
copy : bool
   If `data` is already an `ndarray`, then this flag determines
   whether the data is copied (the default), or whether a view is
   constructed.

See Also
--------
array

Examples
--------
>>> a = np.matrix('1 2; 3 4')
>>> a
matrix([[1, 2],
        [3, 4]])

>>> np.matrix([[1, 2], [3, 4]])
matrix([[1, 2],
        [3, 4]])
*)

val __getitem__ : index:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return self[key].
*)

val __iter__ : [> tag] Obj.t -> Py.Object.t
(**
Implement iter(self).
*)

val __setitem__ : key:Py.Object.t -> value:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Set self[key] to value.
*)

val all : ?axis:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Test whether all matrix elements along a given axis evaluate to True.

Parameters
----------
See `numpy.all` for complete descriptions

See Also
--------
numpy.all

Notes
-----
This is the same as `ndarray.all`, but it returns a `matrix` object.

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3,4))); x
matrix([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> y = x[0]; y
matrix([[0, 1, 2, 3]])
>>> (x == y)
matrix([[ True,  True,  True,  True],
        [False, False, False, False],
        [False, False, False, False]])
>>> (x == y).all()
False
>>> (x == y).all(0)
matrix([[False, False, False, False]])
>>> (x == y).all(1)
matrix([[ True],
        [False],
        [False]])
*)

val any : ?axis:int -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Test whether any array element along a given axis evaluates to True.

Refer to `numpy.any` for full documentation.

Parameters
----------
axis : int, optional
    Axis along which logical OR is performed
out : ndarray, optional
    Output to existing array instead of creating new one, must have
    same shape as expected output

Returns
-------
    any : bool, ndarray
        Returns a single bool if `axis` is ``None``; otherwise,
        returns `ndarray`
*)

val argmax : ?axis:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Indexes of the maximum values along an axis.

Return the indexes of the first occurrences of the maximum values
along the specified axis.  If axis is None, the index is for the
flattened matrix.

Parameters
----------
See `numpy.argmax` for complete descriptions

See Also
--------
numpy.argmax

Notes
-----
This is the same as `ndarray.argmax`, but returns a `matrix` object
where `ndarray.argmax` would return an `ndarray`.

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3,4))); x
matrix([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> x.argmax()
11
>>> x.argmax(0)
matrix([[2, 2, 2, 2]])
>>> x.argmax(1)
matrix([[3],
        [3],
        [3]])
*)

val argmin : ?axis:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Indexes of the minimum values along an axis.

Return the indexes of the first occurrences of the minimum values
along the specified axis.  If axis is None, the index is for the
flattened matrix.

Parameters
----------
See `numpy.argmin` for complete descriptions.

See Also
--------
numpy.argmin

Notes
-----
This is the same as `ndarray.argmin`, but returns a `matrix` object
where `ndarray.argmin` would return an `ndarray`.

Examples
--------
>>> x = -np.matrix(np.arange(12).reshape((3,4))); x
matrix([[  0,  -1,  -2,  -3],
        [ -4,  -5,  -6,  -7],
        [ -8,  -9, -10, -11]])
>>> x.argmin()
11
>>> x.argmin(0)
matrix([[2, 2, 2, 2]])
>>> x.argmin(1)
matrix([[3],
        [3],
        [3]])
*)

val argpartition : ?axis:Py.Object.t -> ?kind:Py.Object.t -> ?order:Py.Object.t -> kth:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.argpartition(kth, axis=-1, kind='introselect', order=None)

Returns the indices that would partition this array.

Refer to `numpy.argpartition` for full documentation.

.. versionadded:: 1.8.0

See Also
--------
numpy.argpartition : equivalent function
*)

val argsort : ?axis:Py.Object.t -> ?kind:Py.Object.t -> ?order:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.argsort(axis=-1, kind=None, order=None)

Returns the indices that would sort this array.

Refer to `numpy.argsort` for full documentation.

See Also
--------
numpy.argsort : equivalent function
*)

val astype : ?order:[`C | `F | `A | `K] -> ?casting:[`No | `Equiv | `Safe | `Same_kind | `Unsafe] -> ?subok:Py.Object.t -> ?copy:bool -> dtype:[`S of string | `Dtype of Np.Dtype.t] -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
a.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)

Copy of the array, cast to a specified type.

Parameters
----------
dtype : str or dtype
    Typecode or data-type to which the array is cast.
order : {'C', 'F', 'A', 'K'}, optional
    Controls the memory layout order of the result.
    'C' means C order, 'F' means Fortran order, 'A'
    means 'F' order if all the arrays are Fortran contiguous,
    'C' order otherwise, and 'K' means as close to the
    order the array elements appear in memory as possible.
    Default is 'K'.
casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
    Controls what kind of data casting may occur. Defaults to 'unsafe'
    for backwards compatibility.

      * 'no' means the data types should not be cast at all.
      * 'equiv' means only byte-order changes are allowed.
      * 'safe' means only casts which can preserve values are allowed.
      * 'same_kind' means only safe casts or casts within a kind,
        like float64 to float32, are allowed.
      * 'unsafe' means any data conversions may be done.
subok : bool, optional
    If True, then sub-classes will be passed-through (default), otherwise
    the returned array will be forced to be a base-class array.
copy : bool, optional
    By default, astype always returns a newly allocated array. If this
    is set to false, and the `dtype`, `order`, and `subok`
    requirements are satisfied, the input array is returned instead
    of a copy.

Returns
-------
arr_t : ndarray
    Unless `copy` is False and the other conditions for returning the input
    array are satisfied (see description for `copy` input parameter), `arr_t`
    is a new array of the same shape as the input array, with dtype, order
    given by `dtype`, `order`.

Notes
-----
.. versionchanged:: 1.17.0
   Casting between a simple data type and a structured one is possible only
   for 'unsafe' casting.  Casting to multiple fields is allowed, but
   casting from multiple fields is not.

.. versionchanged:: 1.9.0
   Casting from numeric to string types in 'safe' casting mode requires
   that the string dtype length is long enough to store the max
   integer/float value converted.

Raises
------
ComplexWarning
    When casting from complex to float or int. To avoid this,
    one should use ``a.real.astype(t)``.

Examples
--------
>>> x = np.array([1, 2, 2.5])
>>> x
array([1. ,  2. ,  2.5])

>>> x.astype(int)
array([1, 2, 2])
*)

val byteswap : ?inplace:bool -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
a.byteswap(inplace=False)

Swap the bytes of the array elements

Toggle between low-endian and big-endian data representation by
returning a byteswapped array, optionally swapped in-place.
Arrays of byte-strings are not swapped. The real and imaginary
parts of a complex number are swapped individually.

Parameters
----------
inplace : bool, optional
    If ``True``, swap bytes in-place, default is ``False``.

Returns
-------
out : ndarray
    The byteswapped array. If `inplace` is ``True``, this is
    a view to self.

Examples
--------
>>> A = np.array([1, 256, 8755], dtype=np.int16)
>>> list(map(hex, A))
['0x1', '0x100', '0x2233']
>>> A.byteswap(inplace=True)
array([  256,     1, 13090], dtype=int16)
>>> list(map(hex, A))
['0x100', '0x1', '0x3322']

Arrays of byte-strings are not swapped

>>> A = np.array([b'ceg', b'fac'])
>>> A.byteswap()
array([b'ceg', b'fac'], dtype='|S3')

``A.newbyteorder().byteswap()`` produces an array with the same values
  but different representation in memory

>>> A = np.array([1, 2, 3])
>>> A.view(np.uint8)
array([1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0,
       0, 0], dtype=uint8)
>>> A.newbyteorder().byteswap(inplace=True)
array([1, 2, 3])
>>> A.view(np.uint8)
array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0,
       0, 3], dtype=uint8)
*)

val choose : ?out:Py.Object.t -> ?mode:Py.Object.t -> choices:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.choose(choices, out=None, mode='raise')

Use an index array to construct a new array from a set of choices.

Refer to `numpy.choose` for full documentation.

See Also
--------
numpy.choose : equivalent function
*)

val clip : ?min:Py.Object.t -> ?max:Py.Object.t -> ?out:Py.Object.t -> ?kwargs:(string * Py.Object.t) list -> [> tag] Obj.t -> Py.Object.t
(**
a.clip(min=None, max=None, out=None, **kwargs)

Return an array whose values are limited to ``[min, max]``.
One of max or min must be given.

Refer to `numpy.clip` for full documentation.

See Also
--------
numpy.clip : equivalent function
*)

val compress : ?axis:Py.Object.t -> ?out:Py.Object.t -> condition:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.compress(condition, axis=None, out=None)

Return selected slices of this array along given axis.

Refer to `numpy.compress` for full documentation.

See Also
--------
numpy.compress : equivalent function
*)

val conj : [> tag] Obj.t -> Py.Object.t
(**
a.conj()

Complex-conjugate all elements.

Refer to `numpy.conjugate` for full documentation.

See Also
--------
numpy.conjugate : equivalent function
*)

val conjugate : [> tag] Obj.t -> Py.Object.t
(**
a.conjugate()

Return the complex conjugate, element-wise.

Refer to `numpy.conjugate` for full documentation.

See Also
--------
numpy.conjugate : equivalent function
*)

val copy : ?order:[`C | `F | `A | `K] -> [> tag] Obj.t -> Py.Object.t
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

val cumprod : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.cumprod(axis=None, dtype=None, out=None)

Return the cumulative product of the elements along the given axis.

Refer to `numpy.cumprod` for full documentation.

See Also
--------
numpy.cumprod : equivalent function
*)

val cumsum : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.cumsum(axis=None, dtype=None, out=None)

Return the cumulative sum of the elements along the given axis.

Refer to `numpy.cumsum` for full documentation.

See Also
--------
numpy.cumsum : equivalent function
*)

val diagonal : ?offset:Py.Object.t -> ?axis1:Py.Object.t -> ?axis2:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
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

val dot : ?out:Py.Object.t -> b:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.dot(b, out=None)

Dot product of two arrays.

Refer to `numpy.dot` for full documentation.

See Also
--------
numpy.dot : equivalent function

Examples
--------
>>> a = np.eye(2)
>>> b = np.ones((2, 2)) * 2
>>> a.dot(b)
array([[2.,  2.],
       [2.,  2.]])

This array method can be conveniently chained:

>>> a.dot(b).dot(b)
array([[8.,  8.],
       [8.,  8.]])
*)

val dump : file:[`S of string | `Path of Py.Object.t] -> [> tag] Obj.t -> Py.Object.t
(**
a.dump(file)

Dump a pickle of the array to the specified file.
The array can be read back with pickle.load or numpy.load.

Parameters
----------
file : str or Path
    A string naming the dump file.

    .. versionchanged:: 1.17.0
        `pathlib.Path` objects are now accepted.
*)

val dumps : [> tag] Obj.t -> Py.Object.t
(**
a.dumps()

Returns the pickle of the array as a string.
pickle.loads or numpy.loads will convert the string back to an array.

Parameters
----------
None
*)

val fill : value:[`F of float | `I of int | `Bool of bool | `S of string] -> [> tag] Obj.t -> Py.Object.t
(**
a.fill(value)

Fill the array with a scalar value.

Parameters
----------
value : scalar
    All elements of `a` will be assigned this value.

Examples
--------
>>> a = np.array([1, 2])
>>> a.fill(0)
>>> a
array([0, 0])
>>> a = np.empty(2)
>>> a.fill(1)
>>> a
array([1.,  1.])
*)

val flatten : ?order:[`C | `F | `A | `K] -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return a flattened copy of the matrix.

All `N` elements of the matrix are placed into a single row.

Parameters
----------
order : {'C', 'F', 'A', 'K'}, optional
    'C' means to flatten in row-major (C-style) order. 'F' means to
    flatten in column-major (Fortran-style) order. 'A' means to
    flatten in column-major order if `m` is Fortran *contiguous* in
    memory, row-major order otherwise. 'K' means to flatten `m` in
    the order the elements occur in memory. The default is 'C'.

Returns
-------
y : matrix
    A copy of the matrix, flattened to a `(1, N)` matrix where `N`
    is the number of elements in the original matrix.

See Also
--------
ravel : Return a flattened array.
flat : A 1-D flat iterator over the matrix.

Examples
--------
>>> m = np.matrix([[1,2], [3,4]])
>>> m.flatten()
matrix([[1, 2, 3, 4]])
>>> m.flatten('F')
matrix([[1, 3, 2, 4]])
*)

val getA : [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return `self` as an `ndarray` object.

Equivalent to ``np.asarray(self)``.

Parameters
----------
None

Returns
-------
ret : ndarray
    `self` as an `ndarray`

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3,4))); x
matrix([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> x.getA()
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
*)

val getA1 : [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return `self` as a flattened `ndarray`.

Equivalent to ``np.asarray(x).ravel()``

Parameters
----------
None

Returns
-------
ret : ndarray
    `self`, 1-D, as an `ndarray`

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3,4))); x
matrix([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> x.getA1()
array([ 0,  1,  2, ...,  9, 10, 11])
*)

val getH : [> tag] Obj.t -> Py.Object.t
(**
Returns the (complex) conjugate transpose of `self`.

Equivalent to ``np.transpose(self)`` if `self` is real-valued.

Parameters
----------
None

Returns
-------
ret : matrix object
    complex conjugate transpose of `self`

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3,4)))
>>> z = x - 1j*x; z
matrix([[  0. +0.j,   1. -1.j,   2. -2.j,   3. -3.j],
        [  4. -4.j,   5. -5.j,   6. -6.j,   7. -7.j],
        [  8. -8.j,   9. -9.j,  10.-10.j,  11.-11.j]])
>>> z.getH()
matrix([[ 0. -0.j,  4. +4.j,  8. +8.j],
        [ 1. +1.j,  5. +5.j,  9. +9.j],
        [ 2. +2.j,  6. +6.j, 10.+10.j],
        [ 3. +3.j,  7. +7.j, 11.+11.j]])
*)

val getI : [> tag] Obj.t -> Py.Object.t
(**
Returns the (multiplicative) inverse of invertible `self`.

Parameters
----------
None

Returns
-------
ret : matrix object
    If `self` is non-singular, `ret` is such that ``ret * self`` ==
    ``self * ret`` == ``np.matrix(np.eye(self[0,:].size)`` all return
    ``True``.

Raises
------
numpy.linalg.LinAlgError: Singular matrix
    If `self` is singular.

See Also
--------
linalg.inv

Examples
--------
>>> m = np.matrix('[1, 2; 3, 4]'); m
matrix([[1, 2],
        [3, 4]])
>>> m.getI()
matrix([[-2. ,  1. ],
        [ 1.5, -0.5]])
>>> m.getI() * m
matrix([[ 1.,  0.], # may vary
        [ 0.,  1.]])
*)

val getT : [> tag] Obj.t -> Py.Object.t
(**
Returns the transpose of the matrix.

Does *not* conjugate!  For the complex conjugate transpose, use ``.H``.

Parameters
----------
None

Returns
-------
ret : matrix object
    The (non-conjugated) transpose of the matrix.

See Also
--------
transpose, getH

Examples
--------
>>> m = np.matrix('[1, 2; 3, 4]')
>>> m
matrix([[1, 2],
        [3, 4]])
>>> m.getT()
matrix([[1, 3],
        [2, 4]])
*)

val getfield : ?offset:int -> dtype:[`S of string | `Dtype of Np.Dtype.t] -> [> tag] Obj.t -> Py.Object.t
(**
a.getfield(dtype, offset=0)

Returns a field of the given array as a certain type.

A field is a view of the array data with a given data-type. The values in
the view are determined by the given type and the offset into the current
array in bytes. The offset needs to be such that the view dtype fits in the
array dtype; for example an array of dtype complex128 has 16-byte elements.
If taking a view with a 32-bit integer (4 bytes), the offset needs to be
between 0 and 12 bytes.

Parameters
----------
dtype : str or dtype
    The data type of the view. The dtype size of the view can not be larger
    than that of the array itself.
offset : int
    Number of bytes to skip before beginning the element view.

Examples
--------
>>> x = np.diag([1.+1.j]*2)
>>> x[1, 1] = 2 + 4.j
>>> x
array([[1.+1.j,  0.+0.j],
       [0.+0.j,  2.+4.j]])
>>> x.getfield(np.float64)
array([[1.,  0.],
       [0.,  2.]])

By choosing an offset of 8 bytes we can select the complex part of the
array for our view:

>>> x.getfield(np.float64, offset=8)
array([[1.,  0.],
       [0.,  4.]])
*)

val item : Py.Object.t list -> [> tag] Obj.t -> Py.Object.t
(**
a.item( *args)

Copy an element of an array to a standard Python scalar and return it.

Parameters
----------
\*args : Arguments (variable number and type)

    * none: in this case, the method only works for arrays
      with one element (`a.size == 1`), which element is
      copied into a standard Python scalar object and returned.

    * int_type: this argument is interpreted as a flat index into
      the array, specifying which element to copy and return.

    * tuple of int_types: functions as does a single int_type argument,
      except that the argument is interpreted as an nd-index into the
      array.

Returns
-------
z : Standard Python scalar object
    A copy of the specified element of the array as a suitable
    Python scalar

Notes
-----
When the data type of `a` is longdouble or clongdouble, item() returns
a scalar array object because there is no available Python scalar that
would not lose information. Void arrays return a buffer object for item(),
unless fields are defined, in which case a tuple is returned.

`item` is very similar to a[args], except, instead of an array scalar,
a standard Python scalar is returned. This can be useful for speeding up
access to elements of the array and doing arithmetic on elements of the
array using Python's optimized math.

Examples
--------
>>> np.random.seed(123)
>>> x = np.random.randint(9, size=(3, 3))
>>> x
array([[2, 2, 6],
       [1, 3, 6],
       [1, 0, 1]])
>>> x.item(3)
1
>>> x.item(7)
0
>>> x.item((0, 1))
2
>>> x.item((2, 2))
1
*)

val itemset : Py.Object.t list -> [> tag] Obj.t -> Py.Object.t
(**
a.itemset( *args)

Insert scalar into an array (scalar is cast to array's dtype, if possible)

There must be at least 1 argument, and define the last argument
as *item*.  Then, ``a.itemset( *args)`` is equivalent to but faster
than ``a[args] = item``.  The item should be a scalar value and `args`
must select a single item in the array `a`.

Parameters
----------
\*args : Arguments
    If one argument: a scalar, only used in case `a` is of size 1.
    If two arguments: the last argument is the value to be set
    and must be a scalar, the first argument specifies a single array
    element location. It is either an int or a tuple.

Notes
-----
Compared to indexing syntax, `itemset` provides some speed increase
for placing a scalar into a particular location in an `ndarray`,
if you must do this.  However, generally this is discouraged:
among other problems, it complicates the appearance of the code.
Also, when using `itemset` (and `item`) inside a loop, be sure
to assign the methods to a local variable to avoid the attribute
look-up at each loop iteration.

Examples
--------
>>> np.random.seed(123)
>>> x = np.random.randint(9, size=(3, 3))
>>> x
array([[2, 2, 6],
       [1, 3, 6],
       [1, 0, 1]])
>>> x.itemset(4, 0)
>>> x.itemset((2, 2), 9)
>>> x
array([[2, 2, 6],
       [1, 0, 6],
       [1, 0, 9]])
*)

val max : ?axis:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return the maximum value along an axis.

Parameters
----------
See `amax` for complete descriptions

See Also
--------
amax, ndarray.max

Notes
-----
This is the same as `ndarray.max`, but returns a `matrix` object
where `ndarray.max` would return an ndarray.

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3,4))); x
matrix([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> x.max()
11
>>> x.max(0)
matrix([[ 8,  9, 10, 11]])
>>> x.max(1)
matrix([[ 3],
        [ 7],
        [11]])
*)

val mean : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns the average of the matrix elements along the given axis.

Refer to `numpy.mean` for full documentation.

See Also
--------
numpy.mean

Notes
-----
Same as `ndarray.mean` except that, where that returns an `ndarray`,
this returns a `matrix` object.

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3, 4)))
>>> x
matrix([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> x.mean()
5.5
>>> x.mean(0)
matrix([[4., 5., 6., 7.]])
>>> x.mean(1)
matrix([[ 1.5],
        [ 5.5],
        [ 9.5]])
*)

val min : ?axis:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return the minimum value along an axis.

Parameters
----------
See `amin` for complete descriptions.

See Also
--------
amin, ndarray.min

Notes
-----
This is the same as `ndarray.min`, but returns a `matrix` object
where `ndarray.min` would return an ndarray.

Examples
--------
>>> x = -np.matrix(np.arange(12).reshape((3,4))); x
matrix([[  0,  -1,  -2,  -3],
        [ -4,  -5,  -6,  -7],
        [ -8,  -9, -10, -11]])
>>> x.min()
-11
>>> x.min(0)
matrix([[ -8,  -9, -10, -11]])
>>> x.min(1)
matrix([[ -3],
        [ -7],
        [-11]])
*)

val newbyteorder : ?new_order:string -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
arr.newbyteorder(new_order='S')

Return the array with the same data viewed with a different byte order.

Equivalent to::

    arr.view(arr.dtype.newbytorder(new_order))

Changes are also made in all fields and sub-arrays of the array data
type.



Parameters
----------
new_order : string, optional
    Byte order to force; a value from the byte order specifications
    below. `new_order` codes can be any of:

    * 'S' - swap dtype from current to opposite endian
    * {'<', 'L'} - little endian
    * {'>', 'B'} - big endian
    * {'=', 'N'} - native order
    * {'|', 'I'} - ignore (no change to byte order)

    The default value ('S') results in swapping the current
    byte order. The code does a case-insensitive check on the first
    letter of `new_order` for the alternatives above.  For example,
    any of 'B' or 'b' or 'biggish' are valid to specify big-endian.


Returns
-------
new_arr : array
    New array object with the dtype reflecting given change to the
    byte order.
*)

val nonzero : [> tag] Obj.t -> Py.Object.t
(**
a.nonzero()

Return the indices of the elements that are non-zero.

Refer to `numpy.nonzero` for full documentation.

See Also
--------
numpy.nonzero : equivalent function
*)

val partition : ?axis:int -> ?kind:[`Introselect] -> ?order:[`StringList of string list | `S of string] -> kth:[`I of int | `Is of int list] -> [> tag] Obj.t -> Py.Object.t
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

val prod : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return the product of the array elements over the given axis.

Refer to `prod` for full documentation.

See Also
--------
prod, ndarray.prod

Notes
-----
Same as `ndarray.prod`, except, where that returns an `ndarray`, this
returns a `matrix` object instead.

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3,4))); x
matrix([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> x.prod()
0
>>> x.prod(0)
matrix([[  0,  45, 120, 231]])
>>> x.prod(1)
matrix([[   0],
        [ 840],
        [7920]])
*)

val ptp : ?axis:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Peak-to-peak (maximum - minimum) value along the given axis.

Refer to `numpy.ptp` for full documentation.

See Also
--------
numpy.ptp

Notes
-----
Same as `ndarray.ptp`, except, where that would return an `ndarray` object,
this returns a `matrix` object.

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3,4))); x
matrix([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> x.ptp()
11
>>> x.ptp(0)
matrix([[8, 8, 8, 8]])
>>> x.ptp(1)
matrix([[3],
        [3],
        [3]])
*)

val put : ?mode:Py.Object.t -> indices:Py.Object.t -> values:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.put(indices, values, mode='raise')

Set ``a.flat[n] = values[n]`` for all `n` in indices.

Refer to `numpy.put` for full documentation.

See Also
--------
numpy.put : equivalent function
*)

val ravel : ?order:[`C | `F | `A | `K] -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return a flattened matrix.

Refer to `numpy.ravel` for more documentation.

Parameters
----------
order : {'C', 'F', 'A', 'K'}, optional
    The elements of `m` are read using this index order. 'C' means to
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
ret : matrix
    Return the matrix flattened to shape `(1, N)` where `N`
    is the number of elements in the original matrix.
    A copy is made only if necessary.

See Also
--------
matrix.flatten : returns a similar output matrix but always a copy
matrix.flat : a flat iterator on the array.
numpy.ravel : related function which returns an ndarray
*)

val repeat : ?axis:Py.Object.t -> repeats:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.repeat(repeats, axis=None)

Repeat elements of an array.

Refer to `numpy.repeat` for full documentation.

See Also
--------
numpy.repeat : equivalent function
*)

val reshape : ?order:Py.Object.t -> shape:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.reshape(shape, order='C')

Returns an array containing the same data with a new shape.

Refer to `numpy.reshape` for full documentation.

See Also
--------
numpy.reshape : equivalent function

Notes
-----
Unlike the free function `numpy.reshape`, this method on `ndarray` allows
the elements of the shape parameter to be passed in as separate arguments.
For example, ``a.reshape(10, 11)`` is equivalent to
``a.reshape((10, 11))``.
*)

val resize : ?refcheck:bool -> new_shape:[`TupleOfInts of int list | `T_n_ints of Py.Object.t] -> [> tag] Obj.t -> Py.Object.t
(**
a.resize(new_shape, refcheck=True)

Change shape and size of array in-place.

Parameters
----------
new_shape : tuple of ints, or `n` ints
    Shape of resized array.
refcheck : bool, optional
    If False, reference count will not be checked. Default is True.

Returns
-------
None

Raises
------
ValueError
    If `a` does not own its own data or references or views to it exist,
    and the data memory must be changed.
    PyPy only: will always raise if the data memory must be changed, since
    there is no reliable way to determine if references or views to it
    exist.

SystemError
    If the `order` keyword argument is specified. This behaviour is a
    bug in NumPy.

See Also
--------
resize : Return a new array with the specified shape.

Notes
-----
This reallocates space for the data area if necessary.

Only contiguous arrays (data elements consecutive in memory) can be
resized.

The purpose of the reference count check is to make sure you
do not use this array as a buffer for another Python object and then
reallocate the memory. However, reference counts can increase in
other ways so if you are sure that you have not shared the memory
for this array with another Python object, then you may safely set
`refcheck` to False.

Examples
--------
Shrinking an array: array is flattened (in the order that the data are
stored in memory), resized, and reshaped:

>>> a = np.array([[0, 1], [2, 3]], order='C')
>>> a.resize((2, 1))
>>> a
array([[0],
       [1]])

>>> a = np.array([[0, 1], [2, 3]], order='F')
>>> a.resize((2, 1))
>>> a
array([[0],
       [2]])

Enlarging an array: as above, but missing entries are filled with zeros:

>>> b = np.array([[0, 1], [2, 3]])
>>> b.resize(2, 3) # new_shape parameter doesn't have to be a tuple
>>> b
array([[0, 1, 2],
       [3, 0, 0]])

Referencing an array prevents resizing...

>>> c = a
>>> a.resize((1, 1))
Traceback (most recent call last):
...
ValueError: cannot resize an array that references or is referenced ...

Unless `refcheck` is False:

>>> a.resize((1, 1), refcheck=False)
>>> a
array([[0]])
>>> c
array([[0]])
*)

val round : ?decimals:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.round(decimals=0, out=None)

Return `a` with each element rounded to the given number of decimals.

Refer to `numpy.around` for full documentation.

See Also
--------
numpy.around : equivalent function
*)

val searchsorted : ?side:Py.Object.t -> ?sorter:Py.Object.t -> v:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.searchsorted(v, side='left', sorter=None)

Find indices where elements of v should be inserted in a to maintain order.

For full documentation, see `numpy.searchsorted`

See Also
--------
numpy.searchsorted : equivalent function
*)

val setfield : ?offset:int -> val_:Py.Object.t -> dtype:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.setfield(val, dtype, offset=0)

Put a value into a specified place in a field defined by a data-type.

Place `val` into `a`'s field defined by `dtype` and beginning `offset`
bytes into the field.

Parameters
----------
val : object
    Value to be placed in field.
dtype : dtype object
    Data-type of the field in which to place `val`.
offset : int, optional
    The number of bytes into the field at which to place `val`.

Returns
-------
None

See Also
--------
getfield

Examples
--------
>>> x = np.eye(3)
>>> x.getfield(np.float64)
array([[1.,  0.,  0.],
       [0.,  1.,  0.],
       [0.,  0.,  1.]])
>>> x.setfield(3, np.int32)
>>> x.getfield(np.int32)
array([[3, 3, 3],
       [3, 3, 3],
       [3, 3, 3]], dtype=int32)
>>> x
array([[1.0e+000, 1.5e-323, 1.5e-323],
       [1.5e-323, 1.0e+000, 1.5e-323],
       [1.5e-323, 1.5e-323, 1.0e+000]])
>>> x.setfield(np.eye(3), np.int32)
>>> x
array([[1.,  0.,  0.],
       [0.,  1.,  0.],
       [0.,  0.,  1.]])
*)

val setflags : ?write:bool -> ?align:bool -> ?uic:bool -> [> tag] Obj.t -> Py.Object.t
(**
a.setflags(write=None, align=None, uic=None)

Set array flags WRITEABLE, ALIGNED, (WRITEBACKIFCOPY and UPDATEIFCOPY),
respectively.

These Boolean-valued flags affect how numpy interprets the memory
area used by `a` (see Notes below). The ALIGNED flag can only
be set to True if the data is actually aligned according to the type.
The WRITEBACKIFCOPY and (deprecated) UPDATEIFCOPY flags can never be set
to True. The flag WRITEABLE can only be set to True if the array owns its
own memory, or the ultimate owner of the memory exposes a writeable buffer
interface, or is a string. (The exception for string is made so that
unpickling can be done without copying memory.)

Parameters
----------
write : bool, optional
    Describes whether or not `a` can be written to.
align : bool, optional
    Describes whether or not `a` is aligned properly for its type.
uic : bool, optional
    Describes whether or not `a` is a copy of another 'base' array.

Notes
-----
Array flags provide information about how the memory area used
for the array is to be interpreted. There are 7 Boolean flags
in use, only four of which can be changed by the user:
WRITEBACKIFCOPY, UPDATEIFCOPY, WRITEABLE, and ALIGNED.

WRITEABLE (W) the data area can be written to;

ALIGNED (A) the data and strides are aligned appropriately for the hardware
(as determined by the compiler);

UPDATEIFCOPY (U) (deprecated), replaced by WRITEBACKIFCOPY;

WRITEBACKIFCOPY (X) this array is a copy of some other array (referenced
by .base). When the C-API function PyArray_ResolveWritebackIfCopy is
called, the base array will be updated with the contents of this array.

All flags can be accessed using the single (upper case) letter as well
as the full name.

Examples
--------
>>> y = np.array([[3, 1, 7],
...               [2, 0, 0],
...               [8, 5, 9]])
>>> y
array([[3, 1, 7],
       [2, 0, 0],
       [8, 5, 9]])
>>> y.flags
  C_CONTIGUOUS : True
  F_CONTIGUOUS : False
  OWNDATA : True
  WRITEABLE : True
  ALIGNED : True
  WRITEBACKIFCOPY : False
  UPDATEIFCOPY : False
>>> y.setflags(write=0, align=0)
>>> y.flags
  C_CONTIGUOUS : True
  F_CONTIGUOUS : False
  OWNDATA : True
  WRITEABLE : False
  ALIGNED : False
  WRITEBACKIFCOPY : False
  UPDATEIFCOPY : False
>>> y.setflags(uic=1)
Traceback (most recent call last):
  File '<stdin>', line 1, in <module>
ValueError: cannot set WRITEBACKIFCOPY flag to True
*)

val sort : ?axis:int -> ?kind:[`Stable | `Quicksort | `Heapsort | `Mergesort] -> ?order:[`StringList of string list | `S of string] -> [> tag] Obj.t -> Py.Object.t
(**
a.sort(axis=-1, kind=None, order=None)

Sort an array in-place. Refer to `numpy.sort` for full documentation.

Parameters
----------
axis : int, optional
    Axis along which to sort. Default is -1, which means sort along the
    last axis.
kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
    Sorting algorithm. The default is 'quicksort'. Note that both 'stable'
    and 'mergesort' use timsort under the covers and, in general, the
    actual implementation will vary with datatype. The 'mergesort' option
    is retained for backwards compatibility.

    .. versionchanged:: 1.15.0.
       The 'stable' option was added.

order : str or list of str, optional
    When `a` is an array with fields defined, this argument specifies
    which fields to compare first, second, etc.  A single field can
    be specified as a string, and not all fields need be specified,
    but unspecified fields will still be used, in the order in which
    they come up in the dtype, to break ties.

See Also
--------
numpy.sort : Return a sorted copy of an array.
numpy.argsort : Indirect sort.
numpy.lexsort : Indirect stable sort on multiple keys.
numpy.searchsorted : Find elements in sorted array.
numpy.partition: Partial sort.

Notes
-----
See `numpy.sort` for notes on the different sorting algorithms.

Examples
--------
>>> a = np.array([[1,4], [3,1]])
>>> a.sort(axis=1)
>>> a
array([[1, 4],
       [1, 3]])
>>> a.sort(axis=0)
>>> a
array([[1, 3],
       [1, 4]])

Use the `order` keyword to specify a field to use when sorting a
structured array:

>>> a = np.array([('a', 2), ('c', 1)], dtype=[('x', 'S1'), ('y', int)])
>>> a.sort(order='y')
>>> a
array([(b'c', 1), (b'a', 2)],
      dtype=[('x', 'S1'), ('y', '<i8')])
*)

val squeeze : ?axis:int list -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return a possibly reshaped matrix.

Refer to `numpy.squeeze` for more documentation.

Parameters
----------
axis : None or int or tuple of ints, optional
    Selects a subset of the single-dimensional entries in the shape.
    If an axis is selected with shape entry greater than one,
    an error is raised.

Returns
-------
squeezed : matrix
    The matrix, but as a (1, N) matrix if it had shape (N, 1).

See Also
--------
numpy.squeeze : related function

Notes
-----
If `m` has a single column then that column is returned
as the single row of a matrix.  Otherwise `m` is returned.
The returned matrix is always either `m` itself or a view into `m`.
Supplying an axis keyword argument will not affect the returned matrix
but it may cause an error to be raised.

Examples
--------
>>> c = np.matrix([[1], [2]])
>>> c
matrix([[1],
        [2]])
>>> c.squeeze()
matrix([[1, 2]])
>>> r = c.T
>>> r
matrix([[1, 2]])
>>> r.squeeze()
matrix([[1, 2]])
>>> m = np.matrix([[1, 2], [3, 4]])
>>> m.squeeze()
matrix([[1, 2],
        [3, 4]])
*)

val std : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> ?ddof:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return the standard deviation of the array elements along the given axis.

Refer to `numpy.std` for full documentation.

See Also
--------
numpy.std

Notes
-----
This is the same as `ndarray.std`, except that where an `ndarray` would
be returned, a `matrix` object is returned instead.

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3, 4)))
>>> x
matrix([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> x.std()
3.4520525295346629 # may vary
>>> x.std(0)
matrix([[ 3.26598632,  3.26598632,  3.26598632,  3.26598632]]) # may vary
>>> x.std(1)
matrix([[ 1.11803399],
        [ 1.11803399],
        [ 1.11803399]])
*)

val sum : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns the sum of the matrix elements, along the given axis.

Refer to `numpy.sum` for full documentation.

See Also
--------
numpy.sum

Notes
-----
This is the same as `ndarray.sum`, except that where an `ndarray` would
be returned, a `matrix` object is returned instead.

Examples
--------
>>> x = np.matrix([[1, 2], [4, 3]])
>>> x.sum()
10
>>> x.sum(axis=1)
matrix([[3],
        [7]])
>>> x.sum(axis=1, dtype='float')
matrix([[3.],
        [7.]])
>>> out = np.zeros((2, 1), dtype='float')
>>> x.sum(axis=1, dtype='float', out=np.asmatrix(out))
matrix([[3.],
        [7.]])
*)

val swapaxes : axis1:Py.Object.t -> axis2:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.swapaxes(axis1, axis2)

Return a view of the array with `axis1` and `axis2` interchanged.

Refer to `numpy.swapaxes` for full documentation.

See Also
--------
numpy.swapaxes : equivalent function
*)

val take : ?axis:Py.Object.t -> ?out:Py.Object.t -> ?mode:Py.Object.t -> indices:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.take(indices, axis=None, out=None, mode='raise')

Return an array formed from the elements of `a` at the given indices.

Refer to `numpy.take` for full documentation.

See Also
--------
numpy.take : equivalent function
*)

val tobytes : ?order:[`F | `C | `None] -> [> tag] Obj.t -> Py.Object.t
(**
a.tobytes(order='C')

Construct Python bytes containing the raw data bytes in the array.

Constructs Python bytes showing a copy of the raw contents of
data memory. The bytes object can be produced in either 'C' or 'Fortran',
or 'Any' order (the default is 'C'-order). 'Any' order means C-order
unless the F_CONTIGUOUS flag in the array is set, in which case it
means 'Fortran' order.

.. versionadded:: 1.9.0

Parameters
----------
order : {'C', 'F', None}, optional
    Order of the data for multidimensional arrays:
    C, Fortran, or the same as for the original array.

Returns
-------
s : bytes
    Python bytes exhibiting a copy of `a`'s raw data.

Examples
--------
>>> x = np.array([[0, 1], [2, 3]], dtype='<u2')
>>> x.tobytes()
b'\x00\x00\x01\x00\x02\x00\x03\x00'
>>> x.tobytes('C') == x.tobytes()
True
>>> x.tobytes('F')
b'\x00\x00\x02\x00\x01\x00\x03\x00'
*)

val tofile : ?sep:string -> ?format:string -> fid:[`S of string | `PyObject of Py.Object.t] -> [> tag] Obj.t -> Py.Object.t
(**
a.tofile(fid, sep='', format='%s')

Write array to a file as text or binary (default).

Data is always written in 'C' order, independent of the order of `a`.
The data produced by this method can be recovered using the function
fromfile().

Parameters
----------
fid : file or str or Path
    An open file object, or a string containing a filename.

    .. versionchanged:: 1.17.0
        `pathlib.Path` objects are now accepted.

sep : str
    Separator between array items for text output.
    If '' (empty), a binary file is written, equivalent to
    ``file.write(a.tobytes())``.
format : str
    Format string for text file output.
    Each entry in the array is formatted to text by first converting
    it to the closest Python type, and then using 'format' % item.

Notes
-----
This is a convenience function for quick storage of array data.
Information on endianness and precision is lost, so this method is not a
good choice for files intended to archive data or transport data between
machines with different endianness. Some of these problems can be overcome
by outputting the data as text files, at the expense of speed and file
size.

When fid is a file object, array contents are directly written to the
file, bypassing the file object's ``write`` method. As a result, tofile
cannot be used with files objects supporting compression (e.g., GzipFile)
or file-like objects that do not support ``fileno()`` (e.g., BytesIO).
*)

val tolist : [> tag] Obj.t -> Py.Object.t
(**
Return the matrix as a (possibly nested) list.

See `ndarray.tolist` for full documentation.

See Also
--------
ndarray.tolist

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3,4))); x
matrix([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> x.tolist()
[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
*)

val tostring : ?order:[`F | `C | `None] -> [> tag] Obj.t -> Py.Object.t
(**
a.tostring(order='C')

Construct Python bytes containing the raw data bytes in the array.

Constructs Python bytes showing a copy of the raw contents of
data memory. The bytes object can be produced in either 'C' or 'Fortran',
or 'Any' order (the default is 'C'-order). 'Any' order means C-order
unless the F_CONTIGUOUS flag in the array is set, in which case it
means 'Fortran' order.

This function is a compatibility alias for tobytes. Despite its name it returns bytes not strings.

Parameters
----------
order : {'C', 'F', None}, optional
    Order of the data for multidimensional arrays:
    C, Fortran, or the same as for the original array.

Returns
-------
s : bytes
    Python bytes exhibiting a copy of `a`'s raw data.

Examples
--------
>>> x = np.array([[0, 1], [2, 3]], dtype='<u2')
>>> x.tobytes()
b'\x00\x00\x01\x00\x02\x00\x03\x00'
>>> x.tobytes('C') == x.tobytes()
True
>>> x.tobytes('F')
b'\x00\x00\x02\x00\x01\x00\x03\x00'
*)

val trace : ?offset:Py.Object.t -> ?axis1:Py.Object.t -> ?axis2:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.trace(offset=0, axis1=0, axis2=1, dtype=None, out=None)

Return the sum along diagonals of the array.

Refer to `numpy.trace` for full documentation.

See Also
--------
numpy.trace : equivalent function
*)

val transpose : Py.Object.t list -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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
   intended simply as a 'convenience' alternative to the tuple form)

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

val var : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> ?ddof:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns the variance of the matrix elements, along the given axis.

Refer to `numpy.var` for full documentation.

See Also
--------
numpy.var

Notes
-----
This is the same as `ndarray.var`, except that where an `ndarray` would
be returned, a `matrix` object is returned instead.

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3, 4)))
>>> x
matrix([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> x.var()
11.916666666666666
>>> x.var(0)
matrix([[ 10.66666667,  10.66666667,  10.66666667,  10.66666667]]) # may vary
>>> x.var(1)
matrix([[1.25],
        [1.25],
        [1.25]])
*)

val view : ?dtype:[`Ndarray_sub_class of Py.Object.t | `Dtype of Np.Dtype.t] -> ?type_:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.view(dtype=None, type=None)

New view of array with the same data.

Parameters
----------
dtype : data-type or ndarray sub-class, optional
    Data-type descriptor of the returned view, e.g., float32 or int16. The
    default, None, results in the view having the same data-type as `a`.
    This argument can also be specified as an ndarray sub-class, which
    then specifies the type of the returned object (this is equivalent to
    setting the ``type`` parameter).
type : Python type, optional
    Type of the returned view, e.g., ndarray or matrix.  Again, the
    default None results in type preservation.

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

For ``a.view(some_dtype)``, if ``some_dtype`` has a different number of
bytes per entry than the previous dtype (for example, converting a
regular array to a structured array), then the behavior of the view
cannot be predicted just from the superficial appearance of ``a`` (shown
by ``print(a)``). It also depends on exactly how ``a`` is stored in
memory. Therefore if ``a`` is C-ordered versus fortran-ordered, versus
defined as a slice or transpose, etc., the view may give different
results.


Examples
--------
>>> x = np.array([(1, 2)], dtype=[('a', np.int8), ('b', np.int8)])

Viewing array data using a different type and dtype:

>>> y = x.view(dtype=np.int16, type=np.matrix)
>>> y
matrix([[513]], dtype=int16)
>>> print(type(y))
<class 'numpy.matrix'>

Creating a view on a structured array so it can be used in calculations

>>> x = np.array([(1, 2),(3,4)], dtype=[('a', np.int8), ('b', np.int8)])
>>> xv = x.view(dtype=np.int8).reshape(-1,2)
>>> xv
array([[1, 2],
       [3, 4]], dtype=int8)
>>> xv.mean(0)
array([2.,  3.])

Making changes to the view changes the underlying array

>>> xv[0,1] = 20
>>> x
array([(1, 20), (3,  4)], dtype=[('a', 'i1'), ('b', 'i1')])

Using a view to convert an array to a recarray:

>>> z = x.view(np.recarray)
>>> z.a
array([1, 3], dtype=int8)

Views share data:

>>> x[0] = (9, 10)
>>> z[0]
(9, 10)

Views that change the dtype size (bytes per entry) should normally be
avoided on arrays defined by slices, transposes, fortran-ordering, etc.:

>>> x = np.array([[1,2,3],[4,5,6]], dtype=np.int16)
>>> y = x[:, 0:2]
>>> y
array([[1, 2],
       [4, 5]], dtype=int16)
>>> y.view(dtype=[('width', np.int16), ('length', np.int16)])
Traceback (most recent call last):
    ...
ValueError: To change to a dtype of a different size, the array must be C-contiguous
>>> z = y.copy()
>>> z.view(dtype=[('width', np.int16), ('length', np.int16)])
array([[(1, 2)],
       [(4, 5)]], dtype=[('width', '<i2'), ('length', '<i2')])
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val array : ?dtype:Np.Dtype.t -> ?copy:bool -> ?order:[`K | `A | `C | `F] -> ?subok:bool -> ?ndmin:int -> object_:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
array(object, dtype=None, copy=True, order='K', subok=False, ndmin=0)

Create an array.

Parameters
----------
object : array_like
    An array, any object exposing the array interface, an object whose
    __array__ method returns an array, or any (nested) sequence.
dtype : data-type, optional
    The desired data-type for the array.  If not given, then the type will
    be determined as the minimum type required to hold the objects in the
    sequence.
copy : bool, optional
    If true (default), then the object is copied.  Otherwise, a copy will
    only be made if __array__ returns a copy, if obj is a nested sequence,
    or if a copy is needed to satisfy any of the other requirements
    (`dtype`, `order`, etc.).
order : {'K', 'A', 'C', 'F'}, optional
    Specify the memory layout of the array. If object is not an array, the
    newly created array will be in C order (row major) unless 'F' is
    specified, in which case it will be in Fortran order (column major).
    If object is an array the following holds.

    ===== ========= ===================================================
    order  no copy                     copy=True
    ===== ========= ===================================================
    'K'   unchanged F & C order preserved, otherwise most similar order
    'A'   unchanged F order if input is F and not C, otherwise C order
    'C'   C order   C order
    'F'   F order   F order
    ===== ========= ===================================================

    When ``copy=False`` and a copy is made for other reasons, the result is
    the same as if ``copy=True``, with some exceptions for `A`, see the
    Notes section. The default order is 'K'.
subok : bool, optional
    If True, then sub-classes will be passed-through, otherwise
    the returned array will be forced to be a base-class array (default).
ndmin : int, optional
    Specifies the minimum number of dimensions that the resulting
    array should have.  Ones will be pre-pended to the shape as
    needed to meet this requirement.

Returns
-------
out : ndarray
    An array object satisfying the specified requirements.

See Also
--------
empty_like : Return an empty array with shape and type of input.
ones_like : Return an array of ones with shape and type of input.
zeros_like : Return an array of zeros with shape and type of input.
full_like : Return a new array with shape of input filled with value.
empty : Return a new uninitialized array.
ones : Return a new array setting values to one.
zeros : Return a new array setting values to zero.
full : Return a new array of given shape filled with value.


Notes
-----
When order is 'A' and `object` is an array in neither 'C' nor 'F' order,
and a copy is forced by a change in dtype, then the order of the result is
not necessarily 'C' as expected. This is likely a bug.

Examples
--------
>>> np.array([1, 2, 3])
array([1, 2, 3])

Upcasting:

>>> np.array([1, 2, 3.0])
array([ 1.,  2.,  3.])

More than one dimension:

>>> np.array([[1, 2], [3, 4]])
array([[1, 2],
       [3, 4]])

Minimum dimensions 2:

>>> np.array([1, 2, 3], ndmin=2)
array([[1, 2, 3]])

Type provided:

>>> np.array([1, 2, 3], dtype=complex)
array([ 1.+0.j,  2.+0.j,  3.+0.j])

Data-type consisting of more than one element:

>>> x = np.array([(1,2),(3,4)],dtype=[('a','<i4'),('b','<i4')])
>>> x['a']
array([1, 3])

Creating an array from sub-classes:

>>> np.array(np.mat('1 2; 3 4'))
array([[1, 2],
       [3, 4]])

>>> np.array(np.mat('1 2; 3 4'), subok=True)
matrix([[1, 2],
        [3, 4]])
*)

val asanyarray : ?dtype:Np.Dtype.t -> ?order:[`F | `C] -> a:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Convert the input to an ndarray, but pass ndarray subclasses through.

Parameters
----------
a : array_like
    Input data, in any form that can be converted to an array.  This
    includes scalars, lists, lists of tuples, tuples, tuples of tuples,
    tuples of lists, and ndarrays.
dtype : data-type, optional
    By default, the data-type is inferred from the input data.
order : {'C', 'F'}, optional
    Whether to use row-major (C-style) or column-major
    (Fortran-style) memory representation.  Defaults to 'C'.

Returns
-------
out : ndarray or an ndarray subclass
    Array interpretation of `a`.  If `a` is an ndarray or a subclass
    of ndarray, it is returned as-is and no copy is performed.

See Also
--------
asarray : Similar function which always returns ndarrays.
ascontiguousarray : Convert input to a contiguous array.
asfarray : Convert input to a floating point ndarray.
asfortranarray : Convert input to an ndarray with column-major
                 memory order.
asarray_chkfinite : Similar function which checks input for NaNs and
                    Infs.
fromiter : Create an array from an iterator.
fromfunction : Construct an array by executing a function on grid
               positions.

Examples
--------
Convert a list into an array:

>>> a = [1, 2]
>>> np.asanyarray(a)
array([1, 2])

Instances of `ndarray` subclasses are passed through as-is:

>>> a = np.array([(1.0, 2), (3.0, 4)], dtype='f4,i4').view(np.recarray)
>>> np.asanyarray(a) is a
True
*)

val asarray : ?dtype:Np.Dtype.t -> ?order:[`F | `C] -> a:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Convert the input to an array.

Parameters
----------
a : array_like
    Input data, in any form that can be converted to an array.  This
    includes lists, lists of tuples, tuples, tuples of tuples, tuples
    of lists and ndarrays.
dtype : data-type, optional
    By default, the data-type is inferred from the input data.
order : {'C', 'F'}, optional
    Whether to use row-major (C-style) or
    column-major (Fortran-style) memory representation.
    Defaults to 'C'.

Returns
-------
out : ndarray
    Array interpretation of `a`.  No copy is performed if the input
    is already an ndarray with matching dtype and order.  If `a` is a
    subclass of ndarray, a base class ndarray is returned.

See Also
--------
asanyarray : Similar function which passes through subclasses.
ascontiguousarray : Convert input to a contiguous array.
asfarray : Convert input to a floating point ndarray.
asfortranarray : Convert input to an ndarray with column-major
                 memory order.
asarray_chkfinite : Similar function which checks input for NaNs and Infs.
fromiter : Create an array from an iterator.
fromfunction : Construct an array by executing a function on grid
               positions.

Examples
--------
Convert a list into an array:

>>> a = [1, 2]
>>> np.asarray(a)
array([1, 2])

Existing arrays are not copied:

>>> a = np.array([1, 2])
>>> np.asarray(a) is a
True

If `dtype` is set, array is copied only if dtype does not match:

>>> a = np.array([1, 2], dtype=np.float32)
>>> np.asarray(a, dtype=np.float32) is a
True
>>> np.asarray(a, dtype=np.float64) is a
False

Contrary to `asanyarray`, ndarray subclasses are not passed through:

>>> issubclass(np.recarray, np.ndarray)
True
>>> a = np.array([(1.0, 2), (3.0, 4)], dtype='f4,i4').view(np.recarray)
>>> np.asarray(a) is a
False
>>> np.asanyarray(a) is a
True
*)

val aslinearoperator : Py.Object.t -> Py.Object.t
(**
Return A as a LinearOperator.

'A' may be any of the following types:
 - ndarray
 - matrix
 - sparse matrix (e.g. csr_matrix, lil_matrix, etc.)
 - LinearOperator
 - An object with .shape and .matvec attributes

See the LinearOperator documentation for additional information.

Notes
-----
If 'A' has no .dtype attribute, the data type is determined by calling
:func:`LinearOperator.matvec()` - set the .dtype attribute to prevent this
call upon the linear operator creation.

Examples
--------
>>> from scipy.sparse.linalg import aslinearoperator
>>> M = np.array([[1,2,3],[4,5,6]], dtype=np.int32)
>>> aslinearoperator(M)
<2x3 MatrixLinearOperator with dtype=int32>
*)

val asmatrix : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> Py.Object.t
(**
None
*)

val coerce : x:Py.Object.t -> y:Py.Object.t -> unit -> Py.Object.t
(**
None
*)

val id : Py.Object.t -> Py.Object.t
(**
None
*)

val make_system : a:Py.Object.t -> m:Py.Object.t -> x0:[`Ndarray of [>`Ndarray] Np.Obj.t | `None] -> b:[>`Ndarray] Np.Obj.t -> unit -> (Py.Object.t * Py.Object.t * Py.Object.t * Py.Object.t)
(**
Make a linear system Ax=b

Parameters
----------
A : LinearOperator
    sparse or dense matrix (or any valid input to aslinearoperator)
M : {LinearOperator, Nones}
    preconditioner
    sparse or dense matrix (or any valid input to aslinearoperator)
x0 : {array_like, None}
    initial guess to iterative method
b : array_like
    right hand side

Returns
-------
(A, M, x, b, postprocess)
    A : LinearOperator
        matrix of the linear system
    M : LinearOperator
        preconditioner
    x : rank 1 ndarray
        initial guess
    b : rank 1 ndarray
        right hand side
    postprocess : function
        converts the solution vector to the appropriate
        type and dimensions (e.g. (N,1) matrix)
*)

val zeros : ?dtype:Np.Dtype.t -> ?order:[`C | `F] -> shape:int list -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
zeros(shape, dtype=float, order='C')

Return a new array of given shape and type, filled with zeros.

Parameters
----------
shape : int or tuple of ints
    Shape of the new array, e.g., ``(2, 3)`` or ``2``.
dtype : data-type, optional
    The desired data-type for the array, e.g., `numpy.int8`.  Default is
    `numpy.float64`.
order : {'C', 'F'}, optional, default: 'C'
    Whether to store multi-dimensional data in row-major
    (C-style) or column-major (Fortran-style) order in
    memory.

Returns
-------
out : ndarray
    Array of zeros with the given shape, dtype, and order.

See Also
--------
zeros_like : Return an array of zeros with shape and type of input.
empty : Return a new uninitialized array.
ones : Return a new array setting values to one.
full : Return a new array of given shape filled with value.

Examples
--------
>>> np.zeros(5)
array([ 0.,  0.,  0.,  0.,  0.])

>>> np.zeros((5,), dtype=int)
array([0, 0, 0, 0, 0])

>>> np.zeros((2, 1))
array([[ 0.],
       [ 0.]])

>>> s = (2,2)
>>> np.zeros(s)
array([[ 0.,  0.],
       [ 0.,  0.]])

>>> np.zeros((2,), dtype=[('x', 'i4'), ('y', 'i4')]) # custom dtype
array([(0, 0), (0, 0)],
      dtype=[('x', '<i4'), ('y', '<i4')])
*)


end

val bicg : ?x0:Py.Object.t -> ?tol:Py.Object.t -> ?maxiter:Py.Object.t -> ?m:Py.Object.t -> ?callback:Py.Object.t -> ?atol:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Use BIConjugate Gradient iteration to solve ``Ax = b``.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real or complex N-by-N matrix of the linear system.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` and ``A^T x`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).

Returns
-------
x : {array, matrix}
    The converged solution.
info : integer
    Provides convergence information:
        0  : successful exit
        >0 : convergence to tolerance not achieved, number of iterations
        <0 : illegal input or breakdown

Other Parameters
----------------
x0  : {array, matrix}
    Starting guess for the solution.
tol, atol : float, optional
    Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
    The default for ``atol`` is ``'legacy'``, which emulates
    a different legacy behavior.

    .. warning::

       The default value for `atol` will be changed in a future release.
       For future compatibility, specify `atol` explicitly.
maxiter : integer
    Maximum number of iterations.  Iteration will stop after maxiter
    steps even if the specified tolerance has not been achieved.
M : {sparse matrix, dense matrix, LinearOperator}
    Preconditioner for A.  The preconditioner should approximate the
    inverse of A.  Effective preconditioning dramatically improves the
    rate of convergence, which implies that fewer iterations are needed
    to reach a given error tolerance.
callback : function
    User-supplied function to call after each iteration.  It is called
    as callback(xk), where xk is the current solution vector.
*)

val bicgstab : ?x0:Py.Object.t -> ?tol:Py.Object.t -> ?maxiter:Py.Object.t -> ?m:Py.Object.t -> ?callback:Py.Object.t -> ?atol:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Use BIConjugate Gradient STABilized iteration to solve ``Ax = b``.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real or complex N-by-N matrix of the linear system.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).

Returns
-------
x : {array, matrix}
    The converged solution.
info : integer
    Provides convergence information:
        0  : successful exit
        >0 : convergence to tolerance not achieved, number of iterations
        <0 : illegal input or breakdown

Other Parameters
----------------
x0  : {array, matrix}
    Starting guess for the solution.
tol, atol : float, optional
    Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
    The default for ``atol`` is ``'legacy'``, which emulates
    a different legacy behavior.

    .. warning::

       The default value for `atol` will be changed in a future release.
       For future compatibility, specify `atol` explicitly.
maxiter : integer
    Maximum number of iterations.  Iteration will stop after maxiter
    steps even if the specified tolerance has not been achieved.
M : {sparse matrix, dense matrix, LinearOperator}
    Preconditioner for A.  The preconditioner should approximate the
    inverse of A.  Effective preconditioning dramatically improves the
    rate of convergence, which implies that fewer iterations are needed
    to reach a given error tolerance.
callback : function
    User-supplied function to call after each iteration.  It is called
    as callback(xk), where xk is the current solution vector.
*)

val cg : ?x0:Py.Object.t -> ?tol:Py.Object.t -> ?maxiter:Py.Object.t -> ?m:Py.Object.t -> ?callback:Py.Object.t -> ?atol:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Use Conjugate Gradient iteration to solve ``Ax = b``.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real or complex N-by-N matrix of the linear system.
    ``A`` must represent a hermitian, positive definite matrix.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).

Returns
-------
x : {array, matrix}
    The converged solution.
info : integer
    Provides convergence information:
        0  : successful exit
        >0 : convergence to tolerance not achieved, number of iterations
        <0 : illegal input or breakdown

Other Parameters
----------------
x0  : {array, matrix}
    Starting guess for the solution.
tol, atol : float, optional
    Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
    The default for ``atol`` is ``'legacy'``, which emulates
    a different legacy behavior.

    .. warning::

       The default value for `atol` will be changed in a future release.
       For future compatibility, specify `atol` explicitly.
maxiter : integer
    Maximum number of iterations.  Iteration will stop after maxiter
    steps even if the specified tolerance has not been achieved.
M : {sparse matrix, dense matrix, LinearOperator}
    Preconditioner for A.  The preconditioner should approximate the
    inverse of A.  Effective preconditioning dramatically improves the
    rate of convergence, which implies that fewer iterations are needed
    to reach a given error tolerance.
callback : function
    User-supplied function to call after each iteration.  It is called
    as callback(xk), where xk is the current solution vector.
*)

val cgs : ?x0:Py.Object.t -> ?tol:Py.Object.t -> ?maxiter:Py.Object.t -> ?m:Py.Object.t -> ?callback:Py.Object.t -> ?atol:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Use Conjugate Gradient Squared iteration to solve ``Ax = b``.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real-valued N-by-N matrix of the linear system.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).

Returns
-------
x : {array, matrix}
    The converged solution.
info : integer
    Provides convergence information:
        0  : successful exit
        >0 : convergence to tolerance not achieved, number of iterations
        <0 : illegal input or breakdown

Other Parameters
----------------
x0  : {array, matrix}
    Starting guess for the solution.
tol, atol : float, optional
    Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
    The default for ``atol`` is ``'legacy'``, which emulates
    a different legacy behavior.

    .. warning::

       The default value for `atol` will be changed in a future release.
       For future compatibility, specify `atol` explicitly.
maxiter : integer
    Maximum number of iterations.  Iteration will stop after maxiter
    steps even if the specified tolerance has not been achieved.
M : {sparse matrix, dense matrix, LinearOperator}
    Preconditioner for A.  The preconditioner should approximate the
    inverse of A.  Effective preconditioning dramatically improves the
    rate of convergence, which implies that fewer iterations are needed
    to reach a given error tolerance.
callback : function
    User-supplied function to call after each iteration.  It is called
    as callback(xk), where xk is the current solution vector.
*)

val gcrotmk : ?x0:[>`Ndarray] Np.Obj.t -> ?tol:Py.Object.t -> ?maxiter:int -> ?m:[`PyObject of Py.Object.t | `Spmatrix of [>`Spmatrix] Np.Obj.t] -> ?callback:Py.Object.t -> ?m':int -> ?k:int -> ?cu:Py.Object.t -> ?discard_C:bool -> ?truncate:[`Oldest | `Smallest] -> ?atol:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Solve a matrix equation using flexible GCROT(m,k) algorithm.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real or complex N-by-N matrix of the linear system.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).
x0  : {array, matrix}
    Starting guess for the solution.
tol, atol : float, optional
    Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
    The default for ``atol`` is `tol`.

    .. warning::

       The default value for `atol` will be changed in a future release.
       For future compatibility, specify `atol` explicitly.
maxiter : int, optional
    Maximum number of iterations.  Iteration will stop after maxiter
    steps even if the specified tolerance has not been achieved.
M : {sparse matrix, dense matrix, LinearOperator}, optional
    Preconditioner for A.  The preconditioner should approximate the
    inverse of A. gcrotmk is a 'flexible' algorithm and the preconditioner
    can vary from iteration to iteration. Effective preconditioning
    dramatically improves the rate of convergence, which implies that
    fewer iterations are needed to reach a given error tolerance.
callback : function, optional
    User-supplied function to call after each iteration.  It is called
    as callback(xk), where xk is the current solution vector.
m : int, optional
    Number of inner FGMRES iterations per each outer iteration.
    Default: 20
k : int, optional
    Number of vectors to carry between inner FGMRES iterations.
    According to [2]_, good values are around m.
    Default: m
CU : list of tuples, optional
    List of tuples ``(c, u)`` which contain the columns of the matrices
    C and U in the GCROT(m,k) algorithm. For details, see [2]_.
    The list given and vectors contained in it are modified in-place.
    If not given, start from empty matrices. The ``c`` elements in the
    tuples can be ``None``, in which case the vectors are recomputed
    via ``c = A u`` on start and orthogonalized as described in [3]_.
discard_C : bool, optional
    Discard the C-vectors at the end. Useful if recycling Krylov subspaces
    for different linear systems.
truncate : {'oldest', 'smallest'}, optional
    Truncation scheme to use. Drop: oldest vectors, or vectors with
    smallest singular values using the scheme discussed in [1,2].
    See [2]_ for detailed comparison.
    Default: 'oldest'

Returns
-------
x : array or matrix
    The solution found.
info : int
    Provides convergence information:

    * 0  : successful exit
    * >0 : convergence to tolerance not achieved, number of iterations

References
----------
.. [1] E. de Sturler, ''Truncation strategies for optimal Krylov subspace
       methods'', SIAM J. Numer. Anal. 36, 864 (1999).
.. [2] J.E. Hicken and D.W. Zingg, ''A simplified and flexible variant
       of GCROT for solving nonsymmetric linear systems'',
       SIAM J. Sci. Comput. 32, 172 (2010).
.. [3] M.L. Parks, E. de Sturler, G. Mackey, D.D. Johnson, S. Maiti,
       ''Recycling Krylov subspaces for sequences of linear systems'',
       SIAM J. Sci. Comput. 28, 1651 (2006).
*)

val gmres : ?x0:Py.Object.t -> ?tol:Py.Object.t -> ?restart:Py.Object.t -> ?maxiter:Py.Object.t -> ?m:Py.Object.t -> ?callback:Py.Object.t -> ?restrt:Py.Object.t -> ?atol:Py.Object.t -> ?callback_type:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Use Generalized Minimal RESidual iteration to solve ``Ax = b``.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real or complex N-by-N matrix of the linear system.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).

Returns
-------
x : {array, matrix}
    The converged solution.
info : int
    Provides convergence information:
      * 0  : successful exit
      * >0 : convergence to tolerance not achieved, number of iterations
      * <0 : illegal input or breakdown

Other parameters
----------------
x0 : {array, matrix}
    Starting guess for the solution (a vector of zeros by default).
tol, atol : float, optional
    Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
    The default for ``atol`` is ``'legacy'``, which emulates
    a different legacy behavior.

    .. warning::

       The default value for `atol` will be changed in a future release.
       For future compatibility, specify `atol` explicitly.
restart : int, optional
    Number of iterations between restarts. Larger values increase
    iteration cost, but may be necessary for convergence.
    Default is 20.
maxiter : int, optional
    Maximum number of iterations (restart cycles).  Iteration will stop
    after maxiter steps even if the specified tolerance has not been
    achieved.
M : {sparse matrix, dense matrix, LinearOperator}
    Inverse of the preconditioner of A.  M should approximate the
    inverse of A and be easy to solve for (see Notes).  Effective
    preconditioning dramatically improves the rate of convergence,
    which implies that fewer iterations are needed to reach a given
    error tolerance.  By default, no preconditioner is used.
callback : function
    User-supplied function to call after each iteration.  It is called
    as `callback(args)`, where `args` are selected by `callback_type`.
callback_type : {'x', 'pr_norm', 'legacy'}, optional
    Callback function argument requested:
      - ``x``: current iterate (ndarray), called on every restart
      - ``pr_norm``: relative (preconditioned) residual norm (float),
        called on every inner iteration
      - ``legacy`` (default): same as ``pr_norm``, but also changes the
        meaning of 'maxiter' to count inner iterations instead of restart
        cycles.
restrt : int, optional
    DEPRECATED - use `restart` instead.

See Also
--------
LinearOperator

Notes
-----
A preconditioner, P, is chosen such that P is close to A but easy to solve
for. The preconditioner parameter required by this routine is
``M = P^-1``. The inverse should preferably not be calculated
explicitly.  Rather, use the following template to produce M::

  # Construct a linear operator that computes P^-1 * x.
  import scipy.sparse.linalg as spla
  M_x = lambda x: spla.spsolve(P, x)
  M = spla.LinearOperator((n, n), M_x)

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import gmres
>>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
>>> b = np.array([2, 4, -1], dtype=float)
>>> x, exitCode = gmres(A, b)
>>> print(exitCode)            # 0 indicates successful convergence
0
>>> np.allclose(A.dot(x), b)
True
*)

val lgmres : ?x0:[>`Ndarray] Np.Obj.t -> ?tol:Py.Object.t -> ?maxiter:int -> ?m:[`PyObject of Py.Object.t | `Spmatrix of [>`Spmatrix] Np.Obj.t] -> ?callback:Py.Object.t -> ?inner_m:int -> ?outer_k:int -> ?outer_v:Py.Object.t -> ?store_outer_Av:bool -> ?prepend_outer_v:bool -> ?atol:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Solve a matrix equation using the LGMRES algorithm.

The LGMRES algorithm [1]_ [2]_ is designed to avoid some problems
in the convergence in restarted GMRES, and often converges in fewer
iterations.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real or complex N-by-N matrix of the linear system.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).
x0  : {array, matrix}
    Starting guess for the solution.
tol, atol : float, optional
    Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
    The default for ``atol`` is `tol`.

    .. warning::

       The default value for `atol` will be changed in a future release.
       For future compatibility, specify `atol` explicitly.
maxiter : int, optional
    Maximum number of iterations.  Iteration will stop after maxiter
    steps even if the specified tolerance has not been achieved.
M : {sparse matrix, dense matrix, LinearOperator}, optional
    Preconditioner for A.  The preconditioner should approximate the
    inverse of A.  Effective preconditioning dramatically improves the
    rate of convergence, which implies that fewer iterations are needed
    to reach a given error tolerance.
callback : function, optional
    User-supplied function to call after each iteration.  It is called
    as callback(xk), where xk is the current solution vector.
inner_m : int, optional
    Number of inner GMRES iterations per each outer iteration.
outer_k : int, optional
    Number of vectors to carry between inner GMRES iterations.
    According to [1]_, good values are in the range of 1...3.
    However, note that if you want to use the additional vectors to
    accelerate solving multiple similar problems, larger values may
    be beneficial.
outer_v : list of tuples, optional
    List containing tuples ``(v, Av)`` of vectors and corresponding
    matrix-vector products, used to augment the Krylov subspace, and
    carried between inner GMRES iterations. The element ``Av`` can
    be `None` if the matrix-vector product should be re-evaluated.
    This parameter is modified in-place by `lgmres`, and can be used
    to pass 'guess' vectors in and out of the algorithm when solving
    similar problems.
store_outer_Av : bool, optional
    Whether LGMRES should store also A*v in addition to vectors `v`
    in the `outer_v` list. Default is True.
prepend_outer_v : bool, optional 
    Whether to put outer_v augmentation vectors before Krylov iterates.
    In standard LGMRES, prepend_outer_v=False.

Returns
-------
x : array or matrix
    The converged solution.
info : int
    Provides convergence information:

        - 0  : successful exit
        - >0 : convergence to tolerance not achieved, number of iterations
        - <0 : illegal input or breakdown

Notes
-----
The LGMRES algorithm [1]_ [2]_ is designed to avoid the
slowing of convergence in restarted GMRES, due to alternating
residual vectors. Typically, it often outperforms GMRES(m) of
comparable memory requirements by some measure, or at least is not
much worse.

Another advantage in this algorithm is that you can supply it with
'guess' vectors in the `outer_v` argument that augment the Krylov
subspace. If the solution lies close to the span of these vectors,
the algorithm converges faster. This can be useful if several very
similar matrices need to be inverted one after another, such as in
Newton-Krylov iteration where the Jacobian matrix often changes
little in the nonlinear steps.

References
----------
.. [1] A.H. Baker and E.R. Jessup and T. Manteuffel, 'A Technique for
         Accelerating the Convergence of Restarted GMRES', SIAM J. Matrix
         Anal. Appl. 26, 962 (2005).
.. [2] A.H. Baker, 'On Improving the Performance of the Linear Solver
         restarted GMRES', PhD thesis, University of Colorado (2003).

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import lgmres
>>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
>>> b = np.array([2, 4, -1], dtype=float)
>>> x, exitCode = lgmres(A, b)
>>> print(exitCode)            # 0 indicates successful convergence
0
>>> np.allclose(A.dot(x), b)
True
*)

val lsmr : ?damp:float -> ?atol:Py.Object.t -> ?btol:Py.Object.t -> ?conlim:float -> ?maxiter:int -> ?show:bool -> ?x0:[>`Ndarray] Np.Obj.t -> a:[`Arr of [>`ArrayLike] Np.Obj.t | `LinearOperator of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> (Py.Object.t * int * int * float * float * float * float * float)
(**
Iterative solver for least-squares problems.

lsmr solves the system of linear equations ``Ax = b``. If the system
is inconsistent, it solves the least-squares problem ``min ||b - Ax||_2``.
A is a rectangular matrix of dimension m-by-n, where all cases are
allowed: m = n, m > n, or m < n. B is a vector of length m.
The matrix A may be dense or sparse (usually sparse).

Parameters
----------
A : {matrix, sparse matrix, ndarray, LinearOperator}
    Matrix A in the linear system.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` and ``A^H x`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : array_like, shape (m,)
    Vector b in the linear system.
damp : float
    Damping factor for regularized least-squares. `lsmr` solves
    the regularized least-squares problem::

     min ||(b) - (  A   )x||
         ||(0)   (damp*I) ||_2

    where damp is a scalar.  If damp is None or 0, the system
    is solved without regularization.
atol, btol : float, optional
    Stopping tolerances. `lsmr` continues iterations until a
    certain backward error estimate is smaller than some quantity
    depending on atol and btol.  Let ``r = b - Ax`` be the
    residual vector for the current approximate solution ``x``.
    If ``Ax = b`` seems to be consistent, ``lsmr`` terminates
    when ``norm(r) <= atol * norm(A) * norm(x) + btol * norm(b)``.
    Otherwise, lsmr terminates when ``norm(A^H r) <=
    atol * norm(A) * norm(r)``.  If both tolerances are 1.0e-6 (say),
    the final ``norm(r)`` should be accurate to about 6
    digits. (The final x will usually have fewer correct digits,
    depending on ``cond(A)`` and the size of LAMBDA.)  If `atol`
    or `btol` is None, a default value of 1.0e-6 will be used.
    Ideally, they should be estimates of the relative error in the
    entries of A and B respectively.  For example, if the entries
    of `A` have 7 correct digits, set atol = 1e-7. This prevents
    the algorithm from doing unnecessary work beyond the
    uncertainty of the input data.
conlim : float, optional
    `lsmr` terminates if an estimate of ``cond(A)`` exceeds
    `conlim`.  For compatible systems ``Ax = b``, conlim could be
    as large as 1.0e+12 (say).  For least-squares problems,
    `conlim` should be less than 1.0e+8. If `conlim` is None, the
    default value is 1e+8.  Maximum precision can be obtained by
    setting ``atol = btol = conlim = 0``, but the number of
    iterations may then be excessive.
maxiter : int, optional
    `lsmr` terminates if the number of iterations reaches
    `maxiter`.  The default is ``maxiter = min(m, n)``.  For
    ill-conditioned systems, a larger value of `maxiter` may be
    needed.
show : bool, optional
    Print iterations logs if ``show=True``.
x0 : array_like, shape (n,), optional
    Initial guess of x, if None zeros are used.

    .. versionadded:: 1.0.0
Returns
-------
x : ndarray of float
    Least-square solution returned.
istop : int
    istop gives the reason for stopping::

      istop   = 0 means x=0 is a solution.  If x0 was given, then x=x0 is a
                  solution.
              = 1 means x is an approximate solution to A*x = B,
                  according to atol and btol.
              = 2 means x approximately solves the least-squares problem
                  according to atol.
              = 3 means COND(A) seems to be greater than CONLIM.
              = 4 is the same as 1 with atol = btol = eps (machine
                  precision)
              = 5 is the same as 2 with atol = eps.
              = 6 is the same as 3 with CONLIM = 1/eps.
              = 7 means ITN reached maxiter before the other stopping
                  conditions were satisfied.

itn : int
    Number of iterations used.
normr : float
    ``norm(b-Ax)``
normar : float
    ``norm(A^H (b - Ax))``
norma : float
    ``norm(A)``
conda : float
    Condition number of A.
normx : float
    ``norm(x)``

Notes
-----

.. versionadded:: 0.11.0

References
----------
.. [1] D. C.-L. Fong and M. A. Saunders,
       'LSMR: An iterative algorithm for sparse least-squares problems',
       SIAM J. Sci. Comput., vol. 33, pp. 2950-2971, 2011.
       https://arxiv.org/abs/1006.0758
.. [2] LSMR Software, https://web.stanford.edu/group/SOL/software/lsmr/

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import lsmr
>>> A = csc_matrix([[1., 0.], [1., 1.], [0., 1.]], dtype=float)

The first example has the trivial solution `[0, 0]`

>>> b = np.array([0., 0., 0.], dtype=float)
>>> x, istop, itn, normr = lsmr(A, b)[:4]
>>> istop
0
>>> x
array([ 0.,  0.])

The stopping code `istop=0` returned indicates that a vector of zeros was
found as a solution. The returned solution `x` indeed contains `[0., 0.]`.
The next example has a non-trivial solution:

>>> b = np.array([1., 0., -1.], dtype=float)
>>> x, istop, itn, normr = lsmr(A, b)[:4]
>>> istop
1
>>> x
array([ 1., -1.])
>>> itn
1
>>> normr
4.440892098500627e-16

As indicated by `istop=1`, `lsmr` found a solution obeying the tolerance
limits. The given solution `[1., -1.]` obviously solves the equation. The
remaining return values include information about the number of iterations
(`itn=1`) and the remaining difference of left and right side of the solved
equation.
The final example demonstrates the behavior in the case where there is no
solution for the equation:

>>> b = np.array([1., 0.01, -1.], dtype=float)
>>> x, istop, itn, normr = lsmr(A, b)[:4]
>>> istop
2
>>> x
array([ 1.00333333, -0.99666667])
>>> A.dot(x)-b
array([ 0.00333333, -0.00333333,  0.00333333])
>>> normr
0.005773502691896255

`istop` indicates that the system is inconsistent and thus `x` is rather an
approximate solution to the corresponding least-squares problem. `normr`
contains the minimal distance that was found.
*)

val lsqr : ?damp:float -> ?atol:Py.Object.t -> ?btol:Py.Object.t -> ?conlim:float -> ?iter_lim:int -> ?show:bool -> ?calc_var:bool -> ?x0:[>`Ndarray] Np.Obj.t -> a:[`Arr of [>`ArrayLike] Np.Obj.t | `LinearOperator of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> (Py.Object.t * int * int * float * float * float * float * float * float * Py.Object.t)
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
If x0 is 'good', norm(r0) will be smaller than norm(b).
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
       'LSQR: An algorithm for sparse linear equations and
       sparse least squares', ACM TOMS 8(1), 43-71.
.. [2] C. C. Paige and M. A. Saunders (1982b).
       'Algorithm 583.  LSQR: Sparse linear equations and least
       squares problems', ACM TOMS 8(2), 195-209.
.. [3] M. A. Saunders (1995).  'Solution of sparse rectangular
       systems using LSQR and CRAIG', BIT 35, 588-604.

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

val minres : ?x0:Py.Object.t -> ?shift:Py.Object.t -> ?tol:Py.Object.t -> ?maxiter:Py.Object.t -> ?m:Py.Object.t -> ?callback:Py.Object.t -> ?show:Py.Object.t -> ?check:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Use MINimum RESidual iteration to solve Ax=b

MINRES minimizes norm(A*x - b) for a real symmetric matrix A.  Unlike
the Conjugate Gradient method, A can be indefinite or singular.

If shift != 0 then the method solves (A - shift*I)x = b

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real symmetric N-by-N matrix of the linear system
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).

Returns
-------
x : {array, matrix}
    The converged solution.
info : integer
    Provides convergence information:
        0  : successful exit
        >0 : convergence to tolerance not achieved, number of iterations
        <0 : illegal input or breakdown

Other Parameters
----------------
x0  : {array, matrix}
    Starting guess for the solution.
tol : float
    Tolerance to achieve. The algorithm terminates when the relative
    residual is below `tol`.
maxiter : integer
    Maximum number of iterations.  Iteration will stop after maxiter
    steps even if the specified tolerance has not been achieved.
M : {sparse matrix, dense matrix, LinearOperator}
    Preconditioner for A.  The preconditioner should approximate the
    inverse of A.  Effective preconditioning dramatically improves the
    rate of convergence, which implies that fewer iterations are needed
    to reach a given error tolerance.
callback : function
    User-supplied function to call after each iteration.  It is called
    as callback(xk), where xk is the current solution vector.

References
----------
Solution of sparse indefinite systems of linear equations,
    C. C. Paige and M. A. Saunders (1975),
    SIAM J. Numer. Anal. 12(4), pp. 617-629.
    https://web.stanford.edu/group/SOL/software/minres/

This file is a translation of the following MATLAB implementation:
    https://web.stanford.edu/group/SOL/software/minres/minres-matlab.zip
*)

val qmr : ?x0:Py.Object.t -> ?tol:Py.Object.t -> ?maxiter:Py.Object.t -> ?m1:Py.Object.t -> ?m2:Py.Object.t -> ?callback:Py.Object.t -> ?atol:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Use Quasi-Minimal Residual iteration to solve ``Ax = b``.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real-valued N-by-N matrix of the linear system.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` and ``A^T x`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).

Returns
-------
x : {array, matrix}
    The converged solution.
info : integer
    Provides convergence information:
        0  : successful exit
        >0 : convergence to tolerance not achieved, number of iterations
        <0 : illegal input or breakdown

Other Parameters
----------------
x0  : {array, matrix}
    Starting guess for the solution.
tol, atol : float, optional
    Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
    The default for ``atol`` is ``'legacy'``, which emulates
    a different legacy behavior.

    .. warning::

       The default value for `atol` will be changed in a future release.
       For future compatibility, specify `atol` explicitly.
maxiter : integer
    Maximum number of iterations.  Iteration will stop after maxiter
    steps even if the specified tolerance has not been achieved.
M1 : {sparse matrix, dense matrix, LinearOperator}
    Left preconditioner for A.
M2 : {sparse matrix, dense matrix, LinearOperator}
    Right preconditioner for A. Used together with the left
    preconditioner M1.  The matrix M1*A*M2 should have better
    conditioned than A alone.
callback : function
    User-supplied function to call after each iteration.  It is called
    as callback(xk), where xk is the current solution vector.

See Also
--------
LinearOperator

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import qmr
>>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
>>> b = np.array([2, 4, -1], dtype=float)
>>> x, exitCode = qmr(A, b)
>>> print(exitCode)            # 0 indicates successful convergence
0
>>> np.allclose(A.dot(x), b)
True
*)


end

module Iterative : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val bicg : ?x0:Py.Object.t -> ?tol:Py.Object.t -> ?maxiter:Py.Object.t -> ?m:Py.Object.t -> ?callback:Py.Object.t -> ?atol:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Use BIConjugate Gradient iteration to solve ``Ax = b``.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real or complex N-by-N matrix of the linear system.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` and ``A^T x`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).

Returns
-------
x : {array, matrix}
    The converged solution.
info : integer
    Provides convergence information:
        0  : successful exit
        >0 : convergence to tolerance not achieved, number of iterations
        <0 : illegal input or breakdown

Other Parameters
----------------
x0  : {array, matrix}
    Starting guess for the solution.
tol, atol : float, optional
    Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
    The default for ``atol`` is ``'legacy'``, which emulates
    a different legacy behavior.

    .. warning::

       The default value for `atol` will be changed in a future release.
       For future compatibility, specify `atol` explicitly.
maxiter : integer
    Maximum number of iterations.  Iteration will stop after maxiter
    steps even if the specified tolerance has not been achieved.
M : {sparse matrix, dense matrix, LinearOperator}
    Preconditioner for A.  The preconditioner should approximate the
    inverse of A.  Effective preconditioning dramatically improves the
    rate of convergence, which implies that fewer iterations are needed
    to reach a given error tolerance.
callback : function
    User-supplied function to call after each iteration.  It is called
    as callback(xk), where xk is the current solution vector.
*)

val bicgstab : ?x0:Py.Object.t -> ?tol:Py.Object.t -> ?maxiter:Py.Object.t -> ?m:Py.Object.t -> ?callback:Py.Object.t -> ?atol:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Use BIConjugate Gradient STABilized iteration to solve ``Ax = b``.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real or complex N-by-N matrix of the linear system.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).

Returns
-------
x : {array, matrix}
    The converged solution.
info : integer
    Provides convergence information:
        0  : successful exit
        >0 : convergence to tolerance not achieved, number of iterations
        <0 : illegal input or breakdown

Other Parameters
----------------
x0  : {array, matrix}
    Starting guess for the solution.
tol, atol : float, optional
    Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
    The default for ``atol`` is ``'legacy'``, which emulates
    a different legacy behavior.

    .. warning::

       The default value for `atol` will be changed in a future release.
       For future compatibility, specify `atol` explicitly.
maxiter : integer
    Maximum number of iterations.  Iteration will stop after maxiter
    steps even if the specified tolerance has not been achieved.
M : {sparse matrix, dense matrix, LinearOperator}
    Preconditioner for A.  The preconditioner should approximate the
    inverse of A.  Effective preconditioning dramatically improves the
    rate of convergence, which implies that fewer iterations are needed
    to reach a given error tolerance.
callback : function
    User-supplied function to call after each iteration.  It is called
    as callback(xk), where xk is the current solution vector.
*)

val cg : ?x0:Py.Object.t -> ?tol:Py.Object.t -> ?maxiter:Py.Object.t -> ?m:Py.Object.t -> ?callback:Py.Object.t -> ?atol:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Use Conjugate Gradient iteration to solve ``Ax = b``.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real or complex N-by-N matrix of the linear system.
    ``A`` must represent a hermitian, positive definite matrix.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).

Returns
-------
x : {array, matrix}
    The converged solution.
info : integer
    Provides convergence information:
        0  : successful exit
        >0 : convergence to tolerance not achieved, number of iterations
        <0 : illegal input or breakdown

Other Parameters
----------------
x0  : {array, matrix}
    Starting guess for the solution.
tol, atol : float, optional
    Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
    The default for ``atol`` is ``'legacy'``, which emulates
    a different legacy behavior.

    .. warning::

       The default value for `atol` will be changed in a future release.
       For future compatibility, specify `atol` explicitly.
maxiter : integer
    Maximum number of iterations.  Iteration will stop after maxiter
    steps even if the specified tolerance has not been achieved.
M : {sparse matrix, dense matrix, LinearOperator}
    Preconditioner for A.  The preconditioner should approximate the
    inverse of A.  Effective preconditioning dramatically improves the
    rate of convergence, which implies that fewer iterations are needed
    to reach a given error tolerance.
callback : function
    User-supplied function to call after each iteration.  It is called
    as callback(xk), where xk is the current solution vector.
*)

val cgs : ?x0:Py.Object.t -> ?tol:Py.Object.t -> ?maxiter:Py.Object.t -> ?m:Py.Object.t -> ?callback:Py.Object.t -> ?atol:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Use Conjugate Gradient Squared iteration to solve ``Ax = b``.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real-valued N-by-N matrix of the linear system.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).

Returns
-------
x : {array, matrix}
    The converged solution.
info : integer
    Provides convergence information:
        0  : successful exit
        >0 : convergence to tolerance not achieved, number of iterations
        <0 : illegal input or breakdown

Other Parameters
----------------
x0  : {array, matrix}
    Starting guess for the solution.
tol, atol : float, optional
    Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
    The default for ``atol`` is ``'legacy'``, which emulates
    a different legacy behavior.

    .. warning::

       The default value for `atol` will be changed in a future release.
       For future compatibility, specify `atol` explicitly.
maxiter : integer
    Maximum number of iterations.  Iteration will stop after maxiter
    steps even if the specified tolerance has not been achieved.
M : {sparse matrix, dense matrix, LinearOperator}
    Preconditioner for A.  The preconditioner should approximate the
    inverse of A.  Effective preconditioning dramatically improves the
    rate of convergence, which implies that fewer iterations are needed
    to reach a given error tolerance.
callback : function
    User-supplied function to call after each iteration.  It is called
    as callback(xk), where xk is the current solution vector.
*)

val gmres : ?x0:Py.Object.t -> ?tol:Py.Object.t -> ?restart:Py.Object.t -> ?maxiter:Py.Object.t -> ?m:Py.Object.t -> ?callback:Py.Object.t -> ?restrt:Py.Object.t -> ?atol:Py.Object.t -> ?callback_type:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Use Generalized Minimal RESidual iteration to solve ``Ax = b``.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real or complex N-by-N matrix of the linear system.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).

Returns
-------
x : {array, matrix}
    The converged solution.
info : int
    Provides convergence information:
      * 0  : successful exit
      * >0 : convergence to tolerance not achieved, number of iterations
      * <0 : illegal input or breakdown

Other parameters
----------------
x0 : {array, matrix}
    Starting guess for the solution (a vector of zeros by default).
tol, atol : float, optional
    Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
    The default for ``atol`` is ``'legacy'``, which emulates
    a different legacy behavior.

    .. warning::

       The default value for `atol` will be changed in a future release.
       For future compatibility, specify `atol` explicitly.
restart : int, optional
    Number of iterations between restarts. Larger values increase
    iteration cost, but may be necessary for convergence.
    Default is 20.
maxiter : int, optional
    Maximum number of iterations (restart cycles).  Iteration will stop
    after maxiter steps even if the specified tolerance has not been
    achieved.
M : {sparse matrix, dense matrix, LinearOperator}
    Inverse of the preconditioner of A.  M should approximate the
    inverse of A and be easy to solve for (see Notes).  Effective
    preconditioning dramatically improves the rate of convergence,
    which implies that fewer iterations are needed to reach a given
    error tolerance.  By default, no preconditioner is used.
callback : function
    User-supplied function to call after each iteration.  It is called
    as `callback(args)`, where `args` are selected by `callback_type`.
callback_type : {'x', 'pr_norm', 'legacy'}, optional
    Callback function argument requested:
      - ``x``: current iterate (ndarray), called on every restart
      - ``pr_norm``: relative (preconditioned) residual norm (float),
        called on every inner iteration
      - ``legacy`` (default): same as ``pr_norm``, but also changes the
        meaning of 'maxiter' to count inner iterations instead of restart
        cycles.
restrt : int, optional
    DEPRECATED - use `restart` instead.

See Also
--------
LinearOperator

Notes
-----
A preconditioner, P, is chosen such that P is close to A but easy to solve
for. The preconditioner parameter required by this routine is
``M = P^-1``. The inverse should preferably not be calculated
explicitly.  Rather, use the following template to produce M::

  # Construct a linear operator that computes P^-1 * x.
  import scipy.sparse.linalg as spla
  M_x = lambda x: spla.spsolve(P, x)
  M = spla.LinearOperator((n, n), M_x)

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import gmres
>>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
>>> b = np.array([2, 4, -1], dtype=float)
>>> x, exitCode = gmres(A, b)
>>> print(exitCode)            # 0 indicates successful convergence
0
>>> np.allclose(A.dot(x), b)
True
*)

val make_system : a:Py.Object.t -> m:Py.Object.t -> x0:[`Ndarray of [>`Ndarray] Np.Obj.t | `None] -> b:[>`Ndarray] Np.Obj.t -> unit -> (Py.Object.t * Py.Object.t * Py.Object.t * Py.Object.t)
(**
Make a linear system Ax=b

Parameters
----------
A : LinearOperator
    sparse or dense matrix (or any valid input to aslinearoperator)
M : {LinearOperator, Nones}
    preconditioner
    sparse or dense matrix (or any valid input to aslinearoperator)
x0 : {array_like, None}
    initial guess to iterative method
b : array_like
    right hand side

Returns
-------
(A, M, x, b, postprocess)
    A : LinearOperator
        matrix of the linear system
    M : LinearOperator
        preconditioner
    x : rank 1 ndarray
        initial guess
    b : rank 1 ndarray
        right hand side
    postprocess : function
        converts the solution vector to the appropriate
        type and dimensions (e.g. (N,1) matrix)
*)

val non_reentrant : ?err_msg:Py.Object.t -> unit -> Py.Object.t
(**
Decorate a function with a threading lock and prevent reentrant calls.
*)

val qmr : ?x0:Py.Object.t -> ?tol:Py.Object.t -> ?maxiter:Py.Object.t -> ?m1:Py.Object.t -> ?m2:Py.Object.t -> ?callback:Py.Object.t -> ?atol:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Use Quasi-Minimal Residual iteration to solve ``Ax = b``.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real-valued N-by-N matrix of the linear system.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` and ``A^T x`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).

Returns
-------
x : {array, matrix}
    The converged solution.
info : integer
    Provides convergence information:
        0  : successful exit
        >0 : convergence to tolerance not achieved, number of iterations
        <0 : illegal input or breakdown

Other Parameters
----------------
x0  : {array, matrix}
    Starting guess for the solution.
tol, atol : float, optional
    Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
    The default for ``atol`` is ``'legacy'``, which emulates
    a different legacy behavior.

    .. warning::

       The default value for `atol` will be changed in a future release.
       For future compatibility, specify `atol` explicitly.
maxiter : integer
    Maximum number of iterations.  Iteration will stop after maxiter
    steps even if the specified tolerance has not been achieved.
M1 : {sparse matrix, dense matrix, LinearOperator}
    Left preconditioner for A.
M2 : {sparse matrix, dense matrix, LinearOperator}
    Right preconditioner for A. Used together with the left
    preconditioner M1.  The matrix M1*A*M2 should have better
    conditioned than A alone.
callback : function
    User-supplied function to call after each iteration.  It is called
    as callback(xk), where xk is the current solution vector.

See Also
--------
LinearOperator

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import qmr
>>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
>>> b = np.array([2, 4, -1], dtype=float)
>>> x, exitCode = qmr(A, b)
>>> print(exitCode)            # 0 indicates successful convergence
0
>>> np.allclose(A.dot(x), b)
True
*)

val set_docstring : ?footer:Py.Object.t -> ?atol_default:Py.Object.t -> header:Py.Object.t -> ainfo:Py.Object.t -> unit -> Py.Object.t
(**
None
*)


end

module Linsolve : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val asarray : ?dtype:Np.Dtype.t -> ?order:[`F | `C] -> a:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Convert the input to an array.

Parameters
----------
a : array_like
    Input data, in any form that can be converted to an array.  This
    includes lists, lists of tuples, tuples, tuples of tuples, tuples
    of lists and ndarrays.
dtype : data-type, optional
    By default, the data-type is inferred from the input data.
order : {'C', 'F'}, optional
    Whether to use row-major (C-style) or
    column-major (Fortran-style) memory representation.
    Defaults to 'C'.

Returns
-------
out : ndarray
    Array interpretation of `a`.  No copy is performed if the input
    is already an ndarray with matching dtype and order.  If `a` is a
    subclass of ndarray, a base class ndarray is returned.

See Also
--------
asanyarray : Similar function which passes through subclasses.
ascontiguousarray : Convert input to a contiguous array.
asfarray : Convert input to a floating point ndarray.
asfortranarray : Convert input to an ndarray with column-major
                 memory order.
asarray_chkfinite : Similar function which checks input for NaNs and Infs.
fromiter : Create an array from an iterator.
fromfunction : Construct an array by executing a function on grid
               positions.

Examples
--------
Convert a list into an array:

>>> a = [1, 2]
>>> np.asarray(a)
array([1, 2])

Existing arrays are not copied:

>>> a = np.array([1, 2])
>>> np.asarray(a) is a
True

If `dtype` is set, array is copied only if dtype does not match:

>>> a = np.array([1, 2], dtype=np.float32)
>>> np.asarray(a, dtype=np.float32) is a
True
>>> np.asarray(a, dtype=np.float64) is a
False

Contrary to `asanyarray`, ndarray subclasses are not passed through:

>>> issubclass(np.recarray, np.ndarray)
True
>>> a = np.array([(1.0, 2), (3.0, 4)], dtype='f4,i4').view(np.recarray)
>>> np.asarray(a) is a
False
>>> np.asanyarray(a) is a
True
*)

val factorized : [>`Ndarray] Np.Obj.t -> Py.Object.t
(**
Return a function for solving a sparse linear system, with A pre-factorized.

Parameters
----------
A : (N, N) array_like
    Input.

Returns
-------
solve : callable
    To solve the linear system of equations given in `A`, the `solve`
    callable should be passed an ndarray of shape (N,).

Examples
--------
>>> from scipy.sparse.linalg import factorized
>>> A = np.array([[ 3. ,  2. , -1. ],
...               [ 2. , -2. ,  4. ],
...               [-1. ,  0.5, -1. ]])
>>> solve = factorized(A) # Makes LU decomposition.
>>> rhs1 = np.array([1, -2, 0])
>>> solve(rhs1) # Uses the LU factors.
array([ 1., -2., -2.])
*)

val is_pydata_spmatrix : Py.Object.t -> Py.Object.t
(**
Check whether object is pydata/sparse matrix, avoiding importing the module.
*)

val isspmatrix : Py.Object.t -> Py.Object.t
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

val isspmatrix_csc : Py.Object.t -> Py.Object.t
(**
Is x of csc_matrix type?

Parameters
----------
x
    object to check for being a csc matrix

Returns
-------
bool
    True if x is a csc matrix, False otherwise

Examples
--------
>>> from scipy.sparse import csc_matrix, isspmatrix_csc
>>> isspmatrix_csc(csc_matrix([[5]]))
True

>>> from scipy.sparse import csc_matrix, csr_matrix, isspmatrix_csc
>>> isspmatrix_csc(csr_matrix([[5]]))
False
*)

val isspmatrix_csr : Py.Object.t -> Py.Object.t
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

val spilu : ?drop_tol:float -> ?fill_factor:float -> ?drop_rule:string -> ?permc_spec:Py.Object.t -> ?diag_pivot_thresh:Py.Object.t -> ?relax:Py.Object.t -> ?panel_size:Py.Object.t -> ?options:Py.Object.t -> a:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Compute an incomplete LU decomposition for a sparse, square matrix.

The resulting object is an approximation to the inverse of `A`.

Parameters
----------
A : (N, N) array_like
    Sparse matrix to factorize
drop_tol : float, optional
    Drop tolerance (0 <= tol <= 1) for an incomplete LU decomposition.
    (default: 1e-4)
fill_factor : float, optional
    Specifies the fill ratio upper bound (>= 1.0) for ILU. (default: 10)
drop_rule : str, optional
    Comma-separated string of drop rules to use.
    Available rules: ``basic``, ``prows``, ``column``, ``area``,
    ``secondary``, ``dynamic``, ``interp``. (Default: ``basic,area``)

    See SuperLU documentation for details.

Remaining other options
    Same as for `splu`

Returns
-------
invA_approx : scipy.sparse.linalg.SuperLU
    Object, which has a ``solve`` method.

See also
--------
splu : complete LU decomposition

Notes
-----
To improve the better approximation to the inverse, you may need to
increase `fill_factor` AND decrease `drop_tol`.

This function uses the SuperLU library.

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import spilu
>>> A = csc_matrix([[1., 0., 0.], [5., 0., 2.], [0., -1., 0.]], dtype=float)
>>> B = spilu(A)
>>> x = np.array([1., 2., 3.], dtype=float)
>>> B.solve(x)
array([ 1. , -3. , -1.5])
>>> A.dot(B.solve(x))
array([ 1.,  2.,  3.])
>>> B.solve(A.dot(x))
array([ 1.,  2.,  3.])
*)

val splu : ?permc_spec:string -> ?diag_pivot_thresh:float -> ?relax:int -> ?panel_size:int -> ?options:Py.Object.t -> a:[>`Spmatrix] Np.Obj.t -> unit -> Py.Object.t
(**
Compute the LU decomposition of a sparse, square matrix.

Parameters
----------
A : sparse matrix
    Sparse matrix to factorize. Should be in CSR or CSC format.
permc_spec : str, optional
    How to permute the columns of the matrix for sparsity preservation.
    (default: 'COLAMD')

    - ``NATURAL``: natural ordering.
    - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
    - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
    - ``COLAMD``: approximate minimum degree column ordering

diag_pivot_thresh : float, optional
    Threshold used for a diagonal entry to be an acceptable pivot.
    See SuperLU user's guide for details [1]_
relax : int, optional
    Expert option for customizing the degree of relaxing supernodes.
    See SuperLU user's guide for details [1]_
panel_size : int, optional
    Expert option for customizing the panel size.
    See SuperLU user's guide for details [1]_
options : dict, optional
    Dictionary containing additional expert options to SuperLU.
    See SuperLU user guide [1]_ (section 2.4 on the 'Options' argument)
    for more details. For example, you can specify
    ``options=dict(Equil=False, IterRefine='SINGLE'))``
    to turn equilibration off and perform a single iterative refinement.

Returns
-------
invA : scipy.sparse.linalg.SuperLU
    Object, which has a ``solve`` method.

See also
--------
spilu : incomplete LU decomposition

Notes
-----
This function uses the SuperLU library.

References
----------
.. [1] SuperLU http://crd.lbl.gov/~xiaoye/SuperLU/

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import splu
>>> A = csc_matrix([[1., 0., 0.], [5., 0., 2.], [0., -1., 0.]], dtype=float)
>>> B = splu(A)
>>> x = np.array([1., 2., 3.], dtype=float)
>>> B.solve(x)
array([ 1. , -3. , -1.5])
>>> A.dot(B.solve(x))
array([ 1.,  2.,  3.])
>>> B.solve(A.dot(x))
array([ 1.,  2.,  3.])
*)

val spsolve : ?permc_spec:string -> ?use_umfpack:bool -> a:[>`ArrayLike] Np.Obj.t -> b:[>`ArrayLike] Np.Obj.t -> unit -> [>`ArrayLike] Np.Obj.t
(**
Solve the sparse linear system Ax=b, where b may be a vector or a matrix.

Parameters
----------
A : ndarray or sparse matrix
    The square matrix A will be converted into CSC or CSR form
b : ndarray or sparse matrix
    The matrix or vector representing the right hand side of the equation.
    If a vector, b.shape must be (n,) or (n, 1).
permc_spec : str, optional
    How to permute the columns of the matrix for sparsity preservation.
    (default: 'COLAMD')

    - ``NATURAL``: natural ordering.
    - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
    - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
    - ``COLAMD``: approximate minimum degree column ordering
use_umfpack : bool, optional
    if True (default) then use umfpack for the solution.  This is
    only referenced if b is a vector and ``scikit-umfpack`` is installed.

Returns
-------
x : ndarray or sparse matrix
    the solution of the sparse linear equation.
    If b is a vector, then x is a vector of size A.shape[1]
    If b is a matrix, then x is a matrix of size (A.shape[1], b.shape[1])

Notes
-----
For solving the matrix expression AX = B, this solver assumes the resulting
matrix X is sparse, as is often the case for very sparse inputs.  If the
resulting X is dense, the construction of this sparse result will be
relatively expensive.  In that case, consider converting A to a dense
matrix and using scipy.linalg.solve or its variants.

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import spsolve
>>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
>>> B = csc_matrix([[2, 0], [-1, 0], [2, 0]], dtype=float)
>>> x = spsolve(A, B)
>>> np.allclose(A.dot(x).todense(), B.todense())
True
*)

val spsolve_triangular : ?lower:bool -> ?overwrite_A:bool -> ?overwrite_b:bool -> ?unit_diagonal:bool -> a:[>`Spmatrix] Np.Obj.t -> b:Py.Object.t -> unit -> Py.Object.t
(**
Solve the equation `A x = b` for `x`, assuming A is a triangular matrix.

Parameters
----------
A : (M, M) sparse matrix
    A sparse square triangular matrix. Should be in CSR format.
b : (M,) or (M, N) array_like
    Right-hand side matrix in `A x = b`
lower : bool, optional
    Whether `A` is a lower or upper triangular matrix.
    Default is lower triangular matrix.
overwrite_A : bool, optional
    Allow changing `A`. The indices of `A` are going to be sorted and zero
    entries are going to be removed.
    Enabling gives a performance gain. Default is False.
overwrite_b : bool, optional
    Allow overwriting data in `b`.
    Enabling gives a performance gain. Default is False.
    If `overwrite_b` is True, it should be ensured that
    `b` has an appropriate dtype to be able to store the result.
unit_diagonal : bool, optional
    If True, diagonal elements of `a` are assumed to be 1 and will not be
    referenced.

    .. versionadded:: 1.4.0

Returns
-------
x : (M,) or (M, N) ndarray
    Solution to the system `A x = b`. Shape of return matches shape of `b`.

Raises
------
LinAlgError
    If `A` is singular or not triangular.
ValueError
    If shape of `A` or shape of `b` do not match the requirements.

Notes
-----
.. versionadded:: 0.19.0

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> from scipy.sparse.linalg import spsolve_triangular
>>> A = csr_matrix([[3, 0, 0], [1, -1, 0], [2, 0, 1]], dtype=float)
>>> B = np.array([[2, 0], [-1, 0], [2, 0]], dtype=float)
>>> x = spsolve_triangular(A, B)
>>> np.allclose(A.dot(x), B)
True
*)

val use_solver : ?kwargs:(string * Py.Object.t) list -> unit -> Py.Object.t
(**
Select default sparse direct solver to be used.

Parameters
----------
useUmfpack : bool, optional
    Use UMFPACK over SuperLU. Has effect only if scikits.umfpack is
    installed. Default: True
assumeSortedIndices : bool, optional
    Allow UMFPACK to skip the step of sorting indices for a CSR/CSC matrix.
    Has effect only if useUmfpack is True and scikits.umfpack is installed.
    Default: False

Notes
-----
The default sparse solver is umfpack when available
(scikits.umfpack is installed). This can be changed by passing
useUmfpack = False, which then causes the always present SuperLU
based solver to be used.

Umfpack requires a CSR/CSC matrix to have sorted column/row indices. If
sure that the matrix fulfills this, pass ``assumeSortedIndices=True``
to gain some speed.
*)

val warn : ?category:Py.Object.t -> ?stacklevel:Py.Object.t -> ?source:Py.Object.t -> message:Py.Object.t -> unit -> Py.Object.t
(**
Issue a warning, or maybe ignore it or raise an exception.
*)


end

module Matfuncs : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module MatrixPowerOperator : sig
type tag = [`MatrixPowerOperator]
type t = [`MatrixPowerOperator | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> t
(**
Common interface for performing matrix vector products

Many iterative methods (e.g. cg, gmres) do not need to know the
individual entries of a matrix to solve a linear system A*x=b.
Such solvers only require the computation of matrix vector
products, A*v where v is a dense vector.  This class serves as
an abstract interface between iterative solvers and matrix-like
objects.

To construct a concrete LinearOperator, either pass appropriate
callables to the constructor of this class, or subclass it.

A subclass must implement either one of the methods ``_matvec``
and ``_matmat``, and the attributes/properties ``shape`` (pair of
integers) and ``dtype`` (may be None). It may call the ``__init__``
on this class to have these attributes validated. Implementing
``_matvec`` automatically implements ``_matmat`` (using a naive
algorithm) and vice-versa.

Optionally, a subclass may implement ``_rmatvec`` or ``_adjoint``
to implement the Hermitian adjoint (conjugate transpose). As with
``_matvec`` and ``_matmat``, implementing either ``_rmatvec`` or
``_adjoint`` implements the other automatically. Implementing
``_adjoint`` is preferable; ``_rmatvec`` is mostly there for
backwards compatibility.

Parameters
----------
shape : tuple
    Matrix dimensions (M, N).
matvec : callable f(v)
    Returns returns A * v.
rmatvec : callable f(v)
    Returns A^H * v, where A^H is the conjugate transpose of A.
matmat : callable f(V)
    Returns A * V, where V is a dense matrix with dimensions (N, K).
dtype : dtype
    Data type of the matrix.
rmatmat : callable f(V)
    Returns A^H * V, where V is a dense matrix with dimensions (M, K).

Attributes
----------
args : tuple
    For linear operators describing products etc. of other linear
    operators, the operands of the binary operation.

See Also
--------
aslinearoperator : Construct LinearOperators

Notes
-----
The user-defined matvec() function must properly handle the case
where v has shape (N,) as well as the (N,1) case.  The shape of
the return type is handled internally by LinearOperator.

LinearOperator instances can also be multiplied, added with each
other and exponentiated, all lazily: the result of these operations
is always a new, composite LinearOperator, that defers linear
operations to the original operators and combines the results.

More details regarding how to subclass a LinearOperator and several
examples of concrete LinearOperator instances can be found in the
external project `PyLops <https://pylops.readthedocs.io>`_.


Examples
--------
>>> import numpy as np
>>> from scipy.sparse.linalg import LinearOperator
>>> def mv(v):
...     return np.array([2*v[0], 3*v[1]])
...
>>> A = LinearOperator((2,2), matvec=mv)
>>> A
<2x2 _CustomLinearOperator with dtype=float64>
>>> A.matvec(np.ones(2))
array([ 2.,  3.])
>>> A * np.ones(2)
array([ 2.,  3.])
*)

val adjoint : [> tag] Obj.t -> Py.Object.t
(**
Hermitian adjoint.

Returns the Hermitian adjoint of self, aka the Hermitian
conjugate or Hermitian transpose. For a complex matrix, the
Hermitian adjoint is equal to the conjugate transpose.

Can be abbreviated self.H instead of self.adjoint().

Returns
-------
A_H : LinearOperator
    Hermitian adjoint of self.
*)

val dot : x:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Matrix-matrix or matrix-vector multiplication.

Parameters
----------
x : array_like
    1-d or 2-d array, representing a vector or matrix.

Returns
-------
Ax : array
    1-d or 2-d array (depending on the shape of x) that represents
    the result of applying this linear operator on x.
*)

val matmat : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Matrix-matrix multiplication.

Performs the operation y=A*X where A is an MxN linear
operator and X dense N*K matrix or ndarray.

Parameters
----------
X : {matrix, ndarray}
    An array with shape (N,K).

Returns
-------
Y : {matrix, ndarray}
    A matrix or ndarray with shape (M,K) depending on
    the type of the X argument.

Notes
-----
This matmat wraps any user-specified matmat routine or overridden
_matmat method to ensure that y has the correct type.
*)

val matvec : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Matrix-vector multiplication.

Performs the operation y=A*x where A is an MxN linear
operator and x is a column vector or 1-d array.

Parameters
----------
x : {matrix, ndarray}
    An array with shape (N,) or (N,1).

Returns
-------
y : {matrix, ndarray}
    A matrix or ndarray with shape (M,) or (M,1) depending
    on the type and shape of the x argument.

Notes
-----
This matvec wraps the user-specified matvec routine or overridden
_matvec method to ensure that y has the correct shape and type.
*)

val rmatmat : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Adjoint matrix-matrix multiplication.

Performs the operation y = A^H * x where A is an MxN linear
operator and x is a column vector or 1-d array, or 2-d array.
The default implementation defers to the adjoint.

Parameters
----------
X : {matrix, ndarray}
    A matrix or 2D array.

Returns
-------
Y : {matrix, ndarray}
    A matrix or 2D array depending on the type of the input.

Notes
-----
This rmatmat wraps the user-specified rmatmat routine.
*)

val rmatvec : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Adjoint matrix-vector multiplication.

Performs the operation y = A^H * x where A is an MxN linear
operator and x is a column vector or 1-d array.

Parameters
----------
x : {matrix, ndarray}
    An array with shape (M,) or (M,1).

Returns
-------
y : {matrix, ndarray}
    A matrix or ndarray with shape (N,) or (N,1) depending
    on the type and shape of the x argument.

Notes
-----
This rmatvec wraps the user-specified rmatvec routine or overridden
_rmatvec method to ensure that y has the correct shape and type.
*)

val transpose : [> tag] Obj.t -> Py.Object.t
(**
Transpose this linear operator.

Returns a LinearOperator that represents the transpose of this one.
Can be abbreviated self.T instead of self.transpose().
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module ProductOperator : sig
type tag = [`ProductOperator]
type t = [`Object | `ProductOperator] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> t
(**
For now, this is limited to products of multiple square matrices.
*)

val adjoint : [> tag] Obj.t -> Py.Object.t
(**
Hermitian adjoint.

Returns the Hermitian adjoint of self, aka the Hermitian
conjugate or Hermitian transpose. For a complex matrix, the
Hermitian adjoint is equal to the conjugate transpose.

Can be abbreviated self.H instead of self.adjoint().

Returns
-------
A_H : LinearOperator
    Hermitian adjoint of self.
*)

val dot : x:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Matrix-matrix or matrix-vector multiplication.

Parameters
----------
x : array_like
    1-d or 2-d array, representing a vector or matrix.

Returns
-------
Ax : array
    1-d or 2-d array (depending on the shape of x) that represents
    the result of applying this linear operator on x.
*)

val matmat : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Matrix-matrix multiplication.

Performs the operation y=A*X where A is an MxN linear
operator and X dense N*K matrix or ndarray.

Parameters
----------
X : {matrix, ndarray}
    An array with shape (N,K).

Returns
-------
Y : {matrix, ndarray}
    A matrix or ndarray with shape (M,K) depending on
    the type of the X argument.

Notes
-----
This matmat wraps any user-specified matmat routine or overridden
_matmat method to ensure that y has the correct type.
*)

val matvec : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Matrix-vector multiplication.

Performs the operation y=A*x where A is an MxN linear
operator and x is a column vector or 1-d array.

Parameters
----------
x : {matrix, ndarray}
    An array with shape (N,) or (N,1).

Returns
-------
y : {matrix, ndarray}
    A matrix or ndarray with shape (M,) or (M,1) depending
    on the type and shape of the x argument.

Notes
-----
This matvec wraps the user-specified matvec routine or overridden
_matvec method to ensure that y has the correct shape and type.
*)

val rmatmat : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Adjoint matrix-matrix multiplication.

Performs the operation y = A^H * x where A is an MxN linear
operator and x is a column vector or 1-d array, or 2-d array.
The default implementation defers to the adjoint.

Parameters
----------
X : {matrix, ndarray}
    A matrix or 2D array.

Returns
-------
Y : {matrix, ndarray}
    A matrix or 2D array depending on the type of the input.

Notes
-----
This rmatmat wraps the user-specified rmatmat routine.
*)

val rmatvec : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Adjoint matrix-vector multiplication.

Performs the operation y = A^H * x where A is an MxN linear
operator and x is a column vector or 1-d array.

Parameters
----------
x : {matrix, ndarray}
    An array with shape (M,) or (M,1).

Returns
-------
y : {matrix, ndarray}
    A matrix or ndarray with shape (N,) or (N,1) depending
    on the type and shape of the x argument.

Notes
-----
This rmatvec wraps the user-specified rmatvec routine or overridden
_rmatvec method to ensure that y has the correct shape and type.
*)

val transpose : [> tag] Obj.t -> Py.Object.t
(**
Transpose this linear operator.

Returns a LinearOperator that represents the transpose of this one.
Can be abbreviated self.T instead of self.transpose().
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val expm : [>`ArrayLike] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the matrix exponential using Pade approximation.

Parameters
----------
A : (M,M) array_like or sparse matrix
    2D Array or Matrix (sparse or dense) to be exponentiated

Returns
-------
expA : (M,M) ndarray
    Matrix exponential of `A`

Notes
-----
This is algorithm (6.1) which is a simplification of algorithm (5.1).

.. versionadded:: 0.12.0

References
----------
.. [1] Awad H. Al-Mohy and Nicholas J. Higham (2009)
       'A New Scaling and Squaring Algorithm for the Matrix Exponential.'
       SIAM Journal on Matrix Analysis and Applications.
       31 (3). pp. 970-989. ISSN 1095-7162

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import expm
>>> A = csc_matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
>>> A.todense()
matrix([[1, 0, 0],
        [0, 2, 0],
        [0, 0, 3]], dtype=int64)
>>> Aexp = expm(A)
>>> Aexp
<3x3 sparse matrix of type '<class 'numpy.float64'>'
    with 3 stored elements in Compressed Sparse Column format>
>>> Aexp.todense()
matrix([[  2.71828183,   0.        ,   0.        ],
        [  0.        ,   7.3890561 ,   0.        ],
        [  0.        ,   0.        ,  20.08553692]])
*)

val inv : [>`ArrayLike] Np.Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Compute the inverse of a sparse matrix

Parameters
----------
A : (M,M) ndarray or sparse matrix
    square matrix to be inverted

Returns
-------
Ainv : (M,M) ndarray or sparse matrix
    inverse of `A`

Notes
-----
This computes the sparse inverse of `A`.  If the inverse of `A` is expected
to be non-sparse, it will likely be faster to convert `A` to dense and use
scipy.linalg.inv.

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import inv
>>> A = csc_matrix([[1., 0.], [1., 2.]])
>>> Ainv = inv(A)
>>> Ainv
<2x2 sparse matrix of type '<class 'numpy.float64'>'
    with 3 stored elements in Compressed Sparse Column format>
>>> A.dot(Ainv)
<2x2 sparse matrix of type '<class 'numpy.float64'>'
    with 2 stored elements in Compressed Sparse Column format>
>>> A.dot(Ainv).todense()
matrix([[ 1.,  0.],
        [ 0.,  1.]])

.. versionadded:: 0.12.0
*)

val is_pydata_spmatrix : Py.Object.t -> Py.Object.t
(**
Check whether object is pydata/sparse matrix, avoiding importing the module.
*)

val isspmatrix : Py.Object.t -> Py.Object.t
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

val solve : ?sym_pos:bool -> ?lower:bool -> ?overwrite_a:bool -> ?overwrite_b:bool -> ?debug:Py.Object.t -> ?check_finite:bool -> ?assume_a:string -> ?transposed:bool -> a:[>`Ndarray] Np.Obj.t -> b:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Solves the linear equation set ``a * x = b`` for the unknown ``x``
for square ``a`` matrix.

If the data matrix is known to be a particular type then supplying the
corresponding string to ``assume_a`` key chooses the dedicated solver.
The available options are

===================  ========
 generic matrix       'gen'
 symmetric            'sym'
 hermitian            'her'
 positive definite    'pos'
===================  ========

If omitted, ``'gen'`` is the default structure.

The datatype of the arrays define which solver is called regardless
of the values. In other words, even when the complex array entries have
precisely zero imaginary parts, the complex solver will be called based
on the data type of the array.

Parameters
----------
a : (N, N) array_like
    Square input data
b : (N, NRHS) array_like
    Input data for the right hand side.
sym_pos : bool, optional
    Assume `a` is symmetric and positive definite. This key is deprecated
    and assume_a = 'pos' keyword is recommended instead. The functionality
    is the same. It will be removed in the future.
lower : bool, optional
    If True, only the data contained in the lower triangle of `a`. Default
    is to use upper triangle. (ignored for ``'gen'``)
overwrite_a : bool, optional
    Allow overwriting data in `a` (may enhance performance).
    Default is False.
overwrite_b : bool, optional
    Allow overwriting data in `b` (may enhance performance).
    Default is False.
check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.
assume_a : str, optional
    Valid entries are explained above.
transposed: bool, optional
    If True, ``a^T x = b`` for real matrices, raises `NotImplementedError`
    for complex matrices (only for True).

Returns
-------
x : (N, NRHS) ndarray
    The solution array.

Raises
------
ValueError
    If size mismatches detected or input a is not square.
LinAlgError
    If the matrix is singular.
LinAlgWarning
    If an ill-conditioned input a is detected.
NotImplementedError
    If transposed is True and input a is a complex matrix.

Examples
--------
Given `a` and `b`, solve for `x`:

>>> a = np.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]])
>>> b = np.array([2, 4, -1])
>>> from scipy import linalg
>>> x = linalg.solve(a, b)
>>> x
array([ 2., -2.,  9.])
>>> np.dot(a, x) == b
array([ True,  True,  True], dtype=bool)

Notes
-----
If the input b matrix is a 1D array with N elements, when supplied
together with an NxN input a, it is assumed as a valid column vector
despite the apparent size mismatch. This is compatible with the
numpy.dot() behavior and the returned result is still 1D array.

The generic, symmetric, hermitian and positive definite solutions are
obtained via calling ?GESV, ?SYSV, ?HESV, and ?POSV routines of
LAPACK respectively.
*)

val solve_triangular : ?trans:[`Zero | `N | `C | `One | `Two | `T] -> ?lower:bool -> ?unit_diagonal:bool -> ?overwrite_b:bool -> ?debug:Py.Object.t -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> b:Py.Object.t -> unit -> Py.Object.t
(**
Solve the equation `a x = b` for `x`, assuming a is a triangular matrix.

Parameters
----------
a : (M, M) array_like
    A triangular matrix
b : (M,) or (M, N) array_like
    Right-hand side matrix in `a x = b`
lower : bool, optional
    Use only data contained in the lower triangle of `a`.
    Default is to use upper triangle.
trans : {0, 1, 2, 'N', 'T', 'C'}, optional
    Type of system to solve:

    ========  =========
    trans     system
    ========  =========
    0 or 'N'  a x  = b
    1 or 'T'  a^T x = b
    2 or 'C'  a^H x = b
    ========  =========
unit_diagonal : bool, optional
    If True, diagonal elements of `a` are assumed to be 1 and
    will not be referenced.
overwrite_b : bool, optional
    Allow overwriting data in `b` (may enhance performance)
check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
x : (M,) or (M, N) ndarray
    Solution to the system `a x = b`.  Shape of return matches `b`.

Raises
------
LinAlgError
    If `a` is singular

Notes
-----
.. versionadded:: 0.9.0

Examples
--------
Solve the lower triangular system a x = b, where::

         [3  0  0  0]       [4]
    a =  [2  1  0  0]   b = [2]
         [1  0  1  0]       [4]
         [1  1  1  1]       [2]

>>> from scipy.linalg import solve_triangular
>>> a = np.array([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
>>> b = np.array([4, 2, 4, 2])
>>> x = solve_triangular(a, b, lower=True)
>>> x
array([ 1.33333333, -0.66666667,  2.66666667, -1.33333333])
>>> a.dot(x)  # Check the result
array([ 4.,  2.,  4.,  2.])
*)

val speye : ?n:int -> ?k:int -> ?dtype:Np.Dtype.t -> ?format:string -> m:int -> unit -> Py.Object.t
(**
Sparse matrix with ones on diagonal

Returns a sparse (m x n) matrix where the k-th diagonal
is all ones and everything else is zeros.

Parameters
----------
m : int
    Number of rows in the matrix.
n : int, optional
    Number of columns. Default: `m`.
k : int, optional
    Diagonal to place ones on. Default: 0 (main diagonal).
dtype : dtype, optional
    Data type of the matrix.
format : str, optional
    Sparse format of the result, e.g. format='csr', etc.

Examples
--------
>>> from scipy import sparse
>>> sparse.eye(3).toarray()
array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]])
>>> sparse.eye(3, dtype=np.int8)
<3x3 sparse matrix of type '<class 'numpy.int8'>'
    with 3 stored elements (1 diagonals) in DIAgonal format>
*)

val spsolve : ?permc_spec:string -> ?use_umfpack:bool -> a:[>`ArrayLike] Np.Obj.t -> b:[>`ArrayLike] Np.Obj.t -> unit -> [>`ArrayLike] Np.Obj.t
(**
Solve the sparse linear system Ax=b, where b may be a vector or a matrix.

Parameters
----------
A : ndarray or sparse matrix
    The square matrix A will be converted into CSC or CSR form
b : ndarray or sparse matrix
    The matrix or vector representing the right hand side of the equation.
    If a vector, b.shape must be (n,) or (n, 1).
permc_spec : str, optional
    How to permute the columns of the matrix for sparsity preservation.
    (default: 'COLAMD')

    - ``NATURAL``: natural ordering.
    - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
    - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
    - ``COLAMD``: approximate minimum degree column ordering
use_umfpack : bool, optional
    if True (default) then use umfpack for the solution.  This is
    only referenced if b is a vector and ``scikit-umfpack`` is installed.

Returns
-------
x : ndarray or sparse matrix
    the solution of the sparse linear equation.
    If b is a vector, then x is a vector of size A.shape[1]
    If b is a matrix, then x is a matrix of size (A.shape[1], b.shape[1])

Notes
-----
For solving the matrix expression AX = B, this solver assumes the resulting
matrix X is sparse, as is often the case for very sparse inputs.  If the
resulting X is dense, the construction of this sparse result will be
relatively expensive.  In that case, consider converting A to a dense
matrix and using scipy.linalg.solve or its variants.

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import spsolve
>>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
>>> B = csc_matrix([[2, 0], [-1, 0], [2, 0]], dtype=float)
>>> x = spsolve(A, B)
>>> np.allclose(A.dot(x).todense(), B.todense())
True
*)


end

module Utils : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module IdentityOperator : sig
type tag = [`IdentityOperator]
type t = [`IdentityOperator | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> t
(**
Common interface for performing matrix vector products

Many iterative methods (e.g. cg, gmres) do not need to know the
individual entries of a matrix to solve a linear system A*x=b.
Such solvers only require the computation of matrix vector
products, A*v where v is a dense vector.  This class serves as
an abstract interface between iterative solvers and matrix-like
objects.

To construct a concrete LinearOperator, either pass appropriate
callables to the constructor of this class, or subclass it.

A subclass must implement either one of the methods ``_matvec``
and ``_matmat``, and the attributes/properties ``shape`` (pair of
integers) and ``dtype`` (may be None). It may call the ``__init__``
on this class to have these attributes validated. Implementing
``_matvec`` automatically implements ``_matmat`` (using a naive
algorithm) and vice-versa.

Optionally, a subclass may implement ``_rmatvec`` or ``_adjoint``
to implement the Hermitian adjoint (conjugate transpose). As with
``_matvec`` and ``_matmat``, implementing either ``_rmatvec`` or
``_adjoint`` implements the other automatically. Implementing
``_adjoint`` is preferable; ``_rmatvec`` is mostly there for
backwards compatibility.

Parameters
----------
shape : tuple
    Matrix dimensions (M, N).
matvec : callable f(v)
    Returns returns A * v.
rmatvec : callable f(v)
    Returns A^H * v, where A^H is the conjugate transpose of A.
matmat : callable f(V)
    Returns A * V, where V is a dense matrix with dimensions (N, K).
dtype : dtype
    Data type of the matrix.
rmatmat : callable f(V)
    Returns A^H * V, where V is a dense matrix with dimensions (M, K).

Attributes
----------
args : tuple
    For linear operators describing products etc. of other linear
    operators, the operands of the binary operation.

See Also
--------
aslinearoperator : Construct LinearOperators

Notes
-----
The user-defined matvec() function must properly handle the case
where v has shape (N,) as well as the (N,1) case.  The shape of
the return type is handled internally by LinearOperator.

LinearOperator instances can also be multiplied, added with each
other and exponentiated, all lazily: the result of these operations
is always a new, composite LinearOperator, that defers linear
operations to the original operators and combines the results.

More details regarding how to subclass a LinearOperator and several
examples of concrete LinearOperator instances can be found in the
external project `PyLops <https://pylops.readthedocs.io>`_.


Examples
--------
>>> import numpy as np
>>> from scipy.sparse.linalg import LinearOperator
>>> def mv(v):
...     return np.array([2*v[0], 3*v[1]])
...
>>> A = LinearOperator((2,2), matvec=mv)
>>> A
<2x2 _CustomLinearOperator with dtype=float64>
>>> A.matvec(np.ones(2))
array([ 2.,  3.])
>>> A * np.ones(2)
array([ 2.,  3.])
*)

val adjoint : [> tag] Obj.t -> Py.Object.t
(**
Hermitian adjoint.

Returns the Hermitian adjoint of self, aka the Hermitian
conjugate or Hermitian transpose. For a complex matrix, the
Hermitian adjoint is equal to the conjugate transpose.

Can be abbreviated self.H instead of self.adjoint().

Returns
-------
A_H : LinearOperator
    Hermitian adjoint of self.
*)

val dot : x:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Matrix-matrix or matrix-vector multiplication.

Parameters
----------
x : array_like
    1-d or 2-d array, representing a vector or matrix.

Returns
-------
Ax : array
    1-d or 2-d array (depending on the shape of x) that represents
    the result of applying this linear operator on x.
*)

val matmat : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Matrix-matrix multiplication.

Performs the operation y=A*X where A is an MxN linear
operator and X dense N*K matrix or ndarray.

Parameters
----------
X : {matrix, ndarray}
    An array with shape (N,K).

Returns
-------
Y : {matrix, ndarray}
    A matrix or ndarray with shape (M,K) depending on
    the type of the X argument.

Notes
-----
This matmat wraps any user-specified matmat routine or overridden
_matmat method to ensure that y has the correct type.
*)

val matvec : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Matrix-vector multiplication.

Performs the operation y=A*x where A is an MxN linear
operator and x is a column vector or 1-d array.

Parameters
----------
x : {matrix, ndarray}
    An array with shape (N,) or (N,1).

Returns
-------
y : {matrix, ndarray}
    A matrix or ndarray with shape (M,) or (M,1) depending
    on the type and shape of the x argument.

Notes
-----
This matvec wraps the user-specified matvec routine or overridden
_matvec method to ensure that y has the correct shape and type.
*)

val rmatmat : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Adjoint matrix-matrix multiplication.

Performs the operation y = A^H * x where A is an MxN linear
operator and x is a column vector or 1-d array, or 2-d array.
The default implementation defers to the adjoint.

Parameters
----------
X : {matrix, ndarray}
    A matrix or 2D array.

Returns
-------
Y : {matrix, ndarray}
    A matrix or 2D array depending on the type of the input.

Notes
-----
This rmatmat wraps the user-specified rmatmat routine.
*)

val rmatvec : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Adjoint matrix-vector multiplication.

Performs the operation y = A^H * x where A is an MxN linear
operator and x is a column vector or 1-d array.

Parameters
----------
x : {matrix, ndarray}
    An array with shape (M,) or (M,1).

Returns
-------
y : {matrix, ndarray}
    A matrix or ndarray with shape (N,) or (N,1) depending
    on the type and shape of the x argument.

Notes
-----
This rmatvec wraps the user-specified rmatvec routine or overridden
_rmatvec method to ensure that y has the correct shape and type.
*)

val transpose : [> tag] Obj.t -> Py.Object.t
(**
Transpose this linear operator.

Returns a LinearOperator that represents the transpose of this one.
Can be abbreviated self.T instead of self.transpose().
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Matrix : sig
type tag = [`Matrix]
type t = [`ArrayLike | `Matrix | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?dtype:Np.Dtype.t -> ?copy:bool -> data:[`Ndarray of [>`Ndarray] Np.Obj.t | `S of string] -> unit -> t
(**
matrix(data, dtype=None, copy=True)

.. note:: It is no longer recommended to use this class, even for linear
          algebra. Instead use regular arrays. The class may be removed
          in the future.

Returns a matrix from an array-like object, or from a string of data.
A matrix is a specialized 2-D array that retains its 2-D nature
through operations.  It has certain special operators, such as ``*``
(matrix multiplication) and ``**`` (matrix power).

Parameters
----------
data : array_like or string
   If `data` is a string, it is interpreted as a matrix with commas
   or spaces separating columns, and semicolons separating rows.
dtype : data-type
   Data-type of the output matrix.
copy : bool
   If `data` is already an `ndarray`, then this flag determines
   whether the data is copied (the default), or whether a view is
   constructed.

See Also
--------
array

Examples
--------
>>> a = np.matrix('1 2; 3 4')
>>> a
matrix([[1, 2],
        [3, 4]])

>>> np.matrix([[1, 2], [3, 4]])
matrix([[1, 2],
        [3, 4]])
*)

val __getitem__ : index:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return self[key].
*)

val __iter__ : [> tag] Obj.t -> Py.Object.t
(**
Implement iter(self).
*)

val __setitem__ : key:Py.Object.t -> value:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Set self[key] to value.
*)

val all : ?axis:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Test whether all matrix elements along a given axis evaluate to True.

Parameters
----------
See `numpy.all` for complete descriptions

See Also
--------
numpy.all

Notes
-----
This is the same as `ndarray.all`, but it returns a `matrix` object.

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3,4))); x
matrix([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> y = x[0]; y
matrix([[0, 1, 2, 3]])
>>> (x == y)
matrix([[ True,  True,  True,  True],
        [False, False, False, False],
        [False, False, False, False]])
>>> (x == y).all()
False
>>> (x == y).all(0)
matrix([[False, False, False, False]])
>>> (x == y).all(1)
matrix([[ True],
        [False],
        [False]])
*)

val any : ?axis:int -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Test whether any array element along a given axis evaluates to True.

Refer to `numpy.any` for full documentation.

Parameters
----------
axis : int, optional
    Axis along which logical OR is performed
out : ndarray, optional
    Output to existing array instead of creating new one, must have
    same shape as expected output

Returns
-------
    any : bool, ndarray
        Returns a single bool if `axis` is ``None``; otherwise,
        returns `ndarray`
*)

val argmax : ?axis:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Indexes of the maximum values along an axis.

Return the indexes of the first occurrences of the maximum values
along the specified axis.  If axis is None, the index is for the
flattened matrix.

Parameters
----------
See `numpy.argmax` for complete descriptions

See Also
--------
numpy.argmax

Notes
-----
This is the same as `ndarray.argmax`, but returns a `matrix` object
where `ndarray.argmax` would return an `ndarray`.

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3,4))); x
matrix([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> x.argmax()
11
>>> x.argmax(0)
matrix([[2, 2, 2, 2]])
>>> x.argmax(1)
matrix([[3],
        [3],
        [3]])
*)

val argmin : ?axis:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Indexes of the minimum values along an axis.

Return the indexes of the first occurrences of the minimum values
along the specified axis.  If axis is None, the index is for the
flattened matrix.

Parameters
----------
See `numpy.argmin` for complete descriptions.

See Also
--------
numpy.argmin

Notes
-----
This is the same as `ndarray.argmin`, but returns a `matrix` object
where `ndarray.argmin` would return an `ndarray`.

Examples
--------
>>> x = -np.matrix(np.arange(12).reshape((3,4))); x
matrix([[  0,  -1,  -2,  -3],
        [ -4,  -5,  -6,  -7],
        [ -8,  -9, -10, -11]])
>>> x.argmin()
11
>>> x.argmin(0)
matrix([[2, 2, 2, 2]])
>>> x.argmin(1)
matrix([[3],
        [3],
        [3]])
*)

val argpartition : ?axis:Py.Object.t -> ?kind:Py.Object.t -> ?order:Py.Object.t -> kth:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.argpartition(kth, axis=-1, kind='introselect', order=None)

Returns the indices that would partition this array.

Refer to `numpy.argpartition` for full documentation.

.. versionadded:: 1.8.0

See Also
--------
numpy.argpartition : equivalent function
*)

val argsort : ?axis:Py.Object.t -> ?kind:Py.Object.t -> ?order:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.argsort(axis=-1, kind=None, order=None)

Returns the indices that would sort this array.

Refer to `numpy.argsort` for full documentation.

See Also
--------
numpy.argsort : equivalent function
*)

val astype : ?order:[`C | `F | `A | `K] -> ?casting:[`No | `Equiv | `Safe | `Same_kind | `Unsafe] -> ?subok:Py.Object.t -> ?copy:bool -> dtype:[`S of string | `Dtype of Np.Dtype.t] -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
a.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)

Copy of the array, cast to a specified type.

Parameters
----------
dtype : str or dtype
    Typecode or data-type to which the array is cast.
order : {'C', 'F', 'A', 'K'}, optional
    Controls the memory layout order of the result.
    'C' means C order, 'F' means Fortran order, 'A'
    means 'F' order if all the arrays are Fortran contiguous,
    'C' order otherwise, and 'K' means as close to the
    order the array elements appear in memory as possible.
    Default is 'K'.
casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
    Controls what kind of data casting may occur. Defaults to 'unsafe'
    for backwards compatibility.

      * 'no' means the data types should not be cast at all.
      * 'equiv' means only byte-order changes are allowed.
      * 'safe' means only casts which can preserve values are allowed.
      * 'same_kind' means only safe casts or casts within a kind,
        like float64 to float32, are allowed.
      * 'unsafe' means any data conversions may be done.
subok : bool, optional
    If True, then sub-classes will be passed-through (default), otherwise
    the returned array will be forced to be a base-class array.
copy : bool, optional
    By default, astype always returns a newly allocated array. If this
    is set to false, and the `dtype`, `order`, and `subok`
    requirements are satisfied, the input array is returned instead
    of a copy.

Returns
-------
arr_t : ndarray
    Unless `copy` is False and the other conditions for returning the input
    array are satisfied (see description for `copy` input parameter), `arr_t`
    is a new array of the same shape as the input array, with dtype, order
    given by `dtype`, `order`.

Notes
-----
.. versionchanged:: 1.17.0
   Casting between a simple data type and a structured one is possible only
   for 'unsafe' casting.  Casting to multiple fields is allowed, but
   casting from multiple fields is not.

.. versionchanged:: 1.9.0
   Casting from numeric to string types in 'safe' casting mode requires
   that the string dtype length is long enough to store the max
   integer/float value converted.

Raises
------
ComplexWarning
    When casting from complex to float or int. To avoid this,
    one should use ``a.real.astype(t)``.

Examples
--------
>>> x = np.array([1, 2, 2.5])
>>> x
array([1. ,  2. ,  2.5])

>>> x.astype(int)
array([1, 2, 2])
*)

val byteswap : ?inplace:bool -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
a.byteswap(inplace=False)

Swap the bytes of the array elements

Toggle between low-endian and big-endian data representation by
returning a byteswapped array, optionally swapped in-place.
Arrays of byte-strings are not swapped. The real and imaginary
parts of a complex number are swapped individually.

Parameters
----------
inplace : bool, optional
    If ``True``, swap bytes in-place, default is ``False``.

Returns
-------
out : ndarray
    The byteswapped array. If `inplace` is ``True``, this is
    a view to self.

Examples
--------
>>> A = np.array([1, 256, 8755], dtype=np.int16)
>>> list(map(hex, A))
['0x1', '0x100', '0x2233']
>>> A.byteswap(inplace=True)
array([  256,     1, 13090], dtype=int16)
>>> list(map(hex, A))
['0x100', '0x1', '0x3322']

Arrays of byte-strings are not swapped

>>> A = np.array([b'ceg', b'fac'])
>>> A.byteswap()
array([b'ceg', b'fac'], dtype='|S3')

``A.newbyteorder().byteswap()`` produces an array with the same values
  but different representation in memory

>>> A = np.array([1, 2, 3])
>>> A.view(np.uint8)
array([1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0,
       0, 0], dtype=uint8)
>>> A.newbyteorder().byteswap(inplace=True)
array([1, 2, 3])
>>> A.view(np.uint8)
array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0,
       0, 3], dtype=uint8)
*)

val choose : ?out:Py.Object.t -> ?mode:Py.Object.t -> choices:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.choose(choices, out=None, mode='raise')

Use an index array to construct a new array from a set of choices.

Refer to `numpy.choose` for full documentation.

See Also
--------
numpy.choose : equivalent function
*)

val clip : ?min:Py.Object.t -> ?max:Py.Object.t -> ?out:Py.Object.t -> ?kwargs:(string * Py.Object.t) list -> [> tag] Obj.t -> Py.Object.t
(**
a.clip(min=None, max=None, out=None, **kwargs)

Return an array whose values are limited to ``[min, max]``.
One of max or min must be given.

Refer to `numpy.clip` for full documentation.

See Also
--------
numpy.clip : equivalent function
*)

val compress : ?axis:Py.Object.t -> ?out:Py.Object.t -> condition:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.compress(condition, axis=None, out=None)

Return selected slices of this array along given axis.

Refer to `numpy.compress` for full documentation.

See Also
--------
numpy.compress : equivalent function
*)

val conj : [> tag] Obj.t -> Py.Object.t
(**
a.conj()

Complex-conjugate all elements.

Refer to `numpy.conjugate` for full documentation.

See Also
--------
numpy.conjugate : equivalent function
*)

val conjugate : [> tag] Obj.t -> Py.Object.t
(**
a.conjugate()

Return the complex conjugate, element-wise.

Refer to `numpy.conjugate` for full documentation.

See Also
--------
numpy.conjugate : equivalent function
*)

val copy : ?order:[`C | `F | `A | `K] -> [> tag] Obj.t -> Py.Object.t
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

val cumprod : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.cumprod(axis=None, dtype=None, out=None)

Return the cumulative product of the elements along the given axis.

Refer to `numpy.cumprod` for full documentation.

See Also
--------
numpy.cumprod : equivalent function
*)

val cumsum : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.cumsum(axis=None, dtype=None, out=None)

Return the cumulative sum of the elements along the given axis.

Refer to `numpy.cumsum` for full documentation.

See Also
--------
numpy.cumsum : equivalent function
*)

val diagonal : ?offset:Py.Object.t -> ?axis1:Py.Object.t -> ?axis2:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
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

val dot : ?out:Py.Object.t -> b:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.dot(b, out=None)

Dot product of two arrays.

Refer to `numpy.dot` for full documentation.

See Also
--------
numpy.dot : equivalent function

Examples
--------
>>> a = np.eye(2)
>>> b = np.ones((2, 2)) * 2
>>> a.dot(b)
array([[2.,  2.],
       [2.,  2.]])

This array method can be conveniently chained:

>>> a.dot(b).dot(b)
array([[8.,  8.],
       [8.,  8.]])
*)

val dump : file:[`S of string | `Path of Py.Object.t] -> [> tag] Obj.t -> Py.Object.t
(**
a.dump(file)

Dump a pickle of the array to the specified file.
The array can be read back with pickle.load or numpy.load.

Parameters
----------
file : str or Path
    A string naming the dump file.

    .. versionchanged:: 1.17.0
        `pathlib.Path` objects are now accepted.
*)

val dumps : [> tag] Obj.t -> Py.Object.t
(**
a.dumps()

Returns the pickle of the array as a string.
pickle.loads or numpy.loads will convert the string back to an array.

Parameters
----------
None
*)

val fill : value:[`F of float | `I of int | `Bool of bool | `S of string] -> [> tag] Obj.t -> Py.Object.t
(**
a.fill(value)

Fill the array with a scalar value.

Parameters
----------
value : scalar
    All elements of `a` will be assigned this value.

Examples
--------
>>> a = np.array([1, 2])
>>> a.fill(0)
>>> a
array([0, 0])
>>> a = np.empty(2)
>>> a.fill(1)
>>> a
array([1.,  1.])
*)

val flatten : ?order:[`C | `F | `A | `K] -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return a flattened copy of the matrix.

All `N` elements of the matrix are placed into a single row.

Parameters
----------
order : {'C', 'F', 'A', 'K'}, optional
    'C' means to flatten in row-major (C-style) order. 'F' means to
    flatten in column-major (Fortran-style) order. 'A' means to
    flatten in column-major order if `m` is Fortran *contiguous* in
    memory, row-major order otherwise. 'K' means to flatten `m` in
    the order the elements occur in memory. The default is 'C'.

Returns
-------
y : matrix
    A copy of the matrix, flattened to a `(1, N)` matrix where `N`
    is the number of elements in the original matrix.

See Also
--------
ravel : Return a flattened array.
flat : A 1-D flat iterator over the matrix.

Examples
--------
>>> m = np.matrix([[1,2], [3,4]])
>>> m.flatten()
matrix([[1, 2, 3, 4]])
>>> m.flatten('F')
matrix([[1, 3, 2, 4]])
*)

val getA : [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return `self` as an `ndarray` object.

Equivalent to ``np.asarray(self)``.

Parameters
----------
None

Returns
-------
ret : ndarray
    `self` as an `ndarray`

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3,4))); x
matrix([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> x.getA()
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
*)

val getA1 : [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return `self` as a flattened `ndarray`.

Equivalent to ``np.asarray(x).ravel()``

Parameters
----------
None

Returns
-------
ret : ndarray
    `self`, 1-D, as an `ndarray`

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3,4))); x
matrix([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> x.getA1()
array([ 0,  1,  2, ...,  9, 10, 11])
*)

val getH : [> tag] Obj.t -> Py.Object.t
(**
Returns the (complex) conjugate transpose of `self`.

Equivalent to ``np.transpose(self)`` if `self` is real-valued.

Parameters
----------
None

Returns
-------
ret : matrix object
    complex conjugate transpose of `self`

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3,4)))
>>> z = x - 1j*x; z
matrix([[  0. +0.j,   1. -1.j,   2. -2.j,   3. -3.j],
        [  4. -4.j,   5. -5.j,   6. -6.j,   7. -7.j],
        [  8. -8.j,   9. -9.j,  10.-10.j,  11.-11.j]])
>>> z.getH()
matrix([[ 0. -0.j,  4. +4.j,  8. +8.j],
        [ 1. +1.j,  5. +5.j,  9. +9.j],
        [ 2. +2.j,  6. +6.j, 10.+10.j],
        [ 3. +3.j,  7. +7.j, 11.+11.j]])
*)

val getI : [> tag] Obj.t -> Py.Object.t
(**
Returns the (multiplicative) inverse of invertible `self`.

Parameters
----------
None

Returns
-------
ret : matrix object
    If `self` is non-singular, `ret` is such that ``ret * self`` ==
    ``self * ret`` == ``np.matrix(np.eye(self[0,:].size)`` all return
    ``True``.

Raises
------
numpy.linalg.LinAlgError: Singular matrix
    If `self` is singular.

See Also
--------
linalg.inv

Examples
--------
>>> m = np.matrix('[1, 2; 3, 4]'); m
matrix([[1, 2],
        [3, 4]])
>>> m.getI()
matrix([[-2. ,  1. ],
        [ 1.5, -0.5]])
>>> m.getI() * m
matrix([[ 1.,  0.], # may vary
        [ 0.,  1.]])
*)

val getT : [> tag] Obj.t -> Py.Object.t
(**
Returns the transpose of the matrix.

Does *not* conjugate!  For the complex conjugate transpose, use ``.H``.

Parameters
----------
None

Returns
-------
ret : matrix object
    The (non-conjugated) transpose of the matrix.

See Also
--------
transpose, getH

Examples
--------
>>> m = np.matrix('[1, 2; 3, 4]')
>>> m
matrix([[1, 2],
        [3, 4]])
>>> m.getT()
matrix([[1, 3],
        [2, 4]])
*)

val getfield : ?offset:int -> dtype:[`S of string | `Dtype of Np.Dtype.t] -> [> tag] Obj.t -> Py.Object.t
(**
a.getfield(dtype, offset=0)

Returns a field of the given array as a certain type.

A field is a view of the array data with a given data-type. The values in
the view are determined by the given type and the offset into the current
array in bytes. The offset needs to be such that the view dtype fits in the
array dtype; for example an array of dtype complex128 has 16-byte elements.
If taking a view with a 32-bit integer (4 bytes), the offset needs to be
between 0 and 12 bytes.

Parameters
----------
dtype : str or dtype
    The data type of the view. The dtype size of the view can not be larger
    than that of the array itself.
offset : int
    Number of bytes to skip before beginning the element view.

Examples
--------
>>> x = np.diag([1.+1.j]*2)
>>> x[1, 1] = 2 + 4.j
>>> x
array([[1.+1.j,  0.+0.j],
       [0.+0.j,  2.+4.j]])
>>> x.getfield(np.float64)
array([[1.,  0.],
       [0.,  2.]])

By choosing an offset of 8 bytes we can select the complex part of the
array for our view:

>>> x.getfield(np.float64, offset=8)
array([[1.,  0.],
       [0.,  4.]])
*)

val item : Py.Object.t list -> [> tag] Obj.t -> Py.Object.t
(**
a.item( *args)

Copy an element of an array to a standard Python scalar and return it.

Parameters
----------
\*args : Arguments (variable number and type)

    * none: in this case, the method only works for arrays
      with one element (`a.size == 1`), which element is
      copied into a standard Python scalar object and returned.

    * int_type: this argument is interpreted as a flat index into
      the array, specifying which element to copy and return.

    * tuple of int_types: functions as does a single int_type argument,
      except that the argument is interpreted as an nd-index into the
      array.

Returns
-------
z : Standard Python scalar object
    A copy of the specified element of the array as a suitable
    Python scalar

Notes
-----
When the data type of `a` is longdouble or clongdouble, item() returns
a scalar array object because there is no available Python scalar that
would not lose information. Void arrays return a buffer object for item(),
unless fields are defined, in which case a tuple is returned.

`item` is very similar to a[args], except, instead of an array scalar,
a standard Python scalar is returned. This can be useful for speeding up
access to elements of the array and doing arithmetic on elements of the
array using Python's optimized math.

Examples
--------
>>> np.random.seed(123)
>>> x = np.random.randint(9, size=(3, 3))
>>> x
array([[2, 2, 6],
       [1, 3, 6],
       [1, 0, 1]])
>>> x.item(3)
1
>>> x.item(7)
0
>>> x.item((0, 1))
2
>>> x.item((2, 2))
1
*)

val itemset : Py.Object.t list -> [> tag] Obj.t -> Py.Object.t
(**
a.itemset( *args)

Insert scalar into an array (scalar is cast to array's dtype, if possible)

There must be at least 1 argument, and define the last argument
as *item*.  Then, ``a.itemset( *args)`` is equivalent to but faster
than ``a[args] = item``.  The item should be a scalar value and `args`
must select a single item in the array `a`.

Parameters
----------
\*args : Arguments
    If one argument: a scalar, only used in case `a` is of size 1.
    If two arguments: the last argument is the value to be set
    and must be a scalar, the first argument specifies a single array
    element location. It is either an int or a tuple.

Notes
-----
Compared to indexing syntax, `itemset` provides some speed increase
for placing a scalar into a particular location in an `ndarray`,
if you must do this.  However, generally this is discouraged:
among other problems, it complicates the appearance of the code.
Also, when using `itemset` (and `item`) inside a loop, be sure
to assign the methods to a local variable to avoid the attribute
look-up at each loop iteration.

Examples
--------
>>> np.random.seed(123)
>>> x = np.random.randint(9, size=(3, 3))
>>> x
array([[2, 2, 6],
       [1, 3, 6],
       [1, 0, 1]])
>>> x.itemset(4, 0)
>>> x.itemset((2, 2), 9)
>>> x
array([[2, 2, 6],
       [1, 0, 6],
       [1, 0, 9]])
*)

val max : ?axis:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return the maximum value along an axis.

Parameters
----------
See `amax` for complete descriptions

See Also
--------
amax, ndarray.max

Notes
-----
This is the same as `ndarray.max`, but returns a `matrix` object
where `ndarray.max` would return an ndarray.

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3,4))); x
matrix([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> x.max()
11
>>> x.max(0)
matrix([[ 8,  9, 10, 11]])
>>> x.max(1)
matrix([[ 3],
        [ 7],
        [11]])
*)

val mean : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns the average of the matrix elements along the given axis.

Refer to `numpy.mean` for full documentation.

See Also
--------
numpy.mean

Notes
-----
Same as `ndarray.mean` except that, where that returns an `ndarray`,
this returns a `matrix` object.

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3, 4)))
>>> x
matrix([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> x.mean()
5.5
>>> x.mean(0)
matrix([[4., 5., 6., 7.]])
>>> x.mean(1)
matrix([[ 1.5],
        [ 5.5],
        [ 9.5]])
*)

val min : ?axis:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return the minimum value along an axis.

Parameters
----------
See `amin` for complete descriptions.

See Also
--------
amin, ndarray.min

Notes
-----
This is the same as `ndarray.min`, but returns a `matrix` object
where `ndarray.min` would return an ndarray.

Examples
--------
>>> x = -np.matrix(np.arange(12).reshape((3,4))); x
matrix([[  0,  -1,  -2,  -3],
        [ -4,  -5,  -6,  -7],
        [ -8,  -9, -10, -11]])
>>> x.min()
-11
>>> x.min(0)
matrix([[ -8,  -9, -10, -11]])
>>> x.min(1)
matrix([[ -3],
        [ -7],
        [-11]])
*)

val newbyteorder : ?new_order:string -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
arr.newbyteorder(new_order='S')

Return the array with the same data viewed with a different byte order.

Equivalent to::

    arr.view(arr.dtype.newbytorder(new_order))

Changes are also made in all fields and sub-arrays of the array data
type.



Parameters
----------
new_order : string, optional
    Byte order to force; a value from the byte order specifications
    below. `new_order` codes can be any of:

    * 'S' - swap dtype from current to opposite endian
    * {'<', 'L'} - little endian
    * {'>', 'B'} - big endian
    * {'=', 'N'} - native order
    * {'|', 'I'} - ignore (no change to byte order)

    The default value ('S') results in swapping the current
    byte order. The code does a case-insensitive check on the first
    letter of `new_order` for the alternatives above.  For example,
    any of 'B' or 'b' or 'biggish' are valid to specify big-endian.


Returns
-------
new_arr : array
    New array object with the dtype reflecting given change to the
    byte order.
*)

val nonzero : [> tag] Obj.t -> Py.Object.t
(**
a.nonzero()

Return the indices of the elements that are non-zero.

Refer to `numpy.nonzero` for full documentation.

See Also
--------
numpy.nonzero : equivalent function
*)

val partition : ?axis:int -> ?kind:[`Introselect] -> ?order:[`StringList of string list | `S of string] -> kth:[`I of int | `Is of int list] -> [> tag] Obj.t -> Py.Object.t
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

val prod : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return the product of the array elements over the given axis.

Refer to `prod` for full documentation.

See Also
--------
prod, ndarray.prod

Notes
-----
Same as `ndarray.prod`, except, where that returns an `ndarray`, this
returns a `matrix` object instead.

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3,4))); x
matrix([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> x.prod()
0
>>> x.prod(0)
matrix([[  0,  45, 120, 231]])
>>> x.prod(1)
matrix([[   0],
        [ 840],
        [7920]])
*)

val ptp : ?axis:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Peak-to-peak (maximum - minimum) value along the given axis.

Refer to `numpy.ptp` for full documentation.

See Also
--------
numpy.ptp

Notes
-----
Same as `ndarray.ptp`, except, where that would return an `ndarray` object,
this returns a `matrix` object.

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3,4))); x
matrix([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> x.ptp()
11
>>> x.ptp(0)
matrix([[8, 8, 8, 8]])
>>> x.ptp(1)
matrix([[3],
        [3],
        [3]])
*)

val put : ?mode:Py.Object.t -> indices:Py.Object.t -> values:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.put(indices, values, mode='raise')

Set ``a.flat[n] = values[n]`` for all `n` in indices.

Refer to `numpy.put` for full documentation.

See Also
--------
numpy.put : equivalent function
*)

val ravel : ?order:[`C | `F | `A | `K] -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return a flattened matrix.

Refer to `numpy.ravel` for more documentation.

Parameters
----------
order : {'C', 'F', 'A', 'K'}, optional
    The elements of `m` are read using this index order. 'C' means to
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
ret : matrix
    Return the matrix flattened to shape `(1, N)` where `N`
    is the number of elements in the original matrix.
    A copy is made only if necessary.

See Also
--------
matrix.flatten : returns a similar output matrix but always a copy
matrix.flat : a flat iterator on the array.
numpy.ravel : related function which returns an ndarray
*)

val repeat : ?axis:Py.Object.t -> repeats:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.repeat(repeats, axis=None)

Repeat elements of an array.

Refer to `numpy.repeat` for full documentation.

See Also
--------
numpy.repeat : equivalent function
*)

val reshape : ?order:Py.Object.t -> shape:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.reshape(shape, order='C')

Returns an array containing the same data with a new shape.

Refer to `numpy.reshape` for full documentation.

See Also
--------
numpy.reshape : equivalent function

Notes
-----
Unlike the free function `numpy.reshape`, this method on `ndarray` allows
the elements of the shape parameter to be passed in as separate arguments.
For example, ``a.reshape(10, 11)`` is equivalent to
``a.reshape((10, 11))``.
*)

val resize : ?refcheck:bool -> new_shape:[`TupleOfInts of int list | `T_n_ints of Py.Object.t] -> [> tag] Obj.t -> Py.Object.t
(**
a.resize(new_shape, refcheck=True)

Change shape and size of array in-place.

Parameters
----------
new_shape : tuple of ints, or `n` ints
    Shape of resized array.
refcheck : bool, optional
    If False, reference count will not be checked. Default is True.

Returns
-------
None

Raises
------
ValueError
    If `a` does not own its own data or references or views to it exist,
    and the data memory must be changed.
    PyPy only: will always raise if the data memory must be changed, since
    there is no reliable way to determine if references or views to it
    exist.

SystemError
    If the `order` keyword argument is specified. This behaviour is a
    bug in NumPy.

See Also
--------
resize : Return a new array with the specified shape.

Notes
-----
This reallocates space for the data area if necessary.

Only contiguous arrays (data elements consecutive in memory) can be
resized.

The purpose of the reference count check is to make sure you
do not use this array as a buffer for another Python object and then
reallocate the memory. However, reference counts can increase in
other ways so if you are sure that you have not shared the memory
for this array with another Python object, then you may safely set
`refcheck` to False.

Examples
--------
Shrinking an array: array is flattened (in the order that the data are
stored in memory), resized, and reshaped:

>>> a = np.array([[0, 1], [2, 3]], order='C')
>>> a.resize((2, 1))
>>> a
array([[0],
       [1]])

>>> a = np.array([[0, 1], [2, 3]], order='F')
>>> a.resize((2, 1))
>>> a
array([[0],
       [2]])

Enlarging an array: as above, but missing entries are filled with zeros:

>>> b = np.array([[0, 1], [2, 3]])
>>> b.resize(2, 3) # new_shape parameter doesn't have to be a tuple
>>> b
array([[0, 1, 2],
       [3, 0, 0]])

Referencing an array prevents resizing...

>>> c = a
>>> a.resize((1, 1))
Traceback (most recent call last):
...
ValueError: cannot resize an array that references or is referenced ...

Unless `refcheck` is False:

>>> a.resize((1, 1), refcheck=False)
>>> a
array([[0]])
>>> c
array([[0]])
*)

val round : ?decimals:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.round(decimals=0, out=None)

Return `a` with each element rounded to the given number of decimals.

Refer to `numpy.around` for full documentation.

See Also
--------
numpy.around : equivalent function
*)

val searchsorted : ?side:Py.Object.t -> ?sorter:Py.Object.t -> v:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.searchsorted(v, side='left', sorter=None)

Find indices where elements of v should be inserted in a to maintain order.

For full documentation, see `numpy.searchsorted`

See Also
--------
numpy.searchsorted : equivalent function
*)

val setfield : ?offset:int -> val_:Py.Object.t -> dtype:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.setfield(val, dtype, offset=0)

Put a value into a specified place in a field defined by a data-type.

Place `val` into `a`'s field defined by `dtype` and beginning `offset`
bytes into the field.

Parameters
----------
val : object
    Value to be placed in field.
dtype : dtype object
    Data-type of the field in which to place `val`.
offset : int, optional
    The number of bytes into the field at which to place `val`.

Returns
-------
None

See Also
--------
getfield

Examples
--------
>>> x = np.eye(3)
>>> x.getfield(np.float64)
array([[1.,  0.,  0.],
       [0.,  1.,  0.],
       [0.,  0.,  1.]])
>>> x.setfield(3, np.int32)
>>> x.getfield(np.int32)
array([[3, 3, 3],
       [3, 3, 3],
       [3, 3, 3]], dtype=int32)
>>> x
array([[1.0e+000, 1.5e-323, 1.5e-323],
       [1.5e-323, 1.0e+000, 1.5e-323],
       [1.5e-323, 1.5e-323, 1.0e+000]])
>>> x.setfield(np.eye(3), np.int32)
>>> x
array([[1.,  0.,  0.],
       [0.,  1.,  0.],
       [0.,  0.,  1.]])
*)

val setflags : ?write:bool -> ?align:bool -> ?uic:bool -> [> tag] Obj.t -> Py.Object.t
(**
a.setflags(write=None, align=None, uic=None)

Set array flags WRITEABLE, ALIGNED, (WRITEBACKIFCOPY and UPDATEIFCOPY),
respectively.

These Boolean-valued flags affect how numpy interprets the memory
area used by `a` (see Notes below). The ALIGNED flag can only
be set to True if the data is actually aligned according to the type.
The WRITEBACKIFCOPY and (deprecated) UPDATEIFCOPY flags can never be set
to True. The flag WRITEABLE can only be set to True if the array owns its
own memory, or the ultimate owner of the memory exposes a writeable buffer
interface, or is a string. (The exception for string is made so that
unpickling can be done without copying memory.)

Parameters
----------
write : bool, optional
    Describes whether or not `a` can be written to.
align : bool, optional
    Describes whether or not `a` is aligned properly for its type.
uic : bool, optional
    Describes whether or not `a` is a copy of another 'base' array.

Notes
-----
Array flags provide information about how the memory area used
for the array is to be interpreted. There are 7 Boolean flags
in use, only four of which can be changed by the user:
WRITEBACKIFCOPY, UPDATEIFCOPY, WRITEABLE, and ALIGNED.

WRITEABLE (W) the data area can be written to;

ALIGNED (A) the data and strides are aligned appropriately for the hardware
(as determined by the compiler);

UPDATEIFCOPY (U) (deprecated), replaced by WRITEBACKIFCOPY;

WRITEBACKIFCOPY (X) this array is a copy of some other array (referenced
by .base). When the C-API function PyArray_ResolveWritebackIfCopy is
called, the base array will be updated with the contents of this array.

All flags can be accessed using the single (upper case) letter as well
as the full name.

Examples
--------
>>> y = np.array([[3, 1, 7],
...               [2, 0, 0],
...               [8, 5, 9]])
>>> y
array([[3, 1, 7],
       [2, 0, 0],
       [8, 5, 9]])
>>> y.flags
  C_CONTIGUOUS : True
  F_CONTIGUOUS : False
  OWNDATA : True
  WRITEABLE : True
  ALIGNED : True
  WRITEBACKIFCOPY : False
  UPDATEIFCOPY : False
>>> y.setflags(write=0, align=0)
>>> y.flags
  C_CONTIGUOUS : True
  F_CONTIGUOUS : False
  OWNDATA : True
  WRITEABLE : False
  ALIGNED : False
  WRITEBACKIFCOPY : False
  UPDATEIFCOPY : False
>>> y.setflags(uic=1)
Traceback (most recent call last):
  File '<stdin>', line 1, in <module>
ValueError: cannot set WRITEBACKIFCOPY flag to True
*)

val sort : ?axis:int -> ?kind:[`Stable | `Quicksort | `Heapsort | `Mergesort] -> ?order:[`StringList of string list | `S of string] -> [> tag] Obj.t -> Py.Object.t
(**
a.sort(axis=-1, kind=None, order=None)

Sort an array in-place. Refer to `numpy.sort` for full documentation.

Parameters
----------
axis : int, optional
    Axis along which to sort. Default is -1, which means sort along the
    last axis.
kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
    Sorting algorithm. The default is 'quicksort'. Note that both 'stable'
    and 'mergesort' use timsort under the covers and, in general, the
    actual implementation will vary with datatype. The 'mergesort' option
    is retained for backwards compatibility.

    .. versionchanged:: 1.15.0.
       The 'stable' option was added.

order : str or list of str, optional
    When `a` is an array with fields defined, this argument specifies
    which fields to compare first, second, etc.  A single field can
    be specified as a string, and not all fields need be specified,
    but unspecified fields will still be used, in the order in which
    they come up in the dtype, to break ties.

See Also
--------
numpy.sort : Return a sorted copy of an array.
numpy.argsort : Indirect sort.
numpy.lexsort : Indirect stable sort on multiple keys.
numpy.searchsorted : Find elements in sorted array.
numpy.partition: Partial sort.

Notes
-----
See `numpy.sort` for notes on the different sorting algorithms.

Examples
--------
>>> a = np.array([[1,4], [3,1]])
>>> a.sort(axis=1)
>>> a
array([[1, 4],
       [1, 3]])
>>> a.sort(axis=0)
>>> a
array([[1, 3],
       [1, 4]])

Use the `order` keyword to specify a field to use when sorting a
structured array:

>>> a = np.array([('a', 2), ('c', 1)], dtype=[('x', 'S1'), ('y', int)])
>>> a.sort(order='y')
>>> a
array([(b'c', 1), (b'a', 2)],
      dtype=[('x', 'S1'), ('y', '<i8')])
*)

val squeeze : ?axis:int list -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return a possibly reshaped matrix.

Refer to `numpy.squeeze` for more documentation.

Parameters
----------
axis : None or int or tuple of ints, optional
    Selects a subset of the single-dimensional entries in the shape.
    If an axis is selected with shape entry greater than one,
    an error is raised.

Returns
-------
squeezed : matrix
    The matrix, but as a (1, N) matrix if it had shape (N, 1).

See Also
--------
numpy.squeeze : related function

Notes
-----
If `m` has a single column then that column is returned
as the single row of a matrix.  Otherwise `m` is returned.
The returned matrix is always either `m` itself or a view into `m`.
Supplying an axis keyword argument will not affect the returned matrix
but it may cause an error to be raised.

Examples
--------
>>> c = np.matrix([[1], [2]])
>>> c
matrix([[1],
        [2]])
>>> c.squeeze()
matrix([[1, 2]])
>>> r = c.T
>>> r
matrix([[1, 2]])
>>> r.squeeze()
matrix([[1, 2]])
>>> m = np.matrix([[1, 2], [3, 4]])
>>> m.squeeze()
matrix([[1, 2],
        [3, 4]])
*)

val std : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> ?ddof:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return the standard deviation of the array elements along the given axis.

Refer to `numpy.std` for full documentation.

See Also
--------
numpy.std

Notes
-----
This is the same as `ndarray.std`, except that where an `ndarray` would
be returned, a `matrix` object is returned instead.

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3, 4)))
>>> x
matrix([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> x.std()
3.4520525295346629 # may vary
>>> x.std(0)
matrix([[ 3.26598632,  3.26598632,  3.26598632,  3.26598632]]) # may vary
>>> x.std(1)
matrix([[ 1.11803399],
        [ 1.11803399],
        [ 1.11803399]])
*)

val sum : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns the sum of the matrix elements, along the given axis.

Refer to `numpy.sum` for full documentation.

See Also
--------
numpy.sum

Notes
-----
This is the same as `ndarray.sum`, except that where an `ndarray` would
be returned, a `matrix` object is returned instead.

Examples
--------
>>> x = np.matrix([[1, 2], [4, 3]])
>>> x.sum()
10
>>> x.sum(axis=1)
matrix([[3],
        [7]])
>>> x.sum(axis=1, dtype='float')
matrix([[3.],
        [7.]])
>>> out = np.zeros((2, 1), dtype='float')
>>> x.sum(axis=1, dtype='float', out=np.asmatrix(out))
matrix([[3.],
        [7.]])
*)

val swapaxes : axis1:Py.Object.t -> axis2:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.swapaxes(axis1, axis2)

Return a view of the array with `axis1` and `axis2` interchanged.

Refer to `numpy.swapaxes` for full documentation.

See Also
--------
numpy.swapaxes : equivalent function
*)

val take : ?axis:Py.Object.t -> ?out:Py.Object.t -> ?mode:Py.Object.t -> indices:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.take(indices, axis=None, out=None, mode='raise')

Return an array formed from the elements of `a` at the given indices.

Refer to `numpy.take` for full documentation.

See Also
--------
numpy.take : equivalent function
*)

val tobytes : ?order:[`F | `C | `None] -> [> tag] Obj.t -> Py.Object.t
(**
a.tobytes(order='C')

Construct Python bytes containing the raw data bytes in the array.

Constructs Python bytes showing a copy of the raw contents of
data memory. The bytes object can be produced in either 'C' or 'Fortran',
or 'Any' order (the default is 'C'-order). 'Any' order means C-order
unless the F_CONTIGUOUS flag in the array is set, in which case it
means 'Fortran' order.

.. versionadded:: 1.9.0

Parameters
----------
order : {'C', 'F', None}, optional
    Order of the data for multidimensional arrays:
    C, Fortran, or the same as for the original array.

Returns
-------
s : bytes
    Python bytes exhibiting a copy of `a`'s raw data.

Examples
--------
>>> x = np.array([[0, 1], [2, 3]], dtype='<u2')
>>> x.tobytes()
b'\x00\x00\x01\x00\x02\x00\x03\x00'
>>> x.tobytes('C') == x.tobytes()
True
>>> x.tobytes('F')
b'\x00\x00\x02\x00\x01\x00\x03\x00'
*)

val tofile : ?sep:string -> ?format:string -> fid:[`S of string | `PyObject of Py.Object.t] -> [> tag] Obj.t -> Py.Object.t
(**
a.tofile(fid, sep='', format='%s')

Write array to a file as text or binary (default).

Data is always written in 'C' order, independent of the order of `a`.
The data produced by this method can be recovered using the function
fromfile().

Parameters
----------
fid : file or str or Path
    An open file object, or a string containing a filename.

    .. versionchanged:: 1.17.0
        `pathlib.Path` objects are now accepted.

sep : str
    Separator between array items for text output.
    If '' (empty), a binary file is written, equivalent to
    ``file.write(a.tobytes())``.
format : str
    Format string for text file output.
    Each entry in the array is formatted to text by first converting
    it to the closest Python type, and then using 'format' % item.

Notes
-----
This is a convenience function for quick storage of array data.
Information on endianness and precision is lost, so this method is not a
good choice for files intended to archive data or transport data between
machines with different endianness. Some of these problems can be overcome
by outputting the data as text files, at the expense of speed and file
size.

When fid is a file object, array contents are directly written to the
file, bypassing the file object's ``write`` method. As a result, tofile
cannot be used with files objects supporting compression (e.g., GzipFile)
or file-like objects that do not support ``fileno()`` (e.g., BytesIO).
*)

val tolist : [> tag] Obj.t -> Py.Object.t
(**
Return the matrix as a (possibly nested) list.

See `ndarray.tolist` for full documentation.

See Also
--------
ndarray.tolist

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3,4))); x
matrix([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> x.tolist()
[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
*)

val tostring : ?order:[`F | `C | `None] -> [> tag] Obj.t -> Py.Object.t
(**
a.tostring(order='C')

Construct Python bytes containing the raw data bytes in the array.

Constructs Python bytes showing a copy of the raw contents of
data memory. The bytes object can be produced in either 'C' or 'Fortran',
or 'Any' order (the default is 'C'-order). 'Any' order means C-order
unless the F_CONTIGUOUS flag in the array is set, in which case it
means 'Fortran' order.

This function is a compatibility alias for tobytes. Despite its name it returns bytes not strings.

Parameters
----------
order : {'C', 'F', None}, optional
    Order of the data for multidimensional arrays:
    C, Fortran, or the same as for the original array.

Returns
-------
s : bytes
    Python bytes exhibiting a copy of `a`'s raw data.

Examples
--------
>>> x = np.array([[0, 1], [2, 3]], dtype='<u2')
>>> x.tobytes()
b'\x00\x00\x01\x00\x02\x00\x03\x00'
>>> x.tobytes('C') == x.tobytes()
True
>>> x.tobytes('F')
b'\x00\x00\x02\x00\x01\x00\x03\x00'
*)

val trace : ?offset:Py.Object.t -> ?axis1:Py.Object.t -> ?axis2:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.trace(offset=0, axis1=0, axis2=1, dtype=None, out=None)

Return the sum along diagonals of the array.

Refer to `numpy.trace` for full documentation.

See Also
--------
numpy.trace : equivalent function
*)

val transpose : Py.Object.t list -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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
   intended simply as a 'convenience' alternative to the tuple form)

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

val var : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> ?ddof:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns the variance of the matrix elements, along the given axis.

Refer to `numpy.var` for full documentation.

See Also
--------
numpy.var

Notes
-----
This is the same as `ndarray.var`, except that where an `ndarray` would
be returned, a `matrix` object is returned instead.

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3, 4)))
>>> x
matrix([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> x.var()
11.916666666666666
>>> x.var(0)
matrix([[ 10.66666667,  10.66666667,  10.66666667,  10.66666667]]) # may vary
>>> x.var(1)
matrix([[1.25],
        [1.25],
        [1.25]])
*)

val view : ?dtype:[`Ndarray_sub_class of Py.Object.t | `Dtype of Np.Dtype.t] -> ?type_:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.view(dtype=None, type=None)

New view of array with the same data.

Parameters
----------
dtype : data-type or ndarray sub-class, optional
    Data-type descriptor of the returned view, e.g., float32 or int16. The
    default, None, results in the view having the same data-type as `a`.
    This argument can also be specified as an ndarray sub-class, which
    then specifies the type of the returned object (this is equivalent to
    setting the ``type`` parameter).
type : Python type, optional
    Type of the returned view, e.g., ndarray or matrix.  Again, the
    default None results in type preservation.

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

For ``a.view(some_dtype)``, if ``some_dtype`` has a different number of
bytes per entry than the previous dtype (for example, converting a
regular array to a structured array), then the behavior of the view
cannot be predicted just from the superficial appearance of ``a`` (shown
by ``print(a)``). It also depends on exactly how ``a`` is stored in
memory. Therefore if ``a`` is C-ordered versus fortran-ordered, versus
defined as a slice or transpose, etc., the view may give different
results.


Examples
--------
>>> x = np.array([(1, 2)], dtype=[('a', np.int8), ('b', np.int8)])

Viewing array data using a different type and dtype:

>>> y = x.view(dtype=np.int16, type=np.matrix)
>>> y
matrix([[513]], dtype=int16)
>>> print(type(y))
<class 'numpy.matrix'>

Creating a view on a structured array so it can be used in calculations

>>> x = np.array([(1, 2),(3,4)], dtype=[('a', np.int8), ('b', np.int8)])
>>> xv = x.view(dtype=np.int8).reshape(-1,2)
>>> xv
array([[1, 2],
       [3, 4]], dtype=int8)
>>> xv.mean(0)
array([2.,  3.])

Making changes to the view changes the underlying array

>>> xv[0,1] = 20
>>> x
array([(1, 20), (3,  4)], dtype=[('a', 'i1'), ('b', 'i1')])

Using a view to convert an array to a recarray:

>>> z = x.view(np.recarray)
>>> z.a
array([1, 3], dtype=int8)

Views share data:

>>> x[0] = (9, 10)
>>> z[0]
(9, 10)

Views that change the dtype size (bytes per entry) should normally be
avoided on arrays defined by slices, transposes, fortran-ordering, etc.:

>>> x = np.array([[1,2,3],[4,5,6]], dtype=np.int16)
>>> y = x[:, 0:2]
>>> y
array([[1, 2],
       [4, 5]], dtype=int16)
>>> y.view(dtype=[('width', np.int16), ('length', np.int16)])
Traceback (most recent call last):
    ...
ValueError: To change to a dtype of a different size, the array must be C-contiguous
>>> z = y.copy()
>>> z.view(dtype=[('width', np.int16), ('length', np.int16)])
array([[(1, 2)],
       [(4, 5)]], dtype=[('width', '<i2'), ('length', '<i2')])
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val array : ?dtype:Np.Dtype.t -> ?copy:bool -> ?order:[`K | `A | `C | `F] -> ?subok:bool -> ?ndmin:int -> object_:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
array(object, dtype=None, copy=True, order='K', subok=False, ndmin=0)

Create an array.

Parameters
----------
object : array_like
    An array, any object exposing the array interface, an object whose
    __array__ method returns an array, or any (nested) sequence.
dtype : data-type, optional
    The desired data-type for the array.  If not given, then the type will
    be determined as the minimum type required to hold the objects in the
    sequence.
copy : bool, optional
    If true (default), then the object is copied.  Otherwise, a copy will
    only be made if __array__ returns a copy, if obj is a nested sequence,
    or if a copy is needed to satisfy any of the other requirements
    (`dtype`, `order`, etc.).
order : {'K', 'A', 'C', 'F'}, optional
    Specify the memory layout of the array. If object is not an array, the
    newly created array will be in C order (row major) unless 'F' is
    specified, in which case it will be in Fortran order (column major).
    If object is an array the following holds.

    ===== ========= ===================================================
    order  no copy                     copy=True
    ===== ========= ===================================================
    'K'   unchanged F & C order preserved, otherwise most similar order
    'A'   unchanged F order if input is F and not C, otherwise C order
    'C'   C order   C order
    'F'   F order   F order
    ===== ========= ===================================================

    When ``copy=False`` and a copy is made for other reasons, the result is
    the same as if ``copy=True``, with some exceptions for `A`, see the
    Notes section. The default order is 'K'.
subok : bool, optional
    If True, then sub-classes will be passed-through, otherwise
    the returned array will be forced to be a base-class array (default).
ndmin : int, optional
    Specifies the minimum number of dimensions that the resulting
    array should have.  Ones will be pre-pended to the shape as
    needed to meet this requirement.

Returns
-------
out : ndarray
    An array object satisfying the specified requirements.

See Also
--------
empty_like : Return an empty array with shape and type of input.
ones_like : Return an array of ones with shape and type of input.
zeros_like : Return an array of zeros with shape and type of input.
full_like : Return a new array with shape of input filled with value.
empty : Return a new uninitialized array.
ones : Return a new array setting values to one.
zeros : Return a new array setting values to zero.
full : Return a new array of given shape filled with value.


Notes
-----
When order is 'A' and `object` is an array in neither 'C' nor 'F' order,
and a copy is forced by a change in dtype, then the order of the result is
not necessarily 'C' as expected. This is likely a bug.

Examples
--------
>>> np.array([1, 2, 3])
array([1, 2, 3])

Upcasting:

>>> np.array([1, 2, 3.0])
array([ 1.,  2.,  3.])

More than one dimension:

>>> np.array([[1, 2], [3, 4]])
array([[1, 2],
       [3, 4]])

Minimum dimensions 2:

>>> np.array([1, 2, 3], ndmin=2)
array([[1, 2, 3]])

Type provided:

>>> np.array([1, 2, 3], dtype=complex)
array([ 1.+0.j,  2.+0.j,  3.+0.j])

Data-type consisting of more than one element:

>>> x = np.array([(1,2),(3,4)],dtype=[('a','<i4'),('b','<i4')])
>>> x['a']
array([1, 3])

Creating an array from sub-classes:

>>> np.array(np.mat('1 2; 3 4'))
array([[1, 2],
       [3, 4]])

>>> np.array(np.mat('1 2; 3 4'), subok=True)
matrix([[1, 2],
        [3, 4]])
*)

val asanyarray : ?dtype:Np.Dtype.t -> ?order:[`F | `C] -> a:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Convert the input to an ndarray, but pass ndarray subclasses through.

Parameters
----------
a : array_like
    Input data, in any form that can be converted to an array.  This
    includes scalars, lists, lists of tuples, tuples, tuples of tuples,
    tuples of lists, and ndarrays.
dtype : data-type, optional
    By default, the data-type is inferred from the input data.
order : {'C', 'F'}, optional
    Whether to use row-major (C-style) or column-major
    (Fortran-style) memory representation.  Defaults to 'C'.

Returns
-------
out : ndarray or an ndarray subclass
    Array interpretation of `a`.  If `a` is an ndarray or a subclass
    of ndarray, it is returned as-is and no copy is performed.

See Also
--------
asarray : Similar function which always returns ndarrays.
ascontiguousarray : Convert input to a contiguous array.
asfarray : Convert input to a floating point ndarray.
asfortranarray : Convert input to an ndarray with column-major
                 memory order.
asarray_chkfinite : Similar function which checks input for NaNs and
                    Infs.
fromiter : Create an array from an iterator.
fromfunction : Construct an array by executing a function on grid
               positions.

Examples
--------
Convert a list into an array:

>>> a = [1, 2]
>>> np.asanyarray(a)
array([1, 2])

Instances of `ndarray` subclasses are passed through as-is:

>>> a = np.array([(1.0, 2), (3.0, 4)], dtype='f4,i4').view(np.recarray)
>>> np.asanyarray(a) is a
True
*)

val asarray : ?dtype:Np.Dtype.t -> ?order:[`F | `C] -> a:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Convert the input to an array.

Parameters
----------
a : array_like
    Input data, in any form that can be converted to an array.  This
    includes lists, lists of tuples, tuples, tuples of tuples, tuples
    of lists and ndarrays.
dtype : data-type, optional
    By default, the data-type is inferred from the input data.
order : {'C', 'F'}, optional
    Whether to use row-major (C-style) or
    column-major (Fortran-style) memory representation.
    Defaults to 'C'.

Returns
-------
out : ndarray
    Array interpretation of `a`.  No copy is performed if the input
    is already an ndarray with matching dtype and order.  If `a` is a
    subclass of ndarray, a base class ndarray is returned.

See Also
--------
asanyarray : Similar function which passes through subclasses.
ascontiguousarray : Convert input to a contiguous array.
asfarray : Convert input to a floating point ndarray.
asfortranarray : Convert input to an ndarray with column-major
                 memory order.
asarray_chkfinite : Similar function which checks input for NaNs and Infs.
fromiter : Create an array from an iterator.
fromfunction : Construct an array by executing a function on grid
               positions.

Examples
--------
Convert a list into an array:

>>> a = [1, 2]
>>> np.asarray(a)
array([1, 2])

Existing arrays are not copied:

>>> a = np.array([1, 2])
>>> np.asarray(a) is a
True

If `dtype` is set, array is copied only if dtype does not match:

>>> a = np.array([1, 2], dtype=np.float32)
>>> np.asarray(a, dtype=np.float32) is a
True
>>> np.asarray(a, dtype=np.float64) is a
False

Contrary to `asanyarray`, ndarray subclasses are not passed through:

>>> issubclass(np.recarray, np.ndarray)
True
>>> a = np.array([(1.0, 2), (3.0, 4)], dtype='f4,i4').view(np.recarray)
>>> np.asarray(a) is a
False
>>> np.asanyarray(a) is a
True
*)

val aslinearoperator : Py.Object.t -> Py.Object.t
(**
Return A as a LinearOperator.

'A' may be any of the following types:
 - ndarray
 - matrix
 - sparse matrix (e.g. csr_matrix, lil_matrix, etc.)
 - LinearOperator
 - An object with .shape and .matvec attributes

See the LinearOperator documentation for additional information.

Notes
-----
If 'A' has no .dtype attribute, the data type is determined by calling
:func:`LinearOperator.matvec()` - set the .dtype attribute to prevent this
call upon the linear operator creation.

Examples
--------
>>> from scipy.sparse.linalg import aslinearoperator
>>> M = np.array([[1,2,3],[4,5,6]], dtype=np.int32)
>>> aslinearoperator(M)
<2x3 MatrixLinearOperator with dtype=int32>
*)

val asmatrix : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> Py.Object.t
(**
None
*)

val coerce : x:Py.Object.t -> y:Py.Object.t -> unit -> Py.Object.t
(**
None
*)

val id : Py.Object.t -> Py.Object.t
(**
None
*)

val make_system : a:Py.Object.t -> m:Py.Object.t -> x0:[`Ndarray of [>`Ndarray] Np.Obj.t | `None] -> b:[>`Ndarray] Np.Obj.t -> unit -> (Py.Object.t * Py.Object.t * Py.Object.t * Py.Object.t)
(**
Make a linear system Ax=b

Parameters
----------
A : LinearOperator
    sparse or dense matrix (or any valid input to aslinearoperator)
M : {LinearOperator, Nones}
    preconditioner
    sparse or dense matrix (or any valid input to aslinearoperator)
x0 : {array_like, None}
    initial guess to iterative method
b : array_like
    right hand side

Returns
-------
(A, M, x, b, postprocess)
    A : LinearOperator
        matrix of the linear system
    M : LinearOperator
        preconditioner
    x : rank 1 ndarray
        initial guess
    b : rank 1 ndarray
        right hand side
    postprocess : function
        converts the solution vector to the appropriate
        type and dimensions (e.g. (N,1) matrix)
*)

val zeros : ?dtype:Np.Dtype.t -> ?order:[`C | `F] -> shape:int list -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
zeros(shape, dtype=float, order='C')

Return a new array of given shape and type, filled with zeros.

Parameters
----------
shape : int or tuple of ints
    Shape of the new array, e.g., ``(2, 3)`` or ``2``.
dtype : data-type, optional
    The desired data-type for the array, e.g., `numpy.int8`.  Default is
    `numpy.float64`.
order : {'C', 'F'}, optional, default: 'C'
    Whether to store multi-dimensional data in row-major
    (C-style) or column-major (Fortran-style) order in
    memory.

Returns
-------
out : ndarray
    Array of zeros with the given shape, dtype, and order.

See Also
--------
zeros_like : Return an array of zeros with shape and type of input.
empty : Return a new uninitialized array.
ones : Return a new array setting values to one.
full : Return a new array of given shape filled with value.

Examples
--------
>>> np.zeros(5)
array([ 0.,  0.,  0.,  0.,  0.])

>>> np.zeros((5,), dtype=int)
array([0, 0, 0, 0, 0])

>>> np.zeros((2, 1))
array([[ 0.],
       [ 0.]])

>>> s = (2,2)
>>> np.zeros(s)
array([[ 0.,  0.],
       [ 0.,  0.]])

>>> np.zeros((2,), dtype=[('x', 'i4'), ('y', 'i4')]) # custom dtype
array([(0, 0), (0, 0)],
      dtype=[('x', '<i4'), ('y', '<i4')])
*)


end

val aslinearoperator : Py.Object.t -> Py.Object.t
(**
Return A as a LinearOperator.

'A' may be any of the following types:
 - ndarray
 - matrix
 - sparse matrix (e.g. csr_matrix, lil_matrix, etc.)
 - LinearOperator
 - An object with .shape and .matvec attributes

See the LinearOperator documentation for additional information.

Notes
-----
If 'A' has no .dtype attribute, the data type is determined by calling
:func:`LinearOperator.matvec()` - set the .dtype attribute to prevent this
call upon the linear operator creation.

Examples
--------
>>> from scipy.sparse.linalg import aslinearoperator
>>> M = np.array([[1,2,3],[4,5,6]], dtype=np.int32)
>>> aslinearoperator(M)
<2x3 MatrixLinearOperator with dtype=int32>
*)

val bicg : ?x0:Py.Object.t -> ?tol:Py.Object.t -> ?maxiter:Py.Object.t -> ?m:Py.Object.t -> ?callback:Py.Object.t -> ?atol:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Use BIConjugate Gradient iteration to solve ``Ax = b``.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real or complex N-by-N matrix of the linear system.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` and ``A^T x`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).

Returns
-------
x : {array, matrix}
    The converged solution.
info : integer
    Provides convergence information:
        0  : successful exit
        >0 : convergence to tolerance not achieved, number of iterations
        <0 : illegal input or breakdown

Other Parameters
----------------
x0  : {array, matrix}
    Starting guess for the solution.
tol, atol : float, optional
    Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
    The default for ``atol`` is ``'legacy'``, which emulates
    a different legacy behavior.

    .. warning::

       The default value for `atol` will be changed in a future release.
       For future compatibility, specify `atol` explicitly.
maxiter : integer
    Maximum number of iterations.  Iteration will stop after maxiter
    steps even if the specified tolerance has not been achieved.
M : {sparse matrix, dense matrix, LinearOperator}
    Preconditioner for A.  The preconditioner should approximate the
    inverse of A.  Effective preconditioning dramatically improves the
    rate of convergence, which implies that fewer iterations are needed
    to reach a given error tolerance.
callback : function
    User-supplied function to call after each iteration.  It is called
    as callback(xk), where xk is the current solution vector.
*)

val bicgstab : ?x0:Py.Object.t -> ?tol:Py.Object.t -> ?maxiter:Py.Object.t -> ?m:Py.Object.t -> ?callback:Py.Object.t -> ?atol:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Use BIConjugate Gradient STABilized iteration to solve ``Ax = b``.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real or complex N-by-N matrix of the linear system.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).

Returns
-------
x : {array, matrix}
    The converged solution.
info : integer
    Provides convergence information:
        0  : successful exit
        >0 : convergence to tolerance not achieved, number of iterations
        <0 : illegal input or breakdown

Other Parameters
----------------
x0  : {array, matrix}
    Starting guess for the solution.
tol, atol : float, optional
    Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
    The default for ``atol`` is ``'legacy'``, which emulates
    a different legacy behavior.

    .. warning::

       The default value for `atol` will be changed in a future release.
       For future compatibility, specify `atol` explicitly.
maxiter : integer
    Maximum number of iterations.  Iteration will stop after maxiter
    steps even if the specified tolerance has not been achieved.
M : {sparse matrix, dense matrix, LinearOperator}
    Preconditioner for A.  The preconditioner should approximate the
    inverse of A.  Effective preconditioning dramatically improves the
    rate of convergence, which implies that fewer iterations are needed
    to reach a given error tolerance.
callback : function
    User-supplied function to call after each iteration.  It is called
    as callback(xk), where xk is the current solution vector.
*)

val cg : ?x0:Py.Object.t -> ?tol:Py.Object.t -> ?maxiter:Py.Object.t -> ?m:Py.Object.t -> ?callback:Py.Object.t -> ?atol:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Use Conjugate Gradient iteration to solve ``Ax = b``.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real or complex N-by-N matrix of the linear system.
    ``A`` must represent a hermitian, positive definite matrix.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).

Returns
-------
x : {array, matrix}
    The converged solution.
info : integer
    Provides convergence information:
        0  : successful exit
        >0 : convergence to tolerance not achieved, number of iterations
        <0 : illegal input or breakdown

Other Parameters
----------------
x0  : {array, matrix}
    Starting guess for the solution.
tol, atol : float, optional
    Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
    The default for ``atol`` is ``'legacy'``, which emulates
    a different legacy behavior.

    .. warning::

       The default value for `atol` will be changed in a future release.
       For future compatibility, specify `atol` explicitly.
maxiter : integer
    Maximum number of iterations.  Iteration will stop after maxiter
    steps even if the specified tolerance has not been achieved.
M : {sparse matrix, dense matrix, LinearOperator}
    Preconditioner for A.  The preconditioner should approximate the
    inverse of A.  Effective preconditioning dramatically improves the
    rate of convergence, which implies that fewer iterations are needed
    to reach a given error tolerance.
callback : function
    User-supplied function to call after each iteration.  It is called
    as callback(xk), where xk is the current solution vector.
*)

val cgs : ?x0:Py.Object.t -> ?tol:Py.Object.t -> ?maxiter:Py.Object.t -> ?m:Py.Object.t -> ?callback:Py.Object.t -> ?atol:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Use Conjugate Gradient Squared iteration to solve ``Ax = b``.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real-valued N-by-N matrix of the linear system.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).

Returns
-------
x : {array, matrix}
    The converged solution.
info : integer
    Provides convergence information:
        0  : successful exit
        >0 : convergence to tolerance not achieved, number of iterations
        <0 : illegal input or breakdown

Other Parameters
----------------
x0  : {array, matrix}
    Starting guess for the solution.
tol, atol : float, optional
    Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
    The default for ``atol`` is ``'legacy'``, which emulates
    a different legacy behavior.

    .. warning::

       The default value for `atol` will be changed in a future release.
       For future compatibility, specify `atol` explicitly.
maxiter : integer
    Maximum number of iterations.  Iteration will stop after maxiter
    steps even if the specified tolerance has not been achieved.
M : {sparse matrix, dense matrix, LinearOperator}
    Preconditioner for A.  The preconditioner should approximate the
    inverse of A.  Effective preconditioning dramatically improves the
    rate of convergence, which implies that fewer iterations are needed
    to reach a given error tolerance.
callback : function
    User-supplied function to call after each iteration.  It is called
    as callback(xk), where xk is the current solution vector.
*)

val eigs : ?k:int -> ?m:[`Arr of [>`ArrayLike] Np.Obj.t | `LinearOperator of Py.Object.t] -> ?sigma:Py.Object.t -> ?which:[`LM | `SM | `LR | `SR | `LI | `SI] -> ?v0:[>`Ndarray] Np.Obj.t -> ?ncv:int -> ?maxiter:int -> ?tol:float -> ?return_eigenvectors:bool -> ?minv:[`Arr of [>`ArrayLike] Np.Obj.t | `LinearOperator of Py.Object.t] -> ?oPinv:[`Arr of [>`ArrayLike] Np.Obj.t | `LinearOperator of Py.Object.t] -> ?oPpart:Py.Object.t -> a:[`Arr of [>`ArrayLike] Np.Obj.t | `LinearOperator of Py.Object.t] -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Find k eigenvalues and eigenvectors of the square matrix A.

Solves ``A * x[i] = w[i] * x[i]``, the standard eigenvalue problem
for w[i] eigenvalues with corresponding eigenvectors x[i].

If M is specified, solves ``A * x[i] = w[i] * M * x[i]``, the
generalized eigenvalue problem for w[i] eigenvalues
with corresponding eigenvectors x[i]

Parameters
----------
A : ndarray, sparse matrix or LinearOperator
    An array, sparse matrix, or LinearOperator representing
    the operation ``A * x``, where A is a real or complex square matrix.
k : int, optional
    The number of eigenvalues and eigenvectors desired.
    `k` must be smaller than N-1. It is not possible to compute all
    eigenvectors of a matrix.
M : ndarray, sparse matrix or LinearOperator, optional
    An array, sparse matrix, or LinearOperator representing
    the operation M*x for the generalized eigenvalue problem

        A * x = w * M * x.

    M must represent a real, symmetric matrix if A is real, and must
    represent a complex, hermitian matrix if A is complex. For best
    results, the data type of M should be the same as that of A.
    Additionally:

        If `sigma` is None, M is positive definite

        If sigma is specified, M is positive semi-definite

    If sigma is None, eigs requires an operator to compute the solution
    of the linear equation ``M * x = b``.  This is done internally via a
    (sparse) LU decomposition for an explicit matrix M, or via an
    iterative solver for a general linear operator.  Alternatively,
    the user can supply the matrix or operator Minv, which gives
    ``x = Minv * b = M^-1 * b``.
sigma : real or complex, optional
    Find eigenvalues near sigma using shift-invert mode.  This requires
    an operator to compute the solution of the linear system
    ``[A - sigma * M] * x = b``, where M is the identity matrix if
    unspecified. This is computed internally via a (sparse) LU
    decomposition for explicit matrices A & M, or via an iterative
    solver if either A or M is a general linear operator.
    Alternatively, the user can supply the matrix or operator OPinv,
    which gives ``x = OPinv * b = [A - sigma * M]^-1 * b``.
    For a real matrix A, shift-invert can either be done in imaginary
    mode or real mode, specified by the parameter OPpart ('r' or 'i').
    Note that when sigma is specified, the keyword 'which' (below)
    refers to the shifted eigenvalues ``w'[i]`` where:

        If A is real and OPpart == 'r' (default),
          ``w'[i] = 1/2 * [1/(w[i]-sigma) + 1/(w[i]-conj(sigma))]``.

        If A is real and OPpart == 'i',
          ``w'[i] = 1/2i * [1/(w[i]-sigma) - 1/(w[i]-conj(sigma))]``.

        If A is complex, ``w'[i] = 1/(w[i]-sigma)``.

v0 : ndarray, optional
    Starting vector for iteration.
    Default: random
ncv : int, optional
    The number of Lanczos vectors generated
    `ncv` must be greater than `k`; it is recommended that ``ncv > 2*k``.
    Default: ``min(n, max(2*k + 1, 20))``
which : str, ['LM' | 'SM' | 'LR' | 'SR' | 'LI' | 'SI'], optional
    Which `k` eigenvectors and eigenvalues to find:

        'LM' : largest magnitude

        'SM' : smallest magnitude

        'LR' : largest real part

        'SR' : smallest real part

        'LI' : largest imaginary part

        'SI' : smallest imaginary part

    When sigma != None, 'which' refers to the shifted eigenvalues w'[i]
    (see discussion in 'sigma', above).  ARPACK is generally better
    at finding large values than small values.  If small eigenvalues are
    desired, consider using shift-invert mode for better performance.
maxiter : int, optional
    Maximum number of Arnoldi update iterations allowed
    Default: ``n*10``
tol : float, optional
    Relative accuracy for eigenvalues (stopping criterion)
    The default value of 0 implies machine precision.
return_eigenvectors : bool, optional
    Return eigenvectors (True) in addition to eigenvalues
Minv : ndarray, sparse matrix or LinearOperator, optional
    See notes in M, above.
OPinv : ndarray, sparse matrix or LinearOperator, optional
    See notes in sigma, above.
OPpart : {'r' or 'i'}, optional
    See notes in sigma, above

Returns
-------
w : ndarray
    Array of k eigenvalues.
v : ndarray
    An array of `k` eigenvectors.
    ``v[:, i]`` is the eigenvector corresponding to the eigenvalue w[i].

Raises
------
ArpackNoConvergence
    When the requested convergence is not obtained.
    The currently converged eigenvalues and eigenvectors can be found
    as ``eigenvalues`` and ``eigenvectors`` attributes of the exception
    object.

See Also
--------
eigsh : eigenvalues and eigenvectors for symmetric matrix A
svds : singular value decomposition for a matrix A

Notes
-----
This function is a wrapper to the ARPACK [1]_ SNEUPD, DNEUPD, CNEUPD,
ZNEUPD, functions which use the Implicitly Restarted Arnoldi Method to
find the eigenvalues and eigenvectors [2]_.

References
----------
.. [1] ARPACK Software, http://www.caam.rice.edu/software/ARPACK/
.. [2] R. B. Lehoucq, D. C. Sorensen, and C. Yang,  ARPACK USERS GUIDE:
   Solution of Large Scale Eigenvalue Problems by Implicitly Restarted
   Arnoldi Methods. SIAM, Philadelphia, PA, 1998.

Examples
--------
Find 6 eigenvectors of the identity matrix:

>>> from scipy.sparse.linalg import eigs
>>> id = np.eye(13)
>>> vals, vecs = eigs(id, k=6)
>>> vals
array([ 1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j])
>>> vecs.shape
(13, 6)
*)

val eigsh : ?k:int -> ?m:Py.Object.t -> ?sigma:Py.Object.t -> ?which:Py.Object.t -> ?v0:Py.Object.t -> ?ncv:Py.Object.t -> ?maxiter:Py.Object.t -> ?tol:Py.Object.t -> ?return_eigenvectors:Py.Object.t -> ?minv:Py.Object.t -> ?oPinv:Py.Object.t -> ?mode:Py.Object.t -> a:[`Arr of [>`ArrayLike] Np.Obj.t | `LinearOperator of Py.Object.t] -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Find k eigenvalues and eigenvectors of the real symmetric square matrix
or complex hermitian matrix A.

Solves ``A * x[i] = w[i] * x[i]``, the standard eigenvalue problem for
w[i] eigenvalues with corresponding eigenvectors x[i].

If M is specified, solves ``A * x[i] = w[i] * M * x[i]``, the
generalized eigenvalue problem for w[i] eigenvalues
with corresponding eigenvectors x[i].

Parameters
----------
A : ndarray, sparse matrix or LinearOperator
    A square operator representing the operation ``A * x``, where ``A`` is
    real symmetric or complex hermitian. For buckling mode (see below)
    ``A`` must additionally be positive-definite.
k : int, optional
    The number of eigenvalues and eigenvectors desired.
    `k` must be smaller than N. It is not possible to compute all
    eigenvectors of a matrix.

Returns
-------
w : array
    Array of k eigenvalues.
v : array
    An array representing the `k` eigenvectors.  The column ``v[:, i]`` is
    the eigenvector corresponding to the eigenvalue ``w[i]``.

Other Parameters
----------------
M : An N x N matrix, array, sparse matrix, or linear operator representing
    the operation ``M @ x`` for the generalized eigenvalue problem

        A @ x = w * M @ x.

    M must represent a real, symmetric matrix if A is real, and must
    represent a complex, hermitian matrix if A is complex. For best
    results, the data type of M should be the same as that of A.
    Additionally:

        If sigma is None, M is symmetric positive definite.

        If sigma is specified, M is symmetric positive semi-definite.

        In buckling mode, M is symmetric indefinite.

    If sigma is None, eigsh requires an operator to compute the solution
    of the linear equation ``M @ x = b``. This is done internally via a
    (sparse) LU decomposition for an explicit matrix M, or via an
    iterative solver for a general linear operator.  Alternatively,
    the user can supply the matrix or operator Minv, which gives
    ``x = Minv @ b = M^-1 @ b``.
sigma : real
    Find eigenvalues near sigma using shift-invert mode.  This requires
    an operator to compute the solution of the linear system
    ``[A - sigma * M] x = b``, where M is the identity matrix if
    unspecified.  This is computed internally via a (sparse) LU
    decomposition for explicit matrices A & M, or via an iterative
    solver if either A or M is a general linear operator.
    Alternatively, the user can supply the matrix or operator OPinv,
    which gives ``x = OPinv @ b = [A - sigma * M]^-1 @ b``.
    Note that when sigma is specified, the keyword 'which' refers to
    the shifted eigenvalues ``w'[i]`` where:

        if mode == 'normal', ``w'[i] = 1 / (w[i] - sigma)``.

        if mode == 'cayley', ``w'[i] = (w[i] + sigma) / (w[i] - sigma)``.

        if mode == 'buckling', ``w'[i] = w[i] / (w[i] - sigma)``.

    (see further discussion in 'mode' below)
v0 : ndarray, optional
    Starting vector for iteration.
    Default: random
ncv : int, optional
    The number of Lanczos vectors generated ncv must be greater than k and
    smaller than n; it is recommended that ``ncv > 2*k``.
    Default: ``min(n, max(2*k + 1, 20))``
which : str ['LM' | 'SM' | 'LA' | 'SA' | 'BE']
    If A is a complex hermitian matrix, 'BE' is invalid.
    Which `k` eigenvectors and eigenvalues to find:

        'LM' : Largest (in magnitude) eigenvalues.

        'SM' : Smallest (in magnitude) eigenvalues.

        'LA' : Largest (algebraic) eigenvalues.

        'SA' : Smallest (algebraic) eigenvalues.

        'BE' : Half (k/2) from each end of the spectrum.

    When k is odd, return one more (k/2+1) from the high end.
    When sigma != None, 'which' refers to the shifted eigenvalues ``w'[i]``
    (see discussion in 'sigma', above).  ARPACK is generally better
    at finding large values than small values.  If small eigenvalues are
    desired, consider using shift-invert mode for better performance.
maxiter : int, optional
    Maximum number of Arnoldi update iterations allowed.
    Default: ``n*10``
tol : float
    Relative accuracy for eigenvalues (stopping criterion).
    The default value of 0 implies machine precision.
Minv : N x N matrix, array, sparse matrix, or LinearOperator
    See notes in M, above.
OPinv : N x N matrix, array, sparse matrix, or LinearOperator
    See notes in sigma, above.
return_eigenvectors : bool
    Return eigenvectors (True) in addition to eigenvalues.
    This value determines the order in which eigenvalues are sorted.
    The sort order is also dependent on the `which` variable.

        For which = 'LM' or 'SA':
            If `return_eigenvectors` is True, eigenvalues are sorted by
            algebraic value.

            If `return_eigenvectors` is False, eigenvalues are sorted by
            absolute value.

        For which = 'BE' or 'LA':
            eigenvalues are always sorted by algebraic value.

        For which = 'SM':
            If `return_eigenvectors` is True, eigenvalues are sorted by
            algebraic value.

            If `return_eigenvectors` is False, eigenvalues are sorted by
            decreasing absolute value.

mode : string ['normal' | 'buckling' | 'cayley']
    Specify strategy to use for shift-invert mode.  This argument applies
    only for real-valued A and sigma != None.  For shift-invert mode,
    ARPACK internally solves the eigenvalue problem
    ``OP * x'[i] = w'[i] * B * x'[i]``
    and transforms the resulting Ritz vectors x'[i] and Ritz values w'[i]
    into the desired eigenvectors and eigenvalues of the problem
    ``A * x[i] = w[i] * M * x[i]``.
    The modes are as follows:

        'normal' :
            OP = [A - sigma * M]^-1 @ M,
            B = M,
            w'[i] = 1 / (w[i] - sigma)

        'buckling' :
            OP = [A - sigma * M]^-1 @ A,
            B = A,
            w'[i] = w[i] / (w[i] - sigma)

        'cayley' :
            OP = [A - sigma * M]^-1 @ [A + sigma * M],
            B = M,
            w'[i] = (w[i] + sigma) / (w[i] - sigma)

    The choice of mode will affect which eigenvalues are selected by
    the keyword 'which', and can also impact the stability of
    convergence (see [2] for a discussion).

Raises
------
ArpackNoConvergence
    When the requested convergence is not obtained.

    The currently converged eigenvalues and eigenvectors can be found
    as ``eigenvalues`` and ``eigenvectors`` attributes of the exception
    object.

See Also
--------
eigs : eigenvalues and eigenvectors for a general (nonsymmetric) matrix A
svds : singular value decomposition for a matrix A

Notes
-----
This function is a wrapper to the ARPACK [1]_ SSEUPD and DSEUPD
functions which use the Implicitly Restarted Lanczos Method to
find the eigenvalues and eigenvectors [2]_.

References
----------
.. [1] ARPACK Software, http://www.caam.rice.edu/software/ARPACK/
.. [2] R. B. Lehoucq, D. C. Sorensen, and C. Yang,  ARPACK USERS GUIDE:
   Solution of Large Scale Eigenvalue Problems by Implicitly Restarted
   Arnoldi Methods. SIAM, Philadelphia, PA, 1998.

Examples
--------
>>> from scipy.sparse.linalg import eigsh
>>> identity = np.eye(13)
>>> eigenvalues, eigenvectors = eigsh(identity, k=6)
>>> eigenvalues
array([1., 1., 1., 1., 1., 1.])
>>> eigenvectors.shape
(13, 6)
*)

val expm : [>`ArrayLike] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the matrix exponential using Pade approximation.

Parameters
----------
A : (M,M) array_like or sparse matrix
    2D Array or Matrix (sparse or dense) to be exponentiated

Returns
-------
expA : (M,M) ndarray
    Matrix exponential of `A`

Notes
-----
This is algorithm (6.1) which is a simplification of algorithm (5.1).

.. versionadded:: 0.12.0

References
----------
.. [1] Awad H. Al-Mohy and Nicholas J. Higham (2009)
       'A New Scaling and Squaring Algorithm for the Matrix Exponential.'
       SIAM Journal on Matrix Analysis and Applications.
       31 (3). pp. 970-989. ISSN 1095-7162

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import expm
>>> A = csc_matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
>>> A.todense()
matrix([[1, 0, 0],
        [0, 2, 0],
        [0, 0, 3]], dtype=int64)
>>> Aexp = expm(A)
>>> Aexp
<3x3 sparse matrix of type '<class 'numpy.float64'>'
    with 3 stored elements in Compressed Sparse Column format>
>>> Aexp.todense()
matrix([[  2.71828183,   0.        ,   0.        ],
        [  0.        ,   7.3890561 ,   0.        ],
        [  0.        ,   0.        ,  20.08553692]])
*)

val expm_multiply : ?start:[`Bool of bool | `S of string | `I of int | `F of float] -> ?stop:[`Bool of bool | `S of string | `I of int | `F of float] -> ?num:int -> ?endpoint:bool -> a:Py.Object.t -> b:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the action of the matrix exponential of A on B.

Parameters
----------
A : transposable linear operator
    The operator whose exponential is of interest.
B : ndarray
    The matrix or vector to be multiplied by the matrix exponential of A.
start : scalar, optional
    The starting time point of the sequence.
stop : scalar, optional
    The end time point of the sequence, unless `endpoint` is set to False.
    In that case, the sequence consists of all but the last of ``num + 1``
    evenly spaced time points, so that `stop` is excluded.
    Note that the step size changes when `endpoint` is False.
num : int, optional
    Number of time points to use.
endpoint : bool, optional
    If True, `stop` is the last time point.  Otherwise, it is not included.

Returns
-------
expm_A_B : ndarray
     The result of the action :math:`e^{t_k A} B`.

Notes
-----
The optional arguments defining the sequence of evenly spaced time points
are compatible with the arguments of `numpy.linspace`.

The output ndarray shape is somewhat complicated so I explain it here.
The ndim of the output could be either 1, 2, or 3.
It would be 1 if you are computing the expm action on a single vector
at a single time point.
It would be 2 if you are computing the expm action on a vector
at multiple time points, or if you are computing the expm action
on a matrix at a single time point.
It would be 3 if you want the action on a matrix with multiple
columns at multiple time points.
If multiple time points are requested, expm_A_B[0] will always
be the action of the expm at the first time point,
regardless of whether the action is on a vector or a matrix.

References
----------
.. [1] Awad H. Al-Mohy and Nicholas J. Higham (2011)
       'Computing the Action of the Matrix Exponential,
       with an Application to Exponential Integrators.'
       SIAM Journal on Scientific Computing,
       33 (2). pp. 488-511. ISSN 1064-8275
       http://eprints.ma.man.ac.uk/1591/

.. [2] Nicholas J. Higham and Awad H. Al-Mohy (2010)
       'Computing Matrix Functions.'
       Acta Numerica,
       19. 159-208. ISSN 0962-4929
       http://eprints.ma.man.ac.uk/1451/

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import expm, expm_multiply
>>> A = csc_matrix([[1, 0], [0, 1]])
>>> A.todense()
matrix([[1, 0],
        [0, 1]], dtype=int64)
>>> B = np.array([np.exp(-1.), np.exp(-2.)])
>>> B
array([ 0.36787944,  0.13533528])
>>> expm_multiply(A, B, start=1, stop=2, num=3, endpoint=True)
array([[ 1.        ,  0.36787944],
       [ 1.64872127,  0.60653066],
       [ 2.71828183,  1.        ]])
>>> expm(A).dot(B)                  # Verify 1st timestep
array([ 1.        ,  0.36787944])
>>> expm(1.5*A).dot(B)              # Verify 2nd timestep
array([ 1.64872127,  0.60653066])
>>> expm(2*A).dot(B)                # Verify 3rd timestep
array([ 2.71828183,  1.        ])
*)

val factorized : [>`Ndarray] Np.Obj.t -> Py.Object.t
(**
Return a function for solving a sparse linear system, with A pre-factorized.

Parameters
----------
A : (N, N) array_like
    Input.

Returns
-------
solve : callable
    To solve the linear system of equations given in `A`, the `solve`
    callable should be passed an ndarray of shape (N,).

Examples
--------
>>> from scipy.sparse.linalg import factorized
>>> A = np.array([[ 3. ,  2. , -1. ],
...               [ 2. , -2. ,  4. ],
...               [-1. ,  0.5, -1. ]])
>>> solve = factorized(A) # Makes LU decomposition.
>>> rhs1 = np.array([1, -2, 0])
>>> solve(rhs1) # Uses the LU factors.
array([ 1., -2., -2.])
*)

val gcrotmk : ?x0:[>`Ndarray] Np.Obj.t -> ?tol:Py.Object.t -> ?maxiter:int -> ?m:[`PyObject of Py.Object.t | `Spmatrix of [>`Spmatrix] Np.Obj.t] -> ?callback:Py.Object.t -> ?m':int -> ?k:int -> ?cu:Py.Object.t -> ?discard_C:bool -> ?truncate:[`Oldest | `Smallest] -> ?atol:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Solve a matrix equation using flexible GCROT(m,k) algorithm.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real or complex N-by-N matrix of the linear system.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).
x0  : {array, matrix}
    Starting guess for the solution.
tol, atol : float, optional
    Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
    The default for ``atol`` is `tol`.

    .. warning::

       The default value for `atol` will be changed in a future release.
       For future compatibility, specify `atol` explicitly.
maxiter : int, optional
    Maximum number of iterations.  Iteration will stop after maxiter
    steps even if the specified tolerance has not been achieved.
M : {sparse matrix, dense matrix, LinearOperator}, optional
    Preconditioner for A.  The preconditioner should approximate the
    inverse of A. gcrotmk is a 'flexible' algorithm and the preconditioner
    can vary from iteration to iteration. Effective preconditioning
    dramatically improves the rate of convergence, which implies that
    fewer iterations are needed to reach a given error tolerance.
callback : function, optional
    User-supplied function to call after each iteration.  It is called
    as callback(xk), where xk is the current solution vector.
m : int, optional
    Number of inner FGMRES iterations per each outer iteration.
    Default: 20
k : int, optional
    Number of vectors to carry between inner FGMRES iterations.
    According to [2]_, good values are around m.
    Default: m
CU : list of tuples, optional
    List of tuples ``(c, u)`` which contain the columns of the matrices
    C and U in the GCROT(m,k) algorithm. For details, see [2]_.
    The list given and vectors contained in it are modified in-place.
    If not given, start from empty matrices. The ``c`` elements in the
    tuples can be ``None``, in which case the vectors are recomputed
    via ``c = A u`` on start and orthogonalized as described in [3]_.
discard_C : bool, optional
    Discard the C-vectors at the end. Useful if recycling Krylov subspaces
    for different linear systems.
truncate : {'oldest', 'smallest'}, optional
    Truncation scheme to use. Drop: oldest vectors, or vectors with
    smallest singular values using the scheme discussed in [1,2].
    See [2]_ for detailed comparison.
    Default: 'oldest'

Returns
-------
x : array or matrix
    The solution found.
info : int
    Provides convergence information:

    * 0  : successful exit
    * >0 : convergence to tolerance not achieved, number of iterations

References
----------
.. [1] E. de Sturler, ''Truncation strategies for optimal Krylov subspace
       methods'', SIAM J. Numer. Anal. 36, 864 (1999).
.. [2] J.E. Hicken and D.W. Zingg, ''A simplified and flexible variant
       of GCROT for solving nonsymmetric linear systems'',
       SIAM J. Sci. Comput. 32, 172 (2010).
.. [3] M.L. Parks, E. de Sturler, G. Mackey, D.D. Johnson, S. Maiti,
       ''Recycling Krylov subspaces for sequences of linear systems'',
       SIAM J. Sci. Comput. 28, 1651 (2006).
*)

val gmres : ?x0:Py.Object.t -> ?tol:Py.Object.t -> ?restart:Py.Object.t -> ?maxiter:Py.Object.t -> ?m:Py.Object.t -> ?callback:Py.Object.t -> ?restrt:Py.Object.t -> ?atol:Py.Object.t -> ?callback_type:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Use Generalized Minimal RESidual iteration to solve ``Ax = b``.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real or complex N-by-N matrix of the linear system.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).

Returns
-------
x : {array, matrix}
    The converged solution.
info : int
    Provides convergence information:
      * 0  : successful exit
      * >0 : convergence to tolerance not achieved, number of iterations
      * <0 : illegal input or breakdown

Other parameters
----------------
x0 : {array, matrix}
    Starting guess for the solution (a vector of zeros by default).
tol, atol : float, optional
    Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
    The default for ``atol`` is ``'legacy'``, which emulates
    a different legacy behavior.

    .. warning::

       The default value for `atol` will be changed in a future release.
       For future compatibility, specify `atol` explicitly.
restart : int, optional
    Number of iterations between restarts. Larger values increase
    iteration cost, but may be necessary for convergence.
    Default is 20.
maxiter : int, optional
    Maximum number of iterations (restart cycles).  Iteration will stop
    after maxiter steps even if the specified tolerance has not been
    achieved.
M : {sparse matrix, dense matrix, LinearOperator}
    Inverse of the preconditioner of A.  M should approximate the
    inverse of A and be easy to solve for (see Notes).  Effective
    preconditioning dramatically improves the rate of convergence,
    which implies that fewer iterations are needed to reach a given
    error tolerance.  By default, no preconditioner is used.
callback : function
    User-supplied function to call after each iteration.  It is called
    as `callback(args)`, where `args` are selected by `callback_type`.
callback_type : {'x', 'pr_norm', 'legacy'}, optional
    Callback function argument requested:
      - ``x``: current iterate (ndarray), called on every restart
      - ``pr_norm``: relative (preconditioned) residual norm (float),
        called on every inner iteration
      - ``legacy`` (default): same as ``pr_norm``, but also changes the
        meaning of 'maxiter' to count inner iterations instead of restart
        cycles.
restrt : int, optional
    DEPRECATED - use `restart` instead.

See Also
--------
LinearOperator

Notes
-----
A preconditioner, P, is chosen such that P is close to A but easy to solve
for. The preconditioner parameter required by this routine is
``M = P^-1``. The inverse should preferably not be calculated
explicitly.  Rather, use the following template to produce M::

  # Construct a linear operator that computes P^-1 * x.
  import scipy.sparse.linalg as spla
  M_x = lambda x: spla.spsolve(P, x)
  M = spla.LinearOperator((n, n), M_x)

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import gmres
>>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
>>> b = np.array([2, 4, -1], dtype=float)
>>> x, exitCode = gmres(A, b)
>>> print(exitCode)            # 0 indicates successful convergence
0
>>> np.allclose(A.dot(x), b)
True
*)

val inv : [>`ArrayLike] Np.Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Compute the inverse of a sparse matrix

Parameters
----------
A : (M,M) ndarray or sparse matrix
    square matrix to be inverted

Returns
-------
Ainv : (M,M) ndarray or sparse matrix
    inverse of `A`

Notes
-----
This computes the sparse inverse of `A`.  If the inverse of `A` is expected
to be non-sparse, it will likely be faster to convert `A` to dense and use
scipy.linalg.inv.

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import inv
>>> A = csc_matrix([[1., 0.], [1., 2.]])
>>> Ainv = inv(A)
>>> Ainv
<2x2 sparse matrix of type '<class 'numpy.float64'>'
    with 3 stored elements in Compressed Sparse Column format>
>>> A.dot(Ainv)
<2x2 sparse matrix of type '<class 'numpy.float64'>'
    with 2 stored elements in Compressed Sparse Column format>
>>> A.dot(Ainv).todense()
matrix([[ 1.,  0.],
        [ 0.,  1.]])

.. versionadded:: 0.12.0
*)

val lgmres : ?x0:[>`Ndarray] Np.Obj.t -> ?tol:Py.Object.t -> ?maxiter:int -> ?m:[`PyObject of Py.Object.t | `Spmatrix of [>`Spmatrix] Np.Obj.t] -> ?callback:Py.Object.t -> ?inner_m:int -> ?outer_k:int -> ?outer_v:Py.Object.t -> ?store_outer_Av:bool -> ?prepend_outer_v:bool -> ?atol:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Solve a matrix equation using the LGMRES algorithm.

The LGMRES algorithm [1]_ [2]_ is designed to avoid some problems
in the convergence in restarted GMRES, and often converges in fewer
iterations.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real or complex N-by-N matrix of the linear system.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).
x0  : {array, matrix}
    Starting guess for the solution.
tol, atol : float, optional
    Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
    The default for ``atol`` is `tol`.

    .. warning::

       The default value for `atol` will be changed in a future release.
       For future compatibility, specify `atol` explicitly.
maxiter : int, optional
    Maximum number of iterations.  Iteration will stop after maxiter
    steps even if the specified tolerance has not been achieved.
M : {sparse matrix, dense matrix, LinearOperator}, optional
    Preconditioner for A.  The preconditioner should approximate the
    inverse of A.  Effective preconditioning dramatically improves the
    rate of convergence, which implies that fewer iterations are needed
    to reach a given error tolerance.
callback : function, optional
    User-supplied function to call after each iteration.  It is called
    as callback(xk), where xk is the current solution vector.
inner_m : int, optional
    Number of inner GMRES iterations per each outer iteration.
outer_k : int, optional
    Number of vectors to carry between inner GMRES iterations.
    According to [1]_, good values are in the range of 1...3.
    However, note that if you want to use the additional vectors to
    accelerate solving multiple similar problems, larger values may
    be beneficial.
outer_v : list of tuples, optional
    List containing tuples ``(v, Av)`` of vectors and corresponding
    matrix-vector products, used to augment the Krylov subspace, and
    carried between inner GMRES iterations. The element ``Av`` can
    be `None` if the matrix-vector product should be re-evaluated.
    This parameter is modified in-place by `lgmres`, and can be used
    to pass 'guess' vectors in and out of the algorithm when solving
    similar problems.
store_outer_Av : bool, optional
    Whether LGMRES should store also A*v in addition to vectors `v`
    in the `outer_v` list. Default is True.
prepend_outer_v : bool, optional 
    Whether to put outer_v augmentation vectors before Krylov iterates.
    In standard LGMRES, prepend_outer_v=False.

Returns
-------
x : array or matrix
    The converged solution.
info : int
    Provides convergence information:

        - 0  : successful exit
        - >0 : convergence to tolerance not achieved, number of iterations
        - <0 : illegal input or breakdown

Notes
-----
The LGMRES algorithm [1]_ [2]_ is designed to avoid the
slowing of convergence in restarted GMRES, due to alternating
residual vectors. Typically, it often outperforms GMRES(m) of
comparable memory requirements by some measure, or at least is not
much worse.

Another advantage in this algorithm is that you can supply it with
'guess' vectors in the `outer_v` argument that augment the Krylov
subspace. If the solution lies close to the span of these vectors,
the algorithm converges faster. This can be useful if several very
similar matrices need to be inverted one after another, such as in
Newton-Krylov iteration where the Jacobian matrix often changes
little in the nonlinear steps.

References
----------
.. [1] A.H. Baker and E.R. Jessup and T. Manteuffel, 'A Technique for
         Accelerating the Convergence of Restarted GMRES', SIAM J. Matrix
         Anal. Appl. 26, 962 (2005).
.. [2] A.H. Baker, 'On Improving the Performance of the Linear Solver
         restarted GMRES', PhD thesis, University of Colorado (2003).

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import lgmres
>>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
>>> b = np.array([2, 4, -1], dtype=float)
>>> x, exitCode = lgmres(A, b)
>>> print(exitCode)            # 0 indicates successful convergence
0
>>> np.allclose(A.dot(x), b)
True
*)

val lobpcg : ?b:[`PyObject of Py.Object.t | `Spmatrix of [>`Spmatrix] Np.Obj.t] -> ?m:[`PyObject of Py.Object.t | `Spmatrix of [>`Spmatrix] Np.Obj.t] -> ?y:[`Ndarray of [>`Ndarray] Np.Obj.t | `PyObject of Py.Object.t] -> ?tol:[`Bool of bool | `S of string | `I of int | `F of float] -> ?maxiter:int -> ?largest:bool -> ?verbosityLevel:int -> ?retLambdaHistory:bool -> ?retResidualNormsHistory:bool -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> x:[`Ndarray of [>`Ndarray] Np.Obj.t | `PyObject of Py.Object.t] -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t * Py.Object.t * Py.Object.t)
(**
Locally Optimal Block Preconditioned Conjugate Gradient Method (LOBPCG)

LOBPCG is a preconditioned eigensolver for large symmetric positive
definite (SPD) generalized eigenproblems.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The symmetric linear operator of the problem, usually a
    sparse matrix.  Often called the 'stiffness matrix'.
X : ndarray, float32 or float64
    Initial approximation to the ``k`` eigenvectors (non-sparse). If `A`
    has ``shape=(n,n)`` then `X` should have shape ``shape=(n,k)``.
B : {dense matrix, sparse matrix, LinearOperator}, optional
    The right hand side operator in a generalized eigenproblem.
    By default, ``B = Identity``.  Often called the 'mass matrix'.
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
iteration by calling the 'standard' dense eigensolver, so if ``m`` is not
small enough compared to ``n``, it does not make sense to call the LOBPCG
code, but rather one should use the 'standard' eigensolver, e.g. numpy or
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

val lsmr : ?damp:float -> ?atol:Py.Object.t -> ?btol:Py.Object.t -> ?conlim:float -> ?maxiter:int -> ?show:bool -> ?x0:[>`Ndarray] Np.Obj.t -> a:[`Arr of [>`ArrayLike] Np.Obj.t | `LinearOperator of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> (Py.Object.t * int * int * float * float * float * float * float)
(**
Iterative solver for least-squares problems.

lsmr solves the system of linear equations ``Ax = b``. If the system
is inconsistent, it solves the least-squares problem ``min ||b - Ax||_2``.
A is a rectangular matrix of dimension m-by-n, where all cases are
allowed: m = n, m > n, or m < n. B is a vector of length m.
The matrix A may be dense or sparse (usually sparse).

Parameters
----------
A : {matrix, sparse matrix, ndarray, LinearOperator}
    Matrix A in the linear system.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` and ``A^H x`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : array_like, shape (m,)
    Vector b in the linear system.
damp : float
    Damping factor for regularized least-squares. `lsmr` solves
    the regularized least-squares problem::

     min ||(b) - (  A   )x||
         ||(0)   (damp*I) ||_2

    where damp is a scalar.  If damp is None or 0, the system
    is solved without regularization.
atol, btol : float, optional
    Stopping tolerances. `lsmr` continues iterations until a
    certain backward error estimate is smaller than some quantity
    depending on atol and btol.  Let ``r = b - Ax`` be the
    residual vector for the current approximate solution ``x``.
    If ``Ax = b`` seems to be consistent, ``lsmr`` terminates
    when ``norm(r) <= atol * norm(A) * norm(x) + btol * norm(b)``.
    Otherwise, lsmr terminates when ``norm(A^H r) <=
    atol * norm(A) * norm(r)``.  If both tolerances are 1.0e-6 (say),
    the final ``norm(r)`` should be accurate to about 6
    digits. (The final x will usually have fewer correct digits,
    depending on ``cond(A)`` and the size of LAMBDA.)  If `atol`
    or `btol` is None, a default value of 1.0e-6 will be used.
    Ideally, they should be estimates of the relative error in the
    entries of A and B respectively.  For example, if the entries
    of `A` have 7 correct digits, set atol = 1e-7. This prevents
    the algorithm from doing unnecessary work beyond the
    uncertainty of the input data.
conlim : float, optional
    `lsmr` terminates if an estimate of ``cond(A)`` exceeds
    `conlim`.  For compatible systems ``Ax = b``, conlim could be
    as large as 1.0e+12 (say).  For least-squares problems,
    `conlim` should be less than 1.0e+8. If `conlim` is None, the
    default value is 1e+8.  Maximum precision can be obtained by
    setting ``atol = btol = conlim = 0``, but the number of
    iterations may then be excessive.
maxiter : int, optional
    `lsmr` terminates if the number of iterations reaches
    `maxiter`.  The default is ``maxiter = min(m, n)``.  For
    ill-conditioned systems, a larger value of `maxiter` may be
    needed.
show : bool, optional
    Print iterations logs if ``show=True``.
x0 : array_like, shape (n,), optional
    Initial guess of x, if None zeros are used.

    .. versionadded:: 1.0.0
Returns
-------
x : ndarray of float
    Least-square solution returned.
istop : int
    istop gives the reason for stopping::

      istop   = 0 means x=0 is a solution.  If x0 was given, then x=x0 is a
                  solution.
              = 1 means x is an approximate solution to A*x = B,
                  according to atol and btol.
              = 2 means x approximately solves the least-squares problem
                  according to atol.
              = 3 means COND(A) seems to be greater than CONLIM.
              = 4 is the same as 1 with atol = btol = eps (machine
                  precision)
              = 5 is the same as 2 with atol = eps.
              = 6 is the same as 3 with CONLIM = 1/eps.
              = 7 means ITN reached maxiter before the other stopping
                  conditions were satisfied.

itn : int
    Number of iterations used.
normr : float
    ``norm(b-Ax)``
normar : float
    ``norm(A^H (b - Ax))``
norma : float
    ``norm(A)``
conda : float
    Condition number of A.
normx : float
    ``norm(x)``

Notes
-----

.. versionadded:: 0.11.0

References
----------
.. [1] D. C.-L. Fong and M. A. Saunders,
       'LSMR: An iterative algorithm for sparse least-squares problems',
       SIAM J. Sci. Comput., vol. 33, pp. 2950-2971, 2011.
       https://arxiv.org/abs/1006.0758
.. [2] LSMR Software, https://web.stanford.edu/group/SOL/software/lsmr/

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import lsmr
>>> A = csc_matrix([[1., 0.], [1., 1.], [0., 1.]], dtype=float)

The first example has the trivial solution `[0, 0]`

>>> b = np.array([0., 0., 0.], dtype=float)
>>> x, istop, itn, normr = lsmr(A, b)[:4]
>>> istop
0
>>> x
array([ 0.,  0.])

The stopping code `istop=0` returned indicates that a vector of zeros was
found as a solution. The returned solution `x` indeed contains `[0., 0.]`.
The next example has a non-trivial solution:

>>> b = np.array([1., 0., -1.], dtype=float)
>>> x, istop, itn, normr = lsmr(A, b)[:4]
>>> istop
1
>>> x
array([ 1., -1.])
>>> itn
1
>>> normr
4.440892098500627e-16

As indicated by `istop=1`, `lsmr` found a solution obeying the tolerance
limits. The given solution `[1., -1.]` obviously solves the equation. The
remaining return values include information about the number of iterations
(`itn=1`) and the remaining difference of left and right side of the solved
equation.
The final example demonstrates the behavior in the case where there is no
solution for the equation:

>>> b = np.array([1., 0.01, -1.], dtype=float)
>>> x, istop, itn, normr = lsmr(A, b)[:4]
>>> istop
2
>>> x
array([ 1.00333333, -0.99666667])
>>> A.dot(x)-b
array([ 0.00333333, -0.00333333,  0.00333333])
>>> normr
0.005773502691896255

`istop` indicates that the system is inconsistent and thus `x` is rather an
approximate solution to the corresponding least-squares problem. `normr`
contains the minimal distance that was found.
*)

val lsqr : ?damp:float -> ?atol:Py.Object.t -> ?btol:Py.Object.t -> ?conlim:float -> ?iter_lim:int -> ?show:bool -> ?calc_var:bool -> ?x0:[>`Ndarray] Np.Obj.t -> a:[`Arr of [>`ArrayLike] Np.Obj.t | `LinearOperator of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> (Py.Object.t * int * int * float * float * float * float * float * float * Py.Object.t)
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
If x0 is 'good', norm(r0) will be smaller than norm(b).
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
       'LSQR: An algorithm for sparse linear equations and
       sparse least squares', ACM TOMS 8(1), 43-71.
.. [2] C. C. Paige and M. A. Saunders (1982b).
       'Algorithm 583.  LSQR: Sparse linear equations and least
       squares problems', ACM TOMS 8(2), 195-209.
.. [3] M. A. Saunders (1995).  'Solution of sparse rectangular
       systems using LSQR and CRAIG', BIT 35, 588-604.

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

val minres : ?x0:Py.Object.t -> ?shift:Py.Object.t -> ?tol:Py.Object.t -> ?maxiter:Py.Object.t -> ?m:Py.Object.t -> ?callback:Py.Object.t -> ?show:Py.Object.t -> ?check:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Use MINimum RESidual iteration to solve Ax=b

MINRES minimizes norm(A*x - b) for a real symmetric matrix A.  Unlike
the Conjugate Gradient method, A can be indefinite or singular.

If shift != 0 then the method solves (A - shift*I)x = b

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real symmetric N-by-N matrix of the linear system
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).

Returns
-------
x : {array, matrix}
    The converged solution.
info : integer
    Provides convergence information:
        0  : successful exit
        >0 : convergence to tolerance not achieved, number of iterations
        <0 : illegal input or breakdown

Other Parameters
----------------
x0  : {array, matrix}
    Starting guess for the solution.
tol : float
    Tolerance to achieve. The algorithm terminates when the relative
    residual is below `tol`.
maxiter : integer
    Maximum number of iterations.  Iteration will stop after maxiter
    steps even if the specified tolerance has not been achieved.
M : {sparse matrix, dense matrix, LinearOperator}
    Preconditioner for A.  The preconditioner should approximate the
    inverse of A.  Effective preconditioning dramatically improves the
    rate of convergence, which implies that fewer iterations are needed
    to reach a given error tolerance.
callback : function
    User-supplied function to call after each iteration.  It is called
    as callback(xk), where xk is the current solution vector.

References
----------
Solution of sparse indefinite systems of linear equations,
    C. C. Paige and M. A. Saunders (1975),
    SIAM J. Numer. Anal. 12(4), pp. 617-629.
    https://web.stanford.edu/group/SOL/software/minres/

This file is a translation of the following MATLAB implementation:
    https://web.stanford.edu/group/SOL/software/minres/minres-matlab.zip
*)

val norm : ?ord:[`Fro | `PyObject of Py.Object.t] -> ?axis:[`T2_tuple_of_ints of Py.Object.t | `I of int] -> x:Py.Object.t -> unit -> Py.Object.t
(**
Norm of a sparse matrix

This function is able to return one of seven different matrix norms,
depending on the value of the ``ord`` parameter.

Parameters
----------
x : a sparse matrix
    Input sparse matrix.
ord : {non-zero int, inf, -inf, 'fro'}, optional
    Order of the norm (see table under ``Notes``). inf means numpy's
    `inf` object.
axis : {int, 2-tuple of ints, None}, optional
    If `axis` is an integer, it specifies the axis of `x` along which to
    compute the vector norms.  If `axis` is a 2-tuple, it specifies the
    axes that hold 2-D matrices, and the matrix norms of these matrices
    are computed.  If `axis` is None then either a vector norm (when `x`
    is 1-D) or a matrix norm (when `x` is 2-D) is returned.

Returns
-------
n : float or ndarray

Notes
-----
Some of the ord are not implemented because some associated functions like, 
_multi_svd_norm, are not yet available for sparse matrix. 

This docstring is modified based on numpy.linalg.norm. 
https://github.com/numpy/numpy/blob/master/numpy/linalg/linalg.py 

The following norms can be calculated:

=====  ============================  
ord    norm for sparse matrices             
=====  ============================  
None   Frobenius norm                
'fro'  Frobenius norm                
inf    max(sum(abs(x), axis=1))      
-inf   min(sum(abs(x), axis=1))      
0      abs(x).sum(axis=axis)                           
1      max(sum(abs(x), axis=0))      
-1     min(sum(abs(x), axis=0))      
2      Not implemented  
-2     Not implemented      
other  Not implemented                               
=====  ============================  

The Frobenius norm is given by [1]_:

    :math:`||A||_F = [\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

References
----------
.. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
    Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15

Examples
--------
>>> from scipy.sparse import *
>>> import numpy as np
>>> from scipy.sparse.linalg import norm
>>> a = np.arange(9) - 4
>>> a
array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
>>> b = a.reshape((3, 3))
>>> b
array([[-4, -3, -2],
       [-1, 0, 1],
       [ 2, 3, 4]])

>>> b = csr_matrix(b)
>>> norm(b)
7.745966692414834
>>> norm(b, 'fro')
7.745966692414834
>>> norm(b, np.inf)
9
>>> norm(b, -np.inf)
2
>>> norm(b, 1)
7
>>> norm(b, -1)
6
*)

val onenormest : ?t:int -> ?itmax:int -> ?compute_v:bool -> ?compute_w:bool -> a:[`Ndarray of [>`Ndarray] Np.Obj.t | `Other_linear_operator of Py.Object.t] -> unit -> (float * [`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Compute a lower bound of the 1-norm of a sparse matrix.

Parameters
----------
A : ndarray or other linear operator
    A linear operator that can be transposed and that can
    produce matrix products.
t : int, optional
    A positive parameter controlling the tradeoff between
    accuracy versus time and memory usage.
    Larger values take longer and use more memory
    but give more accurate output.
itmax : int, optional
    Use at most this many iterations.
compute_v : bool, optional
    Request a norm-maximizing linear operator input vector if True.
compute_w : bool, optional
    Request a norm-maximizing linear operator output vector if True.

Returns
-------
est : float
    An underestimate of the 1-norm of the sparse matrix.
v : ndarray, optional
    The vector such that ||Av||_1 == est*||v||_1.
    It can be thought of as an input to the linear operator
    that gives an output with particularly large norm.
w : ndarray, optional
    The vector Av which has relatively large 1-norm.
    It can be thought of as an output of the linear operator
    that is relatively large in norm compared to the input.

Notes
-----
This is algorithm 2.4 of [1].

In [2] it is described as follows.
'This algorithm typically requires the evaluation of
about 4t matrix-vector products and almost invariably
produces a norm estimate (which is, in fact, a lower
bound on the norm) correct to within a factor 3.'

.. versionadded:: 0.13.0

References
----------
.. [1] Nicholas J. Higham and Francoise Tisseur (2000),
       'A Block Algorithm for Matrix 1-Norm Estimation,
       with an Application to 1-Norm Pseudospectra.'
       SIAM J. Matrix Anal. Appl. Vol. 21, No. 4, pp. 1185-1201.

.. [2] Awad H. Al-Mohy and Nicholas J. Higham (2009),
       'A new scaling and squaring algorithm for the matrix exponential.'
       SIAM J. Matrix Anal. Appl. Vol. 31, No. 3, pp. 970-989.

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import onenormest
>>> A = csc_matrix([[1., 0., 0.], [5., 8., 2.], [0., -1., 0.]], dtype=float)
>>> A.todense()
matrix([[ 1.,  0.,  0.],
        [ 5.,  8.,  2.],
        [ 0., -1.,  0.]])
>>> onenormest(A)
9.0
>>> np.linalg.norm(A.todense(), ord=1)
9.0
*)

val qmr : ?x0:Py.Object.t -> ?tol:Py.Object.t -> ?maxiter:Py.Object.t -> ?m1:Py.Object.t -> ?m2:Py.Object.t -> ?callback:Py.Object.t -> ?atol:Py.Object.t -> a:[`Spmatrix of [>`Spmatrix] Np.Obj.t | `PyObject of Py.Object.t] -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Use Quasi-Minimal Residual iteration to solve ``Ax = b``.

Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}
    The real-valued N-by-N matrix of the linear system.
    Alternatively, ``A`` can be a linear operator which can
    produce ``Ax`` and ``A^T x`` using, e.g.,
    ``scipy.sparse.linalg.LinearOperator``.
b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).

Returns
-------
x : {array, matrix}
    The converged solution.
info : integer
    Provides convergence information:
        0  : successful exit
        >0 : convergence to tolerance not achieved, number of iterations
        <0 : illegal input or breakdown

Other Parameters
----------------
x0  : {array, matrix}
    Starting guess for the solution.
tol, atol : float, optional
    Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
    The default for ``atol`` is ``'legacy'``, which emulates
    a different legacy behavior.

    .. warning::

       The default value for `atol` will be changed in a future release.
       For future compatibility, specify `atol` explicitly.
maxiter : integer
    Maximum number of iterations.  Iteration will stop after maxiter
    steps even if the specified tolerance has not been achieved.
M1 : {sparse matrix, dense matrix, LinearOperator}
    Left preconditioner for A.
M2 : {sparse matrix, dense matrix, LinearOperator}
    Right preconditioner for A. Used together with the left
    preconditioner M1.  The matrix M1*A*M2 should have better
    conditioned than A alone.
callback : function
    User-supplied function to call after each iteration.  It is called
    as callback(xk), where xk is the current solution vector.

See Also
--------
LinearOperator

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import qmr
>>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
>>> b = np.array([2, 4, -1], dtype=float)
>>> x, exitCode = qmr(A, b)
>>> print(exitCode)            # 0 indicates successful convergence
0
>>> np.allclose(A.dot(x), b)
True
*)

val spilu : ?drop_tol:float -> ?fill_factor:float -> ?drop_rule:string -> ?permc_spec:Py.Object.t -> ?diag_pivot_thresh:Py.Object.t -> ?relax:Py.Object.t -> ?panel_size:Py.Object.t -> ?options:Py.Object.t -> a:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Compute an incomplete LU decomposition for a sparse, square matrix.

The resulting object is an approximation to the inverse of `A`.

Parameters
----------
A : (N, N) array_like
    Sparse matrix to factorize
drop_tol : float, optional
    Drop tolerance (0 <= tol <= 1) for an incomplete LU decomposition.
    (default: 1e-4)
fill_factor : float, optional
    Specifies the fill ratio upper bound (>= 1.0) for ILU. (default: 10)
drop_rule : str, optional
    Comma-separated string of drop rules to use.
    Available rules: ``basic``, ``prows``, ``column``, ``area``,
    ``secondary``, ``dynamic``, ``interp``. (Default: ``basic,area``)

    See SuperLU documentation for details.

Remaining other options
    Same as for `splu`

Returns
-------
invA_approx : scipy.sparse.linalg.SuperLU
    Object, which has a ``solve`` method.

See also
--------
splu : complete LU decomposition

Notes
-----
To improve the better approximation to the inverse, you may need to
increase `fill_factor` AND decrease `drop_tol`.

This function uses the SuperLU library.

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import spilu
>>> A = csc_matrix([[1., 0., 0.], [5., 0., 2.], [0., -1., 0.]], dtype=float)
>>> B = spilu(A)
>>> x = np.array([1., 2., 3.], dtype=float)
>>> B.solve(x)
array([ 1. , -3. , -1.5])
>>> A.dot(B.solve(x))
array([ 1.,  2.,  3.])
>>> B.solve(A.dot(x))
array([ 1.,  2.,  3.])
*)

val splu : ?permc_spec:string -> ?diag_pivot_thresh:float -> ?relax:int -> ?panel_size:int -> ?options:Py.Object.t -> a:[>`Spmatrix] Np.Obj.t -> unit -> Py.Object.t
(**
Compute the LU decomposition of a sparse, square matrix.

Parameters
----------
A : sparse matrix
    Sparse matrix to factorize. Should be in CSR or CSC format.
permc_spec : str, optional
    How to permute the columns of the matrix for sparsity preservation.
    (default: 'COLAMD')

    - ``NATURAL``: natural ordering.
    - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
    - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
    - ``COLAMD``: approximate minimum degree column ordering

diag_pivot_thresh : float, optional
    Threshold used for a diagonal entry to be an acceptable pivot.
    See SuperLU user's guide for details [1]_
relax : int, optional
    Expert option for customizing the degree of relaxing supernodes.
    See SuperLU user's guide for details [1]_
panel_size : int, optional
    Expert option for customizing the panel size.
    See SuperLU user's guide for details [1]_
options : dict, optional
    Dictionary containing additional expert options to SuperLU.
    See SuperLU user guide [1]_ (section 2.4 on the 'Options' argument)
    for more details. For example, you can specify
    ``options=dict(Equil=False, IterRefine='SINGLE'))``
    to turn equilibration off and perform a single iterative refinement.

Returns
-------
invA : scipy.sparse.linalg.SuperLU
    Object, which has a ``solve`` method.

See also
--------
spilu : incomplete LU decomposition

Notes
-----
This function uses the SuperLU library.

References
----------
.. [1] SuperLU http://crd.lbl.gov/~xiaoye/SuperLU/

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import splu
>>> A = csc_matrix([[1., 0., 0.], [5., 0., 2.], [0., -1., 0.]], dtype=float)
>>> B = splu(A)
>>> x = np.array([1., 2., 3.], dtype=float)
>>> B.solve(x)
array([ 1. , -3. , -1.5])
>>> A.dot(B.solve(x))
array([ 1.,  2.,  3.])
>>> B.solve(A.dot(x))
array([ 1.,  2.,  3.])
*)

val spsolve : ?permc_spec:string -> ?use_umfpack:bool -> a:[>`ArrayLike] Np.Obj.t -> b:[>`ArrayLike] Np.Obj.t -> unit -> [>`ArrayLike] Np.Obj.t
(**
Solve the sparse linear system Ax=b, where b may be a vector or a matrix.

Parameters
----------
A : ndarray or sparse matrix
    The square matrix A will be converted into CSC or CSR form
b : ndarray or sparse matrix
    The matrix or vector representing the right hand side of the equation.
    If a vector, b.shape must be (n,) or (n, 1).
permc_spec : str, optional
    How to permute the columns of the matrix for sparsity preservation.
    (default: 'COLAMD')

    - ``NATURAL``: natural ordering.
    - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
    - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
    - ``COLAMD``: approximate minimum degree column ordering
use_umfpack : bool, optional
    if True (default) then use umfpack for the solution.  This is
    only referenced if b is a vector and ``scikit-umfpack`` is installed.

Returns
-------
x : ndarray or sparse matrix
    the solution of the sparse linear equation.
    If b is a vector, then x is a vector of size A.shape[1]
    If b is a matrix, then x is a matrix of size (A.shape[1], b.shape[1])

Notes
-----
For solving the matrix expression AX = B, this solver assumes the resulting
matrix X is sparse, as is often the case for very sparse inputs.  If the
resulting X is dense, the construction of this sparse result will be
relatively expensive.  In that case, consider converting A to a dense
matrix and using scipy.linalg.solve or its variants.

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import spsolve
>>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
>>> B = csc_matrix([[2, 0], [-1, 0], [2, 0]], dtype=float)
>>> x = spsolve(A, B)
>>> np.allclose(A.dot(x).todense(), B.todense())
True
*)

val spsolve_triangular : ?lower:bool -> ?overwrite_A:bool -> ?overwrite_b:bool -> ?unit_diagonal:bool -> a:[>`Spmatrix] Np.Obj.t -> b:Py.Object.t -> unit -> Py.Object.t
(**
Solve the equation `A x = b` for `x`, assuming A is a triangular matrix.

Parameters
----------
A : (M, M) sparse matrix
    A sparse square triangular matrix. Should be in CSR format.
b : (M,) or (M, N) array_like
    Right-hand side matrix in `A x = b`
lower : bool, optional
    Whether `A` is a lower or upper triangular matrix.
    Default is lower triangular matrix.
overwrite_A : bool, optional
    Allow changing `A`. The indices of `A` are going to be sorted and zero
    entries are going to be removed.
    Enabling gives a performance gain. Default is False.
overwrite_b : bool, optional
    Allow overwriting data in `b`.
    Enabling gives a performance gain. Default is False.
    If `overwrite_b` is True, it should be ensured that
    `b` has an appropriate dtype to be able to store the result.
unit_diagonal : bool, optional
    If True, diagonal elements of `a` are assumed to be 1 and will not be
    referenced.

    .. versionadded:: 1.4.0

Returns
-------
x : (M,) or (M, N) ndarray
    Solution to the system `A x = b`. Shape of return matches shape of `b`.

Raises
------
LinAlgError
    If `A` is singular or not triangular.
ValueError
    If shape of `A` or shape of `b` do not match the requirements.

Notes
-----
.. versionadded:: 0.19.0

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> from scipy.sparse.linalg import spsolve_triangular
>>> A = csr_matrix([[3, 0, 0], [1, -1, 0], [2, 0, 1]], dtype=float)
>>> B = np.array([[2, 0], [-1, 0], [2, 0]], dtype=float)
>>> x = spsolve_triangular(A, B)
>>> np.allclose(A.dot(x), B)
True
*)

val svds : ?k:int -> ?ncv:int -> ?tol:float -> ?which:[`LM | `SM] -> ?v0:[>`Ndarray] Np.Obj.t -> ?maxiter:int -> ?return_singular_vectors:[`Bool of bool | `S of string] -> ?solver:string -> a:[`LinearOperator of Py.Object.t | `Spmatrix of [>`Spmatrix] Np.Obj.t] -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Compute the largest or smallest k singular values/vectors for a sparse matrix. The order of the singular values is not guaranteed.

Parameters
----------
A : {sparse matrix, LinearOperator}
    Array to compute the SVD on, of shape (M, N)
k : int, optional
    Number of singular values and vectors to compute.
    Must be 1 <= k < min(A.shape).
ncv : int, optional
    The number of Lanczos vectors generated
    ncv must be greater than k+1 and smaller than n;
    it is recommended that ncv > 2*k
    Default: ``min(n, max(2*k + 1, 20))``
tol : float, optional
    Tolerance for singular values. Zero (default) means machine precision.
which : str, ['LM' | 'SM'], optional
    Which `k` singular values to find:

        - 'LM' : largest singular values
        - 'SM' : smallest singular values

    .. versionadded:: 0.12.0
v0 : ndarray, optional
    Starting vector for iteration, of length min(A.shape). Should be an
    (approximate) left singular vector if N > M and a right singular
    vector otherwise.
    Default: random

    .. versionadded:: 0.12.0
maxiter : int, optional
    Maximum number of iterations.

    .. versionadded:: 0.12.0
return_singular_vectors : bool or str, optional
    - True: return singular vectors (True) in addition to singular values.

    .. versionadded:: 0.12.0

    - 'u': only return the u matrix, without computing vh (if N > M).
    - 'vh': only return the vh matrix, without computing u (if N <= M).

    .. versionadded:: 0.16.0
solver : str, optional
        Eigenvalue solver to use. Should be 'arpack' or 'lobpcg'.
        Default: 'arpack'

Returns
-------
u : ndarray, shape=(M, k)
    Unitary matrix having left singular vectors as columns.
    If `return_singular_vectors` is 'vh', this variable is not computed,
    and None is returned instead.
s : ndarray, shape=(k,)
    The singular values.
vt : ndarray, shape=(k, N)
    Unitary matrix having right singular vectors as rows.
    If `return_singular_vectors` is 'u', this variable is not computed,
    and None is returned instead.


Notes
-----
This is a naive implementation using ARPACK or LOBPCG as an eigensolver
on A.H * A or A * A.H, depending on which one is more efficient.

Examples
--------
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import svds, eigs
>>> A = csc_matrix([[1, 0, 0], [5, 0, 2], [0, -1, 0], [0, 0, 3]], dtype=float)
>>> u, s, vt = svds(A, k=2)
>>> s
array([ 2.75193379,  5.6059665 ])
>>> np.sqrt(eigs(A.dot(A.T), k=2)[0]).real
array([ 5.6059665 ,  2.75193379])
*)

val use_solver : ?kwargs:(string * Py.Object.t) list -> unit -> Py.Object.t
(**
Select default sparse direct solver to be used.

Parameters
----------
useUmfpack : bool, optional
    Use UMFPACK over SuperLU. Has effect only if scikits.umfpack is
    installed. Default: True
assumeSortedIndices : bool, optional
    Allow UMFPACK to skip the step of sorting indices for a CSR/CSC matrix.
    Has effect only if useUmfpack is True and scikits.umfpack is installed.
    Default: False

Notes
-----
The default sparse solver is umfpack when available
(scikits.umfpack is installed). This can be changed by passing
useUmfpack = False, which then causes the always present SuperLU
based solver to be used.

Umfpack requires a CSR/CSC matrix to have sorted column/row indices. If
sure that the matrix fulfills this, pass ``assumeSortedIndices=True``
to gain some speed.
*)


end

module Sputils : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val asmatrix : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> Py.Object.t
(**
None
*)

val bmat : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> Py.Object.t
(**
None
*)

val check_reshape_kwargs : Py.Object.t -> Py.Object.t
(**
Unpack keyword arguments for reshape function.

This is useful because keyword arguments after star arguments are not
allowed in Python 2, but star keyword arguments are. This function unpacks
'order' and 'copy' from the star keyword arguments (with defaults) and
throws an error for any remaining.
*)

val check_shape : ?current_shape:Py.Object.t -> args:Py.Object.t -> unit -> Py.Object.t
(**
Imitate numpy.matrix handling of shape arguments
*)

val downcast_intp_index : Py.Object.t -> Py.Object.t
(**
Down-cast index array to np.intp dtype if it is of a larger dtype.

Raise an error if the array contains a value that is too large for
intp.
*)

val get_index_dtype : ?arrays:Py.Object.t -> ?maxval:float -> ?check_contents:bool -> unit -> Np.Dtype.t
(**
Based on input (integer) arrays `a`, determine a suitable index data
type that can hold the data in the arrays.

Parameters
----------
arrays : tuple of array_like
    Input arrays whose types/contents to check
maxval : float, optional
    Maximum value needed
check_contents : bool, optional
    Whether to check the values in the arrays and not just their types.
    Default: False (check only the types)

Returns
-------
dtype : dtype
    Suitable index data type (int32 or int64)
*)

val get_sum_dtype : Py.Object.t -> Py.Object.t
(**
Mimic numpy's casting for np.sum
*)

val getdtype : ?a:Py.Object.t -> ?default:Py.Object.t -> dtype:Py.Object.t -> unit -> Py.Object.t
(**
Function used to simplify argument processing.  If 'dtype' is not
specified (is None), returns a.dtype; otherwise returns a np.dtype
object created from the specified dtype argument.  If 'dtype' and 'a'
are both None, construct a data type out of the 'default' parameter.
Furthermore, 'dtype' must be in 'allowed' set.
*)

val is_pydata_spmatrix : Py.Object.t -> Py.Object.t
(**
Check whether object is pydata/sparse matrix, avoiding importing the module.
*)

val isdense : Py.Object.t -> Py.Object.t
(**
None
*)

val isintlike : Py.Object.t -> Py.Object.t
(**
Is x appropriate as an index into a sparse matrix? Returns True
if it can be cast safely to a machine int.
*)

val ismatrix : Py.Object.t -> Py.Object.t
(**
None
*)

val isscalarlike : Py.Object.t -> Py.Object.t
(**
Is x either a scalar, an array scalar, or a 0-dim array?
*)

val issequence : Py.Object.t -> Py.Object.t
(**
None
*)

val isshape : ?nonneg:Py.Object.t -> x:Py.Object.t -> unit -> Py.Object.t
(**
Is x a valid 2-tuple of dimensions?

If nonneg, also checks that the dimensions are non-negative.
*)

val matrix : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> Py.Object.t
(**
None
*)

val to_native : Py.Object.t -> Py.Object.t
(**
None
*)

val upcast : Py.Object.t list -> Py.Object.t
(**
Returns the nearest supported sparse dtype for the
combination of one or more types.

upcast(t0, t1, ..., tn) -> T  where T is a supported dtype

Examples
--------

>>> upcast('int32')
<type 'numpy.int32'>
>>> upcast('bool')
<type 'numpy.bool_'>
>>> upcast('int32','float32')
<type 'numpy.float64'>
>>> upcast('bool',complex,float)
<type 'numpy.complex128'>
*)

val upcast_char : Py.Object.t list -> Py.Object.t
(**
Same as `upcast` but taking dtype.char as input (faster).
*)

val upcast_scalar : dtype:Py.Object.t -> scalar:Py.Object.t -> unit -> Py.Object.t
(**
Determine data type for binary operation between an array of
type `dtype` and a scalar.
*)

val validateaxis : Py.Object.t -> Py.Object.t
(**
None
*)


end

val block_diag : ?format:string -> ?dtype:Py.Object.t -> mats:Py.Object.t -> unit -> [`ArrayLike|`Object|`Spmatrix] Np.Obj.t
(**
Build a block diagonal sparse matrix from provided matrices.

Parameters
----------
mats : sequence of matrices
    Input matrices.
format : str, optional
    The sparse format of the result (e.g. 'csr').  If not given, the matrix
    is returned in 'coo' format.
dtype : dtype specifier, optional
    The data-type of the output matrix.  If not given, the dtype is
    determined from that of `blocks`.

Returns
-------
res : sparse matrix

Notes
-----

.. versionadded:: 0.11.0

See Also
--------
bmat, diags

Examples
--------
>>> from scipy.sparse import coo_matrix, block_diag
>>> A = coo_matrix([[1, 2], [3, 4]])
>>> B = coo_matrix([[5], [6]])
>>> C = coo_matrix([[7]])
>>> block_diag((A, B, C)).toarray()
array([[1, 2, 0, 0],
       [3, 4, 0, 0],
       [0, 0, 5, 0],
       [0, 0, 6, 0],
       [0, 0, 0, 7]])
*)

val bmat : ?format:[`Lil | `Bsr | `Csr | `Csc | `Coo | `Dia | `Dok] -> ?dtype:Np.Dtype.t -> blocks:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Object|`Spmatrix] Np.Obj.t
(**
Build a sparse matrix from sparse sub-blocks

Parameters
----------
blocks : array_like
    Grid of sparse matrices with compatible shapes.
    An entry of None implies an all-zero matrix.
format : {'bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil'}, optional
    The sparse format of the result (e.g. 'csr').  By default an
    appropriate sparse matrix format is returned.
    This choice is subject to change.
dtype : dtype, optional
    The data-type of the output matrix.  If not given, the dtype is
    determined from that of `blocks`.

Returns
-------
bmat : sparse matrix

See Also
--------
block_diag, diags

Examples
--------
>>> from scipy.sparse import coo_matrix, bmat
>>> A = coo_matrix([[1, 2], [3, 4]])
>>> B = coo_matrix([[5], [6]])
>>> C = coo_matrix([[7]])
>>> bmat([[A, B], [None, C]]).toarray()
array([[1, 2, 5],
       [3, 4, 6],
       [0, 0, 7]])

>>> bmat([[A, None], [None, C]]).toarray()
array([[1, 2, 0],
       [3, 4, 0],
       [0, 0, 7]])
*)

val diags : ?offsets:Py.Object.t -> ?shape:Py.Object.t -> ?format:[`Lil | `Csr | `Csc | `Dia | `T of Py.Object.t] -> ?dtype:Np.Dtype.t -> diagonals:Py.Object.t -> unit -> Py.Object.t
(**
Construct a sparse matrix from diagonals.

Parameters
----------
diagonals : sequence of array_like
    Sequence of arrays containing the matrix diagonals,
    corresponding to `offsets`.
offsets : sequence of int or an int, optional
    Diagonals to set:
      - k = 0  the main diagonal (default)
      - k > 0  the k-th upper diagonal
      - k < 0  the k-th lower diagonal
shape : tuple of int, optional
    Shape of the result. If omitted, a square matrix large enough
    to contain the diagonals is returned.
format : {'dia', 'csr', 'csc', 'lil', ...}, optional
    Matrix format of the result.  By default (format=None) an
    appropriate sparse matrix format is returned.  This choice is
    subject to change.
dtype : dtype, optional
    Data type of the matrix.

See Also
--------
spdiags : construct matrix from diagonals

Notes
-----
This function differs from `spdiags` in the way it handles
off-diagonals.

The result from `diags` is the sparse equivalent of::

    np.diag(diagonals[0], offsets[0])
    + ...
    + np.diag(diagonals[k], offsets[k])

Repeated diagonal offsets are disallowed.

.. versionadded:: 0.11

Examples
--------
>>> from scipy.sparse import diags
>>> diagonals = [[1, 2, 3, 4], [1, 2, 3], [1, 2]]
>>> diags(diagonals, [0, -1, 2]).toarray()
array([[1, 0, 1, 0],
       [1, 2, 0, 2],
       [0, 2, 3, 0],
       [0, 0, 3, 4]])

Broadcasting of scalars is supported (but shape needs to be
specified):

>>> diags([1, -2, 1], [-1, 0, 1], shape=(4, 4)).toarray()
array([[-2.,  1.,  0.,  0.],
       [ 1., -2.,  1.,  0.],
       [ 0.,  1., -2.,  1.],
       [ 0.,  0.,  1., -2.]])


If only one diagonal is wanted (as in `numpy.diag`), the following
works as well:

>>> diags([1, 2, 3], 1).toarray()
array([[ 0.,  1.,  0.,  0.],
       [ 0.,  0.,  2.,  0.],
       [ 0.,  0.,  0.,  3.],
       [ 0.,  0.,  0.,  0.]])
*)

val eye : ?n:int -> ?k:int -> ?dtype:Np.Dtype.t -> ?format:string -> m:int -> unit -> Py.Object.t
(**
Sparse matrix with ones on diagonal

Returns a sparse (m x n) matrix where the k-th diagonal
is all ones and everything else is zeros.

Parameters
----------
m : int
    Number of rows in the matrix.
n : int, optional
    Number of columns. Default: `m`.
k : int, optional
    Diagonal to place ones on. Default: 0 (main diagonal).
dtype : dtype, optional
    Data type of the matrix.
format : str, optional
    Sparse format of the result, e.g. format='csr', etc.

Examples
--------
>>> from scipy import sparse
>>> sparse.eye(3).toarray()
array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]])
>>> sparse.eye(3, dtype=np.int8)
<3x3 sparse matrix of type '<class 'numpy.int8'>'
    with 3 stored elements (1 diagonals) in DIAgonal format>
*)

val find : [`Dense of Py.Object.t | `Spmatrix of [>`Spmatrix] Np.Obj.t] -> Py.Object.t
(**
Return the indices and values of the nonzero elements of a matrix

Parameters
----------
A : dense or sparse matrix
    Matrix whose nonzero elements are desired.

Returns
-------
(I,J,V) : tuple of arrays
    I,J, and V contain the row indices, column indices, and values
    of the nonzero matrix entries.


Examples
--------
>>> from scipy.sparse import csr_matrix, find
>>> A = csr_matrix([[7.0, 8.0, 0],[0, 0, 9.0]])
>>> find(A)
(array([0, 0, 1], dtype=int32), array([0, 1, 2], dtype=int32), array([ 7.,  8.,  9.]))
*)

val hstack : ?format:string -> ?dtype:Np.Dtype.t -> blocks:Py.Object.t -> unit -> Py.Object.t
(**
Stack sparse matrices horizontally (column wise)

Parameters
----------
blocks
    sequence of sparse matrices with compatible shapes
format : str
    sparse format of the result (e.g. 'csr')
    by default an appropriate sparse matrix format is returned.
    This choice is subject to change.
dtype : dtype, optional
    The data-type of the output matrix.  If not given, the dtype is
    determined from that of `blocks`.

See Also
--------
vstack : stack sparse matrices vertically (row wise)

Examples
--------
>>> from scipy.sparse import coo_matrix, hstack
>>> A = coo_matrix([[1, 2], [3, 4]])
>>> B = coo_matrix([[5], [6]])
>>> hstack([A,B]).toarray()
array([[1, 2, 5],
       [3, 4, 6]])
*)

val identity : ?dtype:Np.Dtype.t -> ?format:string -> n:int -> unit -> Py.Object.t
(**
Identity matrix in sparse format

Returns an identity matrix with shape (n,n) using a given
sparse format and dtype.

Parameters
----------
n : int
    Shape of the identity matrix.
dtype : dtype, optional
    Data type of the matrix
format : str, optional
    Sparse format of the result, e.g. format='csr', etc.

Examples
--------
>>> from scipy.sparse import identity
>>> identity(3).toarray()
array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]])
>>> identity(3, dtype='int8', format='dia')
<3x3 sparse matrix of type '<class 'numpy.int8'>'
        with 3 stored elements (1 diagonals) in DIAgonal format>
*)

val issparse : Py.Object.t -> Py.Object.t
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

val isspmatrix : Py.Object.t -> Py.Object.t
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

val isspmatrix_bsr : Py.Object.t -> Py.Object.t
(**
Is x of a bsr_matrix type?

Parameters
----------
x
    object to check for being a bsr matrix

Returns
-------
bool
    True if x is a bsr matrix, False otherwise

Examples
--------
>>> from scipy.sparse import bsr_matrix, isspmatrix_bsr
>>> isspmatrix_bsr(bsr_matrix([[5]]))
True

>>> from scipy.sparse import bsr_matrix, csr_matrix, isspmatrix_bsr
>>> isspmatrix_bsr(csr_matrix([[5]]))
False
*)

val isspmatrix_coo : Py.Object.t -> Py.Object.t
(**
Is x of coo_matrix type?

Parameters
----------
x
    object to check for being a coo matrix

Returns
-------
bool
    True if x is a coo matrix, False otherwise

Examples
--------
>>> from scipy.sparse import coo_matrix, isspmatrix_coo
>>> isspmatrix_coo(coo_matrix([[5]]))
True

>>> from scipy.sparse import coo_matrix, csr_matrix, isspmatrix_coo
>>> isspmatrix_coo(csr_matrix([[5]]))
False
*)

val isspmatrix_csc : Py.Object.t -> Py.Object.t
(**
Is x of csc_matrix type?

Parameters
----------
x
    object to check for being a csc matrix

Returns
-------
bool
    True if x is a csc matrix, False otherwise

Examples
--------
>>> from scipy.sparse import csc_matrix, isspmatrix_csc
>>> isspmatrix_csc(csc_matrix([[5]]))
True

>>> from scipy.sparse import csc_matrix, csr_matrix, isspmatrix_csc
>>> isspmatrix_csc(csr_matrix([[5]]))
False
*)

val isspmatrix_csr : Py.Object.t -> Py.Object.t
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

val isspmatrix_dia : Py.Object.t -> Py.Object.t
(**
Is x of dia_matrix type?

Parameters
----------
x
    object to check for being a dia matrix

Returns
-------
bool
    True if x is a dia matrix, False otherwise

Examples
--------
>>> from scipy.sparse import dia_matrix, isspmatrix_dia
>>> isspmatrix_dia(dia_matrix([[5]]))
True

>>> from scipy.sparse import dia_matrix, csr_matrix, isspmatrix_dia
>>> isspmatrix_dia(csr_matrix([[5]]))
False
*)

val isspmatrix_dok : Py.Object.t -> Py.Object.t
(**
Is x of dok_matrix type?

Parameters
----------
x
    object to check for being a dok matrix

Returns
-------
bool
    True if x is a dok matrix, False otherwise

Examples
--------
>>> from scipy.sparse import dok_matrix, isspmatrix_dok
>>> isspmatrix_dok(dok_matrix([[5]]))
True

>>> from scipy.sparse import dok_matrix, csr_matrix, isspmatrix_dok
>>> isspmatrix_dok(csr_matrix([[5]]))
False
*)

val isspmatrix_lil : Py.Object.t -> Py.Object.t
(**
Is x of lil_matrix type?

Parameters
----------
x
    object to check for being a lil matrix

Returns
-------
bool
    True if x is a lil matrix, False otherwise

Examples
--------
>>> from scipy.sparse import lil_matrix, isspmatrix_lil
>>> isspmatrix_lil(lil_matrix([[5]]))
True

>>> from scipy.sparse import lil_matrix, csr_matrix, isspmatrix_lil
>>> isspmatrix_lil(csr_matrix([[5]]))
False
*)

val kron : ?format:string -> a:Py.Object.t -> b:Py.Object.t -> unit -> Py.Object.t
(**
kronecker product of sparse matrices A and B

Parameters
----------
A : sparse or dense matrix
    first matrix of the product
B : sparse or dense matrix
    second matrix of the product
format : str, optional
    format of the result (e.g. 'csr')

Returns
-------
kronecker product in a sparse matrix format


Examples
--------
>>> from scipy import sparse
>>> A = sparse.csr_matrix(np.array([[0, 2], [5, 0]]))
>>> B = sparse.csr_matrix(np.array([[1, 2], [3, 4]]))
>>> sparse.kron(A, B).toarray()
array([[ 0,  0,  2,  4],
       [ 0,  0,  6,  8],
       [ 5, 10,  0,  0],
       [15, 20,  0,  0]])

>>> sparse.kron(A, [[1, 2], [3, 4]]).toarray()
array([[ 0,  0,  2,  4],
       [ 0,  0,  6,  8],
       [ 5, 10,  0,  0],
       [15, 20,  0,  0]])
*)

val kronsum : ?format:string -> a:Py.Object.t -> b:Py.Object.t -> unit -> Py.Object.t
(**
kronecker sum of sparse matrices A and B

Kronecker sum of two sparse matrices is a sum of two Kronecker
products kron(I_n,A) + kron(B,I_m) where A has shape (m,m)
and B has shape (n,n) and I_m and I_n are identity matrices
of shape (m,m) and (n,n) respectively.

Parameters
----------
A
    square matrix
B
    square matrix
format : str
    format of the result (e.g. 'csr')

Returns
-------
kronecker sum in a sparse matrix format

Examples
--------
*)

val load_npz : [`File_like_object of Py.Object.t | `S of string] -> Py.Object.t
(**
Load a sparse matrix from a file using ``.npz`` format.

Parameters
----------
file : str or file-like object
    Either the file name (string) or an open file (file-like object)
    where the data will be loaded.

Returns
-------
result : csc_matrix, csr_matrix, bsr_matrix, dia_matrix or coo_matrix
    A sparse matrix containing the loaded data.

Raises
------
IOError
    If the input file does not exist or cannot be read.

See Also
--------
scipy.sparse.save_npz: Save a sparse matrix to a file using ``.npz`` format.
numpy.load: Load several arrays from a ``.npz`` archive.

Examples
--------
Store sparse matrix to disk, and load it again:

>>> import scipy.sparse
>>> sparse_matrix = scipy.sparse.csc_matrix(np.array([[0, 0, 3], [4, 0, 0]]))
>>> sparse_matrix
<2x3 sparse matrix of type '<class 'numpy.int64'>'
   with 2 stored elements in Compressed Sparse Column format>
>>> sparse_matrix.todense()
matrix([[0, 0, 3],
        [4, 0, 0]], dtype=int64)

>>> scipy.sparse.save_npz('/tmp/sparse_matrix.npz', sparse_matrix)
>>> sparse_matrix = scipy.sparse.load_npz('/tmp/sparse_matrix.npz')

>>> sparse_matrix
<2x3 sparse matrix of type '<class 'numpy.int64'>'
    with 2 stored elements in Compressed Sparse Column format>
>>> sparse_matrix.todense()
matrix([[0, 0, 3],
        [4, 0, 0]], dtype=int64)
*)

val rand : ?density:Py.Object.t -> ?format:string -> ?dtype:Np.Dtype.t -> ?random_state:[`I of int | `Numpy_random_RandomState of Py.Object.t] -> m:Py.Object.t -> n:Py.Object.t -> unit -> [`ArrayLike|`Object|`Spmatrix] Np.Obj.t
(**
Generate a sparse matrix of the given shape and density with uniformly
distributed values.

Parameters
----------
m, n : int
    shape of the matrix
density : real, optional
    density of the generated matrix: density equal to one means a full
    matrix, density of 0 means a matrix with no non-zero items.
format : str, optional
    sparse matrix format.
dtype : dtype, optional
    type of the returned matrix values.
random_state : {numpy.random.RandomState, int}, optional
    Random number generator or random seed. If not given, the singleton
    numpy.random will be used.

Returns
-------
res : sparse matrix

Notes
-----
Only float types are supported for now.

See Also
--------
scipy.sparse.random : Similar function that allows a user-specified random
    data source.

Examples
--------
>>> from scipy.sparse import rand
>>> matrix = rand(3, 4, density=0.25, format='csr', random_state=42)
>>> matrix
<3x4 sparse matrix of type '<class 'numpy.float64'>'
   with 3 stored elements in Compressed Sparse Row format>
>>> matrix.todense()
matrix([[0.05641158, 0.        , 0.        , 0.65088847],
        [0.        , 0.        , 0.        , 0.14286682],
        [0.        , 0.        , 0.        , 0.        ]])
*)

val random : ?density:Py.Object.t -> ?format:string -> ?dtype:Np.Dtype.t -> ?random_state:[`I of int | `Numpy_random_RandomState of Py.Object.t] -> ?data_rvs:Py.Object.t -> m:Py.Object.t -> n:Py.Object.t -> unit -> [`ArrayLike|`Object|`Spmatrix] Np.Obj.t
(**
Generate a sparse matrix of the given shape and density with randomly
distributed values.

Parameters
----------
m, n : int
    shape of the matrix
density : real, optional
    density of the generated matrix: density equal to one means a full
    matrix, density of 0 means a matrix with no non-zero items.
format : str, optional
    sparse matrix format.
dtype : dtype, optional
    type of the returned matrix values.
random_state : {numpy.random.RandomState, int}, optional
    Random number generator or random seed. If not given, the singleton
    numpy.random will be used.  This random state will be used
    for sampling the sparsity structure, but not necessarily for sampling
    the values of the structurally nonzero entries of the matrix.
data_rvs : callable, optional
    Samples a requested number of random values.
    This function should take a single argument specifying the length
    of the ndarray that it will return.  The structurally nonzero entries
    of the sparse random matrix will be taken from the array sampled
    by this function.  By default, uniform [0, 1) random values will be
    sampled using the same random state as is used for sampling
    the sparsity structure.

Returns
-------
res : sparse matrix

Notes
-----
Only float types are supported for now.

Examples
--------
>>> from scipy.sparse import random
>>> from scipy import stats

>>> class CustomRandomState(np.random.RandomState):
...     def randint(self, k):
...         i = np.random.randint(k)
...         return i - i % 2
>>> np.random.seed(12345)
>>> rs = CustomRandomState()
>>> rvs = stats.poisson(25, loc=10).rvs
>>> S = random(3, 4, density=0.25, random_state=rs, data_rvs=rvs)
>>> S.A
array([[ 36.,   0.,  33.,   0.],   # random
       [  0.,   0.,   0.,   0.],
       [  0.,   0.,  36.,   0.]])

>>> from scipy.sparse import random
>>> from scipy.stats import rv_continuous
>>> class CustomDistribution(rv_continuous):
...     def _rvs(self, *args, **kwargs):
...         return self._random_state.randn( *self._size)
>>> X = CustomDistribution(seed=2906)
>>> Y = X()  # get a frozen version of the distribution
>>> S = random(3, 4, density=0.25, random_state=2906, data_rvs=Y.rvs)
>>> S.A
array([[ 0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.13569738,  1.9467163 , -0.81205367,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ]])
*)

val save_npz : ?compressed:bool -> file:[`File_like_object of Py.Object.t | `S of string] -> matrix:Py.Object.t -> unit -> Py.Object.t
(**
Save a sparse matrix to a file using ``.npz`` format.

Parameters
----------
file : str or file-like object
    Either the file name (string) or an open file (file-like object)
    where the data will be saved. If file is a string, the ``.npz``
    extension will be appended to the file name if it is not already
    there.
matrix: spmatrix (format: ``csc``, ``csr``, ``bsr``, ``dia`` or coo``)
    The sparse matrix to save.
compressed : bool, optional
    Allow compressing the file. Default: True

See Also
--------
scipy.sparse.load_npz: Load a sparse matrix from a file using ``.npz`` format.
numpy.savez: Save several arrays into a ``.npz`` archive.
numpy.savez_compressed : Save several arrays into a compressed ``.npz`` archive.

Examples
--------
Store sparse matrix to disk, and load it again:

>>> import scipy.sparse
>>> sparse_matrix = scipy.sparse.csc_matrix(np.array([[0, 0, 3], [4, 0, 0]]))
>>> sparse_matrix
<2x3 sparse matrix of type '<class 'numpy.int64'>'
   with 2 stored elements in Compressed Sparse Column format>
>>> sparse_matrix.todense()
matrix([[0, 0, 3],
        [4, 0, 0]], dtype=int64)

>>> scipy.sparse.save_npz('/tmp/sparse_matrix.npz', sparse_matrix)
>>> sparse_matrix = scipy.sparse.load_npz('/tmp/sparse_matrix.npz')

>>> sparse_matrix
<2x3 sparse matrix of type '<class 'numpy.int64'>'
   with 2 stored elements in Compressed Sparse Column format>
>>> sparse_matrix.todense()
matrix([[0, 0, 3],
        [4, 0, 0]], dtype=int64)
*)

val spdiags : ?format:string -> data:[>`Ndarray] Np.Obj.t -> diags:Py.Object.t -> m:Py.Object.t -> n:Py.Object.t -> unit -> Py.Object.t
(**
Return a sparse matrix from diagonals.

Parameters
----------
data : array_like
    matrix diagonals stored row-wise
diags : diagonals to set
    - k = 0  the main diagonal
    - k > 0  the k-th upper diagonal
    - k < 0  the k-th lower diagonal
m, n : int
    shape of the result
format : str, optional
    Format of the result. By default (format=None) an appropriate sparse
    matrix format is returned.  This choice is subject to change.

See Also
--------
diags : more convenient form of this function
dia_matrix : the sparse DIAgonal format.

Examples
--------
>>> from scipy.sparse import spdiags
>>> data = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
>>> diags = np.array([0, -1, 2])
>>> spdiags(data, diags, 4, 4).toarray()
array([[1, 0, 3, 0],
       [1, 2, 0, 4],
       [0, 2, 3, 0],
       [0, 0, 3, 4]])
*)

val tril : ?k:Py.Object.t -> ?format:string -> a:[`Dense of Py.Object.t | `Spmatrix of [>`Spmatrix] Np.Obj.t] -> unit -> [`ArrayLike|`Object|`Spmatrix] Np.Obj.t
(**
Return the lower triangular portion of a matrix in sparse format

Returns the elements on or below the k-th diagonal of the matrix A.
    - k = 0 corresponds to the main diagonal
    - k > 0 is above the main diagonal
    - k < 0 is below the main diagonal

Parameters
----------
A : dense or sparse matrix
    Matrix whose lower trianglar portion is desired.
k : integer : optional
    The top-most diagonal of the lower triangle.
format : string
    Sparse format of the result, e.g. format='csr', etc.

Returns
-------
L : sparse matrix
    Lower triangular portion of A in sparse format.

See Also
--------
triu : upper triangle in sparse format

Examples
--------
>>> from scipy.sparse import csr_matrix, tril
>>> A = csr_matrix([[1, 2, 0, 0, 3], [4, 5, 0, 6, 7], [0, 0, 8, 9, 0]],
...                dtype='int32')
>>> A.toarray()
array([[1, 2, 0, 0, 3],
       [4, 5, 0, 6, 7],
       [0, 0, 8, 9, 0]])
>>> tril(A).toarray()
array([[1, 0, 0, 0, 0],
       [4, 5, 0, 0, 0],
       [0, 0, 8, 0, 0]])
>>> tril(A).nnz
4
>>> tril(A, k=1).toarray()
array([[1, 2, 0, 0, 0],
       [4, 5, 0, 0, 0],
       [0, 0, 8, 9, 0]])
>>> tril(A, k=-1).toarray()
array([[0, 0, 0, 0, 0],
       [4, 0, 0, 0, 0],
       [0, 0, 0, 0, 0]])
>>> tril(A, format='csc')
<3x5 sparse matrix of type '<class 'numpy.int32'>'
        with 4 stored elements in Compressed Sparse Column format>
*)

val triu : ?k:Py.Object.t -> ?format:string -> a:[`Dense of Py.Object.t | `Spmatrix of [>`Spmatrix] Np.Obj.t] -> unit -> [`ArrayLike|`Object|`Spmatrix] Np.Obj.t
(**
Return the upper triangular portion of a matrix in sparse format

Returns the elements on or above the k-th diagonal of the matrix A.
    - k = 0 corresponds to the main diagonal
    - k > 0 is above the main diagonal
    - k < 0 is below the main diagonal

Parameters
----------
A : dense or sparse matrix
    Matrix whose upper trianglar portion is desired.
k : integer : optional
    The bottom-most diagonal of the upper triangle.
format : string
    Sparse format of the result, e.g. format='csr', etc.

Returns
-------
L : sparse matrix
    Upper triangular portion of A in sparse format.

See Also
--------
tril : lower triangle in sparse format

Examples
--------
>>> from scipy.sparse import csr_matrix, triu
>>> A = csr_matrix([[1, 2, 0, 0, 3], [4, 5, 0, 6, 7], [0, 0, 8, 9, 0]],
...                dtype='int32')
>>> A.toarray()
array([[1, 2, 0, 0, 3],
       [4, 5, 0, 6, 7],
       [0, 0, 8, 9, 0]])
>>> triu(A).toarray()
array([[1, 2, 0, 0, 3],
       [0, 5, 0, 6, 7],
       [0, 0, 8, 9, 0]])
>>> triu(A).nnz
8
>>> triu(A, k=1).toarray()
array([[0, 2, 0, 0, 3],
       [0, 0, 0, 6, 7],
       [0, 0, 0, 9, 0]])
>>> triu(A, k=-1).toarray()
array([[1, 2, 0, 0, 3],
       [4, 5, 0, 6, 7],
       [0, 0, 8, 9, 0]])
>>> triu(A, format='csc')
<3x5 sparse matrix of type '<class 'numpy.int32'>'
        with 8 stored elements in Compressed Sparse Column format>
*)

val vstack : ?format:string -> ?dtype:Np.Dtype.t -> blocks:Py.Object.t -> unit -> Py.Object.t
(**
Stack sparse matrices vertically (row wise)

Parameters
----------
blocks
    sequence of sparse matrices with compatible shapes
format : str, optional
    sparse format of the result (e.g. 'csr')
    by default an appropriate sparse matrix format is returned.
    This choice is subject to change.
dtype : dtype, optional
    The data-type of the output matrix.  If not given, the dtype is
    determined from that of `blocks`.

See Also
--------
hstack : stack sparse matrices horizontally (column wise)

Examples
--------
>>> from scipy.sparse import coo_matrix, vstack
>>> A = coo_matrix([[1, 2], [3, 4]])
>>> B = coo_matrix([[5, 6]])
>>> vstack([A, B]).toarray()
array([[1, 2],
       [3, 4],
       [5, 6]])
*)

