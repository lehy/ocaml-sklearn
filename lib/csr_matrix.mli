type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?shape:int list -> ?dtype:Py.Object.t -> ?copy:Py.Object.t -> arg1:Py.Object.t -> unit -> t
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

>>> docs = [["hello", "world", "hello"], ["goodbye", "cruel", "world"]]
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

val get_item : key:Py.Object.t -> t -> Py.Object.t
(**
None
*)

val arcsin : t -> Py.Object.t
(**
Element-wise arcsin.

See numpy.arcsin for more information.
*)

val arcsinh : t -> Py.Object.t
(**
Element-wise arcsinh.

See numpy.arcsinh for more information.
*)

val arctan : t -> Py.Object.t
(**
Element-wise arctan.

See numpy.arctan for more information.
*)

val arctanh : t -> Py.Object.t
(**
Element-wise arctanh.

See numpy.arctanh for more information.
*)

val argmax : ?axis:[`None | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> t -> Py.Object.t
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

val argmin : ?axis:[`None | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> t -> Py.Object.t
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

val ceil : t -> Py.Object.t
(**
Element-wise ceil.

See numpy.ceil for more information.
*)

val check_format : ?full_check:bool -> t -> Py.Object.t
(**
check whether the matrix format is valid

Parameters
----------
full_check : bool, optional
    If `True`, rigorous check, O(N) operations. Otherwise
    basic check, O(1) operations (default True).
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

val deg2rad : t -> Py.Object.t
(**
Element-wise deg2rad.

See numpy.deg2rad for more information.
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

val eliminate_zeros : t -> Py.Object.t
(**
Remove zero entries from the matrix

This is an *in place* operation
*)

val expm1 : t -> Py.Object.t
(**
Element-wise expm1.

See numpy.expm1 for more information.
*)

val floor : t -> Py.Object.t
(**
Element-wise floor.

See numpy.floor for more information.
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

val getcol : i:Py.Object.t -> t -> Py.Object.t
(**
Returns a copy of column i of the matrix, as a (m x 1)
CSR matrix (column vector).
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
Returns a copy of row i of the matrix, as a (1 x n)
CSR matrix (row vector).
*)

val log1p : t -> Py.Object.t
(**
Element-wise log1p.

See numpy.log1p for more information.
*)

val max : ?axis:[`None | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> t -> Py.Object.t
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

val min : ?axis:[`None | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> t -> Py.Object.t
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

val minimum : other:Py.Object.t -> t -> Py.Object.t
(**
Element-wise minimum between this and another matrix.
*)

val multiply : other:Py.Object.t -> t -> Py.Object.t
(**
Point-wise multiplication by another matrix, vector, or
scalar.
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
This function performs element-wise power.

Parameters
----------
n : n is a scalar

dtype : If dtype is not specified, the current dtype will be preserved.
*)

val prune : t -> Py.Object.t
(**
Remove empty space after all non-zero elements.
        
*)

val rad2deg : t -> Py.Object.t
(**
Element-wise rad2deg.

See numpy.rad2deg for more information.
*)

val reshape : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> t -> t
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

val rint : t -> Py.Object.t
(**
Element-wise rint.

See numpy.rint for more information.
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

val sign : t -> Py.Object.t
(**
Element-wise sign.

See numpy.sign for more information.
*)

val sin : t -> Py.Object.t
(**
Element-wise sin.

See numpy.sin for more information.
*)

val sinh : t -> Py.Object.t
(**
Element-wise sinh.

See numpy.sinh for more information.
*)

val sort_indices : t -> Py.Object.t
(**
Sort the indices of this matrix *in place*
        
*)

val sorted_indices : t -> Py.Object.t
(**
Return a copy of this matrix with sorted indices
        
*)

val sqrt : t -> Py.Object.t
(**
Element-wise sqrt.

See numpy.sqrt for more information.
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

val sum_duplicates : t -> Py.Object.t
(**
Eliminate duplicate matrix entries by adding them together

The is an *in place* operation
*)

val tan : t -> Py.Object.t
(**
Element-wise tan.

See numpy.tan for more information.
*)

val tanh : t -> Py.Object.t
(**
Element-wise tanh.

See numpy.tanh for more information.
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

val trunc : t -> Py.Object.t
(**
Element-wise trunc.

See numpy.trunc for more information.
*)


(** Attribute dtype: see constructor for documentation *)
val dtype : t -> Py.Object.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]

