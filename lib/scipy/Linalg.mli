(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module LinAlgError : sig
type tag = [`LinAlgError]
type t = [`BaseException | `LinAlgError | `Object] Obj.t
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

module LinAlgWarning : sig
type tag = [`LinAlgWarning]
type t = [`BaseException | `LinAlgWarning | `Object] Obj.t
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

module Basic : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val atleast_1d : Py.Object.t list -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Convert inputs to arrays with at least one dimension.

Scalar inputs are converted to 1-dimensional arrays, whilst
higher-dimensional inputs are preserved.

Parameters
----------
arys1, arys2, ... : array_like
    One or more input arrays.

Returns
-------
ret : ndarray
    An array, or list of arrays, each with ``a.ndim >= 1``.
    Copies are made only if necessary.

See Also
--------
atleast_2d, atleast_3d

Examples
--------
>>> np.atleast_1d(1.0)
array([1.])

>>> x = np.arange(9.0).reshape(3,3)
>>> np.atleast_1d(x)
array([[0., 1., 2.],
       [3., 4., 5.],
       [6., 7., 8.]])
>>> np.atleast_1d(x) is x
True

>>> np.atleast_1d(1, [3, 4])
[array([1]), array([3, 4])]
*)

val atleast_2d : Py.Object.t list -> Py.Object.t
(**
View inputs as arrays with at least two dimensions.

Parameters
----------
arys1, arys2, ... : array_like
    One or more array-like sequences.  Non-array inputs are converted
    to arrays.  Arrays that already have two or more dimensions are
    preserved.

Returns
-------
res, res2, ... : ndarray
    An array, or list of arrays, each with ``a.ndim >= 2``.
    Copies are avoided where possible, and views with two or more
    dimensions are returned.

See Also
--------
atleast_1d, atleast_3d

Examples
--------
>>> np.atleast_2d(3.0)
array([[3.]])

>>> x = np.arange(3.0)
>>> np.atleast_2d(x)
array([[0., 1., 2.]])
>>> np.atleast_2d(x).base is x
True

>>> np.atleast_2d(1, [1, 2], [[1, 2]])
[array([[1]]), array([[1, 2]]), array([[1, 2]])]
*)

val det : ?overwrite_a:bool -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Compute the determinant of a matrix

The determinant of a square matrix is a value derived arithmetically
from the coefficients of the matrix.

The determinant for a 3x3 matrix, for example, is computed as follows::

    a    b    c
    d    e    f = A
    g    h    i

    det(A) = a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h

Parameters
----------
a : (M, M) array_like
    A square matrix.
overwrite_a : bool, optional
    Allow overwriting data in a (may enhance performance).
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
det : float or complex
    Determinant of `a`.

Notes
-----
The determinant is computed via LU factorization, LAPACK routine z/dgetrf.

Examples
--------
>>> from scipy import linalg
>>> a = np.array([[1,2,3], [4,5,6], [7,8,9]])
>>> linalg.det(a)
0.0
>>> a = np.array([[0,2,3], [4,5,6], [7,8,9]])
>>> linalg.det(a)
3.0
*)

val get_flinalg_funcs : ?arrays:Py.Object.t -> ?debug:Py.Object.t -> names:Py.Object.t -> unit -> Py.Object.t
(**
Return optimal available _flinalg function objects with
names. arrays are used to determine optimal prefix.
*)

val get_lapack_funcs : ?arrays:[>`Ndarray] Np.Obj.t list -> ?dtype:[`S of string | `Dtype of Np.Dtype.t] -> names:[`Sequence_of_str of Py.Object.t | `S of string] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return available LAPACK function objects from names.

Arrays are used to determine the optimal prefix of LAPACK routines.

Parameters
----------
names : str or sequence of str
    Name(s) of LAPACK functions without type prefix.

arrays : sequence of ndarrays, optional
    Arrays can be given to determine optimal prefix of LAPACK
    routines. If not given, double-precision routines will be
    used, otherwise the most generic type in arrays will be used.

dtype : str or dtype, optional
    Data-type specifier. Not used if `arrays` is non-empty.

Returns
-------
funcs : list
    List containing the found function(s).

Notes
-----
This routine automatically chooses between Fortran/C
interfaces. Fortran code is used whenever possible for arrays with
column major order. In all other cases, C code is preferred.

In LAPACK, the naming convention is that all functions start with a
type prefix, which depends on the type of the principal
matrix. These can be one of {'s', 'd', 'c', 'z'} for the numpy
types {float32, float64, complex64, complex128} respectively, and
are stored in attribute ``typecode`` of the returned functions.

Examples
--------
Suppose we would like to use '?lange' routine which computes the selected
norm of an array. We pass our array in order to get the correct 'lange'
flavor.

>>> import scipy.linalg as LA
>>> a = np.random.rand(3,2)
>>> x_lange = LA.get_lapack_funcs('lange', (a,))
>>> x_lange.typecode
'd'
>>> x_lange = LA.get_lapack_funcs('lange',(a*1j,))
>>> x_lange.typecode
'z'

Several LAPACK routines work best when its internal WORK array has
the optimal size (big enough for fast computation and small enough to
avoid waste of memory). This size is determined also by a dedicated query
to the function which is often wrapped as a standalone function and
commonly denoted as ``###_lwork``. Below is an example for ``?sysv``

>>> import scipy.linalg as LA
>>> a = np.random.rand(1000,1000)
>>> b = np.random.rand(1000,1)*1j
>>> # We pick up zsysv and zsysv_lwork due to b array
... xsysv, xlwork = LA.get_lapack_funcs(('sysv', 'sysv_lwork'), (a, b))
>>> opt_lwork, _ = xlwork(a.shape[0])  # returns a complex for 'z' prefix
>>> udut, ipiv, x, info = xsysv(a, b, lwork=int(opt_lwork.real))
*)

val inv : ?overwrite_a:bool -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the inverse of a matrix.

Parameters
----------
a : array_like
    Square matrix to be inverted.
overwrite_a : bool, optional
    Discard data in `a` (may improve performance). Default is False.
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
ainv : ndarray
    Inverse of the matrix `a`.

Raises
------
LinAlgError
    If `a` is singular.
ValueError
    If `a` is not square, or not 2-dimensional.

Examples
--------
>>> from scipy import linalg
>>> a = np.array([[1., 2.], [3., 4.]])
>>> linalg.inv(a)
array([[-2. ,  1. ],
       [ 1.5, -0.5]])
>>> np.dot(a, linalg.inv(a))
array([[ 1.,  0.],
       [ 0.,  1.]])
*)

val levinson : a:[`Ndarray of [>`Ndarray] Np.Obj.t | `PyObject of Py.Object.t] -> b:[`Ndarray of [>`Ndarray] Np.Obj.t | `PyObject of Py.Object.t] -> unit -> (Py.Object.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Solve a linear Toeplitz system using Levinson recursion.

Parameters
----------
a : array, dtype=double or complex128, shape=(2n-1,)
    The first column of the matrix in reverse order (without the diagonal)
    followed by the first (see below)
b : array, dtype=double  or complex128, shape=(n,)
    The right hand side vector. Both a and b must have the same type
    (double or complex128).

Notes
-----
For example, the 5x5 toeplitz matrix below should be represented as
the linear array ``a`` on the right ::

    [ a0    a1   a2  a3  a4 ]
    [ a-1   a0   a1  a2  a3 ]
    [ a-2  a-1   a0  a1  a2 ] -> [a-4  a-3  a-2  a-1  a0  a1  a2  a3  a4]
    [ a-3  a-2  a-1  a0  a1 ]
    [ a-4  a-3  a-2  a-1 a0 ]

Returns
-------
x : arrray, shape=(n,)
    The solution vector
reflection_coeff : array, shape=(n+1,)
    Toeplitz reflection coefficients. When a is symmetric Toeplitz and
    ``b`` is ``a[n:]``, as in the solution of autoregressive systems,
    then ``reflection_coeff`` also correspond to the partial
    autocorrelation function.
*)

val lstsq : ?cond:float -> ?overwrite_a:bool -> ?overwrite_b:bool -> ?check_finite:bool -> ?lapack_driver:string -> a:[>`Ndarray] Np.Obj.t -> b:Py.Object.t -> unit -> (Py.Object.t * Py.Object.t * int * Py.Object.t option)
(**
Compute least-squares solution to equation Ax = b.

Compute a vector x such that the 2-norm ``|b - A x|`` is minimized.

Parameters
----------
a : (M, N) array_like
    Left hand side array
b : (M,) or (M, K) array_like
    Right hand side array
cond : float, optional
    Cutoff for 'small' singular values; used to determine effective
    rank of a. Singular values smaller than
    ``rcond * largest_singular_value`` are considered zero.
overwrite_a : bool, optional
    Discard data in `a` (may enhance performance). Default is False.
overwrite_b : bool, optional
    Discard data in `b` (may enhance performance). Default is False.
check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.
lapack_driver : str, optional
    Which LAPACK driver is used to solve the least-squares problem.
    Options are ``'gelsd'``, ``'gelsy'``, ``'gelss'``. Default
    (``'gelsd'``) is a good choice.  However, ``'gelsy'`` can be slightly
    faster on many problems.  ``'gelss'`` was used historically.  It is
    generally slow but uses less memory.

    .. versionadded:: 0.17.0

Returns
-------
x : (N,) or (N, K) ndarray
    Least-squares solution.  Return shape matches shape of `b`.
residues : (K,) ndarray or float
    Square of the 2-norm for each column in ``b - a x``, if ``M > N`` and
    ``ndim(A) == n`` (returns a scalar if b is 1-D). Otherwise a
    (0,)-shaped array is returned.
rank : int
    Effective rank of `a`.
s : (min(M, N),) ndarray or None
    Singular values of `a`. The condition number of a is
    ``abs(s[0] / s[-1])``.

Raises
------
LinAlgError
    If computation does not converge.

ValueError
    When parameters are not compatible.

See Also
--------
scipy.optimize.nnls : linear least squares with non-negativity constraint

Notes
-----
When ``'gelsy'`` is used as a driver, `residues` is set to a (0,)-shaped
array and `s` is always ``None``.

Examples
--------
>>> from scipy.linalg import lstsq
>>> import matplotlib.pyplot as plt

Suppose we have the following data:

>>> x = np.array([1, 2.5, 3.5, 4, 5, 7, 8.5])
>>> y = np.array([0.3, 1.1, 1.5, 2.0, 3.2, 6.6, 8.6])

We want to fit a quadratic polynomial of the form ``y = a + b*x**2``
to this data.  We first form the 'design matrix' M, with a constant
column of 1s and a column containing ``x**2``:

>>> M = x[:, np.newaxis]**[0, 2]
>>> M
array([[  1.  ,   1.  ],
       [  1.  ,   6.25],
       [  1.  ,  12.25],
       [  1.  ,  16.  ],
       [  1.  ,  25.  ],
       [  1.  ,  49.  ],
       [  1.  ,  72.25]])

We want to find the least-squares solution to ``M.dot(p) = y``,
where ``p`` is a vector with length 2 that holds the parameters
``a`` and ``b``.

>>> p, res, rnk, s = lstsq(M, y)
>>> p
array([ 0.20925829,  0.12013861])

Plot the data and the fitted curve.

>>> plt.plot(x, y, 'o', label='data')
>>> xx = np.linspace(0, 9, 101)
>>> yy = p[0] + p[1]*xx**2
>>> plt.plot(xx, yy, label='least squares fit, $y = a + bx^2$')
>>> plt.xlabel('x')
>>> plt.ylabel('y')
>>> plt.legend(framealpha=1, shadow=True)
>>> plt.grid(alpha=0.25)
>>> plt.show()
*)

val matrix_balance : ?permute:bool -> ?scale:float -> ?separate:bool -> ?overwrite_a:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Compute a diagonal similarity transformation for row/column balancing.

The balancing tries to equalize the row and column 1-norms by applying
a similarity transformation such that the magnitude variation of the
matrix entries is reflected to the scaling matrices.

Moreover, if enabled, the matrix is first permuted to isolate the upper
triangular parts of the matrix and, again if scaling is also enabled,
only the remaining subblocks are subjected to scaling.

The balanced matrix satisfies the following equality

.. math::

                    B = T^{-1} A T

The scaling coefficients are approximated to the nearest power of 2
to avoid round-off errors.

Parameters
----------
A : (n, n) array_like
    Square data matrix for the balancing.
permute : bool, optional
    The selector to define whether permutation of A is also performed
    prior to scaling.
scale : bool, optional
    The selector to turn on and off the scaling. If False, the matrix
    will not be scaled.
separate : bool, optional
    This switches from returning a full matrix of the transformation
    to a tuple of two separate 1D permutation and scaling arrays.
overwrite_a : bool, optional
    This is passed to xGEBAL directly. Essentially, overwrites the result
    to the data. It might increase the space efficiency. See LAPACK manual
    for details. This is False by default.

Returns
-------
B : (n, n) ndarray
    Balanced matrix
T : (n, n) ndarray
    A possibly permuted diagonal matrix whose nonzero entries are
    integer powers of 2 to avoid numerical truncation errors.
scale, perm : (n,) ndarray
    If ``separate`` keyword is set to True then instead of the array
    ``T`` above, the scaling and the permutation vectors are given
    separately as a tuple without allocating the full array ``T``.

Notes
-----

This algorithm is particularly useful for eigenvalue and matrix
decompositions and in many cases it is already called by various
LAPACK routines.

The algorithm is based on the well-known technique of [1]_ and has
been modified to account for special cases. See [2]_ for details
which have been implemented since LAPACK v3.5.0. Before this version
there are corner cases where balancing can actually worsen the
conditioning. See [3]_ for such examples.

The code is a wrapper around LAPACK's xGEBAL routine family for matrix
balancing.

.. versionadded:: 0.19.0

Examples
--------
>>> from scipy import linalg
>>> x = np.array([[1,2,0], [9,1,0.01], [1,2,10*np.pi]])

>>> y, permscale = linalg.matrix_balance(x)
>>> np.abs(x).sum(axis=0) / np.abs(x).sum(axis=1)
array([ 3.66666667,  0.4995005 ,  0.91312162])

>>> np.abs(y).sum(axis=0) / np.abs(y).sum(axis=1)
array([ 1.2       ,  1.27041742,  0.92658316])  # may vary

>>> permscale  # only powers of 2 (0.5 == 2^(-1))
array([[  0.5,   0. ,  0. ],  # may vary
       [  0. ,   1. ,  0. ],
       [  0. ,   0. ,  1. ]])

References
----------
.. [1] : B.N. Parlett and C. Reinsch, 'Balancing a Matrix for
   Calculation of Eigenvalues and Eigenvectors', Numerische Mathematik,
   Vol.13(4), 1969, DOI:10.1007/BF02165404

.. [2] : R. James, J. Langou, B.R. Lowery, 'On matrix balancing and
   eigenvector computation', 2014, Available online:
   https://arxiv.org/abs/1401.5766

.. [3] :  D.S. Watkins. A case where balancing is harmful.
   Electron. Trans. Numer. Anal, Vol.23, 2006.
*)

val pinv : ?cond:Py.Object.t -> ?rcond:Py.Object.t -> ?return_rank:bool -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Compute the (Moore-Penrose) pseudo-inverse of a matrix.

Calculate a generalized inverse of a matrix using a least-squares
solver.

Parameters
----------
a : (M, N) array_like
    Matrix to be pseudo-inverted.
cond, rcond : float, optional
    Cutoff factor for 'small' singular values. In `lstsq`,
    singular values less than ``cond*largest_singular_value`` will be
    considered as zero. If both are omitted, the default value
    ``max(M, N) * eps`` is passed to `lstsq` where ``eps`` is the
    corresponding machine precision value of the datatype of ``a``.

    .. versionchanged:: 1.3.0
        Previously the default cutoff value was just `eps` without the
        factor ``max(M, N)``.

return_rank : bool, optional
    if True, return the effective rank of the matrix
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
B : (N, M) ndarray
    The pseudo-inverse of matrix `a`.
rank : int
    The effective rank of the matrix.  Returned if return_rank == True

Raises
------
LinAlgError
    If computation does not converge.

Examples
--------
>>> from scipy import linalg
>>> a = np.random.randn(9, 6)
>>> B = linalg.pinv(a)
>>> np.allclose(a, np.dot(a, np.dot(B, a)))
True
>>> np.allclose(B, np.dot(B, np.dot(a, B)))
True
*)

val pinv2 : ?cond:Py.Object.t -> ?rcond:Py.Object.t -> ?return_rank:bool -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Compute the (Moore-Penrose) pseudo-inverse of a matrix.

Calculate a generalized inverse of a matrix using its
singular-value decomposition and including all 'large' singular
values.

Parameters
----------
a : (M, N) array_like
    Matrix to be pseudo-inverted.
cond, rcond : float or None
    Cutoff for 'small' singular values; singular values smaller than this
    value are considered as zero. If both are omitted, the default value
    ``max(M,N)*largest_singular_value*eps`` is used where ``eps`` is the
    machine precision value of the datatype of ``a``.

    .. versionchanged:: 1.3.0
        Previously the default cutoff value was just ``eps*f`` where ``f``
        was ``1e3`` for single precision and ``1e6`` for double precision.

return_rank : bool, optional
    If True, return the effective rank of the matrix.
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
B : (N, M) ndarray
    The pseudo-inverse of matrix `a`.
rank : int
    The effective rank of the matrix.  Returned if `return_rank` is True.

Raises
------
LinAlgError
    If SVD computation does not converge.

Examples
--------
>>> from scipy import linalg
>>> a = np.random.randn(9, 6)
>>> B = linalg.pinv2(a)
>>> np.allclose(a, np.dot(a, np.dot(B, a)))
True
>>> np.allclose(B, np.dot(B, np.dot(a, B)))
True
*)

val pinvh : ?cond:Py.Object.t -> ?rcond:Py.Object.t -> ?lower:bool -> ?return_rank:bool -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Compute the (Moore-Penrose) pseudo-inverse of a Hermitian matrix.

Calculate a generalized inverse of a Hermitian or real symmetric matrix
using its eigenvalue decomposition and including all eigenvalues with
'large' absolute value.

Parameters
----------
a : (N, N) array_like
    Real symmetric or complex hermetian matrix to be pseudo-inverted
cond, rcond : float or None
    Cutoff for 'small' singular values; singular values smaller than this
    value are considered as zero. If both are omitted, the default
    ``max(M,N)*largest_eigenvalue*eps`` is used where ``eps`` is the
    machine precision value of the datatype of ``a``.

    .. versionchanged:: 1.3.0
        Previously the default cutoff value was just ``eps*f`` where ``f``
        was ``1e3`` for single precision and ``1e6`` for double precision.

lower : bool, optional
    Whether the pertinent array data is taken from the lower or upper
    triangle of `a`. (Default: lower)
return_rank : bool, optional
    If True, return the effective rank of the matrix.
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
B : (N, N) ndarray
    The pseudo-inverse of matrix `a`.
rank : int
    The effective rank of the matrix.  Returned if `return_rank` is True.

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

val solve_banded : ?overwrite_ab:bool -> ?overwrite_b:bool -> ?debug:Py.Object.t -> ?check_finite:bool -> l_and_u:Py.Object.t -> ab:Py.Object.t -> b:Py.Object.t -> unit -> Py.Object.t
(**
Solve the equation a x = b for x, assuming a is banded matrix.

The matrix a is stored in `ab` using the matrix diagonal ordered form::

    ab[u + i - j, j] == a[i,j]

Example of `ab` (shape of a is (6,6), `u` =1, `l` =2)::

    *    a01  a12  a23  a34  a45
    a00  a11  a22  a33  a44  a55
    a10  a21  a32  a43  a54   *
    a20  a31  a42  a53   *    *

Parameters
----------
(l, u) : (integer, integer)
    Number of non-zero lower and upper diagonals
ab : (`l` + `u` + 1, M) array_like
    Banded matrix
b : (M,) or (M, K) array_like
    Right-hand side
overwrite_ab : bool, optional
    Discard data in `ab` (may enhance performance)
overwrite_b : bool, optional
    Discard data in `b` (may enhance performance)
check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
x : (M,) or (M, K) ndarray
    The solution to the system a x = b.  Returned shape depends on the
    shape of `b`.

Examples
--------
Solve the banded system a x = b, where::

        [5  2 -1  0  0]       [0]
        [1  4  2 -1  0]       [1]
    a = [0  1  3  2 -1]   b = [2]
        [0  0  1  2  2]       [2]
        [0  0  0  1  1]       [3]

There is one nonzero diagonal below the main diagonal (l = 1), and
two above (u = 2).  The diagonal banded form of the matrix is::

         [*  * -1 -1 -1]
    ab = [*  2  2  2  2]
         [5  4  3  2  1]
         [1  1  1  1  *]

>>> from scipy.linalg import solve_banded
>>> ab = np.array([[0,  0, -1, -1, -1],
...                [0,  2,  2,  2,  2],
...                [5,  4,  3,  2,  1],
...                [1,  1,  1,  1,  0]])
>>> b = np.array([0, 1, 2, 2, 3])
>>> x = solve_banded((1, 2), ab, b)
>>> x
array([-2.37288136,  3.93220339, -4.        ,  4.3559322 , -1.3559322 ])
*)

val solve_circulant : ?singular:string -> ?tol:float -> ?caxis:int -> ?baxis:int -> ?outaxis:int -> c:[>`Ndarray] Np.Obj.t -> b:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Solve C x = b for x, where C is a circulant matrix.

`C` is the circulant matrix associated with the vector `c`.

The system is solved by doing division in Fourier space.  The
calculation is::

    x = ifft(fft(b) / fft(c))

where `fft` and `ifft` are the fast Fourier transform and its inverse,
respectively.  For a large vector `c`, this is *much* faster than
solving the system with the full circulant matrix.

Parameters
----------
c : array_like
    The coefficients of the circulant matrix.
b : array_like
    Right-hand side matrix in ``a x = b``.
singular : str, optional
    This argument controls how a near singular circulant matrix is
    handled.  If `singular` is 'raise' and the circulant matrix is
    near singular, a `LinAlgError` is raised.  If `singular` is
    'lstsq', the least squares solution is returned.  Default is 'raise'.
tol : float, optional
    If any eigenvalue of the circulant matrix has an absolute value
    that is less than or equal to `tol`, the matrix is considered to be
    near singular.  If not given, `tol` is set to::

        tol = abs_eigs.max() * abs_eigs.size * np.finfo(np.float64).eps

    where `abs_eigs` is the array of absolute values of the eigenvalues
    of the circulant matrix.
caxis : int
    When `c` has dimension greater than 1, it is viewed as a collection
    of circulant vectors.  In this case, `caxis` is the axis of `c` that
    holds the vectors of circulant coefficients.
baxis : int
    When `b` has dimension greater than 1, it is viewed as a collection
    of vectors.  In this case, `baxis` is the axis of `b` that holds the
    right-hand side vectors.
outaxis : int
    When `c` or `b` are multidimensional, the value returned by
    `solve_circulant` is multidimensional.  In this case, `outaxis` is
    the axis of the result that holds the solution vectors.

Returns
-------
x : ndarray
    Solution to the system ``C x = b``.

Raises
------
LinAlgError
    If the circulant matrix associated with `c` is near singular.

See Also
--------
circulant : circulant matrix

Notes
-----
For a one-dimensional vector `c` with length `m`, and an array `b`
with shape ``(m, ...)``,

    solve_circulant(c, b)

returns the same result as

    solve(circulant(c), b)

where `solve` and `circulant` are from `scipy.linalg`.

.. versionadded:: 0.16.0

Examples
--------
>>> from scipy.linalg import solve_circulant, solve, circulant, lstsq

>>> c = np.array([2, 2, 4])
>>> b = np.array([1, 2, 3])
>>> solve_circulant(c, b)
array([ 0.75, -0.25,  0.25])

Compare that result to solving the system with `scipy.linalg.solve`:

>>> solve(circulant(c), b)
array([ 0.75, -0.25,  0.25])

A singular example:

>>> c = np.array([1, 1, 0, 0])
>>> b = np.array([1, 2, 3, 4])

Calling ``solve_circulant(c, b)`` will raise a `LinAlgError`.  For the
least square solution, use the option ``singular='lstsq'``:

>>> solve_circulant(c, b, singular='lstsq')
array([ 0.25,  1.25,  2.25,  1.25])

Compare to `scipy.linalg.lstsq`:

>>> x, resid, rnk, s = lstsq(circulant(c), b)
>>> x
array([ 0.25,  1.25,  2.25,  1.25])

A broadcasting example:

Suppose we have the vectors of two circulant matrices stored in an array
with shape (2, 5), and three `b` vectors stored in an array with shape
(3, 5).  For example,

>>> c = np.array([[1.5, 2, 3, 0, 0], [1, 1, 4, 3, 2]])
>>> b = np.arange(15).reshape(-1, 5)

We want to solve all combinations of circulant matrices and `b` vectors,
with the result stored in an array with shape (2, 3, 5).  When we
disregard the axes of `c` and `b` that hold the vectors of coefficients,
the shapes of the collections are (2,) and (3,), respectively, which are
not compatible for broadcasting.  To have a broadcast result with shape
(2, 3), we add a trivial dimension to `c`: ``c[:, np.newaxis, :]`` has
shape (2, 1, 5).  The last dimension holds the coefficients of the
circulant matrices, so when we call `solve_circulant`, we can use the
default ``caxis=-1``.  The coefficients of the `b` vectors are in the last
dimension of the array `b`, so we use ``baxis=-1``.  If we use the
default `outaxis`, the result will have shape (5, 2, 3), so we'll use
``outaxis=-1`` to put the solution vectors in the last dimension.

>>> x = solve_circulant(c[:, np.newaxis, :], b, baxis=-1, outaxis=-1)
>>> x.shape
(2, 3, 5)
>>> np.set_printoptions(precision=3)  # For compact output of numbers.
>>> x
array([[[-0.118,  0.22 ,  1.277, -0.142,  0.302],
        [ 0.651,  0.989,  2.046,  0.627,  1.072],
        [ 1.42 ,  1.758,  2.816,  1.396,  1.841]],
       [[ 0.401,  0.304,  0.694, -0.867,  0.377],
        [ 0.856,  0.758,  1.149, -0.412,  0.831],
        [ 1.31 ,  1.213,  1.603,  0.042,  1.286]]])

Check by solving one pair of `c` and `b` vectors (cf. ``x[1, 1, :]``):

>>> solve_circulant(c[1], b[1, :])
array([ 0.856,  0.758,  1.149, -0.412,  0.831])
*)

val solve_toeplitz : ?check_finite:bool -> c_or_cr:[`Ndarray of [>`Ndarray] Np.Obj.t | `Tuple_of_array_like_array_like_ of Py.Object.t] -> b:Py.Object.t -> unit -> Py.Object.t
(**
Solve a Toeplitz system using Levinson Recursion

The Toeplitz matrix has constant diagonals, with c as its first column
and r as its first row.  If r is not given, ``r == conjugate(c)`` is
assumed.

Parameters
----------
c_or_cr : array_like or tuple of (array_like, array_like)
    The vector ``c``, or a tuple of arrays (``c``, ``r``). Whatever the
    actual shape of ``c``, it will be converted to a 1-D array. If not
    supplied, ``r = conjugate(c)`` is assumed; in this case, if c[0] is
    real, the Toeplitz matrix is Hermitian. r[0] is ignored; the first row
    of the Toeplitz matrix is ``[c[0], r[1:]]``.  Whatever the actual shape
    of ``r``, it will be converted to a 1-D array.
b : (M,) or (M, K) array_like
    Right-hand side in ``T x = b``.
check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (result entirely NaNs) if the inputs do contain infinities or NaNs.

Returns
-------
x : (M,) or (M, K) ndarray
    The solution to the system ``T x = b``.  Shape of return matches shape
    of `b`.

See Also
--------
toeplitz : Toeplitz matrix

Notes
-----
The solution is computed using Levinson-Durbin recursion, which is faster
than generic least-squares methods, but can be less numerically stable.

Examples
--------
Solve the Toeplitz system T x = b, where::

        [ 1 -1 -2 -3]       [1]
    T = [ 3  1 -1 -2]   b = [2]
        [ 6  3  1 -1]       [2]
        [10  6  3  1]       [5]

To specify the Toeplitz matrix, only the first column and the first
row are needed.

>>> c = np.array([1, 3, 6, 10])    # First column of T
>>> r = np.array([1, -1, -2, -3])  # First row of T
>>> b = np.array([1, 2, 2, 5])

>>> from scipy.linalg import solve_toeplitz, toeplitz
>>> x = solve_toeplitz((c, r), b)
>>> x
array([ 1.66666667, -1.        , -2.66666667,  2.33333333])

Check the result by creating the full Toeplitz matrix and
multiplying it by `x`.  We should get `b`.

>>> T = toeplitz(c, r)
>>> T.dot(x)
array([ 1.,  2.,  2.,  5.])
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

val solveh_banded : ?overwrite_ab:bool -> ?overwrite_b:bool -> ?lower:bool -> ?check_finite:bool -> ab:Py.Object.t -> b:Py.Object.t -> unit -> Py.Object.t
(**
Solve equation a x = b. a is Hermitian positive-definite banded matrix.

The matrix a is stored in `ab` either in lower diagonal or upper
diagonal ordered form:

    ab[u + i - j, j] == a[i,j]        (if upper form; i <= j)
    ab[    i - j, j] == a[i,j]        (if lower form; i >= j)

Example of `ab` (shape of a is (6, 6), `u` =2)::

    upper form:
    *   *   a02 a13 a24 a35
    *   a01 a12 a23 a34 a45
    a00 a11 a22 a33 a44 a55

    lower form:
    a00 a11 a22 a33 a44 a55
    a10 a21 a32 a43 a54 *
    a20 a31 a42 a53 *   *

Cells marked with * are not used.

Parameters
----------
ab : (`u` + 1, M) array_like
    Banded matrix
b : (M,) or (M, K) array_like
    Right-hand side
overwrite_ab : bool, optional
    Discard data in `ab` (may enhance performance)
overwrite_b : bool, optional
    Discard data in `b` (may enhance performance)
lower : bool, optional
    Is the matrix in the lower form. (Default is upper form)
check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
x : (M,) or (M, K) ndarray
    The solution to the system a x = b.  Shape of return matches shape
    of `b`.

Examples
--------
Solve the banded system A x = b, where::

        [ 4  2 -1  0  0  0]       [1]
        [ 2  5  2 -1  0  0]       [2]
    A = [-1  2  6  2 -1  0]   b = [2]
        [ 0 -1  2  7  2 -1]       [3]
        [ 0  0 -1  2  8  2]       [3]
        [ 0  0  0 -1  2  9]       [3]

>>> from scipy.linalg import solveh_banded

`ab` contains the main diagonal and the nonzero diagonals below the
main diagonal.  That is, we use the lower form:

>>> ab = np.array([[ 4,  5,  6,  7, 8, 9],
...                [ 2,  2,  2,  2, 2, 0],
...                [-1, -1, -1, -1, 0, 0]])
>>> b = np.array([1, 2, 2, 3, 3, 3])
>>> x = solveh_banded(ab, b, lower=True)
>>> x
array([ 0.03431373,  0.45938375,  0.05602241,  0.47759104,  0.17577031,
        0.34733894])


Solve the Hermitian banded system H x = b, where::

        [ 8   2-1j   0     0  ]        [ 1  ]
    H = [2+1j  5     1j    0  ]    b = [1+1j]
        [ 0   -1j    9   -2-1j]        [1-2j]
        [ 0    0   -2+1j   6  ]        [ 0  ]

In this example, we put the upper diagonals in the array `hb`:

>>> hb = np.array([[0, 2-1j, 1j, -2-1j],
...                [8,  5,    9,   6  ]])
>>> b = np.array([1, 1+1j, 1-2j, 0])
>>> x = solveh_banded(hb, b)
>>> x
array([ 0.07318536-0.02939412j,  0.11877624+0.17696461j,
        0.10077984-0.23035393j, -0.00479904-0.09358128j])
*)

val warn : ?category:Py.Object.t -> ?stacklevel:Py.Object.t -> ?source:Py.Object.t -> message:Py.Object.t -> unit -> Py.Object.t
(**
Issue a warning, or maybe ignore it or raise an exception.
*)


end

module Blas : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val crotg : a:Py.Object.t -> b:Py.Object.t -> unit -> Py.Object.t
(**
c,s = crotg(a,b)

Wrapper for ``crotg``.

Parameters
----------
a : input complex
b : input complex

Returns
-------
c : complex
s : complex
*)

val drotg : a:Py.Object.t -> b:Py.Object.t -> unit -> float
(**
c,s = drotg(a,b)

Wrapper for ``drotg``.

Parameters
----------
a : input float
b : input float

Returns
-------
c : float
s : float
*)

val find_best_blas_type : ?arrays:[>`Ndarray] Np.Obj.t list -> ?dtype:[`S of string | `Dtype of Np.Dtype.t] -> unit -> (string * Np.Dtype.t * bool)
(**
Find best-matching BLAS/LAPACK type.

Arrays are used to determine the optimal prefix of BLAS routines.

Parameters
----------
arrays : sequence of ndarrays, optional
    Arrays can be given to determine optimal prefix of BLAS
    routines. If not given, double-precision routines will be
    used, otherwise the most generic type in arrays will be used.
dtype : str or dtype, optional
    Data-type specifier. Not used if `arrays` is non-empty.

Returns
-------
prefix : str
    BLAS/LAPACK prefix character.
dtype : dtype
    Inferred Numpy data type.
prefer_fortran : bool
    Whether to prefer Fortran order routines over C order.

Examples
--------
>>> import scipy.linalg.blas as bla
>>> a = np.random.rand(10,15)
>>> b = np.asfortranarray(a)  # Change the memory layout order
>>> bla.find_best_blas_type((a,))
('d', dtype('float64'), False)
>>> bla.find_best_blas_type((a*1j,))
('z', dtype('complex128'), False)
>>> bla.find_best_blas_type((b,))
('d', dtype('float64'), True)
*)

val get_blas_funcs : ?arrays:[>`Ndarray] Np.Obj.t list -> ?dtype:[`S of string | `Dtype of Np.Dtype.t] -> names:[`Sequence_of_str of Py.Object.t | `S of string] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return available BLAS function objects from names.

Arrays are used to determine the optimal prefix of BLAS routines.

Parameters
----------
names : str or sequence of str
    Name(s) of BLAS functions without type prefix.

arrays : sequence of ndarrays, optional
    Arrays can be given to determine optimal prefix of BLAS
    routines. If not given, double-precision routines will be
    used, otherwise the most generic type in arrays will be used.

dtype : str or dtype, optional
    Data-type specifier. Not used if `arrays` is non-empty.


Returns
-------
funcs : list
    List containing the found function(s).


Notes
-----
This routine automatically chooses between Fortran/C
interfaces. Fortran code is used whenever possible for arrays with
column major order. In all other cases, C code is preferred.

In BLAS, the naming convention is that all functions start with a
type prefix, which depends on the type of the principal
matrix. These can be one of {'s', 'd', 'c', 'z'} for the numpy
types {float32, float64, complex64, complex128} respectively.
The code and the dtype are stored in attributes `typecode` and `dtype`
of the returned functions.

Examples
--------
>>> import scipy.linalg as LA
>>> a = np.random.rand(3,2)
>>> x_gemv = LA.get_blas_funcs('gemv', (a,))
>>> x_gemv.typecode
'd'
>>> x_gemv = LA.get_blas_funcs('gemv',(a*1j,))
>>> x_gemv.typecode
'z'
*)

val srotg : a:Py.Object.t -> b:Py.Object.t -> unit -> float
(**
c,s = srotg(a,b)

Wrapper for ``srotg``.

Parameters
----------
a : input float
b : input float

Returns
-------
c : float
s : float
*)

val zrotg : a:Py.Object.t -> b:Py.Object.t -> unit -> Py.Object.t
(**
c,s = zrotg(a,b)

Wrapper for ``zrotg``.

Parameters
----------
a : input complex
b : input complex

Returns
-------
c : complex
s : complex
*)


end

module Cython_blas : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t


end

module Cython_lapack : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t


end

module Decomp : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module Inexact : sig
type tag = [`Inexact]
type t = [`Inexact | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : unit -> t
(**
Abstract base class of all numeric scalar types with a (potentially)
inexact representation of the values in its range, such as
floating-point numbers.
*)

val __getitem__ : key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return self[key].
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val argsort : ?axis:[`I of int | `None] -> ?kind:[`Stable | `Quicksort | `Heapsort | `Mergesort] -> ?order:[`StringList of string list | `S of string] -> a:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Returns the indices that would sort an array.

Perform an indirect sort along the given axis using the algorithm specified
by the `kind` keyword. It returns an array of indices of the same shape as
`a` that index data along the given axis in sorted order.

Parameters
----------
a : array_like
    Array to sort.
axis : int or None, optional
    Axis along which to sort.  The default is -1 (the last axis). If None,
    the flattened array is used.
kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
    Sorting algorithm. The default is 'quicksort'. Note that both 'stable'
    and 'mergesort' use timsort under the covers and, in general, the
    actual implementation will vary with data type. The 'mergesort' option
    is retained for backwards compatibility.

    .. versionchanged:: 1.15.0.
       The 'stable' option was added.
order : str or list of str, optional
    When `a` is an array with fields defined, this argument specifies
    which fields to compare first, second, etc.  A single field can
    be specified as a string, and not all fields need be specified,
    but unspecified fields will still be used, in the order in which
    they come up in the dtype, to break ties.

Returns
-------
index_array : ndarray, int
    Array of indices that sort `a` along the specified `axis`.
    If `a` is one-dimensional, ``a[index_array]`` yields a sorted `a`.
    More generally, ``np.take_along_axis(a, index_array, axis=axis)``
    always yields the sorted `a`, irrespective of dimensionality.

See Also
--------
sort : Describes sorting algorithms used.
lexsort : Indirect stable sort with multiple keys.
ndarray.sort : Inplace sort.
argpartition : Indirect partial sort.
take_along_axis : Apply ``index_array`` from argsort 
                  to an array as if by calling sort.

Notes
-----
See `sort` for notes on the different sorting algorithms.

As of NumPy 1.4.0 `argsort` works with real/complex arrays containing
nan values. The enhanced sort order is documented in `sort`.

Examples
--------
One dimensional array:

>>> x = np.array([3, 1, 2])
>>> np.argsort(x)
array([1, 2, 0])

Two-dimensional array:

>>> x = np.array([[0, 3], [2, 2]])
>>> x
array([[0, 3],
       [2, 2]])

>>> ind = np.argsort(x, axis=0)  # sorts along first axis (down)
>>> ind
array([[0, 1],
       [1, 0]])
>>> np.take_along_axis(x, ind, axis=0)  # same as np.sort(x, axis=0)
array([[0, 2],
       [2, 3]])

>>> ind = np.argsort(x, axis=1)  # sorts along last axis (across)
>>> ind
array([[0, 1],
       [0, 1]])
>>> np.take_along_axis(x, ind, axis=1)  # same as np.sort(x, axis=1)
array([[0, 3],
       [2, 2]])

Indices of the sorted elements of a N-dimensional array:

>>> ind = np.unravel_index(np.argsort(x, axis=None), x.shape)
>>> ind
(array([0, 1, 1, 0]), array([0, 0, 1, 1]))
>>> x[ind]  # same as np.sort(x, axis=None)
array([0, 2, 2, 3])

Sorting with keys:

>>> x = np.array([(1, 0), (0, 1)], dtype=[('x', '<i4'), ('y', '<i4')])
>>> x
array([(1, 0), (0, 1)],
      dtype=[('x', '<i4'), ('y', '<i4')])

>>> np.argsort(x, order=('x','y'))
array([1, 0])

>>> np.argsort(x, order=('y','x'))
array([0, 1])
*)

val argwhere : [>`Ndarray] Np.Obj.t -> Py.Object.t
(**
Find the indices of array elements that are non-zero, grouped by element.

Parameters
----------
a : array_like
    Input data.

Returns
-------
index_array : (N, a.ndim) ndarray
    Indices of elements that are non-zero. Indices are grouped by element.
    This array will have shape ``(N, a.ndim)`` where ``N`` is the number of
    non-zero items.

See Also
--------
where, nonzero

Notes
-----
``np.argwhere(a)`` is almost the same as ``np.transpose(np.nonzero(a))``,
but produces a result of the correct shape for a 0D array.

The output of ``argwhere`` is not suitable for indexing arrays.
For this purpose use ``nonzero(a)`` instead.

Examples
--------
>>> x = np.arange(6).reshape(2,3)
>>> x
array([[0, 1, 2],
       [3, 4, 5]])
>>> np.argwhere(x>1)
array([[0, 2],
       [1, 0],
       [1, 1],
       [1, 2]])
*)

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

val cdf2rdf : w:Py.Object.t -> v:Py.Object.t -> unit -> (Py.Object.t * Py.Object.t)
(**
Converts complex eigenvalues ``w`` and eigenvectors ``v`` to real
eigenvalues in a block diagonal form ``wr`` and the associated real
eigenvectors ``vr``, such that::

    vr @ wr = X @ vr

continues to hold, where ``X`` is the original array for which ``w`` and
``v`` are the eigenvalues and eigenvectors.

.. versionadded:: 1.1.0

Parameters
----------
w : (..., M) array_like
    Complex or real eigenvalues, an array or stack of arrays

    Conjugate pairs must not be interleaved, else the wrong result
    will be produced. So ``[1+1j, 1, 1-1j]`` will give a correct result, but
    ``[1+1j, 2+1j, 1-1j, 2-1j]`` will not.

v : (..., M, M) array_like
    Complex or real eigenvectors, a square array or stack of square arrays.

Returns
-------
wr : (..., M, M) ndarray
    Real diagonal block form of eigenvalues
vr : (..., M, M) ndarray
    Real eigenvectors associated with ``wr``

See Also
--------
eig : Eigenvalues and right eigenvectors for non-symmetric arrays
rsf2csf : Convert real Schur form to complex Schur form

Notes
-----
``w``, ``v`` must be the eigenstructure for some *real* matrix ``X``.
For example, obtained by ``w, v = scipy.linalg.eig(X)`` or
``w, v = numpy.linalg.eig(X)`` in which case ``X`` can also represent
stacked arrays.

.. versionadded:: 1.1.0

Examples
--------
>>> X = np.array([[1, 2, 3], [0, 4, 5], [0, -5, 4]])
>>> X
array([[ 1,  2,  3],
       [ 0,  4,  5],
       [ 0, -5,  4]])

>>> from scipy import linalg
>>> w, v = linalg.eig(X)
>>> w
array([ 1.+0.j,  4.+5.j,  4.-5.j])
>>> v
array([[ 1.00000+0.j     , -0.01906-0.40016j, -0.01906+0.40016j],
       [ 0.00000+0.j     ,  0.00000-0.64788j,  0.00000+0.64788j],
       [ 0.00000+0.j     ,  0.64788+0.j     ,  0.64788-0.j     ]])

>>> wr, vr = linalg.cdf2rdf(w, v)
>>> wr
array([[ 1.,  0.,  0.],
       [ 0.,  4.,  5.],
       [ 0., -5.,  4.]])
>>> vr
array([[ 1.     ,  0.40016, -0.01906],
       [ 0.     ,  0.64788,  0.     ],
       [ 0.     ,  0.     ,  0.64788]])

>>> vr @ wr
array([[ 1.     ,  1.69593,  1.9246 ],
       [ 0.     ,  2.59153,  3.23942],
       [ 0.     , -3.23942,  2.59153]])
>>> X @ vr
array([[ 1.     ,  1.69593,  1.9246 ],
       [ 0.     ,  2.59153,  3.23942],
       [ 0.     , -3.23942,  2.59153]])
*)

val conj : ?out:[`Ndarray of [>`Ndarray] Np.Obj.t | `Tuple_of_ndarray_and_None of Py.Object.t] -> ?where:[>`Ndarray] Np.Obj.t -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
conjugate(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

Return the complex conjugate, element-wise.

The complex conjugate of a complex number is obtained by changing the
sign of its imaginary part.

Parameters
----------
x : array_like
    Input value.
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
    The complex conjugate of `x`, with same dtype as `y`.
    This is a scalar if `x` is a scalar.

Notes
-----
`conj` is an alias for `conjugate`:

>>> np.conj is np.conjugate
True

Examples
--------
>>> np.conjugate(1+2j)
(1-2j)

>>> x = np.eye(2) + 1j * np.eye(2)
>>> np.conjugate(x)
array([[ 1.-1.j,  0.-0.j],
       [ 0.-0.j,  1.-1.j]])
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

val eig_banded : ?lower:bool -> ?eigvals_only:bool -> ?overwrite_a_band:bool -> ?select:[`A | `V | `I] -> ?select_range:Py.Object.t -> ?max_ev:int -> ?check_finite:bool -> a_band:Py.Object.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * Py.Object.t)
(**
Solve real symmetric or complex hermitian band matrix eigenvalue problem.

Find eigenvalues w and optionally right eigenvectors v of a::

    a v[:,i] = w[i] v[:,i]
    v.H v    = identity

The matrix a is stored in a_band either in lower diagonal or upper
diagonal ordered form:

    a_band[u + i - j, j] == a[i,j]        (if upper form; i <= j)
    a_band[    i - j, j] == a[i,j]        (if lower form; i >= j)

where u is the number of bands above the diagonal.

Example of a_band (shape of a is (6,6), u=2)::

    upper form:
    *   *   a02 a13 a24 a35
    *   a01 a12 a23 a34 a45
    a00 a11 a22 a33 a44 a55

    lower form:
    a00 a11 a22 a33 a44 a55
    a10 a21 a32 a43 a54 *
    a20 a31 a42 a53 *   *

Cells marked with * are not used.

Parameters
----------
a_band : (u+1, M) array_like
    The bands of the M by M matrix a.
lower : bool, optional
    Is the matrix in the lower form. (Default is upper form)
eigvals_only : bool, optional
    Compute only the eigenvalues and no eigenvectors.
    (Default: calculate also eigenvectors)
overwrite_a_band : bool, optional
    Discard data in a_band (may enhance performance)
select : {'a', 'v', 'i'}, optional
    Which eigenvalues to calculate

    ======  ========================================
    select  calculated
    ======  ========================================
    'a'     All eigenvalues
    'v'     Eigenvalues in the interval (min, max]
    'i'     Eigenvalues with indices min <= i <= max
    ======  ========================================
select_range : (min, max), optional
    Range of selected eigenvalues
max_ev : int, optional
    For select=='v', maximum number of eigenvalues expected.
    For other values of select, has no meaning.

    In doubt, leave this parameter untouched.

check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
w : (M,) ndarray
    The eigenvalues, in ascending order, each repeated according to its
    multiplicity.
v : (M, M) float or complex ndarray
    The normalized eigenvector corresponding to the eigenvalue w[i] is
    the column v[:,i].

Raises
------
LinAlgError
    If eigenvalue computation does not converge.

See Also
--------
eigvals_banded : eigenvalues for symmetric/Hermitian band matrices
eig : eigenvalues and right eigenvectors of general arrays.
eigh : eigenvalues and right eigenvectors for symmetric/Hermitian arrays
eigh_tridiagonal : eigenvalues and right eiegenvectors for
    symmetric/Hermitian tridiagonal matrices

Examples
--------
>>> from scipy.linalg import eig_banded
>>> A = np.array([[1, 5, 2, 0], [5, 2, 5, 2], [2, 5, 3, 5], [0, 2, 5, 4]])
>>> Ab = np.array([[1, 2, 3, 4], [5, 5, 5, 0], [2, 2, 0, 0]])
>>> w, v = eig_banded(Ab, lower=True)
>>> np.allclose(A @ v - v @ np.diag(w), np.zeros((4, 4)))
True
>>> w = eig_banded(Ab, lower=True, eigvals_only=True)
>>> w
array([-4.26200532, -2.22987175,  3.95222349, 12.53965359])

Request only the eigenvalues between ``[-3, 4]``

>>> w, v = eig_banded(Ab, lower=True, select='v', select_range=[-3, 4])
>>> w
array([-2.22987175,  3.95222349])
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

val eigh_tridiagonal : ?eigvals_only:Py.Object.t -> ?select:[`A | `V | `I] -> ?select_range:Py.Object.t -> ?check_finite:bool -> ?tol:float -> ?lapack_driver:string -> d:[>`Ndarray] Np.Obj.t -> e:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Solve eigenvalue problem for a real symmetric tridiagonal matrix.

Find eigenvalues `w` and optionally right eigenvectors `v` of ``a``::

    a v[:,i] = w[i] v[:,i]
    v.H v    = identity

For a real symmetric matrix ``a`` with diagonal elements `d` and
off-diagonal elements `e`.

Parameters
----------
d : ndarray, shape (ndim,)
    The diagonal elements of the array.
e : ndarray, shape (ndim-1,)
    The off-diagonal elements of the array.
select : {'a', 'v', 'i'}, optional
    Which eigenvalues to calculate

    ======  ========================================
    select  calculated
    ======  ========================================
    'a'     All eigenvalues
    'v'     Eigenvalues in the interval (min, max]
    'i'     Eigenvalues with indices min <= i <= max
    ======  ========================================
select_range : (min, max), optional
    Range of selected eigenvalues
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.
tol : float
    The absolute tolerance to which each eigenvalue is required
    (only used when 'stebz' is the `lapack_driver`).
    An eigenvalue (or cluster) is considered to have converged if it
    lies in an interval of this width. If <= 0. (default),
    the value ``eps*|a|`` is used where eps is the machine precision,
    and ``|a|`` is the 1-norm of the matrix ``a``.
lapack_driver : str
    LAPACK function to use, can be 'auto', 'stemr', 'stebz', 'sterf',
    or 'stev'. When 'auto' (default), it will use 'stemr' if ``select='a'``
    and 'stebz' otherwise. When 'stebz' is used to find the eigenvalues and
    ``eigvals_only=False``, then a second LAPACK call (to ``?STEIN``) is
    used to find the corresponding eigenvectors. 'sterf' can only be
    used when ``eigvals_only=True`` and ``select='a'``. 'stev' can only
    be used when ``select='a'``.

Returns
-------
w : (M,) ndarray
    The eigenvalues, in ascending order, each repeated according to its
    multiplicity.
v : (M, M) ndarray
    The normalized eigenvector corresponding to the eigenvalue ``w[i]`` is
    the column ``v[:,i]``.

Raises
------
LinAlgError
    If eigenvalue computation does not converge.

See Also
--------
eigvalsh_tridiagonal : eigenvalues of symmetric/Hermitian tridiagonal
    matrices
eig : eigenvalues and right eigenvectors for non-symmetric arrays
eigh : eigenvalues and right eigenvectors for symmetric/Hermitian arrays
eig_banded : eigenvalues and right eigenvectors for symmetric/Hermitian
    band matrices

Notes
-----
This function makes use of LAPACK ``S/DSTEMR`` routines.

Examples
--------
>>> from scipy.linalg import eigh_tridiagonal
>>> d = 3*np.ones(4)
>>> e = -1*np.ones(3)
>>> w, v = eigh_tridiagonal(d, e)
>>> A = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)
>>> np.allclose(A @ v - v @ np.diag(w), np.zeros((4, 4)))
True
*)

val eigvals : ?b:[>`Ndarray] Np.Obj.t -> ?overwrite_a:bool -> ?check_finite:bool -> ?homogeneous_eigvals:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Compute eigenvalues from an ordinary or generalized eigenvalue problem.

Find eigenvalues of a general matrix::

    a   vr[:,i] = w[i]        b   vr[:,i]

Parameters
----------
a : (M, M) array_like
    A complex or real matrix whose eigenvalues and eigenvectors
    will be computed.
b : (M, M) array_like, optional
    Right-hand side matrix in a generalized eigenvalue problem.
    If omitted, identity matrix is assumed.
overwrite_a : bool, optional
    Whether to overwrite data in a (may improve performance)
check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities
    or NaNs.
homogeneous_eigvals : bool, optional
    If True, return the eigenvalues in homogeneous coordinates.
    In this case ``w`` is a (2, M) array so that::

        w[1,i] a vr[:,i] = w[0,i] b vr[:,i]

    Default is False.

Returns
-------
w : (M,) or (2, M) double or complex ndarray
    The eigenvalues, each repeated according to its multiplicity
    but not in any specific order. The shape is (M,) unless
    ``homogeneous_eigvals=True``.

Raises
------
LinAlgError
    If eigenvalue computation does not converge

See Also
--------
eig : eigenvalues and right eigenvectors of general arrays.
eigvalsh : eigenvalues of symmetric or Hermitian arrays
eigvals_banded : eigenvalues for symmetric/Hermitian band matrices
eigvalsh_tridiagonal : eigenvalues of symmetric/Hermitian tridiagonal
    matrices

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
*)

val eigvals_banded : ?lower:bool -> ?overwrite_a_band:bool -> ?select:[`A | `V | `I] -> ?select_range:Py.Object.t -> ?check_finite:bool -> a_band:Py.Object.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Solve real symmetric or complex hermitian band matrix eigenvalue problem.

Find eigenvalues w of a::

    a v[:,i] = w[i] v[:,i]
    v.H v    = identity

The matrix a is stored in a_band either in lower diagonal or upper
diagonal ordered form:

    a_band[u + i - j, j] == a[i,j]        (if upper form; i <= j)
    a_band[    i - j, j] == a[i,j]        (if lower form; i >= j)

where u is the number of bands above the diagonal.

Example of a_band (shape of a is (6,6), u=2)::

    upper form:
    *   *   a02 a13 a24 a35
    *   a01 a12 a23 a34 a45
    a00 a11 a22 a33 a44 a55

    lower form:
    a00 a11 a22 a33 a44 a55
    a10 a21 a32 a43 a54 *
    a20 a31 a42 a53 *   *

Cells marked with * are not used.

Parameters
----------
a_band : (u+1, M) array_like
    The bands of the M by M matrix a.
lower : bool, optional
    Is the matrix in the lower form. (Default is upper form)
overwrite_a_band : bool, optional
    Discard data in a_band (may enhance performance)
select : {'a', 'v', 'i'}, optional
    Which eigenvalues to calculate

    ======  ========================================
    select  calculated
    ======  ========================================
    'a'     All eigenvalues
    'v'     Eigenvalues in the interval (min, max]
    'i'     Eigenvalues with indices min <= i <= max
    ======  ========================================
select_range : (min, max), optional
    Range of selected eigenvalues
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
w : (M,) ndarray
    The eigenvalues, in ascending order, each repeated according to its
    multiplicity.

Raises
------
LinAlgError
    If eigenvalue computation does not converge.

See Also
--------
eig_banded : eigenvalues and right eigenvectors for symmetric/Hermitian
    band matrices
eigvalsh_tridiagonal : eigenvalues of symmetric/Hermitian tridiagonal
    matrices
eigvals : eigenvalues of general arrays
eigh : eigenvalues and right eigenvectors for symmetric/Hermitian arrays
eig : eigenvalues and right eigenvectors for non-symmetric arrays

Examples
--------
>>> from scipy.linalg import eigvals_banded
>>> A = np.array([[1, 5, 2, 0], [5, 2, 5, 2], [2, 5, 3, 5], [0, 2, 5, 4]])
>>> Ab = np.array([[1, 2, 3, 4], [5, 5, 5, 0], [2, 2, 0, 0]])
>>> w = eigvals_banded(Ab, lower=True)
>>> w
array([-4.26200532, -2.22987175,  3.95222349, 12.53965359])
*)

val eigvalsh : ?b:[>`Ndarray] Np.Obj.t -> ?lower:bool -> ?overwrite_a:bool -> ?overwrite_b:bool -> ?turbo:bool -> ?eigvals:Py.Object.t -> ?type_:int -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Solve an ordinary or generalized eigenvalue problem for a complex
Hermitian or real symmetric matrix.

Find eigenvalues w of matrix a, where b is positive definite::

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
turbo : bool, optional
    Use divide and conquer algorithm (faster but expensive in memory,
    only for generalized eigenvalue problem and if eigvals=None)
eigvals : tuple (lo, hi), optional
    Indexes of the smallest and largest (in ascending order) eigenvalues
    and corresponding eigenvectors to be returned: 0 <= lo < hi <= M-1.
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

Raises
------
LinAlgError
    If eigenvalue computation does not converge,
    an error occurred, or b matrix is not definite positive. Note that
    if input matrices are not symmetric or hermitian, no error is reported
    but results will be wrong.

See Also
--------
eigh : eigenvalues and right eigenvectors for symmetric/Hermitian arrays
eigvals : eigenvalues of general arrays
eigvals_banded : eigenvalues for symmetric/Hermitian band matrices
eigvalsh_tridiagonal : eigenvalues of symmetric/Hermitian tridiagonal
    matrices

Notes
-----
This function does not check the input array for being hermitian/symmetric
in order to allow for representing arrays with only their upper/lower
triangular parts.

Examples
--------
>>> from scipy.linalg import eigvalsh
>>> A = np.array([[6, 3, 1, 5], [3, 0, 5, 1], [1, 5, 6, 2], [5, 1, 2, 2]])
>>> w = eigvalsh(A)
>>> w
array([-3.74637491, -0.76263923,  6.08502336, 12.42399079])
*)

val eigvalsh_tridiagonal : ?select:[`A | `V | `I] -> ?select_range:Py.Object.t -> ?check_finite:bool -> ?tol:float -> ?lapack_driver:string -> d:[>`Ndarray] Np.Obj.t -> e:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Solve eigenvalue problem for a real symmetric tridiagonal matrix.

Find eigenvalues `w` of ``a``::

    a v[:,i] = w[i] v[:,i]
    v.H v    = identity

For a real symmetric matrix ``a`` with diagonal elements `d` and
off-diagonal elements `e`.

Parameters
----------
d : ndarray, shape (ndim,)
    The diagonal elements of the array.
e : ndarray, shape (ndim-1,)
    The off-diagonal elements of the array.
select : {'a', 'v', 'i'}, optional
    Which eigenvalues to calculate

    ======  ========================================
    select  calculated
    ======  ========================================
    'a'     All eigenvalues
    'v'     Eigenvalues in the interval (min, max]
    'i'     Eigenvalues with indices min <= i <= max
    ======  ========================================
select_range : (min, max), optional
    Range of selected eigenvalues
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.
tol : float
    The absolute tolerance to which each eigenvalue is required
    (only used when ``lapack_driver='stebz'``).
    An eigenvalue (or cluster) is considered to have converged if it
    lies in an interval of this width. If <= 0. (default),
    the value ``eps*|a|`` is used where eps is the machine precision,
    and ``|a|`` is the 1-norm of the matrix ``a``.
lapack_driver : str
    LAPACK function to use, can be 'auto', 'stemr', 'stebz',  'sterf',
    or 'stev'. When 'auto' (default), it will use 'stemr' if ``select='a'``
    and 'stebz' otherwise. 'sterf' and 'stev' can only be used when
    ``select='a'``.

Returns
-------
w : (M,) ndarray
    The eigenvalues, in ascending order, each repeated according to its
    multiplicity.

Raises
------
LinAlgError
    If eigenvalue computation does not converge.

See Also
--------
eigh_tridiagonal : eigenvalues and right eiegenvectors for
    symmetric/Hermitian tridiagonal matrices

Examples
--------
>>> from scipy.linalg import eigvalsh_tridiagonal, eigvalsh
>>> d = 3*np.ones(4)
>>> e = -1*np.ones(3)
>>> w = eigvalsh_tridiagonal(d, e)
>>> A = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)
>>> w2 = eigvalsh(A)  # Verify with other eigenvalue routines
>>> np.allclose(w - w2, np.zeros(4))
True
*)

val einsum : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
einsum(subscripts, *operands, out=None, dtype=None, order='K',
       casting='safe', optimize=False)

Evaluates the Einstein summation convention on the operands.

Using the Einstein summation convention, many common multi-dimensional,
linear algebraic array operations can be represented in a simple fashion.
In *implicit* mode `einsum` computes these values.

In *explicit* mode, `einsum` provides further flexibility to compute
other array operations that might not be considered classical Einstein
summation operations, by disabling, or forcing summation over specified
subscript labels.

See the notes and examples for clarification.

Parameters
----------
subscripts : str
    Specifies the subscripts for summation as comma separated list of
    subscript labels. An implicit (classical Einstein summation)
    calculation is performed unless the explicit indicator '->' is
    included as well as subscript labels of the precise output form.
operands : list of array_like
    These are the arrays for the operation.
out : ndarray, optional
    If provided, the calculation is done into this array.
dtype : {data-type, None}, optional
    If provided, forces the calculation to use the data type specified.
    Note that you may have to also give a more liberal `casting`
    parameter to allow the conversions. Default is None.
order : {'C', 'F', 'A', 'K'}, optional
    Controls the memory layout of the output. 'C' means it should
    be C contiguous. 'F' means it should be Fortran contiguous,
    'A' means it should be 'F' if the inputs are all 'F', 'C' otherwise.
    'K' means it should be as close to the layout as the inputs as
    is possible, including arbitrarily permuted axes.
    Default is 'K'.
casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
    Controls what kind of data casting may occur.  Setting this to
    'unsafe' is not recommended, as it can adversely affect accumulations.

      * 'no' means the data types should not be cast at all.
      * 'equiv' means only byte-order changes are allowed.
      * 'safe' means only casts which can preserve values are allowed.
      * 'same_kind' means only safe casts or casts within a kind,
        like float64 to float32, are allowed.
      * 'unsafe' means any data conversions may be done.

    Default is 'safe'.
optimize : {False, True, 'greedy', 'optimal'}, optional
    Controls if intermediate optimization should occur. No optimization
    will occur if False and True will default to the 'greedy' algorithm.
    Also accepts an explicit contraction list from the ``np.einsum_path``
    function. See ``np.einsum_path`` for more details. Defaults to False.

Returns
-------
output : ndarray
    The calculation based on the Einstein summation convention.

See Also
--------
einsum_path, dot, inner, outer, tensordot, linalg.multi_dot

Notes
-----
.. versionadded:: 1.6.0

The Einstein summation convention can be used to compute
many multi-dimensional, linear algebraic array operations. `einsum`
provides a succinct way of representing these.

A non-exhaustive list of these operations,
which can be computed by `einsum`, is shown below along with examples:

* Trace of an array, :py:func:`numpy.trace`.
* Return a diagonal, :py:func:`numpy.diag`.
* Array axis summations, :py:func:`numpy.sum`.
* Transpositions and permutations, :py:func:`numpy.transpose`.
* Matrix multiplication and dot product, :py:func:`numpy.matmul` :py:func:`numpy.dot`.
* Vector inner and outer products, :py:func:`numpy.inner` :py:func:`numpy.outer`.
* Broadcasting, element-wise and scalar multiplication, :py:func:`numpy.multiply`.
* Tensor contractions, :py:func:`numpy.tensordot`.
* Chained array operations, in efficient calculation order, :py:func:`numpy.einsum_path`.

The subscripts string is a comma-separated list of subscript labels,
where each label refers to a dimension of the corresponding operand.
Whenever a label is repeated it is summed, so ``np.einsum('i,i', a, b)``
is equivalent to :py:func:`np.inner(a,b) <numpy.inner>`. If a label
appears only once, it is not summed, so ``np.einsum('i', a)`` produces a
view of ``a`` with no changes. A further example ``np.einsum('ij,jk', a, b)``
describes traditional matrix multiplication and is equivalent to
:py:func:`np.matmul(a,b) <numpy.matmul>`. Repeated subscript labels in one
operand take the diagonal. For example, ``np.einsum('ii', a)`` is equivalent
to :py:func:`np.trace(a) <numpy.trace>`.

In *implicit mode*, the chosen subscripts are important
since the axes of the output are reordered alphabetically.  This
means that ``np.einsum('ij', a)`` doesn't affect a 2D array, while
``np.einsum('ji', a)`` takes its transpose. Additionally,
``np.einsum('ij,jk', a, b)`` returns a matrix multiplication, while,
``np.einsum('ij,jh', a, b)`` returns the transpose of the
multiplication since subscript 'h' precedes subscript 'i'.

In *explicit mode* the output can be directly controlled by
specifying output subscript labels.  This requires the
identifier '->' as well as the list of output subscript labels.
This feature increases the flexibility of the function since
summing can be disabled or forced when required. The call
``np.einsum('i->', a)`` is like :py:func:`np.sum(a, axis=-1) <numpy.sum>`,
and ``np.einsum('ii->i', a)`` is like :py:func:`np.diag(a) <numpy.diag>`.
The difference is that `einsum` does not allow broadcasting by default.
Additionally ``np.einsum('ij,jh->ih', a, b)`` directly specifies the
order of the output subscript labels and therefore returns matrix
multiplication, unlike the example above in implicit mode.

To enable and control broadcasting, use an ellipsis.  Default
NumPy-style broadcasting is done by adding an ellipsis
to the left of each term, like ``np.einsum('...ii->...i', a)``.
To take the trace along the first and last axes,
you can do ``np.einsum('i...i', a)``, or to do a matrix-matrix
product with the left-most indices instead of rightmost, one can do
``np.einsum('ij...,jk...->ik...', a, b)``.

When there is only one operand, no axes are summed, and no output
parameter is provided, a view into the operand is returned instead
of a new array.  Thus, taking the diagonal as ``np.einsum('ii->i', a)``
produces a view (changed in version 1.10.0).

`einsum` also provides an alternative way to provide the subscripts
and operands as ``einsum(op0, sublist0, op1, sublist1, ..., [sublistout])``.
If the output shape is not provided in this format `einsum` will be
calculated in implicit mode, otherwise it will be performed explicitly.
The examples below have corresponding `einsum` calls with the two
parameter methods.

.. versionadded:: 1.10.0

Views returned from einsum are now writeable whenever the input array
is writeable. For example, ``np.einsum('ijk...->kji...', a)`` will now
have the same effect as :py:func:`np.swapaxes(a, 0, 2) <numpy.swapaxes>`
and ``np.einsum('ii->i', a)`` will return a writeable view of the diagonal
of a 2D array.

.. versionadded:: 1.12.0

Added the ``optimize`` argument which will optimize the contraction order
of an einsum expression. For a contraction with three or more operands this
can greatly increase the computational efficiency at the cost of a larger
memory footprint during computation.

Typically a 'greedy' algorithm is applied which empirical tests have shown
returns the optimal path in the majority of cases. In some cases 'optimal'
will return the superlative path through a more expensive, exhaustive search.
For iterative calculations it may be advisable to calculate the optimal path
once and reuse that path by supplying it as an argument. An example is given
below.

See :py:func:`numpy.einsum_path` for more details.

Examples
--------
>>> a = np.arange(25).reshape(5,5)
>>> b = np.arange(5)
>>> c = np.arange(6).reshape(2,3)

Trace of a matrix:

>>> np.einsum('ii', a)
60
>>> np.einsum(a, [0,0])
60
>>> np.trace(a)
60

Extract the diagonal (requires explicit form):

>>> np.einsum('ii->i', a)
array([ 0,  6, 12, 18, 24])
>>> np.einsum(a, [0,0], [0])
array([ 0,  6, 12, 18, 24])
>>> np.diag(a)
array([ 0,  6, 12, 18, 24])

Sum over an axis (requires explicit form):

>>> np.einsum('ij->i', a)
array([ 10,  35,  60,  85, 110])
>>> np.einsum(a, [0,1], [0])
array([ 10,  35,  60,  85, 110])
>>> np.sum(a, axis=1)
array([ 10,  35,  60,  85, 110])

For higher dimensional arrays summing a single axis can be done with ellipsis:

>>> np.einsum('...j->...', a)
array([ 10,  35,  60,  85, 110])
>>> np.einsum(a, [Ellipsis,1], [Ellipsis])
array([ 10,  35,  60,  85, 110])

Compute a matrix transpose, or reorder any number of axes:

>>> np.einsum('ji', c)
array([[0, 3],
       [1, 4],
       [2, 5]])
>>> np.einsum('ij->ji', c)
array([[0, 3],
       [1, 4],
       [2, 5]])
>>> np.einsum(c, [1,0])
array([[0, 3],
       [1, 4],
       [2, 5]])
>>> np.transpose(c)
array([[0, 3],
       [1, 4],
       [2, 5]])

Vector inner products:

>>> np.einsum('i,i', b, b)
30
>>> np.einsum(b, [0], b, [0])
30
>>> np.inner(b,b)
30

Matrix vector multiplication:

>>> np.einsum('ij,j', a, b)
array([ 30,  80, 130, 180, 230])
>>> np.einsum(a, [0,1], b, [1])
array([ 30,  80, 130, 180, 230])
>>> np.dot(a, b)
array([ 30,  80, 130, 180, 230])
>>> np.einsum('...j,j', a, b)
array([ 30,  80, 130, 180, 230])

Broadcasting and scalar multiplication:

>>> np.einsum('..., ...', 3, c)
array([[ 0,  3,  6],
       [ 9, 12, 15]])
>>> np.einsum(',ij', 3, c)
array([[ 0,  3,  6],
       [ 9, 12, 15]])
>>> np.einsum(3, [Ellipsis], c, [Ellipsis])
array([[ 0,  3,  6],
       [ 9, 12, 15]])
>>> np.multiply(3, c)
array([[ 0,  3,  6],
       [ 9, 12, 15]])

Vector outer product:

>>> np.einsum('i,j', np.arange(2)+1, b)
array([[0, 1, 2, 3, 4],
       [0, 2, 4, 6, 8]])
>>> np.einsum(np.arange(2)+1, [0], b, [1])
array([[0, 1, 2, 3, 4],
       [0, 2, 4, 6, 8]])
>>> np.outer(np.arange(2)+1, b)
array([[0, 1, 2, 3, 4],
       [0, 2, 4, 6, 8]])

Tensor contraction:

>>> a = np.arange(60.).reshape(3,4,5)
>>> b = np.arange(24.).reshape(4,3,2)
>>> np.einsum('ijk,jil->kl', a, b)
array([[4400., 4730.],
       [4532., 4874.],
       [4664., 5018.],
       [4796., 5162.],
       [4928., 5306.]])
>>> np.einsum(a, [0,1,2], b, [1,0,3], [2,3])
array([[4400., 4730.],
       [4532., 4874.],
       [4664., 5018.],
       [4796., 5162.],
       [4928., 5306.]])
>>> np.tensordot(a,b, axes=([1,0],[0,1]))
array([[4400., 4730.],
       [4532., 4874.],
       [4664., 5018.],
       [4796., 5162.],
       [4928., 5306.]])

Writeable returned arrays (since version 1.10.0):

>>> a = np.zeros((3, 3))
>>> np.einsum('ii->i', a)[:] = 1
>>> a
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])

Example of ellipsis use:

>>> a = np.arange(6).reshape((3,2))
>>> b = np.arange(12).reshape((4,3))
>>> np.einsum('ki,jk->ij', a, b)
array([[10, 28, 46, 64],
       [13, 40, 67, 94]])
>>> np.einsum('ki,...k->i...', a, b)
array([[10, 28, 46, 64],
       [13, 40, 67, 94]])
>>> np.einsum('k...,jk', a, b)
array([[10, 28, 46, 64],
       [13, 40, 67, 94]])

Chained array operations. For more complicated contractions, speed ups
might be achieved by repeatedly computing a 'greedy' path or pre-computing the
'optimal' path and repeatedly applying it, using an
`einsum_path` insertion (since version 1.12.0). Performance improvements can be
particularly significant with larger arrays:

>>> a = np.ones(64).reshape(2,4,8)

Basic `einsum`: ~1520ms  (benchmarked on 3.1GHz Intel i5.)

>>> for iteration in range(500):
...     _ = np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a)

Sub-optimal `einsum` (due to repeated path calculation time): ~330ms

>>> for iteration in range(500):
...     _ = np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a, optimize='optimal')

Greedy `einsum` (faster optimal path approximation): ~160ms

>>> for iteration in range(500):
...     _ = np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a, optimize='greedy')

Optimal `einsum` (best usage pattern in some use cases): ~110ms

>>> path = np.einsum_path('ijk,ilm,njm,nlk,abc->',a,a,a,a,a, optimize='optimal')[0]
>>> for iteration in range(500):
...     _ = np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a, optimize=path)
*)

val empty : ?dtype:Np.Dtype.t -> ?order:[`C | `F] -> shape:[`I of int | `Tuple_of_int of Py.Object.t] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
empty(shape, dtype=float, order='C')

Return a new array of given shape and type, without initializing entries.

Parameters
----------
shape : int or tuple of int
    Shape of the empty array, e.g., ``(2, 3)`` or ``2``.
dtype : data-type, optional
    Desired output data-type for the array, e.g, `numpy.int8`. Default is
    `numpy.float64`.
order : {'C', 'F'}, optional, default: 'C'
    Whether to store multi-dimensional data in row-major
    (C-style) or column-major (Fortran-style) order in
    memory.

Returns
-------
out : ndarray
    Array of uninitialized (arbitrary) data of the given shape, dtype, and
    order.  Object arrays will be initialized to None.

See Also
--------
empty_like : Return an empty array with shape and type of input.
ones : Return a new array setting values to one.
zeros : Return a new array setting values to zero.
full : Return a new array of given shape filled with value.


Notes
-----
`empty`, unlike `zeros`, does not set the array values to zero,
and may therefore be marginally faster.  On the other hand, it requires
the user to manually set all the values in the array, and should be
used with caution.

Examples
--------
>>> np.empty([2, 2])
array([[ -9.74499359e+001,   6.69583040e-309],
       [  2.13182611e-314,   3.06959433e-309]])         #uninitialized

>>> np.empty([2, 2], dtype=int)
array([[-1073741821, -1067949133],
       [  496041986,    19249760]])                     #uninitialized
*)

val eye : ?m:int -> ?k:int -> ?dtype:Np.Dtype.t -> ?order:[`C | `F] -> n:int -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return a 2-D array with ones on the diagonal and zeros elsewhere.

Parameters
----------
N : int
  Number of rows in the output.
M : int, optional
  Number of columns in the output. If None, defaults to `N`.
k : int, optional
  Index of the diagonal: 0 (the default) refers to the main diagonal,
  a positive value refers to an upper diagonal, and a negative value
  to a lower diagonal.
dtype : data-type, optional
  Data-type of the returned array.
order : {'C', 'F'}, optional
    Whether the output should be stored in row-major (C-style) or
    column-major (Fortran-style) order in memory.

    .. versionadded:: 1.14.0

Returns
-------
I : ndarray of shape (N,M)
  An array where all elements are equal to zero, except for the `k`-th
  diagonal, whose values are equal to one.

See Also
--------
identity : (almost) equivalent function
diag : diagonal 2-D array from a 1-D array specified by the user.

Examples
--------
>>> np.eye(2, dtype=int)
array([[1, 0],
       [0, 1]])
>>> np.eye(3, k=1)
array([[0.,  1.,  0.],
       [0.,  0.,  1.],
       [0.,  0.,  0.]])
*)

val flatnonzero : [>`Ndarray] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return indices that are non-zero in the flattened version of a.

This is equivalent to np.nonzero(np.ravel(a))[0].

Parameters
----------
a : array_like
    Input data.

Returns
-------
res : ndarray
    Output array, containing the indices of the elements of `a.ravel()`
    that are non-zero.

See Also
--------
nonzero : Return the indices of the non-zero elements of the input array.
ravel : Return a 1-D array containing the elements of the input array.

Examples
--------
>>> x = np.arange(-2, 3)
>>> x
array([-2, -1,  0,  1,  2])
>>> np.flatnonzero(x)
array([0, 1, 3, 4])

Use the indices of the non-zero elements as an index array to extract
these elements:

>>> x.ravel()[np.flatnonzero(x)]
array([-2, -1,  1,  2])
*)

val get_lapack_funcs : ?arrays:[>`Ndarray] Np.Obj.t list -> ?dtype:[`S of string | `Dtype of Np.Dtype.t] -> names:[`Sequence_of_str of Py.Object.t | `S of string] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return available LAPACK function objects from names.

Arrays are used to determine the optimal prefix of LAPACK routines.

Parameters
----------
names : str or sequence of str
    Name(s) of LAPACK functions without type prefix.

arrays : sequence of ndarrays, optional
    Arrays can be given to determine optimal prefix of LAPACK
    routines. If not given, double-precision routines will be
    used, otherwise the most generic type in arrays will be used.

dtype : str or dtype, optional
    Data-type specifier. Not used if `arrays` is non-empty.

Returns
-------
funcs : list
    List containing the found function(s).

Notes
-----
This routine automatically chooses between Fortran/C
interfaces. Fortran code is used whenever possible for arrays with
column major order. In all other cases, C code is preferred.

In LAPACK, the naming convention is that all functions start with a
type prefix, which depends on the type of the principal
matrix. These can be one of {'s', 'd', 'c', 'z'} for the numpy
types {float32, float64, complex64, complex128} respectively, and
are stored in attribute ``typecode`` of the returned functions.

Examples
--------
Suppose we would like to use '?lange' routine which computes the selected
norm of an array. We pass our array in order to get the correct 'lange'
flavor.

>>> import scipy.linalg as LA
>>> a = np.random.rand(3,2)
>>> x_lange = LA.get_lapack_funcs('lange', (a,))
>>> x_lange.typecode
'd'
>>> x_lange = LA.get_lapack_funcs('lange',(a*1j,))
>>> x_lange.typecode
'z'

Several LAPACK routines work best when its internal WORK array has
the optimal size (big enough for fast computation and small enough to
avoid waste of memory). This size is determined also by a dedicated query
to the function which is often wrapped as a standalone function and
commonly denoted as ``###_lwork``. Below is an example for ``?sysv``

>>> import scipy.linalg as LA
>>> a = np.random.rand(1000,1000)
>>> b = np.random.rand(1000,1)*1j
>>> # We pick up zsysv and zsysv_lwork due to b array
... xsysv, xlwork = LA.get_lapack_funcs(('sysv', 'sysv_lwork'), (a, b))
>>> opt_lwork, _ = xlwork(a.shape[0])  # returns a complex for 'z' prefix
>>> udut, ipiv, x, info = xsysv(a, b, lwork=int(opt_lwork.real))
*)

val hessenberg : ?calc_q:bool -> ?overwrite_a:bool -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Compute Hessenberg form of a matrix.

The Hessenberg decomposition is::

    A = Q H Q^H

where `Q` is unitary/orthogonal and `H` has only zero elements below
the first sub-diagonal.

Parameters
----------
a : (M, M) array_like
    Matrix to bring into Hessenberg form.
calc_q : bool, optional
    Whether to compute the transformation matrix.  Default is False.
overwrite_a : bool, optional
    Whether to overwrite `a`; may improve performance.
    Default is False.
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
H : (M, M) ndarray
    Hessenberg form of `a`.
Q : (M, M) ndarray
    Unitary/orthogonal similarity transformation matrix ``A = Q H Q^H``.
    Only returned if ``calc_q=True``.

Examples
--------
>>> from scipy.linalg import hessenberg
>>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
>>> H, Q = hessenberg(A, calc_q=True)
>>> H
array([[  2.        , -11.65843866,   1.42005301,   0.25349066],
       [ -9.94987437,  14.53535354,  -5.31022304,   2.43081618],
       [  0.        ,  -1.83299243,   0.38969961,  -0.51527034],
       [  0.        ,   0.        ,  -3.83189513,   1.07494686]])
>>> np.allclose(Q @ H @ Q.conj().T - A, np.zeros((4, 4)))
True
*)

val iscomplex : [>`Ndarray] Np.Obj.t -> Py.Object.t
(**
Returns a bool array, where True if input element is complex.

What is tested is whether the input has a non-zero imaginary part, not if
the input type is complex.

Parameters
----------
x : array_like
    Input array.

Returns
-------
out : ndarray of bools
    Output array.

See Also
--------
isreal
iscomplexobj : Return True if x is a complex type or an array of complex
               numbers.

Examples
--------
>>> np.iscomplex([1+1j, 1+0j, 4.5, 3, 2, 2j])
array([ True, False, False, False, False,  True])
*)

val iscomplexobj : Py.Object.t -> bool
(**
Check for a complex type or an array of complex numbers.

The type of the input is checked, not the value. Even if the input
has an imaginary part equal to zero, `iscomplexobj` evaluates to True.

Parameters
----------
x : any
    The input can be of any type and shape.

Returns
-------
iscomplexobj : bool
    The return value, True if `x` is of a complex type or has at least
    one complex element.

See Also
--------
isrealobj, iscomplex

Examples
--------
>>> np.iscomplexobj(1)
False
>>> np.iscomplexobj(1+0j)
True
>>> np.iscomplexobj([3, 1+0j, True])
True
*)

val isfinite : ?out:[`Ndarray of [>`Ndarray] Np.Obj.t | `Tuple_of_ndarray_and_None of Py.Object.t] -> ?where:[>`Ndarray] Np.Obj.t -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
isfinite(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

Test element-wise for finiteness (not infinity or not Not a Number).

The result is returned as a boolean array.

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
y : ndarray, bool
    True where ``x`` is not positive infinity, negative infinity,
    or NaN; false otherwise.
    This is a scalar if `x` is a scalar.

See Also
--------
isinf, isneginf, isposinf, isnan

Notes
-----
Not a Number, positive infinity and negative infinity are considered
to be non-finite.

NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
(IEEE 754). This means that Not a Number is not equivalent to infinity.
Also that positive infinity is not equivalent to negative infinity. But
infinity is equivalent to positive infinity.  Errors result if the
second argument is also supplied when `x` is a scalar input, or if
first and second arguments have different shapes.

Examples
--------
>>> np.isfinite(1)
True
>>> np.isfinite(0)
True
>>> np.isfinite(np.nan)
False
>>> np.isfinite(np.inf)
False
>>> np.isfinite(np.NINF)
False
>>> np.isfinite([np.log(-1.),1.,np.log(0)])
array([False,  True, False])

>>> x = np.array([-np.inf, 0., np.inf])
>>> y = np.array([2, 2, 2])
>>> np.isfinite(x, y)
array([0, 1, 0])
>>> y
array([0, 1, 0])
*)

val nonzero : [>`Ndarray] Np.Obj.t -> Py.Object.t
(**
Return the indices of the elements that are non-zero.

Returns a tuple of arrays, one for each dimension of `a`,
containing the indices of the non-zero elements in that
dimension. The values in `a` are always tested and returned in
row-major, C-style order.

To group the indices by element, rather than dimension, use `argwhere`,
which returns a row for each non-zero element.

.. note::

   When called on a zero-d array or scalar, ``nonzero(a)`` is treated
   as ``nonzero(atleast1d(a))``.

   .. deprecated:: 1.17.0

      Use `atleast1d` explicitly if this behavior is deliberate.

Parameters
----------
a : array_like
    Input array.

Returns
-------
tuple_of_arrays : tuple
    Indices of elements that are non-zero.

See Also
--------
flatnonzero :
    Return indices that are non-zero in the flattened version of the input
    array.
ndarray.nonzero :
    Equivalent ndarray method.
count_nonzero :
    Counts the number of non-zero elements in the input array.

Notes
-----
While the nonzero values can be obtained with ``a[nonzero(a)]``, it is
recommended to use ``x[x.astype(bool)]`` or ``x[x != 0]`` instead, which
will correctly handle 0-d arrays.

Examples
--------
>>> x = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
>>> x
array([[3, 0, 0],
       [0, 4, 0],
       [5, 6, 0]])
>>> np.nonzero(x)
(array([0, 1, 2, 2]), array([0, 1, 0, 1]))

>>> x[np.nonzero(x)]
array([3, 4, 5, 6])
>>> np.transpose(np.nonzero(x))
array([[0, 0],
       [1, 1],
       [2, 0],
       [2, 1]])

A common use for ``nonzero`` is to find the indices of an array, where
a condition is True.  Given an array `a`, the condition `a` > 3 is a
boolean array and since False is interpreted as 0, np.nonzero(a > 3)
yields the indices of the `a` where the condition is true.

>>> a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
>>> a > 3
array([[False, False, False],
       [ True,  True,  True],
       [ True,  True,  True]])
>>> np.nonzero(a > 3)
(array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))

Using this result to index `a` is equivalent to using the mask directly:

>>> a[np.nonzero(a > 3)]
array([4, 5, 6, 7, 8, 9])
>>> a[a > 3]  # prefer this spelling
array([4, 5, 6, 7, 8, 9])

``nonzero`` can also be called as a method of the array.

>>> (a > 3).nonzero()
(array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))
*)

val norm : ?ord:[`Fro | `PyObject of Py.Object.t] -> ?axis:[`T2_tuple_of_ints of Py.Object.t | `I of int] -> ?keepdims:bool -> ?check_finite:bool -> a:Py.Object.t -> unit -> Py.Object.t
(**
Matrix or vector norm.

This function is able to return one of seven different matrix norms,
or one of an infinite number of vector norms (described below), depending
on the value of the ``ord`` parameter.

Parameters
----------
a : (M,) or (M, N) array_like
    Input array.  If `axis` is None, `a` must be 1-D or 2-D.
ord : {non-zero int, inf, -inf, 'fro'}, optional
    Order of the norm (see table under ``Notes``). inf means numpy's
    `inf` object
axis : {int, 2-tuple of ints, None}, optional
    If `axis` is an integer, it specifies the axis of `a` along which to
    compute the vector norms.  If `axis` is a 2-tuple, it specifies the
    axes that hold 2-D matrices, and the matrix norms of these matrices
    are computed.  If `axis` is None then either a vector norm (when `a`
    is 1-D) or a matrix norm (when `a` is 2-D) is returned.
keepdims : bool, optional
    If this is set to True, the axes which are normed over are left in the
    result as dimensions with size one.  With this option the result will
    broadcast correctly against the original `a`.
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
n : float or ndarray
    Norm of the matrix or vector(s).

Notes
-----
For values of ``ord <= 0``, the result is, strictly speaking, not a
mathematical 'norm', but it may still be useful for various numerical
purposes.

The following norms can be calculated:

=====  ============================  ==========================
ord    norm for matrices             norm for vectors
=====  ============================  ==========================
None   Frobenius norm                2-norm
'fro'  Frobenius norm                --
inf    max(sum(abs(x), axis=1))      max(abs(x))
-inf   min(sum(abs(x), axis=1))      min(abs(x))
0      --                            sum(x != 0)
1      max(sum(abs(x), axis=0))      as below
-1     min(sum(abs(x), axis=0))      as below
2      2-norm (largest sing. value)  as below
-2     smallest singular value       as below
other  --                            sum(abs(x)**ord)**(1./ord)
=====  ============================  ==========================

The Frobenius norm is given by [1]_:

    :math:`||A||_F = [\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

The ``axis`` and ``keepdims`` arguments are passed directly to
``numpy.linalg.norm`` and are only usable if they are supported
by the version of numpy in use.

References
----------
.. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
       Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15

Examples
--------
>>> from scipy.linalg import norm
>>> a = np.arange(9) - 4.0
>>> a
array([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
>>> b = a.reshape((3, 3))
>>> b
array([[-4., -3., -2.],
       [-1.,  0.,  1.],
       [ 2.,  3.,  4.]])

>>> norm(a)
7.745966692414834
>>> norm(b)
7.745966692414834
>>> norm(b, 'fro')
7.745966692414834
>>> norm(a, np.inf)
4
>>> norm(b, np.inf)
9
>>> norm(a, -np.inf)
0
>>> norm(b, -np.inf)
2

>>> norm(a, 1)
20
>>> norm(b, 1)
7
>>> norm(a, -1)
-4.6566128774142013e-010
>>> norm(b, -1)
6
>>> norm(a, 2)
7.745966692414834
>>> norm(b, 2)
7.3484692283495345

>>> norm(a, -2)
0
>>> norm(b, -2)
1.8570331885190563e-016
>>> norm(a, 3)
5.8480354764257312
>>> norm(a, -3)
0
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

module Decomp_cholesky : sig
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

val asarray_chkfinite : ?dtype:Np.Dtype.t -> ?order:[`F | `C] -> a:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Convert the input to an array, checking for NaNs or Infs.

Parameters
----------
a : array_like
    Input data, in any form that can be converted to an array.  This
    includes lists, lists of tuples, tuples, tuples of tuples, tuples
    of lists and ndarrays.  Success requires no NaNs or Infs.
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
    is already an ndarray.  If `a` is a subclass of ndarray, a base
    class ndarray is returned.

Raises
------
ValueError
    Raises ValueError if `a` contains NaN (Not a Number) or Inf (Infinity).

See Also
--------
asarray : Create and array.
asanyarray : Similar function which passes through subclasses.
ascontiguousarray : Convert input to a contiguous array.
asfarray : Convert input to a floating point ndarray.
asfortranarray : Convert input to an ndarray with column-major
                 memory order.
fromiter : Create an array from an iterator.
fromfunction : Construct an array by executing a function on grid
               positions.

Examples
--------
Convert a list into an array.  If all elements are finite
``asarray_chkfinite`` is identical to ``asarray``.

>>> a = [1, 2]
>>> np.asarray_chkfinite(a, dtype=float)
array([1., 2.])

Raises ValueError if array_like contains Nans or Infs.

>>> a = [1, 2, np.inf]
>>> try:
...     np.asarray_chkfinite(a)
... except ValueError:
...     print('ValueError')
...
ValueError
*)

val atleast_2d : Py.Object.t list -> Py.Object.t
(**
View inputs as arrays with at least two dimensions.

Parameters
----------
arys1, arys2, ... : array_like
    One or more array-like sequences.  Non-array inputs are converted
    to arrays.  Arrays that already have two or more dimensions are
    preserved.

Returns
-------
res, res2, ... : ndarray
    An array, or list of arrays, each with ``a.ndim >= 2``.
    Copies are avoided where possible, and views with two or more
    dimensions are returned.

See Also
--------
atleast_1d, atleast_3d

Examples
--------
>>> np.atleast_2d(3.0)
array([[3.]])

>>> x = np.arange(3.0)
>>> np.atleast_2d(x)
array([[0., 1., 2.]])
>>> np.atleast_2d(x).base is x
True

>>> np.atleast_2d(1, [1, 2], [[1, 2]])
[array([[1]]), array([[1, 2]]), array([[1, 2]])]
*)

val cho_factor : ?lower:bool -> ?overwrite_a:bool -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * bool)
(**
Compute the Cholesky decomposition of a matrix, to use in cho_solve

Returns a matrix containing the Cholesky decomposition,
``A = L L*`` or ``A = U* U`` of a Hermitian positive-definite matrix `a`.
The return value can be directly used as the first parameter to cho_solve.

.. warning::
    The returned matrix also contains random data in the entries not
    used by the Cholesky decomposition. If you need to zero these
    entries, use the function `cholesky` instead.

Parameters
----------
a : (M, M) array_like
    Matrix to be decomposed
lower : bool, optional
    Whether to compute the upper or lower triangular Cholesky factorization
    (Default: upper-triangular)
overwrite_a : bool, optional
    Whether to overwrite data in a (may improve performance)
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
c : (M, M) ndarray
    Matrix whose upper or lower triangle contains the Cholesky factor
    of `a`. Other parts of the matrix contain random data.
lower : bool
    Flag indicating whether the factor is in the lower or upper triangle

Raises
------
LinAlgError
    Raised if decomposition fails.

See also
--------
cho_solve : Solve a linear set equations using the Cholesky factorization
            of a matrix.

Examples
--------
>>> from scipy.linalg import cho_factor
>>> A = np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]])
>>> c, low = cho_factor(A)
>>> c
array([[3.        , 1.        , 0.33333333, 1.66666667],
       [3.        , 2.44948974, 1.90515869, -0.27216553],
       [1.        , 5.        , 2.29330749, 0.8559528 ],
       [5.        , 1.        , 2.        , 1.55418563]])
>>> np.allclose(np.triu(c).T @ np. triu(c) - A, np.zeros((4, 4)))
True
*)

val cho_solve : ?overwrite_b:bool -> ?check_finite:bool -> c_and_lower:Py.Object.t -> b:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Solve the linear equations A x = b, given the Cholesky factorization of A.

Parameters
----------
(c, lower) : tuple, (array, bool)
    Cholesky factorization of a, as given by cho_factor
b : array
    Right-hand side
overwrite_b : bool, optional
    Whether to overwrite data in b (may improve performance)
check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
x : array
    The solution to the system A x = b

See also
--------
cho_factor : Cholesky factorization of a matrix

Examples
--------
>>> from scipy.linalg import cho_factor, cho_solve
>>> A = np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]])
>>> c, low = cho_factor(A)
>>> x = cho_solve((c, low), [1, 1, 1, 1])
>>> np.allclose(A @ x - [1, 1, 1, 1], np.zeros(4))
True
*)

val cho_solve_banded : ?overwrite_b:bool -> ?check_finite:bool -> cb_and_lower:Py.Object.t -> b:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Solve the linear equations ``A x = b``, given the Cholesky factorization of
the banded hermitian ``A``.

Parameters
----------
(cb, lower) : tuple, (ndarray, bool)
    `cb` is the Cholesky factorization of A, as given by cholesky_banded.
    `lower` must be the same value that was given to cholesky_banded.
b : array_like
    Right-hand side
overwrite_b : bool, optional
    If True, the function will overwrite the values in `b`.
check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
x : array
    The solution to the system A x = b

See also
--------
cholesky_banded : Cholesky factorization of a banded matrix

Notes
-----

.. versionadded:: 0.8.0

Examples
--------
>>> from scipy.linalg import cholesky_banded, cho_solve_banded
>>> Ab = np.array([[0, 0, 1j, 2, 3j], [0, -1, -2, 3, 4], [9, 8, 7, 6, 9]])
>>> A = np.diag(Ab[0,2:], k=2) + np.diag(Ab[1,1:], k=1)
>>> A = A + A.conj().T + np.diag(Ab[2, :])
>>> c = cholesky_banded(Ab)
>>> x = cho_solve_banded((c, False), np.ones(5))
>>> np.allclose(A @ x - np.ones(5), np.zeros(5))
True
*)

val cholesky : ?lower:bool -> ?overwrite_a:bool -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the Cholesky decomposition of a matrix.

Returns the Cholesky decomposition, :math:`A = L L^*` or
:math:`A = U^* U` of a Hermitian positive-definite matrix A.

Parameters
----------
a : (M, M) array_like
    Matrix to be decomposed
lower : bool, optional
    Whether to compute the upper or lower triangular Cholesky
    factorization.  Default is upper-triangular.
overwrite_a : bool, optional
    Whether to overwrite data in `a` (may improve performance).
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
c : (M, M) ndarray
    Upper- or lower-triangular Cholesky factor of `a`.

Raises
------
LinAlgError : if decomposition fails.

Examples
--------
>>> from scipy.linalg import cholesky
>>> a = np.array([[1,-2j],[2j,5]])
>>> L = cholesky(a, lower=True)
>>> L
array([[ 1.+0.j,  0.+0.j],
       [ 0.+2.j,  1.+0.j]])
>>> L @ L.T.conj()
array([[ 1.+0.j,  0.-2.j],
       [ 0.+2.j,  5.+0.j]])
*)

val cholesky_banded : ?overwrite_ab:bool -> ?lower:bool -> ?check_finite:bool -> ab:Py.Object.t -> unit -> Py.Object.t
(**
Cholesky decompose a banded Hermitian positive-definite matrix

The matrix a is stored in ab either in lower diagonal or upper
diagonal ordered form::

    ab[u + i - j, j] == a[i,j]        (if upper form; i <= j)
    ab[    i - j, j] == a[i,j]        (if lower form; i >= j)

Example of ab (shape of a is (6,6), u=2)::

    upper form:
    *   *   a02 a13 a24 a35
    *   a01 a12 a23 a34 a45
    a00 a11 a22 a33 a44 a55

    lower form:
    a00 a11 a22 a33 a44 a55
    a10 a21 a32 a43 a54 *
    a20 a31 a42 a53 *   *

Parameters
----------
ab : (u + 1, M) array_like
    Banded matrix
overwrite_ab : bool, optional
    Discard data in ab (may enhance performance)
lower : bool, optional
    Is the matrix in the lower form. (Default is upper form)
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
c : (u + 1, M) ndarray
    Cholesky factorization of a, in the same banded format as ab
    
See also
--------
cho_solve_banded : Solve a linear set equations, given the Cholesky factorization
            of a banded hermitian.

Examples
--------
>>> from scipy.linalg import cholesky_banded
>>> from numpy import allclose, zeros, diag
>>> Ab = np.array([[0, 0, 1j, 2, 3j], [0, -1, -2, 3, 4], [9, 8, 7, 6, 9]])
>>> A = np.diag(Ab[0,2:], k=2) + np.diag(Ab[1,1:], k=1)
>>> A = A + A.conj().T + np.diag(Ab[2, :])
>>> c = cholesky_banded(Ab)
>>> C = np.diag(c[0, 2:], k=2) + np.diag(c[1, 1:], k=1) + np.diag(c[2, :])
>>> np.allclose(C.conj().T @ C - A, np.zeros((5, 5)))
True
*)

val get_lapack_funcs : ?arrays:[>`Ndarray] Np.Obj.t list -> ?dtype:[`S of string | `Dtype of Np.Dtype.t] -> names:[`Sequence_of_str of Py.Object.t | `S of string] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return available LAPACK function objects from names.

Arrays are used to determine the optimal prefix of LAPACK routines.

Parameters
----------
names : str or sequence of str
    Name(s) of LAPACK functions without type prefix.

arrays : sequence of ndarrays, optional
    Arrays can be given to determine optimal prefix of LAPACK
    routines. If not given, double-precision routines will be
    used, otherwise the most generic type in arrays will be used.

dtype : str or dtype, optional
    Data-type specifier. Not used if `arrays` is non-empty.

Returns
-------
funcs : list
    List containing the found function(s).

Notes
-----
This routine automatically chooses between Fortran/C
interfaces. Fortran code is used whenever possible for arrays with
column major order. In all other cases, C code is preferred.

In LAPACK, the naming convention is that all functions start with a
type prefix, which depends on the type of the principal
matrix. These can be one of {'s', 'd', 'c', 'z'} for the numpy
types {float32, float64, complex64, complex128} respectively, and
are stored in attribute ``typecode`` of the returned functions.

Examples
--------
Suppose we would like to use '?lange' routine which computes the selected
norm of an array. We pass our array in order to get the correct 'lange'
flavor.

>>> import scipy.linalg as LA
>>> a = np.random.rand(3,2)
>>> x_lange = LA.get_lapack_funcs('lange', (a,))
>>> x_lange.typecode
'd'
>>> x_lange = LA.get_lapack_funcs('lange',(a*1j,))
>>> x_lange.typecode
'z'

Several LAPACK routines work best when its internal WORK array has
the optimal size (big enough for fast computation and small enough to
avoid waste of memory). This size is determined also by a dedicated query
to the function which is often wrapped as a standalone function and
commonly denoted as ``###_lwork``. Below is an example for ``?sysv``

>>> import scipy.linalg as LA
>>> a = np.random.rand(1000,1000)
>>> b = np.random.rand(1000,1)*1j
>>> # We pick up zsysv and zsysv_lwork due to b array
... xsysv, xlwork = LA.get_lapack_funcs(('sysv', 'sysv_lwork'), (a, b))
>>> opt_lwork, _ = xlwork(a.shape[0])  # returns a complex for 'z' prefix
>>> udut, ipiv, x, info = xsysv(a, b, lwork=int(opt_lwork.real))
*)


end

module Decomp_lu : sig
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

val asarray_chkfinite : ?dtype:Np.Dtype.t -> ?order:[`F | `C] -> a:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Convert the input to an array, checking for NaNs or Infs.

Parameters
----------
a : array_like
    Input data, in any form that can be converted to an array.  This
    includes lists, lists of tuples, tuples, tuples of tuples, tuples
    of lists and ndarrays.  Success requires no NaNs or Infs.
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
    is already an ndarray.  If `a` is a subclass of ndarray, a base
    class ndarray is returned.

Raises
------
ValueError
    Raises ValueError if `a` contains NaN (Not a Number) or Inf (Infinity).

See Also
--------
asarray : Create and array.
asanyarray : Similar function which passes through subclasses.
ascontiguousarray : Convert input to a contiguous array.
asfarray : Convert input to a floating point ndarray.
asfortranarray : Convert input to an ndarray with column-major
                 memory order.
fromiter : Create an array from an iterator.
fromfunction : Construct an array by executing a function on grid
               positions.

Examples
--------
Convert a list into an array.  If all elements are finite
``asarray_chkfinite`` is identical to ``asarray``.

>>> a = [1, 2]
>>> np.asarray_chkfinite(a, dtype=float)
array([1., 2.])

Raises ValueError if array_like contains Nans or Infs.

>>> a = [1, 2, np.inf]
>>> try:
...     np.asarray_chkfinite(a)
... except ValueError:
...     print('ValueError')
...
ValueError
*)

val get_flinalg_funcs : ?arrays:Py.Object.t -> ?debug:Py.Object.t -> names:Py.Object.t -> unit -> Py.Object.t
(**
Return optimal available _flinalg function objects with
names. arrays are used to determine optimal prefix.
*)

val get_lapack_funcs : ?arrays:[>`Ndarray] Np.Obj.t list -> ?dtype:[`S of string | `Dtype of Np.Dtype.t] -> names:[`Sequence_of_str of Py.Object.t | `S of string] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return available LAPACK function objects from names.

Arrays are used to determine the optimal prefix of LAPACK routines.

Parameters
----------
names : str or sequence of str
    Name(s) of LAPACK functions without type prefix.

arrays : sequence of ndarrays, optional
    Arrays can be given to determine optimal prefix of LAPACK
    routines. If not given, double-precision routines will be
    used, otherwise the most generic type in arrays will be used.

dtype : str or dtype, optional
    Data-type specifier. Not used if `arrays` is non-empty.

Returns
-------
funcs : list
    List containing the found function(s).

Notes
-----
This routine automatically chooses between Fortran/C
interfaces. Fortran code is used whenever possible for arrays with
column major order. In all other cases, C code is preferred.

In LAPACK, the naming convention is that all functions start with a
type prefix, which depends on the type of the principal
matrix. These can be one of {'s', 'd', 'c', 'z'} for the numpy
types {float32, float64, complex64, complex128} respectively, and
are stored in attribute ``typecode`` of the returned functions.

Examples
--------
Suppose we would like to use '?lange' routine which computes the selected
norm of an array. We pass our array in order to get the correct 'lange'
flavor.

>>> import scipy.linalg as LA
>>> a = np.random.rand(3,2)
>>> x_lange = LA.get_lapack_funcs('lange', (a,))
>>> x_lange.typecode
'd'
>>> x_lange = LA.get_lapack_funcs('lange',(a*1j,))
>>> x_lange.typecode
'z'

Several LAPACK routines work best when its internal WORK array has
the optimal size (big enough for fast computation and small enough to
avoid waste of memory). This size is determined also by a dedicated query
to the function which is often wrapped as a standalone function and
commonly denoted as ``###_lwork``. Below is an example for ``?sysv``

>>> import scipy.linalg as LA
>>> a = np.random.rand(1000,1000)
>>> b = np.random.rand(1000,1)*1j
>>> # We pick up zsysv and zsysv_lwork due to b array
... xsysv, xlwork = LA.get_lapack_funcs(('sysv', 'sysv_lwork'), (a, b))
>>> opt_lwork, _ = xlwork(a.shape[0])  # returns a complex for 'z' prefix
>>> udut, ipiv, x, info = xsysv(a, b, lwork=int(opt_lwork.real))
*)

val lu : ?permute_l:bool -> ?overwrite_a:bool -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * Py.Object.t * Py.Object.t * Py.Object.t)
(**
Compute pivoted LU decomposition of a matrix.

The decomposition is::

    A = P L U

where P is a permutation matrix, L lower triangular with unit
diagonal elements, and U upper triangular.

Parameters
----------
a : (M, N) array_like
    Array to decompose
permute_l : bool, optional
    Perform the multiplication P*L  (Default: do not permute)
overwrite_a : bool, optional
    Whether to overwrite data in a (may improve performance)
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
**(If permute_l == False)**

p : (M, M) ndarray
    Permutation matrix
l : (M, K) ndarray
    Lower triangular or trapezoidal matrix with unit diagonal.
    K = min(M, N)
u : (K, N) ndarray
    Upper triangular or trapezoidal matrix

**(If permute_l == True)**

pl : (M, K) ndarray
    Permuted L matrix.
    K = min(M, N)
u : (K, N) ndarray
    Upper triangular or trapezoidal matrix

Notes
-----
This is a LU factorization routine written for SciPy.

Examples
--------
>>> from scipy.linalg import lu
>>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
>>> p, l, u = lu(A)
>>> np.allclose(A - p @ l @ u, np.zeros((4, 4)))
True
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

val warn : ?category:Py.Object.t -> ?stacklevel:Py.Object.t -> ?source:Py.Object.t -> message:Py.Object.t -> unit -> Py.Object.t
(**
Issue a warning, or maybe ignore it or raise an exception.
*)


end

module Decomp_qr : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val get_lapack_funcs : ?arrays:[>`Ndarray] Np.Obj.t list -> ?dtype:[`S of string | `Dtype of Np.Dtype.t] -> names:[`Sequence_of_str of Py.Object.t | `S of string] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return available LAPACK function objects from names.

Arrays are used to determine the optimal prefix of LAPACK routines.

Parameters
----------
names : str or sequence of str
    Name(s) of LAPACK functions without type prefix.

arrays : sequence of ndarrays, optional
    Arrays can be given to determine optimal prefix of LAPACK
    routines. If not given, double-precision routines will be
    used, otherwise the most generic type in arrays will be used.

dtype : str or dtype, optional
    Data-type specifier. Not used if `arrays` is non-empty.

Returns
-------
funcs : list
    List containing the found function(s).

Notes
-----
This routine automatically chooses between Fortran/C
interfaces. Fortran code is used whenever possible for arrays with
column major order. In all other cases, C code is preferred.

In LAPACK, the naming convention is that all functions start with a
type prefix, which depends on the type of the principal
matrix. These can be one of {'s', 'd', 'c', 'z'} for the numpy
types {float32, float64, complex64, complex128} respectively, and
are stored in attribute ``typecode`` of the returned functions.

Examples
--------
Suppose we would like to use '?lange' routine which computes the selected
norm of an array. We pass our array in order to get the correct 'lange'
flavor.

>>> import scipy.linalg as LA
>>> a = np.random.rand(3,2)
>>> x_lange = LA.get_lapack_funcs('lange', (a,))
>>> x_lange.typecode
'd'
>>> x_lange = LA.get_lapack_funcs('lange',(a*1j,))
>>> x_lange.typecode
'z'

Several LAPACK routines work best when its internal WORK array has
the optimal size (big enough for fast computation and small enough to
avoid waste of memory). This size is determined also by a dedicated query
to the function which is often wrapped as a standalone function and
commonly denoted as ``###_lwork``. Below is an example for ``?sysv``

>>> import scipy.linalg as LA
>>> a = np.random.rand(1000,1000)
>>> b = np.random.rand(1000,1)*1j
>>> # We pick up zsysv and zsysv_lwork due to b array
... xsysv, xlwork = LA.get_lapack_funcs(('sysv', 'sysv_lwork'), (a, b))
>>> opt_lwork, _ = xlwork(a.shape[0])  # returns a complex for 'z' prefix
>>> udut, ipiv, x, info = xsysv(a, b, lwork=int(opt_lwork.real))
*)

val qr : ?overwrite_a:bool -> ?lwork:int -> ?mode:[`Full | `R | `Economic | `Raw] -> ?pivoting:bool -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> (Py.Object.t * Py.Object.t * Py.Object.t)
(**
Compute QR decomposition of a matrix.

Calculate the decomposition ``A = Q R`` where Q is unitary/orthogonal
and R upper triangular.

Parameters
----------
a : (M, N) array_like
    Matrix to be decomposed
overwrite_a : bool, optional
    Whether data in a is overwritten (may improve performance)
lwork : int, optional
    Work array size, lwork >= a.shape[1]. If None or -1, an optimal size
    is computed.
mode : {'full', 'r', 'economic', 'raw'}, optional
    Determines what information is to be returned: either both Q and R
    ('full', default), only R ('r') or both Q and R but computed in
    economy-size ('economic', see Notes). The final option 'raw'
    (added in SciPy 0.11) makes the function return two matrices
    (Q, TAU) in the internal format used by LAPACK.
pivoting : bool, optional
    Whether or not factorization should include pivoting for rank-revealing
    qr decomposition. If pivoting, compute the decomposition
    ``A P = Q R`` as above, but where P is chosen such that the diagonal
    of R is non-increasing.
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
Q : float or complex ndarray
    Of shape (M, M), or (M, K) for ``mode='economic'``.  Not returned
    if ``mode='r'``.
R : float or complex ndarray
    Of shape (M, N), or (K, N) for ``mode='economic'``.  ``K = min(M, N)``.
P : int ndarray
    Of shape (N,) for ``pivoting=True``. Not returned if
    ``pivoting=False``.

Raises
------
LinAlgError
    Raised if decomposition fails

Notes
-----
This is an interface to the LAPACK routines dgeqrf, zgeqrf,
dorgqr, zungqr, dgeqp3, and zgeqp3.

If ``mode=economic``, the shapes of Q and R are (M, K) and (K, N) instead
of (M,M) and (M,N), with ``K=min(M,N)``.

Examples
--------
>>> from scipy import linalg
>>> a = np.random.randn(9, 6)

>>> q, r = linalg.qr(a)
>>> np.allclose(a, np.dot(q, r))
True
>>> q.shape, r.shape
((9, 9), (9, 6))

>>> r2 = linalg.qr(a, mode='r')
>>> np.allclose(r, r2)
True

>>> q3, r3 = linalg.qr(a, mode='economic')
>>> q3.shape, r3.shape
((9, 6), (6, 6))

>>> q4, r4, p4 = linalg.qr(a, pivoting=True)
>>> d = np.abs(np.diag(r4))
>>> np.all(d[1:] <= d[:-1])
True
>>> np.allclose(a[:, p4], np.dot(q4, r4))
True
>>> q4.shape, r4.shape, p4.shape
((9, 9), (9, 6), (6,))

>>> q5, r5, p5 = linalg.qr(a, mode='economic', pivoting=True)
>>> q5.shape, r5.shape, p5.shape
((9, 6), (6, 6), (6,))
*)

val qr_multiply : ?mode:[`Left | `Right] -> ?pivoting:bool -> ?conjugate:bool -> ?overwrite_a:bool -> ?overwrite_c:bool -> a:[>`Ndarray] Np.Obj.t -> c:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * Py.Object.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Calculate the QR decomposition and multiply Q with a matrix.

Calculate the decomposition ``A = Q R`` where Q is unitary/orthogonal
and R upper triangular. Multiply Q with a vector or a matrix c.

Parameters
----------
a : (M, N), array_like
    Input array
c : array_like
    Input array to be multiplied by ``q``.
mode : {'left', 'right'}, optional
    ``Q @ c`` is returned if mode is 'left', ``c @ Q`` is returned if
    mode is 'right'.
    The shape of c must be appropriate for the matrix multiplications,
    if mode is 'left', ``min(a.shape) == c.shape[0]``,
    if mode is 'right', ``a.shape[0] == c.shape[1]``.
pivoting : bool, optional
    Whether or not factorization should include pivoting for rank-revealing
    qr decomposition, see the documentation of qr.
conjugate : bool, optional
    Whether Q should be complex-conjugated. This might be faster
    than explicit conjugation.
overwrite_a : bool, optional
    Whether data in a is overwritten (may improve performance)
overwrite_c : bool, optional
    Whether data in c is overwritten (may improve performance).
    If this is used, c must be big enough to keep the result,
    i.e. ``c.shape[0]`` = ``a.shape[0]`` if mode is 'left'.

Returns
-------
CQ : ndarray
    The product of ``Q`` and ``c``.
R : (K, N), ndarray
    R array of the resulting QR factorization where ``K = min(M, N)``.
P : (N,) ndarray
    Integer pivot array. Only returned when ``pivoting=True``.

Raises
------
LinAlgError
    Raised if QR decomposition fails.

Notes
-----
This is an interface to the LAPACK routines ``?GEQRF``, ``?ORMQR``,
``?UNMQR``, and ``?GEQP3``.

.. versionadded:: 0.11.0

Examples
--------
>>> from scipy.linalg import qr_multiply, qr
>>> A = np.array([[1, 3, 3], [2, 3, 2], [2, 3, 3], [1, 3, 2]])
>>> qc, r1, piv1 = qr_multiply(A, 2*np.eye(4), pivoting=1)
>>> qc
array([[-1.,  1., -1.],
       [-1., -1.,  1.],
       [-1., -1., -1.],
       [-1.,  1.,  1.]])
>>> r1
array([[-6., -3., -5.            ],
       [ 0., -1., -1.11022302e-16],
       [ 0.,  0., -1.            ]])
>>> piv1
array([1, 0, 2], dtype=int32)
>>> q2, r2, piv2 = qr(A, mode='economic', pivoting=1)
>>> np.allclose(2*q2 - qc, np.zeros((4, 3)))
True
*)

val rq : ?overwrite_a:bool -> ?lwork:int -> ?mode:[`Full | `R | `Economic] -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> (Py.Object.t * Py.Object.t)
(**
Compute RQ decomposition of a matrix.

Calculate the decomposition ``A = R Q`` where Q is unitary/orthogonal
and R upper triangular.

Parameters
----------
a : (M, N) array_like
    Matrix to be decomposed
overwrite_a : bool, optional
    Whether data in a is overwritten (may improve performance)
lwork : int, optional
    Work array size, lwork >= a.shape[1]. If None or -1, an optimal size
    is computed.
mode : {'full', 'r', 'economic'}, optional
    Determines what information is to be returned: either both Q and R
    ('full', default), only R ('r') or both Q and R but computed in
    economy-size ('economic', see Notes).
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
R : float or complex ndarray
    Of shape (M, N) or (M, K) for ``mode='economic'``.  ``K = min(M, N)``.
Q : float or complex ndarray
    Of shape (N, N) or (K, N) for ``mode='economic'``.  Not returned
    if ``mode='r'``.

Raises
------
LinAlgError
    If decomposition fails.

Notes
-----
This is an interface to the LAPACK routines sgerqf, dgerqf, cgerqf, zgerqf,
sorgrq, dorgrq, cungrq and zungrq.

If ``mode=economic``, the shapes of Q and R are (K, N) and (M, K) instead
of (N,N) and (M,N), with ``K=min(M,N)``.

Examples
--------
>>> from scipy import linalg
>>> a = np.random.randn(6, 9)
>>> r, q = linalg.rq(a)
>>> np.allclose(a, r @ q)
True
>>> r.shape, q.shape
((6, 9), (9, 9))
>>> r2 = linalg.rq(a, mode='r')
>>> np.allclose(r, r2)
True
>>> r3, q3 = linalg.rq(a, mode='economic')
>>> r3.shape, q3.shape
((6, 6), (6, 9))
*)

val safecall : ?kwargs:(string * Py.Object.t) list -> f:Py.Object.t -> name:Py.Object.t -> Py.Object.t list -> Py.Object.t
(**
Call a LAPACK routine, determining lwork automatically and handling
error return values
*)


end

module Decomp_schur : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module Single : sig
type tag = [`Float32]
type t = [`Float32 | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val __getitem__ : key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return self[key].
*)

val newbyteorder : ?new_order:string -> [> tag] Obj.t -> Np.Dtype.t
(**
newbyteorder(new_order='S')

Return a new `dtype` with a different byte order.

Changes are also made in all fields and sub-arrays of the data type.

The `new_order` code can be any from the following:

* 'S' - swap dtype from current to opposite endian
* {'<', 'L'} - little endian
* {'>', 'B'} - big endian
* {'=', 'N'} - native order
* {'|', 'I'} - ignore (no change to byte order)

Parameters
----------
new_order : str, optional
    Byte order to force; a value from the byte order specifications
    above.  The default value ('S') results in swapping the current
    byte order. The code does a case-insensitive check on the first
    letter of `new_order` for the alternatives above.  For example,
    any of 'B' or 'b' or 'biggish' are valid to specify big-endian.


Returns
-------
new_dtype : dtype
    New `dtype` object with the given change to the byte order.
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

val asarray_chkfinite : ?dtype:Np.Dtype.t -> ?order:[`F | `C] -> a:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Convert the input to an array, checking for NaNs or Infs.

Parameters
----------
a : array_like
    Input data, in any form that can be converted to an array.  This
    includes lists, lists of tuples, tuples, tuples of tuples, tuples
    of lists and ndarrays.  Success requires no NaNs or Infs.
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
    is already an ndarray.  If `a` is a subclass of ndarray, a base
    class ndarray is returned.

Raises
------
ValueError
    Raises ValueError if `a` contains NaN (Not a Number) or Inf (Infinity).

See Also
--------
asarray : Create and array.
asanyarray : Similar function which passes through subclasses.
ascontiguousarray : Convert input to a contiguous array.
asfarray : Convert input to a floating point ndarray.
asfortranarray : Convert input to an ndarray with column-major
                 memory order.
fromiter : Create an array from an iterator.
fromfunction : Construct an array by executing a function on grid
               positions.

Examples
--------
Convert a list into an array.  If all elements are finite
``asarray_chkfinite`` is identical to ``asarray``.

>>> a = [1, 2]
>>> np.asarray_chkfinite(a, dtype=float)
array([1., 2.])

Raises ValueError if array_like contains Nans or Infs.

>>> a = [1, 2, np.inf]
>>> try:
...     np.asarray_chkfinite(a)
... except ValueError:
...     print('ValueError')
...
ValueError
*)

val callable : Py.Object.t -> Py.Object.t
(**
None
*)

val eigvals : ?b:[>`Ndarray] Np.Obj.t -> ?overwrite_a:bool -> ?check_finite:bool -> ?homogeneous_eigvals:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Compute eigenvalues from an ordinary or generalized eigenvalue problem.

Find eigenvalues of a general matrix::

    a   vr[:,i] = w[i]        b   vr[:,i]

Parameters
----------
a : (M, M) array_like
    A complex or real matrix whose eigenvalues and eigenvectors
    will be computed.
b : (M, M) array_like, optional
    Right-hand side matrix in a generalized eigenvalue problem.
    If omitted, identity matrix is assumed.
overwrite_a : bool, optional
    Whether to overwrite data in a (may improve performance)
check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities
    or NaNs.
homogeneous_eigvals : bool, optional
    If True, return the eigenvalues in homogeneous coordinates.
    In this case ``w`` is a (2, M) array so that::

        w[1,i] a vr[:,i] = w[0,i] b vr[:,i]

    Default is False.

Returns
-------
w : (M,) or (2, M) double or complex ndarray
    The eigenvalues, each repeated according to its multiplicity
    but not in any specific order. The shape is (M,) unless
    ``homogeneous_eigvals=True``.

Raises
------
LinAlgError
    If eigenvalue computation does not converge

See Also
--------
eig : eigenvalues and right eigenvectors of general arrays.
eigvalsh : eigenvalues of symmetric or Hermitian arrays
eigvals_banded : eigenvalues for symmetric/Hermitian band matrices
eigvalsh_tridiagonal : eigenvalues of symmetric/Hermitian tridiagonal
    matrices

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
*)

val get_lapack_funcs : ?arrays:[>`Ndarray] Np.Obj.t list -> ?dtype:[`S of string | `Dtype of Np.Dtype.t] -> names:[`Sequence_of_str of Py.Object.t | `S of string] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return available LAPACK function objects from names.

Arrays are used to determine the optimal prefix of LAPACK routines.

Parameters
----------
names : str or sequence of str
    Name(s) of LAPACK functions without type prefix.

arrays : sequence of ndarrays, optional
    Arrays can be given to determine optimal prefix of LAPACK
    routines. If not given, double-precision routines will be
    used, otherwise the most generic type in arrays will be used.

dtype : str or dtype, optional
    Data-type specifier. Not used if `arrays` is non-empty.

Returns
-------
funcs : list
    List containing the found function(s).

Notes
-----
This routine automatically chooses between Fortran/C
interfaces. Fortran code is used whenever possible for arrays with
column major order. In all other cases, C code is preferred.

In LAPACK, the naming convention is that all functions start with a
type prefix, which depends on the type of the principal
matrix. These can be one of {'s', 'd', 'c', 'z'} for the numpy
types {float32, float64, complex64, complex128} respectively, and
are stored in attribute ``typecode`` of the returned functions.

Examples
--------
Suppose we would like to use '?lange' routine which computes the selected
norm of an array. We pass our array in order to get the correct 'lange'
flavor.

>>> import scipy.linalg as LA
>>> a = np.random.rand(3,2)
>>> x_lange = LA.get_lapack_funcs('lange', (a,))
>>> x_lange.typecode
'd'
>>> x_lange = LA.get_lapack_funcs('lange',(a*1j,))
>>> x_lange.typecode
'z'

Several LAPACK routines work best when its internal WORK array has
the optimal size (big enough for fast computation and small enough to
avoid waste of memory). This size is determined also by a dedicated query
to the function which is often wrapped as a standalone function and
commonly denoted as ``###_lwork``. Below is an example for ``?sysv``

>>> import scipy.linalg as LA
>>> a = np.random.rand(1000,1000)
>>> b = np.random.rand(1000,1)*1j
>>> # We pick up zsysv and zsysv_lwork due to b array
... xsysv, xlwork = LA.get_lapack_funcs(('sysv', 'sysv_lwork'), (a, b))
>>> opt_lwork, _ = xlwork(a.shape[0])  # returns a complex for 'z' prefix
>>> udut, ipiv, x, info = xsysv(a, b, lwork=int(opt_lwork.real))
*)

val norm : ?ord:[`Fro | `PyObject of Py.Object.t | `Nuc] -> ?axis:[`T2_tuple_of_ints of Py.Object.t | `I of int] -> ?keepdims:bool -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Matrix or vector norm.

This function is able to return one of eight different matrix norms,
or one of an infinite number of vector norms (described below), depending
on the value of the ``ord`` parameter.

Parameters
----------
x : array_like
    Input array.  If `axis` is None, `x` must be 1-D or 2-D, unless `ord`
    is None. If both `axis` and `ord` are None, the 2-norm of
    ``x.ravel`` will be returned.
ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
    Order of the norm (see table under ``Notes``). inf means numpy's
    `inf` object. The default is None.
axis : {None, int, 2-tuple of ints}, optional.
    If `axis` is an integer, it specifies the axis of `x` along which to
    compute the vector norms.  If `axis` is a 2-tuple, it specifies the
    axes that hold 2-D matrices, and the matrix norms of these matrices
    are computed.  If `axis` is None then either a vector norm (when `x`
    is 1-D) or a matrix norm (when `x` is 2-D) is returned. The default
    is None.

    .. versionadded:: 1.8.0

keepdims : bool, optional
    If this is set to True, the axes which are normed over are left in the
    result as dimensions with size one.  With this option the result will
    broadcast correctly against the original `x`.

    .. versionadded:: 1.10.0

Returns
-------
n : float or ndarray
    Norm of the matrix or vector(s).

Notes
-----
For values of ``ord <= 0``, the result is, strictly speaking, not a
mathematical 'norm', but it may still be useful for various numerical
purposes.

The following norms can be calculated:

=====  ============================  ==========================
ord    norm for matrices             norm for vectors
=====  ============================  ==========================
None   Frobenius norm                2-norm
'fro'  Frobenius norm                --
'nuc'  nuclear norm                  --
inf    max(sum(abs(x), axis=1))      max(abs(x))
-inf   min(sum(abs(x), axis=1))      min(abs(x))
0      --                            sum(x != 0)
1      max(sum(abs(x), axis=0))      as below
-1     min(sum(abs(x), axis=0))      as below
2      2-norm (largest sing. value)  as below
-2     smallest singular value       as below
other  --                            sum(abs(x)**ord)**(1./ord)
=====  ============================  ==========================

The Frobenius norm is given by [1]_:

    :math:`||A||_F = [\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

The nuclear norm is the sum of the singular values.

References
----------
.. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
       Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15

Examples
--------
>>> from numpy import linalg as LA
>>> a = np.arange(9) - 4
>>> a
array([-4, -3, -2, ...,  2,  3,  4])
>>> b = a.reshape((3, 3))
>>> b
array([[-4, -3, -2],
       [-1,  0,  1],
       [ 2,  3,  4]])

>>> LA.norm(a)
7.745966692414834
>>> LA.norm(b)
7.745966692414834
>>> LA.norm(b, 'fro')
7.745966692414834
>>> LA.norm(a, np.inf)
4.0
>>> LA.norm(b, np.inf)
9.0
>>> LA.norm(a, -np.inf)
0.0
>>> LA.norm(b, -np.inf)
2.0

>>> LA.norm(a, 1)
20.0
>>> LA.norm(b, 1)
7.0
>>> LA.norm(a, -1)
-4.6566128774142013e-010
>>> LA.norm(b, -1)
6.0
>>> LA.norm(a, 2)
7.745966692414834
>>> LA.norm(b, 2)
7.3484692283495345

>>> LA.norm(a, -2)
0.0
>>> LA.norm(b, -2)
1.8570331885190563e-016 # may vary
>>> LA.norm(a, 3)
5.8480354764257312 # may vary
>>> LA.norm(a, -3)
0.0

Using the `axis` argument to compute vector norms:

>>> c = np.array([[ 1, 2, 3],
...               [-1, 1, 4]])
>>> LA.norm(c, axis=0)
array([ 1.41421356,  2.23606798,  5.        ])
>>> LA.norm(c, axis=1)
array([ 3.74165739,  4.24264069])
>>> LA.norm(c, ord=1, axis=1)
array([ 6.,  6.])

Using the `axis` argument to compute matrix norms:

>>> m = np.arange(8).reshape(2,2,2)
>>> LA.norm(m, axis=(1,2))
array([  3.74165739,  11.22497216])
>>> LA.norm(m[0, :, :]), LA.norm(m[1, :, :])
(3.7416573867739413, 11.224972160321824)
*)

val rsf2csf : ?check_finite:bool -> t:[>`Ndarray] Np.Obj.t -> z:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Convert real Schur form to complex Schur form.

Convert a quasi-diagonal real-valued Schur form to the upper triangular
complex-valued Schur form.

Parameters
----------
T : (M, M) array_like
    Real Schur form of the original array
Z : (M, M) array_like
    Schur transformation matrix
check_finite : bool, optional
    Whether to check that the input arrays contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
T : (M, M) ndarray
    Complex Schur form of the original array
Z : (M, M) ndarray
    Schur transformation matrix corresponding to the complex form

See Also
--------
schur : Schur decomposition of an array

Examples
--------
>>> from scipy.linalg import schur, rsf2csf
>>> A = np.array([[0, 2, 2], [0, 1, 2], [1, 0, 1]])
>>> T, Z = schur(A)
>>> T
array([[ 2.65896708,  1.42440458, -1.92933439],
       [ 0.        , -0.32948354, -0.49063704],
       [ 0.        ,  1.31178921, -0.32948354]])
>>> Z
array([[0.72711591, -0.60156188, 0.33079564],
       [0.52839428, 0.79801892, 0.28976765],
       [0.43829436, 0.03590414, -0.89811411]])
>>> T2 , Z2 = rsf2csf(T, Z)
>>> T2
array([[2.65896708+0.j, -1.64592781+0.743164187j, -1.21516887+1.00660462j],
       [0.+0.j , -0.32948354+8.02254558e-01j, -0.82115218-2.77555756e-17j],
       [0.+0.j , 0.+0.j, -0.32948354-0.802254558j]])
>>> Z2
array([[0.72711591+0.j,  0.28220393-0.31385693j,  0.51319638-0.17258824j],
       [0.52839428+0.j,  0.24720268+0.41635578j, -0.68079517-0.15118243j],
       [0.43829436+0.j, -0.76618703+0.01873251j, -0.03063006+0.46857912j]])
*)

val schur : ?output:[`Real | `Complex] -> ?lwork:int -> ?overwrite_a:bool -> ?sort:[`Iuc | `Ouc | `Rhp | `Callable of Py.Object.t | `Lhp] -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Compute Schur decomposition of a matrix.

The Schur decomposition is::

    A = Z T Z^H

where Z is unitary and T is either upper-triangular, or for real
Schur decomposition (output='real'), quasi-upper triangular.  In
the quasi-triangular form, 2x2 blocks describing complex-valued
eigenvalue pairs may extrude from the diagonal.

Parameters
----------
a : (M, M) array_like
    Matrix to decompose
output : {'real', 'complex'}, optional
    Construct the real or complex Schur decomposition (for real matrices).
lwork : int, optional
    Work array size. If None or -1, it is automatically computed.
overwrite_a : bool, optional
    Whether to overwrite data in a (may improve performance).
sort : {None, callable, 'lhp', 'rhp', 'iuc', 'ouc'}, optional
    Specifies whether the upper eigenvalues should be sorted.  A callable
    may be passed that, given a eigenvalue, returns a boolean denoting
    whether the eigenvalue should be sorted to the top-left (True).
    Alternatively, string parameters may be used::

        'lhp'   Left-hand plane (x.real < 0.0)
        'rhp'   Right-hand plane (x.real > 0.0)
        'iuc'   Inside the unit circle (x*x.conjugate() <= 1.0)
        'ouc'   Outside the unit circle (x*x.conjugate() > 1.0)

    Defaults to None (no sorting).
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
T : (M, M) ndarray
    Schur form of A. It is real-valued for the real Schur decomposition.
Z : (M, M) ndarray
    An unitary Schur transformation matrix for A.
    It is real-valued for the real Schur decomposition.
sdim : int
    If and only if sorting was requested, a third return value will
    contain the number of eigenvalues satisfying the sort condition.

Raises
------
LinAlgError
    Error raised under three conditions:

    1. The algorithm failed due to a failure of the QR algorithm to
       compute all eigenvalues
    2. If eigenvalue sorting was requested, the eigenvalues could not be
       reordered due to a failure to separate eigenvalues, usually because
       of poor conditioning
    3. If eigenvalue sorting was requested, roundoff errors caused the
       leading eigenvalues to no longer satisfy the sorting condition

See also
--------
rsf2csf : Convert real Schur form to complex Schur form

Examples
--------
>>> from scipy.linalg import schur, eigvals
>>> A = np.array([[0, 2, 2], [0, 1, 2], [1, 0, 1]])
>>> T, Z = schur(A)
>>> T
array([[ 2.65896708,  1.42440458, -1.92933439],
       [ 0.        , -0.32948354, -0.49063704],
       [ 0.        ,  1.31178921, -0.32948354]])
>>> Z
array([[0.72711591, -0.60156188, 0.33079564],
       [0.52839428, 0.79801892, 0.28976765],
       [0.43829436, 0.03590414, -0.89811411]])

>>> T2, Z2 = schur(A, output='complex')
>>> T2
array([[ 2.65896708, -1.22839825+1.32378589j,  0.42590089+1.51937378j],
       [ 0.        , -0.32948354+0.80225456j, -0.59877807+0.56192146j],
       [ 0.        ,  0.                    , -0.32948354-0.80225456j]])
>>> eigvals(T2)
array([2.65896708, -0.32948354+0.80225456j, -0.32948354-0.80225456j])

An arbitrary custom eig-sorting condition, having positive imaginary part, 
which is satisfied by only one eigenvalue

>>> T3, Z3, sdim = schur(A, output='complex', sort=lambda x: x.imag > 0)
>>> sdim
1
*)


end

module Decomp_svd : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val arccos : ?out:[`Ndarray of [>`Ndarray] Np.Obj.t | `Tuple_of_ndarray_and_None of Py.Object.t] -> ?where:[>`Ndarray] Np.Obj.t -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
arccos(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

Trigonometric inverse cosine, element-wise.

The inverse of `cos` so that, if ``y = cos(x)``, then ``x = arccos(y)``.

Parameters
----------
x : array_like
    `x`-coordinate on the unit circle.
    For real arguments, the domain is [-1, 1].
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
angle : ndarray
    The angle of the ray intersecting the unit circle at the given
    `x`-coordinate in radians [0, pi].
    This is a scalar if `x` is a scalar.

See Also
--------
cos, arctan, arcsin, emath.arccos

Notes
-----
`arccos` is a multivalued function: for each `x` there are infinitely
many numbers `z` such that `cos(z) = x`. The convention is to return
the angle `z` whose real part lies in `[0, pi]`.

For real-valued input data types, `arccos` always returns real output.
For each value that cannot be expressed as a real number or infinity,
it yields ``nan`` and sets the `invalid` floating point error flag.

For complex-valued input, `arccos` is a complex analytic function that
has branch cuts `[-inf, -1]` and `[1, inf]` and is continuous from
above on the former and from below on the latter.

The inverse `cos` is also known as `acos` or cos^-1.

References
----------
M. Abramowitz and I.A. Stegun, 'Handbook of Mathematical Functions',
10th printing, 1964, pp. 79. http://www.math.sfu.ca/~cbm/aands/

Examples
--------
We expect the arccos of 1 to be 0, and of -1 to be pi:

>>> np.arccos([1, -1])
array([ 0.        ,  3.14159265])

Plot arccos:

>>> import matplotlib.pyplot as plt
>>> x = np.linspace(-1, 1, num=100)
>>> plt.plot(x, np.arccos(x))
>>> plt.axis('tight')
>>> plt.show()
*)

val arcsin : ?out:[`Ndarray of [>`Ndarray] Np.Obj.t | `Tuple_of_ndarray_and_None of Py.Object.t] -> ?where:[>`Ndarray] Np.Obj.t -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
arcsin(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

Inverse sine, element-wise.

Parameters
----------
x : array_like
    `y`-coordinate on the unit circle.
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
angle : ndarray
    The inverse sine of each element in `x`, in radians and in the
    closed interval ``[-pi/2, pi/2]``.
    This is a scalar if `x` is a scalar.

See Also
--------
sin, cos, arccos, tan, arctan, arctan2, emath.arcsin

Notes
-----
`arcsin` is a multivalued function: for each `x` there are infinitely
many numbers `z` such that :math:`sin(z) = x`.  The convention is to
return the angle `z` whose real part lies in [-pi/2, pi/2].

For real-valued input data types, *arcsin* always returns real output.
For each value that cannot be expressed as a real number or infinity,
it yields ``nan`` and sets the `invalid` floating point error flag.

For complex-valued input, `arcsin` is a complex analytic function that
has, by convention, the branch cuts [-inf, -1] and [1, inf]  and is
continuous from above on the former and from below on the latter.

The inverse sine is also known as `asin` or sin^{-1}.

References
----------
Abramowitz, M. and Stegun, I. A., *Handbook of Mathematical Functions*,
10th printing, New York: Dover, 1964, pp. 79ff.
http://www.math.sfu.ca/~cbm/aands/

Examples
--------
>>> np.arcsin(1)     # pi/2
1.5707963267948966
>>> np.arcsin(-1)    # -pi/2
-1.5707963267948966
>>> np.arcsin(0)
0.0
*)

val clip : ?out:[>`Ndarray] Np.Obj.t -> ?kwargs:(string * Py.Object.t) list -> a:[>`Ndarray] Np.Obj.t -> a_min:[`S of string | `F of float | `Ndarray of [>`Ndarray] Np.Obj.t | `I of int | `Bool of bool | `None] -> a_max:[`S of string | `F of float | `Ndarray of [>`Ndarray] Np.Obj.t | `I of int | `Bool of bool | `None] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Clip (limit) the values in an array.

Given an interval, values outside the interval are clipped to
the interval edges.  For example, if an interval of ``[0, 1]``
is specified, values smaller than 0 become 0, and values larger
than 1 become 1.

Equivalent to but faster than ``np.maximum(a_min, np.minimum(a, a_max))``.
No check is performed to ensure ``a_min < a_max``.

Parameters
----------
a : array_like
    Array containing elements to clip.
a_min : scalar or array_like or None
    Minimum value. If None, clipping is not performed on lower
    interval edge. Not more than one of `a_min` and `a_max` may be
    None.
a_max : scalar or array_like or None
    Maximum value. If None, clipping is not performed on upper
    interval edge. Not more than one of `a_min` and `a_max` may be
    None. If `a_min` or `a_max` are array_like, then the three
    arrays will be broadcasted to match their shapes.
out : ndarray, optional
    The results will be placed in this array. It may be the input
    array for in-place clipping.  `out` must be of the right shape
    to hold the output.  Its type is preserved.
**kwargs
    For other keyword-only arguments, see the
    :ref:`ufunc docs <ufuncs.kwargs>`.

    .. versionadded:: 1.17.0

Returns
-------
clipped_array : ndarray
    An array with the elements of `a`, but where values
    < `a_min` are replaced with `a_min`, and those > `a_max`
    with `a_max`.

See Also
--------
ufuncs-output-type

Examples
--------
>>> a = np.arange(10)
>>> np.clip(a, 1, 8)
array([1, 1, 2, 3, 4, 5, 6, 7, 8, 8])
>>> a
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> np.clip(a, 3, 6, out=a)
array([3, 3, 3, 3, 4, 5, 6, 6, 6, 6])
>>> a = np.arange(10)
>>> a
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> np.clip(a, [3, 4, 1, 1, 1, 4, 4, 4, 4, 4], 8)
array([3, 4, 2, 3, 4, 5, 6, 7, 8, 8])
*)

val diag : ?k:int -> v:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Extract a diagonal or construct a diagonal array.

See the more detailed documentation for ``numpy.diagonal`` if you use this
function to extract a diagonal and wish to write to the resulting array;
whether it returns a copy or a view depends on what version of numpy you
are using.

Parameters
----------
v : array_like
    If `v` is a 2-D array, return a copy of its `k`-th diagonal.
    If `v` is a 1-D array, return a 2-D array with `v` on the `k`-th
    diagonal.
k : int, optional
    Diagonal in question. The default is 0. Use `k>0` for diagonals
    above the main diagonal, and `k<0` for diagonals below the main
    diagonal.

Returns
-------
out : ndarray
    The extracted diagonal or constructed diagonal array.

See Also
--------
diagonal : Return specified diagonals.
diagflat : Create a 2-D array with the flattened input as a diagonal.
trace : Sum along diagonals.
triu : Upper triangle of an array.
tril : Lower triangle of an array.

Examples
--------
>>> x = np.arange(9).reshape((3,3))
>>> x
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])

>>> np.diag(x)
array([0, 4, 8])
>>> np.diag(x, k=1)
array([1, 5])
>>> np.diag(x, k=-1)
array([3, 7])

>>> np.diag(np.diag(x))
array([[0, 0, 0],
       [0, 4, 0],
       [0, 0, 8]])
*)

val diagsvd : s:Py.Object.t -> m:int -> n:int -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Construct the sigma matrix in SVD from singular values and size M, N.

Parameters
----------
s : (M,) or (N,) array_like
    Singular values
M : int
    Size of the matrix whose singular values are `s`.
N : int
    Size of the matrix whose singular values are `s`.

Returns
-------
S : (M, N) ndarray
    The S-matrix in the singular value decomposition

See Also
--------
svd : Singular value decomposition of a matrix
svdvals : Compute singular values of a matrix.

Examples
--------
>>> from scipy.linalg import diagsvd
>>> vals = np.array([1, 2, 3])  # The array representing the computed svd
>>> diagsvd(vals, 3, 4)
array([[1, 0, 0, 0],
       [0, 2, 0, 0],
       [0, 0, 3, 0]])
>>> diagsvd(vals, 4, 3)
array([[1, 0, 0],
       [0, 2, 0],
       [0, 0, 3],
       [0, 0, 0]])
*)

val dot : ?out:[>`Ndarray] Np.Obj.t -> a:[>`Ndarray] Np.Obj.t -> b:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
dot(a, b, out=None)

Dot product of two arrays. Specifically,

- If both `a` and `b` are 1-D arrays, it is inner product of vectors
  (without complex conjugation).

- If both `a` and `b` are 2-D arrays, it is matrix multiplication,
  but using :func:`matmul` or ``a @ b`` is preferred.

- If either `a` or `b` is 0-D (scalar), it is equivalent to :func:`multiply`
  and using ``numpy.multiply(a, b)`` or ``a * b`` is preferred.

- If `a` is an N-D array and `b` is a 1-D array, it is a sum product over
  the last axis of `a` and `b`.

- If `a` is an N-D array and `b` is an M-D array (where ``M>=2``), it is a
  sum product over the last axis of `a` and the second-to-last axis of `b`::

    dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])

Parameters
----------
a : array_like
    First argument.
b : array_like
    Second argument.
out : ndarray, optional
    Output argument. This must have the exact kind that would be returned
    if it was not used. In particular, it must have the right type, must be
    C-contiguous, and its dtype must be the dtype that would be returned
    for `dot(a,b)`. This is a performance feature. Therefore, if these
    conditions are not met, an exception is raised, instead of attempting
    to be flexible.

Returns
-------
output : ndarray
    Returns the dot product of `a` and `b`.  If `a` and `b` are both
    scalars or both 1-D arrays then a scalar is returned; otherwise
    an array is returned.
    If `out` is given, then it is returned.

Raises
------
ValueError
    If the last dimension of `a` is not the same size as
    the second-to-last dimension of `b`.

See Also
--------
vdot : Complex-conjugating dot product.
tensordot : Sum products over arbitrary axes.
einsum : Einstein summation convention.
matmul : '@' operator as method with out parameter.

Examples
--------
>>> np.dot(3, 4)
12

Neither argument is complex-conjugated:

>>> np.dot([2j, 3j], [2j, 3j])
(-13+0j)

For 2-D arrays it is the matrix product:

>>> a = [[1, 0], [0, 1]]
>>> b = [[4, 1], [2, 2]]
>>> np.dot(a, b)
array([[4, 1],
       [2, 2]])

>>> a = np.arange(3*4*5*6).reshape((3,4,5,6))
>>> b = np.arange(3*4*5*6)[::-1].reshape((5,4,6,3))
>>> np.dot(a, b)[2,3,2,1,2,2]
499128
>>> sum(a[2,3,2,:] * b[1,2,:,2])
499128
*)

val get_lapack_funcs : ?arrays:[>`Ndarray] Np.Obj.t list -> ?dtype:[`S of string | `Dtype of Np.Dtype.t] -> names:[`Sequence_of_str of Py.Object.t | `S of string] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return available LAPACK function objects from names.

Arrays are used to determine the optimal prefix of LAPACK routines.

Parameters
----------
names : str or sequence of str
    Name(s) of LAPACK functions without type prefix.

arrays : sequence of ndarrays, optional
    Arrays can be given to determine optimal prefix of LAPACK
    routines. If not given, double-precision routines will be
    used, otherwise the most generic type in arrays will be used.

dtype : str or dtype, optional
    Data-type specifier. Not used if `arrays` is non-empty.

Returns
-------
funcs : list
    List containing the found function(s).

Notes
-----
This routine automatically chooses between Fortran/C
interfaces. Fortran code is used whenever possible for arrays with
column major order. In all other cases, C code is preferred.

In LAPACK, the naming convention is that all functions start with a
type prefix, which depends on the type of the principal
matrix. These can be one of {'s', 'd', 'c', 'z'} for the numpy
types {float32, float64, complex64, complex128} respectively, and
are stored in attribute ``typecode`` of the returned functions.

Examples
--------
Suppose we would like to use '?lange' routine which computes the selected
norm of an array. We pass our array in order to get the correct 'lange'
flavor.

>>> import scipy.linalg as LA
>>> a = np.random.rand(3,2)
>>> x_lange = LA.get_lapack_funcs('lange', (a,))
>>> x_lange.typecode
'd'
>>> x_lange = LA.get_lapack_funcs('lange',(a*1j,))
>>> x_lange.typecode
'z'

Several LAPACK routines work best when its internal WORK array has
the optimal size (big enough for fast computation and small enough to
avoid waste of memory). This size is determined also by a dedicated query
to the function which is often wrapped as a standalone function and
commonly denoted as ``###_lwork``. Below is an example for ``?sysv``

>>> import scipy.linalg as LA
>>> a = np.random.rand(1000,1000)
>>> b = np.random.rand(1000,1)*1j
>>> # We pick up zsysv and zsysv_lwork due to b array
... xsysv, xlwork = LA.get_lapack_funcs(('sysv', 'sysv_lwork'), (a, b))
>>> opt_lwork, _ = xlwork(a.shape[0])  # returns a complex for 'z' prefix
>>> udut, ipiv, x, info = xsysv(a, b, lwork=int(opt_lwork.real))
*)

val null_space : ?rcond:float -> a:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Construct an orthonormal basis for the null space of A using SVD

Parameters
----------
A : (M, N) array_like
    Input array
rcond : float, optional
    Relative condition number. Singular values ``s`` smaller than
    ``rcond * max(s)`` are considered zero.
    Default: floating point eps * max(M,N).

Returns
-------
Z : (N, K) ndarray
    Orthonormal basis for the null space of A.
    K = dimension of effective null space, as determined by rcond

See also
--------
svd : Singular value decomposition of a matrix
orth : Matrix range

Examples
--------
One-dimensional null space:

>>> from scipy.linalg import null_space
>>> A = np.array([[1, 1], [1, 1]])
>>> ns = null_space(A)
>>> ns * np.sign(ns[0,0])  # Remove the sign ambiguity of the vector
array([[ 0.70710678],
       [-0.70710678]])

Two-dimensional null space:

>>> B = np.random.rand(3, 5)
>>> Z = null_space(B)
>>> Z.shape
(5, 2)
>>> np.allclose(B.dot(Z), 0)
True

The basis vectors are orthonormal (up to rounding error):

>>> Z.T.dot(Z)
array([[  1.00000000e+00,   6.92087741e-17],
       [  6.92087741e-17,   1.00000000e+00]])
*)

val orth : ?rcond:float -> a:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Construct an orthonormal basis for the range of A using SVD

Parameters
----------
A : (M, N) array_like
    Input array
rcond : float, optional
    Relative condition number. Singular values ``s`` smaller than
    ``rcond * max(s)`` are considered zero.
    Default: floating point eps * max(M,N).

Returns
-------
Q : (M, K) ndarray
    Orthonormal basis for the range of A.
    K = effective rank of A, as determined by rcond

See also
--------
svd : Singular value decomposition of a matrix
null_space : Matrix null space

Examples
--------
>>> from scipy.linalg import orth
>>> A = np.array([[2, 0, 0], [0, 5, 0]])  # rank 2 array
>>> orth(A)
array([[0., 1.],
       [1., 0.]])
>>> orth(A.T)
array([[0., 1.],
       [1., 0.],
       [0., 0.]])
*)

val subspace_angles : a:[>`Ndarray] Np.Obj.t -> b:Py.Object.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the subspace angles between two matrices.

Parameters
----------
A : (M, N) array_like
    The first input array.
B : (M, K) array_like
    The second input array.

Returns
-------
angles : ndarray, shape (min(N, K),)
    The subspace angles between the column spaces of `A` and `B` in
    descending order.

See Also
--------
orth
svd

Notes
-----
This computes the subspace angles according to the formula
provided in [1]_. For equivalence with MATLAB and Octave behavior,
use ``angles[0]``.

.. versionadded:: 1.0

References
----------
.. [1] Knyazev A, Argentati M (2002) Principal Angles between Subspaces
       in an A-Based Scalar Product: Algorithms and Perturbation
       Estimates. SIAM J. Sci. Comput. 23:2008-2040.

Examples
--------
A Hadamard matrix, which has orthogonal columns, so we expect that
the suspace angle to be :math:`\frac{\pi}{2}`:

>>> from scipy.linalg import hadamard, subspace_angles
>>> H = hadamard(4)
>>> print(H)
[[ 1  1  1  1]
 [ 1 -1  1 -1]
 [ 1  1 -1 -1]
 [ 1 -1 -1  1]]
>>> np.rad2deg(subspace_angles(H[:, :2], H[:, 2:]))
array([ 90.,  90.])

And the subspace angle of a matrix to itself should be zero:

>>> subspace_angles(H[:, :2], H[:, :2]) <= 2 * np.finfo(float).eps
array([ True,  True], dtype=bool)

The angles between non-orthogonal subspaces are in between these extremes:

>>> x = np.random.RandomState(0).randn(4, 3)
>>> np.rad2deg(subspace_angles(x[:, :2], x[:, [2]]))
array([ 55.832])
*)

val svd : ?full_matrices:bool -> ?compute_uv:bool -> ?overwrite_a:bool -> ?check_finite:bool -> ?lapack_driver:[`Gesdd | `Gesvd] -> a:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
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

val svdvals : ?overwrite_a:bool -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Compute singular values of a matrix.

Parameters
----------
a : (M, N) array_like
    Matrix to decompose.
overwrite_a : bool, optional
    Whether to overwrite `a`; may improve performance.
    Default is False.
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
s : (min(M, N),) ndarray
    The singular values, sorted in decreasing order.

Raises
------
LinAlgError
    If SVD computation does not converge.

Notes
-----
``svdvals(a)`` only differs from ``svd(a, compute_uv=False)`` by its
handling of the edge case of empty ``a``, where it returns an
empty sequence:

>>> a = np.empty((0, 2))
>>> from scipy.linalg import svdvals
>>> svdvals(a)
array([], dtype=float64)

See Also
--------
svd : Compute the full singular value decomposition of a matrix.
diagsvd : Construct the Sigma matrix, given the vector s.

Examples
--------
>>> from scipy.linalg import svdvals
>>> m = np.array([[1.0, 0.0],
...               [2.0, 3.0],
...               [1.0, 1.0],
...               [0.0, 2.0],
...               [1.0, 0.0]])
>>> svdvals(m)
array([ 4.28091555,  1.63516424])

We can verify the maximum singular value of `m` by computing the maximum
length of `m.dot(u)` over all the unit vectors `u` in the (x,y) plane.
We approximate 'all' the unit vectors with a large sample.  Because
of linearity, we only need the unit vectors with angles in [0, pi].

>>> t = np.linspace(0, np.pi, 2000)
>>> u = np.array([np.cos(t), np.sin(t)])
>>> np.linalg.norm(m.dot(u), axis=0).max()
4.2809152422538475

`p` is a projection matrix with rank 1.  With exact arithmetic,
its singular values would be [1, 0, 0, 0].

>>> v = np.array([0.1, 0.3, 0.9, 0.3])
>>> p = np.outer(v, v)
>>> svdvals(p)
array([  1.00000000e+00,   2.02021698e-17,   1.56692500e-17,
         8.15115104e-34])

The singular values of an orthogonal matrix are all 1.  Here we
create a random orthogonal matrix by using the `rvs()` method of
`scipy.stats.ortho_group`.

>>> from scipy.stats import ortho_group
>>> np.random.seed(123)
>>> orth = ortho_group.rvs(4)
>>> svdvals(orth)
array([ 1.,  1.,  1.,  1.])
*)

val where : ?x:Py.Object.t -> ?y:Py.Object.t -> condition:[`Ndarray of [>`Ndarray] Np.Obj.t | `Bool of bool] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
where(condition, [x, y])

Return elements chosen from `x` or `y` depending on `condition`.

.. note::
    When only `condition` is provided, this function is a shorthand for
    ``np.asarray(condition).nonzero()``. Using `nonzero` directly should be
    preferred, as it behaves correctly for subclasses. The rest of this
    documentation covers only the case where all three arguments are
    provided.

Parameters
----------
condition : array_like, bool
    Where True, yield `x`, otherwise yield `y`.
x, y : array_like
    Values from which to choose. `x`, `y` and `condition` need to be
    broadcastable to some shape.

Returns
-------
out : ndarray
    An array with elements from `x` where `condition` is True, and elements
    from `y` elsewhere.

See Also
--------
choose
nonzero : The function that is called when x and y are omitted

Notes
-----
If all the arrays are 1-D, `where` is equivalent to::

    [xv if c else yv
     for c, xv, yv in zip(condition, x, y)]

Examples
--------
>>> a = np.arange(10)
>>> a
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> np.where(a < 5, a, 10*a)
array([ 0,  1,  2,  3,  4, 50, 60, 70, 80, 90])

This can be used on multidimensional arrays too:

>>> np.where([[True, False], [True, True]],
...          [[1, 2], [3, 4]],
...          [[9, 8], [7, 6]])
array([[1, 8],
       [3, 4]])

The shapes of x, y, and the condition are broadcast together:

>>> x, y = np.ogrid[:3, :4]
>>> np.where(x < y, x, 10 + y)  # both x and 10+y are broadcast
array([[10,  0,  0,  0],
       [10, 11,  1,  1],
       [10, 11, 12,  2]])

>>> a = np.array([[0, 1, 2],
...               [0, 2, 4],
...               [0, 3, 6]])
>>> np.where(a < 4, a, -1)  # -1 is broadcast
array([[ 0,  1,  2],
       [ 0,  2, -1],
       [ 0,  3, -1]])
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

module Flinalg : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val get_flinalg_funcs : ?arrays:Py.Object.t -> ?debug:Py.Object.t -> names:Py.Object.t -> unit -> Py.Object.t
(**
Return optimal available _flinalg function objects with
names. arrays are used to determine optimal prefix.
*)

val has_column_major_storage : Py.Object.t -> Py.Object.t
(**
None
*)


end

module Lapack : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val cgegv : ?kwds:(string * Py.Object.t) list -> Py.Object.t list -> Py.Object.t
(**
`cgegv` is deprecated!
The `*gegv` family of routines has been deprecated in
LAPACK 3.6.0 in favor of the `*ggev` family of routines.
The corresponding wrappers will be removed from SciPy in
a future release.

alpha,beta,vl,vr,info = cgegv(a,b,[compute_vl,compute_vr,lwork,overwrite_a,overwrite_b])

Wrapper for ``cgegv``.

Parameters
----------
a : input rank-2 array('F') with bounds (n,n)
b : input rank-2 array('F') with bounds (n,n)

Other Parameters
----------------
compute_vl : input int, optional
    Default: 1
compute_vr : input int, optional
    Default: 1
overwrite_a : input int, optional
    Default: 0
overwrite_b : input int, optional
    Default: 0
lwork : input int, optional
    Default: max(2*n,1)

Returns
-------
alpha : rank-1 array('F') with bounds (n)
beta : rank-1 array('F') with bounds (n)
vl : rank-2 array('F') with bounds (ldvl,n)
vr : rank-2 array('F') with bounds (ldvr,n)
info : int
*)

val dgegv : ?kwds:(string * Py.Object.t) list -> Py.Object.t list -> Py.Object.t
(**
`dgegv` is deprecated!
The `*gegv` family of routines has been deprecated in
LAPACK 3.6.0 in favor of the `*ggev` family of routines.
The corresponding wrappers will be removed from SciPy in
a future release.

alphar,alphai,beta,vl,vr,info = dgegv(a,b,[compute_vl,compute_vr,lwork,overwrite_a,overwrite_b])

Wrapper for ``dgegv``.

Parameters
----------
a : input rank-2 array('d') with bounds (n,n)
b : input rank-2 array('d') with bounds (n,n)

Other Parameters
----------------
compute_vl : input int, optional
    Default: 1
compute_vr : input int, optional
    Default: 1
overwrite_a : input int, optional
    Default: 0
overwrite_b : input int, optional
    Default: 0
lwork : input int, optional
    Default: max(8*n,1)

Returns
-------
alphar : rank-1 array('d') with bounds (n)
alphai : rank-1 array('d') with bounds (n)
beta : rank-1 array('d') with bounds (n)
vl : rank-2 array('d') with bounds (ldvl,n)
vr : rank-2 array('d') with bounds (ldvr,n)
info : int
*)

val find_best_lapack_type : ?arrays:[>`Ndarray] Np.Obj.t list -> ?dtype:[`S of string | `Dtype of Np.Dtype.t] -> unit -> (string * Np.Dtype.t * bool)
(**
Find best-matching BLAS/LAPACK type.

Arrays are used to determine the optimal prefix of BLAS routines.

Parameters
----------
arrays : sequence of ndarrays, optional
    Arrays can be given to determine optimal prefix of BLAS
    routines. If not given, double-precision routines will be
    used, otherwise the most generic type in arrays will be used.
dtype : str or dtype, optional
    Data-type specifier. Not used if `arrays` is non-empty.

Returns
-------
prefix : str
    BLAS/LAPACK prefix character.
dtype : dtype
    Inferred Numpy data type.
prefer_fortran : bool
    Whether to prefer Fortran order routines over C order.

Examples
--------
>>> import scipy.linalg.blas as bla
>>> a = np.random.rand(10,15)
>>> b = np.asfortranarray(a)  # Change the memory layout order
>>> bla.find_best_blas_type((a,))
('d', dtype('float64'), False)
>>> bla.find_best_blas_type((a*1j,))
('z', dtype('complex128'), False)
>>> bla.find_best_blas_type((b,))
('d', dtype('float64'), True)
*)

val get_lapack_funcs : ?arrays:[>`Ndarray] Np.Obj.t list -> ?dtype:[`S of string | `Dtype of Np.Dtype.t] -> names:[`Sequence_of_str of Py.Object.t | `S of string] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return available LAPACK function objects from names.

Arrays are used to determine the optimal prefix of LAPACK routines.

Parameters
----------
names : str or sequence of str
    Name(s) of LAPACK functions without type prefix.

arrays : sequence of ndarrays, optional
    Arrays can be given to determine optimal prefix of LAPACK
    routines. If not given, double-precision routines will be
    used, otherwise the most generic type in arrays will be used.

dtype : str or dtype, optional
    Data-type specifier. Not used if `arrays` is non-empty.

Returns
-------
funcs : list
    List containing the found function(s).

Notes
-----
This routine automatically chooses between Fortran/C
interfaces. Fortran code is used whenever possible for arrays with
column major order. In all other cases, C code is preferred.

In LAPACK, the naming convention is that all functions start with a
type prefix, which depends on the type of the principal
matrix. These can be one of {'s', 'd', 'c', 'z'} for the numpy
types {float32, float64, complex64, complex128} respectively, and
are stored in attribute ``typecode`` of the returned functions.

Examples
--------
Suppose we would like to use '?lange' routine which computes the selected
norm of an array. We pass our array in order to get the correct 'lange'
flavor.

>>> import scipy.linalg as LA
>>> a = np.random.rand(3,2)
>>> x_lange = LA.get_lapack_funcs('lange', (a,))
>>> x_lange.typecode
'd'
>>> x_lange = LA.get_lapack_funcs('lange',(a*1j,))
>>> x_lange.typecode
'z'

Several LAPACK routines work best when its internal WORK array has
the optimal size (big enough for fast computation and small enough to
avoid waste of memory). This size is determined also by a dedicated query
to the function which is often wrapped as a standalone function and
commonly denoted as ``###_lwork``. Below is an example for ``?sysv``

>>> import scipy.linalg as LA
>>> a = np.random.rand(1000,1000)
>>> b = np.random.rand(1000,1)*1j
>>> # We pick up zsysv and zsysv_lwork due to b array
... xsysv, xlwork = LA.get_lapack_funcs(('sysv', 'sysv_lwork'), (a, b))
>>> opt_lwork, _ = xlwork(a.shape[0])  # returns a complex for 'z' prefix
>>> udut, ipiv, x, info = xsysv(a, b, lwork=int(opt_lwork.real))
*)

val sgegv : ?kwds:(string * Py.Object.t) list -> Py.Object.t list -> Py.Object.t
(**
`sgegv` is deprecated!
The `*gegv` family of routines has been deprecated in
LAPACK 3.6.0 in favor of the `*ggev` family of routines.
The corresponding wrappers will be removed from SciPy in
a future release.

alphar,alphai,beta,vl,vr,info = sgegv(a,b,[compute_vl,compute_vr,lwork,overwrite_a,overwrite_b])

Wrapper for ``sgegv``.

Parameters
----------
a : input rank-2 array('f') with bounds (n,n)
b : input rank-2 array('f') with bounds (n,n)

Other Parameters
----------------
compute_vl : input int, optional
    Default: 1
compute_vr : input int, optional
    Default: 1
overwrite_a : input int, optional
    Default: 0
overwrite_b : input int, optional
    Default: 0
lwork : input int, optional
    Default: max(8*n,1)

Returns
-------
alphar : rank-1 array('f') with bounds (n)
alphai : rank-1 array('f') with bounds (n)
beta : rank-1 array('f') with bounds (n)
vl : rank-2 array('f') with bounds (ldvl,n)
vr : rank-2 array('f') with bounds (ldvr,n)
info : int
*)

val zgegv : ?kwds:(string * Py.Object.t) list -> Py.Object.t list -> Py.Object.t
(**
`zgegv` is deprecated!
The `*gegv` family of routines has been deprecated in
LAPACK 3.6.0 in favor of the `*ggev` family of routines.
The corresponding wrappers will be removed from SciPy in
a future release.

alpha,beta,vl,vr,info = zgegv(a,b,[compute_vl,compute_vr,lwork,overwrite_a,overwrite_b])

Wrapper for ``zgegv``.

Parameters
----------
a : input rank-2 array('D') with bounds (n,n)
b : input rank-2 array('D') with bounds (n,n)

Other Parameters
----------------
compute_vl : input int, optional
    Default: 1
compute_vr : input int, optional
    Default: 1
overwrite_a : input int, optional
    Default: 0
overwrite_b : input int, optional
    Default: 0
lwork : input int, optional
    Default: max(2*n,1)

Returns
-------
alpha : rank-1 array('D') with bounds (n)
beta : rank-1 array('D') with bounds (n)
vl : rank-2 array('D') with bounds (ldvl,n)
vr : rank-2 array('D') with bounds (ldvr,n)
info : int
*)


end

module Linalg_version : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t


end

module Matfuncs : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module Single : sig
type tag = [`Float32]
type t = [`Float32 | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val __getitem__ : key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return self[key].
*)

val newbyteorder : ?new_order:string -> [> tag] Obj.t -> Np.Dtype.t
(**
newbyteorder(new_order='S')

Return a new `dtype` with a different byte order.

Changes are also made in all fields and sub-arrays of the data type.

The `new_order` code can be any from the following:

* 'S' - swap dtype from current to opposite endian
* {'<', 'L'} - little endian
* {'>', 'B'} - big endian
* {'=', 'N'} - native order
* {'|', 'I'} - ignore (no change to byte order)

Parameters
----------
new_order : str, optional
    Byte order to force; a value from the byte order specifications
    above.  The default value ('S') results in swapping the current
    byte order. The code does a case-insensitive check on the first
    letter of `new_order` for the alternatives above.  For example,
    any of 'B' or 'b' or 'biggish' are valid to specify big-endian.


Returns
-------
new_dtype : dtype
    New `dtype` object with the given change to the byte order.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val absolute : ?out:[`Ndarray of [>`Ndarray] Np.Obj.t | `Tuple_of_ndarray_and_None of Py.Object.t] -> ?where:[>`Ndarray] Np.Obj.t -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
absolute(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

Calculate the absolute value element-wise.

``np.abs`` is a shorthand for this function.

Parameters
----------
x : array_like
    Input array.
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
absolute : ndarray
    An ndarray containing the absolute value of
    each element in `x`.  For complex input, ``a + ib``, the
    absolute value is :math:`\sqrt{ a^2 + b^2 }`.
    This is a scalar if `x` is a scalar.

Examples
--------
>>> x = np.array([-1.2, 1.2])
>>> np.absolute(x)
array([ 1.2,  1.2])
>>> np.absolute(1.2 + 1j)
1.5620499351813308

Plot the function over ``[-10, 10]``:

>>> import matplotlib.pyplot as plt

>>> x = np.linspace(start=-10, stop=10, num=101)
>>> plt.plot(x, np.absolute(x))
>>> plt.show()

Plot the function over the complex plane:

>>> xx = x + 1j * x[:, np.newaxis]
>>> plt.imshow(np.abs(xx), extent=[-10, 10, -10, 10], cmap='gray')
>>> plt.show()
*)

val amax : ?axis:int list -> ?out:[>`Ndarray] Np.Obj.t -> ?keepdims:bool -> ?initial:[`F of float | `I of int | `Bool of bool | `S of string] -> ?where:Py.Object.t -> a:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return the maximum of an array or maximum along an axis.

Parameters
----------
a : array_like
    Input data.
axis : None or int or tuple of ints, optional
    Axis or axes along which to operate.  By default, flattened input is
    used.

    .. versionadded:: 1.7.0

    If this is a tuple of ints, the maximum is selected over multiple axes,
    instead of a single axis or all the axes as before.
out : ndarray, optional
    Alternative output array in which to place the result.  Must
    be of the same shape and buffer length as the expected output.
    See `ufuncs-output-type` for more details.

keepdims : bool, optional
    If this is set to True, the axes which are reduced are left
    in the result as dimensions with size one. With this option,
    the result will broadcast correctly against the input array.

    If the default value is passed, then `keepdims` will not be
    passed through to the `amax` method of sub-classes of
    `ndarray`, however any non-default value will be.  If the
    sub-class' method does not implement `keepdims` any
    exceptions will be raised.

initial : scalar, optional
    The minimum value of an output element. Must be present to allow
    computation on empty slice. See `~numpy.ufunc.reduce` for details.

    .. versionadded:: 1.15.0

where : array_like of bool, optional
    Elements to compare for the maximum. See `~numpy.ufunc.reduce`
    for details.

    .. versionadded:: 1.17.0

Returns
-------
amax : ndarray or scalar
    Maximum of `a`. If `axis` is None, the result is a scalar value.
    If `axis` is given, the result is an array of dimension
    ``a.ndim - 1``.

See Also
--------
amin :
    The minimum value of an array along a given axis, propagating any NaNs.
nanmax :
    The maximum value of an array along a given axis, ignoring any NaNs.
maximum :
    Element-wise maximum of two arrays, propagating any NaNs.
fmax :
    Element-wise maximum of two arrays, ignoring any NaNs.
argmax :
    Return the indices of the maximum values.

nanmin, minimum, fmin

Notes
-----
NaN values are propagated, that is if at least one item is NaN, the
corresponding max value will be NaN as well. To ignore NaN values
(MATLAB behavior), please use nanmax.

Don't use `amax` for element-wise comparison of 2 arrays; when
``a.shape[0]`` is 2, ``maximum(a[0], a[1])`` is faster than
``amax(a, axis=0)``.

Examples
--------
>>> a = np.arange(4).reshape((2,2))
>>> a
array([[0, 1],
       [2, 3]])
>>> np.amax(a)           # Maximum of the flattened array
3
>>> np.amax(a, axis=0)   # Maxima along the first axis
array([2, 3])
>>> np.amax(a, axis=1)   # Maxima along the second axis
array([1, 3])
>>> np.amax(a, where=[False, True], initial=-1, axis=0)
array([-1,  3])
>>> b = np.arange(5, dtype=float)
>>> b[2] = np.NaN
>>> np.amax(b)
nan
>>> np.amax(b, where=~np.isnan(b), initial=-1)
4.0
>>> np.nanmax(b)
4.0

You can use an initial value to compute the maximum of an empty slice, or
to initialize it to a different value:

>>> np.max([[-50], [10]], axis=-1, initial=0)
array([ 0, 10])

Notice that the initial value is used as one of the elements for which the
maximum is determined, unlike for the default argument Python's max
function, which is only used for empty iterables.

>>> np.max([5], initial=6)
6
>>> max([5], default=6)
5
*)

val conjugate : ?out:[`Ndarray of [>`Ndarray] Np.Obj.t | `Tuple_of_ndarray_and_None of Py.Object.t] -> ?where:[>`Ndarray] Np.Obj.t -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
conjugate(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

Return the complex conjugate, element-wise.

The complex conjugate of a complex number is obtained by changing the
sign of its imaginary part.

Parameters
----------
x : array_like
    Input value.
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
    The complex conjugate of `x`, with same dtype as `y`.
    This is a scalar if `x` is a scalar.

Notes
-----
`conj` is an alias for `conjugate`:

>>> np.conj is np.conjugate
True

Examples
--------
>>> np.conjugate(1+2j)
(1-2j)

>>> x = np.eye(2) + 1j * np.eye(2)
>>> np.conjugate(x)
array([[ 1.-1.j,  0.-0.j],
       [ 0.-0.j,  1.-1.j]])
*)

val coshm : [>`Ndarray] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the hyperbolic matrix cosine.

This routine uses expm to compute the matrix exponentials.

Parameters
----------
A : (N, N) array_like
    Input array.

Returns
-------
coshm : (N, N) ndarray
    Hyperbolic matrix cosine of `A`

Examples
--------
>>> from scipy.linalg import tanhm, sinhm, coshm
>>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
>>> c = coshm(a)
>>> c
array([[ 11.24592233,  38.76236492],
       [ 12.92078831,  50.00828725]])

Verify tanhm(a) = sinhm(a).dot(inv(coshm(a)))

>>> t = tanhm(a)
>>> s = sinhm(a)
>>> t - s.dot(np.linalg.inv(c))
array([[  2.72004641e-15,   4.55191440e-15],
       [  0.00000000e+00,  -5.55111512e-16]])
*)

val cosm : [>`Ndarray] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the matrix cosine.

This routine uses expm to compute the matrix exponentials.

Parameters
----------
A : (N, N) array_like
    Input array

Returns
-------
cosm : (N, N) ndarray
    Matrix cosine of A

Examples
--------
>>> from scipy.linalg import expm, sinm, cosm

Euler's identity (exp(i*theta) = cos(theta) + i*sin(theta))
applied to a matrix:

>>> a = np.array([[1.0, 2.0], [-1.0, 3.0]])
>>> expm(1j*a)
array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
       [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
>>> cosm(a) + 1j*sinm(a)
array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
       [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
*)

val diag : ?k:int -> v:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Extract a diagonal or construct a diagonal array.

See the more detailed documentation for ``numpy.diagonal`` if you use this
function to extract a diagonal and wish to write to the resulting array;
whether it returns a copy or a view depends on what version of numpy you
are using.

Parameters
----------
v : array_like
    If `v` is a 2-D array, return a copy of its `k`-th diagonal.
    If `v` is a 1-D array, return a 2-D array with `v` on the `k`-th
    diagonal.
k : int, optional
    Diagonal in question. The default is 0. Use `k>0` for diagonals
    above the main diagonal, and `k<0` for diagonals below the main
    diagonal.

Returns
-------
out : ndarray
    The extracted diagonal or constructed diagonal array.

See Also
--------
diagonal : Return specified diagonals.
diagflat : Create a 2-D array with the flattened input as a diagonal.
trace : Sum along diagonals.
triu : Upper triangle of an array.
tril : Lower triangle of an array.

Examples
--------
>>> x = np.arange(9).reshape((3,3))
>>> x
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])

>>> np.diag(x)
array([0, 4, 8])
>>> np.diag(x, k=1)
array([1, 5])
>>> np.diag(x, k=-1)
array([3, 7])

>>> np.diag(np.diag(x))
array([[0, 0, 0],
       [0, 4, 0],
       [0, 0, 8]])
*)

val dot : ?out:[>`Ndarray] Np.Obj.t -> a:[>`Ndarray] Np.Obj.t -> b:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
dot(a, b, out=None)

Dot product of two arrays. Specifically,

- If both `a` and `b` are 1-D arrays, it is inner product of vectors
  (without complex conjugation).

- If both `a` and `b` are 2-D arrays, it is matrix multiplication,
  but using :func:`matmul` or ``a @ b`` is preferred.

- If either `a` or `b` is 0-D (scalar), it is equivalent to :func:`multiply`
  and using ``numpy.multiply(a, b)`` or ``a * b`` is preferred.

- If `a` is an N-D array and `b` is a 1-D array, it is a sum product over
  the last axis of `a` and `b`.

- If `a` is an N-D array and `b` is an M-D array (where ``M>=2``), it is a
  sum product over the last axis of `a` and the second-to-last axis of `b`::

    dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])

Parameters
----------
a : array_like
    First argument.
b : array_like
    Second argument.
out : ndarray, optional
    Output argument. This must have the exact kind that would be returned
    if it was not used. In particular, it must have the right type, must be
    C-contiguous, and its dtype must be the dtype that would be returned
    for `dot(a,b)`. This is a performance feature. Therefore, if these
    conditions are not met, an exception is raised, instead of attempting
    to be flexible.

Returns
-------
output : ndarray
    Returns the dot product of `a` and `b`.  If `a` and `b` are both
    scalars or both 1-D arrays then a scalar is returned; otherwise
    an array is returned.
    If `out` is given, then it is returned.

Raises
------
ValueError
    If the last dimension of `a` is not the same size as
    the second-to-last dimension of `b`.

See Also
--------
vdot : Complex-conjugating dot product.
tensordot : Sum products over arbitrary axes.
einsum : Einstein summation convention.
matmul : '@' operator as method with out parameter.

Examples
--------
>>> np.dot(3, 4)
12

Neither argument is complex-conjugated:

>>> np.dot([2j, 3j], [2j, 3j])
(-13+0j)

For 2-D arrays it is the matrix product:

>>> a = [[1, 0], [0, 1]]
>>> b = [[4, 1], [2, 2]]
>>> np.dot(a, b)
array([[4, 1],
       [2, 2]])

>>> a = np.arange(3*4*5*6).reshape((3,4,5,6))
>>> b = np.arange(3*4*5*6)[::-1].reshape((5,4,6,3))
>>> np.dot(a, b)[2,3,2,1,2,2]
499128
>>> sum(a[2,3,2,:] * b[1,2,:,2])
499128
*)

val expm : [>`ArrayLike] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the matrix exponential using Pade approximation.

Parameters
----------
A : (N, N) array_like or sparse matrix
    Matrix to be exponentiated.

Returns
-------
expm : (N, N) ndarray
    Matrix exponential of `A`.

References
----------
.. [1] Awad H. Al-Mohy and Nicholas J. Higham (2009)
       'A New Scaling and Squaring Algorithm for the Matrix Exponential.'
       SIAM Journal on Matrix Analysis and Applications.
       31 (3). pp. 970-989. ISSN 1095-7162

Examples
--------
>>> from scipy.linalg import expm, sinm, cosm

Matrix version of the formula exp(0) = 1:

>>> expm(np.zeros((2,2)))
array([[ 1.,  0.],
       [ 0.,  1.]])

Euler's identity (exp(i*theta) = cos(theta) + i*sin(theta))
applied to a matrix:

>>> a = np.array([[1.0, 2.0], [-1.0, 3.0]])
>>> expm(1j*a)
array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
       [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
>>> cosm(a) + 1j*sinm(a)
array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
       [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
*)

val expm_cond : ?check_finite:bool -> a:Py.Object.t -> unit -> float
(**
Relative condition number of the matrix exponential in the Frobenius norm.

Parameters
----------
A : 2d array_like
    Square input matrix with shape (N, N).
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
kappa : float
    The relative condition number of the matrix exponential
    in the Frobenius norm

Notes
-----
A faster estimate for the condition number in the 1-norm
has been published but is not yet implemented in scipy.

.. versionadded:: 0.14.0

See also
--------
expm : Compute the exponential of a matrix.
expm_frechet : Compute the Frechet derivative of the matrix exponential.

Examples
--------
>>> from scipy.linalg import expm_cond
>>> A = np.array([[-0.3, 0.2, 0.6], [0.6, 0.3, -0.1], [-0.7, 1.2, 0.9]])
>>> k = expm_cond(A)
>>> k
1.7787805864469866
*)

val expm_frechet : ?method_:string -> ?compute_expm:bool -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> e:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Frechet derivative of the matrix exponential of A in the direction E.

Parameters
----------
A : (N, N) array_like
    Matrix of which to take the matrix exponential.
E : (N, N) array_like
    Matrix direction in which to take the Frechet derivative.
method : str, optional
    Choice of algorithm.  Should be one of

    - `SPS` (default)
    - `blockEnlarge`

compute_expm : bool, optional
    Whether to compute also `expm_A` in addition to `expm_frechet_AE`.
    Default is True.
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
expm_A : ndarray
    Matrix exponential of A.
expm_frechet_AE : ndarray
    Frechet derivative of the matrix exponential of A in the direction E.

For ``compute_expm = False``, only `expm_frechet_AE` is returned.

See also
--------
expm : Compute the exponential of a matrix.

Notes
-----
This section describes the available implementations that can be selected
by the `method` parameter. The default method is *SPS*.

Method *blockEnlarge* is a naive algorithm.

Method *SPS* is Scaling-Pade-Squaring [1]_.
It is a sophisticated implementation which should take
only about 3/8 as much time as the naive implementation.
The asymptotics are the same.

.. versionadded:: 0.13.0

References
----------
.. [1] Awad H. Al-Mohy and Nicholas J. Higham (2009)
       Computing the Frechet Derivative of the Matrix Exponential,
       with an application to Condition Number Estimation.
       SIAM Journal On Matrix Analysis and Applications.,
       30 (4). pp. 1639-1657. ISSN 1095-7162

Examples
--------
>>> import scipy.linalg
>>> A = np.random.randn(3, 3)
>>> E = np.random.randn(3, 3)
>>> expm_A, expm_frechet_AE = scipy.linalg.expm_frechet(A, E)
>>> expm_A.shape, expm_frechet_AE.shape
((3, 3), (3, 3))

>>> import scipy.linalg
>>> A = np.random.randn(3, 3)
>>> E = np.random.randn(3, 3)
>>> expm_A, expm_frechet_AE = scipy.linalg.expm_frechet(A, E)
>>> M = np.zeros((6, 6))
>>> M[:3, :3] = A; M[:3, 3:] = E; M[3:, 3:] = A
>>> expm_M = scipy.linalg.expm(M)
>>> np.allclose(expm_A, expm_M[:3, :3])
True
>>> np.allclose(expm_frechet_AE, expm_M[:3, 3:])
True
*)

val fractional_matrix_power : a:[>`Ndarray] Np.Obj.t -> t:float -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the fractional power of a matrix.

Proceeds according to the discussion in section (6) of [1]_.

Parameters
----------
A : (N, N) array_like
    Matrix whose fractional power to evaluate.
t : float
    Fractional power.

Returns
-------
X : (N, N) array_like
    The fractional power of the matrix.

References
----------
.. [1] Nicholas J. Higham and Lijing lin (2011)
       'A Schur-Pade Algorithm for Fractional Powers of a Matrix.'
       SIAM Journal on Matrix Analysis and Applications,
       32 (3). pp. 1056-1078. ISSN 0895-4798

Examples
--------
>>> from scipy.linalg import fractional_matrix_power
>>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
>>> b = fractional_matrix_power(a, 0.5)
>>> b
array([[ 0.75592895,  1.13389342],
       [ 0.37796447,  1.88982237]])
>>> np.dot(b, b)      # Verify square root
array([[ 1.,  3.],
       [ 1.,  4.]])
*)

val funm : ?disp:bool -> a:[>`Ndarray] Np.Obj.t -> func:Py.Object.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * float)
(**
Evaluate a matrix function specified by a callable.

Returns the value of matrix-valued function ``f`` at `A`. The
function ``f`` is an extension of the scalar-valued function `func`
to matrices.

Parameters
----------
A : (N, N) array_like
    Matrix at which to evaluate the function
func : callable
    Callable object that evaluates a scalar function f.
    Must be vectorized (eg. using vectorize).
disp : bool, optional
    Print warning if error in the result is estimated large
    instead of returning estimated error. (Default: True)

Returns
-------
funm : (N, N) ndarray
    Value of the matrix function specified by func evaluated at `A`
errest : float
    (if disp == False)

    1-norm of the estimated error, ||err||_1 / ||A||_1

Examples
--------
>>> from scipy.linalg import funm
>>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
>>> funm(a, lambda x: x*x)
array([[  4.,  15.],
       [  5.,  19.]])
>>> a.dot(a)
array([[  4.,  15.],
       [  5.,  19.]])

Notes
-----
This function implements the general algorithm based on Schur decomposition
(Algorithm 9.1.1. in [1]_).

If the input matrix is known to be diagonalizable, then relying on the
eigendecomposition is likely to be faster. For example, if your matrix is
Hermitian, you can do

>>> from scipy.linalg import eigh
>>> def funm_herm(a, func, check_finite=False):
...     w, v = eigh(a, check_finite=check_finite)
...     ## if you further know that your matrix is positive semidefinite,
...     ## you can optionally guard against precision errors by doing
...     # w = np.maximum(w, 0)
...     w = func(w)
...     return (v * w).dot(v.conj().T)

References
----------
.. [1] Gene H. Golub, Charles F. van Loan, Matrix Computations 4th ed.
*)

val inv : ?overwrite_a:bool -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the inverse of a matrix.

Parameters
----------
a : array_like
    Square matrix to be inverted.
overwrite_a : bool, optional
    Discard data in `a` (may improve performance). Default is False.
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
ainv : ndarray
    Inverse of the matrix `a`.

Raises
------
LinAlgError
    If `a` is singular.
ValueError
    If `a` is not square, or not 2-dimensional.

Examples
--------
>>> from scipy import linalg
>>> a = np.array([[1., 2.], [3., 4.]])
>>> linalg.inv(a)
array([[-2. ,  1. ],
       [ 1.5, -0.5]])
>>> np.dot(a, linalg.inv(a))
array([[ 1.,  0.],
       [ 0.,  1.]])
*)

val isfinite : ?out:[`Ndarray of [>`Ndarray] Np.Obj.t | `Tuple_of_ndarray_and_None of Py.Object.t] -> ?where:[>`Ndarray] Np.Obj.t -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
isfinite(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

Test element-wise for finiteness (not infinity or not Not a Number).

The result is returned as a boolean array.

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
y : ndarray, bool
    True where ``x`` is not positive infinity, negative infinity,
    or NaN; false otherwise.
    This is a scalar if `x` is a scalar.

See Also
--------
isinf, isneginf, isposinf, isnan

Notes
-----
Not a Number, positive infinity and negative infinity are considered
to be non-finite.

NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
(IEEE 754). This means that Not a Number is not equivalent to infinity.
Also that positive infinity is not equivalent to negative infinity. But
infinity is equivalent to positive infinity.  Errors result if the
second argument is also supplied when `x` is a scalar input, or if
first and second arguments have different shapes.

Examples
--------
>>> np.isfinite(1)
True
>>> np.isfinite(0)
True
>>> np.isfinite(np.nan)
False
>>> np.isfinite(np.inf)
False
>>> np.isfinite(np.NINF)
False
>>> np.isfinite([np.log(-1.),1.,np.log(0)])
array([False,  True, False])

>>> x = np.array([-np.inf, 0., np.inf])
>>> y = np.array([2, 2, 2])
>>> np.isfinite(x, y)
array([0, 1, 0])
>>> y
array([0, 1, 0])
*)

val logical_not : ?out:[`Ndarray of [>`Ndarray] Np.Obj.t | `Tuple_of_ndarray_and_None of Py.Object.t] -> ?where:[>`Ndarray] Np.Obj.t -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
logical_not(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

Compute the truth value of NOT x element-wise.

Parameters
----------
x : array_like
    Logical NOT is applied to the elements of `x`.
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
y : bool or ndarray of bool
    Boolean result with the same shape as `x` of the NOT operation
    on elements of `x`.
    This is a scalar if `x` is a scalar.

See Also
--------
logical_and, logical_or, logical_xor

Examples
--------
>>> np.logical_not(3)
False
>>> np.logical_not([True, False, 0, 1])
array([False,  True,  True, False])

>>> x = np.arange(5)
>>> np.logical_not(x<3)
array([False, False, False,  True,  True])
*)

val logm : ?disp:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * float)
(**
Compute matrix logarithm.

The matrix logarithm is the inverse of
expm: expm(logm(`A`)) == `A`

Parameters
----------
A : (N, N) array_like
    Matrix whose logarithm to evaluate
disp : bool, optional
    Print warning if error in the result is estimated large
    instead of returning estimated error. (Default: True)

Returns
-------
logm : (N, N) ndarray
    Matrix logarithm of `A`
errest : float
    (if disp == False)

    1-norm of the estimated error, ||err||_1 / ||A||_1

References
----------
.. [1] Awad H. Al-Mohy and Nicholas J. Higham (2012)
       'Improved Inverse Scaling and Squaring Algorithms
       for the Matrix Logarithm.'
       SIAM Journal on Scientific Computing, 34 (4). C152-C169.
       ISSN 1095-7197

.. [2] Nicholas J. Higham (2008)
       'Functions of Matrices: Theory and Computation'
       ISBN 978-0-898716-46-7

.. [3] Nicholas J. Higham and Lijing lin (2011)
       'A Schur-Pade Algorithm for Fractional Powers of a Matrix.'
       SIAM Journal on Matrix Analysis and Applications,
       32 (3). pp. 1056-1078. ISSN 0895-4798

Examples
--------
>>> from scipy.linalg import logm, expm
>>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
>>> b = logm(a)
>>> b
array([[-1.02571087,  2.05142174],
       [ 0.68380725,  1.02571087]])
>>> expm(b)         # Verify expm(logm(a)) returns a
array([[ 1.,  3.],
       [ 1.,  4.]])
*)

val norm : ?ord:[`Fro | `PyObject of Py.Object.t] -> ?axis:[`T2_tuple_of_ints of Py.Object.t | `I of int] -> ?keepdims:bool -> ?check_finite:bool -> a:Py.Object.t -> unit -> Py.Object.t
(**
Matrix or vector norm.

This function is able to return one of seven different matrix norms,
or one of an infinite number of vector norms (described below), depending
on the value of the ``ord`` parameter.

Parameters
----------
a : (M,) or (M, N) array_like
    Input array.  If `axis` is None, `a` must be 1-D or 2-D.
ord : {non-zero int, inf, -inf, 'fro'}, optional
    Order of the norm (see table under ``Notes``). inf means numpy's
    `inf` object
axis : {int, 2-tuple of ints, None}, optional
    If `axis` is an integer, it specifies the axis of `a` along which to
    compute the vector norms.  If `axis` is a 2-tuple, it specifies the
    axes that hold 2-D matrices, and the matrix norms of these matrices
    are computed.  If `axis` is None then either a vector norm (when `a`
    is 1-D) or a matrix norm (when `a` is 2-D) is returned.
keepdims : bool, optional
    If this is set to True, the axes which are normed over are left in the
    result as dimensions with size one.  With this option the result will
    broadcast correctly against the original `a`.
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
n : float or ndarray
    Norm of the matrix or vector(s).

Notes
-----
For values of ``ord <= 0``, the result is, strictly speaking, not a
mathematical 'norm', but it may still be useful for various numerical
purposes.

The following norms can be calculated:

=====  ============================  ==========================
ord    norm for matrices             norm for vectors
=====  ============================  ==========================
None   Frobenius norm                2-norm
'fro'  Frobenius norm                --
inf    max(sum(abs(x), axis=1))      max(abs(x))
-inf   min(sum(abs(x), axis=1))      min(abs(x))
0      --                            sum(x != 0)
1      max(sum(abs(x), axis=0))      as below
-1     min(sum(abs(x), axis=0))      as below
2      2-norm (largest sing. value)  as below
-2     smallest singular value       as below
other  --                            sum(abs(x)**ord)**(1./ord)
=====  ============================  ==========================

The Frobenius norm is given by [1]_:

    :math:`||A||_F = [\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

The ``axis`` and ``keepdims`` arguments are passed directly to
``numpy.linalg.norm`` and are only usable if they are supported
by the version of numpy in use.

References
----------
.. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
       Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15

Examples
--------
>>> from scipy.linalg import norm
>>> a = np.arange(9) - 4.0
>>> a
array([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
>>> b = a.reshape((3, 3))
>>> b
array([[-4., -3., -2.],
       [-1.,  0.,  1.],
       [ 2.,  3.,  4.]])

>>> norm(a)
7.745966692414834
>>> norm(b)
7.745966692414834
>>> norm(b, 'fro')
7.745966692414834
>>> norm(a, np.inf)
4
>>> norm(b, np.inf)
9
>>> norm(a, -np.inf)
0
>>> norm(b, -np.inf)
2

>>> norm(a, 1)
20
>>> norm(b, 1)
7
>>> norm(a, -1)
-4.6566128774142013e-010
>>> norm(b, -1)
6
>>> norm(a, 2)
7.745966692414834
>>> norm(b, 2)
7.3484692283495345

>>> norm(a, -2)
0
>>> norm(b, -2)
1.8570331885190563e-016
>>> norm(a, 3)
5.8480354764257312
>>> norm(a, -3)
0
*)

val prod : ?axis:int list -> ?dtype:Np.Dtype.t -> ?out:[>`Ndarray] Np.Obj.t -> ?keepdims:bool -> ?initial:[`F of float | `I of int | `Bool of bool | `S of string] -> ?where:Py.Object.t -> a:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Return the product of array elements over a given axis.

Parameters
----------
a : array_like
    Input data.
axis : None or int or tuple of ints, optional
    Axis or axes along which a product is performed.  The default,
    axis=None, will calculate the product of all the elements in the
    input array. If axis is negative it counts from the last to the
    first axis.

    .. versionadded:: 1.7.0

    If axis is a tuple of ints, a product is performed on all of the
    axes specified in the tuple instead of a single axis or all the
    axes as before.
dtype : dtype, optional
    The type of the returned array, as well as of the accumulator in
    which the elements are multiplied.  The dtype of `a` is used by
    default unless `a` has an integer dtype of less precision than the
    default platform integer.  In that case, if `a` is signed then the
    platform integer is used while if `a` is unsigned then an unsigned
    integer of the same precision as the platform integer is used.
out : ndarray, optional
    Alternative output array in which to place the result. It must have
    the same shape as the expected output, but the type of the output
    values will be cast if necessary.
keepdims : bool, optional
    If this is set to True, the axes which are reduced are left in the
    result as dimensions with size one. With this option, the result
    will broadcast correctly against the input array.

    If the default value is passed, then `keepdims` will not be
    passed through to the `prod` method of sub-classes of
    `ndarray`, however any non-default value will be.  If the
    sub-class' method does not implement `keepdims` any
    exceptions will be raised.
initial : scalar, optional
    The starting value for this product. See `~numpy.ufunc.reduce` for details.

    .. versionadded:: 1.15.0

where : array_like of bool, optional
    Elements to include in the product. See `~numpy.ufunc.reduce` for details.

    .. versionadded:: 1.17.0

Returns
-------
product_along_axis : ndarray, see `dtype` parameter above.
    An array shaped as `a` but with the specified axis removed.
    Returns a reference to `out` if specified.

See Also
--------
ndarray.prod : equivalent method
ufuncs-output-type

Notes
-----
Arithmetic is modular when using integer types, and no error is
raised on overflow.  That means that, on a 32-bit platform:

>>> x = np.array([536870910, 536870910, 536870910, 536870910])
>>> np.prod(x)
16 # may vary

The product of an empty array is the neutral element 1:

>>> np.prod([])
1.0

Examples
--------
By default, calculate the product of all elements:

>>> np.prod([1.,2.])
2.0

Even when the input array is two-dimensional:

>>> np.prod([[1.,2.],[3.,4.]])
24.0

But we can also specify the axis over which to multiply:

>>> np.prod([[1.,2.],[3.,4.]], axis=1)
array([  2.,  12.])

Or select specific elements to include:

>>> np.prod([1., np.nan, 3.], where=[True, False, True])
3.0

If the type of `x` is unsigned, then the output type is
the unsigned platform integer:

>>> x = np.array([1, 2, 3], dtype=np.uint8)
>>> np.prod(x).dtype == np.uint
True

If `x` is of a signed integer type, then the output type
is the default platform integer:

>>> x = np.array([1, 2, 3], dtype=np.int8)
>>> np.prod(x).dtype == int
True

You can also start the product with a value other than one:

>>> np.prod([1, 2], initial=5)
10
*)

val ravel : ?order:[`C | `F | `A | `K] -> a:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return a contiguous flattened array.

A 1-D array, containing the elements of the input, is returned.  A copy is
made only if needed.

As of NumPy 1.10, the returned array will have the same type as the input
array. (for example, a masked array will be returned for a masked array
input)

Parameters
----------
a : array_like
    Input array.  The elements in `a` are read in the order specified by
    `order`, and packed as a 1-D array.
order : {'C','F', 'A', 'K'}, optional

    The elements of `a` are read using this index order. 'C' means
    to index the elements in row-major, C-style order,
    with the last axis index changing fastest, back to the first
    axis index changing slowest.  'F' means to index the elements
    in column-major, Fortran-style order, with the
    first index changing fastest, and the last index changing
    slowest. Note that the 'C' and 'F' options take no account of
    the memory layout of the underlying array, and only refer to
    the order of axis indexing.  'A' means to read the elements in
    Fortran-like index order if `a` is Fortran *contiguous* in
    memory, C-like order otherwise.  'K' means to read the
    elements in the order they occur in memory, except for
    reversing the data when strides are negative.  By default, 'C'
    index order is used.

Returns
-------
y : array_like
    y is an array of the same subtype as `a`, with shape ``(a.size,)``.
    Note that matrices are special cased for backward compatibility, if `a`
    is a matrix, then y is a 1-D ndarray.

See Also
--------
ndarray.flat : 1-D iterator over an array.
ndarray.flatten : 1-D array copy of the elements of an array
                  in row-major order.
ndarray.reshape : Change the shape of an array without changing its data.

Notes
-----
In row-major, C-style order, in two dimensions, the row index
varies the slowest, and the column index the quickest.  This can
be generalized to multiple dimensions, where row-major order
implies that the index along the first axis varies slowest, and
the index along the last quickest.  The opposite holds for
column-major, Fortran-style index ordering.

When a view is desired in as many cases as possible, ``arr.reshape(-1)``
may be preferable.

Examples
--------
It is equivalent to ``reshape(-1, order=order)``.

>>> x = np.array([[1, 2, 3], [4, 5, 6]])
>>> np.ravel(x)
array([1, 2, 3, 4, 5, 6])

>>> x.reshape(-1)
array([1, 2, 3, 4, 5, 6])

>>> np.ravel(x, order='F')
array([1, 4, 2, 5, 3, 6])

When ``order`` is 'A', it will preserve the array's 'C' or 'F' ordering:

>>> np.ravel(x.T)
array([1, 4, 2, 5, 3, 6])
>>> np.ravel(x.T, order='A')
array([1, 2, 3, 4, 5, 6])

When ``order`` is 'K', it will preserve orderings that are neither 'C'
nor 'F', but won't reverse axes:

>>> a = np.arange(3)[::-1]; a
array([2, 1, 0])
>>> a.ravel(order='C')
array([2, 1, 0])
>>> a.ravel(order='K')
array([2, 1, 0])

>>> a = np.arange(12).reshape(2,3,2).swapaxes(1,2); a
array([[[ 0,  2,  4],
        [ 1,  3,  5]],
       [[ 6,  8, 10],
        [ 7,  9, 11]]])
>>> a.ravel(order='C')
array([ 0,  2,  4,  1,  3,  5,  6,  8, 10,  7,  9, 11])
>>> a.ravel(order='K')
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
*)

val rsf2csf : ?check_finite:bool -> t:[>`Ndarray] Np.Obj.t -> z:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Convert real Schur form to complex Schur form.

Convert a quasi-diagonal real-valued Schur form to the upper triangular
complex-valued Schur form.

Parameters
----------
T : (M, M) array_like
    Real Schur form of the original array
Z : (M, M) array_like
    Schur transformation matrix
check_finite : bool, optional
    Whether to check that the input arrays contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
T : (M, M) ndarray
    Complex Schur form of the original array
Z : (M, M) ndarray
    Schur transformation matrix corresponding to the complex form

See Also
--------
schur : Schur decomposition of an array

Examples
--------
>>> from scipy.linalg import schur, rsf2csf
>>> A = np.array([[0, 2, 2], [0, 1, 2], [1, 0, 1]])
>>> T, Z = schur(A)
>>> T
array([[ 2.65896708,  1.42440458, -1.92933439],
       [ 0.        , -0.32948354, -0.49063704],
       [ 0.        ,  1.31178921, -0.32948354]])
>>> Z
array([[0.72711591, -0.60156188, 0.33079564],
       [0.52839428, 0.79801892, 0.28976765],
       [0.43829436, 0.03590414, -0.89811411]])
>>> T2 , Z2 = rsf2csf(T, Z)
>>> T2
array([[2.65896708+0.j, -1.64592781+0.743164187j, -1.21516887+1.00660462j],
       [0.+0.j , -0.32948354+8.02254558e-01j, -0.82115218-2.77555756e-17j],
       [0.+0.j , 0.+0.j, -0.32948354-0.802254558j]])
>>> Z2
array([[0.72711591+0.j,  0.28220393-0.31385693j,  0.51319638-0.17258824j],
       [0.52839428+0.j,  0.24720268+0.41635578j, -0.68079517-0.15118243j],
       [0.43829436+0.j, -0.76618703+0.01873251j, -0.03063006+0.46857912j]])
*)

val schur : ?output:[`Real | `Complex] -> ?lwork:int -> ?overwrite_a:bool -> ?sort:[`Iuc | `Ouc | `Rhp | `Callable of Py.Object.t | `Lhp] -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Compute Schur decomposition of a matrix.

The Schur decomposition is::

    A = Z T Z^H

where Z is unitary and T is either upper-triangular, or for real
Schur decomposition (output='real'), quasi-upper triangular.  In
the quasi-triangular form, 2x2 blocks describing complex-valued
eigenvalue pairs may extrude from the diagonal.

Parameters
----------
a : (M, M) array_like
    Matrix to decompose
output : {'real', 'complex'}, optional
    Construct the real or complex Schur decomposition (for real matrices).
lwork : int, optional
    Work array size. If None or -1, it is automatically computed.
overwrite_a : bool, optional
    Whether to overwrite data in a (may improve performance).
sort : {None, callable, 'lhp', 'rhp', 'iuc', 'ouc'}, optional
    Specifies whether the upper eigenvalues should be sorted.  A callable
    may be passed that, given a eigenvalue, returns a boolean denoting
    whether the eigenvalue should be sorted to the top-left (True).
    Alternatively, string parameters may be used::

        'lhp'   Left-hand plane (x.real < 0.0)
        'rhp'   Right-hand plane (x.real > 0.0)
        'iuc'   Inside the unit circle (x*x.conjugate() <= 1.0)
        'ouc'   Outside the unit circle (x*x.conjugate() > 1.0)

    Defaults to None (no sorting).
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
T : (M, M) ndarray
    Schur form of A. It is real-valued for the real Schur decomposition.
Z : (M, M) ndarray
    An unitary Schur transformation matrix for A.
    It is real-valued for the real Schur decomposition.
sdim : int
    If and only if sorting was requested, a third return value will
    contain the number of eigenvalues satisfying the sort condition.

Raises
------
LinAlgError
    Error raised under three conditions:

    1. The algorithm failed due to a failure of the QR algorithm to
       compute all eigenvalues
    2. If eigenvalue sorting was requested, the eigenvalues could not be
       reordered due to a failure to separate eigenvalues, usually because
       of poor conditioning
    3. If eigenvalue sorting was requested, roundoff errors caused the
       leading eigenvalues to no longer satisfy the sorting condition

See also
--------
rsf2csf : Convert real Schur form to complex Schur form

Examples
--------
>>> from scipy.linalg import schur, eigvals
>>> A = np.array([[0, 2, 2], [0, 1, 2], [1, 0, 1]])
>>> T, Z = schur(A)
>>> T
array([[ 2.65896708,  1.42440458, -1.92933439],
       [ 0.        , -0.32948354, -0.49063704],
       [ 0.        ,  1.31178921, -0.32948354]])
>>> Z
array([[0.72711591, -0.60156188, 0.33079564],
       [0.52839428, 0.79801892, 0.28976765],
       [0.43829436, 0.03590414, -0.89811411]])

>>> T2, Z2 = schur(A, output='complex')
>>> T2
array([[ 2.65896708, -1.22839825+1.32378589j,  0.42590089+1.51937378j],
       [ 0.        , -0.32948354+0.80225456j, -0.59877807+0.56192146j],
       [ 0.        ,  0.                    , -0.32948354-0.80225456j]])
>>> eigvals(T2)
array([2.65896708, -0.32948354+0.80225456j, -0.32948354-0.80225456j])

An arbitrary custom eig-sorting condition, having positive imaginary part, 
which is satisfied by only one eigenvalue

>>> T3, Z3, sdim = schur(A, output='complex', sort=lambda x: x.imag > 0)
>>> sdim
1
*)

val sign : ?out:[`Ndarray of [>`Ndarray] Np.Obj.t | `Tuple_of_ndarray_and_None of Py.Object.t] -> ?where:[>`Ndarray] Np.Obj.t -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val signm : ?disp:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * float)
(**
Matrix sign function.

Extension of the scalar sign(x) to matrices.

Parameters
----------
A : (N, N) array_like
    Matrix at which to evaluate the sign function
disp : bool, optional
    Print warning if error in the result is estimated large
    instead of returning estimated error. (Default: True)

Returns
-------
signm : (N, N) ndarray
    Value of the sign function at `A`
errest : float
    (if disp == False)

    1-norm of the estimated error, ||err||_1 / ||A||_1

Examples
--------
>>> from scipy.linalg import signm, eigvals
>>> a = [[1,2,3], [1,2,1], [1,1,1]]
>>> eigvals(a)
array([ 4.12488542+0.j, -0.76155718+0.j,  0.63667176+0.j])
>>> eigvals(signm(a))
array([-1.+0.j,  1.+0.j,  1.+0.j])
*)

val sinhm : [>`Ndarray] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the hyperbolic matrix sine.

This routine uses expm to compute the matrix exponentials.

Parameters
----------
A : (N, N) array_like
    Input array.

Returns
-------
sinhm : (N, N) ndarray
    Hyperbolic matrix sine of `A`

Examples
--------
>>> from scipy.linalg import tanhm, sinhm, coshm
>>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
>>> s = sinhm(a)
>>> s
array([[ 10.57300653,  39.28826594],
       [ 13.09608865,  49.86127247]])

Verify tanhm(a) = sinhm(a).dot(inv(coshm(a)))

>>> t = tanhm(a)
>>> c = coshm(a)
>>> t - s.dot(np.linalg.inv(c))
array([[  2.72004641e-15,   4.55191440e-15],
       [  0.00000000e+00,  -5.55111512e-16]])
*)

val sinm : [>`Ndarray] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the matrix sine.

This routine uses expm to compute the matrix exponentials.

Parameters
----------
A : (N, N) array_like
    Input array.

Returns
-------
sinm : (N, N) ndarray
    Matrix sine of `A`

Examples
--------
>>> from scipy.linalg import expm, sinm, cosm

Euler's identity (exp(i*theta) = cos(theta) + i*sin(theta))
applied to a matrix:

>>> a = np.array([[1.0, 2.0], [-1.0, 3.0]])
>>> expm(1j*a)
array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
       [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
>>> cosm(a) + 1j*sinm(a)
array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
       [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
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

val sqrtm : ?disp:bool -> ?blocksize:int -> a:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * float)
(**
Matrix square root.

Parameters
----------
A : (N, N) array_like
    Matrix whose square root to evaluate
disp : bool, optional
    Print warning if error in the result is estimated large
    instead of returning estimated error. (Default: True)
blocksize : integer, optional
    If the blocksize is not degenerate with respect to the
    size of the input array, then use a blocked algorithm. (Default: 64)

Returns
-------
sqrtm : (N, N) ndarray
    Value of the sqrt function at `A`

errest : float
    (if disp == False)

    Frobenius norm of the estimated error, ||err||_F / ||A||_F

References
----------
.. [1] Edvin Deadman, Nicholas J. Higham, Rui Ralha (2013)
       'Blocked Schur Algorithms for Computing the Matrix Square Root,
       Lecture Notes in Computer Science, 7782. pp. 171-182.

Examples
--------
>>> from scipy.linalg import sqrtm
>>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
>>> r = sqrtm(a)
>>> r
array([[ 0.75592895,  1.13389342],
       [ 0.37796447,  1.88982237]])
>>> r.dot(r)
array([[ 1.,  3.],
       [ 1.,  4.]])
*)

val svd : ?full_matrices:bool -> ?compute_uv:bool -> ?overwrite_a:bool -> ?check_finite:bool -> ?lapack_driver:[`Gesdd | `Gesvd] -> a:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
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

val tanhm : [>`Ndarray] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the hyperbolic matrix tangent.

This routine uses expm to compute the matrix exponentials.

Parameters
----------
A : (N, N) array_like
    Input array

Returns
-------
tanhm : (N, N) ndarray
    Hyperbolic matrix tangent of `A`

Examples
--------
>>> from scipy.linalg import tanhm, sinhm, coshm
>>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
>>> t = tanhm(a)
>>> t
array([[ 0.3428582 ,  0.51987926],
       [ 0.17329309,  0.86273746]])

Verify tanhm(a) = sinhm(a).dot(inv(coshm(a)))

>>> s = sinhm(a)
>>> c = coshm(a)
>>> t - s.dot(np.linalg.inv(c))
array([[  2.72004641e-15,   4.55191440e-15],
       [  0.00000000e+00,  -5.55111512e-16]])
*)

val tanm : [>`Ndarray] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the matrix tangent.

This routine uses expm to compute the matrix exponentials.

Parameters
----------
A : (N, N) array_like
    Input array.

Returns
-------
tanm : (N, N) ndarray
    Matrix tangent of `A`

Examples
--------
>>> from scipy.linalg import tanm, sinm, cosm
>>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
>>> t = tanm(a)
>>> t
array([[ -2.00876993,  -8.41880636],
       [ -2.80626879, -10.42757629]])

Verify tanm(a) = sinm(a).dot(inv(cosm(a)))

>>> s = sinm(a)
>>> c = cosm(a)
>>> s.dot(np.linalg.inv(c))
array([[ -2.00876993,  -8.41880636],
       [ -2.80626879, -10.42757629]])
*)

val transpose : ?axes:Py.Object.t -> a:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Permute the dimensions of an array.

Parameters
----------
a : array_like
    Input array.
axes : list of ints, optional
    By default, reverse the dimensions, otherwise permute the axes
    according to the values given.

Returns
-------
p : ndarray
    `a` with its axes permuted.  A view is returned whenever
    possible.

See Also
--------
moveaxis
argsort

Notes
-----
Use `transpose(a, argsort(axes))` to invert the transposition of tensors
when using the `axes` keyword argument.

Transposing a 1-D array returns an unchanged view of the original array.

Examples
--------
>>> x = np.arange(4).reshape((2,2))
>>> x
array([[0, 1],
       [2, 3]])

>>> np.transpose(x)
array([[0, 2],
       [1, 3]])

>>> x = np.ones((1, 2, 3))
>>> np.transpose(x, (1, 0, 2)).shape
(2, 1, 3)
*)

val triu : ?k:int -> m:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Make a copy of a matrix with elements below the k-th diagonal zeroed.

Parameters
----------
m : array_like
    Matrix whose elements to return
k : int, optional
    Diagonal below which to zero elements.
    `k` == 0 is the main diagonal, `k` < 0 subdiagonal and
    `k` > 0 superdiagonal.

Returns
-------
triu : ndarray
    Return matrix with zeroed elements below the k-th diagonal and has
    same shape and type as `m`.

Examples
--------
>>> from scipy.linalg import triu
>>> triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)
array([[ 1,  2,  3],
       [ 4,  5,  6],
       [ 0,  8,  9],
       [ 0,  0, 12]])
*)


end

module Misc : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val get_blas_funcs : ?arrays:[>`Ndarray] Np.Obj.t list -> ?dtype:[`S of string | `Dtype of Np.Dtype.t] -> names:[`Sequence_of_str of Py.Object.t | `S of string] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return available BLAS function objects from names.

Arrays are used to determine the optimal prefix of BLAS routines.

Parameters
----------
names : str or sequence of str
    Name(s) of BLAS functions without type prefix.

arrays : sequence of ndarrays, optional
    Arrays can be given to determine optimal prefix of BLAS
    routines. If not given, double-precision routines will be
    used, otherwise the most generic type in arrays will be used.

dtype : str or dtype, optional
    Data-type specifier. Not used if `arrays` is non-empty.


Returns
-------
funcs : list
    List containing the found function(s).


Notes
-----
This routine automatically chooses between Fortran/C
interfaces. Fortran code is used whenever possible for arrays with
column major order. In all other cases, C code is preferred.

In BLAS, the naming convention is that all functions start with a
type prefix, which depends on the type of the principal
matrix. These can be one of {'s', 'd', 'c', 'z'} for the numpy
types {float32, float64, complex64, complex128} respectively.
The code and the dtype are stored in attributes `typecode` and `dtype`
of the returned functions.

Examples
--------
>>> import scipy.linalg as LA
>>> a = np.random.rand(3,2)
>>> x_gemv = LA.get_blas_funcs('gemv', (a,))
>>> x_gemv.typecode
'd'
>>> x_gemv = LA.get_blas_funcs('gemv',(a*1j,))
>>> x_gemv.typecode
'z'
*)

val get_lapack_funcs : ?arrays:[>`Ndarray] Np.Obj.t list -> ?dtype:[`S of string | `Dtype of Np.Dtype.t] -> names:[`Sequence_of_str of Py.Object.t | `S of string] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return available LAPACK function objects from names.

Arrays are used to determine the optimal prefix of LAPACK routines.

Parameters
----------
names : str or sequence of str
    Name(s) of LAPACK functions without type prefix.

arrays : sequence of ndarrays, optional
    Arrays can be given to determine optimal prefix of LAPACK
    routines. If not given, double-precision routines will be
    used, otherwise the most generic type in arrays will be used.

dtype : str or dtype, optional
    Data-type specifier. Not used if `arrays` is non-empty.

Returns
-------
funcs : list
    List containing the found function(s).

Notes
-----
This routine automatically chooses between Fortran/C
interfaces. Fortran code is used whenever possible for arrays with
column major order. In all other cases, C code is preferred.

In LAPACK, the naming convention is that all functions start with a
type prefix, which depends on the type of the principal
matrix. These can be one of {'s', 'd', 'c', 'z'} for the numpy
types {float32, float64, complex64, complex128} respectively, and
are stored in attribute ``typecode`` of the returned functions.

Examples
--------
Suppose we would like to use '?lange' routine which computes the selected
norm of an array. We pass our array in order to get the correct 'lange'
flavor.

>>> import scipy.linalg as LA
>>> a = np.random.rand(3,2)
>>> x_lange = LA.get_lapack_funcs('lange', (a,))
>>> x_lange.typecode
'd'
>>> x_lange = LA.get_lapack_funcs('lange',(a*1j,))
>>> x_lange.typecode
'z'

Several LAPACK routines work best when its internal WORK array has
the optimal size (big enough for fast computation and small enough to
avoid waste of memory). This size is determined also by a dedicated query
to the function which is often wrapped as a standalone function and
commonly denoted as ``###_lwork``. Below is an example for ``?sysv``

>>> import scipy.linalg as LA
>>> a = np.random.rand(1000,1000)
>>> b = np.random.rand(1000,1)*1j
>>> # We pick up zsysv and zsysv_lwork due to b array
... xsysv, xlwork = LA.get_lapack_funcs(('sysv', 'sysv_lwork'), (a, b))
>>> opt_lwork, _ = xlwork(a.shape[0])  # returns a complex for 'z' prefix
>>> udut, ipiv, x, info = xsysv(a, b, lwork=int(opt_lwork.real))
*)

val norm : ?ord:[`Fro | `PyObject of Py.Object.t] -> ?axis:[`T2_tuple_of_ints of Py.Object.t | `I of int] -> ?keepdims:bool -> ?check_finite:bool -> a:Py.Object.t -> unit -> Py.Object.t
(**
Matrix or vector norm.

This function is able to return one of seven different matrix norms,
or one of an infinite number of vector norms (described below), depending
on the value of the ``ord`` parameter.

Parameters
----------
a : (M,) or (M, N) array_like
    Input array.  If `axis` is None, `a` must be 1-D or 2-D.
ord : {non-zero int, inf, -inf, 'fro'}, optional
    Order of the norm (see table under ``Notes``). inf means numpy's
    `inf` object
axis : {int, 2-tuple of ints, None}, optional
    If `axis` is an integer, it specifies the axis of `a` along which to
    compute the vector norms.  If `axis` is a 2-tuple, it specifies the
    axes that hold 2-D matrices, and the matrix norms of these matrices
    are computed.  If `axis` is None then either a vector norm (when `a`
    is 1-D) or a matrix norm (when `a` is 2-D) is returned.
keepdims : bool, optional
    If this is set to True, the axes which are normed over are left in the
    result as dimensions with size one.  With this option the result will
    broadcast correctly against the original `a`.
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
n : float or ndarray
    Norm of the matrix or vector(s).

Notes
-----
For values of ``ord <= 0``, the result is, strictly speaking, not a
mathematical 'norm', but it may still be useful for various numerical
purposes.

The following norms can be calculated:

=====  ============================  ==========================
ord    norm for matrices             norm for vectors
=====  ============================  ==========================
None   Frobenius norm                2-norm
'fro'  Frobenius norm                --
inf    max(sum(abs(x), axis=1))      max(abs(x))
-inf   min(sum(abs(x), axis=1))      min(abs(x))
0      --                            sum(x != 0)
1      max(sum(abs(x), axis=0))      as below
-1     min(sum(abs(x), axis=0))      as below
2      2-norm (largest sing. value)  as below
-2     smallest singular value       as below
other  --                            sum(abs(x)**ord)**(1./ord)
=====  ============================  ==========================

The Frobenius norm is given by [1]_:

    :math:`||A||_F = [\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

The ``axis`` and ``keepdims`` arguments are passed directly to
``numpy.linalg.norm`` and are only usable if they are supported
by the version of numpy in use.

References
----------
.. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
       Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15

Examples
--------
>>> from scipy.linalg import norm
>>> a = np.arange(9) - 4.0
>>> a
array([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
>>> b = a.reshape((3, 3))
>>> b
array([[-4., -3., -2.],
       [-1.,  0.,  1.],
       [ 2.,  3.,  4.]])

>>> norm(a)
7.745966692414834
>>> norm(b)
7.745966692414834
>>> norm(b, 'fro')
7.745966692414834
>>> norm(a, np.inf)
4
>>> norm(b, np.inf)
9
>>> norm(a, -np.inf)
0
>>> norm(b, -np.inf)
2

>>> norm(a, 1)
20
>>> norm(b, 1)
7
>>> norm(a, -1)
-4.6566128774142013e-010
>>> norm(b, -1)
6
>>> norm(a, 2)
7.745966692414834
>>> norm(b, 2)
7.3484692283495345

>>> norm(a, -2)
0
>>> norm(b, -2)
1.8570331885190563e-016
>>> norm(a, 3)
5.8480354764257312
>>> norm(a, -3)
0
*)


end

module Special_matrices : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val as_strided : ?shape:Py.Object.t -> ?strides:Py.Object.t -> ?subok:bool -> ?writeable:bool -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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

val block_diag : Py.Object.t list -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Create a block diagonal matrix from provided arrays.

Given the inputs `A`, `B` and `C`, the output will have these
arrays arranged on the diagonal::

    [[A, 0, 0],
     [0, B, 0],
     [0, 0, C]]

Parameters
----------
A, B, C, ... : array_like, up to 2-D
    Input arrays.  A 1-D array or array_like sequence of length `n` is
    treated as a 2-D array with shape ``(1,n)``.

Returns
-------
D : ndarray
    Array with `A`, `B`, `C`, ... on the diagonal.  `D` has the
    same dtype as `A`.

Notes
-----
If all the input arrays are square, the output is known as a
block diagonal matrix.

Empty sequences (i.e., array-likes of zero size) will not be ignored.
Noteworthy, both [] and [[]] are treated as matrices with shape ``(1,0)``.

Examples
--------
>>> from scipy.linalg import block_diag
>>> A = [[1, 0],
...      [0, 1]]
>>> B = [[3, 4, 5],
...      [6, 7, 8]]
>>> C = [[7]]
>>> P = np.zeros((2, 0), dtype='int32')
>>> block_diag(A, B, C)
array([[1, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0],
       [0, 0, 3, 4, 5, 0],
       [0, 0, 6, 7, 8, 0],
       [0, 0, 0, 0, 0, 7]])
>>> block_diag(A, P, B, C)
array([[1, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 3, 4, 5, 0],
       [0, 0, 6, 7, 8, 0],
       [0, 0, 0, 0, 0, 7]])
>>> block_diag(1.0, [2, 3], [[4, 5], [6, 7]])
array([[ 1.,  0.,  0.,  0.,  0.],
       [ 0.,  2.,  3.,  0.,  0.],
       [ 0.,  0.,  0.,  4.,  5.],
       [ 0.,  0.,  0.,  6.,  7.]])
*)

val circulant : [>`Ndarray] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Construct a circulant matrix.

Parameters
----------
c : (N,) array_like
    1-D array, the first column of the matrix.

Returns
-------
A : (N, N) ndarray
    A circulant matrix whose first column is `c`.

See Also
--------
toeplitz : Toeplitz matrix
hankel : Hankel matrix
solve_circulant : Solve a circulant system.

Notes
-----
.. versionadded:: 0.8.0

Examples
--------
>>> from scipy.linalg import circulant
>>> circulant([1, 2, 3])
array([[1, 3, 2],
       [2, 1, 3],
       [3, 2, 1]])
*)

val companion : [>`Ndarray] Np.Obj.t -> Py.Object.t
(**
Create a companion matrix.

Create the companion matrix [1]_ associated with the polynomial whose
coefficients are given in `a`.

Parameters
----------
a : (N,) array_like
    1-D array of polynomial coefficients.  The length of `a` must be
    at least two, and ``a[0]`` must not be zero.

Returns
-------
c : (N-1, N-1) ndarray
    The first row of `c` is ``-a[1:]/a[0]``, and the first
    sub-diagonal is all ones.  The data-type of the array is the same
    as the data-type of ``1.0*a[0]``.

Raises
------
ValueError
    If any of the following are true: a) ``a.ndim != 1``;
    b) ``a.size < 2``; c) ``a[0] == 0``.

Notes
-----
.. versionadded:: 0.8.0

References
----------
.. [1] R. A. Horn & C. R. Johnson, *Matrix Analysis*.  Cambridge, UK:
    Cambridge University Press, 1999, pp. 146-7.

Examples
--------
>>> from scipy.linalg import companion
>>> companion([1, -10, 31, -30])
array([[ 10., -31.,  30.],
       [  1.,   0.,   0.],
       [  0.,   1.,   0.]])
*)

val dft : ?scale:float -> n:int -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Discrete Fourier transform matrix.

Create the matrix that computes the discrete Fourier transform of a
sequence [1]_.  The n-th primitive root of unity used to generate the
matrix is exp(-2*pi*i/n), where i = sqrt(-1).

Parameters
----------
n : int
    Size the matrix to create.
scale : str, optional
    Must be None, 'sqrtn', or 'n'.
    If `scale` is 'sqrtn', the matrix is divided by `sqrt(n)`.
    If `scale` is 'n', the matrix is divided by `n`.
    If `scale` is None (the default), the matrix is not normalized, and the
    return value is simply the Vandermonde matrix of the roots of unity.

Returns
-------
m : (n, n) ndarray
    The DFT matrix.

Notes
-----
When `scale` is None, multiplying a vector by the matrix returned by
`dft` is mathematically equivalent to (but much less efficient than)
the calculation performed by `scipy.fft.fft`.

.. versionadded:: 0.14.0

References
----------
.. [1] 'DFT matrix', https://en.wikipedia.org/wiki/DFT_matrix

Examples
--------
>>> from scipy.linalg import dft
>>> np.set_printoptions(precision=2, suppress=True)  # for compact output
>>> m = dft(5)
>>> m
array([[ 1.  +0.j  ,  1.  +0.j  ,  1.  +0.j  ,  1.  +0.j  ,  1.  +0.j  ],
       [ 1.  +0.j  ,  0.31-0.95j, -0.81-0.59j, -0.81+0.59j,  0.31+0.95j],
       [ 1.  +0.j  , -0.81-0.59j,  0.31+0.95j,  0.31-0.95j, -0.81+0.59j],
       [ 1.  +0.j  , -0.81+0.59j,  0.31-0.95j,  0.31+0.95j, -0.81-0.59j],
       [ 1.  +0.j  ,  0.31+0.95j, -0.81+0.59j, -0.81-0.59j,  0.31-0.95j]])
>>> x = np.array([1, 2, 3, 0, 3])
>>> m @ x  # Compute the DFT of x
array([ 9.  +0.j  ,  0.12-0.81j, -2.12+3.44j, -2.12-3.44j,  0.12+0.81j])

Verify that ``m @ x`` is the same as ``fft(x)``.

>>> from scipy.fft import fft
>>> fft(x)     # Same result as m @ x
array([ 9.  +0.j  ,  0.12-0.81j, -2.12+3.44j, -2.12-3.44j,  0.12+0.81j])
*)

val fiedler : [>`Ndarray] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Returns a symmetric Fiedler matrix

Given an sequence of numbers `a`, Fiedler matrices have the structure
``F[i, j] = np.abs(a[i] - a[j])``, and hence zero diagonals and nonnegative
entries. A Fiedler matrix has a dominant positive eigenvalue and other
eigenvalues are negative. Although not valid generally, for certain inputs,
the inverse and the determinant can be derived explicitly as given in [1]_.

Parameters
----------
a : (n,) array_like
    coefficient array

Returns
-------
F : (n, n) ndarray

See Also
--------
circulant, toeplitz

Notes
-----

.. versionadded:: 1.3.0

References
----------
.. [1] J. Todd, 'Basic Numerical Mathematics: Vol.2 : Numerical Algebra',
    1977, Birkhauser, :doi:`10.1007/978-3-0348-7286-7`

Examples
--------
>>> from scipy.linalg import det, inv, fiedler
>>> a = [1, 4, 12, 45, 77]
>>> n = len(a)
>>> A = fiedler(a)
>>> A
array([[ 0,  3, 11, 44, 76],
       [ 3,  0,  8, 41, 73],
       [11,  8,  0, 33, 65],
       [44, 41, 33,  0, 32],
       [76, 73, 65, 32,  0]])

The explicit formulas for determinant and inverse seem to hold only for
monotonically increasing/decreasing arrays. Note the tridiagonal structure
and the corners.

>>> Ai = inv(A)
>>> Ai[np.abs(Ai) < 1e-12] = 0.  # cleanup the numerical noise for display
>>> Ai
array([[-0.16008772,  0.16666667,  0.        ,  0.        ,  0.00657895],
       [ 0.16666667, -0.22916667,  0.0625    ,  0.        ,  0.        ],
       [ 0.        ,  0.0625    , -0.07765152,  0.01515152,  0.        ],
       [ 0.        ,  0.        ,  0.01515152, -0.03077652,  0.015625  ],
       [ 0.00657895,  0.        ,  0.        ,  0.015625  , -0.00904605]])
>>> det(A)
15409151.999999998
>>> (-1)**(n-1) * 2**(n-2) * np.diff(a).prod() * (a[-1] - a[0])
15409152
*)

val fiedler_companion : [>`Ndarray] Np.Obj.t -> Py.Object.t
(**
Returns a Fiedler companion matrix

Given a polynomial coefficient array ``a``, this function forms a
pentadiagonal matrix with a special structure whose eigenvalues coincides
with the roots of ``a``.

Parameters
----------
a : (N,) array_like
    1-D array of polynomial coefficients in descending order with a nonzero
    leading coefficient. For ``N < 2``, an empty array is returned.

Returns
-------
c : (N-1, N-1) ndarray
    Resulting companion matrix

Notes
-----
Similar to `companion` the leading coefficient should be nonzero. In case
the leading coefficient is not 1., other coefficients are rescaled before
the array generation. To avoid numerical issues, it is best to provide a
monic polynomial.

.. versionadded:: 1.3.0

See Also
--------
companion

References
----------
.. [1] M. Fiedler, ' A note on companion matrices', Linear Algebra and its
    Applications, 2003, :doi:`10.1016/S0024-3795(03)00548-2`

Examples
--------
>>> from scipy.linalg import fiedler_companion, eigvals
>>> p = np.poly(np.arange(1, 9, 2))  # [1., -16., 86., -176., 105.]
>>> fc = fiedler_companion(p)
>>> fc
array([[  16.,  -86.,    1.,    0.],
       [   1.,    0.,    0.,    0.],
       [   0.,  176.,    0., -105.],
       [   0.,    1.,    0.,    0.]])
>>> eigvals(fc)
array([7.+0.j, 5.+0.j, 3.+0.j, 1.+0.j])
*)

val hadamard : ?dtype:Np.Dtype.t -> n:int -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Construct a Hadamard matrix.

Constructs an n-by-n Hadamard matrix, using Sylvester's
construction.  `n` must be a power of 2.

Parameters
----------
n : int
    The order of the matrix.  `n` must be a power of 2.
dtype : dtype, optional
    The data type of the array to be constructed.

Returns
-------
H : (n, n) ndarray
    The Hadamard matrix.

Notes
-----
.. versionadded:: 0.8.0

Examples
--------
>>> from scipy.linalg import hadamard
>>> hadamard(2, dtype=complex)
array([[ 1.+0.j,  1.+0.j],
       [ 1.+0.j, -1.-0.j]])
>>> hadamard(4)
array([[ 1,  1,  1,  1],
       [ 1, -1,  1, -1],
       [ 1,  1, -1, -1],
       [ 1, -1, -1,  1]])
*)

val hankel : ?r:[>`Ndarray] Np.Obj.t -> c:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Construct a Hankel matrix.

The Hankel matrix has constant anti-diagonals, with `c` as its
first column and `r` as its last row.  If `r` is not given, then
`r = zeros_like(c)` is assumed.

Parameters
----------
c : array_like
    First column of the matrix.  Whatever the actual shape of `c`, it
    will be converted to a 1-D array.
r : array_like, optional
    Last row of the matrix. If None, ``r = zeros_like(c)`` is assumed.
    r[0] is ignored; the last row of the returned matrix is
    ``[c[-1], r[1:]]``.  Whatever the actual shape of `r`, it will be
    converted to a 1-D array.

Returns
-------
A : (len(c), len(r)) ndarray
    The Hankel matrix. Dtype is the same as ``(c[0] + r[0]).dtype``.

See Also
--------
toeplitz : Toeplitz matrix
circulant : circulant matrix

Examples
--------
>>> from scipy.linalg import hankel
>>> hankel([1, 17, 99])
array([[ 1, 17, 99],
       [17, 99,  0],
       [99,  0,  0]])
>>> hankel([1,2,3,4], [4,7,7,8,9])
array([[1, 2, 3, 4, 7],
       [2, 3, 4, 7, 7],
       [3, 4, 7, 7, 8],
       [4, 7, 7, 8, 9]])
*)

val helmert : ?full:bool -> n:int -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Create a Helmert matrix of order `n`.

This has applications in statistics, compositional or simplicial analysis,
and in Aitchison geometry.

Parameters
----------
n : int
    The size of the array to create.
full : bool, optional
    If True the (n, n) ndarray will be returned.
    Otherwise the submatrix that does not include the first
    row will be returned.
    Default: False.

Returns
-------
M : ndarray
    The Helmert matrix.
    The shape is (n, n) or (n-1, n) depending on the `full` argument.

Examples
--------
>>> from scipy.linalg import helmert
>>> helmert(5, full=True)
array([[ 0.4472136 ,  0.4472136 ,  0.4472136 ,  0.4472136 ,  0.4472136 ],
       [ 0.70710678, -0.70710678,  0.        ,  0.        ,  0.        ],
       [ 0.40824829,  0.40824829, -0.81649658,  0.        ,  0.        ],
       [ 0.28867513,  0.28867513,  0.28867513, -0.8660254 ,  0.        ],
       [ 0.2236068 ,  0.2236068 ,  0.2236068 ,  0.2236068 , -0.89442719]])
*)

val hilbert : int -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Create a Hilbert matrix of order `n`.

Returns the `n` by `n` array with entries `h[i,j] = 1 / (i + j + 1)`.

Parameters
----------
n : int
    The size of the array to create.

Returns
-------
h : (n, n) ndarray
    The Hilbert matrix.

See Also
--------
invhilbert : Compute the inverse of a Hilbert matrix.

Notes
-----
.. versionadded:: 0.10.0

Examples
--------
>>> from scipy.linalg import hilbert
>>> hilbert(3)
array([[ 1.        ,  0.5       ,  0.33333333],
       [ 0.5       ,  0.33333333,  0.25      ],
       [ 0.33333333,  0.25      ,  0.2       ]])
*)

val invhilbert : ?exact:bool -> n:int -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the inverse of the Hilbert matrix of order `n`.

The entries in the inverse of a Hilbert matrix are integers.  When `n`
is greater than 14, some entries in the inverse exceed the upper limit
of 64 bit integers.  The `exact` argument provides two options for
dealing with these large integers.

Parameters
----------
n : int
    The order of the Hilbert matrix.
exact : bool, optional
    If False, the data type of the array that is returned is np.float64,
    and the array is an approximation of the inverse.
    If True, the array is the exact integer inverse array.  To represent
    the exact inverse when n > 14, the returned array is an object array
    of long integers.  For n <= 14, the exact inverse is returned as an
    array with data type np.int64.

Returns
-------
invh : (n, n) ndarray
    The data type of the array is np.float64 if `exact` is False.
    If `exact` is True, the data type is either np.int64 (for n <= 14)
    or object (for n > 14).  In the latter case, the objects in the
    array will be long integers.

See Also
--------
hilbert : Create a Hilbert matrix.

Notes
-----
.. versionadded:: 0.10.0

Examples
--------
>>> from scipy.linalg import invhilbert
>>> invhilbert(4)
array([[   16.,  -120.,   240.,  -140.],
       [ -120.,  1200., -2700.,  1680.],
       [  240., -2700.,  6480., -4200.],
       [ -140.,  1680., -4200.,  2800.]])
>>> invhilbert(4, exact=True)
array([[   16,  -120,   240,  -140],
       [ -120,  1200, -2700,  1680],
       [  240, -2700,  6480, -4200],
       [ -140,  1680, -4200,  2800]], dtype=int64)
>>> invhilbert(16)[7,7]
4.2475099528537506e+19
>>> invhilbert(16, exact=True)[7,7]
42475099528537378560L
*)

val invpascal : ?kind:string -> ?exact:bool -> n:int -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Returns the inverse of the n x n Pascal matrix.

The Pascal matrix is a matrix containing the binomial coefficients as
its elements.

Parameters
----------
n : int
    The size of the matrix to create; that is, the result is an n x n
    matrix.
kind : str, optional
    Must be one of 'symmetric', 'lower', or 'upper'.
    Default is 'symmetric'.
exact : bool, optional
    If `exact` is True, the result is either an array of type
    ``numpy.int64`` (if `n` <= 35) or an object array of Python integers.
    If `exact` is False, the coefficients in the matrix are computed using
    `scipy.special.comb` with `exact=False`.  The result will be a floating
    point array, and for large `n`, the values in the array will not be the
    exact coefficients.

Returns
-------
invp : (n, n) ndarray
    The inverse of the Pascal matrix.

See Also
--------
pascal

Notes
-----

.. versionadded:: 0.16.0

References
----------
.. [1] 'Pascal matrix', https://en.wikipedia.org/wiki/Pascal_matrix
.. [2] Cohen, A. M., 'The inverse of a Pascal matrix', Mathematical
       Gazette, 59(408), pp. 111-112, 1975.

Examples
--------
>>> from scipy.linalg import invpascal, pascal
>>> invp = invpascal(5)
>>> invp
array([[  5, -10,  10,  -5,   1],
       [-10,  30, -35,  19,  -4],
       [ 10, -35,  46, -27,   6],
       [ -5,  19, -27,  17,  -4],
       [  1,  -4,   6,  -4,   1]])

>>> p = pascal(5)
>>> p.dot(invp)
array([[ 1.,  0.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  0.,  1.]])

An example of the use of `kind` and `exact`:

>>> invpascal(5, kind='lower', exact=False)
array([[ 1., -0.,  0., -0.,  0.],
       [-1.,  1., -0.,  0., -0.],
       [ 1., -2.,  1., -0.,  0.],
       [-1.,  3., -3.,  1., -0.],
       [ 1., -4.,  6., -4.,  1.]])
*)

val kron : a:[>`Ndarray] Np.Obj.t -> b:Py.Object.t -> unit -> Py.Object.t
(**
Kronecker product.

The result is the block matrix::

    a[0,0]*b    a[0,1]*b  ... a[0,-1]*b
    a[1,0]*b    a[1,1]*b  ... a[1,-1]*b
    ...
    a[-1,0]*b   a[-1,1]*b ... a[-1,-1]*b

Parameters
----------
a : (M, N) ndarray
    Input array
b : (P, Q) ndarray
    Input array

Returns
-------
A : (M*P, N*Q) ndarray
    Kronecker product of `a` and `b`.

Examples
--------
>>> from numpy import array
>>> from scipy.linalg import kron
>>> kron(array([[1,2],[3,4]]), array([[1,1,1]]))
array([[1, 1, 1, 2, 2, 2],
       [3, 3, 3, 4, 4, 4]])
*)

val leslie : f:[>`Ndarray] Np.Obj.t -> s:Py.Object.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Create a Leslie matrix.

Given the length n array of fecundity coefficients `f` and the length
n-1 array of survival coefficients `s`, return the associated Leslie
matrix.

Parameters
----------
f : (N,) array_like
    The 'fecundity' coefficients.
s : (N-1,) array_like
    The 'survival' coefficients, has to be 1-D.  The length of `s`
    must be one less than the length of `f`, and it must be at least 1.

Returns
-------
L : (N, N) ndarray
    The array is zero except for the first row,
    which is `f`, and the first sub-diagonal, which is `s`.
    The data-type of the array will be the data-type of ``f[0]+s[0]``.

Notes
-----
.. versionadded:: 0.8.0

The Leslie matrix is used to model discrete-time, age-structured
population growth [1]_ [2]_. In a population with `n` age classes, two sets
of parameters define a Leslie matrix: the `n` 'fecundity coefficients',
which give the number of offspring per-capita produced by each age
class, and the `n` - 1 'survival coefficients', which give the
per-capita survival rate of each age class.

References
----------
.. [1] P. H. Leslie, On the use of matrices in certain population
       mathematics, Biometrika, Vol. 33, No. 3, 183--212 (Nov. 1945)
.. [2] P. H. Leslie, Some further notes on the use of matrices in
       population mathematics, Biometrika, Vol. 35, No. 3/4, 213--245
       (Dec. 1948)

Examples
--------
>>> from scipy.linalg import leslie
>>> leslie([0.1, 2.0, 1.0, 0.1], [0.2, 0.8, 0.7])
array([[ 0.1,  2. ,  1. ,  0.1],
       [ 0.2,  0. ,  0. ,  0. ],
       [ 0. ,  0.8,  0. ,  0. ],
       [ 0. ,  0. ,  0.7,  0. ]])
*)

val pascal : ?kind:string -> ?exact:bool -> n:int -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Returns the n x n Pascal matrix.

The Pascal matrix is a matrix containing the binomial coefficients as
its elements.

Parameters
----------
n : int
    The size of the matrix to create; that is, the result is an n x n
    matrix.
kind : str, optional
    Must be one of 'symmetric', 'lower', or 'upper'.
    Default is 'symmetric'.
exact : bool, optional
    If `exact` is True, the result is either an array of type
    numpy.uint64 (if n < 35) or an object array of Python long integers.
    If `exact` is False, the coefficients in the matrix are computed using
    `scipy.special.comb` with `exact=False`.  The result will be a floating
    point array, and the values in the array will not be the exact
    coefficients, but this version is much faster than `exact=True`.

Returns
-------
p : (n, n) ndarray
    The Pascal matrix.

See Also
--------
invpascal

Notes
-----
See https://en.wikipedia.org/wiki/Pascal_matrix for more information
about Pascal matrices.

.. versionadded:: 0.11.0

Examples
--------
>>> from scipy.linalg import pascal
>>> pascal(4)
array([[ 1,  1,  1,  1],
       [ 1,  2,  3,  4],
       [ 1,  3,  6, 10],
       [ 1,  4, 10, 20]], dtype=uint64)
>>> pascal(4, kind='lower')
array([[1, 0, 0, 0],
       [1, 1, 0, 0],
       [1, 2, 1, 0],
       [1, 3, 3, 1]], dtype=uint64)
>>> pascal(50)[-1, -1]
25477612258980856902730428600L
>>> from scipy.special import comb
>>> comb(98, 49, exact=True)
25477612258980856902730428600L
*)

val toeplitz : ?r:[>`Ndarray] Np.Obj.t -> c:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Construct a Toeplitz matrix.

The Toeplitz matrix has constant diagonals, with c as its first column
and r as its first row.  If r is not given, ``r == conjugate(c)`` is
assumed.

Parameters
----------
c : array_like
    First column of the matrix.  Whatever the actual shape of `c`, it
    will be converted to a 1-D array.
r : array_like, optional
    First row of the matrix. If None, ``r = conjugate(c)`` is assumed;
    in this case, if c[0] is real, the result is a Hermitian matrix.
    r[0] is ignored; the first row of the returned matrix is
    ``[c[0], r[1:]]``.  Whatever the actual shape of `r`, it will be
    converted to a 1-D array.

Returns
-------
A : (len(c), len(r)) ndarray
    The Toeplitz matrix. Dtype is the same as ``(c[0] + r[0]).dtype``.

See Also
--------
circulant : circulant matrix
hankel : Hankel matrix
solve_toeplitz : Solve a Toeplitz system.

Notes
-----
The behavior when `c` or `r` is a scalar, or when `c` is complex and
`r` is None, was changed in version 0.8.0.  The behavior in previous
versions was undocumented and is no longer supported.

Examples
--------
>>> from scipy.linalg import toeplitz
>>> toeplitz([1,2,3], [1,4,5,6])
array([[1, 4, 5, 6],
       [2, 1, 4, 5],
       [3, 2, 1, 4]])
>>> toeplitz([1.0, 2+3j, 4-1j])
array([[ 1.+0.j,  2.-3.j,  4.+1.j],
       [ 2.+3.j,  1.+0.j,  2.-3.j],
       [ 4.-1.j,  2.+3.j,  1.+0.j]])
*)

val tri : ?m:int -> ?k:int -> ?dtype:Np.Dtype.t -> n:int -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Construct (N, M) matrix filled with ones at and below the k-th diagonal.

The matrix has A[i,j] == 1 for i <= j + k

Parameters
----------
N : int
    The size of the first dimension of the matrix.
M : int or None, optional
    The size of the second dimension of the matrix. If `M` is None,
    `M = N` is assumed.
k : int, optional
    Number of subdiagonal below which matrix is filled with ones.
    `k` = 0 is the main diagonal, `k` < 0 subdiagonal and `k` > 0
    superdiagonal.
dtype : dtype, optional
    Data type of the matrix.

Returns
-------
tri : (N, M) ndarray
    Tri matrix.

Examples
--------
>>> from scipy.linalg import tri
>>> tri(3, 5, 2, dtype=int)
array([[1, 1, 1, 0, 0],
       [1, 1, 1, 1, 0],
       [1, 1, 1, 1, 1]])
>>> tri(3, 5, -1, dtype=int)
array([[0, 0, 0, 0, 0],
       [1, 0, 0, 0, 0],
       [1, 1, 0, 0, 0]])
*)

val tril : ?k:int -> m:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Make a copy of a matrix with elements above the k-th diagonal zeroed.

Parameters
----------
m : array_like
    Matrix whose elements to return
k : int, optional
    Diagonal above which to zero elements.
    `k` == 0 is the main diagonal, `k` < 0 subdiagonal and
    `k` > 0 superdiagonal.

Returns
-------
tril : ndarray
    Return is the same shape and type as `m`.

Examples
--------
>>> from scipy.linalg import tril
>>> tril([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)
array([[ 0,  0,  0],
       [ 4,  0,  0],
       [ 7,  8,  0],
       [10, 11, 12]])
*)

val triu : ?k:int -> m:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Make a copy of a matrix with elements below the k-th diagonal zeroed.

Parameters
----------
m : array_like
    Matrix whose elements to return
k : int, optional
    Diagonal below which to zero elements.
    `k` == 0 is the main diagonal, `k` < 0 subdiagonal and
    `k` > 0 superdiagonal.

Returns
-------
triu : ndarray
    Return matrix with zeroed elements below the k-th diagonal and has
    same shape and type as `m`.

Examples
--------
>>> from scipy.linalg import triu
>>> triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)
array([[ 1,  2,  3],
       [ 4,  5,  6],
       [ 0,  8,  9],
       [ 0,  0, 12]])
*)


end

val block_diag : Py.Object.t list -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Create a block diagonal matrix from provided arrays.

Given the inputs `A`, `B` and `C`, the output will have these
arrays arranged on the diagonal::

    [[A, 0, 0],
     [0, B, 0],
     [0, 0, C]]

Parameters
----------
A, B, C, ... : array_like, up to 2-D
    Input arrays.  A 1-D array or array_like sequence of length `n` is
    treated as a 2-D array with shape ``(1,n)``.

Returns
-------
D : ndarray
    Array with `A`, `B`, `C`, ... on the diagonal.  `D` has the
    same dtype as `A`.

Notes
-----
If all the input arrays are square, the output is known as a
block diagonal matrix.

Empty sequences (i.e., array-likes of zero size) will not be ignored.
Noteworthy, both [] and [[]] are treated as matrices with shape ``(1,0)``.

Examples
--------
>>> from scipy.linalg import block_diag
>>> A = [[1, 0],
...      [0, 1]]
>>> B = [[3, 4, 5],
...      [6, 7, 8]]
>>> C = [[7]]
>>> P = np.zeros((2, 0), dtype='int32')
>>> block_diag(A, B, C)
array([[1, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0],
       [0, 0, 3, 4, 5, 0],
       [0, 0, 6, 7, 8, 0],
       [0, 0, 0, 0, 0, 7]])
>>> block_diag(A, P, B, C)
array([[1, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 3, 4, 5, 0],
       [0, 0, 6, 7, 8, 0],
       [0, 0, 0, 0, 0, 7]])
>>> block_diag(1.0, [2, 3], [[4, 5], [6, 7]])
array([[ 1.,  0.,  0.,  0.,  0.],
       [ 0.,  2.,  3.,  0.,  0.],
       [ 0.,  0.,  0.,  4.,  5.],
       [ 0.,  0.,  0.,  6.,  7.]])
*)

val cdf2rdf : w:Py.Object.t -> v:Py.Object.t -> unit -> (Py.Object.t * Py.Object.t)
(**
Converts complex eigenvalues ``w`` and eigenvectors ``v`` to real
eigenvalues in a block diagonal form ``wr`` and the associated real
eigenvectors ``vr``, such that::

    vr @ wr = X @ vr

continues to hold, where ``X`` is the original array for which ``w`` and
``v`` are the eigenvalues and eigenvectors.

.. versionadded:: 1.1.0

Parameters
----------
w : (..., M) array_like
    Complex or real eigenvalues, an array or stack of arrays

    Conjugate pairs must not be interleaved, else the wrong result
    will be produced. So ``[1+1j, 1, 1-1j]`` will give a correct result, but
    ``[1+1j, 2+1j, 1-1j, 2-1j]`` will not.

v : (..., M, M) array_like
    Complex or real eigenvectors, a square array or stack of square arrays.

Returns
-------
wr : (..., M, M) ndarray
    Real diagonal block form of eigenvalues
vr : (..., M, M) ndarray
    Real eigenvectors associated with ``wr``

See Also
--------
eig : Eigenvalues and right eigenvectors for non-symmetric arrays
rsf2csf : Convert real Schur form to complex Schur form

Notes
-----
``w``, ``v`` must be the eigenstructure for some *real* matrix ``X``.
For example, obtained by ``w, v = scipy.linalg.eig(X)`` or
``w, v = numpy.linalg.eig(X)`` in which case ``X`` can also represent
stacked arrays.

.. versionadded:: 1.1.0

Examples
--------
>>> X = np.array([[1, 2, 3], [0, 4, 5], [0, -5, 4]])
>>> X
array([[ 1,  2,  3],
       [ 0,  4,  5],
       [ 0, -5,  4]])

>>> from scipy import linalg
>>> w, v = linalg.eig(X)
>>> w
array([ 1.+0.j,  4.+5.j,  4.-5.j])
>>> v
array([[ 1.00000+0.j     , -0.01906-0.40016j, -0.01906+0.40016j],
       [ 0.00000+0.j     ,  0.00000-0.64788j,  0.00000+0.64788j],
       [ 0.00000+0.j     ,  0.64788+0.j     ,  0.64788-0.j     ]])

>>> wr, vr = linalg.cdf2rdf(w, v)
>>> wr
array([[ 1.,  0.,  0.],
       [ 0.,  4.,  5.],
       [ 0., -5.,  4.]])
>>> vr
array([[ 1.     ,  0.40016, -0.01906],
       [ 0.     ,  0.64788,  0.     ],
       [ 0.     ,  0.     ,  0.64788]])

>>> vr @ wr
array([[ 1.     ,  1.69593,  1.9246 ],
       [ 0.     ,  2.59153,  3.23942],
       [ 0.     , -3.23942,  2.59153]])
>>> X @ vr
array([[ 1.     ,  1.69593,  1.9246 ],
       [ 0.     ,  2.59153,  3.23942],
       [ 0.     , -3.23942,  2.59153]])
*)

val cho_factor : ?lower:bool -> ?overwrite_a:bool -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * bool)
(**
Compute the Cholesky decomposition of a matrix, to use in cho_solve

Returns a matrix containing the Cholesky decomposition,
``A = L L*`` or ``A = U* U`` of a Hermitian positive-definite matrix `a`.
The return value can be directly used as the first parameter to cho_solve.

.. warning::
    The returned matrix also contains random data in the entries not
    used by the Cholesky decomposition. If you need to zero these
    entries, use the function `cholesky` instead.

Parameters
----------
a : (M, M) array_like
    Matrix to be decomposed
lower : bool, optional
    Whether to compute the upper or lower triangular Cholesky factorization
    (Default: upper-triangular)
overwrite_a : bool, optional
    Whether to overwrite data in a (may improve performance)
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
c : (M, M) ndarray
    Matrix whose upper or lower triangle contains the Cholesky factor
    of `a`. Other parts of the matrix contain random data.
lower : bool
    Flag indicating whether the factor is in the lower or upper triangle

Raises
------
LinAlgError
    Raised if decomposition fails.

See also
--------
cho_solve : Solve a linear set equations using the Cholesky factorization
            of a matrix.

Examples
--------
>>> from scipy.linalg import cho_factor
>>> A = np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]])
>>> c, low = cho_factor(A)
>>> c
array([[3.        , 1.        , 0.33333333, 1.66666667],
       [3.        , 2.44948974, 1.90515869, -0.27216553],
       [1.        , 5.        , 2.29330749, 0.8559528 ],
       [5.        , 1.        , 2.        , 1.55418563]])
>>> np.allclose(np.triu(c).T @ np. triu(c) - A, np.zeros((4, 4)))
True
*)

val cho_solve : ?overwrite_b:bool -> ?check_finite:bool -> c_and_lower:Py.Object.t -> b:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Solve the linear equations A x = b, given the Cholesky factorization of A.

Parameters
----------
(c, lower) : tuple, (array, bool)
    Cholesky factorization of a, as given by cho_factor
b : array
    Right-hand side
overwrite_b : bool, optional
    Whether to overwrite data in b (may improve performance)
check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
x : array
    The solution to the system A x = b

See also
--------
cho_factor : Cholesky factorization of a matrix

Examples
--------
>>> from scipy.linalg import cho_factor, cho_solve
>>> A = np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]])
>>> c, low = cho_factor(A)
>>> x = cho_solve((c, low), [1, 1, 1, 1])
>>> np.allclose(A @ x - [1, 1, 1, 1], np.zeros(4))
True
*)

val cho_solve_banded : ?overwrite_b:bool -> ?check_finite:bool -> cb_and_lower:Py.Object.t -> b:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Solve the linear equations ``A x = b``, given the Cholesky factorization of
the banded hermitian ``A``.

Parameters
----------
(cb, lower) : tuple, (ndarray, bool)
    `cb` is the Cholesky factorization of A, as given by cholesky_banded.
    `lower` must be the same value that was given to cholesky_banded.
b : array_like
    Right-hand side
overwrite_b : bool, optional
    If True, the function will overwrite the values in `b`.
check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
x : array
    The solution to the system A x = b

See also
--------
cholesky_banded : Cholesky factorization of a banded matrix

Notes
-----

.. versionadded:: 0.8.0

Examples
--------
>>> from scipy.linalg import cholesky_banded, cho_solve_banded
>>> Ab = np.array([[0, 0, 1j, 2, 3j], [0, -1, -2, 3, 4], [9, 8, 7, 6, 9]])
>>> A = np.diag(Ab[0,2:], k=2) + np.diag(Ab[1,1:], k=1)
>>> A = A + A.conj().T + np.diag(Ab[2, :])
>>> c = cholesky_banded(Ab)
>>> x = cho_solve_banded((c, False), np.ones(5))
>>> np.allclose(A @ x - np.ones(5), np.zeros(5))
True
*)

val cholesky : ?lower:bool -> ?overwrite_a:bool -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the Cholesky decomposition of a matrix.

Returns the Cholesky decomposition, :math:`A = L L^*` or
:math:`A = U^* U` of a Hermitian positive-definite matrix A.

Parameters
----------
a : (M, M) array_like
    Matrix to be decomposed
lower : bool, optional
    Whether to compute the upper or lower triangular Cholesky
    factorization.  Default is upper-triangular.
overwrite_a : bool, optional
    Whether to overwrite data in `a` (may improve performance).
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
c : (M, M) ndarray
    Upper- or lower-triangular Cholesky factor of `a`.

Raises
------
LinAlgError : if decomposition fails.

Examples
--------
>>> from scipy.linalg import cholesky
>>> a = np.array([[1,-2j],[2j,5]])
>>> L = cholesky(a, lower=True)
>>> L
array([[ 1.+0.j,  0.+0.j],
       [ 0.+2.j,  1.+0.j]])
>>> L @ L.T.conj()
array([[ 1.+0.j,  0.-2.j],
       [ 0.+2.j,  5.+0.j]])
*)

val cholesky_banded : ?overwrite_ab:bool -> ?lower:bool -> ?check_finite:bool -> ab:Py.Object.t -> unit -> Py.Object.t
(**
Cholesky decompose a banded Hermitian positive-definite matrix

The matrix a is stored in ab either in lower diagonal or upper
diagonal ordered form::

    ab[u + i - j, j] == a[i,j]        (if upper form; i <= j)
    ab[    i - j, j] == a[i,j]        (if lower form; i >= j)

Example of ab (shape of a is (6,6), u=2)::

    upper form:
    *   *   a02 a13 a24 a35
    *   a01 a12 a23 a34 a45
    a00 a11 a22 a33 a44 a55

    lower form:
    a00 a11 a22 a33 a44 a55
    a10 a21 a32 a43 a54 *
    a20 a31 a42 a53 *   *

Parameters
----------
ab : (u + 1, M) array_like
    Banded matrix
overwrite_ab : bool, optional
    Discard data in ab (may enhance performance)
lower : bool, optional
    Is the matrix in the lower form. (Default is upper form)
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
c : (u + 1, M) ndarray
    Cholesky factorization of a, in the same banded format as ab
    
See also
--------
cho_solve_banded : Solve a linear set equations, given the Cholesky factorization
            of a banded hermitian.

Examples
--------
>>> from scipy.linalg import cholesky_banded
>>> from numpy import allclose, zeros, diag
>>> Ab = np.array([[0, 0, 1j, 2, 3j], [0, -1, -2, 3, 4], [9, 8, 7, 6, 9]])
>>> A = np.diag(Ab[0,2:], k=2) + np.diag(Ab[1,1:], k=1)
>>> A = A + A.conj().T + np.diag(Ab[2, :])
>>> c = cholesky_banded(Ab)
>>> C = np.diag(c[0, 2:], k=2) + np.diag(c[1, 1:], k=1) + np.diag(c[2, :])
>>> np.allclose(C.conj().T @ C - A, np.zeros((5, 5)))
True
*)

val circulant : [>`Ndarray] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Construct a circulant matrix.

Parameters
----------
c : (N,) array_like
    1-D array, the first column of the matrix.

Returns
-------
A : (N, N) ndarray
    A circulant matrix whose first column is `c`.

See Also
--------
toeplitz : Toeplitz matrix
hankel : Hankel matrix
solve_circulant : Solve a circulant system.

Notes
-----
.. versionadded:: 0.8.0

Examples
--------
>>> from scipy.linalg import circulant
>>> circulant([1, 2, 3])
array([[1, 3, 2],
       [2, 1, 3],
       [3, 2, 1]])
*)

val clarkson_woodruff_transform : ?seed:[`T_numpy_random_mtrand_RandomState_instance of Py.Object.t | `I of int] -> input_matrix:[>`Ndarray] Np.Obj.t -> sketch_size:int -> unit -> Py.Object.t
(**
'
Applies a Clarkson-Woodruff Transform/sketch to the input matrix.

Given an input_matrix ``A`` of size ``(n, d)``, compute a matrix ``A'`` of
size (sketch_size, d) so that

.. math:: \|Ax\| \approx \|A'x\|

with high probability via the Clarkson-Woodruff Transform, otherwise
known as the CountSketch matrix.

Parameters
----------
input_matrix: array_like
    Input matrix, of shape ``(n, d)``.
sketch_size: int
    Number of rows for the sketch.
seed : None or int or `numpy.random.mtrand.RandomState` instance, optional
    This parameter defines the ``RandomState`` object to use for drawing
    random variates.
    If None (or ``np.random``), the global ``np.random`` state is used.
    If integer, it is used to seed the local ``RandomState`` instance.
    Default is None.

Returns
-------
A' : array_like
    Sketch of the input matrix ``A``, of size ``(sketch_size, d)``.

Notes
-----
To make the statement

.. math:: \|Ax\| \approx \|A'x\|

precise, observe the following result which is adapted from the
proof of Theorem 14 of [2]_ via Markov's Inequality. If we have
a sketch size ``sketch_size=k`` which is at least

.. math:: k \geq \frac{2}{\epsilon^2\delta}

Then for any fixed vector ``x``,

.. math:: \|Ax\| = (1\pm\epsilon)\|A'x\|

with probability at least one minus delta.

This implementation takes advantage of sparsity: computing
a sketch takes time proportional to ``A.nnz``. Data ``A`` which
is in ``scipy.sparse.csc_matrix`` format gives the quickest
computation time for sparse input.

>>> from scipy import linalg
>>> from scipy import sparse
>>> n_rows, n_columns, density, sketch_n_rows = 15000, 100, 0.01, 200
>>> A = sparse.rand(n_rows, n_columns, density=density, format='csc')
>>> B = sparse.rand(n_rows, n_columns, density=density, format='csr')
>>> C = sparse.rand(n_rows, n_columns, density=density, format='coo')
>>> D = np.random.randn(n_rows, n_columns)
>>> SA = linalg.clarkson_woodruff_transform(A, sketch_n_rows) # fastest
>>> SB = linalg.clarkson_woodruff_transform(B, sketch_n_rows) # fast
>>> SC = linalg.clarkson_woodruff_transform(C, sketch_n_rows) # slower
>>> SD = linalg.clarkson_woodruff_transform(D, sketch_n_rows) # slowest

That said, this method does perform well on dense inputs, just slower
on a relative scale.

Examples
--------
Given a big dense matrix ``A``:

>>> from scipy import linalg
>>> n_rows, n_columns, sketch_n_rows = 15000, 100, 200
>>> A = np.random.randn(n_rows, n_columns)
>>> sketch = linalg.clarkson_woodruff_transform(A, sketch_n_rows)
>>> sketch.shape
(200, 100)
>>> norm_A = np.linalg.norm(A)
>>> norm_sketch = np.linalg.norm(sketch)

Now with high probability, the true norm ``norm_A`` is close to
the sketched norm ``norm_sketch`` in absolute value.

Similarly, applying our sketch preserves the solution to a linear
regression of :math:`\min \|Ax - b\|`.

>>> from scipy import linalg
>>> n_rows, n_columns, sketch_n_rows = 15000, 100, 200
>>> A = np.random.randn(n_rows, n_columns)
>>> b = np.random.randn(n_rows)
>>> x = np.linalg.lstsq(A, b, rcond=None)
>>> Ab = np.hstack((A, b.reshape(-1,1)))
>>> SAb = linalg.clarkson_woodruff_transform(Ab, sketch_n_rows)
>>> SA, Sb = SAb[:,:-1], SAb[:,-1]
>>> x_sketched = np.linalg.lstsq(SA, Sb, rcond=None)

As with the matrix norm example, ``np.linalg.norm(A @ x - b)``
is close to ``np.linalg.norm(A @ x_sketched - b)`` with high
probability.

References
----------
.. [1] Kenneth L. Clarkson and David P. Woodruff. Low rank approximation and
       regression in input sparsity time. In STOC, 2013.

.. [2] David P. Woodruff. Sketching as a tool for numerical linear algebra.
       In Foundations and Trends in Theoretical Computer Science, 2014.
*)

val companion : [>`Ndarray] Np.Obj.t -> Py.Object.t
(**
Create a companion matrix.

Create the companion matrix [1]_ associated with the polynomial whose
coefficients are given in `a`.

Parameters
----------
a : (N,) array_like
    1-D array of polynomial coefficients.  The length of `a` must be
    at least two, and ``a[0]`` must not be zero.

Returns
-------
c : (N-1, N-1) ndarray
    The first row of `c` is ``-a[1:]/a[0]``, and the first
    sub-diagonal is all ones.  The data-type of the array is the same
    as the data-type of ``1.0*a[0]``.

Raises
------
ValueError
    If any of the following are true: a) ``a.ndim != 1``;
    b) ``a.size < 2``; c) ``a[0] == 0``.

Notes
-----
.. versionadded:: 0.8.0

References
----------
.. [1] R. A. Horn & C. R. Johnson, *Matrix Analysis*.  Cambridge, UK:
    Cambridge University Press, 1999, pp. 146-7.

Examples
--------
>>> from scipy.linalg import companion
>>> companion([1, -10, 31, -30])
array([[ 10., -31.,  30.],
       [  1.,   0.,   0.],
       [  0.,   1.,   0.]])
*)

val coshm : [>`Ndarray] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the hyperbolic matrix cosine.

This routine uses expm to compute the matrix exponentials.

Parameters
----------
A : (N, N) array_like
    Input array.

Returns
-------
coshm : (N, N) ndarray
    Hyperbolic matrix cosine of `A`

Examples
--------
>>> from scipy.linalg import tanhm, sinhm, coshm
>>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
>>> c = coshm(a)
>>> c
array([[ 11.24592233,  38.76236492],
       [ 12.92078831,  50.00828725]])

Verify tanhm(a) = sinhm(a).dot(inv(coshm(a)))

>>> t = tanhm(a)
>>> s = sinhm(a)
>>> t - s.dot(np.linalg.inv(c))
array([[  2.72004641e-15,   4.55191440e-15],
       [  0.00000000e+00,  -5.55111512e-16]])
*)

val cosm : [>`Ndarray] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the matrix cosine.

This routine uses expm to compute the matrix exponentials.

Parameters
----------
A : (N, N) array_like
    Input array

Returns
-------
cosm : (N, N) ndarray
    Matrix cosine of A

Examples
--------
>>> from scipy.linalg import expm, sinm, cosm

Euler's identity (exp(i*theta) = cos(theta) + i*sin(theta))
applied to a matrix:

>>> a = np.array([[1.0, 2.0], [-1.0, 3.0]])
>>> expm(1j*a)
array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
       [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
>>> cosm(a) + 1j*sinm(a)
array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
       [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
*)

val det : ?overwrite_a:bool -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Compute the determinant of a matrix

The determinant of a square matrix is a value derived arithmetically
from the coefficients of the matrix.

The determinant for a 3x3 matrix, for example, is computed as follows::

    a    b    c
    d    e    f = A
    g    h    i

    det(A) = a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h

Parameters
----------
a : (M, M) array_like
    A square matrix.
overwrite_a : bool, optional
    Allow overwriting data in a (may enhance performance).
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
det : float or complex
    Determinant of `a`.

Notes
-----
The determinant is computed via LU factorization, LAPACK routine z/dgetrf.

Examples
--------
>>> from scipy import linalg
>>> a = np.array([[1,2,3], [4,5,6], [7,8,9]])
>>> linalg.det(a)
0.0
>>> a = np.array([[0,2,3], [4,5,6], [7,8,9]])
>>> linalg.det(a)
3.0
*)

val dft : ?scale:float -> n:int -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Discrete Fourier transform matrix.

Create the matrix that computes the discrete Fourier transform of a
sequence [1]_.  The n-th primitive root of unity used to generate the
matrix is exp(-2*pi*i/n), where i = sqrt(-1).

Parameters
----------
n : int
    Size the matrix to create.
scale : str, optional
    Must be None, 'sqrtn', or 'n'.
    If `scale` is 'sqrtn', the matrix is divided by `sqrt(n)`.
    If `scale` is 'n', the matrix is divided by `n`.
    If `scale` is None (the default), the matrix is not normalized, and the
    return value is simply the Vandermonde matrix of the roots of unity.

Returns
-------
m : (n, n) ndarray
    The DFT matrix.

Notes
-----
When `scale` is None, multiplying a vector by the matrix returned by
`dft` is mathematically equivalent to (but much less efficient than)
the calculation performed by `scipy.fft.fft`.

.. versionadded:: 0.14.0

References
----------
.. [1] 'DFT matrix', https://en.wikipedia.org/wiki/DFT_matrix

Examples
--------
>>> from scipy.linalg import dft
>>> np.set_printoptions(precision=2, suppress=True)  # for compact output
>>> m = dft(5)
>>> m
array([[ 1.  +0.j  ,  1.  +0.j  ,  1.  +0.j  ,  1.  +0.j  ,  1.  +0.j  ],
       [ 1.  +0.j  ,  0.31-0.95j, -0.81-0.59j, -0.81+0.59j,  0.31+0.95j],
       [ 1.  +0.j  , -0.81-0.59j,  0.31+0.95j,  0.31-0.95j, -0.81+0.59j],
       [ 1.  +0.j  , -0.81+0.59j,  0.31-0.95j,  0.31+0.95j, -0.81-0.59j],
       [ 1.  +0.j  ,  0.31+0.95j, -0.81+0.59j, -0.81-0.59j,  0.31-0.95j]])
>>> x = np.array([1, 2, 3, 0, 3])
>>> m @ x  # Compute the DFT of x
array([ 9.  +0.j  ,  0.12-0.81j, -2.12+3.44j, -2.12-3.44j,  0.12+0.81j])

Verify that ``m @ x`` is the same as ``fft(x)``.

>>> from scipy.fft import fft
>>> fft(x)     # Same result as m @ x
array([ 9.  +0.j  ,  0.12-0.81j, -2.12+3.44j, -2.12-3.44j,  0.12+0.81j])
*)

val diagsvd : s:Py.Object.t -> m:int -> n:int -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Construct the sigma matrix in SVD from singular values and size M, N.

Parameters
----------
s : (M,) or (N,) array_like
    Singular values
M : int
    Size of the matrix whose singular values are `s`.
N : int
    Size of the matrix whose singular values are `s`.

Returns
-------
S : (M, N) ndarray
    The S-matrix in the singular value decomposition

See Also
--------
svd : Singular value decomposition of a matrix
svdvals : Compute singular values of a matrix.

Examples
--------
>>> from scipy.linalg import diagsvd
>>> vals = np.array([1, 2, 3])  # The array representing the computed svd
>>> diagsvd(vals, 3, 4)
array([[1, 0, 0, 0],
       [0, 2, 0, 0],
       [0, 0, 3, 0]])
>>> diagsvd(vals, 4, 3)
array([[1, 0, 0],
       [0, 2, 0],
       [0, 0, 3],
       [0, 0, 0]])
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

val eig_banded : ?lower:bool -> ?eigvals_only:bool -> ?overwrite_a_band:bool -> ?select:[`A | `V | `I] -> ?select_range:Py.Object.t -> ?max_ev:int -> ?check_finite:bool -> a_band:Py.Object.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * Py.Object.t)
(**
Solve real symmetric or complex hermitian band matrix eigenvalue problem.

Find eigenvalues w and optionally right eigenvectors v of a::

    a v[:,i] = w[i] v[:,i]
    v.H v    = identity

The matrix a is stored in a_band either in lower diagonal or upper
diagonal ordered form:

    a_band[u + i - j, j] == a[i,j]        (if upper form; i <= j)
    a_band[    i - j, j] == a[i,j]        (if lower form; i >= j)

where u is the number of bands above the diagonal.

Example of a_band (shape of a is (6,6), u=2)::

    upper form:
    *   *   a02 a13 a24 a35
    *   a01 a12 a23 a34 a45
    a00 a11 a22 a33 a44 a55

    lower form:
    a00 a11 a22 a33 a44 a55
    a10 a21 a32 a43 a54 *
    a20 a31 a42 a53 *   *

Cells marked with * are not used.

Parameters
----------
a_band : (u+1, M) array_like
    The bands of the M by M matrix a.
lower : bool, optional
    Is the matrix in the lower form. (Default is upper form)
eigvals_only : bool, optional
    Compute only the eigenvalues and no eigenvectors.
    (Default: calculate also eigenvectors)
overwrite_a_band : bool, optional
    Discard data in a_band (may enhance performance)
select : {'a', 'v', 'i'}, optional
    Which eigenvalues to calculate

    ======  ========================================
    select  calculated
    ======  ========================================
    'a'     All eigenvalues
    'v'     Eigenvalues in the interval (min, max]
    'i'     Eigenvalues with indices min <= i <= max
    ======  ========================================
select_range : (min, max), optional
    Range of selected eigenvalues
max_ev : int, optional
    For select=='v', maximum number of eigenvalues expected.
    For other values of select, has no meaning.

    In doubt, leave this parameter untouched.

check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
w : (M,) ndarray
    The eigenvalues, in ascending order, each repeated according to its
    multiplicity.
v : (M, M) float or complex ndarray
    The normalized eigenvector corresponding to the eigenvalue w[i] is
    the column v[:,i].

Raises
------
LinAlgError
    If eigenvalue computation does not converge.

See Also
--------
eigvals_banded : eigenvalues for symmetric/Hermitian band matrices
eig : eigenvalues and right eigenvectors of general arrays.
eigh : eigenvalues and right eigenvectors for symmetric/Hermitian arrays
eigh_tridiagonal : eigenvalues and right eiegenvectors for
    symmetric/Hermitian tridiagonal matrices

Examples
--------
>>> from scipy.linalg import eig_banded
>>> A = np.array([[1, 5, 2, 0], [5, 2, 5, 2], [2, 5, 3, 5], [0, 2, 5, 4]])
>>> Ab = np.array([[1, 2, 3, 4], [5, 5, 5, 0], [2, 2, 0, 0]])
>>> w, v = eig_banded(Ab, lower=True)
>>> np.allclose(A @ v - v @ np.diag(w), np.zeros((4, 4)))
True
>>> w = eig_banded(Ab, lower=True, eigvals_only=True)
>>> w
array([-4.26200532, -2.22987175,  3.95222349, 12.53965359])

Request only the eigenvalues between ``[-3, 4]``

>>> w, v = eig_banded(Ab, lower=True, select='v', select_range=[-3, 4])
>>> w
array([-2.22987175,  3.95222349])
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

val eigh_tridiagonal : ?eigvals_only:Py.Object.t -> ?select:[`A | `V | `I] -> ?select_range:Py.Object.t -> ?check_finite:bool -> ?tol:float -> ?lapack_driver:string -> d:[>`Ndarray] Np.Obj.t -> e:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Solve eigenvalue problem for a real symmetric tridiagonal matrix.

Find eigenvalues `w` and optionally right eigenvectors `v` of ``a``::

    a v[:,i] = w[i] v[:,i]
    v.H v    = identity

For a real symmetric matrix ``a`` with diagonal elements `d` and
off-diagonal elements `e`.

Parameters
----------
d : ndarray, shape (ndim,)
    The diagonal elements of the array.
e : ndarray, shape (ndim-1,)
    The off-diagonal elements of the array.
select : {'a', 'v', 'i'}, optional
    Which eigenvalues to calculate

    ======  ========================================
    select  calculated
    ======  ========================================
    'a'     All eigenvalues
    'v'     Eigenvalues in the interval (min, max]
    'i'     Eigenvalues with indices min <= i <= max
    ======  ========================================
select_range : (min, max), optional
    Range of selected eigenvalues
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.
tol : float
    The absolute tolerance to which each eigenvalue is required
    (only used when 'stebz' is the `lapack_driver`).
    An eigenvalue (or cluster) is considered to have converged if it
    lies in an interval of this width. If <= 0. (default),
    the value ``eps*|a|`` is used where eps is the machine precision,
    and ``|a|`` is the 1-norm of the matrix ``a``.
lapack_driver : str
    LAPACK function to use, can be 'auto', 'stemr', 'stebz', 'sterf',
    or 'stev'. When 'auto' (default), it will use 'stemr' if ``select='a'``
    and 'stebz' otherwise. When 'stebz' is used to find the eigenvalues and
    ``eigvals_only=False``, then a second LAPACK call (to ``?STEIN``) is
    used to find the corresponding eigenvectors. 'sterf' can only be
    used when ``eigvals_only=True`` and ``select='a'``. 'stev' can only
    be used when ``select='a'``.

Returns
-------
w : (M,) ndarray
    The eigenvalues, in ascending order, each repeated according to its
    multiplicity.
v : (M, M) ndarray
    The normalized eigenvector corresponding to the eigenvalue ``w[i]`` is
    the column ``v[:,i]``.

Raises
------
LinAlgError
    If eigenvalue computation does not converge.

See Also
--------
eigvalsh_tridiagonal : eigenvalues of symmetric/Hermitian tridiagonal
    matrices
eig : eigenvalues and right eigenvectors for non-symmetric arrays
eigh : eigenvalues and right eigenvectors for symmetric/Hermitian arrays
eig_banded : eigenvalues and right eigenvectors for symmetric/Hermitian
    band matrices

Notes
-----
This function makes use of LAPACK ``S/DSTEMR`` routines.

Examples
--------
>>> from scipy.linalg import eigh_tridiagonal
>>> d = 3*np.ones(4)
>>> e = -1*np.ones(3)
>>> w, v = eigh_tridiagonal(d, e)
>>> A = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)
>>> np.allclose(A @ v - v @ np.diag(w), np.zeros((4, 4)))
True
*)

val eigvals : ?b:[>`Ndarray] Np.Obj.t -> ?overwrite_a:bool -> ?check_finite:bool -> ?homogeneous_eigvals:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Compute eigenvalues from an ordinary or generalized eigenvalue problem.

Find eigenvalues of a general matrix::

    a   vr[:,i] = w[i]        b   vr[:,i]

Parameters
----------
a : (M, M) array_like
    A complex or real matrix whose eigenvalues and eigenvectors
    will be computed.
b : (M, M) array_like, optional
    Right-hand side matrix in a generalized eigenvalue problem.
    If omitted, identity matrix is assumed.
overwrite_a : bool, optional
    Whether to overwrite data in a (may improve performance)
check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities
    or NaNs.
homogeneous_eigvals : bool, optional
    If True, return the eigenvalues in homogeneous coordinates.
    In this case ``w`` is a (2, M) array so that::

        w[1,i] a vr[:,i] = w[0,i] b vr[:,i]

    Default is False.

Returns
-------
w : (M,) or (2, M) double or complex ndarray
    The eigenvalues, each repeated according to its multiplicity
    but not in any specific order. The shape is (M,) unless
    ``homogeneous_eigvals=True``.

Raises
------
LinAlgError
    If eigenvalue computation does not converge

See Also
--------
eig : eigenvalues and right eigenvectors of general arrays.
eigvalsh : eigenvalues of symmetric or Hermitian arrays
eigvals_banded : eigenvalues for symmetric/Hermitian band matrices
eigvalsh_tridiagonal : eigenvalues of symmetric/Hermitian tridiagonal
    matrices

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
*)

val eigvals_banded : ?lower:bool -> ?overwrite_a_band:bool -> ?select:[`A | `V | `I] -> ?select_range:Py.Object.t -> ?check_finite:bool -> a_band:Py.Object.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Solve real symmetric or complex hermitian band matrix eigenvalue problem.

Find eigenvalues w of a::

    a v[:,i] = w[i] v[:,i]
    v.H v    = identity

The matrix a is stored in a_band either in lower diagonal or upper
diagonal ordered form:

    a_band[u + i - j, j] == a[i,j]        (if upper form; i <= j)
    a_band[    i - j, j] == a[i,j]        (if lower form; i >= j)

where u is the number of bands above the diagonal.

Example of a_band (shape of a is (6,6), u=2)::

    upper form:
    *   *   a02 a13 a24 a35
    *   a01 a12 a23 a34 a45
    a00 a11 a22 a33 a44 a55

    lower form:
    a00 a11 a22 a33 a44 a55
    a10 a21 a32 a43 a54 *
    a20 a31 a42 a53 *   *

Cells marked with * are not used.

Parameters
----------
a_band : (u+1, M) array_like
    The bands of the M by M matrix a.
lower : bool, optional
    Is the matrix in the lower form. (Default is upper form)
overwrite_a_band : bool, optional
    Discard data in a_band (may enhance performance)
select : {'a', 'v', 'i'}, optional
    Which eigenvalues to calculate

    ======  ========================================
    select  calculated
    ======  ========================================
    'a'     All eigenvalues
    'v'     Eigenvalues in the interval (min, max]
    'i'     Eigenvalues with indices min <= i <= max
    ======  ========================================
select_range : (min, max), optional
    Range of selected eigenvalues
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
w : (M,) ndarray
    The eigenvalues, in ascending order, each repeated according to its
    multiplicity.

Raises
------
LinAlgError
    If eigenvalue computation does not converge.

See Also
--------
eig_banded : eigenvalues and right eigenvectors for symmetric/Hermitian
    band matrices
eigvalsh_tridiagonal : eigenvalues of symmetric/Hermitian tridiagonal
    matrices
eigvals : eigenvalues of general arrays
eigh : eigenvalues and right eigenvectors for symmetric/Hermitian arrays
eig : eigenvalues and right eigenvectors for non-symmetric arrays

Examples
--------
>>> from scipy.linalg import eigvals_banded
>>> A = np.array([[1, 5, 2, 0], [5, 2, 5, 2], [2, 5, 3, 5], [0, 2, 5, 4]])
>>> Ab = np.array([[1, 2, 3, 4], [5, 5, 5, 0], [2, 2, 0, 0]])
>>> w = eigvals_banded(Ab, lower=True)
>>> w
array([-4.26200532, -2.22987175,  3.95222349, 12.53965359])
*)

val eigvalsh : ?b:[>`Ndarray] Np.Obj.t -> ?lower:bool -> ?overwrite_a:bool -> ?overwrite_b:bool -> ?turbo:bool -> ?eigvals:Py.Object.t -> ?type_:int -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Solve an ordinary or generalized eigenvalue problem for a complex
Hermitian or real symmetric matrix.

Find eigenvalues w of matrix a, where b is positive definite::

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
turbo : bool, optional
    Use divide and conquer algorithm (faster but expensive in memory,
    only for generalized eigenvalue problem and if eigvals=None)
eigvals : tuple (lo, hi), optional
    Indexes of the smallest and largest (in ascending order) eigenvalues
    and corresponding eigenvectors to be returned: 0 <= lo < hi <= M-1.
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

Raises
------
LinAlgError
    If eigenvalue computation does not converge,
    an error occurred, or b matrix is not definite positive. Note that
    if input matrices are not symmetric or hermitian, no error is reported
    but results will be wrong.

See Also
--------
eigh : eigenvalues and right eigenvectors for symmetric/Hermitian arrays
eigvals : eigenvalues of general arrays
eigvals_banded : eigenvalues for symmetric/Hermitian band matrices
eigvalsh_tridiagonal : eigenvalues of symmetric/Hermitian tridiagonal
    matrices

Notes
-----
This function does not check the input array for being hermitian/symmetric
in order to allow for representing arrays with only their upper/lower
triangular parts.

Examples
--------
>>> from scipy.linalg import eigvalsh
>>> A = np.array([[6, 3, 1, 5], [3, 0, 5, 1], [1, 5, 6, 2], [5, 1, 2, 2]])
>>> w = eigvalsh(A)
>>> w
array([-3.74637491, -0.76263923,  6.08502336, 12.42399079])
*)

val eigvalsh_tridiagonal : ?select:[`A | `V | `I] -> ?select_range:Py.Object.t -> ?check_finite:bool -> ?tol:float -> ?lapack_driver:string -> d:[>`Ndarray] Np.Obj.t -> e:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Solve eigenvalue problem for a real symmetric tridiagonal matrix.

Find eigenvalues `w` of ``a``::

    a v[:,i] = w[i] v[:,i]
    v.H v    = identity

For a real symmetric matrix ``a`` with diagonal elements `d` and
off-diagonal elements `e`.

Parameters
----------
d : ndarray, shape (ndim,)
    The diagonal elements of the array.
e : ndarray, shape (ndim-1,)
    The off-diagonal elements of the array.
select : {'a', 'v', 'i'}, optional
    Which eigenvalues to calculate

    ======  ========================================
    select  calculated
    ======  ========================================
    'a'     All eigenvalues
    'v'     Eigenvalues in the interval (min, max]
    'i'     Eigenvalues with indices min <= i <= max
    ======  ========================================
select_range : (min, max), optional
    Range of selected eigenvalues
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.
tol : float
    The absolute tolerance to which each eigenvalue is required
    (only used when ``lapack_driver='stebz'``).
    An eigenvalue (or cluster) is considered to have converged if it
    lies in an interval of this width. If <= 0. (default),
    the value ``eps*|a|`` is used where eps is the machine precision,
    and ``|a|`` is the 1-norm of the matrix ``a``.
lapack_driver : str
    LAPACK function to use, can be 'auto', 'stemr', 'stebz',  'sterf',
    or 'stev'. When 'auto' (default), it will use 'stemr' if ``select='a'``
    and 'stebz' otherwise. 'sterf' and 'stev' can only be used when
    ``select='a'``.

Returns
-------
w : (M,) ndarray
    The eigenvalues, in ascending order, each repeated according to its
    multiplicity.

Raises
------
LinAlgError
    If eigenvalue computation does not converge.

See Also
--------
eigh_tridiagonal : eigenvalues and right eiegenvectors for
    symmetric/Hermitian tridiagonal matrices

Examples
--------
>>> from scipy.linalg import eigvalsh_tridiagonal, eigvalsh
>>> d = 3*np.ones(4)
>>> e = -1*np.ones(3)
>>> w = eigvalsh_tridiagonal(d, e)
>>> A = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)
>>> w2 = eigvalsh(A)  # Verify with other eigenvalue routines
>>> np.allclose(w - w2, np.zeros(4))
True
*)

val expm : [>`ArrayLike] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the matrix exponential using Pade approximation.

Parameters
----------
A : (N, N) array_like or sparse matrix
    Matrix to be exponentiated.

Returns
-------
expm : (N, N) ndarray
    Matrix exponential of `A`.

References
----------
.. [1] Awad H. Al-Mohy and Nicholas J. Higham (2009)
       'A New Scaling and Squaring Algorithm for the Matrix Exponential.'
       SIAM Journal on Matrix Analysis and Applications.
       31 (3). pp. 970-989. ISSN 1095-7162

Examples
--------
>>> from scipy.linalg import expm, sinm, cosm

Matrix version of the formula exp(0) = 1:

>>> expm(np.zeros((2,2)))
array([[ 1.,  0.],
       [ 0.,  1.]])

Euler's identity (exp(i*theta) = cos(theta) + i*sin(theta))
applied to a matrix:

>>> a = np.array([[1.0, 2.0], [-1.0, 3.0]])
>>> expm(1j*a)
array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
       [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
>>> cosm(a) + 1j*sinm(a)
array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
       [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
*)

val expm_cond : ?check_finite:bool -> a:Py.Object.t -> unit -> float
(**
Relative condition number of the matrix exponential in the Frobenius norm.

Parameters
----------
A : 2d array_like
    Square input matrix with shape (N, N).
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
kappa : float
    The relative condition number of the matrix exponential
    in the Frobenius norm

Notes
-----
A faster estimate for the condition number in the 1-norm
has been published but is not yet implemented in scipy.

.. versionadded:: 0.14.0

See also
--------
expm : Compute the exponential of a matrix.
expm_frechet : Compute the Frechet derivative of the matrix exponential.

Examples
--------
>>> from scipy.linalg import expm_cond
>>> A = np.array([[-0.3, 0.2, 0.6], [0.6, 0.3, -0.1], [-0.7, 1.2, 0.9]])
>>> k = expm_cond(A)
>>> k
1.7787805864469866
*)

val expm_frechet : ?method_:string -> ?compute_expm:bool -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> e:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Frechet derivative of the matrix exponential of A in the direction E.

Parameters
----------
A : (N, N) array_like
    Matrix of which to take the matrix exponential.
E : (N, N) array_like
    Matrix direction in which to take the Frechet derivative.
method : str, optional
    Choice of algorithm.  Should be one of

    - `SPS` (default)
    - `blockEnlarge`

compute_expm : bool, optional
    Whether to compute also `expm_A` in addition to `expm_frechet_AE`.
    Default is True.
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
expm_A : ndarray
    Matrix exponential of A.
expm_frechet_AE : ndarray
    Frechet derivative of the matrix exponential of A in the direction E.

For ``compute_expm = False``, only `expm_frechet_AE` is returned.

See also
--------
expm : Compute the exponential of a matrix.

Notes
-----
This section describes the available implementations that can be selected
by the `method` parameter. The default method is *SPS*.

Method *blockEnlarge* is a naive algorithm.

Method *SPS* is Scaling-Pade-Squaring [1]_.
It is a sophisticated implementation which should take
only about 3/8 as much time as the naive implementation.
The asymptotics are the same.

.. versionadded:: 0.13.0

References
----------
.. [1] Awad H. Al-Mohy and Nicholas J. Higham (2009)
       Computing the Frechet Derivative of the Matrix Exponential,
       with an application to Condition Number Estimation.
       SIAM Journal On Matrix Analysis and Applications.,
       30 (4). pp. 1639-1657. ISSN 1095-7162

Examples
--------
>>> import scipy.linalg
>>> A = np.random.randn(3, 3)
>>> E = np.random.randn(3, 3)
>>> expm_A, expm_frechet_AE = scipy.linalg.expm_frechet(A, E)
>>> expm_A.shape, expm_frechet_AE.shape
((3, 3), (3, 3))

>>> import scipy.linalg
>>> A = np.random.randn(3, 3)
>>> E = np.random.randn(3, 3)
>>> expm_A, expm_frechet_AE = scipy.linalg.expm_frechet(A, E)
>>> M = np.zeros((6, 6))
>>> M[:3, :3] = A; M[:3, 3:] = E; M[3:, 3:] = A
>>> expm_M = scipy.linalg.expm(M)
>>> np.allclose(expm_A, expm_M[:3, :3])
True
>>> np.allclose(expm_frechet_AE, expm_M[:3, 3:])
True
*)

val fiedler : [>`Ndarray] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Returns a symmetric Fiedler matrix

Given an sequence of numbers `a`, Fiedler matrices have the structure
``F[i, j] = np.abs(a[i] - a[j])``, and hence zero diagonals and nonnegative
entries. A Fiedler matrix has a dominant positive eigenvalue and other
eigenvalues are negative. Although not valid generally, for certain inputs,
the inverse and the determinant can be derived explicitly as given in [1]_.

Parameters
----------
a : (n,) array_like
    coefficient array

Returns
-------
F : (n, n) ndarray

See Also
--------
circulant, toeplitz

Notes
-----

.. versionadded:: 1.3.0

References
----------
.. [1] J. Todd, 'Basic Numerical Mathematics: Vol.2 : Numerical Algebra',
    1977, Birkhauser, :doi:`10.1007/978-3-0348-7286-7`

Examples
--------
>>> from scipy.linalg import det, inv, fiedler
>>> a = [1, 4, 12, 45, 77]
>>> n = len(a)
>>> A = fiedler(a)
>>> A
array([[ 0,  3, 11, 44, 76],
       [ 3,  0,  8, 41, 73],
       [11,  8,  0, 33, 65],
       [44, 41, 33,  0, 32],
       [76, 73, 65, 32,  0]])

The explicit formulas for determinant and inverse seem to hold only for
monotonically increasing/decreasing arrays. Note the tridiagonal structure
and the corners.

>>> Ai = inv(A)
>>> Ai[np.abs(Ai) < 1e-12] = 0.  # cleanup the numerical noise for display
>>> Ai
array([[-0.16008772,  0.16666667,  0.        ,  0.        ,  0.00657895],
       [ 0.16666667, -0.22916667,  0.0625    ,  0.        ,  0.        ],
       [ 0.        ,  0.0625    , -0.07765152,  0.01515152,  0.        ],
       [ 0.        ,  0.        ,  0.01515152, -0.03077652,  0.015625  ],
       [ 0.00657895,  0.        ,  0.        ,  0.015625  , -0.00904605]])
>>> det(A)
15409151.999999998
>>> (-1)**(n-1) * 2**(n-2) * np.diff(a).prod() * (a[-1] - a[0])
15409152
*)

val fiedler_companion : [>`Ndarray] Np.Obj.t -> Py.Object.t
(**
Returns a Fiedler companion matrix

Given a polynomial coefficient array ``a``, this function forms a
pentadiagonal matrix with a special structure whose eigenvalues coincides
with the roots of ``a``.

Parameters
----------
a : (N,) array_like
    1-D array of polynomial coefficients in descending order with a nonzero
    leading coefficient. For ``N < 2``, an empty array is returned.

Returns
-------
c : (N-1, N-1) ndarray
    Resulting companion matrix

Notes
-----
Similar to `companion` the leading coefficient should be nonzero. In case
the leading coefficient is not 1., other coefficients are rescaled before
the array generation. To avoid numerical issues, it is best to provide a
monic polynomial.

.. versionadded:: 1.3.0

See Also
--------
companion

References
----------
.. [1] M. Fiedler, ' A note on companion matrices', Linear Algebra and its
    Applications, 2003, :doi:`10.1016/S0024-3795(03)00548-2`

Examples
--------
>>> from scipy.linalg import fiedler_companion, eigvals
>>> p = np.poly(np.arange(1, 9, 2))  # [1., -16., 86., -176., 105.]
>>> fc = fiedler_companion(p)
>>> fc
array([[  16.,  -86.,    1.,    0.],
       [   1.,    0.,    0.,    0.],
       [   0.,  176.,    0., -105.],
       [   0.,    1.,    0.,    0.]])
>>> eigvals(fc)
array([7.+0.j, 5.+0.j, 3.+0.j, 1.+0.j])
*)

val find_best_blas_type : ?arrays:[>`Ndarray] Np.Obj.t list -> ?dtype:[`S of string | `Dtype of Np.Dtype.t] -> unit -> (string * Np.Dtype.t * bool)
(**
Find best-matching BLAS/LAPACK type.

Arrays are used to determine the optimal prefix of BLAS routines.

Parameters
----------
arrays : sequence of ndarrays, optional
    Arrays can be given to determine optimal prefix of BLAS
    routines. If not given, double-precision routines will be
    used, otherwise the most generic type in arrays will be used.
dtype : str or dtype, optional
    Data-type specifier. Not used if `arrays` is non-empty.

Returns
-------
prefix : str
    BLAS/LAPACK prefix character.
dtype : dtype
    Inferred Numpy data type.
prefer_fortran : bool
    Whether to prefer Fortran order routines over C order.

Examples
--------
>>> import scipy.linalg.blas as bla
>>> a = np.random.rand(10,15)
>>> b = np.asfortranarray(a)  # Change the memory layout order
>>> bla.find_best_blas_type((a,))
('d', dtype('float64'), False)
>>> bla.find_best_blas_type((a*1j,))
('z', dtype('complex128'), False)
>>> bla.find_best_blas_type((b,))
('d', dtype('float64'), True)
*)

val fractional_matrix_power : a:[>`Ndarray] Np.Obj.t -> t:float -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the fractional power of a matrix.

Proceeds according to the discussion in section (6) of [1]_.

Parameters
----------
A : (N, N) array_like
    Matrix whose fractional power to evaluate.
t : float
    Fractional power.

Returns
-------
X : (N, N) array_like
    The fractional power of the matrix.

References
----------
.. [1] Nicholas J. Higham and Lijing lin (2011)
       'A Schur-Pade Algorithm for Fractional Powers of a Matrix.'
       SIAM Journal on Matrix Analysis and Applications,
       32 (3). pp. 1056-1078. ISSN 0895-4798

Examples
--------
>>> from scipy.linalg import fractional_matrix_power
>>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
>>> b = fractional_matrix_power(a, 0.5)
>>> b
array([[ 0.75592895,  1.13389342],
       [ 0.37796447,  1.88982237]])
>>> np.dot(b, b)      # Verify square root
array([[ 1.,  3.],
       [ 1.,  4.]])
*)

val funm : ?disp:bool -> a:[>`Ndarray] Np.Obj.t -> func:Py.Object.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * float)
(**
Evaluate a matrix function specified by a callable.

Returns the value of matrix-valued function ``f`` at `A`. The
function ``f`` is an extension of the scalar-valued function `func`
to matrices.

Parameters
----------
A : (N, N) array_like
    Matrix at which to evaluate the function
func : callable
    Callable object that evaluates a scalar function f.
    Must be vectorized (eg. using vectorize).
disp : bool, optional
    Print warning if error in the result is estimated large
    instead of returning estimated error. (Default: True)

Returns
-------
funm : (N, N) ndarray
    Value of the matrix function specified by func evaluated at `A`
errest : float
    (if disp == False)

    1-norm of the estimated error, ||err||_1 / ||A||_1

Examples
--------
>>> from scipy.linalg import funm
>>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
>>> funm(a, lambda x: x*x)
array([[  4.,  15.],
       [  5.,  19.]])
>>> a.dot(a)
array([[  4.,  15.],
       [  5.,  19.]])

Notes
-----
This function implements the general algorithm based on Schur decomposition
(Algorithm 9.1.1. in [1]_).

If the input matrix is known to be diagonalizable, then relying on the
eigendecomposition is likely to be faster. For example, if your matrix is
Hermitian, you can do

>>> from scipy.linalg import eigh
>>> def funm_herm(a, func, check_finite=False):
...     w, v = eigh(a, check_finite=check_finite)
...     ## if you further know that your matrix is positive semidefinite,
...     ## you can optionally guard against precision errors by doing
...     # w = np.maximum(w, 0)
...     w = func(w)
...     return (v * w).dot(v.conj().T)

References
----------
.. [1] Gene H. Golub, Charles F. van Loan, Matrix Computations 4th ed.
*)

val get_blas_funcs : ?arrays:[>`Ndarray] Np.Obj.t list -> ?dtype:[`S of string | `Dtype of Np.Dtype.t] -> names:[`Sequence_of_str of Py.Object.t | `S of string] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return available BLAS function objects from names.

Arrays are used to determine the optimal prefix of BLAS routines.

Parameters
----------
names : str or sequence of str
    Name(s) of BLAS functions without type prefix.

arrays : sequence of ndarrays, optional
    Arrays can be given to determine optimal prefix of BLAS
    routines. If not given, double-precision routines will be
    used, otherwise the most generic type in arrays will be used.

dtype : str or dtype, optional
    Data-type specifier. Not used if `arrays` is non-empty.


Returns
-------
funcs : list
    List containing the found function(s).


Notes
-----
This routine automatically chooses between Fortran/C
interfaces. Fortran code is used whenever possible for arrays with
column major order. In all other cases, C code is preferred.

In BLAS, the naming convention is that all functions start with a
type prefix, which depends on the type of the principal
matrix. These can be one of {'s', 'd', 'c', 'z'} for the numpy
types {float32, float64, complex64, complex128} respectively.
The code and the dtype are stored in attributes `typecode` and `dtype`
of the returned functions.

Examples
--------
>>> import scipy.linalg as LA
>>> a = np.random.rand(3,2)
>>> x_gemv = LA.get_blas_funcs('gemv', (a,))
>>> x_gemv.typecode
'd'
>>> x_gemv = LA.get_blas_funcs('gemv',(a*1j,))
>>> x_gemv.typecode
'z'
*)

val get_lapack_funcs : ?arrays:[>`Ndarray] Np.Obj.t list -> ?dtype:[`S of string | `Dtype of Np.Dtype.t] -> names:[`Sequence_of_str of Py.Object.t | `S of string] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return available LAPACK function objects from names.

Arrays are used to determine the optimal prefix of LAPACK routines.

Parameters
----------
names : str or sequence of str
    Name(s) of LAPACK functions without type prefix.

arrays : sequence of ndarrays, optional
    Arrays can be given to determine optimal prefix of LAPACK
    routines. If not given, double-precision routines will be
    used, otherwise the most generic type in arrays will be used.

dtype : str or dtype, optional
    Data-type specifier. Not used if `arrays` is non-empty.

Returns
-------
funcs : list
    List containing the found function(s).

Notes
-----
This routine automatically chooses between Fortran/C
interfaces. Fortran code is used whenever possible for arrays with
column major order. In all other cases, C code is preferred.

In LAPACK, the naming convention is that all functions start with a
type prefix, which depends on the type of the principal
matrix. These can be one of {'s', 'd', 'c', 'z'} for the numpy
types {float32, float64, complex64, complex128} respectively, and
are stored in attribute ``typecode`` of the returned functions.

Examples
--------
Suppose we would like to use '?lange' routine which computes the selected
norm of an array. We pass our array in order to get the correct 'lange'
flavor.

>>> import scipy.linalg as LA
>>> a = np.random.rand(3,2)
>>> x_lange = LA.get_lapack_funcs('lange', (a,))
>>> x_lange.typecode
'd'
>>> x_lange = LA.get_lapack_funcs('lange',(a*1j,))
>>> x_lange.typecode
'z'

Several LAPACK routines work best when its internal WORK array has
the optimal size (big enough for fast computation and small enough to
avoid waste of memory). This size is determined also by a dedicated query
to the function which is often wrapped as a standalone function and
commonly denoted as ``###_lwork``. Below is an example for ``?sysv``

>>> import scipy.linalg as LA
>>> a = np.random.rand(1000,1000)
>>> b = np.random.rand(1000,1)*1j
>>> # We pick up zsysv and zsysv_lwork due to b array
... xsysv, xlwork = LA.get_lapack_funcs(('sysv', 'sysv_lwork'), (a, b))
>>> opt_lwork, _ = xlwork(a.shape[0])  # returns a complex for 'z' prefix
>>> udut, ipiv, x, info = xsysv(a, b, lwork=int(opt_lwork.real))
*)

val hadamard : ?dtype:Np.Dtype.t -> n:int -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Construct a Hadamard matrix.

Constructs an n-by-n Hadamard matrix, using Sylvester's
construction.  `n` must be a power of 2.

Parameters
----------
n : int
    The order of the matrix.  `n` must be a power of 2.
dtype : dtype, optional
    The data type of the array to be constructed.

Returns
-------
H : (n, n) ndarray
    The Hadamard matrix.

Notes
-----
.. versionadded:: 0.8.0

Examples
--------
>>> from scipy.linalg import hadamard
>>> hadamard(2, dtype=complex)
array([[ 1.+0.j,  1.+0.j],
       [ 1.+0.j, -1.-0.j]])
>>> hadamard(4)
array([[ 1,  1,  1,  1],
       [ 1, -1,  1, -1],
       [ 1,  1, -1, -1],
       [ 1, -1, -1,  1]])
*)

val hankel : ?r:[>`Ndarray] Np.Obj.t -> c:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Construct a Hankel matrix.

The Hankel matrix has constant anti-diagonals, with `c` as its
first column and `r` as its last row.  If `r` is not given, then
`r = zeros_like(c)` is assumed.

Parameters
----------
c : array_like
    First column of the matrix.  Whatever the actual shape of `c`, it
    will be converted to a 1-D array.
r : array_like, optional
    Last row of the matrix. If None, ``r = zeros_like(c)`` is assumed.
    r[0] is ignored; the last row of the returned matrix is
    ``[c[-1], r[1:]]``.  Whatever the actual shape of `r`, it will be
    converted to a 1-D array.

Returns
-------
A : (len(c), len(r)) ndarray
    The Hankel matrix. Dtype is the same as ``(c[0] + r[0]).dtype``.

See Also
--------
toeplitz : Toeplitz matrix
circulant : circulant matrix

Examples
--------
>>> from scipy.linalg import hankel
>>> hankel([1, 17, 99])
array([[ 1, 17, 99],
       [17, 99,  0],
       [99,  0,  0]])
>>> hankel([1,2,3,4], [4,7,7,8,9])
array([[1, 2, 3, 4, 7],
       [2, 3, 4, 7, 7],
       [3, 4, 7, 7, 8],
       [4, 7, 7, 8, 9]])
*)

val helmert : ?full:bool -> n:int -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Create a Helmert matrix of order `n`.

This has applications in statistics, compositional or simplicial analysis,
and in Aitchison geometry.

Parameters
----------
n : int
    The size of the array to create.
full : bool, optional
    If True the (n, n) ndarray will be returned.
    Otherwise the submatrix that does not include the first
    row will be returned.
    Default: False.

Returns
-------
M : ndarray
    The Helmert matrix.
    The shape is (n, n) or (n-1, n) depending on the `full` argument.

Examples
--------
>>> from scipy.linalg import helmert
>>> helmert(5, full=True)
array([[ 0.4472136 ,  0.4472136 ,  0.4472136 ,  0.4472136 ,  0.4472136 ],
       [ 0.70710678, -0.70710678,  0.        ,  0.        ,  0.        ],
       [ 0.40824829,  0.40824829, -0.81649658,  0.        ,  0.        ],
       [ 0.28867513,  0.28867513,  0.28867513, -0.8660254 ,  0.        ],
       [ 0.2236068 ,  0.2236068 ,  0.2236068 ,  0.2236068 , -0.89442719]])
*)

val hessenberg : ?calc_q:bool -> ?overwrite_a:bool -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Compute Hessenberg form of a matrix.

The Hessenberg decomposition is::

    A = Q H Q^H

where `Q` is unitary/orthogonal and `H` has only zero elements below
the first sub-diagonal.

Parameters
----------
a : (M, M) array_like
    Matrix to bring into Hessenberg form.
calc_q : bool, optional
    Whether to compute the transformation matrix.  Default is False.
overwrite_a : bool, optional
    Whether to overwrite `a`; may improve performance.
    Default is False.
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
H : (M, M) ndarray
    Hessenberg form of `a`.
Q : (M, M) ndarray
    Unitary/orthogonal similarity transformation matrix ``A = Q H Q^H``.
    Only returned if ``calc_q=True``.

Examples
--------
>>> from scipy.linalg import hessenberg
>>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
>>> H, Q = hessenberg(A, calc_q=True)
>>> H
array([[  2.        , -11.65843866,   1.42005301,   0.25349066],
       [ -9.94987437,  14.53535354,  -5.31022304,   2.43081618],
       [  0.        ,  -1.83299243,   0.38969961,  -0.51527034],
       [  0.        ,   0.        ,  -3.83189513,   1.07494686]])
>>> np.allclose(Q @ H @ Q.conj().T - A, np.zeros((4, 4)))
True
*)

val hilbert : int -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Create a Hilbert matrix of order `n`.

Returns the `n` by `n` array with entries `h[i,j] = 1 / (i + j + 1)`.

Parameters
----------
n : int
    The size of the array to create.

Returns
-------
h : (n, n) ndarray
    The Hilbert matrix.

See Also
--------
invhilbert : Compute the inverse of a Hilbert matrix.

Notes
-----
.. versionadded:: 0.10.0

Examples
--------
>>> from scipy.linalg import hilbert
>>> hilbert(3)
array([[ 1.        ,  0.5       ,  0.33333333],
       [ 0.5       ,  0.33333333,  0.25      ],
       [ 0.33333333,  0.25      ,  0.2       ]])
*)

val inv : ?overwrite_a:bool -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the inverse of a matrix.

Parameters
----------
a : array_like
    Square matrix to be inverted.
overwrite_a : bool, optional
    Discard data in `a` (may improve performance). Default is False.
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
ainv : ndarray
    Inverse of the matrix `a`.

Raises
------
LinAlgError
    If `a` is singular.
ValueError
    If `a` is not square, or not 2-dimensional.

Examples
--------
>>> from scipy import linalg
>>> a = np.array([[1., 2.], [3., 4.]])
>>> linalg.inv(a)
array([[-2. ,  1. ],
       [ 1.5, -0.5]])
>>> np.dot(a, linalg.inv(a))
array([[ 1.,  0.],
       [ 0.,  1.]])
*)

val invhilbert : ?exact:bool -> n:int -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the inverse of the Hilbert matrix of order `n`.

The entries in the inverse of a Hilbert matrix are integers.  When `n`
is greater than 14, some entries in the inverse exceed the upper limit
of 64 bit integers.  The `exact` argument provides two options for
dealing with these large integers.

Parameters
----------
n : int
    The order of the Hilbert matrix.
exact : bool, optional
    If False, the data type of the array that is returned is np.float64,
    and the array is an approximation of the inverse.
    If True, the array is the exact integer inverse array.  To represent
    the exact inverse when n > 14, the returned array is an object array
    of long integers.  For n <= 14, the exact inverse is returned as an
    array with data type np.int64.

Returns
-------
invh : (n, n) ndarray
    The data type of the array is np.float64 if `exact` is False.
    If `exact` is True, the data type is either np.int64 (for n <= 14)
    or object (for n > 14).  In the latter case, the objects in the
    array will be long integers.

See Also
--------
hilbert : Create a Hilbert matrix.

Notes
-----
.. versionadded:: 0.10.0

Examples
--------
>>> from scipy.linalg import invhilbert
>>> invhilbert(4)
array([[   16.,  -120.,   240.,  -140.],
       [ -120.,  1200., -2700.,  1680.],
       [  240., -2700.,  6480., -4200.],
       [ -140.,  1680., -4200.,  2800.]])
>>> invhilbert(4, exact=True)
array([[   16,  -120,   240,  -140],
       [ -120,  1200, -2700,  1680],
       [  240, -2700,  6480, -4200],
       [ -140,  1680, -4200,  2800]], dtype=int64)
>>> invhilbert(16)[7,7]
4.2475099528537506e+19
>>> invhilbert(16, exact=True)[7,7]
42475099528537378560L
*)

val invpascal : ?kind:string -> ?exact:bool -> n:int -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Returns the inverse of the n x n Pascal matrix.

The Pascal matrix is a matrix containing the binomial coefficients as
its elements.

Parameters
----------
n : int
    The size of the matrix to create; that is, the result is an n x n
    matrix.
kind : str, optional
    Must be one of 'symmetric', 'lower', or 'upper'.
    Default is 'symmetric'.
exact : bool, optional
    If `exact` is True, the result is either an array of type
    ``numpy.int64`` (if `n` <= 35) or an object array of Python integers.
    If `exact` is False, the coefficients in the matrix are computed using
    `scipy.special.comb` with `exact=False`.  The result will be a floating
    point array, and for large `n`, the values in the array will not be the
    exact coefficients.

Returns
-------
invp : (n, n) ndarray
    The inverse of the Pascal matrix.

See Also
--------
pascal

Notes
-----

.. versionadded:: 0.16.0

References
----------
.. [1] 'Pascal matrix', https://en.wikipedia.org/wiki/Pascal_matrix
.. [2] Cohen, A. M., 'The inverse of a Pascal matrix', Mathematical
       Gazette, 59(408), pp. 111-112, 1975.

Examples
--------
>>> from scipy.linalg import invpascal, pascal
>>> invp = invpascal(5)
>>> invp
array([[  5, -10,  10,  -5,   1],
       [-10,  30, -35,  19,  -4],
       [ 10, -35,  46, -27,   6],
       [ -5,  19, -27,  17,  -4],
       [  1,  -4,   6,  -4,   1]])

>>> p = pascal(5)
>>> p.dot(invp)
array([[ 1.,  0.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  0.,  1.]])

An example of the use of `kind` and `exact`:

>>> invpascal(5, kind='lower', exact=False)
array([[ 1., -0.,  0., -0.,  0.],
       [-1.,  1., -0.,  0., -0.],
       [ 1., -2.,  1., -0.,  0.],
       [-1.,  3., -3.,  1., -0.],
       [ 1., -4.,  6., -4.,  1.]])
*)

val kron : a:[>`Ndarray] Np.Obj.t -> b:Py.Object.t -> unit -> Py.Object.t
(**
Kronecker product.

The result is the block matrix::

    a[0,0]*b    a[0,1]*b  ... a[0,-1]*b
    a[1,0]*b    a[1,1]*b  ... a[1,-1]*b
    ...
    a[-1,0]*b   a[-1,1]*b ... a[-1,-1]*b

Parameters
----------
a : (M, N) ndarray
    Input array
b : (P, Q) ndarray
    Input array

Returns
-------
A : (M*P, N*Q) ndarray
    Kronecker product of `a` and `b`.

Examples
--------
>>> from numpy import array
>>> from scipy.linalg import kron
>>> kron(array([[1,2],[3,4]]), array([[1,1,1]]))
array([[1, 1, 1, 2, 2, 2],
       [3, 3, 3, 4, 4, 4]])
*)

val ldl : ?lower:bool -> ?hermitian:bool -> ?overwrite_a:bool -> ?check_finite:bool -> a:Py.Object.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Computes the LDLt or Bunch-Kaufman factorization of a symmetric/
hermitian matrix.

This function returns a block diagonal matrix D consisting blocks of size
at most 2x2 and also a possibly permuted unit lower triangular matrix
``L`` such that the factorization ``A = L D L^H`` or ``A = L D L^T``
holds. If ``lower`` is False then (again possibly permuted) upper
triangular matrices are returned as outer factors.

The permutation array can be used to triangularize the outer factors
simply by a row shuffle, i.e., ``lu[perm, :]`` is an upper/lower
triangular matrix. This is also equivalent to multiplication with a
permutation matrix ``P.dot(lu)`` where ``P`` is a column-permuted
identity matrix ``I[:, perm]``.

Depending on the value of the boolean ``lower``, only upper or lower
triangular part of the input array is referenced. Hence a triangular
matrix on entry would give the same result as if the full matrix is
supplied.

Parameters
----------
a : array_like
    Square input array
lower : bool, optional
    This switches between the lower and upper triangular outer factors of
    the factorization. Lower triangular (``lower=True``) is the default.
hermitian : bool, optional
    For complex-valued arrays, this defines whether ``a = a.conj().T`` or
    ``a = a.T`` is assumed. For real-valued arrays, this switch has no
    effect.
overwrite_a : bool, optional
    Allow overwriting data in ``a`` (may enhance performance). The default
    is False.
check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
lu : ndarray
    The (possibly) permuted upper/lower triangular outer factor of the
    factorization.
d : ndarray
    The block diagonal multiplier of the factorization.
perm : ndarray
    The row-permutation index array that brings lu into triangular form.

Raises
------
ValueError
    If input array is not square.
ComplexWarning
    If a complex-valued array with nonzero imaginary parts on the
    diagonal is given and hermitian is set to True.

Examples
--------
Given an upper triangular array `a` that represents the full symmetric
array with its entries, obtain `l`, 'd' and the permutation vector `perm`:

>>> import numpy as np
>>> from scipy.linalg import ldl
>>> a = np.array([[2, -1, 3], [0, 2, 0], [0, 0, 1]])
>>> lu, d, perm = ldl(a, lower=0) # Use the upper part
>>> lu
array([[ 0. ,  0. ,  1. ],
       [ 0. ,  1. , -0.5],
       [ 1. ,  1. ,  1.5]])
>>> d
array([[-5. ,  0. ,  0. ],
       [ 0. ,  1.5,  0. ],
       [ 0. ,  0. ,  2. ]])
>>> perm
array([2, 1, 0])
>>> lu[perm, :]
array([[ 1. ,  1. ,  1.5],
       [ 0. ,  1. , -0.5],
       [ 0. ,  0. ,  1. ]])
>>> lu.dot(d).dot(lu.T)
array([[ 2., -1.,  3.],
       [-1.,  2.,  0.],
       [ 3.,  0.,  1.]])

Notes
-----
This function uses ``?SYTRF`` routines for symmetric matrices and
``?HETRF`` routines for Hermitian matrices from LAPACK. See [1]_ for
the algorithm details.

Depending on the ``lower`` keyword value, only lower or upper triangular
part of the input array is referenced. Moreover, this keyword also defines
the structure of the outer factors of the factorization.

.. versionadded:: 1.1.0

See also
--------
cholesky, lu

References
----------
.. [1] J.R. Bunch, L. Kaufman, Some stable methods for calculating
   inertia and solving symmetric linear systems, Math. Comput. Vol.31,
   1977. DOI: 10.2307/2005787
*)

val leslie : f:[>`Ndarray] Np.Obj.t -> s:Py.Object.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Create a Leslie matrix.

Given the length n array of fecundity coefficients `f` and the length
n-1 array of survival coefficients `s`, return the associated Leslie
matrix.

Parameters
----------
f : (N,) array_like
    The 'fecundity' coefficients.
s : (N-1,) array_like
    The 'survival' coefficients, has to be 1-D.  The length of `s`
    must be one less than the length of `f`, and it must be at least 1.

Returns
-------
L : (N, N) ndarray
    The array is zero except for the first row,
    which is `f`, and the first sub-diagonal, which is `s`.
    The data-type of the array will be the data-type of ``f[0]+s[0]``.

Notes
-----
.. versionadded:: 0.8.0

The Leslie matrix is used to model discrete-time, age-structured
population growth [1]_ [2]_. In a population with `n` age classes, two sets
of parameters define a Leslie matrix: the `n` 'fecundity coefficients',
which give the number of offspring per-capita produced by each age
class, and the `n` - 1 'survival coefficients', which give the
per-capita survival rate of each age class.

References
----------
.. [1] P. H. Leslie, On the use of matrices in certain population
       mathematics, Biometrika, Vol. 33, No. 3, 183--212 (Nov. 1945)
.. [2] P. H. Leslie, Some further notes on the use of matrices in
       population mathematics, Biometrika, Vol. 35, No. 3/4, 213--245
       (Dec. 1948)

Examples
--------
>>> from scipy.linalg import leslie
>>> leslie([0.1, 2.0, 1.0, 0.1], [0.2, 0.8, 0.7])
array([[ 0.1,  2. ,  1. ,  0.1],
       [ 0.2,  0. ,  0. ,  0. ],
       [ 0. ,  0.8,  0. ,  0. ],
       [ 0. ,  0. ,  0.7,  0. ]])
*)

val logm : ?disp:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * float)
(**
Compute matrix logarithm.

The matrix logarithm is the inverse of
expm: expm(logm(`A`)) == `A`

Parameters
----------
A : (N, N) array_like
    Matrix whose logarithm to evaluate
disp : bool, optional
    Print warning if error in the result is estimated large
    instead of returning estimated error. (Default: True)

Returns
-------
logm : (N, N) ndarray
    Matrix logarithm of `A`
errest : float
    (if disp == False)

    1-norm of the estimated error, ||err||_1 / ||A||_1

References
----------
.. [1] Awad H. Al-Mohy and Nicholas J. Higham (2012)
       'Improved Inverse Scaling and Squaring Algorithms
       for the Matrix Logarithm.'
       SIAM Journal on Scientific Computing, 34 (4). C152-C169.
       ISSN 1095-7197

.. [2] Nicholas J. Higham (2008)
       'Functions of Matrices: Theory and Computation'
       ISBN 978-0-898716-46-7

.. [3] Nicholas J. Higham and Lijing lin (2011)
       'A Schur-Pade Algorithm for Fractional Powers of a Matrix.'
       SIAM Journal on Matrix Analysis and Applications,
       32 (3). pp. 1056-1078. ISSN 0895-4798

Examples
--------
>>> from scipy.linalg import logm, expm
>>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
>>> b = logm(a)
>>> b
array([[-1.02571087,  2.05142174],
       [ 0.68380725,  1.02571087]])
>>> expm(b)         # Verify expm(logm(a)) returns a
array([[ 1.,  3.],
       [ 1.,  4.]])
*)

val lstsq : ?cond:float -> ?overwrite_a:bool -> ?overwrite_b:bool -> ?check_finite:bool -> ?lapack_driver:string -> a:[>`Ndarray] Np.Obj.t -> b:Py.Object.t -> unit -> (Py.Object.t * Py.Object.t * int * Py.Object.t option)
(**
Compute least-squares solution to equation Ax = b.

Compute a vector x such that the 2-norm ``|b - A x|`` is minimized.

Parameters
----------
a : (M, N) array_like
    Left hand side array
b : (M,) or (M, K) array_like
    Right hand side array
cond : float, optional
    Cutoff for 'small' singular values; used to determine effective
    rank of a. Singular values smaller than
    ``rcond * largest_singular_value`` are considered zero.
overwrite_a : bool, optional
    Discard data in `a` (may enhance performance). Default is False.
overwrite_b : bool, optional
    Discard data in `b` (may enhance performance). Default is False.
check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.
lapack_driver : str, optional
    Which LAPACK driver is used to solve the least-squares problem.
    Options are ``'gelsd'``, ``'gelsy'``, ``'gelss'``. Default
    (``'gelsd'``) is a good choice.  However, ``'gelsy'`` can be slightly
    faster on many problems.  ``'gelss'`` was used historically.  It is
    generally slow but uses less memory.

    .. versionadded:: 0.17.0

Returns
-------
x : (N,) or (N, K) ndarray
    Least-squares solution.  Return shape matches shape of `b`.
residues : (K,) ndarray or float
    Square of the 2-norm for each column in ``b - a x``, if ``M > N`` and
    ``ndim(A) == n`` (returns a scalar if b is 1-D). Otherwise a
    (0,)-shaped array is returned.
rank : int
    Effective rank of `a`.
s : (min(M, N),) ndarray or None
    Singular values of `a`. The condition number of a is
    ``abs(s[0] / s[-1])``.

Raises
------
LinAlgError
    If computation does not converge.

ValueError
    When parameters are not compatible.

See Also
--------
scipy.optimize.nnls : linear least squares with non-negativity constraint

Notes
-----
When ``'gelsy'`` is used as a driver, `residues` is set to a (0,)-shaped
array and `s` is always ``None``.

Examples
--------
>>> from scipy.linalg import lstsq
>>> import matplotlib.pyplot as plt

Suppose we have the following data:

>>> x = np.array([1, 2.5, 3.5, 4, 5, 7, 8.5])
>>> y = np.array([0.3, 1.1, 1.5, 2.0, 3.2, 6.6, 8.6])

We want to fit a quadratic polynomial of the form ``y = a + b*x**2``
to this data.  We first form the 'design matrix' M, with a constant
column of 1s and a column containing ``x**2``:

>>> M = x[:, np.newaxis]**[0, 2]
>>> M
array([[  1.  ,   1.  ],
       [  1.  ,   6.25],
       [  1.  ,  12.25],
       [  1.  ,  16.  ],
       [  1.  ,  25.  ],
       [  1.  ,  49.  ],
       [  1.  ,  72.25]])

We want to find the least-squares solution to ``M.dot(p) = y``,
where ``p`` is a vector with length 2 that holds the parameters
``a`` and ``b``.

>>> p, res, rnk, s = lstsq(M, y)
>>> p
array([ 0.20925829,  0.12013861])

Plot the data and the fitted curve.

>>> plt.plot(x, y, 'o', label='data')
>>> xx = np.linspace(0, 9, 101)
>>> yy = p[0] + p[1]*xx**2
>>> plt.plot(xx, yy, label='least squares fit, $y = a + bx^2$')
>>> plt.xlabel('x')
>>> plt.ylabel('y')
>>> plt.legend(framealpha=1, shadow=True)
>>> plt.grid(alpha=0.25)
>>> plt.show()
*)

val lu : ?permute_l:bool -> ?overwrite_a:bool -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * Py.Object.t * Py.Object.t * Py.Object.t)
(**
Compute pivoted LU decomposition of a matrix.

The decomposition is::

    A = P L U

where P is a permutation matrix, L lower triangular with unit
diagonal elements, and U upper triangular.

Parameters
----------
a : (M, N) array_like
    Array to decompose
permute_l : bool, optional
    Perform the multiplication P*L  (Default: do not permute)
overwrite_a : bool, optional
    Whether to overwrite data in a (may improve performance)
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
**(If permute_l == False)**

p : (M, M) ndarray
    Permutation matrix
l : (M, K) ndarray
    Lower triangular or trapezoidal matrix with unit diagonal.
    K = min(M, N)
u : (K, N) ndarray
    Upper triangular or trapezoidal matrix

**(If permute_l == True)**

pl : (M, K) ndarray
    Permuted L matrix.
    K = min(M, N)
u : (K, N) ndarray
    Upper triangular or trapezoidal matrix

Notes
-----
This is a LU factorization routine written for SciPy.

Examples
--------
>>> from scipy.linalg import lu
>>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
>>> p, l, u = lu(A)
>>> np.allclose(A - p @ l @ u, np.zeros((4, 4)))
True
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

val matrix_balance : ?permute:bool -> ?scale:float -> ?separate:bool -> ?overwrite_a:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Compute a diagonal similarity transformation for row/column balancing.

The balancing tries to equalize the row and column 1-norms by applying
a similarity transformation such that the magnitude variation of the
matrix entries is reflected to the scaling matrices.

Moreover, if enabled, the matrix is first permuted to isolate the upper
triangular parts of the matrix and, again if scaling is also enabled,
only the remaining subblocks are subjected to scaling.

The balanced matrix satisfies the following equality

.. math::

                    B = T^{-1} A T

The scaling coefficients are approximated to the nearest power of 2
to avoid round-off errors.

Parameters
----------
A : (n, n) array_like
    Square data matrix for the balancing.
permute : bool, optional
    The selector to define whether permutation of A is also performed
    prior to scaling.
scale : bool, optional
    The selector to turn on and off the scaling. If False, the matrix
    will not be scaled.
separate : bool, optional
    This switches from returning a full matrix of the transformation
    to a tuple of two separate 1D permutation and scaling arrays.
overwrite_a : bool, optional
    This is passed to xGEBAL directly. Essentially, overwrites the result
    to the data. It might increase the space efficiency. See LAPACK manual
    for details. This is False by default.

Returns
-------
B : (n, n) ndarray
    Balanced matrix
T : (n, n) ndarray
    A possibly permuted diagonal matrix whose nonzero entries are
    integer powers of 2 to avoid numerical truncation errors.
scale, perm : (n,) ndarray
    If ``separate`` keyword is set to True then instead of the array
    ``T`` above, the scaling and the permutation vectors are given
    separately as a tuple without allocating the full array ``T``.

Notes
-----

This algorithm is particularly useful for eigenvalue and matrix
decompositions and in many cases it is already called by various
LAPACK routines.

The algorithm is based on the well-known technique of [1]_ and has
been modified to account for special cases. See [2]_ for details
which have been implemented since LAPACK v3.5.0. Before this version
there are corner cases where balancing can actually worsen the
conditioning. See [3]_ for such examples.

The code is a wrapper around LAPACK's xGEBAL routine family for matrix
balancing.

.. versionadded:: 0.19.0

Examples
--------
>>> from scipy import linalg
>>> x = np.array([[1,2,0], [9,1,0.01], [1,2,10*np.pi]])

>>> y, permscale = linalg.matrix_balance(x)
>>> np.abs(x).sum(axis=0) / np.abs(x).sum(axis=1)
array([ 3.66666667,  0.4995005 ,  0.91312162])

>>> np.abs(y).sum(axis=0) / np.abs(y).sum(axis=1)
array([ 1.2       ,  1.27041742,  0.92658316])  # may vary

>>> permscale  # only powers of 2 (0.5 == 2^(-1))
array([[  0.5,   0. ,  0. ],  # may vary
       [  0. ,   1. ,  0. ],
       [  0. ,   0. ,  1. ]])

References
----------
.. [1] : B.N. Parlett and C. Reinsch, 'Balancing a Matrix for
   Calculation of Eigenvalues and Eigenvectors', Numerische Mathematik,
   Vol.13(4), 1969, DOI:10.1007/BF02165404

.. [2] : R. James, J. Langou, B.R. Lowery, 'On matrix balancing and
   eigenvector computation', 2014, Available online:
   https://arxiv.org/abs/1401.5766

.. [3] :  D.S. Watkins. A case where balancing is harmful.
   Electron. Trans. Numer. Anal, Vol.23, 2006.
*)

val norm : ?ord:[`Fro | `PyObject of Py.Object.t] -> ?axis:[`T2_tuple_of_ints of Py.Object.t | `I of int] -> ?keepdims:bool -> ?check_finite:bool -> a:Py.Object.t -> unit -> Py.Object.t
(**
Matrix or vector norm.

This function is able to return one of seven different matrix norms,
or one of an infinite number of vector norms (described below), depending
on the value of the ``ord`` parameter.

Parameters
----------
a : (M,) or (M, N) array_like
    Input array.  If `axis` is None, `a` must be 1-D or 2-D.
ord : {non-zero int, inf, -inf, 'fro'}, optional
    Order of the norm (see table under ``Notes``). inf means numpy's
    `inf` object
axis : {int, 2-tuple of ints, None}, optional
    If `axis` is an integer, it specifies the axis of `a` along which to
    compute the vector norms.  If `axis` is a 2-tuple, it specifies the
    axes that hold 2-D matrices, and the matrix norms of these matrices
    are computed.  If `axis` is None then either a vector norm (when `a`
    is 1-D) or a matrix norm (when `a` is 2-D) is returned.
keepdims : bool, optional
    If this is set to True, the axes which are normed over are left in the
    result as dimensions with size one.  With this option the result will
    broadcast correctly against the original `a`.
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
n : float or ndarray
    Norm of the matrix or vector(s).

Notes
-----
For values of ``ord <= 0``, the result is, strictly speaking, not a
mathematical 'norm', but it may still be useful for various numerical
purposes.

The following norms can be calculated:

=====  ============================  ==========================
ord    norm for matrices             norm for vectors
=====  ============================  ==========================
None   Frobenius norm                2-norm
'fro'  Frobenius norm                --
inf    max(sum(abs(x), axis=1))      max(abs(x))
-inf   min(sum(abs(x), axis=1))      min(abs(x))
0      --                            sum(x != 0)
1      max(sum(abs(x), axis=0))      as below
-1     min(sum(abs(x), axis=0))      as below
2      2-norm (largest sing. value)  as below
-2     smallest singular value       as below
other  --                            sum(abs(x)**ord)**(1./ord)
=====  ============================  ==========================

The Frobenius norm is given by [1]_:

    :math:`||A||_F = [\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

The ``axis`` and ``keepdims`` arguments are passed directly to
``numpy.linalg.norm`` and are only usable if they are supported
by the version of numpy in use.

References
----------
.. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
       Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15

Examples
--------
>>> from scipy.linalg import norm
>>> a = np.arange(9) - 4.0
>>> a
array([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
>>> b = a.reshape((3, 3))
>>> b
array([[-4., -3., -2.],
       [-1.,  0.,  1.],
       [ 2.,  3.,  4.]])

>>> norm(a)
7.745966692414834
>>> norm(b)
7.745966692414834
>>> norm(b, 'fro')
7.745966692414834
>>> norm(a, np.inf)
4
>>> norm(b, np.inf)
9
>>> norm(a, -np.inf)
0
>>> norm(b, -np.inf)
2

>>> norm(a, 1)
20
>>> norm(b, 1)
7
>>> norm(a, -1)
-4.6566128774142013e-010
>>> norm(b, -1)
6
>>> norm(a, 2)
7.745966692414834
>>> norm(b, 2)
7.3484692283495345

>>> norm(a, -2)
0
>>> norm(b, -2)
1.8570331885190563e-016
>>> norm(a, 3)
5.8480354764257312
>>> norm(a, -3)
0
*)

val null_space : ?rcond:float -> a:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Construct an orthonormal basis for the null space of A using SVD

Parameters
----------
A : (M, N) array_like
    Input array
rcond : float, optional
    Relative condition number. Singular values ``s`` smaller than
    ``rcond * max(s)`` are considered zero.
    Default: floating point eps * max(M,N).

Returns
-------
Z : (N, K) ndarray
    Orthonormal basis for the null space of A.
    K = dimension of effective null space, as determined by rcond

See also
--------
svd : Singular value decomposition of a matrix
orth : Matrix range

Examples
--------
One-dimensional null space:

>>> from scipy.linalg import null_space
>>> A = np.array([[1, 1], [1, 1]])
>>> ns = null_space(A)
>>> ns * np.sign(ns[0,0])  # Remove the sign ambiguity of the vector
array([[ 0.70710678],
       [-0.70710678]])

Two-dimensional null space:

>>> B = np.random.rand(3, 5)
>>> Z = null_space(B)
>>> Z.shape
(5, 2)
>>> np.allclose(B.dot(Z), 0)
True

The basis vectors are orthonormal (up to rounding error):

>>> Z.T.dot(Z)
array([[  1.00000000e+00,   6.92087741e-17],
       [  6.92087741e-17,   1.00000000e+00]])
*)

val ordqz : ?sort:[`Iuc | `Ouc | `Rhp | `Callable of Py.Object.t | `Lhp] -> ?output:[`Real | `Complex] -> ?overwrite_a:bool -> ?overwrite_b:bool -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
QZ decomposition for a pair of matrices with reordering.

.. versionadded:: 0.17.0

Parameters
----------
A : (N, N) array_like
    2d array to decompose
B : (N, N) array_like
    2d array to decompose
sort : {callable, 'lhp', 'rhp', 'iuc', 'ouc'}, optional
    Specifies whether the upper eigenvalues should be sorted. A
    callable may be passed that, given an ordered pair ``(alpha,
    beta)`` representing the eigenvalue ``x = (alpha/beta)``,
    returns a boolean denoting whether the eigenvalue should be
    sorted to the top-left (True). For the real matrix pairs
    ``beta`` is real while ``alpha`` can be complex, and for
    complex matrix pairs both ``alpha`` and ``beta`` can be
    complex. The callable must be able to accept a numpy
    array. Alternatively, string parameters may be used:

        - 'lhp'   Left-hand plane (x.real < 0.0)
        - 'rhp'   Right-hand plane (x.real > 0.0)
        - 'iuc'   Inside the unit circle (x*x.conjugate() < 1.0)
        - 'ouc'   Outside the unit circle (x*x.conjugate() > 1.0)

    With the predefined sorting functions, an infinite eigenvalue
    (i.e. ``alpha != 0`` and ``beta = 0``) is considered to lie in
    neither the left-hand nor the right-hand plane, but it is
    considered to lie outside the unit circle. For the eigenvalue
    ``(alpha, beta) = (0, 0)`` the predefined sorting functions
    all return `False`.
output : str {'real','complex'}, optional
    Construct the real or complex QZ decomposition for real matrices.
    Default is 'real'.
overwrite_a : bool, optional
    If True, the contents of A are overwritten.
overwrite_b : bool, optional
    If True, the contents of B are overwritten.
check_finite : bool, optional
    If true checks the elements of `A` and `B` are finite numbers. If
    false does no checking and passes matrix through to
    underlying algorithm.

Returns
-------
AA : (N, N) ndarray
    Generalized Schur form of A.
BB : (N, N) ndarray
    Generalized Schur form of B.
alpha : (N,) ndarray
    alpha = alphar + alphai * 1j. See notes.
beta : (N,) ndarray
    See notes.
Q : (N, N) ndarray
    The left Schur vectors.
Z : (N, N) ndarray
    The right Schur vectors.

Notes
-----
On exit, ``(ALPHAR(j) + ALPHAI(j)*i)/BETA(j), j=1,...,N``, will be the
generalized eigenvalues.  ``ALPHAR(j) + ALPHAI(j)*i`` and
``BETA(j),j=1,...,N`` are the diagonals of the complex Schur form (S,T)
that would result if the 2-by-2 diagonal blocks of the real generalized
Schur form of (A,B) were further reduced to triangular form using complex
unitary transformations. If ALPHAI(j) is zero, then the j-th eigenvalue is
real; if positive, then the ``j``-th and ``(j+1)``-st eigenvalues are a
complex conjugate pair, with ``ALPHAI(j+1)`` negative.

See also
--------
qz

Examples
--------
>>> from scipy.linalg import ordqz
>>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
>>> B = np.array([[0, 6, 0, 0], [5, 0, 2, 1], [5, 2, 6, 6], [4, 7, 7, 7]])
>>> AA, BB, alpha, beta, Q, Z = ordqz(A, B, sort='lhp')

Since we have sorted for left half plane eigenvalues, negatives come first

>>> (alpha/beta).real < 0
array([ True,  True, False, False], dtype=bool)
*)

val orth : ?rcond:float -> a:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Construct an orthonormal basis for the range of A using SVD

Parameters
----------
A : (M, N) array_like
    Input array
rcond : float, optional
    Relative condition number. Singular values ``s`` smaller than
    ``rcond * max(s)`` are considered zero.
    Default: floating point eps * max(M,N).

Returns
-------
Q : (M, K) ndarray
    Orthonormal basis for the range of A.
    K = effective rank of A, as determined by rcond

See also
--------
svd : Singular value decomposition of a matrix
null_space : Matrix null space

Examples
--------
>>> from scipy.linalg import orth
>>> A = np.array([[2, 0, 0], [0, 5, 0]])  # rank 2 array
>>> orth(A)
array([[0., 1.],
       [1., 0.]])
>>> orth(A.T)
array([[0., 1.],
       [1., 0.],
       [0., 0.]])
*)

val orthogonal_procrustes : ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * float)
(**
Compute the matrix solution of the orthogonal Procrustes problem.

Given matrices A and B of equal shape, find an orthogonal matrix R
that most closely maps A to B using the algorithm given in [1]_.

Parameters
----------
A : (M, N) array_like
    Matrix to be mapped.
B : (M, N) array_like
    Target matrix.
check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
R : (N, N) ndarray
    The matrix solution of the orthogonal Procrustes problem.
    Minimizes the Frobenius norm of ``(A @ R) - B``, subject to
    ``R.T @ R = I``.
scale : float
    Sum of the singular values of ``A.T @ B``.

Raises
------
ValueError
    If the input array shapes don't match or if check_finite is True and
    the arrays contain Inf or NaN.

Notes
-----
Note that unlike higher level Procrustes analyses of spatial data, this
function only uses orthogonal transformations like rotations and
reflections, and it does not use scaling or translation.

.. versionadded:: 0.15.0

References
----------
.. [1] Peter H. Schonemann, 'A generalized solution of the orthogonal
       Procrustes problem', Psychometrica -- Vol. 31, No. 1, March, 1996.

Examples
--------
>>> from scipy.linalg import orthogonal_procrustes
>>> A = np.array([[ 2,  0,  1], [-2,  0,  0]])

Flip the order of columns and check for the anti-diagonal mapping

>>> R, sca = orthogonal_procrustes(A, np.fliplr(A))
>>> R
array([[-5.34384992e-17,  0.00000000e+00,  1.00000000e+00],
       [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00],
       [ 1.00000000e+00,  0.00000000e+00, -7.85941422e-17]])
>>> sca
9.0
*)

val pascal : ?kind:string -> ?exact:bool -> n:int -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Returns the n x n Pascal matrix.

The Pascal matrix is a matrix containing the binomial coefficients as
its elements.

Parameters
----------
n : int
    The size of the matrix to create; that is, the result is an n x n
    matrix.
kind : str, optional
    Must be one of 'symmetric', 'lower', or 'upper'.
    Default is 'symmetric'.
exact : bool, optional
    If `exact` is True, the result is either an array of type
    numpy.uint64 (if n < 35) or an object array of Python long integers.
    If `exact` is False, the coefficients in the matrix are computed using
    `scipy.special.comb` with `exact=False`.  The result will be a floating
    point array, and the values in the array will not be the exact
    coefficients, but this version is much faster than `exact=True`.

Returns
-------
p : (n, n) ndarray
    The Pascal matrix.

See Also
--------
invpascal

Notes
-----
See https://en.wikipedia.org/wiki/Pascal_matrix for more information
about Pascal matrices.

.. versionadded:: 0.11.0

Examples
--------
>>> from scipy.linalg import pascal
>>> pascal(4)
array([[ 1,  1,  1,  1],
       [ 1,  2,  3,  4],
       [ 1,  3,  6, 10],
       [ 1,  4, 10, 20]], dtype=uint64)
>>> pascal(4, kind='lower')
array([[1, 0, 0, 0],
       [1, 1, 0, 0],
       [1, 2, 1, 0],
       [1, 3, 3, 1]], dtype=uint64)
>>> pascal(50)[-1, -1]
25477612258980856902730428600L
>>> from scipy.special import comb
>>> comb(98, 49, exact=True)
25477612258980856902730428600L
*)

val pinv : ?cond:Py.Object.t -> ?rcond:Py.Object.t -> ?return_rank:bool -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Compute the (Moore-Penrose) pseudo-inverse of a matrix.

Calculate a generalized inverse of a matrix using a least-squares
solver.

Parameters
----------
a : (M, N) array_like
    Matrix to be pseudo-inverted.
cond, rcond : float, optional
    Cutoff factor for 'small' singular values. In `lstsq`,
    singular values less than ``cond*largest_singular_value`` will be
    considered as zero. If both are omitted, the default value
    ``max(M, N) * eps`` is passed to `lstsq` where ``eps`` is the
    corresponding machine precision value of the datatype of ``a``.

    .. versionchanged:: 1.3.0
        Previously the default cutoff value was just `eps` without the
        factor ``max(M, N)``.

return_rank : bool, optional
    if True, return the effective rank of the matrix
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
B : (N, M) ndarray
    The pseudo-inverse of matrix `a`.
rank : int
    The effective rank of the matrix.  Returned if return_rank == True

Raises
------
LinAlgError
    If computation does not converge.

Examples
--------
>>> from scipy import linalg
>>> a = np.random.randn(9, 6)
>>> B = linalg.pinv(a)
>>> np.allclose(a, np.dot(a, np.dot(B, a)))
True
>>> np.allclose(B, np.dot(B, np.dot(a, B)))
True
*)

val pinv2 : ?cond:Py.Object.t -> ?rcond:Py.Object.t -> ?return_rank:bool -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Compute the (Moore-Penrose) pseudo-inverse of a matrix.

Calculate a generalized inverse of a matrix using its
singular-value decomposition and including all 'large' singular
values.

Parameters
----------
a : (M, N) array_like
    Matrix to be pseudo-inverted.
cond, rcond : float or None
    Cutoff for 'small' singular values; singular values smaller than this
    value are considered as zero. If both are omitted, the default value
    ``max(M,N)*largest_singular_value*eps`` is used where ``eps`` is the
    machine precision value of the datatype of ``a``.

    .. versionchanged:: 1.3.0
        Previously the default cutoff value was just ``eps*f`` where ``f``
        was ``1e3`` for single precision and ``1e6`` for double precision.

return_rank : bool, optional
    If True, return the effective rank of the matrix.
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
B : (N, M) ndarray
    The pseudo-inverse of matrix `a`.
rank : int
    The effective rank of the matrix.  Returned if `return_rank` is True.

Raises
------
LinAlgError
    If SVD computation does not converge.

Examples
--------
>>> from scipy import linalg
>>> a = np.random.randn(9, 6)
>>> B = linalg.pinv2(a)
>>> np.allclose(a, np.dot(a, np.dot(B, a)))
True
>>> np.allclose(B, np.dot(B, np.dot(a, B)))
True
*)

val pinvh : ?cond:Py.Object.t -> ?rcond:Py.Object.t -> ?lower:bool -> ?return_rank:bool -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Compute the (Moore-Penrose) pseudo-inverse of a Hermitian matrix.

Calculate a generalized inverse of a Hermitian or real symmetric matrix
using its eigenvalue decomposition and including all eigenvalues with
'large' absolute value.

Parameters
----------
a : (N, N) array_like
    Real symmetric or complex hermetian matrix to be pseudo-inverted
cond, rcond : float or None
    Cutoff for 'small' singular values; singular values smaller than this
    value are considered as zero. If both are omitted, the default
    ``max(M,N)*largest_eigenvalue*eps`` is used where ``eps`` is the
    machine precision value of the datatype of ``a``.

    .. versionchanged:: 1.3.0
        Previously the default cutoff value was just ``eps*f`` where ``f``
        was ``1e3`` for single precision and ``1e6`` for double precision.

lower : bool, optional
    Whether the pertinent array data is taken from the lower or upper
    triangle of `a`. (Default: lower)
return_rank : bool, optional
    If True, return the effective rank of the matrix.
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
B : (N, N) ndarray
    The pseudo-inverse of matrix `a`.
rank : int
    The effective rank of the matrix.  Returned if `return_rank` is True.

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

val polar : ?side:[`Left | `Right] -> a:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Compute the polar decomposition.

Returns the factors of the polar decomposition [1]_ `u` and `p` such
that ``a = up`` (if `side` is 'right') or ``a = pu`` (if `side` is
'left'), where `p` is positive semidefinite.  Depending on the shape
of `a`, either the rows or columns of `u` are orthonormal.  When `a`
is a square array, `u` is a square unitary array.  When `a` is not
square, the 'canonical polar decomposition' [2]_ is computed.

Parameters
----------
a : (m, n) array_like
    The array to be factored.
side : {'left', 'right'}, optional
    Determines whether a right or left polar decomposition is computed.
    If `side` is 'right', then ``a = up``.  If `side` is 'left',  then
    ``a = pu``.  The default is 'right'.

Returns
-------
u : (m, n) ndarray
    If `a` is square, then `u` is unitary.  If m > n, then the columns
    of `a` are orthonormal, and if m < n, then the rows of `u` are
    orthonormal.
p : ndarray
    `p` is Hermitian positive semidefinite.  If `a` is nonsingular, `p`
    is positive definite.  The shape of `p` is (n, n) or (m, m), depending
    on whether `side` is 'right' or 'left', respectively.

References
----------
.. [1] R. A. Horn and C. R. Johnson, 'Matrix Analysis', Cambridge
       University Press, 1985.
.. [2] N. J. Higham, 'Functions of Matrices: Theory and Computation',
       SIAM, 2008.

Examples
--------
>>> from scipy.linalg import polar
>>> a = np.array([[1, -1], [2, 4]])
>>> u, p = polar(a)
>>> u
array([[ 0.85749293, -0.51449576],
       [ 0.51449576,  0.85749293]])
>>> p
array([[ 1.88648444,  1.2004901 ],
       [ 1.2004901 ,  3.94446746]])

A non-square example, with m < n:

>>> b = np.array([[0.5, 1, 2], [1.5, 3, 4]])
>>> u, p = polar(b)
>>> u
array([[-0.21196618, -0.42393237,  0.88054056],
       [ 0.39378971,  0.78757942,  0.4739708 ]])
>>> p
array([[ 0.48470147,  0.96940295,  1.15122648],
       [ 0.96940295,  1.9388059 ,  2.30245295],
       [ 1.15122648,  2.30245295,  3.65696431]])
>>> u.dot(p)   # Verify the decomposition.
array([[ 0.5,  1. ,  2. ],
       [ 1.5,  3. ,  4. ]])
>>> u.dot(u.T)   # The rows of u are orthonormal.
array([[  1.00000000e+00,  -2.07353665e-17],
       [ -2.07353665e-17,   1.00000000e+00]])

Another non-square example, with m > n:

>>> c = b.T
>>> u, p = polar(c)
>>> u
array([[-0.21196618,  0.39378971],
       [-0.42393237,  0.78757942],
       [ 0.88054056,  0.4739708 ]])
>>> p
array([[ 1.23116567,  1.93241587],
       [ 1.93241587,  4.84930602]])
>>> u.dot(p)   # Verify the decomposition.
array([[ 0.5,  1.5],
       [ 1. ,  3. ],
       [ 2. ,  4. ]])
>>> u.T.dot(u)  # The columns of u are orthonormal.
array([[  1.00000000e+00,  -1.26363763e-16],
       [ -1.26363763e-16,   1.00000000e+00]])
*)

val qr : ?overwrite_a:bool -> ?lwork:int -> ?mode:[`Full | `R | `Economic | `Raw] -> ?pivoting:bool -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> (Py.Object.t * Py.Object.t * Py.Object.t)
(**
Compute QR decomposition of a matrix.

Calculate the decomposition ``A = Q R`` where Q is unitary/orthogonal
and R upper triangular.

Parameters
----------
a : (M, N) array_like
    Matrix to be decomposed
overwrite_a : bool, optional
    Whether data in a is overwritten (may improve performance)
lwork : int, optional
    Work array size, lwork >= a.shape[1]. If None or -1, an optimal size
    is computed.
mode : {'full', 'r', 'economic', 'raw'}, optional
    Determines what information is to be returned: either both Q and R
    ('full', default), only R ('r') or both Q and R but computed in
    economy-size ('economic', see Notes). The final option 'raw'
    (added in SciPy 0.11) makes the function return two matrices
    (Q, TAU) in the internal format used by LAPACK.
pivoting : bool, optional
    Whether or not factorization should include pivoting for rank-revealing
    qr decomposition. If pivoting, compute the decomposition
    ``A P = Q R`` as above, but where P is chosen such that the diagonal
    of R is non-increasing.
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
Q : float or complex ndarray
    Of shape (M, M), or (M, K) for ``mode='economic'``.  Not returned
    if ``mode='r'``.
R : float or complex ndarray
    Of shape (M, N), or (K, N) for ``mode='economic'``.  ``K = min(M, N)``.
P : int ndarray
    Of shape (N,) for ``pivoting=True``. Not returned if
    ``pivoting=False``.

Raises
------
LinAlgError
    Raised if decomposition fails

Notes
-----
This is an interface to the LAPACK routines dgeqrf, zgeqrf,
dorgqr, zungqr, dgeqp3, and zgeqp3.

If ``mode=economic``, the shapes of Q and R are (M, K) and (K, N) instead
of (M,M) and (M,N), with ``K=min(M,N)``.

Examples
--------
>>> from scipy import linalg
>>> a = np.random.randn(9, 6)

>>> q, r = linalg.qr(a)
>>> np.allclose(a, np.dot(q, r))
True
>>> q.shape, r.shape
((9, 9), (9, 6))

>>> r2 = linalg.qr(a, mode='r')
>>> np.allclose(r, r2)
True

>>> q3, r3 = linalg.qr(a, mode='economic')
>>> q3.shape, r3.shape
((9, 6), (6, 6))

>>> q4, r4, p4 = linalg.qr(a, pivoting=True)
>>> d = np.abs(np.diag(r4))
>>> np.all(d[1:] <= d[:-1])
True
>>> np.allclose(a[:, p4], np.dot(q4, r4))
True
>>> q4.shape, r4.shape, p4.shape
((9, 9), (9, 6), (6,))

>>> q5, r5, p5 = linalg.qr(a, mode='economic', pivoting=True)
>>> q5.shape, r5.shape, p5.shape
((9, 6), (6, 6), (6,))
*)

val qr_insert : ?which:[`Row | `Col] -> ?rcond:float -> ?overwrite_qru:bool -> ?check_finite:bool -> q:[>`Ndarray] Np.Obj.t -> r:[>`Ndarray] Np.Obj.t -> u:Py.Object.t -> k:int -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
qr_insert(Q, R, u, k, which=u'row', rcond=None, overwrite_qru=False, check_finite=True)

QR update on row or column insertions

If ``A = Q R`` is the QR factorization of ``A``, return the QR
factorization of ``A`` where rows or columns have been inserted starting
at row or column ``k``.

Parameters
----------
Q : (M, M) array_like
    Unitary/orthogonal matrix from the QR decomposition of A.
R : (M, N) array_like
    Upper triangular matrix from the QR decomposition of A.
u : (N,), (p, N), (M,), or (M, p) array_like
    Rows or columns to insert
k : int
    Index before which `u` is to be inserted.
which: {'row', 'col'}, optional
    Determines if rows or columns will be inserted, defaults to 'row'
rcond : float
    Lower bound on the reciprocal condition number of ``Q`` augmented with
    ``u/||u||`` Only used when updating economic mode (thin, (M,N) (N,N))
    decompositions.  If None, machine precision is used.  Defaults to
    None.
overwrite_qru : bool, optional
    If True, consume Q, R, and u, if possible, while performing the update,
    otherwise make copies as necessary. Defaults to False.
check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.
    Default is True.

Returns
-------
Q1 : ndarray
    Updated unitary/orthogonal factor
R1 : ndarray
    Updated upper triangular factor

Raises
------
LinAlgError :
    If updating a (M,N) (N,N) factorization and the reciprocal condition
    number of Q augmented with u/||u|| is smaller than rcond.

See Also
--------
qr, qr_multiply, qr_delete, qr_update

Notes
-----
This routine does not guarantee that the diagonal entries of ``R1`` are
positive.

.. versionadded:: 0.16.0

References
----------

.. [1] Golub, G. H. & Van Loan, C. F. Matrix Computations, 3rd Ed.
       (Johns Hopkins University Press, 1996).

.. [2] Daniel, J. W., Gragg, W. B., Kaufman, L. & Stewart, G. W.
       Reorthogonalization and stable algorithms for updating the
       Gram-Schmidt QR factorization. Math. Comput. 30, 772-795 (1976).

.. [3] Reichel, L. & Gragg, W. B. Algorithm 686: FORTRAN Subroutines for
       Updating the QR Decomposition. ACM Trans. Math. Softw. 16, 369-377
       (1990).

Examples
--------
>>> from scipy import linalg
>>> a = np.array([[  3.,  -2.,  -2.],
...               [  6.,  -7.,   4.],
...               [  7.,   8.,  -6.]])
>>> q, r = linalg.qr(a)

Given this QR decomposition, update q and r when 2 rows are inserted.

>>> u = np.array([[  6.,  -9.,  -3.],
...               [ -3.,  10.,   1.]])
>>> q1, r1 = linalg.qr_insert(q, r, u, 2, 'row')
>>> q1
array([[-0.25445668,  0.02246245,  0.18146236, -0.72798806,  0.60979671],  # may vary (signs)
       [-0.50891336,  0.23226178, -0.82836478, -0.02837033, -0.00828114],
       [-0.50891336,  0.35715302,  0.38937158,  0.58110733,  0.35235345],
       [ 0.25445668, -0.52202743, -0.32165498,  0.36263239,  0.65404509],
       [-0.59373225, -0.73856549,  0.16065817, -0.0063658 , -0.27595554]])
>>> r1
array([[-11.78982612,   6.44623587,   3.81685018],  # may vary (signs)
       [  0.        , -16.01393278,   3.72202865],
       [  0.        ,   0.        ,  -6.13010256],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ]])

The update is equivalent, but faster than the following.

>>> a1 = np.insert(a, 2, u, 0)
>>> a1
array([[  3.,  -2.,  -2.],
       [  6.,  -7.,   4.],
       [  6.,  -9.,  -3.],
       [ -3.,  10.,   1.],
       [  7.,   8.,  -6.]])
>>> q_direct, r_direct = linalg.qr(a1)

Check that we have equivalent results:

>>> np.dot(q1, r1)
array([[  3.,  -2.,  -2.],
       [  6.,  -7.,   4.],
       [  6.,  -9.,  -3.],
       [ -3.,  10.,   1.],
       [  7.,   8.,  -6.]])

>>> np.allclose(np.dot(q1, r1), a1)
True

And the updated Q is still unitary:

>>> np.allclose(np.dot(q1.T, q1), np.eye(5))
True
*)

val qr_multiply : ?mode:[`Left | `Right] -> ?pivoting:bool -> ?conjugate:bool -> ?overwrite_a:bool -> ?overwrite_c:bool -> a:[>`Ndarray] Np.Obj.t -> c:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * Py.Object.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Calculate the QR decomposition and multiply Q with a matrix.

Calculate the decomposition ``A = Q R`` where Q is unitary/orthogonal
and R upper triangular. Multiply Q with a vector or a matrix c.

Parameters
----------
a : (M, N), array_like
    Input array
c : array_like
    Input array to be multiplied by ``q``.
mode : {'left', 'right'}, optional
    ``Q @ c`` is returned if mode is 'left', ``c @ Q`` is returned if
    mode is 'right'.
    The shape of c must be appropriate for the matrix multiplications,
    if mode is 'left', ``min(a.shape) == c.shape[0]``,
    if mode is 'right', ``a.shape[0] == c.shape[1]``.
pivoting : bool, optional
    Whether or not factorization should include pivoting for rank-revealing
    qr decomposition, see the documentation of qr.
conjugate : bool, optional
    Whether Q should be complex-conjugated. This might be faster
    than explicit conjugation.
overwrite_a : bool, optional
    Whether data in a is overwritten (may improve performance)
overwrite_c : bool, optional
    Whether data in c is overwritten (may improve performance).
    If this is used, c must be big enough to keep the result,
    i.e. ``c.shape[0]`` = ``a.shape[0]`` if mode is 'left'.

Returns
-------
CQ : ndarray
    The product of ``Q`` and ``c``.
R : (K, N), ndarray
    R array of the resulting QR factorization where ``K = min(M, N)``.
P : (N,) ndarray
    Integer pivot array. Only returned when ``pivoting=True``.

Raises
------
LinAlgError
    Raised if QR decomposition fails.

Notes
-----
This is an interface to the LAPACK routines ``?GEQRF``, ``?ORMQR``,
``?UNMQR``, and ``?GEQP3``.

.. versionadded:: 0.11.0

Examples
--------
>>> from scipy.linalg import qr_multiply, qr
>>> A = np.array([[1, 3, 3], [2, 3, 2], [2, 3, 3], [1, 3, 2]])
>>> qc, r1, piv1 = qr_multiply(A, 2*np.eye(4), pivoting=1)
>>> qc
array([[-1.,  1., -1.],
       [-1., -1.,  1.],
       [-1., -1., -1.],
       [-1.,  1.,  1.]])
>>> r1
array([[-6., -3., -5.            ],
       [ 0., -1., -1.11022302e-16],
       [ 0.,  0., -1.            ]])
>>> piv1
array([1, 0, 2], dtype=int32)
>>> q2, r2, piv2 = qr(A, mode='economic', pivoting=1)
>>> np.allclose(2*q2 - qc, np.zeros((4, 3)))
True
*)

val qr_update : ?overwrite_qruv:bool -> ?check_finite:bool -> q:Py.Object.t -> r:Py.Object.t -> u:Py.Object.t -> v:Py.Object.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
qr_update(Q, R, u, v, overwrite_qruv=False, check_finite=True)

Rank-k QR update

If ``A = Q R`` is the QR factorization of ``A``, return the QR
factorization of ``A + u v**T`` for real ``A`` or ``A + u v**H``
for complex ``A``.

Parameters
----------
Q : (M, M) or (M, N) array_like
    Unitary/orthogonal matrix from the qr decomposition of A.
R : (M, N) or (N, N) array_like
    Upper triangular matrix from the qr decomposition of A.
u : (M,) or (M, k) array_like
    Left update vector
v : (N,) or (N, k) array_like
    Right update vector
overwrite_qruv : bool, optional
    If True, consume Q, R, u, and v, if possible, while performing the
    update, otherwise make copies as necessary. Defaults to False.
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.
    Default is True.

Returns
-------
Q1 : ndarray
    Updated unitary/orthogonal factor
R1 : ndarray
    Updated upper triangular factor

See Also
--------
qr, qr_multiply, qr_delete, qr_insert

Notes
-----
This routine does not guarantee that the diagonal entries of `R1` are
real or positive.

.. versionadded:: 0.16.0

References
----------
.. [1] Golub, G. H. & Van Loan, C. F. Matrix Computations, 3rd Ed.
       (Johns Hopkins University Press, 1996).

.. [2] Daniel, J. W., Gragg, W. B., Kaufman, L. & Stewart, G. W.
       Reorthogonalization and stable algorithms for updating the
       Gram-Schmidt QR factorization. Math. Comput. 30, 772-795 (1976).

.. [3] Reichel, L. & Gragg, W. B. Algorithm 686: FORTRAN Subroutines for
       Updating the QR Decomposition. ACM Trans. Math. Softw. 16, 369-377
       (1990).

Examples
--------
>>> from scipy import linalg
>>> a = np.array([[  3.,  -2.,  -2.],
...               [  6.,  -9.,  -3.],
...               [ -3.,  10.,   1.],
...               [  6.,  -7.,   4.],
...               [  7.,   8.,  -6.]])
>>> q, r = linalg.qr(a)

Given this q, r decomposition, perform a rank 1 update.

>>> u = np.array([7., -2., 4., 3., 5.])
>>> v = np.array([1., 3., -5.])
>>> q_up, r_up = linalg.qr_update(q, r, u, v, False)
>>> q_up
array([[ 0.54073807,  0.18645997,  0.81707661, -0.02136616,  0.06902409],  # may vary (signs)
       [ 0.21629523, -0.63257324,  0.06567893,  0.34125904, -0.65749222],
       [ 0.05407381,  0.64757787, -0.12781284, -0.20031219, -0.72198188],
       [ 0.48666426, -0.30466718, -0.27487277, -0.77079214,  0.0256951 ],
       [ 0.64888568,  0.23001   , -0.4859845 ,  0.49883891,  0.20253783]])
>>> r_up
array([[ 18.49324201,  24.11691794, -44.98940746],  # may vary (signs)
       [  0.        ,  31.95894662, -27.40998201],
       [  0.        ,   0.        ,  -9.25451794],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ]])

The update is equivalent, but faster than the following.

>>> a_up = a + np.outer(u, v)
>>> q_direct, r_direct = linalg.qr(a_up)

Check that we have equivalent results:

>>> np.allclose(np.dot(q_up, r_up), a_up)
True

And the updated Q is still unitary:

>>> np.allclose(np.dot(q_up.T, q_up), np.eye(5))
True

Updating economic (reduced, thin) decompositions is also possible:

>>> qe, re = linalg.qr(a, mode='economic')
>>> qe_up, re_up = linalg.qr_update(qe, re, u, v, False)
>>> qe_up
array([[ 0.54073807,  0.18645997,  0.81707661],  # may vary (signs)
       [ 0.21629523, -0.63257324,  0.06567893],
       [ 0.05407381,  0.64757787, -0.12781284],
       [ 0.48666426, -0.30466718, -0.27487277],
       [ 0.64888568,  0.23001   , -0.4859845 ]])
>>> re_up
array([[ 18.49324201,  24.11691794, -44.98940746],  # may vary (signs)
       [  0.        ,  31.95894662, -27.40998201],
       [  0.        ,   0.        ,  -9.25451794]])
>>> np.allclose(np.dot(qe_up, re_up), a_up)
True
>>> np.allclose(np.dot(qe_up.T, qe_up), np.eye(3))
True

Similarly to the above, perform a rank 2 update.

>>> u2 = np.array([[ 7., -1,],
...                [-2.,  4.],
...                [ 4.,  2.],
...                [ 3., -6.],
...                [ 5.,  3.]])
>>> v2 = np.array([[ 1., 2.],
...                [ 3., 4.],
...                [-5., 2]])
>>> q_up2, r_up2 = linalg.qr_update(q, r, u2, v2, False)
>>> q_up2
array([[-0.33626508, -0.03477253,  0.61956287, -0.64352987, -0.29618884],  # may vary (signs)
       [-0.50439762,  0.58319694, -0.43010077, -0.33395279,  0.33008064],
       [-0.21016568, -0.63123106,  0.0582249 , -0.13675572,  0.73163206],
       [ 0.12609941,  0.49694436,  0.64590024,  0.31191919,  0.47187344],
       [-0.75659643, -0.11517748,  0.10284903,  0.5986227 , -0.21299983]])
>>> r_up2
array([[-23.79075451, -41.1084062 ,  24.71548348],  # may vary (signs)
       [  0.        , -33.83931057,  11.02226551],
       [  0.        ,   0.        ,  48.91476811],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ]])

This update is also a valid qr decomposition of ``A + U V**T``.

>>> a_up2 = a + np.dot(u2, v2.T)
>>> np.allclose(a_up2, np.dot(q_up2, r_up2))
True
>>> np.allclose(np.dot(q_up2.T, q_up2), np.eye(5))
True
*)

val qz : ?output:[`Real | `Complex] -> ?lwork:int -> ?sort:[`Iuc | `Ouc | `Rhp | `Callable of Py.Object.t | `Lhp] -> ?overwrite_a:bool -> ?overwrite_b:bool -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> b:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
QZ decomposition for generalized eigenvalues of a pair of matrices.

The QZ, or generalized Schur, decomposition for a pair of N x N
nonsymmetric matrices (A,B) is::

    (A,B) = (Q*AA*Z', Q*BB*Z')

where AA, BB is in generalized Schur form if BB is upper-triangular
with non-negative diagonal and AA is upper-triangular, or for real QZ
decomposition (``output='real'``) block upper triangular with 1x1
and 2x2 blocks.  In this case, the 1x1 blocks correspond to real
generalized eigenvalues and 2x2 blocks are 'standardized' by making
the corresponding elements of BB have the form::

    [ a 0 ]
    [ 0 b ]

and the pair of corresponding 2x2 blocks in AA and BB will have a complex
conjugate pair of generalized eigenvalues.  If (``output='complex'``) or
A and B are complex matrices, Z' denotes the conjugate-transpose of Z.
Q and Z are unitary matrices.

Parameters
----------
A : (N, N) array_like
    2d array to decompose
B : (N, N) array_like
    2d array to decompose
output : {'real', 'complex'}, optional
    Construct the real or complex QZ decomposition for real matrices.
    Default is 'real'.
lwork : int, optional
    Work array size.  If None or -1, it is automatically computed.
sort : {None, callable, 'lhp', 'rhp', 'iuc', 'ouc'}, optional
    NOTE: THIS INPUT IS DISABLED FOR NOW. Use ordqz instead.

    Specifies whether the upper eigenvalues should be sorted.  A callable
    may be passed that, given a eigenvalue, returns a boolean denoting
    whether the eigenvalue should be sorted to the top-left (True). For
    real matrix pairs, the sort function takes three real arguments
    (alphar, alphai, beta). The eigenvalue
    ``x = (alphar + alphai*1j)/beta``.  For complex matrix pairs or
    output='complex', the sort function takes two complex arguments
    (alpha, beta). The eigenvalue ``x = (alpha/beta)``.  Alternatively,
    string parameters may be used:

        - 'lhp'   Left-hand plane (x.real < 0.0)
        - 'rhp'   Right-hand plane (x.real > 0.0)
        - 'iuc'   Inside the unit circle (x*x.conjugate() < 1.0)
        - 'ouc'   Outside the unit circle (x*x.conjugate() > 1.0)

    Defaults to None (no sorting).
overwrite_a : bool, optional
    Whether to overwrite data in a (may improve performance)
overwrite_b : bool, optional
    Whether to overwrite data in b (may improve performance)
check_finite : bool, optional
    If true checks the elements of `A` and `B` are finite numbers. If
    false does no checking and passes matrix through to
    underlying algorithm.

Returns
-------
AA : (N, N) ndarray
    Generalized Schur form of A.
BB : (N, N) ndarray
    Generalized Schur form of B.
Q : (N, N) ndarray
    The left Schur vectors.
Z : (N, N) ndarray
    The right Schur vectors.

Notes
-----
Q is transposed versus the equivalent function in Matlab.

.. versionadded:: 0.11.0

Examples
--------
>>> from scipy import linalg
>>> np.random.seed(1234)
>>> A = np.arange(9).reshape((3, 3))
>>> B = np.random.randn(3, 3)

>>> AA, BB, Q, Z = linalg.qz(A, B)
>>> AA
array([[-13.40928183,  -4.62471562,   1.09215523],
       [  0.        ,   0.        ,   1.22805978],
       [  0.        ,   0.        ,   0.31973817]])
>>> BB
array([[ 0.33362547, -1.37393632,  0.02179805],
       [ 0.        ,  1.68144922,  0.74683866],
       [ 0.        ,  0.        ,  0.9258294 ]])
>>> Q
array([[ 0.14134727, -0.97562773,  0.16784365],
       [ 0.49835904, -0.07636948, -0.86360059],
       [ 0.85537081,  0.20571399,  0.47541828]])
>>> Z
array([[-0.24900855, -0.51772687,  0.81850696],
       [-0.79813178,  0.58842606,  0.12938478],
       [-0.54861681, -0.6210585 , -0.55973739]])

See also
--------
ordqz
*)

val rq : ?overwrite_a:bool -> ?lwork:int -> ?mode:[`Full | `R | `Economic] -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> (Py.Object.t * Py.Object.t)
(**
Compute RQ decomposition of a matrix.

Calculate the decomposition ``A = R Q`` where Q is unitary/orthogonal
and R upper triangular.

Parameters
----------
a : (M, N) array_like
    Matrix to be decomposed
overwrite_a : bool, optional
    Whether data in a is overwritten (may improve performance)
lwork : int, optional
    Work array size, lwork >= a.shape[1]. If None or -1, an optimal size
    is computed.
mode : {'full', 'r', 'economic'}, optional
    Determines what information is to be returned: either both Q and R
    ('full', default), only R ('r') or both Q and R but computed in
    economy-size ('economic', see Notes).
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
R : float or complex ndarray
    Of shape (M, N) or (M, K) for ``mode='economic'``.  ``K = min(M, N)``.
Q : float or complex ndarray
    Of shape (N, N) or (K, N) for ``mode='economic'``.  Not returned
    if ``mode='r'``.

Raises
------
LinAlgError
    If decomposition fails.

Notes
-----
This is an interface to the LAPACK routines sgerqf, dgerqf, cgerqf, zgerqf,
sorgrq, dorgrq, cungrq and zungrq.

If ``mode=economic``, the shapes of Q and R are (K, N) and (M, K) instead
of (N,N) and (M,N), with ``K=min(M,N)``.

Examples
--------
>>> from scipy import linalg
>>> a = np.random.randn(6, 9)
>>> r, q = linalg.rq(a)
>>> np.allclose(a, r @ q)
True
>>> r.shape, q.shape
((6, 9), (9, 9))
>>> r2 = linalg.rq(a, mode='r')
>>> np.allclose(r, r2)
True
>>> r3, q3 = linalg.rq(a, mode='economic')
>>> r3.shape, q3.shape
((6, 6), (6, 9))
*)

val rsf2csf : ?check_finite:bool -> t:[>`Ndarray] Np.Obj.t -> z:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Convert real Schur form to complex Schur form.

Convert a quasi-diagonal real-valued Schur form to the upper triangular
complex-valued Schur form.

Parameters
----------
T : (M, M) array_like
    Real Schur form of the original array
Z : (M, M) array_like
    Schur transformation matrix
check_finite : bool, optional
    Whether to check that the input arrays contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
T : (M, M) ndarray
    Complex Schur form of the original array
Z : (M, M) ndarray
    Schur transformation matrix corresponding to the complex form

See Also
--------
schur : Schur decomposition of an array

Examples
--------
>>> from scipy.linalg import schur, rsf2csf
>>> A = np.array([[0, 2, 2], [0, 1, 2], [1, 0, 1]])
>>> T, Z = schur(A)
>>> T
array([[ 2.65896708,  1.42440458, -1.92933439],
       [ 0.        , -0.32948354, -0.49063704],
       [ 0.        ,  1.31178921, -0.32948354]])
>>> Z
array([[0.72711591, -0.60156188, 0.33079564],
       [0.52839428, 0.79801892, 0.28976765],
       [0.43829436, 0.03590414, -0.89811411]])
>>> T2 , Z2 = rsf2csf(T, Z)
>>> T2
array([[2.65896708+0.j, -1.64592781+0.743164187j, -1.21516887+1.00660462j],
       [0.+0.j , -0.32948354+8.02254558e-01j, -0.82115218-2.77555756e-17j],
       [0.+0.j , 0.+0.j, -0.32948354-0.802254558j]])
>>> Z2
array([[0.72711591+0.j,  0.28220393-0.31385693j,  0.51319638-0.17258824j],
       [0.52839428+0.j,  0.24720268+0.41635578j, -0.68079517-0.15118243j],
       [0.43829436+0.j, -0.76618703+0.01873251j, -0.03063006+0.46857912j]])
*)

val schur : ?output:[`Real | `Complex] -> ?lwork:int -> ?overwrite_a:bool -> ?sort:[`Iuc | `Ouc | `Rhp | `Callable of Py.Object.t | `Lhp] -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t * int)
(**
Compute Schur decomposition of a matrix.

The Schur decomposition is::

    A = Z T Z^H

where Z is unitary and T is either upper-triangular, or for real
Schur decomposition (output='real'), quasi-upper triangular.  In
the quasi-triangular form, 2x2 blocks describing complex-valued
eigenvalue pairs may extrude from the diagonal.

Parameters
----------
a : (M, M) array_like
    Matrix to decompose
output : {'real', 'complex'}, optional
    Construct the real or complex Schur decomposition (for real matrices).
lwork : int, optional
    Work array size. If None or -1, it is automatically computed.
overwrite_a : bool, optional
    Whether to overwrite data in a (may improve performance).
sort : {None, callable, 'lhp', 'rhp', 'iuc', 'ouc'}, optional
    Specifies whether the upper eigenvalues should be sorted.  A callable
    may be passed that, given a eigenvalue, returns a boolean denoting
    whether the eigenvalue should be sorted to the top-left (True).
    Alternatively, string parameters may be used::

        'lhp'   Left-hand plane (x.real < 0.0)
        'rhp'   Right-hand plane (x.real > 0.0)
        'iuc'   Inside the unit circle (x*x.conjugate() <= 1.0)
        'ouc'   Outside the unit circle (x*x.conjugate() > 1.0)

    Defaults to None (no sorting).
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
T : (M, M) ndarray
    Schur form of A. It is real-valued for the real Schur decomposition.
Z : (M, M) ndarray
    An unitary Schur transformation matrix for A.
    It is real-valued for the real Schur decomposition.
sdim : int
    If and only if sorting was requested, a third return value will
    contain the number of eigenvalues satisfying the sort condition.

Raises
------
LinAlgError
    Error raised under three conditions:

    1. The algorithm failed due to a failure of the QR algorithm to
       compute all eigenvalues
    2. If eigenvalue sorting was requested, the eigenvalues could not be
       reordered due to a failure to separate eigenvalues, usually because
       of poor conditioning
    3. If eigenvalue sorting was requested, roundoff errors caused the
       leading eigenvalues to no longer satisfy the sorting condition

See also
--------
rsf2csf : Convert real Schur form to complex Schur form

Examples
--------
>>> from scipy.linalg import schur, eigvals
>>> A = np.array([[0, 2, 2], [0, 1, 2], [1, 0, 1]])
>>> T, Z = schur(A)
>>> T
array([[ 2.65896708,  1.42440458, -1.92933439],
       [ 0.        , -0.32948354, -0.49063704],
       [ 0.        ,  1.31178921, -0.32948354]])
>>> Z
array([[0.72711591, -0.60156188, 0.33079564],
       [0.52839428, 0.79801892, 0.28976765],
       [0.43829436, 0.03590414, -0.89811411]])

>>> T2, Z2 = schur(A, output='complex')
>>> T2
array([[ 2.65896708, -1.22839825+1.32378589j,  0.42590089+1.51937378j],
       [ 0.        , -0.32948354+0.80225456j, -0.59877807+0.56192146j],
       [ 0.        ,  0.                    , -0.32948354-0.80225456j]])
>>> eigvals(T2)
array([2.65896708, -0.32948354+0.80225456j, -0.32948354-0.80225456j])

An arbitrary custom eig-sorting condition, having positive imaginary part, 
which is satisfied by only one eigenvalue

>>> T3, Z3, sdim = schur(A, output='complex', sort=lambda x: x.imag > 0)
>>> sdim
1
*)

val signm : ?disp:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * float)
(**
Matrix sign function.

Extension of the scalar sign(x) to matrices.

Parameters
----------
A : (N, N) array_like
    Matrix at which to evaluate the sign function
disp : bool, optional
    Print warning if error in the result is estimated large
    instead of returning estimated error. (Default: True)

Returns
-------
signm : (N, N) ndarray
    Value of the sign function at `A`
errest : float
    (if disp == False)

    1-norm of the estimated error, ||err||_1 / ||A||_1

Examples
--------
>>> from scipy.linalg import signm, eigvals
>>> a = [[1,2,3], [1,2,1], [1,1,1]]
>>> eigvals(a)
array([ 4.12488542+0.j, -0.76155718+0.j,  0.63667176+0.j])
>>> eigvals(signm(a))
array([-1.+0.j,  1.+0.j,  1.+0.j])
*)

val sinhm : [>`Ndarray] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the hyperbolic matrix sine.

This routine uses expm to compute the matrix exponentials.

Parameters
----------
A : (N, N) array_like
    Input array.

Returns
-------
sinhm : (N, N) ndarray
    Hyperbolic matrix sine of `A`

Examples
--------
>>> from scipy.linalg import tanhm, sinhm, coshm
>>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
>>> s = sinhm(a)
>>> s
array([[ 10.57300653,  39.28826594],
       [ 13.09608865,  49.86127247]])

Verify tanhm(a) = sinhm(a).dot(inv(coshm(a)))

>>> t = tanhm(a)
>>> c = coshm(a)
>>> t - s.dot(np.linalg.inv(c))
array([[  2.72004641e-15,   4.55191440e-15],
       [  0.00000000e+00,  -5.55111512e-16]])
*)

val sinm : [>`Ndarray] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the matrix sine.

This routine uses expm to compute the matrix exponentials.

Parameters
----------
A : (N, N) array_like
    Input array.

Returns
-------
sinm : (N, N) ndarray
    Matrix sine of `A`

Examples
--------
>>> from scipy.linalg import expm, sinm, cosm

Euler's identity (exp(i*theta) = cos(theta) + i*sin(theta))
applied to a matrix:

>>> a = np.array([[1.0, 2.0], [-1.0, 3.0]])
>>> expm(1j*a)
array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
       [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
>>> cosm(a) + 1j*sinm(a)
array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
       [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
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

val solve_banded : ?overwrite_ab:bool -> ?overwrite_b:bool -> ?debug:Py.Object.t -> ?check_finite:bool -> l_and_u:Py.Object.t -> ab:Py.Object.t -> b:Py.Object.t -> unit -> Py.Object.t
(**
Solve the equation a x = b for x, assuming a is banded matrix.

The matrix a is stored in `ab` using the matrix diagonal ordered form::

    ab[u + i - j, j] == a[i,j]

Example of `ab` (shape of a is (6,6), `u` =1, `l` =2)::

    *    a01  a12  a23  a34  a45
    a00  a11  a22  a33  a44  a55
    a10  a21  a32  a43  a54   *
    a20  a31  a42  a53   *    *

Parameters
----------
(l, u) : (integer, integer)
    Number of non-zero lower and upper diagonals
ab : (`l` + `u` + 1, M) array_like
    Banded matrix
b : (M,) or (M, K) array_like
    Right-hand side
overwrite_ab : bool, optional
    Discard data in `ab` (may enhance performance)
overwrite_b : bool, optional
    Discard data in `b` (may enhance performance)
check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
x : (M,) or (M, K) ndarray
    The solution to the system a x = b.  Returned shape depends on the
    shape of `b`.

Examples
--------
Solve the banded system a x = b, where::

        [5  2 -1  0  0]       [0]
        [1  4  2 -1  0]       [1]
    a = [0  1  3  2 -1]   b = [2]
        [0  0  1  2  2]       [2]
        [0  0  0  1  1]       [3]

There is one nonzero diagonal below the main diagonal (l = 1), and
two above (u = 2).  The diagonal banded form of the matrix is::

         [*  * -1 -1 -1]
    ab = [*  2  2  2  2]
         [5  4  3  2  1]
         [1  1  1  1  *]

>>> from scipy.linalg import solve_banded
>>> ab = np.array([[0,  0, -1, -1, -1],
...                [0,  2,  2,  2,  2],
...                [5,  4,  3,  2,  1],
...                [1,  1,  1,  1,  0]])
>>> b = np.array([0, 1, 2, 2, 3])
>>> x = solve_banded((1, 2), ab, b)
>>> x
array([-2.37288136,  3.93220339, -4.        ,  4.3559322 , -1.3559322 ])
*)

val solve_circulant : ?singular:string -> ?tol:float -> ?caxis:int -> ?baxis:int -> ?outaxis:int -> c:[>`Ndarray] Np.Obj.t -> b:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Solve C x = b for x, where C is a circulant matrix.

`C` is the circulant matrix associated with the vector `c`.

The system is solved by doing division in Fourier space.  The
calculation is::

    x = ifft(fft(b) / fft(c))

where `fft` and `ifft` are the fast Fourier transform and its inverse,
respectively.  For a large vector `c`, this is *much* faster than
solving the system with the full circulant matrix.

Parameters
----------
c : array_like
    The coefficients of the circulant matrix.
b : array_like
    Right-hand side matrix in ``a x = b``.
singular : str, optional
    This argument controls how a near singular circulant matrix is
    handled.  If `singular` is 'raise' and the circulant matrix is
    near singular, a `LinAlgError` is raised.  If `singular` is
    'lstsq', the least squares solution is returned.  Default is 'raise'.
tol : float, optional
    If any eigenvalue of the circulant matrix has an absolute value
    that is less than or equal to `tol`, the matrix is considered to be
    near singular.  If not given, `tol` is set to::

        tol = abs_eigs.max() * abs_eigs.size * np.finfo(np.float64).eps

    where `abs_eigs` is the array of absolute values of the eigenvalues
    of the circulant matrix.
caxis : int
    When `c` has dimension greater than 1, it is viewed as a collection
    of circulant vectors.  In this case, `caxis` is the axis of `c` that
    holds the vectors of circulant coefficients.
baxis : int
    When `b` has dimension greater than 1, it is viewed as a collection
    of vectors.  In this case, `baxis` is the axis of `b` that holds the
    right-hand side vectors.
outaxis : int
    When `c` or `b` are multidimensional, the value returned by
    `solve_circulant` is multidimensional.  In this case, `outaxis` is
    the axis of the result that holds the solution vectors.

Returns
-------
x : ndarray
    Solution to the system ``C x = b``.

Raises
------
LinAlgError
    If the circulant matrix associated with `c` is near singular.

See Also
--------
circulant : circulant matrix

Notes
-----
For a one-dimensional vector `c` with length `m`, and an array `b`
with shape ``(m, ...)``,

    solve_circulant(c, b)

returns the same result as

    solve(circulant(c), b)

where `solve` and `circulant` are from `scipy.linalg`.

.. versionadded:: 0.16.0

Examples
--------
>>> from scipy.linalg import solve_circulant, solve, circulant, lstsq

>>> c = np.array([2, 2, 4])
>>> b = np.array([1, 2, 3])
>>> solve_circulant(c, b)
array([ 0.75, -0.25,  0.25])

Compare that result to solving the system with `scipy.linalg.solve`:

>>> solve(circulant(c), b)
array([ 0.75, -0.25,  0.25])

A singular example:

>>> c = np.array([1, 1, 0, 0])
>>> b = np.array([1, 2, 3, 4])

Calling ``solve_circulant(c, b)`` will raise a `LinAlgError`.  For the
least square solution, use the option ``singular='lstsq'``:

>>> solve_circulant(c, b, singular='lstsq')
array([ 0.25,  1.25,  2.25,  1.25])

Compare to `scipy.linalg.lstsq`:

>>> x, resid, rnk, s = lstsq(circulant(c), b)
>>> x
array([ 0.25,  1.25,  2.25,  1.25])

A broadcasting example:

Suppose we have the vectors of two circulant matrices stored in an array
with shape (2, 5), and three `b` vectors stored in an array with shape
(3, 5).  For example,

>>> c = np.array([[1.5, 2, 3, 0, 0], [1, 1, 4, 3, 2]])
>>> b = np.arange(15).reshape(-1, 5)

We want to solve all combinations of circulant matrices and `b` vectors,
with the result stored in an array with shape (2, 3, 5).  When we
disregard the axes of `c` and `b` that hold the vectors of coefficients,
the shapes of the collections are (2,) and (3,), respectively, which are
not compatible for broadcasting.  To have a broadcast result with shape
(2, 3), we add a trivial dimension to `c`: ``c[:, np.newaxis, :]`` has
shape (2, 1, 5).  The last dimension holds the coefficients of the
circulant matrices, so when we call `solve_circulant`, we can use the
default ``caxis=-1``.  The coefficients of the `b` vectors are in the last
dimension of the array `b`, so we use ``baxis=-1``.  If we use the
default `outaxis`, the result will have shape (5, 2, 3), so we'll use
``outaxis=-1`` to put the solution vectors in the last dimension.

>>> x = solve_circulant(c[:, np.newaxis, :], b, baxis=-1, outaxis=-1)
>>> x.shape
(2, 3, 5)
>>> np.set_printoptions(precision=3)  # For compact output of numbers.
>>> x
array([[[-0.118,  0.22 ,  1.277, -0.142,  0.302],
        [ 0.651,  0.989,  2.046,  0.627,  1.072],
        [ 1.42 ,  1.758,  2.816,  1.396,  1.841]],
       [[ 0.401,  0.304,  0.694, -0.867,  0.377],
        [ 0.856,  0.758,  1.149, -0.412,  0.831],
        [ 1.31 ,  1.213,  1.603,  0.042,  1.286]]])

Check by solving one pair of `c` and `b` vectors (cf. ``x[1, 1, :]``):

>>> solve_circulant(c[1], b[1, :])
array([ 0.856,  0.758,  1.149, -0.412,  0.831])
*)

val solve_continuous_are : ?e:[>`Ndarray] Np.Obj.t -> ?s:[>`Ndarray] Np.Obj.t -> ?balanced:bool -> a:[>`Ndarray] Np.Obj.t -> b:[>`Ndarray] Np.Obj.t -> q:[>`Ndarray] Np.Obj.t -> r:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Solves the continuous-time algebraic Riccati equation (CARE).

The CARE is defined as

.. math::

      X A + A^H X - X B R^{-1} B^H X + Q = 0

The limitations for a solution to exist are :

    * All eigenvalues of :math:`A` on the right half plane, should be
      controllable.

    * The associated hamiltonian pencil (See Notes), should have
      eigenvalues sufficiently away from the imaginary axis.

Moreover, if ``e`` or ``s`` is not precisely ``None``, then the
generalized version of CARE

.. math::

      E^HXA + A^HXE - (E^HXB + S) R^{-1} (B^HXE + S^H) + Q = 0

is solved. When omitted, ``e`` is assumed to be the identity and ``s``
is assumed to be the zero matrix with sizes compatible with ``a`` and
``b`` respectively.

Parameters
----------
a : (M, M) array_like
    Square matrix
b : (M, N) array_like
    Input
q : (M, M) array_like
    Input
r : (N, N) array_like
    Nonsingular square matrix
e : (M, M) array_like, optional
    Nonsingular square matrix
s : (M, N) array_like, optional
    Input
balanced : bool, optional
    The boolean that indicates whether a balancing step is performed
    on the data. The default is set to True.

Returns
-------
x : (M, M) ndarray
    Solution to the continuous-time algebraic Riccati equation.

Raises
------
LinAlgError
    For cases where the stable subspace of the pencil could not be
    isolated. See Notes section and the references for details.

See Also
--------
solve_discrete_are : Solves the discrete-time algebraic Riccati equation

Notes
-----
The equation is solved by forming the extended hamiltonian matrix pencil,
as described in [1]_, :math:`H - \lambda J` given by the block matrices ::

    [ A    0    B ]             [ E   0    0 ]
    [-Q  -A^H  -S ] - \lambda * [ 0  E^H   0 ]
    [ S^H B^H   R ]             [ 0   0    0 ]

and using a QZ decomposition method.

In this algorithm, the fail conditions are linked to the symmetry
of the product :math:`U_2 U_1^{-1}` and condition number of
:math:`U_1`. Here, :math:`U` is the 2m-by-m matrix that holds the
eigenvectors spanning the stable subspace with 2m rows and partitioned
into two m-row matrices. See [1]_ and [2]_ for more details.

In order to improve the QZ decomposition accuracy, the pencil goes
through a balancing step where the sum of absolute values of
:math:`H` and :math:`J` entries (after removing the diagonal entries of
the sum) is balanced following the recipe given in [3]_.

.. versionadded:: 0.11.0

References
----------
.. [1]  P. van Dooren , 'A Generalized Eigenvalue Approach For Solving
   Riccati Equations.', SIAM Journal on Scientific and Statistical
   Computing, Vol.2(2), DOI: 10.1137/0902010

.. [2] A.J. Laub, 'A Schur Method for Solving Algebraic Riccati
   Equations.', Massachusetts Institute of Technology. Laboratory for
   Information and Decision Systems. LIDS-R ; 859. Available online :
   http://hdl.handle.net/1721.1/1301

.. [3] P. Benner, 'Symplectic Balancing of Hamiltonian Matrices', 2001,
   SIAM J. Sci. Comput., 2001, Vol.22(5), DOI: 10.1137/S1064827500367993

Examples
--------
Given `a`, `b`, `q`, and `r` solve for `x`:

>>> from scipy import linalg
>>> a = np.array([[4, 3], [-4.5, -3.5]])
>>> b = np.array([[1], [-1]])
>>> q = np.array([[9, 6], [6, 4.]])
>>> r = 1
>>> x = linalg.solve_continuous_are(a, b, q, r)
>>> x
array([[ 21.72792206,  14.48528137],
       [ 14.48528137,   9.65685425]])
>>> np.allclose(a.T.dot(x) + x.dot(a)-x.dot(b).dot(b.T).dot(x), -q)
True
*)

val solve_continuous_lyapunov : a:[>`Ndarray] Np.Obj.t -> q:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Solves the continuous Lyapunov equation :math:`AX + XA^H = Q`.

Uses the Bartels-Stewart algorithm to find :math:`X`.

Parameters
----------
a : array_like
    A square matrix

q : array_like
    Right-hand side square matrix

Returns
-------
x : ndarray
    Solution to the continuous Lyapunov equation

See Also
--------
solve_discrete_lyapunov : computes the solution to the discrete-time
    Lyapunov equation
solve_sylvester : computes the solution to the Sylvester equation

Notes
-----
The continuous Lyapunov equation is a special form of the Sylvester
equation, hence this solver relies on LAPACK routine ?TRSYL.

.. versionadded:: 0.11.0

Examples
--------
Given `a` and `q` solve for `x`:

>>> from scipy import linalg
>>> a = np.array([[-3, -2, 0], [-1, -1, 0], [0, -5, -1]])
>>> b = np.array([2, 4, -1])
>>> q = np.eye(3)
>>> x = linalg.solve_continuous_lyapunov(a, q)
>>> x
array([[ -0.75  ,   0.875 ,  -3.75  ],
       [  0.875 ,  -1.375 ,   5.3125],
       [ -3.75  ,   5.3125, -27.0625]])
>>> np.allclose(a.dot(x) + x.dot(a.T), q)
True
*)

val solve_discrete_are : ?e:[>`Ndarray] Np.Obj.t -> ?s:[>`Ndarray] Np.Obj.t -> ?balanced:bool -> a:[>`Ndarray] Np.Obj.t -> b:[>`Ndarray] Np.Obj.t -> q:[>`Ndarray] Np.Obj.t -> r:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Solves the discrete-time algebraic Riccati equation (DARE).

The DARE is defined as

.. math::

      A^HXA - X - (A^HXB) (R + B^HXB)^{-1} (B^HXA) + Q = 0

The limitations for a solution to exist are :

    * All eigenvalues of :math:`A` outside the unit disc, should be
      controllable.

    * The associated symplectic pencil (See Notes), should have
      eigenvalues sufficiently away from the unit circle.

Moreover, if ``e`` and ``s`` are not both precisely ``None``, then the
generalized version of DARE

.. math::

      A^HXA - E^HXE - (A^HXB+S) (R+B^HXB)^{-1} (B^HXA+S^H) + Q = 0

is solved. When omitted, ``e`` is assumed to be the identity and ``s``
is assumed to be the zero matrix.

Parameters
----------
a : (M, M) array_like
    Square matrix
b : (M, N) array_like
    Input
q : (M, M) array_like
    Input
r : (N, N) array_like
    Square matrix
e : (M, M) array_like, optional
    Nonsingular square matrix
s : (M, N) array_like, optional
    Input
balanced : bool
    The boolean that indicates whether a balancing step is performed
    on the data. The default is set to True.

Returns
-------
x : (M, M) ndarray
    Solution to the discrete algebraic Riccati equation.

Raises
------
LinAlgError
    For cases where the stable subspace of the pencil could not be
    isolated. See Notes section and the references for details.

See Also
--------
solve_continuous_are : Solves the continuous algebraic Riccati equation

Notes
-----
The equation is solved by forming the extended symplectic matrix pencil,
as described in [1]_, :math:`H - \lambda J` given by the block matrices ::

       [  A   0   B ]             [ E   0   B ]
       [ -Q  E^H -S ] - \lambda * [ 0  A^H  0 ]
       [ S^H  0   R ]             [ 0 -B^H  0 ]

and using a QZ decomposition method.

In this algorithm, the fail conditions are linked to the symmetry
of the product :math:`U_2 U_1^{-1}` and condition number of
:math:`U_1`. Here, :math:`U` is the 2m-by-m matrix that holds the
eigenvectors spanning the stable subspace with 2m rows and partitioned
into two m-row matrices. See [1]_ and [2]_ for more details.

In order to improve the QZ decomposition accuracy, the pencil goes
through a balancing step where the sum of absolute values of
:math:`H` and :math:`J` rows/cols (after removing the diagonal entries)
is balanced following the recipe given in [3]_. If the data has small
numerical noise, balancing may amplify their effects and some clean up
is required.

.. versionadded:: 0.11.0

References
----------
.. [1]  P. van Dooren , 'A Generalized Eigenvalue Approach For Solving
   Riccati Equations.', SIAM Journal on Scientific and Statistical
   Computing, Vol.2(2), DOI: 10.1137/0902010

.. [2] A.J. Laub, 'A Schur Method for Solving Algebraic Riccati
   Equations.', Massachusetts Institute of Technology. Laboratory for
   Information and Decision Systems. LIDS-R ; 859. Available online :
   http://hdl.handle.net/1721.1/1301

.. [3] P. Benner, 'Symplectic Balancing of Hamiltonian Matrices', 2001,
   SIAM J. Sci. Comput., 2001, Vol.22(5), DOI: 10.1137/S1064827500367993

Examples
--------
Given `a`, `b`, `q`, and `r` solve for `x`:

>>> from scipy import linalg as la
>>> a = np.array([[0, 1], [0, -1]])
>>> b = np.array([[1, 0], [2, 1]])
>>> q = np.array([[-4, -4], [-4, 7]])
>>> r = np.array([[9, 3], [3, 1]])
>>> x = la.solve_discrete_are(a, b, q, r)
>>> x
array([[-4., -4.],
       [-4.,  7.]])
>>> R = la.solve(r + b.T.dot(x).dot(b), b.T.dot(x).dot(a))
>>> np.allclose(a.T.dot(x).dot(a) - x - a.T.dot(x).dot(b).dot(R), -q)
True
*)

val solve_discrete_lyapunov : ?method_:[`Bilinear | `Direct] -> a:Py.Object.t -> q:Py.Object.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Solves the discrete Lyapunov equation :math:`AXA^H - X + Q = 0`.

Parameters
----------
a, q : (M, M) array_like
    Square matrices corresponding to A and Q in the equation
    above respectively. Must have the same shape.

method : {'direct', 'bilinear'}, optional
    Type of solver.

    If not given, chosen to be ``direct`` if ``M`` is less than 10 and
    ``bilinear`` otherwise.

Returns
-------
x : ndarray
    Solution to the discrete Lyapunov equation

See Also
--------
solve_continuous_lyapunov : computes the solution to the continuous-time
    Lyapunov equation

Notes
-----
This section describes the available solvers that can be selected by the
'method' parameter. The default method is *direct* if ``M`` is less than 10
and ``bilinear`` otherwise.

Method *direct* uses a direct analytical solution to the discrete Lyapunov
equation. The algorithm is given in, for example, [1]_. However it requires
the linear solution of a system with dimension :math:`M^2` so that
performance degrades rapidly for even moderately sized matrices.

Method *bilinear* uses a bilinear transformation to convert the discrete
Lyapunov equation to a continuous Lyapunov equation :math:`(BX+XB'=-C)`
where :math:`B=(A-I)(A+I)^{-1}` and
:math:`C=2(A' + I)^{-1} Q (A + I)^{-1}`. The continuous equation can be
efficiently solved since it is a special case of a Sylvester equation.
The transformation algorithm is from Popov (1964) as described in [2]_.

.. versionadded:: 0.11.0

References
----------
.. [1] Hamilton, James D. Time Series Analysis, Princeton: Princeton
   University Press, 1994.  265.  Print.
   http://doc1.lbfl.li/aca/FLMF037168.pdf
.. [2] Gajic, Z., and M.T.J. Qureshi. 2008.
   Lyapunov Matrix Equation in System Stability and Control.
   Dover Books on Engineering Series. Dover Publications.

Examples
--------
Given `a` and `q` solve for `x`:

>>> from scipy import linalg
>>> a = np.array([[0.2, 0.5],[0.7, -0.9]])
>>> q = np.eye(2)
>>> x = linalg.solve_discrete_lyapunov(a, q)
>>> x
array([[ 0.70872893,  1.43518822],
       [ 1.43518822, -2.4266315 ]])
>>> np.allclose(a.dot(x).dot(a.T)-x, -q)
True
*)

val solve_lyapunov : a:[>`Ndarray] Np.Obj.t -> q:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Solves the continuous Lyapunov equation :math:`AX + XA^H = Q`.

Uses the Bartels-Stewart algorithm to find :math:`X`.

Parameters
----------
a : array_like
    A square matrix

q : array_like
    Right-hand side square matrix

Returns
-------
x : ndarray
    Solution to the continuous Lyapunov equation

See Also
--------
solve_discrete_lyapunov : computes the solution to the discrete-time
    Lyapunov equation
solve_sylvester : computes the solution to the Sylvester equation

Notes
-----
The continuous Lyapunov equation is a special form of the Sylvester
equation, hence this solver relies on LAPACK routine ?TRSYL.

.. versionadded:: 0.11.0

Examples
--------
Given `a` and `q` solve for `x`:

>>> from scipy import linalg
>>> a = np.array([[-3, -2, 0], [-1, -1, 0], [0, -5, -1]])
>>> b = np.array([2, 4, -1])
>>> q = np.eye(3)
>>> x = linalg.solve_continuous_lyapunov(a, q)
>>> x
array([[ -0.75  ,   0.875 ,  -3.75  ],
       [  0.875 ,  -1.375 ,   5.3125],
       [ -3.75  ,   5.3125, -27.0625]])
>>> np.allclose(a.dot(x) + x.dot(a.T), q)
True
*)

val solve_sylvester : a:[>`Ndarray] Np.Obj.t -> b:[>`Ndarray] Np.Obj.t -> q:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Computes a solution (X) to the Sylvester equation :math:`AX + XB = Q`.

Parameters
----------
a : (M, M) array_like
    Leading matrix of the Sylvester equation
b : (N, N) array_like
    Trailing matrix of the Sylvester equation
q : (M, N) array_like
    Right-hand side

Returns
-------
x : (M, N) ndarray
    The solution to the Sylvester equation.

Raises
------
LinAlgError
    If solution was not found

Notes
-----
Computes a solution to the Sylvester matrix equation via the Bartels-
Stewart algorithm.  The A and B matrices first undergo Schur
decompositions.  The resulting matrices are used to construct an
alternative Sylvester equation (``RY + YS^T = F``) where the R and S
matrices are in quasi-triangular form (or, when R, S or F are complex,
triangular form).  The simplified equation is then solved using
``*TRSYL`` from LAPACK directly.

.. versionadded:: 0.11.0

Examples
--------
Given `a`, `b`, and `q` solve for `x`:

>>> from scipy import linalg
>>> a = np.array([[-3, -2, 0], [-1, -1, 3], [3, -5, -1]])
>>> b = np.array([[1]])
>>> q = np.array([[1],[2],[3]])
>>> x = linalg.solve_sylvester(a, b, q)
>>> x
array([[ 0.0625],
       [-0.5625],
       [ 0.6875]])
>>> np.allclose(a.dot(x) + x.dot(b), q)
True
*)

val solve_toeplitz : ?check_finite:bool -> c_or_cr:[`Ndarray of [>`Ndarray] Np.Obj.t | `Tuple_of_array_like_array_like_ of Py.Object.t] -> b:Py.Object.t -> unit -> Py.Object.t
(**
Solve a Toeplitz system using Levinson Recursion

The Toeplitz matrix has constant diagonals, with c as its first column
and r as its first row.  If r is not given, ``r == conjugate(c)`` is
assumed.

Parameters
----------
c_or_cr : array_like or tuple of (array_like, array_like)
    The vector ``c``, or a tuple of arrays (``c``, ``r``). Whatever the
    actual shape of ``c``, it will be converted to a 1-D array. If not
    supplied, ``r = conjugate(c)`` is assumed; in this case, if c[0] is
    real, the Toeplitz matrix is Hermitian. r[0] is ignored; the first row
    of the Toeplitz matrix is ``[c[0], r[1:]]``.  Whatever the actual shape
    of ``r``, it will be converted to a 1-D array.
b : (M,) or (M, K) array_like
    Right-hand side in ``T x = b``.
check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (result entirely NaNs) if the inputs do contain infinities or NaNs.

Returns
-------
x : (M,) or (M, K) ndarray
    The solution to the system ``T x = b``.  Shape of return matches shape
    of `b`.

See Also
--------
toeplitz : Toeplitz matrix

Notes
-----
The solution is computed using Levinson-Durbin recursion, which is faster
than generic least-squares methods, but can be less numerically stable.

Examples
--------
Solve the Toeplitz system T x = b, where::

        [ 1 -1 -2 -3]       [1]
    T = [ 3  1 -1 -2]   b = [2]
        [ 6  3  1 -1]       [2]
        [10  6  3  1]       [5]

To specify the Toeplitz matrix, only the first column and the first
row are needed.

>>> c = np.array([1, 3, 6, 10])    # First column of T
>>> r = np.array([1, -1, -2, -3])  # First row of T
>>> b = np.array([1, 2, 2, 5])

>>> from scipy.linalg import solve_toeplitz, toeplitz
>>> x = solve_toeplitz((c, r), b)
>>> x
array([ 1.66666667, -1.        , -2.66666667,  2.33333333])

Check the result by creating the full Toeplitz matrix and
multiplying it by `x`.  We should get `b`.

>>> T = toeplitz(c, r)
>>> T.dot(x)
array([ 1.,  2.,  2.,  5.])
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

val solveh_banded : ?overwrite_ab:bool -> ?overwrite_b:bool -> ?lower:bool -> ?check_finite:bool -> ab:Py.Object.t -> b:Py.Object.t -> unit -> Py.Object.t
(**
Solve equation a x = b. a is Hermitian positive-definite banded matrix.

The matrix a is stored in `ab` either in lower diagonal or upper
diagonal ordered form:

    ab[u + i - j, j] == a[i,j]        (if upper form; i <= j)
    ab[    i - j, j] == a[i,j]        (if lower form; i >= j)

Example of `ab` (shape of a is (6, 6), `u` =2)::

    upper form:
    *   *   a02 a13 a24 a35
    *   a01 a12 a23 a34 a45
    a00 a11 a22 a33 a44 a55

    lower form:
    a00 a11 a22 a33 a44 a55
    a10 a21 a32 a43 a54 *
    a20 a31 a42 a53 *   *

Cells marked with * are not used.

Parameters
----------
ab : (`u` + 1, M) array_like
    Banded matrix
b : (M,) or (M, K) array_like
    Right-hand side
overwrite_ab : bool, optional
    Discard data in `ab` (may enhance performance)
overwrite_b : bool, optional
    Discard data in `b` (may enhance performance)
lower : bool, optional
    Is the matrix in the lower form. (Default is upper form)
check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
x : (M,) or (M, K) ndarray
    The solution to the system a x = b.  Shape of return matches shape
    of `b`.

Examples
--------
Solve the banded system A x = b, where::

        [ 4  2 -1  0  0  0]       [1]
        [ 2  5  2 -1  0  0]       [2]
    A = [-1  2  6  2 -1  0]   b = [2]
        [ 0 -1  2  7  2 -1]       [3]
        [ 0  0 -1  2  8  2]       [3]
        [ 0  0  0 -1  2  9]       [3]

>>> from scipy.linalg import solveh_banded

`ab` contains the main diagonal and the nonzero diagonals below the
main diagonal.  That is, we use the lower form:

>>> ab = np.array([[ 4,  5,  6,  7, 8, 9],
...                [ 2,  2,  2,  2, 2, 0],
...                [-1, -1, -1, -1, 0, 0]])
>>> b = np.array([1, 2, 2, 3, 3, 3])
>>> x = solveh_banded(ab, b, lower=True)
>>> x
array([ 0.03431373,  0.45938375,  0.05602241,  0.47759104,  0.17577031,
        0.34733894])


Solve the Hermitian banded system H x = b, where::

        [ 8   2-1j   0     0  ]        [ 1  ]
    H = [2+1j  5     1j    0  ]    b = [1+1j]
        [ 0   -1j    9   -2-1j]        [1-2j]
        [ 0    0   -2+1j   6  ]        [ 0  ]

In this example, we put the upper diagonals in the array `hb`:

>>> hb = np.array([[0, 2-1j, 1j, -2-1j],
...                [8,  5,    9,   6  ]])
>>> b = np.array([1, 1+1j, 1-2j, 0])
>>> x = solveh_banded(hb, b)
>>> x
array([ 0.07318536-0.02939412j,  0.11877624+0.17696461j,
        0.10077984-0.23035393j, -0.00479904-0.09358128j])
*)

val sqrtm : ?disp:bool -> ?blocksize:int -> a:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * float)
(**
Matrix square root.

Parameters
----------
A : (N, N) array_like
    Matrix whose square root to evaluate
disp : bool, optional
    Print warning if error in the result is estimated large
    instead of returning estimated error. (Default: True)
blocksize : integer, optional
    If the blocksize is not degenerate with respect to the
    size of the input array, then use a blocked algorithm. (Default: 64)

Returns
-------
sqrtm : (N, N) ndarray
    Value of the sqrt function at `A`

errest : float
    (if disp == False)

    Frobenius norm of the estimated error, ||err||_F / ||A||_F

References
----------
.. [1] Edvin Deadman, Nicholas J. Higham, Rui Ralha (2013)
       'Blocked Schur Algorithms for Computing the Matrix Square Root,
       Lecture Notes in Computer Science, 7782. pp. 171-182.

Examples
--------
>>> from scipy.linalg import sqrtm
>>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
>>> r = sqrtm(a)
>>> r
array([[ 0.75592895,  1.13389342],
       [ 0.37796447,  1.88982237]])
>>> r.dot(r)
array([[ 1.,  3.],
       [ 1.,  4.]])
*)

val subspace_angles : a:[>`Ndarray] Np.Obj.t -> b:Py.Object.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the subspace angles between two matrices.

Parameters
----------
A : (M, N) array_like
    The first input array.
B : (M, K) array_like
    The second input array.

Returns
-------
angles : ndarray, shape (min(N, K),)
    The subspace angles between the column spaces of `A` and `B` in
    descending order.

See Also
--------
orth
svd

Notes
-----
This computes the subspace angles according to the formula
provided in [1]_. For equivalence with MATLAB and Octave behavior,
use ``angles[0]``.

.. versionadded:: 1.0

References
----------
.. [1] Knyazev A, Argentati M (2002) Principal Angles between Subspaces
       in an A-Based Scalar Product: Algorithms and Perturbation
       Estimates. SIAM J. Sci. Comput. 23:2008-2040.

Examples
--------
A Hadamard matrix, which has orthogonal columns, so we expect that
the suspace angle to be :math:`\frac{\pi}{2}`:

>>> from scipy.linalg import hadamard, subspace_angles
>>> H = hadamard(4)
>>> print(H)
[[ 1  1  1  1]
 [ 1 -1  1 -1]
 [ 1  1 -1 -1]
 [ 1 -1 -1  1]]
>>> np.rad2deg(subspace_angles(H[:, :2], H[:, 2:]))
array([ 90.,  90.])

And the subspace angle of a matrix to itself should be zero:

>>> subspace_angles(H[:, :2], H[:, :2]) <= 2 * np.finfo(float).eps
array([ True,  True], dtype=bool)

The angles between non-orthogonal subspaces are in between these extremes:

>>> x = np.random.RandomState(0).randn(4, 3)
>>> np.rad2deg(subspace_angles(x[:, :2], x[:, [2]]))
array([ 55.832])
*)

val svd : ?full_matrices:bool -> ?compute_uv:bool -> ?overwrite_a:bool -> ?check_finite:bool -> ?lapack_driver:[`Gesdd | `Gesvd] -> a:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
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

val svdvals : ?overwrite_a:bool -> ?check_finite:bool -> a:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Compute singular values of a matrix.

Parameters
----------
a : (M, N) array_like
    Matrix to decompose.
overwrite_a : bool, optional
    Whether to overwrite `a`; may improve performance.
    Default is False.
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
s : (min(M, N),) ndarray
    The singular values, sorted in decreasing order.

Raises
------
LinAlgError
    If SVD computation does not converge.

Notes
-----
``svdvals(a)`` only differs from ``svd(a, compute_uv=False)`` by its
handling of the edge case of empty ``a``, where it returns an
empty sequence:

>>> a = np.empty((0, 2))
>>> from scipy.linalg import svdvals
>>> svdvals(a)
array([], dtype=float64)

See Also
--------
svd : Compute the full singular value decomposition of a matrix.
diagsvd : Construct the Sigma matrix, given the vector s.

Examples
--------
>>> from scipy.linalg import svdvals
>>> m = np.array([[1.0, 0.0],
...               [2.0, 3.0],
...               [1.0, 1.0],
...               [0.0, 2.0],
...               [1.0, 0.0]])
>>> svdvals(m)
array([ 4.28091555,  1.63516424])

We can verify the maximum singular value of `m` by computing the maximum
length of `m.dot(u)` over all the unit vectors `u` in the (x,y) plane.
We approximate 'all' the unit vectors with a large sample.  Because
of linearity, we only need the unit vectors with angles in [0, pi].

>>> t = np.linspace(0, np.pi, 2000)
>>> u = np.array([np.cos(t), np.sin(t)])
>>> np.linalg.norm(m.dot(u), axis=0).max()
4.2809152422538475

`p` is a projection matrix with rank 1.  With exact arithmetic,
its singular values would be [1, 0, 0, 0].

>>> v = np.array([0.1, 0.3, 0.9, 0.3])
>>> p = np.outer(v, v)
>>> svdvals(p)
array([  1.00000000e+00,   2.02021698e-17,   1.56692500e-17,
         8.15115104e-34])

The singular values of an orthogonal matrix are all 1.  Here we
create a random orthogonal matrix by using the `rvs()` method of
`scipy.stats.ortho_group`.

>>> from scipy.stats import ortho_group
>>> np.random.seed(123)
>>> orth = ortho_group.rvs(4)
>>> svdvals(orth)
array([ 1.,  1.,  1.,  1.])
*)

val tanhm : [>`Ndarray] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the hyperbolic matrix tangent.

This routine uses expm to compute the matrix exponentials.

Parameters
----------
A : (N, N) array_like
    Input array

Returns
-------
tanhm : (N, N) ndarray
    Hyperbolic matrix tangent of `A`

Examples
--------
>>> from scipy.linalg import tanhm, sinhm, coshm
>>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
>>> t = tanhm(a)
>>> t
array([[ 0.3428582 ,  0.51987926],
       [ 0.17329309,  0.86273746]])

Verify tanhm(a) = sinhm(a).dot(inv(coshm(a)))

>>> s = sinhm(a)
>>> c = coshm(a)
>>> t - s.dot(np.linalg.inv(c))
array([[  2.72004641e-15,   4.55191440e-15],
       [  0.00000000e+00,  -5.55111512e-16]])
*)

val tanm : [>`Ndarray] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the matrix tangent.

This routine uses expm to compute the matrix exponentials.

Parameters
----------
A : (N, N) array_like
    Input array.

Returns
-------
tanm : (N, N) ndarray
    Matrix tangent of `A`

Examples
--------
>>> from scipy.linalg import tanm, sinm, cosm
>>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
>>> t = tanm(a)
>>> t
array([[ -2.00876993,  -8.41880636],
       [ -2.80626879, -10.42757629]])

Verify tanm(a) = sinm(a).dot(inv(cosm(a)))

>>> s = sinm(a)
>>> c = cosm(a)
>>> s.dot(np.linalg.inv(c))
array([[ -2.00876993,  -8.41880636],
       [ -2.80626879, -10.42757629]])
*)

val toeplitz : ?r:[>`Ndarray] Np.Obj.t -> c:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Construct a Toeplitz matrix.

The Toeplitz matrix has constant diagonals, with c as its first column
and r as its first row.  If r is not given, ``r == conjugate(c)`` is
assumed.

Parameters
----------
c : array_like
    First column of the matrix.  Whatever the actual shape of `c`, it
    will be converted to a 1-D array.
r : array_like, optional
    First row of the matrix. If None, ``r = conjugate(c)`` is assumed;
    in this case, if c[0] is real, the result is a Hermitian matrix.
    r[0] is ignored; the first row of the returned matrix is
    ``[c[0], r[1:]]``.  Whatever the actual shape of `r`, it will be
    converted to a 1-D array.

Returns
-------
A : (len(c), len(r)) ndarray
    The Toeplitz matrix. Dtype is the same as ``(c[0] + r[0]).dtype``.

See Also
--------
circulant : circulant matrix
hankel : Hankel matrix
solve_toeplitz : Solve a Toeplitz system.

Notes
-----
The behavior when `c` or `r` is a scalar, or when `c` is complex and
`r` is None, was changed in version 0.8.0.  The behavior in previous
versions was undocumented and is no longer supported.

Examples
--------
>>> from scipy.linalg import toeplitz
>>> toeplitz([1,2,3], [1,4,5,6])
array([[1, 4, 5, 6],
       [2, 1, 4, 5],
       [3, 2, 1, 4]])
>>> toeplitz([1.0, 2+3j, 4-1j])
array([[ 1.+0.j,  2.-3.j,  4.+1.j],
       [ 2.+3.j,  1.+0.j,  2.-3.j],
       [ 4.-1.j,  2.+3.j,  1.+0.j]])
*)

val tri : ?m:int -> ?k:int -> ?dtype:Np.Dtype.t -> n:int -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Construct (N, M) matrix filled with ones at and below the k-th diagonal.

The matrix has A[i,j] == 1 for i <= j + k

Parameters
----------
N : int
    The size of the first dimension of the matrix.
M : int or None, optional
    The size of the second dimension of the matrix. If `M` is None,
    `M = N` is assumed.
k : int, optional
    Number of subdiagonal below which matrix is filled with ones.
    `k` = 0 is the main diagonal, `k` < 0 subdiagonal and `k` > 0
    superdiagonal.
dtype : dtype, optional
    Data type of the matrix.

Returns
-------
tri : (N, M) ndarray
    Tri matrix.

Examples
--------
>>> from scipy.linalg import tri
>>> tri(3, 5, 2, dtype=int)
array([[1, 1, 1, 0, 0],
       [1, 1, 1, 1, 0],
       [1, 1, 1, 1, 1]])
>>> tri(3, 5, -1, dtype=int)
array([[0, 0, 0, 0, 0],
       [1, 0, 0, 0, 0],
       [1, 1, 0, 0, 0]])
*)

val tril : ?k:int -> m:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Make a copy of a matrix with elements above the k-th diagonal zeroed.

Parameters
----------
m : array_like
    Matrix whose elements to return
k : int, optional
    Diagonal above which to zero elements.
    `k` == 0 is the main diagonal, `k` < 0 subdiagonal and
    `k` > 0 superdiagonal.

Returns
-------
tril : ndarray
    Return is the same shape and type as `m`.

Examples
--------
>>> from scipy.linalg import tril
>>> tril([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)
array([[ 0,  0,  0],
       [ 4,  0,  0],
       [ 7,  8,  0],
       [10, 11, 12]])
*)

val triu : ?k:int -> m:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Make a copy of a matrix with elements below the k-th diagonal zeroed.

Parameters
----------
m : array_like
    Matrix whose elements to return
k : int, optional
    Diagonal below which to zero elements.
    `k` == 0 is the main diagonal, `k` < 0 subdiagonal and
    `k` > 0 superdiagonal.

Returns
-------
triu : ndarray
    Return matrix with zeroed elements below the k-th diagonal and has
    same shape and type as `m`.

Examples
--------
>>> from scipy.linalg import triu
>>> triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)
array([[ 1,  2,  3],
       [ 4,  5,  6],
       [ 0,  8,  9],
       [ 0,  0, 12]])
*)

