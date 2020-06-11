(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module Basic : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val fft : ?n:int -> ?axis:int -> ?overwrite_x:bool -> x:[>`Ndarray] Np.Obj.t -> unit -> (Py.Object.t * Py.Object.t)
(**
Return discrete Fourier transform of real or complex sequence.

The returned complex array contains ``y(0), y(1),..., y(n-1)`` where

``y(j) = (x * exp(-2*pi*sqrt(-1)*j*np.arange(n)/n)).sum()``.

Parameters
----------
x : array_like
    Array to Fourier transform.
n : int, optional
    Length of the Fourier transform.  If ``n < x.shape[axis]``, `x` is
    truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
    default results in ``n = x.shape[axis]``.
axis : int, optional
    Axis along which the fft's are computed; the default is over the
    last axis (i.e., ``axis=-1``).
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.

Returns
-------
z : complex ndarray
    with the elements::

        [y(0),y(1),..,y(n/2),y(1-n/2),...,y(-1)]        if n is even
        [y(0),y(1),..,y((n-1)/2),y(-(n-1)/2),...,y(-1)]  if n is odd

    where::

        y(j) = sum[k=0..n-1] x[k] * exp(-sqrt(-1)*j*k* 2*pi/n), j = 0..n-1

See Also
--------
ifft : Inverse FFT
rfft : FFT of a real sequence

Notes
-----
The packing of the result is 'standard': If ``A = fft(a, n)``, then
``A[0]`` contains the zero-frequency term, ``A[1:n/2]`` contains the
positive-frequency terms, and ``A[n/2:]`` contains the negative-frequency
terms, in order of decreasingly negative frequency. So for an 8-point
transform, the frequencies of the result are [0, 1, 2, 3, -4, -3, -2, -1].
To rearrange the fft output so that the zero-frequency component is
centered, like [-4, -3, -2, -1,  0,  1,  2,  3], use `fftshift`.

Both single and double precision routines are implemented.  Half precision
inputs will be converted to single precision.  Non floating-point inputs
will be converted to double precision.  Long-double precision inputs are
not supported.

This function is most efficient when `n` is a power of two, and least
efficient when `n` is prime.

Note that if ``x`` is real-valued then ``A[j] == A[n-j].conjugate()``.
If ``x`` is real-valued and ``n`` is even then ``A[n/2]`` is real.

If the data type of `x` is real, a 'real FFT' algorithm is automatically
used, which roughly halves the computation time.  To increase efficiency
a little further, use `rfft`, which does the same calculation, but only
outputs half of the symmetrical spectrum.  If the data is both real and
symmetrical, the `dct` can again double the efficiency, by generating
half of the spectrum from half of the signal.

Examples
--------
>>> from scipy.fftpack import fft, ifft
>>> x = np.arange(5)
>>> np.allclose(fft(ifft(x)), x, atol=1e-15)  # within numerical accuracy.
True
*)

val fft2 : ?shape:Py.Object.t -> ?axes:Py.Object.t -> ?overwrite_x:Py.Object.t -> x:Py.Object.t -> unit -> Py.Object.t
(**
2-D discrete Fourier transform.

Return the two-dimensional discrete Fourier transform of the 2-D argument
`x`.

See Also
--------
fftn : for detailed information.
*)

val fftn : ?shape:[`I of int | `Array_like_of_ints of Py.Object.t] -> ?axes:[`I of int | `Array_like_of_ints of Py.Object.t] -> ?overwrite_x:bool -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Return multidimensional discrete Fourier transform.

The returned array contains::

  y[j_1,..,j_d] = sum[k_1=0..n_1-1, ..., k_d=0..n_d-1]
     x[k_1,..,k_d] * prod[i=1..d] exp(-sqrt(-1)*2*pi/n_i * j_i * k_i)

where d = len(x.shape) and n = x.shape.

Parameters
----------
x : array_like
    The (n-dimensional) array to transform.
shape : int or array_like of ints or None, optional
    The shape of the result.  If both `shape` and `axes` (see below) are
    None, `shape` is ``x.shape``; if `shape` is None but `axes` is
    not None, then `shape` is ``scipy.take(x.shape, axes, axis=0)``.
    If ``shape[i] > x.shape[i]``, the i-th dimension is padded with zeros.
    If ``shape[i] < x.shape[i]``, the i-th dimension is truncated to
    length ``shape[i]``.
    If any element of `shape` is -1, the size of the corresponding
    dimension of `x` is used.
axes : int or array_like of ints or None, optional
    The axes of `x` (`y` if `shape` is not None) along which the
    transform is applied.
    The default is over all axes.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed.  Default is False.

Returns
-------
y : complex-valued n-dimensional numpy array
    The (n-dimensional) DFT of the input array.

See Also
--------
ifftn

Notes
-----
If ``x`` is real-valued, then
``y[..., j_i, ...] == y[..., n_i-j_i, ...].conjugate()``.

Both single and double precision routines are implemented.  Half precision
inputs will be converted to single precision.  Non floating-point inputs
will be converted to double precision.  Long-double precision inputs are
not supported.

Examples
--------
>>> from scipy.fftpack import fftn, ifftn
>>> y = (-np.arange(16), 8 - np.arange(16), np.arange(16))
>>> np.allclose(y, fftn(ifftn(y)))
True
*)

val ifft : ?n:int -> ?axis:int -> ?overwrite_x:bool -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return discrete inverse Fourier transform of real or complex sequence.

The returned complex array contains ``y(0), y(1),..., y(n-1)`` where

``y(j) = (x * exp(2*pi*sqrt(-1)*j*np.arange(n)/n)).mean()``.

Parameters
----------
x : array_like
    Transformed data to invert.
n : int, optional
    Length of the inverse Fourier transform.  If ``n < x.shape[axis]``,
    `x` is truncated.  If ``n > x.shape[axis]``, `x` is zero-padded.
    The default results in ``n = x.shape[axis]``.
axis : int, optional
    Axis along which the ifft's are computed; the default is over the
    last axis (i.e., ``axis=-1``).
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.

Returns
-------
ifft : ndarray of floats
    The inverse discrete Fourier transform.

See Also
--------
fft : Forward FFT

Notes
-----
Both single and double precision routines are implemented.  Half precision
inputs will be converted to single precision.  Non floating-point inputs
will be converted to double precision.  Long-double precision inputs are
not supported.

This function is most efficient when `n` is a power of two, and least
efficient when `n` is prime.

If the data type of `x` is real, a 'real IFFT' algorithm is automatically
used, which roughly halves the computation time.

Examples
--------
>>> from scipy.fftpack import fft, ifft
>>> import numpy as np
>>> x = np.arange(5)
>>> np.allclose(ifft(fft(x)), x, atol=1e-15)  # within numerical accuracy.
True
*)

val ifft2 : ?shape:Py.Object.t -> ?axes:Py.Object.t -> ?overwrite_x:Py.Object.t -> x:Py.Object.t -> unit -> Py.Object.t
(**
2-D discrete inverse Fourier transform of real or complex sequence.

Return inverse two-dimensional discrete Fourier transform of
arbitrary type sequence x.

See `ifft` for more information.

See also
--------
fft2, ifft
*)

val ifftn : ?shape:Py.Object.t -> ?axes:Py.Object.t -> ?overwrite_x:Py.Object.t -> x:Py.Object.t -> unit -> Py.Object.t
(**
Return inverse multi-dimensional discrete Fourier transform.

The sequence can be of an arbitrary type.

The returned array contains::

  y[j_1,..,j_d] = 1/p * sum[k_1=0..n_1-1, ..., k_d=0..n_d-1]
     x[k_1,..,k_d] * prod[i=1..d] exp(sqrt(-1)*2*pi/n_i * j_i * k_i)

where ``d = len(x.shape)``, ``n = x.shape``, and ``p = prod[i=1..d] n_i``.

For description of parameters see `fftn`.

See Also
--------
fftn : for detailed information.

Examples
--------
>>> from scipy.fftpack import fftn, ifftn
>>> import numpy as np
>>> y = (-np.arange(16), 8 - np.arange(16), np.arange(16))
>>> np.allclose(y, ifftn(fftn(y)))
True
*)

val irfft : ?n:int -> ?axis:int -> ?overwrite_x:bool -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return inverse discrete Fourier transform of real sequence x.

The contents of `x` are interpreted as the output of the `rfft`
function.

Parameters
----------
x : array_like
    Transformed data to invert.
n : int, optional
    Length of the inverse Fourier transform.
    If n < x.shape[axis], x is truncated.
    If n > x.shape[axis], x is zero-padded.
    The default results in n = x.shape[axis].
axis : int, optional
    Axis along which the ifft's are computed; the default is over
    the last axis (i.e., axis=-1).
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.

Returns
-------
irfft : ndarray of floats
    The inverse discrete Fourier transform.

See Also
--------
rfft, ifft, scipy.fft.irfft

Notes
-----
The returned real array contains::

    [y(0),y(1),...,y(n-1)]

where for n is even::

    y(j) = 1/n (sum[k=1..n/2-1] (x[2*k-1]+sqrt(-1)*x[2*k])
                                 * exp(sqrt(-1)*j*k* 2*pi/n)
                + c.c. + x[0] + (-1)**(j) x[n-1])

and for n is odd::

    y(j) = 1/n (sum[k=1..(n-1)/2] (x[2*k-1]+sqrt(-1)*x[2*k])
                                 * exp(sqrt(-1)*j*k* 2*pi/n)
                + c.c. + x[0])

c.c. denotes complex conjugate of preceding expression.

For details on input parameters, see `rfft`.

To process (conjugate-symmetric) frequency-domain data with a complex
datatype, consider using the newer function `scipy.fft.irfft`.

Examples
--------
>>> from scipy.fftpack import rfft, irfft
>>> a = [1.0, 2.0, 3.0, 4.0, 5.0]
>>> irfft(a)
array([ 2.6       , -3.16405192,  1.24398433, -1.14955713,  1.46962473])
>>> irfft(rfft(a))
array([1., 2., 3., 4., 5.])
*)

val rfft : ?n:int -> ?axis:int -> ?overwrite_x:bool -> x:[`Ndarray of [>`Ndarray] Np.Obj.t | `Real_valued of Py.Object.t] -> unit -> (Py.Object.t * Py.Object.t)
(**
Discrete Fourier transform of a real sequence.

Parameters
----------
x : array_like, real-valued
    The data to transform.
n : int, optional
    Defines the length of the Fourier transform.  If `n` is not specified
    (the default) then ``n = x.shape[axis]``.  If ``n < x.shape[axis]``,
    `x` is truncated, if ``n > x.shape[axis]``, `x` is zero-padded.
axis : int, optional
    The axis along which the transform is applied.  The default is the
    last axis.
overwrite_x : bool, optional
    If set to true, the contents of `x` can be overwritten. Default is
    False.

Returns
-------
z : real ndarray
    The returned real array contains::

      [y(0),Re(y(1)),Im(y(1)),...,Re(y(n/2))]              if n is even
      [y(0),Re(y(1)),Im(y(1)),...,Re(y(n/2)),Im(y(n/2))]   if n is odd

    where::

      y(j) = sum[k=0..n-1] x[k] * exp(-sqrt(-1)*j*k*2*pi/n)
      j = 0..n-1

See Also
--------
fft, irfft, scipy.fft.rfft

Notes
-----
Within numerical accuracy, ``y == rfft(irfft(y))``.

Both single and double precision routines are implemented.  Half precision
inputs will be converted to single precision.  Non floating-point inputs
will be converted to double precision.  Long-double precision inputs are
not supported.

To get an output with a complex datatype, consider using the newer
function `scipy.fft.rfft`.

Examples
--------
>>> from scipy.fftpack import fft, rfft
>>> a = [9, -9, 1, 3]
>>> fft(a)
array([  4. +0.j,   8.+12.j,  16. +0.j,   8.-12.j])
>>> rfft(a)
array([  4.,   8.,  12.,  16.])
*)


end

module Convolve : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val convolve_z : ?overwrite_x:Py.Object.t -> x:Py.Object.t -> omega_real:Py.Object.t -> omega_imag:Py.Object.t -> unit -> Py.Object.t
(**
y = convolve_z(x,omega_real,omega_imag,[overwrite_x])

Wrapper for ``convolve_z``.

Parameters
----------
x : input rank-1 array('d') with bounds (n)
omega_real : input rank-1 array('d') with bounds (n)
omega_imag : input rank-1 array('d') with bounds (n)

Other Parameters
----------------
overwrite_x : input int, optional
    Default: 0

Returns
-------
y : rank-1 array('d') with bounds (n) and x storage
*)


end

module Helper : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val fftfreq : ?d:[`F of float | `I of int | `Bool of bool | `S of string] -> n:int -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return the Discrete Fourier Transform sample frequencies.

The returned float array `f` contains the frequency bin centers in cycles
per unit of the sample spacing (with zero at the start).  For instance, if
the sample spacing is in seconds, then the frequency unit is cycles/second.

Given a window length `n` and a sample spacing `d`::

  f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
  f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd

Parameters
----------
n : int
    Window length.
d : scalar, optional
    Sample spacing (inverse of the sampling rate). Defaults to 1.

Returns
-------
f : ndarray
    Array of length `n` containing the sample frequencies.

Examples
--------
>>> signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
>>> fourier = np.fft.fft(signal)
>>> n = signal.size
>>> timestep = 0.1
>>> freq = np.fft.fftfreq(n, d=timestep)
>>> freq
array([ 0.  ,  1.25,  2.5 , ..., -3.75, -2.5 , -1.25])
*)

val fftshift : ?axes:[`Shape_tuple of Py.Object.t | `I of int] -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Shift the zero-frequency component to the center of the spectrum.

This function swaps half-spaces for all axes listed (defaults to all).
Note that ``y[0]`` is the Nyquist component only if ``len(x)`` is even.

Parameters
----------
x : array_like
    Input array.
axes : int or shape tuple, optional
    Axes over which to shift.  Default is None, which shifts all axes.

Returns
-------
y : ndarray
    The shifted array.

See Also
--------
ifftshift : The inverse of `fftshift`.

Examples
--------
>>> freqs = np.fft.fftfreq(10, 0.1)
>>> freqs
array([ 0.,  1.,  2., ..., -3., -2., -1.])
>>> np.fft.fftshift(freqs)
array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])

Shift the zero-frequency component only along the second axis:

>>> freqs = np.fft.fftfreq(9, d=1./9).reshape(3, 3)
>>> freqs
array([[ 0.,  1.,  2.],
       [ 3.,  4., -4.],
       [-3., -2., -1.]])
>>> np.fft.fftshift(freqs, axes=(1,))
array([[ 2.,  0.,  1.],
       [-4.,  3.,  4.],
       [-1., -3., -2.]])
*)

val ifftshift : ?axes:[`Shape_tuple of Py.Object.t | `I of int] -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
The inverse of `fftshift`. Although identical for even-length `x`, the
functions differ by one sample for odd-length `x`.

Parameters
----------
x : array_like
    Input array.
axes : int or shape tuple, optional
    Axes over which to calculate.  Defaults to None, which shifts all axes.

Returns
-------
y : ndarray
    The shifted array.

See Also
--------
fftshift : Shift zero-frequency component to the center of the spectrum.

Examples
--------
>>> freqs = np.fft.fftfreq(9, d=1./9).reshape(3, 3)
>>> freqs
array([[ 0.,  1.,  2.],
       [ 3.,  4., -4.],
       [-3., -2., -1.]])
>>> np.fft.ifftshift(np.fft.fftshift(freqs))
array([[ 0.,  1.,  2.],
       [ 3.,  4., -4.],
       [-3., -2., -1.]])
*)

val next_fast_len : int -> int
(**
Find the next fast size of input data to `fft`, for zero-padding, etc.

SciPy's FFTPACK has efficient functions for radix {2, 3, 4, 5}, so this
returns the next composite of the prime factors 2, 3, and 5 which is
greater than or equal to `target`. (These are also known as 5-smooth
numbers, regular numbers, or Hamming numbers.)

Parameters
----------
target : int
    Length to start searching from.  Must be a positive integer.

Returns
-------
out : int
    The first 5-smooth number greater than or equal to `target`.

Notes
-----
.. versionadded:: 0.18.0

Examples
--------
On a particular machine, an FFT of prime length takes 133 ms:

>>> from scipy import fftpack
>>> min_len = 10007  # prime length is worst case for speed
>>> a = np.random.randn(min_len)
>>> b = fftpack.fft(a)

Zero-padding to the next 5-smooth length reduces computation time to
211 us, a speedup of 630 times:

>>> fftpack.helper.next_fast_len(min_len)
10125
>>> b = fftpack.fft(a, 10125)

Rounding up to the next power of 2 is not optimal, taking 367 us to
compute, 1.7 times as long as the 5-smooth size:

>>> b = fftpack.fft(a, 16384)
*)

val rfftfreq : ?d:[`F of float | `I of int | `Bool of bool | `S of string] -> n:int -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
DFT sample frequencies (for usage with rfft, irfft).

The returned float array contains the frequency bins in
cycles/unit (with zero at the start) given a window length `n` and a
sample spacing `d`::

  f = [0,1,1,2,2,...,n/2-1,n/2-1,n/2]/(d*n)   if n is even
  f = [0,1,1,2,2,...,n/2-1,n/2-1,n/2,n/2]/(d*n)   if n is odd

Parameters
----------
n : int
    Window length.
d : scalar, optional
    Sample spacing. Default is 1.

Returns
-------
out : ndarray
    The array of length `n`, containing the sample frequencies.

Examples
--------
>>> from scipy import fftpack
>>> sig = np.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
>>> sig_fft = fftpack.rfft(sig)
>>> n = sig_fft.size
>>> timestep = 0.1
>>> freq = fftpack.rfftfreq(n, d=timestep)
>>> freq
array([ 0.  ,  1.25,  1.25,  2.5 ,  2.5 ,  3.75,  3.75,  5.  ])
*)


end

module Pseudo_diffs : sig
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

val cc_diff : ?period:float -> ?_cache:Py.Object.t -> x:[>`Ndarray] Np.Obj.t -> a:Py.Object.t -> b:Py.Object.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return (a,b)-cosh/cosh pseudo-derivative of a periodic sequence.

If x_j and y_j are Fourier coefficients of periodic functions x
and y, respectively, then::

  y_j = cosh(j*a*2*pi/period)/cosh(j*b*2*pi/period) * x_j

Parameters
----------
x : array_like
    The array to take the pseudo-derivative from.
a,b : float
    Defines the parameters of the sinh/sinh pseudo-differential
    operator.
period : float, optional
    The period of the sequence x. Default is ``2*pi``.

Returns
-------
cc_diff : ndarray
    Pseudo-derivative of periodic sequence `x`.

Notes
-----
``cc_diff(cc_diff(x,a,b),b,a) == x``
*)

val cos : ?out:[`Ndarray of [>`Ndarray] Np.Obj.t | `Tuple_of_ndarray_and_None of Py.Object.t] -> ?where:[>`Ndarray] Np.Obj.t -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
cos(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

Cosine element-wise.

Parameters
----------
x : array_like
    Input array in radians.
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
    The corresponding cosine values.
    This is a scalar if `x` is a scalar.

Notes
-----
If `out` is provided, the function writes the result into it,
and returns a reference to `out`.  (See Examples)

References
----------
M. Abramowitz and I. A. Stegun, Handbook of Mathematical Functions.
New York, NY: Dover, 1972.

Examples
--------
>>> np.cos(np.array([0, np.pi/2, np.pi]))
array([  1.00000000e+00,   6.12303177e-17,  -1.00000000e+00])
>>>
>>> # Example of providing the optional output parameter
>>> out1 = np.array([0], dtype='d')
>>> out2 = np.cos([0.1], out1)
>>> out2 is out1
True
>>>
>>> # Example of ValueError due to provision of shape mis-matched `out`
>>> np.cos(np.zeros((3,3)),np.zeros((2,2)))
Traceback (most recent call last):
  File '<stdin>', line 1, in <module>
ValueError: operands could not be broadcast together with shapes (3,3) (2,2)
*)

val cosh : ?out:[`Ndarray of [>`Ndarray] Np.Obj.t | `Tuple_of_ndarray_and_None of Py.Object.t] -> ?where:[>`Ndarray] Np.Obj.t -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
cosh(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

Hyperbolic cosine, element-wise.

Equivalent to ``1/2 * (np.exp(x) + np.exp(-x))`` and ``np.cos(1j*x)``.

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
out : ndarray or scalar
    Output array of same shape as `x`.
    This is a scalar if `x` is a scalar.

Examples
--------
>>> np.cosh(0)
1.0

The hyperbolic cosine describes the shape of a hanging cable:

>>> import matplotlib.pyplot as plt
>>> x = np.linspace(-4, 4, 1000)
>>> plt.plot(x, np.cosh(x))
>>> plt.show()
*)

val cs_diff : ?period:float -> ?_cache:Py.Object.t -> x:[>`Ndarray] Np.Obj.t -> a:Py.Object.t -> b:Py.Object.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return (a,b)-cosh/sinh pseudo-derivative of a periodic sequence.

If ``x_j`` and ``y_j`` are Fourier coefficients of periodic functions x
and y, respectively, then::

  y_j = -sqrt(-1)*cosh(j*a*2*pi/period)/sinh(j*b*2*pi/period) * x_j
  y_0 = 0

Parameters
----------
x : array_like
    The array to take the pseudo-derivative from.
a, b : float
    Defines the parameters of the cosh/sinh pseudo-differential
    operator.
period : float, optional
    The period of the sequence. Default period is ``2*pi``.

Returns
-------
cs_diff : ndarray
    Pseudo-derivative of periodic sequence `x`.

Notes
-----
For even len(`x`), the Nyquist mode of `x` is taken as zero.
*)

val diff : ?order:int -> ?period:float -> ?_cache:Py.Object.t -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Return k-th derivative (or integral) of a periodic sequence x.

If x_j and y_j are Fourier coefficients of periodic functions x
and y, respectively, then::

  y_j = pow(sqrt(-1)*j*2*pi/period, order) * x_j
  y_0 = 0 if order is not 0.

Parameters
----------
x : array_like
    Input array.
order : int, optional
    The order of differentiation. Default order is 1. If order is
    negative, then integration is carried out under the assumption
    that ``x_0 == 0``.
period : float, optional
    The assumed period of the sequence. Default is ``2*pi``.

Notes
-----
If ``sum(x, axis=0) = 0`` then ``diff(diff(x, k), -k) == x`` (within
numerical accuracy).

For odd order and even ``len(x)``, the Nyquist mode is taken zero.
*)

val hilbert : ?_cache:Py.Object.t -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return Hilbert transform of a periodic sequence x.

If x_j and y_j are Fourier coefficients of periodic functions x
and y, respectively, then::

  y_j = sqrt(-1)*sign(j) * x_j
  y_0 = 0

Parameters
----------
x : array_like
    The input array, should be periodic.
_cache : dict, optional
    Dictionary that contains the kernel used to do a convolution with.

Returns
-------
y : ndarray
    The transformed input.

See Also
--------
scipy.signal.hilbert : Compute the analytic signal, using the Hilbert
                       transform.

Notes
-----
If ``sum(x, axis=0) == 0`` then ``hilbert(ihilbert(x)) == x``.

For even len(x), the Nyquist mode of x is taken zero.

The sign of the returned transform does not have a factor -1 that is more
often than not found in the definition of the Hilbert transform.  Note also
that `scipy.signal.hilbert` does have an extra -1 factor compared to this
function.
*)

val ihilbert : Py.Object.t -> Py.Object.t
(**
Return inverse Hilbert transform of a periodic sequence x.

If ``x_j`` and ``y_j`` are Fourier coefficients of periodic functions x
and y, respectively, then::

  y_j = -sqrt(-1)*sign(j) * x_j
  y_0 = 0
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

val itilbert : ?period:Py.Object.t -> ?_cache:Py.Object.t -> x:Py.Object.t -> h:Py.Object.t -> unit -> Py.Object.t
(**
Return inverse h-Tilbert transform of a periodic sequence x.

If ``x_j`` and ``y_j`` are Fourier coefficients of periodic functions x
and y, respectively, then::

  y_j = -sqrt(-1)*tanh(j*h*2*pi/period) * x_j
  y_0 = 0

For more details, see `tilbert`.
*)

val sc_diff : ?period:float -> ?_cache:Py.Object.t -> x:[>`Ndarray] Np.Obj.t -> a:Py.Object.t -> b:Py.Object.t -> unit -> Py.Object.t
(**
Return (a,b)-sinh/cosh pseudo-derivative of a periodic sequence x.

If x_j and y_j are Fourier coefficients of periodic functions x
and y, respectively, then::

  y_j = sqrt(-1)*sinh(j*a*2*pi/period)/cosh(j*b*2*pi/period) * x_j
  y_0 = 0

Parameters
----------
x : array_like
    Input array.
a,b : float
    Defines the parameters of the sinh/cosh pseudo-differential
    operator.
period : float, optional
    The period of the sequence x. Default is 2*pi.

Notes
-----
``sc_diff(cs_diff(x,a,b),b,a) == x``
For even ``len(x)``, the Nyquist mode of x is taken as zero.
*)

val shift : ?period:float -> ?_cache:Py.Object.t -> x:[>`Ndarray] Np.Obj.t -> a:float -> unit -> Py.Object.t
(**
Shift periodic sequence x by a: y(u) = x(u+a).

If x_j and y_j are Fourier coefficients of periodic functions x
and y, respectively, then::

      y_j = exp(j*a*2*pi/period*sqrt(-1)) * x_f

Parameters
----------
x : array_like
    The array to take the pseudo-derivative from.
a : float
    Defines the parameters of the sinh/sinh pseudo-differential
period : float, optional
    The period of the sequences x and y. Default period is ``2*pi``.
*)

val sin : ?out:[`Ndarray of [>`Ndarray] Np.Obj.t | `Tuple_of_ndarray_and_None of Py.Object.t] -> ?where:[>`Ndarray] Np.Obj.t -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
sin(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

Trigonometric sine, element-wise.

Parameters
----------
x : array_like
    Angle, in radians (:math:`2 \pi` rad equals 360 degrees).
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
y : array_like
    The sine of each element of x.
    This is a scalar if `x` is a scalar.

See Also
--------
arcsin, sinh, cos

Notes
-----
The sine is one of the fundamental functions of trigonometry (the
mathematical study of triangles).  Consider a circle of radius 1
centered on the origin.  A ray comes in from the :math:`+x` axis, makes
an angle at the origin (measured counter-clockwise from that axis), and
departs from the origin.  The :math:`y` coordinate of the outgoing
ray's intersection with the unit circle is the sine of that angle.  It
ranges from -1 for :math:`x=3\pi / 2` to +1 for :math:`\pi / 2.`  The
function has zeroes where the angle is a multiple of :math:`\pi`.
Sines of angles between :math:`\pi` and :math:`2\pi` are negative.
The numerous properties of the sine and related functions are included
in any standard trigonometry text.

Examples
--------
Print sine of one angle:

>>> np.sin(np.pi/2.)
1.0

Print sines of an array of angles given in degrees:

>>> np.sin(np.array((0., 30., 45., 60., 90.)) * np.pi / 180. )
array([ 0.        ,  0.5       ,  0.70710678,  0.8660254 ,  1.        ])

Plot the sine function:

>>> import matplotlib.pylab as plt
>>> x = np.linspace(-np.pi, np.pi, 201)
>>> plt.plot(x, np.sin(x))
>>> plt.xlabel('Angle [rad]')
>>> plt.ylabel('sin(x)')
>>> plt.axis('tight')
>>> plt.show()
*)

val sinh : ?out:[`Ndarray of [>`Ndarray] Np.Obj.t | `Tuple_of_ndarray_and_None of Py.Object.t] -> ?where:[>`Ndarray] Np.Obj.t -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
sinh(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

Hyperbolic sine, element-wise.

Equivalent to ``1/2 * (np.exp(x) - np.exp(-x))`` or
``-1j * np.sin(1j*x)``.

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
y : ndarray
    The corresponding hyperbolic sine values.
    This is a scalar if `x` is a scalar.

Notes
-----
If `out` is provided, the function writes the result into it,
and returns a reference to `out`.  (See Examples)

References
----------
M. Abramowitz and I. A. Stegun, Handbook of Mathematical Functions.
New York, NY: Dover, 1972, pg. 83.

Examples
--------
>>> np.sinh(0)
0.0
>>> np.sinh(np.pi*1j/2)
1j
>>> np.sinh(np.pi*1j) # (exact value is 0)
1.2246063538223773e-016j
>>> # Discrepancy due to vagaries of floating point arithmetic.

>>> # Example of providing the optional output parameter
>>> out1 = np.array([0], dtype='d')
>>> out2 = np.sinh([0.1], out1)
>>> out2 is out1
True

>>> # Example of ValueError due to provision of shape mis-matched `out`
>>> np.sinh(np.zeros((3,3)),np.zeros((2,2)))
Traceback (most recent call last):
  File '<stdin>', line 1, in <module>
ValueError: operands could not be broadcast together with shapes (3,3) (2,2)
*)

val ss_diff : ?period:float -> ?_cache:Py.Object.t -> x:[>`Ndarray] Np.Obj.t -> a:Py.Object.t -> b:Py.Object.t -> unit -> Py.Object.t
(**
Return (a,b)-sinh/sinh pseudo-derivative of a periodic sequence x.

If x_j and y_j are Fourier coefficients of periodic functions x
and y, respectively, then::

  y_j = sinh(j*a*2*pi/period)/sinh(j*b*2*pi/period) * x_j
  y_0 = a/b * x_0

Parameters
----------
x : array_like
    The array to take the pseudo-derivative from.
a,b
    Defines the parameters of the sinh/sinh pseudo-differential
    operator.
period : float, optional
    The period of the sequence x. Default is ``2*pi``.

Notes
-----
``ss_diff(ss_diff(x,a,b),b,a) == x``
*)

val tanh : ?out:[`Ndarray of [>`Ndarray] Np.Obj.t | `Tuple_of_ndarray_and_None of Py.Object.t] -> ?where:[>`Ndarray] Np.Obj.t -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
tanh(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

Compute hyperbolic tangent element-wise.

Equivalent to ``np.sinh(x)/np.cosh(x)`` or ``-1j * np.tan(1j*x)``.

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
y : ndarray
    The corresponding hyperbolic tangent values.
    This is a scalar if `x` is a scalar.

Notes
-----
If `out` is provided, the function writes the result into it,
and returns a reference to `out`.  (See Examples)

References
----------
.. [1] M. Abramowitz and I. A. Stegun, Handbook of Mathematical Functions.
       New York, NY: Dover, 1972, pg. 83.
       http://www.math.sfu.ca/~cbm/aands/

.. [2] Wikipedia, 'Hyperbolic function',
       https://en.wikipedia.org/wiki/Hyperbolic_function

Examples
--------
>>> np.tanh((0, np.pi*1j, np.pi*1j/2))
array([ 0. +0.00000000e+00j,  0. -1.22460635e-16j,  0. +1.63317787e+16j])

>>> # Example of providing the optional output parameter illustrating
>>> # that what is returned is a reference to said parameter
>>> out1 = np.array([0], dtype='d')
>>> out2 = np.tanh([0.1], out1)
>>> out2 is out1
True

>>> # Example of ValueError due to provision of shape mis-matched `out`
>>> np.tanh(np.zeros((3,3)),np.zeros((2,2)))
Traceback (most recent call last):
  File '<stdin>', line 1, in <module>
ValueError: operands could not be broadcast together with shapes (3,3) (2,2)
*)

val tilbert : ?period:float -> ?_cache:Py.Object.t -> x:[>`Ndarray] Np.Obj.t -> h:float -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return h-Tilbert transform of a periodic sequence x.

If x_j and y_j are Fourier coefficients of periodic functions x
and y, respectively, then::

    y_j = sqrt(-1)*coth(j*h*2*pi/period) * x_j
    y_0 = 0

Parameters
----------
x : array_like
    The input array to transform.
h : float
    Defines the parameter of the Tilbert transform.
period : float, optional
    The assumed period of the sequence.  Default period is ``2*pi``.

Returns
-------
tilbert : ndarray
    The result of the transform.

Notes
-----
If ``sum(x, axis=0) == 0`` and ``n = len(x)`` is odd then
``tilbert(itilbert(x)) == x``.

If ``2 * pi * h / period`` is approximately 10 or larger, then
numerically ``tilbert == hilbert``
(theoretically oo-Tilbert == Hilbert).

For even ``len(x)``, the Nyquist mode of ``x`` is taken zero.
*)


end

module Realtransforms : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val dct : ?type_:[`Three | `One | `Four | `Two] -> ?n:int -> ?axis:int -> ?norm:string -> ?overwrite_x:bool -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Return the Discrete Cosine Transform of arbitrary type sequence x.

Parameters
----------
x : array_like
    The input array.
type : {1, 2, 3, 4}, optional
    Type of the DCT (see Notes). Default type is 2.
n : int, optional
    Length of the transform.  If ``n < x.shape[axis]``, `x` is
    truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
    default results in ``n = x.shape[axis]``.
axis : int, optional
    Axis along which the dct is computed; the default is over the
    last axis (i.e., ``axis=-1``).
norm : {None, 'ortho'}, optional
    Normalization mode (see Notes). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.

Returns
-------
y : ndarray of real
    The transformed input array.

See Also
--------
idct : Inverse DCT

Notes
-----
For a single dimension array ``x``, ``dct(x, norm='ortho')`` is equal to
MATLAB ``dct(x)``.

There are theoretically 8 types of the DCT, only the first 4 types are
implemented in scipy. 'The' DCT generally refers to DCT type 2, and 'the'
Inverse DCT generally refers to DCT type 3.

**Type I**

There are several definitions of the DCT-I; we use the following
(for ``norm=None``)

.. math::

   y_k = x_0 + (-1)^k x_{N-1} + 2 \sum_{n=1}^{N-2} x_n \cos\left(
   \frac{\pi k n}{N-1} \right)

If ``norm='ortho'``, ``x[0]`` and ``x[N-1]`` are multiplied by a scaling
factor of :math:`\sqrt{2}`, and ``y[k]`` is multiplied by a scaling factor
``f``

.. math::

    f = \begin{cases}
     \frac{1}{2}\sqrt{\frac{1}{N-1}} & \text{if }k=0\text{ or }N-1, \\
     \frac{1}{2}\sqrt{\frac{2}{N-1}} & \text{otherwise} \end{cases}

.. versionadded:: 1.2.0
   Orthonormalization in DCT-I.

.. note::
   The DCT-I is only supported for input size > 1.

**Type II**

There are several definitions of the DCT-II; we use the following
(for ``norm=None``)

.. math::

   y_k = 2 \sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi k(2n+1)}{2N} \right)

If ``norm='ortho'``, ``y[k]`` is multiplied by a scaling factor ``f``

.. math::
   f = \begin{cases}
   \sqrt{\frac{1}{4N}} & \text{if }k=0, \\
   \sqrt{\frac{1}{2N}} & \text{otherwise} \end{cases}

Which makes the corresponding matrix of coefficients orthonormal
(``O @ O.T = np.eye(N)``).

**Type III**

There are several definitions, we use the following (for ``norm=None``)

.. math::

   y_k = x_0 + 2 \sum_{n=1}^{N-1} x_n \cos\left(\frac{\pi(2k+1)n}{2N}\right)

or, for ``norm='ortho'``

.. math::

   y_k = \frac{x_0}{\sqrt{N}} + \sqrt{\frac{2}{N}} \sum_{n=1}^{N-1} x_n
   \cos\left(\frac{\pi(2k+1)n}{2N}\right)

The (unnormalized) DCT-III is the inverse of the (unnormalized) DCT-II, up
to a factor `2N`. The orthonormalized DCT-III is exactly the inverse of
the orthonormalized DCT-II.

**Type IV**

There are several definitions of the DCT-IV; we use the following
(for ``norm=None``)

.. math::

   y_k = 2 \sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi(2k+1)(2n+1)}{4N} \right)

If ``norm='ortho'``, ``y[k]`` is multiplied by a scaling factor ``f``

.. math::

    f = \frac{1}{\sqrt{2N}}

.. versionadded:: 1.2.0
   Support for DCT-IV.

References
----------
.. [1] 'A Fast Cosine Transform in One and Two Dimensions', by J.
       Makhoul, `IEEE Transactions on acoustics, speech and signal
       processing` vol. 28(1), pp. 27-34,
       :doi:`10.1109/TASSP.1980.1163351` (1980).
.. [2] Wikipedia, 'Discrete cosine transform',
       https://en.wikipedia.org/wiki/Discrete_cosine_transform

Examples
--------
The Type 1 DCT is equivalent to the FFT (though faster) for real,
even-symmetrical inputs.  The output is also real and even-symmetrical.
Half of the FFT input is used to generate half of the FFT output:

>>> from scipy.fftpack import fft, dct
>>> fft(np.array([4., 3., 5., 10., 5., 3.])).real
array([ 30.,  -8.,   6.,  -2.,   6.,  -8.])
>>> dct(np.array([4., 3., 5., 10.]), 1)
array([ 30.,  -8.,   6.,  -2.])
*)

val dctn : ?type_:[`Three | `One | `Four | `Two] -> ?shape:[`I of int | `Array_like_of_ints of Py.Object.t] -> ?axes:[`I of int | `Array_like_of_ints of Py.Object.t] -> ?norm:string -> ?overwrite_x:bool -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Return multidimensional Discrete Cosine Transform along the specified axes.

Parameters
----------
x : array_like
    The input array.
type : {1, 2, 3, 4}, optional
    Type of the DCT (see Notes). Default type is 2.
shape : int or array_like of ints or None, optional
    The shape of the result.  If both `shape` and `axes` (see below) are
    None, `shape` is ``x.shape``; if `shape` is None but `axes` is
    not None, then `shape` is ``scipy.take(x.shape, axes, axis=0)``.
    If ``shape[i] > x.shape[i]``, the i-th dimension is padded with zeros.
    If ``shape[i] < x.shape[i]``, the i-th dimension is truncated to
    length ``shape[i]``.
    If any element of `shape` is -1, the size of the corresponding
    dimension of `x` is used.
axes : int or array_like of ints or None, optional
    Axes along which the DCT is computed.
    The default is over all axes.
norm : {None, 'ortho'}, optional
    Normalization mode (see Notes). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.

Returns
-------
y : ndarray of real
    The transformed input array.

See Also
--------
idctn : Inverse multidimensional DCT

Notes
-----
For full details of the DCT types and normalization modes, as well as
references, see `dct`.

Examples
--------
>>> from scipy.fftpack import dctn, idctn
>>> y = np.random.randn(16, 16)
>>> np.allclose(y, idctn(dctn(y, norm='ortho'), norm='ortho'))
True
*)

val dst : ?type_:[`Three | `One | `Four | `Two] -> ?n:int -> ?axis:int -> ?norm:string -> ?overwrite_x:bool -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return the Discrete Sine Transform of arbitrary type sequence x.

Parameters
----------
x : array_like
    The input array.
type : {1, 2, 3, 4}, optional
    Type of the DST (see Notes). Default type is 2.
n : int, optional
    Length of the transform.  If ``n < x.shape[axis]``, `x` is
    truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
    default results in ``n = x.shape[axis]``.
axis : int, optional
    Axis along which the dst is computed; the default is over the
    last axis (i.e., ``axis=-1``).
norm : {None, 'ortho'}, optional
    Normalization mode (see Notes). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.

Returns
-------
dst : ndarray of reals
    The transformed input array.

See Also
--------
idst : Inverse DST

Notes
-----
For a single dimension array ``x``.

There are theoretically 8 types of the DST for different combinations of
even/odd boundary conditions and boundary off sets [1]_, only the first
4 types are implemented in scipy.

**Type I**

There are several definitions of the DST-I; we use the following
for ``norm=None``. DST-I assumes the input is odd around `n=-1` and `n=N`.

.. math::

    y_k = 2 \sum_{n=0}^{N-1} x_n \sin\left(\frac{\pi(k+1)(n+1)}{N+1}\right)

Note that the DST-I is only supported for input size > 1.
The (unnormalized) DST-I is its own inverse, up to a factor `2(N+1)`.
The orthonormalized DST-I is exactly its own inverse.

**Type II**

There are several definitions of the DST-II; we use the following for
``norm=None``. DST-II assumes the input is odd around `n=-1/2` and
`n=N-1/2`; the output is odd around :math:`k=-1` and even around `k=N-1`

.. math::

    y_k = 2 \sum_{n=0}^{N-1} x_n \sin\left(\frac{\pi(k+1)(2n+1)}{2N}\right)

if ``norm='ortho'``, ``y[k]`` is multiplied by a scaling factor ``f``

.. math::

    f = \begin{cases}
    \sqrt{\frac{1}{4N}} & \text{if }k = 0, \\
    \sqrt{\frac{1}{2N}} & \text{otherwise} \end{cases}

**Type III**

There are several definitions of the DST-III, we use the following (for
``norm=None``). DST-III assumes the input is odd around `n=-1` and even
around `n=N-1`

.. math::

    y_k = (-1)^k x_{N-1} + 2 \sum_{n=0}^{N-2} x_n \sin\left(
    \frac{\pi(2k+1)(n+1)}{2N}\right)

The (unnormalized) DST-III is the inverse of the (unnormalized) DST-II, up
to a factor `2N`. The orthonormalized DST-III is exactly the inverse of the
orthonormalized DST-II.

.. versionadded:: 0.11.0

**Type IV**

There are several definitions of the DST-IV, we use the following (for
``norm=None``). DST-IV assumes the input is odd around `n=-0.5` and even
around `n=N-0.5`

.. math::

    y_k = 2 \sum_{n=0}^{N-1} x_n \sin\left(\frac{\pi(2k+1)(2n+1)}{4N}\right)

The (unnormalized) DST-IV is its own inverse, up to a factor `2N`. The
orthonormalized DST-IV is exactly its own inverse.

.. versionadded:: 1.2.0
   Support for DST-IV.

References
----------
.. [1] Wikipedia, 'Discrete sine transform',
       https://en.wikipedia.org/wiki/Discrete_sine_transform
*)

val dstn : ?type_:[`Three | `One | `Four | `Two] -> ?shape:[`I of int | `Array_like_of_ints of Py.Object.t] -> ?axes:[`I of int | `Array_like_of_ints of Py.Object.t] -> ?norm:string -> ?overwrite_x:bool -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Return multidimensional Discrete Sine Transform along the specified axes.

Parameters
----------
x : array_like
    The input array.
type : {1, 2, 3, 4}, optional
    Type of the DST (see Notes). Default type is 2.
shape : int or array_like of ints or None, optional
    The shape of the result.  If both `shape` and `axes` (see below) are
    None, `shape` is ``x.shape``; if `shape` is None but `axes` is
    not None, then `shape` is ``scipy.take(x.shape, axes, axis=0)``.
    If ``shape[i] > x.shape[i]``, the i-th dimension is padded with zeros.
    If ``shape[i] < x.shape[i]``, the i-th dimension is truncated to
    length ``shape[i]``.
    If any element of `shape` is -1, the size of the corresponding
    dimension of `x` is used.
axes : int or array_like of ints or None, optional
    Axes along which the DCT is computed.
    The default is over all axes.
norm : {None, 'ortho'}, optional
    Normalization mode (see Notes). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.

Returns
-------
y : ndarray of real
    The transformed input array.

See Also
--------
idstn : Inverse multidimensional DST

Notes
-----
For full details of the DST types and normalization modes, as well as
references, see `dst`.

Examples
--------
>>> from scipy.fftpack import dstn, idstn
>>> y = np.random.randn(16, 16)
>>> np.allclose(y, idstn(dstn(y, norm='ortho'), norm='ortho'))
True
*)

val idct : ?type_:[`Three | `One | `Four | `Two] -> ?n:int -> ?axis:int -> ?norm:string -> ?overwrite_x:bool -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Return the Inverse Discrete Cosine Transform of an arbitrary type sequence.

Parameters
----------
x : array_like
    The input array.
type : {1, 2, 3, 4}, optional
    Type of the DCT (see Notes). Default type is 2.
n : int, optional
    Length of the transform.  If ``n < x.shape[axis]``, `x` is
    truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
    default results in ``n = x.shape[axis]``.
axis : int, optional
    Axis along which the idct is computed; the default is over the
    last axis (i.e., ``axis=-1``).
norm : {None, 'ortho'}, optional
    Normalization mode (see Notes). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.

Returns
-------
idct : ndarray of real
    The transformed input array.

See Also
--------
dct : Forward DCT

Notes
-----
For a single dimension array `x`, ``idct(x, norm='ortho')`` is equal to
MATLAB ``idct(x)``.

'The' IDCT is the IDCT of type 2, which is the same as DCT of type 3.

IDCT of type 1 is the DCT of type 1, IDCT of type 2 is the DCT of type
3, and IDCT of type 3 is the DCT of type 2. IDCT of type 4 is the DCT
of type 4. For the definition of these types, see `dct`.

Examples
--------
The Type 1 DCT is equivalent to the DFT for real, even-symmetrical
inputs.  The output is also real and even-symmetrical.  Half of the IFFT
input is used to generate half of the IFFT output:

>>> from scipy.fftpack import ifft, idct
>>> ifft(np.array([ 30.,  -8.,   6.,  -2.,   6.,  -8.])).real
array([  4.,   3.,   5.,  10.,   5.,   3.])
>>> idct(np.array([ 30.,  -8.,   6.,  -2.]), 1) / 6
array([  4.,   3.,   5.,  10.])
*)

val idctn : ?type_:[`Three | `One | `Four | `Two] -> ?shape:[`I of int | `Array_like_of_ints of Py.Object.t] -> ?axes:[`I of int | `Array_like_of_ints of Py.Object.t] -> ?norm:string -> ?overwrite_x:bool -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Return multidimensional Discrete Cosine Transform along the specified axes.

Parameters
----------
x : array_like
    The input array.
type : {1, 2, 3, 4}, optional
    Type of the DCT (see Notes). Default type is 2.
shape : int or array_like of ints or None, optional
    The shape of the result.  If both `shape` and `axes` (see below) are
    None, `shape` is ``x.shape``; if `shape` is None but `axes` is
    not None, then `shape` is ``scipy.take(x.shape, axes, axis=0)``.
    If ``shape[i] > x.shape[i]``, the i-th dimension is padded with zeros.
    If ``shape[i] < x.shape[i]``, the i-th dimension is truncated to
    length ``shape[i]``.
    If any element of `shape` is -1, the size of the corresponding
    dimension of `x` is used.
axes : int or array_like of ints or None, optional
    Axes along which the IDCT is computed.
    The default is over all axes.
norm : {None, 'ortho'}, optional
    Normalization mode (see Notes). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.

Returns
-------
y : ndarray of real
    The transformed input array.

See Also
--------
dctn : multidimensional DCT

Notes
-----
For full details of the IDCT types and normalization modes, as well as
references, see `idct`.

Examples
--------
>>> from scipy.fftpack import dctn, idctn
>>> y = np.random.randn(16, 16)
>>> np.allclose(y, idctn(dctn(y, norm='ortho'), norm='ortho'))
True
*)

val idst : ?type_:[`Three | `One | `Four | `Two] -> ?n:int -> ?axis:int -> ?norm:string -> ?overwrite_x:bool -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Return the Inverse Discrete Sine Transform of an arbitrary type sequence.

Parameters
----------
x : array_like
    The input array.
type : {1, 2, 3, 4}, optional
    Type of the DST (see Notes). Default type is 2.
n : int, optional
    Length of the transform.  If ``n < x.shape[axis]``, `x` is
    truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
    default results in ``n = x.shape[axis]``.
axis : int, optional
    Axis along which the idst is computed; the default is over the
    last axis (i.e., ``axis=-1``).
norm : {None, 'ortho'}, optional
    Normalization mode (see Notes). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.

Returns
-------
idst : ndarray of real
    The transformed input array.

See Also
--------
dst : Forward DST

Notes
-----
'The' IDST is the IDST of type 2, which is the same as DST of type 3.

IDST of type 1 is the DST of type 1, IDST of type 2 is the DST of type
3, and IDST of type 3 is the DST of type 2. For the definition of these
types, see `dst`.

.. versionadded:: 0.11.0
*)

val idstn : ?type_:[`Three | `One | `Four | `Two] -> ?shape:[`I of int | `Array_like_of_ints of Py.Object.t] -> ?axes:[`I of int | `Array_like_of_ints of Py.Object.t] -> ?norm:string -> ?overwrite_x:bool -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Return multidimensional Discrete Sine Transform along the specified axes.

Parameters
----------
x : array_like
    The input array.
type : {1, 2, 3, 4}, optional
    Type of the DST (see Notes). Default type is 2.
shape : int or array_like of ints or None, optional
    The shape of the result.  If both `shape` and `axes` (see below) are
    None, `shape` is ``x.shape``; if `shape` is None but `axes` is
    not None, then `shape` is ``scipy.take(x.shape, axes, axis=0)``.
    If ``shape[i] > x.shape[i]``, the i-th dimension is padded with zeros.
    If ``shape[i] < x.shape[i]``, the i-th dimension is truncated to
    length ``shape[i]``.
    If any element of `shape` is -1, the size of the corresponding
    dimension of `x` is used.
axes : int or array_like of ints or None, optional
    Axes along which the IDST is computed.
    The default is over all axes.
norm : {None, 'ortho'}, optional
    Normalization mode (see Notes). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.

Returns
-------
y : ndarray of real
    The transformed input array.

See Also
--------
dstn : multidimensional DST

Notes
-----
For full details of the IDST types and normalization modes, as well as
references, see `idst`.

Examples
--------
>>> from scipy.fftpack import dstn, idstn
>>> y = np.random.randn(16, 16)
>>> np.allclose(y, idstn(dstn(y, norm='ortho'), norm='ortho'))
True
*)


end

val cc_diff : ?period:float -> ?_cache:Py.Object.t -> x:[>`Ndarray] Np.Obj.t -> a:Py.Object.t -> b:Py.Object.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return (a,b)-cosh/cosh pseudo-derivative of a periodic sequence.

If x_j and y_j are Fourier coefficients of periodic functions x
and y, respectively, then::

  y_j = cosh(j*a*2*pi/period)/cosh(j*b*2*pi/period) * x_j

Parameters
----------
x : array_like
    The array to take the pseudo-derivative from.
a,b : float
    Defines the parameters of the sinh/sinh pseudo-differential
    operator.
period : float, optional
    The period of the sequence x. Default is ``2*pi``.

Returns
-------
cc_diff : ndarray
    Pseudo-derivative of periodic sequence `x`.

Notes
-----
``cc_diff(cc_diff(x,a,b),b,a) == x``
*)

val cs_diff : ?period:float -> ?_cache:Py.Object.t -> x:[>`Ndarray] Np.Obj.t -> a:Py.Object.t -> b:Py.Object.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return (a,b)-cosh/sinh pseudo-derivative of a periodic sequence.

If ``x_j`` and ``y_j`` are Fourier coefficients of periodic functions x
and y, respectively, then::

  y_j = -sqrt(-1)*cosh(j*a*2*pi/period)/sinh(j*b*2*pi/period) * x_j
  y_0 = 0

Parameters
----------
x : array_like
    The array to take the pseudo-derivative from.
a, b : float
    Defines the parameters of the cosh/sinh pseudo-differential
    operator.
period : float, optional
    The period of the sequence. Default period is ``2*pi``.

Returns
-------
cs_diff : ndarray
    Pseudo-derivative of periodic sequence `x`.

Notes
-----
For even len(`x`), the Nyquist mode of `x` is taken as zero.
*)

val dct : ?type_:[`Three | `One | `Four | `Two] -> ?n:int -> ?axis:int -> ?norm:string -> ?overwrite_x:bool -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Return the Discrete Cosine Transform of arbitrary type sequence x.

Parameters
----------
x : array_like
    The input array.
type : {1, 2, 3, 4}, optional
    Type of the DCT (see Notes). Default type is 2.
n : int, optional
    Length of the transform.  If ``n < x.shape[axis]``, `x` is
    truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
    default results in ``n = x.shape[axis]``.
axis : int, optional
    Axis along which the dct is computed; the default is over the
    last axis (i.e., ``axis=-1``).
norm : {None, 'ortho'}, optional
    Normalization mode (see Notes). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.

Returns
-------
y : ndarray of real
    The transformed input array.

See Also
--------
idct : Inverse DCT

Notes
-----
For a single dimension array ``x``, ``dct(x, norm='ortho')`` is equal to
MATLAB ``dct(x)``.

There are theoretically 8 types of the DCT, only the first 4 types are
implemented in scipy. 'The' DCT generally refers to DCT type 2, and 'the'
Inverse DCT generally refers to DCT type 3.

**Type I**

There are several definitions of the DCT-I; we use the following
(for ``norm=None``)

.. math::

   y_k = x_0 + (-1)^k x_{N-1} + 2 \sum_{n=1}^{N-2} x_n \cos\left(
   \frac{\pi k n}{N-1} \right)

If ``norm='ortho'``, ``x[0]`` and ``x[N-1]`` are multiplied by a scaling
factor of :math:`\sqrt{2}`, and ``y[k]`` is multiplied by a scaling factor
``f``

.. math::

    f = \begin{cases}
     \frac{1}{2}\sqrt{\frac{1}{N-1}} & \text{if }k=0\text{ or }N-1, \\
     \frac{1}{2}\sqrt{\frac{2}{N-1}} & \text{otherwise} \end{cases}

.. versionadded:: 1.2.0
   Orthonormalization in DCT-I.

.. note::
   The DCT-I is only supported for input size > 1.

**Type II**

There are several definitions of the DCT-II; we use the following
(for ``norm=None``)

.. math::

   y_k = 2 \sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi k(2n+1)}{2N} \right)

If ``norm='ortho'``, ``y[k]`` is multiplied by a scaling factor ``f``

.. math::
   f = \begin{cases}
   \sqrt{\frac{1}{4N}} & \text{if }k=0, \\
   \sqrt{\frac{1}{2N}} & \text{otherwise} \end{cases}

Which makes the corresponding matrix of coefficients orthonormal
(``O @ O.T = np.eye(N)``).

**Type III**

There are several definitions, we use the following (for ``norm=None``)

.. math::

   y_k = x_0 + 2 \sum_{n=1}^{N-1} x_n \cos\left(\frac{\pi(2k+1)n}{2N}\right)

or, for ``norm='ortho'``

.. math::

   y_k = \frac{x_0}{\sqrt{N}} + \sqrt{\frac{2}{N}} \sum_{n=1}^{N-1} x_n
   \cos\left(\frac{\pi(2k+1)n}{2N}\right)

The (unnormalized) DCT-III is the inverse of the (unnormalized) DCT-II, up
to a factor `2N`. The orthonormalized DCT-III is exactly the inverse of
the orthonormalized DCT-II.

**Type IV**

There are several definitions of the DCT-IV; we use the following
(for ``norm=None``)

.. math::

   y_k = 2 \sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi(2k+1)(2n+1)}{4N} \right)

If ``norm='ortho'``, ``y[k]`` is multiplied by a scaling factor ``f``

.. math::

    f = \frac{1}{\sqrt{2N}}

.. versionadded:: 1.2.0
   Support for DCT-IV.

References
----------
.. [1] 'A Fast Cosine Transform in One and Two Dimensions', by J.
       Makhoul, `IEEE Transactions on acoustics, speech and signal
       processing` vol. 28(1), pp. 27-34,
       :doi:`10.1109/TASSP.1980.1163351` (1980).
.. [2] Wikipedia, 'Discrete cosine transform',
       https://en.wikipedia.org/wiki/Discrete_cosine_transform

Examples
--------
The Type 1 DCT is equivalent to the FFT (though faster) for real,
even-symmetrical inputs.  The output is also real and even-symmetrical.
Half of the FFT input is used to generate half of the FFT output:

>>> from scipy.fftpack import fft, dct
>>> fft(np.array([4., 3., 5., 10., 5., 3.])).real
array([ 30.,  -8.,   6.,  -2.,   6.,  -8.])
>>> dct(np.array([4., 3., 5., 10.]), 1)
array([ 30.,  -8.,   6.,  -2.])
*)

val dctn : ?type_:[`Three | `One | `Four | `Two] -> ?shape:[`I of int | `Array_like_of_ints of Py.Object.t] -> ?axes:[`I of int | `Array_like_of_ints of Py.Object.t] -> ?norm:string -> ?overwrite_x:bool -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Return multidimensional Discrete Cosine Transform along the specified axes.

Parameters
----------
x : array_like
    The input array.
type : {1, 2, 3, 4}, optional
    Type of the DCT (see Notes). Default type is 2.
shape : int or array_like of ints or None, optional
    The shape of the result.  If both `shape` and `axes` (see below) are
    None, `shape` is ``x.shape``; if `shape` is None but `axes` is
    not None, then `shape` is ``scipy.take(x.shape, axes, axis=0)``.
    If ``shape[i] > x.shape[i]``, the i-th dimension is padded with zeros.
    If ``shape[i] < x.shape[i]``, the i-th dimension is truncated to
    length ``shape[i]``.
    If any element of `shape` is -1, the size of the corresponding
    dimension of `x` is used.
axes : int or array_like of ints or None, optional
    Axes along which the DCT is computed.
    The default is over all axes.
norm : {None, 'ortho'}, optional
    Normalization mode (see Notes). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.

Returns
-------
y : ndarray of real
    The transformed input array.

See Also
--------
idctn : Inverse multidimensional DCT

Notes
-----
For full details of the DCT types and normalization modes, as well as
references, see `dct`.

Examples
--------
>>> from scipy.fftpack import dctn, idctn
>>> y = np.random.randn(16, 16)
>>> np.allclose(y, idctn(dctn(y, norm='ortho'), norm='ortho'))
True
*)

val diff : ?order:int -> ?period:float -> ?_cache:Py.Object.t -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Return k-th derivative (or integral) of a periodic sequence x.

If x_j and y_j are Fourier coefficients of periodic functions x
and y, respectively, then::

  y_j = pow(sqrt(-1)*j*2*pi/period, order) * x_j
  y_0 = 0 if order is not 0.

Parameters
----------
x : array_like
    Input array.
order : int, optional
    The order of differentiation. Default order is 1. If order is
    negative, then integration is carried out under the assumption
    that ``x_0 == 0``.
period : float, optional
    The assumed period of the sequence. Default is ``2*pi``.

Notes
-----
If ``sum(x, axis=0) = 0`` then ``diff(diff(x, k), -k) == x`` (within
numerical accuracy).

For odd order and even ``len(x)``, the Nyquist mode is taken zero.
*)

val dst : ?type_:[`Three | `One | `Four | `Two] -> ?n:int -> ?axis:int -> ?norm:string -> ?overwrite_x:bool -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return the Discrete Sine Transform of arbitrary type sequence x.

Parameters
----------
x : array_like
    The input array.
type : {1, 2, 3, 4}, optional
    Type of the DST (see Notes). Default type is 2.
n : int, optional
    Length of the transform.  If ``n < x.shape[axis]``, `x` is
    truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
    default results in ``n = x.shape[axis]``.
axis : int, optional
    Axis along which the dst is computed; the default is over the
    last axis (i.e., ``axis=-1``).
norm : {None, 'ortho'}, optional
    Normalization mode (see Notes). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.

Returns
-------
dst : ndarray of reals
    The transformed input array.

See Also
--------
idst : Inverse DST

Notes
-----
For a single dimension array ``x``.

There are theoretically 8 types of the DST for different combinations of
even/odd boundary conditions and boundary off sets [1]_, only the first
4 types are implemented in scipy.

**Type I**

There are several definitions of the DST-I; we use the following
for ``norm=None``. DST-I assumes the input is odd around `n=-1` and `n=N`.

.. math::

    y_k = 2 \sum_{n=0}^{N-1} x_n \sin\left(\frac{\pi(k+1)(n+1)}{N+1}\right)

Note that the DST-I is only supported for input size > 1.
The (unnormalized) DST-I is its own inverse, up to a factor `2(N+1)`.
The orthonormalized DST-I is exactly its own inverse.

**Type II**

There are several definitions of the DST-II; we use the following for
``norm=None``. DST-II assumes the input is odd around `n=-1/2` and
`n=N-1/2`; the output is odd around :math:`k=-1` and even around `k=N-1`

.. math::

    y_k = 2 \sum_{n=0}^{N-1} x_n \sin\left(\frac{\pi(k+1)(2n+1)}{2N}\right)

if ``norm='ortho'``, ``y[k]`` is multiplied by a scaling factor ``f``

.. math::

    f = \begin{cases}
    \sqrt{\frac{1}{4N}} & \text{if }k = 0, \\
    \sqrt{\frac{1}{2N}} & \text{otherwise} \end{cases}

**Type III**

There are several definitions of the DST-III, we use the following (for
``norm=None``). DST-III assumes the input is odd around `n=-1` and even
around `n=N-1`

.. math::

    y_k = (-1)^k x_{N-1} + 2 \sum_{n=0}^{N-2} x_n \sin\left(
    \frac{\pi(2k+1)(n+1)}{2N}\right)

The (unnormalized) DST-III is the inverse of the (unnormalized) DST-II, up
to a factor `2N`. The orthonormalized DST-III is exactly the inverse of the
orthonormalized DST-II.

.. versionadded:: 0.11.0

**Type IV**

There are several definitions of the DST-IV, we use the following (for
``norm=None``). DST-IV assumes the input is odd around `n=-0.5` and even
around `n=N-0.5`

.. math::

    y_k = 2 \sum_{n=0}^{N-1} x_n \sin\left(\frac{\pi(2k+1)(2n+1)}{4N}\right)

The (unnormalized) DST-IV is its own inverse, up to a factor `2N`. The
orthonormalized DST-IV is exactly its own inverse.

.. versionadded:: 1.2.0
   Support for DST-IV.

References
----------
.. [1] Wikipedia, 'Discrete sine transform',
       https://en.wikipedia.org/wiki/Discrete_sine_transform
*)

val dstn : ?type_:[`Three | `One | `Four | `Two] -> ?shape:[`I of int | `Array_like_of_ints of Py.Object.t] -> ?axes:[`I of int | `Array_like_of_ints of Py.Object.t] -> ?norm:string -> ?overwrite_x:bool -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Return multidimensional Discrete Sine Transform along the specified axes.

Parameters
----------
x : array_like
    The input array.
type : {1, 2, 3, 4}, optional
    Type of the DST (see Notes). Default type is 2.
shape : int or array_like of ints or None, optional
    The shape of the result.  If both `shape` and `axes` (see below) are
    None, `shape` is ``x.shape``; if `shape` is None but `axes` is
    not None, then `shape` is ``scipy.take(x.shape, axes, axis=0)``.
    If ``shape[i] > x.shape[i]``, the i-th dimension is padded with zeros.
    If ``shape[i] < x.shape[i]``, the i-th dimension is truncated to
    length ``shape[i]``.
    If any element of `shape` is -1, the size of the corresponding
    dimension of `x` is used.
axes : int or array_like of ints or None, optional
    Axes along which the DCT is computed.
    The default is over all axes.
norm : {None, 'ortho'}, optional
    Normalization mode (see Notes). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.

Returns
-------
y : ndarray of real
    The transformed input array.

See Also
--------
idstn : Inverse multidimensional DST

Notes
-----
For full details of the DST types and normalization modes, as well as
references, see `dst`.

Examples
--------
>>> from scipy.fftpack import dstn, idstn
>>> y = np.random.randn(16, 16)
>>> np.allclose(y, idstn(dstn(y, norm='ortho'), norm='ortho'))
True
*)

val fft : ?n:int -> ?axis:int -> ?overwrite_x:bool -> x:[>`Ndarray] Np.Obj.t -> unit -> (Py.Object.t * Py.Object.t)
(**
Return discrete Fourier transform of real or complex sequence.

The returned complex array contains ``y(0), y(1),..., y(n-1)`` where

``y(j) = (x * exp(-2*pi*sqrt(-1)*j*np.arange(n)/n)).sum()``.

Parameters
----------
x : array_like
    Array to Fourier transform.
n : int, optional
    Length of the Fourier transform.  If ``n < x.shape[axis]``, `x` is
    truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
    default results in ``n = x.shape[axis]``.
axis : int, optional
    Axis along which the fft's are computed; the default is over the
    last axis (i.e., ``axis=-1``).
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.

Returns
-------
z : complex ndarray
    with the elements::

        [y(0),y(1),..,y(n/2),y(1-n/2),...,y(-1)]        if n is even
        [y(0),y(1),..,y((n-1)/2),y(-(n-1)/2),...,y(-1)]  if n is odd

    where::

        y(j) = sum[k=0..n-1] x[k] * exp(-sqrt(-1)*j*k* 2*pi/n), j = 0..n-1

See Also
--------
ifft : Inverse FFT
rfft : FFT of a real sequence

Notes
-----
The packing of the result is 'standard': If ``A = fft(a, n)``, then
``A[0]`` contains the zero-frequency term, ``A[1:n/2]`` contains the
positive-frequency terms, and ``A[n/2:]`` contains the negative-frequency
terms, in order of decreasingly negative frequency. So for an 8-point
transform, the frequencies of the result are [0, 1, 2, 3, -4, -3, -2, -1].
To rearrange the fft output so that the zero-frequency component is
centered, like [-4, -3, -2, -1,  0,  1,  2,  3], use `fftshift`.

Both single and double precision routines are implemented.  Half precision
inputs will be converted to single precision.  Non floating-point inputs
will be converted to double precision.  Long-double precision inputs are
not supported.

This function is most efficient when `n` is a power of two, and least
efficient when `n` is prime.

Note that if ``x`` is real-valued then ``A[j] == A[n-j].conjugate()``.
If ``x`` is real-valued and ``n`` is even then ``A[n/2]`` is real.

If the data type of `x` is real, a 'real FFT' algorithm is automatically
used, which roughly halves the computation time.  To increase efficiency
a little further, use `rfft`, which does the same calculation, but only
outputs half of the symmetrical spectrum.  If the data is both real and
symmetrical, the `dct` can again double the efficiency, by generating
half of the spectrum from half of the signal.

Examples
--------
>>> from scipy.fftpack import fft, ifft
>>> x = np.arange(5)
>>> np.allclose(fft(ifft(x)), x, atol=1e-15)  # within numerical accuracy.
True
*)

val fft2 : ?shape:Py.Object.t -> ?axes:Py.Object.t -> ?overwrite_x:Py.Object.t -> x:Py.Object.t -> unit -> Py.Object.t
(**
2-D discrete Fourier transform.

Return the two-dimensional discrete Fourier transform of the 2-D argument
`x`.

See Also
--------
fftn : for detailed information.
*)

val fftfreq : ?d:[`F of float | `I of int | `Bool of bool | `S of string] -> n:int -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return the Discrete Fourier Transform sample frequencies.

The returned float array `f` contains the frequency bin centers in cycles
per unit of the sample spacing (with zero at the start).  For instance, if
the sample spacing is in seconds, then the frequency unit is cycles/second.

Given a window length `n` and a sample spacing `d`::

  f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
  f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd

Parameters
----------
n : int
    Window length.
d : scalar, optional
    Sample spacing (inverse of the sampling rate). Defaults to 1.

Returns
-------
f : ndarray
    Array of length `n` containing the sample frequencies.

Examples
--------
>>> signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
>>> fourier = np.fft.fft(signal)
>>> n = signal.size
>>> timestep = 0.1
>>> freq = np.fft.fftfreq(n, d=timestep)
>>> freq
array([ 0.  ,  1.25,  2.5 , ..., -3.75, -2.5 , -1.25])
*)

val fftn : ?shape:[`I of int | `Array_like_of_ints of Py.Object.t] -> ?axes:[`I of int | `Array_like_of_ints of Py.Object.t] -> ?overwrite_x:bool -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Return multidimensional discrete Fourier transform.

The returned array contains::

  y[j_1,..,j_d] = sum[k_1=0..n_1-1, ..., k_d=0..n_d-1]
     x[k_1,..,k_d] * prod[i=1..d] exp(-sqrt(-1)*2*pi/n_i * j_i * k_i)

where d = len(x.shape) and n = x.shape.

Parameters
----------
x : array_like
    The (n-dimensional) array to transform.
shape : int or array_like of ints or None, optional
    The shape of the result.  If both `shape` and `axes` (see below) are
    None, `shape` is ``x.shape``; if `shape` is None but `axes` is
    not None, then `shape` is ``scipy.take(x.shape, axes, axis=0)``.
    If ``shape[i] > x.shape[i]``, the i-th dimension is padded with zeros.
    If ``shape[i] < x.shape[i]``, the i-th dimension is truncated to
    length ``shape[i]``.
    If any element of `shape` is -1, the size of the corresponding
    dimension of `x` is used.
axes : int or array_like of ints or None, optional
    The axes of `x` (`y` if `shape` is not None) along which the
    transform is applied.
    The default is over all axes.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed.  Default is False.

Returns
-------
y : complex-valued n-dimensional numpy array
    The (n-dimensional) DFT of the input array.

See Also
--------
ifftn

Notes
-----
If ``x`` is real-valued, then
``y[..., j_i, ...] == y[..., n_i-j_i, ...].conjugate()``.

Both single and double precision routines are implemented.  Half precision
inputs will be converted to single precision.  Non floating-point inputs
will be converted to double precision.  Long-double precision inputs are
not supported.

Examples
--------
>>> from scipy.fftpack import fftn, ifftn
>>> y = (-np.arange(16), 8 - np.arange(16), np.arange(16))
>>> np.allclose(y, fftn(ifftn(y)))
True
*)

val fftshift : ?axes:[`Shape_tuple of Py.Object.t | `I of int] -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Shift the zero-frequency component to the center of the spectrum.

This function swaps half-spaces for all axes listed (defaults to all).
Note that ``y[0]`` is the Nyquist component only if ``len(x)`` is even.

Parameters
----------
x : array_like
    Input array.
axes : int or shape tuple, optional
    Axes over which to shift.  Default is None, which shifts all axes.

Returns
-------
y : ndarray
    The shifted array.

See Also
--------
ifftshift : The inverse of `fftshift`.

Examples
--------
>>> freqs = np.fft.fftfreq(10, 0.1)
>>> freqs
array([ 0.,  1.,  2., ..., -3., -2., -1.])
>>> np.fft.fftshift(freqs)
array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])

Shift the zero-frequency component only along the second axis:

>>> freqs = np.fft.fftfreq(9, d=1./9).reshape(3, 3)
>>> freqs
array([[ 0.,  1.,  2.],
       [ 3.,  4., -4.],
       [-3., -2., -1.]])
>>> np.fft.fftshift(freqs, axes=(1,))
array([[ 2.,  0.,  1.],
       [-4.,  3.,  4.],
       [-1., -3., -2.]])
*)

val hilbert : ?_cache:Py.Object.t -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return Hilbert transform of a periodic sequence x.

If x_j and y_j are Fourier coefficients of periodic functions x
and y, respectively, then::

  y_j = sqrt(-1)*sign(j) * x_j
  y_0 = 0

Parameters
----------
x : array_like
    The input array, should be periodic.
_cache : dict, optional
    Dictionary that contains the kernel used to do a convolution with.

Returns
-------
y : ndarray
    The transformed input.

See Also
--------
scipy.signal.hilbert : Compute the analytic signal, using the Hilbert
                       transform.

Notes
-----
If ``sum(x, axis=0) == 0`` then ``hilbert(ihilbert(x)) == x``.

For even len(x), the Nyquist mode of x is taken zero.

The sign of the returned transform does not have a factor -1 that is more
often than not found in the definition of the Hilbert transform.  Note also
that `scipy.signal.hilbert` does have an extra -1 factor compared to this
function.
*)

val idct : ?type_:[`Three | `One | `Four | `Two] -> ?n:int -> ?axis:int -> ?norm:string -> ?overwrite_x:bool -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Return the Inverse Discrete Cosine Transform of an arbitrary type sequence.

Parameters
----------
x : array_like
    The input array.
type : {1, 2, 3, 4}, optional
    Type of the DCT (see Notes). Default type is 2.
n : int, optional
    Length of the transform.  If ``n < x.shape[axis]``, `x` is
    truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
    default results in ``n = x.shape[axis]``.
axis : int, optional
    Axis along which the idct is computed; the default is over the
    last axis (i.e., ``axis=-1``).
norm : {None, 'ortho'}, optional
    Normalization mode (see Notes). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.

Returns
-------
idct : ndarray of real
    The transformed input array.

See Also
--------
dct : Forward DCT

Notes
-----
For a single dimension array `x`, ``idct(x, norm='ortho')`` is equal to
MATLAB ``idct(x)``.

'The' IDCT is the IDCT of type 2, which is the same as DCT of type 3.

IDCT of type 1 is the DCT of type 1, IDCT of type 2 is the DCT of type
3, and IDCT of type 3 is the DCT of type 2. IDCT of type 4 is the DCT
of type 4. For the definition of these types, see `dct`.

Examples
--------
The Type 1 DCT is equivalent to the DFT for real, even-symmetrical
inputs.  The output is also real and even-symmetrical.  Half of the IFFT
input is used to generate half of the IFFT output:

>>> from scipy.fftpack import ifft, idct
>>> ifft(np.array([ 30.,  -8.,   6.,  -2.,   6.,  -8.])).real
array([  4.,   3.,   5.,  10.,   5.,   3.])
>>> idct(np.array([ 30.,  -8.,   6.,  -2.]), 1) / 6
array([  4.,   3.,   5.,  10.])
*)

val idctn : ?type_:[`Three | `One | `Four | `Two] -> ?shape:[`I of int | `Array_like_of_ints of Py.Object.t] -> ?axes:[`I of int | `Array_like_of_ints of Py.Object.t] -> ?norm:string -> ?overwrite_x:bool -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Return multidimensional Discrete Cosine Transform along the specified axes.

Parameters
----------
x : array_like
    The input array.
type : {1, 2, 3, 4}, optional
    Type of the DCT (see Notes). Default type is 2.
shape : int or array_like of ints or None, optional
    The shape of the result.  If both `shape` and `axes` (see below) are
    None, `shape` is ``x.shape``; if `shape` is None but `axes` is
    not None, then `shape` is ``scipy.take(x.shape, axes, axis=0)``.
    If ``shape[i] > x.shape[i]``, the i-th dimension is padded with zeros.
    If ``shape[i] < x.shape[i]``, the i-th dimension is truncated to
    length ``shape[i]``.
    If any element of `shape` is -1, the size of the corresponding
    dimension of `x` is used.
axes : int or array_like of ints or None, optional
    Axes along which the IDCT is computed.
    The default is over all axes.
norm : {None, 'ortho'}, optional
    Normalization mode (see Notes). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.

Returns
-------
y : ndarray of real
    The transformed input array.

See Also
--------
dctn : multidimensional DCT

Notes
-----
For full details of the IDCT types and normalization modes, as well as
references, see `idct`.

Examples
--------
>>> from scipy.fftpack import dctn, idctn
>>> y = np.random.randn(16, 16)
>>> np.allclose(y, idctn(dctn(y, norm='ortho'), norm='ortho'))
True
*)

val idst : ?type_:[`Three | `One | `Four | `Two] -> ?n:int -> ?axis:int -> ?norm:string -> ?overwrite_x:bool -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Return the Inverse Discrete Sine Transform of an arbitrary type sequence.

Parameters
----------
x : array_like
    The input array.
type : {1, 2, 3, 4}, optional
    Type of the DST (see Notes). Default type is 2.
n : int, optional
    Length of the transform.  If ``n < x.shape[axis]``, `x` is
    truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
    default results in ``n = x.shape[axis]``.
axis : int, optional
    Axis along which the idst is computed; the default is over the
    last axis (i.e., ``axis=-1``).
norm : {None, 'ortho'}, optional
    Normalization mode (see Notes). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.

Returns
-------
idst : ndarray of real
    The transformed input array.

See Also
--------
dst : Forward DST

Notes
-----
'The' IDST is the IDST of type 2, which is the same as DST of type 3.

IDST of type 1 is the DST of type 1, IDST of type 2 is the DST of type
3, and IDST of type 3 is the DST of type 2. For the definition of these
types, see `dst`.

.. versionadded:: 0.11.0
*)

val idstn : ?type_:[`Three | `One | `Four | `Two] -> ?shape:[`I of int | `Array_like_of_ints of Py.Object.t] -> ?axes:[`I of int | `Array_like_of_ints of Py.Object.t] -> ?norm:string -> ?overwrite_x:bool -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Return multidimensional Discrete Sine Transform along the specified axes.

Parameters
----------
x : array_like
    The input array.
type : {1, 2, 3, 4}, optional
    Type of the DST (see Notes). Default type is 2.
shape : int or array_like of ints or None, optional
    The shape of the result.  If both `shape` and `axes` (see below) are
    None, `shape` is ``x.shape``; if `shape` is None but `axes` is
    not None, then `shape` is ``scipy.take(x.shape, axes, axis=0)``.
    If ``shape[i] > x.shape[i]``, the i-th dimension is padded with zeros.
    If ``shape[i] < x.shape[i]``, the i-th dimension is truncated to
    length ``shape[i]``.
    If any element of `shape` is -1, the size of the corresponding
    dimension of `x` is used.
axes : int or array_like of ints or None, optional
    Axes along which the IDST is computed.
    The default is over all axes.
norm : {None, 'ortho'}, optional
    Normalization mode (see Notes). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.

Returns
-------
y : ndarray of real
    The transformed input array.

See Also
--------
dstn : multidimensional DST

Notes
-----
For full details of the IDST types and normalization modes, as well as
references, see `idst`.

Examples
--------
>>> from scipy.fftpack import dstn, idstn
>>> y = np.random.randn(16, 16)
>>> np.allclose(y, idstn(dstn(y, norm='ortho'), norm='ortho'))
True
*)

val ifft : ?n:int -> ?axis:int -> ?overwrite_x:bool -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return discrete inverse Fourier transform of real or complex sequence.

The returned complex array contains ``y(0), y(1),..., y(n-1)`` where

``y(j) = (x * exp(2*pi*sqrt(-1)*j*np.arange(n)/n)).mean()``.

Parameters
----------
x : array_like
    Transformed data to invert.
n : int, optional
    Length of the inverse Fourier transform.  If ``n < x.shape[axis]``,
    `x` is truncated.  If ``n > x.shape[axis]``, `x` is zero-padded.
    The default results in ``n = x.shape[axis]``.
axis : int, optional
    Axis along which the ifft's are computed; the default is over the
    last axis (i.e., ``axis=-1``).
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.

Returns
-------
ifft : ndarray of floats
    The inverse discrete Fourier transform.

See Also
--------
fft : Forward FFT

Notes
-----
Both single and double precision routines are implemented.  Half precision
inputs will be converted to single precision.  Non floating-point inputs
will be converted to double precision.  Long-double precision inputs are
not supported.

This function is most efficient when `n` is a power of two, and least
efficient when `n` is prime.

If the data type of `x` is real, a 'real IFFT' algorithm is automatically
used, which roughly halves the computation time.

Examples
--------
>>> from scipy.fftpack import fft, ifft
>>> import numpy as np
>>> x = np.arange(5)
>>> np.allclose(ifft(fft(x)), x, atol=1e-15)  # within numerical accuracy.
True
*)

val ifft2 : ?shape:Py.Object.t -> ?axes:Py.Object.t -> ?overwrite_x:Py.Object.t -> x:Py.Object.t -> unit -> Py.Object.t
(**
2-D discrete inverse Fourier transform of real or complex sequence.

Return inverse two-dimensional discrete Fourier transform of
arbitrary type sequence x.

See `ifft` for more information.

See also
--------
fft2, ifft
*)

val ifftn : ?shape:Py.Object.t -> ?axes:Py.Object.t -> ?overwrite_x:Py.Object.t -> x:Py.Object.t -> unit -> Py.Object.t
(**
Return inverse multi-dimensional discrete Fourier transform.

The sequence can be of an arbitrary type.

The returned array contains::

  y[j_1,..,j_d] = 1/p * sum[k_1=0..n_1-1, ..., k_d=0..n_d-1]
     x[k_1,..,k_d] * prod[i=1..d] exp(sqrt(-1)*2*pi/n_i * j_i * k_i)

where ``d = len(x.shape)``, ``n = x.shape``, and ``p = prod[i=1..d] n_i``.

For description of parameters see `fftn`.

See Also
--------
fftn : for detailed information.

Examples
--------
>>> from scipy.fftpack import fftn, ifftn
>>> import numpy as np
>>> y = (-np.arange(16), 8 - np.arange(16), np.arange(16))
>>> np.allclose(y, ifftn(fftn(y)))
True
*)

val ifftshift : ?axes:[`Shape_tuple of Py.Object.t | `I of int] -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
The inverse of `fftshift`. Although identical for even-length `x`, the
functions differ by one sample for odd-length `x`.

Parameters
----------
x : array_like
    Input array.
axes : int or shape tuple, optional
    Axes over which to calculate.  Defaults to None, which shifts all axes.

Returns
-------
y : ndarray
    The shifted array.

See Also
--------
fftshift : Shift zero-frequency component to the center of the spectrum.

Examples
--------
>>> freqs = np.fft.fftfreq(9, d=1./9).reshape(3, 3)
>>> freqs
array([[ 0.,  1.,  2.],
       [ 3.,  4., -4.],
       [-3., -2., -1.]])
>>> np.fft.ifftshift(np.fft.fftshift(freqs))
array([[ 0.,  1.,  2.],
       [ 3.,  4., -4.],
       [-3., -2., -1.]])
*)

val ihilbert : Py.Object.t -> Py.Object.t
(**
Return inverse Hilbert transform of a periodic sequence x.

If ``x_j`` and ``y_j`` are Fourier coefficients of periodic functions x
and y, respectively, then::

  y_j = -sqrt(-1)*sign(j) * x_j
  y_0 = 0
*)

val irfft : ?n:int -> ?axis:int -> ?overwrite_x:bool -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return inverse discrete Fourier transform of real sequence x.

The contents of `x` are interpreted as the output of the `rfft`
function.

Parameters
----------
x : array_like
    Transformed data to invert.
n : int, optional
    Length of the inverse Fourier transform.
    If n < x.shape[axis], x is truncated.
    If n > x.shape[axis], x is zero-padded.
    The default results in n = x.shape[axis].
axis : int, optional
    Axis along which the ifft's are computed; the default is over
    the last axis (i.e., axis=-1).
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.

Returns
-------
irfft : ndarray of floats
    The inverse discrete Fourier transform.

See Also
--------
rfft, ifft, scipy.fft.irfft

Notes
-----
The returned real array contains::

    [y(0),y(1),...,y(n-1)]

where for n is even::

    y(j) = 1/n (sum[k=1..n/2-1] (x[2*k-1]+sqrt(-1)*x[2*k])
                                 * exp(sqrt(-1)*j*k* 2*pi/n)
                + c.c. + x[0] + (-1)**(j) x[n-1])

and for n is odd::

    y(j) = 1/n (sum[k=1..(n-1)/2] (x[2*k-1]+sqrt(-1)*x[2*k])
                                 * exp(sqrt(-1)*j*k* 2*pi/n)
                + c.c. + x[0])

c.c. denotes complex conjugate of preceding expression.

For details on input parameters, see `rfft`.

To process (conjugate-symmetric) frequency-domain data with a complex
datatype, consider using the newer function `scipy.fft.irfft`.

Examples
--------
>>> from scipy.fftpack import rfft, irfft
>>> a = [1.0, 2.0, 3.0, 4.0, 5.0]
>>> irfft(a)
array([ 2.6       , -3.16405192,  1.24398433, -1.14955713,  1.46962473])
>>> irfft(rfft(a))
array([1., 2., 3., 4., 5.])
*)

val itilbert : ?period:Py.Object.t -> ?_cache:Py.Object.t -> x:Py.Object.t -> h:Py.Object.t -> unit -> Py.Object.t
(**
Return inverse h-Tilbert transform of a periodic sequence x.

If ``x_j`` and ``y_j`` are Fourier coefficients of periodic functions x
and y, respectively, then::

  y_j = -sqrt(-1)*tanh(j*h*2*pi/period) * x_j
  y_0 = 0

For more details, see `tilbert`.
*)

val next_fast_len : int -> int
(**
Find the next fast size of input data to `fft`, for zero-padding, etc.

SciPy's FFTPACK has efficient functions for radix {2, 3, 4, 5}, so this
returns the next composite of the prime factors 2, 3, and 5 which is
greater than or equal to `target`. (These are also known as 5-smooth
numbers, regular numbers, or Hamming numbers.)

Parameters
----------
target : int
    Length to start searching from.  Must be a positive integer.

Returns
-------
out : int
    The first 5-smooth number greater than or equal to `target`.

Notes
-----
.. versionadded:: 0.18.0

Examples
--------
On a particular machine, an FFT of prime length takes 133 ms:

>>> from scipy import fftpack
>>> min_len = 10007  # prime length is worst case for speed
>>> a = np.random.randn(min_len)
>>> b = fftpack.fft(a)

Zero-padding to the next 5-smooth length reduces computation time to
211 us, a speedup of 630 times:

>>> fftpack.helper.next_fast_len(min_len)
10125
>>> b = fftpack.fft(a, 10125)

Rounding up to the next power of 2 is not optimal, taking 367 us to
compute, 1.7 times as long as the 5-smooth size:

>>> b = fftpack.fft(a, 16384)
*)

val rfft : ?n:int -> ?axis:int -> ?overwrite_x:bool -> x:[`Ndarray of [>`Ndarray] Np.Obj.t | `Real_valued of Py.Object.t] -> unit -> (Py.Object.t * Py.Object.t)
(**
Discrete Fourier transform of a real sequence.

Parameters
----------
x : array_like, real-valued
    The data to transform.
n : int, optional
    Defines the length of the Fourier transform.  If `n` is not specified
    (the default) then ``n = x.shape[axis]``.  If ``n < x.shape[axis]``,
    `x` is truncated, if ``n > x.shape[axis]``, `x` is zero-padded.
axis : int, optional
    The axis along which the transform is applied.  The default is the
    last axis.
overwrite_x : bool, optional
    If set to true, the contents of `x` can be overwritten. Default is
    False.

Returns
-------
z : real ndarray
    The returned real array contains::

      [y(0),Re(y(1)),Im(y(1)),...,Re(y(n/2))]              if n is even
      [y(0),Re(y(1)),Im(y(1)),...,Re(y(n/2)),Im(y(n/2))]   if n is odd

    where::

      y(j) = sum[k=0..n-1] x[k] * exp(-sqrt(-1)*j*k*2*pi/n)
      j = 0..n-1

See Also
--------
fft, irfft, scipy.fft.rfft

Notes
-----
Within numerical accuracy, ``y == rfft(irfft(y))``.

Both single and double precision routines are implemented.  Half precision
inputs will be converted to single precision.  Non floating-point inputs
will be converted to double precision.  Long-double precision inputs are
not supported.

To get an output with a complex datatype, consider using the newer
function `scipy.fft.rfft`.

Examples
--------
>>> from scipy.fftpack import fft, rfft
>>> a = [9, -9, 1, 3]
>>> fft(a)
array([  4. +0.j,   8.+12.j,  16. +0.j,   8.-12.j])
>>> rfft(a)
array([  4.,   8.,  12.,  16.])
*)

val rfftfreq : ?d:[`F of float | `I of int | `Bool of bool | `S of string] -> n:int -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
DFT sample frequencies (for usage with rfft, irfft).

The returned float array contains the frequency bins in
cycles/unit (with zero at the start) given a window length `n` and a
sample spacing `d`::

  f = [0,1,1,2,2,...,n/2-1,n/2-1,n/2]/(d*n)   if n is even
  f = [0,1,1,2,2,...,n/2-1,n/2-1,n/2,n/2]/(d*n)   if n is odd

Parameters
----------
n : int
    Window length.
d : scalar, optional
    Sample spacing. Default is 1.

Returns
-------
out : ndarray
    The array of length `n`, containing the sample frequencies.

Examples
--------
>>> from scipy import fftpack
>>> sig = np.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
>>> sig_fft = fftpack.rfft(sig)
>>> n = sig_fft.size
>>> timestep = 0.1
>>> freq = fftpack.rfftfreq(n, d=timestep)
>>> freq
array([ 0.  ,  1.25,  1.25,  2.5 ,  2.5 ,  3.75,  3.75,  5.  ])
*)

val sc_diff : ?period:float -> ?_cache:Py.Object.t -> x:[>`Ndarray] Np.Obj.t -> a:Py.Object.t -> b:Py.Object.t -> unit -> Py.Object.t
(**
Return (a,b)-sinh/cosh pseudo-derivative of a periodic sequence x.

If x_j and y_j are Fourier coefficients of periodic functions x
and y, respectively, then::

  y_j = sqrt(-1)*sinh(j*a*2*pi/period)/cosh(j*b*2*pi/period) * x_j
  y_0 = 0

Parameters
----------
x : array_like
    Input array.
a,b : float
    Defines the parameters of the sinh/cosh pseudo-differential
    operator.
period : float, optional
    The period of the sequence x. Default is 2*pi.

Notes
-----
``sc_diff(cs_diff(x,a,b),b,a) == x``
For even ``len(x)``, the Nyquist mode of x is taken as zero.
*)

val shift : ?period:float -> ?_cache:Py.Object.t -> x:[>`Ndarray] Np.Obj.t -> a:float -> unit -> Py.Object.t
(**
Shift periodic sequence x by a: y(u) = x(u+a).

If x_j and y_j are Fourier coefficients of periodic functions x
and y, respectively, then::

      y_j = exp(j*a*2*pi/period*sqrt(-1)) * x_f

Parameters
----------
x : array_like
    The array to take the pseudo-derivative from.
a : float
    Defines the parameters of the sinh/sinh pseudo-differential
period : float, optional
    The period of the sequences x and y. Default period is ``2*pi``.
*)

val ss_diff : ?period:float -> ?_cache:Py.Object.t -> x:[>`Ndarray] Np.Obj.t -> a:Py.Object.t -> b:Py.Object.t -> unit -> Py.Object.t
(**
Return (a,b)-sinh/sinh pseudo-derivative of a periodic sequence x.

If x_j and y_j are Fourier coefficients of periodic functions x
and y, respectively, then::

  y_j = sinh(j*a*2*pi/period)/sinh(j*b*2*pi/period) * x_j
  y_0 = a/b * x_0

Parameters
----------
x : array_like
    The array to take the pseudo-derivative from.
a,b
    Defines the parameters of the sinh/sinh pseudo-differential
    operator.
period : float, optional
    The period of the sequence x. Default is ``2*pi``.

Notes
-----
``ss_diff(ss_diff(x,a,b),b,a) == x``
*)

val tilbert : ?period:float -> ?_cache:Py.Object.t -> x:[>`Ndarray] Np.Obj.t -> h:float -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return h-Tilbert transform of a periodic sequence x.

If x_j and y_j are Fourier coefficients of periodic functions x
and y, respectively, then::

    y_j = sqrt(-1)*coth(j*h*2*pi/period) * x_j
    y_0 = 0

Parameters
----------
x : array_like
    The input array to transform.
h : float
    Defines the parameter of the Tilbert transform.
period : float, optional
    The assumed period of the sequence.  Default period is ``2*pi``.

Returns
-------
tilbert : ndarray
    The result of the transform.

Notes
-----
If ``sum(x, axis=0) == 0`` and ``n = len(x)`` is odd then
``tilbert(itilbert(x)) == x``.

If ``2 * pi * h / period`` is approximately 10 or larger, then
numerically ``tilbert == hilbert``
(theoretically oo-Tilbert == Hilbert).

For even ``len(x)``, the Nyquist mode of ``x`` is taken zero.
*)

