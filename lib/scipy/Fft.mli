(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val dct : ?type_:[`Three | `One | `Four | `Two] -> ?n:int -> ?axis:int -> ?norm:string -> ?overwrite_x:bool -> ?workers:int -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
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
workers : int, optional
    Maximum number of workers to use for parallel computation. If negative,
    the value wraps around from ``os.cpu_count()``.
    See :func:`~scipy.fft.fft` for more details.

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

For ``norm=None``, there is no scaling on `dct` and the `idct` is scaled by
``1/N`` where ``N`` is the 'logical' size of the DCT. For ``norm='ortho'``
both directions are scaled by the same factor ``1/sqrt(N)``.

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

>>> from scipy.fft import fft, dct
>>> fft(np.array([4., 3., 5., 10., 5., 3.])).real
array([ 30.,  -8.,   6.,  -2.,   6.,  -8.])
>>> dct(np.array([4., 3., 5., 10.]), 1)
array([ 30.,  -8.,   6.,  -2.])
*)

val dctn : ?type_:[`Three | `One | `Four | `Two] -> ?s:[`I of int | `Array_like_of_ints of Py.Object.t] -> ?axes:[`I of int | `Array_like_of_ints of Py.Object.t] -> ?norm:string -> ?overwrite_x:bool -> ?workers:int -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Return multidimensional Discrete Cosine Transform along the specified axes.

Parameters
----------
x : array_like
    The input array.
type : {1, 2, 3, 4}, optional
    Type of the DCT (see Notes). Default type is 2.
s : int or array_like of ints or None, optional
    The shape of the result.  If both `s` and `axes` (see below) are None,
    `s` is ``x.shape``; if `s` is None but `axes` is not None, then `s` is
    ``scipy.take(x.shape, axes, axis=0)``.
    If ``s[i] > x.shape[i]``, the i-th dimension is padded with zeros.
    If ``s[i] < x.shape[i]``, the i-th dimension is truncated to length
    ``s[i]``.
    If any element of `s` is -1, the size of the corresponding dimension of
    `x` is used.
axes : int or array_like of ints or None, optional
    Axes over which the DCT is computed.  If not given, the last ``len(s)``
    axes are used, or all axes if `s` is also not specified.
norm : {None, 'ortho'}, optional
    Normalization mode (see Notes). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.
workers : int, optional
    Maximum number of workers to use for parallel computation. If negative,
    the value wraps around from ``os.cpu_count()``.
    See :func:`~scipy.fft.fft` for more details.

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
>>> from scipy.fft import dctn, idctn
>>> y = np.random.randn(16, 16)
>>> np.allclose(y, idctn(dctn(y)))
True
*)

val dst : ?type_:[`Three | `One | `Four | `Two] -> ?n:int -> ?axis:int -> ?norm:string -> ?overwrite_x:bool -> ?workers:int -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
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
workers : int, optional
    Maximum number of workers to use for parallel computation. If negative,
    the value wraps around from ``os.cpu_count()``.
    See :func:`~scipy.fft.fft` for more details.

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

For ``norm=None``, there is no scaling on the `dst` and the `idst` is
scaled by ``1/N`` where ``N`` is the 'logical' size of the DST. For
``norm='ortho'`` both directions are scaled by the same factor
``1/sqrt(N)``.

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

**Type IV**

There are several definitions of the DST-IV, we use the following (for
``norm=None``). DST-IV assumes the input is odd around `n=-0.5` and even
around `n=N-0.5`

.. math::

    y_k = 2 \sum_{n=0}^{N-1} x_n \sin\left(\frac{\pi(2k+1)(2n+1)}{4N}\right)

The (unnormalized) DST-IV is its own inverse, up to a factor `2N`. The
orthonormalized DST-IV is exactly its own inverse.

References
----------
.. [1] Wikipedia, 'Discrete sine transform',
       https://en.wikipedia.org/wiki/Discrete_sine_transform
*)

val dstn : ?type_:[`Three | `One | `Four | `Two] -> ?s:[`I of int | `Array_like_of_ints of Py.Object.t] -> ?axes:[`I of int | `Array_like_of_ints of Py.Object.t] -> ?norm:string -> ?overwrite_x:bool -> ?workers:int -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Return multidimensional Discrete Sine Transform along the specified axes.

Parameters
----------
x : array_like
    The input array.
type : {1, 2, 3, 4}, optional
    Type of the DST (see Notes). Default type is 2.
s : int or array_like of ints or None, optional
    The shape of the result.  If both `s` and `axes` (see below) are None,
    `s` is ``x.shape``; if `s` is None but `axes` is not None, then `s` is
    ``scipy.take(x.shape, axes, axis=0)``.
    If ``s[i] > x.shape[i]``, the i-th dimension is padded with zeros.
    If ``s[i] < x.shape[i]``, the i-th dimension is truncated to length
    ``s[i]``.
    If any element of `shape` is -1, the size of the corresponding dimension
    of `x` is used.
axes : int or array_like of ints or None, optional
    Axes over which the DST is computed.  If not given, the last ``len(s)``
    axes are used, or all axes if `s` is also not specified.
norm : {None, 'ortho'}, optional
    Normalization mode (see Notes). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.
workers : int, optional
    Maximum number of workers to use for parallel computation. If negative,
    the value wraps around from ``os.cpu_count()``.
    See :func:`~scipy.fft.fft` for more details.

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
>>> from scipy.fft import dstn, idstn
>>> y = np.random.randn(16, 16)
>>> np.allclose(y, idstn(dstn(y)))
True
*)

val fft : ?n:int -> ?axis:int -> ?norm:string -> ?overwrite_x:bool -> ?workers:int -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Compute the one-dimensional discrete Fourier Transform.

This function computes the one-dimensional *n*-point discrete Fourier
Transform (DFT) with the efficient Fast Fourier Transform (FFT)
algorithm [1]_.

Parameters
----------
x : array_like
    Input array, can be complex.
n : int, optional
    Length of the transformed axis of the output.
    If `n` is smaller than the length of the input, the input is cropped.
    If it is larger, the input is padded with zeros.  If `n` is not given,
    the length of the input along the axis specified by `axis` is used.
axis : int, optional
    Axis over which to compute the FFT.  If not given, the last axis is
    used.
norm : {None, 'ortho'}, optional
    Normalization mode. Default is None, meaning no normalization on the
    forward transforms and scaling by ``1/n`` on the `ifft`.
    For ``norm='ortho'``, both directions are scaled by ``1/sqrt(n)``.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.
    See the notes below for more details.
workers : int, optional
    Maximum number of workers to use for parallel computation. If negative,
    the value wraps around from ``os.cpu_count()``. See below for more
    details.

Returns
-------
out : complex ndarray
    The truncated or zero-padded input, transformed along the axis
    indicated by `axis`, or the last one if `axis` is not specified.

Raises
------
IndexError
    if `axes` is larger than the last axis of `x`.

See Also
--------
ifft : The inverse of `fft`.
fft2 : The two-dimensional FFT.
fftn : The *n*-dimensional FFT.
rfftn : The *n*-dimensional FFT of real input.
fftfreq : Frequency bins for given FFT parameters.
next_fast_len : Size to pad input to for most efficient transforms

Notes
-----

FFT (Fast Fourier Transform) refers to a way the discrete Fourier Transform
(DFT) can be calculated efficiently, by using symmetries in the calculated
terms. The symmetry is highest when `n` is a power of 2, and the transform
is therefore most efficient for these sizes. For poorly factorizable sizes,
`scipy.fft` uses Bluestein's algorithm [2]_ and so is never worse than
O(`n` log `n`). Further performance improvements may be seen by zero-padding
the input using `next_fast_len`.

If ``x`` is a 1d array, then the `fft` is equivalent to ::

    y[k] = np.sum(x * np.exp(-2j * np.pi * k * np.arange(n)/n))

The frequency term ``f=k/n`` is found at ``y[k]``. At ``y[n/2]`` we reach
the Nyquist frequency and wrap around to the negative-frequency terms. So,
for an 8-point transform, the frequencies of the result are
[0, 1, 2, 3, -4, -3, -2, -1]. To rearrange the fft output so that the
zero-frequency component is centered, like [-4, -3, -2, -1, 0, 1, 2, 3],
use `fftshift`.

Transforms can be done in single, double or extended precision (long
double) floating point. Half precision inputs will be converted to single
precision and non floating-point inputs will be converted to double
precision.

If the data type of ``x`` is real, a 'real FFT' algorithm is automatically
used, which roughly halves the computation time.  To increase efficiency
a little further, use `rfft`, which does the same calculation, but only
outputs half of the symmetrical spectrum.  If the data is both real and
symmetrical, the `dct` can again double the efficiency, by generating
half of the spectrum from half of the signal.

When ``overwrite_x=True`` is specified, the memory referenced by ``x`` may
be used by the implementation in any way. This may include reusing the
memory for the result, but this is in no way guaranteed. You should not
rely on the contents of ``x`` after the transform as this may change in
future without warning.

The ``workers`` argument specifies the maximum number of parallel jobs to
split the FFT computation into. This will execute independent 1-dimensional
FFTs within ``x``. So, ``x`` must be at least 2-dimensional and the
non-transformed axes must be large enough to split into chunks. If ``x`` is
too small, fewer jobs may be used than requested.

References
----------
.. [1] Cooley, James W., and John W. Tukey, 1965, 'An algorithm for the
       machine calculation of complex Fourier series,' *Math. Comput.*
       19: 297-301.
.. [2] Bluestein, L., 1970, 'A linear filtering approach to the
       computation of discrete Fourier transform'. *IEEE Transactions on
       Audio and Electroacoustics.* 18 (4): 451-455.

Examples
--------
>>> import scipy.fft
>>> scipy.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8))
array([-2.33486982e-16+1.14423775e-17j,  8.00000000e+00-1.25557246e-15j,
        2.33486982e-16+2.33486982e-16j,  0.00000000e+00+1.22464680e-16j,
       -1.14423775e-17+2.33486982e-16j,  0.00000000e+00+5.20784380e-16j,
        1.14423775e-17+1.14423775e-17j,  0.00000000e+00+1.22464680e-16j])

In this example, real input has an FFT which is Hermitian, i.e., symmetric
in the real part and anti-symmetric in the imaginary part:

>>> from scipy.fft import fft, fftfreq, fftshift
>>> import matplotlib.pyplot as plt
>>> t = np.arange(256)
>>> sp = fftshift(fft(np.sin(t)))
>>> freq = fftshift(fftfreq(t.shape[-1]))
>>> plt.plot(freq, sp.real, freq, sp.imag)
[<matplotlib.lines.Line2D object at 0x...>, <matplotlib.lines.Line2D object at 0x...>]
>>> plt.show()
*)

val fft2 : ?s:int list -> ?axes:int list -> ?norm:string -> ?overwrite_x:bool -> ?workers:int -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Compute the 2-dimensional discrete Fourier Transform

This function computes the *n*-dimensional discrete Fourier Transform
over any axes in an *M*-dimensional array by means of the
Fast Fourier Transform (FFT).  By default, the transform is computed over
the last two axes of the input array, i.e., a 2-dimensional FFT.

Parameters
----------
x : array_like
    Input array, can be complex
s : sequence of ints, optional
    Shape (length of each transformed axis) of the output
    (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
    This corresponds to ``n`` for ``fft(x, n)``.
    Along each axis, if the given shape is smaller than that of the input,
    the input is cropped.  If it is larger, the input is padded with zeros.
    if `s` is not given, the shape of the input along the axes specified
    by `axes` is used.
axes : sequence of ints, optional
    Axes over which to compute the FFT. If not given, the last two axes are
    used.
norm : {None, 'ortho'}, optional
    Normalization mode (see `fft`). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.
    See :func:`fft` for more details.
workers : int, optional
    Maximum number of workers to use for parallel computation. If negative,
    the value wraps around from ``os.cpu_count()``.
    See :func:`~scipy.fft.fft` for more details.

Returns
-------
out : complex ndarray
    The truncated or zero-padded input, transformed along the axes
    indicated by `axes`, or the last two axes if `axes` is not given.

Raises
------
ValueError
    If `s` and `axes` have different length, or `axes` not given and
    ``len(s) != 2``.
IndexError
    If an element of `axes` is larger than than the number of axes of `x`.

See Also
--------
ifft2 : The inverse two-dimensional FFT.
fft : The one-dimensional FFT.
fftn : The *n*-dimensional FFT.
fftshift : Shifts zero-frequency terms to the center of the array.
    For two-dimensional input, swaps first and third quadrants, and second
    and fourth quadrants.

Notes
-----
`fft2` is just `fftn` with a different default for `axes`.

The output, analogously to `fft`, contains the term for zero frequency in
the low-order corner of the transformed axes, the positive frequency terms
in the first half of these axes, the term for the Nyquist frequency in the
middle of the axes and the negative frequency terms in the second half of
the axes, in order of decreasingly negative frequency.

See `fftn` for details and a plotting example, and `fft` for
definitions and conventions used.


Examples
--------
>>> import scipy.fft
>>> x = np.mgrid[:5, :5][0]
>>> scipy.fft.fft2(x)
array([[ 50.  +0.j        ,   0.  +0.j        ,   0.  +0.j        , # may vary
          0.  +0.j        ,   0.  +0.j        ],
       [-12.5+17.20477401j,   0.  +0.j        ,   0.  +0.j        ,
          0.  +0.j        ,   0.  +0.j        ],
       [-12.5 +4.0614962j ,   0.  +0.j        ,   0.  +0.j        ,
          0.  +0.j        ,   0.  +0.j        ],
       [-12.5 -4.0614962j ,   0.  +0.j        ,   0.  +0.j        ,
          0.  +0.j        ,   0.  +0.j        ],
       [-12.5-17.20477401j,   0.  +0.j        ,   0.  +0.j        ,
          0.  +0.j        ,   0.  +0.j        ]])
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

val fftn : ?s:int list -> ?axes:int list -> ?norm:string -> ?overwrite_x:bool -> ?workers:int -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Compute the N-dimensional discrete Fourier Transform.

This function computes the *N*-dimensional discrete Fourier Transform over
any number of axes in an *M*-dimensional array by means of the Fast Fourier
Transform (FFT).

Parameters
----------
x : array_like
    Input array, can be complex.
s : sequence of ints, optional
    Shape (length of each transformed axis) of the output
    (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
    This corresponds to ``n`` for ``fft(x, n)``.
    Along any axis, if the given shape is smaller than that of the input,
    the input is cropped.  If it is larger, the input is padded with zeros.
    if `s` is not given, the shape of the input along the axes specified
    by `axes` is used.
axes : sequence of ints, optional
    Axes over which to compute the FFT.  If not given, the last ``len(s)``
    axes are used, or all axes if `s` is also not specified.
norm : {None, 'ortho'}, optional
    Normalization mode (see `fft`). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.
    See :func:`fft` for more details.
workers : int, optional
    Maximum number of workers to use for parallel computation. If negative,
    the value wraps around from ``os.cpu_count()``.
    See :func:`~scipy.fft.fft` for more details.

Returns
-------
out : complex ndarray
    The truncated or zero-padded input, transformed along the axes
    indicated by `axes`, or by a combination of `s` and `x`,
    as explained in the parameters section above.

Raises
------
ValueError
    If `s` and `axes` have different length.
IndexError
    If an element of `axes` is larger than than the number of axes of `x`.

See Also
--------
ifftn : The inverse of `fftn`, the inverse *n*-dimensional FFT.
fft : The one-dimensional FFT, with definitions and conventions used.
rfftn : The *n*-dimensional FFT of real input.
fft2 : The two-dimensional FFT.
fftshift : Shifts zero-frequency terms to centre of array

Notes
-----
The output, analogously to `fft`, contains the term for zero frequency in
the low-order corner of all axes, the positive frequency terms in the
first half of all axes, the term for the Nyquist frequency in the middle
of all axes and the negative frequency terms in the second half of all
axes, in order of decreasingly negative frequency.

Examples
--------
>>> import scipy.fft
>>> x = np.mgrid[:3, :3, :3][0]
>>> scipy.fft.fftn(x, axes=(1, 2))
array([[[ 0.+0.j,   0.+0.j,   0.+0.j], # may vary
        [ 0.+0.j,   0.+0.j,   0.+0.j],
        [ 0.+0.j,   0.+0.j,   0.+0.j]],
       [[ 9.+0.j,   0.+0.j,   0.+0.j],
        [ 0.+0.j,   0.+0.j,   0.+0.j],
        [ 0.+0.j,   0.+0.j,   0.+0.j]],
       [[18.+0.j,   0.+0.j,   0.+0.j],
        [ 0.+0.j,   0.+0.j,   0.+0.j],
        [ 0.+0.j,   0.+0.j,   0.+0.j]]])
>>> scipy.fft.fftn(x, (2, 2), axes=(0, 1))
array([[[ 2.+0.j,  2.+0.j,  2.+0.j], # may vary
        [ 0.+0.j,  0.+0.j,  0.+0.j]],
       [[-2.+0.j, -2.+0.j, -2.+0.j],
        [ 0.+0.j,  0.+0.j,  0.+0.j]]])

>>> import matplotlib.pyplot as plt
>>> [X, Y] = np.meshgrid(2 * np.pi * np.arange(200) / 12,
...                      2 * np.pi * np.arange(200) / 34)
>>> S = np.sin(X) + np.cos(Y) + np.random.uniform(0, 1, X.shape)
>>> FS = scipy.fft.fftn(S)
>>> plt.imshow(np.log(np.abs(scipy.fft.fftshift(FS))**2))
<matplotlib.image.AxesImage object at 0x...>
>>> plt.show()
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

val get_workers : unit -> Py.Object.t
(**
Returns the default number of workers within the current context

Examples
--------
>>> from scipy import fft
>>> fft.get_workers()
1
>>> with fft.set_workers(4):
...     fft.get_workers()
4
*)

val hfft : ?n:int -> ?axis:int -> ?norm:string -> ?overwrite_x:bool -> ?workers:int -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the FFT of a signal that has Hermitian symmetry, i.e., a real
spectrum.

Parameters
----------
x : array_like
    The input array.
n : int, optional
    Length of the transformed axis of the output. For `n` output
    points, ``n//2 + 1`` input points are necessary.  If the input is
    longer than this, it is cropped.  If it is shorter than this, it is
    padded with zeros. If `n` is not given, it is taken to be ``2*(m-1)``
    where ``m`` is the length of the input along the axis specified by
    `axis`.
axis : int, optional
    Axis over which to compute the FFT. If not given, the last
    axis is used.
norm : {None, 'ortho'}, optional
    Normalization mode (see `fft`). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.
    See `fft` for more details.
workers : int, optional
    Maximum number of workers to use for parallel computation. If negative,
    the value wraps around from ``os.cpu_count()``.
    See :func:`~scipy.fft.fft` for more details.

Returns
-------
out : ndarray
    The truncated or zero-padded input, transformed along the axis
    indicated by `axis`, or the last one if `axis` is not specified.
    The length of the transformed axis is `n`, or, if `n` is not given,
    ``2*m - 2`` where ``m`` is the length of the transformed axis of
    the input. To get an odd number of output points, `n` must be
    specified, for instance as ``2*m - 1`` in the typical case,

Raises
------
IndexError
    If `axis` is larger than the last axis of `a`.

See also
--------
rfft : Compute the one-dimensional FFT for real input.
ihfft : The inverse of `hfft`.
hfftn : Compute the n-dimensional FFT of a Hermitian signal.

Notes
-----
`hfft`/`ihfft` are a pair analogous to `rfft`/`irfft`, but for the
opposite case: here the signal has Hermitian symmetry in the time
domain and is real in the frequency domain. So here it's `hfft` for
which you must supply the length of the result if it is to be odd.
* even: ``ihfft(hfft(a, 2*len(a) - 2) == a``, within roundoff error,
* odd: ``ihfft(hfft(a, 2*len(a) - 1) == a``, within roundoff error.

Examples
--------
>>> from scipy.fft import fft, hfft
>>> a = 2 * np.pi * np.arange(10) / 10
>>> signal = np.cos(a) + 3j * np.sin(3 * a)
>>> fft(signal).round(10)
array([ -0.+0.j,   5.+0.j,  -0.+0.j,  15.-0.j,   0.+0.j,   0.+0.j,
        -0.+0.j, -15.-0.j,   0.+0.j,   5.+0.j])
>>> hfft(signal[:6]).round(10) # Input first half of signal
array([  0.,   5.,   0.,  15.,  -0.,   0.,   0., -15.,  -0.,   5.])
>>> hfft(signal, 10)  # Input entire signal and truncate
array([  0.,   5.,   0.,  15.,  -0.,   0.,   0., -15.,  -0.,   5.])
*)

val hfft2 : ?s:int list -> ?axes:int list -> ?norm:string -> ?overwrite_x:bool -> ?workers:int -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the 2-dimensional FFT of a Hermitian complex array.

Parameters
----------
x : array
    Input array, taken to be Hermitian complex.
s : sequence of ints, optional
    Shape of the real output.
axes : sequence of ints, optional
    Axes over which to compute the FFT.
norm : {None, 'ortho'}, optional
    Normalization mode (see `fft`). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.
    See `fft` for more details.
workers : int, optional
    Maximum number of workers to use for parallel computation. If negative,
    the value wraps around from ``os.cpu_count()``.
    See :func:`~scipy.fft.fft` for more details.

Returns
-------
out : ndarray
    The real result of the 2D Hermitian complex real FFT.

See Also
--------
hfftn : Compute the N-dimensional discrete Fourier Transform for Hermitian
        complex input.

Notes
-----
This is really just `hfftn` with different default behavior.
For more details see `hfftn`.
*)

val hfftn : ?s:int list -> ?axes:int list -> ?norm:string -> ?overwrite_x:bool -> ?workers:int -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the N-dimensional FFT of Hermitian symmetric complex input, i.e. a
signal with a real spectrum.

This function computes the N-dimensional discrete Fourier Transform for a
Hermitian symmetric complex input over any number of axes in an
M-dimensional array by means of the Fast Fourier Transform (FFT). In other
words, ``ihfftn(hfftn(x, s)) == x`` to within numerical accuracy. (``s``
here is ``x.shape`` with ``s[-1] = x.shape[-1] * 2 - 1``, this is necessary
for the same reason ``x.shape`` would be necessary for `irfft`.)

Parameters
----------
x : array_like
    Input array.
s : sequence of ints, optional
    Shape (length of each transformed axis) of the output
    (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.). `s` is also the
    number of input points used along this axis, except for the last axis,
    where ``s[-1]//2+1`` points of the input are used.
    Along any axis, if the shape indicated by `s` is smaller than that of
    the input, the input is cropped.  If it is larger, the input is padded
    with zeros. If `s` is not given, the shape of the input along the axes
    specified by axes is used. Except for the last axis which is taken to be
    ``2*(m-1)`` where ``m`` is the length of the input along that axis.
axes : sequence of ints, optional
    Axes over which to compute the inverse FFT. If not given, the last
    `len(s)` axes are used, or all axes if `s` is also not specified.
norm : {None, 'ortho'}, optional
    Normalization mode (see `fft`). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.
    See :func:`fft` for more details.
workers : int, optional
    Maximum number of workers to use for parallel computation. If negative,
    the value wraps around from ``os.cpu_count()``.
    See :func:`~scipy.fft.fft` for more details.

Returns
-------
out : ndarray
    The truncated or zero-padded input, transformed along the axes
    indicated by `axes`, or by a combination of `s` or `x`,
    as explained in the parameters section above.
    The length of each transformed axis is as given by the corresponding
    element of `s`, or the length of the input in every axis except for the
    last one if `s` is not given.  In the final transformed axis the length
    of the output when `s` is not given is ``2*(m-1)`` where ``m`` is the
    length of the final transformed axis of the input.  To get an odd
    number of output points in the final axis, `s` must be specified.

Raises
------
ValueError
    If `s` and `axes` have different length.
IndexError
    If an element of `axes` is larger than than the number of axes of `x`.

See Also
--------
ihfftn : The inverse n-dimensional FFT with real spectrum. Inverse of `hfftn`.
fft : The one-dimensional FFT, with definitions and conventions used.
rfft : Forward FFT of real input

Notes
-----

For a 1 dimensional signal ``x`` to have a real spectrum, it must satisfy
the Hermitian property::

    x[i] == np.conj(x[-i]) for all i

This generalizes into higher dimensions by reflecting over each axis in
turn::

    x[i, j, k, ...] == np.conj(x[-i, -j, -k, ...]) for all i, j, k, ...

This should not be confused with a Hermitian matrix, for which the
transpose is it's own conjugate::

    x[i, j] == np.conj(x[j, i]) for all i, j


The default value of `s` assumes an even output length in the final
transformation axis. When performing the final complex to real
transformation, the Hermitian symmetry requires that the last imaginary
component along that axis must be 0 and so it is ignored. To avoid losing
information, the correct length of the real input *must* be given.

Examples
--------
>>> import scipy.fft
>>> x = np.ones((3, 2, 2))
>>> scipy.fft.hfftn(x)
array([[[12.,  0.],
        [ 0.,  0.]],
       [[ 0.,  0.],
        [ 0.,  0.]],
       [[ 0.,  0.],
        [ 0.,  0.]]])
*)

val idct : ?type_:[`Three | `One | `Four | `Two] -> ?n:int -> ?axis:int -> ?norm:string -> ?overwrite_x:bool -> ?workers:int -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
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
workers : int, optional
    Maximum number of workers to use for parallel computation. If negative,
    the value wraps around from ``os.cpu_count()``.
    See :func:`~scipy.fft.fft` for more details.

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

'The' IDCT is the IDCT-II, which is the same as the normalized DCT-III.

The IDCT is equivalent to a normal DCT except for the normalization and
type. DCT type 1 and 4 are their own inverse and DCTs 2 and 3 are each
other's inverses.

Examples
--------
The Type 1 DCT is equivalent to the DFT for real, even-symmetrical
inputs.  The output is also real and even-symmetrical.  Half of the IFFT
input is used to generate half of the IFFT output:

>>> from scipy.fft import ifft, idct
>>> ifft(np.array([ 30.,  -8.,   6.,  -2.,   6.,  -8.])).real
array([  4.,   3.,   5.,  10.,   5.,   3.])
>>> idct(np.array([ 30.,  -8.,   6.,  -2.]), 1)
array([  4.,   3.,   5.,  10.])
*)

val idctn : ?type_:[`Three | `One | `Four | `Two] -> ?s:[`I of int | `Array_like_of_ints of Py.Object.t] -> ?axes:[`I of int | `Array_like_of_ints of Py.Object.t] -> ?norm:string -> ?overwrite_x:bool -> ?workers:int -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Return multidimensional Discrete Cosine Transform along the specified axes.

Parameters
----------
x : array_like
    The input array.
type : {1, 2, 3, 4}, optional
    Type of the DCT (see Notes). Default type is 2.
s : int or array_like of ints or None, optional
    The shape of the result.  If both `s` and `axes` (see below) are
    None, `s` is ``x.shape``; if `s` is None but `axes` is
    not None, then `s` is ``scipy.take(x.shape, axes, axis=0)``.
    If ``s[i] > x.shape[i]``, the i-th dimension is padded with zeros.
    If ``s[i] < x.shape[i]``, the i-th dimension is truncated to length
    ``s[i]``.
    If any element of `s` is -1, the size of the corresponding dimension of
    `x` is used.
axes : int or array_like of ints or None, optional
    Axes over which the IDCT is computed.  If not given, the last ``len(s)``
    axes are used, or all axes if `s` is also not specified.
norm : {None, 'ortho'}, optional
    Normalization mode (see Notes). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.
workers : int, optional
    Maximum number of workers to use for parallel computation. If negative,
    the value wraps around from ``os.cpu_count()``.
    See :func:`~scipy.fft.fft` for more details.

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
>>> from scipy.fft import dctn, idctn
>>> y = np.random.randn(16, 16)
>>> np.allclose(y, idctn(dctn(y)))
True
*)

val idst : ?type_:[`Three | `One | `Four | `Two] -> ?n:int -> ?axis:int -> ?norm:string -> ?overwrite_x:bool -> ?workers:int -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
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
workers : int, optional
    Maximum number of workers to use for parallel computation. If negative,
    the value wraps around from ``os.cpu_count()``.
    See :func:`~scipy.fft.fft` for more details.

Returns
-------
idst : ndarray of real
    The transformed input array.

See Also
--------
dst : Forward DST

Notes
-----

'The' IDST is the IDST-II, which is the same as the normalized DST-III.

The IDST is equivalent to a normal DST except for the normalization and
type. DST type 1 and 4 are their own inverse and DSTs 2 and 3 are each
other's inverses.
*)

val idstn : ?type_:[`Three | `One | `Four | `Two] -> ?s:[`I of int | `Array_like_of_ints of Py.Object.t] -> ?axes:[`I of int | `Array_like_of_ints of Py.Object.t] -> ?norm:string -> ?overwrite_x:bool -> ?workers:int -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Return multidimensional Discrete Sine Transform along the specified axes.

Parameters
----------
x : array_like
    The input array.
type : {1, 2, 3, 4}, optional
    Type of the DST (see Notes). Default type is 2.
s : int or array_like of ints or None, optional
    The shape of the result.  If both `s` and `axes` (see below) are None,
    `s` is ``x.shape``; if `s` is None but `axes` is not None, then `s` is
    ``scipy.take(x.shape, axes, axis=0)``.
    If ``s[i] > x.shape[i]``, the i-th dimension is padded with zeros.
    If ``s[i] < x.shape[i]``, the i-th dimension is truncated to length
    ``s[i]``.
    If any element of `s` is -1, the size of the corresponding dimension of
    `x` is used.
axes : int or array_like of ints or None, optional
    Axes over which the IDST is computed.  If not given, the last ``len(s)``
    axes are used, or all axes if `s` is also not specified.
norm : {None, 'ortho'}, optional
    Normalization mode (see Notes). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.
workers : int, optional
    Maximum number of workers to use for parallel computation. If negative,
    the value wraps around from ``os.cpu_count()``.
    See :func:`~scipy.fft.fft` for more details.

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
>>> from scipy.fft import dstn, idstn
>>> y = np.random.randn(16, 16)
>>> np.allclose(y, idstn(dstn(y)))
True
*)

val ifft : ?n:int -> ?axis:int -> ?norm:string -> ?overwrite_x:bool -> ?workers:int -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Compute the one-dimensional inverse discrete Fourier Transform.

This function computes the inverse of the one-dimensional *n*-point
discrete Fourier transform computed by `fft`.  In other words,
``ifft(fft(x)) == x`` to within numerical accuracy.

The input should be ordered in the same way as is returned by `fft`,
i.e.,

* ``x[0]`` should contain the zero frequency term,
* ``x[1:n//2]`` should contain the positive-frequency terms,
* ``x[n//2 + 1:]`` should contain the negative-frequency terms, in
  increasing order starting from the most negative frequency.

For an even number of input points, ``x[n//2]`` represents the sum of
the values at the positive and negative Nyquist frequencies, as the two
are aliased together. See `fft` for details.

Parameters
----------
x : array_like
    Input array, can be complex.
n : int, optional
    Length of the transformed axis of the output.
    If `n` is smaller than the length of the input, the input is cropped.
    If it is larger, the input is padded with zeros.  If `n` is not given,
    the length of the input along the axis specified by `axis` is used.
    See notes about padding issues.
axis : int, optional
    Axis over which to compute the inverse DFT.  If not given, the last
    axis is used.
norm : {None, 'ortho'}, optional
    Normalization mode (see `fft`). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.
    See :func:`fft` for more details.
workers : int, optional
    Maximum number of workers to use for parallel computation. If negative,
    the value wraps around from ``os.cpu_count()``.
    See :func:`~scipy.fft.fft` for more details.

Returns
-------
out : complex ndarray
    The truncated or zero-padded input, transformed along the axis
    indicated by `axis`, or the last one if `axis` is not specified.

Raises
------
IndexError
    If `axes` is larger than the last axis of `x`.

See Also
--------
fft : The one-dimensional (forward) FFT, of which `ifft` is the inverse
ifft2 : The two-dimensional inverse FFT.
ifftn : The n-dimensional inverse FFT.

Notes
-----
If the input parameter `n` is larger than the size of the input, the input
is padded by appending zeros at the end.  Even though this is the common
approach, it might lead to surprising results.  If a different padding is
desired, it must be performed before calling `ifft`.

If ``x`` is a 1d array, then the `ifft` is equivalent to ::

    y[k] = np.sum(x * np.exp(2j * np.pi * k * np.arange(n)/n)) / len(x)

As with `fft`, `ifft` has support for all floating point types and is
optimized for real input.

Examples
--------
>>> import scipy.fft
>>> scipy.fft.ifft([0, 4, 0, 0])
array([ 1.+0.j,  0.+1.j, -1.+0.j,  0.-1.j]) # may vary

Create and plot a band-limited signal with random phases:

>>> import matplotlib.pyplot as plt
>>> t = np.arange(400)
>>> n = np.zeros((400,), dtype=complex)
>>> n[40:60] = np.exp(1j*np.random.uniform(0, 2*np.pi, (20,)))
>>> s = scipy.fft.ifft(n)
>>> plt.plot(t, s.real, 'b-', t, s.imag, 'r--')
[<matplotlib.lines.Line2D object at ...>, <matplotlib.lines.Line2D object at ...>]
>>> plt.legend(('real', 'imaginary'))
<matplotlib.legend.Legend object at ...>
>>> plt.show()
*)

val ifft2 : ?s:int list -> ?axes:int list -> ?norm:string -> ?overwrite_x:bool -> ?workers:int -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Compute the 2-dimensional inverse discrete Fourier Transform.

This function computes the inverse of the 2-dimensional discrete Fourier
Transform over any number of axes in an M-dimensional array by means of
the Fast Fourier Transform (FFT).  In other words, ``ifft2(fft2(x)) == x``
to within numerical accuracy.  By default, the inverse transform is
computed over the last two axes of the input array.

The input, analogously to `ifft`, should be ordered in the same way as is
returned by `fft2`, i.e. it should have the term for zero frequency
in the low-order corner of the two axes, the positive frequency terms in
the first half of these axes, the term for the Nyquist frequency in the
middle of the axes and the negative frequency terms in the second half of
both axes, in order of decreasingly negative frequency.

Parameters
----------
x : array_like
    Input array, can be complex.
s : sequence of ints, optional
    Shape (length of each axis) of the output (``s[0]`` refers to axis 0,
    ``s[1]`` to axis 1, etc.).  This corresponds to `n` for ``ifft(x, n)``.
    Along each axis, if the given shape is smaller than that of the input,
    the input is cropped.  If it is larger, the input is padded with zeros.
    if `s` is not given, the shape of the input along the axes specified
    by `axes` is used.  See notes for issue on `ifft` zero padding.
axes : sequence of ints, optional
    Axes over which to compute the FFT.  If not given, the last two
    axes are used.
norm : {None, 'ortho'}, optional
    Normalization mode (see `fft`). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.
    See :func:`fft` for more details.
workers : int, optional
    Maximum number of workers to use for parallel computation. If negative,
    the value wraps around from ``os.cpu_count()``.
    See :func:`~scipy.fft.fft` for more details.

Returns
-------
out : complex ndarray
    The truncated or zero-padded input, transformed along the axes
    indicated by `axes`, or the last two axes if `axes` is not given.

Raises
------
ValueError
    If `s` and `axes` have different length, or `axes` not given and
    ``len(s) != 2``.
IndexError
    If an element of `axes` is larger than than the number of axes of `x`.

See Also
--------
fft2 : The forward 2-dimensional FFT, of which `ifft2` is the inverse.
ifftn : The inverse of the *n*-dimensional FFT.
fft : The one-dimensional FFT.
ifft : The one-dimensional inverse FFT.

Notes
-----
`ifft2` is just `ifftn` with a different default for `axes`.

See `ifftn` for details and a plotting example, and `fft` for
definition and conventions used.

Zero-padding, analogously with `ifft`, is performed by appending zeros to
the input along the specified dimension.  Although this is the common
approach, it might lead to surprising results.  If another form of zero
padding is desired, it must be performed before `ifft2` is called.

Examples
--------
>>> import scipy.fft
>>> x = 4 * np.eye(4)
>>> scipy.fft.ifft2(x)
array([[1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j], # may vary
       [0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j],
       [0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
       [0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j]])
*)

val ifftn : ?s:int list -> ?axes:int list -> ?norm:string -> ?overwrite_x:bool -> ?workers:int -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Compute the N-dimensional inverse discrete Fourier Transform.

This function computes the inverse of the N-dimensional discrete
Fourier Transform over any number of axes in an M-dimensional array by
means of the Fast Fourier Transform (FFT).  In other words,
``ifftn(fftn(x)) == x`` to within numerical accuracy.

The input, analogously to `ifft`, should be ordered in the same way as is
returned by `fftn`, i.e. it should have the term for zero frequency
in all axes in the low-order corner, the positive frequency terms in the
first half of all axes, the term for the Nyquist frequency in the middle
of all axes and the negative frequency terms in the second half of all
axes, in order of decreasingly negative frequency.

Parameters
----------
x : array_like
    Input array, can be complex.
s : sequence of ints, optional
    Shape (length of each transformed axis) of the output
    (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
    This corresponds to ``n`` for ``ifft(x, n)``.
    Along any axis, if the given shape is smaller than that of the input,
    the input is cropped.  If it is larger, the input is padded with zeros.
    if `s` is not given, the shape of the input along the axes specified
    by `axes` is used.  See notes for issue on `ifft` zero padding.
axes : sequence of ints, optional
    Axes over which to compute the IFFT.  If not given, the last ``len(s)``
    axes are used, or all axes if `s` is also not specified.
norm : {None, 'ortho'}, optional
    Normalization mode (see `fft`). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.
    See :func:`fft` for more details.
workers : int, optional
    Maximum number of workers to use for parallel computation. If negative,
    the value wraps around from ``os.cpu_count()``.
    See :func:`~scipy.fft.fft` for more details.

Returns
-------
out : complex ndarray
    The truncated or zero-padded input, transformed along the axes
    indicated by `axes`, or by a combination of `s` or `x`,
    as explained in the parameters section above.

Raises
------
ValueError
    If `s` and `axes` have different length.
IndexError
    If an element of `axes` is larger than than the number of axes of `x`.

See Also
--------
fftn : The forward *n*-dimensional FFT, of which `ifftn` is the inverse.
ifft : The one-dimensional inverse FFT.
ifft2 : The two-dimensional inverse FFT.
ifftshift : Undoes `fftshift`, shifts zero-frequency terms to beginning
    of array.

Notes
-----
Zero-padding, analogously with `ifft`, is performed by appending zeros to
the input along the specified dimension.  Although this is the common
approach, it might lead to surprising results.  If another form of zero
padding is desired, it must be performed before `ifftn` is called.

Examples
--------
>>> import scipy.fft
>>> x = np.eye(4)
>>> scipy.fft.ifftn(scipy.fft.fftn(x, axes=(0,)), axes=(1,))
array([[1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j], # may vary
       [0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
       [0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
       [0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j]])


Create and plot an image with band-limited frequency content:

>>> import matplotlib.pyplot as plt
>>> n = np.zeros((200,200), dtype=complex)
>>> n[60:80, 20:40] = np.exp(1j*np.random.uniform(0, 2*np.pi, (20, 20)))
>>> im = scipy.fft.ifftn(n).real
>>> plt.imshow(im)
<matplotlib.image.AxesImage object at 0x...>
>>> plt.show()
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

val ihfft : ?n:int -> ?axis:int -> ?norm:string -> ?overwrite_x:bool -> ?workers:int -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Compute the inverse FFT of a signal that has Hermitian symmetry.

Parameters
----------
x : array_like
    Input array.
n : int, optional
    Length of the inverse FFT, the number of points along
    transformation axis in the input to use.  If `n` is smaller than
    the length of the input, the input is cropped.  If it is larger,
    the input is padded with zeros. If `n` is not given, the length of
    the input along the axis specified by `axis` is used.
axis : int, optional
    Axis over which to compute the inverse FFT. If not given, the last
    axis is used.
norm : {None, 'ortho'}, optional
    Normalization mode (see `fft`). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.
    See `fft` for more details.
workers : int, optional
    Maximum number of workers to use for parallel computation. If negative,
    the value wraps around from ``os.cpu_count()``.
    See :func:`~scipy.fft.fft` for more details.

Returns
-------
out : complex ndarray
    The truncated or zero-padded input, transformed along the axis
    indicated by `axis`, or the last one if `axis` is not specified.
    The length of the transformed axis is ``n//2 + 1``.

See also
--------
hfft, irfft

Notes
-----
`hfft`/`ihfft` are a pair analogous to `rfft`/`irfft`, but for the
opposite case: here the signal has Hermitian symmetry in the time
domain and is real in the frequency domain. So here it's `hfft` for
which you must supply the length of the result if it is to be odd:
* even: ``ihfft(hfft(a, 2*len(a) - 2) == a``, within roundoff error,
* odd: ``ihfft(hfft(a, 2*len(a) - 1) == a``, within roundoff error.

Examples
--------
>>> from scipy.fft import ifft, ihfft
>>> spectrum = np.array([ 15, -4, 0, -1, 0, -4])
>>> ifft(spectrum)
array([1.+0.j,  2.+0.j,  3.+0.j,  4.+0.j,  3.+0.j,  2.+0.j]) # may vary
>>> ihfft(spectrum)
array([ 1.-0.j,  2.-0.j,  3.-0.j,  4.-0.j]) # may vary
*)

val ihfft2 : ?s:int list -> ?axes:int list -> ?norm:string -> ?overwrite_x:bool -> ?workers:int -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the 2-dimensional inverse FFT of a real spectrum.

Parameters
----------
x : array_like
    The input array
s : sequence of ints, optional
    Shape of the real input to the inverse FFT.
axes : sequence of ints, optional
    The axes over which to compute the inverse fft.
    Default is the last two axes.
norm : {None, 'ortho'}, optional
    Normalization mode (see `fft`). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.
    See :func:`fft` for more details.
workers : int, optional
    Maximum number of workers to use for parallel computation. If negative,
    the value wraps around from ``os.cpu_count()``.
    See :func:`~scipy.fft.fft` for more details.

Returns
-------
out : ndarray
    The result of the inverse real 2-D FFT.

See Also
--------
ihfftn : Compute the inverse of the N-dimensional FFT of Hermitian input.

Notes
-----
This is really `ihfftn` with different defaults.
For more details see `ihfftn`.
*)

val ihfftn : ?s:int list -> ?axes:int list -> ?norm:string -> ?overwrite_x:bool -> ?workers:int -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Compute the N-dimensional inverse discrete Fourier Transform for a real
spectrum.

This function computes the N-dimensional inverse discrete Fourier Transform
over any number of axes in an M-dimensional real array by means of the Fast
Fourier Transform (FFT). By default, all axes are transformed, with the
real transform performed over the last axis, while the remaining transforms
are complex.

Parameters
----------
x : array_like
    Input array, taken to be real.
s : sequence of ints, optional
    Shape (length along each transformed axis) to use from the input.
    (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
    Along any axis, if the given shape is smaller than that of the input,
    the input is cropped.  If it is larger, the input is padded with zeros.
    if `s` is not given, the shape of the input along the axes specified
    by `axes` is used.
axes : sequence of ints, optional
    Axes over which to compute the FFT.  If not given, the last ``len(s)``
    axes are used, or all axes if `s` is also not specified.
norm : {None, 'ortho'}, optional
    Normalization mode (see `fft`). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.
    See :func:`fft` for more details.
workers : int, optional
    Maximum number of workers to use for parallel computation. If negative,
    the value wraps around from ``os.cpu_count()``.
    See :func:`~scipy.fft.fft` for more details.

Returns
-------
out : complex ndarray
    The truncated or zero-padded input, transformed along the axes
    indicated by `axes`, or by a combination of `s` and `x`,
    as explained in the parameters section above.
    The length of the last axis transformed will be ``s[-1]//2+1``,
    while the remaining transformed axes will have lengths according to
    `s`, or unchanged from the input.

Raises
------
ValueError
    If `s` and `axes` have different length.
IndexError
    If an element of `axes` is larger than than the number of axes of `x`.

See Also
--------
hfftn : The forward n-dimensional FFT of Hermitian input.
hfft : The one-dimensional FFT of Hermitian input.
fft : The one-dimensional FFT, with definitions and conventions used.
fftn : The n-dimensional FFT.
hfft2 : The two-dimensional FFT of Hermitian input.

Notes
-----

The transform for real input is performed over the last transformation
axis, as by `ihfft`, then the transform over the remaining axes is
performed as by `ifftn`. The order of the output is the positive part of
the Hermitian output signal, in the same format as `rfft`.

Examples
--------
>>> import scipy.fft
>>> x = np.ones((2, 2, 2))
>>> scipy.fft.ihfftn(x)
array([[[1.+0.j,  0.+0.j], # may vary
        [0.+0.j,  0.+0.j]],
       [[0.+0.j,  0.+0.j],
        [0.+0.j,  0.+0.j]]])
>>> scipy.fft.ihfftn(x, axes=(2, 0))
array([[[1.+0.j,  0.+0.j], # may vary
        [1.+0.j,  0.+0.j]],
       [[0.+0.j,  0.+0.j],
        [0.+0.j,  0.+0.j]]])
*)

val irfft : ?n:int -> ?axis:int -> ?norm:string -> ?overwrite_x:bool -> ?workers:int -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the inverse of the n-point DFT for real input.

This function computes the inverse of the one-dimensional *n*-point
discrete Fourier Transform of real input computed by `rfft`.
In other words, ``irfft(rfft(x), len(x)) == x`` to within numerical
accuracy. (See Notes below for why ``len(a)`` is necessary here.)

The input is expected to be in the form returned by `rfft`, i.e. the
real zero-frequency term followed by the complex positive frequency terms
in order of increasing frequency.  Since the discrete Fourier Transform of
real input is Hermitian-symmetric, the negative frequency terms are taken
to be the complex conjugates of the corresponding positive frequency terms.

Parameters
----------
x : array_like
    The input array.
n : int, optional
    Length of the transformed axis of the output.
    For `n` output points, ``n//2+1`` input points are necessary.  If the
    input is longer than this, it is cropped.  If it is shorter than this,
    it is padded with zeros.  If `n` is not given, it is taken to be
    ``2*(m-1)`` where ``m`` is the length of the input along the axis
    specified by `axis`.
axis : int, optional
    Axis over which to compute the inverse FFT. If not given, the last
    axis is used.
norm : {None, 'ortho'}, optional
    Normalization mode (see `fft`). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.
    See :func:`fft` for more details.
workers : int, optional
    Maximum number of workers to use for parallel computation. If negative,
    the value wraps around from ``os.cpu_count()``.
    See :func:`~scipy.fft.fft` for more details.

Returns
-------
out : ndarray
    The truncated or zero-padded input, transformed along the axis
    indicated by `axis`, or the last one if `axis` is not specified.
    The length of the transformed axis is `n`, or, if `n` is not given,
    ``2*(m-1)`` where ``m`` is the length of the transformed axis of the
    input. To get an odd number of output points, `n` must be specified.

Raises
------
IndexError
    If `axis` is larger than the last axis of `x`.

See Also
--------
rfft : The one-dimensional FFT of real input, of which `irfft` is inverse.
fft : The one-dimensional FFT.
irfft2 : The inverse of the two-dimensional FFT of real input.
irfftn : The inverse of the *n*-dimensional FFT of real input.

Notes
-----
Returns the real valued `n`-point inverse discrete Fourier transform
of `x`, where `x` contains the non-negative frequency terms of a
Hermitian-symmetric sequence. `n` is the length of the result, not the
input.

If you specify an `n` such that `a` must be zero-padded or truncated, the
extra/removed values will be added/removed at high frequencies. One can
thus resample a series to `m` points via Fourier interpolation by:
``a_resamp = irfft(rfft(a), m)``.

The default value of `n` assumes an even output length. By the Hermitian
symmetry, the last imaginary component must be 0 and so is ignored. To
avoid losing information, the correct length of the real input *must* be
given.

Examples
--------
>>> import scipy.fft
>>> scipy.fft.ifft([1, -1j, -1, 1j])
array([0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j]) # may vary
>>> scipy.fft.irfft([1, -1j, -1])
array([0.,  1.,  0.,  0.])

Notice how the last term in the input to the ordinary `ifft` is the
complex conjugate of the second term, and the output has zero imaginary
part everywhere.  When calling `irfft`, the negative frequencies are not
specified, and the output array is purely real.
*)

val irfft2 : ?s:int list -> ?axes:int list -> ?norm:string -> ?overwrite_x:bool -> ?workers:int -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the 2-dimensional inverse FFT of a real array.

Parameters
----------
x : array_like
    The input array
s : sequence of ints, optional
    Shape of the real output to the inverse FFT.
axes : sequence of ints, optional
    The axes over which to compute the inverse fft.
    Default is the last two axes.
norm : {None, 'ortho'}, optional
    Normalization mode (see `fft`). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.
    See :func:`fft` for more details.
workers : int, optional
    Maximum number of workers to use for parallel computation. If negative,
    the value wraps around from ``os.cpu_count()``.
    See :func:`~scipy.fft.fft` for more details.

Returns
-------
out : ndarray
    The result of the inverse real 2-D FFT.

See Also
--------
irfftn : Compute the inverse of the N-dimensional FFT of real input.

Notes
-----
This is really `irfftn` with different defaults.
For more details see `irfftn`.
*)

val irfftn : ?s:int list -> ?axes:int list -> ?norm:string -> ?overwrite_x:bool -> ?workers:int -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the inverse of the N-dimensional FFT of real input.

This function computes the inverse of the N-dimensional discrete
Fourier Transform for real input over any number of axes in an
M-dimensional array by means of the Fast Fourier Transform (FFT).  In
other words, ``irfftn(rfftn(x), x.shape) == x`` to within numerical
accuracy. (The ``a.shape`` is necessary like ``len(a)`` is for `irfft`,
and for the same reason.)

The input should be ordered in the same way as is returned by `rfftn`,
i.e. as for `irfft` for the final transformation axis, and as for `ifftn`
along all the other axes.

Parameters
----------
x : array_like
    Input array.
s : sequence of ints, optional
    Shape (length of each transformed axis) of the output
    (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.). `s` is also the
    number of input points used along this axis, except for the last axis,
    where ``s[-1]//2+1`` points of the input are used.
    Along any axis, if the shape indicated by `s` is smaller than that of
    the input, the input is cropped.  If it is larger, the input is padded
    with zeros. If `s` is not given, the shape of the input along the axes
    specified by axes is used. Except for the last axis which is taken to be
    ``2*(m-1)`` where ``m`` is the length of the input along that axis.
axes : sequence of ints, optional
    Axes over which to compute the inverse FFT. If not given, the last
    `len(s)` axes are used, or all axes if `s` is also not specified.
norm : {None, 'ortho'}, optional
    Normalization mode (see `fft`). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.
    See :func:`fft` for more details.
workers : int, optional
    Maximum number of workers to use for parallel computation. If negative,
    the value wraps around from ``os.cpu_count()``.
    See :func:`~scipy.fft.fft` for more details.

Returns
-------
out : ndarray
    The truncated or zero-padded input, transformed along the axes
    indicated by `axes`, or by a combination of `s` or `x`,
    as explained in the parameters section above.
    The length of each transformed axis is as given by the corresponding
    element of `s`, or the length of the input in every axis except for the
    last one if `s` is not given.  In the final transformed axis the length
    of the output when `s` is not given is ``2*(m-1)`` where ``m`` is the
    length of the final transformed axis of the input.  To get an odd
    number of output points in the final axis, `s` must be specified.

Raises
------
ValueError
    If `s` and `axes` have different length.
IndexError
    If an element of `axes` is larger than than the number of axes of `x`.

See Also
--------
rfftn : The forward n-dimensional FFT of real input,
        of which `ifftn` is the inverse.
fft : The one-dimensional FFT, with definitions and conventions used.
irfft : The inverse of the one-dimensional FFT of real input.
irfft2 : The inverse of the two-dimensional FFT of real input.

Notes
-----
See `fft` for definitions and conventions used.

See `rfft` for definitions and conventions used for real input.

The default value of `s` assumes an even output length in the final
transformation axis. When performing the final complex to real
transformation, the Hermitian symmetry requires that the last imaginary
component along that axis must be 0 and so it is ignored. To avoid losing
information, the correct length of the real input *must* be given.

Examples
--------
>>> import scipy.fft
>>> x = np.zeros((3, 2, 2))
>>> x[0, 0, 0] = 3 * 2 * 2
>>> scipy.fft.irfftn(x)
array([[[1.,  1.],
        [1.,  1.]],
       [[1.,  1.],
        [1.,  1.]],
       [[1.,  1.],
        [1.,  1.]]])
*)

val register_backend : [`Scipy | `PyObject of Py.Object.t] -> Py.Object.t
(**
Register a backend for permanent use.

Registered backends have the lowest priority and will be tried after the
global backend.

Parameters
----------
backend: {object, 'scipy'}
    The backend to use.
    Can either be a ``str`` containing the name of a known backend
    {'scipy'}, or an object that implements the uarray protocol.

Raises
------
ValueError: If the backend does not implement ``numpy.scipy.fft``
*)

val rfft : ?n:int -> ?axis:int -> ?norm:string -> ?overwrite_x:bool -> ?workers:int -> x:Py.Object.t -> unit -> Py.Object.t
(**
Compute the one-dimensional discrete Fourier Transform for real input.

This function computes the one-dimensional *n*-point discrete Fourier
Transform (DFT) of a real-valued array by means of an efficient algorithm
called the Fast Fourier Transform (FFT).

Parameters
----------
a : array_like
    Input array
n : int, optional
    Number of points along transformation axis in the input to use.
    If `n` is smaller than the length of the input, the input is cropped.
    If it is larger, the input is padded with zeros. If `n` is not given,
    the length of the input along the axis specified by `axis` is used.
axis : int, optional
    Axis over which to compute the FFT. If not given, the last axis is
    used.
norm : {None, 'ortho'}, optional
    Normalization mode (see `fft`). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.
    See :func:`fft` for more details.
workers : int, optional
    Maximum number of workers to use for parallel computation. If negative,
    the value wraps around from ``os.cpu_count()``.
    See :func:`~scipy.fft.fft` for more details.

Returns
-------
out : complex ndarray
    The truncated or zero-padded input, transformed along the axis
    indicated by `axis`, or the last one if `axis` is not specified.
    If `n` is even, the length of the transformed axis is ``(n/2)+1``.
    If `n` is odd, the length is ``(n+1)/2``.

Raises
------
IndexError
    If `axis` is larger than the last axis of `a`.

See Also
--------
irfft : The inverse of `rfft`.
fft : The one-dimensional FFT of general (complex) input.
fftn : The *n*-dimensional FFT.
rfftn : The *n*-dimensional FFT of real input.

Notes
-----
When the DFT is computed for purely real input, the output is
Hermitian-symmetric, i.e. the negative frequency terms are just the complex
conjugates of the corresponding positive-frequency terms, and the
negative-frequency terms are therefore redundant.  This function does not
compute the negative frequency terms, and the length of the transformed
axis of the output is therefore ``n//2 + 1``.

When ``X = rfft(x)`` and fs is the sampling frequency, ``X[0]`` contains
the zero-frequency term 0*fs, which is real due to Hermitian symmetry.

If `n` is even, ``A[-1]`` contains the term representing both positive
and negative Nyquist frequency (+fs/2 and -fs/2), and must also be purely
real. If `n` is odd, there is no term at fs/2; ``A[-1]`` contains
the largest positive frequency (fs/2*(n-1)/n), and is complex in the
general case.

If the input `a` contains an imaginary part, it is silently discarded.

Examples
--------
>>> import scipy.fft
>>> scipy.fft.fft([0, 1, 0, 0])
array([ 1.+0.j,  0.-1.j, -1.+0.j,  0.+1.j]) # may vary
>>> scipy.fft.rfft([0, 1, 0, 0])
array([ 1.+0.j,  0.-1.j, -1.+0.j]) # may vary

Notice how the final element of the `fft` output is the complex conjugate
of the second element, for real input. For `rfft`, this symmetry is
exploited to compute only the non-negative frequency terms.
*)

val rfft2 : ?s:int list -> ?axes:int list -> ?norm:string -> ?overwrite_x:bool -> ?workers:int -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the 2-dimensional FFT of a real array.

Parameters
----------
x : array
    Input array, taken to be real.
s : sequence of ints, optional
    Shape of the FFT.
axes : sequence of ints, optional
    Axes over which to compute the FFT.
norm : {None, 'ortho'}, optional
    Normalization mode (see `fft`). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.
    See :func:`fft` for more details.
workers : int, optional
    Maximum number of workers to use for parallel computation. If negative,
    the value wraps around from ``os.cpu_count()``.
    See :func:`~scipy.fft.fft` for more details.

Returns
-------
out : ndarray
    The result of the real 2-D FFT.

See Also
--------
rfftn : Compute the N-dimensional discrete Fourier Transform for real
        input.

Notes
-----
This is really just `rfftn` with different default behavior.
For more details see `rfftn`.
*)

val rfftfreq : ?d:[`F of float | `I of int | `Bool of bool | `S of string] -> n:int -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return the Discrete Fourier Transform sample frequencies
(for usage with rfft, irfft).

The returned float array `f` contains the frequency bin centers in cycles
per unit of the sample spacing (with zero at the start).  For instance, if
the sample spacing is in seconds, then the frequency unit is cycles/second.

Given a window length `n` and a sample spacing `d`::

  f = [0, 1, ...,     n/2-1,     n/2] / (d*n)   if n is even
  f = [0, 1, ..., (n-1)/2-1, (n-1)/2] / (d*n)   if n is odd

Unlike `fftfreq` (but like `scipy.fftpack.rfftfreq`)
the Nyquist frequency component is considered to be positive.

Parameters
----------
n : int
    Window length.
d : scalar, optional
    Sample spacing (inverse of the sampling rate). Defaults to 1.

Returns
-------
f : ndarray
    Array of length ``n//2 + 1`` containing the sample frequencies.

Examples
--------
>>> signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5, -3, 4], dtype=float)
>>> fourier = np.fft.rfft(signal)
>>> n = signal.size
>>> sample_rate = 100
>>> freq = np.fft.fftfreq(n, d=1./sample_rate)
>>> freq
array([  0.,  10.,  20., ..., -30., -20., -10.])
>>> freq = np.fft.rfftfreq(n, d=1./sample_rate)
>>> freq
array([  0.,  10.,  20.,  30.,  40.,  50.])
*)

val rfftn : ?s:int list -> ?axes:int list -> ?norm:string -> ?overwrite_x:bool -> ?workers:int -> x:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Compute the N-dimensional discrete Fourier Transform for real input.

This function computes the N-dimensional discrete Fourier Transform over
any number of axes in an M-dimensional real array by means of the Fast
Fourier Transform (FFT).  By default, all axes are transformed, with the
real transform performed over the last axis, while the remaining
transforms are complex.

Parameters
----------
x : array_like
    Input array, taken to be real.
s : sequence of ints, optional
    Shape (length along each transformed axis) to use from the input.
    (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
    The final element of `s` corresponds to `n` for ``rfft(x, n)``, while
    for the remaining axes, it corresponds to `n` for ``fft(x, n)``.
    Along any axis, if the given shape is smaller than that of the input,
    the input is cropped.  If it is larger, the input is padded with zeros.
    if `s` is not given, the shape of the input along the axes specified
    by `axes` is used.
axes : sequence of ints, optional
    Axes over which to compute the FFT.  If not given, the last ``len(s)``
    axes are used, or all axes if `s` is also not specified.
norm : {None, 'ortho'}, optional
    Normalization mode (see `fft`). Default is None.
overwrite_x : bool, optional
    If True, the contents of `x` can be destroyed; the default is False.
    See :func:`fft` for more details.
workers : int, optional
    Maximum number of workers to use for parallel computation. If negative,
    the value wraps around from ``os.cpu_count()``.
    See :func:`~scipy.fft.fft` for more details.

Returns
-------
out : complex ndarray
    The truncated or zero-padded input, transformed along the axes
    indicated by `axes`, or by a combination of `s` and `x`,
    as explained in the parameters section above.
    The length of the last axis transformed will be ``s[-1]//2+1``,
    while the remaining transformed axes will have lengths according to
    `s`, or unchanged from the input.

Raises
------
ValueError
    If `s` and `axes` have different length.
IndexError
    If an element of `axes` is larger than than the number of axes of `x`.

See Also
--------
irfftn : The inverse of `rfftn`, i.e. the inverse of the n-dimensional FFT
     of real input.
fft : The one-dimensional FFT, with definitions and conventions used.
rfft : The one-dimensional FFT of real input.
fftn : The n-dimensional FFT.
rfft2 : The two-dimensional FFT of real input.

Notes
-----
The transform for real input is performed over the last transformation
axis, as by `rfft`, then the transform over the remaining axes is
performed as by `fftn`.  The order of the output is as for `rfft` for the
final transformation axis, and as for `fftn` for the remaining
transformation axes.

See `fft` for details, definitions and conventions used.

Examples
--------
>>> import scipy.fft
>>> x = np.ones((2, 2, 2))
>>> scipy.fft.rfftn(x)
array([[[8.+0.j,  0.+0.j], # may vary
        [0.+0.j,  0.+0.j]],
       [[0.+0.j,  0.+0.j],
        [0.+0.j,  0.+0.j]]])

>>> scipy.fft.rfftn(x, axes=(2, 0))
array([[[4.+0.j,  0.+0.j], # may vary
        [4.+0.j,  0.+0.j]],
       [[0.+0.j,  0.+0.j],
        [0.+0.j,  0.+0.j]]])
*)

val set_backend : ?coerce:bool -> ?only:bool -> backend:[`Scipy | `PyObject of Py.Object.t] -> unit -> Py.Object.t
(**
Context manager to set the backend within a fixed scope.

Upon entering the ``with`` statement, the given backend will be added to
the list of available backends with the highest priority. Upon exit, the
backend is reset to the state before entering the scope.

Parameters
----------
backend: {object, 'scipy'}
    The backend to use.
    Can either be a ``str`` containing the name of a known backend
    {'scipy'}, or an object that implements the uarray protocol.
coerce: bool, optional
    Whether to allow expensive conversions for the ``x`` parameter. e.g.
    copying a numpy array to the GPU for a CuPy backend. Implies ``only``.
only: bool, optional
   If only is ``True`` and this backend returns ``NotImplemented`` then a
   BackendNotImplemented error will be raised immediately. Ignoring any
   lower priority backends.

Examples
--------
>>> import scipy.fft as fft
>>> with fft.set_backend('scipy', only=True):
...     fft.fft([1])  # Always calls the scipy implementation
array([1.+0.j])
*)

val set_global_backend : [`Scipy | `PyObject of Py.Object.t] -> Py.Object.t
(**
Sets the global fft backend

The global backend has higher priority than registered backends, but lower
priority than context-specific backends set with `set_backend`.

Parameters
----------
backend: {object, 'scipy'}
    The backend to use.
    Can either be a ``str`` containing the name of a known backend
    {'scipy'}, or an object that implements the uarray protocol.

Raises
------
ValueError: If the backend does not implement ``numpy.scipy.fft``

Notes
-----
This will overwrite the previously set global backend, which by default is
the SciPy implementation.
*)

val set_workers : int -> Py.Object.t
(**
Context manager for the default number of workers used in `scipy.fft`

Parameters
----------
workers : int
    The default number of workers to use

Examples
--------
>>> from scipy import fft, signal
>>> x = np.random.randn(128, 64)
>>> with fft.set_workers(4):
...     y = signal.fftconvolve(x, x)
*)

val skip_backend : [`Scipy | `PyObject of Py.Object.t] -> Py.Object.t
(**
Context manager to skip a backend within a fixed scope.

Within the context of a ``with`` statement, the given backend will not be
called. This covers backends registered both locally and globally. Upon
exit, the backend will again be considered.

Parameters
----------
backend: {object, 'scipy'}
    The backend to skip.
    Can either be a ``str`` containing the name of a known backend
    {'scipy'}, or an object that implements the uarray protocol.

Examples
--------
>>> import scipy.fft as fft
>>> fft.fft([1])  # Calls default scipy backend
array([1.+0.j])
>>> with fft.skip_backend('scipy'):  # We expicitly skip the scipy backend
...     fft.fft([1])                 # leaving no implementation available
Traceback (most recent call last):
    ...
BackendNotImplementedError: No selected backends had an implementation ...
*)

