(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module Filters : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module Iterable : sig
type tag = [`Iterable]
type t = [`Iterable | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val __iter__ : [> tag] Obj.t -> Py.Object.t
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

val convolve : ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`I of int | `Sequence of Py.Object.t] -> input:[>`Ndarray] Np.Obj.t -> weights:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional convolution.

The array is convolved with the given kernel.

Parameters
----------
input : array_like
    The input array.
weights : array_like
    Array of weights, same number of dimensions as input
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'reflect'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0
origin : int or sequence, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right. By passing a sequence of origins with length equal to
    the number of dimensions of the input array, different shifts can
    be specified along each axis.

Returns
-------
result : ndarray
    The result of convolution of `input` with `weights`.

See Also
--------
correlate : Correlate an image with a kernel.

Notes
-----
Each value in result is :math:`C_i = \sum_j{I_{i+k-j} W_j}`, where
W is the `weights` kernel,
j is the N-D spatial index over :math:`W`,
I is the `input` and k is the coordinate of the center of
W, specified by `origin` in the input parameters.

Examples
--------
Perhaps the simplest case to understand is ``mode='constant', cval=0.0``,
because in this case borders (i.e., where the `weights` kernel, centered
on any one value, extends beyond an edge of `input`) are treated as zeros.

>>> a = np.array([[1, 2, 0, 0],
...               [5, 3, 0, 4],
...               [0, 0, 0, 7],
...               [9, 3, 0, 0]])
>>> k = np.array([[1,1,1],[1,1,0],[1,0,0]])
>>> from scipy import ndimage
>>> ndimage.convolve(a, k, mode='constant', cval=0.0)
array([[11, 10,  7,  4],
       [10,  3, 11, 11],
       [15, 12, 14,  7],
       [12,  3,  7,  0]])

Setting ``cval=1.0`` is equivalent to padding the outer edge of `input`
with 1.0's (and then extracting only the original region of the result).

>>> ndimage.convolve(a, k, mode='constant', cval=1.0)
array([[13, 11,  8,  7],
       [11,  3, 11, 14],
       [16, 12, 14, 10],
       [15,  6, 10,  5]])

With ``mode='reflect'`` (the default), outer values are reflected at the
edge of `input` to fill in missing values.

>>> b = np.array([[2, 0, 0],
...               [1, 0, 0],
...               [0, 0, 0]])
>>> k = np.array([[0,1,0], [0,1,0], [0,1,0]])
>>> ndimage.convolve(b, k, mode='reflect')
array([[5, 0, 0],
       [3, 0, 0],
       [1, 0, 0]])

This includes diagonally at the corners.

>>> k = np.array([[1,0,0],[0,1,0],[0,0,1]])
>>> ndimage.convolve(b, k)
array([[4, 2, 0],
       [3, 2, 0],
       [1, 1, 0]])

With ``mode='nearest'``, the single nearest value in to an edge in
`input` is repeated as many times as needed to match the overlapping
`weights`.

>>> c = np.array([[2, 0, 1],
...               [1, 0, 0],
...               [0, 0, 0]])
>>> k = np.array([[0, 1, 0],
...               [0, 1, 0],
...               [0, 1, 0],
...               [0, 1, 0],
...               [0, 1, 0]])
>>> ndimage.convolve(c, k, mode='nearest')
array([[7, 0, 3],
       [5, 0, 2],
       [3, 0, 1]])
*)

val convolve1d : ?axis:int -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:int -> input:[>`Ndarray] Np.Obj.t -> weights:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Calculate a 1-D convolution along the given axis.

The lines of the array along the given axis are convolved with the
given weights.

Parameters
----------
input : array_like
    The input array.
weights : ndarray
    1-D sequence of numbers.
axis : int, optional
    The axis of `input` along which to calculate. Default is -1.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'reflect'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : int, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right.

Returns
-------
convolve1d : ndarray
    Convolved array with same shape as input

Examples
--------
>>> from scipy.ndimage import convolve1d
>>> convolve1d([2, 8, 0, 4, 1, 9, 9, 0], weights=[1, 3])
array([14, 24,  4, 13, 12, 36, 27,  0])
*)

val correlate : ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`I of int | `Sequence of Py.Object.t] -> input:[>`Ndarray] Np.Obj.t -> weights:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional correlation.

The array is correlated with the given kernel.

Parameters
----------
input : array_like
    The input array.
weights : ndarray
    array of weights, same number of dimensions as input
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'reflect'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : int or sequence, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right. By passing a sequence of origins with length equal to
    the number of dimensions of the input array, different shifts can
    be specified along each axis.

Returns
-------
result : ndarray
    The result of correlation of `input` with `weights`.

See Also
--------
convolve : Convolve an image with a kernel.

Examples
--------
Correlation is the process of moving a filter mask often referred to
as kernel over the image and computing the sum of products at each location.

>>> from scipy.ndimage import correlate
>>> input_img = np.arange(25).reshape(5,5)
>>> print(input_img)
[[ 0  1  2  3  4]
[ 5  6  7  8  9]
[10 11 12 13 14]
[15 16 17 18 19]
[20 21 22 23 24]]

Define a kernel (weights) for correlation. In this example, it is for sum of
center and up, down, left and right next elements.

>>> weights = [[0, 1, 0],
...            [1, 1, 1],
...            [0, 1, 0]]

We can calculate a correlation result:
For example, element ``[2,2]`` is ``7 + 11 + 12 + 13 + 17 = 60``.

>>> correlate(input_img, weights)
array([[  6,  10,  15,  20,  24],
    [ 26,  30,  35,  40,  44],
    [ 51,  55,  60,  65,  69],
    [ 76,  80,  85,  90,  94],
    [ 96, 100, 105, 110, 114]])
*)

val correlate1d : ?axis:int -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:int -> input:[>`Ndarray] Np.Obj.t -> weights:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Calculate a 1-D correlation along the given axis.

The lines of the array along the given axis are correlated with the
given weights.

Parameters
----------
input : array_like
    The input array.
weights : array
    1-D sequence of numbers.
axis : int, optional
    The axis of `input` along which to calculate. Default is -1.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'reflect'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : int, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right.

Examples
--------
>>> from scipy.ndimage import correlate1d
>>> correlate1d([2, 8, 0, 4, 1, 9, 9, 0], weights=[1, 3])
array([ 8, 26,  8, 12,  7, 28, 36,  9])
*)

val gaussian_filter : ?order:[`I of int | `Is of int list] -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Sequence of Py.Object.t | `S of string] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?truncate:float -> input:[>`Ndarray] Np.Obj.t -> sigma:[`Bool of bool | `F of float | `S of string | `Sequence_of_scalars of Py.Object.t | `I of int] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional Gaussian filter.

Parameters
----------
input : array_like
    The input array.
sigma : scalar or sequence of scalars
    Standard deviation for Gaussian kernel. The standard
    deviations of the Gaussian filter are given for each axis as a
    sequence, or as a single number, in which case it is equal for
    all axes.
order : int or sequence of ints, optional
    The order of the filter along each axis is given as a sequence
    of integers, or as a single number. An order of 0 corresponds
    to convolution with a Gaussian kernel. A positive order
    corresponds to convolution with that derivative of a Gaussian.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : str or sequence, optional
    The `mode` parameter determines how the input array is extended
    when the filter overlaps a border. By passing a sequence of modes
    with length equal to the number of dimensions of the input array,
    different modes can be specified along each axis. Default value is
    'reflect'. The valid values and their behavior is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
truncate : float
    Truncate the filter at this many standard deviations.
    Default is 4.0.

Returns
-------
gaussian_filter : ndarray
    Returned array of same shape as `input`.

Notes
-----
The multidimensional filter is implemented as a sequence of
1-D convolution filters. The intermediate arrays are
stored in the same data type as the output. Therefore, for output
types with a limited precision, the results may be imprecise
because intermediate results may be stored with insufficient
precision.

Examples
--------
>>> from scipy.ndimage import gaussian_filter
>>> a = np.arange(50, step=2).reshape((5,5))
>>> a
array([[ 0,  2,  4,  6,  8],
       [10, 12, 14, 16, 18],
       [20, 22, 24, 26, 28],
       [30, 32, 34, 36, 38],
       [40, 42, 44, 46, 48]])
>>> gaussian_filter(a, sigma=1)
array([[ 4,  6,  8,  9, 11],
       [10, 12, 14, 15, 17],
       [20, 22, 24, 25, 27],
       [29, 31, 33, 34, 36],
       [35, 37, 39, 40, 42]])

>>> from scipy import misc
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = misc.ascent()
>>> result = gaussian_filter(ascent, sigma=5)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()
*)

val gaussian_filter1d : ?axis:int -> ?order:int -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?truncate:float -> input:[>`Ndarray] Np.Obj.t -> sigma:[`F of float | `I of int | `Bool of bool | `S of string] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
1-D Gaussian filter.

Parameters
----------
input : array_like
    The input array.
sigma : scalar
    standard deviation for Gaussian kernel
axis : int, optional
    The axis of `input` along which to calculate. Default is -1.
order : int, optional
    An order of 0 corresponds to convolution with a Gaussian
    kernel. A positive order corresponds to convolution with
    that derivative of a Gaussian.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'reflect'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
truncate : float, optional
    Truncate the filter at this many standard deviations.
    Default is 4.0.

Returns
-------
gaussian_filter1d : ndarray

Examples
--------
>>> from scipy.ndimage import gaussian_filter1d
>>> gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 1)
array([ 1.42704095,  2.06782203,  3.        ,  3.93217797,  4.57295905])
>>> gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 4)
array([ 2.91948343,  2.95023502,  3.        ,  3.04976498,  3.08051657])
>>> import matplotlib.pyplot as plt
>>> np.random.seed(280490)
>>> x = np.random.randn(101).cumsum()
>>> y3 = gaussian_filter1d(x, 3)
>>> y6 = gaussian_filter1d(x, 6)
>>> plt.plot(x, 'k', label='original data')
>>> plt.plot(y3, '--', label='filtered, sigma=3')
>>> plt.plot(y6, ':', label='filtered, sigma=6')
>>> plt.legend()
>>> plt.grid()
>>> plt.show()
*)

val gaussian_gradient_magnitude : ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Sequence of Py.Object.t | `S of string] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?kwargs:(string * Py.Object.t) list -> input:[>`Ndarray] Np.Obj.t -> sigma:[`Bool of bool | `F of float | `S of string | `Sequence_of_scalars of Py.Object.t | `I of int] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional gradient magnitude using Gaussian derivatives.

Parameters
----------
input : array_like
    The input array.
sigma : scalar or sequence of scalars
    The standard deviations of the Gaussian filter are given for
    each axis as a sequence, or as a single number, in which case
    it is equal for all axes.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : str or sequence, optional
    The `mode` parameter determines how the input array is extended
    when the filter overlaps a border. By passing a sequence of modes
    with length equal to the number of dimensions of the input array,
    different modes can be specified along each axis. Default value is
    'reflect'. The valid values and their behavior is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
Extra keyword arguments will be passed to gaussian_filter().

Returns
-------
gaussian_gradient_magnitude : ndarray
    Filtered array. Has the same shape as `input`.

Examples
--------
>>> from scipy import ndimage, misc
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = misc.ascent()
>>> result = ndimage.gaussian_gradient_magnitude(ascent, sigma=5)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()
*)

val gaussian_laplace : ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Sequence of Py.Object.t | `S of string] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?kwargs:(string * Py.Object.t) list -> input:[>`Ndarray] Np.Obj.t -> sigma:[`Bool of bool | `F of float | `S of string | `Sequence_of_scalars of Py.Object.t | `I of int] -> unit -> Py.Object.t
(**
Multidimensional Laplace filter using Gaussian second derivatives.

Parameters
----------
input : array_like
    The input array.
sigma : scalar or sequence of scalars
    The standard deviations of the Gaussian filter are given for
    each axis as a sequence, or as a single number, in which case
    it is equal for all axes.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : str or sequence, optional
    The `mode` parameter determines how the input array is extended
    when the filter overlaps a border. By passing a sequence of modes
    with length equal to the number of dimensions of the input array,
    different modes can be specified along each axis. Default value is
    'reflect'. The valid values and their behavior is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
Extra keyword arguments will be passed to gaussian_filter().

Examples
--------
>>> from scipy import ndimage, misc
>>> import matplotlib.pyplot as plt
>>> ascent = misc.ascent()

>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side

>>> result = ndimage.gaussian_laplace(ascent, sigma=1)
>>> ax1.imshow(result)

>>> result = ndimage.gaussian_laplace(ascent, sigma=3)
>>> ax2.imshow(result)
>>> plt.show()
*)

val generic_filter : ?size:[`Bool of bool | `F of float | `S of string | `Tuple of Py.Object.t | `I of int] -> ?footprint:[>`Ndarray] Np.Obj.t -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`I of int | `Sequence of Py.Object.t] -> ?extra_arguments:Py.Object.t -> ?extra_keywords:Py.Object.t -> input:[>`Ndarray] Np.Obj.t -> function_:[`Scipy_LowLevelCallable of Py.Object.t | `Callable of Py.Object.t] -> unit -> Py.Object.t
(**
Calculate a multidimensional filter using the given function.

At each element the provided function is called. The input values
within the filter footprint at that element are passed to the function
as a 1-D array of double values.

Parameters
----------
input : array_like
    The input array.
function : {callable, scipy.LowLevelCallable}
    Function to apply at each element.
size : scalar or tuple, optional
    See footprint, below. Ignored if footprint is given.
footprint : array, optional
    Either `size` or `footprint` must be defined. `size` gives
    the shape that is taken from the input array, at every element
    position, to define the input to the filter function.
    `footprint` is a boolean array that specifies (implicitly) a
    shape, but also which of the elements within this shape will get
    passed to the filter function. Thus ``size=(n,m)`` is equivalent
    to ``footprint=np.ones((n,m))``.  We adjust `size` to the number
    of dimensions of the input array, so that, if the input array is
    shape (10,10,10), and `size` is 2, then the actual size used is
    (2,2,2). When `footprint` is given, `size` is ignored.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'reflect'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : int or sequence, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right. By passing a sequence of origins with length equal to
    the number of dimensions of the input array, different shifts can
    be specified along each axis.
extra_arguments : sequence, optional
    Sequence of extra positional arguments to pass to passed function.
extra_keywords : dict, optional
    dict of extra keyword arguments to pass to passed function.

Notes
-----
This function also accepts low-level callback functions with one of
the following signatures and wrapped in `scipy.LowLevelCallable`:

.. code:: c

   int callback(double *buffer, npy_intp filter_size,
                double *return_value, void *user_data)
   int callback(double *buffer, intptr_t filter_size,
                double *return_value, void *user_data)

The calling function iterates over the elements of the input and
output arrays, calling the callback function at each element. The
elements within the footprint of the filter at the current element are
passed through the ``buffer`` parameter, and the number of elements
within the footprint through ``filter_size``. The calculated value is
returned in ``return_value``. ``user_data`` is the data pointer provided
to `scipy.LowLevelCallable` as-is.

The callback function must return an integer error status that is zero
if something went wrong and one otherwise. If an error occurs, you should
normally set the python error status with an informative message
before returning, otherwise a default error message is set by the
calling function.

In addition, some other low-level function pointer specifications
are accepted, but these are for backward compatibility only and should
not be used in new code.
*)

val generic_filter1d : ?axis:int -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:int -> ?extra_arguments:Py.Object.t -> ?extra_keywords:Py.Object.t -> input:[>`Ndarray] Np.Obj.t -> function_:[`Scipy_LowLevelCallable of Py.Object.t | `Callable of Py.Object.t] -> filter_size:[`F of float | `I of int | `Bool of bool | `S of string] -> unit -> Py.Object.t
(**
Calculate a 1-D filter along the given axis.

`generic_filter1d` iterates over the lines of the array, calling the
given function at each line. The arguments of the line are the
input line, and the output line. The input and output lines are 1-D
double arrays. The input line is extended appropriately according
to the filter size and origin. The output line must be modified
in-place with the result.

Parameters
----------
input : array_like
    The input array.
function : {callable, scipy.LowLevelCallable}
    Function to apply along given axis.
filter_size : scalar
    Length of the filter.
axis : int, optional
    The axis of `input` along which to calculate. Default is -1.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'reflect'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : int, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right.
extra_arguments : sequence, optional
    Sequence of extra positional arguments to pass to passed function.
extra_keywords : dict, optional
    dict of extra keyword arguments to pass to passed function.

Notes
-----
This function also accepts low-level callback functions with one of
the following signatures and wrapped in `scipy.LowLevelCallable`:

.. code:: c

   int function(double *input_line, npy_intp input_length,
                double *output_line, npy_intp output_length,
                void *user_data)
   int function(double *input_line, intptr_t input_length,
                double *output_line, intptr_t output_length,
                void *user_data)

The calling function iterates over the lines of the input and output
arrays, calling the callback function at each line. The current line
is extended according to the border conditions set by the calling
function, and the result is copied into the array that is passed
through ``input_line``. The length of the input line (after extension)
is passed through ``input_length``. The callback function should apply
the filter and store the result in the array passed through
``output_line``. The length of the output line is passed through
``output_length``. ``user_data`` is the data pointer provided
to `scipy.LowLevelCallable` as-is.

The callback function must return an integer error status that is zero
if something went wrong and one otherwise. If an error occurs, you should
normally set the python error status with an informative message
before returning, otherwise a default error message is set by the
calling function.

In addition, some other low-level function pointer specifications
are accepted, but these are for backward compatibility only and should
not be used in new code.
*)

val generic_gradient_magnitude : ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Sequence of Py.Object.t | `S of string] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?extra_arguments:Py.Object.t -> ?extra_keywords:Py.Object.t -> input:[>`Ndarray] Np.Obj.t -> derivative:Py.Object.t -> unit -> Py.Object.t
(**
Gradient magnitude using a provided gradient function.

Parameters
----------
input : array_like
    The input array.
derivative : callable
    Callable with the following signature::

        derivative(input, axis, output, mode, cval,
                   *extra_arguments, **extra_keywords)

    See `extra_arguments`, `extra_keywords` below.
    `derivative` can assume that `input` and `output` are ndarrays.
    Note that the output from `derivative` is modified inplace;
    be careful to copy important inputs before returning them.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : str or sequence, optional
    The `mode` parameter determines how the input array is extended
    when the filter overlaps a border. By passing a sequence of modes
    with length equal to the number of dimensions of the input array,
    different modes can be specified along each axis. Default value is
    'reflect'. The valid values and their behavior is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
extra_keywords : dict, optional
    dict of extra keyword arguments to pass to passed function.
extra_arguments : sequence, optional
    Sequence of extra positional arguments to pass to passed function.
*)

val generic_laplace : ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Sequence of Py.Object.t | `S of string] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?extra_arguments:Py.Object.t -> ?extra_keywords:Py.Object.t -> input:[>`Ndarray] Np.Obj.t -> derivative2:Py.Object.t -> unit -> Py.Object.t
(**
N-D Laplace filter using a provided second derivative function.

Parameters
----------
input : array_like
    The input array.
derivative2 : callable
    Callable with the following signature::

        derivative2(input, axis, output, mode, cval,
                    *extra_arguments, **extra_keywords)

    See `extra_arguments`, `extra_keywords` below.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : str or sequence, optional
    The `mode` parameter determines how the input array is extended
    when the filter overlaps a border. By passing a sequence of modes
    with length equal to the number of dimensions of the input array,
    different modes can be specified along each axis. Default value is
    'reflect'. The valid values and their behavior is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
extra_keywords : dict, optional
    dict of extra keyword arguments to pass to passed function.
extra_arguments : sequence, optional
    Sequence of extra positional arguments to pass to passed function.
*)

val laplace : ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Sequence of Py.Object.t | `S of string] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
N-D Laplace filter based on approximate second derivatives.

Parameters
----------
input : array_like
    The input array.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : str or sequence, optional
    The `mode` parameter determines how the input array is extended
    when the filter overlaps a border. By passing a sequence of modes
    with length equal to the number of dimensions of the input array,
    different modes can be specified along each axis. Default value is
    'reflect'. The valid values and their behavior is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.

Examples
--------
>>> from scipy import ndimage, misc
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = misc.ascent()
>>> result = ndimage.laplace(ascent)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()
*)

val maximum_filter : ?size:[`Bool of bool | `F of float | `S of string | `Tuple of Py.Object.t | `I of int] -> ?footprint:[>`Ndarray] Np.Obj.t -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Sequence of Py.Object.t | `S of string] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`I of int | `Sequence of Py.Object.t] -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Calculate a multidimensional maximum filter.

Parameters
----------
input : array_like
    The input array.
size : scalar or tuple, optional
    See footprint, below. Ignored if footprint is given.
footprint : array, optional
    Either `size` or `footprint` must be defined. `size` gives
    the shape that is taken from the input array, at every element
    position, to define the input to the filter function.
    `footprint` is a boolean array that specifies (implicitly) a
    shape, but also which of the elements within this shape will get
    passed to the filter function. Thus ``size=(n,m)`` is equivalent
    to ``footprint=np.ones((n,m))``.  We adjust `size` to the number
    of dimensions of the input array, so that, if the input array is
    shape (10,10,10), and `size` is 2, then the actual size used is
    (2,2,2). When `footprint` is given, `size` is ignored.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : str or sequence, optional
    The `mode` parameter determines how the input array is extended
    when the filter overlaps a border. By passing a sequence of modes
    with length equal to the number of dimensions of the input array,
    different modes can be specified along each axis. Default value is
    'reflect'. The valid values and their behavior is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : int or sequence, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right. By passing a sequence of origins with length equal to
    the number of dimensions of the input array, different shifts can
    be specified along each axis.

Returns
-------
maximum_filter : ndarray
    Filtered array. Has the same shape as `input`.

Notes
-----
A sequence of modes (one per axis) is only supported when the footprint is
separable. Otherwise, a single mode string must be provided.

Examples
--------
>>> from scipy import ndimage, misc
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = misc.ascent()
>>> result = ndimage.maximum_filter(ascent, size=20)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()
*)

val maximum_filter1d : ?axis:int -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:int -> input:[>`Ndarray] Np.Obj.t -> size:int -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t option
(**
Calculate a 1-D maximum filter along the given axis.

The lines of the array along the given axis are filtered with a
maximum filter of given size.

Parameters
----------
input : array_like
    The input array.
size : int
    Length along which to calculate the 1-D maximum.
axis : int, optional
    The axis of `input` along which to calculate. Default is -1.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'reflect'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : int, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right.

Returns
-------
maximum1d : ndarray, None
    Maximum-filtered array with same shape as input.
    None if `output` is not None

Notes
-----
This function implements the MAXLIST algorithm [1]_, as described by
Richard Harter [2]_, and has a guaranteed O(n) performance, `n` being
the `input` length, regardless of filter size.

References
----------
.. [1] http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.42.2777
.. [2] http://www.richardhartersworld.com/cri/2001/slidingmin.html

Examples
--------
>>> from scipy.ndimage import maximum_filter1d
>>> maximum_filter1d([2, 8, 0, 4, 1, 9, 9, 0], size=3)
array([8, 8, 8, 4, 9, 9, 9, 9])
*)

val median_filter : ?size:[`Bool of bool | `F of float | `S of string | `Tuple of Py.Object.t | `I of int] -> ?footprint:[>`Ndarray] Np.Obj.t -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`I of int | `Sequence of Py.Object.t] -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Calculate a multidimensional median filter.

Parameters
----------
input : array_like
    The input array.
size : scalar or tuple, optional
    See footprint, below. Ignored if footprint is given.
footprint : array, optional
    Either `size` or `footprint` must be defined. `size` gives
    the shape that is taken from the input array, at every element
    position, to define the input to the filter function.
    `footprint` is a boolean array that specifies (implicitly) a
    shape, but also which of the elements within this shape will get
    passed to the filter function. Thus ``size=(n,m)`` is equivalent
    to ``footprint=np.ones((n,m))``.  We adjust `size` to the number
    of dimensions of the input array, so that, if the input array is
    shape (10,10,10), and `size` is 2, then the actual size used is
    (2,2,2). When `footprint` is given, `size` is ignored.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'reflect'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : int or sequence, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right. By passing a sequence of origins with length equal to
    the number of dimensions of the input array, different shifts can
    be specified along each axis.

Returns
-------
median_filter : ndarray
    Filtered array. Has the same shape as `input`.

Examples
--------
>>> from scipy import ndimage, misc
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = misc.ascent()
>>> result = ndimage.median_filter(ascent, size=20)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()
*)

val minimum_filter : ?size:[`Bool of bool | `F of float | `S of string | `Tuple of Py.Object.t | `I of int] -> ?footprint:[>`Ndarray] Np.Obj.t -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Sequence of Py.Object.t | `S of string] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`I of int | `Sequence of Py.Object.t] -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Calculate a multidimensional minimum filter.

Parameters
----------
input : array_like
    The input array.
size : scalar or tuple, optional
    See footprint, below. Ignored if footprint is given.
footprint : array, optional
    Either `size` or `footprint` must be defined. `size` gives
    the shape that is taken from the input array, at every element
    position, to define the input to the filter function.
    `footprint` is a boolean array that specifies (implicitly) a
    shape, but also which of the elements within this shape will get
    passed to the filter function. Thus ``size=(n,m)`` is equivalent
    to ``footprint=np.ones((n,m))``.  We adjust `size` to the number
    of dimensions of the input array, so that, if the input array is
    shape (10,10,10), and `size` is 2, then the actual size used is
    (2,2,2). When `footprint` is given, `size` is ignored.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : str or sequence, optional
    The `mode` parameter determines how the input array is extended
    when the filter overlaps a border. By passing a sequence of modes
    with length equal to the number of dimensions of the input array,
    different modes can be specified along each axis. Default value is
    'reflect'. The valid values and their behavior is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : int or sequence, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right. By passing a sequence of origins with length equal to
    the number of dimensions of the input array, different shifts can
    be specified along each axis.

Returns
-------
minimum_filter : ndarray
    Filtered array. Has the same shape as `input`.

Notes
-----
A sequence of modes (one per axis) is only supported when the footprint is
separable. Otherwise, a single mode string must be provided.

Examples
--------
>>> from scipy import ndimage, misc
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = misc.ascent()
>>> result = ndimage.minimum_filter(ascent, size=20)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()
*)

val minimum_filter1d : ?axis:int -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:int -> input:[>`Ndarray] Np.Obj.t -> size:int -> unit -> Py.Object.t
(**
Calculate a 1-D minimum filter along the given axis.

The lines of the array along the given axis are filtered with a
minimum filter of given size.

Parameters
----------
input : array_like
    The input array.
size : int
    length along which to calculate 1D minimum
axis : int, optional
    The axis of `input` along which to calculate. Default is -1.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'reflect'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : int, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right.

Notes
-----
This function implements the MINLIST algorithm [1]_, as described by
Richard Harter [2]_, and has a guaranteed O(n) performance, `n` being
the `input` length, regardless of filter size.

References
----------
.. [1] http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.42.2777
.. [2] http://www.richardhartersworld.com/cri/2001/slidingmin.html


Examples
--------
>>> from scipy.ndimage import minimum_filter1d
>>> minimum_filter1d([2, 8, 0, 4, 1, 9, 9, 0], size=3)
array([2, 0, 0, 0, 1, 1, 0, 0])
*)

val normalize_axis_index : ?msg_prefix:string -> axis:int -> ndim:int -> unit -> int
(**
normalize_axis_index(axis, ndim, msg_prefix=None)

Normalizes an axis index, `axis`, such that is a valid positive index into
the shape of array with `ndim` dimensions. Raises an AxisError with an
appropriate message if this is not possible.

Used internally by all axis-checking logic.

.. versionadded:: 1.13.0

Parameters
----------
axis : int
    The un-normalized index of the axis. Can be negative
ndim : int
    The number of dimensions of the array that `axis` should be normalized
    against
msg_prefix : str
    A prefix to put before the message, typically the name of the argument

Returns
-------
normalized_axis : int
    The normalized axis index, such that `0 <= normalized_axis < ndim`

Raises
------
AxisError
    If the axis index is invalid, when `-ndim <= axis < ndim` is false.

Examples
--------
>>> normalize_axis_index(0, ndim=3)
0
>>> normalize_axis_index(1, ndim=3)
1
>>> normalize_axis_index(-1, ndim=3)
2

>>> normalize_axis_index(3, ndim=3)
Traceback (most recent call last):
...
AxisError: axis 3 is out of bounds for array of dimension 3
>>> normalize_axis_index(-4, ndim=3, msg_prefix='axes_arg')
Traceback (most recent call last):
...
AxisError: axes_arg: axis -4 is out of bounds for array of dimension 3
*)

val percentile_filter : ?size:[`Bool of bool | `F of float | `S of string | `Tuple of Py.Object.t | `I of int] -> ?footprint:[>`Ndarray] Np.Obj.t -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`I of int | `Sequence of Py.Object.t] -> input:[>`Ndarray] Np.Obj.t -> percentile:[`F of float | `I of int | `Bool of bool | `S of string] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Calculate a multidimensional percentile filter.

Parameters
----------
input : array_like
    The input array.
percentile : scalar
    The percentile parameter may be less then zero, i.e.,
    percentile = -20 equals percentile = 80
size : scalar or tuple, optional
    See footprint, below. Ignored if footprint is given.
footprint : array, optional
    Either `size` or `footprint` must be defined. `size` gives
    the shape that is taken from the input array, at every element
    position, to define the input to the filter function.
    `footprint` is a boolean array that specifies (implicitly) a
    shape, but also which of the elements within this shape will get
    passed to the filter function. Thus ``size=(n,m)`` is equivalent
    to ``footprint=np.ones((n,m))``.  We adjust `size` to the number
    of dimensions of the input array, so that, if the input array is
    shape (10,10,10), and `size` is 2, then the actual size used is
    (2,2,2). When `footprint` is given, `size` is ignored.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'reflect'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : int or sequence, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right. By passing a sequence of origins with length equal to
    the number of dimensions of the input array, different shifts can
    be specified along each axis.

Returns
-------
percentile_filter : ndarray
    Filtered array. Has the same shape as `input`.

Examples
--------
>>> from scipy import ndimage, misc
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = misc.ascent()
>>> result = ndimage.percentile_filter(ascent, percentile=20, size=20)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()
*)

val prewitt : ?axis:int -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Sequence of Py.Object.t | `S of string] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Calculate a Prewitt filter.

Parameters
----------
input : array_like
    The input array.
axis : int, optional
    The axis of `input` along which to calculate. Default is -1.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : str or sequence, optional
    The `mode` parameter determines how the input array is extended
    when the filter overlaps a border. By passing a sequence of modes
    with length equal to the number of dimensions of the input array,
    different modes can be specified along each axis. Default value is
    'reflect'. The valid values and their behavior is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.

Examples
--------
>>> from scipy import ndimage, misc
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = misc.ascent()
>>> result = ndimage.prewitt(ascent)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()
*)

val rank_filter : ?size:[`Bool of bool | `F of float | `S of string | `Tuple of Py.Object.t | `I of int] -> ?footprint:[>`Ndarray] Np.Obj.t -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`I of int | `Sequence of Py.Object.t] -> input:[>`Ndarray] Np.Obj.t -> rank:int -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Calculate a multidimensional rank filter.

Parameters
----------
input : array_like
    The input array.
rank : int
    The rank parameter may be less then zero, i.e., rank = -1
    indicates the largest element.
size : scalar or tuple, optional
    See footprint, below. Ignored if footprint is given.
footprint : array, optional
    Either `size` or `footprint` must be defined. `size` gives
    the shape that is taken from the input array, at every element
    position, to define the input to the filter function.
    `footprint` is a boolean array that specifies (implicitly) a
    shape, but also which of the elements within this shape will get
    passed to the filter function. Thus ``size=(n,m)`` is equivalent
    to ``footprint=np.ones((n,m))``.  We adjust `size` to the number
    of dimensions of the input array, so that, if the input array is
    shape (10,10,10), and `size` is 2, then the actual size used is
    (2,2,2). When `footprint` is given, `size` is ignored.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'reflect'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : int or sequence, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right. By passing a sequence of origins with length equal to
    the number of dimensions of the input array, different shifts can
    be specified along each axis.

Returns
-------
rank_filter : ndarray
    Filtered array. Has the same shape as `input`.

Examples
--------
>>> from scipy import ndimage, misc
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = misc.ascent()
>>> result = ndimage.rank_filter(ascent, rank=42, size=20)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()
*)

val sobel : ?axis:int -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Sequence of Py.Object.t | `S of string] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Calculate a Sobel filter.

Parameters
----------
input : array_like
    The input array.
axis : int, optional
    The axis of `input` along which to calculate. Default is -1.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : str or sequence, optional
    The `mode` parameter determines how the input array is extended
    when the filter overlaps a border. By passing a sequence of modes
    with length equal to the number of dimensions of the input array,
    different modes can be specified along each axis. Default value is
    'reflect'. The valid values and their behavior is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.

Examples
--------
>>> from scipy import ndimage, misc
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = misc.ascent()
>>> result = ndimage.sobel(ascent)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()
*)

val uniform_filter : ?size:[`I of int | `Is of int list] -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Sequence of Py.Object.t | `S of string] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`I of int | `Sequence of Py.Object.t] -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional uniform filter.

Parameters
----------
input : array_like
    The input array.
size : int or sequence of ints, optional
    The sizes of the uniform filter are given for each axis as a
    sequence, or as a single number, in which case the size is
    equal for all axes.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : str or sequence, optional
    The `mode` parameter determines how the input array is extended
    when the filter overlaps a border. By passing a sequence of modes
    with length equal to the number of dimensions of the input array,
    different modes can be specified along each axis. Default value is
    'reflect'. The valid values and their behavior is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : int or sequence, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right. By passing a sequence of origins with length equal to
    the number of dimensions of the input array, different shifts can
    be specified along each axis.

Returns
-------
uniform_filter : ndarray
    Filtered array. Has the same shape as `input`.

Notes
-----
The multidimensional filter is implemented as a sequence of
1-D uniform filters. The intermediate arrays are stored
in the same data type as the output. Therefore, for output types
with a limited precision, the results may be imprecise because
intermediate results may be stored with insufficient precision.

Examples
--------
>>> from scipy import ndimage, misc
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = misc.ascent()
>>> result = ndimage.uniform_filter(ascent, size=20)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()
*)

val uniform_filter1d : ?axis:int -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:int -> input:[>`Ndarray] Np.Obj.t -> size:int -> unit -> Py.Object.t
(**
Calculate a 1-D uniform filter along the given axis.

The lines of the array along the given axis are filtered with a
uniform filter of given size.

Parameters
----------
input : array_like
    The input array.
size : int
    length of uniform filter
axis : int, optional
    The axis of `input` along which to calculate. Default is -1.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'reflect'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : int, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right.

Examples
--------
>>> from scipy.ndimage import uniform_filter1d
>>> uniform_filter1d([2, 8, 0, 4, 1, 9, 9, 0], size=3)
array([4, 3, 4, 1, 4, 6, 6, 3])
*)


end

module Fourier : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val fourier_ellipsoid : ?n:int -> ?axis:int -> ?output:[>`Ndarray] Np.Obj.t -> input:[>`Ndarray] Np.Obj.t -> size:[`F of float | `Sequence of Py.Object.t] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional ellipsoid Fourier filter.

The array is multiplied with the fourier transform of a ellipsoid of
given sizes.

Parameters
----------
input : array_like
    The input array.
size : float or sequence
    The size of the box used for filtering.
    If a float, `size` is the same for all axes. If a sequence, `size` has
    to contain one value for each axis.
n : int, optional
    If `n` is negative (default), then the input is assumed to be the
    result of a complex fft.
    If `n` is larger than or equal to zero, the input is assumed to be the
    result of a real fft, and `n` gives the length of the array before
    transformation along the real transform direction.
axis : int, optional
    The axis of the real transform.
output : ndarray, optional
    If given, the result of filtering the input is placed in this array.
    None is returned in this case.

Returns
-------
fourier_ellipsoid : ndarray
    The filtered input.

Notes
-----
This function is implemented for arrays of rank 1, 2, or 3.

Examples
--------
>>> from scipy import ndimage, misc
>>> import numpy.fft
>>> import matplotlib.pyplot as plt
>>> fig, (ax1, ax2) = plt.subplots(1, 2)
>>> plt.gray()  # show the filtered result in grayscale
>>> ascent = misc.ascent()
>>> input_ = numpy.fft.fft2(ascent)
>>> result = ndimage.fourier_ellipsoid(input_, size=20)
>>> result = numpy.fft.ifft2(result)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result.real)  # the imaginary part is an artifact
>>> plt.show()
*)

val fourier_gaussian : ?n:int -> ?axis:int -> ?output:[>`Ndarray] Np.Obj.t -> input:[>`Ndarray] Np.Obj.t -> sigma:[`F of float | `Sequence of Py.Object.t] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional Gaussian fourier filter.

The array is multiplied with the fourier transform of a Gaussian
kernel.

Parameters
----------
input : array_like
    The input array.
sigma : float or sequence
    The sigma of the Gaussian kernel. If a float, `sigma` is the same for
    all axes. If a sequence, `sigma` has to contain one value for each
    axis.
n : int, optional
    If `n` is negative (default), then the input is assumed to be the
    result of a complex fft.
    If `n` is larger than or equal to zero, the input is assumed to be the
    result of a real fft, and `n` gives the length of the array before
    transformation along the real transform direction.
axis : int, optional
    The axis of the real transform.
output : ndarray, optional
    If given, the result of filtering the input is placed in this array.
    None is returned in this case.

Returns
-------
fourier_gaussian : ndarray
    The filtered input.

Examples
--------
>>> from scipy import ndimage, misc
>>> import numpy.fft
>>> import matplotlib.pyplot as plt
>>> fig, (ax1, ax2) = plt.subplots(1, 2)
>>> plt.gray()  # show the filtered result in grayscale
>>> ascent = misc.ascent()
>>> input_ = numpy.fft.fft2(ascent)
>>> result = ndimage.fourier_gaussian(input_, sigma=4)
>>> result = numpy.fft.ifft2(result)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result.real)  # the imaginary part is an artifact
>>> plt.show()
*)

val fourier_shift : ?n:int -> ?axis:int -> ?output:[>`Ndarray] Np.Obj.t -> input:[>`Ndarray] Np.Obj.t -> shift:[`F of float | `Sequence of Py.Object.t] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional Fourier shift filter.

The array is multiplied with the Fourier transform of a shift operation.

Parameters
----------
input : array_like
    The input array.
shift : float or sequence
    The size of the box used for filtering.
    If a float, `shift` is the same for all axes. If a sequence, `shift`
    has to contain one value for each axis.
n : int, optional
    If `n` is negative (default), then the input is assumed to be the
    result of a complex fft.
    If `n` is larger than or equal to zero, the input is assumed to be the
    result of a real fft, and `n` gives the length of the array before
    transformation along the real transform direction.
axis : int, optional
    The axis of the real transform.
output : ndarray, optional
    If given, the result of shifting the input is placed in this array.
    None is returned in this case.

Returns
-------
fourier_shift : ndarray
    The shifted input.

Examples
--------
>>> from scipy import ndimage, misc
>>> import matplotlib.pyplot as plt
>>> import numpy.fft
>>> fig, (ax1, ax2) = plt.subplots(1, 2)
>>> plt.gray()  # show the filtered result in grayscale
>>> ascent = misc.ascent()
>>> input_ = numpy.fft.fft2(ascent)
>>> result = ndimage.fourier_shift(input_, shift=200)
>>> result = numpy.fft.ifft2(result)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result.real)  # the imaginary part is an artifact
>>> plt.show()
*)

val fourier_uniform : ?n:int -> ?axis:int -> ?output:[>`Ndarray] Np.Obj.t -> input:[>`Ndarray] Np.Obj.t -> size:[`F of float | `Sequence of Py.Object.t] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional uniform fourier filter.

The array is multiplied with the Fourier transform of a box of given
size.

Parameters
----------
input : array_like
    The input array.
size : float or sequence
    The size of the box used for filtering.
    If a float, `size` is the same for all axes. If a sequence, `size` has
    to contain one value for each axis.
n : int, optional
    If `n` is negative (default), then the input is assumed to be the
    result of a complex fft.
    If `n` is larger than or equal to zero, the input is assumed to be the
    result of a real fft, and `n` gives the length of the array before
    transformation along the real transform direction.
axis : int, optional
    The axis of the real transform.
output : ndarray, optional
    If given, the result of filtering the input is placed in this array.
    None is returned in this case.

Returns
-------
fourier_uniform : ndarray
    The filtered input.

Examples
--------
>>> from scipy import ndimage, misc
>>> import numpy.fft
>>> import matplotlib.pyplot as plt
>>> fig, (ax1, ax2) = plt.subplots(1, 2)
>>> plt.gray()  # show the filtered result in grayscale
>>> ascent = misc.ascent()
>>> input_ = numpy.fft.fft2(ascent)
>>> result = ndimage.fourier_uniform(input_, size=20)
>>> result = numpy.fft.ifft2(result)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result.real)  # the imaginary part is an artifact
>>> plt.show()
*)

val normalize_axis_index : ?msg_prefix:string -> axis:int -> ndim:int -> unit -> int
(**
normalize_axis_index(axis, ndim, msg_prefix=None)

Normalizes an axis index, `axis`, such that is a valid positive index into
the shape of array with `ndim` dimensions. Raises an AxisError with an
appropriate message if this is not possible.

Used internally by all axis-checking logic.

.. versionadded:: 1.13.0

Parameters
----------
axis : int
    The un-normalized index of the axis. Can be negative
ndim : int
    The number of dimensions of the array that `axis` should be normalized
    against
msg_prefix : str
    A prefix to put before the message, typically the name of the argument

Returns
-------
normalized_axis : int
    The normalized axis index, such that `0 <= normalized_axis < ndim`

Raises
------
AxisError
    If the axis index is invalid, when `-ndim <= axis < ndim` is false.

Examples
--------
>>> normalize_axis_index(0, ndim=3)
0
>>> normalize_axis_index(1, ndim=3)
1
>>> normalize_axis_index(-1, ndim=3)
2

>>> normalize_axis_index(3, ndim=3)
Traceback (most recent call last):
...
AxisError: axis 3 is out of bounds for array of dimension 3
>>> normalize_axis_index(-4, ndim=3, msg_prefix='axes_arg')
Traceback (most recent call last):
...
AxisError: axes_arg: axis -4 is out of bounds for array of dimension 3
*)


end

module Interpolation : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val affine_transform : ?offset:[`F of float | `Sequence of Py.Object.t] -> ?output_shape:int list -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?order:int -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?prefilter:bool -> input:[>`Ndarray] Np.Obj.t -> matrix:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Apply an affine transformation.

Given an output image pixel index vector ``o``, the pixel value
is determined from the input image at position
``np.dot(matrix, o) + offset``.

This does 'pull' (or 'backward') resampling, transforming the output space
to the input to locate data. Affine transformations are often described in
the 'push' (or 'forward') direction, transforming input to output. If you
have a matrix for the 'push' transformation, use its inverse
(:func:`numpy.linalg.inv`) in this function.

Parameters
----------
input : array_like
    The input array.
matrix : ndarray
    The inverse coordinate transformation matrix, mapping output
    coordinates to input coordinates. If ``ndim`` is the number of
    dimensions of ``input``, the given matrix must have one of the
    following shapes:

        - ``(ndim, ndim)``: the linear transformation matrix for each
          output coordinate.
        - ``(ndim,)``: assume that the 2-D transformation matrix is
          diagonal, with the diagonal specified by the given value. A more
          efficient algorithm is then used that exploits the separability
          of the problem.
        - ``(ndim + 1, ndim + 1)``: assume that the transformation is
          specified using homogeneous coordinates [1]_. In this case, any
          value passed to ``offset`` is ignored.
        - ``(ndim, ndim + 1)``: as above, but the bottom row of a
          homogeneous transformation matrix is always ``[0, 0, ..., 1]``,
          and may be omitted.

offset : float or sequence, optional
    The offset into the array where the transform is applied. If a float,
    `offset` is the same for each axis. If a sequence, `offset` should
    contain one value for each axis.
output_shape : tuple of ints, optional
    Shape tuple.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
order : int, optional
    The order of the spline interpolation, default is 3.
    The order has to be in the range 0-5.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'constant'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
prefilter : bool, optional
    Determines if the input array is prefiltered with `spline_filter`
    before interpolation. The default is True, which will create a
    temporary `float64` array of filtered values if `order > 1`. If
    setting this to False, the output will be slightly blurred if
    `order > 1`, unless the input is prefiltered, i.e. it is the result
    of calling `spline_filter` on the original input.

Returns
-------
affine_transform : ndarray
    The transformed input.

Notes
-----
The given matrix and offset are used to find for each point in the
output the corresponding coordinates in the input by an affine
transformation. The value of the input at those coordinates is
determined by spline interpolation of the requested order. Points
outside the boundaries of the input are filled according to the given
mode.

.. versionchanged:: 0.18.0
    Previously, the exact interpretation of the affine transformation
    depended on whether the matrix was supplied as a 1-D or a
    2-D array. If a 1-D array was supplied
    to the matrix parameter, the output pixel value at index ``o``
    was determined from the input image at position
    ``matrix * (o + offset)``.

References
----------
.. [1] https://en.wikipedia.org/wiki/Homogeneous_coordinates
*)

val docfiller : Py.Object.t -> Py.Object.t
(**
None
*)

val geometric_transform : ?output_shape:int list -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?order:int -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?prefilter:bool -> ?extra_arguments:Py.Object.t -> ?extra_keywords:Py.Object.t -> input:[>`Ndarray] Np.Obj.t -> mapping:[`Scipy_LowLevelCallable of Py.Object.t | `Callable of Py.Object.t] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Apply an arbitrary geometric transform.

The given mapping function is used to find, for each point in the
output, the corresponding coordinates in the input. The value of the
input at those coordinates is determined by spline interpolation of
the requested order.

Parameters
----------
input : array_like
    The input array.
mapping : {callable, scipy.LowLevelCallable}
    A callable object that accepts a tuple of length equal to the output
    array rank, and returns the corresponding input coordinates as a tuple
    of length equal to the input array rank.
output_shape : tuple of ints, optional
    Shape tuple.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
order : int, optional
    The order of the spline interpolation, default is 3.
    The order has to be in the range 0-5.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'constant'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
prefilter : bool, optional
    Determines if the input array is prefiltered with `spline_filter`
    before interpolation. The default is True, which will create a
    temporary `float64` array of filtered values if `order > 1`. If
    setting this to False, the output will be slightly blurred if
    `order > 1`, unless the input is prefiltered, i.e. it is the result
    of calling `spline_filter` on the original input.
extra_arguments : tuple, optional
    Extra arguments passed to `mapping`.
extra_keywords : dict, optional
    Extra keywords passed to `mapping`.

Returns
-------
output : ndarray
    The filtered input.

See Also
--------
map_coordinates, affine_transform, spline_filter1d


Notes
-----
This function also accepts low-level callback functions with one
the following signatures and wrapped in `scipy.LowLevelCallable`:

.. code:: c

   int mapping(npy_intp *output_coordinates, double *input_coordinates,
               int output_rank, int input_rank, void *user_data)
   int mapping(intptr_t *output_coordinates, double *input_coordinates,
               int output_rank, int input_rank, void *user_data)

The calling function iterates over the elements of the output array,
calling the callback function at each element. The coordinates of the
current output element are passed through ``output_coordinates``. The
callback function must return the coordinates at which the input must
be interpolated in ``input_coordinates``. The rank of the input and
output arrays are given by ``input_rank`` and ``output_rank``
respectively. ``user_data`` is the data pointer provided
to `scipy.LowLevelCallable` as-is.

The callback function must return an integer error status that is zero
if something went wrong and one otherwise. If an error occurs, you should
normally set the Python error status with an informative message
before returning, otherwise a default error message is set by the
calling function.

In addition, some other low-level function pointer specifications
are accepted, but these are for backward compatibility only and should
not be used in new code.

Examples
--------
>>> import numpy as np
>>> from scipy.ndimage import geometric_transform
>>> a = np.arange(12.).reshape((4, 3))
>>> def shift_func(output_coords):
...     return (output_coords[0] - 0.5, output_coords[1] - 0.5)
...
>>> geometric_transform(a, shift_func)
array([[ 0.   ,  0.   ,  0.   ],
       [ 0.   ,  1.362,  2.738],
       [ 0.   ,  4.812,  6.187],
       [ 0.   ,  8.263,  9.637]])

>>> b = [1, 2, 3, 4, 5]
>>> def shift_func(output_coords):
...     return (output_coords[0] - 3,)
...
>>> geometric_transform(b, shift_func, mode='constant')
array([0, 0, 0, 1, 2])
>>> geometric_transform(b, shift_func, mode='nearest')
array([1, 1, 1, 1, 2])
>>> geometric_transform(b, shift_func, mode='reflect')
array([3, 2, 1, 1, 2])
>>> geometric_transform(b, shift_func, mode='wrap')
array([2, 3, 4, 1, 2])
*)

val map_coordinates : ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?order:int -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?prefilter:bool -> input:[>`Ndarray] Np.Obj.t -> coordinates:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Map the input array to new coordinates by interpolation.

The array of coordinates is used to find, for each point in the output,
the corresponding coordinates in the input. The value of the input at
those coordinates is determined by spline interpolation of the
requested order.

The shape of the output is derived from that of the coordinate
array by dropping the first axis. The values of the array along
the first axis are the coordinates in the input array at which the
output value is found.

Parameters
----------
input : array_like
    The input array.
coordinates : array_like
    The coordinates at which `input` is evaluated.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
order : int, optional
    The order of the spline interpolation, default is 3.
    The order has to be in the range 0-5.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'constant'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
prefilter : bool, optional
    Determines if the input array is prefiltered with `spline_filter`
    before interpolation. The default is True, which will create a
    temporary `float64` array of filtered values if `order > 1`. If
    setting this to False, the output will be slightly blurred if
    `order > 1`, unless the input is prefiltered, i.e. it is the result
    of calling `spline_filter` on the original input.

Returns
-------
map_coordinates : ndarray
    The result of transforming the input. The shape of the output is
    derived from that of `coordinates` by dropping the first axis.

See Also
--------
spline_filter, geometric_transform, scipy.interpolate

Examples
--------
>>> from scipy import ndimage
>>> a = np.arange(12.).reshape((4, 3))
>>> a
array([[  0.,   1.,   2.],
       [  3.,   4.,   5.],
       [  6.,   7.,   8.],
       [  9.,  10.,  11.]])
>>> ndimage.map_coordinates(a, [[0.5, 2], [0.5, 1]], order=1)
array([ 2.,  7.])

Above, the interpolated value of a[0.5, 0.5] gives output[0], while
a[2, 1] is output[1].

>>> inds = np.array([[0.5, 2], [0.5, 4]])
>>> ndimage.map_coordinates(a, inds, order=1, cval=-33.3)
array([  2. , -33.3])
>>> ndimage.map_coordinates(a, inds, order=1, mode='nearest')
array([ 2.,  8.])
>>> ndimage.map_coordinates(a, inds, order=1, cval=0, output=bool)
array([ True, False], dtype=bool)
*)

val normalize_axis_index : ?msg_prefix:string -> axis:int -> ndim:int -> unit -> int
(**
normalize_axis_index(axis, ndim, msg_prefix=None)

Normalizes an axis index, `axis`, such that is a valid positive index into
the shape of array with `ndim` dimensions. Raises an AxisError with an
appropriate message if this is not possible.

Used internally by all axis-checking logic.

.. versionadded:: 1.13.0

Parameters
----------
axis : int
    The un-normalized index of the axis. Can be negative
ndim : int
    The number of dimensions of the array that `axis` should be normalized
    against
msg_prefix : str
    A prefix to put before the message, typically the name of the argument

Returns
-------
normalized_axis : int
    The normalized axis index, such that `0 <= normalized_axis < ndim`

Raises
------
AxisError
    If the axis index is invalid, when `-ndim <= axis < ndim` is false.

Examples
--------
>>> normalize_axis_index(0, ndim=3)
0
>>> normalize_axis_index(1, ndim=3)
1
>>> normalize_axis_index(-1, ndim=3)
2

>>> normalize_axis_index(3, ndim=3)
Traceback (most recent call last):
...
AxisError: axis 3 is out of bounds for array of dimension 3
>>> normalize_axis_index(-4, ndim=3, msg_prefix='axes_arg')
Traceback (most recent call last):
...
AxisError: axes_arg: axis -4 is out of bounds for array of dimension 3
*)

val rotate : ?axes:Py.Object.t -> ?reshape:bool -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?order:int -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?prefilter:bool -> input:[>`Ndarray] Np.Obj.t -> angle:float -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Rotate an array.

The array is rotated in the plane defined by the two axes given by the
`axes` parameter using spline interpolation of the requested order.

Parameters
----------
input : array_like
    The input array.
angle : float
    The rotation angle in degrees.
axes : tuple of 2 ints, optional
    The two axes that define the plane of rotation. Default is the first
    two axes.
reshape : bool, optional
    If `reshape` is true, the output shape is adapted so that the input
    array is contained completely in the output. Default is True.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
order : int, optional
    The order of the spline interpolation, default is 3.
    The order has to be in the range 0-5.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'constant'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
prefilter : bool, optional
    Determines if the input array is prefiltered with `spline_filter`
    before interpolation. The default is True, which will create a
    temporary `float64` array of filtered values if `order > 1`. If
    setting this to False, the output will be slightly blurred if
    `order > 1`, unless the input is prefiltered, i.e. it is the result
    of calling `spline_filter` on the original input.

Returns
-------
rotate : ndarray
    The rotated input.

Examples
--------
>>> from scipy import ndimage, misc
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure(figsize=(10, 3))
>>> ax1, ax2, ax3 = fig.subplots(1, 3)
>>> img = misc.ascent()
>>> img_45 = ndimage.rotate(img, 45, reshape=False)
>>> full_img_45 = ndimage.rotate(img, 45, reshape=True)
>>> ax1.imshow(img, cmap='gray')
>>> ax1.set_axis_off()
>>> ax2.imshow(img_45, cmap='gray')
>>> ax2.set_axis_off()
>>> ax3.imshow(full_img_45, cmap='gray')
>>> ax3.set_axis_off()
>>> fig.set_tight_layout(True)
>>> plt.show()
>>> print(img.shape)
(512, 512)
>>> print(img_45.shape)
(512, 512)
>>> print(full_img_45.shape)
(724, 724)
*)

val shift : ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?order:int -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?prefilter:bool -> input:[>`Ndarray] Np.Obj.t -> shift:[`F of float | `Sequence of Py.Object.t] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Shift an array.

The array is shifted using spline interpolation of the requested order.
Points outside the boundaries of the input are filled according to the
given mode.

Parameters
----------
input : array_like
    The input array.
shift : float or sequence
    The shift along the axes. If a float, `shift` is the same for each
    axis. If a sequence, `shift` should contain one value for each axis.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
order : int, optional
    The order of the spline interpolation, default is 3.
    The order has to be in the range 0-5.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'constant'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
prefilter : bool, optional
    Determines if the input array is prefiltered with `spline_filter`
    before interpolation. The default is True, which will create a
    temporary `float64` array of filtered values if `order > 1`. If
    setting this to False, the output will be slightly blurred if
    `order > 1`, unless the input is prefiltered, i.e. it is the result
    of calling `spline_filter` on the original input.

Returns
-------
shift : ndarray
    The shifted input.
*)

val spline_filter : ?order:Py.Object.t -> ?output:Py.Object.t -> ?mode:Py.Object.t -> input:Py.Object.t -> unit -> Py.Object.t
(**
Multidimensional spline filter.

For more details, see `spline_filter1d`.

See Also
--------
spline_filter1d : Calculate a 1-D spline filter along the given axis.

Notes
-----
The multidimensional filter is implemented as a sequence of
1-D spline filters. The intermediate arrays are stored
in the same data type as the output. Therefore, for output types
with a limited precision, the results may be imprecise because
intermediate results may be stored with insufficient precision.

Examples
--------
We can filter an image using multidimentional splines:

>>> from scipy.ndimage import spline_filter
>>> import matplotlib.pyplot as plt
>>> orig_img = np.eye(20)  # create an image
>>> orig_img[10, :] = 1.0
>>> sp_filter = spline_filter(orig_img, order=3)
>>> f, ax = plt.subplots(1, 2, sharex=True)
>>> for ind, data in enumerate([[orig_img, 'original image'],
...                             [sp_filter, 'spline filter']]):
...     ax[ind].imshow(data[0], cmap='gray_r')
...     ax[ind].set_title(data[1])
>>> plt.tight_layout()
>>> plt.show()
*)

val spline_filter1d : ?order:int -> ?axis:int -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Calculate a 1-D spline filter along the given axis.

The lines of the array along the given axis are filtered by a
spline filter. The order of the spline must be >= 2 and <= 5.

Parameters
----------
input : array_like
    The input array.
order : int, optional
    The order of the spline, default is 3.
axis : int, optional
    The axis along which the spline filter is applied. Default is the last
    axis.
output : ndarray or dtype, optional
    The array in which to place the output, or the dtype of the returned
    array. Default is ``numpy.float64``.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'constant'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.

Returns
-------
spline_filter1d : ndarray
    The filtered input.

Notes
-----
All functions in `ndimage.interpolation` do spline interpolation of
the input image. If using B-splines of `order > 1`, the input image
values have to be converted to B-spline coefficients first, which is
done by applying this 1-D filter sequentially along all
axes of the input. All functions that require B-spline coefficients
will automatically filter their inputs, a behavior controllable with
the `prefilter` keyword argument. For functions that accept a `mode`
parameter, the result will only be correct if it matches the `mode`
used when filtering.

See Also
--------
spline_filter : Multidimensional spline filter.

Examples
--------
We can filter an image using 1-D spline along the given axis:

>>> from scipy.ndimage import spline_filter1d
>>> import matplotlib.pyplot as plt
>>> orig_img = np.eye(20)  # create an image
>>> orig_img[10, :] = 1.0
>>> sp_filter_axis_0 = spline_filter1d(orig_img, axis=0)
>>> sp_filter_axis_1 = spline_filter1d(orig_img, axis=1)
>>> f, ax = plt.subplots(1, 3, sharex=True)
>>> for ind, data in enumerate([[orig_img, 'original image'],
...             [sp_filter_axis_0, 'spline filter (axis=0)'],
...             [sp_filter_axis_1, 'spline filter (axis=1)']]):
...     ax[ind].imshow(data[0], cmap='gray_r')
...     ax[ind].set_title(data[1])
>>> plt.tight_layout()
>>> plt.show()
*)

val zoom : ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?order:int -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?prefilter:bool -> input:[>`Ndarray] Np.Obj.t -> zoom:[`F of float | `Sequence of Py.Object.t] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Zoom an array.

The array is zoomed using spline interpolation of the requested order.

Parameters
----------
input : array_like
    The input array.
zoom : float or sequence
    The zoom factor along the axes. If a float, `zoom` is the same for each
    axis. If a sequence, `zoom` should contain one value for each axis.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
order : int, optional
    The order of the spline interpolation, default is 3.
    The order has to be in the range 0-5.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'constant'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
prefilter : bool, optional
    Determines if the input array is prefiltered with `spline_filter`
    before interpolation. The default is True, which will create a
    temporary `float64` array of filtered values if `order > 1`. If
    setting this to False, the output will be slightly blurred if
    `order > 1`, unless the input is prefiltered, i.e. it is the result
    of calling `spline_filter` on the original input.

Returns
-------
zoom : ndarray
    The zoomed input.

Examples
--------
>>> from scipy import ndimage, misc
>>> import matplotlib.pyplot as plt

>>> fig = plt.figure()
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = misc.ascent()
>>> result = ndimage.zoom(ascent, 3.0)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()

>>> print(ascent.shape)
(512, 512)

>>> print(result.shape)
(1536, 1536)
*)


end

module Measurements : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val center_of_mass : ?labels:[>`Ndarray] Np.Obj.t -> ?index:[`I of int | `Is of int list] -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Calculate the center of mass of the values of an array at labels.

Parameters
----------
input : ndarray
    Data from which to calculate center-of-mass. The masses can either
    be positive or negative.
labels : ndarray, optional
    Labels for objects in `input`, as generated by `ndimage.label`.
    Only used with `index`. Dimensions must be the same as `input`.
index : int or sequence of ints, optional
    Labels for which to calculate centers-of-mass. If not specified,
    all labels greater than zero are used. Only used with `labels`.

Returns
-------
center_of_mass : tuple, or list of tuples
    Coordinates of centers-of-mass.

Examples
--------
>>> a = np.array(([0,0,0,0],
...               [0,1,1,0],
...               [0,1,1,0],
...               [0,1,1,0]))
>>> from scipy import ndimage
>>> ndimage.measurements.center_of_mass(a)
(2.0, 1.5)

Calculation of multiple objects in an image

>>> b = np.array(([0,1,1,0],
...               [0,1,0,0],
...               [0,0,0,0],
...               [0,0,1,1],
...               [0,0,1,1]))
>>> lbl = ndimage.label(b)[0]
>>> ndimage.measurements.center_of_mass(b, lbl, [1,2])
[(0.33333333333333331, 1.3333333333333333), (3.5, 2.5)]

Negative masses are also accepted, which can occur for example when
bias is removed from measured data due to random noise.

>>> c = np.array(([-1,0,0,0],
...               [0,-1,-1,0],
...               [0,1,-1,0],
...               [0,1,1,0]))
>>> ndimage.measurements.center_of_mass(c)
(-4.0, 1.0)

If there are division by zero issues, the function does not raise an
error but rather issues a RuntimeWarning before returning inf and/or NaN.

>>> d = np.array([-1, 1])
>>> ndimage.measurements.center_of_mass(d)
(inf,)
*)

val extrema : ?labels:[>`Ndarray] Np.Obj.t -> ?index:[`I of int | `Is of int list] -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Calculate the minimums and maximums of the values of an array
at labels, along with their positions.

Parameters
----------
input : ndarray
    N-D image data to process.
labels : ndarray, optional
    Labels of features in input.
    If not None, must be same shape as `input`.
index : int or sequence of ints, optional
    Labels to include in output.  If None (default), all values where
    non-zero `labels` are used.

Returns
-------
minimums, maximums : int or ndarray
    Values of minimums and maximums in each feature.
min_positions, max_positions : tuple or list of tuples
    Each tuple gives the N-D coordinates of the corresponding minimum
    or maximum.

See Also
--------
maximum, minimum, maximum_position, minimum_position, center_of_mass

Examples
--------
>>> a = np.array([[1, 2, 0, 0],
...               [5, 3, 0, 4],
...               [0, 0, 0, 7],
...               [9, 3, 0, 0]])
>>> from scipy import ndimage
>>> ndimage.extrema(a)
(0, 9, (0, 2), (3, 0))

Features to process can be specified using `labels` and `index`:

>>> lbl, nlbl = ndimage.label(a)
>>> ndimage.extrema(a, lbl, index=np.arange(1, nlbl+1))
(array([1, 4, 3]),
 array([5, 7, 9]),
 [(0, 0), (1, 3), (3, 1)],
 [(1, 0), (2, 3), (3, 0)])

If no index is given, non-zero `labels` are processed:

>>> ndimage.extrema(a, lbl)
(1, 9, (0, 0), (3, 0))
*)

val find_objects : ?max_label:int -> input:Py.Object.t -> unit -> Py.Object.t
(**
Find objects in a labeled array.

Parameters
----------
input : ndarray of ints
    Array containing objects defined by different labels. Labels with
    value 0 are ignored.
max_label : int, optional
    Maximum label to be searched for in `input`. If max_label is not
    given, the positions of all objects are returned.

Returns
-------
object_slices : list of tuples
    A list of tuples, with each tuple containing N slices (with N the
    dimension of the input array). Slices correspond to the minimal
    parallelepiped that contains the object. If a number is missing,
    None is returned instead of a slice.

See Also
--------
label, center_of_mass

Notes
-----
This function is very useful for isolating a volume of interest inside
a 3-D array, that cannot be 'seen through'.

Examples
--------
>>> from scipy import ndimage
>>> a = np.zeros((6,6), dtype=int)
>>> a[2:4, 2:4] = 1
>>> a[4, 4] = 1
>>> a[:2, :3] = 2
>>> a[0, 5] = 3
>>> a
array([[2, 2, 2, 0, 0, 3],
       [2, 2, 2, 0, 0, 0],
       [0, 0, 1, 1, 0, 0],
       [0, 0, 1, 1, 0, 0],
       [0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 0]])
>>> ndimage.find_objects(a)
[(slice(2, 5, None), slice(2, 5, None)), (slice(0, 2, None), slice(0, 3, None)), (slice(0, 1, None), slice(5, 6, None))]
>>> ndimage.find_objects(a, max_label=2)
[(slice(2, 5, None), slice(2, 5, None)), (slice(0, 2, None), slice(0, 3, None))]
>>> ndimage.find_objects(a == 1, max_label=2)
[(slice(2, 5, None), slice(2, 5, None)), None]

>>> loc = ndimage.find_objects(a)[0]
>>> a[loc]
array([[1, 1, 0],
       [1, 1, 0],
       [0, 0, 1]])
*)

val histogram : ?labels:[>`Ndarray] Np.Obj.t -> ?index:[`I of int | `Is of int list] -> input:[>`Ndarray] Np.Obj.t -> min:Py.Object.t -> max:Py.Object.t -> bins:int -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Calculate the histogram of the values of an array, optionally at labels.

Histogram calculates the frequency of values in an array within bins
determined by `min`, `max`, and `bins`. The `labels` and `index`
keywords can limit the scope of the histogram to specified sub-regions
within the array.

Parameters
----------
input : array_like
    Data for which to calculate histogram.
min, max : int
    Minimum and maximum values of range of histogram bins.
bins : int
    Number of bins.
labels : array_like, optional
    Labels for objects in `input`.
    If not None, must be same shape as `input`.
index : int or sequence of ints, optional
    Label or labels for which to calculate histogram. If None, all values
    where label is greater than zero are used

Returns
-------
hist : ndarray
    Histogram counts.

Examples
--------
>>> a = np.array([[ 0.    ,  0.2146,  0.5962,  0.    ],
...               [ 0.    ,  0.7778,  0.    ,  0.    ],
...               [ 0.    ,  0.    ,  0.    ,  0.    ],
...               [ 0.    ,  0.    ,  0.7181,  0.2787],
...               [ 0.    ,  0.    ,  0.6573,  0.3094]])
>>> from scipy import ndimage
>>> ndimage.measurements.histogram(a, 0, 1, 10)
array([13,  0,  2,  1,  0,  1,  1,  2,  0,  0])

With labels and no indices, non-zero elements are counted:

>>> lbl, nlbl = ndimage.label(a)
>>> ndimage.measurements.histogram(a, 0, 1, 10, lbl)
array([0, 0, 2, 1, 0, 1, 1, 2, 0, 0])

Indices can be used to count only certain objects:

>>> ndimage.measurements.histogram(a, 0, 1, 10, lbl, 2)
array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
*)

val label : ?structure:[>`Ndarray] Np.Obj.t -> ?output:Py.Object.t -> input:[>`Ndarray] Np.Obj.t -> unit -> (Py.Object.t * int)
(**
Label features in an array.

Parameters
----------
input : array_like
    An array-like object to be labeled. Any non-zero values in `input` are
    counted as features and zero values are considered the background.
structure : array_like, optional
    A structuring element that defines feature connections.
    `structure` must be centrosymmetric
    (see Notes).
    If no structuring element is provided,
    one is automatically generated with a squared connectivity equal to
    one.  That is, for a 2-D `input` array, the default structuring element
    is::

        [[0,1,0],
         [1,1,1],
         [0,1,0]]

output : (None, data-type, array_like), optional
    If `output` is a data type, it specifies the type of the resulting
    labeled feature array.
    If `output` is an array-like object, then `output` will be updated
    with the labeled features from this function.  This function can
    operate in-place, by passing output=input.
    Note that the output must be able to store the largest label, or this
    function will raise an Exception.

Returns
-------
label : ndarray or int
    An integer ndarray where each unique feature in `input` has a unique
    label in the returned array.
num_features : int
    How many objects were found.

    If `output` is None, this function returns a tuple of
    (`labeled_array`, `num_features`).

    If `output` is a ndarray, then it will be updated with values in
    `labeled_array` and only `num_features` will be returned by this
    function.

See Also
--------
find_objects : generate a list of slices for the labeled features (or
               objects); useful for finding features' position or
               dimensions

Notes
-----
A centrosymmetric matrix is a matrix that is symmetric about the center.
See [1]_ for more information.

The `structure` matrix must be centrosymmetric to ensure
two-way connections.
For instance, if the `structure` matrix is not centrosymmetric
and is defined as::

    [[0,1,0],
     [1,1,0],
     [0,0,0]]

and the `input` is::

    [[1,2],
     [0,3]]

then the structure matrix would indicate the
entry 2 in the input is connected to 1,
but 1 is not connected to 2.

Examples
--------
Create an image with some features, then label it using the default
(cross-shaped) structuring element:

>>> from scipy.ndimage import label, generate_binary_structure
>>> a = np.array([[0,0,1,1,0,0],
...               [0,0,0,1,0,0],
...               [1,1,0,0,1,0],
...               [0,0,0,1,0,0]])
>>> labeled_array, num_features = label(a)

Each of the 4 features are labeled with a different integer:

>>> num_features
4
>>> labeled_array
array([[0, 0, 1, 1, 0, 0],
       [0, 0, 0, 1, 0, 0],
       [2, 2, 0, 0, 3, 0],
       [0, 0, 0, 4, 0, 0]])

Generate a structuring element that will consider features connected even
if they touch diagonally:

>>> s = generate_binary_structure(2,2)

or,

>>> s = [[1,1,1],
...      [1,1,1],
...      [1,1,1]]

Label the image using the new structuring element:

>>> labeled_array, num_features = label(a, structure=s)

Show the 2 labeled features (note that features 1, 3, and 4 from above are
now considered a single feature):

>>> num_features
2
>>> labeled_array
array([[0, 0, 1, 1, 0, 0],
       [0, 0, 0, 1, 0, 0],
       [2, 2, 0, 0, 1, 0],
       [0, 0, 0, 1, 0, 0]])

References
----------

.. [1] James R. Weaver, 'Centrosymmetric (cross-symmetric)
   matrices, their basic properties, eigenvalues, and
   eigenvectors.' The American Mathematical Monthly 92.10
   (1985): 711-717.
*)

val labeled_comprehension : ?pass_positions:bool -> input:[>`Ndarray] Np.Obj.t -> labels:[`Ndarray of [>`Ndarray] Np.Obj.t | `None] -> index:[`I of int | `Is of int list | `None] -> func:Py.Object.t -> out_dtype:Np.Dtype.t -> default:[`F of float | `I of int | `None] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Roughly equivalent to [func(input[labels == i]) for i in index].

Sequentially applies an arbitrary function (that works on array_like input)
to subsets of an N-D image array specified by `labels` and `index`.
The option exists to provide the function with positional parameters as the
second argument.

Parameters
----------
input : array_like
    Data from which to select `labels` to process.
labels : array_like or None
    Labels to objects in `input`.
    If not None, array must be same shape as `input`.
    If None, `func` is applied to raveled `input`.
index : int, sequence of ints or None
    Subset of `labels` to which to apply `func`.
    If a scalar, a single value is returned.
    If None, `func` is applied to all non-zero values of `labels`.
func : callable
    Python function to apply to `labels` from `input`.
out_dtype : dtype
    Dtype to use for `result`.
default : int, float or None
    Default return value when a element of `index` does not exist
    in `labels`.
pass_positions : bool, optional
    If True, pass linear indices to `func` as a second argument.
    Default is False.

Returns
-------
result : ndarray
    Result of applying `func` to each of `labels` to `input` in `index`.

Examples
--------
>>> a = np.array([[1, 2, 0, 0],
...               [5, 3, 0, 4],
...               [0, 0, 0, 7],
...               [9, 3, 0, 0]])
>>> from scipy import ndimage
>>> lbl, nlbl = ndimage.label(a)
>>> lbls = np.arange(1, nlbl+1)
>>> ndimage.labeled_comprehension(a, lbl, lbls, np.mean, float, 0)
array([ 2.75,  5.5 ,  6.  ])

Falling back to `default`:

>>> lbls = np.arange(1, nlbl+2)
>>> ndimage.labeled_comprehension(a, lbl, lbls, np.mean, float, -1)
array([ 2.75,  5.5 ,  6.  , -1.  ])

Passing positions:

>>> def fn(val, pos):
...     print('fn says: %s : %s' % (val, pos))
...     return (val.sum()) if (pos.sum() % 2 == 0) else (-val.sum())
...
>>> ndimage.labeled_comprehension(a, lbl, lbls, fn, float, 0, True)
fn says: [1 2 5 3] : [0 1 4 5]
fn says: [4 7] : [ 7 11]
fn says: [9 3] : [12 13]
array([ 11.,  11., -12.,   0.])
*)

val maximum : ?labels:[>`Ndarray] Np.Obj.t -> ?index:[>`Ndarray] Np.Obj.t -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Calculate the maximum of the values of an array over labeled regions.

Parameters
----------
input : array_like
    Array_like of values. For each region specified by `labels`, the
    maximal values of `input` over the region is computed.
labels : array_like, optional
    An array of integers marking different regions over which the
    maximum value of `input` is to be computed. `labels` must have the
    same shape as `input`. If `labels` is not specified, the maximum
    over the whole array is returned.
index : array_like, optional
    A list of region labels that are taken into account for computing the
    maxima. If index is None, the maximum over all elements where `labels`
    is non-zero is returned.

Returns
-------
output : float or list of floats
    List of maxima of `input` over the regions determined by `labels` and
    whose index is in `index`. If `index` or `labels` are not specified, a
    float is returned: the maximal value of `input` if `labels` is None,
    and the maximal value of elements where `labels` is greater than zero
    if `index` is None.

See also
--------
label, minimum, median, maximum_position, extrema, sum, mean, variance,
standard_deviation

Notes
-----
The function returns a Python list and not a NumPy array, use
`np.array` to convert the list to an array.

Examples
--------
>>> a = np.arange(16).reshape((4,4))
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]])
>>> labels = np.zeros_like(a)
>>> labels[:2,:2] = 1
>>> labels[2:, 1:3] = 2
>>> labels
array([[1, 1, 0, 0],
       [1, 1, 0, 0],
       [0, 2, 2, 0],
       [0, 2, 2, 0]])
>>> from scipy import ndimage
>>> ndimage.maximum(a)
15.0
>>> ndimage.maximum(a, labels=labels, index=[1,2])
[5.0, 14.0]
>>> ndimage.maximum(a, labels=labels)
14.0

>>> b = np.array([[1, 2, 0, 0],
...               [5, 3, 0, 4],
...               [0, 0, 0, 7],
...               [9, 3, 0, 0]])
>>> labels, labels_nb = ndimage.label(b)
>>> labels
array([[1, 1, 0, 0],
       [1, 1, 0, 2],
       [0, 0, 0, 2],
       [3, 3, 0, 0]])
>>> ndimage.maximum(b, labels=labels, index=np.arange(1, labels_nb + 1))
[5.0, 7.0, 9.0]
*)

val maximum_position : ?labels:[>`Ndarray] Np.Obj.t -> ?index:[>`Ndarray] Np.Obj.t -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Find the positions of the maximums of the values of an array at labels.

For each region specified by `labels`, the position of the maximum
value of `input` within the region is returned.

Parameters
----------
input : array_like
    Array_like of values.
labels : array_like, optional
    An array of integers marking different regions over which the
    position of the maximum value of `input` is to be computed.
    `labels` must have the same shape as `input`. If `labels` is not
    specified, the location of the first maximum over the whole
    array is returned.

    The `labels` argument only works when `index` is specified.
index : array_like, optional
    A list of region labels that are taken into account for finding the
    location of the maxima. If `index` is None, the first maximum
    over all elements where `labels` is non-zero is returned.

    The `index` argument only works when `labels` is specified.

Returns
-------
output : list of tuples of ints
    List of tuples of ints that specify the location of maxima of
    `input` over the regions determined by `labels` and whose index
    is in `index`.

    If `index` or `labels` are not specified, a tuple of ints is
    returned specifying the location of the ``first`` maximal value
    of `input`.

See also
--------
label, minimum, median, maximum_position, extrema, sum, mean, variance,
standard_deviation

Examples
--------
>>> from scipy import ndimage
>>> a = np.array([[1, 2, 0, 0],
...               [5, 3, 0, 4],
...               [0, 0, 0, 7],
...               [9, 3, 0, 0]])
>>> ndimage.maximum_position(a)
(3, 0)

Features to process can be specified using `labels` and `index`:

>>> lbl = np.array([[0, 1, 2, 3],
...                 [0, 1, 2, 3],
...                 [0, 1, 2, 3],
...                 [0, 1, 2, 3]])
>>> ndimage.maximum_position(a, lbl, 1)
(1, 1)

If no index is given, non-zero `labels` are processed:

>>> ndimage.maximum_position(a, lbl)
(2, 3)

If there are no maxima, the position of the first element is returned:

>>> ndimage.maximum_position(a, lbl, 2)
(0, 2)
*)

val mean : ?labels:[>`Ndarray] Np.Obj.t -> ?index:[`I of int | `Is of int list] -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Calculate the mean of the values of an array at labels.

Parameters
----------
input : array_like
    Array on which to compute the mean of elements over distinct
    regions.
labels : array_like, optional
    Array of labels of same shape, or broadcastable to the same shape as
    `input`. All elements sharing the same label form one region over
    which the mean of the elements is computed.
index : int or sequence of ints, optional
    Labels of the objects over which the mean is to be computed.
    Default is None, in which case the mean for all values where label is
    greater than 0 is calculated.

Returns
-------
out : list
    Sequence of same length as `index`, with the mean of the different
    regions labeled by the labels in `index`.

See also
--------
variance, standard_deviation, minimum, maximum, sum, label

Examples
--------
>>> from scipy import ndimage
>>> a = np.arange(25).reshape((5,5))
>>> labels = np.zeros_like(a)
>>> labels[3:5,3:5] = 1
>>> index = np.unique(labels)
>>> labels
array([[0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 1, 1],
       [0, 0, 0, 1, 1]])
>>> index
array([0, 1])
>>> ndimage.mean(a, labels=labels, index=index)
[10.285714285714286, 21.0]
*)

val median : ?labels:[>`Ndarray] Np.Obj.t -> ?index:[>`Ndarray] Np.Obj.t -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Calculate the median of the values of an array over labeled regions.

Parameters
----------
input : array_like
    Array_like of values. For each region specified by `labels`, the
    median value of `input` over the region is computed.
labels : array_like, optional
    An array_like of integers marking different regions over which the
    median value of `input` is to be computed. `labels` must have the
    same shape as `input`. If `labels` is not specified, the median
    over the whole array is returned.
index : array_like, optional
    A list of region labels that are taken into account for computing the
    medians. If index is None, the median over all elements where `labels`
    is non-zero is returned.

Returns
-------
median : float or list of floats
    List of medians of `input` over the regions determined by `labels` and
    whose index is in `index`. If `index` or `labels` are not specified, a
    float is returned: the median value of `input` if `labels` is None,
    and the median value of elements where `labels` is greater than zero
    if `index` is None.

See also
--------
label, minimum, maximum, extrema, sum, mean, variance, standard_deviation

Notes
-----
The function returns a Python list and not a NumPy array, use
`np.array` to convert the list to an array.

Examples
--------
>>> from scipy import ndimage
>>> a = np.array([[1, 2, 0, 1],
...               [5, 3, 0, 4],
...               [0, 0, 0, 7],
...               [9, 3, 0, 0]])
>>> labels, labels_nb = ndimage.label(a)
>>> labels
array([[1, 1, 0, 2],
       [1, 1, 0, 2],
       [0, 0, 0, 2],
       [3, 3, 0, 0]])
>>> ndimage.median(a, labels=labels, index=np.arange(1, labels_nb + 1))
[2.5, 4.0, 6.0]
>>> ndimage.median(a)
1.0
>>> ndimage.median(a, labels=labels)
3.0
*)

val minimum : ?labels:[>`Ndarray] Np.Obj.t -> ?index:[>`Ndarray] Np.Obj.t -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Calculate the minimum of the values of an array over labeled regions.

Parameters
----------
input : array_like
    Array_like of values. For each region specified by `labels`, the
    minimal values of `input` over the region is computed.
labels : array_like, optional
    An array_like of integers marking different regions over which the
    minimum value of `input` is to be computed. `labels` must have the
    same shape as `input`. If `labels` is not specified, the minimum
    over the whole array is returned.
index : array_like, optional
    A list of region labels that are taken into account for computing the
    minima. If index is None, the minimum over all elements where `labels`
    is non-zero is returned.

Returns
-------
minimum : float or list of floats
    List of minima of `input` over the regions determined by `labels` and
    whose index is in `index`. If `index` or `labels` are not specified, a
    float is returned: the minimal value of `input` if `labels` is None,
    and the minimal value of elements where `labels` is greater than zero
    if `index` is None.

See also
--------
label, maximum, median, minimum_position, extrema, sum, mean, variance,
standard_deviation

Notes
-----
The function returns a Python list and not a NumPy array, use
`np.array` to convert the list to an array.

Examples
--------
>>> from scipy import ndimage
>>> a = np.array([[1, 2, 0, 0],
...               [5, 3, 0, 4],
...               [0, 0, 0, 7],
...               [9, 3, 0, 0]])
>>> labels, labels_nb = ndimage.label(a)
>>> labels
array([[1, 1, 0, 0],
       [1, 1, 0, 2],
       [0, 0, 0, 2],
       [3, 3, 0, 0]])
>>> ndimage.minimum(a, labels=labels, index=np.arange(1, labels_nb + 1))
[1.0, 4.0, 3.0]
>>> ndimage.minimum(a)
0.0
>>> ndimage.minimum(a, labels=labels)
1.0
*)

val minimum_position : ?labels:[>`Ndarray] Np.Obj.t -> ?index:[>`Ndarray] Np.Obj.t -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Find the positions of the minimums of the values of an array at labels.

Parameters
----------
input : array_like
    Array_like of values.
labels : array_like, optional
    An array of integers marking different regions over which the
    position of the minimum value of `input` is to be computed.
    `labels` must have the same shape as `input`. If `labels` is not
    specified, the location of the first minimum over the whole
    array is returned.

    The `labels` argument only works when `index` is specified.
index : array_like, optional
    A list of region labels that are taken into account for finding the
    location of the minima. If `index` is None, the ``first`` minimum
    over all elements where `labels` is non-zero is returned.

    The `index` argument only works when `labels` is specified.

Returns
-------
output : list of tuples of ints
    Tuple of ints or list of tuples of ints that specify the location
    of minima of `input` over the regions determined by `labels` and
    whose index is in `index`.

    If `index` or `labels` are not specified, a tuple of ints is
    returned specifying the location of the first minimal value of `input`.

See also
--------
label, minimum, median, maximum_position, extrema, sum, mean, variance,
standard_deviation

Examples
--------
>>> a = np.array([[10, 20, 30],
...               [40, 80, 100],
...               [1, 100, 200]])
>>> b = np.array([[1, 2, 0, 1],
...               [5, 3, 0, 4],
...               [0, 0, 0, 7],
...               [9, 3, 0, 0]])

>>> from scipy import ndimage

>>> ndimage.minimum_position(a)
(2, 0)
>>> ndimage.minimum_position(b)
(0, 2)

Features to process can be specified using `labels` and `index`:

>>> label, pos = ndimage.label(a)
>>> ndimage.minimum_position(a, label, index=np.arange(1, pos+1))
[(2, 0)]

>>> label, pos = ndimage.label(b)
>>> ndimage.minimum_position(b, label, index=np.arange(1, pos+1))
[(0, 0), (0, 3), (3, 1)]
*)

val standard_deviation : ?labels:[>`Ndarray] Np.Obj.t -> ?index:[`I of int | `Is of int list] -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Calculate the standard deviation of the values of an N-D image array,
optionally at specified sub-regions.

Parameters
----------
input : array_like
    N-D image data to process.
labels : array_like, optional
    Labels to identify sub-regions in `input`.
    If not None, must be same shape as `input`.
index : int or sequence of ints, optional
    `labels` to include in output. If None (default), all values where
    `labels` is non-zero are used.

Returns
-------
standard_deviation : float or ndarray
    Values of standard deviation, for each sub-region if `labels` and
    `index` are specified.

See Also
--------
label, variance, maximum, minimum, extrema

Examples
--------
>>> a = np.array([[1, 2, 0, 0],
...               [5, 3, 0, 4],
...               [0, 0, 0, 7],
...               [9, 3, 0, 0]])
>>> from scipy import ndimage
>>> ndimage.standard_deviation(a)
2.7585095613392387

Features to process can be specified using `labels` and `index`:

>>> lbl, nlbl = ndimage.label(a)
>>> ndimage.standard_deviation(a, lbl, index=np.arange(1, nlbl+1))
array([ 1.479,  1.5  ,  3.   ])

If no index is given, non-zero `labels` are processed:

>>> ndimage.standard_deviation(a, lbl)
2.4874685927665499
*)

val sum : ?labels:Py.Object.t -> ?index:[>`Ndarray] Np.Obj.t -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Calculate the sum of the values of the array.

Parameters
----------
input : array_like
    Values of `input` inside the regions defined by `labels`
    are summed together.
labels : array_like of ints, optional
    Assign labels to the values of the array. Has to have the same shape as
    `input`.
index : array_like, optional
    A single label number or a sequence of label numbers of
    the objects to be measured.

Returns
-------
sum : ndarray or scalar
    An array of the sums of values of `input` inside the regions defined
    by `labels` with the same shape as `index`. If 'index' is None or scalar,
    a scalar is returned.

See also
--------
mean, median

Examples
--------
>>> from scipy import ndimage
>>> input =  [0,1,2,3]
>>> labels = [1,1,2,2]
>>> ndimage.sum(input, labels, index=[1,2])
[1.0, 5.0]
>>> ndimage.sum(input, labels, index=1)
1
>>> ndimage.sum(input, labels)
6
*)

val variance : ?labels:[>`Ndarray] Np.Obj.t -> ?index:[`I of int | `Is of int list] -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Calculate the variance of the values of an N-D image array, optionally at
specified sub-regions.

Parameters
----------
input : array_like
    Nd-image data to process.
labels : array_like, optional
    Labels defining sub-regions in `input`.
    If not None, must be same shape as `input`.
index : int or sequence of ints, optional
    `labels` to include in output.  If None (default), all values where
    `labels` is non-zero are used.

Returns
-------
variance : float or ndarray
    Values of variance, for each sub-region if `labels` and `index` are
    specified.

See Also
--------
label, standard_deviation, maximum, minimum, extrema

Examples
--------
>>> a = np.array([[1, 2, 0, 0],
...               [5, 3, 0, 4],
...               [0, 0, 0, 7],
...               [9, 3, 0, 0]])
>>> from scipy import ndimage
>>> ndimage.variance(a)
7.609375

Features to process can be specified using `labels` and `index`:

>>> lbl, nlbl = ndimage.label(a)
>>> ndimage.variance(a, lbl, index=np.arange(1, nlbl+1))
array([ 2.1875,  2.25  ,  9.    ])

If no index is given, all non-zero `labels` are processed:

>>> ndimage.variance(a, lbl)
6.1875
*)

val watershed_ift : ?structure:Py.Object.t -> ?output:[>`Ndarray] Np.Obj.t -> input:[>`Ndarray] Np.Obj.t -> markers:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Apply watershed from markers using image foresting transform algorithm.

Parameters
----------
input : array_like
    Input.
markers : array_like
    Markers are points within each watershed that form the beginning
    of the process. Negative markers are considered background markers
    which are processed after the other markers.
structure : structure element, optional
    A structuring element defining the connectivity of the object can be
    provided. If None, an element is generated with a squared
    connectivity equal to one.
output : ndarray, optional
    An output array can optionally be provided. The same shape as input.

Returns
-------
watershed_ift : ndarray
    Output.  Same shape as `input`.

References
----------
.. [1] A.X. Falcao, J. Stolfi and R. de Alencar Lotufo, 'The image
       foresting transform: theory, algorithms, and applications',
       Pattern Analysis and Machine Intelligence, vol. 26, pp. 19-29, 2004.
*)


end

module Morphology : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val binary_closing : ?structure:[>`Ndarray] Np.Obj.t -> ?iterations:int -> ?output:[>`Ndarray] Np.Obj.t -> ?origin:int list -> ?mask:[>`Ndarray] Np.Obj.t -> ?border_value:int -> ?brute_force:bool -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Multidimensional binary closing with the given structuring element.

The *closing* of an input image by a structuring element is the
*erosion* of the *dilation* of the image by the structuring element.

Parameters
----------
input : array_like
    Binary array_like to be closed. Non-zero (True) elements form
    the subset to be closed.
structure : array_like, optional
    Structuring element used for the closing. Non-zero elements are
    considered True. If no structuring element is provided an element
    is generated with a square connectivity equal to one (i.e., only
    nearest neighbors are connected to the center, diagonally-connected
    elements are not considered neighbors).
iterations : int, optional
    The dilation step of the closing, then the erosion step are each
    repeated `iterations` times (one, by default). If iterations is
    less than 1, each operations is repeated until the result does
    not change anymore. Only an integer of iterations is accepted.
output : ndarray, optional
    Array of the same shape as input, into which the output is placed.
    By default, a new array is created.
origin : int or tuple of ints, optional
    Placement of the filter, by default 0.
mask : array_like, optional
    If a mask is given, only those elements with a True value at
    the corresponding mask element are modified at each iteration.

    .. versionadded:: 1.1.0
border_value : int (cast to 0 or 1), optional
    Value at the border in the output array.

    .. versionadded:: 1.1.0
brute_force : boolean, optional
    Memory condition: if False, only the pixels whose value was changed in
    the last iteration are tracked as candidates to be updated in the
    current iteration; if true al pixels are considered as candidates for
    update, regardless of what happened in the previous iteration.
    False by default.

    .. versionadded:: 1.1.0

Returns
-------
binary_closing : ndarray of bools
    Closing of the input by the structuring element.

See also
--------
grey_closing, binary_opening, binary_dilation, binary_erosion,
generate_binary_structure

Notes
-----
*Closing* [1]_ is a mathematical morphology operation [2]_ that
consists in the succession of a dilation and an erosion of the
input with the same structuring element. Closing therefore fills
holes smaller than the structuring element.

Together with *opening* (`binary_opening`), closing can be used for
noise removal.

References
----------
.. [1] https://en.wikipedia.org/wiki/Closing_%28morphology%29
.. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

Examples
--------
>>> from scipy import ndimage
>>> a = np.zeros((5,5), dtype=int)
>>> a[1:-1, 1:-1] = 1; a[2,2] = 0
>>> a
array([[0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 0, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 0, 0, 0, 0]])
>>> # Closing removes small holes
>>> ndimage.binary_closing(a).astype(int)
array([[0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 0, 0, 0, 0]])
>>> # Closing is the erosion of the dilation of the input
>>> ndimage.binary_dilation(a).astype(int)
array([[0, 1, 1, 1, 0],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [0, 1, 1, 1, 0]])
>>> ndimage.binary_erosion(ndimage.binary_dilation(a)).astype(int)
array([[0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 0, 0, 0, 0]])


>>> a = np.zeros((7,7), dtype=int)
>>> a[1:6, 2:5] = 1; a[1:3,3] = 0
>>> a
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 0, 1, 0, 0],
       [0, 0, 1, 0, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 0]])
>>> # In addition to removing holes, closing can also
>>> # coarsen boundaries with fine hollows.
>>> ndimage.binary_closing(a).astype(int)
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 0, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 0]])
>>> ndimage.binary_closing(a, structure=np.ones((2,2))).astype(int)
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 0]])
*)

val binary_dilation : ?structure:[>`Ndarray] Np.Obj.t -> ?iterations:int -> ?mask:[>`Ndarray] Np.Obj.t -> ?output:[>`Ndarray] Np.Obj.t -> ?border_value:int -> ?origin:int list -> ?brute_force:bool -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Multidimensional binary dilation with the given structuring element.

Parameters
----------
input : array_like
    Binary array_like to be dilated. Non-zero (True) elements form
    the subset to be dilated.
structure : array_like, optional
    Structuring element used for the dilation. Non-zero elements are
    considered True. If no structuring element is provided an element
    is generated with a square connectivity equal to one.
iterations : int, optional
    The dilation is repeated `iterations` times (one, by default).
    If iterations is less than 1, the dilation is repeated until the
    result does not change anymore. Only an integer of iterations is
    accepted.
mask : array_like, optional
    If a mask is given, only those elements with a True value at
    the corresponding mask element are modified at each iteration.
output : ndarray, optional
    Array of the same shape as input, into which the output is placed.
    By default, a new array is created.
border_value : int (cast to 0 or 1), optional
    Value at the border in the output array.
origin : int or tuple of ints, optional
    Placement of the filter, by default 0.
brute_force : boolean, optional
    Memory condition: if False, only the pixels whose value was changed in
    the last iteration are tracked as candidates to be updated (dilated)
    in the current iteration; if True all pixels are considered as
    candidates for dilation, regardless of what happened in the previous
    iteration. False by default.

Returns
-------
binary_dilation : ndarray of bools
    Dilation of the input by the structuring element.

See also
--------
grey_dilation, binary_erosion, binary_closing, binary_opening,
generate_binary_structure

Notes
-----
Dilation [1]_ is a mathematical morphology operation [2]_ that uses a
structuring element for expanding the shapes in an image. The binary
dilation of an image by a structuring element is the locus of the points
covered by the structuring element, when its center lies within the
non-zero points of the image.

References
----------
.. [1] https://en.wikipedia.org/wiki/Dilation_%28morphology%29
.. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

Examples
--------
>>> from scipy import ndimage
>>> a = np.zeros((5, 5))
>>> a[2, 2] = 1
>>> a
array([[ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.]])
>>> ndimage.binary_dilation(a)
array([[False, False, False, False, False],
       [False, False,  True, False, False],
       [False,  True,  True,  True, False],
       [False, False,  True, False, False],
       [False, False, False, False, False]], dtype=bool)
>>> ndimage.binary_dilation(a).astype(a.dtype)
array([[ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.],
       [ 0.,  1.,  1.,  1.,  0.],
       [ 0.,  0.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.]])
>>> # 3x3 structuring element with connectivity 1, used by default
>>> struct1 = ndimage.generate_binary_structure(2, 1)
>>> struct1
array([[False,  True, False],
       [ True,  True,  True],
       [False,  True, False]], dtype=bool)
>>> # 3x3 structuring element with connectivity 2
>>> struct2 = ndimage.generate_binary_structure(2, 2)
>>> struct2
array([[ True,  True,  True],
       [ True,  True,  True],
       [ True,  True,  True]], dtype=bool)
>>> ndimage.binary_dilation(a, structure=struct1).astype(a.dtype)
array([[ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.],
       [ 0.,  1.,  1.,  1.,  0.],
       [ 0.,  0.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.]])
>>> ndimage.binary_dilation(a, structure=struct2).astype(a.dtype)
array([[ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  1.,  1.,  1.,  0.],
       [ 0.,  1.,  1.,  1.,  0.],
       [ 0.,  1.,  1.,  1.,  0.],
       [ 0.,  0.,  0.,  0.,  0.]])
>>> ndimage.binary_dilation(a, structure=struct1,\
... iterations=2).astype(a.dtype)
array([[ 0.,  0.,  1.,  0.,  0.],
       [ 0.,  1.,  1.,  1.,  0.],
       [ 1.,  1.,  1.,  1.,  1.],
       [ 0.,  1.,  1.,  1.,  0.],
       [ 0.,  0.,  1.,  0.,  0.]])
*)

val binary_erosion : ?structure:[>`Ndarray] Np.Obj.t -> ?iterations:int -> ?mask:[>`Ndarray] Np.Obj.t -> ?output:[>`Ndarray] Np.Obj.t -> ?border_value:int -> ?origin:int list -> ?brute_force:bool -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Multidimensional binary erosion with a given structuring element.

Binary erosion is a mathematical morphology operation used for image
processing.

Parameters
----------
input : array_like
    Binary image to be eroded. Non-zero (True) elements form
    the subset to be eroded.
structure : array_like, optional
    Structuring element used for the erosion. Non-zero elements are
    considered True. If no structuring element is provided, an element
    is generated with a square connectivity equal to one.
iterations : int, optional
    The erosion is repeated `iterations` times (one, by default).
    If iterations is less than 1, the erosion is repeated until the
    result does not change anymore.
mask : array_like, optional
    If a mask is given, only those elements with a True value at
    the corresponding mask element are modified at each iteration.
output : ndarray, optional
    Array of the same shape as input, into which the output is placed.
    By default, a new array is created.
border_value : int (cast to 0 or 1), optional
    Value at the border in the output array.
origin : int or tuple of ints, optional
    Placement of the filter, by default 0.
brute_force : boolean, optional
    Memory condition: if False, only the pixels whose value was changed in
    the last iteration are tracked as candidates to be updated (eroded) in
    the current iteration; if True all pixels are considered as candidates
    for erosion, regardless of what happened in the previous iteration.
    False by default.

Returns
-------
binary_erosion : ndarray of bools
    Erosion of the input by the structuring element.

See also
--------
grey_erosion, binary_dilation, binary_closing, binary_opening,
generate_binary_structure

Notes
-----
Erosion [1]_ is a mathematical morphology operation [2]_ that uses a
structuring element for shrinking the shapes in an image. The binary
erosion of an image by a structuring element is the locus of the points
where a superimposition of the structuring element centered on the point
is entirely contained in the set of non-zero elements of the image.

References
----------
.. [1] https://en.wikipedia.org/wiki/Erosion_%28morphology%29
.. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

Examples
--------
>>> from scipy import ndimage
>>> a = np.zeros((7,7), dtype=int)
>>> a[1:6, 2:5] = 1
>>> a
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 0]])
>>> ndimage.binary_erosion(a).astype(a.dtype)
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0]])
>>> #Erosion removes objects smaller than the structure
>>> ndimage.binary_erosion(a, structure=np.ones((5,5))).astype(a.dtype)
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0]])
*)

val binary_fill_holes : ?structure:[>`Ndarray] Np.Obj.t -> ?output:[>`Ndarray] Np.Obj.t -> ?origin:int list -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Fill the holes in binary objects.


Parameters
----------
input : array_like
    N-D binary array with holes to be filled
structure : array_like, optional
    Structuring element used in the computation; large-size elements
    make computations faster but may miss holes separated from the
    background by thin regions. The default element (with a square
    connectivity equal to one) yields the intuitive result where all
    holes in the input have been filled.
output : ndarray, optional
    Array of the same shape as input, into which the output is placed.
    By default, a new array is created.
origin : int, tuple of ints, optional
    Position of the structuring element.

Returns
-------
out : ndarray
    Transformation of the initial image `input` where holes have been
    filled.

See also
--------
binary_dilation, binary_propagation, label

Notes
-----
The algorithm used in this function consists in invading the complementary
of the shapes in `input` from the outer boundary of the image,
using binary dilations. Holes are not connected to the boundary and are
therefore not invaded. The result is the complementary subset of the
invaded region.

References
----------
.. [1] https://en.wikipedia.org/wiki/Mathematical_morphology


Examples
--------
>>> from scipy import ndimage
>>> a = np.zeros((5, 5), dtype=int)
>>> a[1:4, 1:4] = 1
>>> a[2,2] = 0
>>> a
array([[0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 0, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 0, 0, 0, 0]])
>>> ndimage.binary_fill_holes(a).astype(int)
array([[0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 0, 0, 0, 0]])
>>> # Too big structuring element
>>> ndimage.binary_fill_holes(a, structure=np.ones((5,5))).astype(int)
array([[0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 0, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 0, 0, 0, 0]])
*)

val binary_hit_or_miss : ?structure1:[>`Ndarray] Np.Obj.t -> ?structure2:[>`Ndarray] Np.Obj.t -> ?output:[>`Ndarray] Np.Obj.t -> ?origin1:int list -> ?origin2:int list -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional binary hit-or-miss transform.

The hit-or-miss transform finds the locations of a given pattern
inside the input image.

Parameters
----------
input : array_like (cast to booleans)
    Binary image where a pattern is to be detected.
structure1 : array_like (cast to booleans), optional
    Part of the structuring element to be fitted to the foreground
    (non-zero elements) of `input`. If no value is provided, a
    structure of square connectivity 1 is chosen.
structure2 : array_like (cast to booleans), optional
    Second part of the structuring element that has to miss completely
    the foreground. If no value is provided, the complementary of
    `structure1` is taken.
output : ndarray, optional
    Array of the same shape as input, into which the output is placed.
    By default, a new array is created.
origin1 : int or tuple of ints, optional
    Placement of the first part of the structuring element `structure1`,
    by default 0 for a centered structure.
origin2 : int or tuple of ints, optional
    Placement of the second part of the structuring element `structure2`,
    by default 0 for a centered structure. If a value is provided for
    `origin1` and not for `origin2`, then `origin2` is set to `origin1`.

Returns
-------
binary_hit_or_miss : ndarray
    Hit-or-miss transform of `input` with the given structuring
    element (`structure1`, `structure2`).

See also
--------
binary_erosion

References
----------
.. [1] https://en.wikipedia.org/wiki/Hit-or-miss_transform

Examples
--------
>>> from scipy import ndimage
>>> a = np.zeros((7,7), dtype=int)
>>> a[1, 1] = 1; a[2:4, 2:4] = 1; a[4:6, 4:6] = 1
>>> a
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 0, 0, 0],
       [0, 0, 1, 1, 0, 0, 0],
       [0, 0, 0, 0, 1, 1, 0],
       [0, 0, 0, 0, 1, 1, 0],
       [0, 0, 0, 0, 0, 0, 0]])
>>> structure1 = np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1]])
>>> structure1
array([[1, 0, 0],
       [0, 1, 1],
       [0, 1, 1]])
>>> # Find the matches of structure1 in the array a
>>> ndimage.binary_hit_or_miss(a, structure1=structure1).astype(int)
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0]])
>>> # Change the origin of the filter
>>> # origin1=1 is equivalent to origin1=(1,1) here
>>> ndimage.binary_hit_or_miss(a, structure1=structure1,\
... origin1=1).astype(int)
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 0]])
*)

val binary_opening : ?structure:[>`Ndarray] Np.Obj.t -> ?iterations:int -> ?output:[>`Ndarray] Np.Obj.t -> ?origin:int list -> ?mask:[>`Ndarray] Np.Obj.t -> ?border_value:int -> ?brute_force:bool -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Multidimensional binary opening with the given structuring element.

The *opening* of an input image by a structuring element is the
*dilation* of the *erosion* of the image by the structuring element.

Parameters
----------
input : array_like
    Binary array_like to be opened. Non-zero (True) elements form
    the subset to be opened.
structure : array_like, optional
    Structuring element used for the opening. Non-zero elements are
    considered True. If no structuring element is provided an element
    is generated with a square connectivity equal to one (i.e., only
    nearest neighbors are connected to the center, diagonally-connected
    elements are not considered neighbors).
iterations : int, optional
    The erosion step of the opening, then the dilation step are each
    repeated `iterations` times (one, by default). If `iterations` is
    less than 1, each operation is repeated until the result does
    not change anymore. Only an integer of iterations is accepted.
output : ndarray, optional
    Array of the same shape as input, into which the output is placed.
    By default, a new array is created.
origin : int or tuple of ints, optional
    Placement of the filter, by default 0.
mask : array_like, optional
    If a mask is given, only those elements with a True value at
    the corresponding mask element are modified at each iteration.

    .. versionadded:: 1.1.0
border_value : int (cast to 0 or 1), optional
    Value at the border in the output array.

    .. versionadded:: 1.1.0
brute_force : boolean, optional
    Memory condition: if False, only the pixels whose value was changed in
    the last iteration are tracked as candidates to be updated in the
    current iteration; if true all pixels are considered as candidates for
    update, regardless of what happened in the previous iteration.
    False by default.

    .. versionadded:: 1.1.0

Returns
-------
binary_opening : ndarray of bools
    Opening of the input by the structuring element.

See also
--------
grey_opening, binary_closing, binary_erosion, binary_dilation,
generate_binary_structure

Notes
-----
*Opening* [1]_ is a mathematical morphology operation [2]_ that
consists in the succession of an erosion and a dilation of the
input with the same structuring element. Opening, therefore, removes
objects smaller than the structuring element.

Together with *closing* (`binary_closing`), opening can be used for
noise removal.

References
----------
.. [1] https://en.wikipedia.org/wiki/Opening_%28morphology%29
.. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

Examples
--------
>>> from scipy import ndimage
>>> a = np.zeros((5,5), dtype=int)
>>> a[1:4, 1:4] = 1; a[4, 4] = 1
>>> a
array([[0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 0, 0, 0, 1]])
>>> # Opening removes small objects
>>> ndimage.binary_opening(a, structure=np.ones((3,3))).astype(int)
array([[0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 0, 0, 0, 0]])
>>> # Opening can also smooth corners
>>> ndimage.binary_opening(a).astype(int)
array([[0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 1, 1, 1, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0]])
>>> # Opening is the dilation of the erosion of the input
>>> ndimage.binary_erosion(a).astype(int)
array([[0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0]])
>>> ndimage.binary_dilation(ndimage.binary_erosion(a)).astype(int)
array([[0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 1, 1, 1, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0]])
*)

val binary_propagation : ?structure:[>`Ndarray] Np.Obj.t -> ?mask:[>`Ndarray] Np.Obj.t -> ?output:[>`Ndarray] Np.Obj.t -> ?border_value:int -> ?origin:int list -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional binary propagation with the given structuring element.

Parameters
----------
input : array_like
    Binary image to be propagated inside `mask`.
structure : array_like, optional
    Structuring element used in the successive dilations. The output
    may depend on the structuring element, especially if `mask` has
    several connex components. If no structuring element is
    provided, an element is generated with a squared connectivity equal
    to one.
mask : array_like, optional
    Binary mask defining the region into which `input` is allowed to
    propagate.
output : ndarray, optional
    Array of the same shape as input, into which the output is placed.
    By default, a new array is created.
border_value : int (cast to 0 or 1), optional
    Value at the border in the output array.
origin : int or tuple of ints, optional
    Placement of the filter, by default 0.

Returns
-------
binary_propagation : ndarray
    Binary propagation of `input` inside `mask`.

Notes
-----
This function is functionally equivalent to calling binary_dilation
with the number of iterations less than one: iterative dilation until
the result does not change anymore.

The succession of an erosion and propagation inside the original image
can be used instead of an *opening* for deleting small objects while
keeping the contours of larger objects untouched.

References
----------
.. [1] http://cmm.ensmp.fr/~serra/cours/pdf/en/ch6en.pdf, slide 15.
.. [2] I.T. Young, J.J. Gerbrands, and L.J. van Vliet, 'Fundamentals of
    image processing', 1998
    ftp://qiftp.tudelft.nl/DIPimage/docs/FIP2.3.pdf

Examples
--------
>>> from scipy import ndimage
>>> input = np.zeros((8, 8), dtype=int)
>>> input[2, 2] = 1
>>> mask = np.zeros((8, 8), dtype=int)
>>> mask[1:4, 1:4] = mask[4, 4]  = mask[6:8, 6:8] = 1
>>> input
array([[0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0]])
>>> mask
array([[0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0, 0, 0, 0],
       [0, 1, 1, 1, 0, 0, 0, 0],
       [0, 1, 1, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 1],
       [0, 0, 0, 0, 0, 0, 1, 1]])
>>> ndimage.binary_propagation(input, mask=mask).astype(int)
array([[0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0, 0, 0, 0],
       [0, 1, 1, 1, 0, 0, 0, 0],
       [0, 1, 1, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0]])
>>> ndimage.binary_propagation(input, mask=mask,\
... structure=np.ones((3,3))).astype(int)
array([[0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0, 0, 0, 0],
       [0, 1, 1, 1, 0, 0, 0, 0],
       [0, 1, 1, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0]])

>>> # Comparison between opening and erosion+propagation
>>> a = np.zeros((6,6), dtype=int)
>>> a[2:5, 2:5] = 1; a[0, 0] = 1; a[5, 5] = 1
>>> a
array([[1, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 1, 0],
       [0, 0, 1, 1, 1, 0],
       [0, 0, 1, 1, 1, 0],
       [0, 0, 0, 0, 0, 1]])
>>> ndimage.binary_opening(a).astype(int)
array([[0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0],
       [0, 0, 1, 1, 1, 0],
       [0, 0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0, 0]])
>>> b = ndimage.binary_erosion(a)
>>> b.astype(int)
array([[0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0]])
>>> ndimage.binary_propagation(b, mask=a).astype(int)
array([[0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 1, 0],
       [0, 0, 1, 1, 1, 0],
       [0, 0, 1, 1, 1, 0],
       [0, 0, 0, 0, 0, 0]])
*)

val black_tophat : ?size:int list -> ?footprint:Py.Object.t -> ?structure:Py.Object.t -> ?output:[>`Ndarray] Np.Obj.t -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`F of float | `I of int | `Bool of bool | `S of string] -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional black tophat filter.

Parameters
----------
input : array_like
    Input.
size : tuple of ints, optional
    Shape of a flat and full structuring element used for the filter.
    Optional if `footprint` or `structure` is provided.
footprint : array of ints, optional
    Positions of non-infinite elements of a flat structuring element
    used for the black tophat filter.
structure : array of ints, optional
    Structuring element used for the filter. `structure`
    may be a non-flat structuring element.
output : array, optional
    An array used for storing the output of the filter may be provided.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the array borders are
    handled, where `cval` is the value when mode is equal to
    'constant'. Default is 'reflect'
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : scalar, optional
    The `origin` parameter controls the placement of the filter.
    Default 0

Returns
-------
black_tophat : ndarray
    Result of the filter of `input` with `structure`.

See also
--------
white_tophat, grey_opening, grey_closing
*)

val distance_transform_bf : ?metric:string -> ?sampling:[`I of int | `Is of int list] -> ?return_distances:bool -> ?return_indices:bool -> ?distances:Py.Object.t -> ?indices:Py.Object.t -> input:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Distance transform function by a brute force algorithm.

This function calculates the distance transform of the `input`, by
replacing each foreground (non-zero) element, with its
shortest distance to the background (any zero-valued element).

In addition to the distance transform, the feature transform can
be calculated. In this case the index of the closest background
element is returned along the first axis of the result.

Parameters
----------
input : array_like
    Input
metric : str, optional
    Three types of distance metric are supported: 'euclidean', 'taxicab',
    and 'chessboard'.
sampling : {int, sequence of ints}, optional
    This parameter is only used in the case of the euclidean `metric`
    distance transform.

    The sampling along each axis can be given by the `sampling` parameter
    which should be a sequence of length equal to the input rank, or a
    single number in which the `sampling` is assumed to be equal along all
    axes.
return_distances : bool, optional
    The `return_distances` flag can be used to indicate if the distance
    transform is returned.

    The default is True.
return_indices : bool, optional
    The `return_indices` flags can be used to indicate if the feature
    transform is returned.

    The default is False.
distances : float64 ndarray, optional
    Optional output array to hold distances (if `return_distances` is
    True).
indices : int64 ndarray, optional
    Optional output array to hold indices (if `return_indices` is True).

Returns
-------
distances : ndarray
    Distance array if `return_distances` is True.
indices : ndarray
    Indices array if `return_indices` is True.

Notes
-----
This function employs a slow brute force algorithm, see also the
function distance_transform_cdt for more efficient taxicab and
chessboard algorithms.
*)

val distance_transform_cdt : ?metric:[`Chessboard | `Taxicab] -> ?return_distances:Py.Object.t -> ?return_indices:Py.Object.t -> ?distances:Py.Object.t -> ?indices:Py.Object.t -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Distance transform for chamfer type of transforms.

Parameters
----------
input : array_like
    Input
metric : {'chessboard', 'taxicab'}, optional
    The `metric` determines the type of chamfering that is done. If the
    `metric` is equal to 'taxicab' a structure is generated using
    generate_binary_structure with a squared distance equal to 1. If
    the `metric` is equal to 'chessboard', a `metric` is generated
    using generate_binary_structure with a squared distance equal to
    the dimensionality of the array. These choices correspond to the
    common interpretations of the 'taxicab' and the 'chessboard'
    distance metrics in two dimensions.

    The default for `metric` is 'chessboard'.
return_distances, return_indices : bool, optional
    The `return_distances`, and `return_indices` flags can be used to
    indicate if the distance transform, the feature transform, or both
    must be returned.

    If the feature transform is returned (``return_indices=True``),
    the index of the closest background element is returned along
    the first axis of the result.

    The `return_distances` default is True, and the
    `return_indices` default is False.
distances, indices : ndarrays of int32, optional
    The `distances` and `indices` arguments can be used to give optional
    output arrays that must be the same shape as `input`.
*)

val distance_transform_edt : ?sampling:[`F of float | `I of int | `Sequence_of_same of Py.Object.t] -> ?return_distances:bool -> ?return_indices:bool -> ?distances:[>`Ndarray] Np.Obj.t -> ?indices:[>`Ndarray] Np.Obj.t -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Exact Euclidean distance transform.

In addition to the distance transform, the feature transform can
be calculated. In this case the index of the closest background
element is returned along the first axis of the result.

Parameters
----------
input : array_like
    Input data to transform. Can be any type but will be converted
    into binary: 1 wherever input equates to True, 0 elsewhere.
sampling : float or int, or sequence of same, optional
    Spacing of elements along each dimension. If a sequence, must be of
    length equal to the input rank; if a single number, this is used for
    all axes. If not specified, a grid spacing of unity is implied.
return_distances : bool, optional
    Whether to return distance matrix. At least one of
    return_distances/return_indices must be True. Default is True.
return_indices : bool, optional
    Whether to return indices matrix. Default is False.
distances : ndarray, optional
    Used for output of distance array, must be of type float64.
indices : ndarray, optional
    Used for output of indices, must be of type int32.

Returns
-------
distance_transform_edt : ndarray or list of ndarrays
    Either distance matrix, index matrix, or a list of the two,
    depending on `return_x` flags and `distance` and `indices`
    input parameters.

Notes
-----
The Euclidean distance transform gives values of the Euclidean
distance::

                n
  y_i = sqrt(sum (x[i]-b[i])**2)
                i

where b[i] is the background point (value 0) with the smallest
Euclidean distance to input points x[i], and n is the
number of dimensions.

Examples
--------
>>> from scipy import ndimage
>>> a = np.array(([0,1,1,1,1],
...               [0,0,1,1,1],
...               [0,1,1,1,1],
...               [0,1,1,1,0],
...               [0,1,1,0,0]))
>>> ndimage.distance_transform_edt(a)
array([[ 0.    ,  1.    ,  1.4142,  2.2361,  3.    ],
       [ 0.    ,  0.    ,  1.    ,  2.    ,  2.    ],
       [ 0.    ,  1.    ,  1.4142,  1.4142,  1.    ],
       [ 0.    ,  1.    ,  1.4142,  1.    ,  0.    ],
       [ 0.    ,  1.    ,  1.    ,  0.    ,  0.    ]])

With a sampling of 2 units along x, 1 along y:

>>> ndimage.distance_transform_edt(a, sampling=[2,1])
array([[ 0.    ,  1.    ,  2.    ,  2.8284,  3.6056],
       [ 0.    ,  0.    ,  1.    ,  2.    ,  3.    ],
       [ 0.    ,  1.    ,  2.    ,  2.2361,  2.    ],
       [ 0.    ,  1.    ,  2.    ,  1.    ,  0.    ],
       [ 0.    ,  1.    ,  1.    ,  0.    ,  0.    ]])

Asking for indices as well:

>>> edt, inds = ndimage.distance_transform_edt(a, return_indices=True)
>>> inds
array([[[0, 0, 1, 1, 3],
        [1, 1, 1, 1, 3],
        [2, 2, 1, 3, 3],
        [3, 3, 4, 4, 3],
        [4, 4, 4, 4, 4]],
       [[0, 0, 1, 1, 4],
        [0, 1, 1, 1, 4],
        [0, 0, 1, 4, 4],
        [0, 0, 3, 3, 4],
        [0, 0, 3, 3, 4]]])

With arrays provided for inplace outputs:

>>> indices = np.zeros(((np.ndim(a),) + a.shape), dtype=np.int32)
>>> ndimage.distance_transform_edt(a, return_indices=True, indices=indices)
array([[ 0.    ,  1.    ,  1.4142,  2.2361,  3.    ],
       [ 0.    ,  0.    ,  1.    ,  2.    ,  2.    ],
       [ 0.    ,  1.    ,  1.4142,  1.4142,  1.    ],
       [ 0.    ,  1.    ,  1.4142,  1.    ,  0.    ],
       [ 0.    ,  1.    ,  1.    ,  0.    ,  0.    ]])
>>> indices
array([[[0, 0, 1, 1, 3],
        [1, 1, 1, 1, 3],
        [2, 2, 1, 3, 3],
        [3, 3, 4, 4, 3],
        [4, 4, 4, 4, 4]],
       [[0, 0, 1, 1, 4],
        [0, 1, 1, 1, 4],
        [0, 0, 1, 4, 4],
        [0, 0, 3, 3, 4],
        [0, 0, 3, 3, 4]]])
*)

val generate_binary_structure : rank:int -> connectivity:int -> unit -> Py.Object.t
(**
Generate a binary structure for binary morphological operations.

Parameters
----------
rank : int
     Number of dimensions of the array to which the structuring element
     will be applied, as returned by `np.ndim`.
connectivity : int
     `connectivity` determines which elements of the output array belong
     to the structure, i.e., are considered as neighbors of the central
     element. Elements up to a squared distance of `connectivity` from
     the center are considered neighbors. `connectivity` may range from 1
     (no diagonal elements are neighbors) to `rank` (all elements are
     neighbors).

Returns
-------
output : ndarray of bools
     Structuring element which may be used for binary morphological
     operations, with `rank` dimensions and all dimensions equal to 3.

See also
--------
iterate_structure, binary_dilation, binary_erosion

Notes
-----
`generate_binary_structure` can only create structuring elements with
dimensions equal to 3, i.e., minimal dimensions. For larger structuring
elements, that are useful e.g., for eroding large objects, one may either
use `iterate_structure`, or create directly custom arrays with
numpy functions such as `numpy.ones`.

Examples
--------
>>> from scipy import ndimage
>>> struct = ndimage.generate_binary_structure(2, 1)
>>> struct
array([[False,  True, False],
       [ True,  True,  True],
       [False,  True, False]], dtype=bool)
>>> a = np.zeros((5,5))
>>> a[2, 2] = 1
>>> a
array([[ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.]])
>>> b = ndimage.binary_dilation(a, structure=struct).astype(a.dtype)
>>> b
array([[ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.],
       [ 0.,  1.,  1.,  1.,  0.],
       [ 0.,  0.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.]])
>>> ndimage.binary_dilation(b, structure=struct).astype(a.dtype)
array([[ 0.,  0.,  1.,  0.,  0.],
       [ 0.,  1.,  1.,  1.,  0.],
       [ 1.,  1.,  1.,  1.,  1.],
       [ 0.,  1.,  1.,  1.,  0.],
       [ 0.,  0.,  1.,  0.,  0.]])
>>> struct = ndimage.generate_binary_structure(2, 2)
>>> struct
array([[ True,  True,  True],
       [ True,  True,  True],
       [ True,  True,  True]], dtype=bool)
>>> struct = ndimage.generate_binary_structure(3, 1)
>>> struct # no diagonal elements
array([[[False, False, False],
        [False,  True, False],
        [False, False, False]],
       [[False,  True, False],
        [ True,  True,  True],
        [False,  True, False]],
       [[False, False, False],
        [False,  True, False],
        [False, False, False]]], dtype=bool)
*)

val grey_closing : ?size:int list -> ?footprint:Py.Object.t -> ?structure:Py.Object.t -> ?output:[>`Ndarray] Np.Obj.t -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`F of float | `I of int | `Bool of bool | `S of string] -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional grayscale closing.

A grayscale closing consists in the succession of a grayscale dilation,
and a grayscale erosion.

Parameters
----------
input : array_like
    Array over which the grayscale closing is to be computed.
size : tuple of ints
    Shape of a flat and full structuring element used for the grayscale
    closing. Optional if `footprint` or `structure` is provided.
footprint : array of ints, optional
    Positions of non-infinite elements of a flat structuring element
    used for the grayscale closing.
structure : array of ints, optional
    Structuring element used for the grayscale closing. `structure`
    may be a non-flat structuring element.
output : array, optional
    An array used for storing the output of the closing may be provided.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the array borders are
    handled, where `cval` is the value when mode is equal to
    'constant'. Default is 'reflect'
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : scalar, optional
    The `origin` parameter controls the placement of the filter.
    Default 0

Returns
-------
grey_closing : ndarray
    Result of the grayscale closing of `input` with `structure`.

See also
--------
binary_closing, grey_dilation, grey_erosion, grey_opening,
generate_binary_structure

Notes
-----
The action of a grayscale closing with a flat structuring element amounts
to smoothen deep local minima, whereas binary closing fills small holes.

References
----------
.. [1] https://en.wikipedia.org/wiki/Mathematical_morphology

Examples
--------
>>> from scipy import ndimage
>>> a = np.arange(36).reshape((6,6))
>>> a[3,3] = 0
>>> a
array([[ 0,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11],
       [12, 13, 14, 15, 16, 17],
       [18, 19, 20,  0, 22, 23],
       [24, 25, 26, 27, 28, 29],
       [30, 31, 32, 33, 34, 35]])
>>> ndimage.grey_closing(a, size=(3,3))
array([[ 7,  7,  8,  9, 10, 11],
       [ 7,  7,  8,  9, 10, 11],
       [13, 13, 14, 15, 16, 17],
       [19, 19, 20, 20, 22, 23],
       [25, 25, 26, 27, 28, 29],
       [31, 31, 32, 33, 34, 35]])
>>> # Note that the local minimum a[3,3] has disappeared
*)

val grey_dilation : ?size:int list -> ?footprint:Py.Object.t -> ?structure:Py.Object.t -> ?output:[>`Ndarray] Np.Obj.t -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`F of float | `I of int | `Bool of bool | `S of string] -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Calculate a greyscale dilation, using either a structuring element,
or a footprint corresponding to a flat structuring element.

Grayscale dilation is a mathematical morphology operation. For the
simple case of a full and flat structuring element, it can be viewed
as a maximum filter over a sliding window.

Parameters
----------
input : array_like
    Array over which the grayscale dilation is to be computed.
size : tuple of ints
    Shape of a flat and full structuring element used for the grayscale
    dilation. Optional if `footprint` or `structure` is provided.
footprint : array of ints, optional
    Positions of non-infinite elements of a flat structuring element
    used for the grayscale dilation. Non-zero values give the set of
    neighbors of the center over which the maximum is chosen.
structure : array of ints, optional
    Structuring element used for the grayscale dilation. `structure`
    may be a non-flat structuring element.
output : array, optional
    An array used for storing the output of the dilation may be provided.
mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional
    The `mode` parameter determines how the array borders are
    handled, where `cval` is the value when mode is equal to
    'constant'. Default is 'reflect'
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : scalar, optional
    The `origin` parameter controls the placement of the filter.
    Default 0

Returns
-------
grey_dilation : ndarray
    Grayscale dilation of `input`.

See also
--------
binary_dilation, grey_erosion, grey_closing, grey_opening
generate_binary_structure, maximum_filter

Notes
-----
The grayscale dilation of an image input by a structuring element s defined
over a domain E is given by:

(input+s)(x) = max {input(y) + s(x-y), for y in E}

In particular, for structuring elements defined as
s(y) = 0 for y in E, the grayscale dilation computes the maximum of the
input image inside a sliding window defined by E.

Grayscale dilation [1]_ is a *mathematical morphology* operation [2]_.

References
----------
.. [1] https://en.wikipedia.org/wiki/Dilation_%28morphology%29
.. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

Examples
--------
>>> from scipy import ndimage
>>> a = np.zeros((7,7), dtype=int)
>>> a[2:5, 2:5] = 1
>>> a[4,4] = 2; a[2,3] = 3
>>> a
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 3, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 2, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0]])
>>> ndimage.grey_dilation(a, size=(3,3))
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 1, 3, 3, 3, 1, 0],
       [0, 1, 3, 3, 3, 1, 0],
       [0, 1, 3, 3, 3, 2, 0],
       [0, 1, 1, 2, 2, 2, 0],
       [0, 1, 1, 2, 2, 2, 0],
       [0, 0, 0, 0, 0, 0, 0]])
>>> ndimage.grey_dilation(a, footprint=np.ones((3,3)))
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 1, 3, 3, 3, 1, 0],
       [0, 1, 3, 3, 3, 1, 0],
       [0, 1, 3, 3, 3, 2, 0],
       [0, 1, 1, 2, 2, 2, 0],
       [0, 1, 1, 2, 2, 2, 0],
       [0, 0, 0, 0, 0, 0, 0]])
>>> s = ndimage.generate_binary_structure(2,1)
>>> s
array([[False,  True, False],
       [ True,  True,  True],
       [False,  True, False]], dtype=bool)
>>> ndimage.grey_dilation(a, footprint=s)
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 3, 1, 0, 0],
       [0, 1, 3, 3, 3, 1, 0],
       [0, 1, 1, 3, 2, 1, 0],
       [0, 1, 1, 2, 2, 2, 0],
       [0, 0, 1, 1, 2, 0, 0],
       [0, 0, 0, 0, 0, 0, 0]])
>>> ndimage.grey_dilation(a, size=(3,3), structure=np.ones((3,3)))
array([[1, 1, 1, 1, 1, 1, 1],
       [1, 2, 4, 4, 4, 2, 1],
       [1, 2, 4, 4, 4, 2, 1],
       [1, 2, 4, 4, 4, 3, 1],
       [1, 2, 2, 3, 3, 3, 1],
       [1, 2, 2, 3, 3, 3, 1],
       [1, 1, 1, 1, 1, 1, 1]])
*)

val grey_erosion : ?size:int list -> ?footprint:Py.Object.t -> ?structure:Py.Object.t -> ?output:[>`Ndarray] Np.Obj.t -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`F of float | `I of int | `Bool of bool | `S of string] -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Calculate a greyscale erosion, using either a structuring element,
or a footprint corresponding to a flat structuring element.

Grayscale erosion is a mathematical morphology operation. For the
simple case of a full and flat structuring element, it can be viewed
as a minimum filter over a sliding window.

Parameters
----------
input : array_like
    Array over which the grayscale erosion is to be computed.
size : tuple of ints
    Shape of a flat and full structuring element used for the grayscale
    erosion. Optional if `footprint` or `structure` is provided.
footprint : array of ints, optional
    Positions of non-infinite elements of a flat structuring element
    used for the grayscale erosion. Non-zero values give the set of
    neighbors of the center over which the minimum is chosen.
structure : array of ints, optional
    Structuring element used for the grayscale erosion. `structure`
    may be a non-flat structuring element.
output : array, optional
    An array used for storing the output of the erosion may be provided.
mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional
    The `mode` parameter determines how the array borders are
    handled, where `cval` is the value when mode is equal to
    'constant'. Default is 'reflect'
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : scalar, optional
    The `origin` parameter controls the placement of the filter.
    Default 0

Returns
-------
output : ndarray
    Grayscale erosion of `input`.

See also
--------
binary_erosion, grey_dilation, grey_opening, grey_closing
generate_binary_structure, minimum_filter

Notes
-----
The grayscale erosion of an image input by a structuring element s defined
over a domain E is given by:

(input+s)(x) = min {input(y) - s(x-y), for y in E}

In particular, for structuring elements defined as
s(y) = 0 for y in E, the grayscale erosion computes the minimum of the
input image inside a sliding window defined by E.

Grayscale erosion [1]_ is a *mathematical morphology* operation [2]_.

References
----------
.. [1] https://en.wikipedia.org/wiki/Erosion_%28morphology%29
.. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

Examples
--------
>>> from scipy import ndimage
>>> a = np.zeros((7,7), dtype=int)
>>> a[1:6, 1:6] = 3
>>> a[4,4] = 2; a[2,3] = 1
>>> a
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 3, 3, 3, 3, 3, 0],
       [0, 3, 3, 1, 3, 3, 0],
       [0, 3, 3, 3, 3, 3, 0],
       [0, 3, 3, 3, 2, 3, 0],
       [0, 3, 3, 3, 3, 3, 0],
       [0, 0, 0, 0, 0, 0, 0]])
>>> ndimage.grey_erosion(a, size=(3,3))
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 3, 2, 2, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0]])
>>> footprint = ndimage.generate_binary_structure(2, 1)
>>> footprint
array([[False,  True, False],
       [ True,  True,  True],
       [False,  True, False]], dtype=bool)
>>> # Diagonally-connected elements are not considered neighbors
>>> ndimage.grey_erosion(a, size=(3,3), footprint=footprint)
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 3, 1, 2, 0, 0],
       [0, 0, 3, 2, 2, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0]])
*)

val grey_opening : ?size:int list -> ?footprint:Py.Object.t -> ?structure:Py.Object.t -> ?output:[>`Ndarray] Np.Obj.t -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`F of float | `I of int | `Bool of bool | `S of string] -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional grayscale opening.

A grayscale opening consists in the succession of a grayscale erosion,
and a grayscale dilation.

Parameters
----------
input : array_like
    Array over which the grayscale opening is to be computed.
size : tuple of ints
    Shape of a flat and full structuring element used for the grayscale
    opening. Optional if `footprint` or `structure` is provided.
footprint : array of ints, optional
    Positions of non-infinite elements of a flat structuring element
    used for the grayscale opening.
structure : array of ints, optional
    Structuring element used for the grayscale opening. `structure`
    may be a non-flat structuring element.
output : array, optional
    An array used for storing the output of the opening may be provided.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the array borders are
    handled, where `cval` is the value when mode is equal to
    'constant'. Default is 'reflect'
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : scalar, optional
    The `origin` parameter controls the placement of the filter.
    Default 0

Returns
-------
grey_opening : ndarray
    Result of the grayscale opening of `input` with `structure`.

See also
--------
binary_opening, grey_dilation, grey_erosion, grey_closing
generate_binary_structure

Notes
-----
The action of a grayscale opening with a flat structuring element amounts
to smoothen high local maxima, whereas binary opening erases small objects.

References
----------
.. [1] https://en.wikipedia.org/wiki/Mathematical_morphology

Examples
--------
>>> from scipy import ndimage
>>> a = np.arange(36).reshape((6,6))
>>> a[3, 3] = 50
>>> a
array([[ 0,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11],
       [12, 13, 14, 15, 16, 17],
       [18, 19, 20, 50, 22, 23],
       [24, 25, 26, 27, 28, 29],
       [30, 31, 32, 33, 34, 35]])
>>> ndimage.grey_opening(a, size=(3,3))
array([[ 0,  1,  2,  3,  4,  4],
       [ 6,  7,  8,  9, 10, 10],
       [12, 13, 14, 15, 16, 16],
       [18, 19, 20, 22, 22, 22],
       [24, 25, 26, 27, 28, 28],
       [24, 25, 26, 27, 28, 28]])
>>> # Note that the local maximum a[3,3] has disappeared
*)

val iterate_structure : ?origin:Py.Object.t -> structure:[>`Ndarray] Np.Obj.t -> iterations:int -> unit -> Py.Object.t
(**
Iterate a structure by dilating it with itself.

Parameters
----------
structure : array_like
   Structuring element (an array of bools, for example), to be dilated with
   itself.
iterations : int
   number of dilations performed on the structure with itself
origin : optional
    If origin is None, only the iterated structure is returned. If
    not, a tuple of the iterated structure and the modified origin is
    returned.

Returns
-------
iterate_structure : ndarray of bools
    A new structuring element obtained by dilating `structure`
    (`iterations` - 1) times with itself.

See also
--------
generate_binary_structure

Examples
--------
>>> from scipy import ndimage
>>> struct = ndimage.generate_binary_structure(2, 1)
>>> struct.astype(int)
array([[0, 1, 0],
       [1, 1, 1],
       [0, 1, 0]])
>>> ndimage.iterate_structure(struct, 2).astype(int)
array([[0, 0, 1, 0, 0],
       [0, 1, 1, 1, 0],
       [1, 1, 1, 1, 1],
       [0, 1, 1, 1, 0],
       [0, 0, 1, 0, 0]])
>>> ndimage.iterate_structure(struct, 3).astype(int)
array([[0, 0, 0, 1, 0, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 1, 1, 1, 1, 1, 0],
       [1, 1, 1, 1, 1, 1, 1],
       [0, 1, 1, 1, 1, 1, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 0, 1, 0, 0, 0]])
*)

val morphological_gradient : ?size:int list -> ?footprint:Py.Object.t -> ?structure:Py.Object.t -> ?output:[>`Ndarray] Np.Obj.t -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`F of float | `I of int | `Bool of bool | `S of string] -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional morphological gradient.

The morphological gradient is calculated as the difference between a
dilation and an erosion of the input with a given structuring element.

Parameters
----------
input : array_like
    Array over which to compute the morphlogical gradient.
size : tuple of ints
    Shape of a flat and full structuring element used for the mathematical
    morphology operations. Optional if `footprint` or `structure` is
    provided. A larger `size` yields a more blurred gradient.
footprint : array of ints, optional
    Positions of non-infinite elements of a flat structuring element
    used for the morphology operations. Larger footprints
    give a more blurred morphological gradient.
structure : array of ints, optional
    Structuring element used for the morphology operations.
    `structure` may be a non-flat structuring element.
output : array, optional
    An array used for storing the output of the morphological gradient
    may be provided.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the array borders are
    handled, where `cval` is the value when mode is equal to
    'constant'. Default is 'reflect'
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : scalar, optional
    The `origin` parameter controls the placement of the filter.
    Default 0

Returns
-------
morphological_gradient : ndarray
    Morphological gradient of `input`.

See also
--------
grey_dilation, grey_erosion, gaussian_gradient_magnitude

Notes
-----
For a flat structuring element, the morphological gradient
computed at a given point corresponds to the maximal difference
between elements of the input among the elements covered by the
structuring element centered on the point.

References
----------
.. [1] https://en.wikipedia.org/wiki/Mathematical_morphology

Examples
--------
>>> from scipy import ndimage
>>> a = np.zeros((7,7), dtype=int)
>>> a[2:5, 2:5] = 1
>>> ndimage.morphological_gradient(a, size=(3,3))
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 1, 1, 1, 1, 1, 0],
       [0, 1, 1, 1, 1, 1, 0],
       [0, 1, 1, 0, 1, 1, 0],
       [0, 1, 1, 1, 1, 1, 0],
       [0, 1, 1, 1, 1, 1, 0],
       [0, 0, 0, 0, 0, 0, 0]])
>>> # The morphological gradient is computed as the difference
>>> # between a dilation and an erosion
>>> ndimage.grey_dilation(a, size=(3,3)) -\
...  ndimage.grey_erosion(a, size=(3,3))
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 1, 1, 1, 1, 1, 0],
       [0, 1, 1, 1, 1, 1, 0],
       [0, 1, 1, 0, 1, 1, 0],
       [0, 1, 1, 1, 1, 1, 0],
       [0, 1, 1, 1, 1, 1, 0],
       [0, 0, 0, 0, 0, 0, 0]])
>>> a = np.zeros((7,7), dtype=int)
>>> a[2:5, 2:5] = 1
>>> a[4,4] = 2; a[2,3] = 3
>>> a
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 3, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 2, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0]])
>>> ndimage.morphological_gradient(a, size=(3,3))
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 1, 3, 3, 3, 1, 0],
       [0, 1, 3, 3, 3, 1, 0],
       [0, 1, 3, 2, 3, 2, 0],
       [0, 1, 1, 2, 2, 2, 0],
       [0, 1, 1, 2, 2, 2, 0],
       [0, 0, 0, 0, 0, 0, 0]])
*)

val morphological_laplace : ?size:[`I of int | `Is of int list] -> ?footprint:[`Ndarray of [>`Ndarray] Np.Obj.t | `Bool of bool] -> ?structure:Py.Object.t -> ?output:[>`Ndarray] Np.Obj.t -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:Py.Object.t -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional morphological laplace.

Parameters
----------
input : array_like
    Input.
size : int or sequence of ints, optional
    See `structure`.
footprint : bool or ndarray, optional
    See `structure`.
structure : structure, optional
    Either `size`, `footprint`, or the `structure` must be provided.
output : ndarray, optional
    An output array can optionally be provided.
mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional
    The mode parameter determines how the array borders are handled.
    For 'constant' mode, values beyond borders are set to be `cval`.
    Default is 'reflect'.
cval : scalar, optional
    Value to fill past edges of input if mode is 'constant'.
    Default is 0.0
origin : origin, optional
    The origin parameter controls the placement of the filter.

Returns
-------
morphological_laplace : ndarray
    Output
*)

val white_tophat : ?size:int list -> ?footprint:Py.Object.t -> ?structure:Py.Object.t -> ?output:[>`Ndarray] Np.Obj.t -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`F of float | `I of int | `Bool of bool | `S of string] -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional white tophat filter.

Parameters
----------
input : array_like
    Input.
size : tuple of ints
    Shape of a flat and full structuring element used for the filter.
    Optional if `footprint` or `structure` is provided.
footprint : array of ints, optional
    Positions of elements of a flat structuring element
    used for the white tophat filter.
structure : array of ints, optional
    Structuring element used for the filter. `structure`
    may be a non-flat structuring element.
output : array, optional
    An array used for storing the output of the filter may be provided.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the array borders are
    handled, where `cval` is the value when mode is equal to
    'constant'. Default is 'reflect'
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'.
    Default is 0.0.
origin : scalar, optional
    The `origin` parameter controls the placement of the filter.
    Default is 0.

Returns
-------
output : ndarray
    Result of the filter of `input` with `structure`.

See also
--------
black_tophat
*)


end

val affine_transform : ?offset:[`F of float | `Sequence of Py.Object.t] -> ?output_shape:int list -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?order:int -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?prefilter:bool -> input:[>`Ndarray] Np.Obj.t -> matrix:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Apply an affine transformation.

Given an output image pixel index vector ``o``, the pixel value
is determined from the input image at position
``np.dot(matrix, o) + offset``.

This does 'pull' (or 'backward') resampling, transforming the output space
to the input to locate data. Affine transformations are often described in
the 'push' (or 'forward') direction, transforming input to output. If you
have a matrix for the 'push' transformation, use its inverse
(:func:`numpy.linalg.inv`) in this function.

Parameters
----------
input : array_like
    The input array.
matrix : ndarray
    The inverse coordinate transformation matrix, mapping output
    coordinates to input coordinates. If ``ndim`` is the number of
    dimensions of ``input``, the given matrix must have one of the
    following shapes:

        - ``(ndim, ndim)``: the linear transformation matrix for each
          output coordinate.
        - ``(ndim,)``: assume that the 2-D transformation matrix is
          diagonal, with the diagonal specified by the given value. A more
          efficient algorithm is then used that exploits the separability
          of the problem.
        - ``(ndim + 1, ndim + 1)``: assume that the transformation is
          specified using homogeneous coordinates [1]_. In this case, any
          value passed to ``offset`` is ignored.
        - ``(ndim, ndim + 1)``: as above, but the bottom row of a
          homogeneous transformation matrix is always ``[0, 0, ..., 1]``,
          and may be omitted.

offset : float or sequence, optional
    The offset into the array where the transform is applied. If a float,
    `offset` is the same for each axis. If a sequence, `offset` should
    contain one value for each axis.
output_shape : tuple of ints, optional
    Shape tuple.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
order : int, optional
    The order of the spline interpolation, default is 3.
    The order has to be in the range 0-5.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'constant'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
prefilter : bool, optional
    Determines if the input array is prefiltered with `spline_filter`
    before interpolation. The default is True, which will create a
    temporary `float64` array of filtered values if `order > 1`. If
    setting this to False, the output will be slightly blurred if
    `order > 1`, unless the input is prefiltered, i.e. it is the result
    of calling `spline_filter` on the original input.

Returns
-------
affine_transform : ndarray
    The transformed input.

Notes
-----
The given matrix and offset are used to find for each point in the
output the corresponding coordinates in the input by an affine
transformation. The value of the input at those coordinates is
determined by spline interpolation of the requested order. Points
outside the boundaries of the input are filled according to the given
mode.

.. versionchanged:: 0.18.0
    Previously, the exact interpretation of the affine transformation
    depended on whether the matrix was supplied as a 1-D or a
    2-D array. If a 1-D array was supplied
    to the matrix parameter, the output pixel value at index ``o``
    was determined from the input image at position
    ``matrix * (o + offset)``.

References
----------
.. [1] https://en.wikipedia.org/wiki/Homogeneous_coordinates
*)

val binary_closing : ?structure:[>`Ndarray] Np.Obj.t -> ?iterations:int -> ?output:[>`Ndarray] Np.Obj.t -> ?origin:int list -> ?mask:[>`Ndarray] Np.Obj.t -> ?border_value:int -> ?brute_force:bool -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Multidimensional binary closing with the given structuring element.

The *closing* of an input image by a structuring element is the
*erosion* of the *dilation* of the image by the structuring element.

Parameters
----------
input : array_like
    Binary array_like to be closed. Non-zero (True) elements form
    the subset to be closed.
structure : array_like, optional
    Structuring element used for the closing. Non-zero elements are
    considered True. If no structuring element is provided an element
    is generated with a square connectivity equal to one (i.e., only
    nearest neighbors are connected to the center, diagonally-connected
    elements are not considered neighbors).
iterations : int, optional
    The dilation step of the closing, then the erosion step are each
    repeated `iterations` times (one, by default). If iterations is
    less than 1, each operations is repeated until the result does
    not change anymore. Only an integer of iterations is accepted.
output : ndarray, optional
    Array of the same shape as input, into which the output is placed.
    By default, a new array is created.
origin : int or tuple of ints, optional
    Placement of the filter, by default 0.
mask : array_like, optional
    If a mask is given, only those elements with a True value at
    the corresponding mask element are modified at each iteration.

    .. versionadded:: 1.1.0
border_value : int (cast to 0 or 1), optional
    Value at the border in the output array.

    .. versionadded:: 1.1.0
brute_force : boolean, optional
    Memory condition: if False, only the pixels whose value was changed in
    the last iteration are tracked as candidates to be updated in the
    current iteration; if true al pixels are considered as candidates for
    update, regardless of what happened in the previous iteration.
    False by default.

    .. versionadded:: 1.1.0

Returns
-------
binary_closing : ndarray of bools
    Closing of the input by the structuring element.

See also
--------
grey_closing, binary_opening, binary_dilation, binary_erosion,
generate_binary_structure

Notes
-----
*Closing* [1]_ is a mathematical morphology operation [2]_ that
consists in the succession of a dilation and an erosion of the
input with the same structuring element. Closing therefore fills
holes smaller than the structuring element.

Together with *opening* (`binary_opening`), closing can be used for
noise removal.

References
----------
.. [1] https://en.wikipedia.org/wiki/Closing_%28morphology%29
.. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

Examples
--------
>>> from scipy import ndimage
>>> a = np.zeros((5,5), dtype=int)
>>> a[1:-1, 1:-1] = 1; a[2,2] = 0
>>> a
array([[0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 0, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 0, 0, 0, 0]])
>>> # Closing removes small holes
>>> ndimage.binary_closing(a).astype(int)
array([[0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 0, 0, 0, 0]])
>>> # Closing is the erosion of the dilation of the input
>>> ndimage.binary_dilation(a).astype(int)
array([[0, 1, 1, 1, 0],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [0, 1, 1, 1, 0]])
>>> ndimage.binary_erosion(ndimage.binary_dilation(a)).astype(int)
array([[0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 0, 0, 0, 0]])


>>> a = np.zeros((7,7), dtype=int)
>>> a[1:6, 2:5] = 1; a[1:3,3] = 0
>>> a
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 0, 1, 0, 0],
       [0, 0, 1, 0, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 0]])
>>> # In addition to removing holes, closing can also
>>> # coarsen boundaries with fine hollows.
>>> ndimage.binary_closing(a).astype(int)
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 0, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 0]])
>>> ndimage.binary_closing(a, structure=np.ones((2,2))).astype(int)
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 0]])
*)

val binary_dilation : ?structure:[>`Ndarray] Np.Obj.t -> ?iterations:int -> ?mask:[>`Ndarray] Np.Obj.t -> ?output:[>`Ndarray] Np.Obj.t -> ?border_value:int -> ?origin:int list -> ?brute_force:bool -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Multidimensional binary dilation with the given structuring element.

Parameters
----------
input : array_like
    Binary array_like to be dilated. Non-zero (True) elements form
    the subset to be dilated.
structure : array_like, optional
    Structuring element used for the dilation. Non-zero elements are
    considered True. If no structuring element is provided an element
    is generated with a square connectivity equal to one.
iterations : int, optional
    The dilation is repeated `iterations` times (one, by default).
    If iterations is less than 1, the dilation is repeated until the
    result does not change anymore. Only an integer of iterations is
    accepted.
mask : array_like, optional
    If a mask is given, only those elements with a True value at
    the corresponding mask element are modified at each iteration.
output : ndarray, optional
    Array of the same shape as input, into which the output is placed.
    By default, a new array is created.
border_value : int (cast to 0 or 1), optional
    Value at the border in the output array.
origin : int or tuple of ints, optional
    Placement of the filter, by default 0.
brute_force : boolean, optional
    Memory condition: if False, only the pixels whose value was changed in
    the last iteration are tracked as candidates to be updated (dilated)
    in the current iteration; if True all pixels are considered as
    candidates for dilation, regardless of what happened in the previous
    iteration. False by default.

Returns
-------
binary_dilation : ndarray of bools
    Dilation of the input by the structuring element.

See also
--------
grey_dilation, binary_erosion, binary_closing, binary_opening,
generate_binary_structure

Notes
-----
Dilation [1]_ is a mathematical morphology operation [2]_ that uses a
structuring element for expanding the shapes in an image. The binary
dilation of an image by a structuring element is the locus of the points
covered by the structuring element, when its center lies within the
non-zero points of the image.

References
----------
.. [1] https://en.wikipedia.org/wiki/Dilation_%28morphology%29
.. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

Examples
--------
>>> from scipy import ndimage
>>> a = np.zeros((5, 5))
>>> a[2, 2] = 1
>>> a
array([[ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.]])
>>> ndimage.binary_dilation(a)
array([[False, False, False, False, False],
       [False, False,  True, False, False],
       [False,  True,  True,  True, False],
       [False, False,  True, False, False],
       [False, False, False, False, False]], dtype=bool)
>>> ndimage.binary_dilation(a).astype(a.dtype)
array([[ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.],
       [ 0.,  1.,  1.,  1.,  0.],
       [ 0.,  0.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.]])
>>> # 3x3 structuring element with connectivity 1, used by default
>>> struct1 = ndimage.generate_binary_structure(2, 1)
>>> struct1
array([[False,  True, False],
       [ True,  True,  True],
       [False,  True, False]], dtype=bool)
>>> # 3x3 structuring element with connectivity 2
>>> struct2 = ndimage.generate_binary_structure(2, 2)
>>> struct2
array([[ True,  True,  True],
       [ True,  True,  True],
       [ True,  True,  True]], dtype=bool)
>>> ndimage.binary_dilation(a, structure=struct1).astype(a.dtype)
array([[ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.],
       [ 0.,  1.,  1.,  1.,  0.],
       [ 0.,  0.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.]])
>>> ndimage.binary_dilation(a, structure=struct2).astype(a.dtype)
array([[ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  1.,  1.,  1.,  0.],
       [ 0.,  1.,  1.,  1.,  0.],
       [ 0.,  1.,  1.,  1.,  0.],
       [ 0.,  0.,  0.,  0.,  0.]])
>>> ndimage.binary_dilation(a, structure=struct1,\
... iterations=2).astype(a.dtype)
array([[ 0.,  0.,  1.,  0.,  0.],
       [ 0.,  1.,  1.,  1.,  0.],
       [ 1.,  1.,  1.,  1.,  1.],
       [ 0.,  1.,  1.,  1.,  0.],
       [ 0.,  0.,  1.,  0.,  0.]])
*)

val binary_erosion : ?structure:[>`Ndarray] Np.Obj.t -> ?iterations:int -> ?mask:[>`Ndarray] Np.Obj.t -> ?output:[>`Ndarray] Np.Obj.t -> ?border_value:int -> ?origin:int list -> ?brute_force:bool -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Multidimensional binary erosion with a given structuring element.

Binary erosion is a mathematical morphology operation used for image
processing.

Parameters
----------
input : array_like
    Binary image to be eroded. Non-zero (True) elements form
    the subset to be eroded.
structure : array_like, optional
    Structuring element used for the erosion. Non-zero elements are
    considered True. If no structuring element is provided, an element
    is generated with a square connectivity equal to one.
iterations : int, optional
    The erosion is repeated `iterations` times (one, by default).
    If iterations is less than 1, the erosion is repeated until the
    result does not change anymore.
mask : array_like, optional
    If a mask is given, only those elements with a True value at
    the corresponding mask element are modified at each iteration.
output : ndarray, optional
    Array of the same shape as input, into which the output is placed.
    By default, a new array is created.
border_value : int (cast to 0 or 1), optional
    Value at the border in the output array.
origin : int or tuple of ints, optional
    Placement of the filter, by default 0.
brute_force : boolean, optional
    Memory condition: if False, only the pixels whose value was changed in
    the last iteration are tracked as candidates to be updated (eroded) in
    the current iteration; if True all pixels are considered as candidates
    for erosion, regardless of what happened in the previous iteration.
    False by default.

Returns
-------
binary_erosion : ndarray of bools
    Erosion of the input by the structuring element.

See also
--------
grey_erosion, binary_dilation, binary_closing, binary_opening,
generate_binary_structure

Notes
-----
Erosion [1]_ is a mathematical morphology operation [2]_ that uses a
structuring element for shrinking the shapes in an image. The binary
erosion of an image by a structuring element is the locus of the points
where a superimposition of the structuring element centered on the point
is entirely contained in the set of non-zero elements of the image.

References
----------
.. [1] https://en.wikipedia.org/wiki/Erosion_%28morphology%29
.. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

Examples
--------
>>> from scipy import ndimage
>>> a = np.zeros((7,7), dtype=int)
>>> a[1:6, 2:5] = 1
>>> a
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 0]])
>>> ndimage.binary_erosion(a).astype(a.dtype)
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0]])
>>> #Erosion removes objects smaller than the structure
>>> ndimage.binary_erosion(a, structure=np.ones((5,5))).astype(a.dtype)
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0]])
*)

val binary_fill_holes : ?structure:[>`Ndarray] Np.Obj.t -> ?output:[>`Ndarray] Np.Obj.t -> ?origin:int list -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Fill the holes in binary objects.


Parameters
----------
input : array_like
    N-D binary array with holes to be filled
structure : array_like, optional
    Structuring element used in the computation; large-size elements
    make computations faster but may miss holes separated from the
    background by thin regions. The default element (with a square
    connectivity equal to one) yields the intuitive result where all
    holes in the input have been filled.
output : ndarray, optional
    Array of the same shape as input, into which the output is placed.
    By default, a new array is created.
origin : int, tuple of ints, optional
    Position of the structuring element.

Returns
-------
out : ndarray
    Transformation of the initial image `input` where holes have been
    filled.

See also
--------
binary_dilation, binary_propagation, label

Notes
-----
The algorithm used in this function consists in invading the complementary
of the shapes in `input` from the outer boundary of the image,
using binary dilations. Holes are not connected to the boundary and are
therefore not invaded. The result is the complementary subset of the
invaded region.

References
----------
.. [1] https://en.wikipedia.org/wiki/Mathematical_morphology


Examples
--------
>>> from scipy import ndimage
>>> a = np.zeros((5, 5), dtype=int)
>>> a[1:4, 1:4] = 1
>>> a[2,2] = 0
>>> a
array([[0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 0, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 0, 0, 0, 0]])
>>> ndimage.binary_fill_holes(a).astype(int)
array([[0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 0, 0, 0, 0]])
>>> # Too big structuring element
>>> ndimage.binary_fill_holes(a, structure=np.ones((5,5))).astype(int)
array([[0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 0, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 0, 0, 0, 0]])
*)

val binary_hit_or_miss : ?structure1:[>`Ndarray] Np.Obj.t -> ?structure2:[>`Ndarray] Np.Obj.t -> ?output:[>`Ndarray] Np.Obj.t -> ?origin1:int list -> ?origin2:int list -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional binary hit-or-miss transform.

The hit-or-miss transform finds the locations of a given pattern
inside the input image.

Parameters
----------
input : array_like (cast to booleans)
    Binary image where a pattern is to be detected.
structure1 : array_like (cast to booleans), optional
    Part of the structuring element to be fitted to the foreground
    (non-zero elements) of `input`. If no value is provided, a
    structure of square connectivity 1 is chosen.
structure2 : array_like (cast to booleans), optional
    Second part of the structuring element that has to miss completely
    the foreground. If no value is provided, the complementary of
    `structure1` is taken.
output : ndarray, optional
    Array of the same shape as input, into which the output is placed.
    By default, a new array is created.
origin1 : int or tuple of ints, optional
    Placement of the first part of the structuring element `structure1`,
    by default 0 for a centered structure.
origin2 : int or tuple of ints, optional
    Placement of the second part of the structuring element `structure2`,
    by default 0 for a centered structure. If a value is provided for
    `origin1` and not for `origin2`, then `origin2` is set to `origin1`.

Returns
-------
binary_hit_or_miss : ndarray
    Hit-or-miss transform of `input` with the given structuring
    element (`structure1`, `structure2`).

See also
--------
binary_erosion

References
----------
.. [1] https://en.wikipedia.org/wiki/Hit-or-miss_transform

Examples
--------
>>> from scipy import ndimage
>>> a = np.zeros((7,7), dtype=int)
>>> a[1, 1] = 1; a[2:4, 2:4] = 1; a[4:6, 4:6] = 1
>>> a
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 0, 0, 0],
       [0, 0, 1, 1, 0, 0, 0],
       [0, 0, 0, 0, 1, 1, 0],
       [0, 0, 0, 0, 1, 1, 0],
       [0, 0, 0, 0, 0, 0, 0]])
>>> structure1 = np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1]])
>>> structure1
array([[1, 0, 0],
       [0, 1, 1],
       [0, 1, 1]])
>>> # Find the matches of structure1 in the array a
>>> ndimage.binary_hit_or_miss(a, structure1=structure1).astype(int)
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0]])
>>> # Change the origin of the filter
>>> # origin1=1 is equivalent to origin1=(1,1) here
>>> ndimage.binary_hit_or_miss(a, structure1=structure1,\
... origin1=1).astype(int)
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 0]])
*)

val binary_opening : ?structure:[>`Ndarray] Np.Obj.t -> ?iterations:int -> ?output:[>`Ndarray] Np.Obj.t -> ?origin:int list -> ?mask:[>`Ndarray] Np.Obj.t -> ?border_value:int -> ?brute_force:bool -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Multidimensional binary opening with the given structuring element.

The *opening* of an input image by a structuring element is the
*dilation* of the *erosion* of the image by the structuring element.

Parameters
----------
input : array_like
    Binary array_like to be opened. Non-zero (True) elements form
    the subset to be opened.
structure : array_like, optional
    Structuring element used for the opening. Non-zero elements are
    considered True. If no structuring element is provided an element
    is generated with a square connectivity equal to one (i.e., only
    nearest neighbors are connected to the center, diagonally-connected
    elements are not considered neighbors).
iterations : int, optional
    The erosion step of the opening, then the dilation step are each
    repeated `iterations` times (one, by default). If `iterations` is
    less than 1, each operation is repeated until the result does
    not change anymore. Only an integer of iterations is accepted.
output : ndarray, optional
    Array of the same shape as input, into which the output is placed.
    By default, a new array is created.
origin : int or tuple of ints, optional
    Placement of the filter, by default 0.
mask : array_like, optional
    If a mask is given, only those elements with a True value at
    the corresponding mask element are modified at each iteration.

    .. versionadded:: 1.1.0
border_value : int (cast to 0 or 1), optional
    Value at the border in the output array.

    .. versionadded:: 1.1.0
brute_force : boolean, optional
    Memory condition: if False, only the pixels whose value was changed in
    the last iteration are tracked as candidates to be updated in the
    current iteration; if true all pixels are considered as candidates for
    update, regardless of what happened in the previous iteration.
    False by default.

    .. versionadded:: 1.1.0

Returns
-------
binary_opening : ndarray of bools
    Opening of the input by the structuring element.

See also
--------
grey_opening, binary_closing, binary_erosion, binary_dilation,
generate_binary_structure

Notes
-----
*Opening* [1]_ is a mathematical morphology operation [2]_ that
consists in the succession of an erosion and a dilation of the
input with the same structuring element. Opening, therefore, removes
objects smaller than the structuring element.

Together with *closing* (`binary_closing`), opening can be used for
noise removal.

References
----------
.. [1] https://en.wikipedia.org/wiki/Opening_%28morphology%29
.. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

Examples
--------
>>> from scipy import ndimage
>>> a = np.zeros((5,5), dtype=int)
>>> a[1:4, 1:4] = 1; a[4, 4] = 1
>>> a
array([[0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 0, 0, 0, 1]])
>>> # Opening removes small objects
>>> ndimage.binary_opening(a, structure=np.ones((3,3))).astype(int)
array([[0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 0, 0, 0, 0]])
>>> # Opening can also smooth corners
>>> ndimage.binary_opening(a).astype(int)
array([[0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 1, 1, 1, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0]])
>>> # Opening is the dilation of the erosion of the input
>>> ndimage.binary_erosion(a).astype(int)
array([[0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0]])
>>> ndimage.binary_dilation(ndimage.binary_erosion(a)).astype(int)
array([[0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 1, 1, 1, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0]])
*)

val binary_propagation : ?structure:[>`Ndarray] Np.Obj.t -> ?mask:[>`Ndarray] Np.Obj.t -> ?output:[>`Ndarray] Np.Obj.t -> ?border_value:int -> ?origin:int list -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional binary propagation with the given structuring element.

Parameters
----------
input : array_like
    Binary image to be propagated inside `mask`.
structure : array_like, optional
    Structuring element used in the successive dilations. The output
    may depend on the structuring element, especially if `mask` has
    several connex components. If no structuring element is
    provided, an element is generated with a squared connectivity equal
    to one.
mask : array_like, optional
    Binary mask defining the region into which `input` is allowed to
    propagate.
output : ndarray, optional
    Array of the same shape as input, into which the output is placed.
    By default, a new array is created.
border_value : int (cast to 0 or 1), optional
    Value at the border in the output array.
origin : int or tuple of ints, optional
    Placement of the filter, by default 0.

Returns
-------
binary_propagation : ndarray
    Binary propagation of `input` inside `mask`.

Notes
-----
This function is functionally equivalent to calling binary_dilation
with the number of iterations less than one: iterative dilation until
the result does not change anymore.

The succession of an erosion and propagation inside the original image
can be used instead of an *opening* for deleting small objects while
keeping the contours of larger objects untouched.

References
----------
.. [1] http://cmm.ensmp.fr/~serra/cours/pdf/en/ch6en.pdf, slide 15.
.. [2] I.T. Young, J.J. Gerbrands, and L.J. van Vliet, 'Fundamentals of
    image processing', 1998
    ftp://qiftp.tudelft.nl/DIPimage/docs/FIP2.3.pdf

Examples
--------
>>> from scipy import ndimage
>>> input = np.zeros((8, 8), dtype=int)
>>> input[2, 2] = 1
>>> mask = np.zeros((8, 8), dtype=int)
>>> mask[1:4, 1:4] = mask[4, 4]  = mask[6:8, 6:8] = 1
>>> input
array([[0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0]])
>>> mask
array([[0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0, 0, 0, 0],
       [0, 1, 1, 1, 0, 0, 0, 0],
       [0, 1, 1, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 1],
       [0, 0, 0, 0, 0, 0, 1, 1]])
>>> ndimage.binary_propagation(input, mask=mask).astype(int)
array([[0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0, 0, 0, 0],
       [0, 1, 1, 1, 0, 0, 0, 0],
       [0, 1, 1, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0]])
>>> ndimage.binary_propagation(input, mask=mask,\
... structure=np.ones((3,3))).astype(int)
array([[0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0, 0, 0, 0],
       [0, 1, 1, 1, 0, 0, 0, 0],
       [0, 1, 1, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0]])

>>> # Comparison between opening and erosion+propagation
>>> a = np.zeros((6,6), dtype=int)
>>> a[2:5, 2:5] = 1; a[0, 0] = 1; a[5, 5] = 1
>>> a
array([[1, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 1, 0],
       [0, 0, 1, 1, 1, 0],
       [0, 0, 1, 1, 1, 0],
       [0, 0, 0, 0, 0, 1]])
>>> ndimage.binary_opening(a).astype(int)
array([[0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0],
       [0, 0, 1, 1, 1, 0],
       [0, 0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0, 0]])
>>> b = ndimage.binary_erosion(a)
>>> b.astype(int)
array([[0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0]])
>>> ndimage.binary_propagation(b, mask=a).astype(int)
array([[0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 1, 0],
       [0, 0, 1, 1, 1, 0],
       [0, 0, 1, 1, 1, 0],
       [0, 0, 0, 0, 0, 0]])
*)

val black_tophat : ?size:int list -> ?footprint:Py.Object.t -> ?structure:Py.Object.t -> ?output:[>`Ndarray] Np.Obj.t -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`F of float | `I of int | `Bool of bool | `S of string] -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional black tophat filter.

Parameters
----------
input : array_like
    Input.
size : tuple of ints, optional
    Shape of a flat and full structuring element used for the filter.
    Optional if `footprint` or `structure` is provided.
footprint : array of ints, optional
    Positions of non-infinite elements of a flat structuring element
    used for the black tophat filter.
structure : array of ints, optional
    Structuring element used for the filter. `structure`
    may be a non-flat structuring element.
output : array, optional
    An array used for storing the output of the filter may be provided.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the array borders are
    handled, where `cval` is the value when mode is equal to
    'constant'. Default is 'reflect'
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : scalar, optional
    The `origin` parameter controls the placement of the filter.
    Default 0

Returns
-------
black_tophat : ndarray
    Result of the filter of `input` with `structure`.

See also
--------
white_tophat, grey_opening, grey_closing
*)

val center_of_mass : ?labels:[>`Ndarray] Np.Obj.t -> ?index:[`I of int | `Is of int list] -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Calculate the center of mass of the values of an array at labels.

Parameters
----------
input : ndarray
    Data from which to calculate center-of-mass. The masses can either
    be positive or negative.
labels : ndarray, optional
    Labels for objects in `input`, as generated by `ndimage.label`.
    Only used with `index`. Dimensions must be the same as `input`.
index : int or sequence of ints, optional
    Labels for which to calculate centers-of-mass. If not specified,
    all labels greater than zero are used. Only used with `labels`.

Returns
-------
center_of_mass : tuple, or list of tuples
    Coordinates of centers-of-mass.

Examples
--------
>>> a = np.array(([0,0,0,0],
...               [0,1,1,0],
...               [0,1,1,0],
...               [0,1,1,0]))
>>> from scipy import ndimage
>>> ndimage.measurements.center_of_mass(a)
(2.0, 1.5)

Calculation of multiple objects in an image

>>> b = np.array(([0,1,1,0],
...               [0,1,0,0],
...               [0,0,0,0],
...               [0,0,1,1],
...               [0,0,1,1]))
>>> lbl = ndimage.label(b)[0]
>>> ndimage.measurements.center_of_mass(b, lbl, [1,2])
[(0.33333333333333331, 1.3333333333333333), (3.5, 2.5)]

Negative masses are also accepted, which can occur for example when
bias is removed from measured data due to random noise.

>>> c = np.array(([-1,0,0,0],
...               [0,-1,-1,0],
...               [0,1,-1,0],
...               [0,1,1,0]))
>>> ndimage.measurements.center_of_mass(c)
(-4.0, 1.0)

If there are division by zero issues, the function does not raise an
error but rather issues a RuntimeWarning before returning inf and/or NaN.

>>> d = np.array([-1, 1])
>>> ndimage.measurements.center_of_mass(d)
(inf,)
*)

val convolve : ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`I of int | `Sequence of Py.Object.t] -> input:[>`Ndarray] Np.Obj.t -> weights:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional convolution.

The array is convolved with the given kernel.

Parameters
----------
input : array_like
    The input array.
weights : array_like
    Array of weights, same number of dimensions as input
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'reflect'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0
origin : int or sequence, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right. By passing a sequence of origins with length equal to
    the number of dimensions of the input array, different shifts can
    be specified along each axis.

Returns
-------
result : ndarray
    The result of convolution of `input` with `weights`.

See Also
--------
correlate : Correlate an image with a kernel.

Notes
-----
Each value in result is :math:`C_i = \sum_j{I_{i+k-j} W_j}`, where
W is the `weights` kernel,
j is the N-D spatial index over :math:`W`,
I is the `input` and k is the coordinate of the center of
W, specified by `origin` in the input parameters.

Examples
--------
Perhaps the simplest case to understand is ``mode='constant', cval=0.0``,
because in this case borders (i.e., where the `weights` kernel, centered
on any one value, extends beyond an edge of `input`) are treated as zeros.

>>> a = np.array([[1, 2, 0, 0],
...               [5, 3, 0, 4],
...               [0, 0, 0, 7],
...               [9, 3, 0, 0]])
>>> k = np.array([[1,1,1],[1,1,0],[1,0,0]])
>>> from scipy import ndimage
>>> ndimage.convolve(a, k, mode='constant', cval=0.0)
array([[11, 10,  7,  4],
       [10,  3, 11, 11],
       [15, 12, 14,  7],
       [12,  3,  7,  0]])

Setting ``cval=1.0`` is equivalent to padding the outer edge of `input`
with 1.0's (and then extracting only the original region of the result).

>>> ndimage.convolve(a, k, mode='constant', cval=1.0)
array([[13, 11,  8,  7],
       [11,  3, 11, 14],
       [16, 12, 14, 10],
       [15,  6, 10,  5]])

With ``mode='reflect'`` (the default), outer values are reflected at the
edge of `input` to fill in missing values.

>>> b = np.array([[2, 0, 0],
...               [1, 0, 0],
...               [0, 0, 0]])
>>> k = np.array([[0,1,0], [0,1,0], [0,1,0]])
>>> ndimage.convolve(b, k, mode='reflect')
array([[5, 0, 0],
       [3, 0, 0],
       [1, 0, 0]])

This includes diagonally at the corners.

>>> k = np.array([[1,0,0],[0,1,0],[0,0,1]])
>>> ndimage.convolve(b, k)
array([[4, 2, 0],
       [3, 2, 0],
       [1, 1, 0]])

With ``mode='nearest'``, the single nearest value in to an edge in
`input` is repeated as many times as needed to match the overlapping
`weights`.

>>> c = np.array([[2, 0, 1],
...               [1, 0, 0],
...               [0, 0, 0]])
>>> k = np.array([[0, 1, 0],
...               [0, 1, 0],
...               [0, 1, 0],
...               [0, 1, 0],
...               [0, 1, 0]])
>>> ndimage.convolve(c, k, mode='nearest')
array([[7, 0, 3],
       [5, 0, 2],
       [3, 0, 1]])
*)

val convolve1d : ?axis:int -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:int -> input:[>`Ndarray] Np.Obj.t -> weights:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Calculate a 1-D convolution along the given axis.

The lines of the array along the given axis are convolved with the
given weights.

Parameters
----------
input : array_like
    The input array.
weights : ndarray
    1-D sequence of numbers.
axis : int, optional
    The axis of `input` along which to calculate. Default is -1.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'reflect'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : int, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right.

Returns
-------
convolve1d : ndarray
    Convolved array with same shape as input

Examples
--------
>>> from scipy.ndimage import convolve1d
>>> convolve1d([2, 8, 0, 4, 1, 9, 9, 0], weights=[1, 3])
array([14, 24,  4, 13, 12, 36, 27,  0])
*)

val correlate : ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`I of int | `Sequence of Py.Object.t] -> input:[>`Ndarray] Np.Obj.t -> weights:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional correlation.

The array is correlated with the given kernel.

Parameters
----------
input : array_like
    The input array.
weights : ndarray
    array of weights, same number of dimensions as input
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'reflect'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : int or sequence, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right. By passing a sequence of origins with length equal to
    the number of dimensions of the input array, different shifts can
    be specified along each axis.

Returns
-------
result : ndarray
    The result of correlation of `input` with `weights`.

See Also
--------
convolve : Convolve an image with a kernel.

Examples
--------
Correlation is the process of moving a filter mask often referred to
as kernel over the image and computing the sum of products at each location.

>>> from scipy.ndimage import correlate
>>> input_img = np.arange(25).reshape(5,5)
>>> print(input_img)
[[ 0  1  2  3  4]
[ 5  6  7  8  9]
[10 11 12 13 14]
[15 16 17 18 19]
[20 21 22 23 24]]

Define a kernel (weights) for correlation. In this example, it is for sum of
center and up, down, left and right next elements.

>>> weights = [[0, 1, 0],
...            [1, 1, 1],
...            [0, 1, 0]]

We can calculate a correlation result:
For example, element ``[2,2]`` is ``7 + 11 + 12 + 13 + 17 = 60``.

>>> correlate(input_img, weights)
array([[  6,  10,  15,  20,  24],
    [ 26,  30,  35,  40,  44],
    [ 51,  55,  60,  65,  69],
    [ 76,  80,  85,  90,  94],
    [ 96, 100, 105, 110, 114]])
*)

val correlate1d : ?axis:int -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:int -> input:[>`Ndarray] Np.Obj.t -> weights:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Calculate a 1-D correlation along the given axis.

The lines of the array along the given axis are correlated with the
given weights.

Parameters
----------
input : array_like
    The input array.
weights : array
    1-D sequence of numbers.
axis : int, optional
    The axis of `input` along which to calculate. Default is -1.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'reflect'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : int, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right.

Examples
--------
>>> from scipy.ndimage import correlate1d
>>> correlate1d([2, 8, 0, 4, 1, 9, 9, 0], weights=[1, 3])
array([ 8, 26,  8, 12,  7, 28, 36,  9])
*)

val distance_transform_bf : ?metric:string -> ?sampling:[`I of int | `Is of int list] -> ?return_distances:bool -> ?return_indices:bool -> ?distances:Py.Object.t -> ?indices:Py.Object.t -> input:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Distance transform function by a brute force algorithm.

This function calculates the distance transform of the `input`, by
replacing each foreground (non-zero) element, with its
shortest distance to the background (any zero-valued element).

In addition to the distance transform, the feature transform can
be calculated. In this case the index of the closest background
element is returned along the first axis of the result.

Parameters
----------
input : array_like
    Input
metric : str, optional
    Three types of distance metric are supported: 'euclidean', 'taxicab',
    and 'chessboard'.
sampling : {int, sequence of ints}, optional
    This parameter is only used in the case of the euclidean `metric`
    distance transform.

    The sampling along each axis can be given by the `sampling` parameter
    which should be a sequence of length equal to the input rank, or a
    single number in which the `sampling` is assumed to be equal along all
    axes.
return_distances : bool, optional
    The `return_distances` flag can be used to indicate if the distance
    transform is returned.

    The default is True.
return_indices : bool, optional
    The `return_indices` flags can be used to indicate if the feature
    transform is returned.

    The default is False.
distances : float64 ndarray, optional
    Optional output array to hold distances (if `return_distances` is
    True).
indices : int64 ndarray, optional
    Optional output array to hold indices (if `return_indices` is True).

Returns
-------
distances : ndarray
    Distance array if `return_distances` is True.
indices : ndarray
    Indices array if `return_indices` is True.

Notes
-----
This function employs a slow brute force algorithm, see also the
function distance_transform_cdt for more efficient taxicab and
chessboard algorithms.
*)

val distance_transform_cdt : ?metric:[`Chessboard | `Taxicab] -> ?return_distances:Py.Object.t -> ?return_indices:Py.Object.t -> ?distances:Py.Object.t -> ?indices:Py.Object.t -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Distance transform for chamfer type of transforms.

Parameters
----------
input : array_like
    Input
metric : {'chessboard', 'taxicab'}, optional
    The `metric` determines the type of chamfering that is done. If the
    `metric` is equal to 'taxicab' a structure is generated using
    generate_binary_structure with a squared distance equal to 1. If
    the `metric` is equal to 'chessboard', a `metric` is generated
    using generate_binary_structure with a squared distance equal to
    the dimensionality of the array. These choices correspond to the
    common interpretations of the 'taxicab' and the 'chessboard'
    distance metrics in two dimensions.

    The default for `metric` is 'chessboard'.
return_distances, return_indices : bool, optional
    The `return_distances`, and `return_indices` flags can be used to
    indicate if the distance transform, the feature transform, or both
    must be returned.

    If the feature transform is returned (``return_indices=True``),
    the index of the closest background element is returned along
    the first axis of the result.

    The `return_distances` default is True, and the
    `return_indices` default is False.
distances, indices : ndarrays of int32, optional
    The `distances` and `indices` arguments can be used to give optional
    output arrays that must be the same shape as `input`.
*)

val distance_transform_edt : ?sampling:[`F of float | `I of int | `Sequence_of_same of Py.Object.t] -> ?return_distances:bool -> ?return_indices:bool -> ?distances:[>`Ndarray] Np.Obj.t -> ?indices:[>`Ndarray] Np.Obj.t -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Exact Euclidean distance transform.

In addition to the distance transform, the feature transform can
be calculated. In this case the index of the closest background
element is returned along the first axis of the result.

Parameters
----------
input : array_like
    Input data to transform. Can be any type but will be converted
    into binary: 1 wherever input equates to True, 0 elsewhere.
sampling : float or int, or sequence of same, optional
    Spacing of elements along each dimension. If a sequence, must be of
    length equal to the input rank; if a single number, this is used for
    all axes. If not specified, a grid spacing of unity is implied.
return_distances : bool, optional
    Whether to return distance matrix. At least one of
    return_distances/return_indices must be True. Default is True.
return_indices : bool, optional
    Whether to return indices matrix. Default is False.
distances : ndarray, optional
    Used for output of distance array, must be of type float64.
indices : ndarray, optional
    Used for output of indices, must be of type int32.

Returns
-------
distance_transform_edt : ndarray or list of ndarrays
    Either distance matrix, index matrix, or a list of the two,
    depending on `return_x` flags and `distance` and `indices`
    input parameters.

Notes
-----
The Euclidean distance transform gives values of the Euclidean
distance::

                n
  y_i = sqrt(sum (x[i]-b[i])**2)
                i

where b[i] is the background point (value 0) with the smallest
Euclidean distance to input points x[i], and n is the
number of dimensions.

Examples
--------
>>> from scipy import ndimage
>>> a = np.array(([0,1,1,1,1],
...               [0,0,1,1,1],
...               [0,1,1,1,1],
...               [0,1,1,1,0],
...               [0,1,1,0,0]))
>>> ndimage.distance_transform_edt(a)
array([[ 0.    ,  1.    ,  1.4142,  2.2361,  3.    ],
       [ 0.    ,  0.    ,  1.    ,  2.    ,  2.    ],
       [ 0.    ,  1.    ,  1.4142,  1.4142,  1.    ],
       [ 0.    ,  1.    ,  1.4142,  1.    ,  0.    ],
       [ 0.    ,  1.    ,  1.    ,  0.    ,  0.    ]])

With a sampling of 2 units along x, 1 along y:

>>> ndimage.distance_transform_edt(a, sampling=[2,1])
array([[ 0.    ,  1.    ,  2.    ,  2.8284,  3.6056],
       [ 0.    ,  0.    ,  1.    ,  2.    ,  3.    ],
       [ 0.    ,  1.    ,  2.    ,  2.2361,  2.    ],
       [ 0.    ,  1.    ,  2.    ,  1.    ,  0.    ],
       [ 0.    ,  1.    ,  1.    ,  0.    ,  0.    ]])

Asking for indices as well:

>>> edt, inds = ndimage.distance_transform_edt(a, return_indices=True)
>>> inds
array([[[0, 0, 1, 1, 3],
        [1, 1, 1, 1, 3],
        [2, 2, 1, 3, 3],
        [3, 3, 4, 4, 3],
        [4, 4, 4, 4, 4]],
       [[0, 0, 1, 1, 4],
        [0, 1, 1, 1, 4],
        [0, 0, 1, 4, 4],
        [0, 0, 3, 3, 4],
        [0, 0, 3, 3, 4]]])

With arrays provided for inplace outputs:

>>> indices = np.zeros(((np.ndim(a),) + a.shape), dtype=np.int32)
>>> ndimage.distance_transform_edt(a, return_indices=True, indices=indices)
array([[ 0.    ,  1.    ,  1.4142,  2.2361,  3.    ],
       [ 0.    ,  0.    ,  1.    ,  2.    ,  2.    ],
       [ 0.    ,  1.    ,  1.4142,  1.4142,  1.    ],
       [ 0.    ,  1.    ,  1.4142,  1.    ,  0.    ],
       [ 0.    ,  1.    ,  1.    ,  0.    ,  0.    ]])
>>> indices
array([[[0, 0, 1, 1, 3],
        [1, 1, 1, 1, 3],
        [2, 2, 1, 3, 3],
        [3, 3, 4, 4, 3],
        [4, 4, 4, 4, 4]],
       [[0, 0, 1, 1, 4],
        [0, 1, 1, 1, 4],
        [0, 0, 1, 4, 4],
        [0, 0, 3, 3, 4],
        [0, 0, 3, 3, 4]]])
*)

val extrema : ?labels:[>`Ndarray] Np.Obj.t -> ?index:[`I of int | `Is of int list] -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Calculate the minimums and maximums of the values of an array
at labels, along with their positions.

Parameters
----------
input : ndarray
    N-D image data to process.
labels : ndarray, optional
    Labels of features in input.
    If not None, must be same shape as `input`.
index : int or sequence of ints, optional
    Labels to include in output.  If None (default), all values where
    non-zero `labels` are used.

Returns
-------
minimums, maximums : int or ndarray
    Values of minimums and maximums in each feature.
min_positions, max_positions : tuple or list of tuples
    Each tuple gives the N-D coordinates of the corresponding minimum
    or maximum.

See Also
--------
maximum, minimum, maximum_position, minimum_position, center_of_mass

Examples
--------
>>> a = np.array([[1, 2, 0, 0],
...               [5, 3, 0, 4],
...               [0, 0, 0, 7],
...               [9, 3, 0, 0]])
>>> from scipy import ndimage
>>> ndimage.extrema(a)
(0, 9, (0, 2), (3, 0))

Features to process can be specified using `labels` and `index`:

>>> lbl, nlbl = ndimage.label(a)
>>> ndimage.extrema(a, lbl, index=np.arange(1, nlbl+1))
(array([1, 4, 3]),
 array([5, 7, 9]),
 [(0, 0), (1, 3), (3, 1)],
 [(1, 0), (2, 3), (3, 0)])

If no index is given, non-zero `labels` are processed:

>>> ndimage.extrema(a, lbl)
(1, 9, (0, 0), (3, 0))
*)

val find_objects : ?max_label:int -> input:Py.Object.t -> unit -> Py.Object.t
(**
Find objects in a labeled array.

Parameters
----------
input : ndarray of ints
    Array containing objects defined by different labels. Labels with
    value 0 are ignored.
max_label : int, optional
    Maximum label to be searched for in `input`. If max_label is not
    given, the positions of all objects are returned.

Returns
-------
object_slices : list of tuples
    A list of tuples, with each tuple containing N slices (with N the
    dimension of the input array). Slices correspond to the minimal
    parallelepiped that contains the object. If a number is missing,
    None is returned instead of a slice.

See Also
--------
label, center_of_mass

Notes
-----
This function is very useful for isolating a volume of interest inside
a 3-D array, that cannot be 'seen through'.

Examples
--------
>>> from scipy import ndimage
>>> a = np.zeros((6,6), dtype=int)
>>> a[2:4, 2:4] = 1
>>> a[4, 4] = 1
>>> a[:2, :3] = 2
>>> a[0, 5] = 3
>>> a
array([[2, 2, 2, 0, 0, 3],
       [2, 2, 2, 0, 0, 0],
       [0, 0, 1, 1, 0, 0],
       [0, 0, 1, 1, 0, 0],
       [0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 0]])
>>> ndimage.find_objects(a)
[(slice(2, 5, None), slice(2, 5, None)), (slice(0, 2, None), slice(0, 3, None)), (slice(0, 1, None), slice(5, 6, None))]
>>> ndimage.find_objects(a, max_label=2)
[(slice(2, 5, None), slice(2, 5, None)), (slice(0, 2, None), slice(0, 3, None))]
>>> ndimage.find_objects(a == 1, max_label=2)
[(slice(2, 5, None), slice(2, 5, None)), None]

>>> loc = ndimage.find_objects(a)[0]
>>> a[loc]
array([[1, 1, 0],
       [1, 1, 0],
       [0, 0, 1]])
*)

val fourier_ellipsoid : ?n:int -> ?axis:int -> ?output:[>`Ndarray] Np.Obj.t -> input:[>`Ndarray] Np.Obj.t -> size:[`F of float | `Sequence of Py.Object.t] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional ellipsoid Fourier filter.

The array is multiplied with the fourier transform of a ellipsoid of
given sizes.

Parameters
----------
input : array_like
    The input array.
size : float or sequence
    The size of the box used for filtering.
    If a float, `size` is the same for all axes. If a sequence, `size` has
    to contain one value for each axis.
n : int, optional
    If `n` is negative (default), then the input is assumed to be the
    result of a complex fft.
    If `n` is larger than or equal to zero, the input is assumed to be the
    result of a real fft, and `n` gives the length of the array before
    transformation along the real transform direction.
axis : int, optional
    The axis of the real transform.
output : ndarray, optional
    If given, the result of filtering the input is placed in this array.
    None is returned in this case.

Returns
-------
fourier_ellipsoid : ndarray
    The filtered input.

Notes
-----
This function is implemented for arrays of rank 1, 2, or 3.

Examples
--------
>>> from scipy import ndimage, misc
>>> import numpy.fft
>>> import matplotlib.pyplot as plt
>>> fig, (ax1, ax2) = plt.subplots(1, 2)
>>> plt.gray()  # show the filtered result in grayscale
>>> ascent = misc.ascent()
>>> input_ = numpy.fft.fft2(ascent)
>>> result = ndimage.fourier_ellipsoid(input_, size=20)
>>> result = numpy.fft.ifft2(result)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result.real)  # the imaginary part is an artifact
>>> plt.show()
*)

val fourier_gaussian : ?n:int -> ?axis:int -> ?output:[>`Ndarray] Np.Obj.t -> input:[>`Ndarray] Np.Obj.t -> sigma:[`F of float | `Sequence of Py.Object.t] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional Gaussian fourier filter.

The array is multiplied with the fourier transform of a Gaussian
kernel.

Parameters
----------
input : array_like
    The input array.
sigma : float or sequence
    The sigma of the Gaussian kernel. If a float, `sigma` is the same for
    all axes. If a sequence, `sigma` has to contain one value for each
    axis.
n : int, optional
    If `n` is negative (default), then the input is assumed to be the
    result of a complex fft.
    If `n` is larger than or equal to zero, the input is assumed to be the
    result of a real fft, and `n` gives the length of the array before
    transformation along the real transform direction.
axis : int, optional
    The axis of the real transform.
output : ndarray, optional
    If given, the result of filtering the input is placed in this array.
    None is returned in this case.

Returns
-------
fourier_gaussian : ndarray
    The filtered input.

Examples
--------
>>> from scipy import ndimage, misc
>>> import numpy.fft
>>> import matplotlib.pyplot as plt
>>> fig, (ax1, ax2) = plt.subplots(1, 2)
>>> plt.gray()  # show the filtered result in grayscale
>>> ascent = misc.ascent()
>>> input_ = numpy.fft.fft2(ascent)
>>> result = ndimage.fourier_gaussian(input_, sigma=4)
>>> result = numpy.fft.ifft2(result)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result.real)  # the imaginary part is an artifact
>>> plt.show()
*)

val fourier_shift : ?n:int -> ?axis:int -> ?output:[>`Ndarray] Np.Obj.t -> input:[>`Ndarray] Np.Obj.t -> shift:[`F of float | `Sequence of Py.Object.t] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional Fourier shift filter.

The array is multiplied with the Fourier transform of a shift operation.

Parameters
----------
input : array_like
    The input array.
shift : float or sequence
    The size of the box used for filtering.
    If a float, `shift` is the same for all axes. If a sequence, `shift`
    has to contain one value for each axis.
n : int, optional
    If `n` is negative (default), then the input is assumed to be the
    result of a complex fft.
    If `n` is larger than or equal to zero, the input is assumed to be the
    result of a real fft, and `n` gives the length of the array before
    transformation along the real transform direction.
axis : int, optional
    The axis of the real transform.
output : ndarray, optional
    If given, the result of shifting the input is placed in this array.
    None is returned in this case.

Returns
-------
fourier_shift : ndarray
    The shifted input.

Examples
--------
>>> from scipy import ndimage, misc
>>> import matplotlib.pyplot as plt
>>> import numpy.fft
>>> fig, (ax1, ax2) = plt.subplots(1, 2)
>>> plt.gray()  # show the filtered result in grayscale
>>> ascent = misc.ascent()
>>> input_ = numpy.fft.fft2(ascent)
>>> result = ndimage.fourier_shift(input_, shift=200)
>>> result = numpy.fft.ifft2(result)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result.real)  # the imaginary part is an artifact
>>> plt.show()
*)

val fourier_uniform : ?n:int -> ?axis:int -> ?output:[>`Ndarray] Np.Obj.t -> input:[>`Ndarray] Np.Obj.t -> size:[`F of float | `Sequence of Py.Object.t] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional uniform fourier filter.

The array is multiplied with the Fourier transform of a box of given
size.

Parameters
----------
input : array_like
    The input array.
size : float or sequence
    The size of the box used for filtering.
    If a float, `size` is the same for all axes. If a sequence, `size` has
    to contain one value for each axis.
n : int, optional
    If `n` is negative (default), then the input is assumed to be the
    result of a complex fft.
    If `n` is larger than or equal to zero, the input is assumed to be the
    result of a real fft, and `n` gives the length of the array before
    transformation along the real transform direction.
axis : int, optional
    The axis of the real transform.
output : ndarray, optional
    If given, the result of filtering the input is placed in this array.
    None is returned in this case.

Returns
-------
fourier_uniform : ndarray
    The filtered input.

Examples
--------
>>> from scipy import ndimage, misc
>>> import numpy.fft
>>> import matplotlib.pyplot as plt
>>> fig, (ax1, ax2) = plt.subplots(1, 2)
>>> plt.gray()  # show the filtered result in grayscale
>>> ascent = misc.ascent()
>>> input_ = numpy.fft.fft2(ascent)
>>> result = ndimage.fourier_uniform(input_, size=20)
>>> result = numpy.fft.ifft2(result)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result.real)  # the imaginary part is an artifact
>>> plt.show()
*)

val gaussian_filter : ?order:[`I of int | `Is of int list] -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Sequence of Py.Object.t | `S of string] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?truncate:float -> input:[>`Ndarray] Np.Obj.t -> sigma:[`Bool of bool | `F of float | `S of string | `Sequence_of_scalars of Py.Object.t | `I of int] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional Gaussian filter.

Parameters
----------
input : array_like
    The input array.
sigma : scalar or sequence of scalars
    Standard deviation for Gaussian kernel. The standard
    deviations of the Gaussian filter are given for each axis as a
    sequence, or as a single number, in which case it is equal for
    all axes.
order : int or sequence of ints, optional
    The order of the filter along each axis is given as a sequence
    of integers, or as a single number. An order of 0 corresponds
    to convolution with a Gaussian kernel. A positive order
    corresponds to convolution with that derivative of a Gaussian.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : str or sequence, optional
    The `mode` parameter determines how the input array is extended
    when the filter overlaps a border. By passing a sequence of modes
    with length equal to the number of dimensions of the input array,
    different modes can be specified along each axis. Default value is
    'reflect'. The valid values and their behavior is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
truncate : float
    Truncate the filter at this many standard deviations.
    Default is 4.0.

Returns
-------
gaussian_filter : ndarray
    Returned array of same shape as `input`.

Notes
-----
The multidimensional filter is implemented as a sequence of
1-D convolution filters. The intermediate arrays are
stored in the same data type as the output. Therefore, for output
types with a limited precision, the results may be imprecise
because intermediate results may be stored with insufficient
precision.

Examples
--------
>>> from scipy.ndimage import gaussian_filter
>>> a = np.arange(50, step=2).reshape((5,5))
>>> a
array([[ 0,  2,  4,  6,  8],
       [10, 12, 14, 16, 18],
       [20, 22, 24, 26, 28],
       [30, 32, 34, 36, 38],
       [40, 42, 44, 46, 48]])
>>> gaussian_filter(a, sigma=1)
array([[ 4,  6,  8,  9, 11],
       [10, 12, 14, 15, 17],
       [20, 22, 24, 25, 27],
       [29, 31, 33, 34, 36],
       [35, 37, 39, 40, 42]])

>>> from scipy import misc
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = misc.ascent()
>>> result = gaussian_filter(ascent, sigma=5)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()
*)

val gaussian_filter1d : ?axis:int -> ?order:int -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?truncate:float -> input:[>`Ndarray] Np.Obj.t -> sigma:[`F of float | `I of int | `Bool of bool | `S of string] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
1-D Gaussian filter.

Parameters
----------
input : array_like
    The input array.
sigma : scalar
    standard deviation for Gaussian kernel
axis : int, optional
    The axis of `input` along which to calculate. Default is -1.
order : int, optional
    An order of 0 corresponds to convolution with a Gaussian
    kernel. A positive order corresponds to convolution with
    that derivative of a Gaussian.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'reflect'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
truncate : float, optional
    Truncate the filter at this many standard deviations.
    Default is 4.0.

Returns
-------
gaussian_filter1d : ndarray

Examples
--------
>>> from scipy.ndimage import gaussian_filter1d
>>> gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 1)
array([ 1.42704095,  2.06782203,  3.        ,  3.93217797,  4.57295905])
>>> gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 4)
array([ 2.91948343,  2.95023502,  3.        ,  3.04976498,  3.08051657])
>>> import matplotlib.pyplot as plt
>>> np.random.seed(280490)
>>> x = np.random.randn(101).cumsum()
>>> y3 = gaussian_filter1d(x, 3)
>>> y6 = gaussian_filter1d(x, 6)
>>> plt.plot(x, 'k', label='original data')
>>> plt.plot(y3, '--', label='filtered, sigma=3')
>>> plt.plot(y6, ':', label='filtered, sigma=6')
>>> plt.legend()
>>> plt.grid()
>>> plt.show()
*)

val gaussian_gradient_magnitude : ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Sequence of Py.Object.t | `S of string] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?kwargs:(string * Py.Object.t) list -> input:[>`Ndarray] Np.Obj.t -> sigma:[`Bool of bool | `F of float | `S of string | `Sequence_of_scalars of Py.Object.t | `I of int] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional gradient magnitude using Gaussian derivatives.

Parameters
----------
input : array_like
    The input array.
sigma : scalar or sequence of scalars
    The standard deviations of the Gaussian filter are given for
    each axis as a sequence, or as a single number, in which case
    it is equal for all axes.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : str or sequence, optional
    The `mode` parameter determines how the input array is extended
    when the filter overlaps a border. By passing a sequence of modes
    with length equal to the number of dimensions of the input array,
    different modes can be specified along each axis. Default value is
    'reflect'. The valid values and their behavior is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
Extra keyword arguments will be passed to gaussian_filter().

Returns
-------
gaussian_gradient_magnitude : ndarray
    Filtered array. Has the same shape as `input`.

Examples
--------
>>> from scipy import ndimage, misc
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = misc.ascent()
>>> result = ndimage.gaussian_gradient_magnitude(ascent, sigma=5)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()
*)

val gaussian_laplace : ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Sequence of Py.Object.t | `S of string] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?kwargs:(string * Py.Object.t) list -> input:[>`Ndarray] Np.Obj.t -> sigma:[`Bool of bool | `F of float | `S of string | `Sequence_of_scalars of Py.Object.t | `I of int] -> unit -> Py.Object.t
(**
Multidimensional Laplace filter using Gaussian second derivatives.

Parameters
----------
input : array_like
    The input array.
sigma : scalar or sequence of scalars
    The standard deviations of the Gaussian filter are given for
    each axis as a sequence, or as a single number, in which case
    it is equal for all axes.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : str or sequence, optional
    The `mode` parameter determines how the input array is extended
    when the filter overlaps a border. By passing a sequence of modes
    with length equal to the number of dimensions of the input array,
    different modes can be specified along each axis. Default value is
    'reflect'. The valid values and their behavior is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
Extra keyword arguments will be passed to gaussian_filter().

Examples
--------
>>> from scipy import ndimage, misc
>>> import matplotlib.pyplot as plt
>>> ascent = misc.ascent()

>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side

>>> result = ndimage.gaussian_laplace(ascent, sigma=1)
>>> ax1.imshow(result)

>>> result = ndimage.gaussian_laplace(ascent, sigma=3)
>>> ax2.imshow(result)
>>> plt.show()
*)

val generate_binary_structure : rank:int -> connectivity:int -> unit -> Py.Object.t
(**
Generate a binary structure for binary morphological operations.

Parameters
----------
rank : int
     Number of dimensions of the array to which the structuring element
     will be applied, as returned by `np.ndim`.
connectivity : int
     `connectivity` determines which elements of the output array belong
     to the structure, i.e., are considered as neighbors of the central
     element. Elements up to a squared distance of `connectivity` from
     the center are considered neighbors. `connectivity` may range from 1
     (no diagonal elements are neighbors) to `rank` (all elements are
     neighbors).

Returns
-------
output : ndarray of bools
     Structuring element which may be used for binary morphological
     operations, with `rank` dimensions and all dimensions equal to 3.

See also
--------
iterate_structure, binary_dilation, binary_erosion

Notes
-----
`generate_binary_structure` can only create structuring elements with
dimensions equal to 3, i.e., minimal dimensions. For larger structuring
elements, that are useful e.g., for eroding large objects, one may either
use `iterate_structure`, or create directly custom arrays with
numpy functions such as `numpy.ones`.

Examples
--------
>>> from scipy import ndimage
>>> struct = ndimage.generate_binary_structure(2, 1)
>>> struct
array([[False,  True, False],
       [ True,  True,  True],
       [False,  True, False]], dtype=bool)
>>> a = np.zeros((5,5))
>>> a[2, 2] = 1
>>> a
array([[ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.]])
>>> b = ndimage.binary_dilation(a, structure=struct).astype(a.dtype)
>>> b
array([[ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.],
       [ 0.,  1.,  1.,  1.,  0.],
       [ 0.,  0.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.]])
>>> ndimage.binary_dilation(b, structure=struct).astype(a.dtype)
array([[ 0.,  0.,  1.,  0.,  0.],
       [ 0.,  1.,  1.,  1.,  0.],
       [ 1.,  1.,  1.,  1.,  1.],
       [ 0.,  1.,  1.,  1.,  0.],
       [ 0.,  0.,  1.,  0.,  0.]])
>>> struct = ndimage.generate_binary_structure(2, 2)
>>> struct
array([[ True,  True,  True],
       [ True,  True,  True],
       [ True,  True,  True]], dtype=bool)
>>> struct = ndimage.generate_binary_structure(3, 1)
>>> struct # no diagonal elements
array([[[False, False, False],
        [False,  True, False],
        [False, False, False]],
       [[False,  True, False],
        [ True,  True,  True],
        [False,  True, False]],
       [[False, False, False],
        [False,  True, False],
        [False, False, False]]], dtype=bool)
*)

val generic_filter : ?size:[`Bool of bool | `F of float | `S of string | `Tuple of Py.Object.t | `I of int] -> ?footprint:[>`Ndarray] Np.Obj.t -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`I of int | `Sequence of Py.Object.t] -> ?extra_arguments:Py.Object.t -> ?extra_keywords:Py.Object.t -> input:[>`Ndarray] Np.Obj.t -> function_:[`Scipy_LowLevelCallable of Py.Object.t | `Callable of Py.Object.t] -> unit -> Py.Object.t
(**
Calculate a multidimensional filter using the given function.

At each element the provided function is called. The input values
within the filter footprint at that element are passed to the function
as a 1-D array of double values.

Parameters
----------
input : array_like
    The input array.
function : {callable, scipy.LowLevelCallable}
    Function to apply at each element.
size : scalar or tuple, optional
    See footprint, below. Ignored if footprint is given.
footprint : array, optional
    Either `size` or `footprint` must be defined. `size` gives
    the shape that is taken from the input array, at every element
    position, to define the input to the filter function.
    `footprint` is a boolean array that specifies (implicitly) a
    shape, but also which of the elements within this shape will get
    passed to the filter function. Thus ``size=(n,m)`` is equivalent
    to ``footprint=np.ones((n,m))``.  We adjust `size` to the number
    of dimensions of the input array, so that, if the input array is
    shape (10,10,10), and `size` is 2, then the actual size used is
    (2,2,2). When `footprint` is given, `size` is ignored.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'reflect'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : int or sequence, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right. By passing a sequence of origins with length equal to
    the number of dimensions of the input array, different shifts can
    be specified along each axis.
extra_arguments : sequence, optional
    Sequence of extra positional arguments to pass to passed function.
extra_keywords : dict, optional
    dict of extra keyword arguments to pass to passed function.

Notes
-----
This function also accepts low-level callback functions with one of
the following signatures and wrapped in `scipy.LowLevelCallable`:

.. code:: c

   int callback(double *buffer, npy_intp filter_size,
                double *return_value, void *user_data)
   int callback(double *buffer, intptr_t filter_size,
                double *return_value, void *user_data)

The calling function iterates over the elements of the input and
output arrays, calling the callback function at each element. The
elements within the footprint of the filter at the current element are
passed through the ``buffer`` parameter, and the number of elements
within the footprint through ``filter_size``. The calculated value is
returned in ``return_value``. ``user_data`` is the data pointer provided
to `scipy.LowLevelCallable` as-is.

The callback function must return an integer error status that is zero
if something went wrong and one otherwise. If an error occurs, you should
normally set the python error status with an informative message
before returning, otherwise a default error message is set by the
calling function.

In addition, some other low-level function pointer specifications
are accepted, but these are for backward compatibility only and should
not be used in new code.
*)

val generic_filter1d : ?axis:int -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:int -> ?extra_arguments:Py.Object.t -> ?extra_keywords:Py.Object.t -> input:[>`Ndarray] Np.Obj.t -> function_:[`Scipy_LowLevelCallable of Py.Object.t | `Callable of Py.Object.t] -> filter_size:[`F of float | `I of int | `Bool of bool | `S of string] -> unit -> Py.Object.t
(**
Calculate a 1-D filter along the given axis.

`generic_filter1d` iterates over the lines of the array, calling the
given function at each line. The arguments of the line are the
input line, and the output line. The input and output lines are 1-D
double arrays. The input line is extended appropriately according
to the filter size and origin. The output line must be modified
in-place with the result.

Parameters
----------
input : array_like
    The input array.
function : {callable, scipy.LowLevelCallable}
    Function to apply along given axis.
filter_size : scalar
    Length of the filter.
axis : int, optional
    The axis of `input` along which to calculate. Default is -1.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'reflect'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : int, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right.
extra_arguments : sequence, optional
    Sequence of extra positional arguments to pass to passed function.
extra_keywords : dict, optional
    dict of extra keyword arguments to pass to passed function.

Notes
-----
This function also accepts low-level callback functions with one of
the following signatures and wrapped in `scipy.LowLevelCallable`:

.. code:: c

   int function(double *input_line, npy_intp input_length,
                double *output_line, npy_intp output_length,
                void *user_data)
   int function(double *input_line, intptr_t input_length,
                double *output_line, intptr_t output_length,
                void *user_data)

The calling function iterates over the lines of the input and output
arrays, calling the callback function at each line. The current line
is extended according to the border conditions set by the calling
function, and the result is copied into the array that is passed
through ``input_line``. The length of the input line (after extension)
is passed through ``input_length``. The callback function should apply
the filter and store the result in the array passed through
``output_line``. The length of the output line is passed through
``output_length``. ``user_data`` is the data pointer provided
to `scipy.LowLevelCallable` as-is.

The callback function must return an integer error status that is zero
if something went wrong and one otherwise. If an error occurs, you should
normally set the python error status with an informative message
before returning, otherwise a default error message is set by the
calling function.

In addition, some other low-level function pointer specifications
are accepted, but these are for backward compatibility only and should
not be used in new code.
*)

val generic_gradient_magnitude : ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Sequence of Py.Object.t | `S of string] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?extra_arguments:Py.Object.t -> ?extra_keywords:Py.Object.t -> input:[>`Ndarray] Np.Obj.t -> derivative:Py.Object.t -> unit -> Py.Object.t
(**
Gradient magnitude using a provided gradient function.

Parameters
----------
input : array_like
    The input array.
derivative : callable
    Callable with the following signature::

        derivative(input, axis, output, mode, cval,
                   *extra_arguments, **extra_keywords)

    See `extra_arguments`, `extra_keywords` below.
    `derivative` can assume that `input` and `output` are ndarrays.
    Note that the output from `derivative` is modified inplace;
    be careful to copy important inputs before returning them.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : str or sequence, optional
    The `mode` parameter determines how the input array is extended
    when the filter overlaps a border. By passing a sequence of modes
    with length equal to the number of dimensions of the input array,
    different modes can be specified along each axis. Default value is
    'reflect'. The valid values and their behavior is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
extra_keywords : dict, optional
    dict of extra keyword arguments to pass to passed function.
extra_arguments : sequence, optional
    Sequence of extra positional arguments to pass to passed function.
*)

val generic_laplace : ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Sequence of Py.Object.t | `S of string] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?extra_arguments:Py.Object.t -> ?extra_keywords:Py.Object.t -> input:[>`Ndarray] Np.Obj.t -> derivative2:Py.Object.t -> unit -> Py.Object.t
(**
N-D Laplace filter using a provided second derivative function.

Parameters
----------
input : array_like
    The input array.
derivative2 : callable
    Callable with the following signature::

        derivative2(input, axis, output, mode, cval,
                    *extra_arguments, **extra_keywords)

    See `extra_arguments`, `extra_keywords` below.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : str or sequence, optional
    The `mode` parameter determines how the input array is extended
    when the filter overlaps a border. By passing a sequence of modes
    with length equal to the number of dimensions of the input array,
    different modes can be specified along each axis. Default value is
    'reflect'. The valid values and their behavior is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
extra_keywords : dict, optional
    dict of extra keyword arguments to pass to passed function.
extra_arguments : sequence, optional
    Sequence of extra positional arguments to pass to passed function.
*)

val geometric_transform : ?output_shape:int list -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?order:int -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?prefilter:bool -> ?extra_arguments:Py.Object.t -> ?extra_keywords:Py.Object.t -> input:[>`Ndarray] Np.Obj.t -> mapping:[`Scipy_LowLevelCallable of Py.Object.t | `Callable of Py.Object.t] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Apply an arbitrary geometric transform.

The given mapping function is used to find, for each point in the
output, the corresponding coordinates in the input. The value of the
input at those coordinates is determined by spline interpolation of
the requested order.

Parameters
----------
input : array_like
    The input array.
mapping : {callable, scipy.LowLevelCallable}
    A callable object that accepts a tuple of length equal to the output
    array rank, and returns the corresponding input coordinates as a tuple
    of length equal to the input array rank.
output_shape : tuple of ints, optional
    Shape tuple.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
order : int, optional
    The order of the spline interpolation, default is 3.
    The order has to be in the range 0-5.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'constant'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
prefilter : bool, optional
    Determines if the input array is prefiltered with `spline_filter`
    before interpolation. The default is True, which will create a
    temporary `float64` array of filtered values if `order > 1`. If
    setting this to False, the output will be slightly blurred if
    `order > 1`, unless the input is prefiltered, i.e. it is the result
    of calling `spline_filter` on the original input.
extra_arguments : tuple, optional
    Extra arguments passed to `mapping`.
extra_keywords : dict, optional
    Extra keywords passed to `mapping`.

Returns
-------
output : ndarray
    The filtered input.

See Also
--------
map_coordinates, affine_transform, spline_filter1d


Notes
-----
This function also accepts low-level callback functions with one
the following signatures and wrapped in `scipy.LowLevelCallable`:

.. code:: c

   int mapping(npy_intp *output_coordinates, double *input_coordinates,
               int output_rank, int input_rank, void *user_data)
   int mapping(intptr_t *output_coordinates, double *input_coordinates,
               int output_rank, int input_rank, void *user_data)

The calling function iterates over the elements of the output array,
calling the callback function at each element. The coordinates of the
current output element are passed through ``output_coordinates``. The
callback function must return the coordinates at which the input must
be interpolated in ``input_coordinates``. The rank of the input and
output arrays are given by ``input_rank`` and ``output_rank``
respectively. ``user_data`` is the data pointer provided
to `scipy.LowLevelCallable` as-is.

The callback function must return an integer error status that is zero
if something went wrong and one otherwise. If an error occurs, you should
normally set the Python error status with an informative message
before returning, otherwise a default error message is set by the
calling function.

In addition, some other low-level function pointer specifications
are accepted, but these are for backward compatibility only and should
not be used in new code.

Examples
--------
>>> import numpy as np
>>> from scipy.ndimage import geometric_transform
>>> a = np.arange(12.).reshape((4, 3))
>>> def shift_func(output_coords):
...     return (output_coords[0] - 0.5, output_coords[1] - 0.5)
...
>>> geometric_transform(a, shift_func)
array([[ 0.   ,  0.   ,  0.   ],
       [ 0.   ,  1.362,  2.738],
       [ 0.   ,  4.812,  6.187],
       [ 0.   ,  8.263,  9.637]])

>>> b = [1, 2, 3, 4, 5]
>>> def shift_func(output_coords):
...     return (output_coords[0] - 3,)
...
>>> geometric_transform(b, shift_func, mode='constant')
array([0, 0, 0, 1, 2])
>>> geometric_transform(b, shift_func, mode='nearest')
array([1, 1, 1, 1, 2])
>>> geometric_transform(b, shift_func, mode='reflect')
array([3, 2, 1, 1, 2])
>>> geometric_transform(b, shift_func, mode='wrap')
array([2, 3, 4, 1, 2])
*)

val grey_closing : ?size:int list -> ?footprint:Py.Object.t -> ?structure:Py.Object.t -> ?output:[>`Ndarray] Np.Obj.t -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`F of float | `I of int | `Bool of bool | `S of string] -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional grayscale closing.

A grayscale closing consists in the succession of a grayscale dilation,
and a grayscale erosion.

Parameters
----------
input : array_like
    Array over which the grayscale closing is to be computed.
size : tuple of ints
    Shape of a flat and full structuring element used for the grayscale
    closing. Optional if `footprint` or `structure` is provided.
footprint : array of ints, optional
    Positions of non-infinite elements of a flat structuring element
    used for the grayscale closing.
structure : array of ints, optional
    Structuring element used for the grayscale closing. `structure`
    may be a non-flat structuring element.
output : array, optional
    An array used for storing the output of the closing may be provided.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the array borders are
    handled, where `cval` is the value when mode is equal to
    'constant'. Default is 'reflect'
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : scalar, optional
    The `origin` parameter controls the placement of the filter.
    Default 0

Returns
-------
grey_closing : ndarray
    Result of the grayscale closing of `input` with `structure`.

See also
--------
binary_closing, grey_dilation, grey_erosion, grey_opening,
generate_binary_structure

Notes
-----
The action of a grayscale closing with a flat structuring element amounts
to smoothen deep local minima, whereas binary closing fills small holes.

References
----------
.. [1] https://en.wikipedia.org/wiki/Mathematical_morphology

Examples
--------
>>> from scipy import ndimage
>>> a = np.arange(36).reshape((6,6))
>>> a[3,3] = 0
>>> a
array([[ 0,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11],
       [12, 13, 14, 15, 16, 17],
       [18, 19, 20,  0, 22, 23],
       [24, 25, 26, 27, 28, 29],
       [30, 31, 32, 33, 34, 35]])
>>> ndimage.grey_closing(a, size=(3,3))
array([[ 7,  7,  8,  9, 10, 11],
       [ 7,  7,  8,  9, 10, 11],
       [13, 13, 14, 15, 16, 17],
       [19, 19, 20, 20, 22, 23],
       [25, 25, 26, 27, 28, 29],
       [31, 31, 32, 33, 34, 35]])
>>> # Note that the local minimum a[3,3] has disappeared
*)

val grey_dilation : ?size:int list -> ?footprint:Py.Object.t -> ?structure:Py.Object.t -> ?output:[>`Ndarray] Np.Obj.t -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`F of float | `I of int | `Bool of bool | `S of string] -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Calculate a greyscale dilation, using either a structuring element,
or a footprint corresponding to a flat structuring element.

Grayscale dilation is a mathematical morphology operation. For the
simple case of a full and flat structuring element, it can be viewed
as a maximum filter over a sliding window.

Parameters
----------
input : array_like
    Array over which the grayscale dilation is to be computed.
size : tuple of ints
    Shape of a flat and full structuring element used for the grayscale
    dilation. Optional if `footprint` or `structure` is provided.
footprint : array of ints, optional
    Positions of non-infinite elements of a flat structuring element
    used for the grayscale dilation. Non-zero values give the set of
    neighbors of the center over which the maximum is chosen.
structure : array of ints, optional
    Structuring element used for the grayscale dilation. `structure`
    may be a non-flat structuring element.
output : array, optional
    An array used for storing the output of the dilation may be provided.
mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional
    The `mode` parameter determines how the array borders are
    handled, where `cval` is the value when mode is equal to
    'constant'. Default is 'reflect'
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : scalar, optional
    The `origin` parameter controls the placement of the filter.
    Default 0

Returns
-------
grey_dilation : ndarray
    Grayscale dilation of `input`.

See also
--------
binary_dilation, grey_erosion, grey_closing, grey_opening
generate_binary_structure, maximum_filter

Notes
-----
The grayscale dilation of an image input by a structuring element s defined
over a domain E is given by:

(input+s)(x) = max {input(y) + s(x-y), for y in E}

In particular, for structuring elements defined as
s(y) = 0 for y in E, the grayscale dilation computes the maximum of the
input image inside a sliding window defined by E.

Grayscale dilation [1]_ is a *mathematical morphology* operation [2]_.

References
----------
.. [1] https://en.wikipedia.org/wiki/Dilation_%28morphology%29
.. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

Examples
--------
>>> from scipy import ndimage
>>> a = np.zeros((7,7), dtype=int)
>>> a[2:5, 2:5] = 1
>>> a[4,4] = 2; a[2,3] = 3
>>> a
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 3, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 2, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0]])
>>> ndimage.grey_dilation(a, size=(3,3))
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 1, 3, 3, 3, 1, 0],
       [0, 1, 3, 3, 3, 1, 0],
       [0, 1, 3, 3, 3, 2, 0],
       [0, 1, 1, 2, 2, 2, 0],
       [0, 1, 1, 2, 2, 2, 0],
       [0, 0, 0, 0, 0, 0, 0]])
>>> ndimage.grey_dilation(a, footprint=np.ones((3,3)))
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 1, 3, 3, 3, 1, 0],
       [0, 1, 3, 3, 3, 1, 0],
       [0, 1, 3, 3, 3, 2, 0],
       [0, 1, 1, 2, 2, 2, 0],
       [0, 1, 1, 2, 2, 2, 0],
       [0, 0, 0, 0, 0, 0, 0]])
>>> s = ndimage.generate_binary_structure(2,1)
>>> s
array([[False,  True, False],
       [ True,  True,  True],
       [False,  True, False]], dtype=bool)
>>> ndimage.grey_dilation(a, footprint=s)
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 3, 1, 0, 0],
       [0, 1, 3, 3, 3, 1, 0],
       [0, 1, 1, 3, 2, 1, 0],
       [0, 1, 1, 2, 2, 2, 0],
       [0, 0, 1, 1, 2, 0, 0],
       [0, 0, 0, 0, 0, 0, 0]])
>>> ndimage.grey_dilation(a, size=(3,3), structure=np.ones((3,3)))
array([[1, 1, 1, 1, 1, 1, 1],
       [1, 2, 4, 4, 4, 2, 1],
       [1, 2, 4, 4, 4, 2, 1],
       [1, 2, 4, 4, 4, 3, 1],
       [1, 2, 2, 3, 3, 3, 1],
       [1, 2, 2, 3, 3, 3, 1],
       [1, 1, 1, 1, 1, 1, 1]])
*)

val grey_erosion : ?size:int list -> ?footprint:Py.Object.t -> ?structure:Py.Object.t -> ?output:[>`Ndarray] Np.Obj.t -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`F of float | `I of int | `Bool of bool | `S of string] -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Calculate a greyscale erosion, using either a structuring element,
or a footprint corresponding to a flat structuring element.

Grayscale erosion is a mathematical morphology operation. For the
simple case of a full and flat structuring element, it can be viewed
as a minimum filter over a sliding window.

Parameters
----------
input : array_like
    Array over which the grayscale erosion is to be computed.
size : tuple of ints
    Shape of a flat and full structuring element used for the grayscale
    erosion. Optional if `footprint` or `structure` is provided.
footprint : array of ints, optional
    Positions of non-infinite elements of a flat structuring element
    used for the grayscale erosion. Non-zero values give the set of
    neighbors of the center over which the minimum is chosen.
structure : array of ints, optional
    Structuring element used for the grayscale erosion. `structure`
    may be a non-flat structuring element.
output : array, optional
    An array used for storing the output of the erosion may be provided.
mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional
    The `mode` parameter determines how the array borders are
    handled, where `cval` is the value when mode is equal to
    'constant'. Default is 'reflect'
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : scalar, optional
    The `origin` parameter controls the placement of the filter.
    Default 0

Returns
-------
output : ndarray
    Grayscale erosion of `input`.

See also
--------
binary_erosion, grey_dilation, grey_opening, grey_closing
generate_binary_structure, minimum_filter

Notes
-----
The grayscale erosion of an image input by a structuring element s defined
over a domain E is given by:

(input+s)(x) = min {input(y) - s(x-y), for y in E}

In particular, for structuring elements defined as
s(y) = 0 for y in E, the grayscale erosion computes the minimum of the
input image inside a sliding window defined by E.

Grayscale erosion [1]_ is a *mathematical morphology* operation [2]_.

References
----------
.. [1] https://en.wikipedia.org/wiki/Erosion_%28morphology%29
.. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

Examples
--------
>>> from scipy import ndimage
>>> a = np.zeros((7,7), dtype=int)
>>> a[1:6, 1:6] = 3
>>> a[4,4] = 2; a[2,3] = 1
>>> a
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 3, 3, 3, 3, 3, 0],
       [0, 3, 3, 1, 3, 3, 0],
       [0, 3, 3, 3, 3, 3, 0],
       [0, 3, 3, 3, 2, 3, 0],
       [0, 3, 3, 3, 3, 3, 0],
       [0, 0, 0, 0, 0, 0, 0]])
>>> ndimage.grey_erosion(a, size=(3,3))
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 3, 2, 2, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0]])
>>> footprint = ndimage.generate_binary_structure(2, 1)
>>> footprint
array([[False,  True, False],
       [ True,  True,  True],
       [False,  True, False]], dtype=bool)
>>> # Diagonally-connected elements are not considered neighbors
>>> ndimage.grey_erosion(a, size=(3,3), footprint=footprint)
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 3, 1, 2, 0, 0],
       [0, 0, 3, 2, 2, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0]])
*)

val grey_opening : ?size:int list -> ?footprint:Py.Object.t -> ?structure:Py.Object.t -> ?output:[>`Ndarray] Np.Obj.t -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`F of float | `I of int | `Bool of bool | `S of string] -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional grayscale opening.

A grayscale opening consists in the succession of a grayscale erosion,
and a grayscale dilation.

Parameters
----------
input : array_like
    Array over which the grayscale opening is to be computed.
size : tuple of ints
    Shape of a flat and full structuring element used for the grayscale
    opening. Optional if `footprint` or `structure` is provided.
footprint : array of ints, optional
    Positions of non-infinite elements of a flat structuring element
    used for the grayscale opening.
structure : array of ints, optional
    Structuring element used for the grayscale opening. `structure`
    may be a non-flat structuring element.
output : array, optional
    An array used for storing the output of the opening may be provided.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the array borders are
    handled, where `cval` is the value when mode is equal to
    'constant'. Default is 'reflect'
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : scalar, optional
    The `origin` parameter controls the placement of the filter.
    Default 0

Returns
-------
grey_opening : ndarray
    Result of the grayscale opening of `input` with `structure`.

See also
--------
binary_opening, grey_dilation, grey_erosion, grey_closing
generate_binary_structure

Notes
-----
The action of a grayscale opening with a flat structuring element amounts
to smoothen high local maxima, whereas binary opening erases small objects.

References
----------
.. [1] https://en.wikipedia.org/wiki/Mathematical_morphology

Examples
--------
>>> from scipy import ndimage
>>> a = np.arange(36).reshape((6,6))
>>> a[3, 3] = 50
>>> a
array([[ 0,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11],
       [12, 13, 14, 15, 16, 17],
       [18, 19, 20, 50, 22, 23],
       [24, 25, 26, 27, 28, 29],
       [30, 31, 32, 33, 34, 35]])
>>> ndimage.grey_opening(a, size=(3,3))
array([[ 0,  1,  2,  3,  4,  4],
       [ 6,  7,  8,  9, 10, 10],
       [12, 13, 14, 15, 16, 16],
       [18, 19, 20, 22, 22, 22],
       [24, 25, 26, 27, 28, 28],
       [24, 25, 26, 27, 28, 28]])
>>> # Note that the local maximum a[3,3] has disappeared
*)

val histogram : ?labels:[>`Ndarray] Np.Obj.t -> ?index:[`I of int | `Is of int list] -> input:[>`Ndarray] Np.Obj.t -> min:Py.Object.t -> max:Py.Object.t -> bins:int -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Calculate the histogram of the values of an array, optionally at labels.

Histogram calculates the frequency of values in an array within bins
determined by `min`, `max`, and `bins`. The `labels` and `index`
keywords can limit the scope of the histogram to specified sub-regions
within the array.

Parameters
----------
input : array_like
    Data for which to calculate histogram.
min, max : int
    Minimum and maximum values of range of histogram bins.
bins : int
    Number of bins.
labels : array_like, optional
    Labels for objects in `input`.
    If not None, must be same shape as `input`.
index : int or sequence of ints, optional
    Label or labels for which to calculate histogram. If None, all values
    where label is greater than zero are used

Returns
-------
hist : ndarray
    Histogram counts.

Examples
--------
>>> a = np.array([[ 0.    ,  0.2146,  0.5962,  0.    ],
...               [ 0.    ,  0.7778,  0.    ,  0.    ],
...               [ 0.    ,  0.    ,  0.    ,  0.    ],
...               [ 0.    ,  0.    ,  0.7181,  0.2787],
...               [ 0.    ,  0.    ,  0.6573,  0.3094]])
>>> from scipy import ndimage
>>> ndimage.measurements.histogram(a, 0, 1, 10)
array([13,  0,  2,  1,  0,  1,  1,  2,  0,  0])

With labels and no indices, non-zero elements are counted:

>>> lbl, nlbl = ndimage.label(a)
>>> ndimage.measurements.histogram(a, 0, 1, 10, lbl)
array([0, 0, 2, 1, 0, 1, 1, 2, 0, 0])

Indices can be used to count only certain objects:

>>> ndimage.measurements.histogram(a, 0, 1, 10, lbl, 2)
array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
*)

val iterate_structure : ?origin:Py.Object.t -> structure:[>`Ndarray] Np.Obj.t -> iterations:int -> unit -> Py.Object.t
(**
Iterate a structure by dilating it with itself.

Parameters
----------
structure : array_like
   Structuring element (an array of bools, for example), to be dilated with
   itself.
iterations : int
   number of dilations performed on the structure with itself
origin : optional
    If origin is None, only the iterated structure is returned. If
    not, a tuple of the iterated structure and the modified origin is
    returned.

Returns
-------
iterate_structure : ndarray of bools
    A new structuring element obtained by dilating `structure`
    (`iterations` - 1) times with itself.

See also
--------
generate_binary_structure

Examples
--------
>>> from scipy import ndimage
>>> struct = ndimage.generate_binary_structure(2, 1)
>>> struct.astype(int)
array([[0, 1, 0],
       [1, 1, 1],
       [0, 1, 0]])
>>> ndimage.iterate_structure(struct, 2).astype(int)
array([[0, 0, 1, 0, 0],
       [0, 1, 1, 1, 0],
       [1, 1, 1, 1, 1],
       [0, 1, 1, 1, 0],
       [0, 0, 1, 0, 0]])
>>> ndimage.iterate_structure(struct, 3).astype(int)
array([[0, 0, 0, 1, 0, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 1, 1, 1, 1, 1, 0],
       [1, 1, 1, 1, 1, 1, 1],
       [0, 1, 1, 1, 1, 1, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 0, 1, 0, 0, 0]])
*)

val label : ?structure:[>`Ndarray] Np.Obj.t -> ?output:Py.Object.t -> input:[>`Ndarray] Np.Obj.t -> unit -> (Py.Object.t * int)
(**
Label features in an array.

Parameters
----------
input : array_like
    An array-like object to be labeled. Any non-zero values in `input` are
    counted as features and zero values are considered the background.
structure : array_like, optional
    A structuring element that defines feature connections.
    `structure` must be centrosymmetric
    (see Notes).
    If no structuring element is provided,
    one is automatically generated with a squared connectivity equal to
    one.  That is, for a 2-D `input` array, the default structuring element
    is::

        [[0,1,0],
         [1,1,1],
         [0,1,0]]

output : (None, data-type, array_like), optional
    If `output` is a data type, it specifies the type of the resulting
    labeled feature array.
    If `output` is an array-like object, then `output` will be updated
    with the labeled features from this function.  This function can
    operate in-place, by passing output=input.
    Note that the output must be able to store the largest label, or this
    function will raise an Exception.

Returns
-------
label : ndarray or int
    An integer ndarray where each unique feature in `input` has a unique
    label in the returned array.
num_features : int
    How many objects were found.

    If `output` is None, this function returns a tuple of
    (`labeled_array`, `num_features`).

    If `output` is a ndarray, then it will be updated with values in
    `labeled_array` and only `num_features` will be returned by this
    function.

See Also
--------
find_objects : generate a list of slices for the labeled features (or
               objects); useful for finding features' position or
               dimensions

Notes
-----
A centrosymmetric matrix is a matrix that is symmetric about the center.
See [1]_ for more information.

The `structure` matrix must be centrosymmetric to ensure
two-way connections.
For instance, if the `structure` matrix is not centrosymmetric
and is defined as::

    [[0,1,0],
     [1,1,0],
     [0,0,0]]

and the `input` is::

    [[1,2],
     [0,3]]

then the structure matrix would indicate the
entry 2 in the input is connected to 1,
but 1 is not connected to 2.

Examples
--------
Create an image with some features, then label it using the default
(cross-shaped) structuring element:

>>> from scipy.ndimage import label, generate_binary_structure
>>> a = np.array([[0,0,1,1,0,0],
...               [0,0,0,1,0,0],
...               [1,1,0,0,1,0],
...               [0,0,0,1,0,0]])
>>> labeled_array, num_features = label(a)

Each of the 4 features are labeled with a different integer:

>>> num_features
4
>>> labeled_array
array([[0, 0, 1, 1, 0, 0],
       [0, 0, 0, 1, 0, 0],
       [2, 2, 0, 0, 3, 0],
       [0, 0, 0, 4, 0, 0]])

Generate a structuring element that will consider features connected even
if they touch diagonally:

>>> s = generate_binary_structure(2,2)

or,

>>> s = [[1,1,1],
...      [1,1,1],
...      [1,1,1]]

Label the image using the new structuring element:

>>> labeled_array, num_features = label(a, structure=s)

Show the 2 labeled features (note that features 1, 3, and 4 from above are
now considered a single feature):

>>> num_features
2
>>> labeled_array
array([[0, 0, 1, 1, 0, 0],
       [0, 0, 0, 1, 0, 0],
       [2, 2, 0, 0, 1, 0],
       [0, 0, 0, 1, 0, 0]])

References
----------

.. [1] James R. Weaver, 'Centrosymmetric (cross-symmetric)
   matrices, their basic properties, eigenvalues, and
   eigenvectors.' The American Mathematical Monthly 92.10
   (1985): 711-717.
*)

val labeled_comprehension : ?pass_positions:bool -> input:[>`Ndarray] Np.Obj.t -> labels:[`Ndarray of [>`Ndarray] Np.Obj.t | `None] -> index:[`I of int | `Is of int list | `None] -> func:Py.Object.t -> out_dtype:Np.Dtype.t -> default:[`F of float | `I of int | `None] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Roughly equivalent to [func(input[labels == i]) for i in index].

Sequentially applies an arbitrary function (that works on array_like input)
to subsets of an N-D image array specified by `labels` and `index`.
The option exists to provide the function with positional parameters as the
second argument.

Parameters
----------
input : array_like
    Data from which to select `labels` to process.
labels : array_like or None
    Labels to objects in `input`.
    If not None, array must be same shape as `input`.
    If None, `func` is applied to raveled `input`.
index : int, sequence of ints or None
    Subset of `labels` to which to apply `func`.
    If a scalar, a single value is returned.
    If None, `func` is applied to all non-zero values of `labels`.
func : callable
    Python function to apply to `labels` from `input`.
out_dtype : dtype
    Dtype to use for `result`.
default : int, float or None
    Default return value when a element of `index` does not exist
    in `labels`.
pass_positions : bool, optional
    If True, pass linear indices to `func` as a second argument.
    Default is False.

Returns
-------
result : ndarray
    Result of applying `func` to each of `labels` to `input` in `index`.

Examples
--------
>>> a = np.array([[1, 2, 0, 0],
...               [5, 3, 0, 4],
...               [0, 0, 0, 7],
...               [9, 3, 0, 0]])
>>> from scipy import ndimage
>>> lbl, nlbl = ndimage.label(a)
>>> lbls = np.arange(1, nlbl+1)
>>> ndimage.labeled_comprehension(a, lbl, lbls, np.mean, float, 0)
array([ 2.75,  5.5 ,  6.  ])

Falling back to `default`:

>>> lbls = np.arange(1, nlbl+2)
>>> ndimage.labeled_comprehension(a, lbl, lbls, np.mean, float, -1)
array([ 2.75,  5.5 ,  6.  , -1.  ])

Passing positions:

>>> def fn(val, pos):
...     print('fn says: %s : %s' % (val, pos))
...     return (val.sum()) if (pos.sum() % 2 == 0) else (-val.sum())
...
>>> ndimage.labeled_comprehension(a, lbl, lbls, fn, float, 0, True)
fn says: [1 2 5 3] : [0 1 4 5]
fn says: [4 7] : [ 7 11]
fn says: [9 3] : [12 13]
array([ 11.,  11., -12.,   0.])
*)

val laplace : ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Sequence of Py.Object.t | `S of string] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
N-D Laplace filter based on approximate second derivatives.

Parameters
----------
input : array_like
    The input array.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : str or sequence, optional
    The `mode` parameter determines how the input array is extended
    when the filter overlaps a border. By passing a sequence of modes
    with length equal to the number of dimensions of the input array,
    different modes can be specified along each axis. Default value is
    'reflect'. The valid values and their behavior is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.

Examples
--------
>>> from scipy import ndimage, misc
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = misc.ascent()
>>> result = ndimage.laplace(ascent)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()
*)

val map_coordinates : ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?order:int -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?prefilter:bool -> input:[>`Ndarray] Np.Obj.t -> coordinates:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Map the input array to new coordinates by interpolation.

The array of coordinates is used to find, for each point in the output,
the corresponding coordinates in the input. The value of the input at
those coordinates is determined by spline interpolation of the
requested order.

The shape of the output is derived from that of the coordinate
array by dropping the first axis. The values of the array along
the first axis are the coordinates in the input array at which the
output value is found.

Parameters
----------
input : array_like
    The input array.
coordinates : array_like
    The coordinates at which `input` is evaluated.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
order : int, optional
    The order of the spline interpolation, default is 3.
    The order has to be in the range 0-5.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'constant'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
prefilter : bool, optional
    Determines if the input array is prefiltered with `spline_filter`
    before interpolation. The default is True, which will create a
    temporary `float64` array of filtered values if `order > 1`. If
    setting this to False, the output will be slightly blurred if
    `order > 1`, unless the input is prefiltered, i.e. it is the result
    of calling `spline_filter` on the original input.

Returns
-------
map_coordinates : ndarray
    The result of transforming the input. The shape of the output is
    derived from that of `coordinates` by dropping the first axis.

See Also
--------
spline_filter, geometric_transform, scipy.interpolate

Examples
--------
>>> from scipy import ndimage
>>> a = np.arange(12.).reshape((4, 3))
>>> a
array([[  0.,   1.,   2.],
       [  3.,   4.,   5.],
       [  6.,   7.,   8.],
       [  9.,  10.,  11.]])
>>> ndimage.map_coordinates(a, [[0.5, 2], [0.5, 1]], order=1)
array([ 2.,  7.])

Above, the interpolated value of a[0.5, 0.5] gives output[0], while
a[2, 1] is output[1].

>>> inds = np.array([[0.5, 2], [0.5, 4]])
>>> ndimage.map_coordinates(a, inds, order=1, cval=-33.3)
array([  2. , -33.3])
>>> ndimage.map_coordinates(a, inds, order=1, mode='nearest')
array([ 2.,  8.])
>>> ndimage.map_coordinates(a, inds, order=1, cval=0, output=bool)
array([ True, False], dtype=bool)
*)

val maximum : ?labels:[>`Ndarray] Np.Obj.t -> ?index:[>`Ndarray] Np.Obj.t -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Calculate the maximum of the values of an array over labeled regions.

Parameters
----------
input : array_like
    Array_like of values. For each region specified by `labels`, the
    maximal values of `input` over the region is computed.
labels : array_like, optional
    An array of integers marking different regions over which the
    maximum value of `input` is to be computed. `labels` must have the
    same shape as `input`. If `labels` is not specified, the maximum
    over the whole array is returned.
index : array_like, optional
    A list of region labels that are taken into account for computing the
    maxima. If index is None, the maximum over all elements where `labels`
    is non-zero is returned.

Returns
-------
output : float or list of floats
    List of maxima of `input` over the regions determined by `labels` and
    whose index is in `index`. If `index` or `labels` are not specified, a
    float is returned: the maximal value of `input` if `labels` is None,
    and the maximal value of elements where `labels` is greater than zero
    if `index` is None.

See also
--------
label, minimum, median, maximum_position, extrema, sum, mean, variance,
standard_deviation

Notes
-----
The function returns a Python list and not a NumPy array, use
`np.array` to convert the list to an array.

Examples
--------
>>> a = np.arange(16).reshape((4,4))
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]])
>>> labels = np.zeros_like(a)
>>> labels[:2,:2] = 1
>>> labels[2:, 1:3] = 2
>>> labels
array([[1, 1, 0, 0],
       [1, 1, 0, 0],
       [0, 2, 2, 0],
       [0, 2, 2, 0]])
>>> from scipy import ndimage
>>> ndimage.maximum(a)
15.0
>>> ndimage.maximum(a, labels=labels, index=[1,2])
[5.0, 14.0]
>>> ndimage.maximum(a, labels=labels)
14.0

>>> b = np.array([[1, 2, 0, 0],
...               [5, 3, 0, 4],
...               [0, 0, 0, 7],
...               [9, 3, 0, 0]])
>>> labels, labels_nb = ndimage.label(b)
>>> labels
array([[1, 1, 0, 0],
       [1, 1, 0, 2],
       [0, 0, 0, 2],
       [3, 3, 0, 0]])
>>> ndimage.maximum(b, labels=labels, index=np.arange(1, labels_nb + 1))
[5.0, 7.0, 9.0]
*)

val maximum_filter : ?size:[`Bool of bool | `F of float | `S of string | `Tuple of Py.Object.t | `I of int] -> ?footprint:[>`Ndarray] Np.Obj.t -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Sequence of Py.Object.t | `S of string] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`I of int | `Sequence of Py.Object.t] -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Calculate a multidimensional maximum filter.

Parameters
----------
input : array_like
    The input array.
size : scalar or tuple, optional
    See footprint, below. Ignored if footprint is given.
footprint : array, optional
    Either `size` or `footprint` must be defined. `size` gives
    the shape that is taken from the input array, at every element
    position, to define the input to the filter function.
    `footprint` is a boolean array that specifies (implicitly) a
    shape, but also which of the elements within this shape will get
    passed to the filter function. Thus ``size=(n,m)`` is equivalent
    to ``footprint=np.ones((n,m))``.  We adjust `size` to the number
    of dimensions of the input array, so that, if the input array is
    shape (10,10,10), and `size` is 2, then the actual size used is
    (2,2,2). When `footprint` is given, `size` is ignored.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : str or sequence, optional
    The `mode` parameter determines how the input array is extended
    when the filter overlaps a border. By passing a sequence of modes
    with length equal to the number of dimensions of the input array,
    different modes can be specified along each axis. Default value is
    'reflect'. The valid values and their behavior is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : int or sequence, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right. By passing a sequence of origins with length equal to
    the number of dimensions of the input array, different shifts can
    be specified along each axis.

Returns
-------
maximum_filter : ndarray
    Filtered array. Has the same shape as `input`.

Notes
-----
A sequence of modes (one per axis) is only supported when the footprint is
separable. Otherwise, a single mode string must be provided.

Examples
--------
>>> from scipy import ndimage, misc
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = misc.ascent()
>>> result = ndimage.maximum_filter(ascent, size=20)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()
*)

val maximum_filter1d : ?axis:int -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:int -> input:[>`Ndarray] Np.Obj.t -> size:int -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t option
(**
Calculate a 1-D maximum filter along the given axis.

The lines of the array along the given axis are filtered with a
maximum filter of given size.

Parameters
----------
input : array_like
    The input array.
size : int
    Length along which to calculate the 1-D maximum.
axis : int, optional
    The axis of `input` along which to calculate. Default is -1.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'reflect'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : int, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right.

Returns
-------
maximum1d : ndarray, None
    Maximum-filtered array with same shape as input.
    None if `output` is not None

Notes
-----
This function implements the MAXLIST algorithm [1]_, as described by
Richard Harter [2]_, and has a guaranteed O(n) performance, `n` being
the `input` length, regardless of filter size.

References
----------
.. [1] http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.42.2777
.. [2] http://www.richardhartersworld.com/cri/2001/slidingmin.html

Examples
--------
>>> from scipy.ndimage import maximum_filter1d
>>> maximum_filter1d([2, 8, 0, 4, 1, 9, 9, 0], size=3)
array([8, 8, 8, 4, 9, 9, 9, 9])
*)

val maximum_position : ?labels:[>`Ndarray] Np.Obj.t -> ?index:[>`Ndarray] Np.Obj.t -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Find the positions of the maximums of the values of an array at labels.

For each region specified by `labels`, the position of the maximum
value of `input` within the region is returned.

Parameters
----------
input : array_like
    Array_like of values.
labels : array_like, optional
    An array of integers marking different regions over which the
    position of the maximum value of `input` is to be computed.
    `labels` must have the same shape as `input`. If `labels` is not
    specified, the location of the first maximum over the whole
    array is returned.

    The `labels` argument only works when `index` is specified.
index : array_like, optional
    A list of region labels that are taken into account for finding the
    location of the maxima. If `index` is None, the first maximum
    over all elements where `labels` is non-zero is returned.

    The `index` argument only works when `labels` is specified.

Returns
-------
output : list of tuples of ints
    List of tuples of ints that specify the location of maxima of
    `input` over the regions determined by `labels` and whose index
    is in `index`.

    If `index` or `labels` are not specified, a tuple of ints is
    returned specifying the location of the ``first`` maximal value
    of `input`.

See also
--------
label, minimum, median, maximum_position, extrema, sum, mean, variance,
standard_deviation

Examples
--------
>>> from scipy import ndimage
>>> a = np.array([[1, 2, 0, 0],
...               [5, 3, 0, 4],
...               [0, 0, 0, 7],
...               [9, 3, 0, 0]])
>>> ndimage.maximum_position(a)
(3, 0)

Features to process can be specified using `labels` and `index`:

>>> lbl = np.array([[0, 1, 2, 3],
...                 [0, 1, 2, 3],
...                 [0, 1, 2, 3],
...                 [0, 1, 2, 3]])
>>> ndimage.maximum_position(a, lbl, 1)
(1, 1)

If no index is given, non-zero `labels` are processed:

>>> ndimage.maximum_position(a, lbl)
(2, 3)

If there are no maxima, the position of the first element is returned:

>>> ndimage.maximum_position(a, lbl, 2)
(0, 2)
*)

val mean : ?labels:[>`Ndarray] Np.Obj.t -> ?index:[`I of int | `Is of int list] -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Calculate the mean of the values of an array at labels.

Parameters
----------
input : array_like
    Array on which to compute the mean of elements over distinct
    regions.
labels : array_like, optional
    Array of labels of same shape, or broadcastable to the same shape as
    `input`. All elements sharing the same label form one region over
    which the mean of the elements is computed.
index : int or sequence of ints, optional
    Labels of the objects over which the mean is to be computed.
    Default is None, in which case the mean for all values where label is
    greater than 0 is calculated.

Returns
-------
out : list
    Sequence of same length as `index`, with the mean of the different
    regions labeled by the labels in `index`.

See also
--------
variance, standard_deviation, minimum, maximum, sum, label

Examples
--------
>>> from scipy import ndimage
>>> a = np.arange(25).reshape((5,5))
>>> labels = np.zeros_like(a)
>>> labels[3:5,3:5] = 1
>>> index = np.unique(labels)
>>> labels
array([[0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 1, 1],
       [0, 0, 0, 1, 1]])
>>> index
array([0, 1])
>>> ndimage.mean(a, labels=labels, index=index)
[10.285714285714286, 21.0]
*)

val median : ?labels:[>`Ndarray] Np.Obj.t -> ?index:[>`Ndarray] Np.Obj.t -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Calculate the median of the values of an array over labeled regions.

Parameters
----------
input : array_like
    Array_like of values. For each region specified by `labels`, the
    median value of `input` over the region is computed.
labels : array_like, optional
    An array_like of integers marking different regions over which the
    median value of `input` is to be computed. `labels` must have the
    same shape as `input`. If `labels` is not specified, the median
    over the whole array is returned.
index : array_like, optional
    A list of region labels that are taken into account for computing the
    medians. If index is None, the median over all elements where `labels`
    is non-zero is returned.

Returns
-------
median : float or list of floats
    List of medians of `input` over the regions determined by `labels` and
    whose index is in `index`. If `index` or `labels` are not specified, a
    float is returned: the median value of `input` if `labels` is None,
    and the median value of elements where `labels` is greater than zero
    if `index` is None.

See also
--------
label, minimum, maximum, extrema, sum, mean, variance, standard_deviation

Notes
-----
The function returns a Python list and not a NumPy array, use
`np.array` to convert the list to an array.

Examples
--------
>>> from scipy import ndimage
>>> a = np.array([[1, 2, 0, 1],
...               [5, 3, 0, 4],
...               [0, 0, 0, 7],
...               [9, 3, 0, 0]])
>>> labels, labels_nb = ndimage.label(a)
>>> labels
array([[1, 1, 0, 2],
       [1, 1, 0, 2],
       [0, 0, 0, 2],
       [3, 3, 0, 0]])
>>> ndimage.median(a, labels=labels, index=np.arange(1, labels_nb + 1))
[2.5, 4.0, 6.0]
>>> ndimage.median(a)
1.0
>>> ndimage.median(a, labels=labels)
3.0
*)

val median_filter : ?size:[`Bool of bool | `F of float | `S of string | `Tuple of Py.Object.t | `I of int] -> ?footprint:[>`Ndarray] Np.Obj.t -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`I of int | `Sequence of Py.Object.t] -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Calculate a multidimensional median filter.

Parameters
----------
input : array_like
    The input array.
size : scalar or tuple, optional
    See footprint, below. Ignored if footprint is given.
footprint : array, optional
    Either `size` or `footprint` must be defined. `size` gives
    the shape that is taken from the input array, at every element
    position, to define the input to the filter function.
    `footprint` is a boolean array that specifies (implicitly) a
    shape, but also which of the elements within this shape will get
    passed to the filter function. Thus ``size=(n,m)`` is equivalent
    to ``footprint=np.ones((n,m))``.  We adjust `size` to the number
    of dimensions of the input array, so that, if the input array is
    shape (10,10,10), and `size` is 2, then the actual size used is
    (2,2,2). When `footprint` is given, `size` is ignored.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'reflect'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : int or sequence, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right. By passing a sequence of origins with length equal to
    the number of dimensions of the input array, different shifts can
    be specified along each axis.

Returns
-------
median_filter : ndarray
    Filtered array. Has the same shape as `input`.

Examples
--------
>>> from scipy import ndimage, misc
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = misc.ascent()
>>> result = ndimage.median_filter(ascent, size=20)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()
*)

val minimum : ?labels:[>`Ndarray] Np.Obj.t -> ?index:[>`Ndarray] Np.Obj.t -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Calculate the minimum of the values of an array over labeled regions.

Parameters
----------
input : array_like
    Array_like of values. For each region specified by `labels`, the
    minimal values of `input` over the region is computed.
labels : array_like, optional
    An array_like of integers marking different regions over which the
    minimum value of `input` is to be computed. `labels` must have the
    same shape as `input`. If `labels` is not specified, the minimum
    over the whole array is returned.
index : array_like, optional
    A list of region labels that are taken into account for computing the
    minima. If index is None, the minimum over all elements where `labels`
    is non-zero is returned.

Returns
-------
minimum : float or list of floats
    List of minima of `input` over the regions determined by `labels` and
    whose index is in `index`. If `index` or `labels` are not specified, a
    float is returned: the minimal value of `input` if `labels` is None,
    and the minimal value of elements where `labels` is greater than zero
    if `index` is None.

See also
--------
label, maximum, median, minimum_position, extrema, sum, mean, variance,
standard_deviation

Notes
-----
The function returns a Python list and not a NumPy array, use
`np.array` to convert the list to an array.

Examples
--------
>>> from scipy import ndimage
>>> a = np.array([[1, 2, 0, 0],
...               [5, 3, 0, 4],
...               [0, 0, 0, 7],
...               [9, 3, 0, 0]])
>>> labels, labels_nb = ndimage.label(a)
>>> labels
array([[1, 1, 0, 0],
       [1, 1, 0, 2],
       [0, 0, 0, 2],
       [3, 3, 0, 0]])
>>> ndimage.minimum(a, labels=labels, index=np.arange(1, labels_nb + 1))
[1.0, 4.0, 3.0]
>>> ndimage.minimum(a)
0.0
>>> ndimage.minimum(a, labels=labels)
1.0
*)

val minimum_filter : ?size:[`Bool of bool | `F of float | `S of string | `Tuple of Py.Object.t | `I of int] -> ?footprint:[>`Ndarray] Np.Obj.t -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Sequence of Py.Object.t | `S of string] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`I of int | `Sequence of Py.Object.t] -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Calculate a multidimensional minimum filter.

Parameters
----------
input : array_like
    The input array.
size : scalar or tuple, optional
    See footprint, below. Ignored if footprint is given.
footprint : array, optional
    Either `size` or `footprint` must be defined. `size` gives
    the shape that is taken from the input array, at every element
    position, to define the input to the filter function.
    `footprint` is a boolean array that specifies (implicitly) a
    shape, but also which of the elements within this shape will get
    passed to the filter function. Thus ``size=(n,m)`` is equivalent
    to ``footprint=np.ones((n,m))``.  We adjust `size` to the number
    of dimensions of the input array, so that, if the input array is
    shape (10,10,10), and `size` is 2, then the actual size used is
    (2,2,2). When `footprint` is given, `size` is ignored.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : str or sequence, optional
    The `mode` parameter determines how the input array is extended
    when the filter overlaps a border. By passing a sequence of modes
    with length equal to the number of dimensions of the input array,
    different modes can be specified along each axis. Default value is
    'reflect'. The valid values and their behavior is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : int or sequence, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right. By passing a sequence of origins with length equal to
    the number of dimensions of the input array, different shifts can
    be specified along each axis.

Returns
-------
minimum_filter : ndarray
    Filtered array. Has the same shape as `input`.

Notes
-----
A sequence of modes (one per axis) is only supported when the footprint is
separable. Otherwise, a single mode string must be provided.

Examples
--------
>>> from scipy import ndimage, misc
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = misc.ascent()
>>> result = ndimage.minimum_filter(ascent, size=20)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()
*)

val minimum_filter1d : ?axis:int -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:int -> input:[>`Ndarray] Np.Obj.t -> size:int -> unit -> Py.Object.t
(**
Calculate a 1-D minimum filter along the given axis.

The lines of the array along the given axis are filtered with a
minimum filter of given size.

Parameters
----------
input : array_like
    The input array.
size : int
    length along which to calculate 1D minimum
axis : int, optional
    The axis of `input` along which to calculate. Default is -1.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'reflect'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : int, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right.

Notes
-----
This function implements the MINLIST algorithm [1]_, as described by
Richard Harter [2]_, and has a guaranteed O(n) performance, `n` being
the `input` length, regardless of filter size.

References
----------
.. [1] http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.42.2777
.. [2] http://www.richardhartersworld.com/cri/2001/slidingmin.html


Examples
--------
>>> from scipy.ndimage import minimum_filter1d
>>> minimum_filter1d([2, 8, 0, 4, 1, 9, 9, 0], size=3)
array([2, 0, 0, 0, 1, 1, 0, 0])
*)

val minimum_position : ?labels:[>`Ndarray] Np.Obj.t -> ?index:[>`Ndarray] Np.Obj.t -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Find the positions of the minimums of the values of an array at labels.

Parameters
----------
input : array_like
    Array_like of values.
labels : array_like, optional
    An array of integers marking different regions over which the
    position of the minimum value of `input` is to be computed.
    `labels` must have the same shape as `input`. If `labels` is not
    specified, the location of the first minimum over the whole
    array is returned.

    The `labels` argument only works when `index` is specified.
index : array_like, optional
    A list of region labels that are taken into account for finding the
    location of the minima. If `index` is None, the ``first`` minimum
    over all elements where `labels` is non-zero is returned.

    The `index` argument only works when `labels` is specified.

Returns
-------
output : list of tuples of ints
    Tuple of ints or list of tuples of ints that specify the location
    of minima of `input` over the regions determined by `labels` and
    whose index is in `index`.

    If `index` or `labels` are not specified, a tuple of ints is
    returned specifying the location of the first minimal value of `input`.

See also
--------
label, minimum, median, maximum_position, extrema, sum, mean, variance,
standard_deviation

Examples
--------
>>> a = np.array([[10, 20, 30],
...               [40, 80, 100],
...               [1, 100, 200]])
>>> b = np.array([[1, 2, 0, 1],
...               [5, 3, 0, 4],
...               [0, 0, 0, 7],
...               [9, 3, 0, 0]])

>>> from scipy import ndimage

>>> ndimage.minimum_position(a)
(2, 0)
>>> ndimage.minimum_position(b)
(0, 2)

Features to process can be specified using `labels` and `index`:

>>> label, pos = ndimage.label(a)
>>> ndimage.minimum_position(a, label, index=np.arange(1, pos+1))
[(2, 0)]

>>> label, pos = ndimage.label(b)
>>> ndimage.minimum_position(b, label, index=np.arange(1, pos+1))
[(0, 0), (0, 3), (3, 1)]
*)

val morphological_gradient : ?size:int list -> ?footprint:Py.Object.t -> ?structure:Py.Object.t -> ?output:[>`Ndarray] Np.Obj.t -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`F of float | `I of int | `Bool of bool | `S of string] -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional morphological gradient.

The morphological gradient is calculated as the difference between a
dilation and an erosion of the input with a given structuring element.

Parameters
----------
input : array_like
    Array over which to compute the morphlogical gradient.
size : tuple of ints
    Shape of a flat and full structuring element used for the mathematical
    morphology operations. Optional if `footprint` or `structure` is
    provided. A larger `size` yields a more blurred gradient.
footprint : array of ints, optional
    Positions of non-infinite elements of a flat structuring element
    used for the morphology operations. Larger footprints
    give a more blurred morphological gradient.
structure : array of ints, optional
    Structuring element used for the morphology operations.
    `structure` may be a non-flat structuring element.
output : array, optional
    An array used for storing the output of the morphological gradient
    may be provided.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the array borders are
    handled, where `cval` is the value when mode is equal to
    'constant'. Default is 'reflect'
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : scalar, optional
    The `origin` parameter controls the placement of the filter.
    Default 0

Returns
-------
morphological_gradient : ndarray
    Morphological gradient of `input`.

See also
--------
grey_dilation, grey_erosion, gaussian_gradient_magnitude

Notes
-----
For a flat structuring element, the morphological gradient
computed at a given point corresponds to the maximal difference
between elements of the input among the elements covered by the
structuring element centered on the point.

References
----------
.. [1] https://en.wikipedia.org/wiki/Mathematical_morphology

Examples
--------
>>> from scipy import ndimage
>>> a = np.zeros((7,7), dtype=int)
>>> a[2:5, 2:5] = 1
>>> ndimage.morphological_gradient(a, size=(3,3))
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 1, 1, 1, 1, 1, 0],
       [0, 1, 1, 1, 1, 1, 0],
       [0, 1, 1, 0, 1, 1, 0],
       [0, 1, 1, 1, 1, 1, 0],
       [0, 1, 1, 1, 1, 1, 0],
       [0, 0, 0, 0, 0, 0, 0]])
>>> # The morphological gradient is computed as the difference
>>> # between a dilation and an erosion
>>> ndimage.grey_dilation(a, size=(3,3)) -\
...  ndimage.grey_erosion(a, size=(3,3))
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 1, 1, 1, 1, 1, 0],
       [0, 1, 1, 1, 1, 1, 0],
       [0, 1, 1, 0, 1, 1, 0],
       [0, 1, 1, 1, 1, 1, 0],
       [0, 1, 1, 1, 1, 1, 0],
       [0, 0, 0, 0, 0, 0, 0]])
>>> a = np.zeros((7,7), dtype=int)
>>> a[2:5, 2:5] = 1
>>> a[4,4] = 2; a[2,3] = 3
>>> a
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 3, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 2, 0, 0],
       [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0]])
>>> ndimage.morphological_gradient(a, size=(3,3))
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 1, 3, 3, 3, 1, 0],
       [0, 1, 3, 3, 3, 1, 0],
       [0, 1, 3, 2, 3, 2, 0],
       [0, 1, 1, 2, 2, 2, 0],
       [0, 1, 1, 2, 2, 2, 0],
       [0, 0, 0, 0, 0, 0, 0]])
*)

val morphological_laplace : ?size:[`I of int | `Is of int list] -> ?footprint:[`Ndarray of [>`Ndarray] Np.Obj.t | `Bool of bool] -> ?structure:Py.Object.t -> ?output:[>`Ndarray] Np.Obj.t -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:Py.Object.t -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional morphological laplace.

Parameters
----------
input : array_like
    Input.
size : int or sequence of ints, optional
    See `structure`.
footprint : bool or ndarray, optional
    See `structure`.
structure : structure, optional
    Either `size`, `footprint`, or the `structure` must be provided.
output : ndarray, optional
    An output array can optionally be provided.
mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional
    The mode parameter determines how the array borders are handled.
    For 'constant' mode, values beyond borders are set to be `cval`.
    Default is 'reflect'.
cval : scalar, optional
    Value to fill past edges of input if mode is 'constant'.
    Default is 0.0
origin : origin, optional
    The origin parameter controls the placement of the filter.

Returns
-------
morphological_laplace : ndarray
    Output
*)

val percentile_filter : ?size:[`Bool of bool | `F of float | `S of string | `Tuple of Py.Object.t | `I of int] -> ?footprint:[>`Ndarray] Np.Obj.t -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`I of int | `Sequence of Py.Object.t] -> input:[>`Ndarray] Np.Obj.t -> percentile:[`F of float | `I of int | `Bool of bool | `S of string] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Calculate a multidimensional percentile filter.

Parameters
----------
input : array_like
    The input array.
percentile : scalar
    The percentile parameter may be less then zero, i.e.,
    percentile = -20 equals percentile = 80
size : scalar or tuple, optional
    See footprint, below. Ignored if footprint is given.
footprint : array, optional
    Either `size` or `footprint` must be defined. `size` gives
    the shape that is taken from the input array, at every element
    position, to define the input to the filter function.
    `footprint` is a boolean array that specifies (implicitly) a
    shape, but also which of the elements within this shape will get
    passed to the filter function. Thus ``size=(n,m)`` is equivalent
    to ``footprint=np.ones((n,m))``.  We adjust `size` to the number
    of dimensions of the input array, so that, if the input array is
    shape (10,10,10), and `size` is 2, then the actual size used is
    (2,2,2). When `footprint` is given, `size` is ignored.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'reflect'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : int or sequence, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right. By passing a sequence of origins with length equal to
    the number of dimensions of the input array, different shifts can
    be specified along each axis.

Returns
-------
percentile_filter : ndarray
    Filtered array. Has the same shape as `input`.

Examples
--------
>>> from scipy import ndimage, misc
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = misc.ascent()
>>> result = ndimage.percentile_filter(ascent, percentile=20, size=20)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()
*)

val prewitt : ?axis:int -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Sequence of Py.Object.t | `S of string] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Calculate a Prewitt filter.

Parameters
----------
input : array_like
    The input array.
axis : int, optional
    The axis of `input` along which to calculate. Default is -1.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : str or sequence, optional
    The `mode` parameter determines how the input array is extended
    when the filter overlaps a border. By passing a sequence of modes
    with length equal to the number of dimensions of the input array,
    different modes can be specified along each axis. Default value is
    'reflect'. The valid values and their behavior is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.

Examples
--------
>>> from scipy import ndimage, misc
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = misc.ascent()
>>> result = ndimage.prewitt(ascent)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()
*)

val rank_filter : ?size:[`Bool of bool | `F of float | `S of string | `Tuple of Py.Object.t | `I of int] -> ?footprint:[>`Ndarray] Np.Obj.t -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`I of int | `Sequence of Py.Object.t] -> input:[>`Ndarray] Np.Obj.t -> rank:int -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Calculate a multidimensional rank filter.

Parameters
----------
input : array_like
    The input array.
rank : int
    The rank parameter may be less then zero, i.e., rank = -1
    indicates the largest element.
size : scalar or tuple, optional
    See footprint, below. Ignored if footprint is given.
footprint : array, optional
    Either `size` or `footprint` must be defined. `size` gives
    the shape that is taken from the input array, at every element
    position, to define the input to the filter function.
    `footprint` is a boolean array that specifies (implicitly) a
    shape, but also which of the elements within this shape will get
    passed to the filter function. Thus ``size=(n,m)`` is equivalent
    to ``footprint=np.ones((n,m))``.  We adjust `size` to the number
    of dimensions of the input array, so that, if the input array is
    shape (10,10,10), and `size` is 2, then the actual size used is
    (2,2,2). When `footprint` is given, `size` is ignored.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'reflect'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : int or sequence, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right. By passing a sequence of origins with length equal to
    the number of dimensions of the input array, different shifts can
    be specified along each axis.

Returns
-------
rank_filter : ndarray
    Filtered array. Has the same shape as `input`.

Examples
--------
>>> from scipy import ndimage, misc
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = misc.ascent()
>>> result = ndimage.rank_filter(ascent, rank=42, size=20)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()
*)

val rotate : ?axes:Py.Object.t -> ?reshape:bool -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?order:int -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?prefilter:bool -> input:[>`Ndarray] Np.Obj.t -> angle:float -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Rotate an array.

The array is rotated in the plane defined by the two axes given by the
`axes` parameter using spline interpolation of the requested order.

Parameters
----------
input : array_like
    The input array.
angle : float
    The rotation angle in degrees.
axes : tuple of 2 ints, optional
    The two axes that define the plane of rotation. Default is the first
    two axes.
reshape : bool, optional
    If `reshape` is true, the output shape is adapted so that the input
    array is contained completely in the output. Default is True.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
order : int, optional
    The order of the spline interpolation, default is 3.
    The order has to be in the range 0-5.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'constant'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
prefilter : bool, optional
    Determines if the input array is prefiltered with `spline_filter`
    before interpolation. The default is True, which will create a
    temporary `float64` array of filtered values if `order > 1`. If
    setting this to False, the output will be slightly blurred if
    `order > 1`, unless the input is prefiltered, i.e. it is the result
    of calling `spline_filter` on the original input.

Returns
-------
rotate : ndarray
    The rotated input.

Examples
--------
>>> from scipy import ndimage, misc
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure(figsize=(10, 3))
>>> ax1, ax2, ax3 = fig.subplots(1, 3)
>>> img = misc.ascent()
>>> img_45 = ndimage.rotate(img, 45, reshape=False)
>>> full_img_45 = ndimage.rotate(img, 45, reshape=True)
>>> ax1.imshow(img, cmap='gray')
>>> ax1.set_axis_off()
>>> ax2.imshow(img_45, cmap='gray')
>>> ax2.set_axis_off()
>>> ax3.imshow(full_img_45, cmap='gray')
>>> ax3.set_axis_off()
>>> fig.set_tight_layout(True)
>>> plt.show()
>>> print(img.shape)
(512, 512)
>>> print(img_45.shape)
(512, 512)
>>> print(full_img_45.shape)
(724, 724)
*)

val shift : ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?order:int -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?prefilter:bool -> input:[>`Ndarray] Np.Obj.t -> shift:[`F of float | `Sequence of Py.Object.t] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Shift an array.

The array is shifted using spline interpolation of the requested order.
Points outside the boundaries of the input are filled according to the
given mode.

Parameters
----------
input : array_like
    The input array.
shift : float or sequence
    The shift along the axes. If a float, `shift` is the same for each
    axis. If a sequence, `shift` should contain one value for each axis.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
order : int, optional
    The order of the spline interpolation, default is 3.
    The order has to be in the range 0-5.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'constant'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
prefilter : bool, optional
    Determines if the input array is prefiltered with `spline_filter`
    before interpolation. The default is True, which will create a
    temporary `float64` array of filtered values if `order > 1`. If
    setting this to False, the output will be slightly blurred if
    `order > 1`, unless the input is prefiltered, i.e. it is the result
    of calling `spline_filter` on the original input.

Returns
-------
shift : ndarray
    The shifted input.
*)

val sobel : ?axis:int -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Sequence of Py.Object.t | `S of string] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Calculate a Sobel filter.

Parameters
----------
input : array_like
    The input array.
axis : int, optional
    The axis of `input` along which to calculate. Default is -1.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : str or sequence, optional
    The `mode` parameter determines how the input array is extended
    when the filter overlaps a border. By passing a sequence of modes
    with length equal to the number of dimensions of the input array,
    different modes can be specified along each axis. Default value is
    'reflect'. The valid values and their behavior is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.

Examples
--------
>>> from scipy import ndimage, misc
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = misc.ascent()
>>> result = ndimage.sobel(ascent)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()
*)

val spline_filter : ?order:Py.Object.t -> ?output:Py.Object.t -> ?mode:Py.Object.t -> input:Py.Object.t -> unit -> Py.Object.t
(**
Multidimensional spline filter.

For more details, see `spline_filter1d`.

See Also
--------
spline_filter1d : Calculate a 1-D spline filter along the given axis.

Notes
-----
The multidimensional filter is implemented as a sequence of
1-D spline filters. The intermediate arrays are stored
in the same data type as the output. Therefore, for output types
with a limited precision, the results may be imprecise because
intermediate results may be stored with insufficient precision.

Examples
--------
We can filter an image using multidimentional splines:

>>> from scipy.ndimage import spline_filter
>>> import matplotlib.pyplot as plt
>>> orig_img = np.eye(20)  # create an image
>>> orig_img[10, :] = 1.0
>>> sp_filter = spline_filter(orig_img, order=3)
>>> f, ax = plt.subplots(1, 2, sharex=True)
>>> for ind, data in enumerate([[orig_img, 'original image'],
...                             [sp_filter, 'spline filter']]):
...     ax[ind].imshow(data[0], cmap='gray_r')
...     ax[ind].set_title(data[1])
>>> plt.tight_layout()
>>> plt.show()
*)

val spline_filter1d : ?order:int -> ?axis:int -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Calculate a 1-D spline filter along the given axis.

The lines of the array along the given axis are filtered by a
spline filter. The order of the spline must be >= 2 and <= 5.

Parameters
----------
input : array_like
    The input array.
order : int, optional
    The order of the spline, default is 3.
axis : int, optional
    The axis along which the spline filter is applied. Default is the last
    axis.
output : ndarray or dtype, optional
    The array in which to place the output, or the dtype of the returned
    array. Default is ``numpy.float64``.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'constant'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.

Returns
-------
spline_filter1d : ndarray
    The filtered input.

Notes
-----
All functions in `ndimage.interpolation` do spline interpolation of
the input image. If using B-splines of `order > 1`, the input image
values have to be converted to B-spline coefficients first, which is
done by applying this 1-D filter sequentially along all
axes of the input. All functions that require B-spline coefficients
will automatically filter their inputs, a behavior controllable with
the `prefilter` keyword argument. For functions that accept a `mode`
parameter, the result will only be correct if it matches the `mode`
used when filtering.

See Also
--------
spline_filter : Multidimensional spline filter.

Examples
--------
We can filter an image using 1-D spline along the given axis:

>>> from scipy.ndimage import spline_filter1d
>>> import matplotlib.pyplot as plt
>>> orig_img = np.eye(20)  # create an image
>>> orig_img[10, :] = 1.0
>>> sp_filter_axis_0 = spline_filter1d(orig_img, axis=0)
>>> sp_filter_axis_1 = spline_filter1d(orig_img, axis=1)
>>> f, ax = plt.subplots(1, 3, sharex=True)
>>> for ind, data in enumerate([[orig_img, 'original image'],
...             [sp_filter_axis_0, 'spline filter (axis=0)'],
...             [sp_filter_axis_1, 'spline filter (axis=1)']]):
...     ax[ind].imshow(data[0], cmap='gray_r')
...     ax[ind].set_title(data[1])
>>> plt.tight_layout()
>>> plt.show()
*)

val standard_deviation : ?labels:[>`Ndarray] Np.Obj.t -> ?index:[`I of int | `Is of int list] -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Calculate the standard deviation of the values of an N-D image array,
optionally at specified sub-regions.

Parameters
----------
input : array_like
    N-D image data to process.
labels : array_like, optional
    Labels to identify sub-regions in `input`.
    If not None, must be same shape as `input`.
index : int or sequence of ints, optional
    `labels` to include in output. If None (default), all values where
    `labels` is non-zero are used.

Returns
-------
standard_deviation : float or ndarray
    Values of standard deviation, for each sub-region if `labels` and
    `index` are specified.

See Also
--------
label, variance, maximum, minimum, extrema

Examples
--------
>>> a = np.array([[1, 2, 0, 0],
...               [5, 3, 0, 4],
...               [0, 0, 0, 7],
...               [9, 3, 0, 0]])
>>> from scipy import ndimage
>>> ndimage.standard_deviation(a)
2.7585095613392387

Features to process can be specified using `labels` and `index`:

>>> lbl, nlbl = ndimage.label(a)
>>> ndimage.standard_deviation(a, lbl, index=np.arange(1, nlbl+1))
array([ 1.479,  1.5  ,  3.   ])

If no index is given, non-zero `labels` are processed:

>>> ndimage.standard_deviation(a, lbl)
2.4874685927665499
*)

val sum : ?labels:Py.Object.t -> ?index:[>`Ndarray] Np.Obj.t -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Calculate the sum of the values of the array.

Parameters
----------
input : array_like
    Values of `input` inside the regions defined by `labels`
    are summed together.
labels : array_like of ints, optional
    Assign labels to the values of the array. Has to have the same shape as
    `input`.
index : array_like, optional
    A single label number or a sequence of label numbers of
    the objects to be measured.

Returns
-------
sum : ndarray or scalar
    An array of the sums of values of `input` inside the regions defined
    by `labels` with the same shape as `index`. If 'index' is None or scalar,
    a scalar is returned.

See also
--------
mean, median

Examples
--------
>>> from scipy import ndimage
>>> input =  [0,1,2,3]
>>> labels = [1,1,2,2]
>>> ndimage.sum(input, labels, index=[1,2])
[1.0, 5.0]
>>> ndimage.sum(input, labels, index=1)
1
>>> ndimage.sum(input, labels)
6
*)

val uniform_filter : ?size:[`I of int | `Is of int list] -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Sequence of Py.Object.t | `S of string] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`I of int | `Sequence of Py.Object.t] -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional uniform filter.

Parameters
----------
input : array_like
    The input array.
size : int or sequence of ints, optional
    The sizes of the uniform filter are given for each axis as a
    sequence, or as a single number, in which case the size is
    equal for all axes.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : str or sequence, optional
    The `mode` parameter determines how the input array is extended
    when the filter overlaps a border. By passing a sequence of modes
    with length equal to the number of dimensions of the input array,
    different modes can be specified along each axis. Default value is
    'reflect'. The valid values and their behavior is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : int or sequence, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right. By passing a sequence of origins with length equal to
    the number of dimensions of the input array, different shifts can
    be specified along each axis.

Returns
-------
uniform_filter : ndarray
    Filtered array. Has the same shape as `input`.

Notes
-----
The multidimensional filter is implemented as a sequence of
1-D uniform filters. The intermediate arrays are stored
in the same data type as the output. Therefore, for output types
with a limited precision, the results may be imprecise because
intermediate results may be stored with insufficient precision.

Examples
--------
>>> from scipy import ndimage, misc
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = misc.ascent()
>>> result = ndimage.uniform_filter(ascent, size=20)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()
*)

val uniform_filter1d : ?axis:int -> ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:int -> input:[>`Ndarray] Np.Obj.t -> size:int -> unit -> Py.Object.t
(**
Calculate a 1-D uniform filter along the given axis.

The lines of the array along the given axis are filtered with a
uniform filter of given size.

Parameters
----------
input : array_like
    The input array.
size : int
    length of uniform filter
axis : int, optional
    The axis of `input` along which to calculate. Default is -1.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'reflect'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
origin : int, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right.

Examples
--------
>>> from scipy.ndimage import uniform_filter1d
>>> uniform_filter1d([2, 8, 0, 4, 1, 9, 9, 0], size=3)
array([4, 3, 4, 1, 4, 6, 6, 3])
*)

val variance : ?labels:[>`Ndarray] Np.Obj.t -> ?index:[`I of int | `Is of int list] -> input:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Calculate the variance of the values of an N-D image array, optionally at
specified sub-regions.

Parameters
----------
input : array_like
    Nd-image data to process.
labels : array_like, optional
    Labels defining sub-regions in `input`.
    If not None, must be same shape as `input`.
index : int or sequence of ints, optional
    `labels` to include in output.  If None (default), all values where
    `labels` is non-zero are used.

Returns
-------
variance : float or ndarray
    Values of variance, for each sub-region if `labels` and `index` are
    specified.

See Also
--------
label, standard_deviation, maximum, minimum, extrema

Examples
--------
>>> a = np.array([[1, 2, 0, 0],
...               [5, 3, 0, 4],
...               [0, 0, 0, 7],
...               [9, 3, 0, 0]])
>>> from scipy import ndimage
>>> ndimage.variance(a)
7.609375

Features to process can be specified using `labels` and `index`:

>>> lbl, nlbl = ndimage.label(a)
>>> ndimage.variance(a, lbl, index=np.arange(1, nlbl+1))
array([ 2.1875,  2.25  ,  9.    ])

If no index is given, all non-zero `labels` are processed:

>>> ndimage.variance(a, lbl)
6.1875
*)

val watershed_ift : ?structure:Py.Object.t -> ?output:[>`Ndarray] Np.Obj.t -> input:[>`Ndarray] Np.Obj.t -> markers:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Apply watershed from markers using image foresting transform algorithm.

Parameters
----------
input : array_like
    Input.
markers : array_like
    Markers are points within each watershed that form the beginning
    of the process. Negative markers are considered background markers
    which are processed after the other markers.
structure : structure element, optional
    A structuring element defining the connectivity of the object can be
    provided. If None, an element is generated with a squared
    connectivity equal to one.
output : ndarray, optional
    An output array can optionally be provided. The same shape as input.

Returns
-------
watershed_ift : ndarray
    Output.  Same shape as `input`.

References
----------
.. [1] A.X. Falcao, J. Stolfi and R. de Alencar Lotufo, 'The image
       foresting transform: theory, algorithms, and applications',
       Pattern Analysis and Machine Intelligence, vol. 26, pp. 19-29, 2004.
*)

val white_tophat : ?size:int list -> ?footprint:Py.Object.t -> ?structure:Py.Object.t -> ?output:[>`Ndarray] Np.Obj.t -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?origin:[`F of float | `I of int | `Bool of bool | `S of string] -> input:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Multidimensional white tophat filter.

Parameters
----------
input : array_like
    Input.
size : tuple of ints
    Shape of a flat and full structuring element used for the filter.
    Optional if `footprint` or `structure` is provided.
footprint : array of ints, optional
    Positions of elements of a flat structuring element
    used for the white tophat filter.
structure : array of ints, optional
    Structuring element used for the filter. `structure`
    may be a non-flat structuring element.
output : array, optional
    An array used for storing the output of the filter may be provided.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the array borders are
    handled, where `cval` is the value when mode is equal to
    'constant'. Default is 'reflect'
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'.
    Default is 0.0.
origin : scalar, optional
    The `origin` parameter controls the placement of the filter.
    Default is 0.

Returns
-------
output : ndarray
    Result of the filter of `input` with `structure`.

See also
--------
black_tophat
*)

val zoom : ?output:[`Ndarray of [>`Ndarray] Np.Obj.t | `Dtype of Np.Dtype.t] -> ?order:int -> ?mode:[`Reflect | `Constant | `Nearest | `Mirror | `Wrap] -> ?cval:[`F of float | `I of int | `Bool of bool | `S of string] -> ?prefilter:bool -> input:[>`Ndarray] Np.Obj.t -> zoom:[`F of float | `Sequence of Py.Object.t] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Zoom an array.

The array is zoomed using spline interpolation of the requested order.

Parameters
----------
input : array_like
    The input array.
zoom : float or sequence
    The zoom factor along the axes. If a float, `zoom` is the same for each
    axis. If a sequence, `zoom` should contain one value for each axis.
output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.
order : int, optional
    The order of the spline interpolation, default is 3.
    The order has to be in the range 0-5.
mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'constant'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.
cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.
prefilter : bool, optional
    Determines if the input array is prefiltered with `spline_filter`
    before interpolation. The default is True, which will create a
    temporary `float64` array of filtered values if `order > 1`. If
    setting this to False, the output will be slightly blurred if
    `order > 1`, unless the input is prefiltered, i.e. it is the result
    of calling `spline_filter` on the original input.

Returns
-------
zoom : ndarray
    The zoomed input.

Examples
--------
>>> from scipy import ndimage, misc
>>> import matplotlib.pyplot as plt

>>> fig = plt.figure()
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = misc.ascent()
>>> result = ndimage.zoom(ascent, 3.0)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()

>>> print(ascent.shape)
(512, 512)

>>> print(result.shape)
(1536, 1536)
*)

