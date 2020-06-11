(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module Akima1DInterpolator : sig
type tag = [`Akima1DInterpolator]
type t = [`Akima1DInterpolator | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?axis:int -> x:[>`Ndarray] Np.Obj.t -> y:[>`Ndarray] Np.Obj.t -> unit -> t
(**
Akima interpolator

Fit piecewise cubic polynomials, given vectors x and y. The interpolation
method by Akima uses a continuously differentiable sub-spline built from
piecewise cubic polynomials. The resultant curve passes through the given
data points and will appear smooth and natural.

Parameters
----------
x : ndarray, shape (m, )
    1-D array of monotonically increasing real values.
y : ndarray, shape (m, ...)
    N-D array of real values. The length of ``y`` along the first axis
    must be equal to the length of ``x``.
axis : int, optional
    Specifies the axis of ``y`` along which to interpolate. Interpolation
    defaults to the first axis of ``y``.

Methods
-------
__call__
derivative
antiderivative
roots

See Also
--------
PchipInterpolator
CubicSpline
PPoly

Notes
-----
.. versionadded:: 0.14

Use only for precise data, as the fitted curve passes through the given
points exactly. This routine is useful for plotting a pleasingly smooth
curve through a few given points for purposes of plotting.

References
----------
[1] A new method of interpolation and smooth curve fitting based
    on local procedures. Hiroshi Akima, J. ACM, October 1970, 17(4),
    589-602.
*)

val antiderivative : ?nu:int -> [> tag] Obj.t -> Py.Object.t
(**
Construct a new piecewise polynomial representing the antiderivative.

Antiderivative is also the indefinite integral of the function,
and derivative is its inverse operation.

Parameters
----------
nu : int, optional
    Order of antiderivative to evaluate. Default is 1, i.e. compute
    the first integral. If negative, the derivative is returned.

Returns
-------
pp : PPoly
    Piecewise polynomial of order k2 = k + n representing
    the antiderivative of this polynomial.

Notes
-----
The antiderivative returned by this function is continuous and
continuously differentiable to order n-1, up to floating point
rounding error.

If antiderivative is computed and ``self.extrapolate='periodic'``,
it will be set to False for the returned instance. This is done because
the antiderivative is no longer periodic and its correct evaluation
outside of the initially given x interval is difficult.
*)

val construct_fast : ?extrapolate:Py.Object.t -> ?axis:Py.Object.t -> c:Py.Object.t -> x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Construct the piecewise polynomial without making checks.

Takes the same parameters as the constructor. Input arguments
``c`` and ``x`` must be arrays of the correct shape and type.  The
``c`` array can only be of dtypes float and complex, and ``x``
array must have dtype float.
*)

val derivative : ?nu:int -> [> tag] Obj.t -> Py.Object.t
(**
Construct a new piecewise polynomial representing the derivative.

Parameters
----------
nu : int, optional
    Order of derivative to evaluate. Default is 1, i.e. compute the
    first derivative. If negative, the antiderivative is returned.

Returns
-------
pp : PPoly
    Piecewise polynomial of order k2 = k - n representing the derivative
    of this polynomial.

Notes
-----
Derivatives are evaluated piecewise for each polynomial
segment, even if the polynomial is not differentiable at the
breakpoints. The polynomial intervals are considered half-open,
``[a, b)``, except for the last interval which is closed
``[a, b]``.
*)

val extend : ?right:Py.Object.t -> c:Py.Object.t -> x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Add additional breakpoints and coefficients to the polynomial.

Parameters
----------
c : ndarray, size (k, m, ...)
    Additional coefficients for polynomials in intervals. Note that
    the first additional interval will be formed using one of the
    ``self.x`` end points.
x : ndarray, size (m,)
    Additional breakpoints. Must be sorted in the same order as
    ``self.x`` and either to the right or to the left of the current
    breakpoints.
right
    Deprecated argument. Has no effect.

    .. deprecated:: 0.19
*)

val from_bernstein_basis : ?extrapolate:Py.Object.t -> bp:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Construct a piecewise polynomial in the power basis
from a polynomial in Bernstein basis.

Parameters
----------
bp : BPoly
    A Bernstein basis polynomial, as created by BPoly
extrapolate : bool or 'periodic', optional
    If bool, determines whether to extrapolate to out-of-bounds points
    based on first and last intervals, or to return NaNs.
    If 'periodic', periodic extrapolation is used. Default is True.
*)

val from_spline : ?extrapolate:Py.Object.t -> tck:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Construct a piecewise polynomial from a spline

Parameters
----------
tck
    A spline, as returned by `splrep` or a BSpline object.
extrapolate : bool or 'periodic', optional
    If bool, determines whether to extrapolate to out-of-bounds points
    based on first and last intervals, or to return NaNs.
    If 'periodic', periodic extrapolation is used. Default is True.
*)

val integrate : ?extrapolate:[`Bool of bool | `Periodic] -> a:float -> b:float -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute a definite integral over a piecewise polynomial.

Parameters
----------
a : float
    Lower integration bound
b : float
    Upper integration bound
extrapolate : {bool, 'periodic', None}, optional
    If bool, determines whether to extrapolate to out-of-bounds points
    based on first and last intervals, or to return NaNs.
    If 'periodic', periodic extrapolation is used.
    If None (default), use `self.extrapolate`.

Returns
-------
ig : array_like
    Definite integral of the piecewise polynomial over [a, b]
*)

val roots : ?discontinuity:bool -> ?extrapolate:[`Bool of bool | `Periodic] -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Find real roots of the the piecewise polynomial.

Parameters
----------
discontinuity : bool, optional
    Whether to report sign changes across discontinuities at
    breakpoints as roots.
extrapolate : {bool, 'periodic', None}, optional
    If bool, determines whether to return roots from the polynomial
    extrapolated based on first and last intervals, 'periodic' works
    the same as False. If None (default), use `self.extrapolate`.

Returns
-------
roots : ndarray
    Roots of the polynomial(s).

    If the PPoly object describes multiple polynomials, the
    return value is an object array whose each element is an
    ndarray containing the roots.

See Also
--------
PPoly.solve
*)

val solve : ?y:float -> ?discontinuity:bool -> ?extrapolate:[`Bool of bool | `Periodic] -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Find real solutions of the the equation ``pp(x) == y``.

Parameters
----------
y : float, optional
    Right-hand side. Default is zero.
discontinuity : bool, optional
    Whether to report sign changes across discontinuities at
    breakpoints as roots.
extrapolate : {bool, 'periodic', None}, optional
    If bool, determines whether to return roots from the polynomial
    extrapolated based on first and last intervals, 'periodic' works
    the same as False. If None (default), use `self.extrapolate`.

Returns
-------
roots : ndarray
    Roots of the polynomial(s).

    If the PPoly object describes multiple polynomials, the
    return value is an object array whose each element is an
    ndarray containing the roots.

Notes
-----
This routine works only on real-valued polynomials.

If the piecewise polynomial contains sections that are
identically zero, the root list will contain the start point
of the corresponding interval, followed by a ``nan`` value.

If the polynomial is discontinuous across a breakpoint, and
there is a sign change across the breakpoint, this is reported
if the `discont` parameter is True.

Examples
--------

Finding roots of ``[x**2 - 1, (x - 1)**2]`` defined on intervals
``[-2, 1], [1, 2]``:

>>> from scipy.interpolate import PPoly
>>> pp = PPoly(np.array([[1, -4, 3], [1, 0, 0]]).T, [-2, 1, 2])
>>> pp.solve()
array([-1.,  1.])
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module BPoly : sig
type tag = [`BPoly]
type t = [`BPoly | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?extrapolate:bool -> ?axis:int -> c:[>`Ndarray] Np.Obj.t -> x:[>`Ndarray] Np.Obj.t -> unit -> t
(**
Piecewise polynomial in terms of coefficients and breakpoints.

The polynomial between ``x[i]`` and ``x[i + 1]`` is written in the
Bernstein polynomial basis::

    S = sum(c[a, i] * b(a, k; x) for a in range(k+1)),

where ``k`` is the degree of the polynomial, and::

    b(a, k; x) = binom(k, a) * t**a * (1 - t)**(k - a),

with ``t = (x - x[i]) / (x[i+1] - x[i])`` and ``binom`` is the binomial
coefficient.

Parameters
----------
c : ndarray, shape (k, m, ...)
    Polynomial coefficients, order `k` and `m` intervals
x : ndarray, shape (m+1,)
    Polynomial breakpoints. Must be sorted in either increasing or
    decreasing order.
extrapolate : bool, optional
    If bool, determines whether to extrapolate to out-of-bounds points
    based on first and last intervals, or to return NaNs. If 'periodic',
    periodic extrapolation is used. Default is True.
axis : int, optional
    Interpolation axis. Default is zero.

Attributes
----------
x : ndarray
    Breakpoints.
c : ndarray
    Coefficients of the polynomials. They are reshaped
    to a 3-dimensional array with the last dimension representing
    the trailing dimensions of the original coefficient array.
axis : int
    Interpolation axis.

Methods
-------
__call__
extend
derivative
antiderivative
integrate
construct_fast
from_power_basis
from_derivatives

See also
--------
PPoly : piecewise polynomials in the power basis

Notes
-----
Properties of Bernstein polynomials are well documented in the literature,
see for example [1]_ [2]_ [3]_.

References
----------
.. [1] https://en.wikipedia.org/wiki/Bernstein_polynomial

.. [2] Kenneth I. Joy, Bernstein polynomials,
   http://www.idav.ucdavis.edu/education/CAGDNotes/Bernstein-Polynomials.pdf

.. [3] E. H. Doha, A. H. Bhrawy, and M. A. Saker, Boundary Value Problems,
       vol 2011, article ID 829546, :doi:`10.1155/2011/829543`.

Examples
--------
>>> from scipy.interpolate import BPoly
>>> x = [0, 1]
>>> c = [[1], [2], [3]]
>>> bp = BPoly(c, x)

This creates a 2nd order polynomial

.. math::

    B(x) = 1 \times b_{0, 2}(x) + 2 \times b_{1, 2}(x) + 3 \times b_{2, 2}(x) \\
         = 1 \times (1-x)^2 + 2 \times 2 x (1 - x) + 3 \times x^2
*)

val antiderivative : ?nu:int -> [> tag] Obj.t -> Py.Object.t
(**
Construct a new piecewise polynomial representing the antiderivative.

Parameters
----------
nu : int, optional
    Order of antiderivative to evaluate. Default is 1, i.e. compute
    the first integral. If negative, the derivative is returned.

Returns
-------
bp : BPoly
    Piecewise polynomial of order k + nu representing the
    antiderivative of this polynomial.

Notes
-----
If antiderivative is computed and ``self.extrapolate='periodic'``,
it will be set to False for the returned instance. This is done because
the antiderivative is no longer periodic and its correct evaluation
outside of the initially given x interval is difficult.
*)

val construct_fast : ?extrapolate:Py.Object.t -> ?axis:Py.Object.t -> c:Py.Object.t -> x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Construct the piecewise polynomial without making checks.

Takes the same parameters as the constructor. Input arguments
``c`` and ``x`` must be arrays of the correct shape and type.  The
``c`` array can only be of dtypes float and complex, and ``x``
array must have dtype float.
*)

val derivative : ?nu:int -> [> tag] Obj.t -> Py.Object.t
(**
Construct a new piecewise polynomial representing the derivative.

Parameters
----------
nu : int, optional
    Order of derivative to evaluate. Default is 1, i.e. compute the
    first derivative. If negative, the antiderivative is returned.

Returns
-------
bp : BPoly
    Piecewise polynomial of order k - nu representing the derivative of
    this polynomial.
*)

val extend : ?right:Py.Object.t -> c:[`Ndarray of [>`Ndarray] Np.Obj.t | `Size_k_m_ of Py.Object.t] -> x:[`Ndarray of [>`Ndarray] Np.Obj.t | `Size of Py.Object.t] -> [> tag] Obj.t -> Py.Object.t
(**
Add additional breakpoints and coefficients to the polynomial.

Parameters
----------
c : ndarray, size (k, m, ...)
    Additional coefficients for polynomials in intervals. Note that
    the first additional interval will be formed using one of the
    ``self.x`` end points.
x : ndarray, size (m,)
    Additional breakpoints. Must be sorted in the same order as
    ``self.x`` and either to the right or to the left of the current
    breakpoints.
right
    Deprecated argument. Has no effect.

    .. deprecated:: 0.19
*)

val from_derivatives : ?orders:[`I of int | `Array_like_of_ints of Py.Object.t] -> ?extrapolate:[`Bool of bool | `Periodic] -> xi:[>`Ndarray] Np.Obj.t -> yi:[`Ndarray of [>`Ndarray] Np.Obj.t | `List_of_array_likes of Py.Object.t] -> [> tag] Obj.t -> Py.Object.t
(**
Construct a piecewise polynomial in the Bernstein basis,
compatible with the specified values and derivatives at breakpoints.

Parameters
----------
xi : array_like
    sorted 1D array of x-coordinates
yi : array_like or list of array_likes
    ``yi[i][j]`` is the ``j``-th derivative known at ``xi[i]``
orders : None or int or array_like of ints. Default: None.
    Specifies the degree of local polynomials. If not None, some
    derivatives are ignored.
extrapolate : bool or 'periodic', optional
    If bool, determines whether to extrapolate to out-of-bounds points
    based on first and last intervals, or to return NaNs.
    If 'periodic', periodic extrapolation is used. Default is True.

Notes
-----
If ``k`` derivatives are specified at a breakpoint ``x``, the
constructed polynomial is exactly ``k`` times continuously
differentiable at ``x``, unless the ``order`` is provided explicitly.
In the latter case, the smoothness of the polynomial at
the breakpoint is controlled by the ``order``.

Deduces the number of derivatives to match at each end
from ``order`` and the number of derivatives available. If
possible it uses the same number of derivatives from
each end; if the number is odd it tries to take the
extra one from y2. In any case if not enough derivatives
are available at one end or another it draws enough to
make up the total from the other end.

If the order is too high and not enough derivatives are available,
an exception is raised.

Examples
--------

>>> from scipy.interpolate import BPoly
>>> BPoly.from_derivatives([0, 1], [[1, 2], [3, 4]])

Creates a polynomial `f(x)` of degree 3, defined on `[0, 1]`
such that `f(0) = 1, df/dx(0) = 2, f(1) = 3, df/dx(1) = 4`

>>> BPoly.from_derivatives([0, 1, 2], [[0, 1], [0], [2]])

Creates a piecewise polynomial `f(x)`, such that
`f(0) = f(1) = 0`, `f(2) = 2`, and `df/dx(0) = 1`.
Based on the number of derivatives provided, the order of the
local polynomials is 2 on `[0, 1]` and 1 on `[1, 2]`.
Notice that no restriction is imposed on the derivatives at
``x = 1`` and ``x = 2``.

Indeed, the explicit form of the polynomial is::

    f(x) = | x * (1 - x),  0 <= x < 1
           | 2 * (x - 1),  1 <= x <= 2

So that f'(1-0) = -1 and f'(1+0) = 2
*)

val from_power_basis : ?extrapolate:[`Bool of bool | `Periodic] -> pp:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Construct a piecewise polynomial in Bernstein basis
from a power basis polynomial.

Parameters
----------
pp : PPoly
    A piecewise polynomial in the power basis
extrapolate : bool or 'periodic', optional
    If bool, determines whether to extrapolate to out-of-bounds points
    based on first and last intervals, or to return NaNs.
    If 'periodic', periodic extrapolation is used. Default is True.
*)

val integrate : ?extrapolate:[`Bool of bool | `Periodic] -> a:float -> b:float -> [> tag] Obj.t -> Py.Object.t
(**
Compute a definite integral over a piecewise polynomial.

Parameters
----------
a : float
    Lower integration bound
b : float
    Upper integration bound
extrapolate : {bool, 'periodic', None}, optional
    Whether to extrapolate to out-of-bounds points based on first
    and last intervals, or to return NaNs. If 'periodic', periodic
    extrapolation is used. If None (default), use `self.extrapolate`.

Returns
-------
array_like
    Definite integral of the piecewise polynomial over [a, b]
*)


(** Attribute x: get value or raise Not_found if None.*)
val x : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute x: get value as an option. *)
val x_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute c: get value or raise Not_found if None.*)
val c : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute c: get value as an option. *)
val c_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute axis: get value or raise Not_found if None.*)
val axis : t -> int

(** Attribute axis: get value as an option. *)
val axis_opt : t -> (int) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module BSpline : sig
type tag = [`BSpline]
type t = [`BSpline | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?extrapolate:[`Bool of bool | `Periodic] -> ?axis:int -> t:[>`Ndarray] Np.Obj.t -> c:[>`Ndarray] Np.Obj.t -> k:int -> unit -> t
(**
Univariate spline in the B-spline basis.

.. math::

    S(x) = \sum_{j=0}^{n-1} c_j  B_{j, k; t}(x)

where :math:`B_{j, k; t}` are B-spline basis functions of degree `k`
and knots `t`.

Parameters
----------
t : ndarray, shape (n+k+1,)
    knots
c : ndarray, shape (>=n, ...)
    spline coefficients
k : int
    B-spline order
extrapolate : bool or 'periodic', optional
    whether to extrapolate beyond the base interval, ``t[k] .. t[n]``,
    or to return nans.
    If True, extrapolates the first and last polynomial pieces of b-spline
    functions active on the base interval.
    If 'periodic', periodic extrapolation is used.
    Default is True.
axis : int, optional
    Interpolation axis. Default is zero.

Attributes
----------
t : ndarray
    knot vector
c : ndarray
    spline coefficients
k : int
    spline degree
extrapolate : bool
    If True, extrapolates the first and last polynomial pieces of b-spline
    functions active on the base interval.
axis : int
    Interpolation axis.
tck : tuple
    A read-only equivalent of ``(self.t, self.c, self.k)``

Methods
-------
__call__
basis_element
derivative
antiderivative
integrate
construct_fast

Notes
-----
B-spline basis elements are defined via

.. math::

    B_{i, 0}(x) = 1, \textrm{if $t_i \le x < t_{i+1}$, otherwise $0$,}

    B_{i, k}(x) = \frac{x - t_i}{t_{i+k} - t_i} B_{i, k-1}(x)
             + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B_{i+1, k-1}(x)

**Implementation details**

- At least ``k+1`` coefficients are required for a spline of degree `k`,
  so that ``n >= k+1``. Additional coefficients, ``c[j]`` with
  ``j > n``, are ignored.

- B-spline basis elements of degree `k` form a partition of unity on the
  *base interval*, ``t[k] <= x <= t[n]``.


Examples
--------

Translating the recursive definition of B-splines into Python code, we have:

>>> def B(x, k, i, t):
...    if k == 0:
...       return 1.0 if t[i] <= x < t[i+1] else 0.0
...    if t[i+k] == t[i]:
...       c1 = 0.0
...    else:
...       c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
...    if t[i+k+1] == t[i+1]:
...       c2 = 0.0
...    else:
...       c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
...    return c1 + c2

>>> def bspline(x, t, c, k):
...    n = len(t) - k - 1
...    assert (n >= k+1) and (len(c) >= n)
...    return sum(c[i] * B(x, k, i, t) for i in range(n))

Note that this is an inefficient (if straightforward) way to
evaluate B-splines --- this spline class does it in an equivalent,
but much more efficient way.

Here we construct a quadratic spline function on the base interval
``2 <= x <= 4`` and compare with the naive way of evaluating the spline:

>>> from scipy.interpolate import BSpline
>>> k = 2
>>> t = [0, 1, 2, 3, 4, 5, 6]
>>> c = [-1, 2, 0, -1]
>>> spl = BSpline(t, c, k)
>>> spl(2.5)
array(1.375)
>>> bspline(2.5, t, c, k)
1.375

Note that outside of the base interval results differ. This is because
`BSpline` extrapolates the first and last polynomial pieces of b-spline
functions active on the base interval.

>>> import matplotlib.pyplot as plt
>>> fig, ax = plt.subplots()
>>> xx = np.linspace(1.5, 4.5, 50)
>>> ax.plot(xx, [bspline(x, t, c ,k) for x in xx], 'r-', lw=3, label='naive')
>>> ax.plot(xx, spl(xx), 'b-', lw=4, alpha=0.7, label='BSpline')
>>> ax.grid(True)
>>> ax.legend(loc='best')
>>> plt.show()


References
----------
.. [1] Tom Lyche and Knut Morken, Spline methods,
    http://www.uio.no/studier/emner/matnat/ifi/INF-MAT5340/v05/undervisningsmateriale/
.. [2] Carl de Boor, A practical guide to splines, Springer, 2001.
*)

val antiderivative : ?nu:int -> [> tag] Obj.t -> Py.Object.t
(**
Return a b-spline representing the antiderivative.

Parameters
----------
nu : int, optional
    Antiderivative order. Default is 1.

Returns
-------
b : BSpline object
    A new instance representing the antiderivative.

Notes
-----
If antiderivative is computed and ``self.extrapolate='periodic'``,
it will be set to False for the returned instance. This is done because
the antiderivative is no longer periodic and its correct evaluation
outside of the initially given x interval is difficult.

See Also
--------
splder, splantider
*)

val basis_element : ?extrapolate:[`Bool of bool | `Periodic] -> t:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Return a B-spline basis element ``B(x | t[0], ..., t[k+1])``.

Parameters
----------
t : ndarray, shape (k+1,)
    internal knots
extrapolate : bool or 'periodic', optional
    whether to extrapolate beyond the base interval, ``t[0] .. t[k+1]``,
    or to return nans.
    If 'periodic', periodic extrapolation is used.
    Default is True.

Returns
-------
basis_element : callable
    A callable representing a B-spline basis element for the knot
    vector `t`.

Notes
-----
The order of the b-spline, `k`, is inferred from the length of `t` as
``len(t)-2``. The knot vector is constructed by appending and prepending
``k+1`` elements to internal knots `t`.

Examples
--------

Construct a cubic b-spline:

>>> from scipy.interpolate import BSpline
>>> b = BSpline.basis_element([0, 1, 2, 3, 4])
>>> k = b.k
>>> b.t[k:-k]
array([ 0.,  1.,  2.,  3.,  4.])
>>> k
3

Construct a second order b-spline on ``[0, 1, 1, 2]``, and compare
to its explicit form:

>>> t = [-1, 0, 1, 1, 2]
>>> b = BSpline.basis_element(t[1:])
>>> def f(x):
...     return np.where(x < 1, x*x, (2. - x)**2)

>>> import matplotlib.pyplot as plt
>>> fig, ax = plt.subplots()
>>> x = np.linspace(0, 2, 51)
>>> ax.plot(x, b(x), 'g', lw=3)
>>> ax.plot(x, f(x), 'r', lw=8, alpha=0.4)
>>> ax.grid(True)
>>> plt.show()
*)

val construct_fast : ?extrapolate:Py.Object.t -> ?axis:Py.Object.t -> t:Py.Object.t -> c:Py.Object.t -> k:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Construct a spline without making checks.

Accepts same parameters as the regular constructor. Input arrays
`t` and `c` must of correct shape and dtype.
*)

val derivative : ?nu:int -> [> tag] Obj.t -> Py.Object.t
(**
Return a b-spline representing the derivative.

Parameters
----------
nu : int, optional
    Derivative order.
    Default is 1.

Returns
-------
b : BSpline object
    A new instance representing the derivative.

See Also
--------
splder, splantider
*)

val integrate : ?extrapolate:[`Bool of bool | `Periodic] -> a:float -> b:float -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute a definite integral of the spline.

Parameters
----------
a : float
    Lower limit of integration.
b : float
    Upper limit of integration.
extrapolate : bool or 'periodic', optional
    whether to extrapolate beyond the base interval,
    ``t[k] .. t[-k-1]``, or take the spline to be zero outside of the
    base interval. If 'periodic', periodic extrapolation is used.
    If None (default), use `self.extrapolate`.

Returns
-------
I : array_like
    Definite integral of the spline over the interval ``[a, b]``.

Examples
--------
Construct the linear spline ``x if x < 1 else 2 - x`` on the base
interval :math:`[0, 2]`, and integrate it

>>> from scipy.interpolate import BSpline
>>> b = BSpline.basis_element([0, 1, 2])
>>> b.integrate(0, 1)
array(0.5)

If the integration limits are outside of the base interval, the result
is controlled by the `extrapolate` parameter

>>> b.integrate(-1, 1)
array(0.0)
>>> b.integrate(-1, 1, extrapolate=False)
array(0.5)

>>> import matplotlib.pyplot as plt
>>> fig, ax = plt.subplots()
>>> ax.grid(True)
>>> ax.axvline(0, c='r', lw=5, alpha=0.5)  # base interval
>>> ax.axvline(2, c='r', lw=5, alpha=0.5)
>>> xx = [-1, 1, 2]
>>> ax.plot(xx, b(xx))
>>> plt.show()
*)


(** Attribute t: get value or raise Not_found if None.*)
val t : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute t: get value as an option. *)
val t_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute c: get value or raise Not_found if None.*)
val c : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute c: get value as an option. *)
val c_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute k: get value or raise Not_found if None.*)
val k : t -> int

(** Attribute k: get value as an option. *)
val k_opt : t -> (int) option


(** Attribute extrapolate: get value or raise Not_found if None.*)
val extrapolate : t -> bool

(** Attribute extrapolate: get value as an option. *)
val extrapolate_opt : t -> (bool) option


(** Attribute axis: get value or raise Not_found if None.*)
val axis : t -> int

(** Attribute axis: get value as an option. *)
val axis_opt : t -> (int) option


(** Attribute tck: get value or raise Not_found if None.*)
val tck : t -> Py.Object.t

(** Attribute tck: get value as an option. *)
val tck_opt : t -> (Py.Object.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module BarycentricInterpolator : sig
type tag = [`BarycentricInterpolator]
type t = [`BarycentricInterpolator | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?yi:[>`Ndarray] Np.Obj.t -> ?axis:int -> xi:[>`Ndarray] Np.Obj.t -> unit -> t
(**
The interpolating polynomial for a set of points

Constructs a polynomial that passes through a given set of points.
Allows evaluation of the polynomial, efficient changing of the y
values to be interpolated, and updating by adding more x values.
For reasons of numerical stability, this function does not compute
the coefficients of the polynomial.

The values yi need to be provided before the function is
evaluated, but none of the preprocessing depends on them, so rapid
updates are possible.

Parameters
----------
xi : array_like
    1-d array of x coordinates of the points the polynomial
    should pass through
yi : array_like, optional
    The y coordinates of the points the polynomial should pass through.
    If None, the y values will be supplied later via the `set_y` method.
axis : int, optional
    Axis in the yi array corresponding to the x-coordinate values.

Notes
-----
This class uses a 'barycentric interpolation' method that treats
the problem as a special case of rational function interpolation.
This algorithm is quite stable, numerically, but even in a world of
exact computation, unless the x coordinates are chosen very
carefully - Chebyshev zeros (e.g. cos(i*pi/n)) are a good choice -
polynomial interpolation itself is a very ill-conditioned process
due to the Runge phenomenon.

Based on Berrut and Trefethen 2004, 'Barycentric Lagrange Interpolation'.
*)

val add_xi : ?yi:[>`Ndarray] Np.Obj.t -> xi:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Add more x values to the set to be interpolated

The barycentric interpolation algorithm allows easy updating by
adding more points for the polynomial to pass through.

Parameters
----------
xi : array_like
    The x coordinates of the points that the polynomial should pass
    through.
yi : array_like, optional
    The y coordinates of the points the polynomial should pass through.
    Should have shape ``(xi.size, R)``; if R > 1 then the polynomial is
    vector-valued.
    If `yi` is not given, the y values will be supplied later. `yi` should
    be given if and only if the interpolator has y values specified.
*)

val set_yi : ?axis:int -> yi:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Update the y values to be interpolated

The barycentric interpolation algorithm requires the calculation
of weights, but these depend only on the xi. The yi can be changed
at any time.

Parameters
----------
yi : array_like
    The y coordinates of the points the polynomial should pass through.
    If None, the y values will be supplied later.
axis : int, optional
    Axis in the yi array corresponding to the x-coordinate values.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module BivariateSpline : sig
type tag = [`BivariateSpline]
type t = [`BivariateSpline | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : unit -> t
(**
Base class for bivariate splines.

This describes a spline ``s(x, y)`` of degrees ``kx`` and ``ky`` on
the rectangle ``[xb, xe] * [yb, ye]`` calculated from a given set
of data points ``(x, y, z)``.

This class is meant to be subclassed, not instantiated directly.
To construct these splines, call either `SmoothBivariateSpline` or
`LSQBivariateSpline`.

See Also
--------
UnivariateSpline :
    a similar class for univariate spline interpolation
SmoothBivariateSpline :
    to create a BivariateSpline through the given points
LSQBivariateSpline :
    to create a BivariateSpline using weighted least-squares fitting
RectSphereBivariateSpline
SmoothSphereBivariateSpline :
LSQSphereBivariateSpline
bisplrep : older wrapping of FITPACK
bisplev : older wrapping of FITPACK
*)

val ev : ?dx:int -> ?dy:int -> xi:Py.Object.t -> yi:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Evaluate the spline at points

Returns the interpolated value at ``(xi[i], yi[i]),
i=0,...,len(xi)-1``.

Parameters
----------
xi, yi : array_like
    Input coordinates. Standard Numpy broadcasting is obeyed.
dx : int, optional
    Order of x-derivative

    .. versionadded:: 0.14.0
dy : int, optional
    Order of y-derivative

    .. versionadded:: 0.14.0
*)

val get_coeffs : [> tag] Obj.t -> Py.Object.t
(**
Return spline coefficients.
*)

val get_knots : [> tag] Obj.t -> Py.Object.t
(**
Return a tuple (tx,ty) where tx,ty contain knots positions
of the spline with respect to x-, y-variable, respectively.
The position of interior and additional knots are given as
t[k+1:-k-1] and t[:k+1]=b, t[-k-1:]=e, respectively.
*)

val get_residual : [> tag] Obj.t -> Py.Object.t
(**
Return weighted sum of squared residuals of the spline
approximation: sum ((w[i]*(z[i]-s(x[i],y[i])))**2,axis=0)
*)

val integral : xa:Py.Object.t -> xb:Py.Object.t -> ya:Py.Object.t -> yb:Py.Object.t -> [> tag] Obj.t -> float
(**
Evaluate the integral of the spline over area [xa,xb] x [ya,yb].

Parameters
----------
xa, xb : float
    The end-points of the x integration interval.
ya, yb : float
    The end-points of the y integration interval.

Returns
-------
integ : float
    The value of the resulting integral.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module CloughTocher2DInterpolator : sig
type tag = [`CloughTocher2DInterpolator]
type t = [`CloughTocher2DInterpolator | `NDInterpolatorBase | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_nd_interpolator : t -> [`NDInterpolatorBase] Obj.t
val create : ?fill_value:float -> ?tol:float -> ?maxiter:int -> ?rescale:bool -> points:[`Ndarray of [>`Ndarray] Np.Obj.t | `Delaunay of Py.Object.t] -> values:Py.Object.t -> unit -> t
(**
CloughTocher2DInterpolator(points, values, tol=1e-6)

Piecewise cubic, C1 smooth, curvature-minimizing interpolant in 2D.

.. versionadded:: 0.9

Methods
-------
__call__

Parameters
----------
points : ndarray of floats, shape (npoints, ndims); or Delaunay
    Data point coordinates, or a precomputed Delaunay triangulation.
values : ndarray of float or complex, shape (npoints, ...)
    Data values.
fill_value : float, optional
    Value used to fill in for requested points outside of the
    convex hull of the input points.  If not provided, then
    the default is ``nan``.
tol : float, optional
    Absolute/relative tolerance for gradient estimation.
maxiter : int, optional
    Maximum number of iterations in gradient estimation.
rescale : bool, optional
    Rescale points to unit cube before performing interpolation.
    This is useful if some of the input dimensions have
    incommensurable units and differ by many orders of magnitude.

Notes
-----
The interpolant is constructed by triangulating the input data
with Qhull [1]_, and constructing a piecewise cubic
interpolating Bezier polynomial on each triangle, using a
Clough-Tocher scheme [CT]_.  The interpolant is guaranteed to be
continuously differentiable.

The gradients of the interpolant are chosen so that the curvature
of the interpolating surface is approximatively minimized. The
gradients necessary for this are estimated using the global
algorithm described in [Nielson83,Renka84]_.

References
----------
.. [1] http://www.qhull.org/

.. [CT] See, for example,
   P. Alfeld,
   ''A trivariate Clough-Tocher scheme for tetrahedral data''.
   Computer Aided Geometric Design, 1, 169 (1984);
   G. Farin,
   ''Triangular Bernstein-Bezier patches''.
   Computer Aided Geometric Design, 3, 83 (1986).

.. [Nielson83] G. Nielson,
   ''A method for interpolating scattered data based upon a minimum norm
   network''.
   Math. Comp., 40, 253 (1983).

.. [Renka84] R. J. Renka and A. K. Cline.
   ''A Triangle-based C1 interpolation method.'',
   Rocky Mountain J. Math., 14, 223 (1984).
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module CubicHermiteSpline : sig
type tag = [`CubicHermiteSpline]
type t = [`CubicHermiteSpline | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?axis:int -> ?extrapolate:[`Bool of bool | `Periodic] -> x:[>`Ndarray] Np.Obj.t -> y:[>`Ndarray] Np.Obj.t -> dydx:[>`Ndarray] Np.Obj.t -> unit -> t
(**
Piecewise-cubic interpolator matching values and first derivatives.

The result is represented as a `PPoly` instance.

Parameters
----------
x : array_like, shape (n,)
    1-d array containing values of the independent variable.
    Values must be real, finite and in strictly increasing order.
y : array_like
    Array containing values of the dependent variable. It can have
    arbitrary number of dimensions, but the length along ``axis``
    (see below) must match the length of ``x``. Values must be finite.
dydx : array_like
    Array containing derivatives of the dependent variable. It can have
    arbitrary number of dimensions, but the length along ``axis``
    (see below) must match the length of ``x``. Values must be finite.
axis : int, optional
    Axis along which `y` is assumed to be varying. Meaning that for
    ``x[i]`` the corresponding values are ``np.take(y, i, axis=axis)``.
    Default is 0.
extrapolate : {bool, 'periodic', None}, optional
    If bool, determines whether to extrapolate to out-of-bounds points
    based on first and last intervals, or to return NaNs. If 'periodic',
    periodic extrapolation is used. If None (default), it is set to True.

Attributes
----------
x : ndarray, shape (n,)
    Breakpoints. The same ``x`` which was passed to the constructor.
c : ndarray, shape (4, n-1, ...)
    Coefficients of the polynomials on each segment. The trailing
    dimensions match the dimensions of `y`, excluding ``axis``.
    For example, if `y` is 1-d, then ``c[k, i]`` is a coefficient for
    ``(x-x[i])**(3-k)`` on the segment between ``x[i]`` and ``x[i+1]``.
axis : int
    Interpolation axis. The same axis which was passed to the
    constructor.

Methods
-------
__call__
derivative
antiderivative
integrate
roots

See Also
--------
Akima1DInterpolator
PchipInterpolator
CubicSpline
PPoly

Notes
-----
If you want to create a higher-order spline matching higher-order
derivatives, use `BPoly.from_derivatives`.

References
----------
.. [1] `Cubic Hermite spline
        <https://en.wikipedia.org/wiki/Cubic_Hermite_spline>`_
        on Wikipedia.
*)

val antiderivative : ?nu:int -> [> tag] Obj.t -> Py.Object.t
(**
Construct a new piecewise polynomial representing the antiderivative.

Antiderivative is also the indefinite integral of the function,
and derivative is its inverse operation.

Parameters
----------
nu : int, optional
    Order of antiderivative to evaluate. Default is 1, i.e. compute
    the first integral. If negative, the derivative is returned.

Returns
-------
pp : PPoly
    Piecewise polynomial of order k2 = k + n representing
    the antiderivative of this polynomial.

Notes
-----
The antiderivative returned by this function is continuous and
continuously differentiable to order n-1, up to floating point
rounding error.

If antiderivative is computed and ``self.extrapolate='periodic'``,
it will be set to False for the returned instance. This is done because
the antiderivative is no longer periodic and its correct evaluation
outside of the initially given x interval is difficult.
*)

val construct_fast : ?extrapolate:Py.Object.t -> ?axis:Py.Object.t -> c:Py.Object.t -> x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Construct the piecewise polynomial without making checks.

Takes the same parameters as the constructor. Input arguments
``c`` and ``x`` must be arrays of the correct shape and type.  The
``c`` array can only be of dtypes float and complex, and ``x``
array must have dtype float.
*)

val derivative : ?nu:int -> [> tag] Obj.t -> Py.Object.t
(**
Construct a new piecewise polynomial representing the derivative.

Parameters
----------
nu : int, optional
    Order of derivative to evaluate. Default is 1, i.e. compute the
    first derivative. If negative, the antiderivative is returned.

Returns
-------
pp : PPoly
    Piecewise polynomial of order k2 = k - n representing the derivative
    of this polynomial.

Notes
-----
Derivatives are evaluated piecewise for each polynomial
segment, even if the polynomial is not differentiable at the
breakpoints. The polynomial intervals are considered half-open,
``[a, b)``, except for the last interval which is closed
``[a, b]``.
*)

val extend : ?right:Py.Object.t -> c:[`Ndarray of [>`Ndarray] Np.Obj.t | `Size_k_m_ of Py.Object.t] -> x:[`Ndarray of [>`Ndarray] Np.Obj.t | `Size of Py.Object.t] -> [> tag] Obj.t -> Py.Object.t
(**
Add additional breakpoints and coefficients to the polynomial.

Parameters
----------
c : ndarray, size (k, m, ...)
    Additional coefficients for polynomials in intervals. Note that
    the first additional interval will be formed using one of the
    ``self.x`` end points.
x : ndarray, size (m,)
    Additional breakpoints. Must be sorted in the same order as
    ``self.x`` and either to the right or to the left of the current
    breakpoints.
right
    Deprecated argument. Has no effect.

    .. deprecated:: 0.19
*)

val from_bernstein_basis : ?extrapolate:[`Bool of bool | `Periodic] -> bp:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Construct a piecewise polynomial in the power basis
from a polynomial in Bernstein basis.

Parameters
----------
bp : BPoly
    A Bernstein basis polynomial, as created by BPoly
extrapolate : bool or 'periodic', optional
    If bool, determines whether to extrapolate to out-of-bounds points
    based on first and last intervals, or to return NaNs.
    If 'periodic', periodic extrapolation is used. Default is True.
*)

val from_spline : ?extrapolate:[`Bool of bool | `Periodic] -> tck:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Construct a piecewise polynomial from a spline

Parameters
----------
tck
    A spline, as returned by `splrep` or a BSpline object.
extrapolate : bool or 'periodic', optional
    If bool, determines whether to extrapolate to out-of-bounds points
    based on first and last intervals, or to return NaNs.
    If 'periodic', periodic extrapolation is used. Default is True.
*)

val integrate : ?extrapolate:[`Bool of bool | `Periodic] -> a:float -> b:float -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute a definite integral over a piecewise polynomial.

Parameters
----------
a : float
    Lower integration bound
b : float
    Upper integration bound
extrapolate : {bool, 'periodic', None}, optional
    If bool, determines whether to extrapolate to out-of-bounds points
    based on first and last intervals, or to return NaNs.
    If 'periodic', periodic extrapolation is used.
    If None (default), use `self.extrapolate`.

Returns
-------
ig : array_like
    Definite integral of the piecewise polynomial over [a, b]
*)

val roots : ?discontinuity:bool -> ?extrapolate:[`Bool of bool | `Periodic] -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Find real roots of the the piecewise polynomial.

Parameters
----------
discontinuity : bool, optional
    Whether to report sign changes across discontinuities at
    breakpoints as roots.
extrapolate : {bool, 'periodic', None}, optional
    If bool, determines whether to return roots from the polynomial
    extrapolated based on first and last intervals, 'periodic' works
    the same as False. If None (default), use `self.extrapolate`.

Returns
-------
roots : ndarray
    Roots of the polynomial(s).

    If the PPoly object describes multiple polynomials, the
    return value is an object array whose each element is an
    ndarray containing the roots.

See Also
--------
PPoly.solve
*)

val solve : ?y:float -> ?discontinuity:bool -> ?extrapolate:[`Bool of bool | `Periodic] -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Find real solutions of the the equation ``pp(x) == y``.

Parameters
----------
y : float, optional
    Right-hand side. Default is zero.
discontinuity : bool, optional
    Whether to report sign changes across discontinuities at
    breakpoints as roots.
extrapolate : {bool, 'periodic', None}, optional
    If bool, determines whether to return roots from the polynomial
    extrapolated based on first and last intervals, 'periodic' works
    the same as False. If None (default), use `self.extrapolate`.

Returns
-------
roots : ndarray
    Roots of the polynomial(s).

    If the PPoly object describes multiple polynomials, the
    return value is an object array whose each element is an
    ndarray containing the roots.

Notes
-----
This routine works only on real-valued polynomials.

If the piecewise polynomial contains sections that are
identically zero, the root list will contain the start point
of the corresponding interval, followed by a ``nan`` value.

If the polynomial is discontinuous across a breakpoint, and
there is a sign change across the breakpoint, this is reported
if the `discont` parameter is True.

Examples
--------

Finding roots of ``[x**2 - 1, (x - 1)**2]`` defined on intervals
``[-2, 1], [1, 2]``:

>>> from scipy.interpolate import PPoly
>>> pp = PPoly(np.array([[1, -4, 3], [1, 0, 0]]).T, [-2, 1, 2])
>>> pp.solve()
array([-1.,  1.])
*)


(** Attribute x: get value or raise Not_found if None.*)
val x : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute x: get value as an option. *)
val x_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute c: get value or raise Not_found if None.*)
val c : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute c: get value as an option. *)
val c_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute axis: get value or raise Not_found if None.*)
val axis : t -> int

(** Attribute axis: get value as an option. *)
val axis_opt : t -> (int) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module CubicSpline : sig
type tag = [`CubicSpline]
type t = [`CubicSpline | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?axis:int -> ?bc_type:[`S of string | `T2_tuple of Py.Object.t] -> ?extrapolate:[`Bool of bool | `Periodic] -> x:[>`Ndarray] Np.Obj.t -> y:[>`Ndarray] Np.Obj.t -> unit -> t
(**
Cubic spline data interpolator.

Interpolate data with a piecewise cubic polynomial which is twice
continuously differentiable [1]_. The result is represented as a `PPoly`
instance with breakpoints matching the given data.

Parameters
----------
x : array_like, shape (n,)
    1-d array containing values of the independent variable.
    Values must be real, finite and in strictly increasing order.
y : array_like
    Array containing values of the dependent variable. It can have
    arbitrary number of dimensions, but the length along ``axis``
    (see below) must match the length of ``x``. Values must be finite.
axis : int, optional
    Axis along which `y` is assumed to be varying. Meaning that for
    ``x[i]`` the corresponding values are ``np.take(y, i, axis=axis)``.
    Default is 0.
bc_type : string or 2-tuple, optional
    Boundary condition type. Two additional equations, given by the
    boundary conditions, are required to determine all coefficients of
    polynomials on each segment [2]_.

    If `bc_type` is a string, then the specified condition will be applied
    at both ends of a spline. Available conditions are:

    * 'not-a-knot' (default): The first and second segment at a curve end
      are the same polynomial. It is a good default when there is no
      information on boundary conditions.
    * 'periodic': The interpolated functions is assumed to be periodic
      of period ``x[-1] - x[0]``. The first and last value of `y` must be
      identical: ``y[0] == y[-1]``. This boundary condition will result in
      ``y'[0] == y'[-1]`` and ``y''[0] == y''[-1]``.
    * 'clamped': The first derivative at curves ends are zero. Assuming
      a 1D `y`, ``bc_type=((1, 0.0), (1, 0.0))`` is the same condition.
    * 'natural': The second derivative at curve ends are zero. Assuming
      a 1D `y`, ``bc_type=((2, 0.0), (2, 0.0))`` is the same condition.

    If `bc_type` is a 2-tuple, the first and the second value will be
    applied at the curve start and end respectively. The tuple values can
    be one of the previously mentioned strings (except 'periodic') or a
    tuple `(order, deriv_values)` allowing to specify arbitrary
    derivatives at curve ends:

    * `order`: the derivative order, 1 or 2.
    * `deriv_value`: array_like containing derivative values, shape must
      be the same as `y`, excluding ``axis`` dimension. For example, if
      `y` is 1D, then `deriv_value` must be a scalar. If `y` is 3D with
      the shape (n0, n1, n2) and axis=2, then `deriv_value` must be 2D
      and have the shape (n0, n1).
extrapolate : {bool, 'periodic', None}, optional
    If bool, determines whether to extrapolate to out-of-bounds points
    based on first and last intervals, or to return NaNs. If 'periodic',
    periodic extrapolation is used. If None (default), ``extrapolate`` is
    set to 'periodic' for ``bc_type='periodic'`` and to True otherwise.

Attributes
----------
x : ndarray, shape (n,)
    Breakpoints. The same ``x`` which was passed to the constructor.
c : ndarray, shape (4, n-1, ...)
    Coefficients of the polynomials on each segment. The trailing
    dimensions match the dimensions of `y`, excluding ``axis``.
    For example, if `y` is 1-d, then ``c[k, i]`` is a coefficient for
    ``(x-x[i])**(3-k)`` on the segment between ``x[i]`` and ``x[i+1]``.
axis : int
    Interpolation axis. The same axis which was passed to the
    constructor.

Methods
-------
__call__
derivative
antiderivative
integrate
roots

See Also
--------
Akima1DInterpolator
PchipInterpolator
PPoly

Notes
-----
Parameters `bc_type` and ``interpolate`` work independently, i.e. the
former controls only construction of a spline, and the latter only
evaluation.

When a boundary condition is 'not-a-knot' and n = 2, it is replaced by
a condition that the first derivative is equal to the linear interpolant
slope. When both boundary conditions are 'not-a-knot' and n = 3, the
solution is sought as a parabola passing through given points.

When 'not-a-knot' boundary conditions is applied to both ends, the
resulting spline will be the same as returned by `splrep` (with ``s=0``)
and `InterpolatedUnivariateSpline`, but these two methods use a
representation in B-spline basis.

.. versionadded:: 0.18.0

Examples
--------
In this example the cubic spline is used to interpolate a sampled sinusoid.
You can see that the spline continuity property holds for the first and
second derivatives and violates only for the third derivative.

>>> from scipy.interpolate import CubicSpline
>>> import matplotlib.pyplot as plt
>>> x = np.arange(10)
>>> y = np.sin(x)
>>> cs = CubicSpline(x, y)
>>> xs = np.arange(-0.5, 9.6, 0.1)
>>> fig, ax = plt.subplots(figsize=(6.5, 4))
>>> ax.plot(x, y, 'o', label='data')
>>> ax.plot(xs, np.sin(xs), label='true')
>>> ax.plot(xs, cs(xs), label='S')
>>> ax.plot(xs, cs(xs, 1), label='S'')
>>> ax.plot(xs, cs(xs, 2), label='S''')
>>> ax.plot(xs, cs(xs, 3), label='S'''')
>>> ax.set_xlim(-0.5, 9.5)
>>> ax.legend(loc='lower left', ncol=2)
>>> plt.show()

In the second example, the unit circle is interpolated with a spline. A
periodic boundary condition is used. You can see that the first derivative
values, ds/dx=0, ds/dy=1 at the periodic point (1, 0) are correctly
computed. Note that a circle cannot be exactly represented by a cubic
spline. To increase precision, more breakpoints would be required.

>>> theta = 2 * np.pi * np.linspace(0, 1, 5)
>>> y = np.c_[np.cos(theta), np.sin(theta)]
>>> cs = CubicSpline(theta, y, bc_type='periodic')
>>> print('ds/dx={:.1f} ds/dy={:.1f}'.format(cs(0, 1)[0], cs(0, 1)[1]))
ds/dx=0.0 ds/dy=1.0
>>> xs = 2 * np.pi * np.linspace(0, 1, 100)
>>> fig, ax = plt.subplots(figsize=(6.5, 4))
>>> ax.plot(y[:, 0], y[:, 1], 'o', label='data')
>>> ax.plot(np.cos(xs), np.sin(xs), label='true')
>>> ax.plot(cs(xs)[:, 0], cs(xs)[:, 1], label='spline')
>>> ax.axes.set_aspect('equal')
>>> ax.legend(loc='center')
>>> plt.show()

The third example is the interpolation of a polynomial y = x**3 on the
interval 0 <= x<= 1. A cubic spline can represent this function exactly.
To achieve that we need to specify values and first derivatives at
endpoints of the interval. Note that y' = 3 * x**2 and thus y'(0) = 0 and
y'(1) = 3.

>>> cs = CubicSpline([0, 1], [0, 1], bc_type=((1, 0), (1, 3)))
>>> x = np.linspace(0, 1)
>>> np.allclose(x**3, cs(x))
True

References
----------
.. [1] `Cubic Spline Interpolation
        <https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation>`_
        on Wikiversity.
.. [2] Carl de Boor, 'A Practical Guide to Splines', Springer-Verlag, 1978.
*)

val antiderivative : ?nu:int -> [> tag] Obj.t -> Py.Object.t
(**
Construct a new piecewise polynomial representing the antiderivative.

Antiderivative is also the indefinite integral of the function,
and derivative is its inverse operation.

Parameters
----------
nu : int, optional
    Order of antiderivative to evaluate. Default is 1, i.e. compute
    the first integral. If negative, the derivative is returned.

Returns
-------
pp : PPoly
    Piecewise polynomial of order k2 = k + n representing
    the antiderivative of this polynomial.

Notes
-----
The antiderivative returned by this function is continuous and
continuously differentiable to order n-1, up to floating point
rounding error.

If antiderivative is computed and ``self.extrapolate='periodic'``,
it will be set to False for the returned instance. This is done because
the antiderivative is no longer periodic and its correct evaluation
outside of the initially given x interval is difficult.
*)

val construct_fast : ?extrapolate:Py.Object.t -> ?axis:Py.Object.t -> c:Py.Object.t -> x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Construct the piecewise polynomial without making checks.

Takes the same parameters as the constructor. Input arguments
``c`` and ``x`` must be arrays of the correct shape and type.  The
``c`` array can only be of dtypes float and complex, and ``x``
array must have dtype float.
*)

val derivative : ?nu:int -> [> tag] Obj.t -> Py.Object.t
(**
Construct a new piecewise polynomial representing the derivative.

Parameters
----------
nu : int, optional
    Order of derivative to evaluate. Default is 1, i.e. compute the
    first derivative. If negative, the antiderivative is returned.

Returns
-------
pp : PPoly
    Piecewise polynomial of order k2 = k - n representing the derivative
    of this polynomial.

Notes
-----
Derivatives are evaluated piecewise for each polynomial
segment, even if the polynomial is not differentiable at the
breakpoints. The polynomial intervals are considered half-open,
``[a, b)``, except for the last interval which is closed
``[a, b]``.
*)

val extend : ?right:Py.Object.t -> c:[`Ndarray of [>`Ndarray] Np.Obj.t | `Size_k_m_ of Py.Object.t] -> x:[`Ndarray of [>`Ndarray] Np.Obj.t | `Size of Py.Object.t] -> [> tag] Obj.t -> Py.Object.t
(**
Add additional breakpoints and coefficients to the polynomial.

Parameters
----------
c : ndarray, size (k, m, ...)
    Additional coefficients for polynomials in intervals. Note that
    the first additional interval will be formed using one of the
    ``self.x`` end points.
x : ndarray, size (m,)
    Additional breakpoints. Must be sorted in the same order as
    ``self.x`` and either to the right or to the left of the current
    breakpoints.
right
    Deprecated argument. Has no effect.

    .. deprecated:: 0.19
*)

val from_bernstein_basis : ?extrapolate:[`Bool of bool | `Periodic] -> bp:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Construct a piecewise polynomial in the power basis
from a polynomial in Bernstein basis.

Parameters
----------
bp : BPoly
    A Bernstein basis polynomial, as created by BPoly
extrapolate : bool or 'periodic', optional
    If bool, determines whether to extrapolate to out-of-bounds points
    based on first and last intervals, or to return NaNs.
    If 'periodic', periodic extrapolation is used. Default is True.
*)

val from_spline : ?extrapolate:[`Bool of bool | `Periodic] -> tck:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Construct a piecewise polynomial from a spline

Parameters
----------
tck
    A spline, as returned by `splrep` or a BSpline object.
extrapolate : bool or 'periodic', optional
    If bool, determines whether to extrapolate to out-of-bounds points
    based on first and last intervals, or to return NaNs.
    If 'periodic', periodic extrapolation is used. Default is True.
*)

val integrate : ?extrapolate:[`Bool of bool | `Periodic] -> a:float -> b:float -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute a definite integral over a piecewise polynomial.

Parameters
----------
a : float
    Lower integration bound
b : float
    Upper integration bound
extrapolate : {bool, 'periodic', None}, optional
    If bool, determines whether to extrapolate to out-of-bounds points
    based on first and last intervals, or to return NaNs.
    If 'periodic', periodic extrapolation is used.
    If None (default), use `self.extrapolate`.

Returns
-------
ig : array_like
    Definite integral of the piecewise polynomial over [a, b]
*)

val roots : ?discontinuity:bool -> ?extrapolate:[`Bool of bool | `Periodic] -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Find real roots of the the piecewise polynomial.

Parameters
----------
discontinuity : bool, optional
    Whether to report sign changes across discontinuities at
    breakpoints as roots.
extrapolate : {bool, 'periodic', None}, optional
    If bool, determines whether to return roots from the polynomial
    extrapolated based on first and last intervals, 'periodic' works
    the same as False. If None (default), use `self.extrapolate`.

Returns
-------
roots : ndarray
    Roots of the polynomial(s).

    If the PPoly object describes multiple polynomials, the
    return value is an object array whose each element is an
    ndarray containing the roots.

See Also
--------
PPoly.solve
*)

val solve : ?y:float -> ?discontinuity:bool -> ?extrapolate:[`Bool of bool | `Periodic] -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Find real solutions of the the equation ``pp(x) == y``.

Parameters
----------
y : float, optional
    Right-hand side. Default is zero.
discontinuity : bool, optional
    Whether to report sign changes across discontinuities at
    breakpoints as roots.
extrapolate : {bool, 'periodic', None}, optional
    If bool, determines whether to return roots from the polynomial
    extrapolated based on first and last intervals, 'periodic' works
    the same as False. If None (default), use `self.extrapolate`.

Returns
-------
roots : ndarray
    Roots of the polynomial(s).

    If the PPoly object describes multiple polynomials, the
    return value is an object array whose each element is an
    ndarray containing the roots.

Notes
-----
This routine works only on real-valued polynomials.

If the piecewise polynomial contains sections that are
identically zero, the root list will contain the start point
of the corresponding interval, followed by a ``nan`` value.

If the polynomial is discontinuous across a breakpoint, and
there is a sign change across the breakpoint, this is reported
if the `discont` parameter is True.

Examples
--------

Finding roots of ``[x**2 - 1, (x - 1)**2]`` defined on intervals
``[-2, 1], [1, 2]``:

>>> from scipy.interpolate import PPoly
>>> pp = PPoly(np.array([[1, -4, 3], [1, 0, 0]]).T, [-2, 1, 2])
>>> pp.solve()
array([-1.,  1.])
*)


(** Attribute x: get value or raise Not_found if None.*)
val x : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute x: get value as an option. *)
val x_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute c: get value or raise Not_found if None.*)
val c : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute c: get value as an option. *)
val c_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute axis: get value or raise Not_found if None.*)
val axis : t -> int

(** Attribute axis: get value as an option. *)
val axis_opt : t -> (int) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module InterpolatedUnivariateSpline : sig
type tag = [`InterpolatedUnivariateSpline]
type t = [`InterpolatedUnivariateSpline | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?w:[>`Ndarray] Np.Obj.t -> ?bbox:Py.Object.t -> ?k:int -> ?ext:[`S of string | `I of int] -> ?check_finite:bool -> x:[>`Ndarray] Np.Obj.t -> y:[>`Ndarray] Np.Obj.t -> unit -> t
(**
One-dimensional interpolating spline for a given set of data points.

Fits a spline y = spl(x) of degree `k` to the provided `x`, `y` data.
Spline function passes through all provided points. Equivalent to
`UnivariateSpline` with  s=0.

Parameters
----------
x : (N,) array_like
    Input dimension of data points -- must be strictly increasing
y : (N,) array_like
    input dimension of data points
w : (N,) array_like, optional
    Weights for spline fitting.  Must be positive.  If None (default),
    weights are all equal.
bbox : (2,) array_like, optional
    2-sequence specifying the boundary of the approximation interval. If
    None (default), ``bbox=[x[0], x[-1]]``.
k : int, optional
    Degree of the smoothing spline.  Must be 1 <= `k` <= 5.
ext : int or str, optional
    Controls the extrapolation mode for elements
    not in the interval defined by the knot sequence.

    * if ext=0 or 'extrapolate', return the extrapolated value.
    * if ext=1 or 'zeros', return 0
    * if ext=2 or 'raise', raise a ValueError
    * if ext=3 of 'const', return the boundary value.

    The default value is 0.

check_finite : bool, optional
    Whether to check that the input arrays contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination or non-sensical results) if the inputs
    do contain infinities or NaNs.
    Default is False.

See Also
--------
UnivariateSpline : Superclass -- allows knots to be selected by a
    smoothing condition
LSQUnivariateSpline : spline for which knots are user-selected
splrep : An older, non object-oriented wrapping of FITPACK
splev, sproot, splint, spalde
BivariateSpline : A similar class for two-dimensional spline interpolation

Notes
-----
The number of data points must be larger than the spline degree `k`.

Examples
--------
>>> import matplotlib.pyplot as plt
>>> from scipy.interpolate import InterpolatedUnivariateSpline
>>> x = np.linspace(-3, 3, 50)
>>> y = np.exp(-x**2) + 0.1 * np.random.randn(50)
>>> spl = InterpolatedUnivariateSpline(x, y)
>>> plt.plot(x, y, 'ro', ms=5)
>>> xs = np.linspace(-3, 3, 1000)
>>> plt.plot(xs, spl(xs), 'g', lw=3, alpha=0.7)
>>> plt.show()

Notice that the ``spl(x)`` interpolates `y`:

>>> spl.get_residual()
0.0
*)

val antiderivative : ?n:int -> [> tag] Obj.t -> Py.Object.t
(**
Construct a new spline representing the antiderivative of this spline.

Parameters
----------
n : int, optional
    Order of antiderivative to evaluate. Default: 1

Returns
-------
spline : UnivariateSpline
    Spline of order k2=k+n representing the antiderivative of this
    spline.

Notes
-----

.. versionadded:: 0.13.0

See Also
--------
splantider, derivative

Examples
--------
>>> from scipy.interpolate import UnivariateSpline
>>> x = np.linspace(0, np.pi/2, 70)
>>> y = 1 / np.sqrt(1 - 0.8*np.sin(x)**2)
>>> spl = UnivariateSpline(x, y, s=0)

The derivative is the inverse operation of the antiderivative,
although some floating point error accumulates:

>>> spl(1.7), spl.antiderivative().derivative()(1.7)
(array(2.1565429877197317), array(2.1565429877201865))

Antiderivative can be used to evaluate definite integrals:

>>> ispl = spl.antiderivative()
>>> ispl(np.pi/2) - ispl(0)
2.2572053588768486

This is indeed an approximation to the complete elliptic integral
:math:`K(m) = \int_0^{\pi/2} [1 - m\sin^2 x]^{-1/2} dx`:

>>> from scipy.special import ellipk
>>> ellipk(0.8)
2.2572053268208538
*)

val derivative : ?n:int -> [> tag] Obj.t -> Py.Object.t
(**
Construct a new spline representing the derivative of this spline.

Parameters
----------
n : int, optional
    Order of derivative to evaluate. Default: 1

Returns
-------
spline : UnivariateSpline
    Spline of order k2=k-n representing the derivative of this
    spline.

See Also
--------
splder, antiderivative

Notes
-----

.. versionadded:: 0.13.0

Examples
--------
This can be used for finding maxima of a curve:

>>> from scipy.interpolate import UnivariateSpline
>>> x = np.linspace(0, 10, 70)
>>> y = np.sin(x)
>>> spl = UnivariateSpline(x, y, k=4, s=0)

Now, differentiate the spline and find the zeros of the
derivative. (NB: `sproot` only works for order 3 splines, so we
fit an order 4 spline):

>>> spl.derivative().roots() / np.pi
array([ 0.50000001,  1.5       ,  2.49999998])

This agrees well with roots :math:`\pi/2 + n\pi` of
:math:`\cos(x) = \sin'(x)`.
*)

val derivatives : x:float -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return all derivatives of the spline at the point x.

Parameters
----------
x : float
    The point to evaluate the derivatives at.

Returns
-------
der : ndarray, shape(k+1,)
    Derivatives of the orders 0 to k.

Examples
--------
>>> from scipy.interpolate import UnivariateSpline
>>> x = np.linspace(0, 3, 11)
>>> y = x**2
>>> spl = UnivariateSpline(x, y)
>>> spl.derivatives(1.5)
array([2.25, 3.0, 2.0, 0])
*)

val get_coeffs : [> tag] Obj.t -> Py.Object.t
(**
Return spline coefficients.
*)

val get_knots : [> tag] Obj.t -> Py.Object.t
(**
Return positions of interior knots of the spline.

Internally, the knot vector contains ``2*k`` additional boundary knots.
*)

val get_residual : [> tag] Obj.t -> Py.Object.t
(**
Return weighted sum of squared residuals of the spline approximation.

This is equivalent to::

     sum((w[i] * (y[i]-spl(x[i])))**2, axis=0)
*)

val integral : a:float -> b:float -> [> tag] Obj.t -> float
(**
Return definite integral of the spline between two given points.

Parameters
----------
a : float
    Lower limit of integration.
b : float
    Upper limit of integration.

Returns
-------
integral : float
    The value of the definite integral of the spline between limits.

Examples
--------
>>> from scipy.interpolate import UnivariateSpline
>>> x = np.linspace(0, 3, 11)
>>> y = x**2
>>> spl = UnivariateSpline(x, y)
>>> spl.integral(0, 3)
9.0

which agrees with :math:`\int x^2 dx = x^3 / 3` between the limits
of 0 and 3.

A caveat is that this routine assumes the spline to be zero outside of
the data limits:

>>> spl.integral(-1, 4)
9.0
>>> spl.integral(-1, 0)
0.0
*)

val roots : [> tag] Obj.t -> Py.Object.t
(**
Return the zeros of the spline.

Restriction: only cubic splines are supported by fitpack.
*)

val set_smoothing_factor : s:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Continue spline computation with the given smoothing
factor s and with the knots found at the last call.

This routine modifies the spline in place.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module KroghInterpolator : sig
type tag = [`KroghInterpolator]
type t = [`KroghInterpolator | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?axis:int -> xi:[`Ndarray of [>`Ndarray] Np.Obj.t | `Length_N of Py.Object.t] -> yi:[>`Ndarray] Np.Obj.t -> unit -> t
(**
Interpolating polynomial for a set of points.

The polynomial passes through all the pairs (xi,yi). One may
additionally specify a number of derivatives at each point xi;
this is done by repeating the value xi and specifying the
derivatives as successive yi values.

Allows evaluation of the polynomial and all its derivatives.
For reasons of numerical stability, this function does not compute
the coefficients of the polynomial, although they can be obtained
by evaluating all the derivatives.

Parameters
----------
xi : array_like, length N
    Known x-coordinates. Must be sorted in increasing order.
yi : array_like
    Known y-coordinates. When an xi occurs two or more times in
    a row, the corresponding yi's represent derivative values.
axis : int, optional
    Axis in the yi array corresponding to the x-coordinate values.

Notes
-----
Be aware that the algorithms implemented here are not necessarily
the most numerically stable known. Moreover, even in a world of
exact computation, unless the x coordinates are chosen very
carefully - Chebyshev zeros (e.g. cos(i*pi/n)) are a good choice -
polynomial interpolation itself is a very ill-conditioned process
due to the Runge phenomenon. In general, even with well-chosen
x values, degrees higher than about thirty cause problems with
numerical instability in this code.

Based on [1]_.

References
----------
.. [1] Krogh, 'Efficient Algorithms for Polynomial Interpolation
    and Numerical Differentiation', 1970.

Examples
--------
To produce a polynomial that is zero at 0 and 1 and has
derivative 2 at 0, call

>>> from scipy.interpolate import KroghInterpolator
>>> KroghInterpolator([0,0,1],[0,2,0])

This constructs the quadratic 2*X**2-2*X. The derivative condition
is indicated by the repeated zero in the xi array; the corresponding
yi values are 0, the function value, and 2, the derivative value.

For another example, given xi, yi, and a derivative ypi for each
point, appropriate arrays can be constructed as:

>>> xi = np.linspace(0, 1, 5)
>>> yi, ypi = np.random.rand(2, 5)
>>> xi_k, yi_k = np.repeat(xi, 2), np.ravel(np.dstack((yi,ypi)))
>>> KroghInterpolator(xi_k, yi_k)

To produce a vector-valued polynomial, supply a higher-dimensional
array for yi:

>>> KroghInterpolator([0,1],[[2,3],[4,5]])

This constructs a linear polynomial giving (2,3) at 0 and (4,5) at 1.
*)

val derivative : ?der:int -> x:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Evaluate one derivative of the polynomial at the point x

Parameters
----------
x : array_like
    Point or points at which to evaluate the derivatives

der : integer, optional
    Which derivative to extract. This number includes the
    function value as 0th derivative.

Returns
-------
d : ndarray
    Derivative interpolated at the x-points.  Shape of d is
    determined by replacing the interpolation axis in the
    original array with the shape of x.

Notes
-----
This is computed by evaluating all derivatives up to the desired
one (using self.derivatives()) and then discarding the rest.
*)

val derivatives : ?der:int -> x:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Evaluate many derivatives of the polynomial at the point x

Produce an array of all derivative values at the point x.

Parameters
----------
x : array_like
    Point or points at which to evaluate the derivatives
der : int or None, optional
    How many derivatives to extract; None for all potentially
    nonzero derivatives (that is a number equal to the number
    of points). This number includes the function value as 0th
    derivative.

Returns
-------
d : ndarray
    Array with derivatives; d[j] contains the j-th derivative.
    Shape of d[j] is determined by replacing the interpolation
    axis in the original array with the shape of x.

Examples
--------
>>> from scipy.interpolate import KroghInterpolator
>>> KroghInterpolator([0,0,0],[1,2,3]).derivatives(0)
array([1.0,2.0,3.0])
>>> KroghInterpolator([0,0,0],[1,2,3]).derivatives([0,0])
array([[1.0,1.0],
       [2.0,2.0],
       [3.0,3.0]])
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module LSQBivariateSpline : sig
type tag = [`LSQBivariateSpline]
type t = [`LSQBivariateSpline | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?w:[>`Ndarray] Np.Obj.t -> ?bbox:Py.Object.t -> ?kx:Py.Object.t -> ?ky:Py.Object.t -> ?eps:float -> x:Py.Object.t -> y:Py.Object.t -> z:Py.Object.t -> tx:Py.Object.t -> ty:Py.Object.t -> unit -> t
(**
Weighted least-squares bivariate spline approximation.

Parameters
----------
x, y, z : array_like
    1-D sequences of data points (order is not important).
tx, ty : array_like
    Strictly ordered 1-D sequences of knots coordinates.
w : array_like, optional
    Positive 1-D array of weights, of the same length as `x`, `y` and `z`.
bbox : (4,) array_like, optional
    Sequence of length 4 specifying the boundary of the rectangular
    approximation domain.  By default,
    ``bbox=[min(x,tx),max(x,tx), min(y,ty),max(y,ty)]``.
kx, ky : ints, optional
    Degrees of the bivariate spline. Default is 3.
eps : float, optional
    A threshold for determining the effective rank of an over-determined
    linear system of equations. `eps` should have a value between 0 and 1,
    the default is 1e-16.

See Also
--------
bisplrep : an older wrapping of FITPACK
bisplev : an older wrapping of FITPACK
UnivariateSpline : a similar class for univariate spline interpolation
SmoothBivariateSpline : create a smoothing BivariateSpline

Notes
-----
The length of `x`, `y` and `z` should be at least ``(kx+1) * (ky+1)``.
*)

val ev : ?dx:int -> ?dy:int -> xi:Py.Object.t -> yi:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Evaluate the spline at points

Returns the interpolated value at ``(xi[i], yi[i]),
i=0,...,len(xi)-1``.

Parameters
----------
xi, yi : array_like
    Input coordinates. Standard Numpy broadcasting is obeyed.
dx : int, optional
    Order of x-derivative

    .. versionadded:: 0.14.0
dy : int, optional
    Order of y-derivative

    .. versionadded:: 0.14.0
*)

val get_coeffs : [> tag] Obj.t -> Py.Object.t
(**
Return spline coefficients.
*)

val get_knots : [> tag] Obj.t -> Py.Object.t
(**
Return a tuple (tx,ty) where tx,ty contain knots positions
of the spline with respect to x-, y-variable, respectively.
The position of interior and additional knots are given as
t[k+1:-k-1] and t[:k+1]=b, t[-k-1:]=e, respectively.
*)

val get_residual : [> tag] Obj.t -> Py.Object.t
(**
Return weighted sum of squared residuals of the spline
approximation: sum ((w[i]*(z[i]-s(x[i],y[i])))**2,axis=0)
*)

val integral : xa:Py.Object.t -> xb:Py.Object.t -> ya:Py.Object.t -> yb:Py.Object.t -> [> tag] Obj.t -> float
(**
Evaluate the integral of the spline over area [xa,xb] x [ya,yb].

Parameters
----------
xa, xb : float
    The end-points of the x integration interval.
ya, yb : float
    The end-points of the y integration interval.

Returns
-------
integ : float
    The value of the resulting integral.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module LSQSphereBivariateSpline : sig
type tag = [`LSQSphereBivariateSpline]
type t = [`LSQSphereBivariateSpline | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?w:[>`Ndarray] Np.Obj.t -> ?eps:float -> theta:Py.Object.t -> phi:Py.Object.t -> r:Py.Object.t -> tt:Py.Object.t -> tp:Py.Object.t -> unit -> t
(**
Weighted least-squares bivariate spline approximation in spherical
coordinates.

Determines a smooth bicubic spline according to a given
set of knots in the `theta` and `phi` directions.

.. versionadded:: 0.11.0

Parameters
----------
theta, phi, r : array_like
    1-D sequences of data points (order is not important). Coordinates
    must be given in radians. Theta must lie within the interval (0, pi),
    and phi must lie within the interval (0, 2pi).
tt, tp : array_like
    Strictly ordered 1-D sequences of knots coordinates.
    Coordinates must satisfy ``0 < tt[i] < pi``, ``0 < tp[i] < 2*pi``.
w : array_like, optional
    Positive 1-D sequence of weights, of the same length as `theta`, `phi`
    and `r`.
eps : float, optional
    A threshold for determining the effective rank of an over-determined
    linear system of equations. `eps` should have a value between 0 and 1,
    the default is 1e-16.

Notes
-----
For more information, see the FITPACK_ site about this function.

.. _FITPACK: http://www.netlib.org/dierckx/sphere.f

Examples
--------
Suppose we have global data on a coarse grid (the input data does not
have to be on a grid):

>>> theta = np.linspace(0., np.pi, 7)
>>> phi = np.linspace(0., 2*np.pi, 9)
>>> data = np.empty((theta.shape[0], phi.shape[0]))
>>> data[:,0], data[0,:], data[-1,:] = 0., 0., 0.
>>> data[1:-1,1], data[1:-1,-1] = 1., 1.
>>> data[1,1:-1], data[-2,1:-1] = 1., 1.
>>> data[2:-2,2], data[2:-2,-2] = 2., 2.
>>> data[2,2:-2], data[-3,2:-2] = 2., 2.
>>> data[3,3:-2] = 3.
>>> data = np.roll(data, 4, 1)

We need to set up the interpolator object. Here, we must also specify the
coordinates of the knots to use.

>>> lats, lons = np.meshgrid(theta, phi)
>>> knotst, knotsp = theta.copy(), phi.copy()
>>> knotst[0] += .0001
>>> knotst[-1] -= .0001
>>> knotsp[0] += .0001
>>> knotsp[-1] -= .0001
>>> from scipy.interpolate import LSQSphereBivariateSpline
>>> lut = LSQSphereBivariateSpline(lats.ravel(), lons.ravel(),
...                                data.T.ravel(), knotst, knotsp)

As a first test, we'll see what the algorithm returns when run on the
input coordinates

>>> data_orig = lut(theta, phi)

Finally we interpolate the data to a finer grid

>>> fine_lats = np.linspace(0., np.pi, 70)
>>> fine_lons = np.linspace(0., 2*np.pi, 90)

>>> data_lsq = lut(fine_lats, fine_lons)

>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> ax1 = fig.add_subplot(131)
>>> ax1.imshow(data, interpolation='nearest')
>>> ax2 = fig.add_subplot(132)
>>> ax2.imshow(data_orig, interpolation='nearest')
>>> ax3 = fig.add_subplot(133)
>>> ax3.imshow(data_lsq, interpolation='nearest')
>>> plt.show()
*)

val ev : ?dtheta:int -> ?dphi:int -> theta:Py.Object.t -> phi:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Evaluate the spline at points

Returns the interpolated value at ``(theta[i], phi[i]),
i=0,...,len(theta)-1``.

Parameters
----------
theta, phi : array_like
    Input coordinates. Standard Numpy broadcasting is obeyed.
dtheta : int, optional
    Order of theta-derivative

    .. versionadded:: 0.14.0
dphi : int, optional
    Order of phi-derivative

    .. versionadded:: 0.14.0
*)

val get_coeffs : [> tag] Obj.t -> Py.Object.t
(**
Return spline coefficients.
*)

val get_knots : [> tag] Obj.t -> Py.Object.t
(**
Return a tuple (tx,ty) where tx,ty contain knots positions
of the spline with respect to x-, y-variable, respectively.
The position of interior and additional knots are given as
t[k+1:-k-1] and t[:k+1]=b, t[-k-1:]=e, respectively.
*)

val get_residual : [> tag] Obj.t -> Py.Object.t
(**
Return weighted sum of squared residuals of the spline
approximation: sum ((w[i]*(z[i]-s(x[i],y[i])))**2,axis=0)
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module LSQUnivariateSpline : sig
type tag = [`LSQUnivariateSpline]
type t = [`LSQUnivariateSpline | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?w:[>`Ndarray] Np.Obj.t -> ?bbox:Py.Object.t -> ?k:int -> ?ext:[`S of string | `I of int] -> ?check_finite:bool -> x:[>`Ndarray] Np.Obj.t -> y:[>`Ndarray] Np.Obj.t -> t:[>`Ndarray] Np.Obj.t -> unit -> t
(**
One-dimensional spline with explicit internal knots.

Fits a spline y = spl(x) of degree `k` to the provided `x`, `y` data.  `t`
specifies the internal knots of the spline

Parameters
----------
x : (N,) array_like
    Input dimension of data points -- must be increasing
y : (N,) array_like
    Input dimension of data points
t : (M,) array_like
    interior knots of the spline.  Must be in ascending order and::

        bbox[0] < t[0] < ... < t[-1] < bbox[-1]

w : (N,) array_like, optional
    weights for spline fitting.  Must be positive.  If None (default),
    weights are all equal.
bbox : (2,) array_like, optional
    2-sequence specifying the boundary of the approximation interval. If
    None (default), ``bbox = [x[0], x[-1]]``.
k : int, optional
    Degree of the smoothing spline.  Must be 1 <= `k` <= 5.
    Default is k=3, a cubic spline.
ext : int or str, optional
    Controls the extrapolation mode for elements
    not in the interval defined by the knot sequence.

    * if ext=0 or 'extrapolate', return the extrapolated value.
    * if ext=1 or 'zeros', return 0
    * if ext=2 or 'raise', raise a ValueError
    * if ext=3 of 'const', return the boundary value.

    The default value is 0.

check_finite : bool, optional
    Whether to check that the input arrays contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination or non-sensical results) if the inputs
    do contain infinities or NaNs.
    Default is False.

Raises
------
ValueError
    If the interior knots do not satisfy the Schoenberg-Whitney conditions

See Also
--------
UnivariateSpline : Superclass -- knots are specified by setting a
    smoothing condition
InterpolatedUnivariateSpline : spline passing through all points
splrep : An older, non object-oriented wrapping of FITPACK
splev, sproot, splint, spalde
BivariateSpline : A similar class for two-dimensional spline interpolation

Notes
-----
The number of data points must be larger than the spline degree `k`.

Knots `t` must satisfy the Schoenberg-Whitney conditions,
i.e., there must be a subset of data points ``x[j]`` such that
``t[j] < x[j] < t[j+k+1]``, for ``j=0, 1,...,n-k-2``.

Examples
--------
>>> from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline
>>> import matplotlib.pyplot as plt
>>> x = np.linspace(-3, 3, 50)
>>> y = np.exp(-x**2) + 0.1 * np.random.randn(50)

Fit a smoothing spline with a pre-defined internal knots:

>>> t = [-1, 0, 1]
>>> spl = LSQUnivariateSpline(x, y, t)

>>> xs = np.linspace(-3, 3, 1000)
>>> plt.plot(x, y, 'ro', ms=5)
>>> plt.plot(xs, spl(xs), 'g-', lw=3)
>>> plt.show()

Check the knot vector:

>>> spl.get_knots()
array([-3., -1., 0., 1., 3.])

Constructing lsq spline using the knots from another spline:

>>> x = np.arange(10)
>>> s = UnivariateSpline(x, x, s=0)
>>> s.get_knots()
array([ 0.,  2.,  3.,  4.,  5.,  6.,  7.,  9.])
>>> knt = s.get_knots()
>>> s1 = LSQUnivariateSpline(x, x, knt[1:-1])    # Chop 1st and last knot
>>> s1.get_knots()
array([ 0.,  2.,  3.,  4.,  5.,  6.,  7.,  9.])
*)

val antiderivative : ?n:int -> [> tag] Obj.t -> Py.Object.t
(**
Construct a new spline representing the antiderivative of this spline.

Parameters
----------
n : int, optional
    Order of antiderivative to evaluate. Default: 1

Returns
-------
spline : UnivariateSpline
    Spline of order k2=k+n representing the antiderivative of this
    spline.

Notes
-----

.. versionadded:: 0.13.0

See Also
--------
splantider, derivative

Examples
--------
>>> from scipy.interpolate import UnivariateSpline
>>> x = np.linspace(0, np.pi/2, 70)
>>> y = 1 / np.sqrt(1 - 0.8*np.sin(x)**2)
>>> spl = UnivariateSpline(x, y, s=0)

The derivative is the inverse operation of the antiderivative,
although some floating point error accumulates:

>>> spl(1.7), spl.antiderivative().derivative()(1.7)
(array(2.1565429877197317), array(2.1565429877201865))

Antiderivative can be used to evaluate definite integrals:

>>> ispl = spl.antiderivative()
>>> ispl(np.pi/2) - ispl(0)
2.2572053588768486

This is indeed an approximation to the complete elliptic integral
:math:`K(m) = \int_0^{\pi/2} [1 - m\sin^2 x]^{-1/2} dx`:

>>> from scipy.special import ellipk
>>> ellipk(0.8)
2.2572053268208538
*)

val derivative : ?n:int -> [> tag] Obj.t -> Py.Object.t
(**
Construct a new spline representing the derivative of this spline.

Parameters
----------
n : int, optional
    Order of derivative to evaluate. Default: 1

Returns
-------
spline : UnivariateSpline
    Spline of order k2=k-n representing the derivative of this
    spline.

See Also
--------
splder, antiderivative

Notes
-----

.. versionadded:: 0.13.0

Examples
--------
This can be used for finding maxima of a curve:

>>> from scipy.interpolate import UnivariateSpline
>>> x = np.linspace(0, 10, 70)
>>> y = np.sin(x)
>>> spl = UnivariateSpline(x, y, k=4, s=0)

Now, differentiate the spline and find the zeros of the
derivative. (NB: `sproot` only works for order 3 splines, so we
fit an order 4 spline):

>>> spl.derivative().roots() / np.pi
array([ 0.50000001,  1.5       ,  2.49999998])

This agrees well with roots :math:`\pi/2 + n\pi` of
:math:`\cos(x) = \sin'(x)`.
*)

val derivatives : x:float -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return all derivatives of the spline at the point x.

Parameters
----------
x : float
    The point to evaluate the derivatives at.

Returns
-------
der : ndarray, shape(k+1,)
    Derivatives of the orders 0 to k.

Examples
--------
>>> from scipy.interpolate import UnivariateSpline
>>> x = np.linspace(0, 3, 11)
>>> y = x**2
>>> spl = UnivariateSpline(x, y)
>>> spl.derivatives(1.5)
array([2.25, 3.0, 2.0, 0])
*)

val get_coeffs : [> tag] Obj.t -> Py.Object.t
(**
Return spline coefficients.
*)

val get_knots : [> tag] Obj.t -> Py.Object.t
(**
Return positions of interior knots of the spline.

Internally, the knot vector contains ``2*k`` additional boundary knots.
*)

val get_residual : [> tag] Obj.t -> Py.Object.t
(**
Return weighted sum of squared residuals of the spline approximation.

This is equivalent to::

     sum((w[i] * (y[i]-spl(x[i])))**2, axis=0)
*)

val integral : a:float -> b:float -> [> tag] Obj.t -> float
(**
Return definite integral of the spline between two given points.

Parameters
----------
a : float
    Lower limit of integration.
b : float
    Upper limit of integration.

Returns
-------
integral : float
    The value of the definite integral of the spline between limits.

Examples
--------
>>> from scipy.interpolate import UnivariateSpline
>>> x = np.linspace(0, 3, 11)
>>> y = x**2
>>> spl = UnivariateSpline(x, y)
>>> spl.integral(0, 3)
9.0

which agrees with :math:`\int x^2 dx = x^3 / 3` between the limits
of 0 and 3.

A caveat is that this routine assumes the spline to be zero outside of
the data limits:

>>> spl.integral(-1, 4)
9.0
>>> spl.integral(-1, 0)
0.0
*)

val roots : [> tag] Obj.t -> Py.Object.t
(**
Return the zeros of the spline.

Restriction: only cubic splines are supported by fitpack.
*)

val set_smoothing_factor : s:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Continue spline computation with the given smoothing
factor s and with the knots found at the last call.

This routine modifies the spline in place.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module LinearNDInterpolator : sig
type tag = [`LinearNDInterpolator]
type t = [`LinearNDInterpolator | `NDInterpolatorBase | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_nd_interpolator : t -> [`NDInterpolatorBase] Obj.t
val create : ?fill_value:float -> ?rescale:bool -> points:[`Ndarray of [>`Ndarray] Np.Obj.t | `Delaunay of Py.Object.t] -> values:Py.Object.t -> unit -> t
(**
LinearNDInterpolator(points, values, fill_value=np.nan, rescale=False)

Piecewise linear interpolant in N dimensions.

.. versionadded:: 0.9

Methods
-------
__call__

Parameters
----------
points : ndarray of floats, shape (npoints, ndims); or Delaunay
    Data point coordinates, or a precomputed Delaunay triangulation.
values : ndarray of float or complex, shape (npoints, ...)
    Data values.
fill_value : float, optional
    Value used to fill in for requested points outside of the
    convex hull of the input points.  If not provided, then
    the default is ``nan``.
rescale : bool, optional
    Rescale points to unit cube before performing interpolation.
    This is useful if some of the input dimensions have
    incommensurable units and differ by many orders of magnitude.

Notes
-----
The interpolant is constructed by triangulating the input data
with Qhull [1]_, and on each triangle performing linear
barycentric interpolation.

References
----------
.. [1] http://www.qhull.org/
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module NdPPoly : sig
type tag = [`NdPPoly]
type t = [`NdPPoly | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?extrapolate:bool -> c:[>`Ndarray] Np.Obj.t -> x:Py.Object.t -> unit -> t
(**
Piecewise tensor product polynomial

The value at point ``xp = (x', y', z', ...)`` is evaluated by first
computing the interval indices `i` such that::

    x[0][i[0]] <= x' < x[0][i[0]+1]
    x[1][i[1]] <= y' < x[1][i[1]+1]
    ...

and then computing::

    S = sum(c[k0-m0-1,...,kn-mn-1,i[0],...,i[n]]
            * (xp[0] - x[0][i[0]])**m0
            * ...
            * (xp[n] - x[n][i[n]])**mn
            for m0 in range(k[0]+1)
            ...
            for mn in range(k[n]+1))

where ``k[j]`` is the degree of the polynomial in dimension j. This
representation is the piecewise multivariate power basis.

Parameters
----------
c : ndarray, shape (k0, ..., kn, m0, ..., mn, ...)
    Polynomial coefficients, with polynomial order `kj` and
    `mj+1` intervals for each dimension `j`.
x : ndim-tuple of ndarrays, shapes (mj+1,)
    Polynomial breakpoints for each dimension. These must be
    sorted in increasing order.
extrapolate : bool, optional
    Whether to extrapolate to out-of-bounds points based on first
    and last intervals, or to return NaNs. Default: True.

Attributes
----------
x : tuple of ndarrays
    Breakpoints.
c : ndarray
    Coefficients of the polynomials.

Methods
-------
__call__
construct_fast

See also
--------
PPoly : piecewise polynomials in 1D

Notes
-----
High-order polynomials in the power basis can be numerically
unstable.
*)

val antiderivative : nu:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Construct a new piecewise polynomial representing the antiderivative.

Antiderivative is also the indefinite integral of the function,
and derivative is its inverse operation.

Parameters
----------
nu : ndim-tuple of int
    Order of derivatives to evaluate for each dimension.
    If negative, the derivative is returned.

Returns
-------
pp : PPoly
    Piecewise polynomial of order k2 = k + n representing
    the antiderivative of this polynomial.

Notes
-----
The antiderivative returned by this function is continuous and
continuously differentiable to order n-1, up to floating point
rounding error.
*)

val construct_fast : ?extrapolate:Py.Object.t -> c:Py.Object.t -> x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Construct the piecewise polynomial without making checks.

Takes the same parameters as the constructor. Input arguments
``c`` and ``x`` must be arrays of the correct shape and type.  The
``c`` array can only be of dtypes float and complex, and ``x``
array must have dtype float.
*)

val derivative : nu:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Construct a new piecewise polynomial representing the derivative.

Parameters
----------
nu : ndim-tuple of int
    Order of derivatives to evaluate for each dimension.
    If negative, the antiderivative is returned.

Returns
-------
pp : NdPPoly
    Piecewise polynomial of orders (k[0] - nu[0], ..., k[n] - nu[n])
    representing the derivative of this polynomial.

Notes
-----
Derivatives are evaluated piecewise for each polynomial
segment, even if the polynomial is not differentiable at the
breakpoints. The polynomial intervals in each dimension are
considered half-open, ``[a, b)``, except for the last interval
which is closed ``[a, b]``.
*)

val integrate : ?extrapolate:bool -> ranges:Py.Object.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute a definite integral over a piecewise polynomial.

Parameters
----------
ranges : ndim-tuple of 2-tuples float
    Sequence of lower and upper bounds for each dimension,
    ``[(a[0], b[0]), ..., (a[ndim-1], b[ndim-1])]``
extrapolate : bool, optional
    Whether to extrapolate to out-of-bounds points based on first
    and last intervals, or to return NaNs.

Returns
-------
ig : array_like
    Definite integral of the piecewise polynomial over
    [a[0], b[0]] x ... x [a[ndim-1], b[ndim-1]]
*)

val integrate_1d : ?extrapolate:bool -> a:Py.Object.t -> b:Py.Object.t -> axis:int -> [> tag] Obj.t -> Py.Object.t
(**
Compute NdPPoly representation for one dimensional definite integral

The result is a piecewise polynomial representing the integral:

.. math::

   p(y, z, ...) = \int_a^b dx\, p(x, y, z, ...)

where the dimension integrated over is specified with the
`axis` parameter.

Parameters
----------
a, b : float
    Lower and upper bound for integration.
axis : int
    Dimension over which to compute the 1D integrals
extrapolate : bool, optional
    Whether to extrapolate to out-of-bounds points based on first
    and last intervals, or to return NaNs.

Returns
-------
ig : NdPPoly or array-like
    Definite integral of the piecewise polynomial over [a, b].
    If the polynomial was 1-dimensional, an array is returned,
    otherwise, an NdPPoly object.
*)


(** Attribute x: get value or raise Not_found if None.*)
val x : t -> Py.Object.t

(** Attribute x: get value as an option. *)
val x_opt : t -> (Py.Object.t) option


(** Attribute c: get value or raise Not_found if None.*)
val c : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute c: get value as an option. *)
val c_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module NearestNDInterpolator : sig
type tag = [`NearestNDInterpolator]
type t = [`NDInterpolatorBase | `NearestNDInterpolator | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_nd_interpolator : t -> [`NDInterpolatorBase] Obj.t
val create : ?rescale:bool -> ?tree_options:Py.Object.t -> x:[>`Ndarray] Np.Obj.t -> y:Py.Object.t -> unit -> t
(**
NearestNDInterpolator(x, y)

Nearest-neighbour interpolation in N dimensions.

.. versionadded:: 0.9

Methods
-------
__call__

Parameters
----------
x : (Npoints, Ndims) ndarray of floats
    Data point coordinates.
y : (Npoints,) ndarray of float or complex
    Data values.
rescale : boolean, optional
    Rescale points to unit cube before performing interpolation.
    This is useful if some of the input dimensions have
    incommensurable units and differ by many orders of magnitude.

    .. versionadded:: 0.14.0
tree_options : dict, optional
    Options passed to the underlying ``cKDTree``.

    .. versionadded:: 0.17.0


Notes
-----
Uses ``scipy.spatial.cKDTree``
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module PPoly : sig
type tag = [`PPoly]
type t = [`Object | `PPoly] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?extrapolate:[`Bool of bool | `Periodic] -> ?axis:int -> c:[>`Ndarray] Np.Obj.t -> x:[>`Ndarray] Np.Obj.t -> unit -> t
(**
Piecewise polynomial in terms of coefficients and breakpoints

The polynomial between ``x[i]`` and ``x[i + 1]`` is written in the
local power basis::

    S = sum(c[m, i] * (xp - x[i])**(k-m) for m in range(k+1))

where ``k`` is the degree of the polynomial.

Parameters
----------
c : ndarray, shape (k, m, ...)
    Polynomial coefficients, order `k` and `m` intervals
x : ndarray, shape (m+1,)
    Polynomial breakpoints. Must be sorted in either increasing or
    decreasing order.
extrapolate : bool or 'periodic', optional
    If bool, determines whether to extrapolate to out-of-bounds points
    based on first and last intervals, or to return NaNs. If 'periodic',
    periodic extrapolation is used. Default is True.
axis : int, optional
    Interpolation axis. Default is zero.

Attributes
----------
x : ndarray
    Breakpoints.
c : ndarray
    Coefficients of the polynomials. They are reshaped
    to a 3-dimensional array with the last dimension representing
    the trailing dimensions of the original coefficient array.
axis : int
    Interpolation axis.

Methods
-------
__call__
derivative
antiderivative
integrate
solve
roots
extend
from_spline
from_bernstein_basis
construct_fast

See also
--------
BPoly : piecewise polynomials in the Bernstein basis

Notes
-----
High-order polynomials in the power basis can be numerically
unstable.  Precision problems can start to appear for orders
larger than 20-30.
*)

val antiderivative : ?nu:int -> [> tag] Obj.t -> Py.Object.t
(**
Construct a new piecewise polynomial representing the antiderivative.

Antiderivative is also the indefinite integral of the function,
and derivative is its inverse operation.

Parameters
----------
nu : int, optional
    Order of antiderivative to evaluate. Default is 1, i.e. compute
    the first integral. If negative, the derivative is returned.

Returns
-------
pp : PPoly
    Piecewise polynomial of order k2 = k + n representing
    the antiderivative of this polynomial.

Notes
-----
The antiderivative returned by this function is continuous and
continuously differentiable to order n-1, up to floating point
rounding error.

If antiderivative is computed and ``self.extrapolate='periodic'``,
it will be set to False for the returned instance. This is done because
the antiderivative is no longer periodic and its correct evaluation
outside of the initially given x interval is difficult.
*)

val construct_fast : ?extrapolate:Py.Object.t -> ?axis:Py.Object.t -> c:Py.Object.t -> x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Construct the piecewise polynomial without making checks.

Takes the same parameters as the constructor. Input arguments
``c`` and ``x`` must be arrays of the correct shape and type.  The
``c`` array can only be of dtypes float and complex, and ``x``
array must have dtype float.
*)

val derivative : ?nu:int -> [> tag] Obj.t -> Py.Object.t
(**
Construct a new piecewise polynomial representing the derivative.

Parameters
----------
nu : int, optional
    Order of derivative to evaluate. Default is 1, i.e. compute the
    first derivative. If negative, the antiderivative is returned.

Returns
-------
pp : PPoly
    Piecewise polynomial of order k2 = k - n representing the derivative
    of this polynomial.

Notes
-----
Derivatives are evaluated piecewise for each polynomial
segment, even if the polynomial is not differentiable at the
breakpoints. The polynomial intervals are considered half-open,
``[a, b)``, except for the last interval which is closed
``[a, b]``.
*)

val extend : ?right:Py.Object.t -> c:[`Ndarray of [>`Ndarray] Np.Obj.t | `Size_k_m_ of Py.Object.t] -> x:[`Ndarray of [>`Ndarray] Np.Obj.t | `Size of Py.Object.t] -> [> tag] Obj.t -> Py.Object.t
(**
Add additional breakpoints and coefficients to the polynomial.

Parameters
----------
c : ndarray, size (k, m, ...)
    Additional coefficients for polynomials in intervals. Note that
    the first additional interval will be formed using one of the
    ``self.x`` end points.
x : ndarray, size (m,)
    Additional breakpoints. Must be sorted in the same order as
    ``self.x`` and either to the right or to the left of the current
    breakpoints.
right
    Deprecated argument. Has no effect.

    .. deprecated:: 0.19
*)

val from_bernstein_basis : ?extrapolate:[`Bool of bool | `Periodic] -> bp:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Construct a piecewise polynomial in the power basis
from a polynomial in Bernstein basis.

Parameters
----------
bp : BPoly
    A Bernstein basis polynomial, as created by BPoly
extrapolate : bool or 'periodic', optional
    If bool, determines whether to extrapolate to out-of-bounds points
    based on first and last intervals, or to return NaNs.
    If 'periodic', periodic extrapolation is used. Default is True.
*)

val from_spline : ?extrapolate:[`Bool of bool | `Periodic] -> tck:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Construct a piecewise polynomial from a spline

Parameters
----------
tck
    A spline, as returned by `splrep` or a BSpline object.
extrapolate : bool or 'periodic', optional
    If bool, determines whether to extrapolate to out-of-bounds points
    based on first and last intervals, or to return NaNs.
    If 'periodic', periodic extrapolation is used. Default is True.
*)

val integrate : ?extrapolate:[`Bool of bool | `Periodic] -> a:float -> b:float -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute a definite integral over a piecewise polynomial.

Parameters
----------
a : float
    Lower integration bound
b : float
    Upper integration bound
extrapolate : {bool, 'periodic', None}, optional
    If bool, determines whether to extrapolate to out-of-bounds points
    based on first and last intervals, or to return NaNs.
    If 'periodic', periodic extrapolation is used.
    If None (default), use `self.extrapolate`.

Returns
-------
ig : array_like
    Definite integral of the piecewise polynomial over [a, b]
*)

val roots : ?discontinuity:bool -> ?extrapolate:[`Bool of bool | `Periodic] -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Find real roots of the the piecewise polynomial.

Parameters
----------
discontinuity : bool, optional
    Whether to report sign changes across discontinuities at
    breakpoints as roots.
extrapolate : {bool, 'periodic', None}, optional
    If bool, determines whether to return roots from the polynomial
    extrapolated based on first and last intervals, 'periodic' works
    the same as False. If None (default), use `self.extrapolate`.

Returns
-------
roots : ndarray
    Roots of the polynomial(s).

    If the PPoly object describes multiple polynomials, the
    return value is an object array whose each element is an
    ndarray containing the roots.

See Also
--------
PPoly.solve
*)

val solve : ?y:float -> ?discontinuity:bool -> ?extrapolate:[`Bool of bool | `Periodic] -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Find real solutions of the the equation ``pp(x) == y``.

Parameters
----------
y : float, optional
    Right-hand side. Default is zero.
discontinuity : bool, optional
    Whether to report sign changes across discontinuities at
    breakpoints as roots.
extrapolate : {bool, 'periodic', None}, optional
    If bool, determines whether to return roots from the polynomial
    extrapolated based on first and last intervals, 'periodic' works
    the same as False. If None (default), use `self.extrapolate`.

Returns
-------
roots : ndarray
    Roots of the polynomial(s).

    If the PPoly object describes multiple polynomials, the
    return value is an object array whose each element is an
    ndarray containing the roots.

Notes
-----
This routine works only on real-valued polynomials.

If the piecewise polynomial contains sections that are
identically zero, the root list will contain the start point
of the corresponding interval, followed by a ``nan`` value.

If the polynomial is discontinuous across a breakpoint, and
there is a sign change across the breakpoint, this is reported
if the `discont` parameter is True.

Examples
--------

Finding roots of ``[x**2 - 1, (x - 1)**2]`` defined on intervals
``[-2, 1], [1, 2]``:

>>> from scipy.interpolate import PPoly
>>> pp = PPoly(np.array([[1, -4, 3], [1, 0, 0]]).T, [-2, 1, 2])
>>> pp.solve()
array([-1.,  1.])
*)


(** Attribute x: get value or raise Not_found if None.*)
val x : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute x: get value as an option. *)
val x_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute c: get value or raise Not_found if None.*)
val c : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute c: get value as an option. *)
val c_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute axis: get value or raise Not_found if None.*)
val axis : t -> int

(** Attribute axis: get value as an option. *)
val axis_opt : t -> (int) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module PchipInterpolator : sig
type tag = [`PchipInterpolator]
type t = [`Object | `PchipInterpolator] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?axis:int -> ?extrapolate:bool -> x:[>`Ndarray] Np.Obj.t -> y:[>`Ndarray] Np.Obj.t -> unit -> t
(**
PCHIP 1-d monotonic cubic interpolation.

``x`` and ``y`` are arrays of values used to approximate some function f,
with ``y = f(x)``. The interpolant uses monotonic cubic splines
to find the value of new points. (PCHIP stands for Piecewise Cubic
Hermite Interpolating Polynomial).

Parameters
----------
x : ndarray
    A 1-D array of monotonically increasing real values. ``x`` cannot
    include duplicate values (otherwise f is overspecified)
y : ndarray
    A 1-D array of real values. ``y``'s length along the interpolation
    axis must be equal to the length of ``x``. If N-D array, use ``axis``
    parameter to select correct axis.
axis : int, optional
    Axis in the y array corresponding to the x-coordinate values.
extrapolate : bool, optional
    Whether to extrapolate to out-of-bounds points based on first
    and last intervals, or to return NaNs.

Methods
-------
__call__
derivative
antiderivative
roots

See Also
--------
CubicHermiteSpline
Akima1DInterpolator
CubicSpline
PPoly

Notes
-----
The interpolator preserves monotonicity in the interpolation data and does
not overshoot if the data is not smooth.

The first derivatives are guaranteed to be continuous, but the second
derivatives may jump at :math:`x_k`.

Determines the derivatives at the points :math:`x_k`, :math:`f'_k`,
by using PCHIP algorithm [1]_.

Let :math:`h_k = x_{k+1} - x_k`, and  :math:`d_k = (y_{k+1} - y_k) / h_k`
are the slopes at internal points :math:`x_k`.
If the signs of :math:`d_k` and :math:`d_{k-1}` are different or either of
them equals zero, then :math:`f'_k = 0`. Otherwise, it is given by the
weighted harmonic mean

.. math::

    \frac{w_1 + w_2}{f'_k} = \frac{w_1}{d_{k-1}} + \frac{w_2}{d_k}

where :math:`w_1 = 2 h_k + h_{k-1}` and :math:`w_2 = h_k + 2 h_{k-1}`.

The end slopes are set using a one-sided scheme [2]_.


References
----------
.. [1] F. N. Fritsch and R. E. Carlson, Monotone Piecewise Cubic Interpolation,
       SIAM J. Numer. Anal., 17(2), 238 (1980).
       :doi:`10.1137/0717021`.
.. [2] see, e.g., C. Moler, Numerical Computing with Matlab, 2004.
       :doi:`10.1137/1.9780898717952`
*)

val antiderivative : ?nu:int -> [> tag] Obj.t -> Py.Object.t
(**
Construct a new piecewise polynomial representing the antiderivative.

Antiderivative is also the indefinite integral of the function,
and derivative is its inverse operation.

Parameters
----------
nu : int, optional
    Order of antiderivative to evaluate. Default is 1, i.e. compute
    the first integral. If negative, the derivative is returned.

Returns
-------
pp : PPoly
    Piecewise polynomial of order k2 = k + n representing
    the antiderivative of this polynomial.

Notes
-----
The antiderivative returned by this function is continuous and
continuously differentiable to order n-1, up to floating point
rounding error.

If antiderivative is computed and ``self.extrapolate='periodic'``,
it will be set to False for the returned instance. This is done because
the antiderivative is no longer periodic and its correct evaluation
outside of the initially given x interval is difficult.
*)

val construct_fast : ?extrapolate:Py.Object.t -> ?axis:Py.Object.t -> c:Py.Object.t -> x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Construct the piecewise polynomial without making checks.

Takes the same parameters as the constructor. Input arguments
``c`` and ``x`` must be arrays of the correct shape and type.  The
``c`` array can only be of dtypes float and complex, and ``x``
array must have dtype float.
*)

val derivative : ?nu:int -> [> tag] Obj.t -> Py.Object.t
(**
Construct a new piecewise polynomial representing the derivative.

Parameters
----------
nu : int, optional
    Order of derivative to evaluate. Default is 1, i.e. compute the
    first derivative. If negative, the antiderivative is returned.

Returns
-------
pp : PPoly
    Piecewise polynomial of order k2 = k - n representing the derivative
    of this polynomial.

Notes
-----
Derivatives are evaluated piecewise for each polynomial
segment, even if the polynomial is not differentiable at the
breakpoints. The polynomial intervals are considered half-open,
``[a, b)``, except for the last interval which is closed
``[a, b]``.
*)

val extend : ?right:Py.Object.t -> c:[`Ndarray of [>`Ndarray] Np.Obj.t | `Size_k_m_ of Py.Object.t] -> x:[`Ndarray of [>`Ndarray] Np.Obj.t | `Size of Py.Object.t] -> [> tag] Obj.t -> Py.Object.t
(**
Add additional breakpoints and coefficients to the polynomial.

Parameters
----------
c : ndarray, size (k, m, ...)
    Additional coefficients for polynomials in intervals. Note that
    the first additional interval will be formed using one of the
    ``self.x`` end points.
x : ndarray, size (m,)
    Additional breakpoints. Must be sorted in the same order as
    ``self.x`` and either to the right or to the left of the current
    breakpoints.
right
    Deprecated argument. Has no effect.

    .. deprecated:: 0.19
*)

val from_bernstein_basis : ?extrapolate:[`Bool of bool | `Periodic] -> bp:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Construct a piecewise polynomial in the power basis
from a polynomial in Bernstein basis.

Parameters
----------
bp : BPoly
    A Bernstein basis polynomial, as created by BPoly
extrapolate : bool or 'periodic', optional
    If bool, determines whether to extrapolate to out-of-bounds points
    based on first and last intervals, or to return NaNs.
    If 'periodic', periodic extrapolation is used. Default is True.
*)

val from_spline : ?extrapolate:[`Bool of bool | `Periodic] -> tck:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Construct a piecewise polynomial from a spline

Parameters
----------
tck
    A spline, as returned by `splrep` or a BSpline object.
extrapolate : bool or 'periodic', optional
    If bool, determines whether to extrapolate to out-of-bounds points
    based on first and last intervals, or to return NaNs.
    If 'periodic', periodic extrapolation is used. Default is True.
*)

val integrate : ?extrapolate:[`Bool of bool | `Periodic] -> a:float -> b:float -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute a definite integral over a piecewise polynomial.

Parameters
----------
a : float
    Lower integration bound
b : float
    Upper integration bound
extrapolate : {bool, 'periodic', None}, optional
    If bool, determines whether to extrapolate to out-of-bounds points
    based on first and last intervals, or to return NaNs.
    If 'periodic', periodic extrapolation is used.
    If None (default), use `self.extrapolate`.

Returns
-------
ig : array_like
    Definite integral of the piecewise polynomial over [a, b]
*)

val roots : ?discontinuity:bool -> ?extrapolate:[`Bool of bool | `Periodic] -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Find real roots of the the piecewise polynomial.

Parameters
----------
discontinuity : bool, optional
    Whether to report sign changes across discontinuities at
    breakpoints as roots.
extrapolate : {bool, 'periodic', None}, optional
    If bool, determines whether to return roots from the polynomial
    extrapolated based on first and last intervals, 'periodic' works
    the same as False. If None (default), use `self.extrapolate`.

Returns
-------
roots : ndarray
    Roots of the polynomial(s).

    If the PPoly object describes multiple polynomials, the
    return value is an object array whose each element is an
    ndarray containing the roots.

See Also
--------
PPoly.solve
*)

val solve : ?y:float -> ?discontinuity:bool -> ?extrapolate:[`Bool of bool | `Periodic] -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Find real solutions of the the equation ``pp(x) == y``.

Parameters
----------
y : float, optional
    Right-hand side. Default is zero.
discontinuity : bool, optional
    Whether to report sign changes across discontinuities at
    breakpoints as roots.
extrapolate : {bool, 'periodic', None}, optional
    If bool, determines whether to return roots from the polynomial
    extrapolated based on first and last intervals, 'periodic' works
    the same as False. If None (default), use `self.extrapolate`.

Returns
-------
roots : ndarray
    Roots of the polynomial(s).

    If the PPoly object describes multiple polynomials, the
    return value is an object array whose each element is an
    ndarray containing the roots.

Notes
-----
This routine works only on real-valued polynomials.

If the piecewise polynomial contains sections that are
identically zero, the root list will contain the start point
of the corresponding interval, followed by a ``nan`` value.

If the polynomial is discontinuous across a breakpoint, and
there is a sign change across the breakpoint, this is reported
if the `discont` parameter is True.

Examples
--------

Finding roots of ``[x**2 - 1, (x - 1)**2]`` defined on intervals
``[-2, 1], [1, 2]``:

>>> from scipy.interpolate import PPoly
>>> pp = PPoly(np.array([[1, -4, 3], [1, 0, 0]]).T, [-2, 1, 2])
>>> pp.solve()
array([-1.,  1.])
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Rbf : sig
type tag = [`Rbf]
type t = [`Object | `Rbf] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> t
(**
Rbf( *args)

A class for radial basis function interpolation of functions from
n-dimensional scattered data to an m-dimensional domain.

Parameters
----------
*args : arrays
    x, y, z, ..., d, where x, y, z, ... are the coordinates of the nodes
    and d is the array of values at the nodes
function : str or callable, optional
    The radial basis function, based on the radius, r, given by the norm
    (default is Euclidean distance); the default is 'multiquadric'::

        'multiquadric': sqrt((r/self.epsilon)**2 + 1)
        'inverse': 1.0/sqrt((r/self.epsilon)**2 + 1)
        'gaussian': exp(-(r/self.epsilon)**2)
        'linear': r
        'cubic': r**3
        'quintic': r**5
        'thin_plate': r**2 * log(r)

    If callable, then it must take 2 arguments (self, r).  The epsilon
    parameter will be available as self.epsilon.  Other keyword
    arguments passed in will be available as well.

epsilon : float, optional
    Adjustable constant for gaussian or multiquadrics functions
    - defaults to approximate average distance between nodes (which is
    a good start).
smooth : float, optional
    Values greater than zero increase the smoothness of the
    approximation.  0 is for interpolation (default), the function will
    always go through the nodal points in this case.
norm : str, callable, optional
    A function that returns the 'distance' between two points, with
    inputs as arrays of positions (x, y, z, ...), and an output as an
    array of distance. E.g., the default: 'euclidean', such that the result
    is a matrix of the distances from each point in ``x1`` to each point in
    ``x2``. For more options, see documentation of
    `scipy.spatial.distances.cdist`.
mode : str, optional
    Mode of the interpolation, can be '1-D' (default) or 'N-D'. When it is
    '1-D' the data `d` will be considered as one-dimensional and flattened
    internally. When it is 'N-D' the data `d` is assumed to be an array of
    shape (n_samples, m), where m is the dimension of the target domain.


Attributes
----------
N : int
    The number of data points (as determined by the input arrays).
di : ndarray
    The 1-D array of data values at each of the data coordinates `xi`.
xi : ndarray
    The 2-D array of data coordinates.
function : str or callable
    The radial basis function.  See description under Parameters.
epsilon : float
    Parameter used by gaussian or multiquadrics functions.  See Parameters.
smooth : float
    Smoothing parameter.  See description under Parameters.
norm : str or callable
    The distance function.  See description under Parameters.
mode : str
    Mode of the interpolation.  See description under Parameters.
nodes : ndarray
    A 1-D array of node values for the interpolation.
A : internal property, do not use

Examples
--------
>>> from scipy.interpolate import Rbf
>>> x, y, z, d = np.random.rand(4, 50)
>>> rbfi = Rbf(x, y, z, d)  # radial basis function interpolator instance
>>> xi = yi = zi = np.linspace(0, 1, 20)
>>> di = rbfi(xi, yi, zi)   # interpolated values
>>> di.shape
(20,)
*)


(** Attribute N: get value or raise Not_found if None.*)
val n : t -> int

(** Attribute N: get value as an option. *)
val n_opt : t -> (int) option


(** Attribute di: get value or raise Not_found if None.*)
val di : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute di: get value as an option. *)
val di_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute xi: get value or raise Not_found if None.*)
val xi : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute xi: get value as an option. *)
val xi_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute function: get value or raise Not_found if None.*)
val function_ : t -> Py.Object.t

(** Attribute function: get value as an option. *)
val function_opt : t -> (Py.Object.t) option


(** Attribute epsilon: get value or raise Not_found if None.*)
val epsilon : t -> float

(** Attribute epsilon: get value as an option. *)
val epsilon_opt : t -> (float) option


(** Attribute smooth: get value or raise Not_found if None.*)
val smooth : t -> float

(** Attribute smooth: get value as an option. *)
val smooth_opt : t -> (float) option


(** Attribute norm: get value or raise Not_found if None.*)
val norm : t -> Py.Object.t

(** Attribute norm: get value as an option. *)
val norm_opt : t -> (Py.Object.t) option


(** Attribute mode: get value or raise Not_found if None.*)
val mode : t -> string

(** Attribute mode: get value as an option. *)
val mode_opt : t -> (string) option


(** Attribute nodes: get value or raise Not_found if None.*)
val nodes : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute nodes: get value as an option. *)
val nodes_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute A: get value or raise Not_found if None.*)
val a : t -> Py.Object.t

(** Attribute A: get value as an option. *)
val a_opt : t -> (Py.Object.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module RectBivariateSpline : sig
type tag = [`RectBivariateSpline]
type t = [`Object | `RectBivariateSpline] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?bbox:[>`Ndarray] Np.Obj.t -> ?kx:Py.Object.t -> ?ky:Py.Object.t -> ?s:float -> x:Py.Object.t -> y:Py.Object.t -> z:[>`Ndarray] Np.Obj.t -> unit -> t
(**
Bivariate spline approximation over a rectangular mesh.

Can be used for both smoothing and interpolating data.

Parameters
----------
x,y : array_like
    1-D arrays of coordinates in strictly ascending order.
z : array_like
    2-D array of data with shape (x.size,y.size).
bbox : array_like, optional
    Sequence of length 4 specifying the boundary of the rectangular
    approximation domain.  By default,
    ``bbox=[min(x,tx),max(x,tx), min(y,ty),max(y,ty)]``.
kx, ky : ints, optional
    Degrees of the bivariate spline. Default is 3.
s : float, optional
    Positive smoothing factor defined for estimation condition:
    ``sum((w[i]*(z[i]-s(x[i], y[i])))**2, axis=0) <= s``
    Default is ``s=0``, which is for interpolation.

See Also
--------
SmoothBivariateSpline : a smoothing bivariate spline for scattered data
bisplrep : an older wrapping of FITPACK
bisplev : an older wrapping of FITPACK
UnivariateSpline : a similar class for univariate spline interpolation
*)

val ev : ?dx:int -> ?dy:int -> xi:Py.Object.t -> yi:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Evaluate the spline at points

Returns the interpolated value at ``(xi[i], yi[i]),
i=0,...,len(xi)-1``.

Parameters
----------
xi, yi : array_like
    Input coordinates. Standard Numpy broadcasting is obeyed.
dx : int, optional
    Order of x-derivative

    .. versionadded:: 0.14.0
dy : int, optional
    Order of y-derivative

    .. versionadded:: 0.14.0
*)

val get_coeffs : [> tag] Obj.t -> Py.Object.t
(**
Return spline coefficients.
*)

val get_knots : [> tag] Obj.t -> Py.Object.t
(**
Return a tuple (tx,ty) where tx,ty contain knots positions
of the spline with respect to x-, y-variable, respectively.
The position of interior and additional knots are given as
t[k+1:-k-1] and t[:k+1]=b, t[-k-1:]=e, respectively.
*)

val get_residual : [> tag] Obj.t -> Py.Object.t
(**
Return weighted sum of squared residuals of the spline
approximation: sum ((w[i]*(z[i]-s(x[i],y[i])))**2,axis=0)
*)

val integral : xa:Py.Object.t -> xb:Py.Object.t -> ya:Py.Object.t -> yb:Py.Object.t -> [> tag] Obj.t -> float
(**
Evaluate the integral of the spline over area [xa,xb] x [ya,yb].

Parameters
----------
xa, xb : float
    The end-points of the x integration interval.
ya, yb : float
    The end-points of the y integration interval.

Returns
-------
integ : float
    The value of the resulting integral.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module RectSphereBivariateSpline : sig
type tag = [`RectSphereBivariateSpline]
type t = [`Object | `RectSphereBivariateSpline] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?s:float -> ?pole_continuity:[`T_bool_bool_ of Py.Object.t | `Bool of bool] -> ?pole_values:[`F of float | `Tuple of (float * float)] -> ?pole_exact:[`T_bool_bool_ of Py.Object.t | `Bool of bool] -> ?pole_flat:[`T_bool_bool_ of Py.Object.t | `Bool of bool] -> u:[>`Ndarray] Np.Obj.t -> v:[>`Ndarray] Np.Obj.t -> r:[>`Ndarray] Np.Obj.t -> unit -> t
(**
Bivariate spline approximation over a rectangular mesh on a sphere.

Can be used for smoothing data.

.. versionadded:: 0.11.0

Parameters
----------
u : array_like
    1-D array of latitude coordinates in strictly ascending order.
    Coordinates must be given in radians and lie within the interval
    (0, pi).
v : array_like
    1-D array of longitude coordinates in strictly ascending order.
    Coordinates must be given in radians. First element (v[0]) must lie
    within the interval [-pi, pi). Last element (v[-1]) must satisfy
    v[-1] <= v[0] + 2*pi.
r : array_like
    2-D array of data with shape ``(u.size, v.size)``.
s : float, optional
    Positive smoothing factor defined for estimation condition
    (``s=0`` is for interpolation).
pole_continuity : bool or (bool, bool), optional
    Order of continuity at the poles ``u=0`` (``pole_continuity[0]``) and
    ``u=pi`` (``pole_continuity[1]``).  The order of continuity at the pole
    will be 1 or 0 when this is True or False, respectively.
    Defaults to False.
pole_values : float or (float, float), optional
    Data values at the poles ``u=0`` and ``u=pi``.  Either the whole
    parameter or each individual element can be None.  Defaults to None.
pole_exact : bool or (bool, bool), optional
    Data value exactness at the poles ``u=0`` and ``u=pi``.  If True, the
    value is considered to be the right function value, and it will be
    fitted exactly. If False, the value will be considered to be a data
    value just like the other data values.  Defaults to False.
pole_flat : bool or (bool, bool), optional
    For the poles at ``u=0`` and ``u=pi``, specify whether or not the
    approximation has vanishing derivatives.  Defaults to False.

See Also
--------
RectBivariateSpline : bivariate spline approximation over a rectangular
    mesh

Notes
-----
Currently, only the smoothing spline approximation (``iopt[0] = 0`` and
``iopt[0] = 1`` in the FITPACK routine) is supported.  The exact
least-squares spline approximation is not implemented yet.

When actually performing the interpolation, the requested `v` values must
lie within the same length 2pi interval that the original `v` values were
chosen from.

For more information, see the FITPACK_ site about this function.

.. _FITPACK: http://www.netlib.org/dierckx/spgrid.f

Examples
--------
Suppose we have global data on a coarse grid

>>> lats = np.linspace(10, 170, 9) * np.pi / 180.
>>> lons = np.linspace(0, 350, 18) * np.pi / 180.
>>> data = np.dot(np.atleast_2d(90. - np.linspace(-80., 80., 18)).T,
...               np.atleast_2d(180. - np.abs(np.linspace(0., 350., 9)))).T

We want to interpolate it to a global one-degree grid

>>> new_lats = np.linspace(1, 180, 180) * np.pi / 180
>>> new_lons = np.linspace(1, 360, 360) * np.pi / 180
>>> new_lats, new_lons = np.meshgrid(new_lats, new_lons)

We need to set up the interpolator object

>>> from scipy.interpolate import RectSphereBivariateSpline
>>> lut = RectSphereBivariateSpline(lats, lons, data)

Finally we interpolate the data.  The `RectSphereBivariateSpline` object
only takes 1-D arrays as input, therefore we need to do some reshaping.

>>> data_interp = lut.ev(new_lats.ravel(),
...                      new_lons.ravel()).reshape((360, 180)).T

Looking at the original and the interpolated data, one can see that the
interpolant reproduces the original data very well:

>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> ax1 = fig.add_subplot(211)
>>> ax1.imshow(data, interpolation='nearest')
>>> ax2 = fig.add_subplot(212)
>>> ax2.imshow(data_interp, interpolation='nearest')
>>> plt.show()

Choosing the optimal value of ``s`` can be a delicate task. Recommended
values for ``s`` depend on the accuracy of the data values.  If the user
has an idea of the statistical errors on the data, she can also find a
proper estimate for ``s``. By assuming that, if she specifies the
right ``s``, the interpolator will use a spline ``f(u,v)`` which exactly
reproduces the function underlying the data, she can evaluate
``sum((r(i,j)-s(u(i),v(j)))**2)`` to find a good estimate for this ``s``.
For example, if she knows that the statistical errors on her
``r(i,j)``-values are not greater than 0.1, she may expect that a good
``s`` should have a value not larger than ``u.size * v.size * (0.1)**2``.

If nothing is known about the statistical error in ``r(i,j)``, ``s`` must
be determined by trial and error.  The best is then to start with a very
large value of ``s`` (to determine the least-squares polynomial and the
corresponding upper bound ``fp0`` for ``s``) and then to progressively
decrease the value of ``s`` (say by a factor 10 in the beginning, i.e.
``s = fp0 / 10, fp0 / 100, ...``  and more carefully as the approximation
shows more detail) to obtain closer fits.

The interpolation results for different values of ``s`` give some insight
into this process:

>>> fig2 = plt.figure()
>>> s = [3e9, 2e9, 1e9, 1e8]
>>> for ii in range(len(s)):
...     lut = RectSphereBivariateSpline(lats, lons, data, s=s[ii])
...     data_interp = lut.ev(new_lats.ravel(),
...                          new_lons.ravel()).reshape((360, 180)).T
...     ax = fig2.add_subplot(2, 2, ii+1)
...     ax.imshow(data_interp, interpolation='nearest')
...     ax.set_title('s = %g' % s[ii])
>>> plt.show()
*)

val ev : ?dtheta:int -> ?dphi:int -> theta:Py.Object.t -> phi:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Evaluate the spline at points

Returns the interpolated value at ``(theta[i], phi[i]),
i=0,...,len(theta)-1``.

Parameters
----------
theta, phi : array_like
    Input coordinates. Standard Numpy broadcasting is obeyed.
dtheta : int, optional
    Order of theta-derivative

    .. versionadded:: 0.14.0
dphi : int, optional
    Order of phi-derivative

    .. versionadded:: 0.14.0
*)

val get_coeffs : [> tag] Obj.t -> Py.Object.t
(**
Return spline coefficients.
*)

val get_knots : [> tag] Obj.t -> Py.Object.t
(**
Return a tuple (tx,ty) where tx,ty contain knots positions
of the spline with respect to x-, y-variable, respectively.
The position of interior and additional knots are given as
t[k+1:-k-1] and t[:k+1]=b, t[-k-1:]=e, respectively.
*)

val get_residual : [> tag] Obj.t -> Py.Object.t
(**
Return weighted sum of squared residuals of the spline
approximation: sum ((w[i]*(z[i]-s(x[i],y[i])))**2,axis=0)
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module RegularGridInterpolator : sig
type tag = [`RegularGridInterpolator]
type t = [`Object | `RegularGridInterpolator] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?method_:string -> ?bounds_error:bool -> ?fill_value:[`F of float | `I of int] -> points:Py.Object.t -> values:[>`Ndarray] Np.Obj.t -> unit -> t
(**
Interpolation on a regular grid in arbitrary dimensions

The data must be defined on a regular grid; the grid spacing however may be
uneven.  Linear and nearest-neighbour interpolation are supported. After
setting up the interpolator object, the interpolation method ( *linear* or
*nearest* ) may be chosen at each evaluation.

Parameters
----------
points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
    The points defining the regular grid in n dimensions.

values : array_like, shape (m1, ..., mn, ...)
    The data on the regular grid in n dimensions.

method : str, optional
    The method of interpolation to perform. Supported are 'linear' and
    'nearest'. This parameter will become the default for the object's
    ``__call__`` method. Default is 'linear'.

bounds_error : bool, optional
    If True, when interpolated values are requested outside of the
    domain of the input data, a ValueError is raised.
    If False, then `fill_value` is used.

fill_value : number, optional
    If provided, the value to use for points outside of the
    interpolation domain. If None, values outside
    the domain are extrapolated.

Methods
-------
__call__

Notes
-----
Contrary to LinearNDInterpolator and NearestNDInterpolator, this class
avoids expensive triangulation of the input data by taking advantage of the
regular grid structure.

If any of `points` have a dimension of size 1, linear interpolation will
return an array of `nan` values. Nearest-neighbor interpolation will work
as usual in this case.

.. versionadded:: 0.14

Examples
--------
Evaluate a simple example function on the points of a 3D grid:

>>> from scipy.interpolate import RegularGridInterpolator
>>> def f(x, y, z):
...     return 2 * x**3 + 3 * y**2 - z
>>> x = np.linspace(1, 4, 11)
>>> y = np.linspace(4, 7, 22)
>>> z = np.linspace(7, 9, 33)
>>> data = f( *np.meshgrid(x, y, z, indexing='ij', sparse=True))

``data`` is now a 3D array with ``data[i,j,k] = f(x[i], y[j], z[k])``.
Next, define an interpolating function from this data:

>>> my_interpolating_function = RegularGridInterpolator((x, y, z), data)

Evaluate the interpolating function at the two points
``(x,y,z) = (2.1, 6.2, 8.3)`` and ``(3.3, 5.2, 7.1)``:

>>> pts = np.array([[2.1, 6.2, 8.3], [3.3, 5.2, 7.1]])
>>> my_interpolating_function(pts)
array([ 125.80469388,  146.30069388])

which is indeed a close approximation to
``[f(2.1, 6.2, 8.3), f(3.3, 5.2, 7.1)]``.

See also
--------
NearestNDInterpolator : Nearest neighbour interpolation on unstructured
                        data in N dimensions

LinearNDInterpolator : Piecewise linear interpolant on unstructured data
                       in N dimensions

References
----------
.. [1] Python package *regulargrid* by Johannes Buchner, see
       https://pypi.python.org/pypi/regulargrid/
.. [2] Wikipedia, 'Trilinear interpolation',
       https://en.wikipedia.org/wiki/Trilinear_interpolation
.. [3] Weiser, Alan, and Sergio E. Zarantonello. 'A note on piecewise linear
       and multilinear table interpolation in many dimensions.' MATH.
       COMPUT. 50.181 (1988): 189-196.
       https://www.ams.org/journals/mcom/1988-50-181/S0025-5718-1988-0917826-0/S0025-5718-1988-0917826-0.pdf
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module SmoothBivariateSpline : sig
type tag = [`SmoothBivariateSpline]
type t = [`Object | `SmoothBivariateSpline] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?w:[>`Ndarray] Np.Obj.t -> ?bbox:[>`Ndarray] Np.Obj.t -> ?kx:Py.Object.t -> ?ky:Py.Object.t -> ?s:float -> ?eps:float -> x:Py.Object.t -> y:Py.Object.t -> z:Py.Object.t -> unit -> t
(**
Smooth bivariate spline approximation.

Parameters
----------
x, y, z : array_like
    1-D sequences of data points (order is not important).
w : array_like, optional
    Positive 1-D sequence of weights, of same length as `x`, `y` and `z`.
bbox : array_like, optional
    Sequence of length 4 specifying the boundary of the rectangular
    approximation domain.  By default,
    ``bbox=[min(x,tx),max(x,tx), min(y,ty),max(y,ty)]``.
kx, ky : ints, optional
    Degrees of the bivariate spline. Default is 3.
s : float, optional
    Positive smoothing factor defined for estimation condition:
    ``sum((w[i]*(z[i]-s(x[i], y[i])))**2, axis=0) <= s``
    Default ``s=len(w)`` which should be a good value if ``1/w[i]`` is an
    estimate of the standard deviation of ``z[i]``.
eps : float, optional
    A threshold for determining the effective rank of an over-determined
    linear system of equations. `eps` should have a value between 0 and 1,
    the default is 1e-16.

See Also
--------
bisplrep : an older wrapping of FITPACK
bisplev : an older wrapping of FITPACK
UnivariateSpline : a similar class for univariate spline interpolation
LSQUnivariateSpline : to create a BivariateSpline using weighted

Notes
-----
The length of `x`, `y` and `z` should be at least ``(kx+1) * (ky+1)``.
*)

val ev : ?dx:int -> ?dy:int -> xi:Py.Object.t -> yi:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Evaluate the spline at points

Returns the interpolated value at ``(xi[i], yi[i]),
i=0,...,len(xi)-1``.

Parameters
----------
xi, yi : array_like
    Input coordinates. Standard Numpy broadcasting is obeyed.
dx : int, optional
    Order of x-derivative

    .. versionadded:: 0.14.0
dy : int, optional
    Order of y-derivative

    .. versionadded:: 0.14.0
*)

val get_coeffs : [> tag] Obj.t -> Py.Object.t
(**
Return spline coefficients.
*)

val get_knots : [> tag] Obj.t -> Py.Object.t
(**
Return a tuple (tx,ty) where tx,ty contain knots positions
of the spline with respect to x-, y-variable, respectively.
The position of interior and additional knots are given as
t[k+1:-k-1] and t[:k+1]=b, t[-k-1:]=e, respectively.
*)

val get_residual : [> tag] Obj.t -> Py.Object.t
(**
Return weighted sum of squared residuals of the spline
approximation: sum ((w[i]*(z[i]-s(x[i],y[i])))**2,axis=0)
*)

val integral : xa:Py.Object.t -> xb:Py.Object.t -> ya:Py.Object.t -> yb:Py.Object.t -> [> tag] Obj.t -> float
(**
Evaluate the integral of the spline over area [xa,xb] x [ya,yb].

Parameters
----------
xa, xb : float
    The end-points of the x integration interval.
ya, yb : float
    The end-points of the y integration interval.

Returns
-------
integ : float
    The value of the resulting integral.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module SmoothSphereBivariateSpline : sig
type tag = [`SmoothSphereBivariateSpline]
type t = [`Object | `SmoothSphereBivariateSpline] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?w:[>`Ndarray] Np.Obj.t -> ?s:float -> ?eps:float -> theta:Py.Object.t -> phi:Py.Object.t -> r:Py.Object.t -> unit -> t
(**
Smooth bivariate spline approximation in spherical coordinates.

.. versionadded:: 0.11.0

Parameters
----------
theta, phi, r : array_like
    1-D sequences of data points (order is not important). Coordinates
    must be given in radians. Theta must lie within the interval (0, pi),
    and phi must lie within the interval (0, 2pi).
w : array_like, optional
    Positive 1-D sequence of weights.
s : float, optional
    Positive smoothing factor defined for estimation condition:
    ``sum((w(i)*(r(i) - s(theta(i), phi(i))))**2, axis=0) <= s``
    Default ``s=len(w)`` which should be a good value if 1/w[i] is an
    estimate of the standard deviation of r[i].
eps : float, optional
    A threshold for determining the effective rank of an over-determined
    linear system of equations. `eps` should have a value between 0 and 1,
    the default is 1e-16.

Notes
-----
For more information, see the FITPACK_ site about this function.

.. _FITPACK: http://www.netlib.org/dierckx/sphere.f

Examples
--------
Suppose we have global data on a coarse grid (the input data does not
have to be on a grid):

>>> theta = np.linspace(0., np.pi, 7)
>>> phi = np.linspace(0., 2*np.pi, 9)
>>> data = np.empty((theta.shape[0], phi.shape[0]))
>>> data[:,0], data[0,:], data[-1,:] = 0., 0., 0.
>>> data[1:-1,1], data[1:-1,-1] = 1., 1.
>>> data[1,1:-1], data[-2,1:-1] = 1., 1.
>>> data[2:-2,2], data[2:-2,-2] = 2., 2.
>>> data[2,2:-2], data[-3,2:-2] = 2., 2.
>>> data[3,3:-2] = 3.
>>> data = np.roll(data, 4, 1)

We need to set up the interpolator object

>>> lats, lons = np.meshgrid(theta, phi)
>>> from scipy.interpolate import SmoothSphereBivariateSpline
>>> lut = SmoothSphereBivariateSpline(lats.ravel(), lons.ravel(),
...                                   data.T.ravel(), s=3.5)

As a first test, we'll see what the algorithm returns when run on the
input coordinates

>>> data_orig = lut(theta, phi)

Finally we interpolate the data to a finer grid

>>> fine_lats = np.linspace(0., np.pi, 70)
>>> fine_lons = np.linspace(0., 2 * np.pi, 90)

>>> data_smth = lut(fine_lats, fine_lons)

>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> ax1 = fig.add_subplot(131)
>>> ax1.imshow(data, interpolation='nearest')
>>> ax2 = fig.add_subplot(132)
>>> ax2.imshow(data_orig, interpolation='nearest')
>>> ax3 = fig.add_subplot(133)
>>> ax3.imshow(data_smth, interpolation='nearest')
>>> plt.show()
*)

val ev : ?dtheta:int -> ?dphi:int -> theta:Py.Object.t -> phi:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Evaluate the spline at points

Returns the interpolated value at ``(theta[i], phi[i]),
i=0,...,len(theta)-1``.

Parameters
----------
theta, phi : array_like
    Input coordinates. Standard Numpy broadcasting is obeyed.
dtheta : int, optional
    Order of theta-derivative

    .. versionadded:: 0.14.0
dphi : int, optional
    Order of phi-derivative

    .. versionadded:: 0.14.0
*)

val get_coeffs : [> tag] Obj.t -> Py.Object.t
(**
Return spline coefficients.
*)

val get_knots : [> tag] Obj.t -> Py.Object.t
(**
Return a tuple (tx,ty) where tx,ty contain knots positions
of the spline with respect to x-, y-variable, respectively.
The position of interior and additional knots are given as
t[k+1:-k-1] and t[:k+1]=b, t[-k-1:]=e, respectively.
*)

val get_residual : [> tag] Obj.t -> Py.Object.t
(**
Return weighted sum of squared residuals of the spline
approximation: sum ((w[i]*(z[i]-s(x[i],y[i])))**2,axis=0)
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module UnivariateSpline : sig
type tag = [`UnivariateSpline]
type t = [`Object | `UnivariateSpline] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?w:[>`Ndarray] Np.Obj.t -> ?bbox:Py.Object.t -> ?k:int -> ?s:float -> ?ext:[`S of string | `I of int] -> ?check_finite:bool -> x:[>`Ndarray] Np.Obj.t -> y:[>`Ndarray] Np.Obj.t -> unit -> t
(**
One-dimensional smoothing spline fit to a given set of data points.

Fits a spline y = spl(x) of degree `k` to the provided `x`, `y` data.  `s`
specifies the number of knots by specifying a smoothing condition.

Parameters
----------
x : (N,) array_like
    1-D array of independent input data. Must be increasing;
    must be strictly increasing if `s` is 0.
y : (N,) array_like
    1-D array of dependent input data, of the same length as `x`.
w : (N,) array_like, optional
    Weights for spline fitting.  Must be positive.  If None (default),
    weights are all equal.
bbox : (2,) array_like, optional
    2-sequence specifying the boundary of the approximation interval. If
    None (default), ``bbox=[x[0], x[-1]]``.
k : int, optional
    Degree of the smoothing spline.  Must be <= 5.
    Default is k=3, a cubic spline.
s : float or None, optional
    Positive smoothing factor used to choose the number of knots.  Number
    of knots will be increased until the smoothing condition is satisfied::

        sum((w[i] * (y[i]-spl(x[i])))**2, axis=0) <= s

    If None (default), ``s = len(w)`` which should be a good value if
    ``1/w[i]`` is an estimate of the standard deviation of ``y[i]``.
    If 0, spline will interpolate through all data points.
ext : int or str, optional
    Controls the extrapolation mode for elements
    not in the interval defined by the knot sequence.

    * if ext=0 or 'extrapolate', return the extrapolated value.
    * if ext=1 or 'zeros', return 0
    * if ext=2 or 'raise', raise a ValueError
    * if ext=3 of 'const', return the boundary value.

    The default value is 0.

check_finite : bool, optional
    Whether to check that the input arrays contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination or non-sensical results) if the inputs
    do contain infinities or NaNs.
    Default is False.

See Also
--------
InterpolatedUnivariateSpline : Subclass with smoothing forced to 0
LSQUnivariateSpline : Subclass in which knots are user-selected instead of
    being set by smoothing condition
splrep : An older, non object-oriented wrapping of FITPACK
splev, sproot, splint, spalde
BivariateSpline : A similar class for two-dimensional spline interpolation

Notes
-----
The number of data points must be larger than the spline degree `k`.

**NaN handling**: If the input arrays contain ``nan`` values, the result
is not useful, since the underlying spline fitting routines cannot deal
with ``nan`` . A workaround is to use zero weights for not-a-number
data points:

>>> from scipy.interpolate import UnivariateSpline
>>> x, y = np.array([1, 2, 3, 4]), np.array([1, np.nan, 3, 4])
>>> w = np.isnan(y)
>>> y[w] = 0.
>>> spl = UnivariateSpline(x, y, w=~w)

Notice the need to replace a ``nan`` by a numerical value (precise value
does not matter as long as the corresponding weight is zero.)

Examples
--------
>>> import matplotlib.pyplot as plt
>>> from scipy.interpolate import UnivariateSpline
>>> x = np.linspace(-3, 3, 50)
>>> y = np.exp(-x**2) + 0.1 * np.random.randn(50)
>>> plt.plot(x, y, 'ro', ms=5)

Use the default value for the smoothing parameter:

>>> spl = UnivariateSpline(x, y)
>>> xs = np.linspace(-3, 3, 1000)
>>> plt.plot(xs, spl(xs), 'g', lw=3)

Manually change the amount of smoothing:

>>> spl.set_smoothing_factor(0.5)
>>> plt.plot(xs, spl(xs), 'b', lw=3)
>>> plt.show()
*)

val antiderivative : ?n:int -> [> tag] Obj.t -> Py.Object.t
(**
Construct a new spline representing the antiderivative of this spline.

Parameters
----------
n : int, optional
    Order of antiderivative to evaluate. Default: 1

Returns
-------
spline : UnivariateSpline
    Spline of order k2=k+n representing the antiderivative of this
    spline.

Notes
-----

.. versionadded:: 0.13.0

See Also
--------
splantider, derivative

Examples
--------
>>> from scipy.interpolate import UnivariateSpline
>>> x = np.linspace(0, np.pi/2, 70)
>>> y = 1 / np.sqrt(1 - 0.8*np.sin(x)**2)
>>> spl = UnivariateSpline(x, y, s=0)

The derivative is the inverse operation of the antiderivative,
although some floating point error accumulates:

>>> spl(1.7), spl.antiderivative().derivative()(1.7)
(array(2.1565429877197317), array(2.1565429877201865))

Antiderivative can be used to evaluate definite integrals:

>>> ispl = spl.antiderivative()
>>> ispl(np.pi/2) - ispl(0)
2.2572053588768486

This is indeed an approximation to the complete elliptic integral
:math:`K(m) = \int_0^{\pi/2} [1 - m\sin^2 x]^{-1/2} dx`:

>>> from scipy.special import ellipk
>>> ellipk(0.8)
2.2572053268208538
*)

val derivative : ?n:int -> [> tag] Obj.t -> Py.Object.t
(**
Construct a new spline representing the derivative of this spline.

Parameters
----------
n : int, optional
    Order of derivative to evaluate. Default: 1

Returns
-------
spline : UnivariateSpline
    Spline of order k2=k-n representing the derivative of this
    spline.

See Also
--------
splder, antiderivative

Notes
-----

.. versionadded:: 0.13.0

Examples
--------
This can be used for finding maxima of a curve:

>>> from scipy.interpolate import UnivariateSpline
>>> x = np.linspace(0, 10, 70)
>>> y = np.sin(x)
>>> spl = UnivariateSpline(x, y, k=4, s=0)

Now, differentiate the spline and find the zeros of the
derivative. (NB: `sproot` only works for order 3 splines, so we
fit an order 4 spline):

>>> spl.derivative().roots() / np.pi
array([ 0.50000001,  1.5       ,  2.49999998])

This agrees well with roots :math:`\pi/2 + n\pi` of
:math:`\cos(x) = \sin'(x)`.
*)

val derivatives : x:float -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return all derivatives of the spline at the point x.

Parameters
----------
x : float
    The point to evaluate the derivatives at.

Returns
-------
der : ndarray, shape(k+1,)
    Derivatives of the orders 0 to k.

Examples
--------
>>> from scipy.interpolate import UnivariateSpline
>>> x = np.linspace(0, 3, 11)
>>> y = x**2
>>> spl = UnivariateSpline(x, y)
>>> spl.derivatives(1.5)
array([2.25, 3.0, 2.0, 0])
*)

val get_coeffs : [> tag] Obj.t -> Py.Object.t
(**
Return spline coefficients.
*)

val get_knots : [> tag] Obj.t -> Py.Object.t
(**
Return positions of interior knots of the spline.

Internally, the knot vector contains ``2*k`` additional boundary knots.
*)

val get_residual : [> tag] Obj.t -> Py.Object.t
(**
Return weighted sum of squared residuals of the spline approximation.

This is equivalent to::

     sum((w[i] * (y[i]-spl(x[i])))**2, axis=0)
*)

val integral : a:float -> b:float -> [> tag] Obj.t -> float
(**
Return definite integral of the spline between two given points.

Parameters
----------
a : float
    Lower limit of integration.
b : float
    Upper limit of integration.

Returns
-------
integral : float
    The value of the definite integral of the spline between limits.

Examples
--------
>>> from scipy.interpolate import UnivariateSpline
>>> x = np.linspace(0, 3, 11)
>>> y = x**2
>>> spl = UnivariateSpline(x, y)
>>> spl.integral(0, 3)
9.0

which agrees with :math:`\int x^2 dx = x^3 / 3` between the limits
of 0 and 3.

A caveat is that this routine assumes the spline to be zero outside of
the data limits:

>>> spl.integral(-1, 4)
9.0
>>> spl.integral(-1, 0)
0.0
*)

val roots : [> tag] Obj.t -> Py.Object.t
(**
Return the zeros of the spline.

Restriction: only cubic splines are supported by fitpack.
*)

val set_smoothing_factor : s:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Continue spline computation with the given smoothing
factor s and with the knots found at the last call.

This routine modifies the spline in place.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Interp1d : sig
type tag = [`Interp1d]
type t = [`Interp1d | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?kind:[`I of int | `S of string] -> ?axis:int -> ?copy:bool -> ?bounds_error:bool -> ?fill_value:[`Ndarray of [>`Ndarray] Np.Obj.t | `Extrapolate | `T_array_like_array_like_ of Py.Object.t] -> ?assume_sorted:bool -> x:[>`Ndarray] Np.Obj.t -> y:Py.Object.t -> unit -> t
(**
Interpolate a 1-D function.

`x` and `y` are arrays of values used to approximate some function f:
``y = f(x)``.  This class returns a function whose call method uses
interpolation to find the value of new points.

Note that calling `interp1d` with NaNs present in input values results in
undefined behaviour.

Parameters
----------
x : (N,) array_like
    A 1-D array of real values.
y : (...,N,...) array_like
    A N-D array of real values. The length of `y` along the interpolation
    axis must be equal to the length of `x`.
kind : str or int, optional
    Specifies the kind of interpolation as a string
    ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
    'previous', 'next', where 'zero', 'slinear', 'quadratic' and 'cubic'
    refer to a spline interpolation of zeroth, first, second or third
    order; 'previous' and 'next' simply return the previous or next value
    of the point) or as an integer specifying the order of the spline
    interpolator to use.
    Default is 'linear'.
axis : int, optional
    Specifies the axis of `y` along which to interpolate.
    Interpolation defaults to the last axis of `y`.
copy : bool, optional
    If True, the class makes internal copies of x and y.
    If False, references to `x` and `y` are used. The default is to copy.
bounds_error : bool, optional
    If True, a ValueError is raised any time interpolation is attempted on
    a value outside of the range of x (where extrapolation is
    necessary). If False, out of bounds values are assigned `fill_value`.
    By default, an error is raised unless ``fill_value='extrapolate'``.
fill_value : array-like or (array-like, array_like) or 'extrapolate', optional
    - if a ndarray (or float), this value will be used to fill in for
      requested points outside of the data range. If not provided, then
      the default is NaN. The array-like must broadcast properly to the
      dimensions of the non-interpolation axes.
    - If a two-element tuple, then the first element is used as a
      fill value for ``x_new < x[0]`` and the second element is used for
      ``x_new > x[-1]``. Anything that is not a 2-element tuple (e.g.,
      list or ndarray, regardless of shape) is taken to be a single
      array-like argument meant to be used for both bounds as
      ``below, above = fill_value, fill_value``.

      .. versionadded:: 0.17.0
    - If 'extrapolate', then points outside the data range will be
      extrapolated.

      .. versionadded:: 0.17.0
assume_sorted : bool, optional
    If False, values of `x` can be in any order and they are sorted first.
    If True, `x` has to be an array of monotonically increasing values.

Attributes
----------
fill_value

Methods
-------
__call__

See Also
--------
splrep, splev
    Spline interpolation/smoothing based on FITPACK.
UnivariateSpline : An object-oriented wrapper of the FITPACK routines.
interp2d : 2-D interpolation

Examples
--------
>>> import matplotlib.pyplot as plt
>>> from scipy import interpolate
>>> x = np.arange(0, 10)
>>> y = np.exp(-x/3.0)
>>> f = interpolate.interp1d(x, y)

>>> xnew = np.arange(0, 9, 0.1)
>>> ynew = f(xnew)   # use interpolation function returned by `interp1d`
>>> plt.plot(x, y, 'o', xnew, ynew, '-')
>>> plt.show()
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Interp2d : sig
type tag = [`Interp2d]
type t = [`Interp2d | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?kind:[`Linear | `Cubic | `Quintic] -> ?copy:bool -> ?bounds_error:bool -> ?fill_value:[`I of int | `F of float] -> x:Py.Object.t -> y:Py.Object.t -> z:[>`Ndarray] Np.Obj.t -> unit -> t
(**
interp2d(x, y, z, kind='linear', copy=True, bounds_error=False,
         fill_value=None)

Interpolate over a 2-D grid.

`x`, `y` and `z` are arrays of values used to approximate some function
f: ``z = f(x, y)``. This class returns a function whose call method uses
spline interpolation to find the value of new points.

If `x` and `y` represent a regular grid, consider using
RectBivariateSpline.

Note that calling `interp2d` with NaNs present in input values results in
undefined behaviour.

Methods
-------
__call__

Parameters
----------
x, y : array_like
    Arrays defining the data point coordinates.

    If the points lie on a regular grid, `x` can specify the column
    coordinates and `y` the row coordinates, for example::

      >>> x = [0,1,2];  y = [0,3]; z = [[1,2,3], [4,5,6]]

    Otherwise, `x` and `y` must specify the full coordinates for each
    point, for example::

      >>> x = [0,1,2,0,1,2];  y = [0,0,0,3,3,3]; z = [1,2,3,4,5,6]

    If `x` and `y` are multi-dimensional, they are flattened before use.
z : array_like
    The values of the function to interpolate at the data points. If
    `z` is a multi-dimensional array, it is flattened before use.  The
    length of a flattened `z` array is either
    len(`x`)*len(`y`) if `x` and `y` specify the column and row coordinates
    or ``len(z) == len(x) == len(y)`` if `x` and `y` specify coordinates
    for each point.
kind : {'linear', 'cubic', 'quintic'}, optional
    The kind of spline interpolation to use. Default is 'linear'.
copy : bool, optional
    If True, the class makes internal copies of x, y and z.
    If False, references may be used. The default is to copy.
bounds_error : bool, optional
    If True, when interpolated values are requested outside of the
    domain of the input data (x,y), a ValueError is raised.
    If False, then `fill_value` is used.
fill_value : number, optional
    If provided, the value to use for points outside of the
    interpolation domain. If omitted (None), values outside
    the domain are extrapolated via nearest-neighbor extrapolation.

See Also
--------
RectBivariateSpline :
    Much faster 2D interpolation if your input data is on a grid
bisplrep, bisplev :
    Spline interpolation based on FITPACK
BivariateSpline : a more recent wrapper of the FITPACK routines
interp1d : one dimension version of this function

Notes
-----
The minimum number of data points required along the interpolation
axis is ``(k+1)**2``, with k=1 for linear, k=3 for cubic and k=5 for
quintic interpolation.

The interpolator is constructed by `bisplrep`, with a smoothing factor
of 0. If more control over smoothing is needed, `bisplrep` should be
used directly.

Examples
--------
Construct a 2-D grid and interpolate on it:

>>> from scipy import interpolate
>>> x = np.arange(-5.01, 5.01, 0.25)
>>> y = np.arange(-5.01, 5.01, 0.25)
>>> xx, yy = np.meshgrid(x, y)
>>> z = np.sin(xx**2+yy**2)
>>> f = interpolate.interp2d(x, y, z, kind='cubic')

Now use the obtained interpolation function and plot the result:

>>> import matplotlib.pyplot as plt
>>> xnew = np.arange(-5.01, 5.01, 1e-2)
>>> ynew = np.arange(-5.01, 5.01, 1e-2)
>>> znew = f(xnew, ynew)
>>> plt.plot(x, z[0, :], 'ro-', xnew, znew[0, :], 'b-')
>>> plt.show()
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Pchip : sig
type tag = [`PchipInterpolator]
type t = [`Object | `PchipInterpolator] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?axis:int -> ?extrapolate:bool -> x:[>`Ndarray] Np.Obj.t -> y:[>`Ndarray] Np.Obj.t -> unit -> t
(**
PCHIP 1-d monotonic cubic interpolation.

``x`` and ``y`` are arrays of values used to approximate some function f,
with ``y = f(x)``. The interpolant uses monotonic cubic splines
to find the value of new points. (PCHIP stands for Piecewise Cubic
Hermite Interpolating Polynomial).

Parameters
----------
x : ndarray
    A 1-D array of monotonically increasing real values. ``x`` cannot
    include duplicate values (otherwise f is overspecified)
y : ndarray
    A 1-D array of real values. ``y``'s length along the interpolation
    axis must be equal to the length of ``x``. If N-D array, use ``axis``
    parameter to select correct axis.
axis : int, optional
    Axis in the y array corresponding to the x-coordinate values.
extrapolate : bool, optional
    Whether to extrapolate to out-of-bounds points based on first
    and last intervals, or to return NaNs.

Methods
-------
__call__
derivative
antiderivative
roots

See Also
--------
CubicHermiteSpline
Akima1DInterpolator
CubicSpline
PPoly

Notes
-----
The interpolator preserves monotonicity in the interpolation data and does
not overshoot if the data is not smooth.

The first derivatives are guaranteed to be continuous, but the second
derivatives may jump at :math:`x_k`.

Determines the derivatives at the points :math:`x_k`, :math:`f'_k`,
by using PCHIP algorithm [1]_.

Let :math:`h_k = x_{k+1} - x_k`, and  :math:`d_k = (y_{k+1} - y_k) / h_k`
are the slopes at internal points :math:`x_k`.
If the signs of :math:`d_k` and :math:`d_{k-1}` are different or either of
them equals zero, then :math:`f'_k = 0`. Otherwise, it is given by the
weighted harmonic mean

.. math::

    \frac{w_1 + w_2}{f'_k} = \frac{w_1}{d_{k-1}} + \frac{w_2}{d_k}

where :math:`w_1 = 2 h_k + h_{k-1}` and :math:`w_2 = h_k + 2 h_{k-1}`.

The end slopes are set using a one-sided scheme [2]_.


References
----------
.. [1] F. N. Fritsch and R. E. Carlson, Monotone Piecewise Cubic Interpolation,
       SIAM J. Numer. Anal., 17(2), 238 (1980).
       :doi:`10.1137/0717021`.
.. [2] see, e.g., C. Moler, Numerical Computing with Matlab, 2004.
       :doi:`10.1137/1.9780898717952`
*)

val antiderivative : ?nu:int -> [> tag] Obj.t -> Py.Object.t
(**
Construct a new piecewise polynomial representing the antiderivative.

Antiderivative is also the indefinite integral of the function,
and derivative is its inverse operation.

Parameters
----------
nu : int, optional
    Order of antiderivative to evaluate. Default is 1, i.e. compute
    the first integral. If negative, the derivative is returned.

Returns
-------
pp : PPoly
    Piecewise polynomial of order k2 = k + n representing
    the antiderivative of this polynomial.

Notes
-----
The antiderivative returned by this function is continuous and
continuously differentiable to order n-1, up to floating point
rounding error.

If antiderivative is computed and ``self.extrapolate='periodic'``,
it will be set to False for the returned instance. This is done because
the antiderivative is no longer periodic and its correct evaluation
outside of the initially given x interval is difficult.
*)

val construct_fast : ?extrapolate:Py.Object.t -> ?axis:Py.Object.t -> c:Py.Object.t -> x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Construct the piecewise polynomial without making checks.

Takes the same parameters as the constructor. Input arguments
``c`` and ``x`` must be arrays of the correct shape and type.  The
``c`` array can only be of dtypes float and complex, and ``x``
array must have dtype float.
*)

val derivative : ?nu:int -> [> tag] Obj.t -> Py.Object.t
(**
Construct a new piecewise polynomial representing the derivative.

Parameters
----------
nu : int, optional
    Order of derivative to evaluate. Default is 1, i.e. compute the
    first derivative. If negative, the antiderivative is returned.

Returns
-------
pp : PPoly
    Piecewise polynomial of order k2 = k - n representing the derivative
    of this polynomial.

Notes
-----
Derivatives are evaluated piecewise for each polynomial
segment, even if the polynomial is not differentiable at the
breakpoints. The polynomial intervals are considered half-open,
``[a, b)``, except for the last interval which is closed
``[a, b]``.
*)

val extend : ?right:Py.Object.t -> c:[`Ndarray of [>`Ndarray] Np.Obj.t | `Size_k_m_ of Py.Object.t] -> x:[`Ndarray of [>`Ndarray] Np.Obj.t | `Size of Py.Object.t] -> [> tag] Obj.t -> Py.Object.t
(**
Add additional breakpoints and coefficients to the polynomial.

Parameters
----------
c : ndarray, size (k, m, ...)
    Additional coefficients for polynomials in intervals. Note that
    the first additional interval will be formed using one of the
    ``self.x`` end points.
x : ndarray, size (m,)
    Additional breakpoints. Must be sorted in the same order as
    ``self.x`` and either to the right or to the left of the current
    breakpoints.
right
    Deprecated argument. Has no effect.

    .. deprecated:: 0.19
*)

val from_bernstein_basis : ?extrapolate:[`Bool of bool | `Periodic] -> bp:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Construct a piecewise polynomial in the power basis
from a polynomial in Bernstein basis.

Parameters
----------
bp : BPoly
    A Bernstein basis polynomial, as created by BPoly
extrapolate : bool or 'periodic', optional
    If bool, determines whether to extrapolate to out-of-bounds points
    based on first and last intervals, or to return NaNs.
    If 'periodic', periodic extrapolation is used. Default is True.
*)

val from_spline : ?extrapolate:[`Bool of bool | `Periodic] -> tck:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Construct a piecewise polynomial from a spline

Parameters
----------
tck
    A spline, as returned by `splrep` or a BSpline object.
extrapolate : bool or 'periodic', optional
    If bool, determines whether to extrapolate to out-of-bounds points
    based on first and last intervals, or to return NaNs.
    If 'periodic', periodic extrapolation is used. Default is True.
*)

val integrate : ?extrapolate:[`Bool of bool | `Periodic] -> a:float -> b:float -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute a definite integral over a piecewise polynomial.

Parameters
----------
a : float
    Lower integration bound
b : float
    Upper integration bound
extrapolate : {bool, 'periodic', None}, optional
    If bool, determines whether to extrapolate to out-of-bounds points
    based on first and last intervals, or to return NaNs.
    If 'periodic', periodic extrapolation is used.
    If None (default), use `self.extrapolate`.

Returns
-------
ig : array_like
    Definite integral of the piecewise polynomial over [a, b]
*)

val roots : ?discontinuity:bool -> ?extrapolate:[`Bool of bool | `Periodic] -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Find real roots of the the piecewise polynomial.

Parameters
----------
discontinuity : bool, optional
    Whether to report sign changes across discontinuities at
    breakpoints as roots.
extrapolate : {bool, 'periodic', None}, optional
    If bool, determines whether to return roots from the polynomial
    extrapolated based on first and last intervals, 'periodic' works
    the same as False. If None (default), use `self.extrapolate`.

Returns
-------
roots : ndarray
    Roots of the polynomial(s).

    If the PPoly object describes multiple polynomials, the
    return value is an object array whose each element is an
    ndarray containing the roots.

See Also
--------
PPoly.solve
*)

val solve : ?y:float -> ?discontinuity:bool -> ?extrapolate:[`Bool of bool | `Periodic] -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Find real solutions of the the equation ``pp(x) == y``.

Parameters
----------
y : float, optional
    Right-hand side. Default is zero.
discontinuity : bool, optional
    Whether to report sign changes across discontinuities at
    breakpoints as roots.
extrapolate : {bool, 'periodic', None}, optional
    If bool, determines whether to return roots from the polynomial
    extrapolated based on first and last intervals, 'periodic' works
    the same as False. If None (default), use `self.extrapolate`.

Returns
-------
roots : ndarray
    Roots of the polynomial(s).

    If the PPoly object describes multiple polynomials, the
    return value is an object array whose each element is an
    ndarray containing the roots.

Notes
-----
This routine works only on real-valued polynomials.

If the piecewise polynomial contains sections that are
identically zero, the root list will contain the start point
of the corresponding interval, followed by a ``nan`` value.

If the polynomial is discontinuous across a breakpoint, and
there is a sign change across the breakpoint, this is reported
if the `discont` parameter is True.

Examples
--------

Finding roots of ``[x**2 - 1, (x - 1)**2]`` defined on intervals
``[-2, 1], [1, 2]``:

>>> from scipy.interpolate import PPoly
>>> pp = PPoly(np.array([[1, -4, 3], [1, 0, 0]]).T, [-2, 1, 2])
>>> pp.solve()
array([-1.,  1.])
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Dfitpack : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val splev : ?e:Py.Object.t -> t:Py.Object.t -> c:Py.Object.t -> k:Py.Object.t -> x:Py.Object.t -> unit -> Py.Object.t
(**
y = splev(t,c,k,x,[e])

Wrapper for ``splev``.

Parameters
----------
t : input rank-1 array('d') with bounds (n)
c : input rank-1 array('d') with bounds (n)
k : input int
x : input rank-1 array('d') with bounds (m)

Other Parameters
----------------
e : input int, optional
    Default: 0

Returns
-------
y : rank-1 array('d') with bounds (m)
*)


end

module Fitpack : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val bisplev : ?dx:Py.Object.t -> ?dy:Py.Object.t -> x:Py.Object.t -> y:Py.Object.t -> tck:Py.Object.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Evaluate a bivariate B-spline and its derivatives.

Return a rank-2 array of spline function values (or spline derivative
values) at points given by the cross-product of the rank-1 arrays `x` and
`y`.  In special cases, return an array or just a float if either `x` or
`y` or both are floats.  Based on BISPEV from FITPACK.

Parameters
----------
x, y : ndarray
    Rank-1 arrays specifying the domain over which to evaluate the
    spline or its derivative.
tck : tuple
    A sequence of length 5 returned by `bisplrep` containing the knot
    locations, the coefficients, and the degree of the spline:
    [tx, ty, c, kx, ky].
dx, dy : int, optional
    The orders of the partial derivatives in `x` and `y` respectively.

Returns
-------
vals : ndarray
    The B-spline or its derivative evaluated over the set formed by
    the cross-product of `x` and `y`.

See Also
--------
splprep, splrep, splint, sproot, splev
UnivariateSpline, BivariateSpline

Notes
-----
    See `bisplrep` to generate the `tck` representation.

References
----------
.. [1] Dierckx P. : An algorithm for surface fitting
   with spline functions
   Ima J. Numer. Anal. 1 (1981) 267-283.
.. [2] Dierckx P. : An algorithm for surface fitting
   with spline functions
   report tw50, Dept. Computer Science,K.U.Leuven, 1980.
.. [3] Dierckx P. : Curve and surface fitting with splines,
   Monographs on Numerical Analysis, Oxford University Press, 1993.
*)

val bisplrep : ?w:[>`Ndarray] Np.Obj.t -> ?xb:Py.Object.t -> ?xe:Py.Object.t -> ?yb:Py.Object.t -> ?ye:Py.Object.t -> ?kx:Py.Object.t -> ?ky:Py.Object.t -> ?task:int -> ?s:float -> ?eps:float -> ?tx:Py.Object.t -> ?ty:Py.Object.t -> ?full_output:int -> ?nxest:Py.Object.t -> ?nyest:Py.Object.t -> ?quiet:int -> x:Py.Object.t -> y:Py.Object.t -> z:Py.Object.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t * int * string)
(**
Find a bivariate B-spline representation of a surface.

Given a set of data points (x[i], y[i], z[i]) representing a surface
z=f(x,y), compute a B-spline representation of the surface. Based on
the routine SURFIT from FITPACK.

Parameters
----------
x, y, z : ndarray
    Rank-1 arrays of data points.
w : ndarray, optional
    Rank-1 array of weights. By default ``w=np.ones(len(x))``.
xb, xe : float, optional
    End points of approximation interval in `x`.
    By default ``xb = x.min(), xe=x.max()``.
yb, ye : float, optional
    End points of approximation interval in `y`.
    By default ``yb=y.min(), ye = y.max()``.
kx, ky : int, optional
    The degrees of the spline (1 <= kx, ky <= 5).
    Third order (kx=ky=3) is recommended.
task : int, optional
    If task=0, find knots in x and y and coefficients for a given
    smoothing factor, s.
    If task=1, find knots and coefficients for another value of the
    smoothing factor, s.  bisplrep must have been previously called
    with task=0 or task=1.
    If task=-1, find coefficients for a given set of knots tx, ty.
s : float, optional
    A non-negative smoothing factor.  If weights correspond
    to the inverse of the standard-deviation of the errors in z,
    then a good s-value should be found in the range
    ``(m-sqrt(2*m),m+sqrt(2*m))`` where m=len(x).
eps : float, optional
    A threshold for determining the effective rank of an
    over-determined linear system of equations (0 < eps < 1).
    `eps` is not likely to need changing.
tx, ty : ndarray, optional
    Rank-1 arrays of the knots of the spline for task=-1
full_output : int, optional
    Non-zero to return optional outputs.
nxest, nyest : int, optional
    Over-estimates of the total number of knots. If None then
    ``nxest = max(kx+sqrt(m/2),2*kx+3)``,
    ``nyest = max(ky+sqrt(m/2),2*ky+3)``.
quiet : int, optional
    Non-zero to suppress printing of messages.
    This parameter is deprecated; use standard Python warning filters
    instead.

Returns
-------
tck : array_like
    A list [tx, ty, c, kx, ky] containing the knots (tx, ty) and
    coefficients (c) of the bivariate B-spline representation of the
    surface along with the degree of the spline.
fp : ndarray
    The weighted sum of squared residuals of the spline approximation.
ier : int
    An integer flag about splrep success.  Success is indicated if
    ier<=0. If ier in [1,2,3] an error occurred but was not raised.
    Otherwise an error is raised.
msg : str
    A message corresponding to the integer flag, ier.

See Also
--------
splprep, splrep, splint, sproot, splev
UnivariateSpline, BivariateSpline

Notes
-----
See `bisplev` to evaluate the value of the B-spline given its tck
representation.

References
----------
.. [1] Dierckx P.:An algorithm for surface fitting with spline functions
   Ima J. Numer. Anal. 1 (1981) 267-283.
.. [2] Dierckx P.:An algorithm for surface fitting with spline functions
   report tw50, Dept. Computer Science,K.U.Leuven, 1980.
.. [3] Dierckx P.:Curve and surface fitting with splines, Monographs on
   Numerical Analysis, Oxford University Press, 1993.
*)

val dblint : xa:Py.Object.t -> xb:Py.Object.t -> ya:Py.Object.t -> yb:Py.Object.t -> tck:Py.Object.t -> unit -> float
(**
Evaluate the integral of a spline over area [xa,xb] x [ya,yb].

Parameters
----------
xa, xb : float
    The end-points of the x integration interval.
ya, yb : float
    The end-points of the y integration interval.
tck : list [tx, ty, c, kx, ky]
    A sequence of length 5 returned by bisplrep containing the knot
    locations tx, ty, the coefficients c, and the degrees kx, ky
    of the spline.

Returns
-------
integ : float
    The value of the resulting integral.
*)

val insert : ?m:int -> ?per:int -> x:Py.Object.t -> tck:Py.Object.t -> unit -> Py.Object.t
(**
Insert knots into a B-spline.

Given the knots and coefficients of a B-spline representation, create a
new B-spline with a knot inserted `m` times at point `x`.
This is a wrapper around the FORTRAN routine insert of FITPACK.

Parameters
----------
x (u) : array_like
    A 1-D point at which to insert a new knot(s).  If `tck` was returned
    from ``splprep``, then the parameter values, u should be given.
tck : a `BSpline` instance or a tuple
    If tuple, then it is expected to be a tuple (t,c,k) containing
    the vector of knots, the B-spline coefficients, and the degree of
    the spline.
m : int, optional
    The number of times to insert the given knot (its multiplicity).
    Default is 1.
per : int, optional
    If non-zero, the input spline is considered periodic.

Returns
-------
BSpline instance or a tuple
    A new B-spline with knots t, coefficients c, and degree k.
    ``t(k+1) <= x <= t(n-k)``, where k is the degree of the spline.
    In case of a periodic spline (``per != 0``) there must be
    either at least k interior knots t(j) satisfying ``t(k+1)<t(j)<=x``
    or at least k interior knots t(j) satisfying ``x<=t(j)<t(n-k)``.
    A tuple is returned iff the input argument `tck` is a tuple, otherwise
    a BSpline object is constructed and returned.

Notes
-----
Based on algorithms from [1]_ and [2]_.

Manipulating the tck-tuples directly is not recommended. In new code,
prefer using the `BSpline` objects.

References
----------
.. [1] W. Boehm, 'Inserting new knots into b-spline curves.',
    Computer Aided Design, 12, p.199-201, 1980.
.. [2] P. Dierckx, 'Curve and surface fitting with splines, Monographs on
    Numerical Analysis', Oxford University Press, 1993.
*)

val spalde : x:[>`Ndarray] Np.Obj.t -> tck:Py.Object.t -> unit -> Py.Object.t
(**
Evaluate all derivatives of a B-spline.

Given the knots and coefficients of a cubic B-spline compute all
derivatives up to order k at a point (or set of points).

Parameters
----------
x : array_like
    A point or a set of points at which to evaluate the derivatives.
    Note that ``t(k) <= x <= t(n-k+1)`` must hold for each `x`.
tck : tuple
    A tuple ``(t, c, k)``, containing the vector of knots, the B-spline
    coefficients, and the degree of the spline (see `splev`).

Returns
-------
results : {ndarray, list of ndarrays}
    An array (or a list of arrays) containing all derivatives
    up to order k inclusive for each point `x`.

See Also
--------
splprep, splrep, splint, sproot, splev, bisplrep, bisplev,
BSpline

References
----------
.. [1] C. de Boor: On calculating with b-splines, J. Approximation Theory
   6 (1972) 50-62.
.. [2] M. G. Cox : The numerical evaluation of b-splines, J. Inst. Maths
   applics 10 (1972) 134-149.
.. [3] P. Dierckx : Curve and surface fitting with splines, Monographs on
   Numerical Analysis, Oxford University Press, 1993.
*)

val splantider : ?n:int -> tck:Py.Object.t -> unit -> Py.Object.t
(**
Compute the spline for the antiderivative (integral) of a given spline.

Parameters
----------
tck : BSpline instance or a tuple of (t, c, k)
    Spline whose antiderivative to compute
n : int, optional
    Order of antiderivative to evaluate. Default: 1

Returns
-------
BSpline instance or a tuple of (t2, c2, k2)
    Spline of order k2=k+n representing the antiderivative of the input
    spline.
    A tuple is returned iff the input argument `tck` is a tuple, otherwise
    a BSpline object is constructed and returned.

See Also
--------
splder, splev, spalde
BSpline

Notes
-----
The `splder` function is the inverse operation of this function.
Namely, ``splder(splantider(tck))`` is identical to `tck`, modulo
rounding error.

.. versionadded:: 0.13.0

Examples
--------
>>> from scipy.interpolate import splrep, splder, splantider, splev
>>> x = np.linspace(0, np.pi/2, 70)
>>> y = 1 / np.sqrt(1 - 0.8*np.sin(x)**2)
>>> spl = splrep(x, y)

The derivative is the inverse operation of the antiderivative,
although some floating point error accumulates:

>>> splev(1.7, spl), splev(1.7, splder(splantider(spl)))
(array(2.1565429877197317), array(2.1565429877201865))

Antiderivative can be used to evaluate definite integrals:

>>> ispl = splantider(spl)
>>> splev(np.pi/2, ispl) - splev(0, ispl)
2.2572053588768486

This is indeed an approximation to the complete elliptic integral
:math:`K(m) = \int_0^{\pi/2} [1 - m\sin^2 x]^{-1/2} dx`:

>>> from scipy.special import ellipk
>>> ellipk(0.8)
2.2572053268208538
*)

val splder : ?n:int -> tck:Py.Object.t -> unit -> Py.Object.t
(**
Compute the spline representation of the derivative of a given spline

Parameters
----------
tck : BSpline instance or a tuple of (t, c, k)
    Spline whose derivative to compute
n : int, optional
    Order of derivative to evaluate. Default: 1

Returns
-------
`BSpline` instance or tuple
    Spline of order k2=k-n representing the derivative
    of the input spline.
    A tuple is returned iff the input argument `tck` is a tuple, otherwise
    a BSpline object is constructed and returned.

Notes
-----

.. versionadded:: 0.13.0

See Also
--------
splantider, splev, spalde
BSpline

Examples
--------
This can be used for finding maxima of a curve:

>>> from scipy.interpolate import splrep, splder, sproot
>>> x = np.linspace(0, 10, 70)
>>> y = np.sin(x)
>>> spl = splrep(x, y, k=4)

Now, differentiate the spline and find the zeros of the
derivative. (NB: `sproot` only works for order 3 splines, so we
fit an order 4 spline):

>>> dspl = splder(spl)
>>> sproot(dspl) / np.pi
array([ 0.50000001,  1.5       ,  2.49999998])

This agrees well with roots :math:`\pi/2 + n\pi` of
:math:`\cos(x) = \sin'(x)`.
*)

val splev : ?der:int -> ?ext:int -> x:[>`Ndarray] Np.Obj.t -> tck:Py.Object.t -> unit -> Py.Object.t
(**
Evaluate a B-spline or its derivatives.

Given the knots and coefficients of a B-spline representation, evaluate
the value of the smoothing polynomial and its derivatives.  This is a
wrapper around the FORTRAN routines splev and splder of FITPACK.

Parameters
----------
x : array_like
    An array of points at which to return the value of the smoothed
    spline or its derivatives.  If `tck` was returned from `splprep`,
    then the parameter values, u should be given.
tck : 3-tuple or a BSpline object
    If a tuple, then it should be a sequence of length 3 returned by
    `splrep` or `splprep` containing the knots, coefficients, and degree
    of the spline. (Also see Notes.)
der : int, optional
    The order of derivative of the spline to compute (must be less than
    or equal to k, the degree of the spline).
ext : int, optional
    Controls the value returned for elements of ``x`` not in the
    interval defined by the knot sequence.

    * if ext=0, return the extrapolated value.
    * if ext=1, return 0
    * if ext=2, raise a ValueError
    * if ext=3, return the boundary value.

    The default value is 0.

Returns
-------
y : ndarray or list of ndarrays
    An array of values representing the spline function evaluated at
    the points in `x`.  If `tck` was returned from `splprep`, then this
    is a list of arrays representing the curve in N-dimensional space.

Notes
-----
Manipulating the tck-tuples directly is not recommended. In new code,
prefer using `BSpline` objects.

See Also
--------
splprep, splrep, sproot, spalde, splint
bisplrep, bisplev
BSpline

References
----------
.. [1] C. de Boor, 'On calculating with b-splines', J. Approximation
    Theory, 6, p.50-62, 1972.
.. [2] M. G. Cox, 'The numerical evaluation of b-splines', J. Inst. Maths
    Applics, 10, p.134-149, 1972.
.. [3] P. Dierckx, 'Curve and surface fitting with splines', Monographs
    on Numerical Analysis, Oxford University Press, 1993.
*)

val splint : ?full_output:int -> a:Py.Object.t -> b:Py.Object.t -> tck:Py.Object.t -> unit -> (float * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Evaluate the definite integral of a B-spline between two given points.

Parameters
----------
a, b : float
    The end-points of the integration interval.
tck : tuple or a BSpline instance
    If a tuple, then it should be a sequence of length 3, containing the
    vector of knots, the B-spline coefficients, and the degree of the
    spline (see `splev`).
full_output : int, optional
    Non-zero to return optional output.

Returns
-------
integral : float
    The resulting integral.
wrk : ndarray
    An array containing the integrals of the normalized B-splines
    defined on the set of knots.
    (Only returned if `full_output` is non-zero)

Notes
-----
`splint` silently assumes that the spline function is zero outside the data
interval (`a`, `b`).

Manipulating the tck-tuples directly is not recommended. In new code,
prefer using the `BSpline` objects.

See Also
--------
splprep, splrep, sproot, spalde, splev
bisplrep, bisplev
BSpline

References
----------
.. [1] P.W. Gaffney, The calculation of indefinite integrals of b-splines',
    J. Inst. Maths Applics, 17, p.37-41, 1976.
.. [2] P. Dierckx, 'Curve and surface fitting with splines', Monographs
    on Numerical Analysis, Oxford University Press, 1993.
*)

val splprep : ?w:[>`Ndarray] Np.Obj.t -> ?u:[>`Ndarray] Np.Obj.t -> ?ub:Py.Object.t -> ?ue:Py.Object.t -> ?k:int -> ?task:int -> ?s:float -> ?t:int -> ?full_output:int -> ?nest:int -> ?per:int -> ?quiet:int -> x:[>`Ndarray] Np.Obj.t -> unit -> (Py.Object.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t * float * int * string)
(**
Find the B-spline representation of an N-dimensional curve.

Given a list of N rank-1 arrays, `x`, which represent a curve in
N-dimensional space parametrized by `u`, find a smooth approximating
spline curve g(`u`). Uses the FORTRAN routine parcur from FITPACK.

Parameters
----------
x : array_like
    A list of sample vector arrays representing the curve.
w : array_like, optional
    Strictly positive rank-1 array of weights the same length as `x[0]`.
    The weights are used in computing the weighted least-squares spline
    fit. If the errors in the `x` values have standard-deviation given by
    the vector d, then `w` should be 1/d. Default is ``ones(len(x[0]))``.
u : array_like, optional
    An array of parameter values. If not given, these values are
    calculated automatically as ``M = len(x[0])``, where

        v[0] = 0

        v[i] = v[i-1] + distance(`x[i]`, `x[i-1]`)

        u[i] = v[i] / v[M-1]

ub, ue : int, optional
    The end-points of the parameters interval.  Defaults to
    u[0] and u[-1].
k : int, optional
    Degree of the spline. Cubic splines are recommended.
    Even values of `k` should be avoided especially with a small s-value.
    ``1 <= k <= 5``, default is 3.
task : int, optional
    If task==0 (default), find t and c for a given smoothing factor, s.
    If task==1, find t and c for another value of the smoothing factor, s.
    There must have been a previous call with task=0 or task=1
    for the same set of data.
    If task=-1 find the weighted least square spline for a given set of
    knots, t.
s : float, optional
    A smoothing condition.  The amount of smoothness is determined by
    satisfying the conditions: ``sum((w * (y - g))**2,axis=0) <= s``,
    where g(x) is the smoothed interpolation of (x,y).  The user can
    use `s` to control the trade-off between closeness and smoothness
    of fit.  Larger `s` means more smoothing while smaller values of `s`
    indicate less smoothing. Recommended values of `s` depend on the
    weights, w.  If the weights represent the inverse of the
    standard-deviation of y, then a good `s` value should be found in
    the range ``(m-sqrt(2*m),m+sqrt(2*m))``, where m is the number of
    data points in x, y, and w.
t : int, optional
    The knots needed for task=-1.
full_output : int, optional
    If non-zero, then return optional outputs.
nest : int, optional
    An over-estimate of the total number of knots of the spline to
    help in determining the storage space.  By default nest=m/2.
    Always large enough is nest=m+k+1.
per : int, optional
   If non-zero, data points are considered periodic with period
   ``x[m-1] - x[0]`` and a smooth periodic spline approximation is
   returned.  Values of ``y[m-1]`` and ``w[m-1]`` are not used.
quiet : int, optional
     Non-zero to suppress messages.
     This parameter is deprecated; use standard Python warning filters
     instead.

Returns
-------
tck : tuple
    (t,c,k) a tuple containing the vector of knots, the B-spline
    coefficients, and the degree of the spline.
u : array
    An array of the values of the parameter.
fp : float
    The weighted sum of squared residuals of the spline approximation.
ier : int
    An integer flag about splrep success.  Success is indicated
    if ier<=0. If ier in [1,2,3] an error occurred but was not raised.
    Otherwise an error is raised.
msg : str
    A message corresponding to the integer flag, ier.

See Also
--------
splrep, splev, sproot, spalde, splint,
bisplrep, bisplev
UnivariateSpline, BivariateSpline
BSpline
make_interp_spline

Notes
-----
See `splev` for evaluation of the spline and its derivatives.
The number of dimensions N must be smaller than 11.

The number of coefficients in the `c` array is ``k+1`` less then the number
of knots, ``len(t)``. This is in contrast with `splrep`, which zero-pads
the array of coefficients to have the same length as the array of knots.
These additional coefficients are ignored by evaluation routines, `splev`
and `BSpline`.

References
----------
.. [1] P. Dierckx, 'Algorithms for smoothing data with periodic and
    parametric splines, Computer Graphics and Image Processing',
    20 (1982) 171-184.
.. [2] P. Dierckx, 'Algorithms for smoothing data with periodic and
    parametric splines', report tw55, Dept. Computer Science,
    K.U.Leuven, 1981.
.. [3] P. Dierckx, 'Curve and surface fitting with splines', Monographs on
    Numerical Analysis, Oxford University Press, 1993.

Examples
--------
Generate a discretization of a limacon curve in the polar coordinates:

>>> phi = np.linspace(0, 2.*np.pi, 40)
>>> r = 0.5 + np.cos(phi)         # polar coords
>>> x, y = r * np.cos(phi), r * np.sin(phi)    # convert to cartesian

And interpolate:

>>> from scipy.interpolate import splprep, splev
>>> tck, u = splprep([x, y], s=0)
>>> new_points = splev(u, tck)

Notice that (i) we force interpolation by using `s=0`,
(ii) the parameterization, ``u``, is generated automatically.
Now plot the result:

>>> import matplotlib.pyplot as plt
>>> fig, ax = plt.subplots()
>>> ax.plot(x, y, 'ro')
>>> ax.plot(new_points[0], new_points[1], 'r-')
>>> plt.show()
*)

val splrep : ?w:[>`Ndarray] Np.Obj.t -> ?xb:Py.Object.t -> ?xe:Py.Object.t -> ?k:int -> ?task:[`One | `T_1 of Py.Object.t | `Zero] -> ?s:float -> ?t:[>`Ndarray] Np.Obj.t -> ?full_output:bool -> ?per:bool -> ?quiet:bool -> x:Py.Object.t -> y:Py.Object.t -> unit -> (Py.Object.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t * int * string)
(**
Find the B-spline representation of 1-D curve.

Given the set of data points ``(x[i], y[i])`` determine a smooth spline
approximation of degree k on the interval ``xb <= x <= xe``.

Parameters
----------
x, y : array_like
    The data points defining a curve y = f(x).
w : array_like, optional
    Strictly positive rank-1 array of weights the same length as x and y.
    The weights are used in computing the weighted least-squares spline
    fit. If the errors in the y values have standard-deviation given by the
    vector d, then w should be 1/d. Default is ones(len(x)).
xb, xe : float, optional
    The interval to fit.  If None, these default to x[0] and x[-1]
    respectively.
k : int, optional
    The degree of the spline fit. It is recommended to use cubic splines.
    Even values of k should be avoided especially with small s values.
    1 <= k <= 5
task : {1, 0, -1}, optional
    If task==0 find t and c for a given smoothing factor, s.

    If task==1 find t and c for another value of the smoothing factor, s.
    There must have been a previous call with task=0 or task=1 for the same
    set of data (t will be stored an used internally)

    If task=-1 find the weighted least square spline for a given set of
    knots, t. These should be interior knots as knots on the ends will be
    added automatically.
s : float, optional
    A smoothing condition. The amount of smoothness is determined by
    satisfying the conditions: sum((w * (y - g))**2,axis=0) <= s where g(x)
    is the smoothed interpolation of (x,y). The user can use s to control
    the tradeoff between closeness and smoothness of fit. Larger s means
    more smoothing while smaller values of s indicate less smoothing.
    Recommended values of s depend on the weights, w. If the weights
    represent the inverse of the standard-deviation of y, then a good s
    value should be found in the range (m-sqrt(2*m),m+sqrt(2*m)) where m is
    the number of datapoints in x, y, and w. default : s=m-sqrt(2*m) if
    weights are supplied. s = 0.0 (interpolating) if no weights are
    supplied.
t : array_like, optional
    The knots needed for task=-1. If given then task is automatically set
    to -1.
full_output : bool, optional
    If non-zero, then return optional outputs.
per : bool, optional
    If non-zero, data points are considered periodic with period x[m-1] -
    x[0] and a smooth periodic spline approximation is returned. Values of
    y[m-1] and w[m-1] are not used.
quiet : bool, optional
    Non-zero to suppress messages.
    This parameter is deprecated; use standard Python warning filters
    instead.

Returns
-------
tck : tuple
    A tuple (t,c,k) containing the vector of knots, the B-spline
    coefficients, and the degree of the spline.
fp : array, optional
    The weighted sum of squared residuals of the spline approximation.
ier : int, optional
    An integer flag about splrep success. Success is indicated if ier<=0.
    If ier in [1,2,3] an error occurred but was not raised. Otherwise an
    error is raised.
msg : str, optional
    A message corresponding to the integer flag, ier.

See Also
--------
UnivariateSpline, BivariateSpline
splprep, splev, sproot, spalde, splint
bisplrep, bisplev
BSpline
make_interp_spline

Notes
-----
See `splev` for evaluation of the spline and its derivatives. Uses the
FORTRAN routine ``curfit`` from FITPACK.

The user is responsible for assuring that the values of `x` are unique.
Otherwise, `splrep` will not return sensible results.

If provided, knots `t` must satisfy the Schoenberg-Whitney conditions,
i.e., there must be a subset of data points ``x[j]`` such that
``t[j] < x[j] < t[j+k+1]``, for ``j=0, 1,...,n-k-2``.

This routine zero-pads the coefficients array ``c`` to have the same length
as the array of knots ``t`` (the trailing ``k + 1`` coefficients are ignored
by the evaluation routines, `splev` and `BSpline`.) This is in contrast with
`splprep`, which does not zero-pad the coefficients.

References
----------
Based on algorithms described in [1]_, [2]_, [3]_, and [4]_:

.. [1] P. Dierckx, 'An algorithm for smoothing, differentiation and
   integration of experimental data using spline functions',
   J.Comp.Appl.Maths 1 (1975) 165-184.
.. [2] P. Dierckx, 'A fast algorithm for smoothing data on a rectangular
   grid while using spline functions', SIAM J.Numer.Anal. 19 (1982)
   1286-1304.
.. [3] P. Dierckx, 'An improved algorithm for curve fitting with spline
   functions', report tw54, Dept. Computer Science,K.U. Leuven, 1981.
.. [4] P. Dierckx, 'Curve and surface fitting with splines', Monographs on
   Numerical Analysis, Oxford University Press, 1993.

Examples
--------

>>> import matplotlib.pyplot as plt
>>> from scipy.interpolate import splev, splrep
>>> x = np.linspace(0, 10, 10)
>>> y = np.sin(x)
>>> spl = splrep(x, y)
>>> x2 = np.linspace(0, 10, 200)
>>> y2 = splev(x2, spl)
>>> plt.plot(x, y, 'o', x2, y2)
>>> plt.show()
*)

val sproot : ?mest:int -> tck:Py.Object.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Find the roots of a cubic B-spline.

Given the knots (>=8) and coefficients of a cubic B-spline return the
roots of the spline.

Parameters
----------
tck : tuple or a BSpline object
    If a tuple, then it should be a sequence of length 3, containing the
    vector of knots, the B-spline coefficients, and the degree of the
    spline.
    The number of knots must be >= 8, and the degree must be 3.
    The knots must be a montonically increasing sequence.
mest : int, optional
    An estimate of the number of zeros (Default is 10).

Returns
-------
zeros : ndarray
    An array giving the roots of the spline.

Notes
-----
Manipulating the tck-tuples directly is not recommended. In new code,
prefer using the `BSpline` objects.

See also
--------
splprep, splrep, splint, spalde, splev
bisplrep, bisplev
BSpline


References
----------
.. [1] C. de Boor, 'On calculating with b-splines', J. Approximation
    Theory, 6, p.50-62, 1972.
.. [2] M. G. Cox, 'The numerical evaluation of b-splines', J. Inst. Maths
    Applics, 10, p.134-149, 1972.
.. [3] P. Dierckx, 'Curve and surface fitting with splines', Monographs
    on Numerical Analysis, Oxford University Press, 1993.
*)


end

module Fitpack2 : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module SphereBivariateSpline : sig
type tag = [`SphereBivariateSpline]
type t = [`Object | `SphereBivariateSpline] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : unit -> t
(**
Bivariate spline s(x,y) of degrees 3 on a sphere, calculated from a
given set of data points (theta,phi,r).

.. versionadded:: 0.11.0

See Also
--------
bisplrep, bisplev : an older wrapping of FITPACK
UnivariateSpline : a similar class for univariate spline interpolation
SmoothUnivariateSpline :
    to create a BivariateSpline through the given points
LSQUnivariateSpline :
    to create a BivariateSpline using weighted least-squares fitting
*)

val ev : ?dtheta:int -> ?dphi:int -> theta:Py.Object.t -> phi:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Evaluate the spline at points

Returns the interpolated value at ``(theta[i], phi[i]),
i=0,...,len(theta)-1``.

Parameters
----------
theta, phi : array_like
    Input coordinates. Standard Numpy broadcasting is obeyed.
dtheta : int, optional
    Order of theta-derivative

    .. versionadded:: 0.14.0
dphi : int, optional
    Order of phi-derivative

    .. versionadded:: 0.14.0
*)

val get_coeffs : [> tag] Obj.t -> Py.Object.t
(**
Return spline coefficients.
*)

val get_knots : [> tag] Obj.t -> Py.Object.t
(**
Return a tuple (tx,ty) where tx,ty contain knots positions
of the spline with respect to x-, y-variable, respectively.
The position of interior and additional knots are given as
t[k+1:-k-1] and t[:k+1]=b, t[-k-1:]=e, respectively.
*)

val get_residual : [> tag] Obj.t -> Py.Object.t
(**
Return weighted sum of squared residuals of the spline
approximation: sum ((w[i]*(z[i]-s(x[i],y[i])))**2,axis=0)
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

val concatenate : ?axis:int -> ?out:[>`Ndarray] Np.Obj.t -> a:Py.Object.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
concatenate((a1, a2, ...), axis=0, out=None)

Join a sequence of arrays along an existing axis.

Parameters
----------
a1, a2, ... : sequence of array_like
    The arrays must have the same shape, except in the dimension
    corresponding to `axis` (the first, by default).
axis : int, optional
    The axis along which the arrays will be joined.  If axis is None,
    arrays are flattened before use.  Default is 0.
out : ndarray, optional
    If provided, the destination to place the result. The shape must be
    correct, matching that of what concatenate would have returned if no
    out argument were specified.

Returns
-------
res : ndarray
    The concatenated array.

See Also
--------
ma.concatenate : Concatenate function that preserves input masks.
array_split : Split an array into multiple sub-arrays of equal or
              near-equal size.
split : Split array into a list of multiple sub-arrays of equal size.
hsplit : Split array into multiple sub-arrays horizontally (column wise)
vsplit : Split array into multiple sub-arrays vertically (row wise)
dsplit : Split array into multiple sub-arrays along the 3rd axis (depth).
stack : Stack a sequence of arrays along a new axis.
hstack : Stack arrays in sequence horizontally (column wise)
vstack : Stack arrays in sequence vertically (row wise)
dstack : Stack arrays in sequence depth wise (along third dimension)
block : Assemble arrays from blocks.

Notes
-----
When one or more of the arrays to be concatenated is a MaskedArray,
this function will return a MaskedArray object instead of an ndarray,
but the input masks are *not* preserved. In cases where a MaskedArray
is expected as input, use the ma.concatenate function from the masked
array module instead.

Examples
--------
>>> a = np.array([[1, 2], [3, 4]])
>>> b = np.array([[5, 6]])
>>> np.concatenate((a, b), axis=0)
array([[1, 2],
       [3, 4],
       [5, 6]])
>>> np.concatenate((a, b.T), axis=1)
array([[1, 2, 5],
       [3, 4, 6]])
>>> np.concatenate((a, b), axis=None)
array([1, 2, 3, 4, 5, 6])

This function will not preserve masking of MaskedArray inputs.

>>> a = np.ma.arange(3)
>>> a[1] = np.ma.masked
>>> b = np.arange(2, 5)
>>> a
masked_array(data=[0, --, 2],
             mask=[False,  True, False],
       fill_value=999999)
>>> b
array([2, 3, 4])
>>> np.concatenate([a, b])
masked_array(data=[0, 1, 2, 2, 3, 4],
             mask=False,
       fill_value=999999)
>>> np.ma.concatenate([a, b])
masked_array(data=[0, --, 2, 2, 3, 4],
             mask=[False,  True, False, False, False, False],
       fill_value=999999)
*)

val diff : ?n:int -> ?axis:int -> ?prepend:Py.Object.t -> ?append:Py.Object.t -> a:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Calculate the n-th discrete difference along the given axis.

The first difference is given by ``out[i] = a[i+1] - a[i]`` along
the given axis, higher differences are calculated by using `diff`
recursively.

Parameters
----------
a : array_like
    Input array
n : int, optional
    The number of times values are differenced. If zero, the input
    is returned as-is.
axis : int, optional
    The axis along which the difference is taken, default is the
    last axis.
prepend, append : array_like, optional
    Values to prepend or append to `a` along axis prior to
    performing the difference.  Scalar values are expanded to
    arrays with length 1 in the direction of axis and the shape
    of the input array in along all other axes.  Otherwise the
    dimension and shape must match `a` except along axis.

    .. versionadded:: 1.16.0

Returns
-------
diff : ndarray
    The n-th differences. The shape of the output is the same as `a`
    except along `axis` where the dimension is smaller by `n`. The
    type of the output is the same as the type of the difference
    between any two elements of `a`. This is the same as the type of
    `a` in most cases. A notable exception is `datetime64`, which
    results in a `timedelta64` output array.

See Also
--------
gradient, ediff1d, cumsum

Notes
-----
Type is preserved for boolean arrays, so the result will contain
`False` when consecutive elements are the same and `True` when they
differ.

For unsigned integer arrays, the results will also be unsigned. This
should not be surprising, as the result is consistent with
calculating the difference directly:

>>> u8_arr = np.array([1, 0], dtype=np.uint8)
>>> np.diff(u8_arr)
array([255], dtype=uint8)
>>> u8_arr[1,...] - u8_arr[0,...]
255

If this is not desirable, then the array should be cast to a larger
integer type first:

>>> i16_arr = u8_arr.astype(np.int16)
>>> np.diff(i16_arr)
array([-1], dtype=int16)

Examples
--------
>>> x = np.array([1, 2, 4, 7, 0])
>>> np.diff(x)
array([ 1,  2,  3, -7])
>>> np.diff(x, n=2)
array([  1,   1, -10])

>>> x = np.array([[1, 3, 6, 10], [0, 5, 6, 8]])
>>> np.diff(x)
array([[2, 3, 4],
       [5, 1, 2]])
>>> np.diff(x, axis=0)
array([[-1,  2,  0, -2]])

>>> x = np.arange('1066-10-13', '1066-10-16', dtype=np.datetime64)
>>> np.diff(x)
array([1, 1], dtype='timedelta64[D]')
*)

val ones : ?dtype:Np.Dtype.t -> ?order:[`C | `F] -> shape:[`I of int | `Is of int list] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return a new array of given shape and type, filled with ones.

Parameters
----------
shape : int or sequence of ints
    Shape of the new array, e.g., ``(2, 3)`` or ``2``.
dtype : data-type, optional
    The desired data-type for the array, e.g., `numpy.int8`.  Default is
    `numpy.float64`.
order : {'C', 'F'}, optional, default: C
    Whether to store multi-dimensional data in row-major
    (C-style) or column-major (Fortran-style) order in
    memory.

Returns
-------
out : ndarray
    Array of ones with the given shape, dtype, and order.

See Also
--------
ones_like : Return an array of ones with shape and type of input.
empty : Return a new uninitialized array.
zeros : Return a new array setting values to zero.
full : Return a new array of given shape filled with value.


Examples
--------
>>> np.ones(5)
array([1., 1., 1., 1., 1.])

>>> np.ones((5,), dtype=int)
array([1, 1, 1, 1, 1])

>>> np.ones((2, 1))
array([[1.],
       [1.]])

>>> s = (2,2)
>>> np.ones(s)
array([[1.,  1.],
       [1.,  1.]])
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

module Interpnd : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module GradientEstimationWarning : sig
type tag = [`GradientEstimationWarning]
type t = [`BaseException | `GradientEstimationWarning | `Object] Obj.t
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

module NDInterpolatorBase : sig
type tag = [`NDInterpolatorBase]
type t = [`NDInterpolatorBase | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?fill_value:Py.Object.t -> ?ndim:Py.Object.t -> ?rescale:Py.Object.t -> ?need_contiguous:Py.Object.t -> ?need_values:Py.Object.t -> points:Py.Object.t -> values:Py.Object.t -> unit -> t
(**
Common routines for interpolators.

.. versionadded:: 0.9
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end


end

module Interpolate : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module Intp : sig
type tag = [`Int64]
type t = [`Int64 | `Object] Obj.t
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

module Poly1d : sig
type tag = [`Poly1d]
type t = [`Object | `Poly1d] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?r:bool -> ?variable:string -> c_or_r:[>`Ndarray] Np.Obj.t -> unit -> t
(**
A one-dimensional polynomial class.

A convenience class, used to encapsulate 'natural' operations on
polynomials so that said operations may take on their customary
form in code (see Examples).

Parameters
----------
c_or_r : array_like
    The polynomial's coefficients, in decreasing powers, or if
    the value of the second parameter is True, the polynomial's
    roots (values where the polynomial evaluates to 0).  For example,
    ``poly1d([1, 2, 3])`` returns an object that represents
    :math:`x^2 + 2x + 3`, whereas ``poly1d([1, 2, 3], True)`` returns
    one that represents :math:`(x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x -6`.
r : bool, optional
    If True, `c_or_r` specifies the polynomial's roots; the default
    is False.
variable : str, optional
    Changes the variable used when printing `p` from `x` to `variable`
    (see Examples).

Examples
--------
Construct the polynomial :math:`x^2 + 2x + 3`:

>>> p = np.poly1d([1, 2, 3])
>>> print(np.poly1d(p))
   2
1 x + 2 x + 3

Evaluate the polynomial at :math:`x = 0.5`:

>>> p(0.5)
4.25

Find the roots:

>>> p.r
array([-1.+1.41421356j, -1.-1.41421356j])
>>> p(p.r)
array([ -4.44089210e-16+0.j,  -4.44089210e-16+0.j]) # may vary

These numbers in the previous line represent (0, 0) to machine precision

Show the coefficients:

>>> p.c
array([1, 2, 3])

Display the order (the leading zero-coefficients are removed):

>>> p.order
2

Show the coefficient of the k-th power in the polynomial
(which is equivalent to ``p.c[-(i+1)]``):

>>> p[1]
2

Polynomials can be added, subtracted, multiplied, and divided
(returns quotient and remainder):

>>> p * p
poly1d([ 1,  4, 10, 12,  9])

>>> (p**3 + 4) / p
(poly1d([ 1.,  4., 10., 12.,  9.]), poly1d([4.]))

``asarray(p)`` gives the coefficient array, so polynomials can be
used in all functions that accept arrays:

>>> p**2 # square of polynomial
poly1d([ 1,  4, 10, 12,  9])

>>> np.square(p) # square of individual coefficients
array([1, 4, 9])

The variable used in the string representation of `p` can be modified,
using the `variable` parameter:

>>> p = np.poly1d([1,2,3], variable='z')
>>> print(p)
   2
1 z + 2 z + 3

Construct a polynomial from its roots:

>>> np.poly1d([1, 2], True)
poly1d([ 1., -3.,  2.])

This is the same polynomial as obtained by:

>>> np.poly1d([1, -1]) * np.poly1d([1, -2])
poly1d([ 1, -3,  2])
*)

val __getitem__ : val_:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
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

val deriv : ?m:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return a derivative of this polynomial.

Refer to `polyder` for full documentation.

See Also
--------
polyder : equivalent function
*)

val integ : ?m:Py.Object.t -> ?k:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return an antiderivative (indefinite integral) of this polynomial.

Refer to `polyint` for full documentation.

See Also
--------
polyint : equivalent function
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

val comb : ?exact:bool -> ?repetition:bool -> n:[`Ndarray of [>`Ndarray] Np.Obj.t | `I of int] -> k:[`Ndarray of [>`Ndarray] Np.Obj.t | `I of int] -> unit -> Py.Object.t
(**
The number of combinations of N things taken k at a time.

This is often expressed as 'N choose k'.

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

val interpn : ?method_:string -> ?bounds_error:bool -> ?fill_value:[`F of float | `I of int] -> points:Py.Object.t -> values:[>`Ndarray] Np.Obj.t -> xi:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Multidimensional interpolation on regular grids.

Parameters
----------
points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
    The points defining the regular grid in n dimensions.

values : array_like, shape (m1, ..., mn, ...)
    The data on the regular grid in n dimensions.

xi : ndarray of shape (..., ndim)
    The coordinates to sample the gridded data at

method : str, optional
    The method of interpolation to perform. Supported are 'linear' and
    'nearest', and 'splinef2d'. 'splinef2d' is only supported for
    2-dimensional data.

bounds_error : bool, optional
    If True, when interpolated values are requested outside of the
    domain of the input data, a ValueError is raised.
    If False, then `fill_value` is used.

fill_value : number, optional
    If provided, the value to use for points outside of the
    interpolation domain. If None, values outside
    the domain are extrapolated.  Extrapolation is not supported by method
    'splinef2d'.

Returns
-------
values_x : ndarray, shape xi.shape[:-1] + values.shape[ndim:]
    Interpolated values at input coordinates.

Notes
-----

.. versionadded:: 0.14

See also
--------
NearestNDInterpolator : Nearest neighbour interpolation on unstructured
                        data in N dimensions

LinearNDInterpolator : Piecewise linear interpolant on unstructured data
                       in N dimensions

RegularGridInterpolator : Linear and nearest-neighbor Interpolation on a
                          regular grid in arbitrary dimensions

RectBivariateSpline : Bivariate spline approximation over a rectangular mesh
*)

val lagrange : x:[>`Ndarray] Np.Obj.t -> w:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Return a Lagrange interpolating polynomial.

Given two 1-D arrays `x` and `w,` returns the Lagrange interpolating
polynomial through the points ``(x, w)``.

Warning: This implementation is numerically unstable. Do not expect to
be able to use more than about 20 points even if they are chosen optimally.

Parameters
----------
x : array_like
    `x` represents the x-coordinates of a set of datapoints.
w : array_like
    `w` represents the y-coordinates of a set of datapoints, i.e. f(`x`).

Returns
-------
lagrange : `numpy.poly1d` instance
    The Lagrange interpolating polynomial.

Examples
--------
Interpolate :math:`f(x) = x^3` by 3 points.

>>> from scipy.interpolate import lagrange
>>> x = np.array([0, 1, 2])
>>> y = x**3
>>> poly = lagrange(x, y)

Since there are only 3 points, Lagrange polynomial has degree 2. Explicitly,
it is given by

.. math::

    \begin{aligned}
        L(x) &= 1\times \frac{x (x - 2)}{-1} + 8\times \frac{x (x-1)}{2} \\
             &= x (-2 + 3x)
    \end{aligned}

>>> from numpy.polynomial.polynomial import Polynomial
>>> Polynomial(poly).coef
array([ 3., -2.,  0.])
*)

val make_interp_spline : ?k:int -> ?t:[>`Ndarray] Np.Obj.t -> ?bc_type:Py.Object.t -> ?axis:int -> ?check_finite:bool -> x:[>`Ndarray] Np.Obj.t -> y:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Compute the (coefficients of) interpolating B-spline.

Parameters
----------
x : array_like, shape (n,)
    Abscissas.
y : array_like, shape (n, ...)
    Ordinates.
k : int, optional
    B-spline degree. Default is cubic, k=3.
t : array_like, shape (nt + k + 1,), optional.
    Knots.
    The number of knots needs to agree with the number of datapoints and
    the number of derivatives at the edges. Specifically, ``nt - n`` must
    equal ``len(deriv_l) + len(deriv_r)``.
bc_type : 2-tuple or None
    Boundary conditions.
    Default is None, which means choosing the boundary conditions
    automatically. Otherwise, it must be a length-two tuple where the first
    element sets the boundary conditions at ``x[0]`` and the second
    element sets the boundary conditions at ``x[-1]``. Each of these must
    be an iterable of pairs ``(order, value)`` which gives the values of
    derivatives of specified orders at the given edge of the interpolation
    interval.
    Alternatively, the following string aliases are recognized:

    * ``'clamped'``: The first derivatives at the ends are zero. This is
       equivalent to ``bc_type=([(1, 0.0)], [(1, 0.0)])``.
    * ``'natural'``: The second derivatives at ends are zero. This is
      equivalent to ``bc_type=([(2, 0.0)], [(2, 0.0)])``.
    * ``'not-a-knot'`` (default): The first and second segments are the same
      polynomial. This is equivalent to having ``bc_type=None``.

axis : int, optional
    Interpolation axis. Default is 0.
check_finite : bool, optional
    Whether to check that the input arrays contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.
    Default is True.

Returns
-------
b : a BSpline object of the degree ``k`` and with knots ``t``.

Examples
--------

Use cubic interpolation on Chebyshev nodes:

>>> def cheb_nodes(N):
...     jj = 2.*np.arange(N) + 1
...     x = np.cos(np.pi * jj / 2 / N)[::-1]
...     return x

>>> x = cheb_nodes(20)
>>> y = np.sqrt(1 - x**2)

>>> from scipy.interpolate import BSpline, make_interp_spline
>>> b = make_interp_spline(x, y)
>>> np.allclose(b(x), y)
True

Note that the default is a cubic spline with a not-a-knot boundary condition

>>> b.k
3

Here we use a 'natural' spline, with zero 2nd derivatives at edges:

>>> l, r = [(2, 0.0)], [(2, 0.0)]
>>> b_n = make_interp_spline(x, y, bc_type=(l, r))  # or, bc_type='natural'
>>> np.allclose(b_n(x), y)
True
>>> x0, x1 = x[0], x[-1]
>>> np.allclose([b_n(x0, 2), b_n(x1, 2)], [0, 0])
True

Interpolation of parametric curves is also supported. As an example, we
compute a discretization of a snail curve in polar coordinates

>>> phi = np.linspace(0, 2.*np.pi, 40)
>>> r = 0.3 + np.cos(phi)
>>> x, y = r*np.cos(phi), r*np.sin(phi)  # convert to Cartesian coordinates

Build an interpolating curve, parameterizing it by the angle

>>> from scipy.interpolate import make_interp_spline
>>> spl = make_interp_spline(phi, np.c_[x, y])

Evaluate the interpolant on a finer grid (note that we transpose the result
to unpack it into a pair of x- and y-arrays)

>>> phi_new = np.linspace(0, 2.*np.pi, 100)
>>> x_new, y_new = spl(phi_new).T

Plot the result

>>> import matplotlib.pyplot as plt
>>> plt.plot(x, y, 'o')
>>> plt.plot(x_new, y_new, '-')
>>> plt.show()

See Also
--------
BSpline : base class representing the B-spline objects
CubicSpline : a cubic spline in the polynomial basis
make_lsq_spline : a similar factory function for spline fitting
UnivariateSpline : a wrapper over FITPACK spline fitting routines
splrep : a wrapper over FITPACK spline fitting routines
*)

val prod : Py.Object.t -> Py.Object.t
(**
Product of a list of numbers; ~40x faster vs np.prod for Python tuples
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

val searchsorted : ?side:[`Left | `Right] -> ?sorter:Py.Object.t -> a:Py.Object.t -> v:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Find indices where elements should be inserted to maintain order.

Find the indices into a sorted array `a` such that, if the
corresponding elements in `v` were inserted before the indices, the
order of `a` would be preserved.

Assuming that `a` is sorted:

======  ============================
`side`  returned index `i` satisfies
======  ============================
left    ``a[i-1] < v <= a[i]``
right   ``a[i-1] <= v < a[i]``
======  ============================

Parameters
----------
a : 1-D array_like
    Input array. If `sorter` is None, then it must be sorted in
    ascending order, otherwise `sorter` must be an array of indices
    that sort it.
v : array_like
    Values to insert into `a`.
side : {'left', 'right'}, optional
    If 'left', the index of the first suitable location found is given.
    If 'right', return the last such index.  If there is no suitable
    index, return either 0 or N (where N is the length of `a`).
sorter : 1-D array_like, optional
    Optional array of integer indices that sort array a into ascending
    order. They are typically the result of argsort.

    .. versionadded:: 1.7.0

Returns
-------
indices : array of ints
    Array of insertion points with the same shape as `v`.

See Also
--------
sort : Return a sorted copy of an array.
histogram : Produce histogram from 1-D data.

Notes
-----
Binary search is used to find the required insertion points.

As of NumPy 1.4.0 `searchsorted` works with real/complex arrays containing
`nan` values. The enhanced sort order is documented in `sort`.

This function uses the same algorithm as the builtin python `bisect.bisect_left`
(``side='left'``) and `bisect.bisect_right` (``side='right'``) functions,
which is also vectorized in the `v` argument.

Examples
--------
>>> np.searchsorted([1,2,3,4,5], 3)
2
>>> np.searchsorted([1,2,3,4,5], 3, side='right')
3
>>> np.searchsorted([1,2,3,4,5], [-10, 10, 2, 3])
array([0, 5, 1, 2])
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


end

module Ndgriddata : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module CKDTree : sig
type tag = [`CKDTree]
type t = [`CKDTree | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?leafsize:Py.Object.t -> ?compact_nodes:bool -> ?copy_data:bool -> ?balanced_tree:bool -> ?boxsize:[`S of string | `F of float | `Ndarray of [>`Ndarray] Np.Obj.t | `I of int | `Bool of bool] -> data:[>`Ndarray] Np.Obj.t -> unit -> t
(**
cKDTree(data, leafsize=16, compact_nodes=True, copy_data=False,
        balanced_tree=True, boxsize=None)

kd-tree for quick nearest-neighbor lookup

This class provides an index into a set of k-dimensional points
which can be used to rapidly look up the nearest neighbors of any
point.

The algorithm used is described in Maneewongvatana and Mount 1999.
The general idea is that the kd-tree is a binary trie, each of whose
nodes represents an axis-aligned hyperrectangle. Each node specifies
an axis and splits the set of points based on whether their coordinate
along that axis is greater than or less than a particular value.

During construction, the axis and splitting point are chosen by the
'sliding midpoint' rule, which ensures that the cells do not all
become long and thin.

The tree can be queried for the r closest neighbors of any given point
(optionally returning only those within some maximum distance of the
point). It can also be queried, with a substantial gain in efficiency,
for the r approximate closest neighbors.

For large dimensions (20 is already large) do not expect this to run
significantly faster than brute force. High-dimensional nearest-neighbor
queries are a substantial open problem in computer science.

Parameters
----------
data : array_like, shape (n,m)
    The n data points of dimension m to be indexed. This array is
    not copied unless this is necessary to produce a contiguous
    array of doubles, and so modifying this data will result in
    bogus results. The data are also copied if the kd-tree is built
    with copy_data=True.
leafsize : positive int, optional
    The number of points at which the algorithm switches over to
    brute-force. Default: 16.
compact_nodes : bool, optional
    If True, the kd-tree is built to shrink the hyperrectangles to
    the actual data range. This usually gives a more compact tree that
    is robust against degenerated input data and gives faster queries
    at the expense of longer build time. Default: True.
copy_data : bool, optional
    If True the data is always copied to protect the kd-tree against
    data corruption. Default: False.
balanced_tree : bool, optional
    If True, the median is used to split the hyperrectangles instead of
    the midpoint. This usually gives a more compact tree and
    faster queries at the expense of longer build time. Default: True.
boxsize : array_like or scalar, optional
    Apply a m-d toroidal topology to the KDTree.. The topology is generated
    by :math:`x_i + n_i L_i` where :math:`n_i` are integers and :math:`L_i`
    is the boxsize along i-th dimension. The input data shall be wrapped
    into :math:`[0, L_i)`. A ValueError is raised if any of the data is
    outside of this bound.

Attributes
----------
data : ndarray, shape (n,m)
    The n data points of dimension m to be indexed. This array is
    not copied unless this is necessary to produce a contiguous
    array of doubles. The data are also copied if the kd-tree is built
    with `copy_data=True`.
leafsize : positive int
    The number of points at which the algorithm switches over to
    brute-force.
m : int
    The dimension of a single data-point.
n : int
    The number of data points.
maxes : ndarray, shape (m,)
    The maximum value in each dimension of the n data points.
mins : ndarray, shape (m,)
    The minimum value in each dimension of the n data points.
tree : object, class cKDTreeNode
    This class exposes a Python view of the root node in the cKDTree object.
size : int
    The number of nodes in the tree.

See Also
--------
KDTree : Implementation of `cKDTree` in pure Python
*)

val count_neighbors : ?p:float -> ?weights:[`Ndarray of [>`Ndarray] Np.Obj.t | `Tuple of Py.Object.t] -> ?cumulative:bool -> other:Py.Object.t -> r:[`Ndarray of [>`Ndarray] Np.Obj.t | `F of float] -> [> tag] Obj.t -> Py.Object.t
(**
count_neighbors(self, other, r, p=2., weights=None, cumulative=True)

Count how many nearby pairs can be formed. (pair-counting)

Count the number of pairs (x1,x2) can be formed, with x1 drawn
from self and x2 drawn from ``other``, and where
``distance(x1, x2, p) <= r``.

Data points on self and other are optionally weighted by the ``weights``
argument. (See below)

The algorithm we implement here is based on [1]_. See notes for further discussion.

Parameters
----------
other : cKDTree instance
    The other tree to draw points from, can be the same tree as self.
r : float or one-dimensional array of floats
    The radius to produce a count for. Multiple radii are searched with
    a single tree traversal.
    If the count is non-cumulative(``cumulative=False``), ``r`` defines
    the edges of the bins, and must be non-decreasing.
p : float, optional
    1<=p<=infinity.
    Which Minkowski p-norm to use.
    Default 2.0.
    A finite large p may cause a ValueError if overflow can occur.
weights : tuple, array_like, or None, optional
    If None, the pair-counting is unweighted.
    If given as a tuple, weights[0] is the weights of points in ``self``, and
    weights[1] is the weights of points in ``other``; either can be None to
    indicate the points are unweighted.
    If given as an array_like, weights is the weights of points in ``self``
    and ``other``. For this to make sense, ``self`` and ``other`` must be the
    same tree. If ``self`` and ``other`` are two different trees, a ``ValueError``
    is raised.
    Default: None
cumulative : bool, optional
    Whether the returned counts are cumulative. When cumulative is set to ``False``
    the algorithm is optimized to work with a large number of bins (>10) specified
    by ``r``. When ``cumulative`` is set to True, the algorithm is optimized to work
    with a small number of ``r``. Default: True

Returns
-------
result : scalar or 1-D array
    The number of pairs. For unweighted counts, the result is integer.
    For weighted counts, the result is float.
    If cumulative is False, ``result[i]`` contains the counts with
    ``(-inf if i == 0 else r[i-1]) < R <= r[i]``

Notes
-----
Pair-counting is the basic operation used to calculate the two point
correlation functions from a data set composed of position of objects.

Two point correlation function measures the clustering of objects and
is widely used in cosmology to quantify the large scale structure
in our Universe, but it may be useful for data analysis in other fields
where self-similar assembly of objects also occur.

The Landy-Szalay estimator for the two point correlation function of
``D`` measures the clustering signal in ``D``. [2]_

For example, given the position of two sets of objects,

- objects ``D`` (data) contains the clustering signal, and

- objects ``R`` (random) that contains no signal,

.. math::

     \xi(r) = \frac{<D, D> - 2 f <D, R> + f^2<R, R>}{f^2<R, R>},

where the brackets represents counting pairs between two data sets
in a finite bin around ``r`` (distance), corresponding to setting
`cumulative=False`, and ``f = float(len(D)) / float(len(R))`` is the
ratio between number of objects from data and random.

The algorithm implemented here is loosely based on the dual-tree
algorithm described in [1]_. We switch between two different
pair-cumulation scheme depending on the setting of ``cumulative``.
The computing time of the method we use when for
``cumulative == False`` does not scale with the total number of bins.
The algorithm for ``cumulative == True`` scales linearly with the
number of bins, though it is slightly faster when only
1 or 2 bins are used. [5]_.

As an extension to the naive pair-counting,
weighted pair-counting counts the product of weights instead
of number of pairs.
Weighted pair-counting is used to estimate marked correlation functions
([3]_, section 2.2),
or to properly calculate the average of data per distance bin
(e.g. [4]_, section 2.1 on redshift).

.. [1] Gray and Moore,
       'N-body problems in statistical learning',
       Mining the sky, 2000,
       https://arxiv.org/abs/astro-ph/0012333

.. [2] Landy and Szalay,
       'Bias and variance of angular correlation functions',
       The Astrophysical Journal, 1993,
       http://adsabs.harvard.edu/abs/1993ApJ...412...64L

.. [3] Sheth, Connolly and Skibba,
       'Marked correlations in galaxy formation models',
       Arxiv e-print, 2005,
       https://arxiv.org/abs/astro-ph/0511773

.. [4] Hawkins, et al.,
       'The 2dF Galaxy Redshift Survey: correlation functions,
       peculiar velocities and the matter density of the Universe',
       Monthly Notices of the Royal Astronomical Society, 2002,
       http://adsabs.harvard.edu/abs/2003MNRAS.346...78H

.. [5] https://github.com/scipy/scipy/pull/5647#issuecomment-168474926
*)

val query_ball_point : ?p:float -> ?eps:Py.Object.t -> x:[`Ndarray of [>`Ndarray] Np.Obj.t | `Shape_tuple_self_m_ of Py.Object.t] -> r:[`Ndarray of [>`Ndarray] Np.Obj.t | `F of float] -> [> tag] Obj.t -> Py.Object.t
(**
query_ball_point(self, x, r, p=2., eps=0)

Find all points within distance r of point(s) x.

Parameters
----------
x : array_like, shape tuple + (self.m,)
    The point or points to search for neighbors of.
r : array_like, float
    The radius of points to return, shall broadcast to the length of x.
p : float, optional
    Which Minkowski p-norm to use.  Should be in the range [1, inf].
    A finite large p may cause a ValueError if overflow can occur.
eps : nonnegative float, optional
    Approximate search. Branches of the tree are not explored if their
    nearest points are further than ``r / (1 + eps)``, and branches are
    added in bulk if their furthest points are nearer than
    ``r * (1 + eps)``.
n_jobs : int, optional
    Number of jobs to schedule for parallel processing. If -1 is given
    all processors are used. Default: 1.
return_sorted : bool, optional
    Sorts returned indicies if True and does not sort them if False. If
    None, does not sort single point queries, but does sort
    multi-point queries which was the behavior before this option
    was added.

    .. versionadded:: 1.2.0
return_length: bool, optional
    Return the number of points inside the radius instead of a list
    of the indices.
    .. versionadded:: 1.3.0

Returns
-------
results : list or array of lists
    If `x` is a single point, returns a list of the indices of the
    neighbors of `x`. If `x` is an array of points, returns an object
    array of shape tuple containing lists of neighbors.

Notes
-----
If you have many points whose neighbors you want to find, you may save
substantial amounts of time by putting them in a cKDTree and using
query_ball_tree.

Examples
--------
>>> from scipy import spatial
>>> x, y = np.mgrid[0:4, 0:4]
>>> points = np.c_[x.ravel(), y.ravel()]
>>> tree = spatial.cKDTree(points)
>>> tree.query_ball_point([2, 0], 1)
[4, 8, 9, 12]
*)

val query_ball_tree : ?p:float -> ?eps:float -> other:Py.Object.t -> r:float -> [> tag] Obj.t -> Py.Object.t
(**
query_ball_tree(self, other, r, p=2., eps=0)

Find all pairs of points whose distance is at most r

Parameters
----------
other : cKDTree instance
    The tree containing points to search against.
r : float
    The maximum distance, has to be positive.
p : float, optional
    Which Minkowski norm to use.  `p` has to meet the condition
    ``1 <= p <= infinity``.
    A finite large p may cause a ValueError if overflow can occur.
eps : float, optional
    Approximate search.  Branches of the tree are not explored
    if their nearest points are further than ``r/(1+eps)``, and
    branches are added in bulk if their furthest points are nearer
    than ``r * (1+eps)``.  `eps` has to be non-negative.

Returns
-------
results : list of lists
    For each element ``self.data[i]`` of this tree, ``results[i]`` is a
    list of the indices of its neighbors in ``other.data``.
*)

val query_pairs : ?p:float -> ?eps:float -> r:float -> [> tag] Obj.t -> Py.Object.t
(**
query_pairs(self, r, p=2., eps=0)

Find all pairs of points whose distance is at most r.

Parameters
----------
r : positive float
    The maximum distance.
p : float, optional
    Which Minkowski norm to use.  ``p`` has to meet the condition
    ``1 <= p <= infinity``.
    A finite large p may cause a ValueError if overflow can occur.
eps : float, optional
    Approximate search.  Branches of the tree are not explored
    if their nearest points are further than ``r/(1+eps)``, and
    branches are added in bulk if their furthest points are nearer
    than ``r * (1+eps)``.  `eps` has to be non-negative.
output_type : string, optional
    Choose the output container, 'set' or 'ndarray'. Default: 'set'

Returns
-------
results : set or ndarray
    Set of pairs ``(i,j)``, with ``i < j``, for which the corresponding
    positions are close. If output_type is 'ndarray', an ndarry is
    returned instead of a set.
*)

val sparse_distance_matrix : ?p:[`T1_p_infinity of Py.Object.t | `F of float] -> other:Py.Object.t -> max_distance:float -> [> tag] Obj.t -> Py.Object.t
(**
sparse_distance_matrix(self, other, max_distance, p=2.)

Compute a sparse distance matrix

Computes a distance matrix between two cKDTrees, leaving as zero
any distance greater than max_distance.

Parameters
----------
other : cKDTree

max_distance : positive float

p : float, 1<=p<=infinity
    Which Minkowski p-norm to use.
    A finite large p may cause a ValueError if overflow can occur.

output_type : string, optional
    Which container to use for output data. Options: 'dok_matrix',
    'coo_matrix', 'dict', or 'ndarray'. Default: 'dok_matrix'.

Returns
-------
result : dok_matrix, coo_matrix, dict or ndarray
    Sparse matrix representing the results in 'dictionary of keys'
    format. If a dict is returned the keys are (i,j) tuples of indices.
    If output_type is 'ndarray' a record array with fields 'i', 'j',
    and 'v' is returned,
*)


(** Attribute data: get value or raise Not_found if None.*)
val data : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute data: get value as an option. *)
val data_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute leafsize: get value or raise Not_found if None.*)
val leafsize : t -> Py.Object.t

(** Attribute leafsize: get value as an option. *)
val leafsize_opt : t -> (Py.Object.t) option


(** Attribute m: get value or raise Not_found if None.*)
val m : t -> int

(** Attribute m: get value as an option. *)
val m_opt : t -> (int) option


(** Attribute n: get value or raise Not_found if None.*)
val n : t -> int

(** Attribute n: get value as an option. *)
val n_opt : t -> (int) option


(** Attribute maxes: get value or raise Not_found if None.*)
val maxes : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute maxes: get value as an option. *)
val maxes_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute mins: get value or raise Not_found if None.*)
val mins : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute mins: get value as an option. *)
val mins_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute tree: get value or raise Not_found if None.*)
val tree : t -> Py.Object.t

(** Attribute tree: get value as an option. *)
val tree_opt : t -> (Py.Object.t) option


(** Attribute size: get value or raise Not_found if None.*)
val size : t -> int

(** Attribute size: get value as an option. *)
val size_opt : t -> (int) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val griddata : ?method_:[`Linear | `Nearest | `Cubic] -> ?fill_value:float -> ?rescale:bool -> points:Py.Object.t -> values:Py.Object.t -> xi:Py.Object.t -> unit -> Py.Object.t
(**
Interpolate unstructured D-dimensional data.

Parameters
----------
points : 2-D ndarray of floats with shape (n, D), or length D tuple of 1-D ndarrays with shape (n,).
    Data point coordinates. 
values : ndarray of float or complex, shape (n,)
    Data values.
xi : 2-D ndarray of floats with shape (m, D), or length D tuple of ndarrays broadcastable to the same shape.
    Points at which to interpolate data.
method : {'linear', 'nearest', 'cubic'}, optional
    Method of interpolation. One of

    ``nearest``
      return the value at the data point closest to
      the point of interpolation.  See `NearestNDInterpolator` for
      more details.

    ``linear``
      tessellate the input point set to n-dimensional
      simplices, and interpolate linearly on each simplex.  See
      `LinearNDInterpolator` for more details.

    ``cubic`` (1-D)
      return the value determined from a cubic
      spline.

    ``cubic`` (2-D)
      return the value determined from a
      piecewise cubic, continuously differentiable (C1), and
      approximately curvature-minimizing polynomial surface. See
      `CloughTocher2DInterpolator` for more details.
fill_value : float, optional
    Value used to fill in for requested points outside of the
    convex hull of the input points.  If not provided, then the
    default is ``nan``. This option has no effect for the
    'nearest' method.
rescale : bool, optional
    Rescale points to unit cube before performing interpolation.
    This is useful if some of the input dimensions have
    incommensurable units and differ by many orders of magnitude.

    .. versionadded:: 0.14.0
    
Returns
-------
ndarray
    Array of interpolated values.

Notes
-----

.. versionadded:: 0.9

Examples
--------

Suppose we want to interpolate the 2-D function

>>> def func(x, y):
...     return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2

on a grid in [0, 1]x[0, 1]

>>> grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

but we only know its values at 1000 data points:

>>> points = np.random.rand(1000, 2)
>>> values = func(points[:,0], points[:,1])

This can be done with `griddata` -- below we try out all of the
interpolation methods:

>>> from scipy.interpolate import griddata
>>> grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
>>> grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
>>> grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')

One can see that the exact result is reproduced by all of the
methods to some degree, but for this smooth function the piecewise
cubic interpolant gives the best results:

>>> import matplotlib.pyplot as plt
>>> plt.subplot(221)
>>> plt.imshow(func(grid_x, grid_y).T, extent=(0,1,0,1), origin='lower')
>>> plt.plot(points[:,0], points[:,1], 'k.', ms=1)
>>> plt.title('Original')
>>> plt.subplot(222)
>>> plt.imshow(grid_z0.T, extent=(0,1,0,1), origin='lower')
>>> plt.title('Nearest')
>>> plt.subplot(223)
>>> plt.imshow(grid_z1.T, extent=(0,1,0,1), origin='lower')
>>> plt.title('Linear')
>>> plt.subplot(224)
>>> plt.imshow(grid_z2.T, extent=(0,1,0,1), origin='lower')
>>> plt.title('Cubic')
>>> plt.gcf().set_size_inches(6, 6)
>>> plt.show()
*)


end

module Polyint : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val approximate_taylor_polynomial : ?order:int -> f:Py.Object.t -> x:[`F of float | `I of int | `Bool of bool | `S of string] -> degree:int -> scale:float -> unit -> Py.Object.t
(**
Estimate the Taylor polynomial of f at x by polynomial fitting.

Parameters
----------
f : callable
    The function whose Taylor polynomial is sought. Should accept
    a vector of `x` values.
x : scalar
    The point at which the polynomial is to be evaluated.
degree : int
    The degree of the Taylor polynomial
scale : scalar
    The width of the interval to use to evaluate the Taylor polynomial.
    Function values spread over a range this wide are used to fit the
    polynomial. Must be chosen carefully.
order : int or None, optional
    The order of the polynomial to be used in the fitting; `f` will be
    evaluated ``order+1`` times. If None, use `degree`.

Returns
-------
p : poly1d instance
    The Taylor polynomial (translated to the origin, so that
    for example p(0)=f(x)).

Notes
-----
The appropriate choice of 'scale' is a trade-off; too large and the
function differs from its Taylor polynomial too much to get a good
answer, too small and round-off errors overwhelm the higher-order terms.
The algorithm used becomes numerically unstable around order 30 even
under ideal circumstances.

Choosing order somewhat larger than degree may improve the higher-order
terms.
*)

val barycentric_interpolate : ?axis:int -> xi:[>`Ndarray] Np.Obj.t -> yi:[>`Ndarray] Np.Obj.t -> x:[`S of string | `F of float | `Ndarray of [>`Ndarray] Np.Obj.t | `I of int | `Bool of bool] -> unit -> Py.Object.t
(**
Convenience function for polynomial interpolation.

Constructs a polynomial that passes through a given set of points,
then evaluates the polynomial. For reasons of numerical stability,
this function does not compute the coefficients of the polynomial.

This function uses a 'barycentric interpolation' method that treats
the problem as a special case of rational function interpolation.
This algorithm is quite stable, numerically, but even in a world of
exact computation, unless the `x` coordinates are chosen very
carefully - Chebyshev zeros (e.g. cos(i*pi/n)) are a good choice -
polynomial interpolation itself is a very ill-conditioned process
due to the Runge phenomenon.

Parameters
----------
xi : array_like
    1-d array of x coordinates of the points the polynomial should
    pass through
yi : array_like
    The y coordinates of the points the polynomial should pass through.
x : scalar or array_like
    Points to evaluate the interpolator at.
axis : int, optional
    Axis in the yi array corresponding to the x-coordinate values.

Returns
-------
y : scalar or array_like
    Interpolated values. Shape is determined by replacing
    the interpolation axis in the original array with the shape of x.

See Also
--------
BarycentricInterpolator

Notes
-----
Construction of the interpolation weights is a relatively slow process.
If you want to call this many times with the same xi (but possibly
varying yi or x) you should use the class `BarycentricInterpolator`.
This is what this function uses internally.
*)

val factorial : ?exact:bool -> n:[`I of int | `Array_like_of_ints of Py.Object.t] -> unit -> Py.Object.t
(**
The factorial of a number or array of numbers.

The factorial of non-negative integer `n` is the product of all
positive integers less than or equal to `n`::

    n! = n * (n - 1) * (n - 2) * ... * 1

Parameters
----------
n : int or array_like of ints
    Input values.  If ``n < 0``, the return value is 0.
exact : bool, optional
    If True, calculate the answer exactly using long integer arithmetic.
    If False, result is approximated in floating point rapidly using the
    `gamma` function.
    Default is False.

Returns
-------
nf : float or int or ndarray
    Factorial of `n`, as integer or float depending on `exact`.

Notes
-----
For arrays with ``exact=True``, the factorial is computed only once, for
the largest input, with each other result computed in the process.
The output dtype is increased to ``int64`` or ``object`` if necessary.

With ``exact=False`` the factorial is approximated using the gamma
function:

.. math:: n! = \Gamma(n+1)

Examples
--------
>>> from scipy.special import factorial
>>> arr = np.array([3, 4, 5])
>>> factorial(arr, exact=False)
array([   6.,   24.,  120.])
>>> factorial(arr, exact=True)
array([  6,  24, 120])
>>> factorial(5, exact=True)
120L
*)

val krogh_interpolate : ?der:[`Ndarray of [>`Ndarray] Np.Obj.t | `I of int] -> ?axis:int -> xi:[>`Ndarray] Np.Obj.t -> yi:[>`Ndarray] Np.Obj.t -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Convenience function for polynomial interpolation.

See `KroghInterpolator` for more details.

Parameters
----------
xi : array_like
    Known x-coordinates.
yi : array_like
    Known y-coordinates, of shape ``(xi.size, R)``.  Interpreted as
    vectors of length R, or scalars if R=1.
x : array_like
    Point or points at which to evaluate the derivatives.
der : int or list, optional
    How many derivatives to extract; None for all potentially
    nonzero derivatives (that is a number equal to the number
    of points), or a list of derivatives to extract. This number
    includes the function value as 0th derivative.
axis : int, optional
    Axis in the yi array corresponding to the x-coordinate values.

Returns
-------
d : ndarray
    If the interpolator's values are R-dimensional then the
    returned array will be the number of derivatives by N by R.
    If `x` is a scalar, the middle dimension will be dropped; if
    the `yi` are scalars then the last dimension will be dropped.

See Also
--------
KroghInterpolator

Notes
-----
Construction of the interpolating polynomial is a relatively expensive
process. If you want to evaluate it repeatedly consider using the class
KroghInterpolator (which is what this function uses).
*)


end

module Rbf' : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val callable : Py.Object.t -> Py.Object.t
(**
None
*)

val cdist : ?metric:[`Callable of Py.Object.t | `S of string] -> ?kwargs:(string * Py.Object.t) list -> xa:[>`Ndarray] Np.Obj.t -> xb:[>`Ndarray] Np.Obj.t -> Py.Object.t list -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute distance between each pair of the two collections of inputs.

See Notes for common calling conventions.

Parameters
----------
XA : ndarray
    An :math:`m_A` by :math:`n` array of :math:`m_A`
    original observations in an :math:`n`-dimensional space.
    Inputs are converted to float type.
XB : ndarray
    An :math:`m_B` by :math:`n` array of :math:`m_B`
    original observations in an :math:`n`-dimensional space.
    Inputs are converted to float type.
metric : str or callable, optional
    The distance metric to use.  If a string, the distance function can be
    'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
    'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
    'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
    'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
    'wminkowski', 'yule'.
*args : tuple. Deprecated.
    Additional arguments should be passed as keyword arguments
**kwargs : dict, optional
    Extra arguments to `metric`: refer to each metric documentation for a
    list of all possible arguments.

    Some possible arguments:

    p : scalar
    The p-norm to apply for Minkowski, weighted and unweighted.
    Default: 2.

    w : ndarray
    The weight vector for metrics that support weights (e.g., Minkowski).

    V : ndarray
    The variance vector for standardized Euclidean.
    Default: var(vstack([XA, XB]), axis=0, ddof=1)

    VI : ndarray
    The inverse of the covariance matrix for Mahalanobis.
    Default: inv(cov(vstack([XA, XB].T))).T

    out : ndarray
    The output array
    If not None, the distance matrix Y is stored in this array.
    Note: metric independent, it will become a regular keyword arg in a
    future scipy version

Returns
-------
Y : ndarray
    A :math:`m_A` by :math:`m_B` distance matrix is returned.
    For each :math:`i` and :math:`j`, the metric
    ``dist(u=XA[i], v=XB[j])`` is computed and stored in the
    :math:`ij` th entry.

Raises
------
ValueError
    An exception is thrown if `XA` and `XB` do not have
    the same number of columns.

Notes
-----
The following are common calling conventions:

1. ``Y = cdist(XA, XB, 'euclidean')``

   Computes the distance between :math:`m` points using
   Euclidean distance (2-norm) as the distance metric between the
   points. The points are arranged as :math:`m`
   :math:`n`-dimensional row vectors in the matrix X.

2. ``Y = cdist(XA, XB, 'minkowski', p=2.)``

   Computes the distances using the Minkowski distance
   :math:`||u-v||_p` (:math:`p`-norm) where :math:`p \geq 1`.

3. ``Y = cdist(XA, XB, 'cityblock')``

   Computes the city block or Manhattan distance between the
   points.

4. ``Y = cdist(XA, XB, 'seuclidean', V=None)``

   Computes the standardized Euclidean distance. The standardized
   Euclidean distance between two n-vectors ``u`` and ``v`` is

   .. math::

      \sqrt{\sum {(u_i-v_i)^2 / V[x_i]}}.

   V is the variance vector; V[i] is the variance computed over all
   the i'th components of the points. If not passed, it is
   automatically computed.

5. ``Y = cdist(XA, XB, 'sqeuclidean')``

   Computes the squared Euclidean distance :math:`||u-v||_2^2` between
   the vectors.

6. ``Y = cdist(XA, XB, 'cosine')``

   Computes the cosine distance between vectors u and v,

   .. math::

      1 - \frac{u \cdot v}
               {{ ||u|| }_2 { ||v|| }_2}

   where :math:`||*||_2` is the 2-norm of its argument ``*``, and
   :math:`u \cdot v` is the dot product of :math:`u` and :math:`v`.

7. ``Y = cdist(XA, XB, 'correlation')``

   Computes the correlation distance between vectors u and v. This is

   .. math::

      1 - \frac{(u - \bar{u}) \cdot (v - \bar{v})}
               {{ ||(u - \bar{u})|| }_2 { ||(v - \bar{v})|| }_2}

   where :math:`\bar{v}` is the mean of the elements of vector v,
   and :math:`x \cdot y` is the dot product of :math:`x` and :math:`y`.


8. ``Y = cdist(XA, XB, 'hamming')``

   Computes the normalized Hamming distance, or the proportion of
   those vector elements between two n-vectors ``u`` and ``v``
   which disagree. To save memory, the matrix ``X`` can be of type
   boolean.

9. ``Y = cdist(XA, XB, 'jaccard')``

   Computes the Jaccard distance between the points. Given two
   vectors, ``u`` and ``v``, the Jaccard distance is the
   proportion of those elements ``u[i]`` and ``v[i]`` that
   disagree where at least one of them is non-zero.

10. ``Y = cdist(XA, XB, 'chebyshev')``

   Computes the Chebyshev distance between the points. The
   Chebyshev distance between two n-vectors ``u`` and ``v`` is the
   maximum norm-1 distance between their respective elements. More
   precisely, the distance is given by

   .. math::

      d(u,v) = \max_i { |u_i-v_i| }.

11. ``Y = cdist(XA, XB, 'canberra')``

   Computes the Canberra distance between the points. The
   Canberra distance between two points ``u`` and ``v`` is

   .. math::

     d(u,v) = \sum_i \frac{ |u_i-v_i| }
                          { |u_i|+|v_i| }.

12. ``Y = cdist(XA, XB, 'braycurtis')``

   Computes the Bray-Curtis distance between the points. The
   Bray-Curtis distance between two points ``u`` and ``v`` is


   .. math::

        d(u,v) = \frac{\sum_i (|u_i-v_i|)}
                      {\sum_i (|u_i+v_i|)}

13. ``Y = cdist(XA, XB, 'mahalanobis', VI=None)``

   Computes the Mahalanobis distance between the points. The
   Mahalanobis distance between two points ``u`` and ``v`` is
   :math:`\sqrt{(u-v)(1/V)(u-v)^T}` where :math:`(1/V)` (the ``VI``
   variable) is the inverse covariance. If ``VI`` is not None,
   ``VI`` will be used as the inverse covariance matrix.

14. ``Y = cdist(XA, XB, 'yule')``

   Computes the Yule distance between the boolean
   vectors. (see `yule` function documentation)

15. ``Y = cdist(XA, XB, 'matching')``

   Synonym for 'hamming'.

16. ``Y = cdist(XA, XB, 'dice')``

   Computes the Dice distance between the boolean vectors. (see
   `dice` function documentation)

17. ``Y = cdist(XA, XB, 'kulsinski')``

   Computes the Kulsinski distance between the boolean
   vectors. (see `kulsinski` function documentation)

18. ``Y = cdist(XA, XB, 'rogerstanimoto')``

   Computes the Rogers-Tanimoto distance between the boolean
   vectors. (see `rogerstanimoto` function documentation)

19. ``Y = cdist(XA, XB, 'russellrao')``

   Computes the Russell-Rao distance between the boolean
   vectors. (see `russellrao` function documentation)

20. ``Y = cdist(XA, XB, 'sokalmichener')``

   Computes the Sokal-Michener distance between the boolean
   vectors. (see `sokalmichener` function documentation)

21. ``Y = cdist(XA, XB, 'sokalsneath')``

   Computes the Sokal-Sneath distance between the vectors. (see
   `sokalsneath` function documentation)


22. ``Y = cdist(XA, XB, 'wminkowski', p=2., w=w)``

   Computes the weighted Minkowski distance between the
   vectors. (see `wminkowski` function documentation)

23. ``Y = cdist(XA, XB, f)``

   Computes the distance between all pairs of vectors in X
   using the user supplied 2-arity function f. For example,
   Euclidean distance between the vectors could be computed
   as follows::

     dm = cdist(XA, XB, lambda u, v: np.sqrt(((u-v)**2).sum()))

   Note that you should avoid passing a reference to one of
   the distance functions defined in this library. For example,::

     dm = cdist(XA, XB, sokalsneath)

   would calculate the pair-wise distances between the vectors in
   X using the Python function `sokalsneath`. This would result in
   sokalsneath being called :math:`{n \choose 2}` times, which
   is inefficient. Instead, the optimized C version is more
   efficient, and we call it using the following syntax::

     dm = cdist(XA, XB, 'sokalsneath')

Examples
--------
Find the Euclidean distances between four 2-D coordinates:

>>> from scipy.spatial import distance
>>> coords = [(35.0456, -85.2672),
...           (35.1174, -89.9711),
...           (35.9728, -83.9422),
...           (36.1667, -86.7833)]
>>> distance.cdist(coords, coords, 'euclidean')
array([[ 0.    ,  4.7044,  1.6172,  1.8856],
       [ 4.7044,  0.    ,  6.0893,  3.3561],
       [ 1.6172,  6.0893,  0.    ,  2.8477],
       [ 1.8856,  3.3561,  2.8477,  0.    ]])


Find the Manhattan distance from a 3-D point to the corners of the unit
cube:

>>> a = np.array([[0, 0, 0],
...               [0, 0, 1],
...               [0, 1, 0],
...               [0, 1, 1],
...               [1, 0, 0],
...               [1, 0, 1],
...               [1, 1, 0],
...               [1, 1, 1]])
>>> b = np.array([[ 0.1,  0.2,  0.4]])
>>> distance.cdist(a, b, 'cityblock')
array([[ 0.7],
       [ 0.9],
       [ 1.3],
       [ 1.5],
       [ 1.5],
       [ 1.7],
       [ 2.1],
       [ 2.3]])
*)

val pdist : ?metric:[`Callable of Py.Object.t | `S of string] -> ?kwargs:(string * Py.Object.t) list -> x:[>`Ndarray] Np.Obj.t -> Py.Object.t list -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Pairwise distances between observations in n-dimensional space.

See Notes for common calling conventions.

Parameters
----------
X : ndarray
    An m by n array of m original observations in an
    n-dimensional space.
metric : str or function, optional
    The distance metric to use. The distance function can
    be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
    'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
    'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching',
    'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
    'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'.
*args : tuple. Deprecated.
    Additional arguments should be passed as keyword arguments
**kwargs : dict, optional
    Extra arguments to `metric`: refer to each metric documentation for a
    list of all possible arguments.

    Some possible arguments:

    p : scalar
    The p-norm to apply for Minkowski, weighted and unweighted.
    Default: 2.

    w : ndarray
    The weight vector for metrics that support weights (e.g., Minkowski).

    V : ndarray
    The variance vector for standardized Euclidean.
    Default: var(X, axis=0, ddof=1)

    VI : ndarray
    The inverse of the covariance matrix for Mahalanobis.
    Default: inv(cov(X.T)).T

    out : ndarray.
    The output array
    If not None, condensed distance matrix Y is stored in this array.
    Note: metric independent, it will become a regular keyword arg in a
    future scipy version

Returns
-------
Y : ndarray
    Returns a condensed distance matrix Y.  For
    each :math:`i` and :math:`j` (where :math:`i<j<m`),where m is the number
    of original observations. The metric ``dist(u=X[i], v=X[j])``
    is computed and stored in entry ``ij``.

See Also
--------
squareform : converts between condensed distance matrices and
             square distance matrices.

Notes
-----
See ``squareform`` for information on how to calculate the index of
this entry or to convert the condensed distance matrix to a
redundant square matrix.

The following are common calling conventions.

1. ``Y = pdist(X, 'euclidean')``

   Computes the distance between m points using Euclidean distance
   (2-norm) as the distance metric between the points. The points
   are arranged as m n-dimensional row vectors in the matrix X.

2. ``Y = pdist(X, 'minkowski', p=2.)``

   Computes the distances using the Minkowski distance
   :math:`||u-v||_p` (p-norm) where :math:`p \geq 1`.

3. ``Y = pdist(X, 'cityblock')``

   Computes the city block or Manhattan distance between the
   points.

4. ``Y = pdist(X, 'seuclidean', V=None)``

   Computes the standardized Euclidean distance. The standardized
   Euclidean distance between two n-vectors ``u`` and ``v`` is

   .. math::

      \sqrt{\sum {(u_i-v_i)^2 / V[x_i]}}


   V is the variance vector; V[i] is the variance computed over all
   the i'th components of the points.  If not passed, it is
   automatically computed.

5. ``Y = pdist(X, 'sqeuclidean')``

   Computes the squared Euclidean distance :math:`||u-v||_2^2` between
   the vectors.

6. ``Y = pdist(X, 'cosine')``

   Computes the cosine distance between vectors u and v,

   .. math::

      1 - \frac{u \cdot v}
               {{ ||u|| }_2 { ||v|| }_2}

   where :math:`||*||_2` is the 2-norm of its argument ``*``, and
   :math:`u \cdot v` is the dot product of ``u`` and ``v``.

7. ``Y = pdist(X, 'correlation')``

   Computes the correlation distance between vectors u and v. This is

   .. math::

      1 - \frac{(u - \bar{u}) \cdot (v - \bar{v})}
               {{ ||(u - \bar{u})|| }_2 { ||(v - \bar{v})|| }_2}

   where :math:`\bar{v}` is the mean of the elements of vector v,
   and :math:`x \cdot y` is the dot product of :math:`x` and :math:`y`.

8. ``Y = pdist(X, 'hamming')``

   Computes the normalized Hamming distance, or the proportion of
   those vector elements between two n-vectors ``u`` and ``v``
   which disagree. To save memory, the matrix ``X`` can be of type
   boolean.

9. ``Y = pdist(X, 'jaccard')``

   Computes the Jaccard distance between the points. Given two
   vectors, ``u`` and ``v``, the Jaccard distance is the
   proportion of those elements ``u[i]`` and ``v[i]`` that
   disagree.

10. ``Y = pdist(X, 'chebyshev')``

   Computes the Chebyshev distance between the points. The
   Chebyshev distance between two n-vectors ``u`` and ``v`` is the
   maximum norm-1 distance between their respective elements. More
   precisely, the distance is given by

   .. math::

      d(u,v) = \max_i { |u_i-v_i| }

11. ``Y = pdist(X, 'canberra')``

   Computes the Canberra distance between the points. The
   Canberra distance between two points ``u`` and ``v`` is

   .. math::

     d(u,v) = \sum_i \frac{ |u_i-v_i| }
                          { |u_i|+|v_i| }


12. ``Y = pdist(X, 'braycurtis')``

   Computes the Bray-Curtis distance between the points. The
   Bray-Curtis distance between two points ``u`` and ``v`` is


   .. math::

        d(u,v) = \frac{\sum_i { |u_i-v_i| }}
                       {\sum_i { |u_i+v_i| }}

13. ``Y = pdist(X, 'mahalanobis', VI=None)``

   Computes the Mahalanobis distance between the points. The
   Mahalanobis distance between two points ``u`` and ``v`` is
   :math:`\sqrt{(u-v)(1/V)(u-v)^T}` where :math:`(1/V)` (the ``VI``
   variable) is the inverse covariance. If ``VI`` is not None,
   ``VI`` will be used as the inverse covariance matrix.

14. ``Y = pdist(X, 'yule')``

   Computes the Yule distance between each pair of boolean
   vectors. (see yule function documentation)

15. ``Y = pdist(X, 'matching')``

   Synonym for 'hamming'.

16. ``Y = pdist(X, 'dice')``

   Computes the Dice distance between each pair of boolean
   vectors. (see dice function documentation)

17. ``Y = pdist(X, 'kulsinski')``

   Computes the Kulsinski distance between each pair of
   boolean vectors. (see kulsinski function documentation)

18. ``Y = pdist(X, 'rogerstanimoto')``

   Computes the Rogers-Tanimoto distance between each pair of
   boolean vectors. (see rogerstanimoto function documentation)

19. ``Y = pdist(X, 'russellrao')``

   Computes the Russell-Rao distance between each pair of
   boolean vectors. (see russellrao function documentation)

20. ``Y = pdist(X, 'sokalmichener')``

   Computes the Sokal-Michener distance between each pair of
   boolean vectors. (see sokalmichener function documentation)

21. ``Y = pdist(X, 'sokalsneath')``

   Computes the Sokal-Sneath distance between each pair of
   boolean vectors. (see sokalsneath function documentation)

22. ``Y = pdist(X, 'wminkowski', p=2, w=w)``

   Computes the weighted Minkowski distance between each pair of
   vectors. (see wminkowski function documentation)

23. ``Y = pdist(X, f)``

   Computes the distance between all pairs of vectors in X
   using the user supplied 2-arity function f. For example,
   Euclidean distance between the vectors could be computed
   as follows::

     dm = pdist(X, lambda u, v: np.sqrt(((u-v)**2).sum()))

   Note that you should avoid passing a reference to one of
   the distance functions defined in this library. For example,::

     dm = pdist(X, sokalsneath)

   would calculate the pair-wise distances between the vectors in
   X using the Python function sokalsneath. This would result in
   sokalsneath being called :math:`{n \choose 2}` times, which
   is inefficient. Instead, the optimized C version is more
   efficient, and we call it using the following syntax.::

     dm = pdist(X, 'sokalsneath')
*)

val squareform : ?force:string -> ?checks:bool -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Convert a vector-form distance vector to a square-form distance
matrix, and vice-versa.

Parameters
----------
X : ndarray
    Either a condensed or redundant distance matrix.
force : str, optional
    As with MATLAB(TM), if force is equal to ``'tovector'`` or
    ``'tomatrix'``, the input will be treated as a distance matrix or
    distance vector respectively.
checks : bool, optional
    If set to False, no checks will be made for matrix
    symmetry nor zero diagonals. This is useful if it is known that
    ``X - X.T1`` is small and ``diag(X)`` is close to zero.
    These values are ignored any way so they do not disrupt the
    squareform transformation.

Returns
-------
Y : ndarray
    If a condensed distance matrix is passed, a redundant one is
    returned, or if a redundant one is passed, a condensed distance
    matrix is returned.

Notes
-----
1. v = squareform(X)

   Given a square d-by-d symmetric distance matrix X,
   ``v = squareform(X)`` returns a ``d * (d-1) / 2`` (or
   :math:`{n \choose 2}`) sized vector v.

  :math:`v[{n \choose 2}-{n-i \choose 2} + (j-i-1)]` is the distance
  between points i and j. If X is non-square or asymmetric, an error
  is returned.

2. X = squareform(v)

  Given a ``d*(d-1)/2`` sized v for some integer ``d >= 2`` encoding
  distances as described, ``X = squareform(v)`` returns a d by d distance
  matrix X.  The ``X[i, j]`` and ``X[j, i]`` values are set to
  :math:`v[{n \choose 2}-{n-i \choose 2} + (j-i-1)]` and all
  diagonal elements are zero.

In SciPy 0.19.0, ``squareform`` stopped casting all input types to
float64, and started returning arrays of the same dtype as the input.
*)

val xlogy : ?out:Py.Object.t -> ?where:Py.Object.t -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
xlogy(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

xlogy(x, y)

Compute ``x*log(y)`` so that the result is 0 if ``x = 0``.

Parameters
----------
x : array_like
    Multiplier
y : array_like
    Argument

Returns
-------
z : array_like
    Computed x*log(y)

Notes
-----

.. versionadded:: 0.13.0
*)


end

val approximate_taylor_polynomial : ?order:int -> f:Py.Object.t -> x:[`F of float | `I of int | `Bool of bool | `S of string] -> degree:int -> scale:float -> unit -> Py.Object.t
(**
Estimate the Taylor polynomial of f at x by polynomial fitting.

Parameters
----------
f : callable
    The function whose Taylor polynomial is sought. Should accept
    a vector of `x` values.
x : scalar
    The point at which the polynomial is to be evaluated.
degree : int
    The degree of the Taylor polynomial
scale : scalar
    The width of the interval to use to evaluate the Taylor polynomial.
    Function values spread over a range this wide are used to fit the
    polynomial. Must be chosen carefully.
order : int or None, optional
    The order of the polynomial to be used in the fitting; `f` will be
    evaluated ``order+1`` times. If None, use `degree`.

Returns
-------
p : poly1d instance
    The Taylor polynomial (translated to the origin, so that
    for example p(0)=f(x)).

Notes
-----
The appropriate choice of 'scale' is a trade-off; too large and the
function differs from its Taylor polynomial too much to get a good
answer, too small and round-off errors overwhelm the higher-order terms.
The algorithm used becomes numerically unstable around order 30 even
under ideal circumstances.

Choosing order somewhat larger than degree may improve the higher-order
terms.
*)

val barycentric_interpolate : ?axis:int -> xi:[>`Ndarray] Np.Obj.t -> yi:[>`Ndarray] Np.Obj.t -> x:[`S of string | `F of float | `Ndarray of [>`Ndarray] Np.Obj.t | `I of int | `Bool of bool] -> unit -> Py.Object.t
(**
Convenience function for polynomial interpolation.

Constructs a polynomial that passes through a given set of points,
then evaluates the polynomial. For reasons of numerical stability,
this function does not compute the coefficients of the polynomial.

This function uses a 'barycentric interpolation' method that treats
the problem as a special case of rational function interpolation.
This algorithm is quite stable, numerically, but even in a world of
exact computation, unless the `x` coordinates are chosen very
carefully - Chebyshev zeros (e.g. cos(i*pi/n)) are a good choice -
polynomial interpolation itself is a very ill-conditioned process
due to the Runge phenomenon.

Parameters
----------
xi : array_like
    1-d array of x coordinates of the points the polynomial should
    pass through
yi : array_like
    The y coordinates of the points the polynomial should pass through.
x : scalar or array_like
    Points to evaluate the interpolator at.
axis : int, optional
    Axis in the yi array corresponding to the x-coordinate values.

Returns
-------
y : scalar or array_like
    Interpolated values. Shape is determined by replacing
    the interpolation axis in the original array with the shape of x.

See Also
--------
BarycentricInterpolator

Notes
-----
Construction of the interpolation weights is a relatively slow process.
If you want to call this many times with the same xi (but possibly
varying yi or x) you should use the class `BarycentricInterpolator`.
This is what this function uses internally.
*)

val bisplev : ?dx:Py.Object.t -> ?dy:Py.Object.t -> x:Py.Object.t -> y:Py.Object.t -> tck:Py.Object.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Evaluate a bivariate B-spline and its derivatives.

Return a rank-2 array of spline function values (or spline derivative
values) at points given by the cross-product of the rank-1 arrays `x` and
`y`.  In special cases, return an array or just a float if either `x` or
`y` or both are floats.  Based on BISPEV from FITPACK.

Parameters
----------
x, y : ndarray
    Rank-1 arrays specifying the domain over which to evaluate the
    spline or its derivative.
tck : tuple
    A sequence of length 5 returned by `bisplrep` containing the knot
    locations, the coefficients, and the degree of the spline:
    [tx, ty, c, kx, ky].
dx, dy : int, optional
    The orders of the partial derivatives in `x` and `y` respectively.

Returns
-------
vals : ndarray
    The B-spline or its derivative evaluated over the set formed by
    the cross-product of `x` and `y`.

See Also
--------
splprep, splrep, splint, sproot, splev
UnivariateSpline, BivariateSpline

Notes
-----
    See `bisplrep` to generate the `tck` representation.

References
----------
.. [1] Dierckx P. : An algorithm for surface fitting
   with spline functions
   Ima J. Numer. Anal. 1 (1981) 267-283.
.. [2] Dierckx P. : An algorithm for surface fitting
   with spline functions
   report tw50, Dept. Computer Science,K.U.Leuven, 1980.
.. [3] Dierckx P. : Curve and surface fitting with splines,
   Monographs on Numerical Analysis, Oxford University Press, 1993.
*)

val bisplrep : ?w:[>`Ndarray] Np.Obj.t -> ?xb:Py.Object.t -> ?xe:Py.Object.t -> ?yb:Py.Object.t -> ?ye:Py.Object.t -> ?kx:Py.Object.t -> ?ky:Py.Object.t -> ?task:int -> ?s:float -> ?eps:float -> ?tx:Py.Object.t -> ?ty:Py.Object.t -> ?full_output:int -> ?nxest:Py.Object.t -> ?nyest:Py.Object.t -> ?quiet:int -> x:Py.Object.t -> y:Py.Object.t -> z:Py.Object.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t * int * string)
(**
Find a bivariate B-spline representation of a surface.

Given a set of data points (x[i], y[i], z[i]) representing a surface
z=f(x,y), compute a B-spline representation of the surface. Based on
the routine SURFIT from FITPACK.

Parameters
----------
x, y, z : ndarray
    Rank-1 arrays of data points.
w : ndarray, optional
    Rank-1 array of weights. By default ``w=np.ones(len(x))``.
xb, xe : float, optional
    End points of approximation interval in `x`.
    By default ``xb = x.min(), xe=x.max()``.
yb, ye : float, optional
    End points of approximation interval in `y`.
    By default ``yb=y.min(), ye = y.max()``.
kx, ky : int, optional
    The degrees of the spline (1 <= kx, ky <= 5).
    Third order (kx=ky=3) is recommended.
task : int, optional
    If task=0, find knots in x and y and coefficients for a given
    smoothing factor, s.
    If task=1, find knots and coefficients for another value of the
    smoothing factor, s.  bisplrep must have been previously called
    with task=0 or task=1.
    If task=-1, find coefficients for a given set of knots tx, ty.
s : float, optional
    A non-negative smoothing factor.  If weights correspond
    to the inverse of the standard-deviation of the errors in z,
    then a good s-value should be found in the range
    ``(m-sqrt(2*m),m+sqrt(2*m))`` where m=len(x).
eps : float, optional
    A threshold for determining the effective rank of an
    over-determined linear system of equations (0 < eps < 1).
    `eps` is not likely to need changing.
tx, ty : ndarray, optional
    Rank-1 arrays of the knots of the spline for task=-1
full_output : int, optional
    Non-zero to return optional outputs.
nxest, nyest : int, optional
    Over-estimates of the total number of knots. If None then
    ``nxest = max(kx+sqrt(m/2),2*kx+3)``,
    ``nyest = max(ky+sqrt(m/2),2*ky+3)``.
quiet : int, optional
    Non-zero to suppress printing of messages.
    This parameter is deprecated; use standard Python warning filters
    instead.

Returns
-------
tck : array_like
    A list [tx, ty, c, kx, ky] containing the knots (tx, ty) and
    coefficients (c) of the bivariate B-spline representation of the
    surface along with the degree of the spline.
fp : ndarray
    The weighted sum of squared residuals of the spline approximation.
ier : int
    An integer flag about splrep success.  Success is indicated if
    ier<=0. If ier in [1,2,3] an error occurred but was not raised.
    Otherwise an error is raised.
msg : str
    A message corresponding to the integer flag, ier.

See Also
--------
splprep, splrep, splint, sproot, splev
UnivariateSpline, BivariateSpline

Notes
-----
See `bisplev` to evaluate the value of the B-spline given its tck
representation.

References
----------
.. [1] Dierckx P.:An algorithm for surface fitting with spline functions
   Ima J. Numer. Anal. 1 (1981) 267-283.
.. [2] Dierckx P.:An algorithm for surface fitting with spline functions
   report tw50, Dept. Computer Science,K.U.Leuven, 1980.
.. [3] Dierckx P.:Curve and surface fitting with splines, Monographs on
   Numerical Analysis, Oxford University Press, 1993.
*)

val griddata : ?method_:[`Linear | `Nearest | `Cubic] -> ?fill_value:float -> ?rescale:bool -> points:Py.Object.t -> values:Py.Object.t -> xi:Py.Object.t -> unit -> Py.Object.t
(**
Interpolate unstructured D-dimensional data.

Parameters
----------
points : 2-D ndarray of floats with shape (n, D), or length D tuple of 1-D ndarrays with shape (n,).
    Data point coordinates. 
values : ndarray of float or complex, shape (n,)
    Data values.
xi : 2-D ndarray of floats with shape (m, D), or length D tuple of ndarrays broadcastable to the same shape.
    Points at which to interpolate data.
method : {'linear', 'nearest', 'cubic'}, optional
    Method of interpolation. One of

    ``nearest``
      return the value at the data point closest to
      the point of interpolation.  See `NearestNDInterpolator` for
      more details.

    ``linear``
      tessellate the input point set to n-dimensional
      simplices, and interpolate linearly on each simplex.  See
      `LinearNDInterpolator` for more details.

    ``cubic`` (1-D)
      return the value determined from a cubic
      spline.

    ``cubic`` (2-D)
      return the value determined from a
      piecewise cubic, continuously differentiable (C1), and
      approximately curvature-minimizing polynomial surface. See
      `CloughTocher2DInterpolator` for more details.
fill_value : float, optional
    Value used to fill in for requested points outside of the
    convex hull of the input points.  If not provided, then the
    default is ``nan``. This option has no effect for the
    'nearest' method.
rescale : bool, optional
    Rescale points to unit cube before performing interpolation.
    This is useful if some of the input dimensions have
    incommensurable units and differ by many orders of magnitude.

    .. versionadded:: 0.14.0
    
Returns
-------
ndarray
    Array of interpolated values.

Notes
-----

.. versionadded:: 0.9

Examples
--------

Suppose we want to interpolate the 2-D function

>>> def func(x, y):
...     return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2

on a grid in [0, 1]x[0, 1]

>>> grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

but we only know its values at 1000 data points:

>>> points = np.random.rand(1000, 2)
>>> values = func(points[:,0], points[:,1])

This can be done with `griddata` -- below we try out all of the
interpolation methods:

>>> from scipy.interpolate import griddata
>>> grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
>>> grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
>>> grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')

One can see that the exact result is reproduced by all of the
methods to some degree, but for this smooth function the piecewise
cubic interpolant gives the best results:

>>> import matplotlib.pyplot as plt
>>> plt.subplot(221)
>>> plt.imshow(func(grid_x, grid_y).T, extent=(0,1,0,1), origin='lower')
>>> plt.plot(points[:,0], points[:,1], 'k.', ms=1)
>>> plt.title('Original')
>>> plt.subplot(222)
>>> plt.imshow(grid_z0.T, extent=(0,1,0,1), origin='lower')
>>> plt.title('Nearest')
>>> plt.subplot(223)
>>> plt.imshow(grid_z1.T, extent=(0,1,0,1), origin='lower')
>>> plt.title('Linear')
>>> plt.subplot(224)
>>> plt.imshow(grid_z2.T, extent=(0,1,0,1), origin='lower')
>>> plt.title('Cubic')
>>> plt.gcf().set_size_inches(6, 6)
>>> plt.show()
*)

val insert : ?m:int -> ?per:int -> x:Py.Object.t -> tck:Py.Object.t -> unit -> Py.Object.t
(**
Insert knots into a B-spline.

Given the knots and coefficients of a B-spline representation, create a
new B-spline with a knot inserted `m` times at point `x`.
This is a wrapper around the FORTRAN routine insert of FITPACK.

Parameters
----------
x (u) : array_like
    A 1-D point at which to insert a new knot(s).  If `tck` was returned
    from ``splprep``, then the parameter values, u should be given.
tck : a `BSpline` instance or a tuple
    If tuple, then it is expected to be a tuple (t,c,k) containing
    the vector of knots, the B-spline coefficients, and the degree of
    the spline.
m : int, optional
    The number of times to insert the given knot (its multiplicity).
    Default is 1.
per : int, optional
    If non-zero, the input spline is considered periodic.

Returns
-------
BSpline instance or a tuple
    A new B-spline with knots t, coefficients c, and degree k.
    ``t(k+1) <= x <= t(n-k)``, where k is the degree of the spline.
    In case of a periodic spline (``per != 0``) there must be
    either at least k interior knots t(j) satisfying ``t(k+1)<t(j)<=x``
    or at least k interior knots t(j) satisfying ``x<=t(j)<t(n-k)``.
    A tuple is returned iff the input argument `tck` is a tuple, otherwise
    a BSpline object is constructed and returned.

Notes
-----
Based on algorithms from [1]_ and [2]_.

Manipulating the tck-tuples directly is not recommended. In new code,
prefer using the `BSpline` objects.

References
----------
.. [1] W. Boehm, 'Inserting new knots into b-spline curves.',
    Computer Aided Design, 12, p.199-201, 1980.
.. [2] P. Dierckx, 'Curve and surface fitting with splines, Monographs on
    Numerical Analysis', Oxford University Press, 1993.
*)

val interpn : ?method_:string -> ?bounds_error:bool -> ?fill_value:[`F of float | `I of int] -> points:Py.Object.t -> values:[>`Ndarray] Np.Obj.t -> xi:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Multidimensional interpolation on regular grids.

Parameters
----------
points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
    The points defining the regular grid in n dimensions.

values : array_like, shape (m1, ..., mn, ...)
    The data on the regular grid in n dimensions.

xi : ndarray of shape (..., ndim)
    The coordinates to sample the gridded data at

method : str, optional
    The method of interpolation to perform. Supported are 'linear' and
    'nearest', and 'splinef2d'. 'splinef2d' is only supported for
    2-dimensional data.

bounds_error : bool, optional
    If True, when interpolated values are requested outside of the
    domain of the input data, a ValueError is raised.
    If False, then `fill_value` is used.

fill_value : number, optional
    If provided, the value to use for points outside of the
    interpolation domain. If None, values outside
    the domain are extrapolated.  Extrapolation is not supported by method
    'splinef2d'.

Returns
-------
values_x : ndarray, shape xi.shape[:-1] + values.shape[ndim:]
    Interpolated values at input coordinates.

Notes
-----

.. versionadded:: 0.14

See also
--------
NearestNDInterpolator : Nearest neighbour interpolation on unstructured
                        data in N dimensions

LinearNDInterpolator : Piecewise linear interpolant on unstructured data
                       in N dimensions

RegularGridInterpolator : Linear and nearest-neighbor Interpolation on a
                          regular grid in arbitrary dimensions

RectBivariateSpline : Bivariate spline approximation over a rectangular mesh
*)

val krogh_interpolate : ?der:[`Ndarray of [>`Ndarray] Np.Obj.t | `I of int] -> ?axis:int -> xi:[>`Ndarray] Np.Obj.t -> yi:[>`Ndarray] Np.Obj.t -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Convenience function for polynomial interpolation.

See `KroghInterpolator` for more details.

Parameters
----------
xi : array_like
    Known x-coordinates.
yi : array_like
    Known y-coordinates, of shape ``(xi.size, R)``.  Interpreted as
    vectors of length R, or scalars if R=1.
x : array_like
    Point or points at which to evaluate the derivatives.
der : int or list, optional
    How many derivatives to extract; None for all potentially
    nonzero derivatives (that is a number equal to the number
    of points), or a list of derivatives to extract. This number
    includes the function value as 0th derivative.
axis : int, optional
    Axis in the yi array corresponding to the x-coordinate values.

Returns
-------
d : ndarray
    If the interpolator's values are R-dimensional then the
    returned array will be the number of derivatives by N by R.
    If `x` is a scalar, the middle dimension will be dropped; if
    the `yi` are scalars then the last dimension will be dropped.

See Also
--------
KroghInterpolator

Notes
-----
Construction of the interpolating polynomial is a relatively expensive
process. If you want to evaluate it repeatedly consider using the class
KroghInterpolator (which is what this function uses).
*)

val lagrange : x:[>`Ndarray] Np.Obj.t -> w:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Return a Lagrange interpolating polynomial.

Given two 1-D arrays `x` and `w,` returns the Lagrange interpolating
polynomial through the points ``(x, w)``.

Warning: This implementation is numerically unstable. Do not expect to
be able to use more than about 20 points even if they are chosen optimally.

Parameters
----------
x : array_like
    `x` represents the x-coordinates of a set of datapoints.
w : array_like
    `w` represents the y-coordinates of a set of datapoints, i.e. f(`x`).

Returns
-------
lagrange : `numpy.poly1d` instance
    The Lagrange interpolating polynomial.

Examples
--------
Interpolate :math:`f(x) = x^3` by 3 points.

>>> from scipy.interpolate import lagrange
>>> x = np.array([0, 1, 2])
>>> y = x**3
>>> poly = lagrange(x, y)

Since there are only 3 points, Lagrange polynomial has degree 2. Explicitly,
it is given by

.. math::

    \begin{aligned}
        L(x) &= 1\times \frac{x (x - 2)}{-1} + 8\times \frac{x (x-1)}{2} \\
             &= x (-2 + 3x)
    \end{aligned}

>>> from numpy.polynomial.polynomial import Polynomial
>>> Polynomial(poly).coef
array([ 3., -2.,  0.])
*)

val make_interp_spline : ?k:int -> ?t:[>`Ndarray] Np.Obj.t -> ?bc_type:Py.Object.t -> ?axis:int -> ?check_finite:bool -> x:[>`Ndarray] Np.Obj.t -> y:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Compute the (coefficients of) interpolating B-spline.

Parameters
----------
x : array_like, shape (n,)
    Abscissas.
y : array_like, shape (n, ...)
    Ordinates.
k : int, optional
    B-spline degree. Default is cubic, k=3.
t : array_like, shape (nt + k + 1,), optional.
    Knots.
    The number of knots needs to agree with the number of datapoints and
    the number of derivatives at the edges. Specifically, ``nt - n`` must
    equal ``len(deriv_l) + len(deriv_r)``.
bc_type : 2-tuple or None
    Boundary conditions.
    Default is None, which means choosing the boundary conditions
    automatically. Otherwise, it must be a length-two tuple where the first
    element sets the boundary conditions at ``x[0]`` and the second
    element sets the boundary conditions at ``x[-1]``. Each of these must
    be an iterable of pairs ``(order, value)`` which gives the values of
    derivatives of specified orders at the given edge of the interpolation
    interval.
    Alternatively, the following string aliases are recognized:

    * ``'clamped'``: The first derivatives at the ends are zero. This is
       equivalent to ``bc_type=([(1, 0.0)], [(1, 0.0)])``.
    * ``'natural'``: The second derivatives at ends are zero. This is
      equivalent to ``bc_type=([(2, 0.0)], [(2, 0.0)])``.
    * ``'not-a-knot'`` (default): The first and second segments are the same
      polynomial. This is equivalent to having ``bc_type=None``.

axis : int, optional
    Interpolation axis. Default is 0.
check_finite : bool, optional
    Whether to check that the input arrays contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.
    Default is True.

Returns
-------
b : a BSpline object of the degree ``k`` and with knots ``t``.

Examples
--------

Use cubic interpolation on Chebyshev nodes:

>>> def cheb_nodes(N):
...     jj = 2.*np.arange(N) + 1
...     x = np.cos(np.pi * jj / 2 / N)[::-1]
...     return x

>>> x = cheb_nodes(20)
>>> y = np.sqrt(1 - x**2)

>>> from scipy.interpolate import BSpline, make_interp_spline
>>> b = make_interp_spline(x, y)
>>> np.allclose(b(x), y)
True

Note that the default is a cubic spline with a not-a-knot boundary condition

>>> b.k
3

Here we use a 'natural' spline, with zero 2nd derivatives at edges:

>>> l, r = [(2, 0.0)], [(2, 0.0)]
>>> b_n = make_interp_spline(x, y, bc_type=(l, r))  # or, bc_type='natural'
>>> np.allclose(b_n(x), y)
True
>>> x0, x1 = x[0], x[-1]
>>> np.allclose([b_n(x0, 2), b_n(x1, 2)], [0, 0])
True

Interpolation of parametric curves is also supported. As an example, we
compute a discretization of a snail curve in polar coordinates

>>> phi = np.linspace(0, 2.*np.pi, 40)
>>> r = 0.3 + np.cos(phi)
>>> x, y = r*np.cos(phi), r*np.sin(phi)  # convert to Cartesian coordinates

Build an interpolating curve, parameterizing it by the angle

>>> from scipy.interpolate import make_interp_spline
>>> spl = make_interp_spline(phi, np.c_[x, y])

Evaluate the interpolant on a finer grid (note that we transpose the result
to unpack it into a pair of x- and y-arrays)

>>> phi_new = np.linspace(0, 2.*np.pi, 100)
>>> x_new, y_new = spl(phi_new).T

Plot the result

>>> import matplotlib.pyplot as plt
>>> plt.plot(x, y, 'o')
>>> plt.plot(x_new, y_new, '-')
>>> plt.show()

See Also
--------
BSpline : base class representing the B-spline objects
CubicSpline : a cubic spline in the polynomial basis
make_lsq_spline : a similar factory function for spline fitting
UnivariateSpline : a wrapper over FITPACK spline fitting routines
splrep : a wrapper over FITPACK spline fitting routines
*)

val make_lsq_spline : ?k:int -> ?w:[>`Ndarray] Np.Obj.t -> ?axis:int -> ?check_finite:bool -> x:[>`Ndarray] Np.Obj.t -> y:[>`Ndarray] Np.Obj.t -> t:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Compute the (coefficients of) an LSQ B-spline.

The result is a linear combination

.. math::

        S(x) = \sum_j c_j B_j(x; t)

of the B-spline basis elements, :math:`B_j(x; t)`, which minimizes

.. math::

    \sum_{j} \left( w_j \times (S(x_j) - y_j) \right)^2

Parameters
----------
x : array_like, shape (m,)
    Abscissas.
y : array_like, shape (m, ...)
    Ordinates.
t : array_like, shape (n + k + 1,).
    Knots.
    Knots and data points must satisfy Schoenberg-Whitney conditions.
k : int, optional
    B-spline degree. Default is cubic, k=3.
w : array_like, shape (n,), optional
    Weights for spline fitting. Must be positive. If ``None``,
    then weights are all equal.
    Default is ``None``.
axis : int, optional
    Interpolation axis. Default is zero.
check_finite : bool, optional
    Whether to check that the input arrays contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.
    Default is True.

Returns
-------
b : a BSpline object of the degree `k` with knots `t`.

Notes
-----

The number of data points must be larger than the spline degree `k`.

Knots `t` must satisfy the Schoenberg-Whitney conditions,
i.e., there must be a subset of data points ``x[j]`` such that
``t[j] < x[j] < t[j+k+1]``, for ``j=0, 1,...,n-k-2``.

Examples
--------
Generate some noisy data:

>>> x = np.linspace(-3, 3, 50)
>>> y = np.exp(-x**2) + 0.1 * np.random.randn(50)

Now fit a smoothing cubic spline with a pre-defined internal knots.
Here we make the knot vector (k+1)-regular by adding boundary knots:

>>> from scipy.interpolate import make_lsq_spline, BSpline
>>> t = [-1, 0, 1]
>>> k = 3
>>> t = np.r_[(x[0],)*(k+1),
...           t,
...           (x[-1],)*(k+1)]
>>> spl = make_lsq_spline(x, y, t, k)

For comparison, we also construct an interpolating spline for the same
set of data:

>>> from scipy.interpolate import make_interp_spline
>>> spl_i = make_interp_spline(x, y)

Plot both:

>>> import matplotlib.pyplot as plt
>>> xs = np.linspace(-3, 3, 100)
>>> plt.plot(x, y, 'ro', ms=5)
>>> plt.plot(xs, spl(xs), 'g-', lw=3, label='LSQ spline')
>>> plt.plot(xs, spl_i(xs), 'b-', lw=3, alpha=0.7, label='interp spline')
>>> plt.legend(loc='best')
>>> plt.show()

**NaN handling**: If the input arrays contain ``nan`` values, the result is
not useful since the underlying spline fitting routines cannot deal with
``nan``. A workaround is to use zero weights for not-a-number data points:

>>> y[8] = np.nan
>>> w = np.isnan(y)
>>> y[w] = 0.
>>> tck = make_lsq_spline(x, y, t, w=~w)

Notice the need to replace a ``nan`` by a numerical value (precise value
does not matter as long as the corresponding weight is zero.)

See Also
--------
BSpline : base class representing the B-spline objects
make_interp_spline : a similar factory function for interpolating splines
LSQUnivariateSpline : a FITPACK-based spline fitting routine
splrep : a FITPACK-based fitting routine
*)

val pade : ?n:int -> an:[>`Ndarray] Np.Obj.t -> m:int -> unit -> Py.Object.t
(**
Return Pade approximation to a polynomial as the ratio of two polynomials.

Parameters
----------
an : (N,) array_like
    Taylor series coefficients.
m : int
    The order of the returned approximating polynomial `q`.
n : int, optional
    The order of the returned approximating polynomial `p`. By default, 
    the order is ``len(an)-m``.

Returns
-------
p, q : Polynomial class
    The Pade approximation of the polynomial defined by `an` is
    ``p(x)/q(x)``.

Examples
--------
>>> from scipy.interpolate import pade
>>> e_exp = [1.0, 1.0, 1.0/2.0, 1.0/6.0, 1.0/24.0, 1.0/120.0]
>>> p, q = pade(e_exp, 2)

>>> e_exp.reverse()
>>> e_poly = np.poly1d(e_exp)

Compare ``e_poly(x)`` and the Pade approximation ``p(x)/q(x)``

>>> e_poly(1)
2.7166666666666668

>>> p(1)/q(1)
2.7179487179487181
*)

val pchip_interpolate : ?der:[`Ndarray of [>`Ndarray] Np.Obj.t | `I of int] -> ?axis:int -> xi:[>`Ndarray] Np.Obj.t -> yi:[>`Ndarray] Np.Obj.t -> x:[`S of string | `F of float | `Ndarray of [>`Ndarray] Np.Obj.t | `I of int | `Bool of bool] -> unit -> Py.Object.t
(**
Convenience function for pchip interpolation.

xi and yi are arrays of values used to approximate some function f,
with ``yi = f(xi)``.  The interpolant uses monotonic cubic splines
to find the value of new points x and the derivatives there.

See `scipy.interpolate.PchipInterpolator` for details.

Parameters
----------
xi : array_like
    A sorted list of x-coordinates, of length N.
yi :  array_like
    A 1-D array of real values.  `yi`'s length along the interpolation
    axis must be equal to the length of `xi`. If N-D array, use axis
    parameter to select correct axis.
x : scalar or array_like
    Of length M.
der : int or list, optional
    Derivatives to extract.  The 0-th derivative can be included to
    return the function value.
axis : int, optional
    Axis in the yi array corresponding to the x-coordinate values.

See Also
--------
PchipInterpolator

Returns
-------
y : scalar or array_like
    The result, of length R or length M or M by R,
*)

val spalde : x:[>`Ndarray] Np.Obj.t -> tck:Py.Object.t -> unit -> Py.Object.t
(**
Evaluate all derivatives of a B-spline.

Given the knots and coefficients of a cubic B-spline compute all
derivatives up to order k at a point (or set of points).

Parameters
----------
x : array_like
    A point or a set of points at which to evaluate the derivatives.
    Note that ``t(k) <= x <= t(n-k+1)`` must hold for each `x`.
tck : tuple
    A tuple ``(t, c, k)``, containing the vector of knots, the B-spline
    coefficients, and the degree of the spline (see `splev`).

Returns
-------
results : {ndarray, list of ndarrays}
    An array (or a list of arrays) containing all derivatives
    up to order k inclusive for each point `x`.

See Also
--------
splprep, splrep, splint, sproot, splev, bisplrep, bisplev,
BSpline

References
----------
.. [1] C. de Boor: On calculating with b-splines, J. Approximation Theory
   6 (1972) 50-62.
.. [2] M. G. Cox : The numerical evaluation of b-splines, J. Inst. Maths
   applics 10 (1972) 134-149.
.. [3] P. Dierckx : Curve and surface fitting with splines, Monographs on
   Numerical Analysis, Oxford University Press, 1993.
*)

val splantider : ?n:int -> tck:Py.Object.t -> unit -> Py.Object.t
(**
Compute the spline for the antiderivative (integral) of a given spline.

Parameters
----------
tck : BSpline instance or a tuple of (t, c, k)
    Spline whose antiderivative to compute
n : int, optional
    Order of antiderivative to evaluate. Default: 1

Returns
-------
BSpline instance or a tuple of (t2, c2, k2)
    Spline of order k2=k+n representing the antiderivative of the input
    spline.
    A tuple is returned iff the input argument `tck` is a tuple, otherwise
    a BSpline object is constructed and returned.

See Also
--------
splder, splev, spalde
BSpline

Notes
-----
The `splder` function is the inverse operation of this function.
Namely, ``splder(splantider(tck))`` is identical to `tck`, modulo
rounding error.

.. versionadded:: 0.13.0

Examples
--------
>>> from scipy.interpolate import splrep, splder, splantider, splev
>>> x = np.linspace(0, np.pi/2, 70)
>>> y = 1 / np.sqrt(1 - 0.8*np.sin(x)**2)
>>> spl = splrep(x, y)

The derivative is the inverse operation of the antiderivative,
although some floating point error accumulates:

>>> splev(1.7, spl), splev(1.7, splder(splantider(spl)))
(array(2.1565429877197317), array(2.1565429877201865))

Antiderivative can be used to evaluate definite integrals:

>>> ispl = splantider(spl)
>>> splev(np.pi/2, ispl) - splev(0, ispl)
2.2572053588768486

This is indeed an approximation to the complete elliptic integral
:math:`K(m) = \int_0^{\pi/2} [1 - m\sin^2 x]^{-1/2} dx`:

>>> from scipy.special import ellipk
>>> ellipk(0.8)
2.2572053268208538
*)

val splder : ?n:int -> tck:Py.Object.t -> unit -> Py.Object.t
(**
Compute the spline representation of the derivative of a given spline

Parameters
----------
tck : BSpline instance or a tuple of (t, c, k)
    Spline whose derivative to compute
n : int, optional
    Order of derivative to evaluate. Default: 1

Returns
-------
`BSpline` instance or tuple
    Spline of order k2=k-n representing the derivative
    of the input spline.
    A tuple is returned iff the input argument `tck` is a tuple, otherwise
    a BSpline object is constructed and returned.

Notes
-----

.. versionadded:: 0.13.0

See Also
--------
splantider, splev, spalde
BSpline

Examples
--------
This can be used for finding maxima of a curve:

>>> from scipy.interpolate import splrep, splder, sproot
>>> x = np.linspace(0, 10, 70)
>>> y = np.sin(x)
>>> spl = splrep(x, y, k=4)

Now, differentiate the spline and find the zeros of the
derivative. (NB: `sproot` only works for order 3 splines, so we
fit an order 4 spline):

>>> dspl = splder(spl)
>>> sproot(dspl) / np.pi
array([ 0.50000001,  1.5       ,  2.49999998])

This agrees well with roots :math:`\pi/2 + n\pi` of
:math:`\cos(x) = \sin'(x)`.
*)

val splev : ?der:int -> ?ext:int -> x:[>`Ndarray] Np.Obj.t -> tck:Py.Object.t -> unit -> Py.Object.t
(**
Evaluate a B-spline or its derivatives.

Given the knots and coefficients of a B-spline representation, evaluate
the value of the smoothing polynomial and its derivatives.  This is a
wrapper around the FORTRAN routines splev and splder of FITPACK.

Parameters
----------
x : array_like
    An array of points at which to return the value of the smoothed
    spline or its derivatives.  If `tck` was returned from `splprep`,
    then the parameter values, u should be given.
tck : 3-tuple or a BSpline object
    If a tuple, then it should be a sequence of length 3 returned by
    `splrep` or `splprep` containing the knots, coefficients, and degree
    of the spline. (Also see Notes.)
der : int, optional
    The order of derivative of the spline to compute (must be less than
    or equal to k, the degree of the spline).
ext : int, optional
    Controls the value returned for elements of ``x`` not in the
    interval defined by the knot sequence.

    * if ext=0, return the extrapolated value.
    * if ext=1, return 0
    * if ext=2, raise a ValueError
    * if ext=3, return the boundary value.

    The default value is 0.

Returns
-------
y : ndarray or list of ndarrays
    An array of values representing the spline function evaluated at
    the points in `x`.  If `tck` was returned from `splprep`, then this
    is a list of arrays representing the curve in N-dimensional space.

Notes
-----
Manipulating the tck-tuples directly is not recommended. In new code,
prefer using `BSpline` objects.

See Also
--------
splprep, splrep, sproot, spalde, splint
bisplrep, bisplev
BSpline

References
----------
.. [1] C. de Boor, 'On calculating with b-splines', J. Approximation
    Theory, 6, p.50-62, 1972.
.. [2] M. G. Cox, 'The numerical evaluation of b-splines', J. Inst. Maths
    Applics, 10, p.134-149, 1972.
.. [3] P. Dierckx, 'Curve and surface fitting with splines', Monographs
    on Numerical Analysis, Oxford University Press, 1993.
*)

val splint : ?full_output:int -> a:Py.Object.t -> b:Py.Object.t -> tck:Py.Object.t -> unit -> (float * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Evaluate the definite integral of a B-spline between two given points.

Parameters
----------
a, b : float
    The end-points of the integration interval.
tck : tuple or a BSpline instance
    If a tuple, then it should be a sequence of length 3, containing the
    vector of knots, the B-spline coefficients, and the degree of the
    spline (see `splev`).
full_output : int, optional
    Non-zero to return optional output.

Returns
-------
integral : float
    The resulting integral.
wrk : ndarray
    An array containing the integrals of the normalized B-splines
    defined on the set of knots.
    (Only returned if `full_output` is non-zero)

Notes
-----
`splint` silently assumes that the spline function is zero outside the data
interval (`a`, `b`).

Manipulating the tck-tuples directly is not recommended. In new code,
prefer using the `BSpline` objects.

See Also
--------
splprep, splrep, sproot, spalde, splev
bisplrep, bisplev
BSpline

References
----------
.. [1] P.W. Gaffney, The calculation of indefinite integrals of b-splines',
    J. Inst. Maths Applics, 17, p.37-41, 1976.
.. [2] P. Dierckx, 'Curve and surface fitting with splines', Monographs
    on Numerical Analysis, Oxford University Press, 1993.
*)

val splprep : ?w:[>`Ndarray] Np.Obj.t -> ?u:[>`Ndarray] Np.Obj.t -> ?ub:Py.Object.t -> ?ue:Py.Object.t -> ?k:int -> ?task:int -> ?s:float -> ?t:int -> ?full_output:int -> ?nest:int -> ?per:int -> ?quiet:int -> x:[>`Ndarray] Np.Obj.t -> unit -> (Py.Object.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t * float * int * string)
(**
Find the B-spline representation of an N-dimensional curve.

Given a list of N rank-1 arrays, `x`, which represent a curve in
N-dimensional space parametrized by `u`, find a smooth approximating
spline curve g(`u`). Uses the FORTRAN routine parcur from FITPACK.

Parameters
----------
x : array_like
    A list of sample vector arrays representing the curve.
w : array_like, optional
    Strictly positive rank-1 array of weights the same length as `x[0]`.
    The weights are used in computing the weighted least-squares spline
    fit. If the errors in the `x` values have standard-deviation given by
    the vector d, then `w` should be 1/d. Default is ``ones(len(x[0]))``.
u : array_like, optional
    An array of parameter values. If not given, these values are
    calculated automatically as ``M = len(x[0])``, where

        v[0] = 0

        v[i] = v[i-1] + distance(`x[i]`, `x[i-1]`)

        u[i] = v[i] / v[M-1]

ub, ue : int, optional
    The end-points of the parameters interval.  Defaults to
    u[0] and u[-1].
k : int, optional
    Degree of the spline. Cubic splines are recommended.
    Even values of `k` should be avoided especially with a small s-value.
    ``1 <= k <= 5``, default is 3.
task : int, optional
    If task==0 (default), find t and c for a given smoothing factor, s.
    If task==1, find t and c for another value of the smoothing factor, s.
    There must have been a previous call with task=0 or task=1
    for the same set of data.
    If task=-1 find the weighted least square spline for a given set of
    knots, t.
s : float, optional
    A smoothing condition.  The amount of smoothness is determined by
    satisfying the conditions: ``sum((w * (y - g))**2,axis=0) <= s``,
    where g(x) is the smoothed interpolation of (x,y).  The user can
    use `s` to control the trade-off between closeness and smoothness
    of fit.  Larger `s` means more smoothing while smaller values of `s`
    indicate less smoothing. Recommended values of `s` depend on the
    weights, w.  If the weights represent the inverse of the
    standard-deviation of y, then a good `s` value should be found in
    the range ``(m-sqrt(2*m),m+sqrt(2*m))``, where m is the number of
    data points in x, y, and w.
t : int, optional
    The knots needed for task=-1.
full_output : int, optional
    If non-zero, then return optional outputs.
nest : int, optional
    An over-estimate of the total number of knots of the spline to
    help in determining the storage space.  By default nest=m/2.
    Always large enough is nest=m+k+1.
per : int, optional
   If non-zero, data points are considered periodic with period
   ``x[m-1] - x[0]`` and a smooth periodic spline approximation is
   returned.  Values of ``y[m-1]`` and ``w[m-1]`` are not used.
quiet : int, optional
     Non-zero to suppress messages.
     This parameter is deprecated; use standard Python warning filters
     instead.

Returns
-------
tck : tuple
    (t,c,k) a tuple containing the vector of knots, the B-spline
    coefficients, and the degree of the spline.
u : array
    An array of the values of the parameter.
fp : float
    The weighted sum of squared residuals of the spline approximation.
ier : int
    An integer flag about splrep success.  Success is indicated
    if ier<=0. If ier in [1,2,3] an error occurred but was not raised.
    Otherwise an error is raised.
msg : str
    A message corresponding to the integer flag, ier.

See Also
--------
splrep, splev, sproot, spalde, splint,
bisplrep, bisplev
UnivariateSpline, BivariateSpline
BSpline
make_interp_spline

Notes
-----
See `splev` for evaluation of the spline and its derivatives.
The number of dimensions N must be smaller than 11.

The number of coefficients in the `c` array is ``k+1`` less then the number
of knots, ``len(t)``. This is in contrast with `splrep`, which zero-pads
the array of coefficients to have the same length as the array of knots.
These additional coefficients are ignored by evaluation routines, `splev`
and `BSpline`.

References
----------
.. [1] P. Dierckx, 'Algorithms for smoothing data with periodic and
    parametric splines, Computer Graphics and Image Processing',
    20 (1982) 171-184.
.. [2] P. Dierckx, 'Algorithms for smoothing data with periodic and
    parametric splines', report tw55, Dept. Computer Science,
    K.U.Leuven, 1981.
.. [3] P. Dierckx, 'Curve and surface fitting with splines', Monographs on
    Numerical Analysis, Oxford University Press, 1993.

Examples
--------
Generate a discretization of a limacon curve in the polar coordinates:

>>> phi = np.linspace(0, 2.*np.pi, 40)
>>> r = 0.5 + np.cos(phi)         # polar coords
>>> x, y = r * np.cos(phi), r * np.sin(phi)    # convert to cartesian

And interpolate:

>>> from scipy.interpolate import splprep, splev
>>> tck, u = splprep([x, y], s=0)
>>> new_points = splev(u, tck)

Notice that (i) we force interpolation by using `s=0`,
(ii) the parameterization, ``u``, is generated automatically.
Now plot the result:

>>> import matplotlib.pyplot as plt
>>> fig, ax = plt.subplots()
>>> ax.plot(x, y, 'ro')
>>> ax.plot(new_points[0], new_points[1], 'r-')
>>> plt.show()
*)

val splrep : ?w:[>`Ndarray] Np.Obj.t -> ?xb:Py.Object.t -> ?xe:Py.Object.t -> ?k:int -> ?task:[`One | `T_1 of Py.Object.t | `Zero] -> ?s:float -> ?t:[>`Ndarray] Np.Obj.t -> ?full_output:bool -> ?per:bool -> ?quiet:bool -> x:Py.Object.t -> y:Py.Object.t -> unit -> (Py.Object.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t * int * string)
(**
Find the B-spline representation of 1-D curve.

Given the set of data points ``(x[i], y[i])`` determine a smooth spline
approximation of degree k on the interval ``xb <= x <= xe``.

Parameters
----------
x, y : array_like
    The data points defining a curve y = f(x).
w : array_like, optional
    Strictly positive rank-1 array of weights the same length as x and y.
    The weights are used in computing the weighted least-squares spline
    fit. If the errors in the y values have standard-deviation given by the
    vector d, then w should be 1/d. Default is ones(len(x)).
xb, xe : float, optional
    The interval to fit.  If None, these default to x[0] and x[-1]
    respectively.
k : int, optional
    The degree of the spline fit. It is recommended to use cubic splines.
    Even values of k should be avoided especially with small s values.
    1 <= k <= 5
task : {1, 0, -1}, optional
    If task==0 find t and c for a given smoothing factor, s.

    If task==1 find t and c for another value of the smoothing factor, s.
    There must have been a previous call with task=0 or task=1 for the same
    set of data (t will be stored an used internally)

    If task=-1 find the weighted least square spline for a given set of
    knots, t. These should be interior knots as knots on the ends will be
    added automatically.
s : float, optional
    A smoothing condition. The amount of smoothness is determined by
    satisfying the conditions: sum((w * (y - g))**2,axis=0) <= s where g(x)
    is the smoothed interpolation of (x,y). The user can use s to control
    the tradeoff between closeness and smoothness of fit. Larger s means
    more smoothing while smaller values of s indicate less smoothing.
    Recommended values of s depend on the weights, w. If the weights
    represent the inverse of the standard-deviation of y, then a good s
    value should be found in the range (m-sqrt(2*m),m+sqrt(2*m)) where m is
    the number of datapoints in x, y, and w. default : s=m-sqrt(2*m) if
    weights are supplied. s = 0.0 (interpolating) if no weights are
    supplied.
t : array_like, optional
    The knots needed for task=-1. If given then task is automatically set
    to -1.
full_output : bool, optional
    If non-zero, then return optional outputs.
per : bool, optional
    If non-zero, data points are considered periodic with period x[m-1] -
    x[0] and a smooth periodic spline approximation is returned. Values of
    y[m-1] and w[m-1] are not used.
quiet : bool, optional
    Non-zero to suppress messages.
    This parameter is deprecated; use standard Python warning filters
    instead.

Returns
-------
tck : tuple
    A tuple (t,c,k) containing the vector of knots, the B-spline
    coefficients, and the degree of the spline.
fp : array, optional
    The weighted sum of squared residuals of the spline approximation.
ier : int, optional
    An integer flag about splrep success. Success is indicated if ier<=0.
    If ier in [1,2,3] an error occurred but was not raised. Otherwise an
    error is raised.
msg : str, optional
    A message corresponding to the integer flag, ier.

See Also
--------
UnivariateSpline, BivariateSpline
splprep, splev, sproot, spalde, splint
bisplrep, bisplev
BSpline
make_interp_spline

Notes
-----
See `splev` for evaluation of the spline and its derivatives. Uses the
FORTRAN routine ``curfit`` from FITPACK.

The user is responsible for assuring that the values of `x` are unique.
Otherwise, `splrep` will not return sensible results.

If provided, knots `t` must satisfy the Schoenberg-Whitney conditions,
i.e., there must be a subset of data points ``x[j]`` such that
``t[j] < x[j] < t[j+k+1]``, for ``j=0, 1,...,n-k-2``.

This routine zero-pads the coefficients array ``c`` to have the same length
as the array of knots ``t`` (the trailing ``k + 1`` coefficients are ignored
by the evaluation routines, `splev` and `BSpline`.) This is in contrast with
`splprep`, which does not zero-pad the coefficients.

References
----------
Based on algorithms described in [1]_, [2]_, [3]_, and [4]_:

.. [1] P. Dierckx, 'An algorithm for smoothing, differentiation and
   integration of experimental data using spline functions',
   J.Comp.Appl.Maths 1 (1975) 165-184.
.. [2] P. Dierckx, 'A fast algorithm for smoothing data on a rectangular
   grid while using spline functions', SIAM J.Numer.Anal. 19 (1982)
   1286-1304.
.. [3] P. Dierckx, 'An improved algorithm for curve fitting with spline
   functions', report tw54, Dept. Computer Science,K.U. Leuven, 1981.
.. [4] P. Dierckx, 'Curve and surface fitting with splines', Monographs on
   Numerical Analysis, Oxford University Press, 1993.

Examples
--------

>>> import matplotlib.pyplot as plt
>>> from scipy.interpolate import splev, splrep
>>> x = np.linspace(0, 10, 10)
>>> y = np.sin(x)
>>> spl = splrep(x, y)
>>> x2 = np.linspace(0, 10, 200)
>>> y2 = splev(x2, spl)
>>> plt.plot(x, y, 'o', x2, y2)
>>> plt.show()
*)

val sproot : ?mest:int -> tck:Py.Object.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Find the roots of a cubic B-spline.

Given the knots (>=8) and coefficients of a cubic B-spline return the
roots of the spline.

Parameters
----------
tck : tuple or a BSpline object
    If a tuple, then it should be a sequence of length 3, containing the
    vector of knots, the B-spline coefficients, and the degree of the
    spline.
    The number of knots must be >= 8, and the degree must be 3.
    The knots must be a montonically increasing sequence.
mest : int, optional
    An estimate of the number of zeros (Default is 10).

Returns
-------
zeros : ndarray
    An array giving the roots of the spline.

Notes
-----
Manipulating the tck-tuples directly is not recommended. In new code,
prefer using the `BSpline` objects.

See also
--------
splprep, splrep, splint, spalde, splev
bisplrep, bisplev
BSpline


References
----------
.. [1] C. de Boor, 'On calculating with b-splines', J. Approximation
    Theory, 6, p.50-62, 1972.
.. [2] M. G. Cox, 'The numerical evaluation of b-splines', J. Inst. Maths
    Applics, 10, p.134-149, 1972.
.. [3] P. Dierckx, 'Curve and surface fitting with splines', Monographs
    on Numerical Analysis, Oxford University Press, 1993.
*)

