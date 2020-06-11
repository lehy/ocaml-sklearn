(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module Data : sig
type tag = [`Data]
type t = [`Data | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?y:[>`Ndarray] Np.Obj.t -> ?we:[>`Ndarray] Np.Obj.t -> ?wd:[>`Ndarray] Np.Obj.t -> ?fix:Py.Object.t -> ?meta:Py.Object.t -> x:[>`Ndarray] Np.Obj.t -> unit -> t
(**
The data to fit.

Parameters
----------
x : array_like
    Observed data for the independent variable of the regression
y : array_like, optional
    If array-like, observed data for the dependent variable of the
    regression. A scalar input implies that the model to be used on
    the data is implicit.
we : array_like, optional
    If `we` is a scalar, then that value is used for all data points (and
    all dimensions of the response variable).
    If `we` is a rank-1 array of length q (the dimensionality of the
    response variable), then this vector is the diagonal of the covariant
    weighting matrix for all data points.
    If `we` is a rank-1 array of length n (the number of data points), then
    the i'th element is the weight for the i'th response variable
    observation (single-dimensional only).
    If `we` is a rank-2 array of shape (q, q), then this is the full
    covariant weighting matrix broadcast to each observation.
    If `we` is a rank-2 array of shape (q, n), then `we[:,i]` is the
    diagonal of the covariant weighting matrix for the i'th observation.
    If `we` is a rank-3 array of shape (q, q, n), then `we[:,:,i]` is the
    full specification of the covariant weighting matrix for each
    observation.
    If the fit is implicit, then only a positive scalar value is used.
wd : array_like, optional
    If `wd` is a scalar, then that value is used for all data points
    (and all dimensions of the input variable). If `wd` = 0, then the
    covariant weighting matrix for each observation is set to the identity
    matrix (so each dimension of each observation has the same weight).
    If `wd` is a rank-1 array of length m (the dimensionality of the input
    variable), then this vector is the diagonal of the covariant weighting
    matrix for all data points.
    If `wd` is a rank-1 array of length n (the number of data points), then
    the i'th element is the weight for the i'th input variable observation
    (single-dimensional only).
    If `wd` is a rank-2 array of shape (m, m), then this is the full
    covariant weighting matrix broadcast to each observation.
    If `wd` is a rank-2 array of shape (m, n), then `wd[:,i]` is the
    diagonal of the covariant weighting matrix for the i'th observation.
    If `wd` is a rank-3 array of shape (m, m, n), then `wd[:,:,i]` is the
    full specification of the covariant weighting matrix for each
    observation.
fix : array_like of ints, optional
    The `fix` argument is the same as ifixx in the class ODR. It is an
    array of integers with the same shape as data.x that determines which
    input observations are treated as fixed. One can use a sequence of
    length m (the dimensionality of the input observations) to fix some
    dimensions for all observations. A value of 0 fixes the observation,
    a value > 0 makes it free.
meta : dict, optional
    Free-form dictionary for metadata.

Notes
-----
Each argument is attached to the member of the instance of the same name.
The structures of `x` and `y` are described in the Model class docstring.
If `y` is an integer, then the Data instance can only be used to fit with
implicit models where the dimensionality of the response is equal to the
specified value of `y`.

The `we` argument weights the effect a deviation in the response variable
has on the fit.  The `wd` argument weights the effect a deviation in the
input variable has on the fit. To handle multidimensional inputs and
responses easily, the structure of these arguments has the n'th
dimensional axis first. These arguments heavily use the structured
arguments feature of ODRPACK to conveniently and flexibly support all
options. See the ODRPACK User's Guide for a full explanation of how these
weights are used in the algorithm. Basically, a higher value of the weight
for a particular data point makes a deviation at that point more
detrimental to the fit.
*)

val set_meta : ?kwds:(string * Py.Object.t) list -> [> tag] Obj.t -> Py.Object.t
(**
Update the metadata dictionary with the keywords and data provided
by keywords.

Examples
--------
::

    data.set_meta(lab='Ph 7; Lab 26', title='Ag110 + Ag108 Decay')
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Model : sig
type tag = [`Model]
type t = [`Model | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?fjacb:Py.Object.t -> ?fjacd:Py.Object.t -> ?extra_args:Py.Object.t -> ?estimate:Py.Object.t -> ?implicit:bool -> ?meta:Py.Object.t -> fcn:Py.Object.t -> unit -> t
(**
The Model class stores information about the function you wish to fit.

It stores the function itself, at the least, and optionally stores
functions which compute the Jacobians used during fitting. Also, one
can provide a function that will provide reasonable starting values
for the fit parameters possibly given the set of data.

Parameters
----------
fcn : function
      fcn(beta, x) --> y
fjacb : function
      Jacobian of fcn wrt the fit parameters beta.

      fjacb(beta, x) --> @f_i(x,B)/@B_j
fjacd : function
      Jacobian of fcn wrt the (possibly multidimensional) input
      variable.

      fjacd(beta, x) --> @f_i(x,B)/@x_j
extra_args : tuple, optional
      If specified, `extra_args` should be a tuple of extra
      arguments to pass to `fcn`, `fjacb`, and `fjacd`. Each will be called
      by `apply(fcn, (beta, x) + extra_args)`
estimate : array_like of rank-1
      Provides estimates of the fit parameters from the data

      estimate(data) --> estbeta
implicit : boolean
      If TRUE, specifies that the model
      is implicit; i.e `fcn(beta, x)` ~= 0 and there is no y data to fit
      against
meta : dict, optional
      freeform dictionary of metadata for the model

Notes
-----
Note that the `fcn`, `fjacb`, and `fjacd` operate on NumPy arrays and
return a NumPy array. The `estimate` object takes an instance of the
Data class.

Here are the rules for the shapes of the argument and return
arrays of the callback functions:

`x`
    if the input data is single-dimensional, then `x` is rank-1
    array; i.e. ``x = array([1, 2, 3, ...]); x.shape = (n,)``
    If the input data is multi-dimensional, then `x` is a rank-2 array;
    i.e., ``x = array([[1, 2, ...], [2, 4, ...]]); x.shape = (m, n)``.
    In all cases, it has the same shape as the input data array passed to
    `~scipy.odr.odr`. `m` is the dimensionality of the input data,
    `n` is the number of observations.
`y`
    if the response variable is single-dimensional, then `y` is a
    rank-1 array, i.e., ``y = array([2, 4, ...]); y.shape = (n,)``.
    If the response variable is multi-dimensional, then `y` is a rank-2
    array, i.e., ``y = array([[2, 4, ...], [3, 6, ...]]); y.shape =
    (q, n)`` where `q` is the dimensionality of the response variable.
`beta`
    rank-1 array of length `p` where `p` is the number of parameters;
    i.e. ``beta = array([B_1, B_2, ..., B_p])``
`fjacb`
    if the response variable is multi-dimensional, then the
    return array's shape is `(q, p, n)` such that ``fjacb(x,beta)[l,k,i] =
    d f_l(X,B)/d B_k`` evaluated at the i'th data point.  If `q == 1`, then
    the return array is only rank-2 and with shape `(p, n)`.
`fjacd`
    as with fjacb, only the return array's shape is `(q, m, n)`
    such that ``fjacd(x,beta)[l,j,i] = d f_l(X,B)/d X_j`` at the i'th data
    point.  If `q == 1`, then the return array's shape is `(m, n)`. If
    `m == 1`, the shape is (q, n). If `m == q == 1`, the shape is `(n,)`.
*)

val set_meta : ?kwds:(string * Py.Object.t) list -> [> tag] Obj.t -> Py.Object.t
(**
Update the metadata dictionary with the keywords and data provided
here.

Examples
--------
set_meta(name='Exponential', equation='y = a exp(b x) + c')
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module ODR : sig
type tag = [`ODR]
type t = [`ODR | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?beta0:Py.Object.t -> ?delta0:Py.Object.t -> ?ifixb:Py.Object.t -> ?ifixx:Py.Object.t -> ?job:Py.Object.t -> ?iprint:Py.Object.t -> ?errfile:Py.Object.t -> ?rptfile:Py.Object.t -> ?ndigit:Py.Object.t -> ?taufac:Py.Object.t -> ?sstol:Py.Object.t -> ?partol:Py.Object.t -> ?maxit:Py.Object.t -> ?stpb:Py.Object.t -> ?stpd:Py.Object.t -> ?sclb:Py.Object.t -> ?scld:Py.Object.t -> ?work:Py.Object.t -> ?iwork:Py.Object.t -> data:Py.Object.t -> model:Py.Object.t -> unit -> t
(**
The ODR class gathers all information and coordinates the running of the
main fitting routine.

Members of instances of the ODR class have the same names as the arguments
to the initialization routine.

Parameters
----------
data : Data class instance
    instance of the Data class
model : Model class instance
    instance of the Model class

Other Parameters
----------------
beta0 : array_like of rank-1
    a rank-1 sequence of initial parameter values. Optional if
    model provides an 'estimate' function to estimate these values.
delta0 : array_like of floats of rank-1, optional
    a (double-precision) float array to hold the initial values of
    the errors in the input variables. Must be same shape as data.x
ifixb : array_like of ints of rank-1, optional
    sequence of integers with the same length as beta0 that determines
    which parameters are held fixed. A value of 0 fixes the parameter,
    a value > 0 makes the parameter free.
ifixx : array_like of ints with same shape as data.x, optional
    an array of integers with the same shape as data.x that determines
    which input observations are treated as fixed. One can use a sequence
    of length m (the dimensionality of the input observations) to fix some
    dimensions for all observations. A value of 0 fixes the observation,
    a value > 0 makes it free.
job : int, optional
    an integer telling ODRPACK what tasks to perform. See p. 31 of the
    ODRPACK User's Guide if you absolutely must set the value here. Use the
    method set_job post-initialization for a more readable interface.
iprint : int, optional
    an integer telling ODRPACK what to print. See pp. 33-34 of the
    ODRPACK User's Guide if you absolutely must set the value here. Use the
    method set_iprint post-initialization for a more readable interface.
errfile : str, optional
    string with the filename to print ODRPACK errors to. *Do Not Open
    This File Yourself!*
rptfile : str, optional
    string with the filename to print ODRPACK summaries to. *Do Not
    Open This File Yourself!*
ndigit : int, optional
    integer specifying the number of reliable digits in the computation
    of the function.
taufac : float, optional
    float specifying the initial trust region. The default value is 1.
    The initial trust region is equal to taufac times the length of the
    first computed Gauss-Newton step. taufac must be less than 1.
sstol : float, optional
    float specifying the tolerance for convergence based on the relative
    change in the sum-of-squares. The default value is eps**(1/2) where eps
    is the smallest value such that 1 + eps > 1 for double precision
    computation on the machine. sstol must be less than 1.
partol : float, optional
    float specifying the tolerance for convergence based on the relative
    change in the estimated parameters. The default value is eps**(2/3) for
    explicit models and ``eps**(1/3)`` for implicit models. partol must be less
    than 1.
maxit : int, optional
    integer specifying the maximum number of iterations to perform. For
    first runs, maxit is the total number of iterations performed and
    defaults to 50.  For restarts, maxit is the number of additional
    iterations to perform and defaults to 10.
stpb : array_like, optional
    sequence (``len(stpb) == len(beta0)``) of relative step sizes to compute
    finite difference derivatives wrt the parameters.
stpd : optional
    array (``stpd.shape == data.x.shape`` or ``stpd.shape == (m,)``) of relative
    step sizes to compute finite difference derivatives wrt the input
    variable errors. If stpd is a rank-1 array with length m (the
    dimensionality of the input variable), then the values are broadcast to
    all observations.
sclb : array_like, optional
    sequence (``len(stpb) == len(beta0)``) of scaling factors for the
    parameters.  The purpose of these scaling factors are to scale all of
    the parameters to around unity. Normally appropriate scaling factors
    are computed if this argument is not specified. Specify them yourself
    if the automatic procedure goes awry.
scld : array_like, optional
    array (scld.shape == data.x.shape or scld.shape == (m,)) of scaling
    factors for the *errors* in the input variables. Again, these factors
    are automatically computed if you do not provide them. If scld.shape ==
    (m,), then the scaling factors are broadcast to all observations.
work : ndarray, optional
    array to hold the double-valued working data for ODRPACK. When
    restarting, takes the value of self.output.work.
iwork : ndarray, optional
    array to hold the integer-valued working data for ODRPACK. When
    restarting, takes the value of self.output.iwork.

Attributes
----------
data : Data
    The data for this fit
model : Model
    The model used in fit
output : Output
    An instance if the Output class containing all of the returned
    data from an invocation of ODR.run() or ODR.restart()
*)

val restart : ?iter:int -> [> tag] Obj.t -> Py.Object.t
(**
Restarts the run with iter more iterations.

Parameters
----------
iter : int, optional
    ODRPACK's default for the number of new iterations is 10.

Returns
-------
output : Output instance
    This object is also assigned to the attribute .output .
*)

val run : [> tag] Obj.t -> Py.Object.t
(**
Run the fitting routine with all of the information given and with ``full_output=1``.

Returns
-------
output : Output instance
    This object is also assigned to the attribute .output .
*)

val set_iprint : ?init:Py.Object.t -> ?so_init:Py.Object.t -> ?iter:Py.Object.t -> ?so_iter:Py.Object.t -> ?iter_step:Py.Object.t -> ?final:Py.Object.t -> ?so_final:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Set the iprint parameter for the printing of computation reports.

If any of the arguments are specified here, then they are set in the
iprint member. If iprint is not set manually or with this method, then
ODRPACK defaults to no printing. If no filename is specified with the
member rptfile, then ODRPACK prints to stdout. One can tell ODRPACK to
print to stdout in addition to the specified filename by setting the
so_* arguments to this function, but one cannot specify to print to
stdout but not a file since one can do that by not specifying a rptfile
filename.

There are three reports: initialization, iteration, and final reports.
They are represented by the arguments init, iter, and final
respectively.  The permissible values are 0, 1, and 2 representing 'no
report', 'short report', and 'long report' respectively.

The argument iter_step (0 <= iter_step <= 9) specifies how often to make
the iteration report; the report will be made for every iter_step'th
iteration starting with iteration one. If iter_step == 0, then no
iteration report is made, regardless of the other arguments.

If the rptfile is None, then any so_* arguments supplied will raise an
exception.
*)

val set_job : ?fit_type:[`One | `PyObject of Py.Object.t] -> ?deriv:[`One | `Two | `PyObject of Py.Object.t] -> ?var_calc:[`One | `PyObject of Py.Object.t] -> ?del_init:Py.Object.t -> ?restart:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Sets the 'job' parameter is a hopefully comprehensible way.

If an argument is not specified, then the value is left as is. The
default value from class initialization is for all of these options set
to 0.

Parameters
----------
fit_type : {0, 1, 2} int
    0 -> explicit ODR

    1 -> implicit ODR

    2 -> ordinary least-squares
deriv : {0, 1, 2, 3} int
    0 -> forward finite differences

    1 -> central finite differences

    2 -> user-supplied derivatives (Jacobians) with results
      checked by ODRPACK

    3 -> user-supplied derivatives, no checking
var_calc : {0, 1, 2} int
    0 -> calculate asymptotic covariance matrix and fit
         parameter uncertainties (V_B, s_B) using derivatives
         recomputed at the final solution

    1 -> calculate V_B and s_B using derivatives from last iteration

    2 -> do not calculate V_B and s_B
del_init : {0, 1} int
    0 -> initial input variable offsets set to 0

    1 -> initial offsets provided by user in variable 'work'
restart : {0, 1} int
    0 -> fit is not a restart

    1 -> fit is a restart

Notes
-----
The permissible values are different from those given on pg. 31 of the
ODRPACK User's Guide only in that one cannot specify numbers greater than
the last value for each variable.

If one does not supply functions to compute the Jacobians, the fitting
procedure will change deriv to 0, finite differences, as a default. To
initialize the input variable offsets by yourself, set del_init to 1 and
put the offsets into the 'work' variable correctly.
*)


(** Attribute data: get value or raise Not_found if None.*)
val data : t -> Py.Object.t

(** Attribute data: get value as an option. *)
val data_opt : t -> (Py.Object.t) option


(** Attribute model: get value or raise Not_found if None.*)
val model : t -> Py.Object.t

(** Attribute model: get value as an option. *)
val model_opt : t -> (Py.Object.t) option


(** Attribute output: get value or raise Not_found if None.*)
val output : t -> Py.Object.t

(** Attribute output: get value as an option. *)
val output_opt : t -> (Py.Object.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module OdrError : sig
type tag = [`OdrError]
type t = [`BaseException | `Object | `OdrError] Obj.t
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

module OdrStop : sig
type tag = [`OdrStop]
type t = [`BaseException | `Object | `OdrStop] Obj.t
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

module OdrWarning : sig
type tag = [`OdrWarning]
type t = [`BaseException | `Object | `OdrWarning] Obj.t
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

module Output : sig
type tag = [`Output]
type t = [`Object | `Output] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : Py.Object.t -> t
(**
The Output class stores the output of an ODR run.

Attributes
----------
beta : ndarray
    Estimated parameter values, of shape (q,).
sd_beta : ndarray
    Standard errors of the estimated parameters, of shape (p,).
cov_beta : ndarray
    Covariance matrix of the estimated parameters, of shape (p,p).
delta : ndarray, optional
    Array of estimated errors in input variables, of same shape as `x`.
eps : ndarray, optional
    Array of estimated errors in response variables, of same shape as `y`.
xplus : ndarray, optional
    Array of ``x + delta``.
y : ndarray, optional
    Array ``y = fcn(x + delta)``.
res_var : float, optional
    Residual variance.
sum_square : float, optional
    Sum of squares error.
sum_square_delta : float, optional
    Sum of squares of delta error.
sum_square_eps : float, optional
    Sum of squares of eps error.
inv_condnum : float, optional
    Inverse condition number (cf. ODRPACK UG p. 77).
rel_error : float, optional
    Relative error in function values computed within fcn.
work : ndarray, optional
    Final work array.
work_ind : dict, optional
    Indices into work for drawing out values (cf. ODRPACK UG p. 83).
info : int, optional
    Reason for returning, as output by ODRPACK (cf. ODRPACK UG p. 38).
stopreason : list of str, optional
    `info` interpreted into English.

Notes
-----
Takes one argument for initialization, the return value from the
function `~scipy.odr.odr`. The attributes listed as 'optional' above are
only present if `~scipy.odr.odr` was run with ``full_output=1``.
*)

val pprint : [> tag] Obj.t -> Py.Object.t
(**
Pretty-print important results.
        
*)


(** Attribute beta: get value or raise Not_found if None.*)
val beta : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute beta: get value as an option. *)
val beta_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute sd_beta: get value or raise Not_found if None.*)
val sd_beta : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute sd_beta: get value as an option. *)
val sd_beta_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute cov_beta: get value or raise Not_found if None.*)
val cov_beta : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute cov_beta: get value as an option. *)
val cov_beta_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute delta: get value or raise Not_found if None.*)
val delta : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute delta: get value as an option. *)
val delta_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute eps: get value or raise Not_found if None.*)
val eps : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute eps: get value as an option. *)
val eps_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute xplus: get value or raise Not_found if None.*)
val xplus : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute xplus: get value as an option. *)
val xplus_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute y: get value or raise Not_found if None.*)
val y : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute y: get value as an option. *)
val y_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute res_var: get value or raise Not_found if None.*)
val res_var : t -> float

(** Attribute res_var: get value as an option. *)
val res_var_opt : t -> (float) option


(** Attribute sum_square: get value or raise Not_found if None.*)
val sum_square : t -> float

(** Attribute sum_square: get value as an option. *)
val sum_square_opt : t -> (float) option


(** Attribute sum_square_delta: get value or raise Not_found if None.*)
val sum_square_delta : t -> float

(** Attribute sum_square_delta: get value as an option. *)
val sum_square_delta_opt : t -> (float) option


(** Attribute sum_square_eps: get value or raise Not_found if None.*)
val sum_square_eps : t -> float

(** Attribute sum_square_eps: get value as an option. *)
val sum_square_eps_opt : t -> (float) option


(** Attribute inv_condnum: get value or raise Not_found if None.*)
val inv_condnum : t -> float

(** Attribute inv_condnum: get value as an option. *)
val inv_condnum_opt : t -> (float) option


(** Attribute rel_error: get value or raise Not_found if None.*)
val rel_error : t -> float

(** Attribute rel_error: get value as an option. *)
val rel_error_opt : t -> (float) option


(** Attribute work: get value or raise Not_found if None.*)
val work : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute work: get value as an option. *)
val work_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute work_ind: get value or raise Not_found if None.*)
val work_ind : t -> Py.Object.t

(** Attribute work_ind: get value as an option. *)
val work_ind_opt : t -> (Py.Object.t) option


(** Attribute info: get value or raise Not_found if None.*)
val info : t -> int

(** Attribute info: get value as an option. *)
val info_opt : t -> (int) option


(** Attribute stopreason: get value or raise Not_found if None.*)
val stopreason : t -> string list

(** Attribute stopreason: get value as an option. *)
val stopreason_opt : t -> (string list) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module RealData : sig
type tag = [`RealData]
type t = [`Object | `RealData] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?y:[>`Ndarray] Np.Obj.t -> ?sx:[>`Ndarray] Np.Obj.t -> ?sy:[>`Ndarray] Np.Obj.t -> ?covx:[>`Ndarray] Np.Obj.t -> ?covy:[>`Ndarray] Np.Obj.t -> ?fix:[>`Ndarray] Np.Obj.t -> ?meta:Py.Object.t -> x:[>`Ndarray] Np.Obj.t -> unit -> t
(**
The data, with weightings as actual standard deviations and/or
covariances.

Parameters
----------
x : array_like
    Observed data for the independent variable of the regression
y : array_like, optional
    If array-like, observed data for the dependent variable of the
    regression. A scalar input implies that the model to be used on
    the data is implicit.
sx : array_like, optional
    Standard deviations of `x`.
    `sx` are standard deviations of `x` and are converted to weights by
    dividing 1.0 by their squares.
sy : array_like, optional
    Standard deviations of `y`.
    `sy` are standard deviations of `y` and are converted to weights by
    dividing 1.0 by their squares.
covx : array_like, optional
    Covariance of `x`
    `covx` is an array of covariance matrices of `x` and are converted to
    weights by performing a matrix inversion on each observation's
    covariance matrix.
covy : array_like, optional
    Covariance of `y`
    `covy` is an array of covariance matrices and are converted to
    weights by performing a matrix inversion on each observation's
    covariance matrix.
fix : array_like, optional
    The argument and member fix is the same as Data.fix and ODR.ifixx:
    It is an array of integers with the same shape as `x` that
    determines which input observations are treated as fixed. One can
    use a sequence of length m (the dimensionality of the input
    observations) to fix some dimensions for all observations. A value
    of 0 fixes the observation, a value > 0 makes it free.
meta : dict, optional
    Free-form dictionary for metadata.

Notes
-----
The weights `wd` and `we` are computed from provided values as follows:

`sx` and `sy` are converted to weights by dividing 1.0 by their squares.
For example, ``wd = 1./numpy.power(`sx`, 2)``.

`covx` and `covy` are arrays of covariance matrices and are converted to
weights by performing a matrix inversion on each observation's covariance
matrix.  For example, ``we[i] = numpy.linalg.inv(covy[i])``.

These arguments follow the same structured argument conventions as wd and
we only restricted by their natures: `sx` and `sy` can't be rank-3, but
`covx` and `covy` can be.

Only set *either* `sx` or `covx` (not both). Setting both will raise an
exception.  Same with `sy` and `covy`.
*)

val set_meta : ?kwds:(string * Py.Object.t) list -> [> tag] Obj.t -> Py.Object.t
(**
Update the metadata dictionary with the keywords and data provided
by keywords.

Examples
--------
::

    data.set_meta(lab='Ph 7; Lab 26', title='Ag110 + Ag108 Decay')
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Add_newdocs : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val add_newdoc : ?warn_on_python:bool -> place:string -> obj:string -> doc:[`S of string | `PyObject of Py.Object.t] -> unit -> Py.Object.t
(**
Add documentation to an existing object, typically one defined in C

The purpose is to allow easier editing of the docstrings without requiring
a re-compile. This exists primarily for internal use within numpy itself.

Parameters
----------
place : str
    The absolute name of the module to import from
obj : str
    The name of the object to add documentation to, typically a class or
    function name
doc : {str, Tuple[str, str], List[Tuple[str, str]]}
    If a string, the documentation to apply to `obj`

    If a tuple, then the first element is interpreted as an attribute of
    `obj` and the second as the docstring to apply - ``(method, docstring)``

    If a list, then each element of the list should be a tuple of length
    two - ``[(method1, docstring1), (method2, docstring2), ...]``
warn_on_python : bool
    If True, the default, emit `UserWarning` if this is used to attach
    documentation to a pure-python object.

Notes
-----
This routine never raises an error if the docstring can't be written, but
will raise an error if the object being documented does not exist.

This routine cannot modify read-only docstrings, as appear
in new-style classes or built-in functions. Because this
routine never raises an error the caller must check manually
that the docstrings were changed.

Since this function grabs the ``char *`` from a c-level str object and puts
it into the ``tp_doc`` slot of the type of `obj`, it violates a number of
C-API best-practices, by:

- modifying a `PyTypeObject` after calling `PyType_Ready`
- calling `Py_INCREF` on the str and losing the reference, so the str
  will never be released

If possible it should be avoided.
*)


end

module Models : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val polynomial : [`Sequence of Py.Object.t | `I of int] -> Py.Object.t
(**
Factory function for a general polynomial model.

Parameters
----------
order : int or sequence
    If an integer, it becomes the order of the polynomial to fit. If
    a sequence of numbers, then these are the explicit powers in the
    polynomial.
    A constant term (power 0) is always included, so don't include 0.
    Thus, polynomial(n) is equivalent to polynomial(range(1, n+1)).

Returns
-------
polynomial : Model instance
    Model instance.
*)


end

module Odrpack : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val odr : ?we:Py.Object.t -> ?wd:Py.Object.t -> ?fjacb:Py.Object.t -> ?fjacd:Py.Object.t -> ?extra_args:Py.Object.t -> ?ifixx:Py.Object.t -> ?ifixb:Py.Object.t -> ?job:Py.Object.t -> ?iprint:Py.Object.t -> ?errfile:Py.Object.t -> ?rptfile:Py.Object.t -> ?ndigit:Py.Object.t -> ?taufac:Py.Object.t -> ?sstol:Py.Object.t -> ?partol:Py.Object.t -> ?maxit:Py.Object.t -> ?stpb:Py.Object.t -> ?stpd:Py.Object.t -> ?sclb:Py.Object.t -> ?scld:Py.Object.t -> ?work:Py.Object.t -> ?iwork:Py.Object.t -> ?full_output:Py.Object.t -> fcn:Py.Object.t -> beta0:Py.Object.t -> y:Py.Object.t -> x:Py.Object.t -> unit -> Py.Object.t
(**
odr(fcn, beta0, y, x, we=None, wd=None, fjacb=None, fjacd=None, extra_args=None, ifixx=None, ifixb=None, job=0, iprint=0, errfile=None, rptfile=None, ndigit=0, taufac=0.0, sstol=-1.0, partol=-1.0, maxit=-1, stpb=None, stpd=None, sclb=None, scld=None, work=None, iwork=None, full_output=0)

Low-level function for ODR.

See Also
--------
ODR
Model
Data
RealData

Notes
-----
This is a function performing the same operation as the `ODR`,
`Model` and `Data` classes together. The parameters of this
function are explained in the class documentation.
*)

val warn : ?category:Py.Object.t -> ?stacklevel:Py.Object.t -> ?source:Py.Object.t -> message:Py.Object.t -> unit -> Py.Object.t
(**
Issue a warning, or maybe ignore it or raise an exception.
*)


end

val odr : ?we:Py.Object.t -> ?wd:Py.Object.t -> ?fjacb:Py.Object.t -> ?fjacd:Py.Object.t -> ?extra_args:Py.Object.t -> ?ifixx:Py.Object.t -> ?ifixb:Py.Object.t -> ?job:Py.Object.t -> ?iprint:Py.Object.t -> ?errfile:Py.Object.t -> ?rptfile:Py.Object.t -> ?ndigit:Py.Object.t -> ?taufac:Py.Object.t -> ?sstol:Py.Object.t -> ?partol:Py.Object.t -> ?maxit:Py.Object.t -> ?stpb:Py.Object.t -> ?stpd:Py.Object.t -> ?sclb:Py.Object.t -> ?scld:Py.Object.t -> ?work:Py.Object.t -> ?iwork:Py.Object.t -> ?full_output:Py.Object.t -> fcn:Py.Object.t -> beta0:Py.Object.t -> y:Py.Object.t -> x:Py.Object.t -> unit -> Py.Object.t
(**
odr(fcn, beta0, y, x, we=None, wd=None, fjacb=None, fjacd=None, extra_args=None, ifixx=None, ifixb=None, job=0, iprint=0, errfile=None, rptfile=None, ndigit=0, taufac=0.0, sstol=-1.0, partol=-1.0, maxit=-1, stpb=None, stpd=None, sclb=None, scld=None, work=None, iwork=None, full_output=0)

Low-level function for ODR.

See Also
--------
ODR
Model
Data
RealData

Notes
-----
This is a function performing the same operation as the `ODR`,
`Model` and `Data` classes together. The parameters of this
function are explained in the class documentation.
*)

val polynomial : [`Sequence of Py.Object.t | `I of int] -> Py.Object.t
(**
Factory function for a general polynomial model.

Parameters
----------
order : int or sequence
    If an integer, it becomes the order of the polynomial to fit. If
    a sequence of numbers, then these are the explicit powers in the
    polynomial.
    A constant term (power 0) is always included, so don't include 0.
    Thus, polynomial(n) is equivalent to polynomial(range(1, n+1)).

Returns
-------
polynomial : Model instance
    Model instance.
*)

