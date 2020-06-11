(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module IsotonicRegression : sig
type tag = [`IsotonicRegression]
type t = [`BaseEstimator | `IsotonicRegression | `Object | `RegressorMixin | `TransformerMixin] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_regressor : t -> [`RegressorMixin] Obj.t
val as_transformer : t -> [`TransformerMixin] Obj.t
val as_estimator : t -> [`BaseEstimator] Obj.t
val create : ?y_min:Py.Object.t -> ?y_max:Py.Object.t -> ?increasing:[`S of string | `Bool of bool] -> ?out_of_bounds:string -> unit -> t
(**
Isotonic regression model.

The isotonic regression optimization problem is defined by::

    min sum w_i (y[i] - y_[i]) ** 2

    subject to y_[i] <= y_[j] whenever X[i] <= X[j]
    and min(y_) = y_min, max(y_) = y_max

where:
    - ``y[i]`` are inputs (real numbers)
    - ``y_[i]`` are fitted
    - ``X`` specifies the order.
      If ``X`` is non-decreasing then ``y_`` is non-decreasing.
    - ``w[i]`` are optional strictly positive weights (default to 1.0)

Read more in the :ref:`User Guide <isotonic>`.

.. versionadded:: 0.13

Parameters
----------
y_min : optional, default: None
    If not None, set the lowest value of the fit to y_min.

y_max : optional, default: None
    If not None, set the highest value of the fit to y_max.

increasing : boolean or string, optional, default: True
    If boolean, whether or not to fit the isotonic regression with y
    increasing or decreasing.

    The string value 'auto' determines whether y should
    increase or decrease based on the Spearman correlation estimate's
    sign.

out_of_bounds : string, optional, default: 'nan'
    The ``out_of_bounds`` parameter handles how x-values outside of the
    training domain are handled.  When set to 'nan', predicted y-values
    will be NaN.  When set to 'clip', predicted y-values will be
    set to the value corresponding to the nearest train interval endpoint.
    When set to 'raise', allow ``interp1d`` to throw ValueError.


Attributes
----------
X_min_ : float
    Minimum value of input array `X_` for left bound.

X_max_ : float
    Maximum value of input array `X_` for right bound.

f_ : function
    The stepwise interpolating function that covers the input domain ``X``.

Notes
-----
Ties are broken using the secondary method from Leeuw, 1977.

References
----------
Isotonic Median Regression: A Linear Programming Approach
Nilotpal Chakravarti
Mathematics of Operations Research
Vol. 14, No. 2 (May, 1989), pp. 303-308

Isotone Optimization in R : Pool-Adjacent-Violators
Algorithm (PAVA) and Active Set Methods
Leeuw, Hornik, Mair
Journal of Statistical Software 2009

Correctness of Kruskal's algorithms for monotone regression with ties
Leeuw, Psychometrica, 1977

Examples
--------
>>> from sklearn.datasets import make_regression
>>> from sklearn.isotonic import IsotonicRegression
>>> X, y = make_regression(n_samples=10, n_features=1, random_state=41)
>>> iso_reg = IsotonicRegression().fit(X.flatten(), y)
>>> iso_reg.predict([.1, .2])
array([1.8628..., 3.7256...])
*)

val fit : ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fit the model using X, y as training data.

Parameters
----------
X : array-like of shape (n_samples,)
    Training data.

y : array-like of shape (n_samples,)
    Training target.

sample_weight : array-like of shape (n_samples,), default=None
    Weights. If set to None, all weights will be set to 1 (equal
    weights).

Returns
-------
self : object
    Returns an instance of self.

Notes
-----
X is stored for future use, as :meth:`transform` needs X to interpolate
new input data.
*)

val fit_transform : ?y:[>`ArrayLike] Np.Obj.t -> ?fit_params:(string * Py.Object.t) list -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
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

val get_params : ?deep:bool -> [> tag] Obj.t -> Dict.t
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

val predict : t:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Predict new data by linear interpolation.

Parameters
----------
T : array-like of shape (n_samples,)
    Data to transform.

Returns
-------
T_ : array, shape=(n_samples,)
    Transformed data.
*)

val score : ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> float
(**
Return the coefficient of determination R^2 of the prediction.

The coefficient R^2 is defined as (1 - u/v), where u is the residual
sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
sum of squares ((y_true - y_true.mean()) ** 2).sum().
The best possible score is 1.0 and it can be negative (because the
model can be arbitrarily worse). A constant model that always
predicts the expected value of y, disregarding the input features,
would get a R^2 score of 0.0.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Test samples. For some estimators this may be a
    precomputed kernel matrix or a list of generic objects instead,
    shape = (n_samples, n_samples_fitted),
    where n_samples_fitted is the number of
    samples used in the fitting for the estimator.

y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    True values for X.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

Returns
-------
score : float
    R^2 of self.predict(X) wrt. y.

Notes
-----
The R2 score used when calling ``score`` on a regressor will use
``multioutput='uniform_average'`` from version 0.23 to keep consistent
with :func:`~sklearn.metrics.r2_score`. This will influence the
``score`` method of all the multioutput regressors (except for
:class:`~sklearn.multioutput.MultiOutputRegressor`). To specify the
default value manually and avoid the warning, please either call
:func:`~sklearn.metrics.r2_score` directly or make a custom scorer with
:func:`~sklearn.metrics.make_scorer` (the built-in scorer ``'r2'`` uses
``multioutput='uniform_average'``).
*)

val set_params : ?params:(string * Py.Object.t) list -> [> tag] Obj.t -> t
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

val transform : t:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Transform new data by linear interpolation

Parameters
----------
T : array-like of shape (n_samples,)
    Data to transform.

Returns
-------
T_ : array, shape=(n_samples,)
    The transformed data
*)


(** Attribute X_min_: get value or raise Not_found if None.*)
val x_min_ : t -> float

(** Attribute X_min_: get value as an option. *)
val x_min_opt : t -> (float) option


(** Attribute X_max_: get value or raise Not_found if None.*)
val x_max_ : t -> float

(** Attribute X_max_: get value as an option. *)
val x_max_opt : t -> (float) option


(** Attribute f_: get value or raise Not_found if None.*)
val f_ : t -> Py.Object.t

(** Attribute f_: get value as an option. *)
val f_opt : t -> (Py.Object.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val check_array : ?accept_sparse:[`StringList of string list | `S of string | `Bool of bool] -> ?accept_large_sparse:bool -> ?dtype:[`S of string | `Dtype of Np.Dtype.t | `Dtypes of Np.Dtype.t list | `None] -> ?order:[`F | `C] -> ?copy:bool -> ?force_all_finite:[`Allow_nan | `Bool of bool] -> ?ensure_2d:bool -> ?allow_nd:bool -> ?ensure_min_samples:int -> ?ensure_min_features:int -> ?warn_on_dtype:bool -> ?estimator:[>`BaseEstimator] Np.Obj.t -> array:Py.Object.t -> unit -> Py.Object.t
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

dtype : string, type, list of types or None (default='numeric')
    Data type of result. If None, the dtype of the input is preserved.
    If 'numeric', dtype is preserved unless array.dtype is object.
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

val check_increasing : x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> unit -> bool
(**
Determine whether y is monotonically correlated with x.

y is found increasing or decreasing with respect to x based on a Spearman
correlation test.

Parameters
----------
x : array-like of shape (n_samples,)
        Training data.

y : array-like of shape (n_samples,)
    Training target.

Returns
-------
increasing_bool : boolean
    Whether the relationship is increasing or decreasing.

Notes
-----
The Spearman correlation coefficient is estimated from the data, and the
sign of the resulting estimate is used as the result.

In the event that the 95% confidence interval based on Fisher transform
spans zero, a warning is raised.

References
----------
Fisher transformation. Wikipedia.
https://en.wikipedia.org/wiki/Fisher_transformation
*)

val isotonic_regression : ?sample_weight:Py.Object.t -> ?y_min:Py.Object.t -> ?y_max:Py.Object.t -> ?increasing:bool -> y:Py.Object.t -> unit -> float list
(**
Solve the isotonic regression model::

    min sum w[i] (y[i] - y_[i]) ** 2

    subject to y_min = y_[1] <= y_[2] ... <= y_[n] = y_max

where:
    - y[i] are inputs (real numbers)
    - y_[i] are fitted
    - w[i] are optional strictly positive weights (default to 1.0)

Read more in the :ref:`User Guide <isotonic>`.

Parameters
----------
y : iterable of floats
    The data.

sample_weight : iterable of floats, optional, default: None
    Weights on each point of the regression.
    If None, weight is set to 1 (equal weights).

y_min : optional, default: None
    If not None, set the lowest value of the fit to y_min.

y_max : optional, default: None
    If not None, set the highest value of the fit to y_max.

increasing : boolean, optional, default: True
    Whether to compute ``y_`` is increasing (if set to True) or decreasing
    (if set to False)

Returns
-------
y_ : list of floats
    Isotonic fit of y.

References
----------
'Active set algorithms for isotonic regression; A unifying framework'
by Michael J. Best and Nilotpal Chakravarti, section 3.
*)

val spearmanr : ?b:Py.Object.t -> ?axis:[`I of int | `None] -> ?nan_policy:[`Propagate | `Raise | `Omit] -> a:Py.Object.t -> unit -> (Py.Object.t * float)
(**
Calculate a Spearman correlation coefficient with associated p-value.

The Spearman rank-order correlation coefficient is a nonparametric measure
of the monotonicity of the relationship between two datasets. Unlike the
Pearson correlation, the Spearman correlation does not assume that both
datasets are normally distributed. Like other correlation coefficients,
this one varies between -1 and +1 with 0 implying no correlation.
Correlations of -1 or +1 imply an exact monotonic relationship. Positive
correlations imply that as x increases, so does y. Negative correlations
imply that as x increases, y decreases.

The p-value roughly indicates the probability of an uncorrelated system
producing datasets that have a Spearman correlation at least as extreme
as the one computed from these datasets. The p-values are not entirely
reliable but are probably reasonable for datasets larger than 500 or so.

Parameters
----------
a, b : 1D or 2D array_like, b is optional
    One or two 1-D or 2-D arrays containing multiple variables and
    observations. When these are 1-D, each represents a vector of
    observations of a single variable. For the behavior in the 2-D case,
    see under ``axis``, below.
    Both arrays need to have the same length in the ``axis`` dimension.
axis : int or None, optional
    If axis=0 (default), then each column represents a variable, with
    observations in the rows. If axis=1, the relationship is transposed:
    each row represents a variable, while the columns contain observations.
    If axis=None, then both arrays will be raveled.
nan_policy : {'propagate', 'raise', 'omit'}, optional
    Defines how to handle when input contains nan.
    The following options are available (default is 'propagate'):

      * 'propagate': returns nan
      * 'raise': throws an error
      * 'omit': performs the calculations ignoring nan values

Returns
-------
correlation : float or ndarray (2-D square)
    Spearman correlation matrix or correlation coefficient (if only 2
    variables are given as parameters. Correlation matrix is square with
    length equal to total number of variables (columns or rows) in ``a``
    and ``b`` combined.
pvalue : float
    The two-sided p-value for a hypothesis test whose null hypothesis is
    that two sets of data are uncorrelated, has same dimension as rho.

References
----------
.. [1] Zwillinger, D. and Kokoska, S. (2000). CRC Standard
   Probability and Statistics Tables and Formulae. Chapman & Hall: New
   York. 2000.
   Section  14.7

Examples
--------
>>> from scipy import stats
>>> stats.spearmanr([1,2,3,4,5], [5,6,7,8,7])
(0.82078268166812329, 0.088587005313543798)
>>> np.random.seed(1234321)
>>> x2n = np.random.randn(100, 2)
>>> y2n = np.random.randn(100, 2)
>>> stats.spearmanr(x2n)
(0.059969996999699973, 0.55338590803773591)
>>> stats.spearmanr(x2n[:,0], x2n[:,1])
(0.059969996999699973, 0.55338590803773591)
>>> rho, pval = stats.spearmanr(x2n, y2n)
>>> rho
array([[ 1.        ,  0.05997   ,  0.18569457,  0.06258626],
       [ 0.05997   ,  1.        ,  0.110003  ,  0.02534653],
       [ 0.18569457,  0.110003  ,  1.        ,  0.03488749],
       [ 0.06258626,  0.02534653,  0.03488749,  1.        ]])
>>> pval
array([[ 0.        ,  0.55338591,  0.06435364,  0.53617935],
       [ 0.55338591,  0.        ,  0.27592895,  0.80234077],
       [ 0.06435364,  0.27592895,  0.        ,  0.73039992],
       [ 0.53617935,  0.80234077,  0.73039992,  0.        ]])
>>> rho, pval = stats.spearmanr(x2n.T, y2n.T, axis=1)
>>> rho
array([[ 1.        ,  0.05997   ,  0.18569457,  0.06258626],
       [ 0.05997   ,  1.        ,  0.110003  ,  0.02534653],
       [ 0.18569457,  0.110003  ,  1.        ,  0.03488749],
       [ 0.06258626,  0.02534653,  0.03488749,  1.        ]])
>>> stats.spearmanr(x2n, y2n, axis=None)
(0.10816770419260482, 0.1273562188027364)
>>> stats.spearmanr(x2n.ravel(), y2n.ravel())
(0.10816770419260482, 0.1273562188027364)

>>> xint = np.random.randint(10, size=(100, 2))
>>> stats.spearmanr(xint)
(0.052760927029710199, 0.60213045837062351)
*)

