(** Get an attribute of this module as a Py.Object.t. This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module CCA : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_components:int -> ?scale:bool -> ?max_iter:Py.Object.t -> ?tol:Py.Object.t -> ?copy:bool -> unit -> t
(**
CCA Canonical Correlation Analysis.

CCA inherits from PLS with mode="B" and deflation_mode="canonical".

Read more in the :ref:`User Guide <cross_decomposition>`.

Parameters
----------
n_components : int, (default 2).
    number of components to keep.

scale : boolean, (default True)
    whether to scale the data?

max_iter : an integer, (default 500)
    the maximum number of iterations of the NIPALS inner loop

tol : non-negative real, default 1e-06.
    the tolerance used in the iterative algorithm

copy : boolean
    Whether the deflation be done on a copy. Let the default value
    to True unless you don't care about side effects

Attributes
----------
x_weights_ : array, [p, n_components]
    X block weights vectors.

y_weights_ : array, [q, n_components]
    Y block weights vectors.

x_loadings_ : array, [p, n_components]
    X block loadings vectors.

y_loadings_ : array, [q, n_components]
    Y block loadings vectors.

x_scores_ : array, [n_samples, n_components]
    X scores.

y_scores_ : array, [n_samples, n_components]
    Y scores.

x_rotations_ : array, [p, n_components]
    X block to latents rotations.

y_rotations_ : array, [q, n_components]
    Y block to latents rotations.

n_iter_ : array-like
    Number of iterations of the NIPALS inner loop for each
    component.

Notes
-----
For each component k, find the weights u, v that maximizes
max corr(Xk u, Yk v), such that ``|u| = |v| = 1``

Note that it maximizes only the correlations between the scores.

The residual matrix of X (Xk+1) block is obtained by the deflation on the
current X score: x_score.

The residual matrix of Y (Yk+1) block is obtained by deflation on the
current Y score.

Examples
--------
>>> from sklearn.cross_decomposition import CCA
>>> X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]]
>>> Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
>>> cca = CCA(n_components=1)
>>> cca.fit(X, Y)
CCA(n_components=1)
>>> X_c, Y_c = cca.transform(X, Y)

References
----------

Jacob A. Wegelin. A survey of Partial Least Squares (PLS) methods, with
emphasis on the two-block case. Technical Report 371, Department of
Statistics, University of Washington, Seattle, 2000.

In french but still a reference:
Tenenhaus, M. (1998). La regression PLS: theorie et pratique. Paris:
Editions Technic.

See also
--------
PLSCanonical
PLSSVD
*)

val fit : x:Arr.t -> y:Arr.t -> t -> t
(**
Fit model to data.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples and
    n_features is the number of predictors.

Y : array-like of shape (n_samples, n_targets)
    Target vectors, where n_samples is the number of samples and
    n_targets is the number of response variables.
*)

val fit_transform : ?y:Arr.t -> x:Arr.t -> t -> Arr.t
(**
Learn and apply the dimension reduction on the train data.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples and
    n_features is the number of predictors.

y : array-like of shape (n_samples, n_targets)
    Target vectors, where n_samples is the number of samples and
    n_targets is the number of response variables.

Returns
-------
x_scores if Y is not given, (x_scores, y_scores) otherwise.
*)

val get_params : ?deep:bool -> t -> Dict.t
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

val inverse_transform : x:Arr.t -> t -> Arr.t
(**
Transform data back to its original space.

Parameters
----------
X : array-like of shape (n_samples, n_components)
    New data, where n_samples is the number of samples
    and n_components is the number of pls components.

Returns
-------
x_reconstructed : array-like of shape (n_samples, n_features)

Notes
-----
This transformation will only be exact if n_components=n_features
*)

val predict : ?copy:bool -> x:Arr.t -> t -> Arr.t
(**
Apply the dimension reduction learned on the train data.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples and
    n_features is the number of predictors.

copy : boolean, default True
    Whether to copy X and Y, or perform in-place normalization.

Notes
-----
This call requires the estimation of a p x q matrix, which may
be an issue in high dimensional space.
*)

val score : ?sample_weight:Arr.t -> x:Arr.t -> y:Arr.t -> t -> float
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

val transform : ?y:Arr.t -> ?copy:bool -> x:Arr.t -> t -> Arr.t
(**
Apply the dimension reduction learned on the train data.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples and
    n_features is the number of predictors.

Y : array-like of shape (n_samples, n_targets)
    Target vectors, where n_samples is the number of samples and
    n_targets is the number of response variables.

copy : boolean, default True
    Whether to copy X and Y, or perform in-place normalization.

Returns
-------
x_scores if Y is not given, (x_scores, y_scores) otherwise.
*)


(** Attribute x_weights_: get value or raise Not_found if None.*)
val x_weights_ : t -> Arr.t

(** Attribute x_weights_: get value as an option. *)
val x_weights_opt : t -> (Arr.t) option


(** Attribute y_weights_: get value or raise Not_found if None.*)
val y_weights_ : t -> Arr.t

(** Attribute y_weights_: get value as an option. *)
val y_weights_opt : t -> (Arr.t) option


(** Attribute x_loadings_: get value or raise Not_found if None.*)
val x_loadings_ : t -> Arr.t

(** Attribute x_loadings_: get value as an option. *)
val x_loadings_opt : t -> (Arr.t) option


(** Attribute y_loadings_: get value or raise Not_found if None.*)
val y_loadings_ : t -> Arr.t

(** Attribute y_loadings_: get value as an option. *)
val y_loadings_opt : t -> (Arr.t) option


(** Attribute x_scores_: get value or raise Not_found if None.*)
val x_scores_ : t -> Arr.t

(** Attribute x_scores_: get value as an option. *)
val x_scores_opt : t -> (Arr.t) option


(** Attribute y_scores_: get value or raise Not_found if None.*)
val y_scores_ : t -> Arr.t

(** Attribute y_scores_: get value as an option. *)
val y_scores_opt : t -> (Arr.t) option


(** Attribute x_rotations_: get value or raise Not_found if None.*)
val x_rotations_ : t -> Arr.t

(** Attribute x_rotations_: get value as an option. *)
val x_rotations_opt : t -> (Arr.t) option


(** Attribute y_rotations_: get value or raise Not_found if None.*)
val y_rotations_ : t -> Arr.t

(** Attribute y_rotations_: get value as an option. *)
val y_rotations_opt : t -> (Arr.t) option


(** Attribute n_iter_: get value or raise Not_found if None.*)
val n_iter_ : t -> Arr.t

(** Attribute n_iter_: get value as an option. *)
val n_iter_opt : t -> (Arr.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module PLSCanonical : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_components:int -> ?scale:bool -> ?algorithm:[`Nipals | `Svd] -> ?max_iter:Py.Object.t -> ?tol:Py.Object.t -> ?copy:bool -> unit -> t
(**
PLSCanonical implements the 2 blocks canonical PLS of the original Wold
algorithm [Tenenhaus 1998] p.204, referred as PLS-C2A in [Wegelin 2000].

This class inherits from PLS with mode="A" and deflation_mode="canonical",
norm_y_weights=True and algorithm="nipals", but svd should provide similar
results up to numerical errors.

Read more in the :ref:`User Guide <cross_decomposition>`.

.. versionadded:: 0.8

Parameters
----------
n_components : int, (default 2).
    Number of components to keep

scale : boolean, (default True)
    Option to scale data

algorithm : string, "nipals" or "svd"
    The algorithm used to estimate the weights. It will be called
    n_components times, i.e. once for each iteration of the outer loop.

max_iter : an integer, (default 500)
    the maximum number of iterations of the NIPALS inner loop (used
    only if algorithm="nipals")

tol : non-negative real, default 1e-06
    the tolerance used in the iterative algorithm

copy : boolean, default True
    Whether the deflation should be done on a copy. Let the default
    value to True unless you don't care about side effect

Attributes
----------
x_weights_ : array, shape = [p, n_components]
    X block weights vectors.

y_weights_ : array, shape = [q, n_components]
    Y block weights vectors.

x_loadings_ : array, shape = [p, n_components]
    X block loadings vectors.

y_loadings_ : array, shape = [q, n_components]
    Y block loadings vectors.

x_scores_ : array, shape = [n_samples, n_components]
    X scores.

y_scores_ : array, shape = [n_samples, n_components]
    Y scores.

x_rotations_ : array, shape = [p, n_components]
    X block to latents rotations.

y_rotations_ : array, shape = [q, n_components]
    Y block to latents rotations.

n_iter_ : array-like
    Number of iterations of the NIPALS inner loop for each
    component. Not useful if the algorithm provided is "svd".

Notes
-----
Matrices::

    T: x_scores_
    U: y_scores_
    W: x_weights_
    C: y_weights_
    P: x_loadings_
    Q: y_loadings__

Are computed such that::

    X = T P.T + Err and Y = U Q.T + Err
    T[:, k] = Xk W[:, k] for k in range(n_components)
    U[:, k] = Yk C[:, k] for k in range(n_components)
    x_rotations_ = W (P.T W)^(-1)
    y_rotations_ = C (Q.T C)^(-1)

where Xk and Yk are residual matrices at iteration k.

`Slides explaining PLS
<http://www.eigenvector.com/Docs/Wise_pls_properties.pdf>`_

For each component k, find weights u, v that optimize::

    max corr(Xk u, Yk v) * std(Xk u) std(Yk u), such that ``|u| = |v| = 1``

Note that it maximizes both the correlations between the scores and the
intra-block variances.

The residual matrix of X (Xk+1) block is obtained by the deflation on the
current X score: x_score.

The residual matrix of Y (Yk+1) block is obtained by deflation on the
current Y score. This performs a canonical symmetric version of the PLS
regression. But slightly different than the CCA. This is mostly used
for modeling.

This implementation provides the same results that the "plspm" package
provided in the R language (R-project), using the function plsca(X, Y).
Results are equal or collinear with the function
``pls(..., mode = "canonical")`` of the "mixOmics" package. The difference
relies in the fact that mixOmics implementation does not exactly implement
the Wold algorithm since it does not normalize y_weights to one.

Examples
--------
>>> from sklearn.cross_decomposition import PLSCanonical
>>> X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]]
>>> Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
>>> plsca = PLSCanonical(n_components=2)
>>> plsca.fit(X, Y)
PLSCanonical()
>>> X_c, Y_c = plsca.transform(X, Y)

References
----------

Jacob A. Wegelin. A survey of Partial Least Squares (PLS) methods, with
emphasis on the two-block case. Technical Report 371, Department of
Statistics, University of Washington, Seattle, 2000.

Tenenhaus, M. (1998). La regression PLS: theorie et pratique. Paris:
Editions Technic.

See also
--------
CCA
PLSSVD
*)

val fit : x:Arr.t -> y:Arr.t -> t -> t
(**
Fit model to data.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples and
    n_features is the number of predictors.

Y : array-like of shape (n_samples, n_targets)
    Target vectors, where n_samples is the number of samples and
    n_targets is the number of response variables.
*)

val fit_transform : ?y:Arr.t -> x:Arr.t -> t -> Arr.t
(**
Learn and apply the dimension reduction on the train data.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples and
    n_features is the number of predictors.

y : array-like of shape (n_samples, n_targets)
    Target vectors, where n_samples is the number of samples and
    n_targets is the number of response variables.

Returns
-------
x_scores if Y is not given, (x_scores, y_scores) otherwise.
*)

val get_params : ?deep:bool -> t -> Dict.t
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

val inverse_transform : x:Arr.t -> t -> Arr.t
(**
Transform data back to its original space.

Parameters
----------
X : array-like of shape (n_samples, n_components)
    New data, where n_samples is the number of samples
    and n_components is the number of pls components.

Returns
-------
x_reconstructed : array-like of shape (n_samples, n_features)

Notes
-----
This transformation will only be exact if n_components=n_features
*)

val predict : ?copy:bool -> x:Arr.t -> t -> Arr.t
(**
Apply the dimension reduction learned on the train data.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples and
    n_features is the number of predictors.

copy : boolean, default True
    Whether to copy X and Y, or perform in-place normalization.

Notes
-----
This call requires the estimation of a p x q matrix, which may
be an issue in high dimensional space.
*)

val score : ?sample_weight:Arr.t -> x:Arr.t -> y:Arr.t -> t -> float
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

val transform : ?y:Arr.t -> ?copy:bool -> x:Arr.t -> t -> Arr.t
(**
Apply the dimension reduction learned on the train data.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples and
    n_features is the number of predictors.

Y : array-like of shape (n_samples, n_targets)
    Target vectors, where n_samples is the number of samples and
    n_targets is the number of response variables.

copy : boolean, default True
    Whether to copy X and Y, or perform in-place normalization.

Returns
-------
x_scores if Y is not given, (x_scores, y_scores) otherwise.
*)


(** Attribute x_weights_: get value or raise Not_found if None.*)
val x_weights_ : t -> Arr.t

(** Attribute x_weights_: get value as an option. *)
val x_weights_opt : t -> (Arr.t) option


(** Attribute y_weights_: get value or raise Not_found if None.*)
val y_weights_ : t -> Arr.t

(** Attribute y_weights_: get value as an option. *)
val y_weights_opt : t -> (Arr.t) option


(** Attribute x_loadings_: get value or raise Not_found if None.*)
val x_loadings_ : t -> Arr.t

(** Attribute x_loadings_: get value as an option. *)
val x_loadings_opt : t -> (Arr.t) option


(** Attribute y_loadings_: get value or raise Not_found if None.*)
val y_loadings_ : t -> Arr.t

(** Attribute y_loadings_: get value as an option. *)
val y_loadings_opt : t -> (Arr.t) option


(** Attribute x_scores_: get value or raise Not_found if None.*)
val x_scores_ : t -> Arr.t

(** Attribute x_scores_: get value as an option. *)
val x_scores_opt : t -> (Arr.t) option


(** Attribute y_scores_: get value or raise Not_found if None.*)
val y_scores_ : t -> Arr.t

(** Attribute y_scores_: get value as an option. *)
val y_scores_opt : t -> (Arr.t) option


(** Attribute x_rotations_: get value or raise Not_found if None.*)
val x_rotations_ : t -> Arr.t

(** Attribute x_rotations_: get value as an option. *)
val x_rotations_opt : t -> (Arr.t) option


(** Attribute y_rotations_: get value or raise Not_found if None.*)
val y_rotations_ : t -> Arr.t

(** Attribute y_rotations_: get value as an option. *)
val y_rotations_opt : t -> (Arr.t) option


(** Attribute n_iter_: get value or raise Not_found if None.*)
val n_iter_ : t -> Arr.t

(** Attribute n_iter_: get value as an option. *)
val n_iter_opt : t -> (Arr.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module PLSRegression : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_components:int -> ?scale:bool -> ?max_iter:Py.Object.t -> ?tol:Py.Object.t -> ?copy:bool -> unit -> t
(**
PLS regression

PLSRegression implements the PLS 2 blocks regression known as PLS2 or PLS1
in case of one dimensional response.
This class inherits from _PLS with mode="A", deflation_mode="regression",
norm_y_weights=False and algorithm="nipals".

Read more in the :ref:`User Guide <cross_decomposition>`.

.. versionadded:: 0.8

Parameters
----------
n_components : int, (default 2)
    Number of components to keep.

scale : boolean, (default True)
    whether to scale the data

max_iter : an integer, (default 500)
    the maximum number of iterations of the NIPALS inner loop (used
    only if algorithm="nipals")

tol : non-negative real
    Tolerance used in the iterative algorithm default 1e-06.

copy : boolean, default True
    Whether the deflation should be done on a copy. Let the default
    value to True unless you don't care about side effect

Attributes
----------
x_weights_ : array, [p, n_components]
    X block weights vectors.

y_weights_ : array, [q, n_components]
    Y block weights vectors.

x_loadings_ : array, [p, n_components]
    X block loadings vectors.

y_loadings_ : array, [q, n_components]
    Y block loadings vectors.

x_scores_ : array, [n_samples, n_components]
    X scores.

y_scores_ : array, [n_samples, n_components]
    Y scores.

x_rotations_ : array, [p, n_components]
    X block to latents rotations.

y_rotations_ : array, [q, n_components]
    Y block to latents rotations.

coef_ : array, [p, q]
    The coefficients of the linear model: ``Y = X coef_ + Err``

n_iter_ : array-like
    Number of iterations of the NIPALS inner loop for each
    component.

Notes
-----
Matrices::

    T: x_scores_
    U: y_scores_
    W: x_weights_
    C: y_weights_
    P: x_loadings_
    Q: y_loadings_

Are computed such that::

    X = T P.T + Err and Y = U Q.T + Err
    T[:, k] = Xk W[:, k] for k in range(n_components)
    U[:, k] = Yk C[:, k] for k in range(n_components)
    x_rotations_ = W (P.T W)^(-1)
    y_rotations_ = C (Q.T C)^(-1)

where Xk and Yk are residual matrices at iteration k.

`Slides explaining
PLS <http://www.eigenvector.com/Docs/Wise_pls_properties.pdf>`_


For each component k, find weights u, v that optimizes:
``max corr(Xk u, Yk v) * std(Xk u) std(Yk u)``, such that ``|u| = 1``

Note that it maximizes both the correlations between the scores and the
intra-block variances.

The residual matrix of X (Xk+1) block is obtained by the deflation on
the current X score: x_score.

The residual matrix of Y (Yk+1) block is obtained by deflation on the
current X score. This performs the PLS regression known as PLS2. This
mode is prediction oriented.

This implementation provides the same results that 3 PLS packages
provided in the R language (R-project):

    - "mixOmics" with function pls(X, Y, mode = "regression")
    - "plspm " with function plsreg2(X, Y)
    - "pls" with function oscorespls.fit(X, Y)

Examples
--------
>>> from sklearn.cross_decomposition import PLSRegression
>>> X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]]
>>> Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
>>> pls2 = PLSRegression(n_components=2)
>>> pls2.fit(X, Y)
PLSRegression()
>>> Y_pred = pls2.predict(X)

References
----------

Jacob A. Wegelin. A survey of Partial Least Squares (PLS) methods, with
emphasis on the two-block case. Technical Report 371, Department of
Statistics, University of Washington, Seattle, 2000.

In french but still a reference:
Tenenhaus, M. (1998). La regression PLS: theorie et pratique. Paris:
Editions Technic.
*)

val fit : x:Arr.t -> y:Arr.t -> t -> t
(**
Fit model to data.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples and
    n_features is the number of predictors.

Y : array-like of shape (n_samples, n_targets)
    Target vectors, where n_samples is the number of samples and
    n_targets is the number of response variables.
*)

val fit_transform : ?y:Arr.t -> x:Arr.t -> t -> Arr.t
(**
Learn and apply the dimension reduction on the train data.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples and
    n_features is the number of predictors.

y : array-like of shape (n_samples, n_targets)
    Target vectors, where n_samples is the number of samples and
    n_targets is the number of response variables.

Returns
-------
x_scores if Y is not given, (x_scores, y_scores) otherwise.
*)

val get_params : ?deep:bool -> t -> Dict.t
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

val inverse_transform : x:Arr.t -> t -> Arr.t
(**
Transform data back to its original space.

Parameters
----------
X : array-like of shape (n_samples, n_components)
    New data, where n_samples is the number of samples
    and n_components is the number of pls components.

Returns
-------
x_reconstructed : array-like of shape (n_samples, n_features)

Notes
-----
This transformation will only be exact if n_components=n_features
*)

val predict : ?copy:bool -> x:Arr.t -> t -> Arr.t
(**
Apply the dimension reduction learned on the train data.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples and
    n_features is the number of predictors.

copy : boolean, default True
    Whether to copy X and Y, or perform in-place normalization.

Notes
-----
This call requires the estimation of a p x q matrix, which may
be an issue in high dimensional space.
*)

val score : ?sample_weight:Arr.t -> x:Arr.t -> y:Arr.t -> t -> float
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

val transform : ?y:Arr.t -> ?copy:bool -> x:Arr.t -> t -> Arr.t
(**
Apply the dimension reduction learned on the train data.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples and
    n_features is the number of predictors.

Y : array-like of shape (n_samples, n_targets)
    Target vectors, where n_samples is the number of samples and
    n_targets is the number of response variables.

copy : boolean, default True
    Whether to copy X and Y, or perform in-place normalization.

Returns
-------
x_scores if Y is not given, (x_scores, y_scores) otherwise.
*)


(** Attribute x_weights_: get value or raise Not_found if None.*)
val x_weights_ : t -> Arr.t

(** Attribute x_weights_: get value as an option. *)
val x_weights_opt : t -> (Arr.t) option


(** Attribute y_weights_: get value or raise Not_found if None.*)
val y_weights_ : t -> Arr.t

(** Attribute y_weights_: get value as an option. *)
val y_weights_opt : t -> (Arr.t) option


(** Attribute x_loadings_: get value or raise Not_found if None.*)
val x_loadings_ : t -> Arr.t

(** Attribute x_loadings_: get value as an option. *)
val x_loadings_opt : t -> (Arr.t) option


(** Attribute y_loadings_: get value or raise Not_found if None.*)
val y_loadings_ : t -> Arr.t

(** Attribute y_loadings_: get value as an option. *)
val y_loadings_opt : t -> (Arr.t) option


(** Attribute x_scores_: get value or raise Not_found if None.*)
val x_scores_ : t -> Arr.t

(** Attribute x_scores_: get value as an option. *)
val x_scores_opt : t -> (Arr.t) option


(** Attribute y_scores_: get value or raise Not_found if None.*)
val y_scores_ : t -> Arr.t

(** Attribute y_scores_: get value as an option. *)
val y_scores_opt : t -> (Arr.t) option


(** Attribute x_rotations_: get value or raise Not_found if None.*)
val x_rotations_ : t -> Arr.t

(** Attribute x_rotations_: get value as an option. *)
val x_rotations_opt : t -> (Arr.t) option


(** Attribute y_rotations_: get value or raise Not_found if None.*)
val y_rotations_ : t -> Arr.t

(** Attribute y_rotations_: get value as an option. *)
val y_rotations_opt : t -> (Arr.t) option


(** Attribute coef_: get value or raise Not_found if None.*)
val coef_ : t -> Arr.t

(** Attribute coef_: get value as an option. *)
val coef_opt : t -> (Arr.t) option


(** Attribute n_iter_: get value or raise Not_found if None.*)
val n_iter_ : t -> Arr.t

(** Attribute n_iter_: get value as an option. *)
val n_iter_opt : t -> (Arr.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module PLSSVD : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_components:int -> ?scale:bool -> ?copy:bool -> unit -> t
(**
Partial Least Square SVD

Simply perform a svd on the crosscovariance matrix: X'Y
There are no iterative deflation here.

Read more in the :ref:`User Guide <cross_decomposition>`.

.. versionadded:: 0.8

Parameters
----------
n_components : int, default 2
    Number of components to keep.

scale : boolean, default True
    Whether to scale X and Y.

copy : boolean, default True
    Whether to copy X and Y, or perform in-place computations.

Attributes
----------
x_weights_ : array, [p, n_components]
    X block weights vectors.

y_weights_ : array, [q, n_components]
    Y block weights vectors.

x_scores_ : array, [n_samples, n_components]
    X scores.

y_scores_ : array, [n_samples, n_components]
    Y scores.

Examples
--------
>>> import numpy as np
>>> from sklearn.cross_decomposition import PLSSVD
>>> X = np.array([[0., 0., 1.],
...     [1.,0.,0.],
...     [2.,2.,2.],
...     [2.,5.,4.]])
>>> Y = np.array([[0.1, -0.2],
...     [0.9, 1.1],
...     [6.2, 5.9],
...     [11.9, 12.3]])
>>> plsca = PLSSVD(n_components=2)
>>> plsca.fit(X, Y)
PLSSVD()
>>> X_c, Y_c = plsca.transform(X, Y)
>>> X_c.shape, Y_c.shape
((4, 2), (4, 2))

See also
--------
PLSCanonical
CCA
*)

val fit : x:Arr.t -> y:Arr.t -> t -> t
(**
Fit model to data.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples and
    n_features is the number of predictors.

Y : array-like of shape (n_samples, n_targets)
    Target vectors, where n_samples is the number of samples and
    n_targets is the number of response variables.
*)

val fit_transform : ?y:Arr.t -> x:Arr.t -> t -> Arr.t
(**
Learn and apply the dimension reduction on the train data.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples and
    n_features is the number of predictors.

y : array-like of shape (n_samples, n_targets)
    Target vectors, where n_samples is the number of samples and
    n_targets is the number of response variables.

Returns
-------
x_scores if Y is not given, (x_scores, y_scores) otherwise.
*)

val get_params : ?deep:bool -> t -> Dict.t
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

val transform : ?y:Arr.t -> x:Arr.t -> t -> Arr.t
(**
Apply the dimension reduction learned on the train data.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples and
    n_features is the number of predictors.

Y : array-like of shape (n_samples, n_targets)
    Target vectors, where n_samples is the number of samples and
    n_targets is the number of response variables.
*)


(** Attribute x_weights_: get value or raise Not_found if None.*)
val x_weights_ : t -> Arr.t

(** Attribute x_weights_: get value as an option. *)
val x_weights_opt : t -> (Arr.t) option


(** Attribute y_weights_: get value or raise Not_found if None.*)
val y_weights_ : t -> Arr.t

(** Attribute y_weights_: get value as an option. *)
val y_weights_opt : t -> (Arr.t) option


(** Attribute x_scores_: get value or raise Not_found if None.*)
val x_scores_ : t -> Arr.t

(** Attribute x_scores_: get value as an option. *)
val x_scores_opt : t -> (Arr.t) option


(** Attribute y_scores_: get value or raise Not_found if None.*)
val y_scores_ : t -> Arr.t

(** Attribute y_scores_: get value as an option. *)
val y_scores_opt : t -> (Arr.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

