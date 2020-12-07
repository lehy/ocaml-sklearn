(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module LinearSVC : sig
type tag = [`LinearSVC]
type t = [`BaseEstimator | `ClassifierMixin | `LinearClassifierMixin | `LinearSVC | `Object | `SparseCoefMixin] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_estimator : t -> [`BaseEstimator] Obj.t
val as_classifier : t -> [`ClassifierMixin] Obj.t
val as_linear_classifier : t -> [`LinearClassifierMixin] Obj.t
val as_sparse_coef : t -> [`SparseCoefMixin] Obj.t
val create : ?penalty:[`L1 | `L2] -> ?loss:[`Hinge | `Squared_hinge] -> ?dual:bool -> ?tol:float -> ?c:float -> ?multi_class:[`Ovr | `Crammer_singer] -> ?fit_intercept:bool -> ?intercept_scaling:float -> ?class_weight:[`Balanced | `DictIntToFloat of (int * float) list] -> ?verbose:int -> ?random_state:int -> ?max_iter:int -> unit -> t
(**
Linear Support Vector Classification.

Similar to SVC with parameter kernel='linear', but implemented in terms of
liblinear rather than libsvm, so it has more flexibility in the choice of
penalties and loss functions and should scale better to large numbers of
samples.

This class supports both dense and sparse input and the multiclass support
is handled according to a one-vs-the-rest scheme.

Read more in the :ref:`User Guide <svm_classification>`.

Parameters
----------
penalty : {'l1', 'l2'}, default='l2'
    Specifies the norm used in the penalization. The 'l2'
    penalty is the standard used in SVC. The 'l1' leads to ``coef_``
    vectors that are sparse.

loss : {'hinge', 'squared_hinge'}, default='squared_hinge'
    Specifies the loss function. 'hinge' is the standard SVM loss
    (used e.g. by the SVC class) while 'squared_hinge' is the
    square of the hinge loss.

dual : bool, default=True
    Select the algorithm to either solve the dual or primal
    optimization problem. Prefer dual=False when n_samples > n_features.

tol : float, default=1e-4
    Tolerance for stopping criteria.

C : float, default=1.0
    Regularization parameter. The strength of the regularization is
    inversely proportional to C. Must be strictly positive.

multi_class : {'ovr', 'crammer_singer'}, default='ovr'
    Determines the multi-class strategy if `y` contains more than
    two classes.
    ``'ovr'`` trains n_classes one-vs-rest classifiers, while
    ``'crammer_singer'`` optimizes a joint objective over all classes.
    While `crammer_singer` is interesting from a theoretical perspective
    as it is consistent, it is seldom used in practice as it rarely leads
    to better accuracy and is more expensive to compute.
    If ``'crammer_singer'`` is chosen, the options loss, penalty and dual
    will be ignored.

fit_intercept : bool, default=True
    Whether to calculate the intercept for this model. If set
    to false, no intercept will be used in calculations
    (i.e. data is expected to be already centered).

intercept_scaling : float, default=1
    When self.fit_intercept is True, instance vector x becomes
    ``[x, self.intercept_scaling]``,
    i.e. a 'synthetic' feature with constant value equals to
    intercept_scaling is appended to the instance vector.
    The intercept becomes intercept_scaling * synthetic feature weight
    Note! the synthetic feature weight is subject to l1/l2 regularization
    as all other features.
    To lessen the effect of regularization on synthetic feature weight
    (and therefore on the intercept) intercept_scaling has to be increased.

class_weight : dict or 'balanced', default=None
    Set the parameter C of class i to ``class_weight[i]*C`` for
    SVC. If not given, all classes are supposed to have
    weight one.
    The 'balanced' mode uses the values of y to automatically adjust
    weights inversely proportional to class frequencies in the input data
    as ``n_samples / (n_classes * np.bincount(y))``.

verbose : int, default=0
    Enable verbose output. Note that this setting takes advantage of a
    per-process runtime setting in liblinear that, if enabled, may not work
    properly in a multithreaded context.

random_state : int or RandomState instance, default=None
    Controls the pseudo random number generation for shuffling the data for
    the dual coordinate descent (if ``dual=True``). When ``dual=False`` the
    underlying implementation of :class:`LinearSVC` is not random and
    ``random_state`` has no effect on the results.
    Pass an int for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

max_iter : int, default=1000
    The maximum number of iterations to be run.

Attributes
----------
coef_ : ndarray of shape (1, n_features) if n_classes == 2             else (n_classes, n_features)
    Weights assigned to the features (coefficients in the primal
    problem). This is only available in the case of a linear kernel.

    ``coef_`` is a readonly property derived from ``raw_coef_`` that
    follows the internal memory layout of liblinear.

intercept_ : ndarray of shape (1,) if n_classes == 2 else (n_classes,)
    Constants in decision function.

classes_ : ndarray of shape (n_classes,)
    The unique classes labels.

n_iter_ : int
    Maximum number of iterations run across all classes.

See Also
--------
SVC
    Implementation of Support Vector Machine classifier using libsvm:
    the kernel can be non-linear but its SMO algorithm does not
    scale to large number of samples as LinearSVC does.

    Furthermore SVC multi-class mode is implemented using one
    vs one scheme while LinearSVC uses one vs the rest. It is
    possible to implement one vs the rest with SVC by using the
    :class:`sklearn.multiclass.OneVsRestClassifier` wrapper.

    Finally SVC can fit dense data without memory copy if the input
    is C-contiguous. Sparse data will still incur memory copy though.

sklearn.linear_model.SGDClassifier
    SGDClassifier can optimize the same cost function as LinearSVC
    by adjusting the penalty and loss parameters. In addition it requires
    less memory, allows incremental (online) learning, and implements
    various loss functions and regularization regimes.

Notes
-----
The underlying C implementation uses a random number generator to
select features when fitting the model. It is thus not uncommon
to have slightly different results for the same input data. If
that happens, try with a smaller ``tol`` parameter.

The underlying implementation, liblinear, uses a sparse internal
representation for the data that will incur a memory copy.

Predict output may not match that of standalone liblinear in certain
cases. See :ref:`differences from liblinear <liblinear_differences>`
in the narrative documentation.

References
----------
`LIBLINEAR: A Library for Large Linear Classification
<https://www.csie.ntu.edu.tw/~cjlin/liblinear/>`__

Examples
--------
>>> from sklearn.svm import LinearSVC
>>> from sklearn.pipeline import make_pipeline
>>> from sklearn.preprocessing import StandardScaler
>>> from sklearn.datasets import make_classification
>>> X, y = make_classification(n_features=4, random_state=0)
>>> clf = make_pipeline(StandardScaler(),
...                     LinearSVC(random_state=0, tol=1e-5))
>>> clf.fit(X, y)
Pipeline(steps=[('standardscaler', StandardScaler()),
                ('linearsvc', LinearSVC(random_state=0, tol=1e-05))])

>>> print(clf.named_steps['linearsvc'].coef_)
[[0.141...   0.526... 0.679... 0.493...]]

>>> print(clf.named_steps['linearsvc'].intercept_)
[0.1693...]
>>> print(clf.predict([[0, 0, 0, 0]]))
[1]
*)

val decision_function : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Predict confidence scores for samples.

The confidence score for a sample is the signed distance of that
sample to the hyperplane.

Parameters
----------
X : array_like or sparse matrix, shape (n_samples, n_features)
    Samples.

Returns
-------
array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
    Confidence scores per (sample, class) combination. In the binary
    case, confidence score for self.classes_[1] where >0 means this
    class would be predicted.
*)

val densify : [> tag] Obj.t -> t
(**
Convert coefficient matrix to dense array format.

Converts the ``coef_`` member (back) to a numpy.ndarray. This is the
default format of ``coef_`` and is required for fitting, so calling
this method is only required on models that have previously been
sparsified; otherwise, it is a no-op.

Returns
-------
self
    Fitted estimator.
*)

val fit : ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fit the model according to the given training data.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Training vector, where n_samples in the number of samples and
    n_features is the number of features.

y : array-like of shape (n_samples,)
    Target vector relative to X.

sample_weight : array-like of shape (n_samples,), default=None
    Array of weights that are assigned to individual
    samples. If not provided,
    then each sample is given unit weight.

    .. versionadded:: 0.18

Returns
-------
self : object
    An instance of the estimator.
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

val predict : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Predict class labels for samples in X.

Parameters
----------
X : array_like or sparse matrix, shape (n_samples, n_features)
    Samples.

Returns
-------
C : array, shape [n_samples]
    Predicted class label per sample.
*)

val score : ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> float
(**
Return the mean accuracy on the given test data and labels.

In multi-label classification, this is the subset accuracy
which is a harsh metric since you require for each sample that
each label set be correctly predicted.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Test samples.

y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    True labels for X.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

Returns
-------
score : float
    Mean accuracy of self.predict(X) wrt. y.
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

val sparsify : [> tag] Obj.t -> t
(**
Convert coefficient matrix to sparse format.

Converts the ``coef_`` member to a scipy.sparse matrix, which for
L1-regularized models can be much more memory- and storage-efficient
than the usual numpy.ndarray representation.

The ``intercept_`` member is not converted.

Returns
-------
self
    Fitted estimator.

Notes
-----
For non-sparse models, i.e. when there are not many zeros in ``coef_``,
this may actually *increase* memory usage, so use this method with
care. A rule of thumb is that the number of zero elements, which can
be computed with ``(coef_ == 0).sum()``, must be more than 50% for this
to provide significant benefits.

After calling this method, further fitting with the partial_fit
method (if any) will not work until you call densify.
*)


(** Attribute coef_: get value or raise Not_found if None.*)
val coef_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute coef_: get value as an option. *)
val coef_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute intercept_: get value or raise Not_found if None.*)
val intercept_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute intercept_: get value as an option. *)
val intercept_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute classes_: get value or raise Not_found if None.*)
val classes_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute classes_: get value as an option. *)
val classes_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute n_iter_: get value or raise Not_found if None.*)
val n_iter_ : t -> int

(** Attribute n_iter_: get value as an option. *)
val n_iter_opt : t -> (int) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module LinearSVR : sig
type tag = [`LinearSVR]
type t = [`BaseEstimator | `LinearSVR | `Object | `RegressorMixin] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_estimator : t -> [`BaseEstimator] Obj.t
val as_regressor : t -> [`RegressorMixin] Obj.t
val create : ?epsilon:float -> ?tol:float -> ?c:float -> ?loss:[`Epsilon_insensitive | `Squared_epsilon_insensitive] -> ?fit_intercept:bool -> ?intercept_scaling:float -> ?dual:bool -> ?verbose:int -> ?random_state:int -> ?max_iter:int -> unit -> t
(**
Linear Support Vector Regression.

Similar to SVR with parameter kernel='linear', but implemented in terms of
liblinear rather than libsvm, so it has more flexibility in the choice of
penalties and loss functions and should scale better to large numbers of
samples.

This class supports both dense and sparse input.

Read more in the :ref:`User Guide <svm_regression>`.

.. versionadded:: 0.16

Parameters
----------
epsilon : float, default=0.0
    Epsilon parameter in the epsilon-insensitive loss function. Note
    that the value of this parameter depends on the scale of the target
    variable y. If unsure, set ``epsilon=0``.

tol : float, default=1e-4
    Tolerance for stopping criteria.

C : float, default=1.0
    Regularization parameter. The strength of the regularization is
    inversely proportional to C. Must be strictly positive.

loss : {'epsilon_insensitive', 'squared_epsilon_insensitive'},             default='epsilon_insensitive'
    Specifies the loss function. The epsilon-insensitive loss
    (standard SVR) is the L1 loss, while the squared epsilon-insensitive
    loss ('squared_epsilon_insensitive') is the L2 loss.

fit_intercept : bool, default=True
    Whether to calculate the intercept for this model. If set
    to false, no intercept will be used in calculations
    (i.e. data is expected to be already centered).

intercept_scaling : float, default=1.
    When self.fit_intercept is True, instance vector x becomes
    [x, self.intercept_scaling],
    i.e. a 'synthetic' feature with constant value equals to
    intercept_scaling is appended to the instance vector.
    The intercept becomes intercept_scaling * synthetic feature weight
    Note! the synthetic feature weight is subject to l1/l2 regularization
    as all other features.
    To lessen the effect of regularization on synthetic feature weight
    (and therefore on the intercept) intercept_scaling has to be increased.

dual : bool, default=True
    Select the algorithm to either solve the dual or primal
    optimization problem. Prefer dual=False when n_samples > n_features.

verbose : int, default=0
    Enable verbose output. Note that this setting takes advantage of a
    per-process runtime setting in liblinear that, if enabled, may not work
    properly in a multithreaded context.

random_state : int or RandomState instance, default=None
    Controls the pseudo random number generation for shuffling the data.
    Pass an int for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

max_iter : int, default=1000
    The maximum number of iterations to be run.

Attributes
----------
coef_ : ndarray of shape (n_features) if n_classes == 2             else (n_classes, n_features)
    Weights assigned to the features (coefficients in the primal
    problem). This is only available in the case of a linear kernel.

    `coef_` is a readonly property derived from `raw_coef_` that
    follows the internal memory layout of liblinear.

intercept_ : ndarray of shape (1) if n_classes == 2 else (n_classes)
    Constants in decision function.

n_iter_ : int
    Maximum number of iterations run across all classes.

Examples
--------
>>> from sklearn.svm import LinearSVR
>>> from sklearn.pipeline import make_pipeline
>>> from sklearn.preprocessing import StandardScaler
>>> from sklearn.datasets import make_regression
>>> X, y = make_regression(n_features=4, random_state=0)
>>> regr = make_pipeline(StandardScaler(),
...                      LinearSVR(random_state=0, tol=1e-5))
>>> regr.fit(X, y)
Pipeline(steps=[('standardscaler', StandardScaler()),
                ('linearsvr', LinearSVR(random_state=0, tol=1e-05))])

>>> print(regr.named_steps['linearsvr'].coef_)
[18.582... 27.023... 44.357... 64.522...]
>>> print(regr.named_steps['linearsvr'].intercept_)
[-4...]
>>> print(regr.predict([[0, 0, 0, 0]]))
[-2.384...]


See also
--------
LinearSVC
    Implementation of Support Vector Machine classifier using the
    same library as this class (liblinear).

SVR
    Implementation of Support Vector Machine regression using libsvm:
    the kernel can be non-linear but its SMO algorithm does not
    scale to large number of samples as LinearSVC does.

sklearn.linear_model.SGDRegressor
    SGDRegressor can optimize the same cost function as LinearSVR
    by adjusting the penalty and loss parameters. In addition it requires
    less memory, allows incremental (online) learning, and implements
    various loss functions and regularization regimes.
*)

val fit : ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fit the model according to the given training data.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Training vector, where n_samples in the number of samples and
    n_features is the number of features.

y : array-like of shape (n_samples,)
    Target vector relative to X

sample_weight : array-like of shape (n_samples,), default=None
    Array of weights that are assigned to individual
    samples. If not provided,
    then each sample is given unit weight.

    .. versionadded:: 0.18

Returns
-------
self : object
    An instance of the estimator.
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

val predict : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Predict using the linear model.

Parameters
----------
X : array_like or sparse matrix, shape (n_samples, n_features)
    Samples.

Returns
-------
C : array, shape (n_samples,)
    Returns predicted values.
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
The R2 score used when calling ``score`` on a regressor uses
``multioutput='uniform_average'`` from version 0.23 to keep consistent
with default value of :func:`~sklearn.metrics.r2_score`.
This influences the ``score`` method of all the multioutput
regressors (except for
:class:`~sklearn.multioutput.MultiOutputRegressor`).
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


(** Attribute coef_: get value or raise Not_found if None.*)
val coef_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute coef_: get value as an option. *)
val coef_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute intercept_: get value or raise Not_found if None.*)
val intercept_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute intercept_: get value as an option. *)
val intercept_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute n_iter_: get value or raise Not_found if None.*)
val n_iter_ : t -> int

(** Attribute n_iter_: get value as an option. *)
val n_iter_opt : t -> (int) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module NuSVC : sig
type tag = [`NuSVC]
type t = [`BaseEstimator | `BaseLibSVM | `BaseSVC | `ClassifierMixin | `NuSVC | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_classifier : t -> [`ClassifierMixin] Obj.t
val as_estimator : t -> [`BaseEstimator] Obj.t
val as_lib_svm : t -> [`BaseLibSVM] Obj.t
val as_svc : t -> [`BaseSVC] Obj.t
val create : ?nu:float -> ?kernel:[`Linear | `Poly | `Rbf | `Sigmoid | `Precomputed] -> ?degree:int -> ?gamma:[`Scale | `Auto | `F of float] -> ?coef0:float -> ?shrinking:bool -> ?probability:bool -> ?tol:float -> ?cache_size:float -> ?class_weight:[`Balanced | `DictIntToFloat of (int * float) list] -> ?verbose:int -> ?max_iter:int -> ?decision_function_shape:[`Ovo | `Ovr] -> ?break_ties:bool -> ?random_state:int -> unit -> t
(**
Nu-Support Vector Classification.

Similar to SVC but uses a parameter to control the number of support
vectors.

The implementation is based on libsvm.

Read more in the :ref:`User Guide <svm_classification>`.

Parameters
----------
nu : float, default=0.5
    An upper bound on the fraction of margin errors (see :ref:`User Guide
    <nu_svc>`) and a lower bound of the fraction of support vectors.
    Should be in the interval (0, 1].

kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf'
     Specifies the kernel type to be used in the algorithm.
     It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or
     a callable.
     If none is given, 'rbf' will be used. If a callable is given it is
     used to precompute the kernel matrix.

degree : int, default=3
    Degree of the polynomial kernel function ('poly').
    Ignored by all other kernels.

gamma : {'scale', 'auto'} or float, default='scale'
    Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

    - if ``gamma='scale'`` (default) is passed then it uses
      1 / (n_features * X.var()) as value of gamma,
    - if 'auto', uses 1 / n_features.

    .. versionchanged:: 0.22
       The default value of ``gamma`` changed from 'auto' to 'scale'.

coef0 : float, default=0.0
    Independent term in kernel function.
    It is only significant in 'poly' and 'sigmoid'.

shrinking : bool, default=True
    Whether to use the shrinking heuristic.
    See the :ref:`User Guide <shrinking_svm>`.

probability : bool, default=False
    Whether to enable probability estimates. This must be enabled prior
    to calling `fit`, will slow down that method as it internally uses
    5-fold cross-validation, and `predict_proba` may be inconsistent with
    `predict`. Read more in the :ref:`User Guide <scores_probabilities>`.

tol : float, default=1e-3
    Tolerance for stopping criterion.

cache_size : float, default=200
    Specify the size of the kernel cache (in MB).

class_weight : {dict, 'balanced'}, default=None
    Set the parameter C of class i to class_weight[i]*C for
    SVC. If not given, all classes are supposed to have
    weight one. The 'balanced' mode uses the values of y to automatically
    adjust weights inversely proportional to class frequencies as
    ``n_samples / (n_classes * np.bincount(y))``

verbose : bool, default=False
    Enable verbose output. Note that this setting takes advantage of a
    per-process runtime setting in libsvm that, if enabled, may not work
    properly in a multithreaded context.

max_iter : int, default=-1
    Hard limit on iterations within solver, or -1 for no limit.

decision_function_shape : {'ovo', 'ovr'}, default='ovr'
    Whether to return a one-vs-rest ('ovr') decision function of shape
    (n_samples, n_classes) as all other classifiers, or the original
    one-vs-one ('ovo') decision function of libsvm which has shape
    (n_samples, n_classes * (n_classes - 1) / 2). However, one-vs-one
    ('ovo') is always used as multi-class strategy. The parameter is
    ignored for binary classification.

    .. versionchanged:: 0.19
        decision_function_shape is 'ovr' by default.

    .. versionadded:: 0.17
       *decision_function_shape='ovr'* is recommended.

    .. versionchanged:: 0.17
       Deprecated *decision_function_shape='ovo' and None*.

break_ties : bool, default=False
    If true, ``decision_function_shape='ovr'``, and number of classes > 2,
    :term:`predict` will break ties according to the confidence values of
    :term:`decision_function`; otherwise the first class among the tied
    classes is returned. Please note that breaking ties comes at a
    relatively high computational cost compared to a simple predict.

    .. versionadded:: 0.22

random_state : int or RandomState instance, default=None
    Controls the pseudo random number generation for shuffling the data for
    probability estimates. Ignored when `probability` is False.
    Pass an int for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

Attributes
----------
support_ : ndarray of shape (n_SV,)
    Indices of support vectors.

support_vectors_ : ndarray of shape (n_SV, n_features)
    Support vectors.

n_support_ : ndarray of shape (n_class), dtype=int32
    Number of support vectors for each class.

dual_coef_ : ndarray of shape (n_class-1, n_SV)
    Dual coefficients of the support vector in the decision
    function (see :ref:`sgd_mathematical_formulation`), multiplied by
    their targets.
    For multiclass, coefficient for all 1-vs-1 classifiers.
    The layout of the coefficients in the multiclass case is somewhat
    non-trivial. See the :ref:`multi-class section of the User Guide
    <svm_multi_class>` for details.

coef_ : ndarray of shape (n_class * (n_class-1) / 2, n_features)
    Weights assigned to the features (coefficients in the primal
    problem). This is only available in the case of a linear kernel.

    `coef_` is readonly property derived from `dual_coef_` and
    `support_vectors_`.

intercept_ : ndarray of shape (n_class * (n_class-1) / 2,)
    Constants in decision function.

classes_ : ndarray of shape (n_classes,)
    The unique classes labels.

fit_status_ : int
    0 if correctly fitted, 1 if the algorithm did not converge.

probA_ : ndarray of shape (n_class * (n_class-1) / 2,)
probB_ : ndarray of shape (n_class * (n_class-1) / 2,)
    If `probability=True`, it corresponds to the parameters learned in
    Platt scaling to produce probability estimates from decision values.
    If `probability=False`, it's an empty array. Platt scaling uses the
    logistic function
    ``1 / (1 + exp(decision_value * probA_ + probB_))``
    where ``probA_`` and ``probB_`` are learned from the dataset [2]_. For
    more information on the multiclass case and training procedure see
    section 8 of [1]_.

class_weight_ : ndarray of shape (n_class,)
    Multipliers of parameter C of each class.
    Computed based on the ``class_weight`` parameter.

shape_fit_ : tuple of int of shape (n_dimensions_of_X,)
    Array dimensions of training vector ``X``.

Examples
--------
>>> import numpy as np
>>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
>>> y = np.array([1, 1, 2, 2])
>>> from sklearn.pipeline import make_pipeline
>>> from sklearn.preprocessing import StandardScaler
>>> from sklearn.svm import NuSVC
>>> clf = make_pipeline(StandardScaler(), NuSVC())
>>> clf.fit(X, y)
Pipeline(steps=[('standardscaler', StandardScaler()), ('nusvc', NuSVC())])
>>> print(clf.predict([[-0.8, -1]]))
[1]

See also
--------
SVC
    Support Vector Machine for classification using libsvm.

LinearSVC
    Scalable linear Support Vector Machine for classification using
    liblinear.

References
----------
.. [1] `LIBSVM: A Library for Support Vector Machines
    <http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf>`_

.. [2] `Platt, John (1999). 'Probabilistic outputs for support vector
    machines and comparison to regularizedlikelihood methods.'
    <http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.41.1639>`_
*)

val decision_function : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Evaluates the decision function for the samples in X.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
X : ndarray of shape (n_samples, n_classes * (n_classes-1) / 2)
    Returns the decision function of the sample for each class
    in the model.
    If decision_function_shape='ovr', the shape is (n_samples,
    n_classes).

Notes
-----
If decision_function_shape='ovo', the function values are proportional
to the distance of the samples X to the separating hyperplane. If the
exact distances are required, divide the function values by the norm of
the weight vector (``coef_``). See also `this question
<https://stats.stackexchange.com/questions/14876/
interpreting-distance-from-hyperplane-in-svm>`_ for further details.
If decision_function_shape='ovr', the decision function is a monotonic
transformation of ovo decision function.
*)

val fit : ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fit the SVM model according to the given training data.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)                 or (n_samples, n_samples)
    Training vectors, where n_samples is the number of samples
    and n_features is the number of features.
    For kernel='precomputed', the expected shape of X is
    (n_samples, n_samples).

y : array-like of shape (n_samples,)
    Target values (class labels in classification, real numbers in
    regression)

sample_weight : array-like of shape (n_samples,), default=None
    Per-sample weights. Rescale C per sample. Higher weights
    force the classifier to put more emphasis on these points.

Returns
-------
self : object

Notes
-----
If X and y are not C-ordered and contiguous arrays of np.float64 and
X is not a scipy.sparse.csr_matrix, X and/or y may be copied.

If X is a dense array, then the other methods will not support sparse
matrices as input.
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

val predict : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Perform classification on samples in X.

For an one-class model, +1 or -1 is returned.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features) or                 (n_samples_test, n_samples_train)
    For kernel='precomputed', the expected shape of X is
    (n_samples_test, n_samples_train).

Returns
-------
y_pred : ndarray of shape (n_samples,)
    Class labels for samples in X.
*)

val score : ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> float
(**
Return the mean accuracy on the given test data and labels.

In multi-label classification, this is the subset accuracy
which is a harsh metric since you require for each sample that
each label set be correctly predicted.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Test samples.

y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    True labels for X.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

Returns
-------
score : float
    Mean accuracy of self.predict(X) wrt. y.
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


(** Attribute support_: get value or raise Not_found if None.*)
val support_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute support_: get value as an option. *)
val support_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute support_vectors_: get value or raise Not_found if None.*)
val support_vectors_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute support_vectors_: get value as an option. *)
val support_vectors_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute n_support_: get value or raise Not_found if None.*)
val n_support_ : t -> Py.Object.t

(** Attribute n_support_: get value as an option. *)
val n_support_opt : t -> (Py.Object.t) option


(** Attribute dual_coef_: get value or raise Not_found if None.*)
val dual_coef_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute dual_coef_: get value as an option. *)
val dual_coef_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute coef_: get value or raise Not_found if None.*)
val coef_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute coef_: get value as an option. *)
val coef_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute intercept_: get value or raise Not_found if None.*)
val intercept_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute intercept_: get value as an option. *)
val intercept_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute classes_: get value or raise Not_found if None.*)
val classes_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute classes_: get value as an option. *)
val classes_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute fit_status_: get value or raise Not_found if None.*)
val fit_status_ : t -> int

(** Attribute fit_status_: get value as an option. *)
val fit_status_opt : t -> (int) option


(** Attribute probA_: get value or raise Not_found if None.*)
val probA_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute probA_: get value as an option. *)
val probA_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute class_weight_: get value or raise Not_found if None.*)
val class_weight_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute class_weight_: get value as an option. *)
val class_weight_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute shape_fit_: get value or raise Not_found if None.*)
val shape_fit_ : t -> Py.Object.t

(** Attribute shape_fit_: get value as an option. *)
val shape_fit_opt : t -> (Py.Object.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module NuSVR : sig
type tag = [`NuSVR]
type t = [`BaseEstimator | `BaseLibSVM | `NuSVR | `Object | `RegressorMixin] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_estimator : t -> [`BaseEstimator] Obj.t
val as_regressor : t -> [`RegressorMixin] Obj.t
val as_lib_svm : t -> [`BaseLibSVM] Obj.t
val create : ?nu:float -> ?c:float -> ?kernel:[`Linear | `Poly | `Rbf | `Sigmoid | `Precomputed] -> ?degree:int -> ?gamma:[`Scale | `Auto | `F of float] -> ?coef0:float -> ?shrinking:bool -> ?tol:float -> ?cache_size:float -> ?verbose:int -> ?max_iter:int -> unit -> t
(**
Nu Support Vector Regression.

Similar to NuSVC, for regression, uses a parameter nu to control
the number of support vectors. However, unlike NuSVC, where nu
replaces C, here nu replaces the parameter epsilon of epsilon-SVR.

The implementation is based on libsvm.

Read more in the :ref:`User Guide <svm_regression>`.

Parameters
----------
nu : float, default=0.5
    An upper bound on the fraction of training errors and a lower bound of
    the fraction of support vectors. Should be in the interval (0, 1].  By
    default 0.5 will be taken.

C : float, default=1.0
    Penalty parameter C of the error term.

kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf'
     Specifies the kernel type to be used in the algorithm.
     It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or
     a callable.
     If none is given, 'rbf' will be used. If a callable is given it is
     used to precompute the kernel matrix.

degree : int, default=3
    Degree of the polynomial kernel function ('poly').
    Ignored by all other kernels.

gamma : {'scale', 'auto'} or float, default='scale'
    Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

    - if ``gamma='scale'`` (default) is passed then it uses
      1 / (n_features * X.var()) as value of gamma,
    - if 'auto', uses 1 / n_features.

    .. versionchanged:: 0.22
       The default value of ``gamma`` changed from 'auto' to 'scale'.

coef0 : float, default=0.0
    Independent term in kernel function.
    It is only significant in 'poly' and 'sigmoid'.

shrinking : bool, default=True
    Whether to use the shrinking heuristic.
    See the :ref:`User Guide <shrinking_svm>`.

tol : float, default=1e-3
    Tolerance for stopping criterion.

cache_size : float, default=200
    Specify the size of the kernel cache (in MB).

verbose : bool, default=False
    Enable verbose output. Note that this setting takes advantage of a
    per-process runtime setting in libsvm that, if enabled, may not work
    properly in a multithreaded context.

max_iter : int, default=-1
    Hard limit on iterations within solver, or -1 for no limit.

Attributes
----------
support_ : ndarray of shape (n_SV,)
    Indices of support vectors.

support_vectors_ : ndarray of shape (n_SV, n_features)
    Support vectors.

dual_coef_ : ndarray of shape (1, n_SV)
    Coefficients of the support vector in the decision function.

coef_ : ndarray of shape (1, n_features)
    Weights assigned to the features (coefficients in the primal
    problem). This is only available in the case of a linear kernel.

    `coef_` is readonly property derived from `dual_coef_` and
    `support_vectors_`.

intercept_ : ndarray of shape (1,)
    Constants in decision function.

Examples
--------
>>> from sklearn.svm import NuSVR
>>> from sklearn.pipeline import make_pipeline
>>> from sklearn.preprocessing import StandardScaler
>>> import numpy as np
>>> n_samples, n_features = 10, 5
>>> np.random.seed(0)
>>> y = np.random.randn(n_samples)
>>> X = np.random.randn(n_samples, n_features)
>>> regr = make_pipeline(StandardScaler(), NuSVR(C=1.0, nu=0.1))
>>> regr.fit(X, y)
Pipeline(steps=[('standardscaler', StandardScaler()),
                ('nusvr', NuSVR(nu=0.1))])

See also
--------
NuSVC
    Support Vector Machine for classification implemented with libsvm
    with a parameter to control the number of support vectors.

SVR
    epsilon Support Vector Machine for regression implemented with libsvm.

Notes
-----
**References:**
`LIBSVM: A Library for Support Vector Machines
<http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf>`__
*)

val fit : ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fit the SVM model according to the given training data.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)                 or (n_samples, n_samples)
    Training vectors, where n_samples is the number of samples
    and n_features is the number of features.
    For kernel='precomputed', the expected shape of X is
    (n_samples, n_samples).

y : array-like of shape (n_samples,)
    Target values (class labels in classification, real numbers in
    regression)

sample_weight : array-like of shape (n_samples,), default=None
    Per-sample weights. Rescale C per sample. Higher weights
    force the classifier to put more emphasis on these points.

Returns
-------
self : object

Notes
-----
If X and y are not C-ordered and contiguous arrays of np.float64 and
X is not a scipy.sparse.csr_matrix, X and/or y may be copied.

If X is a dense array, then the other methods will not support sparse
matrices as input.
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

val predict : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Perform regression on samples in X.

For an one-class model, +1 (inlier) or -1 (outlier) is returned.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    For kernel='precomputed', the expected shape of X is
    (n_samples_test, n_samples_train).

Returns
-------
y_pred : ndarray of shape (n_samples,)
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
The R2 score used when calling ``score`` on a regressor uses
``multioutput='uniform_average'`` from version 0.23 to keep consistent
with default value of :func:`~sklearn.metrics.r2_score`.
This influences the ``score`` method of all the multioutput
regressors (except for
:class:`~sklearn.multioutput.MultiOutputRegressor`).
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


(** Attribute support_: get value or raise Not_found if None.*)
val support_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute support_: get value as an option. *)
val support_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute support_vectors_: get value or raise Not_found if None.*)
val support_vectors_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute support_vectors_: get value as an option. *)
val support_vectors_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute dual_coef_: get value or raise Not_found if None.*)
val dual_coef_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute dual_coef_: get value as an option. *)
val dual_coef_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute coef_: get value or raise Not_found if None.*)
val coef_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute coef_: get value as an option. *)
val coef_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute intercept_: get value or raise Not_found if None.*)
val intercept_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute intercept_: get value as an option. *)
val intercept_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module OneClassSVM : sig
type tag = [`OneClassSVM]
type t = [`BaseEstimator | `BaseLibSVM | `Object | `OneClassSVM | `OutlierMixin] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_estimator : t -> [`BaseEstimator] Obj.t
val as_lib_svm : t -> [`BaseLibSVM] Obj.t
val as_outlier : t -> [`OutlierMixin] Obj.t
val create : ?kernel:[`Linear | `Poly | `Rbf | `Sigmoid | `Precomputed] -> ?degree:int -> ?gamma:[`Scale | `Auto | `F of float] -> ?coef0:float -> ?tol:float -> ?nu:float -> ?shrinking:bool -> ?cache_size:float -> ?verbose:int -> ?max_iter:int -> unit -> t
(**
Unsupervised Outlier Detection.

Estimate the support of a high-dimensional distribution.

The implementation is based on libsvm.

Read more in the :ref:`User Guide <outlier_detection>`.

Parameters
----------
kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf'
     Specifies the kernel type to be used in the algorithm.
     It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or
     a callable.
     If none is given, 'rbf' will be used. If a callable is given it is
     used to precompute the kernel matrix.

degree : int, default=3
    Degree of the polynomial kernel function ('poly').
    Ignored by all other kernels.

gamma : {'scale', 'auto'} or float, default='scale'
    Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

    - if ``gamma='scale'`` (default) is passed then it uses
      1 / (n_features * X.var()) as value of gamma,
    - if 'auto', uses 1 / n_features.

    .. versionchanged:: 0.22
       The default value of ``gamma`` changed from 'auto' to 'scale'.

coef0 : float, default=0.0
    Independent term in kernel function.
    It is only significant in 'poly' and 'sigmoid'.

tol : float, default=1e-3
    Tolerance for stopping criterion.

nu : float, default=0.5
    An upper bound on the fraction of training
    errors and a lower bound of the fraction of support
    vectors. Should be in the interval (0, 1]. By default 0.5
    will be taken.

shrinking : bool, default=True
    Whether to use the shrinking heuristic.
    See the :ref:`User Guide <shrinking_svm>`.

cache_size : float, default=200
    Specify the size of the kernel cache (in MB).

verbose : bool, default=False
    Enable verbose output. Note that this setting takes advantage of a
    per-process runtime setting in libsvm that, if enabled, may not work
    properly in a multithreaded context.

max_iter : int, default=-1
    Hard limit on iterations within solver, or -1 for no limit.

Attributes
----------
support_ : ndarray of shape (n_SV,)
    Indices of support vectors.

support_vectors_ : ndarray of shape (n_SV, n_features)
    Support vectors.

dual_coef_ : ndarray of shape (1, n_SV)
    Coefficients of the support vectors in the decision function.

coef_ : ndarray of shape (1, n_features)
    Weights assigned to the features (coefficients in the primal
    problem). This is only available in the case of a linear kernel.

    `coef_` is readonly property derived from `dual_coef_` and
    `support_vectors_`

intercept_ : ndarray of shape (1,)
    Constant in the decision function.

offset_ : float
    Offset used to define the decision function from the raw scores.
    We have the relation: decision_function = score_samples - `offset_`.
    The offset is the opposite of `intercept_` and is provided for
    consistency with other outlier detection algorithms.

    .. versionadded:: 0.20

fit_status_ : int
    0 if correctly fitted, 1 otherwise (will raise warning)

Examples
--------
>>> from sklearn.svm import OneClassSVM
>>> X = [[0], [0.44], [0.45], [0.46], [1]]
>>> clf = OneClassSVM(gamma='auto').fit(X)
>>> clf.predict(X)
array([-1,  1,  1,  1, -1])
>>> clf.score_samples(X)
array([1.7798..., 2.0547..., 2.0556..., 2.0561..., 1.7332...])
*)

val decision_function : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Signed distance to the separating hyperplane.

Signed distance is positive for an inlier and negative for an outlier.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The data matrix.

Returns
-------
dec : ndarray of shape (n_samples,)
    Returns the decision function of the samples.
*)

val fit : ?y:Py.Object.t -> ?sample_weight:[>`ArrayLike] Np.Obj.t -> ?params:(string * Py.Object.t) list -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Detects the soft boundary of the set of samples X.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Set of samples, where n_samples is the number of samples and
    n_features is the number of features.

sample_weight : array-like of shape (n_samples,), default=None
    Per-sample weights. Rescale C per sample. Higher weights
    force the classifier to put more emphasis on these points.

y : Ignored
    not used, present for API consistency by convention.

Returns
-------
self : object

Notes
-----
If X is not a C-ordered contiguous array it is copied.
*)

val fit_predict : ?y:Py.Object.t -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Perform fit on X and returns labels for X.

Returns -1 for outliers and 1 for inliers.

Parameters
----------
X : {array-like, sparse matrix, dataframe} of shape             (n_samples, n_features)

y : Ignored
    Not used, present for API consistency by convention.

Returns
-------
y : ndarray of shape (n_samples,)
    1 for inliers, -1 for outliers.
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

val predict : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Perform classification on samples in X.

For a one-class model, +1 or -1 is returned.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features) or                 (n_samples_test, n_samples_train)
    For kernel='precomputed', the expected shape of X is
    (n_samples_test, n_samples_train).

Returns
-------
y_pred : ndarray of shape (n_samples,)
    Class labels for samples in X.
*)

val score_samples : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Raw scoring function of the samples.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The data matrix.

Returns
-------
score_samples : ndarray of shape (n_samples,)
    Returns the (unshifted) scoring function of the samples.
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


(** Attribute support_: get value or raise Not_found if None.*)
val support_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute support_: get value as an option. *)
val support_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute support_vectors_: get value or raise Not_found if None.*)
val support_vectors_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute support_vectors_: get value as an option. *)
val support_vectors_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute dual_coef_: get value or raise Not_found if None.*)
val dual_coef_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute dual_coef_: get value as an option. *)
val dual_coef_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute coef_: get value or raise Not_found if None.*)
val coef_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute coef_: get value as an option. *)
val coef_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute intercept_: get value or raise Not_found if None.*)
val intercept_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute intercept_: get value as an option. *)
val intercept_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute offset_: get value or raise Not_found if None.*)
val offset_ : t -> float

(** Attribute offset_: get value as an option. *)
val offset_opt : t -> (float) option


(** Attribute fit_status_: get value or raise Not_found if None.*)
val fit_status_ : t -> int

(** Attribute fit_status_: get value as an option. *)
val fit_status_opt : t -> (int) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module SVC : sig
type tag = [`SVC]
type t = [`BaseEstimator | `BaseLibSVM | `BaseSVC | `ClassifierMixin | `Object | `SVC] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_classifier : t -> [`ClassifierMixin] Obj.t
val as_estimator : t -> [`BaseEstimator] Obj.t
val as_lib_svm : t -> [`BaseLibSVM] Obj.t
val as_svc : t -> [`BaseSVC] Obj.t
val create : ?c:float -> ?kernel:[`Linear | `Poly | `Rbf | `Sigmoid | `Precomputed] -> ?degree:int -> ?gamma:[`Scale | `Auto | `F of float] -> ?coef0:float -> ?shrinking:bool -> ?probability:bool -> ?tol:float -> ?cache_size:float -> ?class_weight:[`Balanced | `DictIntToFloat of (int * float) list] -> ?verbose:int -> ?max_iter:int -> ?decision_function_shape:[`Ovo | `Ovr] -> ?break_ties:bool -> ?random_state:int -> unit -> t
(**
C-Support Vector Classification.

The implementation is based on libsvm. The fit time scales at least
quadratically with the number of samples and may be impractical
beyond tens of thousands of samples. For large datasets
consider using :class:`sklearn.svm.LinearSVC` or
:class:`sklearn.linear_model.SGDClassifier` instead, possibly after a
:class:`sklearn.kernel_approximation.Nystroem` transformer.

The multiclass support is handled according to a one-vs-one scheme.

For details on the precise mathematical formulation of the provided
kernel functions and how `gamma`, `coef0` and `degree` affect each
other, see the corresponding section in the narrative documentation:
:ref:`svm_kernels`.

Read more in the :ref:`User Guide <svm_classification>`.

Parameters
----------
C : float, default=1.0
    Regularization parameter. The strength of the regularization is
    inversely proportional to C. Must be strictly positive. The penalty
    is a squared l2 penalty.

kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf'
    Specifies the kernel type to be used in the algorithm.
    It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or
    a callable.
    If none is given, 'rbf' will be used. If a callable is given it is
    used to pre-compute the kernel matrix from data matrices; that matrix
    should be an array of shape ``(n_samples, n_samples)``.

degree : int, default=3
    Degree of the polynomial kernel function ('poly').
    Ignored by all other kernels.

gamma : {'scale', 'auto'} or float, default='scale'
    Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

    - if ``gamma='scale'`` (default) is passed then it uses
      1 / (n_features * X.var()) as value of gamma,
    - if 'auto', uses 1 / n_features.

    .. versionchanged:: 0.22
       The default value of ``gamma`` changed from 'auto' to 'scale'.

coef0 : float, default=0.0
    Independent term in kernel function.
    It is only significant in 'poly' and 'sigmoid'.

shrinking : bool, default=True
    Whether to use the shrinking heuristic.
    See the :ref:`User Guide <shrinking_svm>`.

probability : bool, default=False
    Whether to enable probability estimates. This must be enabled prior
    to calling `fit`, will slow down that method as it internally uses
    5-fold cross-validation, and `predict_proba` may be inconsistent with
    `predict`. Read more in the :ref:`User Guide <scores_probabilities>`.

tol : float, default=1e-3
    Tolerance for stopping criterion.

cache_size : float, default=200
    Specify the size of the kernel cache (in MB).

class_weight : dict or 'balanced', default=None
    Set the parameter C of class i to class_weight[i]*C for
    SVC. If not given, all classes are supposed to have
    weight one.
    The 'balanced' mode uses the values of y to automatically adjust
    weights inversely proportional to class frequencies in the input data
    as ``n_samples / (n_classes * np.bincount(y))``

verbose : bool, default=False
    Enable verbose output. Note that this setting takes advantage of a
    per-process runtime setting in libsvm that, if enabled, may not work
    properly in a multithreaded context.

max_iter : int, default=-1
    Hard limit on iterations within solver, or -1 for no limit.

decision_function_shape : {'ovo', 'ovr'}, default='ovr'
    Whether to return a one-vs-rest ('ovr') decision function of shape
    (n_samples, n_classes) as all other classifiers, or the original
    one-vs-one ('ovo') decision function of libsvm which has shape
    (n_samples, n_classes * (n_classes - 1) / 2). However, one-vs-one
    ('ovo') is always used as multi-class strategy. The parameter is
    ignored for binary classification.

    .. versionchanged:: 0.19
        decision_function_shape is 'ovr' by default.

    .. versionadded:: 0.17
       *decision_function_shape='ovr'* is recommended.

    .. versionchanged:: 0.17
       Deprecated *decision_function_shape='ovo' and None*.

break_ties : bool, default=False
    If true, ``decision_function_shape='ovr'``, and number of classes > 2,
    :term:`predict` will break ties according to the confidence values of
    :term:`decision_function`; otherwise the first class among the tied
    classes is returned. Please note that breaking ties comes at a
    relatively high computational cost compared to a simple predict.

    .. versionadded:: 0.22

random_state : int or RandomState instance, default=None
    Controls the pseudo random number generation for shuffling the data for
    probability estimates. Ignored when `probability` is False.
    Pass an int for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

Attributes
----------
support_ : ndarray of shape (n_SV,)
    Indices of support vectors.

support_vectors_ : ndarray of shape (n_SV, n_features)
    Support vectors.

n_support_ : ndarray of shape (n_class,), dtype=int32
    Number of support vectors for each class.

dual_coef_ : ndarray of shape (n_class-1, n_SV)
    Dual coefficients of the support vector in the decision
    function (see :ref:`sgd_mathematical_formulation`), multiplied by
    their targets.
    For multiclass, coefficient for all 1-vs-1 classifiers.
    The layout of the coefficients in the multiclass case is somewhat
    non-trivial. See the :ref:`multi-class section of the User Guide
    <svm_multi_class>` for details.

coef_ : ndarray of shape (n_class * (n_class-1) / 2, n_features)
    Weights assigned to the features (coefficients in the primal
    problem). This is only available in the case of a linear kernel.

    `coef_` is a readonly property derived from `dual_coef_` and
    `support_vectors_`.

intercept_ : ndarray of shape (n_class * (n_class-1) / 2,)
    Constants in decision function.

fit_status_ : int
    0 if correctly fitted, 1 otherwise (will raise warning)

classes_ : ndarray of shape (n_classes,)
    The classes labels.

probA_ : ndarray of shape (n_class * (n_class-1) / 2)
probB_ : ndarray of shape (n_class * (n_class-1) / 2)
    If `probability=True`, it corresponds to the parameters learned in
    Platt scaling to produce probability estimates from decision values.
    If `probability=False`, it's an empty array. Platt scaling uses the
    logistic function
    ``1 / (1 + exp(decision_value * probA_ + probB_))``
    where ``probA_`` and ``probB_`` are learned from the dataset [2]_. For
    more information on the multiclass case and training procedure see
    section 8 of [1]_.

class_weight_ : ndarray of shape (n_class,)
    Multipliers of parameter C for each class.
    Computed based on the ``class_weight`` parameter.

shape_fit_ : tuple of int of shape (n_dimensions_of_X,)
    Array dimensions of training vector ``X``.

Examples
--------
>>> import numpy as np
>>> from sklearn.pipeline import make_pipeline
>>> from sklearn.preprocessing import StandardScaler
>>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
>>> y = np.array([1, 1, 2, 2])
>>> from sklearn.svm import SVC
>>> clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
>>> clf.fit(X, y)
Pipeline(steps=[('standardscaler', StandardScaler()),
                ('svc', SVC(gamma='auto'))])

>>> print(clf.predict([[-0.8, -1]]))
[1]

See also
--------
SVR
    Support Vector Machine for Regression implemented using libsvm.

LinearSVC
    Scalable Linear Support Vector Machine for classification
    implemented using liblinear. Check the See also section of
    LinearSVC for more comparison element.

References
----------
.. [1] `LIBSVM: A Library for Support Vector Machines
    <http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf>`_

.. [2] `Platt, John (1999). 'Probabilistic outputs for support vector
    machines and comparison to regularizedlikelihood methods.'
    <http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.41.1639>`_
*)

val decision_function : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Evaluates the decision function for the samples in X.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
X : ndarray of shape (n_samples, n_classes * (n_classes-1) / 2)
    Returns the decision function of the sample for each class
    in the model.
    If decision_function_shape='ovr', the shape is (n_samples,
    n_classes).

Notes
-----
If decision_function_shape='ovo', the function values are proportional
to the distance of the samples X to the separating hyperplane. If the
exact distances are required, divide the function values by the norm of
the weight vector (``coef_``). See also `this question
<https://stats.stackexchange.com/questions/14876/
interpreting-distance-from-hyperplane-in-svm>`_ for further details.
If decision_function_shape='ovr', the decision function is a monotonic
transformation of ovo decision function.
*)

val fit : ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fit the SVM model according to the given training data.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)                 or (n_samples, n_samples)
    Training vectors, where n_samples is the number of samples
    and n_features is the number of features.
    For kernel='precomputed', the expected shape of X is
    (n_samples, n_samples).

y : array-like of shape (n_samples,)
    Target values (class labels in classification, real numbers in
    regression)

sample_weight : array-like of shape (n_samples,), default=None
    Per-sample weights. Rescale C per sample. Higher weights
    force the classifier to put more emphasis on these points.

Returns
-------
self : object

Notes
-----
If X and y are not C-ordered and contiguous arrays of np.float64 and
X is not a scipy.sparse.csr_matrix, X and/or y may be copied.

If X is a dense array, then the other methods will not support sparse
matrices as input.
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

val predict : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Perform classification on samples in X.

For an one-class model, +1 or -1 is returned.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features) or                 (n_samples_test, n_samples_train)
    For kernel='precomputed', the expected shape of X is
    (n_samples_test, n_samples_train).

Returns
-------
y_pred : ndarray of shape (n_samples,)
    Class labels for samples in X.
*)

val score : ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> float
(**
Return the mean accuracy on the given test data and labels.

In multi-label classification, this is the subset accuracy
which is a harsh metric since you require for each sample that
each label set be correctly predicted.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Test samples.

y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    True labels for X.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

Returns
-------
score : float
    Mean accuracy of self.predict(X) wrt. y.
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


(** Attribute support_: get value or raise Not_found if None.*)
val support_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute support_: get value as an option. *)
val support_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute support_vectors_: get value or raise Not_found if None.*)
val support_vectors_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute support_vectors_: get value as an option. *)
val support_vectors_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute n_support_: get value or raise Not_found if None.*)
val n_support_ : t -> Py.Object.t

(** Attribute n_support_: get value as an option. *)
val n_support_opt : t -> (Py.Object.t) option


(** Attribute dual_coef_: get value or raise Not_found if None.*)
val dual_coef_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute dual_coef_: get value as an option. *)
val dual_coef_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute coef_: get value or raise Not_found if None.*)
val coef_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute coef_: get value as an option. *)
val coef_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute intercept_: get value or raise Not_found if None.*)
val intercept_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute intercept_: get value as an option. *)
val intercept_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute fit_status_: get value or raise Not_found if None.*)
val fit_status_ : t -> int

(** Attribute fit_status_: get value as an option. *)
val fit_status_opt : t -> (int) option


(** Attribute classes_: get value or raise Not_found if None.*)
val classes_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute classes_: get value as an option. *)
val classes_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute probA_: get value or raise Not_found if None.*)
val probA_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute probA_: get value as an option. *)
val probA_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute class_weight_: get value or raise Not_found if None.*)
val class_weight_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute class_weight_: get value as an option. *)
val class_weight_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute shape_fit_: get value or raise Not_found if None.*)
val shape_fit_ : t -> Py.Object.t

(** Attribute shape_fit_: get value as an option. *)
val shape_fit_opt : t -> (Py.Object.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module SVR : sig
type tag = [`SVR]
type t = [`BaseEstimator | `BaseLibSVM | `Object | `RegressorMixin | `SVR] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_estimator : t -> [`BaseEstimator] Obj.t
val as_regressor : t -> [`RegressorMixin] Obj.t
val as_lib_svm : t -> [`BaseLibSVM] Obj.t
val create : ?kernel:[`Linear | `Poly | `Rbf | `Sigmoid | `Precomputed] -> ?degree:int -> ?gamma:[`Scale | `Auto | `F of float] -> ?coef0:float -> ?tol:float -> ?c:float -> ?epsilon:float -> ?shrinking:bool -> ?cache_size:float -> ?verbose:int -> ?max_iter:int -> unit -> t
(**
Epsilon-Support Vector Regression.

The free parameters in the model are C and epsilon.

The implementation is based on libsvm. The fit time complexity
is more than quadratic with the number of samples which makes it hard
to scale to datasets with more than a couple of 10000 samples. For large
datasets consider using :class:`sklearn.svm.LinearSVR` or
:class:`sklearn.linear_model.SGDRegressor` instead, possibly after a
:class:`sklearn.kernel_approximation.Nystroem` transformer.

Read more in the :ref:`User Guide <svm_regression>`.

Parameters
----------
kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf'
     Specifies the kernel type to be used in the algorithm.
     It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or
     a callable.
     If none is given, 'rbf' will be used. If a callable is given it is
     used to precompute the kernel matrix.

degree : int, default=3
    Degree of the polynomial kernel function ('poly').
    Ignored by all other kernels.

gamma : {'scale', 'auto'} or float, default='scale'
    Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

    - if ``gamma='scale'`` (default) is passed then it uses
      1 / (n_features * X.var()) as value of gamma,
    - if 'auto', uses 1 / n_features.

    .. versionchanged:: 0.22
       The default value of ``gamma`` changed from 'auto' to 'scale'.

coef0 : float, default=0.0
    Independent term in kernel function.
    It is only significant in 'poly' and 'sigmoid'.

tol : float, default=1e-3
    Tolerance for stopping criterion.

C : float, default=1.0
    Regularization parameter. The strength of the regularization is
    inversely proportional to C. Must be strictly positive.
    The penalty is a squared l2 penalty.

epsilon : float, default=0.1
     Epsilon in the epsilon-SVR model. It specifies the epsilon-tube
     within which no penalty is associated in the training loss function
     with points predicted within a distance epsilon from the actual
     value.

shrinking : bool, default=True
    Whether to use the shrinking heuristic.
    See the :ref:`User Guide <shrinking_svm>`.

cache_size : float, default=200
    Specify the size of the kernel cache (in MB).

verbose : bool, default=False
    Enable verbose output. Note that this setting takes advantage of a
    per-process runtime setting in libsvm that, if enabled, may not work
    properly in a multithreaded context.

max_iter : int, default=-1
    Hard limit on iterations within solver, or -1 for no limit.

Attributes
----------
support_ : ndarray of shape (n_SV,)
    Indices of support vectors.

support_vectors_ : ndarray of shape (n_SV, n_features)
    Support vectors.

dual_coef_ : ndarray of shape (1, n_SV)
    Coefficients of the support vector in the decision function.

coef_ : ndarray of shape (1, n_features)
    Weights assigned to the features (coefficients in the primal
    problem). This is only available in the case of a linear kernel.

    `coef_` is readonly property derived from `dual_coef_` and
    `support_vectors_`.

fit_status_ : int
    0 if correctly fitted, 1 otherwise (will raise warning)

intercept_ : ndarray of shape (1,)
    Constants in decision function.

Examples
--------
>>> from sklearn.svm import SVR
>>> from sklearn.pipeline import make_pipeline
>>> from sklearn.preprocessing import StandardScaler
>>> import numpy as np
>>> n_samples, n_features = 10, 5
>>> rng = np.random.RandomState(0)
>>> y = rng.randn(n_samples)
>>> X = rng.randn(n_samples, n_features)
>>> regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
>>> regr.fit(X, y)
Pipeline(steps=[('standardscaler', StandardScaler()),
                ('svr', SVR(epsilon=0.2))])


See also
--------
NuSVR
    Support Vector Machine for regression implemented using libsvm
    using a parameter to control the number of support vectors.

LinearSVR
    Scalable Linear Support Vector Machine for regression
    implemented using liblinear.

Notes
-----
**References:**
`LIBSVM: A Library for Support Vector Machines
<http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf>`__
*)

val fit : ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fit the SVM model according to the given training data.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)                 or (n_samples, n_samples)
    Training vectors, where n_samples is the number of samples
    and n_features is the number of features.
    For kernel='precomputed', the expected shape of X is
    (n_samples, n_samples).

y : array-like of shape (n_samples,)
    Target values (class labels in classification, real numbers in
    regression)

sample_weight : array-like of shape (n_samples,), default=None
    Per-sample weights. Rescale C per sample. Higher weights
    force the classifier to put more emphasis on these points.

Returns
-------
self : object

Notes
-----
If X and y are not C-ordered and contiguous arrays of np.float64 and
X is not a scipy.sparse.csr_matrix, X and/or y may be copied.

If X is a dense array, then the other methods will not support sparse
matrices as input.
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

val predict : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Perform regression on samples in X.

For an one-class model, +1 (inlier) or -1 (outlier) is returned.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    For kernel='precomputed', the expected shape of X is
    (n_samples_test, n_samples_train).

Returns
-------
y_pred : ndarray of shape (n_samples,)
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
The R2 score used when calling ``score`` on a regressor uses
``multioutput='uniform_average'`` from version 0.23 to keep consistent
with default value of :func:`~sklearn.metrics.r2_score`.
This influences the ``score`` method of all the multioutput
regressors (except for
:class:`~sklearn.multioutput.MultiOutputRegressor`).
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


(** Attribute support_: get value or raise Not_found if None.*)
val support_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute support_: get value as an option. *)
val support_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute support_vectors_: get value or raise Not_found if None.*)
val support_vectors_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute support_vectors_: get value as an option. *)
val support_vectors_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute dual_coef_: get value or raise Not_found if None.*)
val dual_coef_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute dual_coef_: get value as an option. *)
val dual_coef_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute coef_: get value or raise Not_found if None.*)
val coef_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute coef_: get value as an option. *)
val coef_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute fit_status_: get value or raise Not_found if None.*)
val fit_status_ : t -> int

(** Attribute fit_status_: get value as an option. *)
val fit_status_opt : t -> (int) option


(** Attribute intercept_: get value or raise Not_found if None.*)
val intercept_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute intercept_: get value as an option. *)
val intercept_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val l1_min_c : ?loss:[`Squared_hinge | `Log] -> ?fit_intercept:bool -> ?intercept_scaling:float -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> unit -> float
(**
Return the lowest bound for C such that for C in (l1_min_C, infinity)
the model is guaranteed not to be empty. This applies to l1 penalized
classifiers, such as LinearSVC with penalty='l1' and
linear_model.LogisticRegression with penalty='l1'.

This value is valid if class_weight parameter in fit() is not set.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Training vector, where n_samples in the number of samples and
    n_features is the number of features.

y : array-like of shape (n_samples,)
    Target vector relative to X.

loss : {'squared_hinge', 'log'}, default='squared_hinge'
    Specifies the loss function.
    With 'squared_hinge' it is the squared hinge loss (a.k.a. L2 loss).
    With 'log' it is the loss of logistic regression models.

fit_intercept : bool, default=True
    Specifies if the intercept should be fitted by the model.
    It must match the fit() method parameter.

intercept_scaling : float, default=1.0
    when fit_intercept is True, instance vector x becomes
    [x, intercept_scaling],
    i.e. a 'synthetic' feature with constant value equals to
    intercept_scaling is appended to the instance vector.
    It must match the fit() method parameter.

Returns
-------
l1_min_c : float
    minimum value for C
*)

