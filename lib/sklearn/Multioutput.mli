(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module ClassifierChain : sig
type tag = [`ClassifierChain]
type t = [`BaseEstimator | `ClassifierChain | `ClassifierMixin | `MetaEstimatorMixin | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_classifier : t -> [`ClassifierMixin] Obj.t
val as_estimator : t -> [`BaseEstimator] Obj.t
val as_meta_estimator : t -> [`MetaEstimatorMixin] Obj.t
val create : ?order:[`Arr of [>`ArrayLike] Np.Obj.t | `Random] -> ?cv:[`Arr of [>`ArrayLike] Np.Obj.t | `BaseCrossValidator of [>`BaseCrossValidator] Np.Obj.t | `I of int] -> ?random_state:int -> base_estimator:[>`BaseEstimator] Np.Obj.t -> unit -> t
(**
A multi-label model that arranges binary classifiers into a chain.

Each model makes a prediction in the order specified by the chain using
all of the available features provided to the model plus the predictions
of models that are earlier in the chain.

Read more in the :ref:`User Guide <classifierchain>`.

.. versionadded:: 0.19

Parameters
----------
base_estimator : estimator
    The base estimator from which the classifier chain is built.

order : array-like of shape (n_outputs,) or 'random', optional
    By default the order will be determined by the order of columns in
    the label matrix Y.::

        order = [0, 1, 2, ..., Y.shape[1] - 1]

    The order of the chain can be explicitly set by providing a list of
    integers. For example, for a chain of length 5.::

        order = [1, 3, 2, 4, 0]

    means that the first model in the chain will make predictions for
    column 1 in the Y matrix, the second model will make predictions
    for column 3, etc.

    If order is 'random' a random ordering will be used.

cv : int, cross-validation generator or an iterable, optional     (default=None)
    Determines whether to use cross validated predictions or true
    labels for the results of previous estimators in the chain.
    If cv is None the true labels are used when fitting. Otherwise
    possible inputs for cv are:

    - integer, to specify the number of folds in a (Stratified)KFold,
    - :term:`CV splitter`,
    - An iterable yielding (train, test) splits as arrays of indices.

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

    The random number generator is used to generate random chain orders.

Attributes
----------
classes_ : list
    A list of arrays of length ``len(estimators_)`` containing the
    class labels for each estimator in the chain.

estimators_ : list
    A list of clones of base_estimator.

order_ : list
    The order of labels in the classifier chain.

See also
--------
RegressorChain: Equivalent for regression
MultioutputClassifier: Classifies each output independently rather than
    chaining.

References
----------
Jesse Read, Bernhard Pfahringer, Geoff Holmes, Eibe Frank, 'Classifier
Chains for Multi-label Classification', 2009.
*)

val decision_function : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Evaluate the decision_function of the models in the chain.

Parameters
----------
X : array-like, shape (n_samples, n_features)

Returns
-------
Y_decision : array-like, shape (n_samples, n_classes )
    Returns the decision function of the sample for each model
    in the chain.
*)

val fit : x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fit the model to data matrix X and targets Y.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    The input data.
Y : array-like, shape (n_samples, n_classes)
    The target values.

Returns
-------
self : object
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
Predict on the data matrix X using the ClassifierChain model.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    The input data.

Returns
-------
Y_pred : array-like, shape (n_samples, n_classes)
    The predicted values.
*)

val predict_proba : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Predict probability estimates.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)

Returns
-------
Y_prob : array-like, shape (n_samples, n_classes)
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


(** Attribute classes_: get value or raise Not_found if None.*)
val classes_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute classes_: get value as an option. *)
val classes_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute estimators_: get value or raise Not_found if None.*)
val estimators_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute estimators_: get value as an option. *)
val estimators_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute order_: get value or raise Not_found if None.*)
val order_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute order_: get value as an option. *)
val order_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MultiOutputClassifier : sig
type tag = [`MultiOutputClassifier]
type t = [`BaseEstimator | `ClassifierMixin | `MetaEstimatorMixin | `MultiOutputClassifier | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_classifier : t -> [`ClassifierMixin] Obj.t
val as_estimator : t -> [`BaseEstimator] Obj.t
val as_meta_estimator : t -> [`MetaEstimatorMixin] Obj.t
val create : ?n_jobs:int -> estimator:[>`BaseEstimator] Np.Obj.t -> unit -> t
(**
Multi target classification

This strategy consists of fitting one classifier per target. This is a
simple strategy for extending classifiers that do not natively support
multi-target classification

Parameters
----------
estimator : estimator object
    An estimator object implementing :term:`fit`, :term:`score` and
    :term:`predict_proba`.

n_jobs : int or None, optional (default=None)
    The number of jobs to use for the computation.
    It does each target variable in y in parallel.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

Attributes
----------
estimators_ : list of ``n_output`` estimators
    Estimators used for predictions.

Examples
--------
>>> import numpy as np
>>> from sklearn.datasets import make_multilabel_classification
>>> from sklearn.multioutput import MultiOutputClassifier
>>> from sklearn.neighbors import KNeighborsClassifier

>>> X, y = make_multilabel_classification(n_classes=3, random_state=0)
>>> clf = MultiOutputClassifier(KNeighborsClassifier()).fit(X, y)
>>> clf.predict(X[-2:])
array([[1, 1, 0], [1, 1, 1]])
*)

val fit : ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fit the model to data matrix X and targets Y.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input data.
Y : array-like of shape (n_samples, n_classes)
    The target values.
sample_weight : array-like of shape (n_samples,) or None
    Sample weights. If None, then samples are equally weighted.
    Only supported if the underlying classifier supports sample
    weights.

Returns
-------
self : object
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

val partial_fit : ?classes:[>`ArrayLike] Np.Obj.t list -> ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`Spmatrix] Np.Obj.t -> y:[>`Spmatrix] Np.Obj.t -> [> tag] Obj.t -> t
(**
Incrementally fit the model to data.
Fit a separate model for each output variable.

Parameters
----------
X : (sparse) array-like, shape (n_samples, n_features)
    Data.

y : (sparse) array-like, shape (n_samples, n_outputs)
    Multi-output targets.

classes : list of numpy arrays, shape (n_outputs)
    Each array is unique classes for one output in str/int
    Can be obtained by via
    ``[np.unique(y[:, i]) for i in range(y.shape[1])]``, where y is the
    target matrix of the entire dataset.
    This argument is required for the first call to partial_fit
    and can be omitted in the subsequent calls.
    Note that y doesn't need to contain all labels in `classes`.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights. If None, then samples are equally weighted.
    Only supported if the underlying regressor supports sample
    weights.

Returns
-------
self : object
*)

val predict : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Predict multi-output variable using a model
 trained for each target variable.

Parameters
----------
X : (sparse) array-like, shape (n_samples, n_features)
    Data.

Returns
-------
y : (sparse) array-like, shape (n_samples, n_outputs)
    Multi-output targets predicted across multiple predictors.
    Note: Separate models are generated for each predictor.
*)

val score : x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> float
(**
Returns the mean accuracy on the given test data and labels.

Parameters
----------
X : array-like, shape [n_samples, n_features]
    Test samples

y : array-like, shape [n_samples, n_outputs]
    True values for X

Returns
-------
scores : float
    accuracy_score of self.predict(X) versus y
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


(** Attribute estimators_: get value or raise Not_found if None.*)
val estimators_ : t -> Py.Object.t

(** Attribute estimators_: get value as an option. *)
val estimators_opt : t -> (Py.Object.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MultiOutputEstimator : sig
type tag = [`MultiOutputEstimator]
type t = [`BaseEstimator | `MetaEstimatorMixin | `MultiOutputEstimator | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_estimator : t -> [`BaseEstimator] Obj.t
val as_meta_estimator : t -> [`MetaEstimatorMixin] Obj.t
val fit : ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`Spmatrix] Np.Obj.t -> y:[>`Spmatrix] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fit the model to data.
Fit a separate model for each output variable.

Parameters
----------
X : (sparse) array-like, shape (n_samples, n_features)
    Data.

y : (sparse) array-like, shape (n_samples, n_outputs)
    Multi-output targets. An indicator matrix turns on multilabel
    estimation.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights. If None, then samples are equally weighted.
    Only supported if the underlying regressor supports sample
    weights.

Returns
-------
self : object
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

val partial_fit : ?classes:[>`ArrayLike] Np.Obj.t list -> ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`Spmatrix] Np.Obj.t -> y:[>`Spmatrix] Np.Obj.t -> [> tag] Obj.t -> t
(**
Incrementally fit the model to data.
Fit a separate model for each output variable.

Parameters
----------
X : (sparse) array-like, shape (n_samples, n_features)
    Data.

y : (sparse) array-like, shape (n_samples, n_outputs)
    Multi-output targets.

classes : list of numpy arrays, shape (n_outputs)
    Each array is unique classes for one output in str/int
    Can be obtained by via
    ``[np.unique(y[:, i]) for i in range(y.shape[1])]``, where y is the
    target matrix of the entire dataset.
    This argument is required for the first call to partial_fit
    and can be omitted in the subsequent calls.
    Note that y doesn't need to contain all labels in `classes`.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights. If None, then samples are equally weighted.
    Only supported if the underlying regressor supports sample
    weights.

Returns
-------
self : object
*)

val predict : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Predict multi-output variable using a model
 trained for each target variable.

Parameters
----------
X : (sparse) array-like, shape (n_samples, n_features)
    Data.

Returns
-------
y : (sparse) array-like, shape (n_samples, n_outputs)
    Multi-output targets predicted across multiple predictors.
    Note: Separate models are generated for each predictor.
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


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MultiOutputRegressor : sig
type tag = [`MultiOutputRegressor]
type t = [`BaseEstimator | `MetaEstimatorMixin | `MultiOutputRegressor | `Object | `RegressorMixin] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_regressor : t -> [`RegressorMixin] Obj.t
val as_estimator : t -> [`BaseEstimator] Obj.t
val as_meta_estimator : t -> [`MetaEstimatorMixin] Obj.t
val create : ?n_jobs:int -> estimator:[>`BaseEstimator] Np.Obj.t -> unit -> t
(**
Multi target regression

This strategy consists of fitting one regressor per target. This is a
simple strategy for extending regressors that do not natively support
multi-target regression.

Parameters
----------
estimator : estimator object
    An estimator object implementing :term:`fit` and :term:`predict`.

n_jobs : int or None, optional (default=None)
    The number of jobs to run in parallel for :meth:`fit`.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

    When individual estimators are fast to train or predict
    using `n_jobs>1` can result in slower performance due
    to the overhead of spawning processes.

Attributes
----------
estimators_ : list of ``n_output`` estimators
    Estimators used for predictions.
*)

val fit : ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`Spmatrix] Np.Obj.t -> y:[>`Spmatrix] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fit the model to data.
Fit a separate model for each output variable.

Parameters
----------
X : (sparse) array-like, shape (n_samples, n_features)
    Data.

y : (sparse) array-like, shape (n_samples, n_outputs)
    Multi-output targets. An indicator matrix turns on multilabel
    estimation.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights. If None, then samples are equally weighted.
    Only supported if the underlying regressor supports sample
    weights.

Returns
-------
self : object
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

val partial_fit : ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`Spmatrix] Np.Obj.t -> y:[>`Spmatrix] Np.Obj.t -> [> tag] Obj.t -> t
(**
Incrementally fit the model to data.
Fit a separate model for each output variable.

Parameters
----------
X : (sparse) array-like, shape (n_samples, n_features)
    Data.

y : (sparse) array-like, shape (n_samples, n_outputs)
    Multi-output targets.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights. If None, then samples are equally weighted.
    Only supported if the underlying regressor supports sample
    weights.

Returns
-------
self : object
*)

val predict : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Predict multi-output variable using a model
 trained for each target variable.

Parameters
----------
X : (sparse) array-like, shape (n_samples, n_features)
    Data.

Returns
-------
y : (sparse) array-like, shape (n_samples, n_outputs)
    Multi-output targets predicted across multiple predictors.
    Note: Separate models are generated for each predictor.
*)

val score : ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> float
(**
Returns the coefficient of determination R^2 of the prediction.

The coefficient R^2 is defined as (1 - u/v), where u is the residual
sum of squares ((y_true - y_pred) ** 2).sum() and v is the regression
sum of squares ((y_true - y_true.mean()) ** 2).sum().
Best possible score is 1.0 and it can be negative (because the
model can be arbitrarily worse). A constant model that always
predicts the expected value of y, disregarding the input features,
would get a R^2 score of 0.0.

Notes
-----
R^2 is calculated by weighting all the targets equally using
`multioutput='uniform_average'`.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Test samples.

y : array-like, shape (n_samples) or (n_samples, n_outputs)
    True values for X.

sample_weight : array-like, shape [n_samples], optional
    Sample weights.

Returns
-------
score : float
    R^2 of self.predict(X) wrt. y.
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


(** Attribute estimators_: get value or raise Not_found if None.*)
val estimators_ : t -> Py.Object.t

(** Attribute estimators_: get value as an option. *)
val estimators_opt : t -> (Py.Object.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module RegressorChain : sig
type tag = [`RegressorChain]
type t = [`BaseEstimator | `MetaEstimatorMixin | `Object | `RegressorChain | `RegressorMixin] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_regressor : t -> [`RegressorMixin] Obj.t
val as_estimator : t -> [`BaseEstimator] Obj.t
val as_meta_estimator : t -> [`MetaEstimatorMixin] Obj.t
val create : ?order:[`Arr of [>`ArrayLike] Np.Obj.t | `Random] -> ?cv:[`Arr of [>`ArrayLike] Np.Obj.t | `BaseCrossValidator of [>`BaseCrossValidator] Np.Obj.t | `I of int] -> ?random_state:int -> base_estimator:[>`BaseEstimator] Np.Obj.t -> unit -> t
(**
A multi-label model that arranges regressions into a chain.

Each model makes a prediction in the order specified by the chain using
all of the available features provided to the model plus the predictions
of models that are earlier in the chain.

Read more in the :ref:`User Guide <regressorchain>`.

Parameters
----------
base_estimator : estimator
    The base estimator from which the classifier chain is built.

order : array-like of shape (n_outputs,) or 'random', optional
    By default the order will be determined by the order of columns in
    the label matrix Y.::

        order = [0, 1, 2, ..., Y.shape[1] - 1]

    The order of the chain can be explicitly set by providing a list of
    integers. For example, for a chain of length 5.::

        order = [1, 3, 2, 4, 0]

    means that the first model in the chain will make predictions for
    column 1 in the Y matrix, the second model will make predictions
    for column 3, etc.

    If order is 'random' a random ordering will be used.

cv : int, cross-validation generator or an iterable, optional     (default=None)
    Determines whether to use cross validated predictions or true
    labels for the results of previous estimators in the chain.
    If cv is None the true labels are used when fitting. Otherwise
    possible inputs for cv are:

    - integer, to specify the number of folds in a (Stratified)KFold,
    - :term:`CV splitter`,
    - An iterable yielding (train, test) splits as arrays of indices.

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

    The random number generator is used to generate random chain orders.

Attributes
----------
estimators_ : list
    A list of clones of base_estimator.

order_ : list
    The order of labels in the classifier chain.

See also
--------
ClassifierChain: Equivalent for classification
MultioutputRegressor: Learns each output independently rather than
    chaining.
*)

val fit : x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fit the model to data matrix X and targets Y.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    The input data.
Y : array-like, shape (n_samples, n_classes)
    The target values.

Returns
-------
self : object
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
Predict on the data matrix X using the ClassifierChain model.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    The input data.

Returns
-------
Y_pred : array-like, shape (n_samples, n_classes)
    The predicted values.
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


(** Attribute estimators_: get value or raise Not_found if None.*)
val estimators_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute estimators_: get value as an option. *)
val estimators_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute order_: get value or raise Not_found if None.*)
val order_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute order_: get value as an option. *)
val order_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val abstractmethod : Py.Object.t -> Py.Object.t
(**
A decorator indicating abstract methods.

Requires that the metaclass is ABCMeta or derived from it.  A
class that has a metaclass derived from ABCMeta cannot be
instantiated unless all of its abstract methods are overridden.
The abstract methods can be called using any of the normal
'super' call mechanisms.  abstractmethod() may be used to declare
abstract methods for properties and descriptors.

Usage:

    class C(metaclass=ABCMeta):
        @abstractmethod
        def my_abstract_method(self, ...):
            ...
*)

val check_X_y : ?accept_sparse:[`StringList of string list | `S of string | `Bool of bool] -> ?accept_large_sparse:bool -> ?dtype:[`S of string | `Dtype of Np.Dtype.t | `Dtypes of Np.Dtype.t list | `None] -> ?order:[`F | `C] -> ?copy:bool -> ?force_all_finite:[`Allow_nan | `Bool of bool] -> ?ensure_2d:bool -> ?allow_nd:bool -> ?multi_output:bool -> ?ensure_min_samples:int -> ?ensure_min_features:int -> ?y_numeric:bool -> ?warn_on_dtype:bool -> ?estimator:[>`BaseEstimator] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> unit -> (Py.Object.t * Py.Object.t)
(**
Input validation for standard estimators.

Checks X and y for consistent length, enforces X to be 2D and y 1D. By
default, X is checked to be non-empty and containing only finite values.
Standard input checks are also applied to y, such as checking that y
does not have np.nan or np.inf targets. For multi-label y, set
multi_output=True to allow 2D and sparse y. If the dtype of X is
object, attempt converting to float, raising on failure.

Parameters
----------
X : nd-array, list or sparse matrix
    Input data.

y : nd-array, list or sparse matrix
    Labels.

accept_sparse : string, boolean or list of string (default=False)
    String[s] representing allowed sparse matrix formats, such as 'csc',
    'csr', etc. If the input is sparse but not in the allowed format,
    it will be converted to the first listed format. True allows the input
    to be any format. False means that a sparse matrix input will
    raise an error.

accept_large_sparse : bool (default=True)
    If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
    accept_sparse, accept_large_sparse will cause it to be accepted only
    if its indices are stored with a 32-bit dtype.

    .. versionadded:: 0.20

dtype : string, type, list of types or None (default='numeric')
    Data type of result. If None, the dtype of the input is preserved.
    If 'numeric', dtype is preserved unless array.dtype is object.
    If dtype is a list of types, conversion on the first type is only
    performed if the dtype of the input is not in the list.

order : 'F', 'C' or None (default=None)
    Whether an array will be forced to be fortran or c-style.

copy : boolean (default=False)
    Whether a forced copy will be triggered. If copy=False, a copy might
    be triggered by a conversion.

force_all_finite : boolean or 'allow-nan', (default=True)
    Whether to raise an error on np.inf and np.nan in X. This parameter
    does not influence whether y can have np.inf or np.nan values.
    The possibilities are:

    - True: Force all values of X to be finite.
    - False: accept both np.inf and np.nan in X.
    - 'allow-nan': accept only np.nan values in X. Values cannot be
      infinite.

    .. versionadded:: 0.20
       ``force_all_finite`` accepts the string ``'allow-nan'``.

ensure_2d : boolean (default=True)
    Whether to raise a value error if X is not 2D.

allow_nd : boolean (default=False)
    Whether to allow X.ndim > 2.

multi_output : boolean (default=False)
    Whether to allow 2D y (array or sparse matrix). If false, y will be
    validated as a vector. y cannot have np.nan or np.inf values if
    multi_output=True.

ensure_min_samples : int (default=1)
    Make sure that X has a minimum number of samples in its first
    axis (rows for a 2D array).

ensure_min_features : int (default=1)
    Make sure that the 2D array has some minimum number of features
    (columns). The default value of 1 rejects empty datasets.
    This check is only enforced when X has effectively 2 dimensions or
    is originally 1D and ``ensure_2d`` is True. Setting to 0 disables
    this check.

y_numeric : boolean (default=False)
    Whether to ensure that y has a numeric type. If dtype of y is object,
    it is converted to float64. Should only be used for regression
    algorithms.

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
X_converted : object
    The converted and validated X.

y_converted : object
    The converted and validated y.
*)

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

val check_classification_targets : [>`ArrayLike] Np.Obj.t -> Py.Object.t
(**
Ensure that target y is of a non-regression type.

Only the following target types (as defined in type_of_target) are allowed:
    'binary', 'multiclass', 'multiclass-multioutput',
    'multilabel-indicator', 'multilabel-sequences'

Parameters
----------
y : array-like
*)

val check_is_fitted : ?attributes:[`S of string | `Arr of [>`ArrayLike] Np.Obj.t | `StringList of string list] -> ?msg:string -> ?all_or_any:[`Callable of Py.Object.t | `PyObject of Py.Object.t] -> estimator:[>`BaseEstimator] Np.Obj.t -> unit -> Py.Object.t
(**
Perform is_fitted validation for estimator.

Checks if the estimator is fitted by verifying the presence of
fitted attributes (ending with a trailing underscore) and otherwise
raises a NotFittedError with the given message.

This utility is meant to be used internally by estimators themselves,
typically in their own predict / transform methods.

Parameters
----------
estimator : estimator instance.
    estimator instance for which the check is performed.

attributes : str, list or tuple of str, default=None
    Attribute name(s) given as string or a list/tuple of strings
    Eg.: ``['coef_', 'estimator_', ...], 'coef_'``

    If `None`, `estimator` is considered fitted if there exist an
    attribute that ends with a underscore and does not start with double
    underscore.

msg : string
    The default error message is, 'This %(name)s instance is not fitted
    yet. Call 'fit' with appropriate arguments before using this
    estimator.'

    For custom messages if '%(name)s' is present in the message string,
    it is substituted for the estimator name.

    Eg. : 'Estimator, %(name)s, must be fitted before sparsifying'.

all_or_any : callable, {all, any}, default all
    Specify whether all or any of the given attributes must exist.

Returns
-------
None

Raises
------
NotFittedError
    If the attributes are not found.
*)

val check_random_state : [`Optional of [`I of int | `None] | `RandomState of Py.Object.t] -> Py.Object.t
(**
Turn seed into a np.random.RandomState instance

Parameters
----------
seed : None | int | instance of RandomState
    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
*)

val clone : ?safe:bool -> estimator:[>`BaseEstimator] Np.Obj.t -> unit -> Py.Object.t
(**
Constructs a new estimator with the same parameters.

Clone does a deep copy of the model in an estimator
without actually copying attached data. It yields a new estimator
with the same parameters that has not been fit on any data.

Parameters
----------
estimator : estimator object, or list, tuple or set of objects
    The estimator or group of estimators to be cloned

safe : boolean, optional
    If safe is false, clone will fall back to a deep copy on objects
    that are not estimators.
*)

val cross_val_predict : ?y:[>`ArrayLike] Np.Obj.t -> ?groups:[>`ArrayLike] Np.Obj.t -> ?cv:[`Arr of [>`ArrayLike] Np.Obj.t | `BaseCrossValidator of [>`BaseCrossValidator] Np.Obj.t | `I of int] -> ?n_jobs:int -> ?verbose:int -> ?fit_params:Dict.t -> ?pre_dispatch:[`S of string | `I of int] -> ?method_:string -> estimator:[>`BaseEstimator] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> unit -> [>`ArrayLike] Np.Obj.t
(**
Generate cross-validated estimates for each input data point

The data is split according to the cv parameter. Each sample belongs
to exactly one test set, and its prediction is computed with an
estimator fitted on the corresponding training set.

Passing these predictions into an evaluation metric may not be a valid
way to measure generalization performance. Results can differ from
:func:`cross_validate` and :func:`cross_val_score` unless all tests sets
have equal size and the metric decomposes over samples.

Read more in the :ref:`User Guide <cross_validation>`.

Parameters
----------
estimator : estimator object implementing 'fit' and 'predict'
    The object to use to fit the data.

X : array-like
    The data to fit. Can be, for example a list, or an array at least 2d.

y : array-like, optional, default: None
    The target variable to try to predict in the case of
    supervised learning.

groups : array-like, with shape (n_samples,), optional
    Group labels for the samples used while splitting the dataset into
    train/test set. Only used in conjunction with a 'Group' :term:`cv`
    instance (e.g., :class:`GroupKFold`).

cv : int, cross-validation generator or an iterable, optional
    Determines the cross-validation splitting strategy.
    Possible inputs for cv are:

    - None, to use the default 5-fold cross validation,
    - integer, to specify the number of folds in a `(Stratified)KFold`,
    - :term:`CV splitter`,
    - An iterable yielding (train, test) splits as arrays of indices.

    For integer/None inputs, if the estimator is a classifier and ``y`` is
    either binary or multiclass, :class:`StratifiedKFold` is used. In all
    other cases, :class:`KFold` is used.

    Refer :ref:`User Guide <cross_validation>` for the various
    cross-validation strategies that can be used here.

    .. versionchanged:: 0.22
        ``cv`` default value if None changed from 3-fold to 5-fold.

n_jobs : int or None, optional (default=None)
    The number of CPUs to use to do the computation.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

verbose : integer, optional
    The verbosity level.

fit_params : dict, optional
    Parameters to pass to the fit method of the estimator.

pre_dispatch : int, or string, optional
    Controls the number of jobs that get dispatched during parallel
    execution. Reducing this number can be useful to avoid an
    explosion of memory consumption when more jobs get dispatched
    than CPUs can process. This parameter can be:

        - None, in which case all the jobs are immediately
          created and spawned. Use this for lightweight and
          fast-running jobs, to avoid delays due to on-demand
          spawning of the jobs

        - An int, giving the exact number of total jobs that are
          spawned

        - A string, giving an expression as a function of n_jobs,
          as in '2*n_jobs'

method : string, optional, default: 'predict'
    Invokes the passed method name of the passed estimator. For
    method='predict_proba', the columns correspond to the classes
    in sorted order.

Returns
-------
predictions : ndarray
    This is the result of calling ``method``

See also
--------
cross_val_score : calculate score for each CV split

cross_validate : calculate one or more scores and timings for each CV split

Notes
-----
In the case that one or more classes are absent in a training portion, a
default score needs to be assigned to all instances for that class if
``method`` produces columns per class, as in {'decision_function',
'predict_proba', 'predict_log_proba'}.  For ``predict_proba`` this value is
0.  In order to ensure finite output, we approximate negative infinity by
the minimum finite float value for the dtype in other cases.

Examples
--------
>>> from sklearn import datasets, linear_model
>>> from sklearn.model_selection import cross_val_predict
>>> diabetes = datasets.load_diabetes()
>>> X = diabetes.data[:150]
>>> y = diabetes.target[:150]
>>> lasso = linear_model.Lasso()
>>> y_pred = cross_val_predict(lasso, X, y, cv=3)
*)

val delayed : ?check_pickle:Py.Object.t -> function_:Py.Object.t -> unit -> Py.Object.t
(**
Decorator used to capture the arguments of a function.
*)

val has_fit_parameter : estimator:[>`BaseEstimator] Np.Obj.t -> parameter:string -> unit -> bool
(**
Checks whether the estimator's fit method supports the given parameter.

Parameters
----------
estimator : object
    An estimator to inspect.

parameter : str
    The searched parameter.

Returns
-------
is_parameter: bool
    Whether the parameter was found to be a named parameter of the
    estimator's fit method.

Examples
--------
>>> from sklearn.svm import SVC
>>> has_fit_parameter(SVC(), 'sample_weight')
True
*)

val if_delegate_has_method : [`S of string | `StringList of string list] -> Py.Object.t
(**
Create a decorator for methods that are delegated to a sub-estimator

This enables ducktyping by hasattr returning True according to the
sub-estimator.

Parameters
----------
delegate : string, list of strings or tuple of strings
    Name of the sub-estimator that can be accessed as an attribute of the
    base object. If a list or a tuple of names are provided, the first
    sub-estimator that is an attribute of the base object will be used.
*)

val is_classifier : [>`BaseEstimator] Np.Obj.t -> bool
(**
Return True if the given estimator is (probably) a classifier.

Parameters
----------
estimator : object
    Estimator object to test.

Returns
-------
out : bool
    True if estimator is a classifier and False otherwise.
*)

