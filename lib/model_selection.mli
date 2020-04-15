module GridSearchCV : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?scoring:[`String of string | `Callable of Py.Object.t | `Dict of Py.Object.t | `None | `PyObject of Py.Object.t] -> ?n_jobs:[`Int of int | `None] -> ?iid:bool -> ?refit:[`Bool of bool | `String of string | `Callable of Py.Object.t] -> ?cv:[`Int of int | `CrossValGenerator of Py.Object.t | `Ndarray of Ndarray.t] -> ?verbose:int -> ?pre_dispatch:[`Int of int | `String of string] -> ?error_score:[`Raise | `PyObject of Py.Object.t] -> ?return_train_score:bool -> estimator:Py.Object.t -> param_grid:[`Dict of Py.Object.t | `PyObject of Py.Object.t] -> unit -> t
(**
Exhaustive search over specified parameter values for an estimator.

Important members are fit, predict.

GridSearchCV implements a "fit" and a "score" method.
It also implements "predict", "predict_proba", "decision_function",
"transform" and "inverse_transform" if they are implemented in the
estimator used.

The parameters of the estimator used to apply these methods are optimized
by cross-validated grid-search over a parameter grid.

Read more in the :ref:`User Guide <grid_search>`.

Parameters
----------
estimator : estimator object.
    This is assumed to implement the scikit-learn estimator interface.
    Either estimator needs to provide a ``score`` function,
    or ``scoring`` must be passed.

param_grid : dict or list of dictionaries
    Dictionary with parameters names (string) as keys and lists of
    parameter settings to try as values, or a list of such
    dictionaries, in which case the grids spanned by each dictionary
    in the list are explored. This enables searching over any sequence
    of parameter settings.

scoring : string, callable, list/tuple, dict or None, default: None
    A single string (see :ref:`scoring_parameter`) or a callable
    (see :ref:`scoring`) to evaluate the predictions on the test set.

    For evaluating multiple metrics, either give a list of (unique) strings
    or a dict with names as keys and callables as values.

    NOTE that when using custom scorers, each scorer should return a single
    value. Metric functions returning a list/array of values can be wrapped
    into multiple scorers that return one value each.

    See :ref:`multimetric_grid_search` for an example.

    If None, the estimator's score method is used.

n_jobs : int or None, optional (default=None)
    Number of jobs to run in parallel.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

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

iid : boolean, default=False
    If True, return the average score across folds, weighted by the number
    of samples in each test set. In this case, the data is assumed to be
    identically distributed across the folds, and the loss minimized is
    the total loss per sample, and not the mean loss across the folds.

    .. deprecated:: 0.22
        Parameter ``iid`` is deprecated in 0.22 and will be removed in 0.24

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

refit : boolean, string, or callable, default=True
    Refit an estimator using the best found parameters on the whole
    dataset.

    For multiple metric evaluation, this needs to be a string denoting the
    scorer that would be used to find the best parameters for refitting
    the estimator at the end.

    Where there are considerations other than maximum score in
    choosing a best estimator, ``refit`` can be set to a function which
    returns the selected ``best_index_`` given ``cv_results_``. In that
    case, the ``best_estimator_`` and ``best_parameters_`` will be set
    according to the returned ``best_index_`` while the ``best_score_``
    attribute will not be available.

    The refitted estimator is made available at the ``best_estimator_``
    attribute and permits using ``predict`` directly on this
    ``GridSearchCV`` instance.

    Also for multiple metric evaluation, the attributes ``best_index_``,
    ``best_score_`` and ``best_params_`` will only be available if
    ``refit`` is set and all of them will be determined w.r.t this specific
    scorer.

    See ``scoring`` parameter to know more about multiple metric
    evaluation.

    .. versionchanged:: 0.20
        Support for callable added.

verbose : integer
    Controls the verbosity: the higher, the more messages.

error_score : 'raise' or numeric
    Value to assign to the score if an error occurs in estimator fitting.
    If set to 'raise', the error is raised. If a numeric value is given,
    FitFailedWarning is raised. This parameter does not affect the refit
    step, which will always raise the error. Default is ``np.nan``.

return_train_score : boolean, default=False
    If ``False``, the ``cv_results_`` attribute will not include training
    scores.
    Computing training scores is used to get insights on how different
    parameter settings impact the overfitting/underfitting trade-off.
    However computing the scores on the training set can be computationally
    expensive and is not strictly required to select the parameters that
    yield the best generalization performance.


Examples
--------
>>> from sklearn import svm, datasets
>>> from sklearn.model_selection import GridSearchCV
>>> iris = datasets.load_iris()
>>> parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
>>> svc = svm.SVC()
>>> clf = GridSearchCV(svc, parameters)
>>> clf.fit(iris.data, iris.target)
GridSearchCV(estimator=SVC(),
             param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf')})
>>> sorted(clf.cv_results_.keys())
['mean_fit_time', 'mean_score_time', 'mean_test_score',...
 'param_C', 'param_kernel', 'params',...
 'rank_test_score', 'split0_test_score',...
 'split2_test_score', ...
 'std_fit_time', 'std_score_time', 'std_test_score']

Attributes
----------
cv_results_ : dict of numpy (masked) ndarrays
    A dict with keys as column headers and values as columns, that can be
    imported into a pandas ``DataFrame``.

    For instance the below given table

    +------------+-----------+------------+-----------------+---+---------+
    |param_kernel|param_gamma|param_degree|split0_test_score|...|rank_t...|
    +============+===========+============+=================+===+=========+
    |  'poly'    |     --    |      2     |       0.80      |...|    2    |
    +------------+-----------+------------+-----------------+---+---------+
    |  'poly'    |     --    |      3     |       0.70      |...|    4    |
    +------------+-----------+------------+-----------------+---+---------+
    |  'rbf'     |     0.1   |     --     |       0.80      |...|    3    |
    +------------+-----------+------------+-----------------+---+---------+
    |  'rbf'     |     0.2   |     --     |       0.93      |...|    1    |
    +------------+-----------+------------+-----------------+---+---------+

    will be represented by a ``cv_results_`` dict of::

        {
        'param_kernel': masked_array(data = ['poly', 'poly', 'rbf', 'rbf'],
                                     mask = [False False False False]...)
        'param_gamma': masked_array(data = [-- -- 0.1 0.2],
                                    mask = [ True  True False False]...),
        'param_degree': masked_array(data = [2.0 3.0 -- --],
                                     mask = [False False  True  True]...),
        'split0_test_score'  : [0.80, 0.70, 0.80, 0.93],
        'split1_test_score'  : [0.82, 0.50, 0.70, 0.78],
        'mean_test_score'    : [0.81, 0.60, 0.75, 0.85],
        'std_test_score'     : [0.01, 0.10, 0.05, 0.08],
        'rank_test_score'    : [2, 4, 3, 1],
        'split0_train_score' : [0.80, 0.92, 0.70, 0.93],
        'split1_train_score' : [0.82, 0.55, 0.70, 0.87],
        'mean_train_score'   : [0.81, 0.74, 0.70, 0.90],
        'std_train_score'    : [0.01, 0.19, 0.00, 0.03],
        'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
        'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
        'mean_score_time'    : [0.01, 0.06, 0.04, 0.04],
        'std_score_time'     : [0.00, 0.00, 0.00, 0.01],
        'params'             : [{'kernel': 'poly', 'degree': 2}, ...],
        }

    NOTE

    The key ``'params'`` is used to store a list of parameter
    settings dicts for all the parameter candidates.

    The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
    ``std_score_time`` are all in seconds.

    For multi-metric evaluation, the scores for all the scorers are
    available in the ``cv_results_`` dict at the keys ending with that
    scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
    above. ('split0_test_precision', 'mean_train_precision' etc.)

best_estimator_ : estimator
    Estimator that was chosen by the search, i.e. estimator
    which gave highest score (or smallest loss if specified)
    on the left out data. Not available if ``refit=False``.

    See ``refit`` parameter for more information on allowed values.

best_score_ : float
    Mean cross-validated score of the best_estimator

    For multi-metric evaluation, this is present only if ``refit`` is
    specified.

    This attribute is not available if ``refit`` is a function.

best_params_ : dict
    Parameter setting that gave the best results on the hold out data.

    For multi-metric evaluation, this is present only if ``refit`` is
    specified.

best_index_ : int
    The index (of the ``cv_results_`` arrays) which corresponds to the best
    candidate parameter setting.

    The dict at ``search.cv_results_['params'][search.best_index_]`` gives
    the parameter setting for the best model, that gives the highest
    mean score (``search.best_score_``).

    For multi-metric evaluation, this is present only if ``refit`` is
    specified.

scorer_ : function or a dict
    Scorer function used on the held out data to choose the best
    parameters for the model.

    For multi-metric evaluation, this attribute holds the validated
    ``scoring`` dict which maps the scorer key to the scorer callable.

n_splits_ : int
    The number of cross-validation splits (folds/iterations).

refit_time_ : float
    Seconds used for refitting the best model on the whole dataset.

    This is present only if ``refit`` is not False.

Notes
-----
The parameters selected are those that maximize the score of the left out
data, unless an explicit score is passed in which case it is used instead.

If `n_jobs` was set to a value higher than one, the data is copied for each
point in the grid (and not `n_jobs` times). This is done for efficiency
reasons if individual jobs take very little time, but may raise errors if
the dataset is large and not enough memory is available.  A workaround in
this case is to set `pre_dispatch`. Then, the memory is copied only
`pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
n_jobs`.

See Also
---------
:class:`ParameterGrid`:
    generates all the combinations of a hyperparameter grid.

:func:`sklearn.model_selection.train_test_split`:
    utility function to split the data into a development set usable
    for fitting a GridSearchCV instance and an evaluation set for
    its final evaluation.

:func:`sklearn.metrics.make_scorer`:
    Make a scorer from a performance metric or loss function.
*)

val decision_function : x:Ndarray.t -> t -> Ndarray.t
(**
Call decision_function on the estimator with the best found parameters.

Only available if ``refit=True`` and the underlying estimator supports
``decision_function``.

Parameters
----------
X : indexable, length n_samples
    Must fulfill the input assumptions of the
    underlying estimator.
*)

val fit : ?y:Ndarray.t -> ?groups:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> ?fit_params:(string * Py.Object.t) list -> x:Ndarray.t -> t -> t
(**
Run fit with all sets of parameters.

Parameters
----------

X : array-like of shape (n_samples, n_features)
    Training vector, where n_samples is the number of samples and
    n_features is the number of features.

y : array-like of shape (n_samples, n_output) or (n_samples,), optional
    Target relative to X for classification or regression;
    None for unsupervised learning.

groups : array-like, with shape (n_samples,), optional
    Group labels for the samples used while splitting the dataset into
    train/test set. Only used in conjunction with a "Group" :term:`cv`
    instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).

**fit_params : dict of string -> object
    Parameters passed to the ``fit`` method of the estimator
*)

val get_params : ?deep:bool -> t -> Py.Object.t
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

val inverse_transform : xt:Ndarray.t -> t -> Py.Object.t
(**
Call inverse_transform on the estimator with the best found params.

Only available if the underlying estimator implements
``inverse_transform`` and ``refit=True``.

Parameters
----------
Xt : indexable, length n_samples
    Must fulfill the input assumptions of the
    underlying estimator.
*)

val predict : x:Ndarray.t -> t -> Ndarray.t
(**
Call predict on the estimator with the best found parameters.

Only available if ``refit=True`` and the underlying estimator supports
``predict``.

Parameters
----------
X : indexable, length n_samples
    Must fulfill the input assumptions of the
    underlying estimator.
*)

val predict_log_proba : x:Ndarray.t -> t -> Py.Object.t
(**
Call predict_log_proba on the estimator with the best found parameters.

Only available if ``refit=True`` and the underlying estimator supports
``predict_log_proba``.

Parameters
----------
X : indexable, length n_samples
    Must fulfill the input assumptions of the
    underlying estimator.
*)

val predict_proba : x:Ndarray.t -> t -> Ndarray.t
(**
Call predict_proba on the estimator with the best found parameters.

Only available if ``refit=True`` and the underlying estimator supports
``predict_proba``.

Parameters
----------
X : indexable, length n_samples
    Must fulfill the input assumptions of the
    underlying estimator.
*)

val score : ?y:Ndarray.t -> x:Ndarray.t -> t -> float
(**
Returns the score on the given data, if the estimator has been refit.

This uses the score defined by ``scoring`` where provided, and the
``best_estimator_.score`` method otherwise.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Input data, where n_samples is the number of samples and
    n_features is the number of features.

y : array-like of shape (n_samples, n_output) or (n_samples,), optional
    Target relative to X for classification or regression;
    None for unsupervised learning.

Returns
-------
score : float
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

val transform : x:Ndarray.t -> t -> Ndarray.t
(**
Call transform on the estimator with the best found parameters.

Only available if the underlying estimator supports ``transform`` and
``refit=True``.

Parameters
----------
X : indexable, length n_samples
    Must fulfill the input assumptions of the
    underlying estimator.
*)


(** Attribute cv_results_: see constructor for documentation *)
val cv_results_ : t -> Py.Object.t

(** Attribute best_estimator_: see constructor for documentation *)
val best_estimator_ : t -> Py.Object.t

(** Attribute best_score_: see constructor for documentation *)
val best_score_ : t -> float

(** Attribute best_params_: see constructor for documentation *)
val best_params_ : t -> Py.Object.t

(** Attribute best_index_: see constructor for documentation *)
val best_index_ : t -> int

(** Attribute scorer_: see constructor for documentation *)
val scorer_ : t -> Py.Object.t

(** Attribute n_splits_: see constructor for documentation *)
val n_splits_ : t -> int

(** Attribute refit_time_: see constructor for documentation *)
val refit_time_ : t -> float

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module GroupKFold : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_splits:int -> unit -> t
(**
K-fold iterator variant with non-overlapping groups.

The same group will not appear in two different folds (the number of
distinct groups has to be at least equal to the number of folds).

The folds are approximately balanced in the sense that the number of
distinct groups is approximately the same in each fold.

Parameters
----------
n_splits : int, default=5
    Number of folds. Must be at least 2.

    .. versionchanged:: 0.22
        ``n_splits`` default value changed from 3 to 5.

Examples
--------
>>> import numpy as np
>>> from sklearn.model_selection import GroupKFold
>>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
>>> y = np.array([1, 2, 3, 4])
>>> groups = np.array([0, 0, 2, 2])
>>> group_kfold = GroupKFold(n_splits=2)
>>> group_kfold.get_n_splits(X, y, groups)
2
>>> print(group_kfold)
GroupKFold(n_splits=2)
>>> for train_index, test_index in group_kfold.split(X, y, groups):
...     print("TRAIN:", train_index, "TEST:", test_index)
...     X_train, X_test = X[train_index], X[test_index]
...     y_train, y_test = y[train_index], y[test_index]
...     print(X_train, X_test, y_train, y_test)
...
TRAIN: [0 1] TEST: [2 3]
[[1 2]
 [3 4]] [[5 6]
 [7 8]] [1 2] [3 4]
TRAIN: [2 3] TEST: [0 1]
[[5 6]
 [7 8]] [[1 2]
 [3 4]] [3 4] [1 2]

See also
--------
LeaveOneGroupOut
    For splitting the data according to explicit domain-specific
    stratification of the dataset.
*)

val get_n_splits : ?x:Py.Object.t -> ?y:Py.Object.t -> ?groups:Py.Object.t -> t -> int
(**
Returns the number of splitting iterations in the cross-validator

Parameters
----------
X : object
    Always ignored, exists for compatibility.

y : object
    Always ignored, exists for compatibility.

groups : object
    Always ignored, exists for compatibility.

Returns
-------
n_splits : int
    Returns the number of splitting iterations in the cross-validator.
*)

val split : ?y:Ndarray.t -> ?groups:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> x:Ndarray.t -> t -> Py.Object.t
(**
Generate indices to split data into training and test set.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training data, where n_samples is the number of samples
    and n_features is the number of features.

y : array-like, shape (n_samples,), optional
    The target variable for supervised learning problems.

groups : array-like, with shape (n_samples,)
    Group labels for the samples used while splitting the dataset into
    train/test set.

Yields
------
train : ndarray
    The training set indices for that split.

test : ndarray
    The testing set indices for that split.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module GroupShuffleSplit : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_splits:int -> ?test_size:[`Float of float | `Int of int | `None] -> ?train_size:[`Float of float | `Int of int | `None] -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> unit -> t
(**
Shuffle-Group(s)-Out cross-validation iterator

Provides randomized train/test indices to split data according to a
third-party provided group. This group information can be used to encode
arbitrary domain specific stratifications of the samples as integers.

For instance the groups could be the year of collection of the samples
and thus allow for cross-validation against time-based splits.

The difference between LeavePGroupsOut and GroupShuffleSplit is that
the former generates splits using all subsets of size ``p`` unique groups,
whereas GroupShuffleSplit generates a user-determined number of random
test splits, each with a user-determined fraction of unique groups.

For example, a less computationally intensive alternative to
``LeavePGroupsOut(p=10)`` would be
``GroupShuffleSplit(test_size=10, n_splits=100)``.

Note: The parameters ``test_size`` and ``train_size`` refer to groups, and
not to samples, as in ShuffleSplit.


Parameters
----------
n_splits : int (default 5)
    Number of re-shuffling & splitting iterations.

test_size : float, int, None, optional (default=None)
    If float, should be between 0.0 and 1.0 and represent the proportion
    of groups to include in the test split (rounded up). If int,
    represents the absolute number of test groups. If None, the value is
    set to the complement of the train size. By default, the value is set
    to 0.2.
    The default will change in version 0.21. It will remain 0.2 only
    if ``train_size`` is unspecified, otherwise it will complement
    the specified ``train_size``.

train_size : float, int, or None, default is None
    If float, should be between 0.0 and 1.0 and represent the
    proportion of the groups to include in the train split. If
    int, represents the absolute number of train groups. If None,
    the value is automatically set to the complement of the test size.

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

Examples
--------
>>> import numpy as np
>>> from sklearn.model_selection import GroupShuffleSplit
>>> X = np.ones(shape=(8, 2))
>>> y = np.ones(shape=(8, 1))
>>> groups = np.array([1, 1, 2, 2, 2, 3, 3, 3])
>>> print(groups.shape)
(8,)
>>> gss = GroupShuffleSplit(n_splits=2, train_size=.7, random_state=42)
>>> gss.get_n_splits()
2
>>> for train_idx, test_idx in gss.split(X, y, groups):
...     print("TRAIN:", train_idx, "TEST:", test_idx)
TRAIN: [2 3 4 5 6 7] TEST: [0 1]
TRAIN: [0 1 5 6 7] TEST: [2 3 4]
*)

val get_n_splits : ?x:Py.Object.t -> ?y:Py.Object.t -> ?groups:Py.Object.t -> t -> int
(**
Returns the number of splitting iterations in the cross-validator

Parameters
----------
X : object
    Always ignored, exists for compatibility.

y : object
    Always ignored, exists for compatibility.

groups : object
    Always ignored, exists for compatibility.

Returns
-------
n_splits : int
    Returns the number of splitting iterations in the cross-validator.
*)

val split : ?y:Ndarray.t -> ?groups:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> x:Ndarray.t -> t -> Py.Object.t
(**
Generate indices to split data into training and test set.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training data, where n_samples is the number of samples
    and n_features is the number of features.

y : array-like, shape (n_samples,), optional
    The target variable for supervised learning problems.

groups : array-like, with shape (n_samples,)
    Group labels for the samples used while splitting the dataset into
    train/test set.

Yields
------
train : ndarray
    The training set indices for that split.

test : ndarray
    The testing set indices for that split.

Notes
-----
Randomized CV splitters may return different results for each call of
split. You can make the results identical by setting ``random_state``
to an integer.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module KFold : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_splits:int -> ?shuffle:bool -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> unit -> t
(**
K-Folds cross-validator

Provides train/test indices to split data in train/test sets. Split
dataset into k consecutive folds (without shuffling by default).

Each fold is then used once as a validation while the k - 1 remaining
folds form the training set.

Read more in the :ref:`User Guide <cross_validation>`.

Parameters
----------
n_splits : int, default=5
    Number of folds. Must be at least 2.

    .. versionchanged:: 0.22
        ``n_splits`` default value changed from 3 to 5.

shuffle : boolean, optional
    Whether to shuffle the data before splitting into batches.

random_state : int, RandomState instance or None, optional, default=None
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`. Only used when ``shuffle`` is True. This should be left
    to None if ``shuffle`` is False.

Examples
--------
>>> import numpy as np
>>> from sklearn.model_selection import KFold
>>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
>>> y = np.array([1, 2, 3, 4])
>>> kf = KFold(n_splits=2)
>>> kf.get_n_splits(X)
2
>>> print(kf)
KFold(n_splits=2, random_state=None, shuffle=False)
>>> for train_index, test_index in kf.split(X):
...     print("TRAIN:", train_index, "TEST:", test_index)
...     X_train, X_test = X[train_index], X[test_index]
...     y_train, y_test = y[train_index], y[test_index]
TRAIN: [2 3] TEST: [0 1]
TRAIN: [0 1] TEST: [2 3]

Notes
-----
The first ``n_samples % n_splits`` folds have size
``n_samples // n_splits + 1``, other folds have size
``n_samples // n_splits``, where ``n_samples`` is the number of samples.

Randomized CV splitters may return different results for each call of
split. You can make the results identical by setting ``random_state``
to an integer.

See also
--------
StratifiedKFold
    Takes group information into account to avoid building folds with
    imbalanced class distributions (for binary or multiclass
    classification tasks).

GroupKFold: K-fold iterator variant with non-overlapping groups.

RepeatedKFold: Repeats K-Fold n times.
*)

val get_n_splits : ?x:Py.Object.t -> ?y:Py.Object.t -> ?groups:Py.Object.t -> t -> int
(**
Returns the number of splitting iterations in the cross-validator

Parameters
----------
X : object
    Always ignored, exists for compatibility.

y : object
    Always ignored, exists for compatibility.

groups : object
    Always ignored, exists for compatibility.

Returns
-------
n_splits : int
    Returns the number of splitting iterations in the cross-validator.
*)

val split : ?y:Ndarray.t -> ?groups:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> x:Ndarray.t -> t -> Py.Object.t
(**
Generate indices to split data into training and test set.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training data, where n_samples is the number of samples
    and n_features is the number of features.

y : array-like, shape (n_samples,)
    The target variable for supervised learning problems.

groups : array-like, with shape (n_samples,), optional
    Group labels for the samples used while splitting the dataset into
    train/test set.

Yields
------
train : ndarray
    The training set indices for that split.

test : ndarray
    The testing set indices for that split.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module LeaveOneGroupOut : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : unit -> t
(**
Leave One Group Out cross-validator

Provides train/test indices to split data according to a third-party
provided group. This group information can be used to encode arbitrary
domain specific stratifications of the samples as integers.

For instance the groups could be the year of collection of the samples
and thus allow for cross-validation against time-based splits.

Read more in the :ref:`User Guide <cross_validation>`.

Examples
--------
>>> import numpy as np
>>> from sklearn.model_selection import LeaveOneGroupOut
>>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
>>> y = np.array([1, 2, 1, 2])
>>> groups = np.array([1, 1, 2, 2])
>>> logo = LeaveOneGroupOut()
>>> logo.get_n_splits(X, y, groups)
2
>>> logo.get_n_splits(groups=groups)  # 'groups' is always required
2
>>> print(logo)
LeaveOneGroupOut()
>>> for train_index, test_index in logo.split(X, y, groups):
...     print("TRAIN:", train_index, "TEST:", test_index)
...     X_train, X_test = X[train_index], X[test_index]
...     y_train, y_test = y[train_index], y[test_index]
...     print(X_train, X_test, y_train, y_test)
TRAIN: [2 3] TEST: [0 1]
[[5 6]
 [7 8]] [[1 2]
 [3 4]] [1 2] [1 2]
TRAIN: [0 1] TEST: [2 3]
[[1 2]
 [3 4]] [[5 6]
 [7 8]] [1 2] [1 2]
*)

val get_n_splits : ?x:Py.Object.t -> ?y:Py.Object.t -> ?groups:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> t -> int
(**
Returns the number of splitting iterations in the cross-validator

Parameters
----------
X : object
    Always ignored, exists for compatibility.

y : object
    Always ignored, exists for compatibility.

groups : array-like, with shape (n_samples,)
    Group labels for the samples used while splitting the dataset into
    train/test set. This 'groups' parameter must always be specified to
    calculate the number of splits, though the other parameters can be
    omitted.

Returns
-------
n_splits : int
    Returns the number of splitting iterations in the cross-validator.
*)

val split : ?y:Ndarray.t -> ?groups:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> x:Ndarray.t -> t -> Py.Object.t
(**
Generate indices to split data into training and test set.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training data, where n_samples is the number of samples
    and n_features is the number of features.

y : array-like, of length n_samples, optional
    The target variable for supervised learning problems.

groups : array-like, with shape (n_samples,)
    Group labels for the samples used while splitting the dataset into
    train/test set.

Yields
------
train : ndarray
    The training set indices for that split.

test : ndarray
    The testing set indices for that split.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module LeaveOneOut : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : unit -> t
(**
Leave-One-Out cross-validator

Provides train/test indices to split data in train/test sets. Each
sample is used once as a test set (singleton) while the remaining
samples form the training set.

Note: ``LeaveOneOut()`` is equivalent to ``KFold(n_splits=n)`` and
``LeavePOut(p=1)`` where ``n`` is the number of samples.

Due to the high number of test sets (which is the same as the
number of samples) this cross-validation method can be very costly.
For large datasets one should favor :class:`KFold`, :class:`ShuffleSplit`
or :class:`StratifiedKFold`.

Read more in the :ref:`User Guide <cross_validation>`.

Examples
--------
>>> import numpy as np
>>> from sklearn.model_selection import LeaveOneOut
>>> X = np.array([[1, 2], [3, 4]])
>>> y = np.array([1, 2])
>>> loo = LeaveOneOut()
>>> loo.get_n_splits(X)
2
>>> print(loo)
LeaveOneOut()
>>> for train_index, test_index in loo.split(X):
...     print("TRAIN:", train_index, "TEST:", test_index)
...     X_train, X_test = X[train_index], X[test_index]
...     y_train, y_test = y[train_index], y[test_index]
...     print(X_train, X_test, y_train, y_test)
TRAIN: [1] TEST: [0]
[[3 4]] [[1 2]] [2] [1]
TRAIN: [0] TEST: [1]
[[1 2]] [[3 4]] [1] [2]

See also
--------
LeaveOneGroupOut
    For splitting the data according to explicit, domain-specific
    stratification of the dataset.

GroupKFold: K-fold iterator variant with non-overlapping groups.
*)

val get_n_splits : ?y:Py.Object.t -> ?groups:Py.Object.t -> x:Ndarray.t -> t -> int
(**
Returns the number of splitting iterations in the cross-validator

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training data, where n_samples is the number of samples
    and n_features is the number of features.

y : object
    Always ignored, exists for compatibility.

groups : object
    Always ignored, exists for compatibility.

Returns
-------
n_splits : int
    Returns the number of splitting iterations in the cross-validator.
*)

val split : ?y:Ndarray.t -> ?groups:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> x:Ndarray.t -> t -> Py.Object.t
(**
Generate indices to split data into training and test set.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training data, where n_samples is the number of samples
    and n_features is the number of features.

y : array-like, of length n_samples
    The target variable for supervised learning problems.

groups : array-like, with shape (n_samples,), optional
    Group labels for the samples used while splitting the dataset into
    train/test set.

Yields
------
train : ndarray
    The training set indices for that split.

test : ndarray
    The testing set indices for that split.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module LeavePGroupsOut : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : n_groups:int -> unit -> t
(**
Leave P Group(s) Out cross-validator

Provides train/test indices to split data according to a third-party
provided group. This group information can be used to encode arbitrary
domain specific stratifications of the samples as integers.

For instance the groups could be the year of collection of the samples
and thus allow for cross-validation against time-based splits.

The difference between LeavePGroupsOut and LeaveOneGroupOut is that
the former builds the test sets with all the samples assigned to
``p`` different values of the groups while the latter uses samples
all assigned the same groups.

Read more in the :ref:`User Guide <cross_validation>`.

Parameters
----------
n_groups : int
    Number of groups (``p``) to leave out in the test split.

Examples
--------
>>> import numpy as np
>>> from sklearn.model_selection import LeavePGroupsOut
>>> X = np.array([[1, 2], [3, 4], [5, 6]])
>>> y = np.array([1, 2, 1])
>>> groups = np.array([1, 2, 3])
>>> lpgo = LeavePGroupsOut(n_groups=2)
>>> lpgo.get_n_splits(X, y, groups)
3
>>> lpgo.get_n_splits(groups=groups)  # 'groups' is always required
3
>>> print(lpgo)
LeavePGroupsOut(n_groups=2)
>>> for train_index, test_index in lpgo.split(X, y, groups):
...     print("TRAIN:", train_index, "TEST:", test_index)
...     X_train, X_test = X[train_index], X[test_index]
...     y_train, y_test = y[train_index], y[test_index]
...     print(X_train, X_test, y_train, y_test)
TRAIN: [2] TEST: [0 1]
[[5 6]] [[1 2]
 [3 4]] [1] [1 2]
TRAIN: [1] TEST: [0 2]
[[3 4]] [[1 2]
 [5 6]] [2] [1 1]
TRAIN: [0] TEST: [1 2]
[[1 2]] [[3 4]
 [5 6]] [1] [2 1]

See also
--------
GroupKFold: K-fold iterator variant with non-overlapping groups.
*)

val get_n_splits : ?x:Py.Object.t -> ?y:Py.Object.t -> ?groups:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> t -> int
(**
Returns the number of splitting iterations in the cross-validator

Parameters
----------
X : object
    Always ignored, exists for compatibility.

y : object
    Always ignored, exists for compatibility.

groups : array-like, with shape (n_samples,)
    Group labels for the samples used while splitting the dataset into
    train/test set. This 'groups' parameter must always be specified to
    calculate the number of splits, though the other parameters can be
    omitted.

Returns
-------
n_splits : int
    Returns the number of splitting iterations in the cross-validator.
*)

val split : ?y:Ndarray.t -> ?groups:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> x:Ndarray.t -> t -> Py.Object.t
(**
Generate indices to split data into training and test set.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training data, where n_samples is the number of samples
    and n_features is the number of features.

y : array-like, of length n_samples, optional
    The target variable for supervised learning problems.

groups : array-like, with shape (n_samples,)
    Group labels for the samples used while splitting the dataset into
    train/test set.

Yields
------
train : ndarray
    The training set indices for that split.

test : ndarray
    The testing set indices for that split.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module LeavePOut : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : p:int -> unit -> t
(**
Leave-P-Out cross-validator

Provides train/test indices to split data in train/test sets. This results
in testing on all distinct samples of size p, while the remaining n - p
samples form the training set in each iteration.

Note: ``LeavePOut(p)`` is NOT equivalent to
``KFold(n_splits=n_samples // p)`` which creates non-overlapping test sets.

Due to the high number of iterations which grows combinatorically with the
number of samples this cross-validation method can be very costly. For
large datasets one should favor :class:`KFold`, :class:`StratifiedKFold`
or :class:`ShuffleSplit`.

Read more in the :ref:`User Guide <cross_validation>`.

Parameters
----------
p : int
    Size of the test sets. Must be strictly less than the number of
    samples.

Examples
--------
>>> import numpy as np
>>> from sklearn.model_selection import LeavePOut
>>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
>>> y = np.array([1, 2, 3, 4])
>>> lpo = LeavePOut(2)
>>> lpo.get_n_splits(X)
6
>>> print(lpo)
LeavePOut(p=2)
>>> for train_index, test_index in lpo.split(X):
...     print("TRAIN:", train_index, "TEST:", test_index)
...     X_train, X_test = X[train_index], X[test_index]
...     y_train, y_test = y[train_index], y[test_index]
TRAIN: [2 3] TEST: [0 1]
TRAIN: [1 3] TEST: [0 2]
TRAIN: [1 2] TEST: [0 3]
TRAIN: [0 3] TEST: [1 2]
TRAIN: [0 2] TEST: [1 3]
TRAIN: [0 1] TEST: [2 3]
*)

val get_n_splits : ?y:Py.Object.t -> ?groups:Py.Object.t -> x:Ndarray.t -> t -> Py.Object.t
(**
Returns the number of splitting iterations in the cross-validator

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training data, where n_samples is the number of samples
    and n_features is the number of features.

y : object
    Always ignored, exists for compatibility.

groups : object
    Always ignored, exists for compatibility.
*)

val split : ?y:Ndarray.t -> ?groups:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> x:Ndarray.t -> t -> Py.Object.t
(**
Generate indices to split data into training and test set.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training data, where n_samples is the number of samples
    and n_features is the number of features.

y : array-like, of length n_samples
    The target variable for supervised learning problems.

groups : array-like, with shape (n_samples,), optional
    Group labels for the samples used while splitting the dataset into
    train/test set.

Yields
------
train : ndarray
    The training set indices for that split.

test : ndarray
    The testing set indices for that split.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module ParameterGrid : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : param_grid:Py.Object.t -> unit -> t
(**
Grid of parameters with a discrete number of values for each.

Can be used to iterate over parameter value combinations with the
Python built-in function iter.

Read more in the :ref:`User Guide <grid_search>`.

Parameters
----------
param_grid : dict of string to sequence, or sequence of such
    The parameter grid to explore, as a dictionary mapping estimator
    parameters to sequences of allowed values.

    An empty dict signifies default parameters.

    A sequence of dicts signifies a sequence of grids to search, and is
    useful to avoid exploring parameter combinations that make no sense
    or have no effect. See the examples below.

Examples
--------
>>> from sklearn.model_selection import ParameterGrid
>>> param_grid = {'a': [1, 2], 'b': [True, False]}
>>> list(ParameterGrid(param_grid)) == (
...    [{'a': 1, 'b': True}, {'a': 1, 'b': False},
...     {'a': 2, 'b': True}, {'a': 2, 'b': False}])
True

>>> grid = [{'kernel': ['linear']}, {'kernel': ['rbf'], 'gamma': [1, 10]}]
>>> list(ParameterGrid(grid)) == [{'kernel': 'linear'},
...                               {'kernel': 'rbf', 'gamma': 1},
...                               {'kernel': 'rbf', 'gamma': 10}]
True
>>> ParameterGrid(grid)[1] == {'kernel': 'rbf', 'gamma': 1}
True

See also
--------
:class:`GridSearchCV`:
    Uses :class:`ParameterGrid` to perform a full parallelized parameter
    search.
*)

val get_item : ind:int -> t -> Py.Object.t
(**
Get the parameters that would be ``ind``th in iteration

Parameters
----------
ind : int
    The iteration index

Returns
-------
params : dict of string to any
    Equal to list(self)[ind]
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module ParameterSampler : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> param_distributions:Py.Object.t -> n_iter:int -> unit -> t
(**
Generator on parameters sampled from given distributions.

Non-deterministic iterable over random candidate combinations for hyper-
parameter search. If all parameters are presented as a list,
sampling without replacement is performed. If at least one parameter
is given as a distribution, sampling with replacement is used.
It is highly recommended to use continuous distributions for continuous
parameters.

Read more in the :ref:`User Guide <search>`.

Parameters
----------
param_distributions : dict
    Dictionary with parameters names (string) as keys and distributions
    or lists of parameters to try. Distributions must provide a ``rvs``
    method for sampling (such as those from scipy.stats.distributions).
    If a list is given, it is sampled uniformly.
    If a list of dicts is given, first a dict is sampled uniformly, and
    then a parameter is sampled using that dict as above.

n_iter : integer
    Number of parameter settings that are produced.

random_state : int, RandomState instance or None, optional (default=None)
    Pseudo random number generator state used for random uniform sampling
    from lists of possible values instead of scipy.stats distributions.
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

Returns
-------
params : dict of string to any
    **Yields** dictionaries mapping each estimator parameter to
    as sampled value.

Examples
--------
>>> from sklearn.model_selection import ParameterSampler
>>> from scipy.stats.distributions import expon
>>> import numpy as np
>>> rng = np.random.RandomState(0)
>>> param_grid = {'a':[1, 2], 'b': expon()}
>>> param_list = list(ParameterSampler(param_grid, n_iter=4,
...                                    random_state=rng))
>>> rounded_list = [dict((k, round(v, 6)) for (k, v) in d.items())
...                 for d in param_list]
>>> rounded_list == [{'b': 0.89856, 'a': 1},
...                  {'b': 0.923223, 'a': 1},
...                  {'b': 1.878964, 'a': 2},
...                  {'b': 1.038159, 'a': 2}]
True
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module PredefinedSplit : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : test_fold:Ndarray.t -> unit -> t
(**
Predefined split cross-validator

Provides train/test indices to split data into train/test sets using a
predefined scheme specified by the user with the ``test_fold`` parameter.

Read more in the :ref:`User Guide <cross_validation>`.

.. versionadded:: 0.16

Parameters
----------
test_fold : array-like, shape (n_samples,)
    The entry ``test_fold[i]`` represents the index of the test set that
    sample ``i`` belongs to. It is possible to exclude sample ``i`` from
    any test set (i.e. include sample ``i`` in every training set) by
    setting ``test_fold[i]`` equal to -1.

Examples
--------
>>> import numpy as np
>>> from sklearn.model_selection import PredefinedSplit
>>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
>>> y = np.array([0, 0, 1, 1])
>>> test_fold = [0, 1, -1, 1]
>>> ps = PredefinedSplit(test_fold)
>>> ps.get_n_splits()
2
>>> print(ps)
PredefinedSplit(test_fold=array([ 0,  1, -1,  1]))
>>> for train_index, test_index in ps.split():
...     print("TRAIN:", train_index, "TEST:", test_index)
...     X_train, X_test = X[train_index], X[test_index]
...     y_train, y_test = y[train_index], y[test_index]
TRAIN: [1 2 3] TEST: [0]
TRAIN: [0 2] TEST: [1 3]
*)

val get_n_splits : ?x:Py.Object.t -> ?y:Py.Object.t -> ?groups:Py.Object.t -> t -> int
(**
Returns the number of splitting iterations in the cross-validator

Parameters
----------
X : object
    Always ignored, exists for compatibility.

y : object
    Always ignored, exists for compatibility.

groups : object
    Always ignored, exists for compatibility.

Returns
-------
n_splits : int
    Returns the number of splitting iterations in the cross-validator.
*)

val split : ?x:Py.Object.t -> ?y:Py.Object.t -> ?groups:Py.Object.t -> t -> Py.Object.t
(**
Generate indices to split data into training and test set.

Parameters
----------
X : object
    Always ignored, exists for compatibility.

y : object
    Always ignored, exists for compatibility.

groups : object
    Always ignored, exists for compatibility.

Yields
------
train : ndarray
    The training set indices for that split.

test : ndarray
    The testing set indices for that split.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module RandomizedSearchCV : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_iter:int -> ?scoring:[`String of string | `Callable of Py.Object.t | `Dict of Py.Object.t | `None | `PyObject of Py.Object.t] -> ?n_jobs:[`Int of int | `None] -> ?iid:bool -> ?refit:[`Bool of bool | `String of string | `Callable of Py.Object.t] -> ?cv:[`Int of int | `CrossValGenerator of Py.Object.t | `Ndarray of Ndarray.t] -> ?verbose:int -> ?pre_dispatch:[`Int of int | `String of string] -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> ?error_score:[`Raise | `PyObject of Py.Object.t] -> ?return_train_score:bool -> estimator:Py.Object.t -> param_distributions:[`Dict of Py.Object.t | `PyObject of Py.Object.t] -> unit -> t
(**
Randomized search on hyper parameters.

RandomizedSearchCV implements a "fit" and a "score" method.
It also implements "predict", "predict_proba", "decision_function",
"transform" and "inverse_transform" if they are implemented in the
estimator used.

The parameters of the estimator used to apply these methods are optimized
by cross-validated search over parameter settings.

In contrast to GridSearchCV, not all parameter values are tried out, but
rather a fixed number of parameter settings is sampled from the specified
distributions. The number of parameter settings that are tried is
given by n_iter.

If all parameters are presented as a list,
sampling without replacement is performed. If at least one parameter
is given as a distribution, sampling with replacement is used.
It is highly recommended to use continuous distributions for continuous
parameters.

Read more in the :ref:`User Guide <randomized_parameter_search>`.

.. versionadded:: 0.14

Parameters
----------
estimator : estimator object.
    A object of that type is instantiated for each grid point.
    This is assumed to implement the scikit-learn estimator interface.
    Either estimator needs to provide a ``score`` function,
    or ``scoring`` must be passed.

param_distributions : dict or list of dicts
    Dictionary with parameters names (string) as keys and distributions
    or lists of parameters to try. Distributions must provide a ``rvs``
    method for sampling (such as those from scipy.stats.distributions).
    If a list is given, it is sampled uniformly.
    If a list of dicts is given, first a dict is sampled uniformly, and
    then a parameter is sampled using that dict as above.

n_iter : int, default=10
    Number of parameter settings that are sampled. n_iter trades
    off runtime vs quality of the solution.

scoring : string, callable, list/tuple, dict or None, default: None
    A single string (see :ref:`scoring_parameter`) or a callable
    (see :ref:`scoring`) to evaluate the predictions on the test set.

    For evaluating multiple metrics, either give a list of (unique) strings
    or a dict with names as keys and callables as values.

    NOTE that when using custom scorers, each scorer should return a single
    value. Metric functions returning a list/array of values can be wrapped
    into multiple scorers that return one value each.

    See :ref:`multimetric_grid_search` for an example.

    If None, the estimator's score method is used.

n_jobs : int or None, optional (default=None)
    Number of jobs to run in parallel.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

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

iid : boolean, default=False
    If True, return the average score across folds, weighted by the number
    of samples in each test set. In this case, the data is assumed to be
    identically distributed across the folds, and the loss minimized is
    the total loss per sample, and not the mean loss across the folds.

    .. deprecated:: 0.22
        Parameter ``iid`` is deprecated in 0.22 and will be removed in 0.24

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

refit : boolean, string, or callable, default=True
    Refit an estimator using the best found parameters on the whole
    dataset.

    For multiple metric evaluation, this needs to be a string denoting the
    scorer that would be used to find the best parameters for refitting
    the estimator at the end.

    Where there are considerations other than maximum score in
    choosing a best estimator, ``refit`` can be set to a function which
    returns the selected ``best_index_`` given the ``cv_results``. In that
    case, the ``best_estimator_`` and ``best_parameters_`` will be set
    according to the returned ``best_index_`` while the ``best_score_``
    attribute will not be available.

    The refitted estimator is made available at the ``best_estimator_``
    attribute and permits using ``predict`` directly on this
    ``RandomizedSearchCV`` instance.

    Also for multiple metric evaluation, the attributes ``best_index_``,
    ``best_score_`` and ``best_params_`` will only be available if
    ``refit`` is set and all of them will be determined w.r.t this specific
    scorer.

    See ``scoring`` parameter to know more about multiple metric
    evaluation.

    .. versionchanged:: 0.20
        Support for callable added.

verbose : integer
    Controls the verbosity: the higher, the more messages.

random_state : int, RandomState instance or None, optional, default=None
    Pseudo random number generator state used for random uniform sampling
    from lists of possible values instead of scipy.stats distributions.
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

error_score : 'raise' or numeric
    Value to assign to the score if an error occurs in estimator fitting.
    If set to 'raise', the error is raised. If a numeric value is given,
    FitFailedWarning is raised. This parameter does not affect the refit
    step, which will always raise the error. Default is ``np.nan``.

return_train_score : boolean, default=False
    If ``False``, the ``cv_results_`` attribute will not include training
    scores.
    Computing training scores is used to get insights on how different
    parameter settings impact the overfitting/underfitting trade-off.
    However computing the scores on the training set can be computationally
    expensive and is not strictly required to select the parameters that
    yield the best generalization performance.

Attributes
----------
cv_results_ : dict of numpy (masked) ndarrays
    A dict with keys as column headers and values as columns, that can be
    imported into a pandas ``DataFrame``.

    For instance the below given table

    +--------------+-------------+-------------------+---+---------------+
    | param_kernel | param_gamma | split0_test_score |...|rank_test_score|
    +==============+=============+===================+===+===============+
    |    'rbf'     |     0.1     |       0.80        |...|       2       |
    +--------------+-------------+-------------------+---+---------------+
    |    'rbf'     |     0.2     |       0.90        |...|       1       |
    +--------------+-------------+-------------------+---+---------------+
    |    'rbf'     |     0.3     |       0.70        |...|       1       |
    +--------------+-------------+-------------------+---+---------------+

    will be represented by a ``cv_results_`` dict of::

        {
        'param_kernel' : masked_array(data = ['rbf', 'rbf', 'rbf'],
                                      mask = False),
        'param_gamma'  : masked_array(data = [0.1 0.2 0.3], mask = False),
        'split0_test_score'  : [0.80, 0.90, 0.70],
        'split1_test_score'  : [0.82, 0.50, 0.70],
        'mean_test_score'    : [0.81, 0.70, 0.70],
        'std_test_score'     : [0.01, 0.20, 0.00],
        'rank_test_score'    : [3, 1, 1],
        'split0_train_score' : [0.80, 0.92, 0.70],
        'split1_train_score' : [0.82, 0.55, 0.70],
        'mean_train_score'   : [0.81, 0.74, 0.70],
        'std_train_score'    : [0.01, 0.19, 0.00],
        'mean_fit_time'      : [0.73, 0.63, 0.43],
        'std_fit_time'       : [0.01, 0.02, 0.01],
        'mean_score_time'    : [0.01, 0.06, 0.04],
        'std_score_time'     : [0.00, 0.00, 0.00],
        'params'             : [{'kernel' : 'rbf', 'gamma' : 0.1}, ...],
        }

    NOTE

    The key ``'params'`` is used to store a list of parameter
    settings dicts for all the parameter candidates.

    The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
    ``std_score_time`` are all in seconds.

    For multi-metric evaluation, the scores for all the scorers are
    available in the ``cv_results_`` dict at the keys ending with that
    scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
    above. ('split0_test_precision', 'mean_train_precision' etc.)

best_estimator_ : estimator
    Estimator that was chosen by the search, i.e. estimator
    which gave highest score (or smallest loss if specified)
    on the left out data. Not available if ``refit=False``.

    For multi-metric evaluation, this attribute is present only if
    ``refit`` is specified.

    See ``refit`` parameter for more information on allowed values.

best_score_ : float
    Mean cross-validated score of the best_estimator.

    For multi-metric evaluation, this is not available if ``refit`` is
    ``False``. See ``refit`` parameter for more information.

    This attribute is not available if ``refit`` is a function.

best_params_ : dict
    Parameter setting that gave the best results on the hold out data.

    For multi-metric evaluation, this is not available if ``refit`` is
    ``False``. See ``refit`` parameter for more information.

best_index_ : int
    The index (of the ``cv_results_`` arrays) which corresponds to the best
    candidate parameter setting.

    The dict at ``search.cv_results_['params'][search.best_index_]`` gives
    the parameter setting for the best model, that gives the highest
    mean score (``search.best_score_``).

    For multi-metric evaluation, this is not available if ``refit`` is
    ``False``. See ``refit`` parameter for more information.

scorer_ : function or a dict
    Scorer function used on the held out data to choose the best
    parameters for the model.

    For multi-metric evaluation, this attribute holds the validated
    ``scoring`` dict which maps the scorer key to the scorer callable.

n_splits_ : int
    The number of cross-validation splits (folds/iterations).

refit_time_ : float
    Seconds used for refitting the best model on the whole dataset.

    This is present only if ``refit`` is not False.

Notes
-----
The parameters selected are those that maximize the score of the held-out
data, according to the scoring parameter.

If `n_jobs` was set to a value higher than one, the data is copied for each
parameter setting(and not `n_jobs` times). This is done for efficiency
reasons if individual jobs take very little time, but may raise errors if
the dataset is large and not enough memory is available.  A workaround in
this case is to set `pre_dispatch`. Then, the memory is copied only
`pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
n_jobs`.

See Also
--------
:class:`GridSearchCV`:
    Does exhaustive search over a grid of parameters.

:class:`ParameterSampler`:
    A generator over parameter settings, constructed from
    param_distributions.


Examples
--------
>>> from sklearn.datasets import load_iris
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.model_selection import RandomizedSearchCV
>>> from scipy.stats import uniform
>>> iris = load_iris()
>>> logistic = LogisticRegression(solver='saga', tol=1e-2, max_iter=200,
...                               random_state=0)
>>> distributions = dict(C=uniform(loc=0, scale=4),
...                      penalty=['l2', 'l1'])
>>> clf = RandomizedSearchCV(logistic, distributions, random_state=0)
>>> search = clf.fit(iris.data, iris.target)
>>> search.best_params_
{'C': 2..., 'penalty': 'l1'}
*)

val decision_function : x:Ndarray.t -> t -> Ndarray.t
(**
Call decision_function on the estimator with the best found parameters.

Only available if ``refit=True`` and the underlying estimator supports
``decision_function``.

Parameters
----------
X : indexable, length n_samples
    Must fulfill the input assumptions of the
    underlying estimator.
*)

val fit : ?y:Ndarray.t -> ?groups:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> ?fit_params:(string * Py.Object.t) list -> x:Ndarray.t -> t -> t
(**
Run fit with all sets of parameters.

Parameters
----------

X : array-like of shape (n_samples, n_features)
    Training vector, where n_samples is the number of samples and
    n_features is the number of features.

y : array-like of shape (n_samples, n_output) or (n_samples,), optional
    Target relative to X for classification or regression;
    None for unsupervised learning.

groups : array-like, with shape (n_samples,), optional
    Group labels for the samples used while splitting the dataset into
    train/test set. Only used in conjunction with a "Group" :term:`cv`
    instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).

**fit_params : dict of string -> object
    Parameters passed to the ``fit`` method of the estimator
*)

val get_params : ?deep:bool -> t -> Py.Object.t
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

val inverse_transform : xt:Ndarray.t -> t -> Py.Object.t
(**
Call inverse_transform on the estimator with the best found params.

Only available if the underlying estimator implements
``inverse_transform`` and ``refit=True``.

Parameters
----------
Xt : indexable, length n_samples
    Must fulfill the input assumptions of the
    underlying estimator.
*)

val predict : x:Ndarray.t -> t -> Ndarray.t
(**
Call predict on the estimator with the best found parameters.

Only available if ``refit=True`` and the underlying estimator supports
``predict``.

Parameters
----------
X : indexable, length n_samples
    Must fulfill the input assumptions of the
    underlying estimator.
*)

val predict_log_proba : x:Ndarray.t -> t -> Py.Object.t
(**
Call predict_log_proba on the estimator with the best found parameters.

Only available if ``refit=True`` and the underlying estimator supports
``predict_log_proba``.

Parameters
----------
X : indexable, length n_samples
    Must fulfill the input assumptions of the
    underlying estimator.
*)

val predict_proba : x:Ndarray.t -> t -> Ndarray.t
(**
Call predict_proba on the estimator with the best found parameters.

Only available if ``refit=True`` and the underlying estimator supports
``predict_proba``.

Parameters
----------
X : indexable, length n_samples
    Must fulfill the input assumptions of the
    underlying estimator.
*)

val score : ?y:Ndarray.t -> x:Ndarray.t -> t -> float
(**
Returns the score on the given data, if the estimator has been refit.

This uses the score defined by ``scoring`` where provided, and the
``best_estimator_.score`` method otherwise.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Input data, where n_samples is the number of samples and
    n_features is the number of features.

y : array-like of shape (n_samples, n_output) or (n_samples,), optional
    Target relative to X for classification or regression;
    None for unsupervised learning.

Returns
-------
score : float
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

val transform : x:Ndarray.t -> t -> Ndarray.t
(**
Call transform on the estimator with the best found parameters.

Only available if the underlying estimator supports ``transform`` and
``refit=True``.

Parameters
----------
X : indexable, length n_samples
    Must fulfill the input assumptions of the
    underlying estimator.
*)


(** Attribute cv_results_: see constructor for documentation *)
val cv_results_ : t -> Py.Object.t

(** Attribute best_estimator_: see constructor for documentation *)
val best_estimator_ : t -> Py.Object.t

(** Attribute best_score_: see constructor for documentation *)
val best_score_ : t -> float

(** Attribute best_params_: see constructor for documentation *)
val best_params_ : t -> Py.Object.t

(** Attribute best_index_: see constructor for documentation *)
val best_index_ : t -> int

(** Attribute scorer_: see constructor for documentation *)
val scorer_ : t -> Py.Object.t

(** Attribute n_splits_: see constructor for documentation *)
val n_splits_ : t -> int

(** Attribute refit_time_: see constructor for documentation *)
val refit_time_ : t -> float

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module RepeatedKFold : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_splits:int -> ?n_repeats:int -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> unit -> t
(**
Repeated K-Fold cross validator.

Repeats K-Fold n times with different randomization in each repetition.

Read more in the :ref:`User Guide <cross_validation>`.

Parameters
----------
n_splits : int, default=5
    Number of folds. Must be at least 2.

n_repeats : int, default=10
    Number of times cross-validator needs to be repeated.

random_state : int, RandomState instance or None, optional, default=None
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

Examples
--------
>>> import numpy as np
>>> from sklearn.model_selection import RepeatedKFold
>>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
>>> y = np.array([0, 0, 1, 1])
>>> rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=2652124)
>>> for train_index, test_index in rkf.split(X):
...     print("TRAIN:", train_index, "TEST:", test_index)
...     X_train, X_test = X[train_index], X[test_index]
...     y_train, y_test = y[train_index], y[test_index]
...
TRAIN: [0 1] TEST: [2 3]
TRAIN: [2 3] TEST: [0 1]
TRAIN: [1 2] TEST: [0 3]
TRAIN: [0 3] TEST: [1 2]

Notes
-----
Randomized CV splitters may return different results for each call of
split. You can make the results identical by setting ``random_state``
to an integer.

See also
--------
RepeatedStratifiedKFold: Repeats Stratified K-Fold n times.
*)

val get_n_splits : ?x:Py.Object.t -> ?y:Py.Object.t -> ?groups:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> t -> int
(**
Returns the number of splitting iterations in the cross-validator

Parameters
----------
X : object
    Always ignored, exists for compatibility.
    ``np.zeros(n_samples)`` may be used as a placeholder.

y : object
    Always ignored, exists for compatibility.
    ``np.zeros(n_samples)`` may be used as a placeholder.

groups : array-like, with shape (n_samples,), optional
    Group labels for the samples used while splitting the dataset into
    train/test set.

Returns
-------
n_splits : int
    Returns the number of splitting iterations in the cross-validator.
*)

val split : ?y:Ndarray.t -> ?groups:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> x:Ndarray.t -> t -> Py.Object.t
(**
Generates indices to split data into training and test set.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training data, where n_samples is the number of samples
    and n_features is the number of features.

y : array-like, of length n_samples
    The target variable for supervised learning problems.

groups : array-like, with shape (n_samples,), optional
    Group labels for the samples used while splitting the dataset into
    train/test set.

Yields
------
train : ndarray
    The training set indices for that split.

test : ndarray
    The testing set indices for that split.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module RepeatedStratifiedKFold : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_splits:int -> ?n_repeats:int -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> unit -> t
(**
Repeated Stratified K-Fold cross validator.

Repeats Stratified K-Fold n times with different randomization in each
repetition.

Read more in the :ref:`User Guide <cross_validation>`.

Parameters
----------
n_splits : int, default=5
    Number of folds. Must be at least 2.

n_repeats : int, default=10
    Number of times cross-validator needs to be repeated.

random_state : None, int or RandomState, default=None
    Random state to be used to generate random state for each
    repetition.

Examples
--------
>>> import numpy as np
>>> from sklearn.model_selection import RepeatedStratifiedKFold
>>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
>>> y = np.array([0, 0, 1, 1])
>>> rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=2,
...     random_state=36851234)
>>> for train_index, test_index in rskf.split(X, y):
...     print("TRAIN:", train_index, "TEST:", test_index)
...     X_train, X_test = X[train_index], X[test_index]
...     y_train, y_test = y[train_index], y[test_index]
...
TRAIN: [1 2] TEST: [0 3]
TRAIN: [0 3] TEST: [1 2]
TRAIN: [1 3] TEST: [0 2]
TRAIN: [0 2] TEST: [1 3]

Notes
-----
Randomized CV splitters may return different results for each call of
split. You can make the results identical by setting ``random_state``
to an integer.

See also
--------
RepeatedKFold: Repeats K-Fold n times.
*)

val get_n_splits : ?x:Py.Object.t -> ?y:Py.Object.t -> ?groups:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> t -> int
(**
Returns the number of splitting iterations in the cross-validator

Parameters
----------
X : object
    Always ignored, exists for compatibility.
    ``np.zeros(n_samples)`` may be used as a placeholder.

y : object
    Always ignored, exists for compatibility.
    ``np.zeros(n_samples)`` may be used as a placeholder.

groups : array-like, with shape (n_samples,), optional
    Group labels for the samples used while splitting the dataset into
    train/test set.

Returns
-------
n_splits : int
    Returns the number of splitting iterations in the cross-validator.
*)

val split : ?y:Ndarray.t -> ?groups:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> x:Ndarray.t -> t -> Py.Object.t
(**
Generates indices to split data into training and test set.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training data, where n_samples is the number of samples
    and n_features is the number of features.

y : array-like, of length n_samples
    The target variable for supervised learning problems.

groups : array-like, with shape (n_samples,), optional
    Group labels for the samples used while splitting the dataset into
    train/test set.

Yields
------
train : ndarray
    The training set indices for that split.

test : ndarray
    The testing set indices for that split.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module ShuffleSplit : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_splits:int -> ?test_size:[`Float of float | `Int of int | `None] -> ?train_size:[`Float of float | `Int of int | `None] -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> unit -> t
(**
Random permutation cross-validator

Yields indices to split data into training and test sets.

Note: contrary to other cross-validation strategies, random splits
do not guarantee that all folds will be different, although this is
still very likely for sizeable datasets.

Read more in the :ref:`User Guide <cross_validation>`.

Parameters
----------
n_splits : int, default 10
    Number of re-shuffling & splitting iterations.

test_size : float, int, None, default=None
    If float, should be between 0.0 and 1.0 and represent the proportion
    of the dataset to include in the test split. If int, represents the
    absolute number of test samples. If None, the value is set to the
    complement of the train size. If ``train_size`` is also None, it will
    be set to 0.1.

train_size : float, int, or None, default=None
    If float, should be between 0.0 and 1.0 and represent the
    proportion of the dataset to include in the train split. If
    int, represents the absolute number of train samples. If None,
    the value is automatically set to the complement of the test size.

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

Examples
--------
>>> import numpy as np
>>> from sklearn.model_selection import ShuffleSplit
>>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [3, 4], [5, 6]])
>>> y = np.array([1, 2, 1, 2, 1, 2])
>>> rs = ShuffleSplit(n_splits=5, test_size=.25, random_state=0)
>>> rs.get_n_splits(X)
5
>>> print(rs)
ShuffleSplit(n_splits=5, random_state=0, test_size=0.25, train_size=None)
>>> for train_index, test_index in rs.split(X):
...     print("TRAIN:", train_index, "TEST:", test_index)
TRAIN: [1 3 0 4] TEST: [5 2]
TRAIN: [4 0 2 5] TEST: [1 3]
TRAIN: [1 2 4 0] TEST: [3 5]
TRAIN: [3 4 1 0] TEST: [5 2]
TRAIN: [3 5 1 0] TEST: [2 4]
>>> rs = ShuffleSplit(n_splits=5, train_size=0.5, test_size=.25,
...                   random_state=0)
>>> for train_index, test_index in rs.split(X):
...     print("TRAIN:", train_index, "TEST:", test_index)
TRAIN: [1 3 0] TEST: [5 2]
TRAIN: [4 0 2] TEST: [1 3]
TRAIN: [1 2 4] TEST: [3 5]
TRAIN: [3 4 1] TEST: [5 2]
TRAIN: [3 5 1] TEST: [2 4]
*)

val get_n_splits : ?x:Py.Object.t -> ?y:Py.Object.t -> ?groups:Py.Object.t -> t -> int
(**
Returns the number of splitting iterations in the cross-validator

Parameters
----------
X : object
    Always ignored, exists for compatibility.

y : object
    Always ignored, exists for compatibility.

groups : object
    Always ignored, exists for compatibility.

Returns
-------
n_splits : int
    Returns the number of splitting iterations in the cross-validator.
*)

val split : ?y:Ndarray.t -> ?groups:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> x:Ndarray.t -> t -> Py.Object.t
(**
Generate indices to split data into training and test set.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training data, where n_samples is the number of samples
    and n_features is the number of features.

y : array-like, shape (n_samples,)
    The target variable for supervised learning problems.

groups : array-like, with shape (n_samples,), optional
    Group labels for the samples used while splitting the dataset into
    train/test set.

Yields
------
train : ndarray
    The training set indices for that split.

test : ndarray
    The testing set indices for that split.

Notes
-----
Randomized CV splitters may return different results for each call of
split. You can make the results identical by setting ``random_state``
to an integer.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module StratifiedKFold : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_splits:int -> ?shuffle:bool -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> unit -> t
(**
Stratified K-Folds cross-validator

Provides train/test indices to split data in train/test sets.

This cross-validation object is a variation of KFold that returns
stratified folds. The folds are made by preserving the percentage of
samples for each class.

Read more in the :ref:`User Guide <cross_validation>`.

Parameters
----------
n_splits : int, default=5
    Number of folds. Must be at least 2.

    .. versionchanged:: 0.22
        ``n_splits`` default value changed from 3 to 5.

shuffle : boolean, optional
    Whether to shuffle each class's samples before splitting into batches.

random_state : int, RandomState instance or None, optional, default=None
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`. Only used when ``shuffle`` is True. This should be left
    to None if ``shuffle`` is False.

Examples
--------
>>> import numpy as np
>>> from sklearn.model_selection import StratifiedKFold
>>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
>>> y = np.array([0, 0, 1, 1])
>>> skf = StratifiedKFold(n_splits=2)
>>> skf.get_n_splits(X, y)
2
>>> print(skf)
StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
>>> for train_index, test_index in skf.split(X, y):
...     print("TRAIN:", train_index, "TEST:", test_index)
...     X_train, X_test = X[train_index], X[test_index]
...     y_train, y_test = y[train_index], y[test_index]
TRAIN: [1 3] TEST: [0 2]
TRAIN: [0 2] TEST: [1 3]

Notes
-----
The implementation is designed to:

* Generate test sets such that all contain the same distribution of
  classes, or as close as possible.
* Be invariant to class label: relabelling ``y = ["Happy", "Sad"]`` to
  ``y = [1, 0]`` should not change the indices generated.
* Preserve order dependencies in the dataset ordering, when
  ``shuffle=False``: all samples from class k in some test set were
  contiguous in y, or separated in y by samples from classes other than k.
* Generate test sets where the smallest and largest differ by at most one
  sample.

.. versionchanged:: 0.22
    The previous implementation did not follow the last constraint.

See also
--------
RepeatedStratifiedKFold: Repeats Stratified K-Fold n times.
*)

val get_n_splits : ?x:Py.Object.t -> ?y:Py.Object.t -> ?groups:Py.Object.t -> t -> int
(**
Returns the number of splitting iterations in the cross-validator

Parameters
----------
X : object
    Always ignored, exists for compatibility.

y : object
    Always ignored, exists for compatibility.

groups : object
    Always ignored, exists for compatibility.

Returns
-------
n_splits : int
    Returns the number of splitting iterations in the cross-validator.
*)

val split : ?groups:Py.Object.t -> x:Ndarray.t -> y:Ndarray.t -> t -> Py.Object.t
(**
Generate indices to split data into training and test set.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training data, where n_samples is the number of samples
    and n_features is the number of features.

    Note that providing ``y`` is sufficient to generate the splits and
    hence ``np.zeros(n_samples)`` may be used as a placeholder for
    ``X`` instead of actual training data.

y : array-like, shape (n_samples,)
    The target variable for supervised learning problems.
    Stratification is done based on the y labels.

groups : object
    Always ignored, exists for compatibility.

Yields
------
train : ndarray
    The training set indices for that split.

test : ndarray
    The testing set indices for that split.

Notes
-----
Randomized CV splitters may return different results for each call of
split. You can make the results identical by setting ``random_state``
to an integer.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module StratifiedShuffleSplit : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_splits:int -> ?test_size:[`Float of float | `Int of int | `None] -> ?train_size:[`Float of float | `Int of int | `None] -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> unit -> t
(**
Stratified ShuffleSplit cross-validator

Provides train/test indices to split data in train/test sets.

This cross-validation object is a merge of StratifiedKFold and
ShuffleSplit, which returns stratified randomized folds. The folds
are made by preserving the percentage of samples for each class.

Note: like the ShuffleSplit strategy, stratified random splits
do not guarantee that all folds will be different, although this is
still very likely for sizeable datasets.

Read more in the :ref:`User Guide <cross_validation>`.

Parameters
----------
n_splits : int, default 10
    Number of re-shuffling & splitting iterations.

test_size : float, int, None, optional (default=None)
    If float, should be between 0.0 and 1.0 and represent the proportion
    of the dataset to include in the test split. If int, represents the
    absolute number of test samples. If None, the value is set to the
    complement of the train size. If ``train_size`` is also None, it will
    be set to 0.1.

train_size : float, int, or None, default is None
    If float, should be between 0.0 and 1.0 and represent the
    proportion of the dataset to include in the train split. If
    int, represents the absolute number of train samples. If None,
    the value is automatically set to the complement of the test size.

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

Examples
--------
>>> import numpy as np
>>> from sklearn.model_selection import StratifiedShuffleSplit
>>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
>>> y = np.array([0, 0, 0, 1, 1, 1])
>>> sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
>>> sss.get_n_splits(X, y)
5
>>> print(sss)
StratifiedShuffleSplit(n_splits=5, random_state=0, ...)
>>> for train_index, test_index in sss.split(X, y):
...     print("TRAIN:", train_index, "TEST:", test_index)
...     X_train, X_test = X[train_index], X[test_index]
...     y_train, y_test = y[train_index], y[test_index]
TRAIN: [5 2 3] TEST: [4 1 0]
TRAIN: [5 1 4] TEST: [0 2 3]
TRAIN: [5 0 2] TEST: [4 3 1]
TRAIN: [4 1 0] TEST: [2 3 5]
TRAIN: [0 5 1] TEST: [3 4 2]
*)

val get_n_splits : ?x:Py.Object.t -> ?y:Py.Object.t -> ?groups:Py.Object.t -> t -> int
(**
Returns the number of splitting iterations in the cross-validator

Parameters
----------
X : object
    Always ignored, exists for compatibility.

y : object
    Always ignored, exists for compatibility.

groups : object
    Always ignored, exists for compatibility.

Returns
-------
n_splits : int
    Returns the number of splitting iterations in the cross-validator.
*)

val split : ?groups:Py.Object.t -> x:Ndarray.t -> y:Ndarray.t -> t -> Py.Object.t
(**
Generate indices to split data into training and test set.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training data, where n_samples is the number of samples
    and n_features is the number of features.

    Note that providing ``y`` is sufficient to generate the splits and
    hence ``np.zeros(n_samples)`` may be used as a placeholder for
    ``X`` instead of actual training data.

y : array-like, shape (n_samples,) or (n_samples, n_labels)
    The target variable for supervised learning problems.
    Stratification is done based on the y labels.

groups : object
    Always ignored, exists for compatibility.

Yields
------
train : ndarray
    The training set indices for that split.

test : ndarray
    The testing set indices for that split.

Notes
-----
Randomized CV splitters may return different results for each call of
split. You can make the results identical by setting ``random_state``
to an integer.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module TimeSeriesSplit : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_splits:int -> ?max_train_size:int -> unit -> t
(**
Time Series cross-validator

Provides train/test indices to split time series data samples
that are observed at fixed time intervals, in train/test sets.
In each split, test indices must be higher than before, and thus shuffling
in cross validator is inappropriate.

This cross-validation object is a variation of :class:`KFold`.
In the kth split, it returns first k folds as train set and the
(k+1)th fold as test set.

Note that unlike standard cross-validation methods, successive
training sets are supersets of those that come before them.

Read more in the :ref:`User Guide <cross_validation>`.

Parameters
----------
n_splits : int, default=5
    Number of splits. Must be at least 2.

    .. versionchanged:: 0.22
        ``n_splits`` default value changed from 3 to 5.

max_train_size : int, optional
    Maximum size for a single training set.

Examples
--------
>>> import numpy as np
>>> from sklearn.model_selection import TimeSeriesSplit
>>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
>>> y = np.array([1, 2, 3, 4, 5, 6])
>>> tscv = TimeSeriesSplit()
>>> print(tscv)
TimeSeriesSplit(max_train_size=None, n_splits=5)
>>> for train_index, test_index in tscv.split(X):
...     print("TRAIN:", train_index, "TEST:", test_index)
...     X_train, X_test = X[train_index], X[test_index]
...     y_train, y_test = y[train_index], y[test_index]
TRAIN: [0] TEST: [1]
TRAIN: [0 1] TEST: [2]
TRAIN: [0 1 2] TEST: [3]
TRAIN: [0 1 2 3] TEST: [4]
TRAIN: [0 1 2 3 4] TEST: [5]

Notes
-----
The training set has size ``i * n_samples // (n_splits + 1)
+ n_samples % (n_splits + 1)`` in the ``i``th split,
with a test set of size ``n_samples//(n_splits + 1)``,
where ``n_samples`` is the number of samples.
*)

val get_n_splits : ?x:Py.Object.t -> ?y:Py.Object.t -> ?groups:Py.Object.t -> t -> int
(**
Returns the number of splitting iterations in the cross-validator

Parameters
----------
X : object
    Always ignored, exists for compatibility.

y : object
    Always ignored, exists for compatibility.

groups : object
    Always ignored, exists for compatibility.

Returns
-------
n_splits : int
    Returns the number of splitting iterations in the cross-validator.
*)

val split : ?y:Ndarray.t -> ?groups:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> x:Ndarray.t -> t -> Py.Object.t
(**
Generate indices to split data into training and test set.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training data, where n_samples is the number of samples
    and n_features is the number of features.

y : array-like, shape (n_samples,)
    Always ignored, exists for compatibility.

groups : array-like, with shape (n_samples,)
    Always ignored, exists for compatibility.

Yields
------
train : ndarray
    The training set indices for that split.

test : ndarray
    The testing set indices for that split.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val check_cv : ?cv:[`Int of int | `CrossValGenerator of Py.Object.t | `Ndarray of Ndarray.t] -> ?y:Ndarray.t -> ?classifier:bool -> unit -> Py.Object.t
(**
Input checker utility for building a cross-validator

Parameters
----------
cv : int, cross-validation generator or an iterable, optional
    Determines the cross-validation splitting strategy.
    Possible inputs for cv are:

    - None, to use the default 5-fold cross-validation,
    - integer, to specify the number of folds.
    - :term:`CV splitter`,
    - An iterable yielding (train, test) splits as arrays of indices.

    For integer/None inputs, if classifier is True and ``y`` is either
    binary or multiclass, :class:`StratifiedKFold` is used. In all other
    cases, :class:`KFold` is used.

    Refer :ref:`User Guide <cross_validation>` for the various
    cross-validation strategies that can be used here.

    .. versionchanged:: 0.22
        ``cv`` default value changed from 3-fold to 5-fold.

y : array-like, optional
    The target variable for supervised learning problems.

classifier : boolean, optional, default False
    Whether the task is a classification task, in which case
    stratified KFold will be used.

Returns
-------
checked_cv : a cross-validator instance.
    The return value is a cross-validator which generates the train/test
    splits via the ``split`` method.
*)

val cross_val_predict : ?y:Ndarray.t -> ?groups:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> ?cv:[`Int of int | `CrossValGenerator of Py.Object.t | `Ndarray of Ndarray.t] -> ?n_jobs:[`Int of int | `None] -> ?verbose:int -> ?fit_params:Py.Object.t -> ?pre_dispatch:[`Int of int | `String of string] -> ?method_:string -> estimator:Py.Object.t -> x:Ndarray.t -> unit -> Ndarray.t
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
    train/test set. Only used in conjunction with a "Group" :term:`cv`
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

val cross_val_score : ?y:Ndarray.t -> ?groups:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> ?scoring:[`String of string | `Callable of Py.Object.t | `None] -> ?cv:[`Int of int | `CrossValGenerator of Py.Object.t | `Ndarray of Ndarray.t] -> ?n_jobs:[`Int of int | `None] -> ?verbose:int -> ?fit_params:Py.Object.t -> ?pre_dispatch:[`Int of int | `String of string] -> ?error_score:[`Raise | `PyObject of Py.Object.t] -> estimator:Py.Object.t -> x:Ndarray.t -> unit -> Py.Object.t
(**
Evaluate a score by cross-validation

Read more in the :ref:`User Guide <cross_validation>`.

Parameters
----------
estimator : estimator object implementing 'fit'
    The object to use to fit the data.

X : array-like
    The data to fit. Can be for example a list, or an array.

y : array-like, optional, default: None
    The target variable to try to predict in the case of
    supervised learning.

groups : array-like, with shape (n_samples,), optional
    Group labels for the samples used while splitting the dataset into
    train/test set. Only used in conjunction with a "Group" :term:`cv`
    instance (e.g., :class:`GroupKFold`).

scoring : string, callable or None, optional, default: None
    A string (see model evaluation documentation) or
    a scorer callable object / function with signature
    ``scorer(estimator, X, y)`` which should return only
    a single value.

    Similar to :func:`cross_validate`
    but only a single metric is permitted.

    If None, the estimator's default scorer (if available) is used.

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

error_score : 'raise' or numeric
    Value to assign to the score if an error occurs in estimator fitting.
    If set to 'raise', the error is raised.
    If a numeric value is given, FitFailedWarning is raised. This parameter
    does not affect the refit step, which will always raise the error.

Returns
-------
scores : array of float, shape=(len(list(cv)),)
    Array of scores of the estimator for each run of the cross validation.

Examples
--------
>>> from sklearn import datasets, linear_model
>>> from sklearn.model_selection import cross_val_score
>>> diabetes = datasets.load_diabetes()
>>> X = diabetes.data[:150]
>>> y = diabetes.target[:150]
>>> lasso = linear_model.Lasso()
>>> print(cross_val_score(lasso, X, y, cv=3))
[0.33150734 0.08022311 0.03531764]

See Also
---------
:func:`sklearn.model_selection.cross_validate`:
    To run cross-validation on multiple metrics and also to return
    train scores, fit times and score times.

:func:`sklearn.model_selection.cross_val_predict`:
    Get predictions from each split of cross-validation for diagnostic
    purposes.

:func:`sklearn.metrics.make_scorer`:
    Make a scorer from a performance metric or loss function.
*)

val cross_validate : ?y:Ndarray.t -> ?groups:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> ?scoring:[`String of string | `Callable of Py.Object.t | `Dict of Py.Object.t | `None | `PyObject of Py.Object.t] -> ?cv:[`Int of int | `CrossValGenerator of Py.Object.t | `Ndarray of Ndarray.t] -> ?n_jobs:[`Int of int | `None] -> ?verbose:int -> ?fit_params:Py.Object.t -> ?pre_dispatch:[`Int of int | `String of string] -> ?return_train_score:bool -> ?return_estimator:bool -> ?error_score:[`Raise | `PyObject of Py.Object.t] -> estimator:Py.Object.t -> x:Ndarray.t -> unit -> Py.Object.t
(**
Evaluate metric(s) by cross-validation and also record fit/score times.

Read more in the :ref:`User Guide <multimetric_cross_validation>`.

Parameters
----------
estimator : estimator object implementing 'fit'
    The object to use to fit the data.

X : array-like
    The data to fit. Can be for example a list, or an array.

y : array-like, optional, default: None
    The target variable to try to predict in the case of
    supervised learning.

groups : array-like, with shape (n_samples,), optional
    Group labels for the samples used while splitting the dataset into
    train/test set. Only used in conjunction with a "Group" :term:`cv`
    instance (e.g., :class:`GroupKFold`).

scoring : string, callable, list/tuple, dict or None, default: None
    A single string (see :ref:`scoring_parameter`) or a callable
    (see :ref:`scoring`) to evaluate the predictions on the test set.

    For evaluating multiple metrics, either give a list of (unique) strings
    or a dict with names as keys and callables as values.

    NOTE that when using custom scorers, each scorer should return a single
    value. Metric functions returning a list/array of values can be wrapped
    into multiple scorers that return one value each.

    See :ref:`multimetric_grid_search` for an example.

    If None, the estimator's score method is used.

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

return_train_score : boolean, default=False
    Whether to include train scores.
    Computing training scores is used to get insights on how different
    parameter settings impact the overfitting/underfitting trade-off.
    However computing the scores on the training set can be computationally
    expensive and is not strictly required to select the parameters that
    yield the best generalization performance.

return_estimator : boolean, default False
    Whether to return the estimators fitted on each split.

error_score : 'raise' or numeric
    Value to assign to the score if an error occurs in estimator fitting.
    If set to 'raise', the error is raised.
    If a numeric value is given, FitFailedWarning is raised. This parameter
    does not affect the refit step, which will always raise the error.

Returns
-------
scores : dict of float arrays of shape (n_splits,)
    Array of scores of the estimator for each run of the cross validation.

    A dict of arrays containing the score/time arrays for each scorer is
    returned. The possible keys for this ``dict`` are:

        ``test_score``
            The score array for test scores on each cv split.
            Suffix ``_score`` in ``test_score`` changes to a specific
            metric like ``test_r2`` or ``test_auc`` if there are
            multiple scoring metrics in the scoring parameter.
        ``train_score``
            The score array for train scores on each cv split.
            Suffix ``_score`` in ``train_score`` changes to a specific
            metric like ``train_r2`` or ``train_auc`` if there are
            multiple scoring metrics in the scoring parameter.
            This is available only if ``return_train_score`` parameter
            is ``True``.
        ``fit_time``
            The time for fitting the estimator on the train
            set for each cv split.
        ``score_time``
            The time for scoring the estimator on the test set for each
            cv split. (Note time for scoring on the train set is not
            included even if ``return_train_score`` is set to ``True``
        ``estimator``
            The estimator objects for each cv split.
            This is available only if ``return_estimator`` parameter
            is set to ``True``.

Examples
--------
>>> from sklearn import datasets, linear_model
>>> from sklearn.model_selection import cross_validate
>>> from sklearn.metrics import make_scorer
>>> from sklearn.metrics import confusion_matrix
>>> from sklearn.svm import LinearSVC
>>> diabetes = datasets.load_diabetes()
>>> X = diabetes.data[:150]
>>> y = diabetes.target[:150]
>>> lasso = linear_model.Lasso()

Single metric evaluation using ``cross_validate``

>>> cv_results = cross_validate(lasso, X, y, cv=3)
>>> sorted(cv_results.keys())
['fit_time', 'score_time', 'test_score']
>>> cv_results['test_score']
array([0.33150734, 0.08022311, 0.03531764])

Multiple metric evaluation using ``cross_validate``
(please refer the ``scoring`` parameter doc for more information)

>>> scores = cross_validate(lasso, X, y, cv=3,
...                         scoring=('r2', 'neg_mean_squared_error'),
...                         return_train_score=True)
>>> print(scores['test_neg_mean_squared_error'])
[-3635.5... -3573.3... -6114.7...]
>>> print(scores['train_r2'])
[0.28010158 0.39088426 0.22784852]

See Also
---------
:func:`sklearn.model_selection.cross_val_score`:
    Run cross-validation for single metric evaluation.

:func:`sklearn.model_selection.cross_val_predict`:
    Get predictions from each split of cross-validation for diagnostic
    purposes.

:func:`sklearn.metrics.make_scorer`:
    Make a scorer from a performance metric or loss function.
*)

val fit_grid_point : ?error_score:[`Raise | `PyObject of Py.Object.t] -> ?fit_params:(string * Py.Object.t) list -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t | `ArrayLike of Py.Object.t] -> y:[`Ndarray of Ndarray.t | `None] -> estimator:Py.Object.t -> parameters:Py.Object.t -> train:[`Ndarray of Ndarray.t | `Bool of bool | `PyObject of Py.Object.t] -> test:[`Ndarray of Ndarray.t | `Bool of bool | `PyObject of Py.Object.t] -> scorer:[`Callable of Py.Object.t | `None] -> verbose:int -> unit -> (float * Py.Object.t * int)
(**
Run fit on one set of parameters.

Parameters
----------
X : array-like, sparse matrix or list
    Input data.

y : array-like or None
    Targets for input data.

estimator : estimator object
    A object of that type is instantiated for each grid point.
    This is assumed to implement the scikit-learn estimator interface.
    Either estimator needs to provide a ``score`` function,
    or ``scoring`` must be passed.

parameters : dict
    Parameters to be set on estimator for this grid point.

train : ndarray, dtype int or bool
    Boolean mask or indices for training set.

test : ndarray, dtype int or bool
    Boolean mask or indices for test set.

scorer : callable or None
    The scorer callable object / function must have its signature as
    ``scorer(estimator, X, y)``.

    If ``None`` the estimator's score method is used.

verbose : int
    Verbosity level.

**fit_params : kwargs
    Additional parameter passed to the fit function of the estimator.

error_score : 'raise' or numeric
    Value to assign to the score if an error occurs in estimator fitting.
    If set to 'raise', the error is raised. If a numeric value is given,
    FitFailedWarning is raised. This parameter does not affect the refit
    step, which will always raise the error. Default is ``np.nan``.

Returns
-------
score : float
     Score of this parameter setting on given test split.

parameters : dict
    The parameters that have been evaluated.

n_samples_test : int
    Number of test samples in this split.
*)

val learning_curve : ?groups:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> ?train_sizes:[`Ndarray of Ndarray.t | `Int of int | `PyObject of Py.Object.t] -> ?cv:[`Int of int | `CrossValGenerator of Py.Object.t | `Ndarray of Ndarray.t] -> ?scoring:[`String of string | `Callable of Py.Object.t | `None] -> ?exploit_incremental_learning:bool -> ?n_jobs:[`Int of int | `None] -> ?pre_dispatch:[`Int of int | `String of string] -> ?verbose:int -> ?shuffle:bool -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> ?error_score:[`Raise | `PyObject of Py.Object.t] -> ?return_times:bool -> estimator:Py.Object.t -> x:Ndarray.t -> y:Ndarray.t -> unit -> (Py.Object.t * Ndarray.t * Ndarray.t * Ndarray.t * Ndarray.t)
(**
Learning curve.

Determines cross-validated training and test scores for different training
set sizes.

A cross-validation generator splits the whole dataset k times in training
and test data. Subsets of the training set with varying sizes will be used
to train the estimator and a score for each training subset size and the
test set will be computed. Afterwards, the scores will be averaged over
all k runs for each training subset size.

Read more in the :ref:`User Guide <learning_curve>`.

Parameters
----------
estimator : object type that implements the "fit" and "predict" methods
    An object of that type which is cloned for each validation.

X : array-like, shape (n_samples, n_features)
    Training vector, where n_samples is the number of samples and
    n_features is the number of features.

y : array-like, shape (n_samples) or (n_samples, n_features), optional
    Target relative to X for classification or regression;
    None for unsupervised learning.

groups : array-like, with shape (n_samples,), optional
    Group labels for the samples used while splitting the dataset into
    train/test set. Only used in conjunction with a "Group" :term:`cv`
    instance (e.g., :class:`GroupKFold`).

train_sizes : array-like, shape (n_ticks,), dtype float or int
    Relative or absolute numbers of training examples that will be used to
    generate the learning curve. If the dtype is float, it is regarded as a
    fraction of the maximum size of the training set (that is determined
    by the selected validation method), i.e. it has to be within (0, 1].
    Otherwise it is interpreted as absolute sizes of the training sets.
    Note that for classification the number of samples usually have to
    be big enough to contain at least one sample from each class.
    (default: np.linspace(0.1, 1.0, 5))

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

scoring : string, callable or None, optional, default: None
    A string (see model evaluation documentation) or
    a scorer callable object / function with signature
    ``scorer(estimator, X, y)``.

exploit_incremental_learning : boolean, optional, default: False
    If the estimator supports incremental learning, this will be
    used to speed up fitting for different training set sizes.

n_jobs : int or None, optional (default=None)
    Number of jobs to run in parallel.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

pre_dispatch : integer or string, optional
    Number of predispatched jobs for parallel execution (default is
    all). The option can reduce the allocated memory. The string can
    be an expression like '2*n_jobs'.

verbose : integer, optional
    Controls the verbosity: the higher, the more messages.

shuffle : boolean, optional
    Whether to shuffle training data before taking prefixes of it
    based on``train_sizes``.

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`. Used when ``shuffle`` is True.

error_score : 'raise' or numeric
    Value to assign to the score if an error occurs in estimator fitting.
    If set to 'raise', the error is raised.
    If a numeric value is given, FitFailedWarning is raised. This parameter
    does not affect the refit step, which will always raise the error.

return_times : boolean, optional (default: False)
    Whether to return the fit and score times.

Returns
-------
train_sizes_abs : array, shape (n_unique_ticks,), dtype int
    Numbers of training examples that has been used to generate the
    learning curve. Note that the number of ticks might be less
    than n_ticks because duplicate entries will be removed.

train_scores : array, shape (n_ticks, n_cv_folds)
    Scores on training sets.

test_scores : array, shape (n_ticks, n_cv_folds)
    Scores on test set.

fit_times : array, shape (n_ticks, n_cv_folds)
    Times spent for fitting in seconds. Only present if ``return_times``
    is True.

score_times : array, shape (n_ticks, n_cv_folds)
    Times spent for scoring in seconds. Only present if ``return_times``
    is True.

Notes
-----
See :ref:`examples/model_selection/plot_learning_curve.py
<sphx_glr_auto_examples_model_selection_plot_learning_curve.py>`
*)

val permutation_test_score : ?groups:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> ?cv:[`Int of int | `CrossValGenerator of Py.Object.t | `Ndarray of Ndarray.t] -> ?n_permutations:int -> ?n_jobs:[`Int of int | `None] -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> ?verbose:int -> ?scoring:[`String of string | `Callable of Py.Object.t | `None] -> estimator:Py.Object.t -> x:Py.Object.t -> y:Ndarray.t -> unit -> (float * Ndarray.t * float)
(**
Evaluate the significance of a cross-validated score with permutations

Read more in the :ref:`User Guide <cross_validation>`.

Parameters
----------
estimator : estimator object implementing 'fit'
    The object to use to fit the data.

X : array-like of shape at least 2D
    The data to fit.

y : array-like
    The target variable to try to predict in the case of
    supervised learning.

groups : array-like, with shape (n_samples,), optional
    Labels to constrain permutation within groups, i.e. ``y`` values
    are permuted among samples with the same group identifier.
    When not specified, ``y`` values are permuted among all samples.

    When a grouped cross-validator is used, the group labels are
    also passed on to the ``split`` method of the cross-validator. The
    cross-validator uses them for grouping the samples  while splitting
    the dataset into train/test set.

scoring : string, callable or None, optional, default: None
    A single string (see :ref:`scoring_parameter`) or a callable
    (see :ref:`scoring`) to evaluate the predictions on the test set.

    If None the estimator's score method is used.

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

n_permutations : integer, optional
    Number of times to permute ``y``.

n_jobs : int or None, optional (default=None)
    The number of CPUs to use to do the computation.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

random_state : int, RandomState instance or None, optional (default=0)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

verbose : integer, optional
    The verbosity level.

Returns
-------
score : float
    The true score without permuting targets.

permutation_scores : array, shape (n_permutations,)
    The scores obtained for each permutations.

pvalue : float
    The p-value, which approximates the probability that the score would
    be obtained by chance. This is calculated as:

    `(C + 1) / (n_permutations + 1)`

    Where C is the number of permutations whose score >= the true score.

    The best possible p-value is 1/(n_permutations + 1), the worst is 1.0.

Notes
-----
This function implements Test 1 in:

    Ojala and Garriga. Permutation Tests for Studying Classifier
    Performance.  The Journal of Machine Learning Research (2010)
    vol. 11
*)

val train_test_split : ?test_size:[`Float of float | `Int of int | `None] -> ?train_size:[`Float of float | `Int of int | `None] -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> ?shuffle:bool -> ?stratify:[`Ndarray of Ndarray.t | `None] -> Ndarray.t list -> Ndarray.t array
(**
Split arrays or matrices into random train and test subsets

Quick utility that wraps input validation and
``next(ShuffleSplit().split(X, y))`` and application to input data
into a single call for splitting (and optionally subsampling) data in a
oneliner.

Read more in the :ref:`User Guide <cross_validation>`.

Parameters
----------
*arrays : sequence of indexables with same length / shape[0]
    Allowed inputs are lists, numpy arrays, scipy-sparse
    matrices or pandas dataframes.

test_size : float, int or None, optional (default=None)
    If float, should be between 0.0 and 1.0 and represent the proportion
    of the dataset to include in the test split. If int, represents the
    absolute number of test samples. If None, the value is set to the
    complement of the train size. If ``train_size`` is also None, it will
    be set to 0.25.

train_size : float, int, or None, (default=None)
    If float, should be between 0.0 and 1.0 and represent the
    proportion of the dataset to include in the train split. If
    int, represents the absolute number of train samples. If None,
    the value is automatically set to the complement of the test size.

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

shuffle : boolean, optional (default=True)
    Whether or not to shuffle the data before splitting. If shuffle=False
    then stratify must be None.

stratify : array-like or None (default=None)
    If not None, data is split in a stratified fashion, using this as
    the class labels.

Returns
-------
splitting : list, length=2 * len(arrays)
    List containing train-test split of inputs.

    .. versionadded:: 0.16
        If the input is sparse, the output will be a
        ``scipy.sparse.csr_matrix``. Else, output type is the same as the
        input type.

Examples
--------
>>> import numpy as np
>>> from sklearn.model_selection import train_test_split
>>> X, y = np.arange(10).reshape((5, 2)), range(5)
>>> X
array([[0, 1],
       [2, 3],
       [4, 5],
       [6, 7],
       [8, 9]])
>>> list(y)
[0, 1, 2, 3, 4]

>>> X_train, X_test, y_train, y_test = train_test_split(
...     X, y, test_size=0.33, random_state=42)
...
>>> X_train
array([[4, 5],
       [0, 1],
       [6, 7]])
>>> y_train
[2, 0, 3]
>>> X_test
array([[2, 3],
       [8, 9]])
>>> y_test
[1, 4]

>>> train_test_split(y, shuffle=False)
[[0, 1, 2], [3, 4]]
*)

val validation_curve : ?groups:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> ?cv:[`Int of int | `CrossValGenerator of Py.Object.t | `Ndarray of Ndarray.t] -> ?scoring:[`String of string | `Callable of Py.Object.t | `None] -> ?n_jobs:[`Int of int | `None] -> ?pre_dispatch:[`Int of int | `String of string] -> ?verbose:int -> ?error_score:[`Raise | `PyObject of Py.Object.t] -> estimator:Py.Object.t -> x:Ndarray.t -> y:Ndarray.t -> param_name:string -> param_range:Ndarray.t -> unit -> (Ndarray.t * Ndarray.t)
(**
Validation curve.

Determine training and test scores for varying parameter values.

Compute scores for an estimator with different values of a specified
parameter. This is similar to grid search with one parameter. However, this
will also compute training scores and is merely a utility for plotting the
results.

Read more in the :ref:`User Guide <learning_curve>`.

Parameters
----------
estimator : object type that implements the "fit" and "predict" methods
    An object of that type which is cloned for each validation.

X : array-like, shape (n_samples, n_features)
    Training vector, where n_samples is the number of samples and
    n_features is the number of features.

y : array-like, shape (n_samples) or (n_samples, n_features), optional
    Target relative to X for classification or regression;
    None for unsupervised learning.

param_name : string
    Name of the parameter that will be varied.

param_range : array-like, shape (n_values,)
    The values of the parameter that will be evaluated.

groups : array-like, with shape (n_samples,), optional
    Group labels for the samples used while splitting the dataset into
    train/test set. Only used in conjunction with a "Group" :term:`cv`
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

scoring : string, callable or None, optional, default: None
    A string (see model evaluation documentation) or
    a scorer callable object / function with signature
    ``scorer(estimator, X, y)``.

n_jobs : int or None, optional (default=None)
    Number of jobs to run in parallel.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

pre_dispatch : integer or string, optional
    Number of predispatched jobs for parallel execution (default is
    all). The option can reduce the allocated memory. The string can
    be an expression like '2*n_jobs'.

verbose : integer, optional
    Controls the verbosity: the higher, the more messages.

error_score : 'raise' or numeric
    Value to assign to the score if an error occurs in estimator fitting.
    If set to 'raise', the error is raised.
    If a numeric value is given, FitFailedWarning is raised. This parameter
    does not affect the refit step, which will always raise the error.

Returns
-------
train_scores : array, shape (n_ticks, n_cv_folds)
    Scores on training sets.

test_scores : array, shape (n_ticks, n_cv_folds)
    Scores on test set.

Notes
-----
See :ref:`sphx_glr_auto_examples_model_selection_plot_validation_curve.py`
*)

