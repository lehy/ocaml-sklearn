module PartialDependenceDisplay : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : pd_results:Py.Object.t -> features:Py.Object.t -> feature_names:string list -> target_idx:int -> pdp_lim:Py.Object.t -> deciles:Py.Object.t -> unit -> t
(**
Partial Dependence Plot (PDP) visualization.

It is recommended to use
:func:`~sklearn.inspection.plot_partial_dependence` to create a
:class:`~sklearn.inspection.PartialDependenceDisplay`. All parameters are
stored as attributes.

Read more in
:ref:`sphx_glr_auto_examples_plot_partial_dependence_visualization_api.py`
and the :ref:`User Guide <visualizations>`.

    .. versionadded:: 0.22

Parameters
----------
pd_results : list of (ndarray, ndarray)
    Results of :func:`~sklearn.inspection.partial_dependence` for
    ``features``. Each tuple corresponds to a (averaged_predictions, grid).

features : list of (int,) or list of (int, int)
    Indices of features for a given plot. A tuple of one integer will plot
    a partial dependence curve of one feature. A tuple of two integers will
    plot a two-way partial dependence curve as a contour plot.

feature_names : list of str
    Feature names corresponding to the indices in ``features``.

target_idx : int

    - In a multiclass setting, specifies the class for which the PDPs
      should be computed. Note that for binary classification, the
      positive class (index 1) is always used.
    - In a multioutput setting, specifies the task for which the PDPs
      should be computed.

    Ignored in binary classification or classical regression settings.

pdp_lim : dict
    Global min and max average predictions, such that all plots will have
    the same scale and y limits. `pdp_lim[1]` is the global min and max for
    single partial dependence curves. `pdp_lim[2]` is the global min and
    max for two-way partial dependence curves.

deciles : dict
    Deciles for feature indices in ``features``.

Attributes
----------
bounding_ax_ : matplotlib Axes or None
    If `ax` is an axes or None, the `bounding_ax_` is the axes where the
    grid of partial dependence plots are drawn. If `ax` is a list of axes
    or a numpy array of axes, `bounding_ax_` is None.

axes_ : ndarray of matplotlib Axes
    If `ax` is an axes or None, `axes_[i, j]` is the axes on the i-th row
    and j-th column. If `ax` is a list of axes, `axes_[i]` is the i-th item
    in `ax`. Elements that are None corresponds to a nonexisting axes in
    that position.

lines_ : ndarray of matplotlib Artists
    If `ax` is an axes or None, `line_[i, j]` is the partial dependence
    curve on the i-th row and j-th column. If `ax` is a list of axes,
    `lines_[i]` is the partial dependence curve corresponding to the i-th
    item in `ax`. Elements that are None corresponds to a nonexisting axes
    or an axes that does not include a line plot.

contours_ : ndarray of matplotlib Artists
    If `ax` is an axes or None, `contours_[i, j]` is the partial dependence
    plot on the i-th row and j-th column. If `ax` is a list of axes,
    `contours_[i]` is the partial dependence plot corresponding to the i-th
    item in `ax`. Elements that are None corresponds to a nonexisting axes
    or an axes that does not include a contour plot.

figure_ : matplotlib Figure
    Figure containing partial dependence plots.
*)

val plot : ?ax:Py.Object.t -> ?n_cols:int -> ?line_kw:Py.Object.t -> ?contour_kw:Py.Object.t -> t -> Py.Object.t
(**
Plot partial dependence plots.

Parameters
----------
ax : Matplotlib axes or array-like of Matplotlib axes, default=None
    - If a single axis is passed in, it is treated as a bounding axes
        and a grid of partial dependence plots will be drawn within
        these bounds. The `n_cols` parameter controls the number of
        columns in the grid.
    - If an array-like of axes are passed in, the partial dependence
        plots will be drawn directly into these axes.
    - If `None`, a figure and a bounding axes is created and treated
        as the single axes case.

n_cols : int, default=3
    The maximum number of columns in the grid plot. Only active when
    `ax` is a single axes or `None`.

line_kw : dict, default=None
    Dict with keywords passed to the `matplotlib.pyplot.plot` call.
    For one-way partial dependence plots.

contour_kw : dict, default=None
    Dict with keywords passed to the `matplotlib.pyplot.contourf`
    call for two-way partial dependence plots.

Returns
-------
display: :class:`~sklearn.inspection.PartialDependenceDisplay`
*)


(** Attribute bounding_ax_: see constructor for documentation *)
val bounding_ax_ : t -> Py.Object.t

(** Attribute axes_: see constructor for documentation *)
val axes_ : t -> Py.Object.t

(** Attribute lines_: see constructor for documentation *)
val lines_ : t -> Py.Object.t

(** Attribute contours_: see constructor for documentation *)
val contours_ : t -> Py.Object.t

(** Attribute figure_: see constructor for documentation *)
val figure_ : t -> Py.Object.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

val partial_dependence : ?response_method:[`Auto | `Predict_proba | `Decision_function] -> ?percentiles:Py.Object.t -> ?grid_resolution:int -> ?method_:string -> estimator:Py.Object.t -> x:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> features:Py.Object.t -> unit -> (Ndarray.t * Py.Object.t)
(**
Partial dependence of ``features``.

Partial dependence of a feature (or a set of features) corresponds to
the average response of an estimator for each possible value of the
feature.

Read more in the :ref:`User Guide <partial_dependence>`.

Parameters
----------
estimator : BaseEstimator
    A fitted estimator object implementing :term:`predict`,
    :term:`predict_proba`, or :term:`decision_function`.
    Multioutput-multiclass classifiers are not supported.

X : {array-like or dataframe} of shape (n_samples, n_features)
    ``X`` is used both to generate a grid of values for the
    ``features``, and to compute the averaged predictions when
    method is 'brute'.

features : array-like of {int, str}
    The feature (e.g. `[0]`) or pair of interacting features
    (e.g. `[(0, 1)]`) for which the partial dependency should be computed.

response_method : 'auto', 'predict_proba' or 'decision_function',             optional (default='auto')
    Specifies whether to use :term:`predict_proba` or
    :term:`decision_function` as the target response. For regressors
    this parameter is ignored and the response is always the output of
    :term:`predict`. By default, :term:`predict_proba` is tried first
    and we revert to :term:`decision_function` if it doesn't exist. If
    ``method`` is 'recursion', the response is always the output of
    :term:`decision_function`.

percentiles : tuple of float, optional (default=(0.05, 0.95))
    The lower and upper percentile used to create the extreme values
    for the grid. Must be in [0, 1].

grid_resolution : int, optional (default=100)
    The number of equally spaced points on the grid, for each target
    feature.

method : str, optional (default='auto')
    The method used to calculate the averaged predictions:

    - 'recursion' is only supported for gradient boosting estimator (namely
      :class:`GradientBoostingClassifier<sklearn.ensemble.GradientBoostingClassifier>`,
      :class:`GradientBoostingRegressor<sklearn.ensemble.GradientBoostingRegressor>`,
      :class:`HistGradientBoostingClassifier<sklearn.ensemble.HistGradientBoostingClassifier>`,
      :class:`HistGradientBoostingRegressor<sklearn.ensemble.HistGradientBoostingRegressor>`)
      but is more efficient in terms of speed.
      With this method, ``X`` is only used to build the
      grid and the partial dependences are computed using the training
      data. This method does not account for the ``init`` predictor of
      the boosting process, which may lead to incorrect values (see
      warning below). With this method, the target response of a
      classifier is always the decision function, not the predicted
      probabilities.

    - 'brute' is supported for any estimator, but is more
      computationally intensive.

    - 'auto':

      - 'recursion' is used for
        :class:`GradientBoostingClassifier<sklearn.ensemble.GradientBoostingClassifier>`
        and
        :class:`GradientBoostingRegressor<sklearn.ensemble.GradientBoostingRegressor>`
        if ``init=None``, and for
        :class:`HistGradientBoostingClassifier<sklearn.ensemble.HistGradientBoostingClassifier>`
        and
        :class:`HistGradientBoostingRegressor<sklearn.ensemble.HistGradientBoostingRegressor>`.
      - 'brute' is used for all other estimators.

Returns
-------
averaged_predictions : ndarray,             shape (n_outputs, len(values[0]), len(values[1]), ...)
    The predictions for all the points in the grid, averaged over all
    samples in X (or over the training data if ``method`` is
    'recursion'). ``n_outputs`` corresponds to the number of classes in
    a multi-class setting, or to the number of tasks for multi-output
    regression. For classical regression and binary classification
    ``n_outputs==1``. ``n_values_feature_j`` corresponds to the size
    ``values[j]``.

values : seq of 1d ndarrays
    The values with which the grid has been created. The generated grid
    is a cartesian product of the arrays in ``values``. ``len(values) ==
    len(features)``. The size of each array ``values[j]`` is either
    ``grid_resolution``, or the number of unique values in ``X[:, j]``,
    whichever is smaller.

Examples
--------
>>> X = [[0, 0, 2], [1, 0, 0]]
>>> y = [0, 1]
>>> from sklearn.ensemble import GradientBoostingClassifier
>>> gb = GradientBoostingClassifier(random_state=0).fit(X, y)
>>> partial_dependence(gb, features=[0], X=X, percentiles=(0, 1),
...                    grid_resolution=2) # doctest: +SKIP
(array([[-4.52...,  4.52...]]), [array([ 0.,  1.])])

See also
--------
sklearn.inspection.plot_partial_dependence: Plot partial dependence

Warnings
--------
The 'recursion' method only works for gradient boosting estimators, and
unlike the 'brute' method, it does not account for the ``init``
predictor of the boosting process. In practice this will produce the
same values as 'brute' up to a constant offset in the target response,
provided that ``init`` is a consant estimator (which is the default).
However, as soon as ``init`` is not a constant estimator, the partial
dependence values are incorrect for 'recursion'. This is not relevant for
:class:`HistGradientBoostingClassifier
<sklearn.ensemble.HistGradientBoostingClassifier>` and
:class:`HistGradientBoostingRegressor
<sklearn.ensemble.HistGradientBoostingRegressor>`, which do not have an
``init`` parameter.
*)

val permutation_importance : ?scoring:[`String of string | `Callable of Py.Object.t | `None] -> ?n_repeats:int -> ?n_jobs:[`Int of int | `None] -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> estimator:Py.Object.t -> x:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> y:[`Ndarray of Ndarray.t | `None] -> unit -> (Py.Object.t * Ndarray.t)
(**
Permutation importance for feature evaluation [BRE]_.

The :term:`estimator` is required to be a fitted estimator. `X` can be the
data set used to train the estimator or a hold-out set. The permutation
importance of a feature is calculated as follows. First, a baseline metric,
defined by :term:`scoring`, is evaluated on a (potentially different)
dataset defined by the `X`. Next, a feature column from the validation set
is permuted and the metric is evaluated again. The permutation importance
is defined to be the difference between the baseline metric and metric from
permutating the feature column.

Read more in the :ref:`User Guide <permutation_importance>`.

Parameters
----------
estimator : object
    An estimator that has already been :term:`fitted` and is compatible
    with :term:`scorer`.

X : ndarray or DataFrame, shape (n_samples, n_features)
    Data on which permutation importance will be computed.

y : array-like or None, shape (n_samples, ) or (n_samples, n_classes)
    Targets for supervised or `None` for unsupervised.

scoring : string, callable or None, default=None
    Scorer to use. It can be a single
    string (see :ref:`scoring_parameter`) or a callable (see
    :ref:`scoring`). If None, the estimator's default scorer is used.

n_repeats : int, default=5
    Number of times to permute a feature.

n_jobs : int or None, default=None
    The number of jobs to use for the computation.
    `None` means 1 unless in a :obj:`joblib.parallel_backend` context.
    `-1` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

random_state : int, RandomState instance, or None, default=None
    Pseudo-random number generator to control the permutations of each
    feature. See :term:`random_state`.

Returns
-------
result : Bunch
    Dictionary-like object, with attributes:

    importances_mean : ndarray, shape (n_features, )
        Mean of feature importance over `n_repeats`.
    importances_std : ndarray, shape (n_features, )
        Standard deviation over `n_repeats`.
    importances : ndarray, shape (n_features, n_repeats)
        Raw permutation importance scores.

References
----------
.. [BRE] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32,
         2001. https://doi.org/10.1023/A:1010933404324
*)

val plot_partial_dependence : ?feature_names:[`StringList of string list | `PyObject of Py.Object.t] -> ?target:int -> ?response_method:[`Auto | `Predict_proba | `Decision_function] -> ?n_cols:int -> ?grid_resolution:int -> ?percentiles:Py.Object.t -> ?method_:string -> ?n_jobs:int -> ?verbose:int -> ?fig:Py.Object.t -> ?line_kw:Py.Object.t -> ?contour_kw:Py.Object.t -> ?ax:Py.Object.t -> estimator:Py.Object.t -> x:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> features:[`String of string | `PyObject of Py.Object.t] -> unit -> Py.Object.t
(**
Partial dependence plots.

The ``len(features)`` plots are arranged in a grid with ``n_cols``
columns. Two-way partial dependence plots are plotted as contour plots. The
deciles of the feature values will be shown with tick marks on the x-axes
for one-way plots, and on both axes for two-way plots.

.. note::

    :func:`plot_partial_dependence` does not support using the same axes
    with multiple calls. To plot the the partial dependence for multiple
    estimators, please pass the axes created by the first call to the
    second call::

      >>> from sklearn.inspection import plot_partial_dependence
      >>> from sklearn.datasets import make_friedman1
      >>> from sklearn.linear_model import LinearRegression
      >>> X, y = make_friedman1()
      >>> est = LinearRegression().fit(X, y)
      >>> disp1 = plot_partial_dependence(est, X)  # doctest: +SKIP
      >>> disp2 = plot_partial_dependence(est, X,
      ...                                 ax=disp1.axes_)  # doctest: +SKIP

Read more in the :ref:`User Guide <partial_dependence>`.

Parameters
----------
estimator : BaseEstimator
    A fitted estimator object implementing :term:`predict`,
    :term:`predict_proba`, or :term:`decision_function`.
    Multioutput-multiclass classifiers are not supported.

X : {array-like or dataframe} of shape (n_samples, n_features)
    The data to use to build the grid of values on which the dependence
    will be evaluated. This is usually the training data.

features : list of {int, str, pair of int, pair of str}
    The target features for which to create the PDPs.
    If features[i] is an int or a string, a one-way PDP is created; if
    features[i] is a tuple, a two-way PDP is created. Each tuple must be
    of size 2.
    if any entry is a string, then it must be in ``feature_names``.

feature_names : array-like of shape (n_features,), dtype=str, default=None
    Name of each feature; feature_names[i] holds the name of the feature
    with index i.
    By default, the name of the feature corresponds to their numerical
    index for NumPy array and their column name for pandas dataframe.

target : int, optional (default=None)
    - In a multiclass setting, specifies the class for which the PDPs
      should be computed. Note that for binary classification, the
      positive class (index 1) is always used.
    - In a multioutput setting, specifies the task for which the PDPs
      should be computed.

    Ignored in binary classification or classical regression settings.

response_method : 'auto', 'predict_proba' or 'decision_function',             optional (default='auto')
    Specifies whether to use :term:`predict_proba` or
    :term:`decision_function` as the target response. For regressors
    this parameter is ignored and the response is always the output of
    :term:`predict`. By default, :term:`predict_proba` is tried first
    and we revert to :term:`decision_function` if it doesn't exist. If
    ``method`` is 'recursion', the response is always the output of
    :term:`decision_function`.

n_cols : int, optional (default=3)
    The maximum number of columns in the grid plot. Only active when `ax`
    is a single axis or `None`.

grid_resolution : int, optional (default=100)
    The number of equally spaced points on the axes of the plots, for each
    target feature.

percentiles : tuple of float, optional (default=(0.05, 0.95))
    The lower and upper percentile used to create the extreme values
    for the PDP axes. Must be in [0, 1].

method : str, optional (default='auto')
    The method to use to calculate the partial dependence predictions:

    - 'recursion' is only supported for gradient boosting estimator (namely
      :class:`GradientBoostingClassifier<sklearn.ensemble.GradientBoostingClassifier>`,
      :class:`GradientBoostingRegressor<sklearn.ensemble.GradientBoostingRegressor>`,
      :class:`HistGradientBoostingClassifier<sklearn.ensemble.HistGradientBoostingClassifier>`,
      :class:`HistGradientBoostingRegressor<sklearn.ensemble.HistGradientBoostingRegressor>`)
      but is more efficient in terms of speed.
      With this method, ``X`` is optional and is only used to build the
      grid and the partial dependences are computed using the training
      data. This method does not account for the ``init`` predictor of
      the boosting process, which may lead to incorrect values (see
      warning below. With this method, the target response of a
      classifier is always the decision function, not the predicted
      probabilities.

    - 'brute' is supported for any estimator, but is more
      computationally intensive.

    - 'auto':
      - 'recursion' is used for estimators that supports it.
      - 'brute' is used for all other estimators.

n_jobs : int, optional (default=None)
    The number of CPUs to use to compute the partial dependences.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

verbose : int, optional (default=0)
    Verbose output during PD computations.

fig : Matplotlib figure object, optional (default=None)
    A figure object onto which the plots will be drawn, after the figure
    has been cleared. By default, a new one is created.

    .. deprecated:: 0.22
       ``fig`` will be removed in 0.24.

line_kw : dict, optional
    Dict with keywords passed to the ``matplotlib.pyplot.plot`` call.
    For one-way partial dependence plots.

contour_kw : dict, optional
    Dict with keywords passed to the ``matplotlib.pyplot.contourf`` call.
    For two-way partial dependence plots.

ax : Matplotlib axes or array-like of Matplotlib axes, default=None
    - If a single axis is passed in, it is treated as a bounding axes
        and a grid of partial dependence plots will be drawn within
        these bounds. The `n_cols` parameter controls the number of
        columns in the grid.
    - If an array-like of axes are passed in, the partial dependence
        plots will be drawn directly into these axes.
    - If `None`, a figure and a bounding axes is created and treated
        as the single axes case.

    .. versionadded:: 0.22

Returns
-------
display: :class:`~sklearn.inspection.PartialDependenceDisplay`

Examples
--------
>>> from sklearn.datasets import make_friedman1
>>> from sklearn.ensemble import GradientBoostingRegressor
>>> X, y = make_friedman1()
>>> clf = GradientBoostingRegressor(n_estimators=10).fit(X, y)
>>> plot_partial_dependence(clf, X, [0, (0, 1)]) #doctest: +SKIP

See also
--------
sklearn.inspection.partial_dependence: Return raw partial
  dependence values

Warnings
--------
The 'recursion' method only works for gradient boosting estimators, and
unlike the 'brute' method, it does not account for the ``init``
predictor of the boosting process. In practice this will produce the
same values as 'brute' up to a constant offset in the target response,
provided that ``init`` is a consant estimator (which is the default).
However, as soon as ``init`` is not a constant estimator, the partial
dependence values are incorrect for 'recursion'. This is not relevant for
:class:`HistGradientBoostingClassifier
<sklearn.ensemble.HistGradientBoostingClassifier>` and
:class:`HistGradientBoostingRegressor
<sklearn.ensemble.HistGradientBoostingRegressor>`, which do not have an
``init`` parameter.
*)

