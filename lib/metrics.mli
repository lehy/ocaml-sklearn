module ConfusionMatrixDisplay : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : confusion_matrix:Ndarray.t -> display_labels:Ndarray.t -> unit -> t
(**
Confusion Matrix visualization.

It is recommend to use :func:`~sklearn.metrics.plot_confusion_matrix` to
create a :class:`ConfusionMatrixDisplay`. All parameters are stored as
attributes.

Read more in the :ref:`User Guide <visualizations>`.

Parameters
----------
confusion_matrix : ndarray of shape (n_classes, n_classes)
    Confusion matrix.

display_labels : ndarray of shape (n_classes,)
    Display labels for plot.

Attributes
----------
im_ : matplotlib AxesImage
    Image representing the confusion matrix.

text_ : ndarray of shape (n_classes, n_classes), dtype=matplotlib Text,             or None
    Array of matplotlib axes. `None` if `include_values` is false.

ax_ : matplotlib Axes
    Axes with confusion matrix.

figure_ : matplotlib Figure
    Figure containing the confusion matrix.
*)

val plot : ?include_values:bool -> ?cmap:[`String of string | `PyObject of Py.Object.t] -> ?xticks_rotation:[`Vertical | `Horizontal | `Float of float] -> ?values_format:string -> ?ax:Py.Object.t -> t -> Py.Object.t
(**
Plot visualization.

Parameters
----------
include_values : bool, default=True
    Includes values in confusion matrix.

cmap : str or matplotlib Colormap, default='viridis'
    Colormap recognized by matplotlib.

xticks_rotation : {'vertical', 'horizontal'} or float,                          default='horizontal'
    Rotation of xtick labels.

values_format : str, default=None
    Format specification for values in confusion matrix. If `None`,
    the format specification is '.2g'.

ax : matplotlib axes, default=None
    Axes object to plot on. If `None`, a new figure and axes is
    created.

Returns
-------
display : :class:`~sklearn.metrics.ConfusionMatrixDisplay`
*)


(** Attribute im_: see constructor for documentation *)
val im_ : t -> Py.Object.t

(** Attribute text_: see constructor for documentation *)
val text_ : t -> Py.Object.t

(** Attribute ax_: see constructor for documentation *)
val ax_ : t -> Py.Object.t

(** Attribute figure_: see constructor for documentation *)
val figure_ : t -> Py.Object.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

module PrecisionRecallDisplay : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : precision:Ndarray.t -> recall:Ndarray.t -> average_precision:float -> estimator_name:string -> unit -> t
(**
Precision Recall visualization.

It is recommend to use :func:`~sklearn.metrics.plot_precision_recall_curve`
to create a visualizer. All parameters are stored as attributes.

Read more in the :ref:`User Guide <visualizations>`.

Parameters
-----------
precision : ndarray
    Precision values.

recall : ndarray
    Recall values.

average_precision : float
    Average precision.

estimator_name : str
    Name of estimator.

Attributes
----------
line_ : matplotlib Artist
    Precision recall curve.

ax_ : matplotlib Axes
    Axes with precision recall curve.

figure_ : matplotlib Figure
    Figure containing the curve.
*)

val plot : ?ax:Py.Object.t -> ?name:string -> ?kwargs:(string * Py.Object.t) list -> t -> Py.Object.t
(**
Plot visualization.

Extra keyword arguments will be passed to matplotlib's `plot`.

Parameters
----------
ax : Matplotlib Axes, default=None
    Axes object to plot on. If `None`, a new figure and axes is
    created.

name : str, default=None
    Name of precision recall curve for labeling. If `None`, use the
    name of the estimator.

**kwargs : dict
    Keyword arguments to be passed to matplotlib's `plot`.

Returns
-------
display : :class:`~sklearn.metrics.PrecisionRecallDisplay`
    Object that stores computed values.
*)


(** Attribute line_: see constructor for documentation *)
val line_ : t -> Py.Object.t

(** Attribute ax_: see constructor for documentation *)
val ax_ : t -> Py.Object.t

(** Attribute figure_: see constructor for documentation *)
val figure_ : t -> Py.Object.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

module RocCurveDisplay : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : fpr:Ndarray.t -> tpr:Ndarray.t -> roc_auc:float -> estimator_name:string -> unit -> t
(**
ROC Curve visualization.

It is recommend to use :func:`~sklearn.metrics.plot_roc_curve` to create a
visualizer. All parameters are stored as attributes.

Read more in the :ref:`User Guide <visualizations>`.

Parameters
----------
fpr : ndarray
    False positive rate.

tpr : ndarray
    True positive rate.

roc_auc : float
    Area under ROC curve.

estimator_name : str
    Name of estimator.

Attributes
----------
line_ : matplotlib Artist
    ROC Curve.

ax_ : matplotlib Axes
    Axes with ROC Curve.

figure_ : matplotlib Figure
    Figure containing the curve.

Examples
--------
>>> import matplotlib.pyplot as plt  # doctest: +SKIP
>>> import numpy as np
>>> from sklearn import metrics
>>> y = np.array([0, 0, 1, 1])
>>> pred = np.array([0.1, 0.4, 0.35, 0.8])
>>> fpr, tpr, thresholds = metrics.roc_curve(y, pred)
>>> roc_auc = metrics.auc(fpr, tpr)
>>> display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,                                          estimator_name='example estimator')
>>> display.plot()  # doctest: +SKIP
>>> plt.show()      # doctest: +SKIP
*)

val plot : ?ax:Py.Object.t -> ?name:string -> ?kwargs:(string * Py.Object.t) list -> t -> Py.Object.t
(**
Plot visualization

Extra keyword arguments will be passed to matplotlib's ``plot``.

Parameters
----------
ax : matplotlib axes, default=None
    Axes object to plot on. If `None`, a new figure and axes is
    created.

name : str, default=None
    Name of ROC Curve for labeling. If `None`, use the name of the
    estimator.

Returns
-------
display : :class:`~sklearn.metrics.plot.RocCurveDisplay`
    Object that stores computed values.
*)


(** Attribute line_: see constructor for documentation *)
val line_ : t -> Py.Object.t

(** Attribute ax_: see constructor for documentation *)
val ax_ : t -> Py.Object.t

(** Attribute figure_: see constructor for documentation *)
val figure_ : t -> Py.Object.t

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

val accuracy_score : ?normalize:bool -> ?sample_weight:Ndarray.t -> y_true:Py.Object.t -> y_pred:Py.Object.t -> unit -> float
(**
Accuracy classification score.

In multilabel classification, this function computes subset accuracy:
the set of labels predicted for a sample must *exactly* match the
corresponding set of labels in y_true.

Read more in the :ref:`User Guide <accuracy_score>`.

Parameters
----------
y_true : 1d array-like, or label indicator array / sparse matrix
    Ground truth (correct) labels.

y_pred : 1d array-like, or label indicator array / sparse matrix
    Predicted labels, as returned by a classifier.

normalize : bool, optional (default=True)
    If ``False``, return the number of correctly classified samples.
    Otherwise, return the fraction of correctly classified samples.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

Returns
-------
score : float
    If ``normalize == True``, return the fraction of correctly
    classified samples (float), else returns the number of correctly
    classified samples (int).

    The best performance is 1 with ``normalize == True`` and the number
    of samples with ``normalize == False``.

See also
--------
jaccard_score, hamming_loss, zero_one_loss

Notes
-----
In binary and multiclass classification, this function is equal
to the ``jaccard_score`` function.

Examples
--------
>>> from sklearn.metrics import accuracy_score
>>> y_pred = [0, 2, 1, 3]
>>> y_true = [0, 1, 2, 3]
>>> accuracy_score(y_true, y_pred)
0.5
>>> accuracy_score(y_true, y_pred, normalize=False)
2

In the multilabel case with binary label indicators:

>>> import numpy as np
>>> accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))
0.5
*)

val adjusted_mutual_info_score : ?average_method:string -> labels_true:Py.Object.t -> labels_pred:Py.Object.t -> unit -> Py.Object.t
(**
Adjusted Mutual Information between two clusterings.

Adjusted Mutual Information (AMI) is an adjustment of the Mutual
Information (MI) score to account for chance. It accounts for the fact that
the MI is generally higher for two clusterings with a larger number of
clusters, regardless of whether there is actually more information shared.
For two clusterings :math:`U` and :math:`V`, the AMI is given as::

    AMI(U, V) = [MI(U, V) - E(MI(U, V))] / [avg(H(U), H(V)) - E(MI(U, V))]

This metric is independent of the absolute values of the labels:
a permutation of the class or cluster label values won't change the
score value in any way.

This metric is furthermore symmetric: switching ``label_true`` with
``label_pred`` will return the same score value. This can be useful to
measure the agreement of two independent label assignments strategies
on the same dataset when the real ground truth is not known.

Be mindful that this function is an order of magnitude slower than other
metrics, such as the Adjusted Rand Index.

Read more in the :ref:`User Guide <mutual_info_score>`.

Parameters
----------
labels_true : int array, shape = [n_samples]
    A clustering of the data into disjoint subsets.

labels_pred : int array-like of shape (n_samples,)
    A clustering of the data into disjoint subsets.

average_method : string, optional (default: 'arithmetic')
    How to compute the normalizer in the denominator. Possible options
    are 'min', 'geometric', 'arithmetic', and 'max'.

    .. versionadded:: 0.20

    .. versionchanged:: 0.22
       The default value of ``average_method`` changed from 'max' to
       'arithmetic'.

Returns
-------
ami: float (upperlimited by 1.0)
   The AMI returns a value of 1 when the two partitions are identical
   (ie perfectly matched). Random partitions (independent labellings) have
   an expected AMI around 0 on average hence can be negative.

See also
--------
adjusted_rand_score: Adjusted Rand Index
mutual_info_score: Mutual Information (not adjusted for chance)

Examples
--------

Perfect labelings are both homogeneous and complete, hence have
score 1.0::

  >>> from sklearn.metrics.cluster import adjusted_mutual_info_score
  >>> adjusted_mutual_info_score([0, 0, 1, 1], [0, 0, 1, 1])
  ... # doctest: +SKIP
  1.0
  >>> adjusted_mutual_info_score([0, 0, 1, 1], [1, 1, 0, 0])
  ... # doctest: +SKIP
  1.0

If classes members are completely split across different clusters,
the assignment is totally in-complete, hence the AMI is null::

  >>> adjusted_mutual_info_score([0, 0, 0, 0], [0, 1, 2, 3])
  ... # doctest: +SKIP
  0.0

References
----------
.. [1] `Vinh, Epps, and Bailey, (2010). Information Theoretic Measures for
   Clusterings Comparison: Variants, Properties, Normalization and
   Correction for Chance, JMLR
   <http://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf>`_

.. [2] `Wikipedia entry for the Adjusted Mutual Information
   <https://en.wikipedia.org/wiki/Adjusted_Mutual_Information>`_
*)

val adjusted_rand_score : labels_true:Py.Object.t -> labels_pred:Ndarray.t -> unit -> float
(**
Rand index adjusted for chance.

The Rand Index computes a similarity measure between two clusterings
by considering all pairs of samples and counting pairs that are
assigned in the same or different clusters in the predicted and
true clusterings.

The raw RI score is then "adjusted for chance" into the ARI score
using the following scheme::

    ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)

The adjusted Rand index is thus ensured to have a value close to
0.0 for random labeling independently of the number of clusters and
samples and exactly 1.0 when the clusterings are identical (up to
a permutation).

ARI is a symmetric measure::

    adjusted_rand_score(a, b) == adjusted_rand_score(b, a)

Read more in the :ref:`User Guide <adjusted_rand_score>`.

Parameters
----------
labels_true : int array, shape = [n_samples]
    Ground truth class labels to be used as a reference

labels_pred : array-like of shape (n_samples,)
    Cluster labels to evaluate

Returns
-------
ari : float
   Similarity score between -1.0 and 1.0. Random labelings have an ARI
   close to 0.0. 1.0 stands for perfect match.

Examples
--------

Perfectly matching labelings have a score of 1 even

  >>> from sklearn.metrics.cluster import adjusted_rand_score
  >>> adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 1])
  1.0
  >>> adjusted_rand_score([0, 0, 1, 1], [1, 1, 0, 0])
  1.0

Labelings that assign all classes members to the same clusters
are complete be not always pure, hence penalized::

  >>> adjusted_rand_score([0, 0, 1, 2], [0, 0, 1, 1])
  0.57...

ARI is symmetric, so labelings that have pure clusters with members
coming from the same classes but unnecessary splits are penalized::

  >>> adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 2])
  0.57...

If classes members are completely split across different clusters, the
assignment is totally incomplete, hence the ARI is very low::

  >>> adjusted_rand_score([0, 0, 0, 0], [0, 1, 2, 3])
  0.0

References
----------

.. [Hubert1985] L. Hubert and P. Arabie, Comparing Partitions,
  Journal of Classification 1985
  https://link.springer.com/article/10.1007%2FBF01908075

.. [wk] https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index

See also
--------
adjusted_mutual_info_score: Adjusted Mutual Information
*)

val auc : x:Ndarray.t -> y:Py.Object.t -> unit -> float
(**
Compute Area Under the Curve (AUC) using the trapezoidal rule

This is a general function, given points on a curve.  For computing the
area under the ROC-curve, see :func:`roc_auc_score`.  For an alternative
way to summarize a precision-recall curve, see
:func:`average_precision_score`.

Parameters
----------
x : array, shape = [n]
    x coordinates. These must be either monotonic increasing or monotonic
    decreasing.
y : array, shape = [n]
    y coordinates.

Returns
-------
auc : float

Examples
--------
>>> import numpy as np
>>> from sklearn import metrics
>>> y = np.array([1, 1, 2, 2])
>>> pred = np.array([0.1, 0.4, 0.35, 0.8])
>>> fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
>>> metrics.auc(fpr, tpr)
0.75

See also
--------
roc_auc_score : Compute the area under the ROC curve
average_precision_score : Compute average precision from prediction scores
precision_recall_curve :
    Compute precision-recall pairs for different probability thresholds
*)

val average_precision_score : ?average:[`String of string | `Micro | `Macro | `PyObject of Py.Object.t] -> ?pos_label:[`Int of int | `String of string] -> ?sample_weight:Ndarray.t -> y_true:Ndarray.t -> y_score:Ndarray.t -> unit -> float
(**
Compute average precision (AP) from prediction scores

AP summarizes a precision-recall curve as the weighted mean of precisions
achieved at each threshold, with the increase in recall from the previous
threshold used as the weight:

.. math::
    \text{AP} = \sum_n (R_n - R_{n-1}) P_n

where :math:`P_n` and :math:`R_n` are the precision and recall at the nth
threshold [1]_. This implementation is not interpolated and is different
from computing the area under the precision-recall curve with the
trapezoidal rule, which uses linear interpolation and can be too
optimistic.

Note: this implementation is restricted to the binary classification task
or multilabel classification task.

Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

Parameters
----------
y_true : array, shape = [n_samples] or [n_samples, n_classes]
    True binary labels or binary label indicators.

y_score : array, shape = [n_samples] or [n_samples, n_classes]
    Target scores, can either be probability estimates of the positive
    class, confidence values, or non-thresholded measure of decisions
    (as returned by "decision_function" on some classifiers).

average : string, [None, 'micro', 'macro' (default), 'samples', 'weighted']
    If ``None``, the scores for each class are returned. Otherwise,
    this determines the type of averaging performed on the data:

    ``'micro'``:
        Calculate metrics globally by considering each element of the label
        indicator matrix as a label.
    ``'macro'``:
        Calculate metrics for each label, and find their unweighted
        mean.  This does not take label imbalance into account.
    ``'weighted'``:
        Calculate metrics for each label, and find their average, weighted
        by support (the number of true instances for each label).
    ``'samples'``:
        Calculate metrics for each instance, and find their average.

    Will be ignored when ``y_true`` is binary.

pos_label : int or str (default=1)
    The label of the positive class. Only applied to binary ``y_true``.
    For multilabel-indicator ``y_true``, ``pos_label`` is fixed to 1.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

Returns
-------
average_precision : float

References
----------
.. [1] `Wikipedia entry for the Average precision
       <https://en.wikipedia.org/w/index.php?title=Information_retrieval&
       oldid=793358396#Average_precision>`_

See also
--------
roc_auc_score : Compute the area under the ROC curve

precision_recall_curve :
    Compute precision-recall pairs for different probability thresholds

Examples
--------
>>> import numpy as np
>>> from sklearn.metrics import average_precision_score
>>> y_true = np.array([0, 0, 1, 1])
>>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
>>> average_precision_score(y_true, y_scores)
0.83...

Notes
-----
.. versionchanged:: 0.19
  Instead of linearly interpolating between operating points, precisions
  are weighted by the change in recall since the last operating point.
*)

val balanced_accuracy_score : ?sample_weight:Ndarray.t -> ?adjusted:bool -> y_true:Py.Object.t -> y_pred:Py.Object.t -> unit -> float
(**
Compute the balanced accuracy

The balanced accuracy in binary and multiclass classification problems to
deal with imbalanced datasets. It is defined as the average of recall
obtained on each class.

The best value is 1 and the worst value is 0 when ``adjusted=False``.

Read more in the :ref:`User Guide <balanced_accuracy_score>`.

Parameters
----------
y_true : 1d array-like
    Ground truth (correct) target values.

y_pred : 1d array-like
    Estimated targets as returned by a classifier.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

adjusted : bool, default=False
    When true, the result is adjusted for chance, so that random
    performance would score 0, and perfect performance scores 1.

Returns
-------
balanced_accuracy : float

See also
--------
recall_score, roc_auc_score

Notes
-----
Some literature promotes alternative definitions of balanced accuracy. Our
definition is equivalent to :func:`accuracy_score` with class-balanced
sample weights, and shares desirable properties with the binary case.
See the :ref:`User Guide <balanced_accuracy_score>`.

References
----------
.. [1] Brodersen, K.H.; Ong, C.S.; Stephan, K.E.; Buhmann, J.M. (2010).
       The balanced accuracy and its posterior distribution.
       Proceedings of the 20th International Conference on Pattern
       Recognition, 3121-24.
.. [2] John. D. Kelleher, Brian Mac Namee, Aoife D'Arcy, (2015).
       `Fundamentals of Machine Learning for Predictive Data Analytics:
       Algorithms, Worked Examples, and Case Studies
       <https://mitpress.mit.edu/books/fundamentals-machine-learning-predictive-data-analytics>`_.

Examples
--------
>>> from sklearn.metrics import balanced_accuracy_score
>>> y_true = [0, 1, 0, 0, 1, 0]
>>> y_pred = [0, 1, 0, 0, 0, 1]
>>> balanced_accuracy_score(y_true, y_pred)
0.625
*)

val brier_score_loss : ?sample_weight:Ndarray.t -> ?pos_label:[`Int of int | `String of string] -> y_true:Ndarray.t -> y_prob:Ndarray.t -> unit -> float
(**
Compute the Brier score.
The smaller the Brier score, the better, hence the naming with "loss".
Across all items in a set N predictions, the Brier score measures the
mean squared difference between (1) the predicted probability assigned
to the possible outcomes for item i, and (2) the actual outcome.
Therefore, the lower the Brier score is for a set of predictions, the
better the predictions are calibrated. Note that the Brier score always
takes on a value between zero and one, since this is the largest
possible difference between a predicted probability (which must be
between zero and one) and the actual outcome (which can take on values
of only 0 and 1). The Brier loss is composed of refinement loss and
calibration loss.
The Brier score is appropriate for binary and categorical outcomes that
can be structured as true or false, but is inappropriate for ordinal
variables which can take on three or more values (this is because the
Brier score assumes that all possible outcomes are equivalently
"distant" from one another). Which label is considered to be the positive
label is controlled via the parameter pos_label, which defaults to 1.
Read more in the :ref:`User Guide <calibration>`.

Parameters
----------
y_true : array, shape (n_samples,)
    True targets.

y_prob : array, shape (n_samples,)
    Probabilities of the positive class.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

pos_label : int or str, default=None
    Label of the positive class.
    Defaults to the greater label unless y_true is all 0 or all -1
    in which case pos_label defaults to 1.

Returns
-------
score : float
    Brier score

Examples
--------
>>> import numpy as np
>>> from sklearn.metrics import brier_score_loss
>>> y_true = np.array([0, 1, 1, 0])
>>> y_true_categorical = np.array(["spam", "ham", "ham", "spam"])
>>> y_prob = np.array([0.1, 0.9, 0.8, 0.3])
>>> brier_score_loss(y_true, y_prob)
0.037...
>>> brier_score_loss(y_true, 1-y_prob, pos_label=0)
0.037...
>>> brier_score_loss(y_true_categorical, y_prob, pos_label="ham")
0.037...
>>> brier_score_loss(y_true, np.array(y_prob) > 0.5)
0.0

References
----------
.. [1] `Wikipedia entry for the Brier score.
        <https://en.wikipedia.org/wiki/Brier_score>`_
*)

val calinski_harabasz_score : x:Ndarray.t -> labels:Ndarray.t -> unit -> float
(**
Compute the Calinski and Harabasz score.

It is also known as the Variance Ratio Criterion.

The score is defined as ratio between the within-cluster dispersion and
the between-cluster dispersion.

Read more in the :ref:`User Guide <calinski_harabasz_index>`.

Parameters
----------
X : array-like, shape (``n_samples``, ``n_features``)
    List of ``n_features``-dimensional data points. Each row corresponds
    to a single data point.

labels : array-like, shape (``n_samples``,)
    Predicted labels for each sample.

Returns
-------
score : float
    The resulting Calinski-Harabasz score.

References
----------
.. [1] `T. Calinski and J. Harabasz, 1974. "A dendrite method for cluster
   analysis". Communications in Statistics
   <https://www.tandfonline.com/doi/abs/10.1080/03610927408827101>`_
*)

val calinski_harabaz_score : x:Py.Object.t -> labels:Py.Object.t -> unit -> Py.Object.t
(**
DEPRECATED: Function 'calinski_harabaz_score' has been renamed to 'calinski_harabasz_score' and will be removed in version 0.23.
*)

val check_scoring : ?scoring:[`String of string | `Callable of Py.Object.t | `None] -> ?allow_none:bool -> estimator:Py.Object.t -> unit -> Py.Object.t
(**
Determine scorer from user options.

A TypeError will be thrown if the estimator cannot be scored.

Parameters
----------
estimator : estimator object implementing 'fit'
    The object to use to fit the data.

scoring : string, callable or None, optional, default: None
    A string (see model evaluation documentation) or
    a scorer callable object / function with signature
    ``scorer(estimator, X, y)``.

allow_none : boolean, optional, default: False
    If no scoring is specified and the estimator has no score function, we
    can either return None or raise an exception.

Returns
-------
scoring : callable
    A scorer callable object / function with signature
    ``scorer(estimator, X, y)``.
*)

val classification_report : ?labels:Ndarray.t -> ?target_names:string list -> ?sample_weight:Ndarray.t -> ?digits:int -> ?output_dict:bool -> ?zero_division:[`Warn | `PyObject of Py.Object.t] -> y_true:Py.Object.t -> y_pred:Py.Object.t -> unit -> Py.Object.t
(**
Build a text report showing the main classification metrics

Read more in the :ref:`User Guide <classification_report>`.

Parameters
----------
y_true : 1d array-like, or label indicator array / sparse matrix
    Ground truth (correct) target values.

y_pred : 1d array-like, or label indicator array / sparse matrix
    Estimated targets as returned by a classifier.

labels : array, shape = [n_labels]
    Optional list of label indices to include in the report.

target_names : list of strings
    Optional display names matching the labels (same order).

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

digits : int
    Number of digits for formatting output floating point values.
    When ``output_dict`` is ``True``, this will be ignored and the
    returned values will not be rounded.

output_dict : bool (default = False)
    If True, return output as dict

zero_division : "warn", 0 or 1, default="warn"
    Sets the value to return when there is a zero division. If set to
    "warn", this acts as 0, but warnings are also raised.

Returns
-------
report : string / dict
    Text summary of the precision, recall, F1 score for each class.
    Dictionary returned if output_dict is True. Dictionary has the
    following structure::

        {'label 1': {'precision':0.5,
                     'recall':1.0,
                     'f1-score':0.67,
                     'support':1},
         'label 2': { ... },
          ...
        }

    The reported averages include macro average (averaging the unweighted
    mean per label), weighted average (averaging the support-weighted mean
    per label), and sample average (only for multilabel classification).
    Micro average (averaging the total true positives, false negatives and
    false positives) is only shown for multi-label or multi-class
    with a subset of classes, because it corresponds to accuracy otherwise.
    See also :func:`precision_recall_fscore_support` for more details
    on averages.

    Note that in binary classification, recall of the positive class
    is also known as "sensitivity"; recall of the negative class is
    "specificity".

See also
--------
precision_recall_fscore_support, confusion_matrix,
multilabel_confusion_matrix

Examples
--------
>>> from sklearn.metrics import classification_report
>>> y_true = [0, 1, 2, 2, 2]
>>> y_pred = [0, 0, 2, 2, 1]
>>> target_names = ['class 0', 'class 1', 'class 2']
>>> print(classification_report(y_true, y_pred, target_names=target_names))
              precision    recall  f1-score   support
<BLANKLINE>
     class 0       0.50      1.00      0.67         1
     class 1       0.00      0.00      0.00         1
     class 2       1.00      0.67      0.80         3
<BLANKLINE>
    accuracy                           0.60         5
   macro avg       0.50      0.56      0.49         5
weighted avg       0.70      0.60      0.61         5
<BLANKLINE>
>>> y_pred = [1, 1, 0]
>>> y_true = [1, 1, 1]
>>> print(classification_report(y_true, y_pred, labels=[1, 2, 3]))
              precision    recall  f1-score   support
<BLANKLINE>
           1       1.00      0.67      0.80         3
           2       0.00      0.00      0.00         0
           3       0.00      0.00      0.00         0
<BLANKLINE>
   micro avg       1.00      0.67      0.80         3
   macro avg       0.33      0.22      0.27         3
weighted avg       1.00      0.67      0.80         3
<BLANKLINE>
*)

module Cluster : sig
val adjusted_mutual_info_score : ?average_method:string -> labels_true:Py.Object.t -> labels_pred:Py.Object.t -> unit -> Py.Object.t
(**
Adjusted Mutual Information between two clusterings.

Adjusted Mutual Information (AMI) is an adjustment of the Mutual
Information (MI) score to account for chance. It accounts for the fact that
the MI is generally higher for two clusterings with a larger number of
clusters, regardless of whether there is actually more information shared.
For two clusterings :math:`U` and :math:`V`, the AMI is given as::

    AMI(U, V) = [MI(U, V) - E(MI(U, V))] / [avg(H(U), H(V)) - E(MI(U, V))]

This metric is independent of the absolute values of the labels:
a permutation of the class or cluster label values won't change the
score value in any way.

This metric is furthermore symmetric: switching ``label_true`` with
``label_pred`` will return the same score value. This can be useful to
measure the agreement of two independent label assignments strategies
on the same dataset when the real ground truth is not known.

Be mindful that this function is an order of magnitude slower than other
metrics, such as the Adjusted Rand Index.

Read more in the :ref:`User Guide <mutual_info_score>`.

Parameters
----------
labels_true : int array, shape = [n_samples]
    A clustering of the data into disjoint subsets.

labels_pred : int array-like of shape (n_samples,)
    A clustering of the data into disjoint subsets.

average_method : string, optional (default: 'arithmetic')
    How to compute the normalizer in the denominator. Possible options
    are 'min', 'geometric', 'arithmetic', and 'max'.

    .. versionadded:: 0.20

    .. versionchanged:: 0.22
       The default value of ``average_method`` changed from 'max' to
       'arithmetic'.

Returns
-------
ami: float (upperlimited by 1.0)
   The AMI returns a value of 1 when the two partitions are identical
   (ie perfectly matched). Random partitions (independent labellings) have
   an expected AMI around 0 on average hence can be negative.

See also
--------
adjusted_rand_score: Adjusted Rand Index
mutual_info_score: Mutual Information (not adjusted for chance)

Examples
--------

Perfect labelings are both homogeneous and complete, hence have
score 1.0::

  >>> from sklearn.metrics.cluster import adjusted_mutual_info_score
  >>> adjusted_mutual_info_score([0, 0, 1, 1], [0, 0, 1, 1])
  ... # doctest: +SKIP
  1.0
  >>> adjusted_mutual_info_score([0, 0, 1, 1], [1, 1, 0, 0])
  ... # doctest: +SKIP
  1.0

If classes members are completely split across different clusters,
the assignment is totally in-complete, hence the AMI is null::

  >>> adjusted_mutual_info_score([0, 0, 0, 0], [0, 1, 2, 3])
  ... # doctest: +SKIP
  0.0

References
----------
.. [1] `Vinh, Epps, and Bailey, (2010). Information Theoretic Measures for
   Clusterings Comparison: Variants, Properties, Normalization and
   Correction for Chance, JMLR
   <http://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf>`_

.. [2] `Wikipedia entry for the Adjusted Mutual Information
   <https://en.wikipedia.org/wiki/Adjusted_Mutual_Information>`_
*)

val adjusted_rand_score : labels_true:Py.Object.t -> labels_pred:Ndarray.t -> unit -> float
(**
Rand index adjusted for chance.

The Rand Index computes a similarity measure between two clusterings
by considering all pairs of samples and counting pairs that are
assigned in the same or different clusters in the predicted and
true clusterings.

The raw RI score is then "adjusted for chance" into the ARI score
using the following scheme::

    ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)

The adjusted Rand index is thus ensured to have a value close to
0.0 for random labeling independently of the number of clusters and
samples and exactly 1.0 when the clusterings are identical (up to
a permutation).

ARI is a symmetric measure::

    adjusted_rand_score(a, b) == adjusted_rand_score(b, a)

Read more in the :ref:`User Guide <adjusted_rand_score>`.

Parameters
----------
labels_true : int array, shape = [n_samples]
    Ground truth class labels to be used as a reference

labels_pred : array-like of shape (n_samples,)
    Cluster labels to evaluate

Returns
-------
ari : float
   Similarity score between -1.0 and 1.0. Random labelings have an ARI
   close to 0.0. 1.0 stands for perfect match.

Examples
--------

Perfectly matching labelings have a score of 1 even

  >>> from sklearn.metrics.cluster import adjusted_rand_score
  >>> adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 1])
  1.0
  >>> adjusted_rand_score([0, 0, 1, 1], [1, 1, 0, 0])
  1.0

Labelings that assign all classes members to the same clusters
are complete be not always pure, hence penalized::

  >>> adjusted_rand_score([0, 0, 1, 2], [0, 0, 1, 1])
  0.57...

ARI is symmetric, so labelings that have pure clusters with members
coming from the same classes but unnecessary splits are penalized::

  >>> adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 2])
  0.57...

If classes members are completely split across different clusters, the
assignment is totally incomplete, hence the ARI is very low::

  >>> adjusted_rand_score([0, 0, 0, 0], [0, 1, 2, 3])
  0.0

References
----------

.. [Hubert1985] L. Hubert and P. Arabie, Comparing Partitions,
  Journal of Classification 1985
  https://link.springer.com/article/10.1007%2FBF01908075

.. [wk] https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index

See also
--------
adjusted_mutual_info_score: Adjusted Mutual Information
*)

val calinski_harabasz_score : x:Ndarray.t -> labels:Ndarray.t -> unit -> float
(**
Compute the Calinski and Harabasz score.

It is also known as the Variance Ratio Criterion.

The score is defined as ratio between the within-cluster dispersion and
the between-cluster dispersion.

Read more in the :ref:`User Guide <calinski_harabasz_index>`.

Parameters
----------
X : array-like, shape (``n_samples``, ``n_features``)
    List of ``n_features``-dimensional data points. Each row corresponds
    to a single data point.

labels : array-like, shape (``n_samples``,)
    Predicted labels for each sample.

Returns
-------
score : float
    The resulting Calinski-Harabasz score.

References
----------
.. [1] `T. Calinski and J. Harabasz, 1974. "A dendrite method for cluster
   analysis". Communications in Statistics
   <https://www.tandfonline.com/doi/abs/10.1080/03610927408827101>`_
*)

val calinski_harabaz_score : x:Py.Object.t -> labels:Py.Object.t -> unit -> Py.Object.t
(**
DEPRECATED: Function 'calinski_harabaz_score' has been renamed to 'calinski_harabasz_score' and will be removed in version 0.23.
*)

val completeness_score : labels_true:Py.Object.t -> labels_pred:Ndarray.t -> unit -> float
(**
Completeness metric of a cluster labeling given a ground truth.

A clustering result satisfies completeness if all the data points
that are members of a given class are elements of the same cluster.

This metric is independent of the absolute values of the labels:
a permutation of the class or cluster label values won't change the
score value in any way.

This metric is not symmetric: switching ``label_true`` with ``label_pred``
will return the :func:`homogeneity_score` which will be different in
general.

Read more in the :ref:`User Guide <homogeneity_completeness>`.

Parameters
----------
labels_true : int array, shape = [n_samples]
    ground truth class labels to be used as a reference

labels_pred : array-like of shape (n_samples,)
    cluster labels to evaluate

Returns
-------
completeness : float
   score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling

References
----------

.. [1] `Andrew Rosenberg and Julia Hirschberg, 2007. V-Measure: A
   conditional entropy-based external cluster evaluation measure
   <https://aclweb.org/anthology/D/D07/D07-1043.pdf>`_

See also
--------
homogeneity_score
v_measure_score

Examples
--------

Perfect labelings are complete::

  >>> from sklearn.metrics.cluster import completeness_score
  >>> completeness_score([0, 0, 1, 1], [1, 1, 0, 0])
  1.0

Non-perfect labelings that assign all classes members to the same clusters
are still complete::

  >>> print(completeness_score([0, 0, 1, 1], [0, 0, 0, 0]))
  1.0
  >>> print(completeness_score([0, 1, 2, 3], [0, 0, 1, 1]))
  0.999...

If classes members are split across different clusters, the
assignment cannot be complete::

  >>> print(completeness_score([0, 0, 1, 1], [0, 1, 0, 1]))
  0.0
  >>> print(completeness_score([0, 0, 0, 0], [0, 1, 2, 3]))
  0.0
*)

val consensus_score : ?similarity:[`String of string | `Callable of Py.Object.t] -> a:Py.Object.t -> b:Py.Object.t -> unit -> Py.Object.t
(**
The similarity of two sets of biclusters.

Similarity between individual biclusters is computed. Then the
best matching between sets is found using the Hungarian algorithm.
The final score is the sum of similarities divided by the size of
the larger set.

Read more in the :ref:`User Guide <biclustering>`.

Parameters
----------
a : (rows, columns)
    Tuple of row and column indicators for a set of biclusters.

b : (rows, columns)
    Another set of biclusters like ``a``.

similarity : string or function, optional, default: "jaccard"
    May be the string "jaccard" to use the Jaccard coefficient, or
    any function that takes four arguments, each of which is a 1d
    indicator vector: (a_rows, a_columns, b_rows, b_columns).

References
----------

* Hochreiter, Bodenhofer, et. al., 2010. `FABIA: factor analysis
  for bicluster acquisition
  <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2881408/>`__.
*)

val contingency_matrix : ?eps:[`Float of float | `None] -> ?sparse:bool -> labels_true:Py.Object.t -> labels_pred:Ndarray.t -> unit -> Py.Object.t
(**
Build a contingency matrix describing the relationship between labels.

Parameters
----------
labels_true : int array, shape = [n_samples]
    Ground truth class labels to be used as a reference

labels_pred : array-like of shape (n_samples,)
    Cluster labels to evaluate

eps : None or float, optional.
    If a float, that value is added to all values in the contingency
    matrix. This helps to stop NaN propagation.
    If ``None``, nothing is adjusted.

sparse : boolean, optional.
    If True, return a sparse CSR continency matrix. If ``eps is not None``,
    and ``sparse is True``, will throw ValueError.

    .. versionadded:: 0.18

Returns
-------
contingency : {array-like, sparse}, shape=[n_classes_true, n_classes_pred]
    Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
    true class :math:`i` and in predicted class :math:`j`. If
    ``eps is None``, the dtype of this array will be integer. If ``eps`` is
    given, the dtype will be float.
    Will be a ``scipy.sparse.csr_matrix`` if ``sparse=True``.
*)

val davies_bouldin_score : x:Ndarray.t -> labels:Ndarray.t -> unit -> float
(**
Computes the Davies-Bouldin score.

The score is defined as the average similarity measure of each cluster with
its most similar cluster, where similarity is the ratio of within-cluster
distances to between-cluster distances. Thus, clusters which are farther
apart and less dispersed will result in a better score.

The minimum score is zero, with lower values indicating better clustering.

Read more in the :ref:`User Guide <davies-bouldin_index>`.

Parameters
----------
X : array-like, shape (``n_samples``, ``n_features``)
    List of ``n_features``-dimensional data points. Each row corresponds
    to a single data point.

labels : array-like, shape (``n_samples``,)
    Predicted labels for each sample.

Returns
-------
score: float
    The resulting Davies-Bouldin score.

References
----------
.. [1] Davies, David L.; Bouldin, Donald W. (1979).
   `"A Cluster Separation Measure"
   <https://ieeexplore.ieee.org/document/4766909>`__.
   IEEE Transactions on Pattern Analysis and Machine Intelligence.
   PAMI-1 (2): 224-227
*)

val entropy : labels:Py.Object.t -> unit -> Py.Object.t
(**
Calculates the entropy for a labeling.

Parameters
----------
labels : int array, shape = [n_samples]
    The labels

Notes
-----
The logarithm used is the natural logarithm (base-e).
*)

val fowlkes_mallows_score : ?sparse:bool -> labels_true:Py.Object.t -> labels_pred:Ndarray.t -> unit -> float
(**
Measure the similarity of two clusterings of a set of points.

The Fowlkes-Mallows index (FMI) is defined as the geometric mean between of
the precision and recall::

    FMI = TP / sqrt((TP + FP) * (TP + FN))

Where ``TP`` is the number of **True Positive** (i.e. the number of pair of
points that belongs in the same clusters in both ``labels_true`` and
``labels_pred``), ``FP`` is the number of **False Positive** (i.e. the
number of pair of points that belongs in the same clusters in
``labels_true`` and not in ``labels_pred``) and ``FN`` is the number of
**False Negative** (i.e the number of pair of points that belongs in the
same clusters in ``labels_pred`` and not in ``labels_True``).

The score ranges from 0 to 1. A high value indicates a good similarity
between two clusters.

Read more in the :ref:`User Guide <fowlkes_mallows_scores>`.

Parameters
----------
labels_true : int array, shape = (``n_samples``,)
    A clustering of the data into disjoint subsets.

labels_pred : array, shape = (``n_samples``, )
    A clustering of the data into disjoint subsets.

sparse : bool
    Compute contingency matrix internally with sparse matrix.

Returns
-------
score : float
   The resulting Fowlkes-Mallows score.

Examples
--------

Perfect labelings are both homogeneous and complete, hence have
score 1.0::

  >>> from sklearn.metrics.cluster import fowlkes_mallows_score
  >>> fowlkes_mallows_score([0, 0, 1, 1], [0, 0, 1, 1])
  1.0
  >>> fowlkes_mallows_score([0, 0, 1, 1], [1, 1, 0, 0])
  1.0

If classes members are completely split across different clusters,
the assignment is totally random, hence the FMI is null::

  >>> fowlkes_mallows_score([0, 0, 0, 0], [0, 1, 2, 3])
  0.0

References
----------
.. [1] `E. B. Fowkles and C. L. Mallows, 1983. "A method for comparing two
   hierarchical clusterings". Journal of the American Statistical
   Association
   <http://wildfire.stat.ucla.edu/pdflibrary/fowlkes.pdf>`_

.. [2] `Wikipedia entry for the Fowlkes-Mallows Index
       <https://en.wikipedia.org/wiki/Fowlkes-Mallows_index>`_
*)

val homogeneity_completeness_v_measure : ?beta:float -> labels_true:Py.Object.t -> labels_pred:Ndarray.t -> unit -> (float * float * float)
(**
Compute the homogeneity and completeness and V-Measure scores at once.

Those metrics are based on normalized conditional entropy measures of
the clustering labeling to evaluate given the knowledge of a Ground
Truth class labels of the same samples.

A clustering result satisfies homogeneity if all of its clusters
contain only data points which are members of a single class.

A clustering result satisfies completeness if all the data points
that are members of a given class are elements of the same cluster.

Both scores have positive values between 0.0 and 1.0, larger values
being desirable.

Those 3 metrics are independent of the absolute values of the labels:
a permutation of the class or cluster label values won't change the
score values in any way.

V-Measure is furthermore symmetric: swapping ``labels_true`` and
``label_pred`` will give the same score. This does not hold for
homogeneity and completeness. V-Measure is identical to
:func:`normalized_mutual_info_score` with the arithmetic averaging
method.

Read more in the :ref:`User Guide <homogeneity_completeness>`.

Parameters
----------
labels_true : int array, shape = [n_samples]
    ground truth class labels to be used as a reference

labels_pred : array-like of shape (n_samples,)
    cluster labels to evaluate

beta : float
    Ratio of weight attributed to ``homogeneity`` vs ``completeness``.
    If ``beta`` is greater than 1, ``completeness`` is weighted more
    strongly in the calculation. If ``beta`` is less than 1,
    ``homogeneity`` is weighted more strongly.

Returns
-------
homogeneity : float
   score between 0.0 and 1.0. 1.0 stands for perfectly homogeneous labeling

completeness : float
   score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling

v_measure : float
    harmonic mean of the first two

See also
--------
homogeneity_score
completeness_score
v_measure_score
*)

val homogeneity_score : labels_true:Py.Object.t -> labels_pred:Ndarray.t -> unit -> float
(**
Homogeneity metric of a cluster labeling given a ground truth.

A clustering result satisfies homogeneity if all of its clusters
contain only data points which are members of a single class.

This metric is independent of the absolute values of the labels:
a permutation of the class or cluster label values won't change the
score value in any way.

This metric is not symmetric: switching ``label_true`` with ``label_pred``
will return the :func:`completeness_score` which will be different in
general.

Read more in the :ref:`User Guide <homogeneity_completeness>`.

Parameters
----------
labels_true : int array, shape = [n_samples]
    ground truth class labels to be used as a reference

labels_pred : array-like of shape (n_samples,)
    cluster labels to evaluate

Returns
-------
homogeneity : float
   score between 0.0 and 1.0. 1.0 stands for perfectly homogeneous labeling

References
----------

.. [1] `Andrew Rosenberg and Julia Hirschberg, 2007. V-Measure: A
   conditional entropy-based external cluster evaluation measure
   <https://aclweb.org/anthology/D/D07/D07-1043.pdf>`_

See also
--------
completeness_score
v_measure_score

Examples
--------

Perfect labelings are homogeneous::

  >>> from sklearn.metrics.cluster import homogeneity_score
  >>> homogeneity_score([0, 0, 1, 1], [1, 1, 0, 0])
  1.0

Non-perfect labelings that further split classes into more clusters can be
perfectly homogeneous::

  >>> print("%.6f" % homogeneity_score([0, 0, 1, 1], [0, 0, 1, 2]))
  1.000000
  >>> print("%.6f" % homogeneity_score([0, 0, 1, 1], [0, 1, 2, 3]))
  1.000000

Clusters that include samples from different classes do not make for an
homogeneous labeling::

  >>> print("%.6f" % homogeneity_score([0, 0, 1, 1], [0, 1, 0, 1]))
  0.0...
  >>> print("%.6f" % homogeneity_score([0, 0, 1, 1], [0, 0, 0, 0]))
  0.0...
*)

val mutual_info_score : ?contingency:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t | `None] -> labels_true:Py.Object.t -> labels_pred:Py.Object.t -> unit -> float
(**
Mutual Information between two clusterings.

The Mutual Information is a measure of the similarity between two labels of
the same data. Where :math:`|U_i|` is the number of the samples
in cluster :math:`U_i` and :math:`|V_j|` is the number of the
samples in cluster :math:`V_j`, the Mutual Information
between clusterings :math:`U` and :math:`V` is given as:

.. math::

    MI(U,V)=\sum_{i=1}^{ |U| } \sum_{j=1}^{ |V| } \frac{ |U_i\cap V_j| }{N}
    \log\frac{N|U_i \cap V_j| }{ |U_i||V_j| }

This metric is independent of the absolute values of the labels:
a permutation of the class or cluster label values won't change the
score value in any way.

This metric is furthermore symmetric: switching ``label_true`` with
``label_pred`` will return the same score value. This can be useful to
measure the agreement of two independent label assignments strategies
on the same dataset when the real ground truth is not known.

Read more in the :ref:`User Guide <mutual_info_score>`.

Parameters
----------
labels_true : int array, shape = [n_samples]
    A clustering of the data into disjoint subsets.

labels_pred : int array-like of shape (n_samples,)
    A clustering of the data into disjoint subsets.

contingency : {None, array, sparse matrix},                   shape = [n_classes_true, n_classes_pred]
    A contingency matrix given by the :func:`contingency_matrix` function.
    If value is ``None``, it will be computed, otherwise the given value is
    used, with ``labels_true`` and ``labels_pred`` ignored.

Returns
-------
mi : float
   Mutual information, a non-negative value

Notes
-----
The logarithm used is the natural logarithm (base-e).

See also
--------
adjusted_mutual_info_score: Adjusted against chance Mutual Information
normalized_mutual_info_score: Normalized Mutual Information
*)

val normalized_mutual_info_score : ?average_method:string -> labels_true:Py.Object.t -> labels_pred:Py.Object.t -> unit -> float
(**
Normalized Mutual Information between two clusterings.

Normalized Mutual Information (NMI) is a normalization of the Mutual
Information (MI) score to scale the results between 0 (no mutual
information) and 1 (perfect correlation). In this function, mutual
information is normalized by some generalized mean of ``H(labels_true)``
and ``H(labels_pred))``, defined by the `average_method`.

This measure is not adjusted for chance. Therefore
:func:`adjusted_mutual_info_score` might be preferred.

This metric is independent of the absolute values of the labels:
a permutation of the class or cluster label values won't change the
score value in any way.

This metric is furthermore symmetric: switching ``label_true`` with
``label_pred`` will return the same score value. This can be useful to
measure the agreement of two independent label assignments strategies
on the same dataset when the real ground truth is not known.

Read more in the :ref:`User Guide <mutual_info_score>`.

Parameters
----------
labels_true : int array, shape = [n_samples]
    A clustering of the data into disjoint subsets.

labels_pred : int array-like of shape (n_samples,)
    A clustering of the data into disjoint subsets.

average_method : string, optional (default: 'arithmetic')
    How to compute the normalizer in the denominator. Possible options
    are 'min', 'geometric', 'arithmetic', and 'max'.

    .. versionadded:: 0.20

    .. versionchanged:: 0.22
       The default value of ``average_method`` changed from 'geometric' to
       'arithmetic'.

Returns
-------
nmi : float
   score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling

See also
--------
v_measure_score: V-Measure (NMI with arithmetic mean option.)
adjusted_rand_score: Adjusted Rand Index
adjusted_mutual_info_score: Adjusted Mutual Information (adjusted
    against chance)

Examples
--------

Perfect labelings are both homogeneous and complete, hence have
score 1.0::

  >>> from sklearn.metrics.cluster import normalized_mutual_info_score
  >>> normalized_mutual_info_score([0, 0, 1, 1], [0, 0, 1, 1])
  ... # doctest: +SKIP
  1.0
  >>> normalized_mutual_info_score([0, 0, 1, 1], [1, 1, 0, 0])
  ... # doctest: +SKIP
  1.0

If classes members are completely split across different clusters,
the assignment is totally in-complete, hence the NMI is null::

  >>> normalized_mutual_info_score([0, 0, 0, 0], [0, 1, 2, 3])
  ... # doctest: +SKIP
  0.0
*)

val silhouette_samples : ?metric:[`String of string | `Callable of Py.Object.t] -> ?kwds:(string * Py.Object.t) list -> x:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> labels:Ndarray.t -> unit -> Ndarray.t
(**
Compute the Silhouette Coefficient for each sample.

The Silhouette Coefficient is a measure of how well samples are clustered
with samples that are similar to themselves. Clustering models with a high
Silhouette Coefficient are said to be dense, where samples in the same
cluster are similar to each other, and well separated, where samples in
different clusters are not very similar to each other.

The Silhouette Coefficient is calculated using the mean intra-cluster
distance (``a``) and the mean nearest-cluster distance (``b``) for each
sample.  The Silhouette Coefficient for a sample is ``(b - a) / max(a,
b)``.
Note that Silhouette Coefficient is only defined if number of labels
is 2 <= n_labels <= n_samples - 1.

This function returns the Silhouette Coefficient for each sample.

The best value is 1 and the worst value is -1. Values near 0 indicate
overlapping clusters.

Read more in the :ref:`User Guide <silhouette_coefficient>`.

Parameters
----------
X : array [n_samples_a, n_samples_a] if metric == "precomputed", or,              [n_samples_a, n_features] otherwise
    Array of pairwise distances between samples, or a feature array.

labels : array, shape = [n_samples]
         label values for each sample

metric : string, or callable
    The metric to use when calculating distance between instances in a
    feature array. If metric is a string, it must be one of the options
    allowed by :func:`sklearn.metrics.pairwise.pairwise_distances`. If X is
    the distance array itself, use "precomputed" as the metric. Precomputed
    distance matrices must have 0 along the diagonal.

`**kwds` : optional keyword parameters
    Any further parameters are passed directly to the distance function.
    If using a ``scipy.spatial.distance`` metric, the parameters are still
    metric dependent. See the scipy docs for usage examples.

Returns
-------
silhouette : array, shape = [n_samples]
    Silhouette Coefficient for each samples.

References
----------

.. [1] `Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
   Interpretation and Validation of Cluster Analysis". Computational
   and Applied Mathematics 20: 53-65.
   <https://www.sciencedirect.com/science/article/pii/0377042787901257>`_

.. [2] `Wikipedia entry on the Silhouette Coefficient
   <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_
*)

val silhouette_score : ?metric:[`String of string | `Callable of Py.Object.t] -> ?sample_size:[`Int of int | `None] -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> ?kwds:(string * Py.Object.t) list -> x:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> labels:Ndarray.t -> unit -> float
(**
Compute the mean Silhouette Coefficient of all samples.

The Silhouette Coefficient is calculated using the mean intra-cluster
distance (``a``) and the mean nearest-cluster distance (``b``) for each
sample.  The Silhouette Coefficient for a sample is ``(b - a) / max(a,
b)``.  To clarify, ``b`` is the distance between a sample and the nearest
cluster that the sample is not a part of.
Note that Silhouette Coefficient is only defined if number of labels
is 2 <= n_labels <= n_samples - 1.

This function returns the mean Silhouette Coefficient over all samples.
To obtain the values for each sample, use :func:`silhouette_samples`.

The best value is 1 and the worst value is -1. Values near 0 indicate
overlapping clusters. Negative values generally indicate that a sample has
been assigned to the wrong cluster, as a different cluster is more similar.

Read more in the :ref:`User Guide <silhouette_coefficient>`.

Parameters
----------
X : array [n_samples_a, n_samples_a] if metric == "precomputed", or,              [n_samples_a, n_features] otherwise
    Array of pairwise distances between samples, or a feature array.

labels : array, shape = [n_samples]
     Predicted labels for each sample.

metric : string, or callable
    The metric to use when calculating distance between instances in a
    feature array. If metric is a string, it must be one of the options
    allowed by :func:`metrics.pairwise.pairwise_distances
    <sklearn.metrics.pairwise.pairwise_distances>`. If X is the distance
    array itself, use ``metric="precomputed"``.

sample_size : int or None
    The size of the sample to use when computing the Silhouette Coefficient
    on a random subset of the data.
    If ``sample_size is None``, no sampling is used.

random_state : int, RandomState instance or None, optional (default=None)
    The generator used to randomly select a subset of samples.  If int,
    random_state is the seed used by the random number generator; If
    RandomState instance, random_state is the random number generator; If
    None, the random number generator is the RandomState instance used by
    `np.random`. Used when ``sample_size is not None``.

**kwds : optional keyword parameters
    Any further parameters are passed directly to the distance function.
    If using a scipy.spatial.distance metric, the parameters are still
    metric dependent. See the scipy docs for usage examples.

Returns
-------
silhouette : float
    Mean Silhouette Coefficient for all samples.

References
----------

.. [1] `Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
   Interpretation and Validation of Cluster Analysis". Computational
   and Applied Mathematics 20: 53-65.
   <https://www.sciencedirect.com/science/article/pii/0377042787901257>`_

.. [2] `Wikipedia entry on the Silhouette Coefficient
       <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_
*)

val v_measure_score : ?beta:float -> labels_true:Py.Object.t -> labels_pred:Ndarray.t -> unit -> float
(**
V-measure cluster labeling given a ground truth.

This score is identical to :func:`normalized_mutual_info_score` with
the ``'arithmetic'`` option for averaging.

The V-measure is the harmonic mean between homogeneity and completeness::

    v = (1 + beta) * homogeneity * completeness
         / (beta * homogeneity + completeness)

This metric is independent of the absolute values of the labels:
a permutation of the class or cluster label values won't change the
score value in any way.

This metric is furthermore symmetric: switching ``label_true`` with
``label_pred`` will return the same score value. This can be useful to
measure the agreement of two independent label assignments strategies
on the same dataset when the real ground truth is not known.


Read more in the :ref:`User Guide <homogeneity_completeness>`.

Parameters
----------
labels_true : int array, shape = [n_samples]
    ground truth class labels to be used as a reference

labels_pred : array-like of shape (n_samples,)
    cluster labels to evaluate

beta : float
    Ratio of weight attributed to ``homogeneity`` vs ``completeness``.
    If ``beta`` is greater than 1, ``completeness`` is weighted more
    strongly in the calculation. If ``beta`` is less than 1,
    ``homogeneity`` is weighted more strongly.

Returns
-------
v_measure : float
   score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling

References
----------

.. [1] `Andrew Rosenberg and Julia Hirschberg, 2007. V-Measure: A
   conditional entropy-based external cluster evaluation measure
   <https://aclweb.org/anthology/D/D07/D07-1043.pdf>`_

See also
--------
homogeneity_score
completeness_score
normalized_mutual_info_score

Examples
--------

Perfect labelings are both homogeneous and complete, hence have score 1.0::

  >>> from sklearn.metrics.cluster import v_measure_score
  >>> v_measure_score([0, 0, 1, 1], [0, 0, 1, 1])
  1.0
  >>> v_measure_score([0, 0, 1, 1], [1, 1, 0, 0])
  1.0

Labelings that assign all classes members to the same clusters
are complete be not homogeneous, hence penalized::

  >>> print("%.6f" % v_measure_score([0, 0, 1, 2], [0, 0, 1, 1]))
  0.8...
  >>> print("%.6f" % v_measure_score([0, 1, 2, 3], [0, 0, 1, 1]))
  0.66...

Labelings that have pure clusters with members coming from the same
classes are homogeneous but un-necessary splits harms completeness
and thus penalize V-measure as well::

  >>> print("%.6f" % v_measure_score([0, 0, 1, 1], [0, 0, 1, 2]))
  0.8...
  >>> print("%.6f" % v_measure_score([0, 0, 1, 1], [0, 1, 2, 3]))
  0.66...

If classes members are completely split across different clusters,
the assignment is totally incomplete, hence the V-Measure is null::

  >>> print("%.6f" % v_measure_score([0, 0, 0, 0], [0, 1, 2, 3]))
  0.0...

Clusters that include samples from totally different classes totally
destroy the homogeneity of the labeling, hence::

  >>> print("%.6f" % v_measure_score([0, 0, 1, 1], [0, 0, 0, 0]))
  0.0...
*)


end

val cohen_kappa_score : ?labels:Ndarray.t -> ?weights:string -> ?sample_weight:Ndarray.t -> y1:Ndarray.t -> y2:Ndarray.t -> unit -> float
(**
Cohen's kappa: a statistic that measures inter-annotator agreement.

This function computes Cohen's kappa [1]_, a score that expresses the level
of agreement between two annotators on a classification problem. It is
defined as

.. math::
    \kappa = (p_o - p_e) / (1 - p_e)

where :math:`p_o` is the empirical probability of agreement on the label
assigned to any sample (the observed agreement ratio), and :math:`p_e` is
the expected agreement when both annotators assign labels randomly.
:math:`p_e` is estimated using a per-annotator empirical prior over the
class labels [2]_.

Read more in the :ref:`User Guide <cohen_kappa>`.

Parameters
----------
y1 : array, shape = [n_samples]
    Labels assigned by the first annotator.

y2 : array, shape = [n_samples]
    Labels assigned by the second annotator. The kappa statistic is
    symmetric, so swapping ``y1`` and ``y2`` doesn't change the value.

labels : array, shape = [n_classes], optional
    List of labels to index the matrix. This may be used to select a
    subset of labels. If None, all labels that appear at least once in
    ``y1`` or ``y2`` are used.

weights : str, optional
    Weighting type to calculate the score. None means no weighted;
    "linear" means linear weighted; "quadratic" means quadratic weighted.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

Returns
-------
kappa : float
    The kappa statistic, which is a number between -1 and 1. The maximum
    value means complete agreement; zero or lower means chance agreement.

References
----------
.. [1] J. Cohen (1960). "A coefficient of agreement for nominal scales".
       Educational and Psychological Measurement 20(1):37-46.
       doi:10.1177/001316446002000104.
.. [2] `R. Artstein and M. Poesio (2008). "Inter-coder agreement for
       computational linguistics". Computational Linguistics 34(4):555-596.
       <https://www.mitpressjournals.org/doi/pdf/10.1162/coli.07-034-R2>`_
.. [3] `Wikipedia entry for the Cohen's kappa.
        <https://en.wikipedia.org/wiki/Cohen%27s_kappa>`_
*)

val completeness_score : labels_true:Py.Object.t -> labels_pred:Ndarray.t -> unit -> float
(**
Completeness metric of a cluster labeling given a ground truth.

A clustering result satisfies completeness if all the data points
that are members of a given class are elements of the same cluster.

This metric is independent of the absolute values of the labels:
a permutation of the class or cluster label values won't change the
score value in any way.

This metric is not symmetric: switching ``label_true`` with ``label_pred``
will return the :func:`homogeneity_score` which will be different in
general.

Read more in the :ref:`User Guide <homogeneity_completeness>`.

Parameters
----------
labels_true : int array, shape = [n_samples]
    ground truth class labels to be used as a reference

labels_pred : array-like of shape (n_samples,)
    cluster labels to evaluate

Returns
-------
completeness : float
   score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling

References
----------

.. [1] `Andrew Rosenberg and Julia Hirschberg, 2007. V-Measure: A
   conditional entropy-based external cluster evaluation measure
   <https://aclweb.org/anthology/D/D07/D07-1043.pdf>`_

See also
--------
homogeneity_score
v_measure_score

Examples
--------

Perfect labelings are complete::

  >>> from sklearn.metrics.cluster import completeness_score
  >>> completeness_score([0, 0, 1, 1], [1, 1, 0, 0])
  1.0

Non-perfect labelings that assign all classes members to the same clusters
are still complete::

  >>> print(completeness_score([0, 0, 1, 1], [0, 0, 0, 0]))
  1.0
  >>> print(completeness_score([0, 1, 2, 3], [0, 0, 1, 1]))
  0.999...

If classes members are split across different clusters, the
assignment cannot be complete::

  >>> print(completeness_score([0, 0, 1, 1], [0, 1, 0, 1]))
  0.0
  >>> print(completeness_score([0, 0, 0, 0], [0, 1, 2, 3]))
  0.0
*)

val confusion_matrix : ?labels:Ndarray.t -> ?sample_weight:Ndarray.t -> ?normalize:[`True | `Pred | `All] -> y_true:Ndarray.t -> y_pred:Ndarray.t -> unit -> Ndarray.t
(**
Compute confusion matrix to evaluate the accuracy of a classification.

By definition a confusion matrix :math:`C` is such that :math:`C_{i, j}`
is equal to the number of observations known to be in group :math:`i` and
predicted to be in group :math:`j`.

Thus in binary classification, the count of true negatives is
:math:`C_{0,0}`, false negatives is :math:`C_{1,0}`, true positives is
:math:`C_{1,1}` and false positives is :math:`C_{0,1}`.

Read more in the :ref:`User Guide <confusion_matrix>`.

Parameters
----------
y_true : array-like of shape (n_samples,)
    Ground truth (correct) target values.

y_pred : array-like of shape (n_samples,)
    Estimated targets as returned by a classifier.

labels : array-like of shape (n_classes), default=None
    List of labels to index the matrix. This may be used to reorder
    or select a subset of labels.
    If ``None`` is given, those that appear at least once
    in ``y_true`` or ``y_pred`` are used in sorted order.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

normalize : {'true', 'pred', 'all'}, default=None
    Normalizes confusion matrix over the true (rows), predicted (columns)
    conditions or all the population. If None, confusion matrix will not be
    normalized.

Returns
-------
C : ndarray of shape (n_classes, n_classes)
    Confusion matrix.

References
----------
.. [1] `Wikipedia entry for the Confusion matrix
       <https://en.wikipedia.org/wiki/Confusion_matrix>`_
       (Wikipedia and other references may use a different
       convention for axes)

Examples
--------
>>> from sklearn.metrics import confusion_matrix
>>> y_true = [2, 0, 2, 2, 0, 1]
>>> y_pred = [0, 0, 2, 2, 0, 2]
>>> confusion_matrix(y_true, y_pred)
array([[2, 0, 0],
       [0, 0, 1],
       [1, 0, 2]])

>>> y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
>>> y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
>>> confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
array([[2, 0, 0],
       [0, 0, 1],
       [1, 0, 2]])

In the binary case, we can extract true positives, etc as follows:

>>> tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
>>> (tn, fp, fn, tp)
(0, 2, 1, 1)
*)

val consensus_score : ?similarity:[`String of string | `Callable of Py.Object.t] -> a:Py.Object.t -> b:Py.Object.t -> unit -> Py.Object.t
(**
The similarity of two sets of biclusters.

Similarity between individual biclusters is computed. Then the
best matching between sets is found using the Hungarian algorithm.
The final score is the sum of similarities divided by the size of
the larger set.

Read more in the :ref:`User Guide <biclustering>`.

Parameters
----------
a : (rows, columns)
    Tuple of row and column indicators for a set of biclusters.

b : (rows, columns)
    Another set of biclusters like ``a``.

similarity : string or function, optional, default: "jaccard"
    May be the string "jaccard" to use the Jaccard coefficient, or
    any function that takes four arguments, each of which is a 1d
    indicator vector: (a_rows, a_columns, b_rows, b_columns).

References
----------

* Hochreiter, Bodenhofer, et. al., 2010. `FABIA: factor analysis
  for bicluster acquisition
  <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2881408/>`__.
*)

val coverage_error : ?sample_weight:Ndarray.t -> y_true:Ndarray.t -> y_score:Ndarray.t -> unit -> float
(**
Coverage error measure

Compute how far we need to go through the ranked scores to cover all
true labels. The best value is equal to the average number
of labels in ``y_true`` per sample.

Ties in ``y_scores`` are broken by giving maximal rank that would have
been assigned to all tied values.

Note: Our implementation's score is 1 greater than the one given in
Tsoumakas et al., 2010. This extends it to handle the degenerate case
in which an instance has 0 true labels.

Read more in the :ref:`User Guide <coverage_error>`.

Parameters
----------
y_true : array, shape = [n_samples, n_labels]
    True binary labels in binary indicator format.

y_score : array, shape = [n_samples, n_labels]
    Target scores, can either be probability estimates of the positive
    class, confidence values, or non-thresholded measure of decisions
    (as returned by "decision_function" on some classifiers).

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

Returns
-------
coverage_error : float

References
----------
.. [1] Tsoumakas, G., Katakis, I., & Vlahavas, I. (2010).
       Mining multi-label data. In Data mining and knowledge discovery
       handbook (pp. 667-685). Springer US.
*)

val davies_bouldin_score : x:Ndarray.t -> labels:Ndarray.t -> unit -> float
(**
Computes the Davies-Bouldin score.

The score is defined as the average similarity measure of each cluster with
its most similar cluster, where similarity is the ratio of within-cluster
distances to between-cluster distances. Thus, clusters which are farther
apart and less dispersed will result in a better score.

The minimum score is zero, with lower values indicating better clustering.

Read more in the :ref:`User Guide <davies-bouldin_index>`.

Parameters
----------
X : array-like, shape (``n_samples``, ``n_features``)
    List of ``n_features``-dimensional data points. Each row corresponds
    to a single data point.

labels : array-like, shape (``n_samples``,)
    Predicted labels for each sample.

Returns
-------
score: float
    The resulting Davies-Bouldin score.

References
----------
.. [1] Davies, David L.; Bouldin, Donald W. (1979).
   `"A Cluster Separation Measure"
   <https://ieeexplore.ieee.org/document/4766909>`__.
   IEEE Transactions on Pattern Analysis and Machine Intelligence.
   PAMI-1 (2): 224-227
*)

val dcg_score : ?k:int -> ?log_base:float -> ?sample_weight:Ndarray.t -> ?ignore_ties:bool -> y_true:Ndarray.t -> y_score:Ndarray.t -> unit -> float
(**
Compute Discounted Cumulative Gain.

Sum the true scores ranked in the order induced by the predicted scores,
after applying a logarithmic discount.

This ranking metric yields a high value if true labels are ranked high by
``y_score``.

Usually the Normalized Discounted Cumulative Gain (NDCG, computed by
ndcg_score) is preferred.

Parameters
----------
y_true : ndarray, shape (n_samples, n_labels)
    True targets of multilabel classification, or true scores of entities
    to be ranked.

y_score : ndarray, shape (n_samples, n_labels)
    Target scores, can either be probability estimates, confidence values,
    or non-thresholded measure of decisions (as returned by
    "decision_function" on some classifiers).

k : int, optional (default=None)
    Only consider the highest k scores in the ranking. If None, use all
    outputs.

log_base : float, optional (default=2)
    Base of the logarithm used for the discount. A low value means a
    sharper discount (top results are more important).

sample_weight : ndarray, shape (n_samples,), optional (default=None)
    Sample weights. If None, all samples are given the same weight.

ignore_ties : bool, optional (default=False)
    Assume that there are no ties in y_score (which is likely to be the
    case if y_score is continuous) for efficiency gains.

Returns
-------
discounted_cumulative_gain : float
    The averaged sample DCG scores.

See also
--------
ndcg_score :
    The Discounted Cumulative Gain divided by the Ideal Discounted
    Cumulative Gain (the DCG obtained for a perfect ranking), in order to
    have a score between 0 and 1.

References
----------
`Wikipedia entry for Discounted Cumulative Gain
<https://en.wikipedia.org/wiki/Discounted_cumulative_gain>`_

Jarvelin, K., & Kekalainen, J. (2002).
Cumulated gain-based evaluation of IR techniques. ACM Transactions on
Information Systems (TOIS), 20(4), 422-446.

Wang, Y., Wang, L., Li, Y., He, D., Chen, W., & Liu, T. Y. (2013, May).
A theoretical analysis of NDCG ranking measures. In Proceedings of the 26th
Annual Conference on Learning Theory (COLT 2013)

McSherry, F., & Najork, M. (2008, March). Computing information retrieval
performance measures efficiently in the presence of tied scores. In
European conference on information retrieval (pp. 414-421). Springer,
Berlin, Heidelberg.

Examples
--------
>>> from sklearn.metrics import dcg_score
>>> # we have groud-truth relevance of some answers to a query:
>>> true_relevance = np.asarray([[10, 0, 0, 1, 5]])
>>> # we predict scores for the answers
>>> scores = np.asarray([[.1, .2, .3, 4, 70]])
>>> dcg_score(true_relevance, scores) # doctest: +ELLIPSIS
9.49...
>>> # we can set k to truncate the sum; only top k answers contribute
>>> dcg_score(true_relevance, scores, k=2) # doctest: +ELLIPSIS
5.63...
>>> # now we have some ties in our prediction
>>> scores = np.asarray([[1, 0, 0, 0, 1]])
>>> # by default ties are averaged, so here we get the average true
>>> # relevance of our top predictions: (10 + 5) / 2 = 7.5
>>> dcg_score(true_relevance, scores, k=1) # doctest: +ELLIPSIS
7.5
>>> # we can choose to ignore ties for faster results, but only
>>> # if we know there aren't ties in our scores, otherwise we get
>>> # wrong results:
>>> dcg_score(true_relevance,
...           scores, k=1, ignore_ties=True) # doctest: +ELLIPSIS
5.0
*)

val euclidean_distances : ?y:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> ?y_norm_squared:Ndarray.t -> ?squared:bool -> ?x_norm_squared:Ndarray.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> unit -> Ndarray.t
(**
Considering the rows of X (and Y=X) as vectors, compute the
distance matrix between each pair of vectors.

For efficiency reasons, the euclidean distance between a pair of row
vector x and y is computed as::

    dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))

This formulation has two advantages over other ways of computing distances.
First, it is computationally efficient when dealing with sparse data.
Second, if one argument varies but the other remains unchanged, then
`dot(x, x)` and/or `dot(y, y)` can be pre-computed.

However, this is not the most precise way of doing this computation, and
the distance matrix returned by this function may not be exactly
symmetric as required by, e.g., ``scipy.spatial.distance`` functions.

Read more in the :ref:`User Guide <metrics>`.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples_1, n_features)

Y : {array-like, sparse matrix}, shape (n_samples_2, n_features)

Y_norm_squared : array-like, shape (n_samples_2, ), optional
    Pre-computed dot-products of vectors in Y (e.g.,
    ``(Y**2).sum(axis=1)``)
    May be ignored in some cases, see the note below.

squared : boolean, optional
    Return squared Euclidean distances.

X_norm_squared : array-like of shape (n_samples,), optional
    Pre-computed dot-products of vectors in X (e.g.,
    ``(X**2).sum(axis=1)``)
    May be ignored in some cases, see the note below.

Notes
-----
To achieve better accuracy, `X_norm_squared` and `Y_norm_squared` may be
unused if they are passed as ``float32``.

Returns
-------
distances : array, shape (n_samples_1, n_samples_2)

Examples
--------
>>> from sklearn.metrics.pairwise import euclidean_distances
>>> X = [[0, 1], [1, 1]]
>>> # distance between rows of X
>>> euclidean_distances(X, X)
array([[0., 1.],
       [1., 0.]])
>>> # get distance to origin
>>> euclidean_distances(X, [[0, 0]])
array([[1.        ],
       [1.41421356]])

See also
--------
paired_distances : distances betweens pairs of elements of X and Y.
*)

val explained_variance_score : ?sample_weight:Ndarray.t -> ?multioutput:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> y_true:Ndarray.t -> y_pred:Ndarray.t -> unit -> Py.Object.t
(**
Explained variance regression score function

Best possible score is 1.0, lower values are worse.

Read more in the :ref:`User Guide <explained_variance_score>`.

Parameters
----------
y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
    Ground truth (correct) target values.

y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
    Estimated target values.

sample_weight : array-like of shape (n_samples,), optional
    Sample weights.

multioutput : string in ['raw_values', 'uniform_average',                 'variance_weighted'] or array-like of shape (n_outputs)
    Defines aggregating of multiple output scores.
    Array-like value defines weights used to average scores.

    'raw_values' :
        Returns a full set of scores in case of multioutput input.

    'uniform_average' :
        Scores of all outputs are averaged with uniform weight.

    'variance_weighted' :
        Scores of all outputs are averaged, weighted by the variances
        of each individual output.

Returns
-------
score : float or ndarray of floats
    The explained variance or ndarray if 'multioutput' is 'raw_values'.

Notes
-----
This is not a symmetric function.

Examples
--------
>>> from sklearn.metrics import explained_variance_score
>>> y_true = [3, -0.5, 2, 7]
>>> y_pred = [2.5, 0.0, 2, 8]
>>> explained_variance_score(y_true, y_pred)
0.957...
>>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
>>> y_pred = [[0, 2], [-1, 2], [8, -5]]
>>> explained_variance_score(y_true, y_pred, multioutput='uniform_average')
0.983...
*)

val f1_score : ?labels:Py.Object.t -> ?pos_label:[`String of string | `Int of int] -> ?average:[`String of string | `Binary | `PyObject of Py.Object.t] -> ?sample_weight:Ndarray.t -> ?zero_division:[`Warn | `PyObject of Py.Object.t] -> y_true:Py.Object.t -> y_pred:Py.Object.t -> unit -> Py.Object.t
(**
Compute the F1 score, also known as balanced F-score or F-measure

The F1 score can be interpreted as a weighted average of the precision and
recall, where an F1 score reaches its best value at 1 and worst score at 0.
The relative contribution of precision and recall to the F1 score are
equal. The formula for the F1 score is::

    F1 = 2 * (precision * recall) / (precision + recall)

In the multi-class and multi-label case, this is the average of
the F1 score of each class with weighting depending on the ``average``
parameter.

Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

Parameters
----------
y_true : 1d array-like, or label indicator array / sparse matrix
    Ground truth (correct) target values.

y_pred : 1d array-like, or label indicator array / sparse matrix
    Estimated targets as returned by a classifier.

labels : list, optional
    The set of labels to include when ``average != 'binary'``, and their
    order if ``average is None``. Labels present in the data can be
    excluded, for example to calculate a multiclass average ignoring a
    majority negative class, while labels not present in the data will
    result in 0 components in a macro average. For multilabel targets,
    labels are column indices. By default, all labels in ``y_true`` and
    ``y_pred`` are used in sorted order.

    .. versionchanged:: 0.17
       parameter *labels* improved for multiclass problem.

pos_label : str or int, 1 by default
    The class to report if ``average='binary'`` and the data is binary.
    If the data are multiclass or multilabel, this will be ignored;
    setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
    scores for that label only.

average : string, [None, 'binary' (default), 'micro', 'macro', 'samples',                        'weighted']
    This parameter is required for multiclass/multilabel targets.
    If ``None``, the scores for each class are returned. Otherwise, this
    determines the type of averaging performed on the data:

    ``'binary'``:
        Only report results for the class specified by ``pos_label``.
        This is applicable only if targets (``y_{true,pred}``) are binary.
    ``'micro'``:
        Calculate metrics globally by counting the total true positives,
        false negatives and false positives.
    ``'macro'``:
        Calculate metrics for each label, and find their unweighted
        mean.  This does not take label imbalance into account.
    ``'weighted'``:
        Calculate metrics for each label, and find their average weighted
        by support (the number of true instances for each label). This
        alters 'macro' to account for label imbalance; it can result in an
        F-score that is not between precision and recall.
    ``'samples'``:
        Calculate metrics for each instance, and find their average (only
        meaningful for multilabel classification where this differs from
        :func:`accuracy_score`).

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

zero_division : "warn", 0 or 1, default="warn"
    Sets the value to return when there is a zero division, i.e. when all
    predictions and labels are negative. If set to "warn", this acts as 0,
    but warnings are also raised.

Returns
-------
f1_score : float or array of float, shape = [n_unique_labels]
    F1 score of the positive class in binary classification or weighted
    average of the F1 scores of each class for the multiclass task.

See also
--------
fbeta_score, precision_recall_fscore_support, jaccard_score,
multilabel_confusion_matrix

References
----------
.. [1] `Wikipedia entry for the F1-score
       <https://en.wikipedia.org/wiki/F1_score>`_

Examples
--------
>>> from sklearn.metrics import f1_score
>>> y_true = [0, 1, 2, 0, 1, 2]
>>> y_pred = [0, 2, 1, 0, 0, 1]
>>> f1_score(y_true, y_pred, average='macro')
0.26...
>>> f1_score(y_true, y_pred, average='micro')
0.33...
>>> f1_score(y_true, y_pred, average='weighted')
0.26...
>>> f1_score(y_true, y_pred, average=None)
array([0.8, 0. , 0. ])
>>> y_true = [0, 0, 0, 0, 0, 0]
>>> y_pred = [0, 0, 0, 0, 0, 0]
>>> f1_score(y_true, y_pred, zero_division=1)
1.0...

Notes
-----
When ``true positive + false positive == 0``, precision is undefined;
When ``true positive + false negative == 0``, recall is undefined.
In such cases, by default the metric will be set to 0, as will f-score,
and ``UndefinedMetricWarning`` will be raised. This behavior can be
modified with ``zero_division``.
*)

val fbeta_score : ?labels:Py.Object.t -> ?pos_label:[`String of string | `Int of int] -> ?average:[`String of string | `Binary | `PyObject of Py.Object.t] -> ?sample_weight:Ndarray.t -> ?zero_division:[`Warn | `PyObject of Py.Object.t] -> y_true:Py.Object.t -> y_pred:Py.Object.t -> beta:float -> unit -> Py.Object.t
(**
Compute the F-beta score

The F-beta score is the weighted harmonic mean of precision and recall,
reaching its optimal value at 1 and its worst value at 0.

The `beta` parameter determines the weight of recall in the combined
score. ``beta < 1`` lends more weight to precision, while ``beta > 1``
favors recall (``beta -> 0`` considers only precision, ``beta -> +inf``
only recall).

Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

Parameters
----------
y_true : 1d array-like, or label indicator array / sparse matrix
    Ground truth (correct) target values.

y_pred : 1d array-like, or label indicator array / sparse matrix
    Estimated targets as returned by a classifier.

beta : float
    Determines the weight of recall in the combined score.

labels : list, optional
    The set of labels to include when ``average != 'binary'``, and their
    order if ``average is None``. Labels present in the data can be
    excluded, for example to calculate a multiclass average ignoring a
    majority negative class, while labels not present in the data will
    result in 0 components in a macro average. For multilabel targets,
    labels are column indices. By default, all labels in ``y_true`` and
    ``y_pred`` are used in sorted order.

    .. versionchanged:: 0.17
       parameter *labels* improved for multiclass problem.

pos_label : str or int, 1 by default
    The class to report if ``average='binary'`` and the data is binary.
    If the data are multiclass or multilabel, this will be ignored;
    setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
    scores for that label only.

average : string, [None, 'binary' (default), 'micro', 'macro', 'samples',                        'weighted']
    This parameter is required for multiclass/multilabel targets.
    If ``None``, the scores for each class are returned. Otherwise, this
    determines the type of averaging performed on the data:

    ``'binary'``:
        Only report results for the class specified by ``pos_label``.
        This is applicable only if targets (``y_{true,pred}``) are binary.
    ``'micro'``:
        Calculate metrics globally by counting the total true positives,
        false negatives and false positives.
    ``'macro'``:
        Calculate metrics for each label, and find their unweighted
        mean.  This does not take label imbalance into account.
    ``'weighted'``:
        Calculate metrics for each label, and find their average weighted
        by support (the number of true instances for each label). This
        alters 'macro' to account for label imbalance; it can result in an
        F-score that is not between precision and recall.
    ``'samples'``:
        Calculate metrics for each instance, and find their average (only
        meaningful for multilabel classification where this differs from
        :func:`accuracy_score`).

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

zero_division : "warn", 0 or 1, default="warn"
    Sets the value to return when there is a zero division, i.e. when all
    predictions and labels are negative. If set to "warn", this acts as 0,
    but warnings are also raised.

Returns
-------
fbeta_score : float (if average is not None) or array of float, shape =        [n_unique_labels]
    F-beta score of the positive class in binary classification or weighted
    average of the F-beta score of each class for the multiclass task.

See also
--------
precision_recall_fscore_support, multilabel_confusion_matrix

References
----------
.. [1] R. Baeza-Yates and B. Ribeiro-Neto (2011).
       Modern Information Retrieval. Addison Wesley, pp. 327-328.

.. [2] `Wikipedia entry for the F1-score
       <https://en.wikipedia.org/wiki/F1_score>`_

Examples
--------
>>> from sklearn.metrics import fbeta_score
>>> y_true = [0, 1, 2, 0, 1, 2]
>>> y_pred = [0, 2, 1, 0, 0, 1]
>>> fbeta_score(y_true, y_pred, average='macro', beta=0.5)
0.23...
>>> fbeta_score(y_true, y_pred, average='micro', beta=0.5)
0.33...
>>> fbeta_score(y_true, y_pred, average='weighted', beta=0.5)
0.23...
>>> fbeta_score(y_true, y_pred, average=None, beta=0.5)
array([0.71..., 0.        , 0.        ])

Notes
-----
When ``true positive + false positive == 0`` or
``true positive + false negative == 0``, f-score returns 0 and raises
``UndefinedMetricWarning``. This behavior can be
modified with ``zero_division``.
*)

val fowlkes_mallows_score : ?sparse:bool -> labels_true:Py.Object.t -> labels_pred:Ndarray.t -> unit -> float
(**
Measure the similarity of two clusterings of a set of points.

The Fowlkes-Mallows index (FMI) is defined as the geometric mean between of
the precision and recall::

    FMI = TP / sqrt((TP + FP) * (TP + FN))

Where ``TP`` is the number of **True Positive** (i.e. the number of pair of
points that belongs in the same clusters in both ``labels_true`` and
``labels_pred``), ``FP`` is the number of **False Positive** (i.e. the
number of pair of points that belongs in the same clusters in
``labels_true`` and not in ``labels_pred``) and ``FN`` is the number of
**False Negative** (i.e the number of pair of points that belongs in the
same clusters in ``labels_pred`` and not in ``labels_True``).

The score ranges from 0 to 1. A high value indicates a good similarity
between two clusters.

Read more in the :ref:`User Guide <fowlkes_mallows_scores>`.

Parameters
----------
labels_true : int array, shape = (``n_samples``,)
    A clustering of the data into disjoint subsets.

labels_pred : array, shape = (``n_samples``, )
    A clustering of the data into disjoint subsets.

sparse : bool
    Compute contingency matrix internally with sparse matrix.

Returns
-------
score : float
   The resulting Fowlkes-Mallows score.

Examples
--------

Perfect labelings are both homogeneous and complete, hence have
score 1.0::

  >>> from sklearn.metrics.cluster import fowlkes_mallows_score
  >>> fowlkes_mallows_score([0, 0, 1, 1], [0, 0, 1, 1])
  1.0
  >>> fowlkes_mallows_score([0, 0, 1, 1], [1, 1, 0, 0])
  1.0

If classes members are completely split across different clusters,
the assignment is totally random, hence the FMI is null::

  >>> fowlkes_mallows_score([0, 0, 0, 0], [0, 1, 2, 3])
  0.0

References
----------
.. [1] `E. B. Fowkles and C. L. Mallows, 1983. "A method for comparing two
   hierarchical clusterings". Journal of the American Statistical
   Association
   <http://wildfire.stat.ucla.edu/pdflibrary/fowlkes.pdf>`_

.. [2] `Wikipedia entry for the Fowlkes-Mallows Index
       <https://en.wikipedia.org/wiki/Fowlkes-Mallows_index>`_
*)

val get_scorer : scoring:[`String of string | `Callable of Py.Object.t] -> unit -> Py.Object.t
(**
Get a scorer from string.

Read more in the :ref:`User Guide <scoring_parameter>`.

Parameters
----------
scoring : str | callable
    scoring method as string. If callable it is returned as is.

Returns
-------
scorer : callable
    The scorer.
*)

val hamming_loss : ?labels:Ndarray.t -> ?sample_weight:Ndarray.t -> y_true:Py.Object.t -> y_pred:Py.Object.t -> unit -> Py.Object.t
(**
Compute the average Hamming loss.

The Hamming loss is the fraction of labels that are incorrectly predicted.

Read more in the :ref:`User Guide <hamming_loss>`.

Parameters
----------
y_true : 1d array-like, or label indicator array / sparse matrix
    Ground truth (correct) labels.

y_pred : 1d array-like, or label indicator array / sparse matrix
    Predicted labels, as returned by a classifier.

labels : array, shape = [n_labels], optional (default='deprecated')
    Integer array of labels. If not provided, labels will be inferred
    from y_true and y_pred.

    .. versionadded:: 0.18
    .. deprecated:: 0.21
       This parameter ``labels`` is deprecated in version 0.21 and will
       be removed in version 0.23. Hamming loss uses ``y_true.shape[1]``
       for the number of labels when y_true is binary label indicators,
       so it is unnecessary for the user to specify.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

    .. versionadded:: 0.18

Returns
-------
loss : float or int,
    Return the average Hamming loss between element of ``y_true`` and
    ``y_pred``.

See Also
--------
accuracy_score, jaccard_score, zero_one_loss

Notes
-----
In multiclass classification, the Hamming loss corresponds to the Hamming
distance between ``y_true`` and ``y_pred`` which is equivalent to the
subset ``zero_one_loss`` function, when `normalize` parameter is set to
True.

In multilabel classification, the Hamming loss is different from the
subset zero-one loss. The zero-one loss considers the entire set of labels
for a given sample incorrect if it does not entirely match the true set of
labels. Hamming loss is more forgiving in that it penalizes only the
individual labels.

The Hamming loss is upperbounded by the subset zero-one loss, when
`normalize` parameter is set to True. It is always between 0 and 1,
lower being better.

References
----------
.. [1] Grigorios Tsoumakas, Ioannis Katakis. Multi-Label Classification:
       An Overview. International Journal of Data Warehousing & Mining,
       3(3), 1-13, July-September 2007.

.. [2] `Wikipedia entry on the Hamming distance
       <https://en.wikipedia.org/wiki/Hamming_distance>`_

Examples
--------
>>> from sklearn.metrics import hamming_loss
>>> y_pred = [1, 2, 3, 4]
>>> y_true = [2, 2, 3, 4]
>>> hamming_loss(y_true, y_pred)
0.25

In the multilabel case with binary label indicators:

>>> import numpy as np
>>> hamming_loss(np.array([[0, 1], [1, 1]]), np.zeros((2, 2)))
0.75
*)

val hinge_loss : ?labels:Ndarray.t -> ?sample_weight:Ndarray.t -> y_true:Ndarray.t -> pred_decision:Ndarray.t -> unit -> float
(**
Average hinge loss (non-regularized)

In binary class case, assuming labels in y_true are encoded with +1 and -1,
when a prediction mistake is made, ``margin = y_true * pred_decision`` is
always negative (since the signs disagree), implying ``1 - margin`` is
always greater than 1.  The cumulated hinge loss is therefore an upper
bound of the number of mistakes made by the classifier.

In multiclass case, the function expects that either all the labels are
included in y_true or an optional labels argument is provided which
contains all the labels. The multilabel margin is calculated according
to Crammer-Singer's method. As in the binary case, the cumulated hinge loss
is an upper bound of the number of mistakes made by the classifier.

Read more in the :ref:`User Guide <hinge_loss>`.

Parameters
----------
y_true : array, shape = [n_samples]
    True target, consisting of integers of two values. The positive label
    must be greater than the negative label.

pred_decision : array, shape = [n_samples] or [n_samples, n_classes]
    Predicted decisions, as output by decision_function (floats).

labels : array, optional, default None
    Contains all the labels for the problem. Used in multiclass hinge loss.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

Returns
-------
loss : float

References
----------
.. [1] `Wikipedia entry on the Hinge loss
       <https://en.wikipedia.org/wiki/Hinge_loss>`_

.. [2] Koby Crammer, Yoram Singer. On the Algorithmic
       Implementation of Multiclass Kernel-based Vector
       Machines. Journal of Machine Learning Research 2,
       (2001), 265-292

.. [3] `L1 AND L2 Regularization for Multiclass Hinge Loss Models
       by Robert C. Moore, John DeNero.
       <http://www.ttic.edu/sigml/symposium2011/papers/
       Moore+DeNero_Regularization.pdf>`_

Examples
--------
>>> from sklearn import svm
>>> from sklearn.metrics import hinge_loss
>>> X = [[0], [1]]
>>> y = [-1, 1]
>>> est = svm.LinearSVC(random_state=0)
>>> est.fit(X, y)
LinearSVC(random_state=0)
>>> pred_decision = est.decision_function([[-2], [3], [0.5]])
>>> pred_decision
array([-2.18...,  2.36...,  0.09...])
>>> hinge_loss([-1, 1, 1], pred_decision)
0.30...

In the multiclass case:

>>> import numpy as np
>>> X = np.array([[0], [1], [2], [3]])
>>> Y = np.array([0, 1, 2, 3])
>>> labels = np.array([0, 1, 2, 3])
>>> est = svm.LinearSVC()
>>> est.fit(X, Y)
LinearSVC()
>>> pred_decision = est.decision_function([[-1], [2], [3]])
>>> y_true = [0, 2, 3]
>>> hinge_loss(y_true, pred_decision, labels)
0.56...
*)

val homogeneity_completeness_v_measure : ?beta:float -> labels_true:Py.Object.t -> labels_pred:Ndarray.t -> unit -> (float * float * float)
(**
Compute the homogeneity and completeness and V-Measure scores at once.

Those metrics are based on normalized conditional entropy measures of
the clustering labeling to evaluate given the knowledge of a Ground
Truth class labels of the same samples.

A clustering result satisfies homogeneity if all of its clusters
contain only data points which are members of a single class.

A clustering result satisfies completeness if all the data points
that are members of a given class are elements of the same cluster.

Both scores have positive values between 0.0 and 1.0, larger values
being desirable.

Those 3 metrics are independent of the absolute values of the labels:
a permutation of the class or cluster label values won't change the
score values in any way.

V-Measure is furthermore symmetric: swapping ``labels_true`` and
``label_pred`` will give the same score. This does not hold for
homogeneity and completeness. V-Measure is identical to
:func:`normalized_mutual_info_score` with the arithmetic averaging
method.

Read more in the :ref:`User Guide <homogeneity_completeness>`.

Parameters
----------
labels_true : int array, shape = [n_samples]
    ground truth class labels to be used as a reference

labels_pred : array-like of shape (n_samples,)
    cluster labels to evaluate

beta : float
    Ratio of weight attributed to ``homogeneity`` vs ``completeness``.
    If ``beta`` is greater than 1, ``completeness`` is weighted more
    strongly in the calculation. If ``beta`` is less than 1,
    ``homogeneity`` is weighted more strongly.

Returns
-------
homogeneity : float
   score between 0.0 and 1.0. 1.0 stands for perfectly homogeneous labeling

completeness : float
   score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling

v_measure : float
    harmonic mean of the first two

See also
--------
homogeneity_score
completeness_score
v_measure_score
*)

val homogeneity_score : labels_true:Py.Object.t -> labels_pred:Ndarray.t -> unit -> float
(**
Homogeneity metric of a cluster labeling given a ground truth.

A clustering result satisfies homogeneity if all of its clusters
contain only data points which are members of a single class.

This metric is independent of the absolute values of the labels:
a permutation of the class or cluster label values won't change the
score value in any way.

This metric is not symmetric: switching ``label_true`` with ``label_pred``
will return the :func:`completeness_score` which will be different in
general.

Read more in the :ref:`User Guide <homogeneity_completeness>`.

Parameters
----------
labels_true : int array, shape = [n_samples]
    ground truth class labels to be used as a reference

labels_pred : array-like of shape (n_samples,)
    cluster labels to evaluate

Returns
-------
homogeneity : float
   score between 0.0 and 1.0. 1.0 stands for perfectly homogeneous labeling

References
----------

.. [1] `Andrew Rosenberg and Julia Hirschberg, 2007. V-Measure: A
   conditional entropy-based external cluster evaluation measure
   <https://aclweb.org/anthology/D/D07/D07-1043.pdf>`_

See also
--------
completeness_score
v_measure_score

Examples
--------

Perfect labelings are homogeneous::

  >>> from sklearn.metrics.cluster import homogeneity_score
  >>> homogeneity_score([0, 0, 1, 1], [1, 1, 0, 0])
  1.0

Non-perfect labelings that further split classes into more clusters can be
perfectly homogeneous::

  >>> print("%.6f" % homogeneity_score([0, 0, 1, 1], [0, 0, 1, 2]))
  1.000000
  >>> print("%.6f" % homogeneity_score([0, 0, 1, 1], [0, 1, 2, 3]))
  1.000000

Clusters that include samples from different classes do not make for an
homogeneous labeling::

  >>> print("%.6f" % homogeneity_score([0, 0, 1, 1], [0, 1, 0, 1]))
  0.0...
  >>> print("%.6f" % homogeneity_score([0, 0, 1, 1], [0, 0, 0, 0]))
  0.0...
*)

val jaccard_score : ?labels:Py.Object.t -> ?pos_label:[`String of string | `Int of int] -> ?average:[`String of string | `Binary | `PyObject of Py.Object.t] -> ?sample_weight:Ndarray.t -> y_true:Py.Object.t -> y_pred:Py.Object.t -> unit -> Py.Object.t
(**
Jaccard similarity coefficient score

The Jaccard index [1], or Jaccard similarity coefficient, defined as
the size of the intersection divided by the size of the union of two label
sets, is used to compare set of predicted labels for a sample to the
corresponding set of labels in ``y_true``.

Read more in the :ref:`User Guide <jaccard_similarity_score>`.

Parameters
----------
y_true : 1d array-like, or label indicator array / sparse matrix
    Ground truth (correct) labels.

y_pred : 1d array-like, or label indicator array / sparse matrix
    Predicted labels, as returned by a classifier.

labels : list, optional
    The set of labels to include when ``average != 'binary'``, and their
    order if ``average is None``. Labels present in the data can be
    excluded, for example to calculate a multiclass average ignoring a
    majority negative class, while labels not present in the data will
    result in 0 components in a macro average. For multilabel targets,
    labels are column indices. By default, all labels in ``y_true`` and
    ``y_pred`` are used in sorted order.

pos_label : str or int, 1 by default
    The class to report if ``average='binary'`` and the data is binary.
    If the data are multiclass or multilabel, this will be ignored;
    setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
    scores for that label only.

average : string, [None, 'binary' (default), 'micro', 'macro', 'samples',                        'weighted']
    If ``None``, the scores for each class are returned. Otherwise, this
    determines the type of averaging performed on the data:

    ``'binary'``:
        Only report results for the class specified by ``pos_label``.
        This is applicable only if targets (``y_{true,pred}``) are binary.
    ``'micro'``:
        Calculate metrics globally by counting the total true positives,
        false negatives and false positives.
    ``'macro'``:
        Calculate metrics for each label, and find their unweighted
        mean.  This does not take label imbalance into account.
    ``'weighted'``:
        Calculate metrics for each label, and find their average, weighted
        by support (the number of true instances for each label). This
        alters 'macro' to account for label imbalance.
    ``'samples'``:
        Calculate metrics for each instance, and find their average (only
        meaningful for multilabel classification).

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

Returns
-------
score : float (if average is not None) or array of floats, shape =            [n_unique_labels]

See also
--------
accuracy_score, f_score, multilabel_confusion_matrix

Notes
-----
:func:`jaccard_score` may be a poor metric if there are no
positives for some samples or classes. Jaccard is undefined if there are
no true or predicted labels, and our implementation will return a score
of 0 with a warning.

References
----------
.. [1] `Wikipedia entry for the Jaccard index
       <https://en.wikipedia.org/wiki/Jaccard_index>`_

Examples
--------
>>> import numpy as np
>>> from sklearn.metrics import jaccard_score
>>> y_true = np.array([[0, 1, 1],
...                    [1, 1, 0]])
>>> y_pred = np.array([[1, 1, 1],
...                    [1, 0, 0]])

In the binary case:

>>> jaccard_score(y_true[0], y_pred[0])
0.6666...

In the multilabel case:

>>> jaccard_score(y_true, y_pred, average='samples')
0.5833...
>>> jaccard_score(y_true, y_pred, average='macro')
0.6666...
>>> jaccard_score(y_true, y_pred, average=None)
array([0.5, 0.5, 1. ])

In the multiclass case:

>>> y_pred = [0, 2, 1, 2]
>>> y_true = [0, 1, 2, 2]
>>> jaccard_score(y_true, y_pred, average=None)
array([1. , 0. , 0.33...])
*)

val jaccard_similarity_score : ?normalize:bool -> ?sample_weight:Ndarray.t -> y_true:Py.Object.t -> y_pred:Py.Object.t -> unit -> float
(**
Jaccard similarity coefficient score

.. deprecated:: 0.21
    This is deprecated to be removed in 0.23, since its handling of
    binary and multiclass inputs was broken. `jaccard_score` has an API
    that is consistent with precision_score, f_score, etc.

Read more in the :ref:`User Guide <jaccard_similarity_score>`.

Parameters
----------
y_true : 1d array-like, or label indicator array / sparse matrix
    Ground truth (correct) labels.

y_pred : 1d array-like, or label indicator array / sparse matrix
    Predicted labels, as returned by a classifier.

normalize : bool, optional (default=True)
    If ``False``, return the sum of the Jaccard similarity coefficient
    over the sample set. Otherwise, return the average of Jaccard
    similarity coefficient.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

Returns
-------
score : float
    If ``normalize == True``, return the average Jaccard similarity
    coefficient, else it returns the sum of the Jaccard similarity
    coefficient over the sample set.

    The best performance is 1 with ``normalize == True`` and the number
    of samples with ``normalize == False``.

See also
--------
accuracy_score, hamming_loss, zero_one_loss

Notes
-----
In binary and multiclass classification, this function is equivalent
to the ``accuracy_score``. It differs in the multilabel classification
problem.

References
----------
.. [1] `Wikipedia entry for the Jaccard index
       <https://en.wikipedia.org/wiki/Jaccard_index>`_
*)

val label_ranking_average_precision_score : ?sample_weight:Ndarray.t -> y_true:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y_score:Ndarray.t -> unit -> float
(**
Compute ranking-based average precision

Label ranking average precision (LRAP) is the average over each ground
truth label assigned to each sample, of the ratio of true vs. total
labels with lower score.

This metric is used in multilabel ranking problem, where the goal
is to give better rank to the labels associated to each sample.

The obtained score is always strictly greater than 0 and
the best value is 1.

Read more in the :ref:`User Guide <label_ranking_average_precision>`.

Parameters
----------
y_true : array or sparse matrix, shape = [n_samples, n_labels]
    True binary labels in binary indicator format.

y_score : array, shape = [n_samples, n_labels]
    Target scores, can either be probability estimates of the positive
    class, confidence values, or non-thresholded measure of decisions
    (as returned by "decision_function" on some classifiers).

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

Returns
-------
score : float

Examples
--------
>>> import numpy as np
>>> from sklearn.metrics import label_ranking_average_precision_score
>>> y_true = np.array([[1, 0, 0], [0, 0, 1]])
>>> y_score = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
>>> label_ranking_average_precision_score(y_true, y_score)
0.416...
*)

val label_ranking_loss : ?sample_weight:Ndarray.t -> y_true:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y_score:Ndarray.t -> unit -> float
(**
Compute Ranking loss measure

Compute the average number of label pairs that are incorrectly ordered
given y_score weighted by the size of the label set and the number of
labels not in the label set.

This is similar to the error set size, but weighted by the number of
relevant and irrelevant labels. The best performance is achieved with
a ranking loss of zero.

Read more in the :ref:`User Guide <label_ranking_loss>`.

.. versionadded:: 0.17
   A function *label_ranking_loss*

Parameters
----------
y_true : array or sparse matrix, shape = [n_samples, n_labels]
    True binary labels in binary indicator format.

y_score : array, shape = [n_samples, n_labels]
    Target scores, can either be probability estimates of the positive
    class, confidence values, or non-thresholded measure of decisions
    (as returned by "decision_function" on some classifiers).

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

Returns
-------
loss : float

References
----------
.. [1] Tsoumakas, G., Katakis, I., & Vlahavas, I. (2010).
       Mining multi-label data. In Data mining and knowledge discovery
       handbook (pp. 667-685). Springer US.
*)

val log_loss : ?eps:float -> ?normalize:bool -> ?sample_weight:Ndarray.t -> ?labels:Ndarray.t -> y_true:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> y_pred:Py.Object.t -> unit -> float
(**
Log loss, aka logistic loss or cross-entropy loss.

This is the loss function used in (multinomial) logistic regression
and extensions of it such as neural networks, defined as the negative
log-likelihood of the true labels given a probabilistic classifier's
predictions. The log loss is only defined for two or more labels.
For a single sample with true label yt in {0,1} and
estimated probability yp that yt = 1, the log loss is

    -log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))

Read more in the :ref:`User Guide <log_loss>`.

Parameters
----------
y_true : array-like or label indicator matrix
    Ground truth (correct) labels for n_samples samples.

y_pred : array-like of float, shape = (n_samples, n_classes) or (n_samples,)
    Predicted probabilities, as returned by a classifier's
    predict_proba method. If ``y_pred.shape = (n_samples,)``
    the probabilities provided are assumed to be that of the
    positive class. The labels in ``y_pred`` are assumed to be
    ordered alphabetically, as done by
    :class:`preprocessing.LabelBinarizer`.

eps : float
    Log loss is undefined for p=0 or p=1, so probabilities are
    clipped to max(eps, min(1 - eps, p)).

normalize : bool, optional (default=True)
    If true, return the mean loss per sample.
    Otherwise, return the sum of the per-sample losses.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

labels : array-like, optional (default=None)
    If not provided, labels will be inferred from y_true. If ``labels``
    is ``None`` and ``y_pred`` has shape (n_samples,) the labels are
    assumed to be binary and are inferred from ``y_true``.
    .. versionadded:: 0.18

Returns
-------
loss : float

Examples
--------
>>> from sklearn.metrics import log_loss
>>> log_loss(["spam", "ham", "ham", "spam"],
...          [[.1, .9], [.9, .1], [.8, .2], [.35, .65]])
0.21616...

References
----------
C.M. Bishop (2006). Pattern Recognition and Machine Learning. Springer,
p. 209.

Notes
-----
The logarithm used is the natural logarithm (base-e).
*)

val make_scorer : ?greater_is_better:bool -> ?needs_proba:bool -> ?needs_threshold:bool -> ?kwargs:(string * Py.Object.t) list -> score_func:Py.Object.t -> unit -> Py.Object.t
(**
Make a scorer from a performance metric or loss function.

This factory function wraps scoring functions for use in GridSearchCV
and cross_val_score. It takes a score function, such as ``accuracy_score``,
``mean_squared_error``, ``adjusted_rand_index`` or ``average_precision``
and returns a callable that scores an estimator's output.

Read more in the :ref:`User Guide <scoring>`.

Parameters
----------
score_func : callable,
    Score function (or loss function) with signature
    ``score_func(y, y_pred, **kwargs)``.

greater_is_better : boolean, default=True
    Whether score_func is a score function (default), meaning high is good,
    or a loss function, meaning low is good. In the latter case, the
    scorer object will sign-flip the outcome of the score_func.

needs_proba : boolean, default=False
    Whether score_func requires predict_proba to get probability estimates
    out of a classifier.

    If True, for binary `y_true`, the score function is supposed to accept
    a 1D `y_pred` (i.e., probability of the positive class, shape
    `(n_samples,)`).

needs_threshold : boolean, default=False
    Whether score_func takes a continuous decision certainty.
    This only works for binary classification using estimators that
    have either a decision_function or predict_proba method.

    If True, for binary `y_true`, the score function is supposed to accept
    a 1D `y_pred` (i.e., probability of the positive class or the decision
    function, shape `(n_samples,)`).

    For example ``average_precision`` or the area under the roc curve
    can not be computed using discrete predictions alone.

**kwargs : additional arguments
    Additional parameters to be passed to score_func.

Returns
-------
scorer : callable
    Callable object that returns a scalar score; greater is better.

Examples
--------
>>> from sklearn.metrics import fbeta_score, make_scorer
>>> ftwo_scorer = make_scorer(fbeta_score, beta=2)
>>> ftwo_scorer
make_scorer(fbeta_score, beta=2)
>>> from sklearn.model_selection import GridSearchCV
>>> from sklearn.svm import LinearSVC
>>> grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]},
...                     scoring=ftwo_scorer)

Notes
-----
If `needs_proba=False` and `needs_threshold=False`, the score
function is supposed to accept the output of :term:`predict`. If
`needs_proba=True`, the score function is supposed to accept the
output of :term:`predict_proba` (For binary `y_true`, the score function is
supposed to accept probability of the positive class). If
`needs_threshold=True`, the score function is supposed to accept the
output of :term:`decision_function`.
*)

val matthews_corrcoef : ?sample_weight:Ndarray.t -> y_true:Ndarray.t -> y_pred:Ndarray.t -> unit -> float
(**
Compute the Matthews correlation coefficient (MCC)

The Matthews correlation coefficient is used in machine learning as a
measure of the quality of binary and multiclass classifications. It takes
into account true and false positives and negatives and is generally
regarded as a balanced measure which can be used even if the classes are of
very different sizes. The MCC is in essence a correlation coefficient value
between -1 and +1. A coefficient of +1 represents a perfect prediction, 0
an average random prediction and -1 an inverse prediction.  The statistic
is also known as the phi coefficient. [source: Wikipedia]

Binary and multiclass labels are supported.  Only in the binary case does
this relate to information about true and false positives and negatives.
See references below.

Read more in the :ref:`User Guide <matthews_corrcoef>`.

Parameters
----------
y_true : array, shape = [n_samples]
    Ground truth (correct) target values.

y_pred : array, shape = [n_samples]
    Estimated targets as returned by a classifier.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

Returns
-------
mcc : float
    The Matthews correlation coefficient (+1 represents a perfect
    prediction, 0 an average random prediction and -1 and inverse
    prediction).

References
----------
.. [1] `Baldi, Brunak, Chauvin, Andersen and Nielsen, (2000). Assessing the
   accuracy of prediction algorithms for classification: an overview
   <https://doi.org/10.1093/bioinformatics/16.5.412>`_

.. [2] `Wikipedia entry for the Matthews Correlation Coefficient
   <https://en.wikipedia.org/wiki/Matthews_correlation_coefficient>`_

.. [3] `Gorodkin, (2004). Comparing two K-category assignments by a
    K-category correlation coefficient
    <https://www.sciencedirect.com/science/article/pii/S1476927104000799>`_

.. [4] `Jurman, Riccadonna, Furlanello, (2012). A Comparison of MCC and CEN
    Error Measures in MultiClass Prediction
    <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0041882>`_

Examples
--------
>>> from sklearn.metrics import matthews_corrcoef
>>> y_true = [+1, +1, +1, -1]
>>> y_pred = [+1, -1, +1, +1]
>>> matthews_corrcoef(y_true, y_pred)
-0.33...
*)

val max_error : y_true:Ndarray.t -> y_pred:Ndarray.t -> unit -> float
(**
max_error metric calculates the maximum residual error.

Read more in the :ref:`User Guide <max_error>`.

Parameters
----------
y_true : array-like of shape (n_samples,)
    Ground truth (correct) target values.

y_pred : array-like of shape (n_samples,)
    Estimated target values.

Returns
-------
max_error : float
    A positive floating point value (the best value is 0.0).

Examples
--------
>>> from sklearn.metrics import max_error
>>> y_true = [3, 2, 7, 1]
>>> y_pred = [4, 2, 7, 1]
>>> max_error(y_true, y_pred)
1
*)

val mean_absolute_error : ?sample_weight:Ndarray.t -> ?multioutput:Py.Object.t -> y_true:Ndarray.t -> y_pred:Ndarray.t -> unit -> Py.Object.t
(**
Mean absolute error regression loss

Read more in the :ref:`User Guide <mean_absolute_error>`.

Parameters
----------
y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
    Ground truth (correct) target values.

y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
    Estimated target values.

sample_weight : array-like of shape (n_samples,), optional
    Sample weights.

multioutput : string in ['raw_values', 'uniform_average']
    or array-like of shape (n_outputs)
    Defines aggregating of multiple output values.
    Array-like value defines weights used to average errors.

    'raw_values' :
        Returns a full set of errors in case of multioutput input.

    'uniform_average' :
        Errors of all outputs are averaged with uniform weight.


Returns
-------
loss : float or ndarray of floats
    If multioutput is 'raw_values', then mean absolute error is returned
    for each output separately.
    If multioutput is 'uniform_average' or an ndarray of weights, then the
    weighted average of all output errors is returned.

    MAE output is non-negative floating point. The best value is 0.0.

Examples
--------
>>> from sklearn.metrics import mean_absolute_error
>>> y_true = [3, -0.5, 2, 7]
>>> y_pred = [2.5, 0.0, 2, 8]
>>> mean_absolute_error(y_true, y_pred)
0.5
>>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
>>> y_pred = [[0, 2], [-1, 2], [8, -5]]
>>> mean_absolute_error(y_true, y_pred)
0.75
>>> mean_absolute_error(y_true, y_pred, multioutput='raw_values')
array([0.5, 1. ])
>>> mean_absolute_error(y_true, y_pred, multioutput=[0.3, 0.7])
0.85...
*)

val mean_gamma_deviance : ?sample_weight:Ndarray.t -> y_true:Ndarray.t -> y_pred:Ndarray.t -> unit -> float
(**
Mean Gamma deviance regression loss.

Gamma deviance is equivalent to the Tweedie deviance with
the power parameter `p=2`. It is invariant to scaling of
the target variable, and measures relative errors.

Read more in the :ref:`User Guide <mean_tweedie_deviance>`.

Parameters
----------
y_true : array-like of shape (n_samples,)
    Ground truth (correct) target values. Requires y_true > 0.

y_pred : array-like of shape (n_samples,)
    Estimated target values. Requires y_pred > 0.

sample_weight : array-like, shape (n_samples,), optional
    Sample weights.

Returns
-------
loss : float
    A non-negative floating point value (the best value is 0.0).

Examples
--------
>>> from sklearn.metrics import mean_gamma_deviance
>>> y_true = [2, 0.5, 1, 4]
>>> y_pred = [0.5, 0.5, 2., 2.]
>>> mean_gamma_deviance(y_true, y_pred)
1.0568...
*)

val mean_poisson_deviance : ?sample_weight:Ndarray.t -> y_true:Ndarray.t -> y_pred:Ndarray.t -> unit -> float
(**
Mean Poisson deviance regression loss.

Poisson deviance is equivalent to the Tweedie deviance with
the power parameter `p=1`.

Read more in the :ref:`User Guide <mean_tweedie_deviance>`.

Parameters
----------
y_true : array-like of shape (n_samples,)
    Ground truth (correct) target values. Requires y_true >= 0.

y_pred : array-like of shape (n_samples,)
    Estimated target values. Requires y_pred > 0.

sample_weight : array-like, shape (n_samples,), optional
    Sample weights.

Returns
-------
loss : float
    A non-negative floating point value (the best value is 0.0).

Examples
--------
>>> from sklearn.metrics import mean_poisson_deviance
>>> y_true = [2, 0, 1, 4]
>>> y_pred = [0.5, 0.5, 2., 2.]
>>> mean_poisson_deviance(y_true, y_pred)
1.4260...
*)

val mean_squared_error : ?sample_weight:Ndarray.t -> ?multioutput:Py.Object.t -> ?squared:Py.Object.t -> y_true:Ndarray.t -> y_pred:Ndarray.t -> unit -> Py.Object.t
(**
Mean squared error regression loss

Read more in the :ref:`User Guide <mean_squared_error>`.

Parameters
----------
y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
    Ground truth (correct) target values.

y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
    Estimated target values.

sample_weight : array-like of shape (n_samples,), optional
    Sample weights.

multioutput : string in ['raw_values', 'uniform_average']
    or array-like of shape (n_outputs)
    Defines aggregating of multiple output values.
    Array-like value defines weights used to average errors.

    'raw_values' :
        Returns a full set of errors in case of multioutput input.

    'uniform_average' :
        Errors of all outputs are averaged with uniform weight.

squared : boolean value, optional (default = True)
    If True returns MSE value, if False returns RMSE value.

Returns
-------
loss : float or ndarray of floats
    A non-negative floating point value (the best value is 0.0), or an
    array of floating point values, one for each individual target.

Examples
--------
>>> from sklearn.metrics import mean_squared_error
>>> y_true = [3, -0.5, 2, 7]
>>> y_pred = [2.5, 0.0, 2, 8]
>>> mean_squared_error(y_true, y_pred)
0.375
>>> y_true = [3, -0.5, 2, 7]
>>> y_pred = [2.5, 0.0, 2, 8]
>>> mean_squared_error(y_true, y_pred, squared=False)
0.612...
>>> y_true = [[0.5, 1],[-1, 1],[7, -6]]
>>> y_pred = [[0, 2],[-1, 2],[8, -5]]
>>> mean_squared_error(y_true, y_pred)
0.708...
>>> mean_squared_error(y_true, y_pred, multioutput='raw_values')
array([0.41666667, 1.        ])
>>> mean_squared_error(y_true, y_pred, multioutput=[0.3, 0.7])
0.825...
*)

val mean_squared_log_error : ?sample_weight:Ndarray.t -> ?multioutput:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> y_true:Ndarray.t -> y_pred:Ndarray.t -> unit -> Py.Object.t
(**
Mean squared logarithmic error regression loss

Read more in the :ref:`User Guide <mean_squared_log_error>`.

Parameters
----------
y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
    Ground truth (correct) target values.

y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
    Estimated target values.

sample_weight : array-like of shape (n_samples,), optional
    Sample weights.

multioutput : string in ['raw_values', 'uniform_average']             or array-like of shape (n_outputs)

    Defines aggregating of multiple output values.
    Array-like value defines weights used to average errors.

    'raw_values' :
        Returns a full set of errors when the input is of multioutput
        format.

    'uniform_average' :
        Errors of all outputs are averaged with uniform weight.

Returns
-------
loss : float or ndarray of floats
    A non-negative floating point value (the best value is 0.0), or an
    array of floating point values, one for each individual target.

Examples
--------
>>> from sklearn.metrics import mean_squared_log_error
>>> y_true = [3, 5, 2.5, 7]
>>> y_pred = [2.5, 5, 4, 8]
>>> mean_squared_log_error(y_true, y_pred)
0.039...
>>> y_true = [[0.5, 1], [1, 2], [7, 6]]
>>> y_pred = [[0.5, 2], [1, 2.5], [8, 8]]
>>> mean_squared_log_error(y_true, y_pred)
0.044...
>>> mean_squared_log_error(y_true, y_pred, multioutput='raw_values')
array([0.00462428, 0.08377444])
>>> mean_squared_log_error(y_true, y_pred, multioutput=[0.3, 0.7])
0.060...
*)

val mean_tweedie_deviance : ?sample_weight:Ndarray.t -> ?power:float -> y_true:Ndarray.t -> y_pred:Ndarray.t -> unit -> float
(**
Mean Tweedie deviance regression loss.

Read more in the :ref:`User Guide <mean_tweedie_deviance>`.

Parameters
----------
y_true : array-like of shape (n_samples,)
    Ground truth (correct) target values.

y_pred : array-like of shape (n_samples,)
    Estimated target values.

sample_weight : array-like, shape (n_samples,), optional
    Sample weights.

power : float, default=0
    Tweedie power parameter. Either power <= 0 or power >= 1.

    The higher `p` the less weight is given to extreme
    deviations between true and predicted targets.

    - power < 0: Extreme stable distribution. Requires: y_pred > 0.
    - power = 0 : Normal distribution, output corresponds to
      mean_squared_error. y_true and y_pred can be any real numbers.
    - power = 1 : Poisson distribution. Requires: y_true >= 0 and
      y_pred > 0.
    - 1 < p < 2 : Compound Poisson distribution. Requires: y_true >= 0
      and y_pred > 0.
    - power = 2 : Gamma distribution. Requires: y_true > 0 and y_pred > 0.
    - power = 3 : Inverse Gaussian distribution. Requires: y_true > 0
      and y_pred > 0.
    - otherwise : Positive stable distribution. Requires: y_true > 0
      and y_pred > 0.

Returns
-------
loss : float
    A non-negative floating point value (the best value is 0.0).

Examples
--------
>>> from sklearn.metrics import mean_tweedie_deviance
>>> y_true = [2, 0, 1, 4]
>>> y_pred = [0.5, 0.5, 2., 2.]
>>> mean_tweedie_deviance(y_true, y_pred, power=1)
1.4260...
*)

val median_absolute_error : ?multioutput:[`Raw_values | `Uniform_average | `PyObject of Py.Object.t] -> y_true:Ndarray.t -> y_pred:Ndarray.t -> unit -> Py.Object.t
(**
Median absolute error regression loss

Median absolute error output is non-negative floating point. The best value
is 0.0. Read more in the :ref:`User Guide <median_absolute_error>`.

Parameters
----------
y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
    Ground truth (correct) target values.

y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
    Estimated target values.

multioutput : {'raw_values', 'uniform_average'} or array-like of shape
    (n_outputs,)
    Defines aggregating of multiple output values. Array-like value defines
    weights used to average errors.

    'raw_values' :
        Returns a full set of errors in case of multioutput input.

    'uniform_average' :
        Errors of all outputs are averaged with uniform weight.

Returns
-------
loss : float or ndarray of floats
    If multioutput is 'raw_values', then mean absolute error is returned
    for each output separately.
    If multioutput is 'uniform_average' or an ndarray of weights, then the
    weighted average of all output errors is returned.

Examples
--------
>>> from sklearn.metrics import median_absolute_error
>>> y_true = [3, -0.5, 2, 7]
>>> y_pred = [2.5, 0.0, 2, 8]
>>> median_absolute_error(y_true, y_pred)
0.5
>>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
>>> y_pred = [[0, 2], [-1, 2], [8, -5]]
>>> median_absolute_error(y_true, y_pred)
0.75
>>> median_absolute_error(y_true, y_pred, multioutput='raw_values')
array([0.5, 1. ])
>>> median_absolute_error(y_true, y_pred, multioutput=[0.3, 0.7])
0.85
*)

val multilabel_confusion_matrix : ?sample_weight:Ndarray.t -> ?labels:Ndarray.t -> ?samplewise:bool -> y_true:Py.Object.t -> y_pred:Py.Object.t -> unit -> Ndarray.t
(**
Compute a confusion matrix for each class or sample

.. versionadded:: 0.21

Compute class-wise (default) or sample-wise (samplewise=True) multilabel
confusion matrix to evaluate the accuracy of a classification, and output
confusion matrices for each class or sample.

In multilabel confusion matrix :math:`MCM`, the count of true negatives
is :math:`MCM_{:,0,0}`, false negatives is :math:`MCM_{:,1,0}`,
true positives is :math:`MCM_{:,1,1}` and false positives is
:math:`MCM_{:,0,1}`.

Multiclass data will be treated as if binarized under a one-vs-rest
transformation. Returned confusion matrices will be in the order of
sorted unique labels in the union of (y_true, y_pred).

Read more in the :ref:`User Guide <multilabel_confusion_matrix>`.

Parameters
----------
y_true : 1d array-like, or label indicator array / sparse matrix
    of shape (n_samples, n_outputs) or (n_samples,)
    Ground truth (correct) target values.

y_pred : 1d array-like, or label indicator array / sparse matrix
    of shape (n_samples, n_outputs) or (n_samples,)
    Estimated targets as returned by a classifier

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights

labels : array-like
    A list of classes or column indices to select some (or to force
    inclusion of classes absent from the data)

samplewise : bool, default=False
    In the multilabel case, this calculates a confusion matrix per sample

Returns
-------
multi_confusion : array, shape (n_outputs, 2, 2)
    A 2x2 confusion matrix corresponding to each output in the input.
    When calculating class-wise multi_confusion (default), then
    n_outputs = n_labels; when calculating sample-wise multi_confusion
    (samplewise=True), n_outputs = n_samples. If ``labels`` is defined,
    the results will be returned in the order specified in ``labels``,
    otherwise the results will be returned in sorted order by default.

See also
--------
confusion_matrix

Notes
-----
The multilabel_confusion_matrix calculates class-wise or sample-wise
multilabel confusion matrices, and in multiclass tasks, labels are
binarized under a one-vs-rest way; while confusion_matrix calculates
one confusion matrix for confusion between every two classes.

Examples
--------

Multilabel-indicator case:

>>> import numpy as np
>>> from sklearn.metrics import multilabel_confusion_matrix
>>> y_true = np.array([[1, 0, 1],
...                    [0, 1, 0]])
>>> y_pred = np.array([[1, 0, 0],
...                    [0, 1, 1]])
>>> multilabel_confusion_matrix(y_true, y_pred)
array([[[1, 0],
        [0, 1]],
<BLANKLINE>
       [[1, 0],
        [0, 1]],
<BLANKLINE>
       [[0, 1],
        [1, 0]]])

Multiclass case:

>>> y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
>>> y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
>>> multilabel_confusion_matrix(y_true, y_pred,
...                             labels=["ant", "bird", "cat"])
array([[[3, 1],
        [0, 2]],
<BLANKLINE>
       [[5, 0],
        [1, 0]],
<BLANKLINE>
       [[2, 1],
        [1, 2]]])
*)

val mutual_info_score : ?contingency:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t | `None] -> labels_true:Py.Object.t -> labels_pred:Py.Object.t -> unit -> float
(**
Mutual Information between two clusterings.

The Mutual Information is a measure of the similarity between two labels of
the same data. Where :math:`|U_i|` is the number of the samples
in cluster :math:`U_i` and :math:`|V_j|` is the number of the
samples in cluster :math:`V_j`, the Mutual Information
between clusterings :math:`U` and :math:`V` is given as:

.. math::

    MI(U,V)=\sum_{i=1}^{ |U| } \sum_{j=1}^{ |V| } \frac{ |U_i\cap V_j| }{N}
    \log\frac{N|U_i \cap V_j| }{ |U_i||V_j| }

This metric is independent of the absolute values of the labels:
a permutation of the class or cluster label values won't change the
score value in any way.

This metric is furthermore symmetric: switching ``label_true`` with
``label_pred`` will return the same score value. This can be useful to
measure the agreement of two independent label assignments strategies
on the same dataset when the real ground truth is not known.

Read more in the :ref:`User Guide <mutual_info_score>`.

Parameters
----------
labels_true : int array, shape = [n_samples]
    A clustering of the data into disjoint subsets.

labels_pred : int array-like of shape (n_samples,)
    A clustering of the data into disjoint subsets.

contingency : {None, array, sparse matrix},                   shape = [n_classes_true, n_classes_pred]
    A contingency matrix given by the :func:`contingency_matrix` function.
    If value is ``None``, it will be computed, otherwise the given value is
    used, with ``labels_true`` and ``labels_pred`` ignored.

Returns
-------
mi : float
   Mutual information, a non-negative value

Notes
-----
The logarithm used is the natural logarithm (base-e).

See also
--------
adjusted_mutual_info_score: Adjusted against chance Mutual Information
normalized_mutual_info_score: Normalized Mutual Information
*)

val nan_euclidean_distances : ?y:Ndarray.t -> ?squared:bool -> ?missing_values:[`Int of int | `PyObject of Py.Object.t] -> ?copy:bool -> x:Ndarray.t -> unit -> Ndarray.t
(**
Calculate the euclidean distances in the presence of missing values.

Compute the euclidean distance between each pair of samples in X and Y,
where Y=X is assumed if Y=None. When calculating the distance between a
pair of samples, this formulation ignores feature coordinates with a
missing value in either sample and scales up the weight of the remaining
coordinates:

    dist(x,y) = sqrt(weight * sq. distance from present coordinates)
    where,
    weight = Total # of coordinates / # of present coordinates

For example, the distance between ``[3, na, na, 6]`` and ``[1, na, 4, 5]``
is:

    .. math::
        \sqrt{\frac{4}{2}((3-1)^2 + (6-5)^2)}

If all the coordinates are missing or if there are no common present
coordinates then NaN is returned for that pair.

Read more in the :ref:`User Guide <metrics>`.

.. versionadded:: 0.22

Parameters
----------
X : array-like, shape=(n_samples_1, n_features)

Y : array-like, shape=(n_samples_2, n_features)

squared : bool, default=False
    Return squared Euclidean distances.

missing_values : np.nan or int, default=np.nan
    Representation of missing value

copy : boolean, default=True
    Make and use a deep copy of X and Y (if Y exists)

Returns
-------
distances : array, shape (n_samples_1, n_samples_2)

Examples
--------
>>> from sklearn.metrics.pairwise import nan_euclidean_distances
>>> nan = float("NaN")
>>> X = [[0, 1], [1, nan]]
>>> nan_euclidean_distances(X, X) # distance between rows of X
array([[0.        , 1.41421356],
       [1.41421356, 0.        ]])

>>> # get distance to origin
>>> nan_euclidean_distances(X, [[0, 0]])
array([[1.        ],
       [1.41421356]])

References
----------
* John K. Dixon, "Pattern Recognition with Partly Missing Data",
  IEEE Transactions on Systems, Man, and Cybernetics, Volume: 9, Issue:
  10, pp. 617 - 621, Oct. 1979.
  http://ieeexplore.ieee.org/abstract/document/4310090/

See also
--------
paired_distances : distances between pairs of elements of X and Y.
*)

val ndcg_score : ?k:int -> ?sample_weight:Ndarray.t -> ?ignore_ties:bool -> y_true:Ndarray.t -> y_score:Ndarray.t -> unit -> Py.Object.t
(**
Compute Normalized Discounted Cumulative Gain.

Sum the true scores ranked in the order induced by the predicted scores,
after applying a logarithmic discount. Then divide by the best possible
score (Ideal DCG, obtained for a perfect ranking) to obtain a score between
0 and 1.

This ranking metric yields a high value if true labels are ranked high by
``y_score``.

Parameters
----------
y_true : ndarray, shape (n_samples, n_labels)
    True targets of multilabel classification, or true scores of entities
    to be ranked.

y_score : ndarray, shape (n_samples, n_labels)
    Target scores, can either be probability estimates, confidence values,
    or non-thresholded measure of decisions (as returned by
    "decision_function" on some classifiers).

k : int, optional (default=None)
    Only consider the highest k scores in the ranking. If None, use all
    outputs.

sample_weight : ndarray, shape (n_samples,), optional (default=None)
    Sample weights. If None, all samples are given the same weight.

ignore_ties : bool, optional (default=False)
    Assume that there are no ties in y_score (which is likely to be the
    case if y_score is continuous) for efficiency gains.

Returns
-------
normalized_discounted_cumulative_gain : float in [0., 1.]
    The averaged NDCG scores for all samples.

See also
--------
dcg_score : Discounted Cumulative Gain (not normalized).

References
----------
`Wikipedia entry for Discounted Cumulative Gain
<https://en.wikipedia.org/wiki/Discounted_cumulative_gain>`_

Jarvelin, K., & Kekalainen, J. (2002).
Cumulated gain-based evaluation of IR techniques. ACM Transactions on
Information Systems (TOIS), 20(4), 422-446.

Wang, Y., Wang, L., Li, Y., He, D., Chen, W., & Liu, T. Y. (2013, May).
A theoretical analysis of NDCG ranking measures. In Proceedings of the 26th
Annual Conference on Learning Theory (COLT 2013)

McSherry, F., & Najork, M. (2008, March). Computing information retrieval
performance measures efficiently in the presence of tied scores. In
European conference on information retrieval (pp. 414-421). Springer,
Berlin, Heidelberg.

Examples
--------
>>> from sklearn.metrics import ndcg_score
>>> # we have groud-truth relevance of some answers to a query:
>>> true_relevance = np.asarray([[10, 0, 0, 1, 5]])
>>> # we predict some scores (relevance) for the answers
>>> scores = np.asarray([[.1, .2, .3, 4, 70]])
>>> ndcg_score(true_relevance, scores) # doctest: +ELLIPSIS
0.69...
>>> scores = np.asarray([[.05, 1.1, 1., .5, .0]])
>>> ndcg_score(true_relevance, scores) # doctest: +ELLIPSIS
0.49...
>>> # we can set k to truncate the sum; only top k answers contribute.
>>> ndcg_score(true_relevance, scores, k=4) # doctest: +ELLIPSIS
0.35...
>>> # the normalization takes k into account so a perfect answer
>>> # would still get 1.0
>>> ndcg_score(true_relevance, true_relevance, k=4) # doctest: +ELLIPSIS
1.0
>>> # now we have some ties in our prediction
>>> scores = np.asarray([[1, 0, 0, 0, 1]])
>>> # by default ties are averaged, so here we get the average (normalized)
>>> # true relevance of our top predictions: (10 / 10 + 5 / 10) / 2 = .75
>>> ndcg_score(true_relevance, scores, k=1) # doctest: +ELLIPSIS
0.75
>>> # we can choose to ignore ties for faster results, but only
>>> # if we know there aren't ties in our scores, otherwise we get
>>> # wrong results:
>>> ndcg_score(true_relevance,
...           scores, k=1, ignore_ties=True) # doctest: +ELLIPSIS
0.5
*)

val normalized_mutual_info_score : ?average_method:string -> labels_true:Py.Object.t -> labels_pred:Py.Object.t -> unit -> float
(**
Normalized Mutual Information between two clusterings.

Normalized Mutual Information (NMI) is a normalization of the Mutual
Information (MI) score to scale the results between 0 (no mutual
information) and 1 (perfect correlation). In this function, mutual
information is normalized by some generalized mean of ``H(labels_true)``
and ``H(labels_pred))``, defined by the `average_method`.

This measure is not adjusted for chance. Therefore
:func:`adjusted_mutual_info_score` might be preferred.

This metric is independent of the absolute values of the labels:
a permutation of the class or cluster label values won't change the
score value in any way.

This metric is furthermore symmetric: switching ``label_true`` with
``label_pred`` will return the same score value. This can be useful to
measure the agreement of two independent label assignments strategies
on the same dataset when the real ground truth is not known.

Read more in the :ref:`User Guide <mutual_info_score>`.

Parameters
----------
labels_true : int array, shape = [n_samples]
    A clustering of the data into disjoint subsets.

labels_pred : int array-like of shape (n_samples,)
    A clustering of the data into disjoint subsets.

average_method : string, optional (default: 'arithmetic')
    How to compute the normalizer in the denominator. Possible options
    are 'min', 'geometric', 'arithmetic', and 'max'.

    .. versionadded:: 0.20

    .. versionchanged:: 0.22
       The default value of ``average_method`` changed from 'geometric' to
       'arithmetic'.

Returns
-------
nmi : float
   score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling

See also
--------
v_measure_score: V-Measure (NMI with arithmetic mean option.)
adjusted_rand_score: Adjusted Rand Index
adjusted_mutual_info_score: Adjusted Mutual Information (adjusted
    against chance)

Examples
--------

Perfect labelings are both homogeneous and complete, hence have
score 1.0::

  >>> from sklearn.metrics.cluster import normalized_mutual_info_score
  >>> normalized_mutual_info_score([0, 0, 1, 1], [0, 0, 1, 1])
  ... # doctest: +SKIP
  1.0
  >>> normalized_mutual_info_score([0, 0, 1, 1], [1, 1, 0, 0])
  ... # doctest: +SKIP
  1.0

If classes members are completely split across different clusters,
the assignment is totally in-complete, hence the NMI is null::

  >>> normalized_mutual_info_score([0, 0, 0, 0], [0, 1, 2, 3])
  ... # doctest: +SKIP
  0.0
*)

module Pairwise : sig
module Parallel : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_jobs:int -> ?backend:Py.Object.t -> ?verbose:int -> ?timeout:float -> ?pre_dispatch:[`All | `Int of int | `PyObject of Py.Object.t] -> ?batch_size:[`Int of int | `Auto] -> ?temp_folder:string -> ?max_nbytes:Py.Object.t -> ?mmap_mode:[`R_ | `R | `W_ | `C | `None] -> ?prefer:[`Processes | `Threads | `None] -> ?require:[`Sharedmem | `None] -> unit -> t
(**
Helper class for readable parallel mapping.

Read more in the :ref:`User Guide <parallel>`.

Parameters
-----------
n_jobs: int, default: None
    The maximum number of concurrently running jobs, such as the number
    of Python worker processes when backend="multiprocessing"
    or the size of the thread-pool when backend="threading".
    If -1 all CPUs are used. If 1 is given, no parallel computing code
    is used at all, which is useful for debugging. For n_jobs below -1,
    (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all
    CPUs but one are used.
    None is a marker for 'unset' that will be interpreted as n_jobs=1
    (sequential execution) unless the call is performed under a
    parallel_backend context manager that sets another value for
    n_jobs.
backend: str, ParallelBackendBase instance or None, default: 'loky'
    Specify the parallelization backend implementation.
    Supported backends are:

    - "loky" used by default, can induce some
      communication and memory overhead when exchanging input and
      output data with the worker Python processes.
    - "multiprocessing" previous process-based backend based on
      `multiprocessing.Pool`. Less robust than `loky`.
    - "threading" is a very low-overhead backend but it suffers
      from the Python Global Interpreter Lock if the called function
      relies a lot on Python objects. "threading" is mostly useful
      when the execution bottleneck is a compiled extension that
      explicitly releases the GIL (for instance a Cython loop wrapped
      in a "with nogil" block or an expensive call to a library such
      as NumPy).
    - finally, you can register backends by calling
      register_parallel_backend. This will allow you to implement
      a backend of your liking.

    It is not recommended to hard-code the backend name in a call to
    Parallel in a library. Instead it is recommended to set soft hints
    (prefer) or hard constraints (require) so as to make it possible
    for library users to change the backend from the outside using the
    parallel_backend context manager.
prefer: str in {'processes', 'threads'} or None, default: None
    Soft hint to choose the default backend if no specific backend
    was selected with the parallel_backend context manager. The
    default process-based backend is 'loky' and the default
    thread-based backend is 'threading'. Ignored if the ``backend``
    parameter is specified.
require: 'sharedmem' or None, default None
    Hard constraint to select the backend. If set to 'sharedmem',
    the selected backend will be single-host and thread-based even
    if the user asked for a non-thread based backend with
    parallel_backend.
verbose: int, optional
    The verbosity level: if non zero, progress messages are
    printed. Above 50, the output is sent to stdout.
    The frequency of the messages increases with the verbosity level.
    If it more than 10, all iterations are reported.
timeout: float, optional
    Timeout limit for each task to complete.  If any task takes longer
    a TimeOutError will be raised. Only applied when n_jobs != 1
pre_dispatch: {'all', integer, or expression, as in '3*n_jobs'}
    The number of batches (of tasks) to be pre-dispatched.
    Default is '2*n_jobs'. When batch_size="auto" this is reasonable
    default and the workers should never starve.
batch_size: int or 'auto', default: 'auto'
    The number of atomic tasks to dispatch at once to each
    worker. When individual evaluations are very fast, dispatching
    calls to workers can be slower than sequential computation because
    of the overhead. Batching fast computations together can mitigate
    this.
    The ``'auto'`` strategy keeps track of the time it takes for a batch
    to complete, and dynamically adjusts the batch size to keep the time
    on the order of half a second, using a heuristic. The initial batch
    size is 1.
    ``batch_size="auto"`` with ``backend="threading"`` will dispatch
    batches of a single task at a time as the threading backend has
    very little overhead and using larger batch size has not proved to
    bring any gain in that case.
temp_folder: str, optional
    Folder to be used by the pool for memmapping large arrays
    for sharing memory with worker processes. If None, this will try in
    order:

    - a folder pointed by the JOBLIB_TEMP_FOLDER environment
      variable,
    - /dev/shm if the folder exists and is writable: this is a
      RAM disk filesystem available by default on modern Linux
      distributions,
    - the default system temporary folder that can be
      overridden with TMP, TMPDIR or TEMP environment
      variables, typically /tmp under Unix operating systems.

    Only active when backend="loky" or "multiprocessing".
max_nbytes int, str, or None, optional, 1M by default
    Threshold on the size of arrays passed to the workers that
    triggers automated memory mapping in temp_folder. Can be an int
    in Bytes, or a human-readable string, e.g., '1M' for 1 megabyte.
    Use None to disable memmapping of large arrays.
    Only active when backend="loky" or "multiprocessing".
mmap_mode: {None, 'r+', 'r', 'w+', 'c'}
    Memmapping mode for numpy arrays passed to workers.
    See 'max_nbytes' parameter documentation for more details.

Notes
-----

This object uses workers to compute in parallel the application of a
function to many different arguments. The main functionality it brings
in addition to using the raw multiprocessing or concurrent.futures API
are (see examples for details):

* More readable code, in particular since it avoids
  constructing list of arguments.

* Easier debugging:
    - informative tracebacks even when the error happens on
      the client side
    - using 'n_jobs=1' enables to turn off parallel computing
      for debugging without changing the codepath
    - early capture of pickling errors

* An optional progress meter.

* Interruption of multiprocesses jobs with 'Ctrl-C'

* Flexible pickling control for the communication to and from
  the worker processes.

* Ability to use shared memory efficiently with worker
  processes for large numpy-based datastructures.

Examples
--------

A simple example:

>>> from math import sqrt
>>> from joblib import Parallel, delayed
>>> Parallel(n_jobs=1)(delayed(sqrt)(i**2) for i in range(10))
[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

Reshaping the output when the function has several return
values:

>>> from math import modf
>>> from joblib import Parallel, delayed
>>> r = Parallel(n_jobs=1)(delayed(modf)(i/2.) for i in range(10))
>>> res, i = zip( *r)
>>> res
(0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5)
>>> i
(0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0)

The progress meter: the higher the value of `verbose`, the more
messages:

>>> from time import sleep
>>> from joblib import Parallel, delayed
>>> r = Parallel(n_jobs=2, verbose=10)(delayed(sleep)(.2) for _ in range(10)) #doctest: +SKIP
[Parallel(n_jobs=2)]: Done   1 tasks      | elapsed:    0.6s
[Parallel(n_jobs=2)]: Done   4 tasks      | elapsed:    0.8s
[Parallel(n_jobs=2)]: Done  10 out of  10 | elapsed:    1.4s finished

Traceback example, note how the line of the error is indicated
as well as the values of the parameter passed to the function that
triggered the exception, even though the traceback happens in the
child process:

>>> from heapq import nlargest
>>> from joblib import Parallel, delayed
>>> Parallel(n_jobs=2)(delayed(nlargest)(2, n) for n in (range(4), 'abcde', 3)) #doctest: +SKIP
#...
---------------------------------------------------------------------------
Sub-process traceback:
---------------------------------------------------------------------------
TypeError                                          Mon Nov 12 11:37:46 2012
PID: 12934                                    Python 2.7.3: /usr/bin/python
...........................................................................
/usr/lib/python2.7/heapq.pyc in nlargest(n=2, iterable=3, key=None)
    419         if n >= size:
    420             return sorted(iterable, key=key, reverse=True)[:n]
    421
    422     # When key is none, use simpler decoration
    423     if key is None:
--> 424         it = izip(iterable, count(0,-1))                    # decorate
    425         result = _nlargest(n, it)
    426         return map(itemgetter(0), result)                   # undecorate
    427
    428     # General case, slowest method
 TypeError: izip argument #1 must support iteration
___________________________________________________________________________


Using pre_dispatch in a producer/consumer situation, where the
data is generated on the fly. Note how the producer is first
called 3 times before the parallel loop is initiated, and then
called to generate new data on the fly:

>>> from math import sqrt
>>> from joblib import Parallel, delayed
>>> def producer():
...     for i in range(6):
...         print('Produced %s' % i)
...         yield i
>>> out = Parallel(n_jobs=2, verbose=100, pre_dispatch='1.5*n_jobs')(
...                delayed(sqrt)(i) for i in producer()) #doctest: +SKIP
Produced 0
Produced 1
Produced 2
[Parallel(n_jobs=2)]: Done 1 jobs     | elapsed:  0.0s
Produced 3
[Parallel(n_jobs=2)]: Done 2 jobs     | elapsed:  0.0s
Produced 4
[Parallel(n_jobs=2)]: Done 3 jobs     | elapsed:  0.0s
Produced 5
[Parallel(n_jobs=2)]: Done 4 jobs     | elapsed:  0.0s
[Parallel(n_jobs=2)]: Done 6 out of 6 | elapsed:  0.0s remaining: 0.0s
[Parallel(n_jobs=2)]: Done 6 out of 6 | elapsed:  0.0s finished
*)

val debug : msg:Py.Object.t -> t -> Py.Object.t
(**
None
*)

val dispatch_next : t -> Py.Object.t
(**
Dispatch more data for parallel processing

This method is meant to be called concurrently by the multiprocessing
callback. We rely on the thread-safety of dispatch_one_batch to protect
against concurrent consumption of the unprotected iterator.
*)

val dispatch_one_batch : iterator:Py.Object.t -> t -> Py.Object.t
(**
Prefetch the tasks for the next batch and dispatch them.

The effective size of the batch is computed here.
If there are no more jobs to dispatch, return False, else return True.

The iterator consumption and dispatching is protected by the same
lock so calling this function should be thread safe.
*)

val format : ?indent:Py.Object.t -> obj:Py.Object.t -> t -> Py.Object.t
(**
Return the formatted representation of the object.
*)

val print_progress : t -> Py.Object.t
(**
Display the process of the parallel execution only a fraction
of time, controlled by self.verbose.
*)

val retrieve : t -> Py.Object.t
(**
None
*)

val warn : msg:Py.Object.t -> t -> Py.Object.t
(**
None
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit


end

val additive_chi2_kernel : ?y:Ndarray.t -> x:Ndarray.t -> unit -> Ndarray.t
(**
Computes the additive chi-squared kernel between observations in X and Y

The chi-squared kernel is computed between each pair of rows in X and Y.  X
and Y have to be non-negative. This kernel is most commonly applied to
histograms.

The chi-squared kernel is given by::

    k(x, y) = -Sum [(x - y)^2 / (x + y)]

It can be interpreted as a weighted difference per entry.

Read more in the :ref:`User Guide <chi2_kernel>`.

Notes
-----
As the negative of a distance, this kernel is only conditionally positive
definite.


Parameters
----------
X : array-like of shape (n_samples_X, n_features)

Y : array of shape (n_samples_Y, n_features)

Returns
-------
kernel_matrix : array of shape (n_samples_X, n_samples_Y)

References
----------
* Zhang, J. and Marszalek, M. and Lazebnik, S. and Schmid, C.
  Local features and kernels for classification of texture and object
  categories: A comprehensive study
  International Journal of Computer Vision 2007
  https://research.microsoft.com/en-us/um/people/manik/projects/trade-off/papers/ZhangIJCV06.pdf


See also
--------
chi2_kernel : The exponentiated version of the kernel, which is usually
    preferable.

sklearn.kernel_approximation.AdditiveChi2Sampler : A Fourier approximation
    to this kernel.
*)

val check_array : ?accept_sparse:[`String of string | `Bool of bool | `StringList of string list] -> ?accept_large_sparse:bool -> ?dtype:[`String of string | `Dtype of Py.Object.t | `TypeList of Py.Object.t | `None] -> ?order:[`F | `C | `None] -> ?copy:bool -> ?force_all_finite:[`Bool of bool | `Allow_nan] -> ?ensure_2d:bool -> ?allow_nd:bool -> ?ensure_min_samples:int -> ?ensure_min_features:int -> ?warn_on_dtype:[`Bool of bool | `None] -> ?estimator:[`String of string | `Estimator of Py.Object.t] -> array:Py.Object.t -> unit -> Py.Object.t
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

dtype : string, type, list of types or None (default="numeric")
    Data type of result. If None, the dtype of the input is preserved.
    If "numeric", dtype is preserved unless array.dtype is object.
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

val check_non_negative : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> whom:string -> unit -> Py.Object.t
(**
Check if there is any negative value in an array.

Parameters
----------
X : array-like or sparse matrix
    Input data.

whom : string
    Who passed X to this function.
*)

val check_paired_arrays : x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> unit -> (Py.Object.t * Py.Object.t)
(**
Set X and Y appropriately and checks inputs for paired distances

All paired distance metrics should use this function first to assert that
the given parameters are correct and safe to use.

Specifically, this function first ensures that both X and Y are arrays,
then checks that they are at least two dimensional while ensuring that
their elements are floats. Finally, the function checks that the size
of the dimensions of the two arrays are equal.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples_a, n_features)

Y : {array-like, sparse matrix}, shape (n_samples_b, n_features)

Returns
-------
safe_X : {array-like, sparse matrix}, shape (n_samples_a, n_features)
    An array equal to X, guaranteed to be a numpy array.

safe_Y : {array-like, sparse matrix}, shape (n_samples_b, n_features)
    An array equal to Y if Y was not None, guaranteed to be a numpy array.
    If Y was None, safe_Y will be a pointer to X.
*)

val check_pairwise_arrays : ?precomputed:bool -> ?dtype:[`String of string | `Dtype of Py.Object.t | `TypeList of Py.Object.t | `None] -> ?accept_sparse:[`String of string | `Bool of bool | `StringList of string list] -> ?force_all_finite:[`Bool of bool | `Allow_nan] -> ?copy:bool -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> unit -> (Py.Object.t * Py.Object.t)
(**
Set X and Y appropriately and checks inputs

If Y is None, it is set as a pointer to X (i.e. not a copy).
If Y is given, this does not happen.
All distance metrics should use this function first to assert that the
given parameters are correct and safe to use.

Specifically, this function first ensures that both X and Y are arrays,
then checks that they are at least two dimensional while ensuring that
their elements are floats (or dtype if provided). Finally, the function
checks that the size of the second dimension of the two arrays is equal, or
the equivalent check for a precomputed distance matrix.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples_a, n_features)

Y : {array-like, sparse matrix}, shape (n_samples_b, n_features)

precomputed : bool
    True if X is to be treated as precomputed distances to the samples in
    Y.

dtype : string, type, list of types or None (default=None)
    Data type required for X and Y. If None, the dtype will be an
    appropriate float type selected by _return_float_dtype.

    .. versionadded:: 0.18

accept_sparse : string, boolean or list/tuple of strings
    String[s] representing allowed sparse matrix formats, such as 'csc',
    'csr', etc. If the input is sparse but not in the allowed format,
    it will be converted to the first listed format. True allows the input
    to be any format. False means that a sparse matrix input will
    raise an error.

force_all_finite : boolean or 'allow-nan', (default=True)
    Whether to raise an error on np.inf and np.nan in array. The
    possibilities are:

    - True: Force all values of array to be finite.
    - False: accept both np.inf and np.nan in array.
    - 'allow-nan': accept only np.nan values in array. Values cannot
      be infinite.

    .. versionadded:: 0.22
       ``force_all_finite`` accepts the string ``'allow-nan'``.

copy : bool
    Whether a forced copy will be triggered. If copy=False, a copy might
    be triggered by a conversion.

    .. versionadded:: 0.22

Returns
-------
safe_X : {array-like, sparse matrix}, shape (n_samples_a, n_features)
    An array equal to X, guaranteed to be a numpy array.

safe_Y : {array-like, sparse matrix}, shape (n_samples_b, n_features)
    An array equal to Y if Y was not None, guaranteed to be a numpy array.
    If Y was None, safe_Y will be a pointer to X.
*)

val chi2_kernel : ?y:Ndarray.t -> ?gamma:float -> x:Ndarray.t -> unit -> Ndarray.t
(**
Computes the exponential chi-squared kernel X and Y.

The chi-squared kernel is computed between each pair of rows in X and Y.  X
and Y have to be non-negative. This kernel is most commonly applied to
histograms.

The chi-squared kernel is given by::

    k(x, y) = exp(-gamma Sum [(x - y)^2 / (x + y)])

It can be interpreted as a weighted difference per entry.

Read more in the :ref:`User Guide <chi2_kernel>`.

Parameters
----------
X : array-like of shape (n_samples_X, n_features)

Y : array of shape (n_samples_Y, n_features)

gamma : float, default=1.
    Scaling parameter of the chi2 kernel.

Returns
-------
kernel_matrix : array of shape (n_samples_X, n_samples_Y)

References
----------
* Zhang, J. and Marszalek, M. and Lazebnik, S. and Schmid, C.
  Local features and kernels for classification of texture and object
  categories: A comprehensive study
  International Journal of Computer Vision 2007
  https://research.microsoft.com/en-us/um/people/manik/projects/trade-off/papers/ZhangIJCV06.pdf

See also
--------
additive_chi2_kernel : The additive version of this kernel

sklearn.kernel_approximation.AdditiveChi2Sampler : A Fourier approximation
    to the additive version of this kernel.
*)

val cosine_distances : ?y:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> unit -> Py.Object.t
(**
Compute cosine distance between samples in X and Y.

Cosine distance is defined as 1.0 minus the cosine similarity.

Read more in the :ref:`User Guide <metrics>`.

Parameters
----------
X : array_like, sparse matrix
    with shape (n_samples_X, n_features).

Y : array_like, sparse matrix (optional)
    with shape (n_samples_Y, n_features).

Returns
-------
distance matrix : array
    An array with shape (n_samples_X, n_samples_Y).

See also
--------
sklearn.metrics.pairwise.cosine_similarity
scipy.spatial.distance.cosine : dense matrices only
*)

val cosine_similarity : ?y:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> ?dense_output:bool -> x:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> unit -> Py.Object.t
(**
Compute cosine similarity between samples in X and Y.

Cosine similarity, or the cosine kernel, computes similarity as the
normalized dot product of X and Y:

    K(X, Y) = <X, Y> / (||X||*||Y||)

On L2-normalized data, this function is equivalent to linear_kernel.

Read more in the :ref:`User Guide <cosine_similarity>`.

Parameters
----------
X : ndarray or sparse array, shape: (n_samples_X, n_features)
    Input data.

Y : ndarray or sparse array, shape: (n_samples_Y, n_features)
    Input data. If ``None``, the output will be the pairwise
    similarities between all samples in ``X``.

dense_output : boolean (optional), default True
    Whether to return dense output even when the input is sparse. If
    ``False``, the output is sparse if both input arrays are sparse.

    .. versionadded:: 0.17
       parameter ``dense_output`` for dense output.

Returns
-------
kernel matrix : array
    An array with shape (n_samples_X, n_samples_Y).
*)

val delayed : ?check_pickle:Py.Object.t -> function_:Py.Object.t -> unit -> Py.Object.t
(**
Decorator used to capture the arguments of a function.
*)

val distance_metrics : unit -> Py.Object.t
(**
Valid metrics for pairwise_distances.

This function simply returns the valid pairwise distance metrics.
It exists to allow for a description of the mapping for
each of the valid strings.

The valid distance metrics, and the function they map to, are:

=============== ========================================
metric          Function
=============== ========================================
'cityblock'     metrics.pairwise.manhattan_distances
'cosine'        metrics.pairwise.cosine_distances
'euclidean'     metrics.pairwise.euclidean_distances
'haversine'     metrics.pairwise.haversine_distances
'l1'            metrics.pairwise.manhattan_distances
'l2'            metrics.pairwise.euclidean_distances
'manhattan'     metrics.pairwise.manhattan_distances
'nan_euclidean' metrics.pairwise.nan_euclidean_distances
=============== ========================================

Read more in the :ref:`User Guide <metrics>`.
*)

val effective_n_jobs : ?n_jobs:Py.Object.t -> unit -> Py.Object.t
(**
Determine the number of jobs that can actually run in parallel

n_jobs is the number of workers requested by the callers. Passing n_jobs=-1
means requesting all available workers for instance matching the number of
CPU cores on the worker host(s).

This method should return a guesstimate of the number of workers that can
actually perform work concurrently with the currently enabled default
backend. The primary use case is to make it possible for the caller to know
in how many chunks to slice the work.

In general working on larger data chunks is more efficient (less scheduling
overhead and better use of CPU cache prefetching heuristics) as long as all
the workers have enough work to do.

Warning: this function is experimental and subject to change in a future
version of joblib.

.. versionadded:: 0.10
*)

val euclidean_distances : ?y:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> ?y_norm_squared:Ndarray.t -> ?squared:bool -> ?x_norm_squared:Ndarray.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> unit -> Ndarray.t
(**
Considering the rows of X (and Y=X) as vectors, compute the
distance matrix between each pair of vectors.

For efficiency reasons, the euclidean distance between a pair of row
vector x and y is computed as::

    dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))

This formulation has two advantages over other ways of computing distances.
First, it is computationally efficient when dealing with sparse data.
Second, if one argument varies but the other remains unchanged, then
`dot(x, x)` and/or `dot(y, y)` can be pre-computed.

However, this is not the most precise way of doing this computation, and
the distance matrix returned by this function may not be exactly
symmetric as required by, e.g., ``scipy.spatial.distance`` functions.

Read more in the :ref:`User Guide <metrics>`.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples_1, n_features)

Y : {array-like, sparse matrix}, shape (n_samples_2, n_features)

Y_norm_squared : array-like, shape (n_samples_2, ), optional
    Pre-computed dot-products of vectors in Y (e.g.,
    ``(Y**2).sum(axis=1)``)
    May be ignored in some cases, see the note below.

squared : boolean, optional
    Return squared Euclidean distances.

X_norm_squared : array-like of shape (n_samples,), optional
    Pre-computed dot-products of vectors in X (e.g.,
    ``(X**2).sum(axis=1)``)
    May be ignored in some cases, see the note below.

Notes
-----
To achieve better accuracy, `X_norm_squared` and `Y_norm_squared` may be
unused if they are passed as ``float32``.

Returns
-------
distances : array, shape (n_samples_1, n_samples_2)

Examples
--------
>>> from sklearn.metrics.pairwise import euclidean_distances
>>> X = [[0, 1], [1, 1]]
>>> # distance between rows of X
>>> euclidean_distances(X, X)
array([[0., 1.],
       [1., 0.]])
>>> # get distance to origin
>>> euclidean_distances(X, [[0, 0]])
array([[1.        ],
       [1.41421356]])

See also
--------
paired_distances : distances betweens pairs of elements of X and Y.
*)

val gen_batches : ?min_batch_size:Py.Object.t -> n:int -> batch_size:Py.Object.t -> unit -> Py.Object.t
(**
Generator to create slices containing batch_size elements, from 0 to n.

The last slice may contain less than batch_size elements, when batch_size
does not divide n.

Parameters
----------
n : int
batch_size : int
    Number of element in each batch
min_batch_size : int, default=0
    Minimum batch size to produce.

Yields
------
slice of batch_size elements

Examples
--------
>>> from sklearn.utils import gen_batches
>>> list(gen_batches(7, 3))
[slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]
>>> list(gen_batches(6, 3))
[slice(0, 3, None), slice(3, 6, None)]
>>> list(gen_batches(2, 3))
[slice(0, 2, None)]
>>> list(gen_batches(7, 3, min_batch_size=0))
[slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]
>>> list(gen_batches(7, 3, min_batch_size=2))
[slice(0, 3, None), slice(3, 7, None)]
*)

val gen_even_slices : ?n_samples:Py.Object.t -> n:int -> n_packs:Py.Object.t -> unit -> Py.Object.t
(**
Generator to create n_packs slices going up to n.

Parameters
----------
n : int
n_packs : int
    Number of slices to generate.
n_samples : int or None (default = None)
    Number of samples. Pass n_samples when the slices are to be used for
    sparse matrix indexing; slicing off-the-end raises an exception, while
    it works for NumPy arrays.

Yields
------
slice

Examples
--------
>>> from sklearn.utils import gen_even_slices
>>> list(gen_even_slices(10, 1))
[slice(0, 10, None)]
>>> list(gen_even_slices(10, 10))
[slice(0, 1, None), slice(1, 2, None), ..., slice(9, 10, None)]
>>> list(gen_even_slices(10, 5))
[slice(0, 2, None), slice(2, 4, None), ..., slice(8, 10, None)]
>>> list(gen_even_slices(10, 3))
[slice(0, 4, None), slice(4, 7, None), slice(7, 10, None)]
*)

val get_chunk_n_rows : ?max_n_rows:Py.Object.t -> ?working_memory:Py.Object.t -> row_bytes:int -> unit -> Py.Object.t
(**
Calculates how many rows can be processed within working_memory

Parameters
----------
row_bytes : int
    The expected number of bytes of memory that will be consumed
    during the processing of each row.
max_n_rows : int, optional
    The maximum return value.
working_memory : int or float, optional
    The number of rows to fit inside this number of MiB will be returned.
    When None (default), the value of
    ``sklearn.get_config()['working_memory']`` is used.

Returns
-------
int or the value of n_samples

Warns
-----
Issues a UserWarning if ``row_bytes`` exceeds ``working_memory`` MiB.
*)

val haversine_distances : ?y:Ndarray.t -> x:Ndarray.t -> unit -> Ndarray.t
(**
Compute the Haversine distance between samples in X and Y

The Haversine (or great circle) distance is the angular distance between
two points on the surface of a sphere. The first distance of each point is
assumed to be the latitude, the second is the longitude, given in radians.
The dimension of the data must be 2.

.. math::
   D(x, y) = 2\arcsin[\sqrt{\sin^2((x1 - y1) / 2)
                            + \cos(x1)\cos(y1)\sin^2((x2 - y2) / 2)}]

Parameters
----------
X : array_like, shape (n_samples_1, 2)

Y : array_like, shape (n_samples_2, 2), optional

Returns
-------
distance : {array}, shape (n_samples_1, n_samples_2)

Notes
-----
As the Earth is nearly spherical, the haversine formula provides a good
approximation of the distance between two points of the Earth surface, with
a less than 1% error on average.

Examples
--------
We want to calculate the distance between the Ezeiza Airport
(Buenos Aires, Argentina) and the Charles de Gaulle Airport (Paris, France)

>>> from sklearn.metrics.pairwise import haversine_distances
>>> from math import radians
>>> bsas = [-34.83333, -58.5166646]
>>> paris = [49.0083899664, 2.53844117956]
>>> bsas_in_radians = [radians(_) for _ in bsas]
>>> paris_in_radians = [radians(_) for _ in paris]
>>> result = haversine_distances([bsas_in_radians, paris_in_radians])
>>> result * 6371000/1000  # multiply by Earth radius to get kilometers
array([[    0.        , 11099.54035582],
       [11099.54035582,     0.        ]])
*)

val is_scalar_nan : x:Py.Object.t -> unit -> Py.Object.t
(**
Tests if x is NaN

This function is meant to overcome the issue that np.isnan does not allow
non-numerical types as input, and that np.nan is not np.float('nan').

Parameters
----------
x : any type

Returns
-------
boolean

Examples
--------
>>> is_scalar_nan(np.nan)
True
>>> is_scalar_nan(float("nan"))
True
>>> is_scalar_nan(None)
False
>>> is_scalar_nan("")
False
>>> is_scalar_nan([np.nan])
False
*)

val issparse : x:Py.Object.t -> unit -> Py.Object.t
(**
Is x of a sparse matrix type?

Parameters
----------
x
    object to check for being a sparse matrix

Returns
-------
bool
    True if x is a sparse matrix, False otherwise

Notes
-----
issparse and isspmatrix are aliases for the same function.

Examples
--------
>>> from scipy.sparse import csr_matrix, isspmatrix
>>> isspmatrix(csr_matrix([[5]]))
True

>>> from scipy.sparse import isspmatrix
>>> isspmatrix(5)
False
*)

val kernel_metrics : unit -> Py.Object.t
(**
Valid metrics for pairwise_kernels

This function simply returns the valid pairwise distance metrics.
It exists, however, to allow for a verbose description of the mapping for
each of the valid strings.

The valid distance metrics, and the function they map to, are:
  ===============   ========================================
  metric            Function
  ===============   ========================================
  'additive_chi2'   sklearn.pairwise.additive_chi2_kernel
  'chi2'            sklearn.pairwise.chi2_kernel
  'linear'          sklearn.pairwise.linear_kernel
  'poly'            sklearn.pairwise.polynomial_kernel
  'polynomial'      sklearn.pairwise.polynomial_kernel
  'rbf'             sklearn.pairwise.rbf_kernel
  'laplacian'       sklearn.pairwise.laplacian_kernel
  'sigmoid'         sklearn.pairwise.sigmoid_kernel
  'cosine'          sklearn.pairwise.cosine_similarity
  ===============   ========================================

Read more in the :ref:`User Guide <metrics>`.
*)

val laplacian_kernel : ?y:Ndarray.t -> ?gamma:float -> x:Ndarray.t -> unit -> Ndarray.t
(**
Compute the laplacian kernel between X and Y.

The laplacian kernel is defined as::

    K(x, y) = exp(-gamma ||x-y||_1)

for each pair of rows x in X and y in Y.
Read more in the :ref:`User Guide <laplacian_kernel>`.

.. versionadded:: 0.17

Parameters
----------
X : array of shape (n_samples_X, n_features)

Y : array of shape (n_samples_Y, n_features)

gamma : float, default None
    If None, defaults to 1.0 / n_features

Returns
-------
kernel_matrix : array of shape (n_samples_X, n_samples_Y)
*)

val linear_kernel : ?y:Ndarray.t -> ?dense_output:bool -> x:Ndarray.t -> unit -> Py.Object.t
(**
Compute the linear kernel between X and Y.

Read more in the :ref:`User Guide <linear_kernel>`.

Parameters
----------
X : array of shape (n_samples_1, n_features)

Y : array of shape (n_samples_2, n_features)

dense_output : boolean (optional), default True
    Whether to return dense output even when the input is sparse. If
    ``False``, the output is sparse if both input arrays are sparse.

    .. versionadded:: 0.20

Returns
-------
Gram matrix : array of shape (n_samples_1, n_samples_2)
*)

val manhattan_distances : ?y:Ndarray.t -> ?sum_over_features:bool -> x:Ndarray.t -> unit -> Ndarray.t
(**
Compute the L1 distances between the vectors in X and Y.

With sum_over_features equal to False it returns the componentwise
distances.

Read more in the :ref:`User Guide <metrics>`.

Parameters
----------
X : array_like
    An array with shape (n_samples_X, n_features).

Y : array_like, optional
    An array with shape (n_samples_Y, n_features).

sum_over_features : bool, default=True
    If True the function returns the pairwise distance matrix
    else it returns the componentwise L1 pairwise-distances.
    Not supported for sparse matrix inputs.

Returns
-------
D : array
    If sum_over_features is False shape is
    (n_samples_X * n_samples_Y, n_features) and D contains the
    componentwise L1 pairwise-distances (ie. absolute difference),
    else shape is (n_samples_X, n_samples_Y) and D contains
    the pairwise L1 distances.

Notes
--------
When X and/or Y are CSR sparse matrices and they are not already
in canonical format, this function modifies them in-place to
make them canonical.

Examples
--------
>>> from sklearn.metrics.pairwise import manhattan_distances
>>> manhattan_distances([[3]], [[3]])
array([[0.]])
>>> manhattan_distances([[3]], [[2]])
array([[1.]])
>>> manhattan_distances([[2]], [[3]])
array([[1.]])
>>> manhattan_distances([[1, 2], [3, 4]],         [[1, 2], [0, 3]])
array([[0., 2.],
       [4., 4.]])
>>> import numpy as np
>>> X = np.ones((1, 2))
>>> y = np.full((2, 2), 2.)
>>> manhattan_distances(X, y, sum_over_features=False)
array([[1., 1.],
       [1., 1.]])
*)

val nan_euclidean_distances : ?y:Ndarray.t -> ?squared:bool -> ?missing_values:[`Int of int | `PyObject of Py.Object.t] -> ?copy:bool -> x:Ndarray.t -> unit -> Ndarray.t
(**
Calculate the euclidean distances in the presence of missing values.

Compute the euclidean distance between each pair of samples in X and Y,
where Y=X is assumed if Y=None. When calculating the distance between a
pair of samples, this formulation ignores feature coordinates with a
missing value in either sample and scales up the weight of the remaining
coordinates:

    dist(x,y) = sqrt(weight * sq. distance from present coordinates)
    where,
    weight = Total # of coordinates / # of present coordinates

For example, the distance between ``[3, na, na, 6]`` and ``[1, na, 4, 5]``
is:

    .. math::
        \sqrt{\frac{4}{2}((3-1)^2 + (6-5)^2)}

If all the coordinates are missing or if there are no common present
coordinates then NaN is returned for that pair.

Read more in the :ref:`User Guide <metrics>`.

.. versionadded:: 0.22

Parameters
----------
X : array-like, shape=(n_samples_1, n_features)

Y : array-like, shape=(n_samples_2, n_features)

squared : bool, default=False
    Return squared Euclidean distances.

missing_values : np.nan or int, default=np.nan
    Representation of missing value

copy : boolean, default=True
    Make and use a deep copy of X and Y (if Y exists)

Returns
-------
distances : array, shape (n_samples_1, n_samples_2)

Examples
--------
>>> from sklearn.metrics.pairwise import nan_euclidean_distances
>>> nan = float("NaN")
>>> X = [[0, 1], [1, nan]]
>>> nan_euclidean_distances(X, X) # distance between rows of X
array([[0.        , 1.41421356],
       [1.41421356, 0.        ]])

>>> # get distance to origin
>>> nan_euclidean_distances(X, [[0, 0]])
array([[1.        ],
       [1.41421356]])

References
----------
* John K. Dixon, "Pattern Recognition with Partly Missing Data",
  IEEE Transactions on Systems, Man, and Cybernetics, Volume: 9, Issue:
  10, pp. 617 - 621, Oct. 1979.
  http://ieeexplore.ieee.org/abstract/document/4310090/

See also
--------
paired_distances : distances between pairs of elements of X and Y.
*)

val normalize : ?norm:[`L1 | `L2 | `Max | `PyObject of Py.Object.t] -> ?axis:Py.Object.t -> ?copy:bool -> ?return_norm:bool -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> unit -> (Py.Object.t * Py.Object.t)
(**
Scale input vectors individually to unit norm (vector length).

Read more in the :ref:`User Guide <preprocessing_normalization>`.

Parameters
----------
X : {array-like, sparse matrix}, shape [n_samples, n_features]
    The data to normalize, element by element.
    scipy.sparse matrices should be in CSR format to avoid an
    un-necessary copy.

norm : 'l1', 'l2', or 'max', optional ('l2' by default)
    The norm to use to normalize each non zero sample (or each non-zero
    feature if axis is 0).

axis : 0 or 1, optional (1 by default)
    axis used to normalize the data along. If 1, independently normalize
    each sample, otherwise (if 0) normalize each feature.

copy : boolean, optional, default True
    set to False to perform inplace row normalization and avoid a
    copy (if the input is already a numpy array or a scipy.sparse
    CSR matrix and if axis is 1).

return_norm : boolean, default False
    whether to return the computed norms

Returns
-------
X : {array-like, sparse matrix}, shape [n_samples, n_features]
    Normalized input X.

norms : array, shape [n_samples] if axis=1 else [n_features]
    An array of norms along given axis for X.
    When X is sparse, a NotImplementedError will be raised
    for norm 'l1' or 'l2'.

See also
--------
Normalizer: Performs normalization using the ``Transformer`` API
    (e.g. as part of a preprocessing :class:`sklearn.pipeline.Pipeline`).

Notes
-----
For a comparison of the different scalers, transformers, and normalizers,
see :ref:`examples/preprocessing/plot_all_scaling.py
<sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
*)

val paired_cosine_distances : x:Ndarray.t -> y:Ndarray.t -> unit -> Ndarray.t
(**
Computes the paired cosine distances between X and Y

Read more in the :ref:`User Guide <metrics>`.

Parameters
----------
X : array-like, shape (n_samples, n_features)

Y : array-like, shape (n_samples, n_features)

Returns
-------
distances : ndarray, shape (n_samples, )

Notes
-----
The cosine distance is equivalent to the half the squared
euclidean distance if each sample is normalized to unit norm
*)

val paired_distances : ?metric:[`String of string | `Callable of Py.Object.t] -> ?kwds:(string * Py.Object.t) list -> x:Py.Object.t -> y:Py.Object.t -> unit -> Py.Object.t
(**
Computes the paired distances between X and Y.

Computes the distances between (X[0], Y[0]), (X[1], Y[1]), etc...

Read more in the :ref:`User Guide <metrics>`.

Parameters
----------
X : ndarray (n_samples, n_features)
    Array 1 for distance computation.

Y : ndarray (n_samples, n_features)
    Array 2 for distance computation.

metric : string or callable
    The metric to use when calculating distance between instances in a
    feature array. If metric is a string, it must be one of the options
    specified in PAIRED_DISTANCES, including "euclidean",
    "manhattan", or "cosine".
    Alternatively, if metric is a callable function, it is called on each
    pair of instances (rows) and the resulting value recorded. The callable
    should take two arrays from X as input and return a value indicating
    the distance between them.

Returns
-------
distances : ndarray (n_samples, )

Examples
--------
>>> from sklearn.metrics.pairwise import paired_distances
>>> X = [[0, 1], [1, 1]]
>>> Y = [[0, 1], [2, 1]]
>>> paired_distances(X, Y)
array([0., 1.])

See also
--------
pairwise_distances : Computes the distance between every pair of samples
*)

val paired_euclidean_distances : x:Ndarray.t -> y:Ndarray.t -> unit -> Py.Object.t
(**
Computes the paired euclidean distances between X and Y

Read more in the :ref:`User Guide <metrics>`.

Parameters
----------
X : array-like, shape (n_samples, n_features)

Y : array-like, shape (n_samples, n_features)

Returns
-------
distances : ndarray (n_samples, )
*)

val paired_manhattan_distances : x:Ndarray.t -> y:Ndarray.t -> unit -> Py.Object.t
(**
Compute the L1 distances between the vectors in X and Y.

Read more in the :ref:`User Guide <metrics>`.

Parameters
----------
X : array-like, shape (n_samples, n_features)

Y : array-like, shape (n_samples, n_features)

Returns
-------
distances : ndarray (n_samples, )
*)

val pairwise_distances : ?y:Ndarray.t -> ?metric:[`String of string | `Callable of Py.Object.t] -> ?n_jobs:[`Int of int | `None] -> ?force_all_finite:[`Bool of bool | `Allow_nan] -> ?kwds:(string * Py.Object.t) list -> x:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> unit -> Py.Object.t
(**
Compute the distance matrix from a vector array X and optional Y.

This method takes either a vector array or a distance matrix, and returns
a distance matrix. If the input is a vector array, the distances are
computed. If the input is a distances matrix, it is returned instead.

This method provides a safe way to take a distance matrix as input, while
preserving compatibility with many other algorithms that take a vector
array.

If Y is given (default is None), then the returned matrix is the pairwise
distance between the arrays from both X and Y.

Valid values for metric are:

- From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
  'manhattan']. These metrics support sparse matrix
  inputs.
  ['nan_euclidean'] but it does not yet support sparse matrices.

- From scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
  'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
  'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
  'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
  See the documentation for scipy.spatial.distance for details on these
  metrics. These metrics do not support sparse matrix inputs.

Note that in the case of 'cityblock', 'cosine' and 'euclidean' (which are
valid scipy.spatial.distance metrics), the scikit-learn implementation
will be used, which is faster and has support for sparse matrices (except
for 'cityblock'). For a verbose description of the metrics from
scikit-learn, see the __doc__ of the sklearn.pairwise.distance_metrics
function.

Read more in the :ref:`User Guide <metrics>`.

Parameters
----------
X : array [n_samples_a, n_samples_a] if metric == "precomputed", or,              [n_samples_a, n_features] otherwise
    Array of pairwise distances between samples, or a feature array.

Y : array [n_samples_b, n_features], optional
    An optional second feature array. Only allowed if
    metric != "precomputed".

metric : string, or callable
    The metric to use when calculating distance between instances in a
    feature array. If metric is a string, it must be one of the options
    allowed by scipy.spatial.distance.pdist for its metric parameter, or
    a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
    If metric is "precomputed", X is assumed to be a distance matrix.
    Alternatively, if metric is a callable function, it is called on each
    pair of instances (rows) and the resulting value recorded. The callable
    should take two arrays from X as input and return a value indicating
    the distance between them.

n_jobs : int or None, optional (default=None)
    The number of jobs to use for the computation. This works by breaking
    down the pairwise matrix into n_jobs even slices and computing them in
    parallel.

    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

force_all_finite : boolean or 'allow-nan', (default=True)
    Whether to raise an error on np.inf and np.nan in array. The
    possibilities are:

    - True: Force all values of array to be finite.
    - False: accept both np.inf and np.nan in array.
    - 'allow-nan': accept only np.nan values in array. Values cannot
      be infinite.

    .. versionadded:: 0.22

**kwds : optional keyword parameters
    Any further parameters are passed directly to the distance function.
    If using a scipy.spatial.distance metric, the parameters are still
    metric dependent. See the scipy docs for usage examples.

Returns
-------
D : array [n_samples_a, n_samples_a] or [n_samples_a, n_samples_b]
    A distance matrix D such that D_{i, j} is the distance between the
    ith and jth vectors of the given matrix X, if Y is None.
    If Y is not None, then D_{i, j} is the distance between the ith array
    from X and the jth array from Y.

See also
--------
pairwise_distances_chunked : performs the same calculation as this
    function, but returns a generator of chunks of the distance matrix, in
    order to limit memory usage.
paired_distances : Computes the distances between corresponding
                   elements of two arrays
*)

val pairwise_distances_argmin : ?axis:int -> ?metric:[`String of string | `Callable of Py.Object.t] -> ?metric_kwargs:Py.Object.t -> x:Ndarray.t -> y:Ndarray.t -> unit -> Py.Object.t
(**
Compute minimum distances between one point and a set of points.

This function computes for each row in X, the index of the row of Y which
is closest (according to the specified distance).

This is mostly equivalent to calling:

    pairwise_distances(X, Y=Y, metric=metric).argmin(axis=axis)

but uses much less memory, and is faster for large arrays.

This function works with dense 2D arrays only.

Parameters
----------
X : array-like
    Arrays containing points. Respective shapes (n_samples1, n_features)
    and (n_samples2, n_features)

Y : array-like
    Arrays containing points. Respective shapes (n_samples1, n_features)
    and (n_samples2, n_features)

axis : int, optional, default 1
    Axis along which the argmin and distances are to be computed.

metric : string or callable
    metric to use for distance computation. Any metric from scikit-learn
    or scipy.spatial.distance can be used.

    If metric is a callable function, it is called on each
    pair of instances (rows) and the resulting value recorded. The callable
    should take two arrays as input and return one value indicating the
    distance between them. This works for Scipy's metrics, but is less
    efficient than passing the metric name as a string.

    Distance matrices are not supported.

    Valid values for metric are:

    - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
      'manhattan']

    - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
      'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
      'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao',
      'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
      'yule']

    See the documentation for scipy.spatial.distance for details on these
    metrics.

metric_kwargs : dict
    keyword arguments to pass to specified metric function.

Returns
-------
argmin : numpy.ndarray
    Y[argmin[i], :] is the row in Y that is closest to X[i, :].

See also
--------
sklearn.metrics.pairwise_distances
sklearn.metrics.pairwise_distances_argmin_min
*)

val pairwise_distances_argmin_min : ?axis:int -> ?metric:[`String of string | `Callable of Py.Object.t] -> ?metric_kwargs:Py.Object.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> unit -> (Py.Object.t * Py.Object.t)
(**
Compute minimum distances between one point and a set of points.

This function computes for each row in X, the index of the row of Y which
is closest (according to the specified distance). The minimal distances are
also returned.

This is mostly equivalent to calling:

    (pairwise_distances(X, Y=Y, metric=metric).argmin(axis=axis),
     pairwise_distances(X, Y=Y, metric=metric).min(axis=axis))

but uses much less memory, and is faster for large arrays.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples1, n_features)
    Array containing points.

Y : {array-like, sparse matrix}, shape (n_samples2, n_features)
    Arrays containing points.

axis : int, optional, default 1
    Axis along which the argmin and distances are to be computed.

metric : string or callable, default 'euclidean'
    metric to use for distance computation. Any metric from scikit-learn
    or scipy.spatial.distance can be used.

    If metric is a callable function, it is called on each
    pair of instances (rows) and the resulting value recorded. The callable
    should take two arrays as input and return one value indicating the
    distance between them. This works for Scipy's metrics, but is less
    efficient than passing the metric name as a string.

    Distance matrices are not supported.

    Valid values for metric are:

    - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
      'manhattan']

    - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
      'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
      'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao',
      'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
      'yule']

    See the documentation for scipy.spatial.distance for details on these
    metrics.

metric_kwargs : dict, optional
    Keyword arguments to pass to specified metric function.

Returns
-------
argmin : numpy.ndarray
    Y[argmin[i], :] is the row in Y that is closest to X[i, :].

distances : numpy.ndarray
    distances[i] is the distance between the i-th row in X and the
    argmin[i]-th row in Y.

See also
--------
sklearn.metrics.pairwise_distances
sklearn.metrics.pairwise_distances_argmin
*)

val pairwise_distances_chunked : ?y:Ndarray.t -> ?reduce_func:Py.Object.t -> ?metric:[`String of string | `Callable of Py.Object.t] -> ?n_jobs:[`Int of int | `None] -> ?working_memory:int -> ?kwds:(string * Py.Object.t) list -> x:Py.Object.t -> unit -> Py.Object.t
(**
Generate a distance matrix chunk by chunk with optional reduction

In cases where not all of a pairwise distance matrix needs to be stored at
once, this is used to calculate pairwise distances in
``working_memory``-sized chunks.  If ``reduce_func`` is given, it is run
on each chunk and its return values are concatenated into lists, arrays
or sparse matrices.

Parameters
----------
X : array [n_samples_a, n_samples_a] if metric == "precomputed", or,
    [n_samples_a, n_features] otherwise
    Array of pairwise distances between samples, or a feature array.

Y : array [n_samples_b, n_features], optional
    An optional second feature array. Only allowed if
    metric != "precomputed".

reduce_func : callable, optional
    The function which is applied on each chunk of the distance matrix,
    reducing it to needed values.  ``reduce_func(D_chunk, start)``
    is called repeatedly, where ``D_chunk`` is a contiguous vertical
    slice of the pairwise distance matrix, starting at row ``start``.
    It should return one of: None; an array, a list, or a sparse matrix
    of length ``D_chunk.shape[0]``; or a tuple of such objects. Returning
    None is useful for in-place operations, rather than reductions.

    If None, pairwise_distances_chunked returns a generator of vertical
    chunks of the distance matrix.

metric : string, or callable
    The metric to use when calculating distance between instances in a
    feature array. If metric is a string, it must be one of the options
    allowed by scipy.spatial.distance.pdist for its metric parameter, or
    a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
    If metric is "precomputed", X is assumed to be a distance matrix.
    Alternatively, if metric is a callable function, it is called on each
    pair of instances (rows) and the resulting value recorded. The callable
    should take two arrays from X as input and return a value indicating
    the distance between them.

n_jobs : int or None, optional (default=None)
    The number of jobs to use for the computation. This works by breaking
    down the pairwise matrix into n_jobs even slices and computing them in
    parallel.

    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

working_memory : int, optional
    The sought maximum memory for temporary distance matrix chunks.
    When None (default), the value of
    ``sklearn.get_config()['working_memory']`` is used.

`**kwds` : optional keyword parameters
    Any further parameters are passed directly to the distance function.
    If using a scipy.spatial.distance metric, the parameters are still
    metric dependent. See the scipy docs for usage examples.

Yields
------
D_chunk : array or sparse matrix
    A contiguous slice of distance matrix, optionally processed by
    ``reduce_func``.

Examples
--------
Without reduce_func:

>>> import numpy as np
>>> from sklearn.metrics import pairwise_distances_chunked
>>> X = np.random.RandomState(0).rand(5, 3)
>>> D_chunk = next(pairwise_distances_chunked(X))
>>> D_chunk
array([[0.  ..., 0.29..., 0.41..., 0.19..., 0.57...],
       [0.29..., 0.  ..., 0.57..., 0.41..., 0.76...],
       [0.41..., 0.57..., 0.  ..., 0.44..., 0.90...],
       [0.19..., 0.41..., 0.44..., 0.  ..., 0.51...],
       [0.57..., 0.76..., 0.90..., 0.51..., 0.  ...]])

Retrieve all neighbors and average distance within radius r:

>>> r = .2
>>> def reduce_func(D_chunk, start):
...     neigh = [np.flatnonzero(d < r) for d in D_chunk]
...     avg_dist = (D_chunk * (D_chunk < r)).mean(axis=1)
...     return neigh, avg_dist
>>> gen = pairwise_distances_chunked(X, reduce_func=reduce_func)
>>> neigh, avg_dist = next(gen)
>>> neigh
[array([0, 3]), array([1]), array([2]), array([0, 3]), array([4])]
>>> avg_dist
array([0.039..., 0.        , 0.        , 0.039..., 0.        ])

Where r is defined per sample, we need to make use of ``start``:

>>> r = [.2, .4, .4, .3, .1]
>>> def reduce_func(D_chunk, start):
...     neigh = [np.flatnonzero(d < r[i])
...              for i, d in enumerate(D_chunk, start)]
...     return neigh
>>> neigh = next(pairwise_distances_chunked(X, reduce_func=reduce_func))
>>> neigh
[array([0, 3]), array([0, 1]), array([2]), array([0, 3]), array([4])]

Force row-by-row generation by reducing ``working_memory``:

>>> gen = pairwise_distances_chunked(X, reduce_func=reduce_func,
...                                  working_memory=0)
>>> next(gen)
[array([0, 3])]
>>> next(gen)
[array([0, 1])]
*)

val pairwise_kernels : ?y:Ndarray.t -> ?metric:[`String of string | `Callable of Py.Object.t] -> ?filter_params:bool -> ?n_jobs:[`Int of int | `None] -> ?kwds:(string * Py.Object.t) list -> x:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> unit -> Py.Object.t
(**
Compute the kernel between arrays X and optional array Y.

This method takes either a vector array or a kernel matrix, and returns
a kernel matrix. If the input is a vector array, the kernels are
computed. If the input is a kernel matrix, it is returned instead.

This method provides a safe way to take a kernel matrix as input, while
preserving compatibility with many other algorithms that take a vector
array.

If Y is given (default is None), then the returned matrix is the pairwise
kernel between the arrays from both X and Y.

Valid values for metric are:
    ['additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf',
    'laplacian', 'sigmoid', 'cosine']

Read more in the :ref:`User Guide <metrics>`.

Parameters
----------
X : array [n_samples_a, n_samples_a] if metric == "precomputed", or,              [n_samples_a, n_features] otherwise
    Array of pairwise kernels between samples, or a feature array.

Y : array [n_samples_b, n_features]
    A second feature array only if X has shape [n_samples_a, n_features].

metric : string, or callable
    The metric to use when calculating kernel between instances in a
    feature array. If metric is a string, it must be one of the metrics
    in pairwise.PAIRWISE_KERNEL_FUNCTIONS.
    If metric is "precomputed", X is assumed to be a kernel matrix.
    Alternatively, if metric is a callable function, it is called on each
    pair of instances (rows) and the resulting value recorded. The callable
    should take two rows from X as input and return the corresponding
    kernel value as a single number. This means that callables from
    :mod:`sklearn.metrics.pairwise` are not allowed, as they operate on
    matrices, not single samples. Use the string identifying the kernel
    instead.

filter_params : boolean
    Whether to filter invalid parameters or not.

n_jobs : int or None, optional (default=None)
    The number of jobs to use for the computation. This works by breaking
    down the pairwise matrix into n_jobs even slices and computing them in
    parallel.

    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

**kwds : optional keyword parameters
    Any further parameters are passed directly to the kernel function.

Returns
-------
K : array [n_samples_a, n_samples_a] or [n_samples_a, n_samples_b]
    A kernel matrix K such that K_{i, j} is the kernel between the
    ith and jth vectors of the given matrix X, if Y is None.
    If Y is not None, then K_{i, j} is the kernel between the ith array
    from X and the jth array from Y.

Notes
-----
If metric is 'precomputed', Y is ignored and X is returned.
*)

val polynomial_kernel : ?y:Ndarray.t -> ?degree:int -> ?gamma:float -> ?coef0:float -> x:Ndarray.t -> unit -> Py.Object.t
(**
Compute the polynomial kernel between X and Y::

    K(X, Y) = (gamma <X, Y> + coef0)^degree

Read more in the :ref:`User Guide <polynomial_kernel>`.

Parameters
----------
X : ndarray of shape (n_samples_1, n_features)

Y : ndarray of shape (n_samples_2, n_features)

degree : int, default 3

gamma : float, default None
    if None, defaults to 1.0 / n_features

coef0 : float, default 1

Returns
-------
Gram matrix : array of shape (n_samples_1, n_samples_2)
*)

val rbf_kernel : ?y:Ndarray.t -> ?gamma:float -> x:Ndarray.t -> unit -> Ndarray.t
(**
Compute the rbf (gaussian) kernel between X and Y::

    K(x, y) = exp(-gamma ||x-y||^2)

for each pair of rows x in X and y in Y.

Read more in the :ref:`User Guide <rbf_kernel>`.

Parameters
----------
X : array of shape (n_samples_X, n_features)

Y : array of shape (n_samples_Y, n_features)

gamma : float, default None
    If None, defaults to 1.0 / n_features

Returns
-------
kernel_matrix : array of shape (n_samples_X, n_samples_Y)
*)

val row_norms : ?squared:Py.Object.t -> x:Ndarray.t -> unit -> Py.Object.t
(**
Row-wise (squared) Euclidean norm of X.

Equivalent to np.sqrt((X * X).sum(axis=1)), but also supports sparse
matrices and does not create an X.shape-sized temporary.

Performs no input validation.

Parameters
----------
X : array_like
    The input array
squared : bool, optional (default = False)
    If True, return squared norms.

Returns
-------
array_like
    The row-wise (squared) Euclidean norm of X.
*)

val safe_sparse_dot : ?dense_output:Py.Object.t -> a:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> b:Py.Object.t -> unit -> Py.Object.t
(**
Dot product that handle the sparse matrix case correctly

Parameters
----------
a : array or sparse matrix
b : array or sparse matrix
dense_output : boolean, (default=False)
    When False, ``a`` and ``b`` both being sparse will yield sparse output.
    When True, output will always be a dense array.

Returns
-------
dot_product : array or sparse matrix
    sparse if ``a`` and ``b`` are sparse and ``dense_output=False``.
*)

val sigmoid_kernel : ?y:Ndarray.t -> ?gamma:float -> ?coef0:float -> x:Ndarray.t -> unit -> Py.Object.t
(**
Compute the sigmoid kernel between X and Y::

    K(X, Y) = tanh(gamma <X, Y> + coef0)

Read more in the :ref:`User Guide <sigmoid_kernel>`.

Parameters
----------
X : ndarray of shape (n_samples_1, n_features)

Y : ndarray of shape (n_samples_2, n_features)

gamma : float, default None
    If None, defaults to 1.0 / n_features

coef0 : float, default 1

Returns
-------
Gram matrix : array of shape (n_samples_1, n_samples_2)
*)


end

val pairwise_distances : ?y:Ndarray.t -> ?metric:[`String of string | `Callable of Py.Object.t] -> ?n_jobs:[`Int of int | `None] -> ?force_all_finite:[`Bool of bool | `Allow_nan] -> ?kwds:(string * Py.Object.t) list -> x:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> unit -> Py.Object.t
(**
Compute the distance matrix from a vector array X and optional Y.

This method takes either a vector array or a distance matrix, and returns
a distance matrix. If the input is a vector array, the distances are
computed. If the input is a distances matrix, it is returned instead.

This method provides a safe way to take a distance matrix as input, while
preserving compatibility with many other algorithms that take a vector
array.

If Y is given (default is None), then the returned matrix is the pairwise
distance between the arrays from both X and Y.

Valid values for metric are:

- From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
  'manhattan']. These metrics support sparse matrix
  inputs.
  ['nan_euclidean'] but it does not yet support sparse matrices.

- From scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
  'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
  'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
  'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
  See the documentation for scipy.spatial.distance for details on these
  metrics. These metrics do not support sparse matrix inputs.

Note that in the case of 'cityblock', 'cosine' and 'euclidean' (which are
valid scipy.spatial.distance metrics), the scikit-learn implementation
will be used, which is faster and has support for sparse matrices (except
for 'cityblock'). For a verbose description of the metrics from
scikit-learn, see the __doc__ of the sklearn.pairwise.distance_metrics
function.

Read more in the :ref:`User Guide <metrics>`.

Parameters
----------
X : array [n_samples_a, n_samples_a] if metric == "precomputed", or,              [n_samples_a, n_features] otherwise
    Array of pairwise distances between samples, or a feature array.

Y : array [n_samples_b, n_features], optional
    An optional second feature array. Only allowed if
    metric != "precomputed".

metric : string, or callable
    The metric to use when calculating distance between instances in a
    feature array. If metric is a string, it must be one of the options
    allowed by scipy.spatial.distance.pdist for its metric parameter, or
    a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
    If metric is "precomputed", X is assumed to be a distance matrix.
    Alternatively, if metric is a callable function, it is called on each
    pair of instances (rows) and the resulting value recorded. The callable
    should take two arrays from X as input and return a value indicating
    the distance between them.

n_jobs : int or None, optional (default=None)
    The number of jobs to use for the computation. This works by breaking
    down the pairwise matrix into n_jobs even slices and computing them in
    parallel.

    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

force_all_finite : boolean or 'allow-nan', (default=True)
    Whether to raise an error on np.inf and np.nan in array. The
    possibilities are:

    - True: Force all values of array to be finite.
    - False: accept both np.inf and np.nan in array.
    - 'allow-nan': accept only np.nan values in array. Values cannot
      be infinite.

    .. versionadded:: 0.22

**kwds : optional keyword parameters
    Any further parameters are passed directly to the distance function.
    If using a scipy.spatial.distance metric, the parameters are still
    metric dependent. See the scipy docs for usage examples.

Returns
-------
D : array [n_samples_a, n_samples_a] or [n_samples_a, n_samples_b]
    A distance matrix D such that D_{i, j} is the distance between the
    ith and jth vectors of the given matrix X, if Y is None.
    If Y is not None, then D_{i, j} is the distance between the ith array
    from X and the jth array from Y.

See also
--------
pairwise_distances_chunked : performs the same calculation as this
    function, but returns a generator of chunks of the distance matrix, in
    order to limit memory usage.
paired_distances : Computes the distances between corresponding
                   elements of two arrays
*)

val pairwise_distances_argmin : ?axis:int -> ?metric:[`String of string | `Callable of Py.Object.t] -> ?metric_kwargs:Py.Object.t -> x:Ndarray.t -> y:Ndarray.t -> unit -> Py.Object.t
(**
Compute minimum distances between one point and a set of points.

This function computes for each row in X, the index of the row of Y which
is closest (according to the specified distance).

This is mostly equivalent to calling:

    pairwise_distances(X, Y=Y, metric=metric).argmin(axis=axis)

but uses much less memory, and is faster for large arrays.

This function works with dense 2D arrays only.

Parameters
----------
X : array-like
    Arrays containing points. Respective shapes (n_samples1, n_features)
    and (n_samples2, n_features)

Y : array-like
    Arrays containing points. Respective shapes (n_samples1, n_features)
    and (n_samples2, n_features)

axis : int, optional, default 1
    Axis along which the argmin and distances are to be computed.

metric : string or callable
    metric to use for distance computation. Any metric from scikit-learn
    or scipy.spatial.distance can be used.

    If metric is a callable function, it is called on each
    pair of instances (rows) and the resulting value recorded. The callable
    should take two arrays as input and return one value indicating the
    distance between them. This works for Scipy's metrics, but is less
    efficient than passing the metric name as a string.

    Distance matrices are not supported.

    Valid values for metric are:

    - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
      'manhattan']

    - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
      'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
      'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao',
      'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
      'yule']

    See the documentation for scipy.spatial.distance for details on these
    metrics.

metric_kwargs : dict
    keyword arguments to pass to specified metric function.

Returns
-------
argmin : numpy.ndarray
    Y[argmin[i], :] is the row in Y that is closest to X[i, :].

See also
--------
sklearn.metrics.pairwise_distances
sklearn.metrics.pairwise_distances_argmin_min
*)

val pairwise_distances_argmin_min : ?axis:int -> ?metric:[`String of string | `Callable of Py.Object.t] -> ?metric_kwargs:Py.Object.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> unit -> (Py.Object.t * Py.Object.t)
(**
Compute minimum distances between one point and a set of points.

This function computes for each row in X, the index of the row of Y which
is closest (according to the specified distance). The minimal distances are
also returned.

This is mostly equivalent to calling:

    (pairwise_distances(X, Y=Y, metric=metric).argmin(axis=axis),
     pairwise_distances(X, Y=Y, metric=metric).min(axis=axis))

but uses much less memory, and is faster for large arrays.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples1, n_features)
    Array containing points.

Y : {array-like, sparse matrix}, shape (n_samples2, n_features)
    Arrays containing points.

axis : int, optional, default 1
    Axis along which the argmin and distances are to be computed.

metric : string or callable, default 'euclidean'
    metric to use for distance computation. Any metric from scikit-learn
    or scipy.spatial.distance can be used.

    If metric is a callable function, it is called on each
    pair of instances (rows) and the resulting value recorded. The callable
    should take two arrays as input and return one value indicating the
    distance between them. This works for Scipy's metrics, but is less
    efficient than passing the metric name as a string.

    Distance matrices are not supported.

    Valid values for metric are:

    - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
      'manhattan']

    - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
      'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
      'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao',
      'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
      'yule']

    See the documentation for scipy.spatial.distance for details on these
    metrics.

metric_kwargs : dict, optional
    Keyword arguments to pass to specified metric function.

Returns
-------
argmin : numpy.ndarray
    Y[argmin[i], :] is the row in Y that is closest to X[i, :].

distances : numpy.ndarray
    distances[i] is the distance between the i-th row in X and the
    argmin[i]-th row in Y.

See also
--------
sklearn.metrics.pairwise_distances
sklearn.metrics.pairwise_distances_argmin
*)

val pairwise_distances_chunked : ?y:Ndarray.t -> ?reduce_func:Py.Object.t -> ?metric:[`String of string | `Callable of Py.Object.t] -> ?n_jobs:[`Int of int | `None] -> ?working_memory:int -> ?kwds:(string * Py.Object.t) list -> x:Py.Object.t -> unit -> Py.Object.t
(**
Generate a distance matrix chunk by chunk with optional reduction

In cases where not all of a pairwise distance matrix needs to be stored at
once, this is used to calculate pairwise distances in
``working_memory``-sized chunks.  If ``reduce_func`` is given, it is run
on each chunk and its return values are concatenated into lists, arrays
or sparse matrices.

Parameters
----------
X : array [n_samples_a, n_samples_a] if metric == "precomputed", or,
    [n_samples_a, n_features] otherwise
    Array of pairwise distances between samples, or a feature array.

Y : array [n_samples_b, n_features], optional
    An optional second feature array. Only allowed if
    metric != "precomputed".

reduce_func : callable, optional
    The function which is applied on each chunk of the distance matrix,
    reducing it to needed values.  ``reduce_func(D_chunk, start)``
    is called repeatedly, where ``D_chunk`` is a contiguous vertical
    slice of the pairwise distance matrix, starting at row ``start``.
    It should return one of: None; an array, a list, or a sparse matrix
    of length ``D_chunk.shape[0]``; or a tuple of such objects. Returning
    None is useful for in-place operations, rather than reductions.

    If None, pairwise_distances_chunked returns a generator of vertical
    chunks of the distance matrix.

metric : string, or callable
    The metric to use when calculating distance between instances in a
    feature array. If metric is a string, it must be one of the options
    allowed by scipy.spatial.distance.pdist for its metric parameter, or
    a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
    If metric is "precomputed", X is assumed to be a distance matrix.
    Alternatively, if metric is a callable function, it is called on each
    pair of instances (rows) and the resulting value recorded. The callable
    should take two arrays from X as input and return a value indicating
    the distance between them.

n_jobs : int or None, optional (default=None)
    The number of jobs to use for the computation. This works by breaking
    down the pairwise matrix into n_jobs even slices and computing them in
    parallel.

    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

working_memory : int, optional
    The sought maximum memory for temporary distance matrix chunks.
    When None (default), the value of
    ``sklearn.get_config()['working_memory']`` is used.

`**kwds` : optional keyword parameters
    Any further parameters are passed directly to the distance function.
    If using a scipy.spatial.distance metric, the parameters are still
    metric dependent. See the scipy docs for usage examples.

Yields
------
D_chunk : array or sparse matrix
    A contiguous slice of distance matrix, optionally processed by
    ``reduce_func``.

Examples
--------
Without reduce_func:

>>> import numpy as np
>>> from sklearn.metrics import pairwise_distances_chunked
>>> X = np.random.RandomState(0).rand(5, 3)
>>> D_chunk = next(pairwise_distances_chunked(X))
>>> D_chunk
array([[0.  ..., 0.29..., 0.41..., 0.19..., 0.57...],
       [0.29..., 0.  ..., 0.57..., 0.41..., 0.76...],
       [0.41..., 0.57..., 0.  ..., 0.44..., 0.90...],
       [0.19..., 0.41..., 0.44..., 0.  ..., 0.51...],
       [0.57..., 0.76..., 0.90..., 0.51..., 0.  ...]])

Retrieve all neighbors and average distance within radius r:

>>> r = .2
>>> def reduce_func(D_chunk, start):
...     neigh = [np.flatnonzero(d < r) for d in D_chunk]
...     avg_dist = (D_chunk * (D_chunk < r)).mean(axis=1)
...     return neigh, avg_dist
>>> gen = pairwise_distances_chunked(X, reduce_func=reduce_func)
>>> neigh, avg_dist = next(gen)
>>> neigh
[array([0, 3]), array([1]), array([2]), array([0, 3]), array([4])]
>>> avg_dist
array([0.039..., 0.        , 0.        , 0.039..., 0.        ])

Where r is defined per sample, we need to make use of ``start``:

>>> r = [.2, .4, .4, .3, .1]
>>> def reduce_func(D_chunk, start):
...     neigh = [np.flatnonzero(d < r[i])
...              for i, d in enumerate(D_chunk, start)]
...     return neigh
>>> neigh = next(pairwise_distances_chunked(X, reduce_func=reduce_func))
>>> neigh
[array([0, 3]), array([0, 1]), array([2]), array([0, 3]), array([4])]

Force row-by-row generation by reducing ``working_memory``:

>>> gen = pairwise_distances_chunked(X, reduce_func=reduce_func,
...                                  working_memory=0)
>>> next(gen)
[array([0, 3])]
>>> next(gen)
[array([0, 1])]
*)

val pairwise_kernels : ?y:Ndarray.t -> ?metric:[`String of string | `Callable of Py.Object.t] -> ?filter_params:bool -> ?n_jobs:[`Int of int | `None] -> ?kwds:(string * Py.Object.t) list -> x:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> unit -> Py.Object.t
(**
Compute the kernel between arrays X and optional array Y.

This method takes either a vector array or a kernel matrix, and returns
a kernel matrix. If the input is a vector array, the kernels are
computed. If the input is a kernel matrix, it is returned instead.

This method provides a safe way to take a kernel matrix as input, while
preserving compatibility with many other algorithms that take a vector
array.

If Y is given (default is None), then the returned matrix is the pairwise
kernel between the arrays from both X and Y.

Valid values for metric are:
    ['additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf',
    'laplacian', 'sigmoid', 'cosine']

Read more in the :ref:`User Guide <metrics>`.

Parameters
----------
X : array [n_samples_a, n_samples_a] if metric == "precomputed", or,              [n_samples_a, n_features] otherwise
    Array of pairwise kernels between samples, or a feature array.

Y : array [n_samples_b, n_features]
    A second feature array only if X has shape [n_samples_a, n_features].

metric : string, or callable
    The metric to use when calculating kernel between instances in a
    feature array. If metric is a string, it must be one of the metrics
    in pairwise.PAIRWISE_KERNEL_FUNCTIONS.
    If metric is "precomputed", X is assumed to be a kernel matrix.
    Alternatively, if metric is a callable function, it is called on each
    pair of instances (rows) and the resulting value recorded. The callable
    should take two rows from X as input and return the corresponding
    kernel value as a single number. This means that callables from
    :mod:`sklearn.metrics.pairwise` are not allowed, as they operate on
    matrices, not single samples. Use the string identifying the kernel
    instead.

filter_params : boolean
    Whether to filter invalid parameters or not.

n_jobs : int or None, optional (default=None)
    The number of jobs to use for the computation. This works by breaking
    down the pairwise matrix into n_jobs even slices and computing them in
    parallel.

    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

**kwds : optional keyword parameters
    Any further parameters are passed directly to the kernel function.

Returns
-------
K : array [n_samples_a, n_samples_a] or [n_samples_a, n_samples_b]
    A kernel matrix K such that K_{i, j} is the kernel between the
    ith and jth vectors of the given matrix X, if Y is None.
    If Y is not None, then K_{i, j} is the kernel between the ith array
    from X and the jth array from Y.

Notes
-----
If metric is 'precomputed', Y is ignored and X is returned.
*)

val plot_confusion_matrix : ?labels:Ndarray.t -> ?sample_weight:Ndarray.t -> ?normalize:[`True | `Pred | `All] -> ?display_labels:Ndarray.t -> ?include_values:bool -> ?xticks_rotation:[`Vertical | `Horizontal | `Float of float] -> ?values_format:string -> ?cmap:[`String of string | `PyObject of Py.Object.t] -> ?ax:Py.Object.t -> estimator:Py.Object.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y_true:Py.Object.t -> unit -> Py.Object.t
(**
Plot Confusion Matrix.

Read more in the :ref:`User Guide <confusion_matrix>`.

Parameters
----------
estimator : estimator instance
    Trained classifier.

X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Input values.

y : array-like of shape (n_samples,)
    Target values.

labels : array-like of shape (n_classes,), default=None
    List of labels to index the matrix. This may be used to reorder or
    select a subset of labels. If `None` is given, those that appear at
    least once in `y_true` or `y_pred` are used in sorted order.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

normalize : {'true', 'pred', 'all'}, default=None
    Normalizes confusion matrix over the true (rows), predicted (columns)
    conditions or all the population. If None, confusion matrix will not be
    normalized.

display_labels : array-like of shape (n_classes,), default=None
    Target names used for plotting. By default, `labels` will be used if
    it is defined, otherwise the unique labels of `y_true` and `y_pred`
    will be used.

include_values : bool, default=True
    Includes values in confusion matrix.

xticks_rotation : {'vertical', 'horizontal'} or float,                         default='horizontal'
    Rotation of xtick labels.

values_format : str, default=None
    Format specification for values in confusion matrix. If `None`,
    the format specification is '.2g'.

cmap : str or matplotlib Colormap, default='viridis'
    Colormap recognized by matplotlib.

ax : matplotlib Axes, default=None
    Axes object to plot on. If `None`, a new figure and axes is
    created.

Returns
-------
display : :class:`~sklearn.metrics.ConfusionMatrixDisplay`
*)

val plot_precision_recall_curve : ?sample_weight:Ndarray.t -> ?response_method:[`Predict_proba | `Decision_function | `Auto] -> ?name:string -> ?ax:Py.Object.t -> ?kwargs:(string * Py.Object.t) list -> estimator:Py.Object.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:Ndarray.t -> unit -> Py.Object.t
(**
Plot Precision Recall Curve for binary classifiers.

Extra keyword arguments will be passed to matplotlib's `plot`.

Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

Parameters
----------
estimator : estimator instance
    Trained classifier.

X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Input values.

y : array-like of shape (n_samples,)
    Binary target values.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

response_method : {'predict_proba', 'decision_function', 'auto'},                       default='auto'
    Specifies whether to use :term:`predict_proba` or
    :term:`decision_function` as the target response. If set to 'auto',
    :term:`predict_proba` is tried first and if it does not exist
    :term:`decision_function` is tried next.

name : str, default=None
    Name for labeling curve. If `None`, the name of the
    estimator is used.

ax : matplotlib axes, default=None
    Axes object to plot on. If `None`, a new figure and axes is created.

**kwargs : dict
    Keyword arguments to be passed to matplotlib's `plot`.

Returns
-------
display : :class:`~sklearn.metrics.PrecisionRecallDisplay`
    Object that stores computed values.
*)

val plot_roc_curve : ?sample_weight:Ndarray.t -> ?drop_intermediate:bool -> ?response_method:[`Predict_proba | `Decision_function | `Auto] -> ?name:string -> ?ax:Py.Object.t -> ?kwargs:(string * Py.Object.t) list -> estimator:Py.Object.t -> x:[`Ndarray of Ndarray.t | `SparseMatrix of Csr_matrix.t] -> y:Ndarray.t -> unit -> Py.Object.t
(**
Plot Receiver operating characteristic (ROC) curve.

Extra keyword arguments will be passed to matplotlib's `plot`.

Read more in the :ref:`User Guide <visualizations>`.

Parameters
----------
estimator : estimator instance
    Trained classifier.

X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Input values.

y : array-like of shape (n_samples,)
    Target values.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

drop_intermediate : boolean, default=True
    Whether to drop some suboptimal thresholds which would not appear
    on a plotted ROC curve. This is useful in order to create lighter
    ROC curves.

response_method : {'predict_proba', 'decision_function', 'auto'}     default='auto'
    Specifies whether to use :term:`predict_proba` or
    :term:`decision_function` as the target response. If set to 'auto',
    :term:`predict_proba` is tried first and if it does not exist
    :term:`decision_function` is tried next.

name : str, default=None
    Name of ROC Curve for labeling. If `None`, use the name of the
    estimator.

ax : matplotlib axes, default=None
    Axes object to plot on. If `None`, a new figure and axes is created.

Returns
-------
display : :class:`~sklearn.metrics.RocCurveDisplay`
    Object that stores computed values.

Examples
--------
>>> import matplotlib.pyplot as plt  # doctest: +SKIP
>>> from sklearn import datasets, metrics, model_selection, svm
>>> X, y = datasets.make_classification(random_state=0)
>>> X_train, X_test, y_train, y_test = model_selection.train_test_split(            X, y, random_state=0)
>>> clf = svm.SVC(random_state=0)
>>> clf.fit(X_train, y_train)
SVC(random_state=0)
>>> metrics.plot_roc_curve(clf, X_test, y_test)  # doctest: +SKIP
>>> plt.show()                                   # doctest: +SKIP
*)

val precision_recall_curve : ?pos_label:[`Int of int | `String of string] -> ?sample_weight:Ndarray.t -> y_true:Ndarray.t -> probas_pred:Ndarray.t -> unit -> (Py.Object.t * Py.Object.t * Py.Object.t)
(**
Compute precision-recall pairs for different probability thresholds

Note: this implementation is restricted to the binary classification task.

The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
true positives and ``fp`` the number of false positives. The precision is
intuitively the ability of the classifier not to label as positive a sample
that is negative.

The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
true positives and ``fn`` the number of false negatives. The recall is
intuitively the ability of the classifier to find all the positive samples.

The last precision and recall values are 1. and 0. respectively and do not
have a corresponding threshold.  This ensures that the graph starts on the
y axis.

Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

Parameters
----------
y_true : array, shape = [n_samples]
    True binary labels. If labels are not either {-1, 1} or {0, 1}, then
    pos_label should be explicitly given.

probas_pred : array, shape = [n_samples]
    Estimated probabilities or decision function.

pos_label : int or str, default=None
    The label of the positive class.
    When ``pos_label=None``, if y_true is in {-1, 1} or {0, 1},
    ``pos_label`` is set to 1, otherwise an error will be raised.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

Returns
-------
precision : array, shape = [n_thresholds + 1]
    Precision values such that element i is the precision of
    predictions with score >= thresholds[i] and the last element is 1.

recall : array, shape = [n_thresholds + 1]
    Decreasing recall values such that element i is the recall of
    predictions with score >= thresholds[i] and the last element is 0.

thresholds : array, shape = [n_thresholds <= len(np.unique(probas_pred))]
    Increasing thresholds on the decision function used to compute
    precision and recall.

See also
--------
average_precision_score : Compute average precision from prediction scores

roc_curve : Compute Receiver operating characteristic (ROC) curve

Examples
--------
>>> import numpy as np
>>> from sklearn.metrics import precision_recall_curve
>>> y_true = np.array([0, 0, 1, 1])
>>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
>>> precision, recall, thresholds = precision_recall_curve(
...     y_true, y_scores)
>>> precision
array([0.66666667, 0.5       , 1.        , 1.        ])
>>> recall
array([1. , 0.5, 0.5, 0. ])
>>> thresholds
array([0.35, 0.4 , 0.8 ])
*)

val precision_recall_fscore_support : ?beta:float -> ?labels:Py.Object.t -> ?pos_label:[`String of string | `Int of int] -> ?average:[`String of string | `PyObject of Py.Object.t] -> ?warn_for:Py.Object.t -> ?sample_weight:Ndarray.t -> ?zero_division:[`Warn | `PyObject of Py.Object.t] -> y_true:Py.Object.t -> y_pred:Py.Object.t -> unit -> (Py.Object.t * Py.Object.t * Py.Object.t * Py.Object.t)
(**
Compute precision, recall, F-measure and support for each class

The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
true positives and ``fp`` the number of false positives. The precision is
intuitively the ability of the classifier not to label as positive a sample
that is negative.

The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
true positives and ``fn`` the number of false negatives. The recall is
intuitively the ability of the classifier to find all the positive samples.

The F-beta score can be interpreted as a weighted harmonic mean of
the precision and recall, where an F-beta score reaches its best
value at 1 and worst score at 0.

The F-beta score weights recall more than precision by a factor of
``beta``. ``beta == 1.0`` means recall and precision are equally important.

The support is the number of occurrences of each class in ``y_true``.

If ``pos_label is None`` and in binary classification, this function
returns the average precision, recall and F-measure if ``average``
is one of ``'micro'``, ``'macro'``, ``'weighted'`` or ``'samples'``.

Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

Parameters
----------
y_true : 1d array-like, or label indicator array / sparse matrix
    Ground truth (correct) target values.

y_pred : 1d array-like, or label indicator array / sparse matrix
    Estimated targets as returned by a classifier.

beta : float, 1.0 by default
    The strength of recall versus precision in the F-score.

labels : list, optional
    The set of labels to include when ``average != 'binary'``, and their
    order if ``average is None``. Labels present in the data can be
    excluded, for example to calculate a multiclass average ignoring a
    majority negative class, while labels not present in the data will
    result in 0 components in a macro average. For multilabel targets,
    labels are column indices. By default, all labels in ``y_true`` and
    ``y_pred`` are used in sorted order.

pos_label : str or int, 1 by default
    The class to report if ``average='binary'`` and the data is binary.
    If the data are multiclass or multilabel, this will be ignored;
    setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
    scores for that label only.

average : string, [None (default), 'binary', 'micro', 'macro', 'samples',                        'weighted']
    If ``None``, the scores for each class are returned. Otherwise, this
    determines the type of averaging performed on the data:

    ``'binary'``:
        Only report results for the class specified by ``pos_label``.
        This is applicable only if targets (``y_{true,pred}``) are binary.
    ``'micro'``:
        Calculate metrics globally by counting the total true positives,
        false negatives and false positives.
    ``'macro'``:
        Calculate metrics for each label, and find their unweighted
        mean.  This does not take label imbalance into account.
    ``'weighted'``:
        Calculate metrics for each label, and find their average weighted
        by support (the number of true instances for each label). This
        alters 'macro' to account for label imbalance; it can result in an
        F-score that is not between precision and recall.
    ``'samples'``:
        Calculate metrics for each instance, and find their average (only
        meaningful for multilabel classification where this differs from
        :func:`accuracy_score`).

warn_for : tuple or set, for internal use
    This determines which warnings will be made in the case that this
    function is being used to return only one of its metrics.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

zero_division : "warn", 0 or 1, default="warn"
    Sets the value to return when there is a zero division:
       - recall: when there are no positive labels
       - precision: when there are no positive predictions
       - f-score: both

    If set to "warn", this acts as 0, but warnings are also raised.

Returns
-------
precision : float (if average is not None) or array of float, shape =        [n_unique_labels]

recall : float (if average is not None) or array of float, , shape =        [n_unique_labels]

fbeta_score : float (if average is not None) or array of float, shape =        [n_unique_labels]

support : None (if average is not None) or array of int, shape =        [n_unique_labels]
    The number of occurrences of each label in ``y_true``.

References
----------
.. [1] `Wikipedia entry for the Precision and recall
       <https://en.wikipedia.org/wiki/Precision_and_recall>`_

.. [2] `Wikipedia entry for the F1-score
       <https://en.wikipedia.org/wiki/F1_score>`_

.. [3] `Discriminative Methods for Multi-labeled Classification Advances
       in Knowledge Discovery and Data Mining (2004), pp. 22-30 by Shantanu
       Godbole, Sunita Sarawagi
       <http://www.godbole.net/shantanu/pubs/multilabelsvm-pakdd04.pdf>`_

Examples
--------
>>> import numpy as np
>>> from sklearn.metrics import precision_recall_fscore_support
>>> y_true = np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'pig'])
>>> y_pred = np.array(['cat', 'pig', 'dog', 'cat', 'cat', 'dog'])
>>> precision_recall_fscore_support(y_true, y_pred, average='macro')
(0.22..., 0.33..., 0.26..., None)
>>> precision_recall_fscore_support(y_true, y_pred, average='micro')
(0.33..., 0.33..., 0.33..., None)
>>> precision_recall_fscore_support(y_true, y_pred, average='weighted')
(0.22..., 0.33..., 0.26..., None)

It is possible to compute per-label precisions, recalls, F1-scores and
supports instead of averaging:

>>> precision_recall_fscore_support(y_true, y_pred, average=None,
... labels=['pig', 'dog', 'cat'])
(array([0.        , 0.        , 0.66...]),
 array([0., 0., 1.]), array([0. , 0. , 0.8]),
 array([2, 2, 2]))

Notes
-----
When ``true positive + false positive == 0``, precision is undefined;
When ``true positive + false negative == 0``, recall is undefined.
In such cases, by default the metric will be set to 0, as will f-score,
and ``UndefinedMetricWarning`` will be raised. This behavior can be
modified with ``zero_division``.
*)

val precision_score : ?labels:Py.Object.t -> ?pos_label:[`String of string | `Int of int] -> ?average:[`String of string | `Binary | `PyObject of Py.Object.t] -> ?sample_weight:Ndarray.t -> ?zero_division:[`Warn | `PyObject of Py.Object.t] -> y_true:Py.Object.t -> y_pred:Py.Object.t -> unit -> Py.Object.t
(**
Compute the precision

The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
true positives and ``fp`` the number of false positives. The precision is
intuitively the ability of the classifier not to label as positive a sample
that is negative.

The best value is 1 and the worst value is 0.

Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

Parameters
----------
y_true : 1d array-like, or label indicator array / sparse matrix
    Ground truth (correct) target values.

y_pred : 1d array-like, or label indicator array / sparse matrix
    Estimated targets as returned by a classifier.

labels : list, optional
    The set of labels to include when ``average != 'binary'``, and their
    order if ``average is None``. Labels present in the data can be
    excluded, for example to calculate a multiclass average ignoring a
    majority negative class, while labels not present in the data will
    result in 0 components in a macro average. For multilabel targets,
    labels are column indices. By default, all labels in ``y_true`` and
    ``y_pred`` are used in sorted order.

    .. versionchanged:: 0.17
       parameter *labels* improved for multiclass problem.

pos_label : str or int, 1 by default
    The class to report if ``average='binary'`` and the data is binary.
    If the data are multiclass or multilabel, this will be ignored;
    setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
    scores for that label only.

average : string, [None, 'binary' (default), 'micro', 'macro', 'samples',                        'weighted']
    This parameter is required for multiclass/multilabel targets.
    If ``None``, the scores for each class are returned. Otherwise, this
    determines the type of averaging performed on the data:

    ``'binary'``:
        Only report results for the class specified by ``pos_label``.
        This is applicable only if targets (``y_{true,pred}``) are binary.
    ``'micro'``:
        Calculate metrics globally by counting the total true positives,
        false negatives and false positives.
    ``'macro'``:
        Calculate metrics for each label, and find their unweighted
        mean.  This does not take label imbalance into account.
    ``'weighted'``:
        Calculate metrics for each label, and find their average weighted
        by support (the number of true instances for each label). This
        alters 'macro' to account for label imbalance; it can result in an
        F-score that is not between precision and recall.
    ``'samples'``:
        Calculate metrics for each instance, and find their average (only
        meaningful for multilabel classification where this differs from
        :func:`accuracy_score`).

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

zero_division : "warn", 0 or 1, default="warn"
    Sets the value to return when there is a zero division. If set to
    "warn", this acts as 0, but warnings are also raised.

Returns
-------
precision : float (if average is not None) or array of float, shape =        [n_unique_labels]
    Precision of the positive class in binary classification or weighted
    average of the precision of each class for the multiclass task.

See also
--------
precision_recall_fscore_support, multilabel_confusion_matrix

Examples
--------
>>> from sklearn.metrics import precision_score
>>> y_true = [0, 1, 2, 0, 1, 2]
>>> y_pred = [0, 2, 1, 0, 0, 1]
>>> precision_score(y_true, y_pred, average='macro')
0.22...
>>> precision_score(y_true, y_pred, average='micro')
0.33...
>>> precision_score(y_true, y_pred, average='weighted')
0.22...
>>> precision_score(y_true, y_pred, average=None)
array([0.66..., 0.        , 0.        ])
>>> y_pred = [0, 0, 0, 0, 0, 0]
>>> precision_score(y_true, y_pred, average=None)
array([0.33..., 0.        , 0.        ])
>>> precision_score(y_true, y_pred, average=None, zero_division=1)
array([0.33..., 1.        , 1.        ])

Notes
-----
When ``true positive + false positive == 0``, precision returns 0 and
raises ``UndefinedMetricWarning``. This behavior can be
modified with ``zero_division``.
*)

val r2_score : ?sample_weight:Ndarray.t -> ?multioutput:[`Ndarray of Ndarray.t | `None | `PyObject of Py.Object.t] -> y_true:Ndarray.t -> y_pred:Ndarray.t -> unit -> Py.Object.t
(**
R^2 (coefficient of determination) regression score function.

Best possible score is 1.0 and it can be negative (because the
model can be arbitrarily worse). A constant model that always
predicts the expected value of y, disregarding the input features,
would get a R^2 score of 0.0.

Read more in the :ref:`User Guide <r2_score>`.

Parameters
----------
y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
    Ground truth (correct) target values.

y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
    Estimated target values.

sample_weight : array-like of shape (n_samples,), optional
    Sample weights.

multioutput : string in ['raw_values', 'uniform_average', 'variance_weighted'] or None or array-like of shape (n_outputs)

    Defines aggregating of multiple output scores.
    Array-like value defines weights used to average scores.
    Default is "uniform_average".

    'raw_values' :
        Returns a full set of scores in case of multioutput input.

    'uniform_average' :
        Scores of all outputs are averaged with uniform weight.

    'variance_weighted' :
        Scores of all outputs are averaged, weighted by the variances
        of each individual output.

    .. versionchanged:: 0.19
        Default value of multioutput is 'uniform_average'.

Returns
-------
z : float or ndarray of floats
    The R^2 score or ndarray of scores if 'multioutput' is
    'raw_values'.

Notes
-----
This is not a symmetric function.

Unlike most other scores, R^2 score may be negative (it need not actually
be the square of a quantity R).

This metric is not well-defined for single samples and will return a NaN
value if n_samples is less than two.

References
----------
.. [1] `Wikipedia entry on the Coefficient of determination
        <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_

Examples
--------
>>> from sklearn.metrics import r2_score
>>> y_true = [3, -0.5, 2, 7]
>>> y_pred = [2.5, 0.0, 2, 8]
>>> r2_score(y_true, y_pred)
0.948...
>>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
>>> y_pred = [[0, 2], [-1, 2], [8, -5]]
>>> r2_score(y_true, y_pred,
...          multioutput='variance_weighted')
0.938...
>>> y_true = [1, 2, 3]
>>> y_pred = [1, 2, 3]
>>> r2_score(y_true, y_pred)
1.0
>>> y_true = [1, 2, 3]
>>> y_pred = [2, 2, 2]
>>> r2_score(y_true, y_pred)
0.0
>>> y_true = [1, 2, 3]
>>> y_pred = [3, 2, 1]
>>> r2_score(y_true, y_pred)
-3.0
*)

val recall_score : ?labels:Py.Object.t -> ?pos_label:[`String of string | `Int of int] -> ?average:[`String of string | `Binary | `PyObject of Py.Object.t] -> ?sample_weight:Ndarray.t -> ?zero_division:[`Warn | `PyObject of Py.Object.t] -> y_true:Py.Object.t -> y_pred:Py.Object.t -> unit -> Py.Object.t
(**
Compute the recall

The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
true positives and ``fn`` the number of false negatives. The recall is
intuitively the ability of the classifier to find all the positive samples.

The best value is 1 and the worst value is 0.

Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

Parameters
----------
y_true : 1d array-like, or label indicator array / sparse matrix
    Ground truth (correct) target values.

y_pred : 1d array-like, or label indicator array / sparse matrix
    Estimated targets as returned by a classifier.

labels : list, optional
    The set of labels to include when ``average != 'binary'``, and their
    order if ``average is None``. Labels present in the data can be
    excluded, for example to calculate a multiclass average ignoring a
    majority negative class, while labels not present in the data will
    result in 0 components in a macro average. For multilabel targets,
    labels are column indices. By default, all labels in ``y_true`` and
    ``y_pred`` are used in sorted order.

    .. versionchanged:: 0.17
       parameter *labels* improved for multiclass problem.

pos_label : str or int, 1 by default
    The class to report if ``average='binary'`` and the data is binary.
    If the data are multiclass or multilabel, this will be ignored;
    setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
    scores for that label only.

average : string, [None, 'binary' (default), 'micro', 'macro', 'samples',                        'weighted']
    This parameter is required for multiclass/multilabel targets.
    If ``None``, the scores for each class are returned. Otherwise, this
    determines the type of averaging performed on the data:

    ``'binary'``:
        Only report results for the class specified by ``pos_label``.
        This is applicable only if targets (``y_{true,pred}``) are binary.
    ``'micro'``:
        Calculate metrics globally by counting the total true positives,
        false negatives and false positives.
    ``'macro'``:
        Calculate metrics for each label, and find their unweighted
        mean.  This does not take label imbalance into account.
    ``'weighted'``:
        Calculate metrics for each label, and find their average weighted
        by support (the number of true instances for each label). This
        alters 'macro' to account for label imbalance; it can result in an
        F-score that is not between precision and recall.
    ``'samples'``:
        Calculate metrics for each instance, and find their average (only
        meaningful for multilabel classification where this differs from
        :func:`accuracy_score`).

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

zero_division : "warn", 0 or 1, default="warn"
    Sets the value to return when there is a zero division. If set to
    "warn", this acts as 0, but warnings are also raised.

Returns
-------
recall : float (if average is not None) or array of float, shape =        [n_unique_labels]
    Recall of the positive class in binary classification or weighted
    average of the recall of each class for the multiclass task.

See also
--------
precision_recall_fscore_support, balanced_accuracy_score,
multilabel_confusion_matrix

Examples
--------
>>> from sklearn.metrics import recall_score
>>> y_true = [0, 1, 2, 0, 1, 2]
>>> y_pred = [0, 2, 1, 0, 0, 1]
>>> recall_score(y_true, y_pred, average='macro')
0.33...
>>> recall_score(y_true, y_pred, average='micro')
0.33...
>>> recall_score(y_true, y_pred, average='weighted')
0.33...
>>> recall_score(y_true, y_pred, average=None)
array([1., 0., 0.])
>>> y_true = [0, 0, 0, 0, 0, 0]
>>> recall_score(y_true, y_pred, average=None)
array([0.5, 0. , 0. ])
>>> recall_score(y_true, y_pred, average=None, zero_division=1)
array([0.5, 1. , 1. ])

Notes
-----
When ``true positive + false negative == 0``, recall returns 0 and raises
``UndefinedMetricWarning``. This behavior can be modified with
``zero_division``.
*)

val roc_auc_score : ?average:[`Micro | `Macro | `Samples | `Weighted | `None] -> ?sample_weight:Ndarray.t -> ?max_fpr:Py.Object.t -> ?multi_class:[`Raise | `Ovr | `Ovo] -> ?labels:Ndarray.t -> y_true:Ndarray.t -> y_score:Ndarray.t -> unit -> float
(**
Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
from prediction scores.

Note: this implementation can be used with binary, multiclass and
multilabel classification, but some restrictions apply (see Parameters).

Read more in the :ref:`User Guide <roc_metrics>`.

Parameters
----------
y_true : array-like of shape (n_samples,) or (n_samples, n_classes)
    True labels or binary label indicators. The binary and multiclass cases
    expect labels with shape (n_samples,) while the multilabel case expects
    binary label indicators with shape (n_samples, n_classes).

y_score : array-like of shape (n_samples,) or (n_samples, n_classes)
    Target scores. In the binary and multilabel cases, these can be either
    probability estimates or non-thresholded decision values (as returned
    by `decision_function` on some classifiers). In the multiclass case,
    these must be probability estimates which sum to 1. The binary
    case expects a shape (n_samples,), and the scores must be the scores of
    the class with the greater label. The multiclass and multilabel
    cases expect a shape (n_samples, n_classes). In the multiclass case,
    the order of the class scores must correspond to the order of
    ``labels``, if provided, or else to the numerical or lexicographical
    order of the labels in ``y_true``.

average : {'micro', 'macro', 'samples', 'weighted'} or None,             default='macro'
    If ``None``, the scores for each class are returned. Otherwise,
    this determines the type of averaging performed on the data:
    Note: multiclass ROC AUC currently only handles the 'macro' and
    'weighted' averages.

    ``'micro'``:
        Calculate metrics globally by considering each element of the label
        indicator matrix as a label.
    ``'macro'``:
        Calculate metrics for each label, and find their unweighted
        mean.  This does not take label imbalance into account.
    ``'weighted'``:
        Calculate metrics for each label, and find their average, weighted
        by support (the number of true instances for each label).
    ``'samples'``:
        Calculate metrics for each instance, and find their average.

    Will be ignored when ``y_true`` is binary.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

max_fpr : float > 0 and <= 1, default=None
    If not ``None``, the standardized partial AUC [2]_ over the range
    [0, max_fpr] is returned. For the multiclass case, ``max_fpr``,
    should be either equal to ``None`` or ``1.0`` as AUC ROC partial
    computation currently is not supported for multiclass.

multi_class : {'raise', 'ovr', 'ovo'}, default='raise'
    Multiclass only. Determines the type of configuration to use. The
    default value raises an error, so either ``'ovr'`` or ``'ovo'`` must be
    passed explicitly.

    ``'ovr'``:
        Computes the AUC of each class against the rest [3]_ [4]_. This
        treats the multiclass case in the same way as the multilabel case.
        Sensitive to class imbalance even when ``average == 'macro'``,
        because class imbalance affects the composition of each of the
        'rest' groupings.
    ``'ovo'``:
        Computes the average AUC of all possible pairwise combinations of
        classes [5]_. Insensitive to class imbalance when
        ``average == 'macro'``.

labels : array-like of shape (n_classes,), default=None
    Multiclass only. List of labels that index the classes in ``y_score``.
    If ``None``, the numerical or lexicographical order of the labels in
    ``y_true`` is used.

Returns
-------
auc : float

References
----------
.. [1] `Wikipedia entry for the Receiver operating characteristic
        <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_

.. [2] `Analyzing a portion of the ROC curve. McClish, 1989
        <https://www.ncbi.nlm.nih.gov/pubmed/2668680>`_

.. [3] Provost, F., Domingos, P. (2000). Well-trained PETs: Improving
       probability estimation trees (Section 6.2), CeDER Working Paper
       #IS-00-04, Stern School of Business, New York University.

.. [4] `Fawcett, T. (2006). An introduction to ROC analysis. Pattern
        Recognition Letters, 27(8), 861-874.
        <https://www.sciencedirect.com/science/article/pii/S016786550500303X>`_

.. [5] `Hand, D.J., Till, R.J. (2001). A Simple Generalisation of the Area
        Under the ROC Curve for Multiple Class Classification Problems.
        Machine Learning, 45(2), 171-186.
        <http://link.springer.com/article/10.1023/A:1010920819831>`_

See also
--------
average_precision_score : Area under the precision-recall curve

roc_curve : Compute Receiver operating characteristic (ROC) curve

Examples
--------
>>> import numpy as np
>>> from sklearn.metrics import roc_auc_score
>>> y_true = np.array([0, 0, 1, 1])
>>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
>>> roc_auc_score(y_true, y_scores)
0.75
*)

val roc_curve : ?pos_label:[`Int of int | `String of string] -> ?sample_weight:Ndarray.t -> ?drop_intermediate:bool -> y_true:Ndarray.t -> y_score:Ndarray.t -> unit -> (Ndarray.t * Ndarray.t * Ndarray.t)
(**
Compute Receiver operating characteristic (ROC)

Note: this implementation is restricted to the binary classification task.

Read more in the :ref:`User Guide <roc_metrics>`.

Parameters
----------

y_true : array, shape = [n_samples]
    True binary labels. If labels are not either {-1, 1} or {0, 1}, then
    pos_label should be explicitly given.

y_score : array, shape = [n_samples]
    Target scores, can either be probability estimates of the positive
    class, confidence values, or non-thresholded measure of decisions
    (as returned by "decision_function" on some classifiers).

pos_label : int or str, default=None
    The label of the positive class.
    When ``pos_label=None``, if y_true is in {-1, 1} or {0, 1},
    ``pos_label`` is set to 1, otherwise an error will be raised.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

drop_intermediate : boolean, optional (default=True)
    Whether to drop some suboptimal thresholds which would not appear
    on a plotted ROC curve. This is useful in order to create lighter
    ROC curves.

    .. versionadded:: 0.17
       parameter *drop_intermediate*.

Returns
-------
fpr : array, shape = [>2]
    Increasing false positive rates such that element i is the false
    positive rate of predictions with score >= thresholds[i].

tpr : array, shape = [>2]
    Increasing true positive rates such that element i is the true
    positive rate of predictions with score >= thresholds[i].

thresholds : array, shape = [n_thresholds]
    Decreasing thresholds on the decision function used to compute
    fpr and tpr. `thresholds[0]` represents no instances being predicted
    and is arbitrarily set to `max(y_score) + 1`.

See also
--------
roc_auc_score : Compute the area under the ROC curve

Notes
-----
Since the thresholds are sorted from low to high values, they
are reversed upon returning them to ensure they correspond to both ``fpr``
and ``tpr``, which are sorted in reversed order during their calculation.

References
----------
.. [1] `Wikipedia entry for the Receiver operating characteristic
        <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_

.. [2] Fawcett T. An introduction to ROC analysis[J]. Pattern Recognition
       Letters, 2006, 27(8):861-874.

Examples
--------
>>> import numpy as np
>>> from sklearn import metrics
>>> y = np.array([1, 1, 2, 2])
>>> scores = np.array([0.1, 0.4, 0.35, 0.8])
>>> fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
>>> fpr
array([0. , 0. , 0.5, 0.5, 1. ])
>>> tpr
array([0. , 0.5, 0.5, 1. , 1. ])
>>> thresholds
array([1.8 , 0.8 , 0.4 , 0.35, 0.1 ])
*)

val silhouette_samples : ?metric:[`String of string | `Callable of Py.Object.t] -> ?kwds:(string * Py.Object.t) list -> x:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> labels:Ndarray.t -> unit -> Ndarray.t
(**
Compute the Silhouette Coefficient for each sample.

The Silhouette Coefficient is a measure of how well samples are clustered
with samples that are similar to themselves. Clustering models with a high
Silhouette Coefficient are said to be dense, where samples in the same
cluster are similar to each other, and well separated, where samples in
different clusters are not very similar to each other.

The Silhouette Coefficient is calculated using the mean intra-cluster
distance (``a``) and the mean nearest-cluster distance (``b``) for each
sample.  The Silhouette Coefficient for a sample is ``(b - a) / max(a,
b)``.
Note that Silhouette Coefficient is only defined if number of labels
is 2 <= n_labels <= n_samples - 1.

This function returns the Silhouette Coefficient for each sample.

The best value is 1 and the worst value is -1. Values near 0 indicate
overlapping clusters.

Read more in the :ref:`User Guide <silhouette_coefficient>`.

Parameters
----------
X : array [n_samples_a, n_samples_a] if metric == "precomputed", or,              [n_samples_a, n_features] otherwise
    Array of pairwise distances between samples, or a feature array.

labels : array, shape = [n_samples]
         label values for each sample

metric : string, or callable
    The metric to use when calculating distance between instances in a
    feature array. If metric is a string, it must be one of the options
    allowed by :func:`sklearn.metrics.pairwise.pairwise_distances`. If X is
    the distance array itself, use "precomputed" as the metric. Precomputed
    distance matrices must have 0 along the diagonal.

`**kwds` : optional keyword parameters
    Any further parameters are passed directly to the distance function.
    If using a ``scipy.spatial.distance`` metric, the parameters are still
    metric dependent. See the scipy docs for usage examples.

Returns
-------
silhouette : array, shape = [n_samples]
    Silhouette Coefficient for each samples.

References
----------

.. [1] `Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
   Interpretation and Validation of Cluster Analysis". Computational
   and Applied Mathematics 20: 53-65.
   <https://www.sciencedirect.com/science/article/pii/0377042787901257>`_

.. [2] `Wikipedia entry on the Silhouette Coefficient
   <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_
*)

val silhouette_score : ?metric:[`String of string | `Callable of Py.Object.t] -> ?sample_size:[`Int of int | `None] -> ?random_state:[`Int of int | `RandomState of Py.Object.t | `None] -> ?kwds:(string * Py.Object.t) list -> x:[`Ndarray of Ndarray.t | `PyObject of Py.Object.t] -> labels:Ndarray.t -> unit -> float
(**
Compute the mean Silhouette Coefficient of all samples.

The Silhouette Coefficient is calculated using the mean intra-cluster
distance (``a``) and the mean nearest-cluster distance (``b``) for each
sample.  The Silhouette Coefficient for a sample is ``(b - a) / max(a,
b)``.  To clarify, ``b`` is the distance between a sample and the nearest
cluster that the sample is not a part of.
Note that Silhouette Coefficient is only defined if number of labels
is 2 <= n_labels <= n_samples - 1.

This function returns the mean Silhouette Coefficient over all samples.
To obtain the values for each sample, use :func:`silhouette_samples`.

The best value is 1 and the worst value is -1. Values near 0 indicate
overlapping clusters. Negative values generally indicate that a sample has
been assigned to the wrong cluster, as a different cluster is more similar.

Read more in the :ref:`User Guide <silhouette_coefficient>`.

Parameters
----------
X : array [n_samples_a, n_samples_a] if metric == "precomputed", or,              [n_samples_a, n_features] otherwise
    Array of pairwise distances between samples, or a feature array.

labels : array, shape = [n_samples]
     Predicted labels for each sample.

metric : string, or callable
    The metric to use when calculating distance between instances in a
    feature array. If metric is a string, it must be one of the options
    allowed by :func:`metrics.pairwise.pairwise_distances
    <sklearn.metrics.pairwise.pairwise_distances>`. If X is the distance
    array itself, use ``metric="precomputed"``.

sample_size : int or None
    The size of the sample to use when computing the Silhouette Coefficient
    on a random subset of the data.
    If ``sample_size is None``, no sampling is used.

random_state : int, RandomState instance or None, optional (default=None)
    The generator used to randomly select a subset of samples.  If int,
    random_state is the seed used by the random number generator; If
    RandomState instance, random_state is the random number generator; If
    None, the random number generator is the RandomState instance used by
    `np.random`. Used when ``sample_size is not None``.

**kwds : optional keyword parameters
    Any further parameters are passed directly to the distance function.
    If using a scipy.spatial.distance metric, the parameters are still
    metric dependent. See the scipy docs for usage examples.

Returns
-------
silhouette : float
    Mean Silhouette Coefficient for all samples.

References
----------

.. [1] `Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
   Interpretation and Validation of Cluster Analysis". Computational
   and Applied Mathematics 20: 53-65.
   <https://www.sciencedirect.com/science/article/pii/0377042787901257>`_

.. [2] `Wikipedia entry on the Silhouette Coefficient
       <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_
*)

val v_measure_score : ?beta:float -> labels_true:Py.Object.t -> labels_pred:Ndarray.t -> unit -> float
(**
V-measure cluster labeling given a ground truth.

This score is identical to :func:`normalized_mutual_info_score` with
the ``'arithmetic'`` option for averaging.

The V-measure is the harmonic mean between homogeneity and completeness::

    v = (1 + beta) * homogeneity * completeness
         / (beta * homogeneity + completeness)

This metric is independent of the absolute values of the labels:
a permutation of the class or cluster label values won't change the
score value in any way.

This metric is furthermore symmetric: switching ``label_true`` with
``label_pred`` will return the same score value. This can be useful to
measure the agreement of two independent label assignments strategies
on the same dataset when the real ground truth is not known.


Read more in the :ref:`User Guide <homogeneity_completeness>`.

Parameters
----------
labels_true : int array, shape = [n_samples]
    ground truth class labels to be used as a reference

labels_pred : array-like of shape (n_samples,)
    cluster labels to evaluate

beta : float
    Ratio of weight attributed to ``homogeneity`` vs ``completeness``.
    If ``beta`` is greater than 1, ``completeness`` is weighted more
    strongly in the calculation. If ``beta`` is less than 1,
    ``homogeneity`` is weighted more strongly.

Returns
-------
v_measure : float
   score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling

References
----------

.. [1] `Andrew Rosenberg and Julia Hirschberg, 2007. V-Measure: A
   conditional entropy-based external cluster evaluation measure
   <https://aclweb.org/anthology/D/D07/D07-1043.pdf>`_

See also
--------
homogeneity_score
completeness_score
normalized_mutual_info_score

Examples
--------

Perfect labelings are both homogeneous and complete, hence have score 1.0::

  >>> from sklearn.metrics.cluster import v_measure_score
  >>> v_measure_score([0, 0, 1, 1], [0, 0, 1, 1])
  1.0
  >>> v_measure_score([0, 0, 1, 1], [1, 1, 0, 0])
  1.0

Labelings that assign all classes members to the same clusters
are complete be not homogeneous, hence penalized::

  >>> print("%.6f" % v_measure_score([0, 0, 1, 2], [0, 0, 1, 1]))
  0.8...
  >>> print("%.6f" % v_measure_score([0, 1, 2, 3], [0, 0, 1, 1]))
  0.66...

Labelings that have pure clusters with members coming from the same
classes are homogeneous but un-necessary splits harms completeness
and thus penalize V-measure as well::

  >>> print("%.6f" % v_measure_score([0, 0, 1, 1], [0, 0, 1, 2]))
  0.8...
  >>> print("%.6f" % v_measure_score([0, 0, 1, 1], [0, 1, 2, 3]))
  0.66...

If classes members are completely split across different clusters,
the assignment is totally incomplete, hence the V-Measure is null::

  >>> print("%.6f" % v_measure_score([0, 0, 0, 0], [0, 1, 2, 3]))
  0.0...

Clusters that include samples from totally different classes totally
destroy the homogeneity of the labeling, hence::

  >>> print("%.6f" % v_measure_score([0, 0, 1, 1], [0, 0, 0, 0]))
  0.0...
*)

val zero_one_loss : ?normalize:bool -> ?sample_weight:Ndarray.t -> y_true:Py.Object.t -> y_pred:Py.Object.t -> unit -> Py.Object.t
(**
Zero-one classification loss.

If normalize is ``True``, return the fraction of misclassifications
(float), else it returns the number of misclassifications (int). The best
performance is 0.

Read more in the :ref:`User Guide <zero_one_loss>`.

Parameters
----------
y_true : 1d array-like, or label indicator array / sparse matrix
    Ground truth (correct) labels.

y_pred : 1d array-like, or label indicator array / sparse matrix
    Predicted labels, as returned by a classifier.

normalize : bool, optional (default=True)
    If ``False``, return the number of misclassifications.
    Otherwise, return the fraction of misclassifications.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

Returns
-------
loss : float or int,
    If ``normalize == True``, return the fraction of misclassifications
    (float), else it returns the number of misclassifications (int).

Notes
-----
In multilabel classification, the zero_one_loss function corresponds to
the subset zero-one loss: for each sample, the entire set of labels must be
correctly predicted, otherwise the loss for that sample is equal to one.

See also
--------
accuracy_score, hamming_loss, jaccard_score

Examples
--------
>>> from sklearn.metrics import zero_one_loss
>>> y_pred = [1, 2, 3, 4]
>>> y_true = [2, 2, 3, 4]
>>> zero_one_loss(y_true, y_pred)
0.25
>>> zero_one_loss(y_true, y_pred, normalize=False)
1

In the multilabel case with binary label indicators:

>>> import numpy as np
>>> zero_one_loss(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))
0.5
*)

