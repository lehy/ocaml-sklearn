(** Get an attribute of this module as a Py.Object.t. This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module BaseEstimator : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : unit -> t
(**
Base class for all estimators in scikit-learn

Notes
-----
All estimators should specify all the parameters that can be set
at the class level in their ``__init__`` as explicit keyword
arguments (no ``*args`` or ``**kwargs``).
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


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module ClassifierMixin : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : unit -> t
(**
Mixin class for all classifiers in scikit-learn.
*)

val score : ?sample_weight:Arr.t -> x:Arr.t -> y:Arr.t -> t -> float
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


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module LabelBinarizer : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?neg_label:int -> ?pos_label:int -> ?sparse_output:bool -> unit -> t
(**
Binarize labels in a one-vs-all fashion

Several regression and binary classification algorithms are
available in scikit-learn. A simple way to extend these algorithms
to the multi-class classification case is to use the so-called
one-vs-all scheme.

At learning time, this simply consists in learning one regressor
or binary classifier per class. In doing so, one needs to convert
multi-class labels to binary labels (belong or does not belong
to the class). LabelBinarizer makes this process easy with the
transform method.

At prediction time, one assigns the class for which the corresponding
model gave the greatest confidence. LabelBinarizer makes this easy
with the inverse_transform method.

Read more in the :ref:`User Guide <preprocessing_targets>`.

Parameters
----------

neg_label : int (default: 0)
    Value with which negative labels must be encoded.

pos_label : int (default: 1)
    Value with which positive labels must be encoded.

sparse_output : boolean (default: False)
    True if the returned array from transform is desired to be in sparse
    CSR format.

Attributes
----------

classes_ : array of shape [n_class]
    Holds the label for each class.

y_type_ : str,
    Represents the type of the target data as evaluated by
    utils.multiclass.type_of_target. Possible type are 'continuous',
    'continuous-multioutput', 'binary', 'multiclass',
    'multiclass-multioutput', 'multilabel-indicator', and 'unknown'.

sparse_input_ : boolean,
    True if the input data to transform is given as a sparse matrix, False
    otherwise.

Examples
--------
>>> from sklearn import preprocessing
>>> lb = preprocessing.LabelBinarizer()
>>> lb.fit([1, 2, 6, 4, 2])
LabelBinarizer()
>>> lb.classes_
array([1, 2, 4, 6])
>>> lb.transform([1, 6])
array([[1, 0, 0, 0],
       [0, 0, 0, 1]])

Binary targets transform to a column vector

>>> lb = preprocessing.LabelBinarizer()
>>> lb.fit_transform(['yes', 'no', 'no', 'yes'])
array([[1],
       [0],
       [0],
       [1]])

Passing a 2D matrix for multilabel classification

>>> import numpy as np
>>> lb.fit(np.array([[0, 1, 1], [1, 0, 0]]))
LabelBinarizer()
>>> lb.classes_
array([0, 1, 2])
>>> lb.transform([0, 1, 2, 1])
array([[1, 0, 0],
       [0, 1, 0],
       [0, 0, 1],
       [0, 1, 0]])

See also
--------
label_binarize : function to perform the transform operation of
    LabelBinarizer with fixed classes.
sklearn.preprocessing.OneHotEncoder : encode categorical features
    using a one-hot aka one-of-K scheme.
*)

val fit : y:Arr.t -> t -> t
(**
Fit label binarizer

Parameters
----------
y : array of shape [n_samples,] or [n_samples, n_classes]
    Target values. The 2-d matrix should only contain 0 and 1,
    represents multilabel classification.

Returns
-------
self : returns an instance of self.
*)

val fit_transform : y:Arr.t -> t -> Arr.t
(**
Fit label binarizer and transform multi-class labels to binary
labels.

The output of transform is sometimes referred to as
the 1-of-K coding scheme.

Parameters
----------
y : array or sparse matrix of shape [n_samples,] or             [n_samples, n_classes]
    Target values. The 2-d matrix should only contain 0 and 1,
    represents multilabel classification. Sparse matrix can be
    CSR, CSC, COO, DOK, or LIL.

Returns
-------
Y : array or CSR matrix of shape [n_samples, n_classes]
    Shape will be [n_samples, 1] for binary problems.
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

val inverse_transform : ?threshold:float -> y:Arr.t -> t -> Py.Object.t
(**
Transform binary labels back to multi-class labels

Parameters
----------
Y : numpy array or sparse matrix with shape [n_samples, n_classes]
    Target values. All sparse matrices are converted to CSR before
    inverse transformation.

threshold : float or None
    Threshold used in the binary and multi-label cases.

    Use 0 when ``Y`` contains the output of decision_function
    (classifier).
    Use 0.5 when ``Y`` contains the output of predict_proba.

    If None, the threshold is assumed to be half way between
    neg_label and pos_label.

Returns
-------
y : numpy array or CSR matrix of shape [n_samples] Target values.

Notes
-----
In the case when the binary labels are fractional
(probabilistic), inverse_transform chooses the class with the
greatest value. Typically, this allows to use the output of a
linear model's decision_function method directly as the input
of inverse_transform.
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

val transform : y:Arr.t -> t -> Arr.t
(**
Transform multi-class labels to binary labels

The output of transform is sometimes referred to by some authors as
the 1-of-K coding scheme.

Parameters
----------
y : array or sparse matrix of shape [n_samples,] or             [n_samples, n_classes]
    Target values. The 2-d matrix should only contain 0 and 1,
    represents multilabel classification. Sparse matrix can be
    CSR, CSC, COO, DOK, or LIL.

Returns
-------
Y : numpy array or CSR matrix of shape [n_samples, n_classes]
    Shape will be [n_samples, 1] for binary problems.
*)


(** Attribute classes_: get value or raise Not_found if None.*)
val classes_ : t -> Arr.t

(** Attribute classes_: get value as an option. *)
val classes_opt : t -> (Arr.t) option


(** Attribute y_type_: get value or raise Not_found if None.*)
val y_type_ : t -> string

(** Attribute y_type_: get value as an option. *)
val y_type_opt : t -> (string) option


(** Attribute sparse_input_: get value or raise Not_found if None.*)
val sparse_input_ : t -> bool

(** Attribute sparse_input_: get value as an option. *)
val sparse_input_opt : t -> (bool) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MetaEstimatorMixin : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : unit -> t
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

module MultiOutputMixin : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : unit -> t
(**
Mixin to mark estimators that support multioutput.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module OneVsOneClassifier : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_jobs:int -> estimator:Py.Object.t -> unit -> t
(**
One-vs-one multiclass strategy

This strategy consists in fitting one classifier per class pair.
At prediction time, the class which received the most votes is selected.
Since it requires to fit `n_classes * (n_classes - 1) / 2` classifiers,
this method is usually slower than one-vs-the-rest, due to its
O(n_classes^2) complexity. However, this method may be advantageous for
algorithms such as kernel algorithms which don't scale well with
`n_samples`. This is because each individual learning problem only involves
a small subset of the data whereas, with one-vs-the-rest, the complete
dataset is used `n_classes` times.

Read more in the :ref:`User Guide <ovo_classification>`.

Parameters
----------
estimator : estimator object
    An estimator object implementing :term:`fit` and one of
    :term:`decision_function` or :term:`predict_proba`.

n_jobs : int or None, optional (default=None)
    The number of jobs to use for the computation.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

Attributes
----------
estimators_ : list of ``n_classes * (n_classes - 1) / 2`` estimators
    Estimators used for predictions.

classes_ : numpy array of shape [n_classes]
    Array containing labels.

n_classes_ : int
    Number of classes

pairwise_indices_ : list, length = ``len(estimators_)``, or ``None``
    Indices of samples used when training the estimators.
    ``None`` when ``estimator`` does not have ``_pairwise`` attribute.
*)

val decision_function : x:Arr.t -> t -> Arr.t
(**
Decision function for the OneVsOneClassifier.

The decision values for the samples are computed by adding the
normalized sum of pair-wise classification confidence levels to the
votes in order to disambiguate between the decision values when the
votes for all the classes are equal leading to a tie.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
Y : array-like of shape (n_samples, n_classes)
*)

val fit : x:Arr.t -> y:Arr.t -> t -> t
(**
Fit underlying estimators.

Parameters
----------
X : (sparse) array-like of shape (n_samples, n_features)
    Data.

y : array-like of shape (n_samples,)
    Multi-class targets.

Returns
-------
self
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

val partial_fit : ?classes:Arr.t -> x:Arr.t -> y:Arr.t -> t -> Py.Object.t
(**
Partially fit underlying estimators

Should be used when memory is inefficient to train all data. Chunks
of data can be passed in several iteration, where the first call
should have an array of all target variables.


Parameters
----------
X : (sparse) array-like of shape (n_samples, n_features)
    Data.

y : array-like of shape (n_samples,)
    Multi-class targets.

classes : array, shape (n_classes, )
    Classes across all calls to partial_fit.
    Can be obtained via `np.unique(y_all)`, where y_all is the
    target vector of the entire dataset.
    This argument is only required in the first call of partial_fit
    and can be omitted in the subsequent calls.

Returns
-------
self
*)

val predict : x:Arr.t -> t -> Arr.t
(**
Estimate the best class label for each sample in X.

This is implemented as ``argmax(decision_function(X), axis=1)`` which
will return the label of the class with most votes by estimators
predicting the outcome of a decision for each possible class pair.

Parameters
----------
X : (sparse) array-like of shape (n_samples, n_features)
    Data.

Returns
-------
y : numpy array of shape [n_samples]
    Predicted multi-class targets.
*)

val score : ?sample_weight:Arr.t -> x:Arr.t -> y:Arr.t -> t -> float
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


(** Attribute estimators_: get value or raise Not_found if None.*)
val estimators_ : t -> Py.Object.t

(** Attribute estimators_: get value as an option. *)
val estimators_opt : t -> (Py.Object.t) option


(** Attribute classes_: get value or raise Not_found if None.*)
val classes_ : t -> Arr.t

(** Attribute classes_: get value as an option. *)
val classes_opt : t -> (Arr.t) option


(** Attribute n_classes_: get value or raise Not_found if None.*)
val n_classes_ : t -> int

(** Attribute n_classes_: get value as an option. *)
val n_classes_opt : t -> (int) option


(** Attribute pairwise_indices_: get value or raise Not_found if None.*)
val pairwise_indices_ : t -> Py.Object.t

(** Attribute pairwise_indices_: get value as an option. *)
val pairwise_indices_opt : t -> (Py.Object.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module OneVsRestClassifier : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_jobs:int -> estimator:Py.Object.t -> unit -> t
(**
One-vs-the-rest (OvR) multiclass/multilabel strategy

Also known as one-vs-all, this strategy consists in fitting one classifier
per class. For each classifier, the class is fitted against all the other
classes. In addition to its computational efficiency (only `n_classes`
classifiers are needed), one advantage of this approach is its
interpretability. Since each class is represented by one and one classifier
only, it is possible to gain knowledge about the class by inspecting its
corresponding classifier. This is the most commonly used strategy for
multiclass classification and is a fair default choice.

This strategy can also be used for multilabel learning, where a classifier
is used to predict multiple labels for instance, by fitting on a 2-d matrix
in which cell [i, j] is 1 if sample i has label j and 0 otherwise.

In the multilabel learning literature, OvR is also known as the binary
relevance method.

Read more in the :ref:`User Guide <ovr_classification>`.

Parameters
----------
estimator : estimator object
    An estimator object implementing :term:`fit` and one of
    :term:`decision_function` or :term:`predict_proba`.

n_jobs : int or None, optional (default=None)
    The number of jobs to use for the computation.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

Attributes
----------
estimators_ : list of `n_classes` estimators
    Estimators used for predictions.

classes_ : array, shape = [`n_classes`]
    Class labels.

n_classes_ : int
    Number of classes.

label_binarizer_ : LabelBinarizer object
    Object used to transform multiclass labels to binary labels and
    vice-versa.

multilabel_ : boolean
    Whether a OneVsRestClassifier is a multilabel classifier.

Examples
--------
>>> import numpy as np
>>> from sklearn.multiclass import OneVsRestClassifier
>>> from sklearn.svm import SVC
>>> X = np.array([
...     [10, 10],
...     [8, 10],
...     [-5, 5.5],
...     [-5.4, 5.5],
...     [-20, -20],
...     [-15, -20]
... ])
>>> y = np.array([0, 0, 1, 1, 2, 2])
>>> clf = OneVsRestClassifier(SVC()).fit(X, y)
>>> clf.predict([[-19, -20], [9, 9], [-5, 5]])
array([2, 0, 1])
*)

val decision_function : x:Arr.t -> t -> Arr.t
(**
Returns the distance of each sample from the decision boundary for
each class. This can only be used with estimators which implement the
decision_function method.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
T : array-like of shape (n_samples, n_classes)
*)

val fit : x:Arr.t -> y:Arr.t -> t -> t
(**
Fit underlying estimators.

Parameters
----------
X : (sparse) array-like of shape (n_samples, n_features)
    Data.

y : (sparse) array-like of shape (n_samples,) or (n_samples, n_classes)
    Multi-class targets. An indicator matrix turns on multilabel
    classification.

Returns
-------
self
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

val partial_fit : ?classes:Arr.t -> x:Arr.t -> y:Arr.t -> t -> Py.Object.t
(**
Partially fit underlying estimators

Should be used when memory is inefficient to train all data.
Chunks of data can be passed in several iteration.

Parameters
----------
X : (sparse) array-like of shape (n_samples, n_features)
    Data.

y : (sparse) array-like of shape (n_samples,) or (n_samples, n_classes)
    Multi-class targets. An indicator matrix turns on multilabel
    classification.

classes : array, shape (n_classes, )
    Classes across all calls to partial_fit.
    Can be obtained via `np.unique(y_all)`, where y_all is the
    target vector of the entire dataset.
    This argument is only required in the first call of partial_fit
    and can be omitted in the subsequent calls.

Returns
-------
self
*)

val predict : x:Arr.t -> t -> Arr.t
(**
Predict multi-class targets using underlying estimators.

Parameters
----------
X : (sparse) array-like of shape (n_samples, n_features)
    Data.

Returns
-------
y : (sparse) array-like of shape (n_samples,) or (n_samples, n_classes)
    Predicted multi-class targets.
*)

val predict_proba : x:Arr.t -> t -> Arr.t
(**
Probability estimates.

The returned estimates for all classes are ordered by label of classes.

Note that in the multilabel case, each sample can have any number of
labels. This returns the marginal probability that the given sample has
the label in question. For example, it is entirely consistent that two
labels both have a 90% probability of applying to a given sample.

In the single label multiclass case, the rows of the returned matrix
sum to 1.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
T : (sparse) array-like of shape (n_samples, n_classes)
    Returns the probability of the sample for each class in the model,
    where classes are ordered as they are in `self.classes_`.
*)

val score : ?sample_weight:Arr.t -> x:Arr.t -> y:Arr.t -> t -> float
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


(** Attribute estimators_: get value or raise Not_found if None.*)
val estimators_ : t -> Py.Object.t

(** Attribute estimators_: get value as an option. *)
val estimators_opt : t -> (Py.Object.t) option


(** Attribute classes_: get value or raise Not_found if None.*)
val classes_ : t -> Arr.t

(** Attribute classes_: get value as an option. *)
val classes_opt : t -> (Arr.t) option


(** Attribute n_classes_: get value or raise Not_found if None.*)
val n_classes_ : t -> int

(** Attribute n_classes_: get value as an option. *)
val n_classes_opt : t -> (int) option


(** Attribute label_binarizer_: get value or raise Not_found if None.*)
val label_binarizer_ : t -> Py.Object.t

(** Attribute label_binarizer_: get value as an option. *)
val label_binarizer_opt : t -> (Py.Object.t) option


(** Attribute multilabel_: get value or raise Not_found if None.*)
val multilabel_ : t -> bool

(** Attribute multilabel_: get value as an option. *)
val multilabel_opt : t -> (bool) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module OutputCodeClassifier : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?code_size:float -> ?random_state:int -> ?n_jobs:int -> estimator:Py.Object.t -> unit -> t
(**
(Error-Correcting) Output-Code multiclass strategy

Output-code based strategies consist in representing each class with a
binary code (an array of 0s and 1s). At fitting time, one binary
classifier per bit in the code book is fitted.  At prediction time, the
classifiers are used to project new points in the class space and the class
closest to the points is chosen. The main advantage of these strategies is
that the number of classifiers used can be controlled by the user, either
for compressing the model (0 < code_size < 1) or for making the model more
robust to errors (code_size > 1). See the documentation for more details.

Read more in the :ref:`User Guide <ecoc>`.

Parameters
----------
estimator : estimator object
    An estimator object implementing :term:`fit` and one of
    :term:`decision_function` or :term:`predict_proba`.

code_size : float
    Percentage of the number of classes to be used to create the code book.
    A number between 0 and 1 will require fewer classifiers than
    one-vs-the-rest. A number greater than 1 will require more classifiers
    than one-vs-the-rest.

random_state : int, RandomState instance or None, optional, default: None
    The generator used to initialize the codebook.  If int, random_state is
    the seed used by the random number generator; If RandomState instance,
    random_state is the random number generator; If None, the random number
    generator is the RandomState instance used by `np.random`.

n_jobs : int or None, optional (default=None)
    The number of jobs to use for the computation.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

Attributes
----------
estimators_ : list of `int(n_classes * code_size)` estimators
    Estimators used for predictions.

classes_ : numpy array of shape [n_classes]
    Array containing labels.

code_book_ : numpy array of shape [n_classes, code_size]
    Binary array containing the code of each class.

Examples
--------
>>> from sklearn.multiclass import OutputCodeClassifier
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.datasets import make_classification
>>> X, y = make_classification(n_samples=100, n_features=4,
...                            n_informative=2, n_redundant=0,
...                            random_state=0, shuffle=False)
>>> clf = OutputCodeClassifier(
...     estimator=RandomForestClassifier(random_state=0),
...     random_state=0).fit(X, y)
>>> clf.predict([[0, 0, 0, 0]])
array([1])

References
----------

.. [1] "Solving multiclass learning problems via error-correcting output
   codes",
   Dietterich T., Bakiri G.,
   Journal of Artificial Intelligence Research 2,
   1995.

.. [2] "The error coding method and PICTs",
   James G., Hastie T.,
   Journal of Computational and Graphical statistics 7,
   1998.

.. [3] "The Elements of Statistical Learning",
   Hastie T., Tibshirani R., Friedman J., page 606 (second-edition)
   2008.
*)

val fit : x:Arr.t -> y:Arr.t -> t -> t
(**
Fit underlying estimators.

Parameters
----------
X : (sparse) array-like of shape (n_samples, n_features)
    Data.

y : numpy array of shape [n_samples]
    Multi-class targets.

Returns
-------
self
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

val predict : x:Arr.t -> t -> Arr.t
(**
Predict multi-class targets using underlying estimators.

Parameters
----------
X : (sparse) array-like of shape (n_samples, n_features)
    Data.

Returns
-------
y : numpy array of shape [n_samples]
    Predicted multi-class targets.
*)

val score : ?sample_weight:Arr.t -> x:Arr.t -> y:Arr.t -> t -> float
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


(** Attribute estimators_: get value or raise Not_found if None.*)
val estimators_ : t -> Py.Object.t

(** Attribute estimators_: get value as an option. *)
val estimators_opt : t -> (Py.Object.t) option


(** Attribute classes_: get value or raise Not_found if None.*)
val classes_ : t -> Arr.t

(** Attribute classes_: get value as an option. *)
val classes_opt : t -> (Arr.t) option


(** Attribute code_book_: get value or raise Not_found if None.*)
val code_book_ : t -> Arr.t

(** Attribute code_book_: get value as an option. *)
val code_book_opt : t -> (Arr.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Parallel : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_jobs:int -> ?backend:Py.Object.t -> ?verbose:int -> ?timeout:float -> ?pre_dispatch:[`All | `I of int | `PyObject of Py.Object.t] -> ?batch_size:[`I of int | `Auto] -> ?temp_folder:string -> ?max_nbytes:Py.Object.t -> ?mmap_mode:[`R_ | `R | `W_ | `C | `None] -> ?prefer:[`Processes | `Threads] -> ?require:string -> unit -> t
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
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val check_X_y : ?accept_sparse:[`S of string | `Bool of bool | `StringList of string list] -> ?accept_large_sparse:bool -> ?dtype:[`S of string | `Dtype of Py.Object.t | `TypeList of Py.Object.t | `None] -> ?order:[`F | `C] -> ?copy:bool -> ?force_all_finite:[`Bool of bool | `Allow_nan] -> ?ensure_2d:bool -> ?allow_nd:bool -> ?multi_output:bool -> ?ensure_min_samples:int -> ?ensure_min_features:int -> ?y_numeric:bool -> ?warn_on_dtype:bool -> ?estimator:[`S of string | `Estimator of Py.Object.t] -> x:Arr.t -> y:Arr.t -> unit -> (Py.Object.t * Py.Object.t)
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

dtype : string, type, list of types or None (default="numeric")
    Data type of result. If None, the dtype of the input is preserved.
    If "numeric", dtype is preserved unless array.dtype is object.
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

val check_array : ?accept_sparse:[`S of string | `Bool of bool | `StringList of string list] -> ?accept_large_sparse:bool -> ?dtype:[`S of string | `Dtype of Py.Object.t | `TypeList of Py.Object.t | `None] -> ?order:[`F | `C] -> ?copy:bool -> ?force_all_finite:[`Bool of bool | `Allow_nan] -> ?ensure_2d:bool -> ?allow_nd:bool -> ?ensure_min_samples:int -> ?ensure_min_features:int -> ?warn_on_dtype:bool -> ?estimator:[`S of string | `Estimator of Py.Object.t] -> array:Py.Object.t -> unit -> Py.Object.t
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

val check_classification_targets : y:Arr.t -> unit -> Py.Object.t
(**
Ensure that target y is of a non-regression type.

Only the following target types (as defined in type_of_target) are allowed:
    'binary', 'multiclass', 'multiclass-multioutput',
    'multilabel-indicator', 'multilabel-sequences'

Parameters
----------
y : array-like
*)

val check_is_fitted : ?attributes:[`S of string | `Arr of Arr.t | `StringList of string list] -> ?msg:string -> ?all_or_any:[`Callable of Py.Object.t | `PyObject of Py.Object.t] -> estimator:Py.Object.t -> unit -> Py.Object.t
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
    Eg.: ``["coef_", "estimator_", ...], "coef_"``

    If `None`, `estimator` is considered fitted if there exist an
    attribute that ends with a underscore and does not start with double
    underscore.

msg : string
    The default error message is, "This %(name)s instance is not fitted
    yet. Call 'fit' with appropriate arguments before using this
    estimator."

    For custom messages if "%(name)s" is present in the message string,
    it is substituted for the estimator name.

    Eg. : "Estimator, %(name)s, must be fitted before sparsifying".

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

val check_random_state : seed:[`I of int | `RandomState of Py.Object.t | `None] -> unit -> Py.Object.t
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

val clone : ?safe:bool -> estimator:[`Estimator of Py.Object.t | `Arr of Arr.t | `PyObject of Py.Object.t] -> unit -> Py.Object.t
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

val delayed : ?check_pickle:Py.Object.t -> function_:Py.Object.t -> unit -> Py.Object.t
(**
Decorator used to capture the arguments of a function.
*)

val euclidean_distances : ?y:Arr.t -> ?y_norm_squared:Arr.t -> ?squared:bool -> ?x_norm_squared:Arr.t -> x:Arr.t -> unit -> Arr.t
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

val if_delegate_has_method : delegate:[`S of string | `StringList of string list | `Tuple_of_strings of Py.Object.t] -> unit -> Py.Object.t
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

val is_classifier : estimator:Py.Object.t -> unit -> bool
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

val is_regressor : estimator:Py.Object.t -> unit -> bool
(**
Return True if the given estimator is (probably) a regressor.

Parameters
----------
estimator : object
    Estimator object to test.

Returns
-------
out : bool
    True if estimator is a regressor and False otherwise.
*)

