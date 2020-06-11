(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module LabelBinarizer : sig
type tag = [`LabelBinarizer]
type t = [`BaseEstimator | `LabelBinarizer | `Object | `TransformerMixin] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_transformer : t -> [`TransformerMixin] Obj.t
val as_estimator : t -> [`BaseEstimator] Obj.t
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

val fit : y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
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

val fit_transform : y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
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

val inverse_transform : ?threshold:float -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
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

val transform : y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
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
val classes_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute classes_: get value as an option. *)
val classes_opt : t -> ([>`ArrayLike] Np.Obj.t) option


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

module OneVsOneClassifier : sig
type tag = [`OneVsOneClassifier]
type t = [`BaseEstimator | `ClassifierMixin | `MetaEstimatorMixin | `Object | `OneVsOneClassifier] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_classifier : t -> [`ClassifierMixin] Obj.t
val as_estimator : t -> [`BaseEstimator] Obj.t
val as_meta_estimator : t -> [`MetaEstimatorMixin] Obj.t
val create : ?n_jobs:int -> estimator:[>`BaseEstimator] Np.Obj.t -> unit -> t
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

val decision_function : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
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

val fit : x:[>`Spmatrix] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
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

val partial_fit : ?classes:[>`ArrayLike] Np.Obj.t -> x:[>`Spmatrix] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
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

val predict : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
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


(** Attribute estimators_: get value or raise Not_found if None.*)
val estimators_ : t -> Py.Object.t

(** Attribute estimators_: get value as an option. *)
val estimators_opt : t -> (Py.Object.t) option


(** Attribute classes_: get value or raise Not_found if None.*)
val classes_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute classes_: get value as an option. *)
val classes_opt : t -> ([>`ArrayLike] Np.Obj.t) option


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
type tag = [`OneVsRestClassifier]
type t = [`BaseEstimator | `ClassifierMixin | `MetaEstimatorMixin | `MultiOutputMixin | `Object | `OneVsRestClassifier] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_classifier : t -> [`ClassifierMixin] Obj.t
val as_multi_output : t -> [`MultiOutputMixin] Obj.t
val as_estimator : t -> [`BaseEstimator] Obj.t
val as_meta_estimator : t -> [`MetaEstimatorMixin] Obj.t
val create : ?n_jobs:int -> estimator:[>`BaseEstimator] Np.Obj.t -> unit -> t
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

val decision_function : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
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

val fit : x:[>`Spmatrix] Np.Obj.t -> y:[>`Spmatrix] Np.Obj.t -> [> tag] Obj.t -> t
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

val partial_fit : ?classes:[>`ArrayLike] Np.Obj.t -> x:[>`Spmatrix] Np.Obj.t -> y:[>`Spmatrix] Np.Obj.t -> [> tag] Obj.t -> t
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

val predict : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
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

val predict_proba : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
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


(** Attribute estimators_: get value or raise Not_found if None.*)
val estimators_ : t -> Py.Object.t

(** Attribute estimators_: get value as an option. *)
val estimators_opt : t -> (Py.Object.t) option


(** Attribute classes_: get value or raise Not_found if None.*)
val classes_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute classes_: get value as an option. *)
val classes_opt : t -> ([>`ArrayLike] Np.Obj.t) option


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
type tag = [`OutputCodeClassifier]
type t = [`BaseEstimator | `ClassifierMixin | `MetaEstimatorMixin | `Object | `OutputCodeClassifier] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_classifier : t -> [`ClassifierMixin] Obj.t
val as_estimator : t -> [`BaseEstimator] Obj.t
val as_meta_estimator : t -> [`MetaEstimatorMixin] Obj.t
val create : ?code_size:float -> ?random_state:int -> ?n_jobs:int -> estimator:[>`BaseEstimator] Np.Obj.t -> unit -> t
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

.. [1] 'Solving multiclass learning problems via error-correcting output
   codes',
   Dietterich T., Bakiri G.,
   Journal of Artificial Intelligence Research 2,
   1995.

.. [2] 'The error coding method and PICTs',
   James G., Hastie T.,
   Journal of Computational and Graphical statistics 7,
   1998.

.. [3] 'The Elements of Statistical Learning',
   Hastie T., Tibshirani R., Friedman J., page 606 (second-edition)
   2008.
*)

val fit : x:[>`Spmatrix] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
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


(** Attribute estimators_: get value or raise Not_found if None.*)
val estimators_ : t -> Py.Object.t

(** Attribute estimators_: get value as an option. *)
val estimators_opt : t -> (Py.Object.t) option


(** Attribute classes_: get value or raise Not_found if None.*)
val classes_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute classes_: get value as an option. *)
val classes_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute code_book_: get value or raise Not_found if None.*)
val code_book_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute code_book_: get value as an option. *)
val code_book_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

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

val delayed : ?check_pickle:Py.Object.t -> function_:Py.Object.t -> unit -> Py.Object.t
(**
Decorator used to capture the arguments of a function.
*)

val euclidean_distances : ?y:[>`ArrayLike] Np.Obj.t -> ?y_norm_squared:[>`ArrayLike] Np.Obj.t -> ?squared:bool -> ?x_norm_squared:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> unit -> [>`ArrayLike] Np.Obj.t
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

val is_regressor : [>`BaseEstimator] Np.Obj.t -> bool
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

