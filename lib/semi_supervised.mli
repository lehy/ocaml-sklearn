module LabelPropagation : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?kernel:[`Knn | `Rbf | `Callable of Py.Object.t] -> ?gamma:float -> ?n_neighbors:int -> ?max_iter:int -> ?tol:float -> ?n_jobs:[`Int of int | `None] -> unit -> t
(**
Label Propagation classifier

Read more in the :ref:`User Guide <label_propagation>`.

Parameters
----------
kernel : {'knn', 'rbf', callable}
    String identifier for kernel function to use or the kernel function
    itself. Only 'rbf' and 'knn' strings are valid inputs. The function
    passed should take two inputs, each of shape [n_samples, n_features],
    and return a [n_samples, n_samples] shaped weight matrix.

gamma : float
    Parameter for rbf kernel

n_neighbors : integer > 0
    Parameter for knn kernel

max_iter : integer
    Change maximum number of iterations allowed

tol : float
    Convergence tolerance: threshold to consider the system at steady
    state

n_jobs : int or None, optional (default=None)
    The number of parallel jobs to run.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

Attributes
----------
X_ : array, shape = [n_samples, n_features]
    Input array.

classes_ : array, shape = [n_classes]
    The distinct labels used in classifying instances.

label_distributions_ : array, shape = [n_samples, n_classes]
    Categorical distribution for each item.

transduction_ : array, shape = [n_samples]
    Label assigned to each item via the transduction.

n_iter_ : int
    Number of iterations run.

Examples
--------
>>> import numpy as np
>>> from sklearn import datasets
>>> from sklearn.semi_supervised import LabelPropagation
>>> label_prop_model = LabelPropagation()
>>> iris = datasets.load_iris()
>>> rng = np.random.RandomState(42)
>>> random_unlabeled_points = rng.rand(len(iris.target)) < 0.3
>>> labels = np.copy(iris.target)
>>> labels[random_unlabeled_points] = -1
>>> label_prop_model.fit(iris.data, labels)
LabelPropagation(...)

References
----------
Xiaojin Zhu and Zoubin Ghahramani. Learning from labeled and unlabeled data
with label propagation. Technical Report CMU-CALD-02-107, Carnegie Mellon
University, 2002 http://pages.cs.wisc.edu/~jerryzhu/pub/CMU-CALD-02-107.pdf

See Also
--------
LabelSpreading : Alternate label propagation strategy more robust to noise
*)

val fit : x:Py.Object.t -> y:Py.Object.t -> t -> t
(**
Fit a semi-supervised label propagation model based

All the input data is provided matrix X (labeled and unlabeled)
and corresponding label matrix y with a dedicated marker value for
unlabeled samples.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    A {n_samples by n_samples} size matrix will be created from this

y : array_like, shape = [n_samples]
    n_labeled_samples (unlabeled points are marked as -1)
    All unlabeled samples will be transductively assigned labels

Returns
-------
self : returns an instance of self.
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

val predict : x:Ndarray.t -> t -> Ndarray.t
(**
Performs inductive inference across the model.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
y : array_like, shape = [n_samples]
    Predictions for input data
*)

val predict_proba : x:Ndarray.t -> t -> Ndarray.t
(**
Predict probability for each possible outcome.

Compute the probability estimates for each single sample in X
and each possible outcome seen during training (categorical
distribution).

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
probabilities : array, shape = [n_samples, n_classes]
    Normalized probability distributions across
    class labels
*)

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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


(** Attribute X_: see constructor for documentation *)
val x_ : t -> Ndarray.t

(** Attribute classes_: see constructor for documentation *)
val classes_ : t -> Ndarray.t

(** Attribute label_distributions_: see constructor for documentation *)
val label_distributions_ : t -> Ndarray.t

(** Attribute transduction_: see constructor for documentation *)
val transduction_ : t -> Ndarray.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> int

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module LabelSpreading : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?kernel:[`Knn | `Rbf | `Callable of Py.Object.t] -> ?gamma:float -> ?n_neighbors:int -> ?alpha:float -> ?max_iter:int -> ?tol:float -> ?n_jobs:[`Int of int | `None] -> unit -> t
(**
LabelSpreading model for semi-supervised learning

This model is similar to the basic Label Propagation algorithm,
but uses affinity matrix based on the normalized graph Laplacian
and soft clamping across the labels.

Read more in the :ref:`User Guide <label_propagation>`.

Parameters
----------
kernel : {'knn', 'rbf', callable}
    String identifier for kernel function to use or the kernel function
    itself. Only 'rbf' and 'knn' strings are valid inputs. The function
    passed should take two inputs, each of shape [n_samples, n_features],
    and return a [n_samples, n_samples] shaped weight matrix

gamma : float
  parameter for rbf kernel

n_neighbors : integer > 0
  parameter for knn kernel

alpha : float
  Clamping factor. A value in (0, 1) that specifies the relative amount
  that an instance should adopt the information from its neighbors as
  opposed to its initial label.
  alpha=0 means keeping the initial label information; alpha=1 means
  replacing all initial information.

max_iter : integer
  maximum number of iterations allowed

tol : float
  Convergence tolerance: threshold to consider the system at steady
  state

n_jobs : int or None, optional (default=None)
    The number of parallel jobs to run.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

Attributes
----------
X_ : array, shape = [n_samples, n_features]
    Input array.

classes_ : array, shape = [n_classes]
    The distinct labels used in classifying instances.

label_distributions_ : array, shape = [n_samples, n_classes]
    Categorical distribution for each item.

transduction_ : array, shape = [n_samples]
    Label assigned to each item via the transduction.

n_iter_ : int
    Number of iterations run.

Examples
--------
>>> import numpy as np
>>> from sklearn import datasets
>>> from sklearn.semi_supervised import LabelSpreading
>>> label_prop_model = LabelSpreading()
>>> iris = datasets.load_iris()
>>> rng = np.random.RandomState(42)
>>> random_unlabeled_points = rng.rand(len(iris.target)) < 0.3
>>> labels = np.copy(iris.target)
>>> labels[random_unlabeled_points] = -1
>>> label_prop_model.fit(iris.data, labels)
LabelSpreading(...)

References
----------
Dengyong Zhou, Olivier Bousquet, Thomas Navin Lal, Jason Weston,
Bernhard Schoelkopf. Learning with local and global consistency (2004)
http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.115.3219

See Also
--------
LabelPropagation : Unregularized graph based semi-supervised learning
*)

val fit : x:Ndarray.t -> y:Ndarray.t -> t -> t
(**
Fit a semi-supervised label propagation model based

All the input data is provided matrix X (labeled and unlabeled)
and corresponding label matrix y with a dedicated marker value for
unlabeled samples.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    A {n_samples by n_samples} size matrix will be created from this

y : array_like, shape = [n_samples]
    n_labeled_samples (unlabeled points are marked as -1)
    All unlabeled samples will be transductively assigned labels

Returns
-------
self : returns an instance of self.
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

val predict : x:Ndarray.t -> t -> Ndarray.t
(**
Performs inductive inference across the model.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
y : array_like, shape = [n_samples]
    Predictions for input data
*)

val predict_proba : x:Ndarray.t -> t -> Ndarray.t
(**
Predict probability for each possible outcome.

Compute the probability estimates for each single sample in X
and each possible outcome seen during training (categorical
distribution).

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
probabilities : array, shape = [n_samples, n_classes]
    Normalized probability distributions across
    class labels
*)

val score : ?sample_weight:Ndarray.t -> x:Ndarray.t -> y:Ndarray.t -> t -> float
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


(** Attribute X_: see constructor for documentation *)
val x_ : t -> Ndarray.t

(** Attribute classes_: see constructor for documentation *)
val classes_ : t -> Ndarray.t

(** Attribute label_distributions_: see constructor for documentation *)
val label_distributions_ : t -> Ndarray.t

(** Attribute transduction_: see constructor for documentation *)
val transduction_ : t -> Ndarray.t

(** Attribute n_iter_: see constructor for documentation *)
val n_iter_ : t -> int

(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

