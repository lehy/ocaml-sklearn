(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module BaseDiscreteNB : sig
type tag = [`BaseDiscreteNB]
type t = [`BaseDiscreteNB | `BaseEstimator | `ClassifierMixin | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_classifier : t -> [`ClassifierMixin] Obj.t
val as_estimator : t -> [`BaseEstimator] Obj.t
val fit : ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fit Naive Bayes classifier according to X, y

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

y : array-like of shape (n_samples,)
    Target values.

sample_weight : array-like of shape (n_samples,), default=None
    Weights applied to individual samples (1. for unweighted).

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

val partial_fit : ?classes:[>`ArrayLike] Np.Obj.t -> ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Incremental fit on a batch of samples.

This method is expected to be called several times consecutively
on different chunks of a dataset so as to implement out-of-core
or online learning.

This is especially useful when the whole dataset is too big to fit in
memory at once.

This method has some performance overhead hence it is better to call
partial_fit on chunks of data that are as large as possible
(as long as fitting in the memory budget) to hide the overhead.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

y : array-like of shape (n_samples,)
    Target values.

classes : array-like of shape (n_classes) (default=None)
    List of all the classes that can possibly appear in the y vector.

    Must be provided at the first call to partial_fit, can be omitted
    in subsequent calls.

sample_weight : array-like of shape (n_samples,), default=None
    Weights applied to individual samples (1. for unweighted).

Returns
-------
self : object
*)

val predict : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Perform classification on an array of test vectors X.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
C : ndarray of shape (n_samples,)
    Predicted target values for X
*)

val predict_log_proba : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Return log-probability estimates for the test vector X.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
C : array-like of shape (n_samples, n_classes)
    Returns the log-probability of the samples for each class in
    the model. The columns correspond to the classes in sorted
    order, as they appear in the attribute :term:`classes_`.
*)

val predict_proba : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Return probability estimates for the test vector X.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
C : array-like of shape (n_samples, n_classes)
    Returns the probability of the samples for each class in
    the model. The columns correspond to the classes in sorted
    order, as they appear in the attribute :term:`classes_`.
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


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module BaseNB : sig
type tag = [`BaseNB]
type t = [`BaseEstimator | `BaseNB | `ClassifierMixin | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_classifier : t -> [`ClassifierMixin] Obj.t
val as_estimator : t -> [`BaseEstimator] Obj.t
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
Perform classification on an array of test vectors X.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
C : ndarray of shape (n_samples,)
    Predicted target values for X
*)

val predict_log_proba : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Return log-probability estimates for the test vector X.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
C : array-like of shape (n_samples, n_classes)
    Returns the log-probability of the samples for each class in
    the model. The columns correspond to the classes in sorted
    order, as they appear in the attribute :term:`classes_`.
*)

val predict_proba : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Return probability estimates for the test vector X.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
C : array-like of shape (n_samples, n_classes)
    Returns the probability of the samples for each class in
    the model. The columns correspond to the classes in sorted
    order, as they appear in the attribute :term:`classes_`.
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


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module BernoulliNB : sig
type tag = [`BernoulliNB]
type t = [`BaseEstimator | `BernoulliNB | `ClassifierMixin | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_classifier : t -> [`ClassifierMixin] Obj.t
val as_estimator : t -> [`BaseEstimator] Obj.t
val create : ?alpha:float -> ?binarize:[`F of float | `None] -> ?fit_prior:bool -> ?class_prior:[`Arr of [>`ArrayLike] Np.Obj.t | `Size_ of Py.Object.t] -> unit -> t
(**
Naive Bayes classifier for multivariate Bernoulli models.

Like MultinomialNB, this classifier is suitable for discrete data. The
difference is that while MultinomialNB works with occurrence counts,
BernoulliNB is designed for binary/boolean features.

Read more in the :ref:`User Guide <bernoulli_naive_bayes>`.

Parameters
----------
alpha : float, optional (default=1.0)
    Additive (Laplace/Lidstone) smoothing parameter
    (0 for no smoothing).

binarize : float or None, optional (default=0.0)
    Threshold for binarizing (mapping to booleans) of sample features.
    If None, input is presumed to already consist of binary vectors.

fit_prior : bool, optional (default=True)
    Whether to learn class prior probabilities or not.
    If false, a uniform prior will be used.

class_prior : array-like, size=[n_classes,], optional (default=None)
    Prior probabilities of the classes. If specified the priors are not
    adjusted according to the data.

Attributes
----------
class_count_ : array, shape = [n_classes]
    Number of samples encountered for each class during fitting. This
    value is weighted by the sample weight when provided.

class_log_prior_ : array, shape = [n_classes]
    Log probability of each class (smoothed).

classes_ : array, shape (n_classes,)
    Class labels known to the classifier

feature_count_ : array, shape = [n_classes, n_features]
    Number of samples encountered for each (class, feature)
    during fitting. This value is weighted by the sample weight when
    provided.

feature_log_prob_ : array, shape = [n_classes, n_features]
    Empirical log probability of features given a class, P(x_i|y).

n_features_ : int
    Number of features of each sample.


Examples
--------
>>> import numpy as np
>>> rng = np.random.RandomState(1)
>>> X = rng.randint(5, size=(6, 100))
>>> Y = np.array([1, 2, 3, 4, 4, 5])
>>> from sklearn.naive_bayes import BernoulliNB
>>> clf = BernoulliNB()
>>> clf.fit(X, Y)
BernoulliNB()
>>> print(clf.predict(X[2:3]))
[3]

References
----------
C.D. Manning, P. Raghavan and H. Schuetze (2008). Introduction to
Information Retrieval. Cambridge University Press, pp. 234-265.
https://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html

A. McCallum and K. Nigam (1998). A comparison of event models for naive
Bayes text classification. Proc. AAAI/ICML-98 Workshop on Learning for
Text Categorization, pp. 41-48.

V. Metsis, I. Androutsopoulos and G. Paliouras (2006). Spam filtering with
naive Bayes -- Which naive Bayes? 3rd Conf. on Email and Anti-Spam (CEAS).
*)

val fit : ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fit Naive Bayes classifier according to X, y

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

y : array-like of shape (n_samples,)
    Target values.

sample_weight : array-like of shape (n_samples,), default=None
    Weights applied to individual samples (1. for unweighted).

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

val partial_fit : ?classes:[>`ArrayLike] Np.Obj.t -> ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Incremental fit on a batch of samples.

This method is expected to be called several times consecutively
on different chunks of a dataset so as to implement out-of-core
or online learning.

This is especially useful when the whole dataset is too big to fit in
memory at once.

This method has some performance overhead hence it is better to call
partial_fit on chunks of data that are as large as possible
(as long as fitting in the memory budget) to hide the overhead.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

y : array-like of shape (n_samples,)
    Target values.

classes : array-like of shape (n_classes) (default=None)
    List of all the classes that can possibly appear in the y vector.

    Must be provided at the first call to partial_fit, can be omitted
    in subsequent calls.

sample_weight : array-like of shape (n_samples,), default=None
    Weights applied to individual samples (1. for unweighted).

Returns
-------
self : object
*)

val predict : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Perform classification on an array of test vectors X.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
C : ndarray of shape (n_samples,)
    Predicted target values for X
*)

val predict_log_proba : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Return log-probability estimates for the test vector X.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
C : array-like of shape (n_samples, n_classes)
    Returns the log-probability of the samples for each class in
    the model. The columns correspond to the classes in sorted
    order, as they appear in the attribute :term:`classes_`.
*)

val predict_proba : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Return probability estimates for the test vector X.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
C : array-like of shape (n_samples, n_classes)
    Returns the probability of the samples for each class in
    the model. The columns correspond to the classes in sorted
    order, as they appear in the attribute :term:`classes_`.
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


(** Attribute class_count_: get value or raise Not_found if None.*)
val class_count_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute class_count_: get value as an option. *)
val class_count_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute class_log_prior_: get value or raise Not_found if None.*)
val class_log_prior_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute class_log_prior_: get value as an option. *)
val class_log_prior_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute classes_: get value or raise Not_found if None.*)
val classes_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute classes_: get value as an option. *)
val classes_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute feature_count_: get value or raise Not_found if None.*)
val feature_count_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute feature_count_: get value as an option. *)
val feature_count_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute feature_log_prob_: get value or raise Not_found if None.*)
val feature_log_prob_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute feature_log_prob_: get value as an option. *)
val feature_log_prob_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute n_features_: get value or raise Not_found if None.*)
val n_features_ : t -> int

(** Attribute n_features_: get value as an option. *)
val n_features_opt : t -> (int) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module CategoricalNB : sig
type tag = [`CategoricalNB]
type t = [`BaseEstimator | `CategoricalNB | `ClassifierMixin | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_classifier : t -> [`ClassifierMixin] Obj.t
val as_estimator : t -> [`BaseEstimator] Obj.t
val create : ?alpha:float -> ?fit_prior:bool -> ?class_prior:[`Size of Py.Object.t | `Arr of [>`ArrayLike] Np.Obj.t] -> unit -> t
(**
Naive Bayes classifier for categorical features

The categorical Naive Bayes classifier is suitable for classification with
discrete features that are categorically distributed. The categories of
each feature are drawn from a categorical distribution.

Read more in the :ref:`User Guide <categorical_naive_bayes>`.

Parameters
----------
alpha : float, optional (default=1.0)
    Additive (Laplace/Lidstone) smoothing parameter
    (0 for no smoothing).

fit_prior : boolean, optional (default=True)
    Whether to learn class prior probabilities or not.
    If false, a uniform prior will be used.

class_prior : array-like, size (n_classes,), optional (default=None)
    Prior probabilities of the classes. If specified the priors are not
    adjusted according to the data.

Attributes
----------
category_count_ : list of arrays, len n_features
    Holds arrays of shape (n_classes, n_categories of respective feature)
    for each feature. Each array provides the number of samples
    encountered for each class and category of the specific feature.

class_count_ : array, shape (n_classes,)
    Number of samples encountered for each class during fitting. This
    value is weighted by the sample weight when provided.

class_log_prior_ : array, shape (n_classes, )
    Smoothed empirical log probability for each class.

classes_ : array, shape (n_classes,)
    Class labels known to the classifier

feature_log_prob_ : list of arrays, len n_features
    Holds arrays of shape (n_classes, n_categories of respective feature)
    for each feature. Each array provides the empirical log probability
    of categories given the respective feature and class, ``P(x_i|y)``.

n_features_ : int
    Number of features of each sample.

Examples
--------
>>> import numpy as np
>>> rng = np.random.RandomState(1)
>>> X = rng.randint(5, size=(6, 100))
>>> y = np.array([1, 2, 3, 4, 5, 6])
>>> from sklearn.naive_bayes import CategoricalNB
>>> clf = CategoricalNB()
>>> clf.fit(X, y)
CategoricalNB()
>>> print(clf.predict(X[2:3]))
[3]
*)

val fit : ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fit Naive Bayes classifier according to X, y

Parameters
----------
X : {array-like, sparse matrix}, shape = [n_samples, n_features]
    Training vectors, where n_samples is the number of samples and
    n_features is the number of features. Here, each feature of X is
    assumed to be from a different categorical distribution.
    It is further assumed that all categories of each feature are
    represented by the numbers 0, ..., n - 1, where n refers to the
    total number of categories for the given feature. This can, for
    instance, be achieved with the help of OrdinalEncoder.

y : array-like, shape = [n_samples]
    Target values.

sample_weight : array-like, shape = [n_samples], (default=None)
    Weights applied to individual samples (1. for unweighted).

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

val partial_fit : ?classes:[>`ArrayLike] Np.Obj.t -> ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Incremental fit on a batch of samples.

This method is expected to be called several times consecutively
on different chunks of a dataset so as to implement out-of-core
or online learning.

This is especially useful when the whole dataset is too big to fit in
memory at once.

This method has some performance overhead hence it is better to call
partial_fit on chunks of data that are as large as possible
(as long as fitting in the memory budget) to hide the overhead.

Parameters
----------
X : {array-like, sparse matrix}, shape = [n_samples, n_features]
    Training vectors, where n_samples is the number of samples and
    n_features is the number of features. Here, each feature of X is
    assumed to be from a different categorical distribution.
    It is further assumed that all categories of each feature are
    represented by the numbers 0, ..., n - 1, where n refers to the
    total number of categories for the given feature. This can, for
    instance, be achieved with the help of OrdinalEncoder.

y : array-like, shape = [n_samples]
    Target values.

classes : array-like, shape = [n_classes] (default=None)
    List of all the classes that can possibly appear in the y vector.

    Must be provided at the first call to partial_fit, can be omitted
    in subsequent calls.

sample_weight : array-like, shape = [n_samples], (default=None)
    Weights applied to individual samples (1. for unweighted).

Returns
-------
self : object
*)

val predict : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Perform classification on an array of test vectors X.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
C : ndarray of shape (n_samples,)
    Predicted target values for X
*)

val predict_log_proba : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Return log-probability estimates for the test vector X.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
C : array-like of shape (n_samples, n_classes)
    Returns the log-probability of the samples for each class in
    the model. The columns correspond to the classes in sorted
    order, as they appear in the attribute :term:`classes_`.
*)

val predict_proba : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Return probability estimates for the test vector X.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
C : array-like of shape (n_samples, n_classes)
    Returns the probability of the samples for each class in
    the model. The columns correspond to the classes in sorted
    order, as they appear in the attribute :term:`classes_`.
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


(** Attribute category_count_: get value or raise Not_found if None.*)
val category_count_ : t -> Py.Object.t

(** Attribute category_count_: get value as an option. *)
val category_count_opt : t -> (Py.Object.t) option


(** Attribute class_count_: get value or raise Not_found if None.*)
val class_count_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute class_count_: get value as an option. *)
val class_count_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute class_log_prior_: get value or raise Not_found if None.*)
val class_log_prior_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute class_log_prior_: get value as an option. *)
val class_log_prior_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute classes_: get value or raise Not_found if None.*)
val classes_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute classes_: get value as an option. *)
val classes_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute feature_log_prob_: get value or raise Not_found if None.*)
val feature_log_prob_ : t -> Py.Object.t

(** Attribute feature_log_prob_: get value as an option. *)
val feature_log_prob_opt : t -> (Py.Object.t) option


(** Attribute n_features_: get value or raise Not_found if None.*)
val n_features_ : t -> int

(** Attribute n_features_: get value as an option. *)
val n_features_opt : t -> (int) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module ComplementNB : sig
type tag = [`ComplementNB]
type t = [`BaseEstimator | `ClassifierMixin | `ComplementNB | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_classifier : t -> [`ClassifierMixin] Obj.t
val as_estimator : t -> [`BaseEstimator] Obj.t
val create : ?alpha:float -> ?fit_prior:bool -> ?class_prior:[`Size of Py.Object.t | `Arr of [>`ArrayLike] Np.Obj.t] -> ?norm:bool -> unit -> t
(**
The Complement Naive Bayes classifier described in Rennie et al. (2003).

The Complement Naive Bayes classifier was designed to correct the 'severe
assumptions' made by the standard Multinomial Naive Bayes classifier. It is
particularly suited for imbalanced data sets.

Read more in the :ref:`User Guide <complement_naive_bayes>`.

Parameters
----------
alpha : float, optional (default=1.0)
    Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).

fit_prior : boolean, optional (default=True)
    Only used in edge case with a single class in the training set.

class_prior : array-like, size (n_classes,), optional (default=None)
    Prior probabilities of the classes. Not used.

norm : boolean, optional (default=False)
    Whether or not a second normalization of the weights is performed. The
    default behavior mirrors the implementations found in Mahout and Weka,
    which do not follow the full algorithm described in Table 9 of the
    paper.

Attributes
----------
class_count_ : array, shape (n_classes,)
    Number of samples encountered for each class during fitting. This
    value is weighted by the sample weight when provided.

class_log_prior_ : array, shape (n_classes, )
    Smoothed empirical log probability for each class. Only used in edge
    case with a single class in the training set.

classes_ : array, shape (n_classes,)
    Class labels known to the classifier

feature_all_ : array, shape (n_features,)
    Number of samples encountered for each feature during fitting. This
    value is weighted by the sample weight when provided.

feature_count_ : array, shape (n_classes, n_features)
    Number of samples encountered for each (class, feature) during fitting.
    This value is weighted by the sample weight when provided.

feature_log_prob_ : array, shape (n_classes, n_features)
    Empirical weights for class complements.

n_features_ : int
    Number of features of each sample.

Examples
--------
>>> import numpy as np
>>> rng = np.random.RandomState(1)
>>> X = rng.randint(5, size=(6, 100))
>>> y = np.array([1, 2, 3, 4, 5, 6])
>>> from sklearn.naive_bayes import ComplementNB
>>> clf = ComplementNB()
>>> clf.fit(X, y)
ComplementNB()
>>> print(clf.predict(X[2:3]))
[3]

References
----------
Rennie, J. D., Shih, L., Teevan, J., & Karger, D. R. (2003).
Tackling the poor assumptions of naive bayes text classifiers. In ICML
(Vol. 3, pp. 616-623).
https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf
*)

val fit : ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fit Naive Bayes classifier according to X, y

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

y : array-like of shape (n_samples,)
    Target values.

sample_weight : array-like of shape (n_samples,), default=None
    Weights applied to individual samples (1. for unweighted).

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

val partial_fit : ?classes:[>`ArrayLike] Np.Obj.t -> ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Incremental fit on a batch of samples.

This method is expected to be called several times consecutively
on different chunks of a dataset so as to implement out-of-core
or online learning.

This is especially useful when the whole dataset is too big to fit in
memory at once.

This method has some performance overhead hence it is better to call
partial_fit on chunks of data that are as large as possible
(as long as fitting in the memory budget) to hide the overhead.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

y : array-like of shape (n_samples,)
    Target values.

classes : array-like of shape (n_classes) (default=None)
    List of all the classes that can possibly appear in the y vector.

    Must be provided at the first call to partial_fit, can be omitted
    in subsequent calls.

sample_weight : array-like of shape (n_samples,), default=None
    Weights applied to individual samples (1. for unweighted).

Returns
-------
self : object
*)

val predict : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Perform classification on an array of test vectors X.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
C : ndarray of shape (n_samples,)
    Predicted target values for X
*)

val predict_log_proba : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Return log-probability estimates for the test vector X.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
C : array-like of shape (n_samples, n_classes)
    Returns the log-probability of the samples for each class in
    the model. The columns correspond to the classes in sorted
    order, as they appear in the attribute :term:`classes_`.
*)

val predict_proba : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Return probability estimates for the test vector X.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
C : array-like of shape (n_samples, n_classes)
    Returns the probability of the samples for each class in
    the model. The columns correspond to the classes in sorted
    order, as they appear in the attribute :term:`classes_`.
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


(** Attribute class_count_: get value or raise Not_found if None.*)
val class_count_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute class_count_: get value as an option. *)
val class_count_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute class_log_prior_: get value or raise Not_found if None.*)
val class_log_prior_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute class_log_prior_: get value as an option. *)
val class_log_prior_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute classes_: get value or raise Not_found if None.*)
val classes_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute classes_: get value as an option. *)
val classes_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute feature_all_: get value or raise Not_found if None.*)
val feature_all_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute feature_all_: get value as an option. *)
val feature_all_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute feature_count_: get value or raise Not_found if None.*)
val feature_count_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute feature_count_: get value as an option. *)
val feature_count_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute feature_log_prob_: get value or raise Not_found if None.*)
val feature_log_prob_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute feature_log_prob_: get value as an option. *)
val feature_log_prob_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute n_features_: get value or raise Not_found if None.*)
val n_features_ : t -> int

(** Attribute n_features_: get value as an option. *)
val n_features_opt : t -> (int) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module GaussianNB : sig
type tag = [`GaussianNB]
type t = [`BaseEstimator | `ClassifierMixin | `GaussianNB | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_classifier : t -> [`ClassifierMixin] Obj.t
val as_estimator : t -> [`BaseEstimator] Obj.t
val create : ?priors:[>`ArrayLike] Np.Obj.t -> ?var_smoothing:float -> unit -> t
(**
Gaussian Naive Bayes (GaussianNB)

Can perform online updates to model parameters via :meth:`partial_fit`.
For details on algorithm used to update feature means and variance online,
see Stanford CS tech report STAN-CS-79-773 by Chan, Golub, and LeVeque:

    http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf

Read more in the :ref:`User Guide <gaussian_naive_bayes>`.

Parameters
----------
priors : array-like, shape (n_classes,)
    Prior probabilities of the classes. If specified the priors are not
    adjusted according to the data.

var_smoothing : float, optional (default=1e-9)
    Portion of the largest variance of all features that is added to
    variances for calculation stability.

Attributes
----------
class_count_ : array, shape (n_classes,)
    number of training samples observed in each class.

class_prior_ : array, shape (n_classes,)
    probability of each class.

classes_ : array, shape (n_classes,)
    class labels known to the classifier

epsilon_ : float
    absolute additive value to variances

sigma_ : array, shape (n_classes, n_features)
    variance of each feature per class

theta_ : array, shape (n_classes, n_features)
    mean of each feature per class

Examples
--------
>>> import numpy as np
>>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
>>> Y = np.array([1, 1, 1, 2, 2, 2])
>>> from sklearn.naive_bayes import GaussianNB
>>> clf = GaussianNB()
>>> clf.fit(X, Y)
GaussianNB()
>>> print(clf.predict([[-0.8, -1]]))
[1]
>>> clf_pf = GaussianNB()
>>> clf_pf.partial_fit(X, Y, np.unique(Y))
GaussianNB()
>>> print(clf_pf.predict([[-0.8, -1]]))
[1]
*)

val fit : ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fit Gaussian Naive Bayes according to X, y

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples
    and n_features is the number of features.

y : array-like, shape (n_samples,)
    Target values.

sample_weight : array-like, shape (n_samples,), optional (default=None)
    Weights applied to individual samples (1. for unweighted).

    .. versionadded:: 0.17
       Gaussian Naive Bayes supports fitting with *sample_weight*.

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

val partial_fit : ?classes:[>`ArrayLike] Np.Obj.t -> ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Incremental fit on a batch of samples.

This method is expected to be called several times consecutively
on different chunks of a dataset so as to implement out-of-core
or online learning.

This is especially useful when the whole dataset is too big to fit in
memory at once.

This method has some performance and numerical stability overhead,
hence it is better to call partial_fit on chunks of data that are
as large as possible (as long as fitting in the memory budget) to
hide the overhead.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

y : array-like, shape (n_samples,)
    Target values.

classes : array-like, shape (n_classes,), optional (default=None)
    List of all the classes that can possibly appear in the y vector.

    Must be provided at the first call to partial_fit, can be omitted
    in subsequent calls.

sample_weight : array-like, shape (n_samples,), optional (default=None)
    Weights applied to individual samples (1. for unweighted).

    .. versionadded:: 0.17

Returns
-------
self : object
*)

val predict : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Perform classification on an array of test vectors X.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
C : ndarray of shape (n_samples,)
    Predicted target values for X
*)

val predict_log_proba : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Return log-probability estimates for the test vector X.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
C : array-like of shape (n_samples, n_classes)
    Returns the log-probability of the samples for each class in
    the model. The columns correspond to the classes in sorted
    order, as they appear in the attribute :term:`classes_`.
*)

val predict_proba : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Return probability estimates for the test vector X.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
C : array-like of shape (n_samples, n_classes)
    Returns the probability of the samples for each class in
    the model. The columns correspond to the classes in sorted
    order, as they appear in the attribute :term:`classes_`.
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


(** Attribute class_count_: get value or raise Not_found if None.*)
val class_count_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute class_count_: get value as an option. *)
val class_count_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute class_prior_: get value or raise Not_found if None.*)
val class_prior_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute class_prior_: get value as an option. *)
val class_prior_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute classes_: get value or raise Not_found if None.*)
val classes_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute classes_: get value as an option. *)
val classes_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute epsilon_: get value or raise Not_found if None.*)
val epsilon_ : t -> float

(** Attribute epsilon_: get value as an option. *)
val epsilon_opt : t -> (float) option


(** Attribute sigma_: get value or raise Not_found if None.*)
val sigma_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute sigma_: get value as an option. *)
val sigma_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute theta_: get value or raise Not_found if None.*)
val theta_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute theta_: get value as an option. *)
val theta_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MultinomialNB : sig
type tag = [`MultinomialNB]
type t = [`BaseEstimator | `ClassifierMixin | `MultinomialNB | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_classifier : t -> [`ClassifierMixin] Obj.t
val as_estimator : t -> [`BaseEstimator] Obj.t
val create : ?alpha:float -> ?fit_prior:bool -> ?class_prior:[`Size of Py.Object.t | `Arr of [>`ArrayLike] Np.Obj.t] -> unit -> t
(**
Naive Bayes classifier for multinomial models

The multinomial Naive Bayes classifier is suitable for classification with
discrete features (e.g., word counts for text classification). The
multinomial distribution normally requires integer feature counts. However,
in practice, fractional counts such as tf-idf may also work.

Read more in the :ref:`User Guide <multinomial_naive_bayes>`.

Parameters
----------
alpha : float, optional (default=1.0)
    Additive (Laplace/Lidstone) smoothing parameter
    (0 for no smoothing).

fit_prior : boolean, optional (default=True)
    Whether to learn class prior probabilities or not.
    If false, a uniform prior will be used.

class_prior : array-like, size (n_classes,), optional (default=None)
    Prior probabilities of the classes. If specified the priors are not
    adjusted according to the data.

Attributes
----------
class_count_ : array, shape (n_classes,)
    Number of samples encountered for each class during fitting. This
    value is weighted by the sample weight when provided.

class_log_prior_ : array, shape (n_classes, )
    Smoothed empirical log probability for each class.

classes_ : array, shape (n_classes,)
    Class labels known to the classifier

coef_ : array, shape (n_classes, n_features)
    Mirrors ``feature_log_prob_`` for interpreting MultinomialNB
    as a linear model.

feature_count_ : array, shape (n_classes, n_features)
    Number of samples encountered for each (class, feature)
    during fitting. This value is weighted by the sample weight when
    provided.

feature_log_prob_ : array, shape (n_classes, n_features)
    Empirical log probability of features
    given a class, ``P(x_i|y)``.

intercept_ : array, shape (n_classes, )
    Mirrors ``class_log_prior_`` for interpreting MultinomialNB
    as a linear model.

n_features_ : int
    Number of features of each sample.

Examples
--------
>>> import numpy as np
>>> rng = np.random.RandomState(1)
>>> X = rng.randint(5, size=(6, 100))
>>> y = np.array([1, 2, 3, 4, 5, 6])
>>> from sklearn.naive_bayes import MultinomialNB
>>> clf = MultinomialNB()
>>> clf.fit(X, y)
MultinomialNB()
>>> print(clf.predict(X[2:3]))
[3]

Notes
-----
For the rationale behind the names `coef_` and `intercept_`, i.e.
naive Bayes as a linear classifier, see J. Rennie et al. (2003),
Tackling the poor assumptions of naive Bayes text classifiers, ICML.

References
----------
C.D. Manning, P. Raghavan and H. Schuetze (2008). Introduction to
Information Retrieval. Cambridge University Press, pp. 234-265.
https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html
*)

val fit : ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fit Naive Bayes classifier according to X, y

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

y : array-like of shape (n_samples,)
    Target values.

sample_weight : array-like of shape (n_samples,), default=None
    Weights applied to individual samples (1. for unweighted).

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

val partial_fit : ?classes:[>`ArrayLike] Np.Obj.t -> ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Incremental fit on a batch of samples.

This method is expected to be called several times consecutively
on different chunks of a dataset so as to implement out-of-core
or online learning.

This is especially useful when the whole dataset is too big to fit in
memory at once.

This method has some performance overhead hence it is better to call
partial_fit on chunks of data that are as large as possible
(as long as fitting in the memory budget) to hide the overhead.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

y : array-like of shape (n_samples,)
    Target values.

classes : array-like of shape (n_classes) (default=None)
    List of all the classes that can possibly appear in the y vector.

    Must be provided at the first call to partial_fit, can be omitted
    in subsequent calls.

sample_weight : array-like of shape (n_samples,), default=None
    Weights applied to individual samples (1. for unweighted).

Returns
-------
self : object
*)

val predict : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Perform classification on an array of test vectors X.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
C : ndarray of shape (n_samples,)
    Predicted target values for X
*)

val predict_log_proba : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Return log-probability estimates for the test vector X.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
C : array-like of shape (n_samples, n_classes)
    Returns the log-probability of the samples for each class in
    the model. The columns correspond to the classes in sorted
    order, as they appear in the attribute :term:`classes_`.
*)

val predict_proba : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Return probability estimates for the test vector X.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
C : array-like of shape (n_samples, n_classes)
    Returns the probability of the samples for each class in
    the model. The columns correspond to the classes in sorted
    order, as they appear in the attribute :term:`classes_`.
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


(** Attribute class_count_: get value or raise Not_found if None.*)
val class_count_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute class_count_: get value as an option. *)
val class_count_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute class_log_prior_: get value or raise Not_found if None.*)
val class_log_prior_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute class_log_prior_: get value as an option. *)
val class_log_prior_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute classes_: get value or raise Not_found if None.*)
val classes_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute classes_: get value as an option. *)
val classes_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute coef_: get value or raise Not_found if None.*)
val coef_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute coef_: get value as an option. *)
val coef_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute feature_count_: get value or raise Not_found if None.*)
val feature_count_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute feature_count_: get value as an option. *)
val feature_count_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute feature_log_prob_: get value or raise Not_found if None.*)
val feature_log_prob_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute feature_log_prob_: get value as an option. *)
val feature_log_prob_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute intercept_: get value or raise Not_found if None.*)
val intercept_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute intercept_: get value as an option. *)
val intercept_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute n_features_: get value or raise Not_found if None.*)
val n_features_ : t -> int

(** Attribute n_features_: get value as an option. *)
val n_features_opt : t -> (int) option


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

val binarize : ?threshold:float -> ?copy:bool -> x:[>`ArrayLike] Np.Obj.t -> unit -> Py.Object.t
(**
Boolean thresholding of array-like or scipy.sparse matrix

Read more in the :ref:`User Guide <preprocessing_binarization>`.

Parameters
----------
X : {array-like, sparse matrix}, shape [n_samples, n_features]
    The data to binarize, element by element.
    scipy.sparse matrices should be in CSR or CSC format to avoid an
    un-necessary copy.

threshold : float, optional (0.0 by default)
    Feature values below or equal to this are replaced by 0, above it by 1.
    Threshold may not be less than 0 for operations on sparse matrices.

copy : boolean, optional, default True
    set to False to perform inplace binarization and avoid a copy
    (if the input is already a numpy array or a scipy.sparse CSR / CSC
    matrix and if axis is 1).

See also
--------
Binarizer: Performs binarization using the ``Transformer`` API
    (e.g. as part of a preprocessing :class:`sklearn.pipeline.Pipeline`).
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

val check_non_negative : x:[>`ArrayLike] Np.Obj.t -> whom:string -> unit -> Py.Object.t
(**
Check if there is any negative value in an array.

Parameters
----------
X : array-like or sparse matrix
    Input data.

whom : string
    Who passed X to this function.
*)

val column_or_1d : ?warn:bool -> y:[>`ArrayLike] Np.Obj.t -> unit -> [>`ArrayLike] Np.Obj.t
(**
Ravel column or 1d numpy array, else raises an error

Parameters
----------
y : array-like

warn : boolean, default False
   To control display of warnings.

Returns
-------
y : array
*)

val label_binarize : ?neg_label:int -> ?pos_label:int -> ?sparse_output:bool -> y:[>`ArrayLike] Np.Obj.t -> classes:[>`ArrayLike] Np.Obj.t -> unit -> [>`ArrayLike] Np.Obj.t
(**
Binarize labels in a one-vs-all fashion

Several regression and binary classification algorithms are
available in scikit-learn. A simple way to extend these algorithms
to the multi-class classification case is to use the so-called
one-vs-all scheme.

This function makes it possible to compute this transformation for a
fixed set of class labels known ahead of time.

Parameters
----------
y : array-like
    Sequence of integer labels or multilabel data to encode.

classes : array-like of shape [n_classes]
    Uniquely holds the label for each class.

neg_label : int (default: 0)
    Value with which negative labels must be encoded.

pos_label : int (default: 1)
    Value with which positive labels must be encoded.

sparse_output : boolean (default: False),
    Set to true if output binary array is desired in CSR sparse format

Returns
-------
Y : numpy array or CSR matrix of shape [n_samples, n_classes]
    Shape will be [n_samples, 1] for binary problems.

Examples
--------
>>> from sklearn.preprocessing import label_binarize
>>> label_binarize([1, 6], classes=[1, 2, 4, 6])
array([[1, 0, 0, 0],
       [0, 0, 0, 1]])

The class ordering is preserved:

>>> label_binarize([1, 6], classes=[1, 6, 4, 2])
array([[1, 0, 0, 0],
       [0, 1, 0, 0]])

Binary targets transform to a column vector

>>> label_binarize(['yes', 'no', 'no', 'yes'], classes=['no', 'yes'])
array([[1],
       [0],
       [0],
       [1]])

See also
--------
LabelBinarizer : class used to wrap the functionality of label_binarize and
    allow for fitting to classes independently of the transform operation
*)

val logsumexp : ?axis:int list -> ?b:[>`ArrayLike] Np.Obj.t -> ?keepdims:bool -> ?return_sign:bool -> a:[>`ArrayLike] Np.Obj.t -> unit -> ([>`ArrayLike] Np.Obj.t * [>`ArrayLike] Np.Obj.t)
(**
Compute the log of the sum of exponentials of input elements.

Parameters
----------
a : array_like
    Input array.
axis : None or int or tuple of ints, optional
    Axis or axes over which the sum is taken. By default `axis` is None,
    and all elements are summed.

    .. versionadded:: 0.11.0
keepdims : bool, optional
    If this is set to True, the axes which are reduced are left in the
    result as dimensions with size one. With this option, the result
    will broadcast correctly against the original array.

    .. versionadded:: 0.15.0
b : array-like, optional
    Scaling factor for exp(`a`) must be of the same shape as `a` or
    broadcastable to `a`. These values may be negative in order to
    implement subtraction.

    .. versionadded:: 0.12.0
return_sign : bool, optional
    If this is set to True, the result will be a pair containing sign
    information; if False, results that are negative will be returned
    as NaN. Default is False (no sign information).

    .. versionadded:: 0.16.0

Returns
-------
res : ndarray
    The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
    more stable way. If `b` is given then ``np.log(np.sum(b*np.exp(a)))``
    is returned.
sgn : ndarray
    If return_sign is True, this will be an array of floating-point
    numbers matching res and +1, 0, or -1 depending on the sign
    of the result. If False, only one result is returned.

See Also
--------
numpy.logaddexp, numpy.logaddexp2

Notes
-----
NumPy has a logaddexp function which is very similar to `logsumexp`, but
only handles two arguments. `logaddexp.reduce` is similar to this
function, but may be less stable.

Examples
--------
>>> from scipy.special import logsumexp
>>> a = np.arange(10)
>>> np.log(np.sum(np.exp(a)))
9.4586297444267107
>>> logsumexp(a)
9.4586297444267107

With weights

>>> a = np.arange(10)
>>> b = np.arange(10, 0, -1)
>>> logsumexp(a, b=b)
9.9170178533034665
>>> np.log(np.sum(b*np.exp(a)))
9.9170178533034647

Returning a sign flag

>>> logsumexp([1,2],b=[1,-1],return_sign=True)
(1.5413248546129181, -1.0)

Notice that `logsumexp` does not directly support masked arrays. To use it
on a masked array, convert the mask into zero weights:

>>> a = np.ma.array([np.log(2), 2, np.log(3)],
...                  mask=[False, True, False])
>>> b = (~a.mask).astype(int)
>>> logsumexp(a.data, b=b), np.log(5)
1.6094379124341005, 1.6094379124341005
*)

val safe_sparse_dot : ?dense_output:Py.Object.t -> a:[>`ArrayLike] Np.Obj.t -> b:Py.Object.t -> unit -> [>`ArrayLike] Np.Obj.t
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

