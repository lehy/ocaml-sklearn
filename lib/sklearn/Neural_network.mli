(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module BernoulliRBM : sig
type tag = [`BernoulliRBM]
type t = [`BaseEstimator | `BernoulliRBM | `Object | `TransformerMixin] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_transformer : t -> [`TransformerMixin] Obj.t
val as_estimator : t -> [`BaseEstimator] Obj.t
val create : ?n_components:int -> ?learning_rate:float -> ?batch_size:int -> ?n_iter:int -> ?verbose:int -> ?random_state:int -> unit -> t
(**
Bernoulli Restricted Boltzmann Machine (RBM).

A Restricted Boltzmann Machine with binary visible units and
binary hidden units. Parameters are estimated using Stochastic Maximum
Likelihood (SML), also known as Persistent Contrastive Divergence (PCD)
[2].

The time complexity of this implementation is ``O(d ** 2)`` assuming
d ~ n_features ~ n_components.

Read more in the :ref:`User Guide <rbm>`.

Parameters
----------
n_components : int, default=256
    Number of binary hidden units.

learning_rate : float, default=0.1
    The learning rate for weight updates. It is *highly* recommended
    to tune this hyper-parameter. Reasonable values are in the
    10**[0., -3.] range.

batch_size : int, default=10
    Number of examples per minibatch.

n_iter : int, default=10
    Number of iterations/sweeps over the training dataset to perform
    during training.

verbose : int, default=0
    The verbosity level. The default, zero, means silent mode.

random_state : integer or RandomState, default=None
    Determines random number generation for:

    - Gibbs sampling from visible and hidden layers.

    - Initializing components, sampling from layers during fit.

    - Corrupting the data when scoring samples.

    Pass an int for reproducible results across multiple function calls.
    See :term:`Glossary <random_state>`.

Attributes
----------
intercept_hidden_ : array-like, shape (n_components,)
    Biases of the hidden units.

intercept_visible_ : array-like, shape (n_features,)
    Biases of the visible units.

components_ : array-like, shape (n_components, n_features)
    Weight matrix, where n_features in the number of
    visible units and n_components is the number of hidden units.

h_samples_ : array-like, shape (batch_size, n_components)
    Hidden Activation sampled from the model distribution,
    where batch_size in the number of examples per minibatch and
    n_components is the number of hidden units.

Examples
--------

>>> import numpy as np
>>> from sklearn.neural_network import BernoulliRBM
>>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
>>> model = BernoulliRBM(n_components=2)
>>> model.fit(X)
BernoulliRBM(n_components=2)

References
----------

[1] Hinton, G. E., Osindero, S. and Teh, Y. A fast learning algorithm for
    deep belief nets. Neural Computation 18, pp 1527-1554.
    https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

[2] Tieleman, T. Training Restricted Boltzmann Machines using
    Approximations to the Likelihood Gradient. International Conference
    on Machine Learning (ICML) 2008
*)

val fit : ?y:Py.Object.t -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fit the model to the data X.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Training data.

Returns
-------
self : BernoulliRBM
    The fitted model.
*)

val fit_transform : ?y:[>`ArrayLike] Np.Obj.t -> ?fit_params:(string * Py.Object.t) list -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Fit to data, then transform it.

Fits transformer to X and y with optional parameters fit_params
and returns a transformed version of X.

Parameters
----------
X : {array-like, sparse matrix, dataframe} of shape                 (n_samples, n_features)

y : ndarray of shape (n_samples,), default=None
    Target values.

**fit_params : dict
    Additional fit parameters.

Returns
-------
X_new : ndarray array of shape (n_samples, n_features_new)
    Transformed array.
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

val gibbs : v:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Perform one Gibbs sampling step.

Parameters
----------
v : ndarray of shape (n_samples, n_features)
    Values of the visible layer to start from.

Returns
-------
v_new : ndarray of shape (n_samples, n_features)
    Values of the visible layer after one Gibbs step.
*)

val partial_fit : ?y:Py.Object.t -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fit the model to the data X which should contain a partial
segment of the data.

Parameters
----------
X : ndarray of shape (n_samples, n_features)
    Training data.

Returns
-------
self : BernoulliRBM
    The fitted model.
*)

val score_samples : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Compute the pseudo-likelihood of X.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Values of the visible layer. Must be all-boolean (not checked).

Returns
-------
pseudo_likelihood : ndarray of shape (n_samples,)
    Value of the pseudo-likelihood (proxy for likelihood).

Notes
-----
This method is not deterministic: it computes a quantity called the
free energy on X, then on a randomly corrupted version of X, and
returns the log of the logistic function of the difference.
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

val transform : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Compute the hidden layer activation probabilities, P(h=1|v=X).

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The data to be transformed.

Returns
-------
h : ndarray of shape (n_samples, n_components)
    Latent representations of the data.
*)


(** Attribute intercept_hidden_: get value or raise Not_found if None.*)
val intercept_hidden_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute intercept_hidden_: get value as an option. *)
val intercept_hidden_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute intercept_visible_: get value or raise Not_found if None.*)
val intercept_visible_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute intercept_visible_: get value as an option. *)
val intercept_visible_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute components_: get value or raise Not_found if None.*)
val components_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute components_: get value as an option. *)
val components_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute h_samples_: get value or raise Not_found if None.*)
val h_samples_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute h_samples_: get value as an option. *)
val h_samples_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MLPClassifier : sig
type tag = [`MLPClassifier]
type t = [`BaseEstimator | `BaseMultilayerPerceptron | `ClassifierMixin | `MLPClassifier | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_classifier : t -> [`ClassifierMixin] Obj.t
val as_multilayer_perceptron : t -> [`BaseMultilayerPerceptron] Obj.t
val as_estimator : t -> [`BaseEstimator] Obj.t
val create : ?hidden_layer_sizes:Py.Object.t -> ?activation:[`Identity | `Logistic | `Tanh | `Relu] -> ?solver:[`Lbfgs | `Sgd | `Adam] -> ?alpha:float -> ?batch_size:int -> ?learning_rate:[`Constant | `Invscaling | `Adaptive] -> ?learning_rate_init:float -> ?power_t:float -> ?max_iter:int -> ?shuffle:bool -> ?random_state:int -> ?tol:float -> ?verbose:int -> ?warm_start:bool -> ?momentum:float -> ?nesterovs_momentum:bool -> ?early_stopping:bool -> ?validation_fraction:float -> ?beta_1:float -> ?beta_2:float -> ?epsilon:float -> ?n_iter_no_change:int -> ?max_fun:int -> unit -> t
(**
Multi-layer Perceptron classifier.

This model optimizes the log-loss function using LBFGS or stochastic
gradient descent.

.. versionadded:: 0.18

Parameters
----------
hidden_layer_sizes : tuple, length = n_layers - 2, default=(100,)
    The ith element represents the number of neurons in the ith
    hidden layer.

activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
    Activation function for the hidden layer.

    - 'identity', no-op activation, useful to implement linear bottleneck,
      returns f(x) = x

    - 'logistic', the logistic sigmoid function,
      returns f(x) = 1 / (1 + exp(-x)).

    - 'tanh', the hyperbolic tan function,
      returns f(x) = tanh(x).

    - 'relu', the rectified linear unit function,
      returns f(x) = max(0, x)

solver : {'lbfgs', 'sgd', 'adam'}, default='adam'
    The solver for weight optimization.

    - 'lbfgs' is an optimizer in the family of quasi-Newton methods.

    - 'sgd' refers to stochastic gradient descent.

    - 'adam' refers to a stochastic gradient-based optimizer proposed
      by Kingma, Diederik, and Jimmy Ba

    Note: The default solver 'adam' works pretty well on relatively
    large datasets (with thousands of training samples or more) in terms of
    both training time and validation score.
    For small datasets, however, 'lbfgs' can converge faster and perform
    better.

alpha : float, default=0.0001
    L2 penalty (regularization term) parameter.

batch_size : int, default='auto'
    Size of minibatches for stochastic optimizers.
    If the solver is 'lbfgs', the classifier will not use minibatch.
    When set to 'auto', `batch_size=min(200, n_samples)`

learning_rate : {'constant', 'invscaling', 'adaptive'}, default='constant'
    Learning rate schedule for weight updates.

    - 'constant' is a constant learning rate given by
      'learning_rate_init'.

    - 'invscaling' gradually decreases the learning rate at each
      time step 't' using an inverse scaling exponent of 'power_t'.
      effective_learning_rate = learning_rate_init / pow(t, power_t)

    - 'adaptive' keeps the learning rate constant to
      'learning_rate_init' as long as training loss keeps decreasing.
      Each time two consecutive epochs fail to decrease training loss by at
      least tol, or fail to increase validation score by at least tol if
      'early_stopping' is on, the current learning rate is divided by 5.

    Only used when ``solver='sgd'``.

learning_rate_init : double, default=0.001
    The initial learning rate used. It controls the step-size
    in updating the weights. Only used when solver='sgd' or 'adam'.

power_t : double, default=0.5
    The exponent for inverse scaling learning rate.
    It is used in updating effective learning rate when the learning_rate
    is set to 'invscaling'. Only used when solver='sgd'.

max_iter : int, default=200
    Maximum number of iterations. The solver iterates until convergence
    (determined by 'tol') or this number of iterations. For stochastic
    solvers ('sgd', 'adam'), note that this determines the number of epochs
    (how many times each data point will be used), not the number of
    gradient steps.

shuffle : bool, default=True
    Whether to shuffle samples in each iteration. Only used when
    solver='sgd' or 'adam'.

random_state : int, RandomState instance, default=None
    Determines random number generation for weights and bias
    initialization, train-test split if early stopping is used, and batch
    sampling when solver='sgd' or 'adam'.
    Pass an int for reproducible results across multiple function calls.
    See :term:`Glossary <random_state>`.

tol : float, default=1e-4
    Tolerance for the optimization. When the loss or score is not improving
    by at least ``tol`` for ``n_iter_no_change`` consecutive iterations,
    unless ``learning_rate`` is set to 'adaptive', convergence is
    considered to be reached and training stops.

verbose : bool, default=False
    Whether to print progress messages to stdout.

warm_start : bool, default=False
    When set to True, reuse the solution of the previous
    call to fit as initialization, otherwise, just erase the
    previous solution. See :term:`the Glossary <warm_start>`.

momentum : float, default=0.9
    Momentum for gradient descent update. Should be between 0 and 1. Only
    used when solver='sgd'.

nesterovs_momentum : boolean, default=True
    Whether to use Nesterov's momentum. Only used when solver='sgd' and
    momentum > 0.

early_stopping : bool, default=False
    Whether to use early stopping to terminate training when validation
    score is not improving. If set to true, it will automatically set
    aside 10% of training data as validation and terminate training when
    validation score is not improving by at least tol for
    ``n_iter_no_change`` consecutive epochs. The split is stratified,
    except in a multilabel setting.
    Only effective when solver='sgd' or 'adam'

validation_fraction : float, default=0.1
    The proportion of training data to set aside as validation set for
    early stopping. Must be between 0 and 1.
    Only used if early_stopping is True

beta_1 : float, default=0.9
    Exponential decay rate for estimates of first moment vector in adam,
    should be in [0, 1). Only used when solver='adam'

beta_2 : float, default=0.999
    Exponential decay rate for estimates of second moment vector in adam,
    should be in [0, 1). Only used when solver='adam'

epsilon : float, default=1e-8
    Value for numerical stability in adam. Only used when solver='adam'

n_iter_no_change : int, default=10
    Maximum number of epochs to not meet ``tol`` improvement.
    Only effective when solver='sgd' or 'adam'

    .. versionadded:: 0.20

max_fun : int, default=15000
    Only used when solver='lbfgs'. Maximum number of loss function calls.
    The solver iterates until convergence (determined by 'tol'), number
    of iterations reaches max_iter, or this number of loss function calls.
    Note that number of loss function calls will be greater than or equal
    to the number of iterations for the `MLPClassifier`.

    .. versionadded:: 0.22

Attributes
----------
classes_ : ndarray or list of ndarray of shape (n_classes,)
    Class labels for each output.

loss_ : float
    The current loss computed with the loss function.

coefs_ : list, length n_layers - 1
    The ith element in the list represents the weight matrix corresponding
    to layer i.

intercepts_ : list, length n_layers - 1
    The ith element in the list represents the bias vector corresponding to
    layer i + 1.

n_iter_ : int,
    The number of iterations the solver has ran.

n_layers_ : int
    Number of layers.

n_outputs_ : int
    Number of outputs.

out_activation_ : string
    Name of the output activation function.


Examples
--------
>>> from sklearn.neural_network import MLPClassifier
>>> from sklearn.datasets import make_classification
>>> from sklearn.model_selection import train_test_split
>>> X, y = make_classification(n_samples=100, random_state=1)
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
...                                                     random_state=1)
>>> clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
>>> clf.predict_proba(X_test[:1])
array([[0.038..., 0.961...]])
>>> clf.predict(X_test[:5, :])
array([1, 0, 1, 0, 1])
>>> clf.score(X_test, y_test)
0.8...

Notes
-----
MLPClassifier trains iteratively since at each time step
the partial derivatives of the loss function with respect to the model
parameters are computed to update the parameters.

It can also have a regularization term added to the loss function
that shrinks model parameters to prevent overfitting.

This implementation works with data represented as dense numpy arrays or
sparse scipy arrays of floating point values.

References
----------
Hinton, Geoffrey E.
    'Connectionist learning procedures.' Artificial intelligence 40.1
    (1989): 185-234.

Glorot, Xavier, and Yoshua Bengio. 'Understanding the difficulty of
    training deep feedforward neural networks.' International Conference
    on Artificial Intelligence and Statistics. 2010.

He, Kaiming, et al. 'Delving deep into rectifiers: Surpassing human-level
    performance on imagenet classification.' arXiv preprint
    arXiv:1502.01852 (2015).

Kingma, Diederik, and Jimmy Ba. 'Adam: A method for stochastic
    optimization.' arXiv preprint arXiv:1412.6980 (2014).
*)

val fit : x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fit the model to data matrix X and target(s) y.

Parameters
----------
X : ndarray or sparse matrix of shape (n_samples, n_features)
    The input data.

y : ndarray, shape (n_samples,) or (n_samples, n_outputs)
    The target values (class labels in classification, real numbers in
    regression).

Returns
-------
self : returns a trained MLP model.
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

val partial_fit : ?classes:Py.Object.t -> x:Py.Object.t -> y:Py.Object.t -> [> tag] Obj.t -> t
(**
None
*)

val predict : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Predict using the multi-layer perceptron classifier

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input data.

Returns
-------
y : ndarray, shape (n_samples,) or (n_samples, n_classes)
    The predicted classes.
*)

val predict_log_proba : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Return the log of probability estimates.

Parameters
----------
X : ndarray of shape (n_samples, n_features)
    The input data.

Returns
-------
log_y_prob : ndarray of shape (n_samples, n_classes)
    The predicted log-probability of the sample for each class
    in the model, where classes are ordered as they are in
    `self.classes_`. Equivalent to log(predict_proba(X))
*)

val predict_proba : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Probability estimates.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input data.

Returns
-------
y_prob : ndarray of shape (n_samples, n_classes)
    The predicted probability of the sample for each class in the
    model, where classes are ordered as they are in `self.classes_`.
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


(** Attribute loss_: get value or raise Not_found if None.*)
val loss_ : t -> float

(** Attribute loss_: get value as an option. *)
val loss_opt : t -> (float) option


(** Attribute coefs_: get value or raise Not_found if None.*)
val coefs_ : t -> Py.Object.t

(** Attribute coefs_: get value as an option. *)
val coefs_opt : t -> (Py.Object.t) option


(** Attribute intercepts_: get value or raise Not_found if None.*)
val intercepts_ : t -> Py.Object.t

(** Attribute intercepts_: get value as an option. *)
val intercepts_opt : t -> (Py.Object.t) option


(** Attribute n_iter_: get value or raise Not_found if None.*)
val n_iter_ : t -> int

(** Attribute n_iter_: get value as an option. *)
val n_iter_opt : t -> (int) option


(** Attribute n_layers_: get value or raise Not_found if None.*)
val n_layers_ : t -> int

(** Attribute n_layers_: get value as an option. *)
val n_layers_opt : t -> (int) option


(** Attribute n_outputs_: get value or raise Not_found if None.*)
val n_outputs_ : t -> int

(** Attribute n_outputs_: get value as an option. *)
val n_outputs_opt : t -> (int) option


(** Attribute out_activation_: get value or raise Not_found if None.*)
val out_activation_ : t -> string

(** Attribute out_activation_: get value as an option. *)
val out_activation_opt : t -> (string) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MLPRegressor : sig
type tag = [`MLPRegressor]
type t = [`BaseEstimator | `BaseMultilayerPerceptron | `MLPRegressor | `Object | `RegressorMixin] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_estimator : t -> [`BaseEstimator] Obj.t
val as_multilayer_perceptron : t -> [`BaseMultilayerPerceptron] Obj.t
val as_regressor : t -> [`RegressorMixin] Obj.t
val create : ?hidden_layer_sizes:Py.Object.t -> ?activation:[`Identity | `Logistic | `Tanh | `Relu] -> ?solver:[`Lbfgs | `Sgd | `Adam] -> ?alpha:float -> ?batch_size:int -> ?learning_rate:[`Constant | `Invscaling | `Adaptive] -> ?learning_rate_init:float -> ?power_t:float -> ?max_iter:int -> ?shuffle:bool -> ?random_state:int -> ?tol:float -> ?verbose:int -> ?warm_start:bool -> ?momentum:float -> ?nesterovs_momentum:bool -> ?early_stopping:bool -> ?validation_fraction:float -> ?beta_1:float -> ?beta_2:float -> ?epsilon:float -> ?n_iter_no_change:int -> ?max_fun:int -> unit -> t
(**
Multi-layer Perceptron regressor.

This model optimizes the squared-loss using LBFGS or stochastic gradient
descent.

.. versionadded:: 0.18

Parameters
----------
hidden_layer_sizes : tuple, length = n_layers - 2, default=(100,)
    The ith element represents the number of neurons in the ith
    hidden layer.

activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
    Activation function for the hidden layer.

    - 'identity', no-op activation, useful to implement linear bottleneck,
      returns f(x) = x

    - 'logistic', the logistic sigmoid function,
      returns f(x) = 1 / (1 + exp(-x)).

    - 'tanh', the hyperbolic tan function,
      returns f(x) = tanh(x).

    - 'relu', the rectified linear unit function,
      returns f(x) = max(0, x)

solver : {'lbfgs', 'sgd', 'adam'}, default='adam'
    The solver for weight optimization.

    - 'lbfgs' is an optimizer in the family of quasi-Newton methods.

    - 'sgd' refers to stochastic gradient descent.

    - 'adam' refers to a stochastic gradient-based optimizer proposed by
      Kingma, Diederik, and Jimmy Ba

    Note: The default solver 'adam' works pretty well on relatively
    large datasets (with thousands of training samples or more) in terms of
    both training time and validation score.
    For small datasets, however, 'lbfgs' can converge faster and perform
    better.

alpha : float, default=0.0001
    L2 penalty (regularization term) parameter.

batch_size : int, default='auto'
    Size of minibatches for stochastic optimizers.
    If the solver is 'lbfgs', the classifier will not use minibatch.
    When set to 'auto', `batch_size=min(200, n_samples)`

learning_rate : {'constant', 'invscaling', 'adaptive'}, default='constant'
    Learning rate schedule for weight updates.

    - 'constant' is a constant learning rate given by
      'learning_rate_init'.

    - 'invscaling' gradually decreases the learning rate ``learning_rate_``
      at each time step 't' using an inverse scaling exponent of 'power_t'.
      effective_learning_rate = learning_rate_init / pow(t, power_t)

    - 'adaptive' keeps the learning rate constant to
      'learning_rate_init' as long as training loss keeps decreasing.
      Each time two consecutive epochs fail to decrease training loss by at
      least tol, or fail to increase validation score by at least tol if
      'early_stopping' is on, the current learning rate is divided by 5.

    Only used when solver='sgd'.

learning_rate_init : double, default=0.001
    The initial learning rate used. It controls the step-size
    in updating the weights. Only used when solver='sgd' or 'adam'.

power_t : double, default=0.5
    The exponent for inverse scaling learning rate.
    It is used in updating effective learning rate when the learning_rate
    is set to 'invscaling'. Only used when solver='sgd'.

max_iter : int, default=200
    Maximum number of iterations. The solver iterates until convergence
    (determined by 'tol') or this number of iterations. For stochastic
    solvers ('sgd', 'adam'), note that this determines the number of epochs
    (how many times each data point will be used), not the number of
    gradient steps.

shuffle : bool, default=True
    Whether to shuffle samples in each iteration. Only used when
    solver='sgd' or 'adam'.

random_state : int, RandomState instance, default=None
    Determines random number generation for weights and bias
    initialization, train-test split if early stopping is used, and batch
    sampling when solver='sgd' or 'adam'.
    Pass an int for reproducible results across multiple function calls.
    See :term:`Glossary <random_state>`.

tol : float, default=1e-4
    Tolerance for the optimization. When the loss or score is not improving
    by at least ``tol`` for ``n_iter_no_change`` consecutive iterations,
    unless ``learning_rate`` is set to 'adaptive', convergence is
    considered to be reached and training stops.

verbose : bool, default=False
    Whether to print progress messages to stdout.

warm_start : bool, default=False
    When set to True, reuse the solution of the previous
    call to fit as initialization, otherwise, just erase the
    previous solution. See :term:`the Glossary <warm_start>`.

momentum : float, default=0.9
    Momentum for gradient descent update.  Should be between 0 and 1. Only
    used when solver='sgd'.

nesterovs_momentum : boolean, default=True
    Whether to use Nesterov's momentum. Only used when solver='sgd' and
    momentum > 0.

early_stopping : bool, default=False
    Whether to use early stopping to terminate training when validation
    score is not improving. If set to true, it will automatically set
    aside 10% of training data as validation and terminate training when
    validation score is not improving by at least ``tol`` for
    ``n_iter_no_change`` consecutive epochs.
    Only effective when solver='sgd' or 'adam'

validation_fraction : float, default=0.1
    The proportion of training data to set aside as validation set for
    early stopping. Must be between 0 and 1.
    Only used if early_stopping is True

beta_1 : float, default=0.9
    Exponential decay rate for estimates of first moment vector in adam,
    should be in [0, 1). Only used when solver='adam'

beta_2 : float, default=0.999
    Exponential decay rate for estimates of second moment vector in adam,
    should be in [0, 1). Only used when solver='adam'

epsilon : float, default=1e-8
    Value for numerical stability in adam. Only used when solver='adam'

n_iter_no_change : int, default=10
    Maximum number of epochs to not meet ``tol`` improvement.
    Only effective when solver='sgd' or 'adam'

    .. versionadded:: 0.20

max_fun : int, default=15000
    Only used when solver='lbfgs'. Maximum number of function calls.
    The solver iterates until convergence (determined by 'tol'), number
    of iterations reaches max_iter, or this number of function calls.
    Note that number of function calls will be greater than or equal to
    the number of iterations for the MLPRegressor.

    .. versionadded:: 0.22

Attributes
----------
loss_ : float
    The current loss computed with the loss function.

coefs_ : list, length n_layers - 1
    The ith element in the list represents the weight matrix corresponding
    to layer i.

intercepts_ : list, length n_layers - 1
    The ith element in the list represents the bias vector corresponding to
    layer i + 1.

n_iter_ : int,
    The number of iterations the solver has ran.

n_layers_ : int
    Number of layers.

n_outputs_ : int
    Number of outputs.

out_activation_ : string
    Name of the output activation function.

Examples
--------
>>> from sklearn.neural_network import MLPRegressor
>>> from sklearn.datasets import make_regression
>>> from sklearn.model_selection import train_test_split
>>> X, y = make_regression(n_samples=200, random_state=1)
>>> X_train, X_test, y_train, y_test = train_test_split(X, y,
...                                                     random_state=1)
>>> regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
>>> regr.predict(X_test[:2])
array([-0.9..., -7.1...])
>>> regr.score(X_test, y_test)
0.4...

Notes
-----
MLPRegressor trains iteratively since at each time step
the partial derivatives of the loss function with respect to the model
parameters are computed to update the parameters.

It can also have a regularization term added to the loss function
that shrinks model parameters to prevent overfitting.

This implementation works with data represented as dense and sparse numpy
arrays of floating point values.

References
----------
Hinton, Geoffrey E.
    'Connectionist learning procedures.' Artificial intelligence 40.1
    (1989): 185-234.

Glorot, Xavier, and Yoshua Bengio. 'Understanding the difficulty of
    training deep feedforward neural networks.' International Conference
    on Artificial Intelligence and Statistics. 2010.

He, Kaiming, et al. 'Delving deep into rectifiers: Surpassing human-level
    performance on imagenet classification.' arXiv preprint
    arXiv:1502.01852 (2015).

Kingma, Diederik, and Jimmy Ba. 'Adam: A method for stochastic
    optimization.' arXiv preprint arXiv:1412.6980 (2014).
*)

val fit : x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Fit the model to data matrix X and target(s) y.

Parameters
----------
X : ndarray or sparse matrix of shape (n_samples, n_features)
    The input data.

y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
    The target values (class labels in classification, real numbers in
    regression).

Returns
-------
self : returns a trained MLP model.
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

val partial_fit : x:Py.Object.t -> y:Py.Object.t -> [> tag] Obj.t -> t
(**
None
*)

val predict : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Predict using the multi-layer perceptron model.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input data.

Returns
-------
y : ndarray of shape (n_samples, n_outputs)
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


(** Attribute loss_: get value or raise Not_found if None.*)
val loss_ : t -> float

(** Attribute loss_: get value as an option. *)
val loss_opt : t -> (float) option


(** Attribute coefs_: get value or raise Not_found if None.*)
val coefs_ : t -> Py.Object.t

(** Attribute coefs_: get value as an option. *)
val coefs_opt : t -> (Py.Object.t) option


(** Attribute intercepts_: get value or raise Not_found if None.*)
val intercepts_ : t -> Py.Object.t

(** Attribute intercepts_: get value as an option. *)
val intercepts_opt : t -> (Py.Object.t) option


(** Attribute n_iter_: get value or raise Not_found if None.*)
val n_iter_ : t -> int

(** Attribute n_iter_: get value as an option. *)
val n_iter_opt : t -> (int) option


(** Attribute n_layers_: get value or raise Not_found if None.*)
val n_layers_ : t -> int

(** Attribute n_layers_: get value as an option. *)
val n_layers_opt : t -> (int) option


(** Attribute n_outputs_: get value or raise Not_found if None.*)
val n_outputs_ : t -> int

(** Attribute n_outputs_: get value as an option. *)
val n_outputs_opt : t -> (int) option


(** Attribute out_activation_: get value or raise Not_found if None.*)
val out_activation_ : t -> string

(** Attribute out_activation_: get value as an option. *)
val out_activation_opt : t -> (string) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

