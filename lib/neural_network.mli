(** Get an attribute of this module as a Py.Object.t. This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module BernoulliRBM : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

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
    A random number generator instance to define the state of the
    random permutations generator. If an integer is given, it fixes the
    seed. Defaults to the global numpy random number generator.

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

val fit : ?y:Py.Object.t -> x:Arr.t -> t -> t
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

val fit_transform : ?y:Arr.t -> ?fit_params:(string * Py.Object.t) list -> x:Arr.t -> t -> Arr.t
(**
Fit to data, then transform it.

Fits transformer to X and y with optional parameters fit_params
and returns a transformed version of X.

Parameters
----------
X : numpy array of shape [n_samples, n_features]
    Training set.

y : numpy array of shape [n_samples]
    Target values.

**fit_params : dict
    Additional fit parameters.

Returns
-------
X_new : numpy array of shape [n_samples, n_features_new]
    Transformed array.
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

val gibbs : v:Arr.t -> t -> Arr.t
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

val partial_fit : ?y:Py.Object.t -> x:Arr.t -> t -> t
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

val score_samples : x:Arr.t -> t -> Arr.t
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

val transform : x:Arr.t -> t -> Arr.t
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
val intercept_hidden_ : t -> Arr.t

(** Attribute intercept_hidden_: get value as an option. *)
val intercept_hidden_opt : t -> (Arr.t) option


(** Attribute intercept_visible_: get value or raise Not_found if None.*)
val intercept_visible_ : t -> Arr.t

(** Attribute intercept_visible_: get value as an option. *)
val intercept_visible_opt : t -> (Arr.t) option


(** Attribute components_: get value or raise Not_found if None.*)
val components_ : t -> Arr.t

(** Attribute components_: get value as an option. *)
val components_opt : t -> (Arr.t) option


(** Attribute h_samples_: get value or raise Not_found if None.*)
val h_samples_ : t -> Arr.t

(** Attribute h_samples_: get value as an option. *)
val h_samples_opt : t -> (Arr.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

