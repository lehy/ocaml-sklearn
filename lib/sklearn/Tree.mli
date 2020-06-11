(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module BaseDecisionTree : sig
type tag = [`BaseDecisionTree]
type t = [`BaseDecisionTree | `BaseEstimator | `MultiOutputMixin | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_multi_output : t -> [`MultiOutputMixin] Obj.t
val as_estimator : t -> [`BaseEstimator] Obj.t
val apply : ?check_input:bool -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Return the index of the leaf that each sample is predicted as.

.. versionadded:: 0.17

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csr_matrix``.

check_input : bool, default=True
    Allow to bypass several input checking.
    Don't use this parameter unless you know what you do.

Returns
-------
X_leaves : array-like of shape (n_samples,)
    For each datapoint x in X, return the index of the leaf x
    ends up in. Leaves are numbered within
    ``[0; self.tree_.node_count)``, possibly with gaps in the
    numbering.
*)

val cost_complexity_pruning_path : ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> (Py.Object.t * [>`ArrayLike] Np.Obj.t * [>`ArrayLike] Np.Obj.t)
(**
Compute the pruning path during Minimal Cost-Complexity Pruning.

See :ref:`minimal_cost_complexity_pruning` for details on the pruning
process.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The training input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csc_matrix``.

y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    The target values (class labels) as integers or strings.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights. If None, then samples are equally weighted. Splits
    that would create child nodes with net zero or negative weight are
    ignored while searching for a split in each node. Splits are also
    ignored if they would result in any single class carrying a
    negative weight in either child node.

Returns
-------
ccp_path : Bunch
    Dictionary-like object, with attributes:

    ccp_alphas : ndarray
        Effective alphas of subtree during pruning.

    impurities : ndarray
        Sum of the impurities of the subtree leaves for the
        corresponding alpha value in ``ccp_alphas``.
*)

val decision_path : ?check_input:bool -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Object|`Spmatrix] Np.Obj.t
(**
Return the decision path in the tree.

.. versionadded:: 0.18

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csr_matrix``.

check_input : bool, default=True
    Allow to bypass several input checking.
    Don't use this parameter unless you know what you do.

Returns
-------
indicator : sparse matrix of shape (n_samples, n_nodes)
    Return a node indicator CSR matrix where non zero elements
    indicates that the samples goes through the nodes.
*)

val fit : ?sample_weight:Py.Object.t -> ?check_input:Py.Object.t -> ?x_idx_sorted:Py.Object.t -> x:Py.Object.t -> y:Py.Object.t -> [> tag] Obj.t -> t
(**
None
*)

val get_depth : [> tag] Obj.t -> Py.Object.t
(**
Return the depth of the decision tree.

The depth of a tree is the maximum distance between the root
and any leaf.

Returns
-------
self.tree_.max_depth : int
    The maximum depth of the tree.
*)

val get_n_leaves : [> tag] Obj.t -> Py.Object.t
(**
Return the number of leaves of the decision tree.

Returns
-------
self.tree_.n_leaves : int
    Number of leaves.
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

val predict : ?check_input:bool -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Predict class or regression value for X.

For a classification model, the predicted class for each sample in X is
returned. For a regression model, the predicted value based on X is
returned.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csr_matrix``.

check_input : bool, default=True
    Allow to bypass several input checking.
    Don't use this parameter unless you know what you do.

Returns
-------
y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    The predicted classes, or the predict values.
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

module DecisionTreeClassifier : sig
type tag = [`DecisionTreeClassifier]
type t = [`BaseDecisionTree | `BaseEstimator | `ClassifierMixin | `DecisionTreeClassifier | `MultiOutputMixin | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_classifier : t -> [`ClassifierMixin] Obj.t
val as_multi_output : t -> [`MultiOutputMixin] Obj.t
val as_estimator : t -> [`BaseEstimator] Obj.t
val as_decision_tree : t -> [`BaseDecisionTree] Obj.t
val create : ?criterion:[`Gini | `Entropy] -> ?splitter:[`Best | `Random] -> ?max_depth:int -> ?min_samples_split:[`F of float | `I of int] -> ?min_samples_leaf:[`F of float | `I of int] -> ?min_weight_fraction_leaf:float -> ?max_features:[`I of int | `Log2 | `F of float | `Auto | `Sqrt] -> ?random_state:int -> ?max_leaf_nodes:int -> ?min_impurity_decrease:float -> ?min_impurity_split:float -> ?class_weight:[`List_of_dict of Py.Object.t | `DictIntToFloat of (int * float) list | `Balanced] -> ?presort:Py.Object.t -> ?ccp_alpha:float -> unit -> t
(**
A decision tree classifier.

Read more in the :ref:`User Guide <tree>`.

Parameters
----------
criterion : {'gini', 'entropy'}, default='gini'
    The function to measure the quality of a split. Supported criteria are
    'gini' for the Gini impurity and 'entropy' for the information gain.

splitter : {'best', 'random'}, default='best'
    The strategy used to choose the split at each node. Supported
    strategies are 'best' to choose the best split and 'random' to choose
    the best random split.

max_depth : int, default=None
    The maximum depth of the tree. If None, then nodes are expanded until
    all leaves are pure or until all leaves contain less than
    min_samples_split samples.

min_samples_split : int or float, default=2
    The minimum number of samples required to split an internal node:

    - If int, then consider `min_samples_split` as the minimum number.
    - If float, then `min_samples_split` is a fraction and
      `ceil(min_samples_split * n_samples)` are the minimum
      number of samples for each split.

    .. versionchanged:: 0.18
       Added float values for fractions.

min_samples_leaf : int or float, default=1
    The minimum number of samples required to be at a leaf node.
    A split point at any depth will only be considered if it leaves at
    least ``min_samples_leaf`` training samples in each of the left and
    right branches.  This may have the effect of smoothing the model,
    especially in regression.

    - If int, then consider `min_samples_leaf` as the minimum number.
    - If float, then `min_samples_leaf` is a fraction and
      `ceil(min_samples_leaf * n_samples)` are the minimum
      number of samples for each node.

    .. versionchanged:: 0.18
       Added float values for fractions.

min_weight_fraction_leaf : float, default=0.0
    The minimum weighted fraction of the sum total of weights (of all
    the input samples) required to be at a leaf node. Samples have
    equal weight when sample_weight is not provided.

max_features : int, float or {'auto', 'sqrt', 'log2'}, default=None
    The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If 'auto', then `max_features=sqrt(n_features)`.
        - If 'sqrt', then `max_features=sqrt(n_features)`.
        - If 'log2', then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

    Note: the search for a split does not stop until at least one
    valid partition of the node samples is found, even if it requires to
    effectively inspect more than ``max_features`` features.

random_state : int or RandomState, default=None
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

max_leaf_nodes : int, default=None
    Grow a tree with ``max_leaf_nodes`` in best-first fashion.
    Best nodes are defined as relative reduction in impurity.
    If None then unlimited number of leaf nodes.

min_impurity_decrease : float, default=0.0
    A node will be split if this split induces a decrease of the impurity
    greater than or equal to this value.

    The weighted impurity decrease equation is the following::

        N_t / N * (impurity - N_t_R / N_t * right_impurity
                            - N_t_L / N_t * left_impurity)

    where ``N`` is the total number of samples, ``N_t`` is the number of
    samples at the current node, ``N_t_L`` is the number of samples in the
    left child, and ``N_t_R`` is the number of samples in the right child.

    ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
    if ``sample_weight`` is passed.

    .. versionadded:: 0.19

min_impurity_split : float, default=1e-7
    Threshold for early stopping in tree growth. A node will split
    if its impurity is above the threshold, otherwise it is a leaf.

    .. deprecated:: 0.19
       ``min_impurity_split`` has been deprecated in favor of
       ``min_impurity_decrease`` in 0.19. The default value of
       ``min_impurity_split`` will change from 1e-7 to 0 in 0.23 and it
       will be removed in 0.25. Use ``min_impurity_decrease`` instead.

class_weight : dict, list of dict or 'balanced', default=None
    Weights associated with classes in the form ``{class_label: weight}``.
    If None, all classes are supposed to have weight one. For
    multi-output problems, a list of dicts can be provided in the same
    order as the columns of y.

    Note that for multioutput (including multilabel) weights should be
    defined for each class of every column in its own dict. For example,
    for four-class multilabel classification weights should be
    [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
    [{1:1}, {2:5}, {3:1}, {4:1}].

    The 'balanced' mode uses the values of y to automatically adjust
    weights inversely proportional to class frequencies in the input data
    as ``n_samples / (n_classes * np.bincount(y))``

    For multi-output, the weights of each column of y will be multiplied.

    Note that these weights will be multiplied with sample_weight (passed
    through the fit method) if sample_weight is specified.

presort : deprecated, default='deprecated'
    This parameter is deprecated and will be removed in v0.24.

    .. deprecated:: 0.22

ccp_alpha : non-negative float, default=0.0
    Complexity parameter used for Minimal Cost-Complexity Pruning. The
    subtree with the largest cost complexity that is smaller than
    ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
    :ref:`minimal_cost_complexity_pruning` for details.

    .. versionadded:: 0.22

Attributes
----------
classes_ : ndarray of shape (n_classes,) or list of ndarray
    The classes labels (single output problem),
    or a list of arrays of class labels (multi-output problem).

feature_importances_ : ndarray of shape (n_features,)
    The feature importances. The higher, the more important the
    feature. The importance of a feature is computed as the (normalized)
    total reduction of the criterion brought by that feature.  It is also
    known as the Gini importance [4]_.

max_features_ : int
    The inferred value of max_features.

n_classes_ : int or list of int
    The number of classes (for single output problems),
    or a list containing the number of classes for each
    output (for multi-output problems).

n_features_ : int
    The number of features when ``fit`` is performed.

n_outputs_ : int
    The number of outputs when ``fit`` is performed.

tree_ : Tree
    The underlying Tree object. Please refer to
    ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
    :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
    for basic usage of these attributes.

See Also
--------
DecisionTreeRegressor : A decision tree regressor.

Notes
-----
The default values for the parameters controlling the size of the trees
(e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
unpruned trees which can potentially be very large on some data sets. To
reduce memory consumption, the complexity and size of the trees should be
controlled by setting those parameter values.

The features are always randomly permuted at each split. Therefore,
the best found split may vary, even with the same training data and
``max_features=n_features``, if the improvement of the criterion is
identical for several splits enumerated during the search of the best
split. To obtain a deterministic behaviour during fitting,
``random_state`` has to be fixed.

References
----------

.. [1] https://en.wikipedia.org/wiki/Decision_tree_learning

.. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, 'Classification
       and Regression Trees', Wadsworth, Belmont, CA, 1984.

.. [3] T. Hastie, R. Tibshirani and J. Friedman. 'Elements of Statistical
       Learning', Springer, 2009.

.. [4] L. Breiman, and A. Cutler, 'Random Forests',
       https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

Examples
--------
>>> from sklearn.datasets import load_iris
>>> from sklearn.model_selection import cross_val_score
>>> from sklearn.tree import DecisionTreeClassifier
>>> clf = DecisionTreeClassifier(random_state=0)
>>> iris = load_iris()
>>> cross_val_score(clf, iris.data, iris.target, cv=10)
...                             # doctest: +SKIP
...
array([ 1.     ,  0.93...,  0.86...,  0.93...,  0.93...,
        0.93...,  0.93...,  1.     ,  0.93...,  1.      ])
*)

val apply : ?check_input:bool -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Return the index of the leaf that each sample is predicted as.

.. versionadded:: 0.17

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csr_matrix``.

check_input : bool, default=True
    Allow to bypass several input checking.
    Don't use this parameter unless you know what you do.

Returns
-------
X_leaves : array-like of shape (n_samples,)
    For each datapoint x in X, return the index of the leaf x
    ends up in. Leaves are numbered within
    ``[0; self.tree_.node_count)``, possibly with gaps in the
    numbering.
*)

val cost_complexity_pruning_path : ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> (Py.Object.t * [>`ArrayLike] Np.Obj.t * [>`ArrayLike] Np.Obj.t)
(**
Compute the pruning path during Minimal Cost-Complexity Pruning.

See :ref:`minimal_cost_complexity_pruning` for details on the pruning
process.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The training input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csc_matrix``.

y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    The target values (class labels) as integers or strings.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights. If None, then samples are equally weighted. Splits
    that would create child nodes with net zero or negative weight are
    ignored while searching for a split in each node. Splits are also
    ignored if they would result in any single class carrying a
    negative weight in either child node.

Returns
-------
ccp_path : Bunch
    Dictionary-like object, with attributes:

    ccp_alphas : ndarray
        Effective alphas of subtree during pruning.

    impurities : ndarray
        Sum of the impurities of the subtree leaves for the
        corresponding alpha value in ``ccp_alphas``.
*)

val decision_path : ?check_input:bool -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Object|`Spmatrix] Np.Obj.t
(**
Return the decision path in the tree.

.. versionadded:: 0.18

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csr_matrix``.

check_input : bool, default=True
    Allow to bypass several input checking.
    Don't use this parameter unless you know what you do.

Returns
-------
indicator : sparse matrix of shape (n_samples, n_nodes)
    Return a node indicator CSR matrix where non zero elements
    indicates that the samples goes through the nodes.
*)

val fit : ?sample_weight:[>`ArrayLike] Np.Obj.t -> ?check_input:bool -> ?x_idx_sorted:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Build a decision tree classifier from the training set (X, y).

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The training input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csc_matrix``.

y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    The target values (class labels) as integers or strings.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights. If None, then samples are equally weighted. Splits
    that would create child nodes with net zero or negative weight are
    ignored while searching for a split in each node. Splits are also
    ignored if they would result in any single class carrying a
    negative weight in either child node.

check_input : bool, default=True
    Allow to bypass several input checking.
    Don't use this parameter unless you know what you do.

X_idx_sorted : array-like of shape (n_samples, n_features),                 default=None
    The indexes of the sorted training input samples. If many tree
    are grown on the same dataset, this allows the ordering to be
    cached between trees. If None, the data will be sorted here.
    Don't use this parameter unless you know what to do.

Returns
-------
self : DecisionTreeClassifier
    Fitted estimator.
*)

val get_depth : [> tag] Obj.t -> Py.Object.t
(**
Return the depth of the decision tree.

The depth of a tree is the maximum distance between the root
and any leaf.

Returns
-------
self.tree_.max_depth : int
    The maximum depth of the tree.
*)

val get_n_leaves : [> tag] Obj.t -> Py.Object.t
(**
Return the number of leaves of the decision tree.

Returns
-------
self.tree_.n_leaves : int
    Number of leaves.
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

val predict : ?check_input:bool -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Predict class or regression value for X.

For a classification model, the predicted class for each sample in X is
returned. For a regression model, the predicted value based on X is
returned.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csr_matrix``.

check_input : bool, default=True
    Allow to bypass several input checking.
    Don't use this parameter unless you know what you do.

Returns
-------
y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    The predicted classes, or the predict values.
*)

val predict_log_proba : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Predict class log-probabilities of the input samples X.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csr_matrix``.

Returns
-------
proba : ndarray of shape (n_samples, n_classes) or list of n_outputs             such arrays if n_outputs > 1
    The class log-probabilities of the input samples. The order of the
    classes corresponds to that in the attribute :term:`classes_`.
*)

val predict_proba : ?check_input:bool -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Predict class probabilities of the input samples X.

The predicted class probability is the fraction of samples of the same
class in a leaf.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csr_matrix``.

check_input : bool, default=True
    Allow to bypass several input checking.
    Don't use this parameter unless you know what you do.

Returns
-------
proba : ndarray of shape (n_samples, n_classes) or list of n_outputs             such arrays if n_outputs > 1
    The class probabilities of the input samples. The order of the
    classes corresponds to that in the attribute :term:`classes_`.
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


(** Attribute feature_importances_: get value or raise Not_found if None.*)
val feature_importances_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute feature_importances_: get value as an option. *)
val feature_importances_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute max_features_: get value or raise Not_found if None.*)
val max_features_ : t -> int

(** Attribute max_features_: get value as an option. *)
val max_features_opt : t -> (int) option


(** Attribute n_classes_: get value or raise Not_found if None.*)
val n_classes_ : t -> Py.Object.t

(** Attribute n_classes_: get value as an option. *)
val n_classes_opt : t -> (Py.Object.t) option


(** Attribute n_features_: get value or raise Not_found if None.*)
val n_features_ : t -> int

(** Attribute n_features_: get value as an option. *)
val n_features_opt : t -> (int) option


(** Attribute n_outputs_: get value or raise Not_found if None.*)
val n_outputs_ : t -> int

(** Attribute n_outputs_: get value as an option. *)
val n_outputs_opt : t -> (int) option


(** Attribute tree_: get value or raise Not_found if None.*)
val tree_ : t -> Py.Object.t

(** Attribute tree_: get value as an option. *)
val tree_opt : t -> (Py.Object.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module DecisionTreeRegressor : sig
type tag = [`DecisionTreeRegressor]
type t = [`BaseDecisionTree | `BaseEstimator | `DecisionTreeRegressor | `MultiOutputMixin | `Object | `RegressorMixin] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_regressor : t -> [`RegressorMixin] Obj.t
val as_multi_output : t -> [`MultiOutputMixin] Obj.t
val as_estimator : t -> [`BaseEstimator] Obj.t
val as_decision_tree : t -> [`BaseDecisionTree] Obj.t
val create : ?criterion:[`Mse | `Friedman_mse | `Mae] -> ?splitter:[`Best | `Random] -> ?max_depth:int -> ?min_samples_split:[`F of float | `I of int] -> ?min_samples_leaf:[`F of float | `I of int] -> ?min_weight_fraction_leaf:float -> ?max_features:[`I of int | `Log2 | `F of float | `Auto | `Sqrt] -> ?random_state:int -> ?max_leaf_nodes:int -> ?min_impurity_decrease:float -> ?min_impurity_split:float -> ?presort:Py.Object.t -> ?ccp_alpha:float -> unit -> t
(**
A decision tree regressor.

Read more in the :ref:`User Guide <tree>`.

Parameters
----------
criterion : {'mse', 'friedman_mse', 'mae'}, default='mse'
    The function to measure the quality of a split. Supported criteria
    are 'mse' for the mean squared error, which is equal to variance
    reduction as feature selection criterion and minimizes the L2 loss
    using the mean of each terminal node, 'friedman_mse', which uses mean
    squared error with Friedman's improvement score for potential splits,
    and 'mae' for the mean absolute error, which minimizes the L1 loss
    using the median of each terminal node.

    .. versionadded:: 0.18
       Mean Absolute Error (MAE) criterion.

splitter : {'best', 'random'}, default='best'
    The strategy used to choose the split at each node. Supported
    strategies are 'best' to choose the best split and 'random' to choose
    the best random split.

max_depth : int, default=None
    The maximum depth of the tree. If None, then nodes are expanded until
    all leaves are pure or until all leaves contain less than
    min_samples_split samples.

min_samples_split : int or float, default=2
    The minimum number of samples required to split an internal node:

    - If int, then consider `min_samples_split` as the minimum number.
    - If float, then `min_samples_split` is a fraction and
      `ceil(min_samples_split * n_samples)` are the minimum
      number of samples for each split.

    .. versionchanged:: 0.18
       Added float values for fractions.

min_samples_leaf : int or float, default=1
    The minimum number of samples required to be at a leaf node.
    A split point at any depth will only be considered if it leaves at
    least ``min_samples_leaf`` training samples in each of the left and
    right branches.  This may have the effect of smoothing the model,
    especially in regression.

    - If int, then consider `min_samples_leaf` as the minimum number.
    - If float, then `min_samples_leaf` is a fraction and
      `ceil(min_samples_leaf * n_samples)` are the minimum
      number of samples for each node.

    .. versionchanged:: 0.18
       Added float values for fractions.

min_weight_fraction_leaf : float, default=0.0
    The minimum weighted fraction of the sum total of weights (of all
    the input samples) required to be at a leaf node. Samples have
    equal weight when sample_weight is not provided.

max_features : int, float or {'auto', 'sqrt', 'log2'}, default=None
    The number of features to consider when looking for the best split:

    - If int, then consider `max_features` features at each split.
    - If float, then `max_features` is a fraction and
      `int(max_features * n_features)` features are considered at each
      split.
    - If 'auto', then `max_features=n_features`.
    - If 'sqrt', then `max_features=sqrt(n_features)`.
    - If 'log2', then `max_features=log2(n_features)`.
    - If None, then `max_features=n_features`.

    Note: the search for a split does not stop until at least one
    valid partition of the node samples is found, even if it requires to
    effectively inspect more than ``max_features`` features.

random_state : int or RandomState, default=None
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

max_leaf_nodes : int, default=None
    Grow a tree with ``max_leaf_nodes`` in best-first fashion.
    Best nodes are defined as relative reduction in impurity.
    If None then unlimited number of leaf nodes.

min_impurity_decrease : float, default=0.0
    A node will be split if this split induces a decrease of the impurity
    greater than or equal to this value.

    The weighted impurity decrease equation is the following::

        N_t / N * (impurity - N_t_R / N_t * right_impurity
                            - N_t_L / N_t * left_impurity)

    where ``N`` is the total number of samples, ``N_t`` is the number of
    samples at the current node, ``N_t_L`` is the number of samples in the
    left child, and ``N_t_R`` is the number of samples in the right child.

    ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
    if ``sample_weight`` is passed.

    .. versionadded:: 0.19

min_impurity_split : float, (default=1e-7)
    Threshold for early stopping in tree growth. A node will split
    if its impurity is above the threshold, otherwise it is a leaf.

    .. deprecated:: 0.19
       ``min_impurity_split`` has been deprecated in favor of
       ``min_impurity_decrease`` in 0.19. The default value of
       ``min_impurity_split`` will change from 1e-7 to 0 in 0.23 and it
       will be removed in 0.25. Use ``min_impurity_decrease`` instead.

presort : deprecated, default='deprecated'
    This parameter is deprecated and will be removed in v0.24.

    .. deprecated:: 0.22

ccp_alpha : non-negative float, default=0.0
    Complexity parameter used for Minimal Cost-Complexity Pruning. The
    subtree with the largest cost complexity that is smaller than
    ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
    :ref:`minimal_cost_complexity_pruning` for details.

    .. versionadded:: 0.22

Attributes
----------
feature_importances_ : ndarray of shape (n_features,)
    The feature importances.
    The higher, the more important the feature.
    The importance of a feature is computed as the
    (normalized) total reduction of the criterion brought
    by that feature. It is also known as the Gini importance [4]_.

max_features_ : int
    The inferred value of max_features.

n_features_ : int
    The number of features when ``fit`` is performed.

n_outputs_ : int
    The number of outputs when ``fit`` is performed.

tree_ : Tree
    The underlying Tree object. Please refer to
    ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
    :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
    for basic usage of these attributes.

See Also
--------
DecisionTreeClassifier : A decision tree classifier.

Notes
-----
The default values for the parameters controlling the size of the trees
(e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
unpruned trees which can potentially be very large on some data sets. To
reduce memory consumption, the complexity and size of the trees should be
controlled by setting those parameter values.

The features are always randomly permuted at each split. Therefore,
the best found split may vary, even with the same training data and
``max_features=n_features``, if the improvement of the criterion is
identical for several splits enumerated during the search of the best
split. To obtain a deterministic behaviour during fitting,
``random_state`` has to be fixed.

References
----------

.. [1] https://en.wikipedia.org/wiki/Decision_tree_learning

.. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, 'Classification
       and Regression Trees', Wadsworth, Belmont, CA, 1984.

.. [3] T. Hastie, R. Tibshirani and J. Friedman. 'Elements of Statistical
       Learning', Springer, 2009.

.. [4] L. Breiman, and A. Cutler, 'Random Forests',
       https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

Examples
--------
>>> from sklearn.datasets import load_boston
>>> from sklearn.model_selection import cross_val_score
>>> from sklearn.tree import DecisionTreeRegressor
>>> X, y = load_boston(return_X_y=True)
>>> regressor = DecisionTreeRegressor(random_state=0)
>>> cross_val_score(regressor, X, y, cv=10)
...                    # doctest: +SKIP
...
array([ 0.61..., 0.57..., -0.34..., 0.41..., 0.75...,
        0.07..., 0.29..., 0.33..., -1.42..., -1.77...])
*)

val apply : ?check_input:bool -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Return the index of the leaf that each sample is predicted as.

.. versionadded:: 0.17

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csr_matrix``.

check_input : bool, default=True
    Allow to bypass several input checking.
    Don't use this parameter unless you know what you do.

Returns
-------
X_leaves : array-like of shape (n_samples,)
    For each datapoint x in X, return the index of the leaf x
    ends up in. Leaves are numbered within
    ``[0; self.tree_.node_count)``, possibly with gaps in the
    numbering.
*)

val cost_complexity_pruning_path : ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> (Py.Object.t * [>`ArrayLike] Np.Obj.t * [>`ArrayLike] Np.Obj.t)
(**
Compute the pruning path during Minimal Cost-Complexity Pruning.

See :ref:`minimal_cost_complexity_pruning` for details on the pruning
process.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The training input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csc_matrix``.

y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    The target values (class labels) as integers or strings.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights. If None, then samples are equally weighted. Splits
    that would create child nodes with net zero or negative weight are
    ignored while searching for a split in each node. Splits are also
    ignored if they would result in any single class carrying a
    negative weight in either child node.

Returns
-------
ccp_path : Bunch
    Dictionary-like object, with attributes:

    ccp_alphas : ndarray
        Effective alphas of subtree during pruning.

    impurities : ndarray
        Sum of the impurities of the subtree leaves for the
        corresponding alpha value in ``ccp_alphas``.
*)

val decision_path : ?check_input:bool -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Object|`Spmatrix] Np.Obj.t
(**
Return the decision path in the tree.

.. versionadded:: 0.18

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csr_matrix``.

check_input : bool, default=True
    Allow to bypass several input checking.
    Don't use this parameter unless you know what you do.

Returns
-------
indicator : sparse matrix of shape (n_samples, n_nodes)
    Return a node indicator CSR matrix where non zero elements
    indicates that the samples goes through the nodes.
*)

val fit : ?sample_weight:[>`ArrayLike] Np.Obj.t -> ?check_input:bool -> ?x_idx_sorted:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Build a decision tree regressor from the training set (X, y).

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The training input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csc_matrix``.

y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    The target values (real numbers). Use ``dtype=np.float64`` and
    ``order='C'`` for maximum efficiency.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights. If None, then samples are equally weighted. Splits
    that would create child nodes with net zero or negative weight are
    ignored while searching for a split in each node.

check_input : bool, default=True
    Allow to bypass several input checking.
    Don't use this parameter unless you know what you do.

X_idx_sorted : array-like of shape (n_samples, n_features),             default=None
    The indexes of the sorted training input samples. If many tree
    are grown on the same dataset, this allows the ordering to be
    cached between trees. If None, the data will be sorted here.
    Don't use this parameter unless you know what to do.

Returns
-------
self : DecisionTreeRegressor
    Fitted estimator.
*)

val get_depth : [> tag] Obj.t -> Py.Object.t
(**
Return the depth of the decision tree.

The depth of a tree is the maximum distance between the root
and any leaf.

Returns
-------
self.tree_.max_depth : int
    The maximum depth of the tree.
*)

val get_n_leaves : [> tag] Obj.t -> Py.Object.t
(**
Return the number of leaves of the decision tree.

Returns
-------
self.tree_.n_leaves : int
    Number of leaves.
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

val predict : ?check_input:bool -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Predict class or regression value for X.

For a classification model, the predicted class for each sample in X is
returned. For a regression model, the predicted value based on X is
returned.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csr_matrix``.

check_input : bool, default=True
    Allow to bypass several input checking.
    Don't use this parameter unless you know what you do.

Returns
-------
y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    The predicted classes, or the predict values.
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


(** Attribute feature_importances_: get value or raise Not_found if None.*)
val feature_importances_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute feature_importances_: get value as an option. *)
val feature_importances_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute max_features_: get value or raise Not_found if None.*)
val max_features_ : t -> int

(** Attribute max_features_: get value as an option. *)
val max_features_opt : t -> (int) option


(** Attribute n_features_: get value or raise Not_found if None.*)
val n_features_ : t -> int

(** Attribute n_features_: get value as an option. *)
val n_features_opt : t -> (int) option


(** Attribute n_outputs_: get value or raise Not_found if None.*)
val n_outputs_ : t -> int

(** Attribute n_outputs_: get value as an option. *)
val n_outputs_opt : t -> (int) option


(** Attribute tree_: get value or raise Not_found if None.*)
val tree_ : t -> Py.Object.t

(** Attribute tree_: get value as an option. *)
val tree_opt : t -> (Py.Object.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module ExtraTreeClassifier : sig
type tag = [`ExtraTreeClassifier]
type t = [`BaseDecisionTree | `BaseEstimator | `ClassifierMixin | `DecisionTreeClassifier | `ExtraTreeClassifier | `MultiOutputMixin | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_classifier : t -> [`ClassifierMixin] Obj.t
val as_multi_output : t -> [`MultiOutputMixin] Obj.t
val as_decision_tree_classifier : t -> [`DecisionTreeClassifier] Obj.t
val as_decision_tree : t -> [`BaseDecisionTree] Obj.t
val as_estimator : t -> [`BaseEstimator] Obj.t
val create : ?criterion:[`Gini | `Entropy] -> ?splitter:[`Random | `Best] -> ?max_depth:int -> ?min_samples_split:[`F of float | `I of int] -> ?min_samples_leaf:[`F of float | `I of int] -> ?min_weight_fraction_leaf:float -> ?max_features:[`I of int | `F of float | `Sqrt | `PyObject of Py.Object.t | `None] -> ?random_state:int -> ?max_leaf_nodes:int -> ?min_impurity_decrease:float -> ?min_impurity_split:float -> ?class_weight:[`List_of_dict of Py.Object.t | `DictIntToFloat of (int * float) list | `Balanced] -> ?ccp_alpha:float -> unit -> t
(**
An extremely randomized tree classifier.

Extra-trees differ from classic decision trees in the way they are built.
When looking for the best split to separate the samples of a node into two
groups, random splits are drawn for each of the `max_features` randomly
selected features and the best split among those is chosen. When
`max_features` is set 1, this amounts to building a totally random
decision tree.

Warning: Extra-trees should only be used within ensemble methods.

Read more in the :ref:`User Guide <tree>`.

Parameters
----------
criterion : {'gini', 'entropy'}, default='gini'
    The function to measure the quality of a split. Supported criteria are
    'gini' for the Gini impurity and 'entropy' for the information gain.

splitter : {'random', 'best'}, default='random'
    The strategy used to choose the split at each node. Supported
    strategies are 'best' to choose the best split and 'random' to choose
    the best random split.

max_depth : int, default=None
    The maximum depth of the tree. If None, then nodes are expanded until
    all leaves are pure or until all leaves contain less than
    min_samples_split samples.

min_samples_split : int or float, default=2
    The minimum number of samples required to split an internal node:

    - If int, then consider `min_samples_split` as the minimum number.
    - If float, then `min_samples_split` is a fraction and
      `ceil(min_samples_split * n_samples)` are the minimum
      number of samples for each split.

    .. versionchanged:: 0.18
       Added float values for fractions.

min_samples_leaf : int or float, default=1
    The minimum number of samples required to be at a leaf node.
    A split point at any depth will only be considered if it leaves at
    least ``min_samples_leaf`` training samples in each of the left and
    right branches.  This may have the effect of smoothing the model,
    especially in regression.

    - If int, then consider `min_samples_leaf` as the minimum number.
    - If float, then `min_samples_leaf` is a fraction and
      `ceil(min_samples_leaf * n_samples)` are the minimum
      number of samples for each node.

    .. versionchanged:: 0.18
       Added float values for fractions.

min_weight_fraction_leaf : float, default=0.0
    The minimum weighted fraction of the sum total of weights (of all
    the input samples) required to be at a leaf node. Samples have
    equal weight when sample_weight is not provided.

max_features : int, float, {'auto', 'sqrt', 'log2'} or None, default='auto'
    The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If 'auto', then `max_features=sqrt(n_features)`.
        - If 'sqrt', then `max_features=sqrt(n_features)`.
        - If 'log2', then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

    Note: the search for a split does not stop until at least one
    valid partition of the node samples is found, even if it requires to
    effectively inspect more than ``max_features`` features.

random_state : int or RandomState, default=None
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

max_leaf_nodes : int, default=None
    Grow a tree with ``max_leaf_nodes`` in best-first fashion.
    Best nodes are defined as relative reduction in impurity.
    If None then unlimited number of leaf nodes.

min_impurity_decrease : float, default=0.0
    A node will be split if this split induces a decrease of the impurity
    greater than or equal to this value.

    The weighted impurity decrease equation is the following::

        N_t / N * (impurity - N_t_R / N_t * right_impurity
                            - N_t_L / N_t * left_impurity)

    where ``N`` is the total number of samples, ``N_t`` is the number of
    samples at the current node, ``N_t_L`` is the number of samples in the
    left child, and ``N_t_R`` is the number of samples in the right child.

    ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
    if ``sample_weight`` is passed.

    .. versionadded:: 0.19

min_impurity_split : float, (default=1e-7)
    Threshold for early stopping in tree growth. A node will split
    if its impurity is above the threshold, otherwise it is a leaf.

    .. deprecated:: 0.19
       ``min_impurity_split`` has been deprecated in favor of
       ``min_impurity_decrease`` in 0.19. The default value of
       ``min_impurity_split`` will change from 1e-7 to 0 in 0.23 and it
       will be removed in 0.25. Use ``min_impurity_decrease`` instead.

class_weight : dict, list of dict or 'balanced', default=None
    Weights associated with classes in the form ``{class_label: weight}``.
    If None, all classes are supposed to have weight one. For
    multi-output problems, a list of dicts can be provided in the same
    order as the columns of y.

    Note that for multioutput (including multilabel) weights should be
    defined for each class of every column in its own dict. For example,
    for four-class multilabel classification weights should be
    [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
    [{1:1}, {2:5}, {3:1}, {4:1}].

    The 'balanced' mode uses the values of y to automatically adjust
    weights inversely proportional to class frequencies in the input data
    as ``n_samples / (n_classes * np.bincount(y))``

    For multi-output, the weights of each column of y will be multiplied.

    Note that these weights will be multiplied with sample_weight (passed
    through the fit method) if sample_weight is specified.

ccp_alpha : non-negative float, default=0.0
    Complexity parameter used for Minimal Cost-Complexity Pruning. The
    subtree with the largest cost complexity that is smaller than
    ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
    :ref:`minimal_cost_complexity_pruning` for details.

    .. versionadded:: 0.22

Attributes
----------
classes_ : ndarray of shape (n_classes,) or list of ndarray
    The classes labels (single output problem),
    or a list of arrays of class labels (multi-output problem).

max_features_ : int
    The inferred value of max_features.

n_classes_ : int or list of int
    The number of classes (for single output problems),
    or a list containing the number of classes for each
    output (for multi-output problems).

feature_importances_ : ndarray of shape (n_features,)
    Return the feature importances (the higher, the more important the
    feature).

n_features_ : int
    The number of features when ``fit`` is performed.

n_outputs_ : int
    The number of outputs when ``fit`` is performed.

tree_ : Tree
    The underlying Tree object. Please refer to
    ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
    :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
    for basic usage of these attributes.

See Also
--------
ExtraTreeRegressor : An extremely randomized tree regressor.
sklearn.ensemble.ExtraTreesClassifier : An extra-trees classifier.
sklearn.ensemble.ExtraTreesRegressor : An extra-trees regressor.

Notes
-----
The default values for the parameters controlling the size of the trees
(e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
unpruned trees which can potentially be very large on some data sets. To
reduce memory consumption, the complexity and size of the trees should be
controlled by setting those parameter values.

References
----------

.. [1] P. Geurts, D. Ernst., and L. Wehenkel, 'Extremely randomized trees',
       Machine Learning, 63(1), 3-42, 2006.
*)

val apply : ?check_input:bool -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Return the index of the leaf that each sample is predicted as.

.. versionadded:: 0.17

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csr_matrix``.

check_input : bool, default=True
    Allow to bypass several input checking.
    Don't use this parameter unless you know what you do.

Returns
-------
X_leaves : array-like of shape (n_samples,)
    For each datapoint x in X, return the index of the leaf x
    ends up in. Leaves are numbered within
    ``[0; self.tree_.node_count)``, possibly with gaps in the
    numbering.
*)

val cost_complexity_pruning_path : ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> (Py.Object.t * [>`ArrayLike] Np.Obj.t * [>`ArrayLike] Np.Obj.t)
(**
Compute the pruning path during Minimal Cost-Complexity Pruning.

See :ref:`minimal_cost_complexity_pruning` for details on the pruning
process.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The training input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csc_matrix``.

y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    The target values (class labels) as integers or strings.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights. If None, then samples are equally weighted. Splits
    that would create child nodes with net zero or negative weight are
    ignored while searching for a split in each node. Splits are also
    ignored if they would result in any single class carrying a
    negative weight in either child node.

Returns
-------
ccp_path : Bunch
    Dictionary-like object, with attributes:

    ccp_alphas : ndarray
        Effective alphas of subtree during pruning.

    impurities : ndarray
        Sum of the impurities of the subtree leaves for the
        corresponding alpha value in ``ccp_alphas``.
*)

val decision_path : ?check_input:bool -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Object|`Spmatrix] Np.Obj.t
(**
Return the decision path in the tree.

.. versionadded:: 0.18

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csr_matrix``.

check_input : bool, default=True
    Allow to bypass several input checking.
    Don't use this parameter unless you know what you do.

Returns
-------
indicator : sparse matrix of shape (n_samples, n_nodes)
    Return a node indicator CSR matrix where non zero elements
    indicates that the samples goes through the nodes.
*)

val fit : ?sample_weight:[>`ArrayLike] Np.Obj.t -> ?check_input:bool -> ?x_idx_sorted:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Build a decision tree classifier from the training set (X, y).

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The training input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csc_matrix``.

y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    The target values (class labels) as integers or strings.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights. If None, then samples are equally weighted. Splits
    that would create child nodes with net zero or negative weight are
    ignored while searching for a split in each node. Splits are also
    ignored if they would result in any single class carrying a
    negative weight in either child node.

check_input : bool, default=True
    Allow to bypass several input checking.
    Don't use this parameter unless you know what you do.

X_idx_sorted : array-like of shape (n_samples, n_features),                 default=None
    The indexes of the sorted training input samples. If many tree
    are grown on the same dataset, this allows the ordering to be
    cached between trees. If None, the data will be sorted here.
    Don't use this parameter unless you know what to do.

Returns
-------
self : DecisionTreeClassifier
    Fitted estimator.
*)

val get_depth : [> tag] Obj.t -> Py.Object.t
(**
Return the depth of the decision tree.

The depth of a tree is the maximum distance between the root
and any leaf.

Returns
-------
self.tree_.max_depth : int
    The maximum depth of the tree.
*)

val get_n_leaves : [> tag] Obj.t -> Py.Object.t
(**
Return the number of leaves of the decision tree.

Returns
-------
self.tree_.n_leaves : int
    Number of leaves.
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

val predict : ?check_input:bool -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Predict class or regression value for X.

For a classification model, the predicted class for each sample in X is
returned. For a regression model, the predicted value based on X is
returned.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csr_matrix``.

check_input : bool, default=True
    Allow to bypass several input checking.
    Don't use this parameter unless you know what you do.

Returns
-------
y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    The predicted classes, or the predict values.
*)

val predict_log_proba : x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Predict class log-probabilities of the input samples X.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csr_matrix``.

Returns
-------
proba : ndarray of shape (n_samples, n_classes) or list of n_outputs             such arrays if n_outputs > 1
    The class log-probabilities of the input samples. The order of the
    classes corresponds to that in the attribute :term:`classes_`.
*)

val predict_proba : ?check_input:bool -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Predict class probabilities of the input samples X.

The predicted class probability is the fraction of samples of the same
class in a leaf.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csr_matrix``.

check_input : bool, default=True
    Allow to bypass several input checking.
    Don't use this parameter unless you know what you do.

Returns
-------
proba : ndarray of shape (n_samples, n_classes) or list of n_outputs             such arrays if n_outputs > 1
    The class probabilities of the input samples. The order of the
    classes corresponds to that in the attribute :term:`classes_`.
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


(** Attribute max_features_: get value or raise Not_found if None.*)
val max_features_ : t -> int

(** Attribute max_features_: get value as an option. *)
val max_features_opt : t -> (int) option


(** Attribute n_classes_: get value or raise Not_found if None.*)
val n_classes_ : t -> Py.Object.t

(** Attribute n_classes_: get value as an option. *)
val n_classes_opt : t -> (Py.Object.t) option


(** Attribute feature_importances_: get value or raise Not_found if None.*)
val feature_importances_ : t -> [>`ArrayLike] Np.Obj.t

(** Attribute feature_importances_: get value as an option. *)
val feature_importances_opt : t -> ([>`ArrayLike] Np.Obj.t) option


(** Attribute n_features_: get value or raise Not_found if None.*)
val n_features_ : t -> int

(** Attribute n_features_: get value as an option. *)
val n_features_opt : t -> (int) option


(** Attribute n_outputs_: get value or raise Not_found if None.*)
val n_outputs_ : t -> int

(** Attribute n_outputs_: get value as an option. *)
val n_outputs_opt : t -> (int) option


(** Attribute tree_: get value or raise Not_found if None.*)
val tree_ : t -> Py.Object.t

(** Attribute tree_: get value as an option. *)
val tree_opt : t -> (Py.Object.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module ExtraTreeRegressor : sig
type tag = [`ExtraTreeRegressor]
type t = [`BaseDecisionTree | `BaseEstimator | `DecisionTreeRegressor | `ExtraTreeRegressor | `MultiOutputMixin | `Object | `RegressorMixin] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_regressor : t -> [`RegressorMixin] Obj.t
val as_multi_output : t -> [`MultiOutputMixin] Obj.t
val as_decision_tree : t -> [`BaseDecisionTree] Obj.t
val as_estimator : t -> [`BaseEstimator] Obj.t
val as_decision_tree_regressor : t -> [`DecisionTreeRegressor] Obj.t
val create : ?criterion:[`Mse | `Friedman_mse | `Mae] -> ?splitter:[`Random | `Best] -> ?max_depth:int -> ?min_samples_split:[`F of float | `I of int] -> ?min_samples_leaf:[`F of float | `I of int] -> ?min_weight_fraction_leaf:float -> ?max_features:[`I of int | `F of float | `Sqrt | `PyObject of Py.Object.t | `None] -> ?random_state:int -> ?min_impurity_decrease:float -> ?min_impurity_split:float -> ?max_leaf_nodes:int -> ?ccp_alpha:float -> unit -> t
(**
An extremely randomized tree regressor.

Extra-trees differ from classic decision trees in the way they are built.
When looking for the best split to separate the samples of a node into two
groups, random splits are drawn for each of the `max_features` randomly
selected features and the best split among those is chosen. When
`max_features` is set 1, this amounts to building a totally random
decision tree.

Warning: Extra-trees should only be used within ensemble methods.

Read more in the :ref:`User Guide <tree>`.

Parameters
----------
criterion : {'mse', 'friedman_mse', 'mae'}, default='mse'
    The function to measure the quality of a split. Supported criteria
    are 'mse' for the mean squared error, which is equal to variance
    reduction as feature selection criterion, and 'mae' for the mean
    absolute error.

    .. versionadded:: 0.18
       Mean Absolute Error (MAE) criterion.

splitter : {'random', 'best'}, default='random'
    The strategy used to choose the split at each node. Supported
    strategies are 'best' to choose the best split and 'random' to choose
    the best random split.

max_depth : int, default=None
    The maximum depth of the tree. If None, then nodes are expanded until
    all leaves are pure or until all leaves contain less than
    min_samples_split samples.

min_samples_split : int or float, default=2
    The minimum number of samples required to split an internal node:

    - If int, then consider `min_samples_split` as the minimum number.
    - If float, then `min_samples_split` is a fraction and
      `ceil(min_samples_split * n_samples)` are the minimum
      number of samples for each split.

    .. versionchanged:: 0.18
       Added float values for fractions.

min_samples_leaf : int or float, default=1
    The minimum number of samples required to be at a leaf node.
    A split point at any depth will only be considered if it leaves at
    least ``min_samples_leaf`` training samples in each of the left and
    right branches.  This may have the effect of smoothing the model,
    especially in regression.

    - If int, then consider `min_samples_leaf` as the minimum number.
    - If float, then `min_samples_leaf` is a fraction and
      `ceil(min_samples_leaf * n_samples)` are the minimum
      number of samples for each node.

    .. versionchanged:: 0.18
       Added float values for fractions.

min_weight_fraction_leaf : float, default=0.0
    The minimum weighted fraction of the sum total of weights (of all
    the input samples) required to be at a leaf node. Samples have
    equal weight when sample_weight is not provided.

max_features : int, float, {'auto', 'sqrt', 'log2'} or None, default='auto'
    The number of features to consider when looking for the best split:

    - If int, then consider `max_features` features at each split.
    - If float, then `max_features` is a fraction and
      `int(max_features * n_features)` features are considered at each
      split.
    - If 'auto', then `max_features=n_features`.
    - If 'sqrt', then `max_features=sqrt(n_features)`.
    - If 'log2', then `max_features=log2(n_features)`.
    - If None, then `max_features=n_features`.

    Note: the search for a split does not stop until at least one
    valid partition of the node samples is found, even if it requires to
    effectively inspect more than ``max_features`` features.

random_state : int or RandomState, default=None
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

min_impurity_decrease : float, default=0.0
    A node will be split if this split induces a decrease of the impurity
    greater than or equal to this value.

    The weighted impurity decrease equation is the following::

        N_t / N * (impurity - N_t_R / N_t * right_impurity
                            - N_t_L / N_t * left_impurity)

    where ``N`` is the total number of samples, ``N_t`` is the number of
    samples at the current node, ``N_t_L`` is the number of samples in the
    left child, and ``N_t_R`` is the number of samples in the right child.

    ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
    if ``sample_weight`` is passed.

    .. versionadded:: 0.19

min_impurity_split : float, (default=1e-7)
    Threshold for early stopping in tree growth. A node will split
    if its impurity is above the threshold, otherwise it is a leaf.

    .. deprecated:: 0.19
       ``min_impurity_split`` has been deprecated in favor of
       ``min_impurity_decrease`` in 0.19. The default value of
       ``min_impurity_split`` will change from 1e-7 to 0 in 0.23 and it
       will be removed in 0.25. Use ``min_impurity_decrease`` instead.

max_leaf_nodes : int, default=None
    Grow a tree with ``max_leaf_nodes`` in best-first fashion.
    Best nodes are defined as relative reduction in impurity.
    If None then unlimited number of leaf nodes.

ccp_alpha : non-negative float, default=0.0
    Complexity parameter used for Minimal Cost-Complexity Pruning. The
    subtree with the largest cost complexity that is smaller than
    ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
    :ref:`minimal_cost_complexity_pruning` for details.

    .. versionadded:: 0.22

Attributes
----------
max_features_ : int
    The inferred value of max_features.

n_features_ : int
    The number of features when ``fit`` is performed.

n_outputs_ : int
    The number of outputs when ``fit`` is performed.

tree_ : Tree
    The underlying Tree object. Please refer to
    ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
    :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
    for basic usage of these attributes.

See Also
--------
ExtraTreeClassifier : An extremely randomized tree classifier.
sklearn.ensemble.ExtraTreesClassifier : An extra-trees classifier.
sklearn.ensemble.ExtraTreesRegressor : An extra-trees regressor.

Notes
-----
The default values for the parameters controlling the size of the trees
(e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
unpruned trees which can potentially be very large on some data sets. To
reduce memory consumption, the complexity and size of the trees should be
controlled by setting those parameter values.

References
----------

.. [1] P. Geurts, D. Ernst., and L. Wehenkel, 'Extremely randomized trees',
       Machine Learning, 63(1), 3-42, 2006.

Examples
--------
>>> from sklearn.datasets import load_boston
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.ensemble import BaggingRegressor
>>> from sklearn.tree import ExtraTreeRegressor
>>> X, y = load_boston(return_X_y=True)
>>> X_train, X_test, y_train, y_test = train_test_split(
...     X, y, random_state=0)
>>> extra_tree = ExtraTreeRegressor(random_state=0)
>>> reg = BaggingRegressor(extra_tree, random_state=0).fit(
...     X_train, y_train)
>>> reg.score(X_test, y_test)
0.7823...
*)

val apply : ?check_input:bool -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Return the index of the leaf that each sample is predicted as.

.. versionadded:: 0.17

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csr_matrix``.

check_input : bool, default=True
    Allow to bypass several input checking.
    Don't use this parameter unless you know what you do.

Returns
-------
X_leaves : array-like of shape (n_samples,)
    For each datapoint x in X, return the index of the leaf x
    ends up in. Leaves are numbered within
    ``[0; self.tree_.node_count)``, possibly with gaps in the
    numbering.
*)

val cost_complexity_pruning_path : ?sample_weight:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> (Py.Object.t * [>`ArrayLike] Np.Obj.t * [>`ArrayLike] Np.Obj.t)
(**
Compute the pruning path during Minimal Cost-Complexity Pruning.

See :ref:`minimal_cost_complexity_pruning` for details on the pruning
process.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The training input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csc_matrix``.

y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    The target values (class labels) as integers or strings.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights. If None, then samples are equally weighted. Splits
    that would create child nodes with net zero or negative weight are
    ignored while searching for a split in each node. Splits are also
    ignored if they would result in any single class carrying a
    negative weight in either child node.

Returns
-------
ccp_path : Bunch
    Dictionary-like object, with attributes:

    ccp_alphas : ndarray
        Effective alphas of subtree during pruning.

    impurities : ndarray
        Sum of the impurities of the subtree leaves for the
        corresponding alpha value in ``ccp_alphas``.
*)

val decision_path : ?check_input:bool -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Object|`Spmatrix] Np.Obj.t
(**
Return the decision path in the tree.

.. versionadded:: 0.18

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csr_matrix``.

check_input : bool, default=True
    Allow to bypass several input checking.
    Don't use this parameter unless you know what you do.

Returns
-------
indicator : sparse matrix of shape (n_samples, n_nodes)
    Return a node indicator CSR matrix where non zero elements
    indicates that the samples goes through the nodes.
*)

val fit : ?sample_weight:[>`ArrayLike] Np.Obj.t -> ?check_input:bool -> ?x_idx_sorted:[>`ArrayLike] Np.Obj.t -> x:[>`ArrayLike] Np.Obj.t -> y:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> t
(**
Build a decision tree regressor from the training set (X, y).

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The training input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csc_matrix``.

y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    The target values (real numbers). Use ``dtype=np.float64`` and
    ``order='C'`` for maximum efficiency.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights. If None, then samples are equally weighted. Splits
    that would create child nodes with net zero or negative weight are
    ignored while searching for a split in each node.

check_input : bool, default=True
    Allow to bypass several input checking.
    Don't use this parameter unless you know what you do.

X_idx_sorted : array-like of shape (n_samples, n_features),             default=None
    The indexes of the sorted training input samples. If many tree
    are grown on the same dataset, this allows the ordering to be
    cached between trees. If None, the data will be sorted here.
    Don't use this parameter unless you know what to do.

Returns
-------
self : DecisionTreeRegressor
    Fitted estimator.
*)

val get_depth : [> tag] Obj.t -> Py.Object.t
(**
Return the depth of the decision tree.

The depth of a tree is the maximum distance between the root
and any leaf.

Returns
-------
self.tree_.max_depth : int
    The maximum depth of the tree.
*)

val get_n_leaves : [> tag] Obj.t -> Py.Object.t
(**
Return the number of leaves of the decision tree.

Returns
-------
self.tree_.n_leaves : int
    Number of leaves.
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

val predict : ?check_input:bool -> x:[>`ArrayLike] Np.Obj.t -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Predict class or regression value for X.

For a classification model, the predicted class for each sample in X is
returned. For a regression model, the predicted value based on X is
returned.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csr_matrix``.

check_input : bool, default=True
    Allow to bypass several input checking.
    Don't use this parameter unless you know what you do.

Returns
-------
y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    The predicted classes, or the predict values.
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


(** Attribute max_features_: get value or raise Not_found if None.*)
val max_features_ : t -> int

(** Attribute max_features_: get value as an option. *)
val max_features_opt : t -> (int) option


(** Attribute n_features_: get value or raise Not_found if None.*)
val n_features_ : t -> int

(** Attribute n_features_: get value as an option. *)
val n_features_opt : t -> (int) option


(** Attribute n_outputs_: get value or raise Not_found if None.*)
val n_outputs_ : t -> int

(** Attribute n_outputs_: get value as an option. *)
val n_outputs_opt : t -> (int) option


(** Attribute tree_: get value or raise Not_found if None.*)
val tree_ : t -> Py.Object.t

(** Attribute tree_: get value as an option. *)
val tree_opt : t -> (Py.Object.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val export_graphviz : ?out_file:[`S of string | `File_object of Py.Object.t] -> ?max_depth:int -> ?feature_names:string list -> ?class_names:[`Bool of bool | `StringList of string list] -> ?label:[`All | `Root | `None] -> ?filled:bool -> ?leaves_parallel:bool -> ?impurity:bool -> ?node_ids:bool -> ?proportion:bool -> ?rotate:bool -> ?rounded:bool -> ?special_characters:bool -> ?precision:int -> decision_tree:[>`DecisionTreeClassifier] Np.Obj.t -> unit -> string option
(**
Export a decision tree in DOT format.

This function generates a GraphViz representation of the decision tree,
which is then written into `out_file`. Once exported, graphical renderings
can be generated using, for example::

    $ dot -Tps tree.dot -o tree.ps      (PostScript format)
    $ dot -Tpng tree.dot -o tree.png    (PNG format)

The sample counts that are shown are weighted with any sample_weights that
might be present.

Read more in the :ref:`User Guide <tree>`.

Parameters
----------
decision_tree : decision tree classifier
    The decision tree to be exported to GraphViz.

out_file : file object or string, optional (default=None)
    Handle or name of the output file. If ``None``, the result is
    returned as a string.

    .. versionchanged:: 0.20
        Default of out_file changed from 'tree.dot' to None.

max_depth : int, optional (default=None)
    The maximum depth of the representation. If None, the tree is fully
    generated.

feature_names : list of strings, optional (default=None)
    Names of each of the features.

class_names : list of strings, bool or None, optional (default=None)
    Names of each of the target classes in ascending numerical order.
    Only relevant for classification and not supported for multi-output.
    If ``True``, shows a symbolic representation of the class name.

label : {'all', 'root', 'none'}, optional (default='all')
    Whether to show informative labels for impurity, etc.
    Options include 'all' to show at every node, 'root' to show only at
    the top root node, or 'none' to not show at any node.

filled : bool, optional (default=False)
    When set to ``True``, paint nodes to indicate majority class for
    classification, extremity of values for regression, or purity of node
    for multi-output.

leaves_parallel : bool, optional (default=False)
    When set to ``True``, draw all leaf nodes at the bottom of the tree.

impurity : bool, optional (default=True)
    When set to ``True``, show the impurity at each node.

node_ids : bool, optional (default=False)
    When set to ``True``, show the ID number on each node.

proportion : bool, optional (default=False)
    When set to ``True``, change the display of 'values' and/or 'samples'
    to be proportions and percentages respectively.

rotate : bool, optional (default=False)
    When set to ``True``, orient tree left to right rather than top-down.

rounded : bool, optional (default=False)
    When set to ``True``, draw node boxes with rounded corners and use
    Helvetica fonts instead of Times-Roman.

special_characters : bool, optional (default=False)
    When set to ``False``, ignore special characters for PostScript
    compatibility.

precision : int, optional (default=3)
    Number of digits of precision for floating point in the values of
    impurity, threshold and value attributes of each node.

Returns
-------
dot_data : string
    String representation of the input tree in GraphViz dot format.
    Only returned if ``out_file`` is None.

    .. versionadded:: 0.18

Examples
--------
>>> from sklearn.datasets import load_iris
>>> from sklearn import tree

>>> clf = tree.DecisionTreeClassifier()
>>> iris = load_iris()

>>> clf = clf.fit(iris.data, iris.target)
>>> tree.export_graphviz(clf)
'digraph Tree {...
*)

val export_text : ?feature_names:string list -> ?max_depth:int -> ?spacing:int -> ?decimals:int -> ?show_weights:bool -> decision_tree:[>`BaseDecisionTree] Np.Obj.t -> unit -> string
(**
Build a text report showing the rules of a decision tree.

Note that backwards compatibility may not be supported.

Parameters
----------
decision_tree : object
    The decision tree estimator to be exported.
    It can be an instance of
    DecisionTreeClassifier or DecisionTreeRegressor.

feature_names : list, optional (default=None)
    A list of length n_features containing the feature names.
    If None generic names will be used ('feature_0', 'feature_1', ...).

max_depth : int, optional (default=10)
    Only the first max_depth levels of the tree are exported.
    Truncated branches will be marked with '...'.

spacing : int, optional (default=3)
    Number of spaces between edges. The higher it is, the wider the result.

decimals : int, optional (default=2)
    Number of decimal digits to display.

show_weights : bool, optional (default=False)
    If true the classification weights will be exported on each leaf.
    The classification weights are the number of samples each class.

Returns
-------
report : string
    Text summary of all the rules in the decision tree.

Examples
--------

>>> from sklearn.datasets import load_iris
>>> from sklearn.tree import DecisionTreeClassifier
>>> from sklearn.tree import export_text
>>> iris = load_iris()
>>> X = iris['data']
>>> y = iris['target']
>>> decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
>>> decision_tree = decision_tree.fit(X, y)
>>> r = export_text(decision_tree, feature_names=iris['feature_names'])
>>> print(r)
|--- petal width (cm) <= 0.80
|   |--- class: 0
|--- petal width (cm) >  0.80
|   |--- petal width (cm) <= 1.75
|   |   |--- class: 1
|   |--- petal width (cm) >  1.75
|   |   |--- class: 2
*)

val plot_tree : ?max_depth:int -> ?feature_names:string list -> ?class_names:[`Bool of bool | `StringList of string list] -> ?label:[`All | `Root | `None] -> ?filled:bool -> ?impurity:bool -> ?node_ids:bool -> ?proportion:bool -> ?rotate:bool -> ?rounded:bool -> ?precision:int -> ?ax:Py.Object.t -> ?fontsize:int -> decision_tree:[>`BaseDecisionTree] Np.Obj.t -> unit -> Py.Object.t
(**
Plot a decision tree.

The sample counts that are shown are weighted with any sample_weights that
might be present.

The visualization is fit automatically to the size of the axis.
Use the ``figsize`` or ``dpi`` arguments of ``plt.figure``  to control
the size of the rendering.

Read more in the :ref:`User Guide <tree>`.

.. versionadded:: 0.21

Parameters
----------
decision_tree : decision tree regressor or classifier
    The decision tree to be plotted.

max_depth : int, optional (default=None)
    The maximum depth of the representation. If None, the tree is fully
    generated.

feature_names : list of strings, optional (default=None)
    Names of each of the features.

class_names : list of strings, bool or None, optional (default=None)
    Names of each of the target classes in ascending numerical order.
    Only relevant for classification and not supported for multi-output.
    If ``True``, shows a symbolic representation of the class name.

label : {'all', 'root', 'none'}, optional (default='all')
    Whether to show informative labels for impurity, etc.
    Options include 'all' to show at every node, 'root' to show only at
    the top root node, or 'none' to not show at any node.

filled : bool, optional (default=False)
    When set to ``True``, paint nodes to indicate majority class for
    classification, extremity of values for regression, or purity of node
    for multi-output.

impurity : bool, optional (default=True)
    When set to ``True``, show the impurity at each node.

node_ids : bool, optional (default=False)
    When set to ``True``, show the ID number on each node.

proportion : bool, optional (default=False)
    When set to ``True``, change the display of 'values' and/or 'samples'
    to be proportions and percentages respectively.

rotate : bool, optional (default=False)
    When set to ``True``, orient tree left to right rather than top-down.

rounded : bool, optional (default=False)
    When set to ``True``, draw node boxes with rounded corners and use
    Helvetica fonts instead of Times-Roman.

precision : int, optional (default=3)
    Number of digits of precision for floating point in the values of
    impurity, threshold and value attributes of each node.

ax : matplotlib axis, optional (default=None)
    Axes to plot to. If None, use current axis. Any previous content
    is cleared.

fontsize : int, optional (default=None)
    Size of text font. If None, determined automatically to fit figure.

Returns
-------
annotations : list of artists
    List containing the artists for the annotation boxes making up the
    tree.

Examples
--------
>>> from sklearn.datasets import load_iris
>>> from sklearn import tree

>>> clf = tree.DecisionTreeClassifier(random_state=0)
>>> iris = load_iris()

>>> clf = clf.fit(iris.data, iris.target)
>>> tree.plot_tree(clf)  # doctest: +SKIP
[Text(251.5,345.217,'X[3] <= 0.8...
*)

