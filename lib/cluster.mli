(** Get an attribute of this module as a Py.Object.t. This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module AffinityPropagation : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?damping:float -> ?max_iter:int -> ?convergence_iter:int -> ?copy:bool -> ?preference:[`Arr of Arr.t | `F of float] -> ?affinity:[`Euclidean | `Precomputed] -> ?verbose:int -> unit -> t
(**
Perform Affinity Propagation Clustering of data.

Read more in the :ref:`User Guide <affinity_propagation>`.

Parameters
----------
damping : float, default=0.5
    Damping factor (between 0.5 and 1) is the extent to
    which the current value is maintained relative to
    incoming values (weighted 1 - damping). This in order
    to avoid numerical oscillations when updating these
    values (messages).

max_iter : int, default=200
    Maximum number of iterations.

convergence_iter : int, default=15
    Number of iterations with no change in the number
    of estimated clusters that stops the convergence.

copy : bool, default=True
    Make a copy of input data.

preference : array-like of shape (n_samples,) or float, default=None
    Preferences for each point - points with larger values of
    preferences are more likely to be chosen as exemplars. The number
    of exemplars, ie of clusters, is influenced by the input
    preferences value. If the preferences are not passed as arguments,
    they will be set to the median of the input similarities.

affinity : {'euclidean', 'precomputed'}, default='euclidean'
    Which affinity to use. At the moment 'precomputed' and
    ``euclidean`` are supported. 'euclidean' uses the
    negative squared euclidean distance between points.

verbose : bool, default=False
    Whether to be verbose.


Attributes
----------
cluster_centers_indices_ : ndarray of shape (n_clusters,)
    Indices of cluster centers

cluster_centers_ : ndarray of shape (n_clusters, n_features)
    Cluster centers (if affinity != ``precomputed``).

labels_ : ndarray of shape (n_samples,)
    Labels of each point

affinity_matrix_ : ndarray of shape (n_samples, n_samples)
    Stores the affinity matrix used in ``fit``.

n_iter_ : int
    Number of iterations taken to converge.

Examples
--------
>>> from sklearn.cluster import AffinityPropagation
>>> import numpy as np
>>> X = np.array([[1, 2], [1, 4], [1, 0],
...               [4, 2], [4, 4], [4, 0]])
>>> clustering = AffinityPropagation().fit(X)
>>> clustering
AffinityPropagation()
>>> clustering.labels_
array([0, 0, 0, 1, 1, 1])
>>> clustering.predict([[0, 0], [4, 4]])
array([0, 1])
>>> clustering.cluster_centers_
array([[1, 2],
       [4, 2]])

Notes
-----
For an example, see :ref:`examples/cluster/plot_affinity_propagation.py
<sphx_glr_auto_examples_cluster_plot_affinity_propagation.py>`.

The algorithmic complexity of affinity propagation is quadratic
in the number of points.

When ``fit`` does not converge, ``cluster_centers_`` becomes an empty
array and all training samples will be labelled as ``-1``. In addition,
``predict`` will then label every sample as ``-1``.

When all training samples have equal similarities and equal preferences,
the assignment of cluster centers and labels depends on the preference.
If the preference is smaller than the similarities, ``fit`` will result in
a single cluster center and label ``0`` for every sample. Otherwise, every
training sample becomes its own cluster center and is assigned a unique
label.

References
----------

Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
Between Data Points", Science Feb. 2007
*)

val fit : ?y:Py.Object.t -> x:Arr.t -> t -> t
(**
Fit the clustering from features, or affinity matrix.

Parameters
----------
X : array-like or sparse matrix, shape (n_samples, n_features), or             array-like, shape (n_samples, n_samples)
    Training instances to cluster, or similarities / affinities between
    instances if ``affinity='precomputed'``. If a sparse feature matrix
    is provided, it will be converted into a sparse ``csr_matrix``.

y : Ignored
    Not used, present here for API consistency by convention.

Returns
-------
self
*)

val fit_predict : ?y:Py.Object.t -> x:Arr.t -> t -> Arr.t
(**
Fit the clustering from features or affinity matrix, and return
cluster labels.

Parameters
----------
X : array-like or sparse matrix, shape (n_samples, n_features), or             array-like, shape (n_samples, n_samples)
    Training instances to cluster, or similarities / affinities between
    instances if ``affinity='precomputed'``. If a sparse feature matrix
    is provided, it will be converted into a sparse ``csr_matrix``.

y : Ignored
    Not used, present here for API consistency by convention.

Returns
-------
labels : ndarray, shape (n_samples,)
    Cluster labels.
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
Predict the closest cluster each sample in X belongs to.

Parameters
----------
X : array-like or sparse matrix, shape (n_samples, n_features)
    New data to predict. If a sparse matrix is provided, it will be
    converted into a sparse ``csr_matrix``.

Returns
-------
labels : ndarray, shape (n_samples,)
    Cluster labels.
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


(** Attribute cluster_centers_indices_: get value or raise Not_found if None.*)
val cluster_centers_indices_ : t -> Arr.t

(** Attribute cluster_centers_indices_: get value as an option. *)
val cluster_centers_indices_opt : t -> (Arr.t) option


(** Attribute cluster_centers_: get value or raise Not_found if None.*)
val cluster_centers_ : t -> Arr.t

(** Attribute cluster_centers_: get value as an option. *)
val cluster_centers_opt : t -> (Arr.t) option


(** Attribute labels_: get value or raise Not_found if None.*)
val labels_ : t -> Arr.t

(** Attribute labels_: get value as an option. *)
val labels_opt : t -> (Arr.t) option


(** Attribute affinity_matrix_: get value or raise Not_found if None.*)
val affinity_matrix_ : t -> Arr.t

(** Attribute affinity_matrix_: get value as an option. *)
val affinity_matrix_opt : t -> (Arr.t) option


(** Attribute n_iter_: get value or raise Not_found if None.*)
val n_iter_ : t -> int

(** Attribute n_iter_: get value as an option. *)
val n_iter_opt : t -> (int) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module AgglomerativeClustering : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_clusters:[`I of int | `None] -> ?affinity:[`S of string | `Callable of Py.Object.t] -> ?memory:[`S of string | `JoblibMemory of Py.Object.t] -> ?connectivity:[`Arr of Arr.t | `Callable of Py.Object.t] -> ?compute_full_tree:[`Auto | `Bool of bool] -> ?linkage:[`Ward | `Complete | `Average | `Single] -> ?distance_threshold:float -> unit -> t
(**
Agglomerative Clustering

Recursively merges the pair of clusters that minimally increases
a given linkage distance.

Read more in the :ref:`User Guide <hierarchical_clustering>`.

Parameters
----------
n_clusters : int or None, default=2
    The number of clusters to find. It must be ``None`` if
    ``distance_threshold`` is not ``None``.

affinity : str or callable, default='euclidean'
    Metric used to compute the linkage. Can be "euclidean", "l1", "l2",
    "manhattan", "cosine", or "precomputed".
    If linkage is "ward", only "euclidean" is accepted.
    If "precomputed", a distance matrix (instead of a similarity matrix)
    is needed as input for the fit method.

memory : str or object with the joblib.Memory interface, default=None
    Used to cache the output of the computation of the tree.
    By default, no caching is done. If a string is given, it is the
    path to the caching directory.

connectivity : array-like or callable, default=None
    Connectivity matrix. Defines for each sample the neighboring
    samples following a given structure of the data.
    This can be a connectivity matrix itself or a callable that transforms
    the data into a connectivity matrix, such as derived from
    kneighbors_graph. Default is None, i.e, the
    hierarchical clustering algorithm is unstructured.

compute_full_tree : 'auto' or bool, default='auto'
    Stop early the construction of the tree at n_clusters. This is useful
    to decrease computation time if the number of clusters is not small
    compared to the number of samples. This option is useful only when
    specifying a connectivity matrix. Note also that when varying the
    number of clusters and using caching, it may be advantageous to compute
    the full tree. It must be ``True`` if ``distance_threshold`` is not
    ``None``. By default `compute_full_tree` is "auto", which is equivalent
    to `True` when `distance_threshold` is not `None` or that `n_clusters`
    is inferior to the maximum between 100 or `0.02 * n_samples`.
    Otherwise, "auto" is equivalent to `False`.

linkage : {"ward", "complete", "average", "single"}, default="ward"
    Which linkage criterion to use. The linkage criterion determines which
    distance to use between sets of observation. The algorithm will merge
    the pairs of cluster that minimize this criterion.

    - ward minimizes the variance of the clusters being merged.
    - average uses the average of the distances of each observation of
      the two sets.
    - complete or maximum linkage uses the maximum distances between
      all observations of the two sets.
    - single uses the minimum of the distances between all observations
      of the two sets.

distance_threshold : float, default=None
    The linkage distance threshold above which, clusters will not be
    merged. If not ``None``, ``n_clusters`` must be ``None`` and
    ``compute_full_tree`` must be ``True``.

    .. versionadded:: 0.21

Attributes
----------
n_clusters_ : int
    The number of clusters found by the algorithm. If
    ``distance_threshold=None``, it will be equal to the given
    ``n_clusters``.

labels_ : ndarray of shape (n_samples)
    cluster labels for each point

n_leaves_ : int
    Number of leaves in the hierarchical tree.

n_connected_components_ : int
    The estimated number of connected components in the graph.

children_ : array-like of shape (n_samples-1, 2)
    The children of each non-leaf node. Values less than `n_samples`
    correspond to leaves of the tree which are the original samples.
    A node `i` greater than or equal to `n_samples` is a non-leaf
    node and has children `children_[i - n_samples]`. Alternatively
    at the i-th iteration, children[i][0] and children[i][1]
    are merged to form node `n_samples + i`

Examples
--------
>>> from sklearn.cluster import AgglomerativeClustering
>>> import numpy as np
>>> X = np.array([[1, 2], [1, 4], [1, 0],
...               [4, 2], [4, 4], [4, 0]])
>>> clustering = AgglomerativeClustering().fit(X)
>>> clustering
AgglomerativeClustering()
>>> clustering.labels_
array([1, 1, 1, 0, 0, 0])
*)

val fit : ?y:Py.Object.t -> x:Arr.t -> t -> t
(**
Fit the hierarchical clustering from features, or distance matrix.

Parameters
----------
X : array-like, shape (n_samples, n_features) or (n_samples, n_samples)
    Training instances to cluster, or distances between instances if
    ``affinity='precomputed'``.

y : Ignored
    Not used, present here for API consistency by convention.

Returns
-------
self
*)

val fit_predict : ?y:Py.Object.t -> x:Arr.t -> t -> Arr.t
(**
Fit the hierarchical clustering from features or distance matrix,
and return cluster labels.

Parameters
----------
X : array-like, shape (n_samples, n_features) or (n_samples, n_samples)
    Training instances to cluster, or distances between instances if
    ``affinity='precomputed'``.

y : Ignored
    Not used, present here for API consistency by convention.

Returns
-------
labels : ndarray, shape (n_samples,)
    Cluster labels.
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


(** Attribute n_clusters_: get value or raise Not_found if None.*)
val n_clusters_ : t -> int

(** Attribute n_clusters_: get value as an option. *)
val n_clusters_opt : t -> (int) option


(** Attribute labels_: get value or raise Not_found if None.*)
val labels_ : t -> Arr.t

(** Attribute labels_: get value as an option. *)
val labels_opt : t -> (Arr.t) option


(** Attribute n_leaves_: get value or raise Not_found if None.*)
val n_leaves_ : t -> int

(** Attribute n_leaves_: get value as an option. *)
val n_leaves_opt : t -> (int) option


(** Attribute n_connected_components_: get value or raise Not_found if None.*)
val n_connected_components_ : t -> int

(** Attribute n_connected_components_: get value as an option. *)
val n_connected_components_opt : t -> (int) option


(** Attribute children_: get value or raise Not_found if None.*)
val children_ : t -> Arr.t

(** Attribute children_: get value as an option. *)
val children_opt : t -> (Arr.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Birch : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?threshold:float -> ?branching_factor:int -> ?n_clusters:[`I of int | `Instance_of_sklearn_cluster_model of Py.Object.t] -> ?compute_labels:bool -> ?copy:bool -> unit -> t
(**
Implements the Birch clustering algorithm.

It is a memory-efficient, online-learning algorithm provided as an
alternative to :class:`MiniBatchKMeans`. It constructs a tree
data structure with the cluster centroids being read off the leaf.
These can be either the final cluster centroids or can be provided as input
to another clustering algorithm such as :class:`AgglomerativeClustering`.

Read more in the :ref:`User Guide <birch>`.

.. versionadded:: 0.16

Parameters
----------
threshold : float, default=0.5
    The radius of the subcluster obtained by merging a new sample and the
    closest subcluster should be lesser than the threshold. Otherwise a new
    subcluster is started. Setting this value to be very low promotes
    splitting and vice-versa.

branching_factor : int, default=50
    Maximum number of CF subclusters in each node. If a new samples enters
    such that the number of subclusters exceed the branching_factor then
    that node is split into two nodes with the subclusters redistributed
    in each. The parent subcluster of that node is removed and two new
    subclusters are added as parents of the 2 split nodes.

n_clusters : int, instance of sklearn.cluster model, default=3
    Number of clusters after the final clustering step, which treats the
    subclusters from the leaves as new samples.

    - `None` : the final clustering step is not performed and the
      subclusters are returned as they are.

    - :mod:`sklearn.cluster` Estimator : If a model is provided, the model
      is fit treating the subclusters as new samples and the initial data
      is mapped to the label of the closest subcluster.

    - `int` : the model fit is :class:`AgglomerativeClustering` with
      `n_clusters` set to be equal to the int.

compute_labels : bool, default=True
    Whether or not to compute labels for each fit.

copy : bool, default=True
    Whether or not to make a copy of the given data. If set to False,
    the initial data will be overwritten.

Attributes
----------
root_ : _CFNode
    Root of the CFTree.

dummy_leaf_ : _CFNode
    Start pointer to all the leaves.

subcluster_centers_ : ndarray,
    Centroids of all subclusters read directly from the leaves.

subcluster_labels_ : ndarray,
    Labels assigned to the centroids of the subclusters after
    they are clustered globally.

labels_ : ndarray, shape (n_samples,)
    Array of labels assigned to the input data.
    if partial_fit is used instead of fit, they are assigned to the
    last batch of data.

See Also
--------

MiniBatchKMeans
    Alternative  implementation that does incremental updates
    of the centers' positions using mini-batches.

Notes
-----
The tree data structure consists of nodes with each node consisting of
a number of subclusters. The maximum number of subclusters in a node
is determined by the branching factor. Each subcluster maintains a
linear sum, squared sum and the number of samples in that subcluster.
In addition, each subcluster can also have a node as its child, if the
subcluster is not a member of a leaf node.

For a new point entering the root, it is merged with the subcluster closest
to it and the linear sum, squared sum and the number of samples of that
subcluster are updated. This is done recursively till the properties of
the leaf node are updated.

References
----------
* Tian Zhang, Raghu Ramakrishnan, Maron Livny
  BIRCH: An efficient data clustering method for large databases.
  https://www.cs.sfu.ca/CourseCentral/459/han/papers/zhang96.pdf

* Roberto Perdisci
  JBirch - Java implementation of BIRCH clustering algorithm
  https://code.google.com/archive/p/jbirch

Examples
--------
>>> from sklearn.cluster import Birch
>>> X = [[0, 1], [0.3, 1], [-0.3, 1], [0, -1], [0.3, -1], [-0.3, -1]]
>>> brc = Birch(n_clusters=None)
>>> brc.fit(X)
Birch(n_clusters=None)
>>> brc.predict(X)
array([0, 0, 0, 1, 1, 1])
*)

val fit : ?y:Py.Object.t -> x:Arr.t -> t -> t
(**
Build a CF Tree for the input data.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Input data.

y : Ignored
    Not used, present here for API consistency by convention.

Returns
-------
self
    Fitted estimator.
*)

val fit_predict : ?y:Py.Object.t -> x:Arr.t -> t -> Arr.t
(**
Perform clustering on X and returns cluster labels.

Parameters
----------
X : ndarray, shape (n_samples, n_features)
    Input data.

y : Ignored
    Not used, present for API consistency by convention.

Returns
-------
labels : ndarray, shape (n_samples,)
    Cluster labels.
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

val partial_fit : ?x:Arr.t -> ?y:Py.Object.t -> t -> t
(**
Online learning. Prevents rebuilding of CFTree from scratch.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features), None
    Input data. If X is not provided, only the global clustering
    step is done.

y : Ignored
    Not used, present here for API consistency by convention.

Returns
-------
self
    Fitted estimator.
*)

val predict : x:Arr.t -> t -> Arr.t
(**
Predict data using the ``centroids_`` of subclusters.

Avoid computation of the row norms of X.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Input data.

Returns
-------
labels : ndarray, shape(n_samples)
    Labelled data.
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
Transform X into subcluster centroids dimension.

Each dimension represents the distance from the sample point to each
cluster centroid.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Input data.

Returns
-------
X_trans : {array-like, sparse matrix}, shape (n_samples, n_clusters)
    Transformed data.
*)


(** Attribute root_: get value or raise Not_found if None.*)
val root_ : t -> Py.Object.t

(** Attribute root_: get value as an option. *)
val root_opt : t -> (Py.Object.t) option


(** Attribute dummy_leaf_: get value or raise Not_found if None.*)
val dummy_leaf_ : t -> Py.Object.t

(** Attribute dummy_leaf_: get value as an option. *)
val dummy_leaf_opt : t -> (Py.Object.t) option


(** Attribute subcluster_centers_: get value or raise Not_found if None.*)
val subcluster_centers_ : t -> Arr.t

(** Attribute subcluster_centers_: get value as an option. *)
val subcluster_centers_opt : t -> (Arr.t) option


(** Attribute subcluster_labels_: get value or raise Not_found if None.*)
val subcluster_labels_ : t -> Arr.t

(** Attribute subcluster_labels_: get value as an option. *)
val subcluster_labels_opt : t -> (Arr.t) option


(** Attribute labels_: get value or raise Not_found if None.*)
val labels_ : t -> Arr.t

(** Attribute labels_: get value as an option. *)
val labels_opt : t -> (Arr.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module DBSCAN : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?eps:float -> ?min_samples:int -> ?metric:[`S of string | `Callable of Py.Object.t] -> ?metric_params:Dict.t -> ?algorithm:[`Auto | `Ball_tree | `Kd_tree | `Brute] -> ?leaf_size:int -> ?p:float -> ?n_jobs:int -> unit -> t
(**
Perform DBSCAN clustering from vector array or distance matrix.

DBSCAN - Density-Based Spatial Clustering of Applications with Noise.
Finds core samples of high density and expands clusters from them.
Good for data which contains clusters of similar density.

Read more in the :ref:`User Guide <dbscan>`.

Parameters
----------
eps : float, default=0.5
    The maximum distance between two samples for one to be considered
    as in the neighborhood of the other. This is not a maximum bound
    on the distances of points within a cluster. This is the most
    important DBSCAN parameter to choose appropriately for your data set
    and distance function.

min_samples : int, default=5
    The number of samples (or total weight) in a neighborhood for a point
    to be considered as a core point. This includes the point itself.

metric : string, or callable, default='euclidean'
    The metric to use when calculating distance between instances in a
    feature array. If metric is a string or callable, it must be one of
    the options allowed by :func:`sklearn.metrics.pairwise_distances` for
    its metric parameter.
    If metric is "precomputed", X is assumed to be a distance matrix and
    must be square. X may be a :term:`Glossary <sparse graph>`, in which
    case only "nonzero" elements may be considered neighbors for DBSCAN.

    .. versionadded:: 0.17
       metric *precomputed* to accept precomputed sparse matrix.

metric_params : dict, default=None
    Additional keyword arguments for the metric function.

    .. versionadded:: 0.19

algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
    The algorithm to be used by the NearestNeighbors module
    to compute pointwise distances and find nearest neighbors.
    See NearestNeighbors module documentation for details.

leaf_size : int, default=30
    Leaf size passed to BallTree or cKDTree. This can affect the speed
    of the construction and query, as well as the memory required
    to store the tree. The optimal value depends
    on the nature of the problem.

p : float, default=None
    The power of the Minkowski metric to be used to calculate distance
    between points.

n_jobs : int or None, default=None
    The number of parallel jobs to run.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

Attributes
----------
core_sample_indices_ : array, shape = [n_core_samples]
    Indices of core samples.

components_ : array, shape = [n_core_samples, n_features]
    Copy of each core sample found by training.

labels_ : array, shape = [n_samples]
    Cluster labels for each point in the dataset given to fit().
    Noisy samples are given the label -1.

Examples
--------
>>> from sklearn.cluster import DBSCAN
>>> import numpy as np
>>> X = np.array([[1, 2], [2, 2], [2, 3],
...               [8, 7], [8, 8], [25, 80]])
>>> clustering = DBSCAN(eps=3, min_samples=2).fit(X)
>>> clustering.labels_
array([ 0,  0,  0,  1,  1, -1])
>>> clustering
DBSCAN(eps=3, min_samples=2)

See also
--------
OPTICS
    A similar clustering at multiple values of eps. Our implementation
    is optimized for memory usage.

Notes
-----
For an example, see :ref:`examples/cluster/plot_dbscan.py
<sphx_glr_auto_examples_cluster_plot_dbscan.py>`.

This implementation bulk-computes all neighborhood queries, which increases
the memory complexity to O(n.d) where d is the average number of neighbors,
while original DBSCAN had memory complexity O(n). It may attract a higher
memory complexity when querying these nearest neighborhoods, depending
on the ``algorithm``.

One way to avoid the query complexity is to pre-compute sparse
neighborhoods in chunks using
:func:`NearestNeighbors.radius_neighbors_graph
<sklearn.neighbors.NearestNeighbors.radius_neighbors_graph>` with
``mode='distance'``, then using ``metric='precomputed'`` here.

Another way to reduce memory and computation time is to remove
(near-)duplicate points and use ``sample_weight`` instead.

:class:`cluster.OPTICS` provides a similar clustering with lower memory
usage.

References
----------
Ester, M., H. P. Kriegel, J. Sander, and X. Xu, "A Density-Based
Algorithm for Discovering Clusters in Large Spatial Databases with Noise".
In: Proceedings of the 2nd International Conference on Knowledge Discovery
and Data Mining, Portland, OR, AAAI Press, pp. 226-231. 1996

Schubert, E., Sander, J., Ester, M., Kriegel, H. P., & Xu, X. (2017).
DBSCAN revisited, revisited: why and how you should (still) use DBSCAN.
ACM Transactions on Database Systems (TODS), 42(3), 19.
*)

val fit : ?y:Py.Object.t -> ?sample_weight:Arr.t -> x:Arr.t -> t -> t
(**
Perform DBSCAN clustering from features, or distance matrix.

Parameters
----------
X : array-like or sparse matrix, shape (n_samples, n_features), or             (n_samples, n_samples)
    Training instances to cluster, or distances between instances if
    ``metric='precomputed'``. If a sparse matrix is provided, it will
    be converted into a sparse ``csr_matrix``.

sample_weight : array, shape (n_samples,), optional
    Weight of each sample, such that a sample with a weight of at least
    ``min_samples`` is by itself a core sample; a sample with a
    negative weight may inhibit its eps-neighbor from being core.
    Note that weights are absolute, and default to 1.

y : Ignored
    Not used, present here for API consistency by convention.

Returns
-------
self
*)

val fit_predict : ?y:Py.Object.t -> ?sample_weight:Arr.t -> x:Arr.t -> t -> Arr.t
(**
Perform DBSCAN clustering from features or distance matrix,
and return cluster labels.

Parameters
----------
X : array-like or sparse matrix, shape (n_samples, n_features), or             (n_samples, n_samples)
    Training instances to cluster, or distances between instances if
    ``metric='precomputed'``. If a sparse matrix is provided, it will
    be converted into a sparse ``csr_matrix``.

sample_weight : array, shape (n_samples,), optional
    Weight of each sample, such that a sample with a weight of at least
    ``min_samples`` is by itself a core sample; a sample with a
    negative weight may inhibit its eps-neighbor from being core.
    Note that weights are absolute, and default to 1.

y : Ignored
    Not used, present here for API consistency by convention.

Returns
-------
labels : ndarray, shape (n_samples,)
    Cluster labels. Noisy samples are given the label -1.
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


(** Attribute core_sample_indices_: get value or raise Not_found if None.*)
val core_sample_indices_ : t -> Arr.t

(** Attribute core_sample_indices_: get value as an option. *)
val core_sample_indices_opt : t -> (Arr.t) option


(** Attribute components_: get value or raise Not_found if None.*)
val components_ : t -> Arr.t

(** Attribute components_: get value as an option. *)
val components_opt : t -> (Arr.t) option


(** Attribute labels_: get value or raise Not_found if None.*)
val labels_ : t -> Arr.t

(** Attribute labels_: get value as an option. *)
val labels_opt : t -> (Arr.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module FeatureAgglomeration : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_clusters:int -> ?affinity:[`S of string | `Callable of Py.Object.t] -> ?memory:[`S of string | `JoblibMemory of Py.Object.t] -> ?connectivity:[`Arr of Arr.t | `Callable of Py.Object.t] -> ?compute_full_tree:[`Auto | `Bool of bool] -> ?linkage:[`Ward | `Complete | `Average | `Single] -> ?pooling_func:Py.Object.t -> ?distance_threshold:float -> unit -> t
(**
Agglomerate features.

Similar to AgglomerativeClustering, but recursively merges features
instead of samples.

Read more in the :ref:`User Guide <hierarchical_clustering>`.

Parameters
----------
n_clusters : int, default=2
    The number of clusters to find. It must be ``None`` if
    ``distance_threshold`` is not ``None``.

affinity : str or callable, default='euclidean'
    Metric used to compute the linkage. Can be "euclidean", "l1", "l2",
    "manhattan", "cosine", or 'precomputed'.
    If linkage is "ward", only "euclidean" is accepted.

memory : str or object with the joblib.Memory interface, default=None
    Used to cache the output of the computation of the tree.
    By default, no caching is done. If a string is given, it is the
    path to the caching directory.

connectivity : array-like or callable, default=None
    Connectivity matrix. Defines for each feature the neighboring
    features following a given structure of the data.
    This can be a connectivity matrix itself or a callable that transforms
    the data into a connectivity matrix, such as derived from
    kneighbors_graph. Default is None, i.e, the
    hierarchical clustering algorithm is unstructured.

compute_full_tree : 'auto' or bool, optional, default='auto'
    Stop early the construction of the tree at n_clusters. This is useful
    to decrease computation time if the number of clusters is not small
    compared to the number of features. This option is useful only when
    specifying a connectivity matrix. Note also that when varying the
    number of clusters and using caching, it may be advantageous to compute
    the full tree. It must be ``True`` if ``distance_threshold`` is not
    ``None``. By default `compute_full_tree` is "auto", which is equivalent
    to `True` when `distance_threshold` is not `None` or that `n_clusters`
    is inferior to the maximum between 100 or `0.02 * n_samples`.
    Otherwise, "auto" is equivalent to `False`.

linkage : {'ward', 'complete', 'average', 'single'}, default='ward'
    Which linkage criterion to use. The linkage criterion determines which
    distance to use between sets of features. The algorithm will merge
    the pairs of cluster that minimize this criterion.

    - ward minimizes the variance of the clusters being merged.
    - average uses the average of the distances of each feature of
      the two sets.
    - complete or maximum linkage uses the maximum distances between
      all features of the two sets.
    - single uses the minimum of the distances between all observations
      of the two sets.

pooling_func : callable, default=np.mean
    This combines the values of agglomerated features into a single
    value, and should accept an array of shape [M, N] and the keyword
    argument `axis=1`, and reduce it to an array of size [M].

distance_threshold : float, default=None
    The linkage distance threshold above which, clusters will not be
    merged. If not ``None``, ``n_clusters`` must be ``None`` and
    ``compute_full_tree`` must be ``True``.

    .. versionadded:: 0.21

Attributes
----------
n_clusters_ : int
    The number of clusters found by the algorithm. If
    ``distance_threshold=None``, it will be equal to the given
    ``n_clusters``.

labels_ : array-like of (n_features,)
    cluster labels for each feature.

n_leaves_ : int
    Number of leaves in the hierarchical tree.

n_connected_components_ : int
    The estimated number of connected components in the graph.

children_ : array-like of shape (n_nodes-1, 2)
    The children of each non-leaf node. Values less than `n_features`
    correspond to leaves of the tree which are the original samples.
    A node `i` greater than or equal to `n_features` is a non-leaf
    node and has children `children_[i - n_features]`. Alternatively
    at the i-th iteration, children[i][0] and children[i][1]
    are merged to form node `n_features + i`

distances_ : array-like of shape (n_nodes-1,)
    Distances between nodes in the corresponding place in `children_`.
    Only computed if distance_threshold is not None.

Examples
--------
>>> import numpy as np
>>> from sklearn import datasets, cluster
>>> digits = datasets.load_digits()
>>> images = digits.images
>>> X = np.reshape(images, (len(images), -1))
>>> agglo = cluster.FeatureAgglomeration(n_clusters=32)
>>> agglo.fit(X)
FeatureAgglomeration(n_clusters=32)
>>> X_reduced = agglo.transform(X)
>>> X_reduced.shape
(1797, 32)
*)

val fit : ?y:Py.Object.t -> ?params:(string * Py.Object.t) list -> x:Arr.t -> t -> t
(**
Fit the hierarchical clustering on the data

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The data

y : Ignored

Returns
-------
self
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

val inverse_transform : xred:Arr.t -> t -> Arr.t
(**
Inverse the transformation.
Return a vector of size nb_features with the values of Xred assigned
to each group of features

Parameters
----------
Xred : array-like of shape (n_samples, n_clusters) or (n_clusters,)
    The values to be assigned to each cluster of samples

Returns
-------
X : array, shape=[n_samples, n_features] or [n_features]
    A vector of size n_samples with the values of Xred assigned to
    each of the cluster of samples.
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
Transform a new matrix using the built clustering

Parameters
----------
X : array-like of shape (n_samples, n_features) or (n_samples,)
    A M by N array of M observations in N dimensions or a length
    M array of M one-dimensional observations.

Returns
-------
Y : array, shape = [n_samples, n_clusters] or [n_clusters]
    The pooled values for each feature cluster.
*)


(** Attribute n_clusters_: get value or raise Not_found if None.*)
val n_clusters_ : t -> int

(** Attribute n_clusters_: get value as an option. *)
val n_clusters_opt : t -> (int) option


(** Attribute labels_: get value or raise Not_found if None.*)
val labels_ : t -> Py.Object.t

(** Attribute labels_: get value as an option. *)
val labels_opt : t -> (Py.Object.t) option


(** Attribute n_leaves_: get value or raise Not_found if None.*)
val n_leaves_ : t -> int

(** Attribute n_leaves_: get value as an option. *)
val n_leaves_opt : t -> (int) option


(** Attribute n_connected_components_: get value or raise Not_found if None.*)
val n_connected_components_ : t -> int

(** Attribute n_connected_components_: get value as an option. *)
val n_connected_components_opt : t -> (int) option


(** Attribute children_: get value or raise Not_found if None.*)
val children_ : t -> Arr.t

(** Attribute children_: get value as an option. *)
val children_opt : t -> (Arr.t) option


(** Attribute distances_: get value or raise Not_found if None.*)
val distances_ : t -> Arr.t

(** Attribute distances_: get value as an option. *)
val distances_opt : t -> (Arr.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module KMeans : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_clusters:int -> ?init:[`K_means_ | `Random | `Arr of Arr.t] -> ?n_init:int -> ?max_iter:int -> ?tol:float -> ?precompute_distances:[`Auto | `Bool of bool] -> ?verbose:int -> ?random_state:int -> ?copy_x:bool -> ?n_jobs:int -> ?algorithm:[`Auto | `Full | `Elkan] -> unit -> t
(**
K-Means clustering.

Read more in the :ref:`User Guide <k_means>`.

Parameters
----------

n_clusters : int, default=8
    The number of clusters to form as well as the number of
    centroids to generate.

init : {'k-means++', 'random'} or ndarray of shape             (n_clusters, n_features), default='k-means++'
    Method for initialization, defaults to 'k-means++':

    'k-means++' : selects initial cluster centers for k-mean
    clustering in a smart way to speed up convergence. See section
    Notes in k_init for more details.

    'random': choose k observations (rows) at random from data for
    the initial centroids.

    If an ndarray is passed, it should be of shape (n_clusters, n_features)
    and gives the initial centers.

n_init : int, default=10
    Number of time the k-means algorithm will be run with different
    centroid seeds. The final results will be the best output of
    n_init consecutive runs in terms of inertia.

max_iter : int, default=300
    Maximum number of iterations of the k-means algorithm for a
    single run.

tol : float, default=1e-4
    Relative tolerance with regards to inertia to declare convergence.

precompute_distances : 'auto' or bool, default='auto'
    Precompute distances (faster but takes more memory).

    'auto' : do not precompute distances if n_samples * n_clusters > 12
    million. This corresponds to about 100MB overhead per job using
    double precision.

    True : always precompute distances.

    False : never precompute distances.

verbose : int, default=0
    Verbosity mode.

random_state : int, RandomState instance, default=None
    Determines random number generation for centroid initialization. Use
    an int to make the randomness deterministic.
    See :term:`Glossary <random_state>`.

copy_x : bool, default=True
    When pre-computing distances it is more numerically accurate to center
    the data first.  If copy_x is True (default), then the original data is
    not modified, ensuring X is C-contiguous.  If False, the original data
    is modified, and put back before the function returns, but small
    numerical differences may be introduced by subtracting and then adding
    the data mean, in this case it will also not ensure that data is
    C-contiguous which may cause a significant slowdown.

n_jobs : int, default=None
    The number of jobs to use for the computation. This works by computing
    each of the n_init runs in parallel.

    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

algorithm : {"auto", "full", "elkan"}, default="auto"
    K-means algorithm to use. The classical EM-style algorithm is "full".
    The "elkan" variation is more efficient by using the triangle
    inequality, but currently doesn't support sparse data. "auto" chooses
    "elkan" for dense data and "full" for sparse data.

Attributes
----------
cluster_centers_ : ndarray of shape (n_clusters, n_features)
    Coordinates of cluster centers. If the algorithm stops before fully
    converging (see ``tol`` and ``max_iter``), these will not be
    consistent with ``labels_``.

labels_ : ndarray of shape (n_samples,)
    Labels of each point

inertia_ : float
    Sum of squared distances of samples to their closest cluster center.

n_iter_ : int
    Number of iterations run.

See Also
--------

MiniBatchKMeans
    Alternative online implementation that does incremental updates
    of the centers positions using mini-batches.
    For large scale learning (say n_samples > 10k) MiniBatchKMeans is
    probably much faster than the default batch implementation.

Notes
-----
The k-means problem is solved using either Lloyd's or Elkan's algorithm.

The average complexity is given by O(k n T), were n is the number of
samples and T is the number of iteration.

The worst case complexity is given by O(n^(k+2/p)) with
n = n_samples, p = n_features. (D. Arthur and S. Vassilvitskii,
'How slow is the k-means method?' SoCG2006)

In practice, the k-means algorithm is very fast (one of the fastest
clustering algorithms available), but it falls in local minima. That's why
it can be useful to restart it several times.

If the algorithm stops before fully converging (because of ``tol`` or
``max_iter``), ``labels_`` and ``cluster_centers_`` will not be consistent,
i.e. the ``cluster_centers_`` will not be the means of the points in each
cluster. Also, the estimator will reassign ``labels_`` after the last
iteration to make ``labels_`` consistent with ``predict`` on the training
set.

Examples
--------

>>> from sklearn.cluster import KMeans
>>> import numpy as np
>>> X = np.array([[1, 2], [1, 4], [1, 0],
...               [10, 2], [10, 4], [10, 0]])
>>> kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
>>> kmeans.labels_
array([1, 1, 1, 0, 0, 0], dtype=int32)
>>> kmeans.predict([[0, 0], [12, 3]])
array([1, 0], dtype=int32)
>>> kmeans.cluster_centers_
array([[10.,  2.],
       [ 1.,  2.]])
*)

val fit : ?y:Py.Object.t -> ?sample_weight:Arr.t -> x:Arr.t -> t -> t
(**
Compute k-means clustering.

Parameters
----------
X : array-like or sparse matrix, shape=(n_samples, n_features)
    Training instances to cluster. It must be noted that the data
    will be converted to C ordering, which will cause a memory
    copy if the given data is not C-contiguous.

y : Ignored
    Not used, present here for API consistency by convention.

sample_weight : array-like, shape (n_samples,), optional
    The weights for each observation in X. If None, all observations
    are assigned equal weight (default: None).

Returns
-------
self
    Fitted estimator.
*)

val fit_predict : ?y:Py.Object.t -> ?sample_weight:Arr.t -> x:Arr.t -> t -> Arr.t
(**
Compute cluster centers and predict cluster index for each sample.

Convenience method; equivalent to calling fit(X) followed by
predict(X).

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    New data to transform.

y : Ignored
    Not used, present here for API consistency by convention.

sample_weight : array-like, shape (n_samples,), optional
    The weights for each observation in X. If None, all observations
    are assigned equal weight (default: None).

Returns
-------
labels : array, shape [n_samples,]
    Index of the cluster each sample belongs to.
*)

val fit_transform : ?y:Py.Object.t -> ?sample_weight:Arr.t -> x:Arr.t -> t -> Arr.t
(**
Compute clustering and transform X to cluster-distance space.

Equivalent to fit(X).transform(X), but more efficiently implemented.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    New data to transform.

y : Ignored
    Not used, present here for API consistency by convention.

sample_weight : array-like, shape (n_samples,), optional
    The weights for each observation in X. If None, all observations
    are assigned equal weight (default: None).

Returns
-------
X_new : array, shape [n_samples, k]
    X transformed in the new space.
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

val predict : ?sample_weight:Arr.t -> x:Arr.t -> t -> Arr.t
(**
Predict the closest cluster each sample in X belongs to.

In the vector quantization literature, `cluster_centers_` is called
the code book and each value returned by `predict` is the index of
the closest code in the code book.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    New data to predict.

sample_weight : array-like, shape (n_samples,), optional
    The weights for each observation in X. If None, all observations
    are assigned equal weight (default: None).

Returns
-------
labels : array, shape [n_samples,]
    Index of the cluster each sample belongs to.
*)

val score : ?y:Py.Object.t -> ?sample_weight:Arr.t -> x:Arr.t -> t -> float
(**
Opposite of the value of X on the K-means objective.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    New data.

y : Ignored
    Not used, present here for API consistency by convention.

sample_weight : array-like, shape (n_samples,), optional
    The weights for each observation in X. If None, all observations
    are assigned equal weight (default: None).

Returns
-------
score : float
    Opposite of the value of X on the K-means objective.
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
Transform X to a cluster-distance space.

In the new space, each dimension is the distance to the cluster
centers.  Note that even if X is sparse, the array returned by
`transform` will typically be dense.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    New data to transform.

Returns
-------
X_new : array, shape [n_samples, k]
    X transformed in the new space.
*)


(** Attribute cluster_centers_: get value or raise Not_found if None.*)
val cluster_centers_ : t -> Arr.t

(** Attribute cluster_centers_: get value as an option. *)
val cluster_centers_opt : t -> (Arr.t) option


(** Attribute labels_: get value or raise Not_found if None.*)
val labels_ : t -> Arr.t

(** Attribute labels_: get value as an option. *)
val labels_opt : t -> (Arr.t) option


(** Attribute inertia_: get value or raise Not_found if None.*)
val inertia_ : t -> float

(** Attribute inertia_: get value as an option. *)
val inertia_opt : t -> (float) option


(** Attribute n_iter_: get value or raise Not_found if None.*)
val n_iter_ : t -> int

(** Attribute n_iter_: get value as an option. *)
val n_iter_opt : t -> (int) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MeanShift : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?bandwidth:float -> ?seeds:Arr.t -> ?bin_seeding:bool -> ?min_bin_freq:int -> ?cluster_all:bool -> ?n_jobs:int -> ?max_iter:int -> unit -> t
(**
Mean shift clustering using a flat kernel.

Mean shift clustering aims to discover "blobs" in a smooth density of
samples. It is a centroid-based algorithm, which works by updating
candidates for centroids to be the mean of the points within a given
region. These candidates are then filtered in a post-processing stage to
eliminate near-duplicates to form the final set of centroids.

Seeding is performed using a binning technique for scalability.

Read more in the :ref:`User Guide <mean_shift>`.

Parameters
----------
bandwidth : float, optional
    Bandwidth used in the RBF kernel.

    If not given, the bandwidth is estimated using
    sklearn.cluster.estimate_bandwidth; see the documentation for that
    function for hints on scalability (see also the Notes, below).

seeds : array, shape=[n_samples, n_features], optional
    Seeds used to initialize kernels. If not set,
    the seeds are calculated by clustering.get_bin_seeds
    with bandwidth as the grid size and default values for
    other parameters.

bin_seeding : boolean, optional
    If true, initial kernel locations are not locations of all
    points, but rather the location of the discretized version of
    points, where points are binned onto a grid whose coarseness
    corresponds to the bandwidth. Setting this option to True will speed
    up the algorithm because fewer seeds will be initialized.
    default value: False
    Ignored if seeds argument is not None.

min_bin_freq : int, optional
   To speed up the algorithm, accept only those bins with at least
   min_bin_freq points as seeds. If not defined, set to 1.

cluster_all : boolean, default True
    If true, then all points are clustered, even those orphans that are
    not within any kernel. Orphans are assigned to the nearest kernel.
    If false, then orphans are given cluster label -1.

n_jobs : int or None, optional (default=None)
    The number of jobs to use for the computation. This works by computing
    each of the n_init runs in parallel.

    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

max_iter : int, default=300
    Maximum number of iterations, per seed point before the clustering
    operation terminates (for that seed point), if has not converged yet.

    .. versionadded:: 0.22

Attributes
----------
cluster_centers_ : array, [n_clusters, n_features]
    Coordinates of cluster centers.

labels_ :
    Labels of each point.

n_iter_ : int
    Maximum number of iterations performed on each seed.

    .. versionadded:: 0.22

Examples
--------
>>> from sklearn.cluster import MeanShift
>>> import numpy as np
>>> X = np.array([[1, 1], [2, 1], [1, 0],
...               [4, 7], [3, 5], [3, 6]])
>>> clustering = MeanShift(bandwidth=2).fit(X)
>>> clustering.labels_
array([1, 1, 1, 0, 0, 0])
>>> clustering.predict([[0, 0], [5, 5]])
array([1, 0])
>>> clustering
MeanShift(bandwidth=2)

Notes
-----

Scalability:

Because this implementation uses a flat kernel and
a Ball Tree to look up members of each kernel, the complexity will tend
towards O(T*n*log(n)) in lower dimensions, with n the number of samples
and T the number of points. In higher dimensions the complexity will
tend towards O(T*n^2).

Scalability can be boosted by using fewer seeds, for example by using
a higher value of min_bin_freq in the get_bin_seeds function.

Note that the estimate_bandwidth function is much less scalable than the
mean shift algorithm and will be the bottleneck if it is used.

References
----------

Dorin Comaniciu and Peter Meer, "Mean Shift: A robust approach toward
feature space analysis". IEEE Transactions on Pattern Analysis and
Machine Intelligence. 2002. pp. 603-619.
*)

val fit : ?y:Py.Object.t -> x:Arr.t -> t -> t
(**
Perform clustering.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Samples to cluster.

y : Ignored
*)

val fit_predict : ?y:Py.Object.t -> x:Arr.t -> t -> Arr.t
(**
Perform clustering on X and returns cluster labels.

Parameters
----------
X : ndarray, shape (n_samples, n_features)
    Input data.

y : Ignored
    Not used, present for API consistency by convention.

Returns
-------
labels : ndarray, shape (n_samples,)
    Cluster labels.
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
Predict the closest cluster each sample in X belongs to.

Parameters
----------
X : {array-like, sparse matrix}, shape=[n_samples, n_features]
    New data to predict.

Returns
-------
labels : array, shape [n_samples,]
    Index of the cluster each sample belongs to.
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


(** Attribute cluster_centers_: get value or raise Not_found if None.*)
val cluster_centers_ : t -> Arr.t

(** Attribute cluster_centers_: get value as an option. *)
val cluster_centers_opt : t -> (Arr.t) option


(** Attribute labels_: get value or raise Not_found if None.*)
val labels_ : t -> Py.Object.t

(** Attribute labels_: get value as an option. *)
val labels_opt : t -> (Py.Object.t) option


(** Attribute n_iter_: get value or raise Not_found if None.*)
val n_iter_ : t -> int

(** Attribute n_iter_: get value as an option. *)
val n_iter_opt : t -> (int) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MiniBatchKMeans : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_clusters:int -> ?init:[`K_means_ | `Random | `Arr of Arr.t] -> ?max_iter:int -> ?batch_size:int -> ?verbose:int -> ?compute_labels:bool -> ?random_state:int -> ?tol:float -> ?max_no_improvement:int -> ?init_size:int -> ?n_init:int -> ?reassignment_ratio:float -> unit -> t
(**
Mini-Batch K-Means clustering.

Read more in the :ref:`User Guide <mini_batch_kmeans>`.

Parameters
----------

n_clusters : int, default=8
    The number of clusters to form as well as the number of
    centroids to generate.

init : {'k-means++', 'random'} or ndarray of shape             (n_clusters, n_features), default='k-means++'
    Method for initialization

    'k-means++' : selects initial cluster centers for k-mean
    clustering in a smart way to speed up convergence. See section
    Notes in k_init for more details.

    'random': choose k observations (rows) at random from data for
    the initial centroids.

    If an ndarray is passed, it should be of shape (n_clusters, n_features)
    and gives the initial centers.

max_iter : int, default=100
    Maximum number of iterations over the complete dataset before
    stopping independently of any early stopping criterion heuristics.

batch_size : int, default=100
    Size of the mini batches.

verbose : int, default=0
    Verbosity mode.

compute_labels : bool, default=True
    Compute label assignment and inertia for the complete dataset
    once the minibatch optimization has converged in fit.

random_state : int, RandomState instance, default=None
    Determines random number generation for centroid initialization and
    random reassignment. Use an int to make the randomness deterministic.
    See :term:`Glossary <random_state>`.

tol : float, default=0.0
    Control early stopping based on the relative center changes as
    measured by a smoothed, variance-normalized of the mean center
    squared position changes. This early stopping heuristics is
    closer to the one used for the batch variant of the algorithms
    but induces a slight computational and memory overhead over the
    inertia heuristic.

    To disable convergence detection based on normalized center
    change, set tol to 0.0 (default).

max_no_improvement : int, default=10
    Control early stopping based on the consecutive number of mini
    batches that does not yield an improvement on the smoothed inertia.

    To disable convergence detection based on inertia, set
    max_no_improvement to None.

init_size : int, default=None
    Number of samples to randomly sample for speeding up the
    initialization (sometimes at the expense of accuracy): the
    only algorithm is initialized by running a batch KMeans on a
    random subset of the data. This needs to be larger than n_clusters.

    If `None`, `init_size= 3 * batch_size`.

n_init : int, default=3
    Number of random initializations that are tried.
    In contrast to KMeans, the algorithm is only run once, using the
    best of the ``n_init`` initializations as measured by inertia.

reassignment_ratio : float, default=0.01
    Control the fraction of the maximum number of counts for a
    center to be reassigned. A higher value means that low count
    centers are more easily reassigned, which means that the
    model will take longer to converge, but should converge in a
    better clustering.

Attributes
----------

cluster_centers_ : ndarray of shape (n_clusters, n_features)
    Coordinates of cluster centers

labels_ : int
    Labels of each point (if compute_labels is set to True).

inertia_ : float
    The value of the inertia criterion associated with the chosen
    partition (if compute_labels is set to True). The inertia is
    defined as the sum of square distances of samples to their nearest
    neighbor.

See Also
--------
KMeans
    The classic implementation of the clustering method based on the
    Lloyd's algorithm. It consumes the whole set of input data at each
    iteration.

Notes
-----
See https://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf

Examples
--------
>>> from sklearn.cluster import MiniBatchKMeans
>>> import numpy as np
>>> X = np.array([[1, 2], [1, 4], [1, 0],
...               [4, 2], [4, 0], [4, 4],
...               [4, 5], [0, 1], [2, 2],
...               [3, 2], [5, 5], [1, -1]])
>>> # manually fit on batches
>>> kmeans = MiniBatchKMeans(n_clusters=2,
...                          random_state=0,
...                          batch_size=6)
>>> kmeans = kmeans.partial_fit(X[0:6,:])
>>> kmeans = kmeans.partial_fit(X[6:12,:])
>>> kmeans.cluster_centers_
array([[2. , 1. ],
       [3.5, 4.5]])
>>> kmeans.predict([[0, 0], [4, 4]])
array([0, 1], dtype=int32)
>>> # fit on the whole data
>>> kmeans = MiniBatchKMeans(n_clusters=2,
...                          random_state=0,
...                          batch_size=6,
...                          max_iter=10).fit(X)
>>> kmeans.cluster_centers_
array([[3.95918367, 2.40816327],
       [1.12195122, 1.3902439 ]])
>>> kmeans.predict([[0, 0], [4, 4]])
array([1, 0], dtype=int32)
*)

val fit : ?y:Py.Object.t -> ?sample_weight:Arr.t -> x:Arr.t -> t -> t
(**
Compute the centroids on X by chunking it into mini-batches.

Parameters
----------
X : array-like or sparse matrix, shape=(n_samples, n_features)
    Training instances to cluster. It must be noted that the data
    will be converted to C ordering, which will cause a memory copy
    if the given data is not C-contiguous.

y : Ignored
    Not used, present here for API consistency by convention.

sample_weight : array-like, shape (n_samples,), optional
    The weights for each observation in X. If None, all observations
    are assigned equal weight (default: None).

Returns
-------
self
*)

val fit_predict : ?y:Py.Object.t -> ?sample_weight:Arr.t -> x:Arr.t -> t -> Arr.t
(**
Compute cluster centers and predict cluster index for each sample.

Convenience method; equivalent to calling fit(X) followed by
predict(X).

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    New data to transform.

y : Ignored
    Not used, present here for API consistency by convention.

sample_weight : array-like, shape (n_samples,), optional
    The weights for each observation in X. If None, all observations
    are assigned equal weight (default: None).

Returns
-------
labels : array, shape [n_samples,]
    Index of the cluster each sample belongs to.
*)

val fit_transform : ?y:Py.Object.t -> ?sample_weight:Arr.t -> x:Arr.t -> t -> Arr.t
(**
Compute clustering and transform X to cluster-distance space.

Equivalent to fit(X).transform(X), but more efficiently implemented.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    New data to transform.

y : Ignored
    Not used, present here for API consistency by convention.

sample_weight : array-like, shape (n_samples,), optional
    The weights for each observation in X. If None, all observations
    are assigned equal weight (default: None).

Returns
-------
X_new : array, shape [n_samples, k]
    X transformed in the new space.
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

val partial_fit : ?y:Py.Object.t -> ?sample_weight:Arr.t -> x:Arr.t -> t -> Py.Object.t
(**
Update k means estimate on a single mini-batch X.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Coordinates of the data points to cluster. It must be noted that
    X will be copied if it is not C-contiguous.

y : Ignored
    Not used, present here for API consistency by convention.

sample_weight : array-like, shape (n_samples,), optional
    The weights for each observation in X. If None, all observations
    are assigned equal weight (default: None).

Returns
-------
self
*)

val predict : ?sample_weight:Arr.t -> x:Arr.t -> t -> Arr.t
(**
Predict the closest cluster each sample in X belongs to.

In the vector quantization literature, `cluster_centers_` is called
the code book and each value returned by `predict` is the index of
the closest code in the code book.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    New data to predict.

sample_weight : array-like, shape (n_samples,), optional
    The weights for each observation in X. If None, all observations
    are assigned equal weight (default: None).

Returns
-------
labels : array, shape [n_samples,]
    Index of the cluster each sample belongs to.
*)

val score : ?y:Py.Object.t -> ?sample_weight:Arr.t -> x:Arr.t -> t -> float
(**
Opposite of the value of X on the K-means objective.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    New data.

y : Ignored
    Not used, present here for API consistency by convention.

sample_weight : array-like, shape (n_samples,), optional
    The weights for each observation in X. If None, all observations
    are assigned equal weight (default: None).

Returns
-------
score : float
    Opposite of the value of X on the K-means objective.
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
Transform X to a cluster-distance space.

In the new space, each dimension is the distance to the cluster
centers.  Note that even if X is sparse, the array returned by
`transform` will typically be dense.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    New data to transform.

Returns
-------
X_new : array, shape [n_samples, k]
    X transformed in the new space.
*)


(** Attribute cluster_centers_: get value or raise Not_found if None.*)
val cluster_centers_ : t -> Arr.t

(** Attribute cluster_centers_: get value as an option. *)
val cluster_centers_opt : t -> (Arr.t) option


(** Attribute labels_: get value or raise Not_found if None.*)
val labels_ : t -> int

(** Attribute labels_: get value as an option. *)
val labels_opt : t -> (int) option


(** Attribute inertia_: get value or raise Not_found if None.*)
val inertia_ : t -> float

(** Attribute inertia_: get value as an option. *)
val inertia_opt : t -> (float) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module OPTICS : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?min_samples:[`I of int | `Float_between_0_and_1 of Py.Object.t] -> ?max_eps:float -> ?metric:[`S of string | `Callable of Py.Object.t] -> ?p:int -> ?metric_params:Dict.t -> ?cluster_method:string -> ?eps:float -> ?xi:[`F of float | `Between_0_and_1 of Py.Object.t] -> ?predecessor_correction:bool -> ?min_cluster_size:[`I of int | `Float_between_0_and_1 of Py.Object.t] -> ?algorithm:[`Auto | `Ball_tree | `Kd_tree | `Brute] -> ?leaf_size:int -> ?n_jobs:int -> unit -> t
(**
Estimate clustering structure from vector array.

OPTICS (Ordering Points To Identify the Clustering Structure), closely
related to DBSCAN, finds core sample of high density and expands clusters
from them [1]_. Unlike DBSCAN, keeps cluster hierarchy for a variable
neighborhood radius. Better suited for usage on large datasets than the
current sklearn implementation of DBSCAN.

Clusters are then extracted using a DBSCAN-like method
(cluster_method = 'dbscan') or an automatic
technique proposed in [1]_ (cluster_method = 'xi').

This implementation deviates from the original OPTICS by first performing
k-nearest-neighborhood searches on all points to identify core sizes, then
computing only the distances to unprocessed points when constructing the
cluster order. Note that we do not employ a heap to manage the expansion
candidates, so the time complexity will be O(n^2).

Read more in the :ref:`User Guide <optics>`.

Parameters
----------
min_samples : int > 1 or float between 0 and 1 (default=5)
    The number of samples in a neighborhood for a point to be considered as
    a core point. Also, up and down steep regions can't have more then
    ``min_samples`` consecutive non-steep points. Expressed as an absolute
    number or a fraction of the number of samples (rounded to be at least
    2).

max_eps : float, optional (default=np.inf)
    The maximum distance between two samples for one to be considered as
    in the neighborhood of the other. Default value of ``np.inf`` will
    identify clusters across all scales; reducing ``max_eps`` will result
    in shorter run times.

metric : str or callable, optional (default='minkowski')
    Metric to use for distance computation. Any metric from scikit-learn
    or scipy.spatial.distance can be used.

    If metric is a callable function, it is called on each
    pair of instances (rows) and the resulting value recorded. The callable
    should take two arrays as input and return one value indicating the
    distance between them. This works for Scipy's metrics, but is less
    efficient than passing the metric name as a string. If metric is
    "precomputed", X is assumed to be a distance matrix and must be square.

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

p : int, optional (default=2)
    Parameter for the Minkowski metric from
    :class:`sklearn.metrics.pairwise_distances`. When p = 1, this is
    equivalent to using manhattan_distance (l1), and euclidean_distance
    (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

metric_params : dict, optional (default=None)
    Additional keyword arguments for the metric function.

cluster_method : str, optional (default='xi')
    The extraction method used to extract clusters using the calculated
    reachability and ordering. Possible values are "xi" and "dbscan".

eps : float, optional (default=None)
    The maximum distance between two samples for one to be considered as
    in the neighborhood of the other. By default it assumes the same value
    as ``max_eps``.
    Used only when ``cluster_method='dbscan'``.

xi : float, between 0 and 1, optional (default=0.05)
    Determines the minimum steepness on the reachability plot that
    constitutes a cluster boundary. For example, an upwards point in the
    reachability plot is defined by the ratio from one point to its
    successor being at most 1-xi.
    Used only when ``cluster_method='xi'``.

predecessor_correction : bool, optional (default=True)
    Correct clusters according to the predecessors calculated by OPTICS
    [2]_. This parameter has minimal effect on most datasets.
    Used only when ``cluster_method='xi'``.

min_cluster_size : int > 1 or float between 0 and 1 (default=None)
    Minimum number of samples in an OPTICS cluster, expressed as an
    absolute number or a fraction of the number of samples (rounded to be
    at least 2). If ``None``, the value of ``min_samples`` is used instead.
    Used only when ``cluster_method='xi'``.

algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
    Algorithm used to compute the nearest neighbors:

    - 'ball_tree' will use :class:`BallTree`
    - 'kd_tree' will use :class:`KDTree`
    - 'brute' will use a brute-force search.
    - 'auto' will attempt to decide the most appropriate algorithm
      based on the values passed to :meth:`fit` method. (default)

    Note: fitting on sparse input will override the setting of
    this parameter, using brute force.

leaf_size : int, optional (default=30)
    Leaf size passed to :class:`BallTree` or :class:`KDTree`. This can
    affect the speed of the construction and query, as well as the memory
    required to store the tree. The optimal value depends on the
    nature of the problem.

n_jobs : int or None, optional (default=None)
    The number of parallel jobs to run for neighbors search.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

Attributes
----------
labels_ : array, shape (n_samples,)
    Cluster labels for each point in the dataset given to fit().
    Noisy samples and points which are not included in a leaf cluster
    of ``cluster_hierarchy_`` are labeled as -1.

reachability_ : array, shape (n_samples,)
    Reachability distances per sample, indexed by object order. Use
    ``clust.reachability_[clust.ordering_]`` to access in cluster order.

ordering_ : array, shape (n_samples,)
    The cluster ordered list of sample indices.

core_distances_ : array, shape (n_samples,)
    Distance at which each sample becomes a core point, indexed by object
    order. Points which will never be core have a distance of inf. Use
    ``clust.core_distances_[clust.ordering_]`` to access in cluster order.

predecessor_ : array, shape (n_samples,)
    Point that a sample was reached from, indexed by object order.
    Seed points have a predecessor of -1.

cluster_hierarchy_ : array, shape (n_clusters, 2)
    The list of clusters in the form of ``[start, end]`` in each row, with
    all indices inclusive. The clusters are ordered according to
    ``(end, -start)`` (ascending) so that larger clusters encompassing
    smaller clusters come after those smaller ones. Since ``labels_`` does
    not reflect the hierarchy, usually
    ``len(cluster_hierarchy_) > np.unique(optics.labels_)``. Please also
    note that these indices are of the ``ordering_``, i.e.
    ``X[ordering_][start:end + 1]`` form a cluster.
    Only available when ``cluster_method='xi'``.

See Also
--------
DBSCAN
    A similar clustering for a specified neighborhood radius (eps).
    Our implementation is optimized for runtime.

References
----------
.. [1] Ankerst, Mihael, Markus M. Breunig, Hans-Peter Kriegel,
   and Jörg Sander. "OPTICS: ordering points to identify the clustering
   structure." ACM SIGMOD Record 28, no. 2 (1999): 49-60.

.. [2] Schubert, Erich, Michael Gertz.
   "Improving the Cluster Structure Extracted from OPTICS Plots." Proc. of
   the Conference "Lernen, Wissen, Daten, Analysen" (LWDA) (2018): 318-329.

Examples
--------
>>> from sklearn.cluster import OPTICS
>>> import numpy as np
>>> X = np.array([[1, 2], [2, 5], [3, 6],
...               [8, 7], [8, 8], [7, 3]])
>>> clustering = OPTICS(min_samples=2).fit(X)
>>> clustering.labels_
array([0, 0, 0, 1, 1, 1])
*)

val fit : ?y:Py.Object.t -> x:Arr.t -> t -> t
(**
Perform OPTICS clustering.

Extracts an ordered list of points and reachability distances, and
performs initial clustering using ``max_eps`` distance specified at
OPTICS object instantiation.

Parameters
----------
X : array, shape (n_samples, n_features), or (n_samples, n_samples)          if metric=’precomputed’
    A feature array, or array of distances between samples if
    metric='precomputed'.

y : ignored
    Ignored.

Returns
-------
self : instance of OPTICS
    The instance.
*)

val fit_predict : ?y:Py.Object.t -> x:Arr.t -> t -> Arr.t
(**
Perform clustering on X and returns cluster labels.

Parameters
----------
X : ndarray, shape (n_samples, n_features)
    Input data.

y : Ignored
    Not used, present for API consistency by convention.

Returns
-------
labels : ndarray, shape (n_samples,)
    Cluster labels.
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


(** Attribute labels_: get value or raise Not_found if None.*)
val labels_ : t -> Arr.t

(** Attribute labels_: get value as an option. *)
val labels_opt : t -> (Arr.t) option


(** Attribute reachability_: get value or raise Not_found if None.*)
val reachability_ : t -> Arr.t

(** Attribute reachability_: get value as an option. *)
val reachability_opt : t -> (Arr.t) option


(** Attribute ordering_: get value or raise Not_found if None.*)
val ordering_ : t -> Arr.t

(** Attribute ordering_: get value as an option. *)
val ordering_opt : t -> (Arr.t) option


(** Attribute core_distances_: get value or raise Not_found if None.*)
val core_distances_ : t -> Arr.t

(** Attribute core_distances_: get value as an option. *)
val core_distances_opt : t -> (Arr.t) option


(** Attribute predecessor_: get value or raise Not_found if None.*)
val predecessor_ : t -> Arr.t

(** Attribute predecessor_: get value as an option. *)
val predecessor_opt : t -> (Arr.t) option


(** Attribute cluster_hierarchy_: get value or raise Not_found if None.*)
val cluster_hierarchy_ : t -> Arr.t

(** Attribute cluster_hierarchy_: get value as an option. *)
val cluster_hierarchy_opt : t -> (Arr.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module SpectralBiclustering : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_clusters:[`I of int | `PyObject of Py.Object.t] -> ?method_:[`Bistochastic | `Scale | `Log] -> ?n_components:int -> ?n_best:int -> ?svd_method:[`Randomized | `Arpack] -> ?n_svd_vecs:int -> ?mini_batch:bool -> ?init:[`K_means_ | `Random | `PyObject of Py.Object.t] -> ?n_init:int -> ?n_jobs:int -> ?random_state:int -> unit -> t
(**
Spectral biclustering (Kluger, 2003).

Partitions rows and columns under the assumption that the data has
an underlying checkerboard structure. For instance, if there are
two row partitions and three column partitions, each row will
belong to three biclusters, and each column will belong to two
biclusters. The outer product of the corresponding row and column
label vectors gives this checkerboard structure.

Read more in the :ref:`User Guide <spectral_biclustering>`.

Parameters
----------
n_clusters : int or tuple (n_row_clusters, n_column_clusters), default=3
    The number of row and column clusters in the checkerboard
    structure.

method : {'bistochastic', 'scale', 'log'}, default='bistochastic'
    Method of normalizing and converting singular vectors into
    biclusters. May be one of 'scale', 'bistochastic', or 'log'.
    The authors recommend using 'log'. If the data is sparse,
    however, log normalization will not work, which is why the
    default is 'bistochastic'.

    .. warning::
       if `method='log'`, the data must be sparse.

n_components : int, default=6
    Number of singular vectors to check.

n_best : int, default=3
    Number of best singular vectors to which to project the data
    for clustering.

svd_method : {'randomized', 'arpack'}, default='randomized'
    Selects the algorithm for finding singular vectors. May be
    'randomized' or 'arpack'. If 'randomized', uses
    :func:`~sklearn.utils.extmath.randomized_svd`, which may be faster
    for large matrices. If 'arpack', uses
    `scipy.sparse.linalg.svds`, which is more accurate, but
    possibly slower in some cases.

n_svd_vecs : int, default=None
    Number of vectors to use in calculating the SVD. Corresponds
    to `ncv` when `svd_method=arpack` and `n_oversamples` when
    `svd_method` is 'randomized`.

mini_batch : bool, default=False
    Whether to use mini-batch k-means, which is faster but may get
    different results.

init : {'k-means++', 'random'} or ndarray of (n_clusters, n_features),             default='k-means++'
    Method for initialization of k-means algorithm; defaults to
    'k-means++'.

n_init : int, default=10
    Number of random initializations that are tried with the
    k-means algorithm.

    If mini-batch k-means is used, the best initialization is
    chosen and the algorithm runs once. Otherwise, the algorithm
    is run for each initialization and the best solution chosen.

n_jobs : int, default=None
    The number of jobs to use for the computation. This works by breaking
    down the pairwise matrix into n_jobs even slices and computing them in
    parallel.

    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

random_state : int, RandomState instance, default=None
    Used for randomizing the singular value decomposition and the k-means
    initialization. Use an int to make the randomness deterministic.
    See :term:`Glossary <random_state>`.

Attributes
----------
rows_ : array-like of shape (n_row_clusters, n_rows)
    Results of the clustering. `rows[i, r]` is True if
    cluster `i` contains row `r`. Available only after calling ``fit``.

columns_ : array-like of shape (n_column_clusters, n_columns)
    Results of the clustering, like `rows`.

row_labels_ : array-like of shape (n_rows,)
    Row partition labels.

column_labels_ : array-like of shape (n_cols,)
    Column partition labels.

Examples
--------
>>> from sklearn.cluster import SpectralBiclustering
>>> import numpy as np
>>> X = np.array([[1, 1], [2, 1], [1, 0],
...               [4, 7], [3, 5], [3, 6]])
>>> clustering = SpectralBiclustering(n_clusters=2, random_state=0).fit(X)
>>> clustering.row_labels_
array([1, 1, 1, 0, 0, 0], dtype=int32)
>>> clustering.column_labels_
array([0, 1], dtype=int32)
>>> clustering
SpectralBiclustering(n_clusters=2, random_state=0)

References
----------

* Kluger, Yuval, et. al., 2003. `Spectral biclustering of microarray
  data: coclustering genes and conditions
  <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.135.1608>`__.
*)

val fit : ?y:Py.Object.t -> x:Arr.t -> t -> t
(**
Creates a biclustering for X.

Parameters
----------
X : array-like, shape (n_samples, n_features)

y : Ignored
*)

val get_indices : i:int -> t -> Py.Object.t
(**
Row and column indices of the i'th bicluster.

Only works if ``rows_`` and ``columns_`` attributes exist.

Parameters
----------
i : int
    The index of the cluster.

Returns
-------
row_ind : np.array, dtype=np.intp
    Indices of rows in the dataset that belong to the bicluster.
col_ind : np.array, dtype=np.intp
    Indices of columns in the dataset that belong to the bicluster.
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

val get_shape : i:int -> t -> int
(**
Shape of the i'th bicluster.

Parameters
----------
i : int
    The index of the cluster.

Returns
-------
shape : (int, int)
    Number of rows and columns (resp.) in the bicluster.
*)

val get_submatrix : i:int -> data:Py.Object.t -> t -> Arr.t
(**
Return the submatrix corresponding to bicluster `i`.

Parameters
----------
i : int
    The index of the cluster.
data : array
    The data.

Returns
-------
submatrix : array
    The submatrix corresponding to bicluster i.

Notes
-----
Works with sparse matrices. Only works if ``rows_`` and
``columns_`` attributes exist.
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


(** Attribute rows_: get value or raise Not_found if None.*)
val rows_ : t -> Arr.t

(** Attribute rows_: get value as an option. *)
val rows_opt : t -> (Arr.t) option


(** Attribute columns_: get value or raise Not_found if None.*)
val columns_ : t -> Arr.t

(** Attribute columns_: get value as an option. *)
val columns_opt : t -> (Arr.t) option


(** Attribute row_labels_: get value or raise Not_found if None.*)
val row_labels_ : t -> Arr.t

(** Attribute row_labels_: get value as an option. *)
val row_labels_opt : t -> (Arr.t) option


(** Attribute column_labels_: get value or raise Not_found if None.*)
val column_labels_ : t -> Arr.t

(** Attribute column_labels_: get value as an option. *)
val column_labels_opt : t -> (Arr.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module SpectralClustering : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_clusters:int -> ?eigen_solver:[`Arpack | `Lobpcg | `Amg] -> ?n_components:int -> ?random_state:int -> ?n_init:int -> ?gamma:float -> ?affinity:[`S of string | `Callable of Py.Object.t] -> ?n_neighbors:int -> ?eigen_tol:float -> ?assign_labels:[`Kmeans | `Discretize] -> ?degree:float -> ?coef0:float -> ?kernel_params:Py.Object.t -> ?n_jobs:int -> unit -> t
(**
Apply clustering to a projection of the normalized Laplacian.

In practice Spectral Clustering is very useful when the structure of
the individual clusters is highly non-convex or more generally when
a measure of the center and spread of the cluster is not a suitable
description of the complete cluster. For instance when clusters are
nested circles on the 2D plane.

If affinity is the adjacency matrix of a graph, this method can be
used to find normalized graph cuts.

When calling ``fit``, an affinity matrix is constructed using either
kernel function such the Gaussian (aka RBF) kernel of the euclidean
distanced ``d(X, X)``::

        np.exp(-gamma * d(X,X) ** 2)

or a k-nearest neighbors connectivity matrix.

Alternatively, using ``precomputed``, a user-provided affinity
matrix can be used.

Read more in the :ref:`User Guide <spectral_clustering>`.

Parameters
----------
n_clusters : integer, optional
    The dimension of the projection subspace.

eigen_solver : {None, 'arpack', 'lobpcg', or 'amg'}
    The eigenvalue decomposition strategy to use. AMG requires pyamg
    to be installed. It can be faster on very large, sparse problems,
    but may also lead to instabilities.

n_components : integer, optional, default=n_clusters
    Number of eigen vectors to use for the spectral embedding

random_state : int, RandomState instance or None (default)
    A pseudo random number generator used for the initialization of the
    lobpcg eigen vectors decomposition when ``eigen_solver='amg'`` and by
    the K-Means initialization. Use an int to make the randomness
    deterministic.
    See :term:`Glossary <random_state>`.

n_init : int, optional, default: 10
    Number of time the k-means algorithm will be run with different
    centroid seeds. The final results will be the best output of
    n_init consecutive runs in terms of inertia.

gamma : float, default=1.0
    Kernel coefficient for rbf, poly, sigmoid, laplacian and chi2 kernels.
    Ignored for ``affinity='nearest_neighbors'``.

affinity : string or callable, default 'rbf'
    How to construct the affinity matrix.
     - 'nearest_neighbors' : construct the affinity matrix by computing a
       graph of nearest neighbors.
     - 'rbf' : construct the affinity matrix using a radial basis function
       (RBF) kernel.
     - 'precomputed' : interpret ``X`` as a precomputed affinity matrix.
     - 'precomputed_nearest_neighbors' : interpret ``X`` as a sparse graph
       of precomputed nearest neighbors, and constructs the affinity matrix
       by selecting the ``n_neighbors`` nearest neighbors.
     - one of the kernels supported by
       :func:`~sklearn.metrics.pairwise_kernels`.

    Only kernels that produce similarity scores (non-negative values that
    increase with similarity) should be used. This property is not checked
    by the clustering algorithm.

n_neighbors : integer
    Number of neighbors to use when constructing the affinity matrix using
    the nearest neighbors method. Ignored for ``affinity='rbf'``.

eigen_tol : float, optional, default: 0.0
    Stopping criterion for eigendecomposition of the Laplacian matrix
    when ``eigen_solver='arpack'``.

assign_labels : {'kmeans', 'discretize'}, default: 'kmeans'
    The strategy to use to assign labels in the embedding
    space. There are two ways to assign labels after the laplacian
    embedding. k-means can be applied and is a popular choice. But it can
    also be sensitive to initialization. Discretization is another approach
    which is less sensitive to random initialization.

degree : float, default=3
    Degree of the polynomial kernel. Ignored by other kernels.

coef0 : float, default=1
    Zero coefficient for polynomial and sigmoid kernels.
    Ignored by other kernels.

kernel_params : dictionary of string to any, optional
    Parameters (keyword arguments) and values for kernel passed as
    callable object. Ignored by other kernels.

n_jobs : int or None, optional (default=None)
    The number of parallel jobs to run.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

Attributes
----------
affinity_matrix_ : array-like, shape (n_samples, n_samples)
    Affinity matrix used for clustering. Available only if after calling
    ``fit``.

labels_ : array, shape (n_samples,)
    Labels of each point

Examples
--------
>>> from sklearn.cluster import SpectralClustering
>>> import numpy as np
>>> X = np.array([[1, 1], [2, 1], [1, 0],
...               [4, 7], [3, 5], [3, 6]])
>>> clustering = SpectralClustering(n_clusters=2,
...         assign_labels="discretize",
...         random_state=0).fit(X)
>>> clustering.labels_
array([1, 1, 1, 0, 0, 0])
>>> clustering
SpectralClustering(assign_labels='discretize', n_clusters=2,
    random_state=0)

Notes
-----
If you have an affinity matrix, such as a distance matrix,
for which 0 means identical elements, and high values means
very dissimilar elements, it can be transformed in a
similarity matrix that is well suited for the algorithm by
applying the Gaussian (RBF, heat) kernel::

    np.exp(- dist_matrix ** 2 / (2. * delta ** 2))

Where ``delta`` is a free parameter representing the width of the Gaussian
kernel.

Another alternative is to take a symmetric version of the k
nearest neighbors connectivity matrix of the points.

If the pyamg package is installed, it is used: this greatly
speeds up computation.

References
----------

- Normalized cuts and image segmentation, 2000
  Jianbo Shi, Jitendra Malik
  http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.160.2324

- A Tutorial on Spectral Clustering, 2007
  Ulrike von Luxburg
  http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.9323

- Multiclass spectral clustering, 2003
  Stella X. Yu, Jianbo Shi
  https://www1.icsi.berkeley.edu/~stellayu/publication/doc/2003kwayICCV.pdf
*)

val fit : ?y:Py.Object.t -> x:Arr.t -> t -> t
(**
Perform spectral clustering from features, or affinity matrix.

Parameters
----------
X : array-like or sparse matrix, shape (n_samples, n_features), or             array-like, shape (n_samples, n_samples)
    Training instances to cluster, or similarities / affinities between
    instances if ``affinity='precomputed'``. If a sparse matrix is
    provided in a format other than ``csr_matrix``, ``csc_matrix``,
    or ``coo_matrix``, it will be converted into a sparse
    ``csr_matrix``.

y : Ignored
    Not used, present here for API consistency by convention.

Returns
-------
self
*)

val fit_predict : ?y:Py.Object.t -> x:Arr.t -> t -> Arr.t
(**
Perform spectral clustering from features, or affinity matrix,
and return cluster labels.

Parameters
----------
X : array-like or sparse matrix, shape (n_samples, n_features), or             array-like, shape (n_samples, n_samples)
    Training instances to cluster, or similarities / affinities between
    instances if ``affinity='precomputed'``. If a sparse matrix is
    provided in a format other than ``csr_matrix``, ``csc_matrix``,
    or ``coo_matrix``, it will be converted into a sparse
    ``csr_matrix``.

y : Ignored
    Not used, present here for API consistency by convention.

Returns
-------
labels : ndarray, shape (n_samples,)
    Cluster labels.
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


(** Attribute affinity_matrix_: get value or raise Not_found if None.*)
val affinity_matrix_ : t -> Arr.t

(** Attribute affinity_matrix_: get value as an option. *)
val affinity_matrix_opt : t -> (Arr.t) option


(** Attribute labels_: get value or raise Not_found if None.*)
val labels_ : t -> Arr.t

(** Attribute labels_: get value as an option. *)
val labels_opt : t -> (Arr.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module SpectralCoclustering : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_clusters:int -> ?svd_method:[`Randomized | `Arpack] -> ?n_svd_vecs:int -> ?mini_batch:bool -> ?init:[`T_k_means_ of Py.Object.t | `Random | `Arr of Arr.t] -> ?n_init:int -> ?n_jobs:int -> ?random_state:int -> unit -> t
(**
Spectral Co-Clustering algorithm (Dhillon, 2001).

Clusters rows and columns of an array `X` to solve the relaxed
normalized cut of the bipartite graph created from `X` as follows:
the edge between row vertex `i` and column vertex `j` has weight
`X[i, j]`.

The resulting bicluster structure is block-diagonal, since each
row and each column belongs to exactly one bicluster.

Supports sparse matrices, as long as they are nonnegative.

Read more in the :ref:`User Guide <spectral_coclustering>`.

Parameters
----------
n_clusters : int, default=3
    The number of biclusters to find.

svd_method : {'randomized', 'arpack'}, default='randomized'
    Selects the algorithm for finding singular vectors. May be
    'randomized' or 'arpack'. If 'randomized', use
    :func:`sklearn.utils.extmath.randomized_svd`, which may be faster
    for large matrices. If 'arpack', use
    :func:`scipy.sparse.linalg.svds`, which is more accurate, but
    possibly slower in some cases.

n_svd_vecs : int, default=None
    Number of vectors to use in calculating the SVD. Corresponds
    to `ncv` when `svd_method=arpack` and `n_oversamples` when
    `svd_method` is 'randomized`.

mini_batch : bool, default=False
    Whether to use mini-batch k-means, which is faster but may get
    different results.

init : {'k-means++', 'random', or ndarray of shape             (n_clusters, n_features), default='k-means++'
    Method for initialization of k-means algorithm; defaults to
    'k-means++'.

n_init : int, default=10
    Number of random initializations that are tried with the
    k-means algorithm.

    If mini-batch k-means is used, the best initialization is
    chosen and the algorithm runs once. Otherwise, the algorithm
    is run for each initialization and the best solution chosen.

n_jobs : int, default=None
    The number of jobs to use for the computation. This works by breaking
    down the pairwise matrix into n_jobs even slices and computing them in
    parallel.

    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

random_state : int, RandomState instance, default=None
    Used for randomizing the singular value decomposition and the k-means
    initialization. Use an int to make the randomness deterministic.
    See :term:`Glossary <random_state>`.

Attributes
----------
rows_ : array-like of shape (n_row_clusters, n_rows)
    Results of the clustering. `rows[i, r]` is True if
    cluster `i` contains row `r`. Available only after calling ``fit``.

columns_ : array-like of shape (n_column_clusters, n_columns)
    Results of the clustering, like `rows`.

row_labels_ : array-like of shape (n_rows,)
    The bicluster label of each row.

column_labels_ : array-like of shape (n_cols,)
    The bicluster label of each column.

Examples
--------
>>> from sklearn.cluster import SpectralCoclustering
>>> import numpy as np
>>> X = np.array([[1, 1], [2, 1], [1, 0],
...               [4, 7], [3, 5], [3, 6]])
>>> clustering = SpectralCoclustering(n_clusters=2, random_state=0).fit(X)
>>> clustering.row_labels_ #doctest: +SKIP
array([0, 1, 1, 0, 0, 0], dtype=int32)
>>> clustering.column_labels_ #doctest: +SKIP
array([0, 0], dtype=int32)
>>> clustering
SpectralCoclustering(n_clusters=2, random_state=0)

References
----------

* Dhillon, Inderjit S, 2001. `Co-clustering documents and words using
  bipartite spectral graph partitioning
  <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.140.3011>`__.
*)

val fit : ?y:Py.Object.t -> x:Arr.t -> t -> t
(**
Creates a biclustering for X.

Parameters
----------
X : array-like, shape (n_samples, n_features)

y : Ignored
*)

val get_indices : i:int -> t -> Py.Object.t
(**
Row and column indices of the i'th bicluster.

Only works if ``rows_`` and ``columns_`` attributes exist.

Parameters
----------
i : int
    The index of the cluster.

Returns
-------
row_ind : np.array, dtype=np.intp
    Indices of rows in the dataset that belong to the bicluster.
col_ind : np.array, dtype=np.intp
    Indices of columns in the dataset that belong to the bicluster.
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

val get_shape : i:int -> t -> int
(**
Shape of the i'th bicluster.

Parameters
----------
i : int
    The index of the cluster.

Returns
-------
shape : (int, int)
    Number of rows and columns (resp.) in the bicluster.
*)

val get_submatrix : i:int -> data:Py.Object.t -> t -> Arr.t
(**
Return the submatrix corresponding to bicluster `i`.

Parameters
----------
i : int
    The index of the cluster.
data : array
    The data.

Returns
-------
submatrix : array
    The submatrix corresponding to bicluster i.

Notes
-----
Works with sparse matrices. Only works if ``rows_`` and
``columns_`` attributes exist.
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


(** Attribute rows_: get value or raise Not_found if None.*)
val rows_ : t -> Arr.t

(** Attribute rows_: get value as an option. *)
val rows_opt : t -> (Arr.t) option


(** Attribute columns_: get value or raise Not_found if None.*)
val columns_ : t -> Arr.t

(** Attribute columns_: get value as an option. *)
val columns_opt : t -> (Arr.t) option


(** Attribute row_labels_: get value or raise Not_found if None.*)
val row_labels_ : t -> Arr.t

(** Attribute row_labels_: get value as an option. *)
val row_labels_opt : t -> (Arr.t) option


(** Attribute column_labels_: get value or raise Not_found if None.*)
val column_labels_ : t -> Arr.t

(** Attribute column_labels_: get value as an option. *)
val column_labels_opt : t -> (Arr.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val affinity_propagation : ?preference:[`Arr of Arr.t | `F of float] -> ?convergence_iter:int -> ?max_iter:int -> ?damping:float -> ?copy:bool -> ?verbose:int -> ?return_n_iter:bool -> s:Arr.t -> unit -> (Arr.t * Arr.t * int)
(**
Perform Affinity Propagation Clustering of data

Read more in the :ref:`User Guide <affinity_propagation>`.

Parameters
----------

S : array-like, shape (n_samples, n_samples)
    Matrix of similarities between points

preference : array-like, shape (n_samples,) or float, optional
    Preferences for each point - points with larger values of
    preferences are more likely to be chosen as exemplars. The number of
    exemplars, i.e. of clusters, is influenced by the input preferences
    value. If the preferences are not passed as arguments, they will be
    set to the median of the input similarities (resulting in a moderate
    number of clusters). For a smaller amount of clusters, this can be set
    to the minimum value of the similarities.

convergence_iter : int, optional, default: 15
    Number of iterations with no change in the number
    of estimated clusters that stops the convergence.

max_iter : int, optional, default: 200
    Maximum number of iterations

damping : float, optional, default: 0.5
    Damping factor between 0.5 and 1.

copy : boolean, optional, default: True
    If copy is False, the affinity matrix is modified inplace by the
    algorithm, for memory efficiency

verbose : boolean, optional, default: False
    The verbosity level

return_n_iter : bool, default False
    Whether or not to return the number of iterations.

Returns
-------

cluster_centers_indices : array, shape (n_clusters,)
    index of clusters centers

labels : array, shape (n_samples,)
    cluster labels for each point

n_iter : int
    number of iterations run. Returned only if `return_n_iter` is
    set to True.

Notes
-----
For an example, see :ref:`examples/cluster/plot_affinity_propagation.py
<sphx_glr_auto_examples_cluster_plot_affinity_propagation.py>`.

When the algorithm does not converge, it returns an empty array as
``cluster_center_indices`` and ``-1`` as label for each training sample.

When all training samples have equal similarities and equal preferences,
the assignment of cluster centers and labels depends on the preference.
If the preference is smaller than the similarities, a single cluster center
and label ``0`` for every sample will be returned. Otherwise, every
training sample becomes its own cluster center and is assigned a unique
label.

References
----------
Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
Between Data Points", Science Feb. 2007
*)

val cluster_optics_dbscan : reachability:Arr.t -> core_distances:Arr.t -> ordering:Arr.t -> eps:float -> unit -> Arr.t
(**
Performs DBSCAN extraction for an arbitrary epsilon.

Extracting the clusters runs in linear time. Note that this results in
``labels_`` which are close to a :class:`~sklearn.cluster.DBSCAN` with
similar settings and ``eps``, only if ``eps`` is close to ``max_eps``.

Parameters
----------
reachability : array, shape (n_samples,)
    Reachability distances calculated by OPTICS (``reachability_``)

core_distances : array, shape (n_samples,)
    Distances at which points become core (``core_distances_``)

ordering : array, shape (n_samples,)
    OPTICS ordered point indices (``ordering_``)

eps : float
    DBSCAN ``eps`` parameter. Must be set to < ``max_eps``. Results
    will be close to DBSCAN algorithm if ``eps`` and ``max_eps`` are close
    to one another.

Returns
-------
labels_ : array, shape (n_samples,)
    The estimated labels.
*)

val cluster_optics_xi : ?min_cluster_size:[`I of int | `Float_between_0_and_1 of Py.Object.t] -> ?xi:[`F of float | `Between_0_and_1 of Py.Object.t] -> ?predecessor_correction:bool -> reachability:Arr.t -> predecessor:Arr.t -> ordering:Arr.t -> min_samples:[`I of int | `Float_between_0_and_1 of Py.Object.t] -> unit -> (Arr.t * Arr.t)
(**
Automatically extract clusters according to the Xi-steep method.

Parameters
----------
reachability : array, shape (n_samples,)
    Reachability distances calculated by OPTICS (`reachability_`)

predecessor : array, shape (n_samples,)
    Predecessors calculated by OPTICS.

ordering : array, shape (n_samples,)
    OPTICS ordered point indices (`ordering_`)

min_samples : int > 1 or float between 0 and 1
    The same as the min_samples given to OPTICS. Up and down steep regions
    can't have more then ``min_samples`` consecutive non-steep points.
    Expressed as an absolute number or a fraction of the number of samples
    (rounded to be at least 2).

min_cluster_size : int > 1 or float between 0 and 1 (default=None)
    Minimum number of samples in an OPTICS cluster, expressed as an
    absolute number or a fraction of the number of samples (rounded to be
    at least 2). If ``None``, the value of ``min_samples`` is used instead.

xi : float, between 0 and 1, optional (default=0.05)
    Determines the minimum steepness on the reachability plot that
    constitutes a cluster boundary. For example, an upwards point in the
    reachability plot is defined by the ratio from one point to its
    successor being at most 1-xi.

predecessor_correction : bool, optional (default=True)
    Correct clusters based on the calculated predecessors.

Returns
-------
labels : array, shape (n_samples)
    The labels assigned to samples. Points which are not included
    in any cluster are labeled as -1.

clusters : array, shape (n_clusters, 2)
    The list of clusters in the form of ``[start, end]`` in each row, with
    all indices inclusive. The clusters are ordered according to ``(end,
    -start)`` (ascending) so that larger clusters encompassing smaller
    clusters come after such nested smaller clusters. Since ``labels`` does
    not reflect the hierarchy, usually ``len(clusters) >
    np.unique(labels)``.
*)

val compute_optics_graph : x:Arr.t -> min_samples:[`I of int | `Float_between_0_and_1 of Py.Object.t] -> max_eps:float -> metric:[`S of string | `Callable of Py.Object.t] -> p:int -> metric_params:Dict.t -> algorithm:[`Auto | `Ball_tree | `Kd_tree | `Brute] -> leaf_size:int -> n_jobs:[`I of int | `None] -> unit -> (Arr.t * Arr.t * Arr.t * Arr.t)
(**
Computes the OPTICS reachability graph.

Read more in the :ref:`User Guide <optics>`.

Parameters
----------
X : array, shape (n_samples, n_features), or (n_samples, n_samples)  if metric=’precomputed’.
    A feature array, or array of distances between samples if
    metric='precomputed'

min_samples : int > 1 or float between 0 and 1
    The number of samples in a neighborhood for a point to be considered
    as a core point. Expressed as an absolute number or a fraction of the
    number of samples (rounded to be at least 2).

max_eps : float, optional (default=np.inf)
    The maximum distance between two samples for one to be considered as
    in the neighborhood of the other. Default value of ``np.inf`` will
    identify clusters across all scales; reducing ``max_eps`` will result
    in shorter run times.

metric : string or callable, optional (default='minkowski')
    Metric to use for distance computation. Any metric from scikit-learn
    or scipy.spatial.distance can be used.

    If metric is a callable function, it is called on each
    pair of instances (rows) and the resulting value recorded. The callable
    should take two arrays as input and return one value indicating the
    distance between them. This works for Scipy's metrics, but is less
    efficient than passing the metric name as a string. If metric is
    "precomputed", X is assumed to be a distance matrix and must be square.

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

p : integer, optional (default=2)
    Parameter for the Minkowski metric from
    :class:`sklearn.metrics.pairwise_distances`. When p = 1, this is
    equivalent to using manhattan_distance (l1), and euclidean_distance
    (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

metric_params : dict, optional (default=None)
    Additional keyword arguments for the metric function.

algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
    Algorithm used to compute the nearest neighbors:

    - 'ball_tree' will use :class:`BallTree`
    - 'kd_tree' will use :class:`KDTree`
    - 'brute' will use a brute-force search.
    - 'auto' will attempt to decide the most appropriate algorithm
      based on the values passed to :meth:`fit` method. (default)

    Note: fitting on sparse input will override the setting of
    this parameter, using brute force.

leaf_size : int, optional (default=30)
    Leaf size passed to :class:`BallTree` or :class:`KDTree`. This can
    affect the speed of the construction and query, as well as the memory
    required to store the tree. The optimal value depends on the
    nature of the problem.

n_jobs : int or None, optional (default=None)
    The number of parallel jobs to run for neighbors search.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

Returns
-------
ordering_ : array, shape (n_samples,)
    The cluster ordered list of sample indices.

core_distances_ : array, shape (n_samples,)
    Distance at which each sample becomes a core point, indexed by object
    order. Points which will never be core have a distance of inf. Use
    ``clust.core_distances_[clust.ordering_]`` to access in cluster order.

reachability_ : array, shape (n_samples,)
    Reachability distances per sample, indexed by object order. Use
    ``clust.reachability_[clust.ordering_]`` to access in cluster order.

predecessor_ : array, shape (n_samples,)
    Point that a sample was reached from, indexed by object order.
    Seed points have a predecessor of -1.

References
----------
.. [1] Ankerst, Mihael, Markus M. Breunig, Hans-Peter Kriegel,
   and Jörg Sander. "OPTICS: ordering points to identify the clustering
   structure." ACM SIGMOD Record 28, no. 2 (1999): 49-60.
*)

val dbscan : ?eps:float -> ?min_samples:int -> ?metric:[`S of string | `Callable of Py.Object.t] -> ?metric_params:Dict.t -> ?algorithm:[`Auto | `Ball_tree | `Kd_tree | `Brute] -> ?leaf_size:int -> ?p:float -> ?sample_weight:Arr.t -> ?n_jobs:int -> x:[`Arr of Arr.t | `Sparse_CSR_matrix of Py.Object.t] -> unit -> (Py.Object.t * Py.Object.t)
(**
Perform DBSCAN clustering from vector array or distance matrix.

Read more in the :ref:`User Guide <dbscan>`.

Parameters
----------
X : array or sparse (CSR) matrix of shape (n_samples, n_features), or             array of shape (n_samples, n_samples)
    A feature array, or array of distances between samples if
    ``metric='precomputed'``.

eps : float, optional
    The maximum distance between two samples for one to be considered
    as in the neighborhood of the other. This is not a maximum bound
    on the distances of points within a cluster. This is the most
    important DBSCAN parameter to choose appropriately for your data set
    and distance function.

min_samples : int, optional
    The number of samples (or total weight) in a neighborhood for a point
    to be considered as a core point. This includes the point itself.

metric : string, or callable
    The metric to use when calculating distance between instances in a
    feature array. If metric is a string or callable, it must be one of
    the options allowed by :func:`sklearn.metrics.pairwise_distances` for
    its metric parameter.
    If metric is "precomputed", X is assumed to be a distance matrix and
    must be square during fit. X may be a :term:`Glossary <sparse graph>`,
    in which case only "nonzero" elements may be considered neighbors.

metric_params : dict, optional
    Additional keyword arguments for the metric function.

    .. versionadded:: 0.19

algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
    The algorithm to be used by the NearestNeighbors module
    to compute pointwise distances and find nearest neighbors.
    See NearestNeighbors module documentation for details.

leaf_size : int, optional (default = 30)
    Leaf size passed to BallTree or cKDTree. This can affect the speed
    of the construction and query, as well as the memory required
    to store the tree. The optimal value depends
    on the nature of the problem.

p : float, optional
    The power of the Minkowski metric to be used to calculate distance
    between points.

sample_weight : array, shape (n_samples,), optional
    Weight of each sample, such that a sample with a weight of at least
    ``min_samples`` is by itself a core sample; a sample with negative
    weight may inhibit its eps-neighbor from being core.
    Note that weights are absolute, and default to 1.

n_jobs : int or None, optional (default=None)
    The number of parallel jobs to run for neighbors search.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

Returns
-------
core_samples : array [n_core_samples]
    Indices of core samples.

labels : array [n_samples]
    Cluster labels for each point.  Noisy samples are given the label -1.

See also
--------
DBSCAN
    An estimator interface for this clustering algorithm.
OPTICS
    A similar estimator interface clustering at multiple values of eps. Our
    implementation is optimized for memory usage.

Notes
-----
For an example, see :ref:`examples/cluster/plot_dbscan.py
<sphx_glr_auto_examples_cluster_plot_dbscan.py>`.

This implementation bulk-computes all neighborhood queries, which increases
the memory complexity to O(n.d) where d is the average number of neighbors,
while original DBSCAN had memory complexity O(n). It may attract a higher
memory complexity when querying these nearest neighborhoods, depending
on the ``algorithm``.

One way to avoid the query complexity is to pre-compute sparse
neighborhoods in chunks using
:func:`NearestNeighbors.radius_neighbors_graph
<sklearn.neighbors.NearestNeighbors.radius_neighbors_graph>` with
``mode='distance'``, then using ``metric='precomputed'`` here.

Another way to reduce memory and computation time is to remove
(near-)duplicate points and use ``sample_weight`` instead.

:func:`cluster.optics <sklearn.cluster.optics>` provides a similar
clustering with lower memory usage.

References
----------
Ester, M., H. P. Kriegel, J. Sander, and X. Xu, "A Density-Based
Algorithm for Discovering Clusters in Large Spatial Databases with Noise".
In: Proceedings of the 2nd International Conference on Knowledge Discovery
and Data Mining, Portland, OR, AAAI Press, pp. 226-231. 1996

Schubert, E., Sander, J., Ester, M., Kriegel, H. P., & Xu, X. (2017).
DBSCAN revisited, revisited: why and how you should (still) use DBSCAN.
ACM Transactions on Database Systems (TODS), 42(3), 19.
*)

val estimate_bandwidth : ?quantile:float -> ?n_samples:int -> ?random_state:int -> ?n_jobs:int -> x:Arr.t -> unit -> float
(**
Estimate the bandwidth to use with the mean-shift algorithm.

That this function takes time at least quadratic in n_samples. For large
datasets, it's wise to set that parameter to a small value.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Input points.

quantile : float, default 0.3
    should be between [0, 1]
    0.5 means that the median of all pairwise distances is used.

n_samples : int, optional
    The number of samples to use. If not given, all samples are used.

random_state : int, RandomState instance or None (default)
    The generator used to randomly select the samples from input points
    for bandwidth estimation. Use an int to make the randomness
    deterministic.
    See :term:`Glossary <random_state>`.

n_jobs : int or None, optional (default=None)
    The number of parallel jobs to run for neighbors search.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

Returns
-------
bandwidth : float
    The bandwidth parameter.
*)

val get_bin_seeds : ?min_bin_freq:int -> x:Arr.t -> bin_size:float -> unit -> Arr.t
(**
Finds seeds for mean_shift.

Finds seeds by first binning data onto a grid whose lines are
spaced bin_size apart, and then choosing those bins with at least
min_bin_freq points.

Parameters
----------

X : array-like of shape (n_samples, n_features)
    Input points, the same points that will be used in mean_shift.

bin_size : float
    Controls the coarseness of the binning. Smaller values lead
    to more seeding (which is computationally more expensive). If you're
    not sure how to set this, set it to the value of the bandwidth used
    in clustering.mean_shift.

min_bin_freq : integer, optional
    Only bins with at least min_bin_freq will be selected as seeds.
    Raising this value decreases the number of seeds found, which
    makes mean_shift computationally cheaper.

Returns
-------
bin_seeds : array-like of shape (n_samples, n_features)
    Points used as initial kernel positions in clustering.mean_shift.
*)

val k_means : ?sample_weight:Arr.t -> ?init:[`K_means_ | `Random | `Arr of Arr.t | `A_callable of Py.Object.t] -> ?precompute_distances:[`Bool of bool | `Auto] -> ?n_init:int -> ?max_iter:int -> ?verbose:int -> ?tol:float -> ?random_state:int -> ?copy_x:bool -> ?n_jobs:int -> ?algorithm:[`Auto | `Full | `Elkan] -> ?return_n_iter:bool -> x:Arr.t -> n_clusters:int -> unit -> (Py.Object.t * Py.Object.t * float * int)
(**
K-means clustering algorithm.

Read more in the :ref:`User Guide <k_means>`.

Parameters
----------
X : array-like or sparse matrix, shape (n_samples, n_features)
    The observations to cluster. It must be noted that the data
    will be converted to C ordering, which will cause a memory copy
    if the given data is not C-contiguous.

n_clusters : int
    The number of clusters to form as well as the number of
    centroids to generate.

sample_weight : array-like, shape (n_samples,), optional
    The weights for each observation in X. If None, all observations
    are assigned equal weight (default: None)

init : {'k-means++', 'random', or ndarray, or a callable}, optional
    Method for initialization, default to 'k-means++':

    'k-means++' : selects initial cluster centers for k-mean
    clustering in a smart way to speed up convergence. See section
    Notes in k_init for more details.

    'random': choose k observations (rows) at random from data for
    the initial centroids.

    If an ndarray is passed, it should be of shape (n_clusters, n_features)
    and gives the initial centers.

    If a callable is passed, it should take arguments X, k and
    and a random state and return an initialization.

precompute_distances : {'auto', True, False}
    Precompute distances (faster but takes more memory).

    'auto' : do not precompute distances if n_samples * n_clusters > 12
    million. This corresponds to about 100MB overhead per job using
    double precision.

    True : always precompute distances

    False : never precompute distances

n_init : int, optional, default: 10
    Number of time the k-means algorithm will be run with different
    centroid seeds. The final results will be the best output of
    n_init consecutive runs in terms of inertia.

max_iter : int, optional, default 300
    Maximum number of iterations of the k-means algorithm to run.

verbose : boolean, optional
    Verbosity mode.

tol : float, optional
    The relative increment in the results before declaring convergence.

random_state : int, RandomState instance or None (default)
    Determines random number generation for centroid initialization. Use
    an int to make the randomness deterministic.
    See :term:`Glossary <random_state>`.

copy_x : bool, optional
    When pre-computing distances it is more numerically accurate to center
    the data first.  If copy_x is True (default), then the original data is
    not modified, ensuring X is C-contiguous.  If False, the original data
    is modified, and put back before the function returns, but small
    numerical differences may be introduced by subtracting and then adding
    the data mean, in this case it will also not ensure that data is
    C-contiguous which may cause a significant slowdown.

n_jobs : int or None, optional (default=None)
    The number of jobs to use for the computation. This works by computing
    each of the n_init runs in parallel.

    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

algorithm : "auto", "full" or "elkan", default="auto"
    K-means algorithm to use. The classical EM-style algorithm is "full".
    The "elkan" variation is more efficient by using the triangle
    inequality, but currently doesn't support sparse data. "auto" chooses
    "elkan" for dense data and "full" for sparse data.

return_n_iter : bool, optional
    Whether or not to return the number of iterations.

Returns
-------
centroid : float ndarray with shape (k, n_features)
    Centroids found at the last iteration of k-means.

label : integer ndarray with shape (n_samples,)
    label[i] is the code or index of the centroid the
    i'th observation is closest to.

inertia : float
    The final value of the inertia criterion (sum of squared distances to
    the closest centroid for all observations in the training set).

best_n_iter : int
    Number of iterations corresponding to the best results.
    Returned only if `return_n_iter` is set to True.
*)

val linkage_tree : ?connectivity:Csr_matrix.t -> ?n_clusters:int -> ?linkage:[`Average | `Complete | `Single] -> ?affinity:[`S of string | `Callable of Py.Object.t] -> x:Arr.t -> unit -> (Py.Object.t * int * int * Py.Object.t * Arr.t)
(**
Linkage agglomerative clustering based on a Feature matrix.

The inertia matrix uses a Heapq-based representation.

This is the structured version, that takes into account some topological
structure between samples.

Read more in the :ref:`User Guide <hierarchical_clustering>`.

Parameters
----------
X : array, shape (n_samples, n_features)
    feature matrix representing n_samples samples to be clustered

connectivity : sparse matrix (optional).
    connectivity matrix. Defines for each sample the neighboring samples
    following a given structure of the data. The matrix is assumed to
    be symmetric and only the upper triangular half is used.
    Default is None, i.e, the Ward algorithm is unstructured.

n_clusters : int (optional)
    Stop early the construction of the tree at n_clusters. This is
    useful to decrease computation time if the number of clusters is
    not small compared to the number of samples. In this case, the
    complete tree is not computed, thus the 'children' output is of
    limited use, and the 'parents' output should rather be used.
    This option is valid only when specifying a connectivity matrix.

linkage : {"average", "complete", "single"}, optional, default: "complete"
    Which linkage criteria to use. The linkage criterion determines which
    distance to use between sets of observation.
        - average uses the average of the distances of each observation of
          the two sets
        - complete or maximum linkage uses the maximum distances between
          all observations of the two sets.
        - single uses the minimum of the distances between all observations
          of the two sets.

affinity : string or callable, optional, default: "euclidean".
    which metric to use. Can be "euclidean", "manhattan", or any
    distance know to paired distance (see metric.pairwise)

return_distance : bool, default False
    whether or not to return the distances between the clusters.

Returns
-------
children : 2D array, shape (n_nodes-1, 2)
    The children of each non-leaf node. Values less than `n_samples`
    correspond to leaves of the tree which are the original samples.
    A node `i` greater than or equal to `n_samples` is a non-leaf
    node and has children `children_[i - n_samples]`. Alternatively
    at the i-th iteration, children[i][0] and children[i][1]
    are merged to form node `n_samples + i`

n_connected_components : int
    The number of connected components in the graph.

n_leaves : int
    The number of leaves in the tree.

parents : 1D array, shape (n_nodes, ) or None
    The parent of each node. Only returned when a connectivity matrix
    is specified, elsewhere 'None' is returned.

distances : ndarray, shape (n_nodes-1,)
    Returned when return_distance is set to True.

    distances[i] refers to the distance between children[i][0] and
    children[i][1] when they are merged.

See also
--------
ward_tree : hierarchical clustering with ward linkage
*)

val mean_shift : ?bandwidth:float -> ?seeds:Arr.t -> ?bin_seeding:bool -> ?min_bin_freq:int -> ?cluster_all:bool -> ?max_iter:int -> ?n_jobs:int -> x:Arr.t -> unit -> (Arr.t * Arr.t)
(**
Perform mean shift clustering of data using a flat kernel.

Read more in the :ref:`User Guide <mean_shift>`.

Parameters
----------

X : array-like of shape (n_samples, n_features)
    Input data.

bandwidth : float, optional
    Kernel bandwidth.

    If bandwidth is not given, it is determined using a heuristic based on
    the median of all pairwise distances. This will take quadratic time in
    the number of samples. The sklearn.cluster.estimate_bandwidth function
    can be used to do this more efficiently.

seeds : array-like of shape (n_seeds, n_features) or None
    Point used as initial kernel locations. If None and bin_seeding=False,
    each data point is used as a seed. If None and bin_seeding=True,
    see bin_seeding.

bin_seeding : boolean, default=False
    If true, initial kernel locations are not locations of all
    points, but rather the location of the discretized version of
    points, where points are binned onto a grid whose coarseness
    corresponds to the bandwidth. Setting this option to True will speed
    up the algorithm because fewer seeds will be initialized.
    Ignored if seeds argument is not None.

min_bin_freq : int, default=1
   To speed up the algorithm, accept only those bins with at least
   min_bin_freq points as seeds.

cluster_all : boolean, default True
    If true, then all points are clustered, even those orphans that are
    not within any kernel. Orphans are assigned to the nearest kernel.
    If false, then orphans are given cluster label -1.

max_iter : int, default 300
    Maximum number of iterations, per seed point before the clustering
    operation terminates (for that seed point), if has not converged yet.

n_jobs : int or None, optional (default=None)
    The number of jobs to use for the computation. This works by computing
    each of the n_init runs in parallel.

    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

    .. versionadded:: 0.17
       Parallel Execution using *n_jobs*.

Returns
-------

cluster_centers : array, shape=[n_clusters, n_features]
    Coordinates of cluster centers.

labels : array, shape=[n_samples]
    Cluster labels for each point.

Notes
-----
For an example, see :ref:`examples/cluster/plot_mean_shift.py
<sphx_glr_auto_examples_cluster_plot_mean_shift.py>`.
*)

val spectral_clustering : ?n_clusters:int -> ?n_components:int -> ?eigen_solver:[`Arpack | `Lobpcg | `Amg] -> ?random_state:int -> ?n_init:int -> ?eigen_tol:float -> ?assign_labels:[`Kmeans | `Discretize] -> affinity:Arr.t -> unit -> Py.Object.t
(**
Apply clustering to a projection of the normalized Laplacian.

In practice Spectral Clustering is very useful when the structure of
the individual clusters is highly non-convex or more generally when
a measure of the center and spread of the cluster is not a suitable
description of the complete cluster. For instance, when clusters are
nested circles on the 2D plane.

If affinity is the adjacency matrix of a graph, this method can be
used to find normalized graph cuts.

Read more in the :ref:`User Guide <spectral_clustering>`.

Parameters
----------
affinity : array-like or sparse matrix, shape: (n_samples, n_samples)
    The affinity matrix describing the relationship of the samples to
    embed. **Must be symmetric**.

    Possible examples:
      - adjacency matrix of a graph,
      - heat kernel of the pairwise distance matrix of the samples,
      - symmetric k-nearest neighbours connectivity matrix of the samples.

n_clusters : integer, optional
    Number of clusters to extract.

n_components : integer, optional, default is n_clusters
    Number of eigen vectors to use for the spectral embedding

eigen_solver : {None, 'arpack', 'lobpcg', or 'amg'}
    The eigenvalue decomposition strategy to use. AMG requires pyamg
    to be installed. It can be faster on very large, sparse problems,
    but may also lead to instabilities

random_state : int, RandomState instance or None (default)
    A pseudo random number generator used for the initialization of the
    lobpcg eigen vectors decomposition when eigen_solver == 'amg' and by
    the K-Means initialization. Use an int to make the randomness
    deterministic.
    See :term:`Glossary <random_state>`.

n_init : int, optional, default: 10
    Number of time the k-means algorithm will be run with different
    centroid seeds. The final results will be the best output of
    n_init consecutive runs in terms of inertia.

eigen_tol : float, optional, default: 0.0
    Stopping criterion for eigendecomposition of the Laplacian matrix
    when using arpack eigen_solver.

assign_labels : {'kmeans', 'discretize'}, default: 'kmeans'
    The strategy to use to assign labels in the embedding
    space.  There are two ways to assign labels after the laplacian
    embedding.  k-means can be applied and is a popular choice. But it can
    also be sensitive to initialization. Discretization is another
    approach which is less sensitive to random initialization. See
    the 'Multiclass spectral clustering' paper referenced below for
    more details on the discretization approach.

Returns
-------
labels : array of integers, shape: n_samples
    The labels of the clusters.

References
----------

- Normalized cuts and image segmentation, 2000
  Jianbo Shi, Jitendra Malik
  http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.160.2324

- A Tutorial on Spectral Clustering, 2007
  Ulrike von Luxburg
  http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.9323

- Multiclass spectral clustering, 2003
  Stella X. Yu, Jianbo Shi
  https://www1.icsi.berkeley.edu/~stellayu/publication/doc/2003kwayICCV.pdf

Notes
-----
The graph should contain only one connect component, elsewhere
the results make little sense.

This algorithm solves the normalized cut for k=2: it is a
normalized spectral clustering.
*)

val ward_tree : ?connectivity:Csr_matrix.t -> ?n_clusters:int -> x:Arr.t -> unit -> (Py.Object.t * int * int * Py.Object.t * Py.Object.t * Py.Object.t)
(**
Ward clustering based on a Feature matrix.

Recursively merges the pair of clusters that minimally increases
within-cluster variance.

The inertia matrix uses a Heapq-based representation.

This is the structured version, that takes into account some topological
structure between samples.

Read more in the :ref:`User Guide <hierarchical_clustering>`.

Parameters
----------
X : array, shape (n_samples, n_features)
    feature matrix representing n_samples samples to be clustered

connectivity : sparse matrix (optional).
    connectivity matrix. Defines for each sample the neighboring samples
    following a given structure of the data. The matrix is assumed to
    be symmetric and only the upper triangular half is used.
    Default is None, i.e, the Ward algorithm is unstructured.

n_clusters : int (optional)
    Stop early the construction of the tree at n_clusters. This is
    useful to decrease computation time if the number of clusters is
    not small compared to the number of samples. In this case, the
    complete tree is not computed, thus the 'children' output is of
    limited use, and the 'parents' output should rather be used.
    This option is valid only when specifying a connectivity matrix.

return_distance : bool (optional)
    If True, return the distance between the clusters.

Returns
-------
children : 2D array, shape (n_nodes-1, 2)
    The children of each non-leaf node. Values less than `n_samples`
    correspond to leaves of the tree which are the original samples.
    A node `i` greater than or equal to `n_samples` is a non-leaf
    node and has children `children_[i - n_samples]`. Alternatively
    at the i-th iteration, children[i][0] and children[i][1]
    are merged to form node `n_samples + i`

n_connected_components : int
    The number of connected components in the graph.

n_leaves : int
    The number of leaves in the tree

parents : 1D array, shape (n_nodes, ) or None
    The parent of each node. Only returned when a connectivity matrix
    is specified, elsewhere 'None' is returned.

distances : 1D array, shape (n_nodes-1, )
    Only returned if return_distance is set to True (for compatibility).
    The distances between the centers of the nodes. `distances[i]`
    corresponds to a weighted euclidean distance between
    the nodes `children[i, 1]` and `children[i, 2]`. If the nodes refer to
    leaves of the tree, then `distances[i]` is their unweighted euclidean
    distance. Distances are updated in the following way
    (from scipy.hierarchy.linkage):

    The new entry :math:`d(u,v)` is computed as follows,

    .. math::

       d(u,v) = \sqrt{\frac{ |v|+|s| }
                           {T}d(v,s)^2
                    + \frac{ |v|+|t| }
                           {T}d(v,t)^2
                    - \frac{ |v| }
                           {T}d(s,t)^2}

    where :math:`u` is the newly joined cluster consisting of
    clusters :math:`s` and :math:`t`, :math:`v` is an unused
    cluster in the forest, :math:`T=|v|+|s|+|t|`, and
    :math:`|*|` is the cardinality of its argument. This is also
    known as the incremental algorithm.
*)

