(** Get an attribute of this module as a Py.Object.t. This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module Isomap : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_neighbors:int -> ?n_components:int -> ?eigen_solver:[`Auto | `Arpack | `Dense] -> ?tol:float -> ?max_iter:int -> ?path_method:[`Auto | `FW | `D] -> ?neighbors_algorithm:[`Auto | `Brute | `Kd_tree | `Ball_tree] -> ?n_jobs:int -> ?metric:[`S of string | `Callable of Py.Object.t] -> ?p:int -> ?metric_params:Dict.t -> unit -> t
(**
Isomap Embedding

Non-linear dimensionality reduction through Isometric Mapping

Read more in the :ref:`User Guide <isomap>`.

Parameters
----------
n_neighbors : integer
    number of neighbors to consider for each point.

n_components : integer
    number of coordinates for the manifold

eigen_solver : ['auto'|'arpack'|'dense']
    'auto' : Attempt to choose the most efficient solver
    for the given problem.

    'arpack' : Use Arnoldi decomposition to find the eigenvalues
    and eigenvectors.

    'dense' : Use a direct solver (i.e. LAPACK)
    for the eigenvalue decomposition.

tol : float
    Convergence tolerance passed to arpack or lobpcg.
    not used if eigen_solver == 'dense'.

max_iter : integer
    Maximum number of iterations for the arpack solver.
    not used if eigen_solver == 'dense'.

path_method : string ['auto'|'FW'|'D']
    Method to use in finding shortest path.

    'auto' : attempt to choose the best algorithm automatically.

    'FW' : Floyd-Warshall algorithm.

    'D' : Dijkstra's algorithm.

neighbors_algorithm : string ['auto'|'brute'|'kd_tree'|'ball_tree']
    Algorithm to use for nearest neighbors search,
    passed to neighbors.NearestNeighbors instance.

n_jobs : int or None, default=None
    The number of parallel jobs to run.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

metric : string, or callable, default="minkowski"
    The metric to use when calculating distance between instances in a
    feature array. If metric is a string or callable, it must be one of
    the options allowed by :func:`sklearn.metrics.pairwise_distances` for
    its metric parameter.
    If metric is "precomputed", X is assumed to be a distance matrix and
    must be square. X may be a :term:`Glossary <sparse graph>`.

    .. versionadded:: 0.22

p : int, default=2
    Parameter for the Minkowski metric from
    sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
    equivalent to using manhattan_distance (l1), and euclidean_distance
    (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    .. versionadded:: 0.22

metric_params : dict, default=None
    Additional keyword arguments for the metric function.

    .. versionadded:: 0.22

Attributes
----------
embedding_ : array-like, shape (n_samples, n_components)
    Stores the embedding vectors.

kernel_pca_ : object
    :class:`~sklearn.decomposition.KernelPCA` object used to implement the
    embedding.

nbrs_ : sklearn.neighbors.NearestNeighbors instance
    Stores nearest neighbors instance, including BallTree or KDtree
    if applicable.

dist_matrix_ : array-like, shape (n_samples, n_samples)
    Stores the geodesic distance matrix of training data.

Examples
--------
>>> from sklearn.datasets import load_digits
>>> from sklearn.manifold import Isomap
>>> X, _ = load_digits(return_X_y=True)
>>> X.shape
(1797, 64)
>>> embedding = Isomap(n_components=2)
>>> X_transformed = embedding.fit_transform(X[:100])
>>> X_transformed.shape
(100, 2)

References
----------

.. [1] Tenenbaum, J.B.; De Silva, V.; & Langford, J.C. A global geometric
       framework for nonlinear dimensionality reduction. Science 290 (5500)
*)

val fit : ?y:Py.Object.t -> x:[`Arr of Arr.t | `PyObject of Py.Object.t] -> t -> t
(**
Compute the embedding vectors for data X

Parameters
----------
X : {array-like, sparse graph, BallTree, KDTree, NearestNeighbors}
    Sample data, shape = (n_samples, n_features), in the form of a
    numpy array, sparse graph, precomputed tree, or NearestNeighbors
    object.

y : Ignored

Returns
-------
self : returns an instance of self.
*)

val fit_transform : ?y:Py.Object.t -> x:Arr.t -> t -> Arr.t
(**
Fit the model from data in X and transform X.

Parameters
----------
X : {array-like, sparse graph, BallTree, KDTree}
    Training vector, where n_samples in the number of samples
    and n_features is the number of features.

y : Ignored

Returns
-------
X_new : array-like, shape (n_samples, n_components)
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

val reconstruction_error : t -> float
(**
Compute the reconstruction error for the embedding.

Returns
-------
reconstruction_error : float

Notes
-----
The cost function of an isomap embedding is

``E = frobenius_norm[K(D) - K(D_fit)] / n_samples``

Where D is the matrix of distances for the input data X,
D_fit is the matrix of distances for the output embedding X_fit,
and K is the isomap kernel:

``K(D) = -0.5 * (I - 1/n_samples) * D^2 * (I - 1/n_samples)``
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
Transform X.

This is implemented by linking the points X into the graph of geodesic
distances of the training data. First the `n_neighbors` nearest
neighbors of X are found in the training data, and from these the
shortest geodesic distances from each point in X to each point in
the training data are computed in order to construct the kernel.
The embedding of X is the projection of this kernel onto the
embedding vectors of the training set.

Parameters
----------
X : array-like, shape (n_queries, n_features)
    If neighbors_algorithm='precomputed', X is assumed to be a
    distance matrix or a sparse graph of shape
    (n_queries, n_samples_fit).

Returns
-------
X_new : array-like, shape (n_queries, n_components)
*)


(** Attribute embedding_: get value or raise Not_found if None.*)
val embedding_ : t -> Arr.t

(** Attribute embedding_: get value as an option. *)
val embedding_opt : t -> (Arr.t) option


(** Attribute kernel_pca_: get value or raise Not_found if None.*)
val kernel_pca_ : t -> Py.Object.t

(** Attribute kernel_pca_: get value as an option. *)
val kernel_pca_opt : t -> (Py.Object.t) option


(** Attribute nbrs_: get value or raise Not_found if None.*)
val nbrs_ : t -> Py.Object.t

(** Attribute nbrs_: get value as an option. *)
val nbrs_opt : t -> (Py.Object.t) option


(** Attribute dist_matrix_: get value or raise Not_found if None.*)
val dist_matrix_ : t -> Arr.t

(** Attribute dist_matrix_: get value as an option. *)
val dist_matrix_opt : t -> (Arr.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module LocallyLinearEmbedding : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_neighbors:int -> ?n_components:int -> ?reg:float -> ?eigen_solver:[`Auto | `Arpack | `Dense] -> ?tol:float -> ?max_iter:int -> ?method_:[`Standard | `Hessian | `Modified | `Ltsa] -> ?hessian_tol:float -> ?modified_tol:float -> ?neighbors_algorithm:[`Auto | `Brute | `Kd_tree | `Ball_tree] -> ?random_state:int -> ?n_jobs:int -> unit -> t
(**
Locally Linear Embedding

Read more in the :ref:`User Guide <locally_linear_embedding>`.

Parameters
----------
n_neighbors : integer
    number of neighbors to consider for each point.

n_components : integer
    number of coordinates for the manifold

reg : float
    regularization constant, multiplies the trace of the local covariance
    matrix of the distances.

eigen_solver : string, {'auto', 'arpack', 'dense'}
    auto : algorithm will attempt to choose the best method for input data

    arpack : use arnoldi iteration in shift-invert mode.
                For this method, M may be a dense matrix, sparse matrix,
                or general linear operator.
                Warning: ARPACK can be unstable for some problems.  It is
                best to try several random seeds in order to check results.

    dense  : use standard dense matrix operations for the eigenvalue
                decomposition.  For this method, M must be an array
                or matrix type.  This method should be avoided for
                large problems.

tol : float, optional
    Tolerance for 'arpack' method
    Not used if eigen_solver=='dense'.

max_iter : integer
    maximum number of iterations for the arpack solver.
    Not used if eigen_solver=='dense'.

method : string ('standard', 'hessian', 'modified' or 'ltsa')
    standard : use the standard locally linear embedding algorithm.  see
               reference [1]
    hessian  : use the Hessian eigenmap method. This method requires
               ``n_neighbors > n_components * (1 + (n_components + 1) / 2``
               see reference [2]
    modified : use the modified locally linear embedding algorithm.
               see reference [3]
    ltsa     : use local tangent space alignment algorithm
               see reference [4]

hessian_tol : float, optional
    Tolerance for Hessian eigenmapping method.
    Only used if ``method == 'hessian'``

modified_tol : float, optional
    Tolerance for modified LLE method.
    Only used if ``method == 'modified'``

neighbors_algorithm : string ['auto'|'brute'|'kd_tree'|'ball_tree']
    algorithm to use for nearest neighbors search,
    passed to neighbors.NearestNeighbors instance

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`. Used when ``eigen_solver`` == 'arpack'.

n_jobs : int or None, optional (default=None)
    The number of parallel jobs to run.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

Attributes
----------
embedding_ : array-like, shape [n_samples, n_components]
    Stores the embedding vectors

reconstruction_error_ : float
    Reconstruction error associated with `embedding_`

nbrs_ : NearestNeighbors object
    Stores nearest neighbors instance, including BallTree or KDtree
    if applicable.

Examples
--------
>>> from sklearn.datasets import load_digits
>>> from sklearn.manifold import LocallyLinearEmbedding
>>> X, _ = load_digits(return_X_y=True)
>>> X.shape
(1797, 64)
>>> embedding = LocallyLinearEmbedding(n_components=2)
>>> X_transformed = embedding.fit_transform(X[:100])
>>> X_transformed.shape
(100, 2)

References
----------

.. [1] Roweis, S. & Saul, L. Nonlinear dimensionality reduction
    by locally linear embedding.  Science 290:2323 (2000).
.. [2] Donoho, D. & Grimes, C. Hessian eigenmaps: Locally
    linear embedding techniques for high-dimensional data.
    Proc Natl Acad Sci U S A.  100:5591 (2003).
.. [3] Zhang, Z. & Wang, J. MLLE: Modified Locally Linear
    Embedding Using Multiple Weights.
    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.70.382
.. [4] Zhang, Z. & Zha, H. Principal manifolds and nonlinear
    dimensionality reduction via tangent space alignment.
    Journal of Shanghai Univ.  8:406 (2004)
*)

val fit : ?y:Py.Object.t -> x:Arr.t -> t -> t
(**
Compute the embedding vectors for data X

Parameters
----------
X : array-like of shape [n_samples, n_features]
    training set.

y : Ignored

Returns
-------
self : returns an instance of self.
*)

val fit_transform : ?y:Py.Object.t -> x:Arr.t -> t -> Arr.t
(**
Compute the embedding vectors for data X and transform X.

Parameters
----------
X : array-like of shape [n_samples, n_features]
    training set.

y : Ignored

Returns
-------
X_new : array-like, shape (n_samples, n_components)
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

val transform : x:Arr.t -> t -> Arr.t
(**
Transform new points into embedding space.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
X_new : array, shape = [n_samples, n_components]

Notes
-----
Because of scaling performed by this method, it is discouraged to use
it together with methods that are not scale-invariant (like SVMs)
*)


(** Attribute embedding_: get value or raise Not_found if None.*)
val embedding_ : t -> Arr.t

(** Attribute embedding_: get value as an option. *)
val embedding_opt : t -> (Arr.t) option


(** Attribute reconstruction_error_: get value or raise Not_found if None.*)
val reconstruction_error_ : t -> float

(** Attribute reconstruction_error_: get value as an option. *)
val reconstruction_error_opt : t -> (float) option


(** Attribute nbrs_: get value or raise Not_found if None.*)
val nbrs_ : t -> Py.Object.t

(** Attribute nbrs_: get value as an option. *)
val nbrs_opt : t -> (Py.Object.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MDS : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_components:int -> ?metric:bool -> ?n_init:int -> ?max_iter:int -> ?verbose:int -> ?eps:float -> ?n_jobs:int -> ?random_state:int -> ?dissimilarity:[`Euclidean | `Precomputed] -> unit -> t
(**
Multidimensional scaling

Read more in the :ref:`User Guide <multidimensional_scaling>`.

Parameters
----------
n_components : int, optional, default: 2
    Number of dimensions in which to immerse the dissimilarities.

metric : boolean, optional, default: True
    If ``True``, perform metric MDS; otherwise, perform nonmetric MDS.

n_init : int, optional, default: 4
    Number of times the SMACOF algorithm will be run with different
    initializations. The final results will be the best output of the runs,
    determined by the run with the smallest final stress.

max_iter : int, optional, default: 300
    Maximum number of iterations of the SMACOF algorithm for a single run.

verbose : int, optional, default: 0
    Level of verbosity.

eps : float, optional, default: 1e-3
    Relative tolerance with respect to stress at which to declare
    convergence.

n_jobs : int or None, optional (default=None)
    The number of jobs to use for the computation. If multiple
    initializations are used (``n_init``), each run of the algorithm is
    computed in parallel.

    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

random_state : int, RandomState instance or None, optional, default: None
    The generator used to initialize the centers.  If int, random_state is
    the seed used by the random number generator; If RandomState instance,
    random_state is the random number generator; If None, the random number
    generator is the RandomState instance used by `np.random`.

dissimilarity : 'euclidean' | 'precomputed', optional, default: 'euclidean'
    Dissimilarity measure to use:

    - 'euclidean':
        Pairwise Euclidean distances between points in the dataset.

    - 'precomputed':
        Pre-computed dissimilarities are passed directly to ``fit`` and
        ``fit_transform``.

Attributes
----------
embedding_ : array-like, shape (n_samples, n_components)
    Stores the position of the dataset in the embedding space.

stress_ : float
    The final value of the stress (sum of squared distance of the
    disparities and the distances for all constrained points).

Examples
--------
>>> from sklearn.datasets import load_digits
>>> from sklearn.manifold import MDS
>>> X, _ = load_digits(return_X_y=True)
>>> X.shape
(1797, 64)
>>> embedding = MDS(n_components=2)
>>> X_transformed = embedding.fit_transform(X[:100])
>>> X_transformed.shape
(100, 2)

References
----------
"Modern Multidimensional Scaling - Theory and Applications" Borg, I.;
Groenen P. Springer Series in Statistics (1997)

"Nonmetric multidimensional scaling: a numerical method" Kruskal, J.
Psychometrika, 29 (1964)

"Multidimensional scaling by optimizing goodness of fit to a nonmetric
hypothesis" Kruskal, J. Psychometrika, 29, (1964)
*)

val fit : ?y:Py.Object.t -> ?init:Arr.t -> x:Arr.t -> t -> t
(**
Computes the position of the points in the embedding space

Parameters
----------
X : array, shape (n_samples, n_features) or (n_samples, n_samples)
    Input data. If ``dissimilarity=='precomputed'``, the input should
    be the dissimilarity matrix.

y : Ignored

init : ndarray, shape (n_samples,), optional, default: None
    Starting configuration of the embedding to initialize the SMACOF
    algorithm. By default, the algorithm is initialized with a randomly
    chosen array.
*)

val fit_transform : ?y:Py.Object.t -> ?init:Arr.t -> x:Arr.t -> t -> Arr.t
(**
Fit the data from X, and returns the embedded coordinates

Parameters
----------
X : array, shape (n_samples, n_features) or (n_samples, n_samples)
    Input data. If ``dissimilarity=='precomputed'``, the input should
    be the dissimilarity matrix.

y : Ignored

init : ndarray, shape (n_samples,), optional, default: None
    Starting configuration of the embedding to initialize the SMACOF
    algorithm. By default, the algorithm is initialized with a randomly
    chosen array.
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


(** Attribute embedding_: get value or raise Not_found if None.*)
val embedding_ : t -> Arr.t

(** Attribute embedding_: get value as an option. *)
val embedding_opt : t -> (Arr.t) option


(** Attribute stress_: get value or raise Not_found if None.*)
val stress_ : t -> float

(** Attribute stress_: get value as an option. *)
val stress_opt : t -> (float) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module SpectralEmbedding : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_components:int -> ?affinity:[`S of string | `Callable of Py.Object.t] -> ?gamma:float -> ?random_state:int -> ?eigen_solver:[`Arpack | `Lobpcg | `Amg] -> ?n_neighbors:int -> ?n_jobs:int -> unit -> t
(**
Spectral embedding for non-linear dimensionality reduction.

Forms an affinity matrix given by the specified function and
applies spectral decomposition to the corresponding graph laplacian.
The resulting transformation is given by the value of the
eigenvectors for each data point.

Note : Laplacian Eigenmaps is the actual algorithm implemented here.

Read more in the :ref:`User Guide <spectral_embedding>`.

Parameters
----------
n_components : integer, default: 2
    The dimension of the projected subspace.

affinity : string or callable, default : "nearest_neighbors"
    How to construct the affinity matrix.
     - 'nearest_neighbors' : construct the affinity matrix by computing a
       graph of nearest neighbors.
     - 'rbf' : construct the affinity matrix by computing a radial basis
       function (RBF) kernel.
     - 'precomputed' : interpret ``X`` as a precomputed affinity matrix.
     - 'precomputed_nearest_neighbors' : interpret ``X`` as a sparse graph
       of precomputed nearest neighbors, and constructs the affinity matrix
       by selecting the ``n_neighbors`` nearest neighbors.
     - callable : use passed in function as affinity
       the function takes in data matrix (n_samples, n_features)
       and return affinity matrix (n_samples, n_samples).

gamma : float, optional, default : 1/n_features
    Kernel coefficient for rbf kernel.

random_state : int, RandomState instance or None, optional, default: None
    A pseudo random number generator used for the initialization of the
    lobpcg eigenvectors.  If int, random_state is the seed used by the
    random number generator; If RandomState instance, random_state is the
    random number generator; If None, the random number generator is the
    RandomState instance used by `np.random`. Used when ``solver`` ==
    'amg'.

eigen_solver : {None, 'arpack', 'lobpcg', or 'amg'}
    The eigenvalue decomposition strategy to use. AMG requires pyamg
    to be installed. It can be faster on very large, sparse problems.

n_neighbors : int, default : max(n_samples/10 , 1)
    Number of nearest neighbors for nearest_neighbors graph building.

n_jobs : int or None, optional (default=None)
    The number of parallel jobs to run.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

Attributes
----------

embedding_ : array, shape = (n_samples, n_components)
    Spectral embedding of the training matrix.

affinity_matrix_ : array, shape = (n_samples, n_samples)
    Affinity_matrix constructed from samples or precomputed.

n_neighbors_ : int
    Number of nearest neighbors effectively used.

Examples
--------
>>> from sklearn.datasets import load_digits
>>> from sklearn.manifold import SpectralEmbedding
>>> X, _ = load_digits(return_X_y=True)
>>> X.shape
(1797, 64)
>>> embedding = SpectralEmbedding(n_components=2)
>>> X_transformed = embedding.fit_transform(X[:100])
>>> X_transformed.shape
(100, 2)

References
----------

- A Tutorial on Spectral Clustering, 2007
  Ulrike von Luxburg
  http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.9323

- On Spectral Clustering: Analysis and an algorithm, 2001
  Andrew Y. Ng, Michael I. Jordan, Yair Weiss
  http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.19.8100

- Normalized cuts and image segmentation, 2000
  Jianbo Shi, Jitendra Malik
  http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.160.2324
*)

val fit : ?y:Py.Object.t -> x:Arr.t -> t -> t
(**
Fit the model from data in X.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Training vector, where n_samples is the number of samples
    and n_features is the number of features.

    If affinity is "precomputed"
    X : {array-like, sparse matrix}, shape (n_samples, n_samples),
    Interpret X as precomputed adjacency graph computed from
    samples.

Returns
-------
self : object
    Returns the instance itself.
*)

val fit_transform : ?y:Py.Object.t -> x:Arr.t -> t -> Arr.t
(**
Fit the model from data in X and transform X.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Training vector, where n_samples is the number of samples
    and n_features is the number of features.

    If affinity is "precomputed"
    X : {array-like, sparse matrix}, shape (n_samples, n_samples),
    Interpret X as precomputed adjacency graph computed from
    samples.

Returns
-------
X_new : array-like, shape (n_samples, n_components)
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


(** Attribute embedding_: get value or raise Not_found if None.*)
val embedding_ : t -> Arr.t

(** Attribute embedding_: get value as an option. *)
val embedding_opt : t -> (Arr.t) option


(** Attribute affinity_matrix_: get value or raise Not_found if None.*)
val affinity_matrix_ : t -> Arr.t

(** Attribute affinity_matrix_: get value as an option. *)
val affinity_matrix_opt : t -> (Arr.t) option


(** Attribute n_neighbors_: get value or raise Not_found if None.*)
val n_neighbors_ : t -> int

(** Attribute n_neighbors_: get value as an option. *)
val n_neighbors_opt : t -> (int) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module TSNE : sig
type t
val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t

val create : ?n_components:int -> ?perplexity:float -> ?early_exaggeration:float -> ?learning_rate:float -> ?n_iter:int -> ?n_iter_without_progress:int -> ?min_grad_norm:float -> ?metric:[`S of string | `Callable of Py.Object.t] -> ?init:[`S of string | `Arr of Arr.t] -> ?verbose:int -> ?random_state:int -> ?method_:string -> ?angle:float -> ?n_jobs:int -> unit -> t
(**
t-distributed Stochastic Neighbor Embedding.

t-SNE [1] is a tool to visualize high-dimensional data. It converts
similarities between data points to joint probabilities and tries
to minimize the Kullback-Leibler divergence between the joint
probabilities of the low-dimensional embedding and the
high-dimensional data. t-SNE has a cost function that is not convex,
i.e. with different initializations we can get different results.

It is highly recommended to use another dimensionality reduction
method (e.g. PCA for dense data or TruncatedSVD for sparse data)
to reduce the number of dimensions to a reasonable amount (e.g. 50)
if the number of features is very high. This will suppress some
noise and speed up the computation of pairwise distances between
samples. For more tips see Laurens van der Maaten's FAQ [2].

Read more in the :ref:`User Guide <t_sne>`.

Parameters
----------
n_components : int, optional (default: 2)
    Dimension of the embedded space.

perplexity : float, optional (default: 30)
    The perplexity is related to the number of nearest neighbors that
    is used in other manifold learning algorithms. Larger datasets
    usually require a larger perplexity. Consider selecting a value
    between 5 and 50. Different values can result in significanlty
    different results.

early_exaggeration : float, optional (default: 12.0)
    Controls how tight natural clusters in the original space are in
    the embedded space and how much space will be between them. For
    larger values, the space between natural clusters will be larger
    in the embedded space. Again, the choice of this parameter is not
    very critical. If the cost function increases during initial
    optimization, the early exaggeration factor or the learning rate
    might be too high.

learning_rate : float, optional (default: 200.0)
    The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
    the learning rate is too high, the data may look like a 'ball' with any
    point approximately equidistant from its nearest neighbours. If the
    learning rate is too low, most points may look compressed in a dense
    cloud with few outliers. If the cost function gets stuck in a bad local
    minimum increasing the learning rate may help.

n_iter : int, optional (default: 1000)
    Maximum number of iterations for the optimization. Should be at
    least 250.

n_iter_without_progress : int, optional (default: 300)
    Maximum number of iterations without progress before we abort the
    optimization, used after 250 initial iterations with early
    exaggeration. Note that progress is only checked every 50 iterations so
    this value is rounded to the next multiple of 50.

    .. versionadded:: 0.17
       parameter *n_iter_without_progress* to control stopping criteria.

min_grad_norm : float, optional (default: 1e-7)
    If the gradient norm is below this threshold, the optimization will
    be stopped.

metric : string or callable, optional
    The metric to use when calculating distance between instances in a
    feature array. If metric is a string, it must be one of the options
    allowed by scipy.spatial.distance.pdist for its metric parameter, or
    a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
    If metric is "precomputed", X is assumed to be a distance matrix.
    Alternatively, if metric is a callable function, it is called on each
    pair of instances (rows) and the resulting value recorded. The callable
    should take two arrays from X as input and return a value indicating
    the distance between them. The default is "euclidean" which is
    interpreted as squared euclidean distance.

init : string or numpy array, optional (default: "random")
    Initialization of embedding. Possible options are 'random', 'pca',
    and a numpy array of shape (n_samples, n_components).
    PCA initialization cannot be used with precomputed distances and is
    usually more globally stable than random initialization.

verbose : int, optional (default: 0)
    Verbosity level.

random_state : int, RandomState instance or None, optional (default: None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.  Note that different initializations might result in
    different local minima of the cost function.

method : string (default: 'barnes_hut')
    By default the gradient calculation algorithm uses Barnes-Hut
    approximation running in O(NlogN) time. method='exact'
    will run on the slower, but exact, algorithm in O(N^2) time. The
    exact algorithm should be used when nearest-neighbor errors need
    to be better than 3%. However, the exact method cannot scale to
    millions of examples.

    .. versionadded:: 0.17
       Approximate optimization *method* via the Barnes-Hut.

angle : float (default: 0.5)
    Only used if method='barnes_hut'
    This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
    'angle' is the angular size (referred to as theta in [3]) of a distant
    node as measured from a point. If this size is below 'angle' then it is
    used as a summary node of all points contained within it.
    This method is not very sensitive to changes in this parameter
    in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
    computation time and angle greater 0.8 has quickly increasing error.

n_jobs : int or None, optional (default=None)
    The number of parallel jobs to run for neighbors search. This parameter
    has no impact when ``metric="precomputed"`` or
    (``metric="euclidean"`` and ``method="exact"``).
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

    .. versionadded:: 0.22

Attributes
----------
embedding_ : array-like, shape (n_samples, n_components)
    Stores the embedding vectors.

kl_divergence_ : float
    Kullback-Leibler divergence after optimization.

n_iter_ : int
    Number of iterations run.

Examples
--------

>>> import numpy as np
>>> from sklearn.manifold import TSNE
>>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
>>> X_embedded = TSNE(n_components=2).fit_transform(X)
>>> X_embedded.shape
(4, 2)

References
----------

[1] van der Maaten, L.J.P.; Hinton, G.E. Visualizing High-Dimensional Data
    Using t-SNE. Journal of Machine Learning Research 9:2579-2605, 2008.

[2] van der Maaten, L.J.P. t-Distributed Stochastic Neighbor Embedding
    https://lvdmaaten.github.io/tsne/

[3] L.J.P. van der Maaten. Accelerating t-SNE using Tree-Based Algorithms.
    Journal of Machine Learning Research 15(Oct):3221-3245, 2014.
    https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf
*)

val fit : ?y:Py.Object.t -> x:Arr.t -> t -> t
(**
Fit X into an embedded space.

Parameters
----------
X : array, shape (n_samples, n_features) or (n_samples, n_samples)
    If the metric is 'precomputed' X must be a square distance
    matrix. Otherwise it contains a sample per row. If the method
    is 'exact', X may be a sparse matrix of type 'csr', 'csc'
    or 'coo'. If the method is 'barnes_hut' and the metric is
    'precomputed', X may be a precomputed sparse graph.

y : Ignored
*)

val fit_transform : ?y:Py.Object.t -> x:Arr.t -> t -> Arr.t
(**
Fit X into an embedded space and return that transformed
output.

Parameters
----------
X : array, shape (n_samples, n_features) or (n_samples, n_samples)
    If the metric is 'precomputed' X must be a square distance
    matrix. Otherwise it contains a sample per row. If the method
    is 'exact', X may be a sparse matrix of type 'csr', 'csc'
    or 'coo'. If the method is 'barnes_hut' and the metric is
    'precomputed', X may be a precomputed sparse graph.

y : Ignored

Returns
-------
X_new : array, shape (n_samples, n_components)
    Embedding of the training data in low-dimensional space.
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


(** Attribute embedding_: get value or raise Not_found if None.*)
val embedding_ : t -> Arr.t

(** Attribute embedding_: get value as an option. *)
val embedding_opt : t -> (Arr.t) option


(** Attribute kl_divergence_: get value or raise Not_found if None.*)
val kl_divergence_ : t -> float

(** Attribute kl_divergence_: get value as an option. *)
val kl_divergence_opt : t -> (float) option


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

val locally_linear_embedding : ?reg:float -> ?eigen_solver:[`Auto | `Arpack | `Dense] -> ?tol:float -> ?max_iter:int -> ?method_:[`Standard | `Hessian | `Modified | `Ltsa] -> ?hessian_tol:float -> ?modified_tol:float -> ?random_state:int -> ?n_jobs:int -> x:[`Arr of Arr.t | `NearestNeighbors of Py.Object.t] -> n_neighbors:int -> n_components:int -> unit -> (Arr.t * float)
(**
Perform a Locally Linear Embedding analysis on the data.

Read more in the :ref:`User Guide <locally_linear_embedding>`.

Parameters
----------
X : {array-like, NearestNeighbors}
    Sample data, shape = (n_samples, n_features), in the form of a
    numpy array or a NearestNeighbors object.

n_neighbors : integer
    number of neighbors to consider for each point.

n_components : integer
    number of coordinates for the manifold.

reg : float
    regularization constant, multiplies the trace of the local covariance
    matrix of the distances.

eigen_solver : string, {'auto', 'arpack', 'dense'}
    auto : algorithm will attempt to choose the best method for input data

    arpack : use arnoldi iteration in shift-invert mode.
                For this method, M may be a dense matrix, sparse matrix,
                or general linear operator.
                Warning: ARPACK can be unstable for some problems.  It is
                best to try several random seeds in order to check results.

    dense  : use standard dense matrix operations for the eigenvalue
                decomposition.  For this method, M must be an array
                or matrix type.  This method should be avoided for
                large problems.

tol : float, optional
    Tolerance for 'arpack' method
    Not used if eigen_solver=='dense'.

max_iter : integer
    maximum number of iterations for the arpack solver.

method : {'standard', 'hessian', 'modified', 'ltsa'}
    standard : use the standard locally linear embedding algorithm.
               see reference [1]_
    hessian  : use the Hessian eigenmap method.  This method requires
               n_neighbors > n_components * (1 + (n_components + 1) / 2.
               see reference [2]_
    modified : use the modified locally linear embedding algorithm.
               see reference [3]_
    ltsa     : use local tangent space alignment algorithm
               see reference [4]_

hessian_tol : float, optional
    Tolerance for Hessian eigenmapping method.
    Only used if method == 'hessian'

modified_tol : float, optional
    Tolerance for modified LLE method.
    Only used if method == 'modified'

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`. Used when ``solver`` == 'arpack'.

n_jobs : int or None, optional (default=None)
    The number of parallel jobs to run for neighbors search.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

Returns
-------
Y : array-like, shape [n_samples, n_components]
    Embedding vectors.

squared_error : float
    Reconstruction error for the embedding vectors. Equivalent to
    ``norm(Y - W Y, 'fro')**2``, where W are the reconstruction weights.

References
----------

.. [1] Roweis, S. & Saul, L. Nonlinear dimensionality reduction
    by locally linear embedding.  Science 290:2323 (2000).
.. [2] Donoho, D. & Grimes, C. Hessian eigenmaps: Locally
    linear embedding techniques for high-dimensional data.
    Proc Natl Acad Sci U S A.  100:5591 (2003).
.. [3] Zhang, Z. & Wang, J. MLLE: Modified Locally Linear
    Embedding Using Multiple Weights.
    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.70.382
.. [4] Zhang, Z. & Zha, H. Principal manifolds and nonlinear
    dimensionality reduction via tangent space alignment.
    Journal of Shanghai Univ.  8:406 (2004)
*)

val smacof : ?metric:bool -> ?n_components:int -> ?init:Arr.t -> ?n_init:int -> ?n_jobs:int -> ?max_iter:int -> ?verbose:int -> ?eps:float -> ?random_state:int -> ?return_n_iter:bool -> dissimilarities:Arr.t -> unit -> (Arr.t * float * int)
(**
Computes multidimensional scaling using the SMACOF algorithm.

The SMACOF (Scaling by MAjorizing a COmplicated Function) algorithm is a
multidimensional scaling algorithm which minimizes an objective function
(the *stress* ) using a majorization technique. Stress majorization, also
known as the Guttman Transform, guarantees a monotone convergence of
stress, and is more powerful than traditional techniques such as gradient
descent.

The SMACOF algorithm for metric MDS can summarized by the following steps:

1. Set an initial start configuration, randomly or not.
2. Compute the stress
3. Compute the Guttman Transform
4. Iterate 2 and 3 until convergence.

The nonmetric algorithm adds a monotonic regression step before computing
the stress.

Parameters
----------
dissimilarities : ndarray, shape (n_samples, n_samples)
    Pairwise dissimilarities between the points. Must be symmetric.

metric : boolean, optional, default: True
    Compute metric or nonmetric SMACOF algorithm.

n_components : int, optional, default: 2
    Number of dimensions in which to immerse the dissimilarities. If an
    ``init`` array is provided, this option is overridden and the shape of
    ``init`` is used to determine the dimensionality of the embedding
    space.

init : ndarray, shape (n_samples, n_components), optional, default: None
    Starting configuration of the embedding to initialize the algorithm. By
    default, the algorithm is initialized with a randomly chosen array.

n_init : int, optional, default: 8
    Number of times the SMACOF algorithm will be run with different
    initializations. The final results will be the best output of the runs,
    determined by the run with the smallest final stress. If ``init`` is
    provided, this option is overridden and a single run is performed.

n_jobs : int or None, optional (default=None)
    The number of jobs to use for the computation. If multiple
    initializations are used (``n_init``), each run of the algorithm is
    computed in parallel.

    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

max_iter : int, optional, default: 300
    Maximum number of iterations of the SMACOF algorithm for a single run.

verbose : int, optional, default: 0
    Level of verbosity.

eps : float, optional, default: 1e-3
    Relative tolerance with respect to stress at which to declare
    convergence.

random_state : int, RandomState instance or None, optional, default: None
    The generator used to initialize the centers.  If int, random_state is
    the seed used by the random number generator; If RandomState instance,
    random_state is the random number generator; If None, the random number
    generator is the RandomState instance used by `np.random`.

return_n_iter : bool, optional, default: False
    Whether or not to return the number of iterations.

Returns
-------
X : ndarray, shape (n_samples, n_components)
    Coordinates of the points in a ``n_components``-space.

stress : float
    The final value of the stress (sum of squared distance of the
    disparities and the distances for all constrained points).

n_iter : int
    The number of iterations corresponding to the best stress. Returned
    only if ``return_n_iter`` is set to ``True``.

Notes
-----
"Modern Multidimensional Scaling - Theory and Applications" Borg, I.;
Groenen P. Springer Series in Statistics (1997)

"Nonmetric multidimensional scaling: a numerical method" Kruskal, J.
Psychometrika, 29 (1964)

"Multidimensional scaling by optimizing goodness of fit to a nonmetric
hypothesis" Kruskal, J. Psychometrika, 29, (1964)
*)

val spectral_embedding : ?n_components:int -> ?eigen_solver:[`Arpack | `Lobpcg | `Amg] -> ?random_state:int -> ?eigen_tol:float -> ?norm_laplacian:bool -> ?drop_first:bool -> adjacency:[`Arr of Arr.t | `Sparse_graph of Py.Object.t] -> unit -> Arr.t
(**
Project the sample on the first eigenvectors of the graph Laplacian.

The adjacency matrix is used to compute a normalized graph Laplacian
whose spectrum (especially the eigenvectors associated to the
smallest eigenvalues) has an interpretation in terms of minimal
number of cuts necessary to split the graph into comparably sized
components.

This embedding can also 'work' even if the ``adjacency`` variable is
not strictly the adjacency matrix of a graph but more generally
an affinity or similarity matrix between samples (for instance the
heat kernel of a euclidean distance matrix or a k-NN matrix).

However care must taken to always make the affinity matrix symmetric
so that the eigenvector decomposition works as expected.

Note : Laplacian Eigenmaps is the actual algorithm implemented here.

Read more in the :ref:`User Guide <spectral_embedding>`.

Parameters
----------
adjacency : array-like or sparse graph, shape: (n_samples, n_samples)
    The adjacency matrix of the graph to embed.

n_components : integer, optional, default 8
    The dimension of the projection subspace.

eigen_solver : {None, 'arpack', 'lobpcg', or 'amg'}, default None
    The eigenvalue decomposition strategy to use. AMG requires pyamg
    to be installed. It can be faster on very large, sparse problems,
    but may also lead to instabilities.

random_state : int, RandomState instance or None, optional, default: None
    A pseudo random number generator used for the initialization of the
    lobpcg eigenvectors decomposition.  If int, random_state is the seed
    used by the random number generator; If RandomState instance,
    random_state is the random number generator; If None, the random number
    generator is the RandomState instance used by `np.random`. Used when
    ``solver`` == 'amg'.

eigen_tol : float, optional, default=0.0
    Stopping criterion for eigendecomposition of the Laplacian matrix
    when using arpack eigen_solver.

norm_laplacian : bool, optional, default=True
    If True, then compute normalized Laplacian.

drop_first : bool, optional, default=True
    Whether to drop the first eigenvector. For spectral embedding, this
    should be True as the first eigenvector should be constant vector for
    connected graph, but for spectral clustering, this should be kept as
    False to retain the first eigenvector.

Returns
-------
embedding : array, shape=(n_samples, n_components)
    The reduced samples.

Notes
-----
Spectral Embedding (Laplacian Eigenmaps) is most useful when the graph
has one connected component. If there graph has many components, the first
few eigenvectors will simply uncover the connected components of the graph.

References
----------
* https://en.wikipedia.org/wiki/LOBPCG

* Toward the Optimal Preconditioned Eigensolver: Locally Optimal
  Block Preconditioned Conjugate Gradient Method
  Andrew V. Knyazev
  https://doi.org/10.1137%2FS1064827500366124
*)

val trustworthiness : ?n_neighbors:int -> ?metric:[`S of string | `Callable of Py.Object.t] -> x:Arr.t -> x_embedded:Arr.t -> unit -> float
(**
Expresses to what extent the local structure is retained.

The trustworthiness is within [0, 1]. It is defined as

.. math::

    T(k) = 1 - \frac{2}{nk (2n - 3k - 1)} \sum^n_{i=1}
        \sum_{j \in \mathcal{N}_{i}^{k}} \max(0, (r(i, j) - k))

where for each sample i, :math:`\mathcal{N}_{i}^{k}` are its k nearest
neighbors in the output space, and every sample j is its :math:`r(i, j)`-th
nearest neighbor in the input space. In other words, any unexpected nearest
neighbors in the output space are penalised in proportion to their rank in
the input space.

* "Neighborhood Preservation in Nonlinear Projection Methods: An
  Experimental Study"
  J. Venna, S. Kaski
* "Learning a Parametric Embedding by Preserving Local Structure"
  L.J.P. van der Maaten

Parameters
----------
X : array, shape (n_samples, n_features) or (n_samples, n_samples)
    If the metric is 'precomputed' X must be a square distance
    matrix. Otherwise it contains a sample per row.

X_embedded : array, shape (n_samples, n_components)
    Embedding of the training data in low-dimensional space.

n_neighbors : int, optional (default: 5)
    Number of neighbors k that will be considered.

metric : string, or callable, optional, default 'euclidean'
    Which metric to use for computing pairwise distances between samples
    from the original input space. If metric is 'precomputed', X must be a
    matrix of pairwise distances or squared distances. Otherwise, see the
    documentation of argument metric in sklearn.pairwise.pairwise_distances
    for a list of available metrics.

Returns
-------
trustworthiness : float
    Trustworthiness of the low-dimensional embedding.
*)

