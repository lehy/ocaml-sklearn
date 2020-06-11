(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module Hierarchy : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module ClusterNode : sig
type tag = [`ClusterNode]
type t = [`ClusterNode | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?left:Py.Object.t -> ?right:Py.Object.t -> ?dist:float -> ?count:int -> id:int -> unit -> t
(**
A tree node class for representing a cluster.

Leaf nodes correspond to original observations, while non-leaf nodes
correspond to non-singleton clusters.

The `to_tree` function converts a matrix returned by the linkage
function into an easy-to-use tree representation.

All parameter names are also attributes.

Parameters
----------
id : int
    The node id.
left : ClusterNode instance, optional
    The left child tree node.
right : ClusterNode instance, optional
    The right child tree node.
dist : float, optional
    Distance for this cluster in the linkage matrix.
count : int, optional
    The number of samples in this cluster.

See Also
--------
to_tree : for converting a linkage matrix ``Z`` into a tree object.
*)

val get_count : [> tag] Obj.t -> int
(**
The number of leaf nodes (original observations) belonging to
the cluster node nd. If the target node is a leaf, 1 is
returned.

Returns
-------
get_count : int
    The number of leaf nodes below the target node.
*)

val get_id : [> tag] Obj.t -> int
(**
The identifier of the target node.

For ``0 <= i < n``, `i` corresponds to original observation i.
For ``n <= i < 2n-1``, `i` corresponds to non-singleton cluster formed
at iteration ``i-n``.

Returns
-------
id : int
    The identifier of the target node.
*)

val get_left : [> tag] Obj.t -> Py.Object.t
(**
Return a reference to the left child tree object.

Returns
-------
left : ClusterNode
    The left child of the target node.  If the node is a leaf,
    None is returned.
*)

val get_right : [> tag] Obj.t -> Py.Object.t
(**
Return a reference to the right child tree object.

Returns
-------
right : ClusterNode
    The left child of the target node.  If the node is a leaf,
    None is returned.
*)

val is_leaf : [> tag] Obj.t -> bool
(**
Return True if the target node is a leaf.

Returns
-------
leafness : bool
    True if the target node is a leaf node.
*)

val pre_order : ?func:Py.Object.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Perform pre-order traversal without recursive function calls.

When a leaf node is first encountered, ``func`` is called with
the leaf node as its argument, and its result is appended to
the list.

For example, the statement::

   ids = root.pre_order(lambda x: x.id)

returns a list of the node ids corresponding to the leaf nodes
of the tree as they appear from left to right.

Parameters
----------
func : function
    Applied to each leaf ClusterNode object in the pre-order traversal.
    Given the ``i``-th leaf node in the pre-order traversal ``n[i]``,
    the result of ``func(n[i])`` is stored in ``L[i]``. If not
    provided, the index of the original observation to which the node
    corresponds is used.

Returns
-------
L : list
    The pre-order traversal.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module ClusterWarning : sig
type tag = [`ClusterWarning]
type t = [`BaseException | `ClusterWarning | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_exception : t -> [`BaseException] Obj.t
val with_traceback : tb:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Exception.with_traceback(tb) --
set self.__traceback__ to tb and return self.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Deque : sig
type tag = [`Deque]
type t = [`Deque | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val __getitem__ : key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return self[key].
*)

val __iter__ : [> tag] Obj.t -> Py.Object.t
(**
Implement iter(self).
*)

val __setitem__ : key:Py.Object.t -> value:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Set self[key] to value.
*)

val count : value:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
D.count(value) -> integer -- return number of occurrences of value
*)

val insert : index:Py.Object.t -> object_:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
D.insert(index, object) -- insert object before index
*)

val remove : value:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
D.remove(value) -- remove first occurrence of value.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Xrange : sig
type tag = [`Range]
type t = [`Object | `Range] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : Py.Object.t -> t
(**
range(stop) -> range object
range(start, stop[, step]) -> range object

Return an object that produces a sequence of integers from start (inclusive)
to stop (exclusive) by step.  range(i, j) produces i, i+1, i+2, ..., j-1.
start defaults to 0, and stop is omitted!  range(4) produces 0, 1, 2, 3.
These are exactly the valid indices for a list of 4 elements.
When step is given, it specifies the increment (or decrement).
*)

val __getitem__ : key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return self[key].
*)

val __iter__ : [> tag] Obj.t -> Py.Object.t
(**
Implement iter(self).
*)

val count : value:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
rangeobject.count(value) -> integer -- return number of occurrences of value
*)

val index : value:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
rangeobject.index(value) -> integer -- return index of value.
Raise ValueError if the value is not present.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val average : [>`Ndarray] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Perform average/UPGMA linkage on a condensed distance matrix.

Parameters
----------
y : ndarray
    The upper triangular of the distance matrix. The result of
    ``pdist`` is returned in this form.

Returns
-------
Z : ndarray
    A linkage matrix containing the hierarchical clustering. See
    `linkage` for more information on its structure.

See Also
--------
linkage: for advanced creation of hierarchical clusterings.
scipy.spatial.distance.pdist : pairwise distance metrics

Examples
--------
>>> from scipy.cluster.hierarchy import average, fcluster
>>> from scipy.spatial.distance import pdist

First we need a toy dataset to play with::

    x x    x x
    x        x

    x        x
    x x    x x

>>> X = [[0, 0], [0, 1], [1, 0],
...      [0, 4], [0, 3], [1, 4],
...      [4, 0], [3, 0], [4, 1],
...      [4, 4], [3, 4], [4, 3]]

Then we get a condensed distance matrix from this dataset:

>>> y = pdist(X)

Finally, we can perform the clustering:

>>> Z = average(y)
>>> Z
array([[ 0.        ,  1.        ,  1.        ,  2.        ],
       [ 3.        ,  4.        ,  1.        ,  2.        ],
       [ 6.        ,  7.        ,  1.        ,  2.        ],
       [ 9.        , 10.        ,  1.        ,  2.        ],
       [ 2.        , 12.        ,  1.20710678,  3.        ],
       [ 5.        , 13.        ,  1.20710678,  3.        ],
       [ 8.        , 14.        ,  1.20710678,  3.        ],
       [11.        , 15.        ,  1.20710678,  3.        ],
       [16.        , 17.        ,  3.39675184,  6.        ],
       [18.        , 19.        ,  3.39675184,  6.        ],
       [20.        , 21.        ,  4.09206523, 12.        ]])

The linkage matrix ``Z`` represents a dendrogram - see
`scipy.cluster.hierarchy.linkage` for a detailed explanation of its
contents.

We can use `scipy.cluster.hierarchy.fcluster` to see to which cluster
each initial point would belong given a distance threshold:

>>> fcluster(Z, 0.9, criterion='distance')
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12], dtype=int32)
>>> fcluster(Z, 1.5, criterion='distance')
array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], dtype=int32)
>>> fcluster(Z, 4, criterion='distance')
array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2], dtype=int32)
>>> fcluster(Z, 6, criterion='distance')
array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)

Also `scipy.cluster.hierarchy.dendrogram` can be used to generate a
plot of the dendrogram.
*)

val centroid : [>`Ndarray] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Perform centroid/UPGMC linkage.

See `linkage` for more information on the input matrix,
return structure, and algorithm.

The following are common calling conventions:

1. ``Z = centroid(y)``

   Performs centroid/UPGMC linkage on the condensed distance
   matrix ``y``.

2. ``Z = centroid(X)``

   Performs centroid/UPGMC linkage on the observation matrix ``X``
   using Euclidean distance as the distance metric.

Parameters
----------
y : ndarray
    A condensed distance matrix. A condensed
    distance matrix is a flat array containing the upper
    triangular of the distance matrix. This is the form that
    ``pdist`` returns. Alternatively, a collection of
    m observation vectors in n dimensions may be passed as
    a m by n array.

Returns
-------
Z : ndarray
    A linkage matrix containing the hierarchical clustering. See
    the `linkage` function documentation for more information
    on its structure.

See Also
--------
linkage: for advanced creation of hierarchical clusterings.
scipy.spatial.distance.pdist : pairwise distance metrics

Examples
--------
>>> from scipy.cluster.hierarchy import centroid, fcluster
>>> from scipy.spatial.distance import pdist

First we need a toy dataset to play with::

    x x    x x
    x        x

    x        x
    x x    x x

>>> X = [[0, 0], [0, 1], [1, 0],
...      [0, 4], [0, 3], [1, 4],
...      [4, 0], [3, 0], [4, 1],
...      [4, 4], [3, 4], [4, 3]]

Then we get a condensed distance matrix from this dataset:

>>> y = pdist(X)

Finally, we can perform the clustering:

>>> Z = centroid(y)
>>> Z
array([[ 0.        ,  1.        ,  1.        ,  2.        ],
       [ 3.        ,  4.        ,  1.        ,  2.        ],
       [ 9.        , 10.        ,  1.        ,  2.        ],
       [ 6.        ,  7.        ,  1.        ,  2.        ],
       [ 2.        , 12.        ,  1.11803399,  3.        ],
       [ 5.        , 13.        ,  1.11803399,  3.        ],
       [ 8.        , 15.        ,  1.11803399,  3.        ],
       [11.        , 14.        ,  1.11803399,  3.        ],
       [18.        , 19.        ,  3.33333333,  6.        ],
       [16.        , 17.        ,  3.33333333,  6.        ],
       [20.        , 21.        ,  3.33333333, 12.        ]])

The linkage matrix ``Z`` represents a dendrogram - see
`scipy.cluster.hierarchy.linkage` for a detailed explanation of its
contents.

We can use `scipy.cluster.hierarchy.fcluster` to see to which cluster
each initial point would belong given a distance threshold:

>>> fcluster(Z, 0.9, criterion='distance')
array([ 7,  8,  9, 10, 11, 12,  1,  2,  3,  4,  5,  6], dtype=int32)
>>> fcluster(Z, 1.1, criterion='distance')
array([5, 5, 6, 7, 7, 8, 1, 1, 2, 3, 3, 4], dtype=int32)
>>> fcluster(Z, 2, criterion='distance')
array([3, 3, 3, 4, 4, 4, 1, 1, 1, 2, 2, 2], dtype=int32)
>>> fcluster(Z, 4, criterion='distance')
array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)

Also `scipy.cluster.hierarchy.dendrogram` can be used to generate a
plot of the dendrogram.
*)

val complete : [>`Ndarray] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Perform complete/max/farthest point linkage on a condensed distance matrix.

Parameters
----------
y : ndarray
    The upper triangular of the distance matrix. The result of
    ``pdist`` is returned in this form.

Returns
-------
Z : ndarray
    A linkage matrix containing the hierarchical clustering. See
    the `linkage` function documentation for more information
    on its structure.

See Also
--------
linkage: for advanced creation of hierarchical clusterings.
scipy.spatial.distance.pdist : pairwise distance metrics

Examples
--------
>>> from scipy.cluster.hierarchy import complete, fcluster
>>> from scipy.spatial.distance import pdist

First we need a toy dataset to play with::

    x x    x x
    x        x

    x        x
    x x    x x

>>> X = [[0, 0], [0, 1], [1, 0],
...      [0, 4], [0, 3], [1, 4],
...      [4, 0], [3, 0], [4, 1],
...      [4, 4], [3, 4], [4, 3]]

Then we get a condensed distance matrix from this dataset:

>>> y = pdist(X)

Finally, we can perform the clustering:

>>> Z = complete(y)
>>> Z
array([[ 0.        ,  1.        ,  1.        ,  2.        ],
       [ 3.        ,  4.        ,  1.        ,  2.        ],
       [ 6.        ,  7.        ,  1.        ,  2.        ],
       [ 9.        , 10.        ,  1.        ,  2.        ],
       [ 2.        , 12.        ,  1.41421356,  3.        ],
       [ 5.        , 13.        ,  1.41421356,  3.        ],
       [ 8.        , 14.        ,  1.41421356,  3.        ],
       [11.        , 15.        ,  1.41421356,  3.        ],
       [16.        , 17.        ,  4.12310563,  6.        ],
       [18.        , 19.        ,  4.12310563,  6.        ],
       [20.        , 21.        ,  5.65685425, 12.        ]])

The linkage matrix ``Z`` represents a dendrogram - see
`scipy.cluster.hierarchy.linkage` for a detailed explanation of its
contents.

We can use `scipy.cluster.hierarchy.fcluster` to see to which cluster
each initial point would belong given a distance threshold:

>>> fcluster(Z, 0.9, criterion='distance')
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12], dtype=int32)
>>> fcluster(Z, 1.5, criterion='distance')
array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], dtype=int32)
>>> fcluster(Z, 4.5, criterion='distance')
array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2], dtype=int32)
>>> fcluster(Z, 6, criterion='distance')
array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)

Also `scipy.cluster.hierarchy.dendrogram` can be used to generate a
plot of the dendrogram.
*)

val cophenet : ?y:[>`Ndarray] Np.Obj.t -> z:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Calculate the cophenetic distances between each observation in
the hierarchical clustering defined by the linkage ``Z``.

Suppose ``p`` and ``q`` are original observations in
disjoint clusters ``s`` and ``t``, respectively and
``s`` and ``t`` are joined by a direct parent cluster
``u``. The cophenetic distance between observations
``i`` and ``j`` is simply the distance between
clusters ``s`` and ``t``.

Parameters
----------
Z : ndarray
    The hierarchical clustering encoded as an array
    (see `linkage` function).
Y : ndarray (optional)
    Calculates the cophenetic correlation coefficient ``c`` of a
    hierarchical clustering defined by the linkage matrix `Z`
    of a set of :math:`n` observations in :math:`m`
    dimensions. `Y` is the condensed distance matrix from which
    `Z` was generated.

Returns
-------
c : ndarray
    The cophentic correlation distance (if ``Y`` is passed).
d : ndarray
    The cophenetic distance matrix in condensed form. The
    :math:`ij` th entry is the cophenetic distance between
    original observations :math:`i` and :math:`j`.

See Also
--------
linkage: for a description of what a linkage matrix is.
scipy.spatial.distance.squareform: transforming condensed matrices into square ones.

Examples
--------
>>> from scipy.cluster.hierarchy import single, cophenet
>>> from scipy.spatial.distance import pdist, squareform

Given a dataset ``X`` and a linkage matrix ``Z``, the cophenetic distance
between two points of ``X`` is the distance between the largest two
distinct clusters that each of the points:

>>> X = [[0, 0], [0, 1], [1, 0],
...      [0, 4], [0, 3], [1, 4],
...      [4, 0], [3, 0], [4, 1],
...      [4, 4], [3, 4], [4, 3]]

``X`` corresponds to this dataset ::

    x x    x x
    x        x

    x        x
    x x    x x

>>> Z = single(pdist(X))
>>> Z
array([[ 0.,  1.,  1.,  2.],
       [ 2., 12.,  1.,  3.],
       [ 3.,  4.,  1.,  2.],
       [ 5., 14.,  1.,  3.],
       [ 6.,  7.,  1.,  2.],
       [ 8., 16.,  1.,  3.],
       [ 9., 10.,  1.,  2.],
       [11., 18.,  1.,  3.],
       [13., 15.,  2.,  6.],
       [17., 20.,  2.,  9.],
       [19., 21.,  2., 12.]])
>>> cophenet(Z)
array([1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 1., 2., 2., 2., 2., 2.,
       2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 1., 1., 2., 2.,
       2., 2., 2., 2., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
       1., 1., 2., 2., 2., 1., 2., 2., 2., 2., 2., 2., 1., 1., 1.])

The output of the `scipy.cluster.hierarchy.cophenet` method is
represented in condensed form. We can use
`scipy.spatial.distance.squareform` to see the output as a
regular matrix (where each element ``ij`` denotes the cophenetic distance
between each ``i``, ``j`` pair of points in ``X``):

>>> squareform(cophenet(Z))
array([[0., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
       [1., 0., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
       [1., 1., 0., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
       [2., 2., 2., 0., 1., 1., 2., 2., 2., 2., 2., 2.],
       [2., 2., 2., 1., 0., 1., 2., 2., 2., 2., 2., 2.],
       [2., 2., 2., 1., 1., 0., 2., 2., 2., 2., 2., 2.],
       [2., 2., 2., 2., 2., 2., 0., 1., 1., 2., 2., 2.],
       [2., 2., 2., 2., 2., 2., 1., 0., 1., 2., 2., 2.],
       [2., 2., 2., 2., 2., 2., 1., 1., 0., 2., 2., 2.],
       [2., 2., 2., 2., 2., 2., 2., 2., 2., 0., 1., 1.],
       [2., 2., 2., 2., 2., 2., 2., 2., 2., 1., 0., 1.],
       [2., 2., 2., 2., 2., 2., 2., 2., 2., 1., 1., 0.]])

In this example, the cophenetic distance between points on ``X`` that are
very close (i.e. in the same corner) is 1. For other pairs of points is 2,
because the points will be located in clusters at different
corners - thus the distance between these clusters will be larger.
*)

val correspond : z:[>`Ndarray] Np.Obj.t -> y:[>`Ndarray] Np.Obj.t -> unit -> bool
(**
Check for correspondence between linkage and condensed distance matrices.

They must have the same number of original observations for
the check to succeed.

This function is useful as a sanity check in algorithms that make
extensive use of linkage and distance matrices that must
correspond to the same set of original observations.

Parameters
----------
Z : array_like
    The linkage matrix to check for correspondence.
Y : array_like
    The condensed distance matrix to check for correspondence.

Returns
-------
b : bool
    A boolean indicating whether the linkage matrix and distance
    matrix could possibly correspond to one another.

See Also
--------
linkage: for a description of what a linkage matrix is.

Examples
--------
>>> from scipy.cluster.hierarchy import ward, correspond
>>> from scipy.spatial.distance import pdist

This method can be used to check if a given linkage matrix ``Z`` has been
obtained from the application of a cluster method over a dataset ``X``:

>>> X = [[0, 0], [0, 1], [1, 0],
...      [0, 4], [0, 3], [1, 4],
...      [4, 0], [3, 0], [4, 1],
...      [4, 4], [3, 4], [4, 3]]
>>> X_condensed = pdist(X)
>>> Z = ward(X_condensed)

Here we can compare ``Z`` and ``X`` (in condensed form):

>>> correspond(Z, X_condensed)
True
*)

val cut_tree : ?n_clusters:[>`Ndarray] Np.Obj.t -> ?height:[>`Ndarray] Np.Obj.t -> z:Py.Object.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Given a linkage matrix Z, return the cut tree.

Parameters
----------
Z : scipy.cluster.linkage array
    The linkage matrix.
n_clusters : array_like, optional
    Number of clusters in the tree at the cut point.
height : array_like, optional
    The height at which to cut the tree.  Only possible for ultrametric
    trees.

Returns
-------
cutree : array
    An array indicating group membership at each agglomeration step.  I.e.,
    for a full cut tree, in the first column each data point is in its own
    cluster.  At the next step, two nodes are merged.  Finally all
    singleton and non-singleton clusters are in one group.  If `n_clusters`
    or `height` is given, the columns correspond to the columns of
    `n_clusters` or `height`.

Examples
--------
>>> from scipy import cluster
>>> np.random.seed(23)
>>> X = np.random.randn(50, 4)
>>> Z = cluster.hierarchy.ward(X)
>>> cutree = cluster.hierarchy.cut_tree(Z, n_clusters=[5, 10])
>>> cutree[:10]
array([[0, 0],
       [1, 1],
       [2, 2],
       [3, 3],
       [3, 4],
       [2, 2],
       [0, 0],
       [1, 5],
       [3, 6],
       [4, 7]])
*)

val dendrogram : ?p:int -> ?truncate_mode:string -> ?color_threshold:float -> ?get_leaves:bool -> ?orientation:string -> ?labels:[>`Ndarray] Np.Obj.t -> ?count_sort:[`Bool of bool | `S of string] -> ?distance_sort:[`Bool of bool | `S of string] -> ?show_leaf_counts:bool -> ?no_plot:bool -> ?no_labels:bool -> ?leaf_font_size:int -> ?leaf_rotation:float -> ?leaf_label_func:[`Lambda of Py.Object.t | `Callable of Py.Object.t] -> ?show_contracted:bool -> ?link_color_func:Py.Object.t -> ?ax:Py.Object.t -> ?above_threshold_color:string -> z:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Plot the hierarchical clustering as a dendrogram.

The dendrogram illustrates how each cluster is
composed by drawing a U-shaped link between a non-singleton
cluster and its children.  The top of the U-link indicates a
cluster merge.  The two legs of the U-link indicate which clusters
were merged.  The length of the two legs of the U-link represents
the distance between the child clusters.  It is also the
cophenetic distance between original observations in the two
children clusters.

Parameters
----------
Z : ndarray
    The linkage matrix encoding the hierarchical clustering to
    render as a dendrogram. See the ``linkage`` function for more
    information on the format of ``Z``.
p : int, optional
    The ``p`` parameter for ``truncate_mode``.
truncate_mode : str, optional
    The dendrogram can be hard to read when the original
    observation matrix from which the linkage is derived is
    large. Truncation is used to condense the dendrogram. There
    are several modes:

    ``None``
      No truncation is performed (default).
      Note: ``'none'`` is an alias for ``None`` that's kept for
      backward compatibility.

    ``'lastp'``
      The last ``p`` non-singleton clusters formed in the linkage are the
      only non-leaf nodes in the linkage; they correspond to rows
      ``Z[n-p-2:end]`` in ``Z``. All other non-singleton clusters are
      contracted into leaf nodes.

    ``'level'``
      No more than ``p`` levels of the dendrogram tree are displayed.
      A 'level' includes all nodes with ``p`` merges from the last merge.

      Note: ``'mtica'`` is an alias for ``'level'`` that's kept for
      backward compatibility.

color_threshold : double, optional
    For brevity, let :math:`t` be the ``color_threshold``.
    Colors all the descendent links below a cluster node
    :math:`k` the same color if :math:`k` is the first node below
    the cut threshold :math:`t`. All links connecting nodes with
    distances greater than or equal to the threshold are colored
    blue. If :math:`t` is less than or equal to zero, all nodes
    are colored blue. If ``color_threshold`` is None or
    'default', corresponding with MATLAB(TM) behavior, the
    threshold is set to ``0.7*max(Z[:,2])``.
get_leaves : bool, optional
    Includes a list ``R['leaves']=H`` in the result
    dictionary. For each :math:`i`, ``H[i] == j``, cluster node
    ``j`` appears in position ``i`` in the left-to-right traversal
    of the leaves, where :math:`j < 2n-1` and :math:`i < n`.
orientation : str, optional
    The direction to plot the dendrogram, which can be any
    of the following strings:

    ``'top'``
      Plots the root at the top, and plot descendent links going downwards.
      (default).

    ``'bottom'``
      Plots the root at the bottom, and plot descendent links going
      upwards.

    ``'left'``
      Plots the root at the left, and plot descendent links going right.

    ``'right'``
      Plots the root at the right, and plot descendent links going left.

labels : ndarray, optional
    By default ``labels`` is None so the index of the original observation
    is used to label the leaf nodes.  Otherwise, this is an :math:`n`
    -sized list (or tuple). The ``labels[i]`` value is the text to put
    under the :math:`i` th leaf node only if it corresponds to an original
    observation and not a non-singleton cluster.
count_sort : str or bool, optional
    For each node n, the order (visually, from left-to-right) n's
    two descendent links are plotted is determined by this
    parameter, which can be any of the following values:

    ``False``
      Nothing is done.

    ``'ascending'`` or ``True``
      The child with the minimum number of original objects in its cluster
      is plotted first.

    ``'descending'``
      The child with the maximum number of original objects in its cluster
      is plotted first.

    Note ``distance_sort`` and ``count_sort`` cannot both be True.
distance_sort : str or bool, optional
    For each node n, the order (visually, from left-to-right) n's
    two descendent links are plotted is determined by this
    parameter, which can be any of the following values:

    ``False``
      Nothing is done.

    ``'ascending'`` or ``True``
      The child with the minimum distance between its direct descendents is
      plotted first.

    ``'descending'``
      The child with the maximum distance between its direct descendents is
      plotted first.

    Note ``distance_sort`` and ``count_sort`` cannot both be True.
show_leaf_counts : bool, optional
     When True, leaf nodes representing :math:`k>1` original
     observation are labeled with the number of observations they
     contain in parentheses.
no_plot : bool, optional
    When True, the final rendering is not performed. This is
    useful if only the data structures computed for the rendering
    are needed or if matplotlib is not available.
no_labels : bool, optional
    When True, no labels appear next to the leaf nodes in the
    rendering of the dendrogram.
leaf_rotation : double, optional
    Specifies the angle (in degrees) to rotate the leaf
    labels. When unspecified, the rotation is based on the number of
    nodes in the dendrogram (default is 0).
leaf_font_size : int, optional
    Specifies the font size (in points) of the leaf labels. When
    unspecified, the size based on the number of nodes in the
    dendrogram.
leaf_label_func : lambda or function, optional
    When leaf_label_func is a callable function, for each
    leaf with cluster index :math:`k < 2n-1`. The function
    is expected to return a string with the label for the
    leaf.

    Indices :math:`k < n` correspond to original observations
    while indices :math:`k \geq n` correspond to non-singleton
    clusters.

    For example, to label singletons with their node id and
    non-singletons with their id, count, and inconsistency
    coefficient, simply do::

        # First define the leaf label function.
        def llf(id):
            if id < n:
                return str(id)
            else:
                return '[%d %d %1.2f]' % (id, count, R[n-id,3])
        # The text for the leaf nodes is going to be big so force
        # a rotation of 90 degrees.
        dendrogram(Z, leaf_label_func=llf, leaf_rotation=90)

show_contracted : bool, optional
    When True the heights of non-singleton nodes contracted
    into a leaf node are plotted as crosses along the link
    connecting that leaf node.  This really is only useful when
    truncation is used (see ``truncate_mode`` parameter).
link_color_func : callable, optional
    If given, `link_color_function` is called with each non-singleton id
    corresponding to each U-shaped link it will paint. The function is
    expected to return the color to paint the link, encoded as a matplotlib
    color string code. For example::

        dendrogram(Z, link_color_func=lambda k: colors[k])

    colors the direct links below each untruncated non-singleton node
    ``k`` using ``colors[k]``.
ax : matplotlib Axes instance, optional
    If None and `no_plot` is not True, the dendrogram will be plotted
    on the current axes.  Otherwise if `no_plot` is not True the
    dendrogram will be plotted on the given ``Axes`` instance. This can be
    useful if the dendrogram is part of a more complex figure.
above_threshold_color : str, optional
    This matplotlib color string sets the color of the links above the
    color_threshold. The default is 'b'.

Returns
-------
R : dict
    A dictionary of data structures computed to render the
    dendrogram. Its has the following keys:

    ``'color_list'``
      A list of color names. The k'th element represents the color of the
      k'th link.

    ``'icoord'`` and ``'dcoord'``
      Each of them is a list of lists. Let ``icoord = [I1, I2, ..., Ip]``
      where ``Ik = [xk1, xk2, xk3, xk4]`` and ``dcoord = [D1, D2, ..., Dp]``
      where ``Dk = [yk1, yk2, yk3, yk4]``, then the k'th link painted is
      ``(xk1, yk1)`` - ``(xk2, yk2)`` - ``(xk3, yk3)`` - ``(xk4, yk4)``.

    ``'ivl'``
      A list of labels corresponding to the leaf nodes.

    ``'leaves'``
      For each i, ``H[i] == j``, cluster node ``j`` appears in position
      ``i`` in the left-to-right traversal of the leaves, where
      :math:`j < 2n-1` and :math:`i < n`. If ``j`` is less than ``n``, the
      ``i``-th leaf node corresponds to an original observation.
      Otherwise, it corresponds to a non-singleton cluster.

See Also
--------
linkage, set_link_color_palette

Notes
-----
It is expected that the distances in ``Z[:,2]`` be monotonic, otherwise
crossings appear in the dendrogram.

Examples
--------
>>> from scipy.cluster import hierarchy
>>> import matplotlib.pyplot as plt

A very basic example:

>>> ytdist = np.array([662., 877., 255., 412., 996., 295., 468., 268.,
...                    400., 754., 564., 138., 219., 869., 669.])
>>> Z = hierarchy.linkage(ytdist, 'single')
>>> plt.figure()
>>> dn = hierarchy.dendrogram(Z)

Now plot in given axes, improve the color scheme and use both vertical and
horizontal orientations:

>>> hierarchy.set_link_color_palette(['m', 'c', 'y', 'k'])
>>> fig, axes = plt.subplots(1, 2, figsize=(8, 3))
>>> dn1 = hierarchy.dendrogram(Z, ax=axes[0], above_threshold_color='y',
...                            orientation='top')
>>> dn2 = hierarchy.dendrogram(Z, ax=axes[1],
...                            above_threshold_color='#bcbddc',
...                            orientation='right')
>>> hierarchy.set_link_color_palette(None)  # reset to default after use
>>> plt.show()
*)

val fcluster : ?criterion:string -> ?depth:int -> ?r:[>`Ndarray] Np.Obj.t -> ?monocrit:[>`Ndarray] Np.Obj.t -> z:[>`Ndarray] Np.Obj.t -> t:[`F of float | `I of int | `Bool of bool | `S of string] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Form flat clusters from the hierarchical clustering defined by
the given linkage matrix.

Parameters
----------
Z : ndarray
    The hierarchical clustering encoded with the matrix returned
    by the `linkage` function.
t : scalar
    For criteria 'inconsistent', 'distance' or 'monocrit',
     this is the threshold to apply when forming flat clusters.
    For 'maxclust' or 'maxclust_monocrit' criteria,
     this would be max number of clusters requested.
criterion : str, optional
    The criterion to use in forming flat clusters. This can
    be any of the following values:

      ``inconsistent`` :
          If a cluster node and all its
          descendants have an inconsistent value less than or equal
          to `t` then all its leaf descendants belong to the
          same flat cluster. When no non-singleton cluster meets
          this criterion, every node is assigned to its own
          cluster. (Default)

      ``distance`` :
          Forms flat clusters so that the original
          observations in each flat cluster have no greater a
          cophenetic distance than `t`.

      ``maxclust`` :
          Finds a minimum threshold ``r`` so that
          the cophenetic distance between any two original
          observations in the same flat cluster is no more than
          ``r`` and no more than `t` flat clusters are formed.

      ``monocrit`` :
          Forms a flat cluster from a cluster node c
          with index i when ``monocrit[j] <= t``.

          For example, to threshold on the maximum mean distance
          as computed in the inconsistency matrix R with a
          threshold of 0.8 do::

              MR = maxRstat(Z, R, 3)
              cluster(Z, t=0.8, criterion='monocrit', monocrit=MR)

      ``maxclust_monocrit`` :
          Forms a flat cluster from a
          non-singleton cluster node ``c`` when ``monocrit[i] <=
          r`` for all cluster indices ``i`` below and including
          ``c``. ``r`` is minimized such that no more than ``t``
          flat clusters are formed. monocrit must be
          monotonic. For example, to minimize the threshold t on
          maximum inconsistency values so that no more than 3 flat
          clusters are formed, do::

              MI = maxinconsts(Z, R)
              cluster(Z, t=3, criterion='maxclust_monocrit', monocrit=MI)

depth : int, optional
    The maximum depth to perform the inconsistency calculation.
    It has no meaning for the other criteria. Default is 2.
R : ndarray, optional
    The inconsistency matrix to use for the 'inconsistent'
    criterion. This matrix is computed if not provided.
monocrit : ndarray, optional
    An array of length n-1. `monocrit[i]` is the
    statistics upon which non-singleton i is thresholded. The
    monocrit vector must be monotonic, i.e. given a node c with
    index i, for all node indices j corresponding to nodes
    below c, ``monocrit[i] >= monocrit[j]``.

Returns
-------
fcluster : ndarray
    An array of length ``n``. ``T[i]`` is the flat cluster number to
    which original observation ``i`` belongs.

See Also
--------
linkage : for information about hierarchical clustering methods work.

Examples
--------
>>> from scipy.cluster.hierarchy import ward, fcluster
>>> from scipy.spatial.distance import pdist

All cluster linkage methods - e.g. `scipy.cluster.hierarchy.ward`
generate a linkage matrix ``Z`` as their output:

>>> X = [[0, 0], [0, 1], [1, 0],
...      [0, 4], [0, 3], [1, 4],
...      [4, 0], [3, 0], [4, 1],
...      [4, 4], [3, 4], [4, 3]]

>>> Z = ward(pdist(X))

>>> Z
array([[ 0.        ,  1.        ,  1.        ,  2.        ],
       [ 3.        ,  4.        ,  1.        ,  2.        ],
       [ 6.        ,  7.        ,  1.        ,  2.        ],
       [ 9.        , 10.        ,  1.        ,  2.        ],
       [ 2.        , 12.        ,  1.29099445,  3.        ],
       [ 5.        , 13.        ,  1.29099445,  3.        ],
       [ 8.        , 14.        ,  1.29099445,  3.        ],
       [11.        , 15.        ,  1.29099445,  3.        ],
       [16.        , 17.        ,  5.77350269,  6.        ],
       [18.        , 19.        ,  5.77350269,  6.        ],
       [20.        , 21.        ,  8.16496581, 12.        ]])

This matrix represents a dendrogram, where the first and second elements
are the two clusters merged at each step, the third element is the
distance between these clusters, and the fourth element is the size of
the new cluster - the number of original data points included.

`scipy.cluster.hierarchy.fcluster` can be used to flatten the
dendrogram, obtaining as a result an assignation of the original data
points to single clusters.

This assignation mostly depends on a distance threshold ``t`` - the maximum
inter-cluster distance allowed:

>>> fcluster(Z, t=0.9, criterion='distance')
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12], dtype=int32)

>>> fcluster(Z, t=1.1, criterion='distance')
array([1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8], dtype=int32)

>>> fcluster(Z, t=3, criterion='distance')
array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], dtype=int32)

>>> fcluster(Z, t=9, criterion='distance')
array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)

In the first case, the threshold ``t`` is too small to allow any two
samples in the data to form a cluster, so 12 different clusters are
returned.

In the second case, the threshold is large enough to allow the first
4 points to be merged with their nearest neighbors. So here only 8
clusters are returned.

The third case, with a much higher threshold, allows for up to 8 data
points to be connected - so 4 clusters are returned here.

Lastly, the threshold of the fourth case is large enough to allow for
all data points to be merged together - so a single cluster is returned.
*)

val fclusterdata : ?criterion:string -> ?metric:string -> ?depth:int -> ?method_:string -> ?r:[>`Ndarray] Np.Obj.t -> x:[>`Ndarray] Np.Obj.t -> t:[`F of float | `I of int | `Bool of bool | `S of string] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Cluster observation data using a given metric.

Clusters the original observations in the n-by-m data
matrix X (n observations in m dimensions), using the euclidean
distance metric to calculate distances between original observations,
performs hierarchical clustering using the single linkage algorithm,
and forms flat clusters using the inconsistency method with `t` as the
cut-off threshold.

A one-dimensional array ``T`` of length ``n`` is returned. ``T[i]`` is
the index of the flat cluster to which the original observation ``i``
belongs.

Parameters
----------
X : (N, M) ndarray
    N by M data matrix with N observations in M dimensions.
t : scalar
    For criteria 'inconsistent', 'distance' or 'monocrit',
     this is the threshold to apply when forming flat clusters.
    For 'maxclust' or 'maxclust_monocrit' criteria,
     this would be max number of clusters requested.
criterion : str, optional
    Specifies the criterion for forming flat clusters.  Valid
    values are 'inconsistent' (default), 'distance', or 'maxclust'
    cluster formation algorithms. See `fcluster` for descriptions.
metric : str, optional
    The distance metric for calculating pairwise distances. See
    ``distance.pdist`` for descriptions and linkage to verify
    compatibility with the linkage method.
depth : int, optional
    The maximum depth for the inconsistency calculation. See
    `inconsistent` for more information.
method : str, optional
    The linkage method to use (single, complete, average,
    weighted, median centroid, ward). See `linkage` for more
    information. Default is 'single'.
R : ndarray, optional
    The inconsistency matrix. It will be computed if necessary
    if it is not passed.

Returns
-------
fclusterdata : ndarray
    A vector of length n. T[i] is the flat cluster number to
    which original observation i belongs.

See Also
--------
scipy.spatial.distance.pdist : pairwise distance metrics

Notes
-----
This function is similar to the MATLAB function ``clusterdata``.

Examples
--------
>>> from scipy.cluster.hierarchy import fclusterdata

This is a convenience method that abstracts all the steps to perform in a
typical SciPy's hierarchical clustering workflow.

* Transform the input data into a condensed matrix with `scipy.spatial.distance.pdist`.

* Apply a clustering method.

* Obtain flat clusters at a user defined distance threshold ``t`` using `scipy.cluster.hierarchy.fcluster`.

>>> X = [[0, 0], [0, 1], [1, 0],
...      [0, 4], [0, 3], [1, 4],
...      [4, 0], [3, 0], [4, 1],
...      [4, 4], [3, 4], [4, 3]]

>>> fclusterdata(X, t=1)
array([3, 3, 3, 4, 4, 4, 2, 2, 2, 1, 1, 1], dtype=int32)

The output here (for the dataset ``X``, distance threshold ``t``, and the
default settings) is four clusters with three data points each.
*)

val from_mlab_linkage : [>`Ndarray] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Convert a linkage matrix generated by MATLAB(TM) to a new
linkage matrix compatible with this module.

The conversion does two things:

 * the indices are converted from ``1..N`` to ``0..(N-1)`` form,
   and

 * a fourth column ``Z[:,3]`` is added where ``Z[i,3]`` represents the
   number of original observations (leaves) in the non-singleton
   cluster ``i``.

This function is useful when loading in linkages from legacy data
files generated by MATLAB.

Parameters
----------
Z : ndarray
    A linkage matrix generated by MATLAB(TM).

Returns
-------
ZS : ndarray
    A linkage matrix compatible with ``scipy.cluster.hierarchy``.

See Also
--------
linkage: for a description of what a linkage matrix is.
to_mlab_linkage: transform from SciPy to MATLAB format.

Examples
--------
>>> import numpy as np
>>> from scipy.cluster.hierarchy import ward, from_mlab_linkage

Given a linkage matrix in MATLAB format ``mZ``, we can use
`scipy.cluster.hierarchy.from_mlab_linkage` to import
it into SciPy format:

>>> mZ = np.array([[1, 2, 1], [4, 5, 1], [7, 8, 1],
...                [10, 11, 1], [3, 13, 1.29099445],
...                [6, 14, 1.29099445],
...                [9, 15, 1.29099445],
...                [12, 16, 1.29099445],
...                [17, 18, 5.77350269],
...                [19, 20, 5.77350269],
...                [21, 22,  8.16496581]])

>>> Z = from_mlab_linkage(mZ)
>>> Z
array([[  0.        ,   1.        ,   1.        ,   2.        ],
       [  3.        ,   4.        ,   1.        ,   2.        ],
       [  6.        ,   7.        ,   1.        ,   2.        ],
       [  9.        ,  10.        ,   1.        ,   2.        ],
       [  2.        ,  12.        ,   1.29099445,   3.        ],
       [  5.        ,  13.        ,   1.29099445,   3.        ],
       [  8.        ,  14.        ,   1.29099445,   3.        ],
       [ 11.        ,  15.        ,   1.29099445,   3.        ],
       [ 16.        ,  17.        ,   5.77350269,   6.        ],
       [ 18.        ,  19.        ,   5.77350269,   6.        ],
       [ 20.        ,  21.        ,   8.16496581,  12.        ]])

As expected, the linkage matrix ``Z`` returned includes an
additional column counting the number of original samples in
each cluster. Also, all cluster indexes are reduced by 1
(MATLAB format uses 1-indexing, whereas SciPy uses 0-indexing).
*)

val inconsistent : ?d:int -> z:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Calculate inconsistency statistics on a linkage matrix.

Parameters
----------
Z : ndarray
    The :math:`(n-1)` by 4 matrix encoding the linkage (hierarchical
    clustering).  See `linkage` documentation for more information on its
    form.
d : int, optional
    The number of links up to `d` levels below each non-singleton cluster.

Returns
-------
R : ndarray
    A :math:`(n-1)` by 4 matrix where the ``i``'th row contains the link
    statistics for the non-singleton cluster ``i``. The link statistics are
    computed over the link heights for links :math:`d` levels below the
    cluster ``i``. ``R[i,0]`` and ``R[i,1]`` are the mean and standard
    deviation of the link heights, respectively; ``R[i,2]`` is the number
    of links included in the calculation; and ``R[i,3]`` is the
    inconsistency coefficient,

    .. math:: \frac{\mathtt{Z[i,2]} - \mathtt{R[i,0]}} {R[i,1]}

Notes
-----
This function behaves similarly to the MATLAB(TM) ``inconsistent``
function.

Examples
--------
>>> from scipy.cluster.hierarchy import inconsistent, linkage
>>> from matplotlib import pyplot as plt
>>> X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]
>>> Z = linkage(X, 'ward')
>>> print(Z)
[[ 5.          6.          0.          2.        ]
 [ 2.          7.          0.          2.        ]
 [ 0.          4.          1.          2.        ]
 [ 1.          8.          1.15470054  3.        ]
 [ 9.         10.          2.12132034  4.        ]
 [ 3.         12.          4.11096096  5.        ]
 [11.         13.         14.07183949  8.        ]]
>>> inconsistent(Z)
array([[ 0.        ,  0.        ,  1.        ,  0.        ],
       [ 0.        ,  0.        ,  1.        ,  0.        ],
       [ 1.        ,  0.        ,  1.        ,  0.        ],
       [ 0.57735027,  0.81649658,  2.        ,  0.70710678],
       [ 1.04044011,  1.06123822,  3.        ,  1.01850858],
       [ 3.11614065,  1.40688837,  2.        ,  0.70710678],
       [ 6.44583366,  6.76770586,  3.        ,  1.12682288]])
*)

val is_isomorphic : t1:[>`Ndarray] Np.Obj.t -> t2:[>`Ndarray] Np.Obj.t -> unit -> bool
(**
Determine if two different cluster assignments are equivalent.

Parameters
----------
T1 : array_like
    An assignment of singleton cluster ids to flat cluster ids.
T2 : array_like
    An assignment of singleton cluster ids to flat cluster ids.

Returns
-------
b : bool
    Whether the flat cluster assignments `T1` and `T2` are
    equivalent.

See Also
--------
linkage: for a description of what a linkage matrix is.
fcluster: for the creation of flat cluster assignments.

Examples
--------
>>> from scipy.cluster.hierarchy import fcluster, is_isomorphic
>>> from scipy.cluster.hierarchy import single, complete
>>> from scipy.spatial.distance import pdist

Two flat cluster assignments can be isomorphic if they represent the same
cluster assignment, with different labels.

For example, we can use the `scipy.cluster.hierarchy.single`: method
and flatten the output to four clusters:

>>> X = [[0, 0], [0, 1], [1, 0],
...      [0, 4], [0, 3], [1, 4],
...      [4, 0], [3, 0], [4, 1],
...      [4, 4], [3, 4], [4, 3]]

>>> Z = single(pdist(X))
>>> T = fcluster(Z, 1, criterion='distance')
>>> T
array([3, 3, 3, 4, 4, 4, 2, 2, 2, 1, 1, 1], dtype=int32)

We can then do the same using the
`scipy.cluster.hierarchy.complete`: method:

>>> Z = complete(pdist(X))
>>> T_ = fcluster(Z, 1.5, criterion='distance')
>>> T_
array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], dtype=int32)

As we can see, in both cases we obtain four clusters and all the data
points are distributed in the same way - the only thing that changes
are the flat cluster labels (3 => 1, 4 =>2, 2 =>3 and 4 =>1), so both
cluster assignments are isomorphic:

>>> is_isomorphic(T, T_)
True
*)

val is_monotonic : [>`Ndarray] Np.Obj.t -> bool
(**
Return True if the linkage passed is monotonic.

The linkage is monotonic if for every cluster :math:`s` and :math:`t`
joined, the distance between them is no less than the distance
between any previously joined clusters.

Parameters
----------
Z : ndarray
    The linkage matrix to check for monotonicity.

Returns
-------
b : bool
    A boolean indicating whether the linkage is monotonic.

See Also
--------
linkage: for a description of what a linkage matrix is.

Examples
--------
>>> from scipy.cluster.hierarchy import median, ward, is_monotonic
>>> from scipy.spatial.distance import pdist

By definition, some hierarchical clustering algorithms - such as
`scipy.cluster.hierarchy.ward` - produce monotonic assignments of
samples to clusters; however, this is not always true for other
hierarchical methods - e.g. `scipy.cluster.hierarchy.median`.

Given a linkage matrix ``Z`` (as the result of a hierarchical clustering
method) we can test programmatically whether if is has the monotonicity
property or not, using `scipy.cluster.hierarchy.is_monotonic`:

>>> X = [[0, 0], [0, 1], [1, 0],
...      [0, 4], [0, 3], [1, 4],
...      [4, 0], [3, 0], [4, 1],
...      [4, 4], [3, 4], [4, 3]]

>>> Z = ward(pdist(X))
>>> Z
array([[ 0.        ,  1.        ,  1.        ,  2.        ],
       [ 3.        ,  4.        ,  1.        ,  2.        ],
       [ 6.        ,  7.        ,  1.        ,  2.        ],
       [ 9.        , 10.        ,  1.        ,  2.        ],
       [ 2.        , 12.        ,  1.29099445,  3.        ],
       [ 5.        , 13.        ,  1.29099445,  3.        ],
       [ 8.        , 14.        ,  1.29099445,  3.        ],
       [11.        , 15.        ,  1.29099445,  3.        ],
       [16.        , 17.        ,  5.77350269,  6.        ],
       [18.        , 19.        ,  5.77350269,  6.        ],
       [20.        , 21.        ,  8.16496581, 12.        ]])
>>> is_monotonic(Z)
True

>>> Z = median(pdist(X))
>>> Z
array([[ 0.        ,  1.        ,  1.        ,  2.        ],
       [ 3.        ,  4.        ,  1.        ,  2.        ],
       [ 9.        , 10.        ,  1.        ,  2.        ],
       [ 6.        ,  7.        ,  1.        ,  2.        ],
       [ 2.        , 12.        ,  1.11803399,  3.        ],
       [ 5.        , 13.        ,  1.11803399,  3.        ],
       [ 8.        , 15.        ,  1.11803399,  3.        ],
       [11.        , 14.        ,  1.11803399,  3.        ],
       [18.        , 19.        ,  3.        ,  6.        ],
       [16.        , 17.        ,  3.5       ,  6.        ],
       [20.        , 21.        ,  3.25      , 12.        ]])
>>> is_monotonic(Z)
False

Note that this method is equivalent to just verifying that the distances
in the third column of the linkage matrix appear in a monotonically
increasing order.
*)

val is_valid_im : ?warning:bool -> ?throw:bool -> ?name:string -> r:[>`Ndarray] Np.Obj.t -> unit -> bool
(**
Return True if the inconsistency matrix passed is valid.

It must be a :math:`n` by 4 array of doubles. The standard
deviations ``R[:,1]`` must be nonnegative. The link counts
``R[:,2]`` must be positive and no greater than :math:`n-1`.

Parameters
----------
R : ndarray
    The inconsistency matrix to check for validity.
warning : bool, optional
     When True, issues a Python warning if the linkage
     matrix passed is invalid.
throw : bool, optional
     When True, throws a Python exception if the linkage
     matrix passed is invalid.
name : str, optional
     This string refers to the variable name of the invalid
     linkage matrix.

Returns
-------
b : bool
    True if the inconsistency matrix is valid.

See Also
--------
linkage: for a description of what a linkage matrix is.
inconsistent: for the creation of a inconsistency matrix.

Examples
--------
>>> from scipy.cluster.hierarchy import ward, inconsistent, is_valid_im
>>> from scipy.spatial.distance import pdist

Given a data set ``X``, we can apply a clustering method to obtain a
linkage matrix ``Z``. `scipy.cluster.hierarchy.inconsistent` can
be also used to obtain the inconsistency matrix ``R`` associated to
this clustering process:

>>> X = [[0, 0], [0, 1], [1, 0],
...      [0, 4], [0, 3], [1, 4],
...      [4, 0], [3, 0], [4, 1],
...      [4, 4], [3, 4], [4, 3]]

>>> Z = ward(pdist(X))
>>> R = inconsistent(Z)
>>> Z
array([[ 0.        ,  1.        ,  1.        ,  2.        ],
       [ 3.        ,  4.        ,  1.        ,  2.        ],
       [ 6.        ,  7.        ,  1.        ,  2.        ],
       [ 9.        , 10.        ,  1.        ,  2.        ],
       [ 2.        , 12.        ,  1.29099445,  3.        ],
       [ 5.        , 13.        ,  1.29099445,  3.        ],
       [ 8.        , 14.        ,  1.29099445,  3.        ],
       [11.        , 15.        ,  1.29099445,  3.        ],
       [16.        , 17.        ,  5.77350269,  6.        ],
       [18.        , 19.        ,  5.77350269,  6.        ],
       [20.        , 21.        ,  8.16496581, 12.        ]])
>>> R
array([[1.        , 0.        , 1.        , 0.        ],
       [1.        , 0.        , 1.        , 0.        ],
       [1.        , 0.        , 1.        , 0.        ],
       [1.        , 0.        , 1.        , 0.        ],
       [1.14549722, 0.20576415, 2.        , 0.70710678],
       [1.14549722, 0.20576415, 2.        , 0.70710678],
       [1.14549722, 0.20576415, 2.        , 0.70710678],
       [1.14549722, 0.20576415, 2.        , 0.70710678],
       [2.78516386, 2.58797734, 3.        , 1.15470054],
       [2.78516386, 2.58797734, 3.        , 1.15470054],
       [6.57065706, 1.38071187, 3.        , 1.15470054]])

Now we can use `scipy.cluster.hierarchy.is_valid_im` to verify that
``R`` is correct:

>>> is_valid_im(R)
True

However, if ``R`` is wrongly constructed (e.g one of the standard
deviations is set to a negative value) then the check will fail:

>>> R[-1,1] = R[-1,1] * -1
>>> is_valid_im(R)
False
*)

val is_valid_linkage : ?warning:bool -> ?throw:bool -> ?name:string -> z:[>`Ndarray] Np.Obj.t -> unit -> bool
(**
Check the validity of a linkage matrix.

A linkage matrix is valid if it is a two dimensional array (type double)
with :math:`n` rows and 4 columns.  The first two columns must contain
indices between 0 and :math:`2n-1`. For a given row ``i``, the following
two expressions have to hold:

.. math::

    0 \leq \mathtt{Z[i,0]} \leq i+n-1
    0 \leq Z[i,1] \leq i+n-1

I.e. a cluster cannot join another cluster unless the cluster being joined
has been generated.

Parameters
----------
Z : array_like
    Linkage matrix.
warning : bool, optional
    When True, issues a Python warning if the linkage
    matrix passed is invalid.
throw : bool, optional
    When True, throws a Python exception if the linkage
    matrix passed is invalid.
name : str, optional
    This string refers to the variable name of the invalid
    linkage matrix.

Returns
-------
b : bool
    True if the inconsistency matrix is valid.

See Also
--------
linkage: for a description of what a linkage matrix is.

Examples
--------
>>> from scipy.cluster.hierarchy import ward, is_valid_linkage
>>> from scipy.spatial.distance import pdist

All linkage matrices generated by the clustering methods in this module
will be valid (i.e. they will have the appropriate dimensions and the two
required expressions will hold for all the rows).

We can check this using `scipy.cluster.hierarchy.is_valid_linkage`:

>>> X = [[0, 0], [0, 1], [1, 0],
...      [0, 4], [0, 3], [1, 4],
...      [4, 0], [3, 0], [4, 1],
...      [4, 4], [3, 4], [4, 3]]

>>> Z = ward(pdist(X))
>>> Z
array([[ 0.        ,  1.        ,  1.        ,  2.        ],
       [ 3.        ,  4.        ,  1.        ,  2.        ],
       [ 6.        ,  7.        ,  1.        ,  2.        ],
       [ 9.        , 10.        ,  1.        ,  2.        ],
       [ 2.        , 12.        ,  1.29099445,  3.        ],
       [ 5.        , 13.        ,  1.29099445,  3.        ],
       [ 8.        , 14.        ,  1.29099445,  3.        ],
       [11.        , 15.        ,  1.29099445,  3.        ],
       [16.        , 17.        ,  5.77350269,  6.        ],
       [18.        , 19.        ,  5.77350269,  6.        ],
       [20.        , 21.        ,  8.16496581, 12.        ]])
>>> is_valid_linkage(Z)
True

However, is we create a linkage matrix in a wrong way - or if we modify
a valid one in a way that any of the required expressions don't hold
anymore, then the check will fail:

>>> Z[3][1] = 20    # the cluster number 20 is not defined at this point
>>> is_valid_linkage(Z)
False
*)

val leaders : z:[>`Ndarray] Np.Obj.t -> t:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Return the root nodes in a hierarchical clustering.

Returns the root nodes in a hierarchical clustering corresponding
to a cut defined by a flat cluster assignment vector ``T``. See
the ``fcluster`` function for more information on the format of ``T``.

For each flat cluster :math:`j` of the :math:`k` flat clusters
represented in the n-sized flat cluster assignment vector ``T``,
this function finds the lowest cluster node :math:`i` in the linkage
tree Z such that:

  * leaf descendants belong only to flat cluster j
    (i.e. ``T[p]==j`` for all :math:`p` in :math:`S(i)` where
    :math:`S(i)` is the set of leaf ids of descendant leaf nodes
    with cluster node :math:`i`)

  * there does not exist a leaf that is not a descendant with
    :math:`i` that also belongs to cluster :math:`j`
    (i.e. ``T[q]!=j`` for all :math:`q` not in :math:`S(i)`).  If
    this condition is violated, ``T`` is not a valid cluster
    assignment vector, and an exception will be thrown.

Parameters
----------
Z : ndarray
    The hierarchical clustering encoded as a matrix. See
    `linkage` for more information.
T : ndarray
    The flat cluster assignment vector.

Returns
-------
L : ndarray
    The leader linkage node id's stored as a k-element 1-D array
    where ``k`` is the number of flat clusters found in ``T``.

    ``L[j]=i`` is the linkage cluster node id that is the
    leader of flat cluster with id M[j].  If ``i < n``, ``i``
    corresponds to an original observation, otherwise it
    corresponds to a non-singleton cluster.

M : ndarray
    The leader linkage node id's stored as a k-element 1-D array where
    ``k`` is the number of flat clusters found in ``T``. This allows the
    set of flat cluster ids to be any arbitrary set of ``k`` integers.

    For example: if ``L[3]=2`` and ``M[3]=8``, the flat cluster with
    id 8's leader is linkage node 2.

See Also
--------
fcluster: for the creation of flat cluster assignments.

Examples
--------
>>> from scipy.cluster.hierarchy import ward, fcluster, leaders
>>> from scipy.spatial.distance import pdist

Given a linkage matrix ``Z`` - obtained after apply a clustering method
to a dataset ``X`` - and a flat cluster assignment array ``T``:

>>> X = [[0, 0], [0, 1], [1, 0],
...      [0, 4], [0, 3], [1, 4],
...      [4, 0], [3, 0], [4, 1],
...      [4, 4], [3, 4], [4, 3]]

>>> Z = ward(pdist(X))
>>> Z
array([[ 0.        ,  1.        ,  1.        ,  2.        ],
       [ 3.        ,  4.        ,  1.        ,  2.        ],
       [ 6.        ,  7.        ,  1.        ,  2.        ],
       [ 9.        , 10.        ,  1.        ,  2.        ],
       [ 2.        , 12.        ,  1.29099445,  3.        ],
       [ 5.        , 13.        ,  1.29099445,  3.        ],
       [ 8.        , 14.        ,  1.29099445,  3.        ],
       [11.        , 15.        ,  1.29099445,  3.        ],
       [16.        , 17.        ,  5.77350269,  6.        ],
       [18.        , 19.        ,  5.77350269,  6.        ],
       [20.        , 21.        ,  8.16496581, 12.        ]])


>>> T = fcluster(Z, 3, criterion='distance')
>>> T
array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], dtype=int32)

`scipy.cluster.hierarchy.leaders` returns the indexes of the nodes
in the dendrogram that are the leaders of each flat cluster:

>>> L, M = leaders(Z, T)
>>> L
array([16, 17, 18, 19], dtype=int32)

(remember that indexes 0-11 point to the 12 data points in ``X``
whereas indexes 12-22 point to the 11 rows of ``Z``)

`scipy.cluster.hierarchy.leaders` also returns the indexes of
the flat clusters in ``T``:

>>> M
array([1, 2, 3, 4], dtype=int32)
*)

val leaves_list : [>`Ndarray] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return a list of leaf node ids.

The return corresponds to the observation vector index as it appears
in the tree from left to right. Z is a linkage matrix.

Parameters
----------
Z : ndarray
    The hierarchical clustering encoded as a matrix.  `Z` is
    a linkage matrix.  See `linkage` for more information.

Returns
-------
leaves_list : ndarray
    The list of leaf node ids.

See Also
--------
dendrogram: for information about dendrogram structure.

Examples
--------
>>> from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
>>> from scipy.spatial.distance import pdist
>>> from matplotlib import pyplot as plt

>>> X = [[0, 0], [0, 1], [1, 0],
...      [0, 4], [0, 3], [1, 4],
...      [4, 0], [3, 0], [4, 1],
...      [4, 4], [3, 4], [4, 3]]

>>> Z = ward(pdist(X))

The linkage matrix ``Z`` represents a dendrogram, that is, a tree that
encodes the structure of the clustering performed.
`scipy.cluster.hierarchy.leaves_list` shows the mapping between
indexes in the ``X`` dataset and leaves in the dendrogram:

>>> leaves_list(Z)
array([ 2,  0,  1,  5,  3,  4,  8,  6,  7, 11,  9, 10], dtype=int32)

>>> fig = plt.figure(figsize=(25, 10))
>>> dn = dendrogram(Z)
>>> plt.show()
*)

val linkage : ?method_:string -> ?metric:[`Callable of Py.Object.t | `S of string] -> ?optimal_ordering:bool -> y:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Perform hierarchical/agglomerative clustering.

The input y may be either a 1d condensed distance matrix
or a 2d array of observation vectors.

If y is a 1d condensed distance matrix,
then y must be a :math:`\binom{n}{2}` sized
vector where n is the number of original observations paired
in the distance matrix. The behavior of this function is very
similar to the MATLAB linkage function.

A :math:`(n-1)` by 4 matrix ``Z`` is returned. At the
:math:`i`-th iteration, clusters with indices ``Z[i, 0]`` and
``Z[i, 1]`` are combined to form cluster :math:`n + i`. A
cluster with an index less than :math:`n` corresponds to one of
the :math:`n` original observations. The distance between
clusters ``Z[i, 0]`` and ``Z[i, 1]`` is given by ``Z[i, 2]``. The
fourth value ``Z[i, 3]`` represents the number of original
observations in the newly formed cluster.

The following linkage methods are used to compute the distance
:math:`d(s, t)` between two clusters :math:`s` and
:math:`t`. The algorithm begins with a forest of clusters that
have yet to be used in the hierarchy being formed. When two
clusters :math:`s` and :math:`t` from this forest are combined
into a single cluster :math:`u`, :math:`s` and :math:`t` are
removed from the forest, and :math:`u` is added to the
forest. When only one cluster remains in the forest, the algorithm
stops, and this cluster becomes the root.

A distance matrix is maintained at each iteration. The ``d[i,j]``
entry corresponds to the distance between cluster :math:`i` and
:math:`j` in the original forest.

At each iteration, the algorithm must update the distance matrix
to reflect the distance of the newly formed cluster u with the
remaining clusters in the forest.

Suppose there are :math:`|u|` original observations
:math:`u[0], \ldots, u[|u|-1]` in cluster :math:`u` and
:math:`|v|` original objects :math:`v[0], \ldots, v[|v|-1]` in
cluster :math:`v`. Recall :math:`s` and :math:`t` are
combined to form cluster :math:`u`. Let :math:`v` be any
remaining cluster in the forest that is not :math:`u`.

The following are methods for calculating the distance between the
newly formed cluster :math:`u` and each :math:`v`.

  * method='single' assigns

    .. math::
       d(u,v) = \min(dist(u[i],v[j]))

    for all points :math:`i` in cluster :math:`u` and
    :math:`j` in cluster :math:`v`. This is also known as the
    Nearest Point Algorithm.

  * method='complete' assigns

    .. math::
       d(u, v) = \max(dist(u[i],v[j]))

    for all points :math:`i` in cluster u and :math:`j` in
    cluster :math:`v`. This is also known by the Farthest Point
    Algorithm or Voor Hees Algorithm.

  * method='average' assigns

    .. math::
       d(u,v) = \sum_{ij} \frac{d(u[i], v[j])}
                               {(|u|*|v|)}

    for all points :math:`i` and :math:`j` where :math:`|u|`
    and :math:`|v|` are the cardinalities of clusters :math:`u`
    and :math:`v`, respectively. This is also called the UPGMA
    algorithm.

  * method='weighted' assigns

    .. math::
       d(u,v) = (dist(s,v) + dist(t,v))/2

    where cluster u was formed with cluster s and t and v
    is a remaining cluster in the forest. (also called WPGMA)

  * method='centroid' assigns

    .. math::
       dist(s,t) = ||c_s-c_t||_2

    where :math:`c_s` and :math:`c_t` are the centroids of
    clusters :math:`s` and :math:`t`, respectively. When two
    clusters :math:`s` and :math:`t` are combined into a new
    cluster :math:`u`, the new centroid is computed over all the
    original objects in clusters :math:`s` and :math:`t`. The
    distance then becomes the Euclidean distance between the
    centroid of :math:`u` and the centroid of a remaining cluster
    :math:`v` in the forest. This is also known as the UPGMC
    algorithm.

  * method='median' assigns :math:`d(s,t)` like the ``centroid``
    method. When two clusters :math:`s` and :math:`t` are combined
    into a new cluster :math:`u`, the average of centroids s and t
    give the new centroid :math:`u`. This is also known as the
    WPGMC algorithm.

  * method='ward' uses the Ward variance minimization algorithm.
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

Warning: When the minimum distance pair in the forest is chosen, there
may be two or more pairs with the same minimum distance. This
implementation may choose a different minimum than the MATLAB
version.

Parameters
----------
y : ndarray
    A condensed distance matrix. A condensed distance matrix
    is a flat array containing the upper triangular of the distance matrix.
    This is the form that ``pdist`` returns. Alternatively, a collection of
    :math:`m` observation vectors in :math:`n` dimensions may be passed as
    an :math:`m` by :math:`n` array. All elements of the condensed distance
    matrix must be finite, i.e. no NaNs or infs.
method : str, optional
    The linkage algorithm to use. See the ``Linkage Methods`` section below
    for full descriptions.
metric : str or function, optional
    The distance metric to use in the case that y is a collection of
    observation vectors; ignored otherwise. See the ``pdist``
    function for a list of valid distance metrics. A custom distance
    function can also be used.
optimal_ordering : bool, optional
    If True, the linkage matrix will be reordered so that the distance
    between successive leaves is minimal. This results in a more intuitive
    tree structure when the data are visualized. defaults to False, because
    this algorithm can be slow, particularly on large datasets [2]_. See
    also the `optimal_leaf_ordering` function.

    .. versionadded:: 1.0.0

Returns
-------
Z : ndarray
    The hierarchical clustering encoded as a linkage matrix.

Notes
-----
1. For method 'single' an optimized algorithm based on minimum spanning
   tree is implemented. It has time complexity :math:`O(n^2)`.
   For methods 'complete', 'average', 'weighted' and 'ward' an algorithm
   called nearest-neighbors chain is implemented. It also has time
   complexity :math:`O(n^2)`.
   For other methods a naive algorithm is implemented with :math:`O(n^3)`
   time complexity.
   All algorithms use :math:`O(n^2)` memory.
   Refer to [1]_ for details about the algorithms.
2. Methods 'centroid', 'median' and 'ward' are correctly defined only if
   Euclidean pairwise metric is used. If `y` is passed as precomputed
   pairwise distances, then it is a user responsibility to assure that
   these distances are in fact Euclidean, otherwise the produced result
   will be incorrect.

See Also
--------
scipy.spatial.distance.pdist : pairwise distance metrics

References
----------
.. [1] Daniel Mullner, 'Modern hierarchical, agglomerative clustering
       algorithms', :arXiv:`1109.2378v1`.
.. [2] Ziv Bar-Joseph, David K. Gifford, Tommi S. Jaakkola, 'Fast optimal
       leaf ordering for hierarchical clustering', 2001. Bioinformatics
       :doi:`10.1093/bioinformatics/17.suppl_1.S22`

Examples
--------
>>> from scipy.cluster.hierarchy import dendrogram, linkage
>>> from matplotlib import pyplot as plt
>>> X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]

>>> Z = linkage(X, 'ward')
>>> fig = plt.figure(figsize=(25, 10))
>>> dn = dendrogram(Z)

>>> Z = linkage(X, 'single')
>>> fig = plt.figure(figsize=(25, 10))
>>> dn = dendrogram(Z)
>>> plt.show()
*)

val maxRstat : z:[>`Ndarray] Np.Obj.t -> r:[>`Ndarray] Np.Obj.t -> i:int -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return the maximum statistic for each non-singleton cluster and its
children.

Parameters
----------
Z : array_like
    The hierarchical clustering encoded as a matrix. See `linkage` for more
    information.
R : array_like
    The inconsistency matrix.
i : int
    The column of `R` to use as the statistic.

Returns
-------
MR : ndarray
    Calculates the maximum statistic for the i'th column of the
    inconsistency matrix `R` for each non-singleton cluster
    node. ``MR[j]`` is the maximum over ``R[Q(j)-n, i]`` where
    ``Q(j)`` the set of all node ids corresponding to nodes below
    and including ``j``.

See Also
--------
linkage: for a description of what a linkage matrix is.
inconsistent: for the creation of a inconsistency matrix.

Examples
--------
>>> from scipy.cluster.hierarchy import median, inconsistent, maxRstat
>>> from scipy.spatial.distance import pdist

Given a data set ``X``, we can apply a clustering method to obtain a
linkage matrix ``Z``. `scipy.cluster.hierarchy.inconsistent` can
be also used to obtain the inconsistency matrix ``R`` associated to
this clustering process:

>>> X = [[0, 0], [0, 1], [1, 0],
...      [0, 4], [0, 3], [1, 4],
...      [4, 0], [3, 0], [4, 1],
...      [4, 4], [3, 4], [4, 3]]

>>> Z = median(pdist(X))
>>> R = inconsistent(Z)
>>> R
array([[1.        , 0.        , 1.        , 0.        ],
       [1.        , 0.        , 1.        , 0.        ],
       [1.        , 0.        , 1.        , 0.        ],
       [1.        , 0.        , 1.        , 0.        ],
       [1.05901699, 0.08346263, 2.        , 0.70710678],
       [1.05901699, 0.08346263, 2.        , 0.70710678],
       [1.05901699, 0.08346263, 2.        , 0.70710678],
       [1.05901699, 0.08346263, 2.        , 0.70710678],
       [1.74535599, 1.08655358, 3.        , 1.15470054],
       [1.91202266, 1.37522872, 3.        , 1.15470054],
       [3.25      , 0.25      , 3.        , 0.        ]])

`scipy.cluster.hierarchy.maxRstat` can be used to compute
the maximum value of each column of ``R``, for each non-singleton
cluster and its children:

>>> maxRstat(Z, R, 0)
array([1.        , 1.        , 1.        , 1.        , 1.05901699,
       1.05901699, 1.05901699, 1.05901699, 1.74535599, 1.91202266,
       3.25      ])
>>> maxRstat(Z, R, 1)
array([0.        , 0.        , 0.        , 0.        , 0.08346263,
       0.08346263, 0.08346263, 0.08346263, 1.08655358, 1.37522872,
       1.37522872])
>>> maxRstat(Z, R, 3)
array([0.        , 0.        , 0.        , 0.        , 0.70710678,
       0.70710678, 0.70710678, 0.70710678, 1.15470054, 1.15470054,
       1.15470054])
*)

val maxdists : [>`Ndarray] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return the maximum distance between any non-singleton cluster.

Parameters
----------
Z : ndarray
    The hierarchical clustering encoded as a matrix. See
    ``linkage`` for more information.

Returns
-------
maxdists : ndarray
    A ``(n-1)`` sized numpy array of doubles; ``MD[i]`` represents
    the maximum distance between any cluster (including
    singletons) below and including the node with index i. More
    specifically, ``MD[i] = Z[Q(i)-n, 2].max()`` where ``Q(i)`` is the
    set of all node indices below and including node i.

See Also
--------
linkage: for a description of what a linkage matrix is.
is_monotonic: for testing for monotonicity of a linkage matrix.

Examples
--------
>>> from scipy.cluster.hierarchy import median, maxdists
>>> from scipy.spatial.distance import pdist

Given a linkage matrix ``Z``, `scipy.cluster.hierarchy.maxdists`
computes for each new cluster generated (i.e. for each row of the linkage
matrix) what is the maximum distance between any two child clusters.

Due to the nature of hierarchical clustering, in many cases this is going
to be just the distance between the two child clusters that were merged
to form the current one - that is, Z[:,2].

However, for non-monotonic cluster assignments such as
`scipy.cluster.hierarchy.median` clustering this is not always the
case: There may be cluster formations were the distance between the two
clusters merged is smaller than the distance between their children.

We can see this in an example:

>>> X = [[0, 0], [0, 1], [1, 0],
...      [0, 4], [0, 3], [1, 4],
...      [4, 0], [3, 0], [4, 1],
...      [4, 4], [3, 4], [4, 3]]

>>> Z = median(pdist(X))
>>> Z
array([[ 0.        ,  1.        ,  1.        ,  2.        ],
       [ 3.        ,  4.        ,  1.        ,  2.        ],
       [ 9.        , 10.        ,  1.        ,  2.        ],
       [ 6.        ,  7.        ,  1.        ,  2.        ],
       [ 2.        , 12.        ,  1.11803399,  3.        ],
       [ 5.        , 13.        ,  1.11803399,  3.        ],
       [ 8.        , 15.        ,  1.11803399,  3.        ],
       [11.        , 14.        ,  1.11803399,  3.        ],
       [18.        , 19.        ,  3.        ,  6.        ],
       [16.        , 17.        ,  3.5       ,  6.        ],
       [20.        , 21.        ,  3.25      , 12.        ]])
>>> maxdists(Z)
array([1.        , 1.        , 1.        , 1.        , 1.11803399,
       1.11803399, 1.11803399, 1.11803399, 3.        , 3.5       ,
       3.5       ])

Note that while the distance between the two clusters merged when creating the
last cluster is 3.25, there are two children (clusters 16 and 17) whose distance
is larger (3.5). Thus, `scipy.cluster.hierarchy.maxdists` returns 3.5 in
this case.
*)

val maxinconsts : z:[>`Ndarray] Np.Obj.t -> r:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return the maximum inconsistency coefficient for each
non-singleton cluster and its children.

Parameters
----------
Z : ndarray
    The hierarchical clustering encoded as a matrix. See
    `linkage` for more information.
R : ndarray
    The inconsistency matrix.

Returns
-------
MI : ndarray
    A monotonic ``(n-1)``-sized numpy array of doubles.

See Also
--------
linkage: for a description of what a linkage matrix is.
inconsistent: for the creation of a inconsistency matrix.

Examples
--------
>>> from scipy.cluster.hierarchy import median, inconsistent, maxinconsts
>>> from scipy.spatial.distance import pdist

Given a data set ``X``, we can apply a clustering method to obtain a
linkage matrix ``Z``. `scipy.cluster.hierarchy.inconsistent` can
be also used to obtain the inconsistency matrix ``R`` associated to
this clustering process:

>>> X = [[0, 0], [0, 1], [1, 0],
...      [0, 4], [0, 3], [1, 4],
...      [4, 0], [3, 0], [4, 1],
...      [4, 4], [3, 4], [4, 3]]

>>> Z = median(pdist(X))
>>> R = inconsistent(Z)
>>> Z
array([[ 0.        ,  1.        ,  1.        ,  2.        ],
       [ 3.        ,  4.        ,  1.        ,  2.        ],
       [ 9.        , 10.        ,  1.        ,  2.        ],
       [ 6.        ,  7.        ,  1.        ,  2.        ],
       [ 2.        , 12.        ,  1.11803399,  3.        ],
       [ 5.        , 13.        ,  1.11803399,  3.        ],
       [ 8.        , 15.        ,  1.11803399,  3.        ],
       [11.        , 14.        ,  1.11803399,  3.        ],
       [18.        , 19.        ,  3.        ,  6.        ],
       [16.        , 17.        ,  3.5       ,  6.        ],
       [20.        , 21.        ,  3.25      , 12.        ]])
>>> R
array([[1.        , 0.        , 1.        , 0.        ],
       [1.        , 0.        , 1.        , 0.        ],
       [1.        , 0.        , 1.        , 0.        ],
       [1.        , 0.        , 1.        , 0.        ],
       [1.05901699, 0.08346263, 2.        , 0.70710678],
       [1.05901699, 0.08346263, 2.        , 0.70710678],
       [1.05901699, 0.08346263, 2.        , 0.70710678],
       [1.05901699, 0.08346263, 2.        , 0.70710678],
       [1.74535599, 1.08655358, 3.        , 1.15470054],
       [1.91202266, 1.37522872, 3.        , 1.15470054],
       [3.25      , 0.25      , 3.        , 0.        ]])

Here `scipy.cluster.hierarchy.maxinconsts` can be used to compute
the maximum value of the inconsistency statistic (the last column of
``R``) for each non-singleton cluster and its children:

>>> maxinconsts(Z, R)
array([0.        , 0.        , 0.        , 0.        , 0.70710678,
       0.70710678, 0.70710678, 0.70710678, 1.15470054, 1.15470054,
       1.15470054])
*)

val median : [>`Ndarray] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Perform median/WPGMC linkage.

See `linkage` for more information on the return structure
and algorithm.

 The following are common calling conventions:

 1. ``Z = median(y)``

    Performs median/WPGMC linkage on the condensed distance matrix
    ``y``.  See ``linkage`` for more information on the return
    structure and algorithm.

 2. ``Z = median(X)``

    Performs median/WPGMC linkage on the observation matrix ``X``
    using Euclidean distance as the distance metric. See `linkage`
    for more information on the return structure and algorithm.

Parameters
----------
y : ndarray
    A condensed distance matrix. A condensed
    distance matrix is a flat array containing the upper
    triangular of the distance matrix. This is the form that
    ``pdist`` returns.  Alternatively, a collection of
    m observation vectors in n dimensions may be passed as
    a m by n array.

Returns
-------
Z : ndarray
    The hierarchical clustering encoded as a linkage matrix.

See Also
--------
linkage: for advanced creation of hierarchical clusterings.
scipy.spatial.distance.pdist : pairwise distance metrics

Examples
--------
>>> from scipy.cluster.hierarchy import median, fcluster
>>> from scipy.spatial.distance import pdist

First we need a toy dataset to play with::

    x x    x x
    x        x

    x        x
    x x    x x

>>> X = [[0, 0], [0, 1], [1, 0],
...      [0, 4], [0, 3], [1, 4],
...      [4, 0], [3, 0], [4, 1],
...      [4, 4], [3, 4], [4, 3]]

Then we get a condensed distance matrix from this dataset:

>>> y = pdist(X)

Finally, we can perform the clustering:

>>> Z = median(y)
>>> Z
array([[ 0.        ,  1.        ,  1.        ,  2.        ],
       [ 3.        ,  4.        ,  1.        ,  2.        ],
       [ 9.        , 10.        ,  1.        ,  2.        ],
       [ 6.        ,  7.        ,  1.        ,  2.        ],
       [ 2.        , 12.        ,  1.11803399,  3.        ],
       [ 5.        , 13.        ,  1.11803399,  3.        ],
       [ 8.        , 15.        ,  1.11803399,  3.        ],
       [11.        , 14.        ,  1.11803399,  3.        ],
       [18.        , 19.        ,  3.        ,  6.        ],
       [16.        , 17.        ,  3.5       ,  6.        ],
       [20.        , 21.        ,  3.25      , 12.        ]])

The linkage matrix ``Z`` represents a dendrogram - see
`scipy.cluster.hierarchy.linkage` for a detailed explanation of its
contents.

We can use `scipy.cluster.hierarchy.fcluster` to see to which cluster
each initial point would belong given a distance threshold:

>>> fcluster(Z, 0.9, criterion='distance')
array([ 7,  8,  9, 10, 11, 12,  1,  2,  3,  4,  5,  6], dtype=int32)
>>> fcluster(Z, 1.1, criterion='distance')
array([5, 5, 6, 7, 7, 8, 1, 1, 2, 3, 3, 4], dtype=int32)
>>> fcluster(Z, 2, criterion='distance')
array([3, 3, 3, 4, 4, 4, 1, 1, 1, 2, 2, 2], dtype=int32)
>>> fcluster(Z, 4, criterion='distance')
array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)

Also `scipy.cluster.hierarchy.dendrogram` can be used to generate a
plot of the dendrogram.
*)

val num_obs_linkage : [>`Ndarray] Np.Obj.t -> int
(**
Return the number of original observations of the linkage matrix passed.

Parameters
----------
Z : ndarray
    The linkage matrix on which to perform the operation.

Returns
-------
n : int
    The number of original observations in the linkage.

Examples
--------
>>> from scipy.cluster.hierarchy import ward, num_obs_linkage
>>> from scipy.spatial.distance import pdist

>>> X = [[0, 0], [0, 1], [1, 0],
...      [0, 4], [0, 3], [1, 4],
...      [4, 0], [3, 0], [4, 1],
...      [4, 4], [3, 4], [4, 3]]

>>> Z = ward(pdist(X))

``Z`` is a linkage matrix obtained after using the Ward clustering method
with ``X``, a dataset with 12 data points.

>>> num_obs_linkage(Z)
12
*)

val optimal_leaf_ordering : ?metric:[`Callable of Py.Object.t | `S of string] -> z:[>`Ndarray] Np.Obj.t -> y:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Given a linkage matrix Z and distance, reorder the cut tree.

Parameters
----------
Z : ndarray
    The hierarchical clustering encoded as a linkage matrix. See
    `linkage` for more information on the return structure and
    algorithm.
y : ndarray
    The condensed distance matrix from which Z was generated.
    Alternatively, a collection of m observation vectors in n
    dimensions may be passed as a m by n array.
metric : str or function, optional
    The distance metric to use in the case that y is a collection of
    observation vectors; ignored otherwise. See the ``pdist``
    function for a list of valid distance metrics. A custom distance
    function can also be used.

Returns
-------
Z_ordered : ndarray
    A copy of the linkage matrix Z, reordered to minimize the distance
    between adjacent leaves.

Examples
--------
>>> from scipy.cluster import hierarchy
>>> np.random.seed(23)
>>> X = np.random.randn(10,10)
>>> Z = hierarchy.ward(X)
>>> hierarchy.leaves_list(Z)
array([0, 5, 3, 9, 6, 8, 1, 4, 2, 7], dtype=int32)
>>> hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z, X))
array([3, 9, 0, 5, 8, 2, 7, 4, 1, 6], dtype=int32)
*)

val set_link_color_palette : [`StringList of string list | `None] -> Py.Object.t
(**
Set list of matplotlib color codes for use by dendrogram.

Note that this palette is global (i.e. setting it once changes the colors
for all subsequent calls to `dendrogram`) and that it affects only the
the colors below ``color_threshold``.

Note that `dendrogram` also accepts a custom coloring function through its
``link_color_func`` keyword, which is more flexible and non-global.

Parameters
----------
palette : list of str or None
    A list of matplotlib color codes.  The order of the color codes is the
    order in which the colors are cycled through when color thresholding in
    the dendrogram.

    If ``None``, resets the palette to its default (which is
    ``['g', 'r', 'c', 'm', 'y', 'k']``).

Returns
-------
None

See Also
--------
dendrogram

Notes
-----
Ability to reset the palette with ``None`` added in SciPy 0.17.0.

Examples
--------
>>> from scipy.cluster import hierarchy
>>> ytdist = np.array([662., 877., 255., 412., 996., 295., 468., 268.,
...                    400., 754., 564., 138., 219., 869., 669.])
>>> Z = hierarchy.linkage(ytdist, 'single')
>>> dn = hierarchy.dendrogram(Z, no_plot=True)
>>> dn['color_list']
['g', 'b', 'b', 'b', 'b']
>>> hierarchy.set_link_color_palette(['c', 'm', 'y', 'k'])
>>> dn = hierarchy.dendrogram(Z, no_plot=True)
>>> dn['color_list']
['c', 'b', 'b', 'b', 'b']
>>> dn = hierarchy.dendrogram(Z, no_plot=True, color_threshold=267,
...                           above_threshold_color='k')
>>> dn['color_list']
['c', 'm', 'm', 'k', 'k']

Now reset the color palette to its default:

>>> hierarchy.set_link_color_palette(None)
*)

val single : [>`Ndarray] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Perform single/min/nearest linkage on the condensed distance matrix ``y``.

Parameters
----------
y : ndarray
    The upper triangular of the distance matrix. The result of
    ``pdist`` is returned in this form.

Returns
-------
Z : ndarray
    The linkage matrix.

See Also
--------
linkage: for advanced creation of hierarchical clusterings.
scipy.spatial.distance.pdist : pairwise distance metrics

Examples
--------
>>> from scipy.cluster.hierarchy import single, fcluster
>>> from scipy.spatial.distance import pdist

First we need a toy dataset to play with::

    x x    x x
    x        x

    x        x
    x x    x x

>>> X = [[0, 0], [0, 1], [1, 0],
...      [0, 4], [0, 3], [1, 4],
...      [4, 0], [3, 0], [4, 1],
...      [4, 4], [3, 4], [4, 3]]

Then we get a condensed distance matrix from this dataset:

>>> y = pdist(X)

Finally, we can perform the clustering:

>>> Z = single(y)
>>> Z
array([[ 0.,  1.,  1.,  2.],
       [ 2., 12.,  1.,  3.],
       [ 3.,  4.,  1.,  2.],
       [ 5., 14.,  1.,  3.],
       [ 6.,  7.,  1.,  2.],
       [ 8., 16.,  1.,  3.],
       [ 9., 10.,  1.,  2.],
       [11., 18.,  1.,  3.],
       [13., 15.,  2.,  6.],
       [17., 20.,  2.,  9.],
       [19., 21.,  2., 12.]])

The linkage matrix ``Z`` represents a dendrogram - see
`scipy.cluster.hierarchy.linkage` for a detailed explanation of its
contents.

We can use `scipy.cluster.hierarchy.fcluster` to see to which cluster
each initial point would belong given a distance threshold:

>>> fcluster(Z, 0.9, criterion='distance')
array([ 7,  8,  9, 10, 11, 12,  4,  5,  6,  1,  2,  3], dtype=int32)
>>> fcluster(Z, 1, criterion='distance')
array([3, 3, 3, 4, 4, 4, 2, 2, 2, 1, 1, 1], dtype=int32)
>>> fcluster(Z, 2, criterion='distance')
array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)

Also `scipy.cluster.hierarchy.dendrogram` can be used to generate a
plot of the dendrogram.
*)

val to_mlab_linkage : [>`Ndarray] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Convert a linkage matrix to a MATLAB(TM) compatible one.

Converts a linkage matrix ``Z`` generated by the linkage function
of this module to a MATLAB(TM) compatible one. The return linkage
matrix has the last column removed and the cluster indices are
converted to ``1..N`` indexing.

Parameters
----------
Z : ndarray
    A linkage matrix generated by ``scipy.cluster.hierarchy``.

Returns
-------
to_mlab_linkage : ndarray
    A linkage matrix compatible with MATLAB(TM)'s hierarchical
    clustering functions.

    The return linkage matrix has the last column removed
    and the cluster indices are converted to ``1..N`` indexing.

See Also
--------
linkage: for a description of what a linkage matrix is.
from_mlab_linkage: transform from Matlab to SciPy format.

Examples
--------
>>> from scipy.cluster.hierarchy import ward, to_mlab_linkage
>>> from scipy.spatial.distance import pdist

>>> X = [[0, 0], [0, 1], [1, 0],
...      [0, 4], [0, 3], [1, 4],
...      [4, 0], [3, 0], [4, 1],
...      [4, 4], [3, 4], [4, 3]]

>>> Z = ward(pdist(X))
>>> Z
array([[ 0.        ,  1.        ,  1.        ,  2.        ],
       [ 3.        ,  4.        ,  1.        ,  2.        ],
       [ 6.        ,  7.        ,  1.        ,  2.        ],
       [ 9.        , 10.        ,  1.        ,  2.        ],
       [ 2.        , 12.        ,  1.29099445,  3.        ],
       [ 5.        , 13.        ,  1.29099445,  3.        ],
       [ 8.        , 14.        ,  1.29099445,  3.        ],
       [11.        , 15.        ,  1.29099445,  3.        ],
       [16.        , 17.        ,  5.77350269,  6.        ],
       [18.        , 19.        ,  5.77350269,  6.        ],
       [20.        , 21.        ,  8.16496581, 12.        ]])

After a linkage matrix ``Z`` has been created, we can use
`scipy.cluster.hierarchy.to_mlab_linkage` to convert it
into MATLAB format:

>>> mZ = to_mlab_linkage(Z)
>>> mZ
array([[  1.        ,   2.        ,   1.        ],
       [  4.        ,   5.        ,   1.        ],
       [  7.        ,   8.        ,   1.        ],
       [ 10.        ,  11.        ,   1.        ],
       [  3.        ,  13.        ,   1.29099445],
       [  6.        ,  14.        ,   1.29099445],
       [  9.        ,  15.        ,   1.29099445],
       [ 12.        ,  16.        ,   1.29099445],
       [ 17.        ,  18.        ,   5.77350269],
       [ 19.        ,  20.        ,   5.77350269],
       [ 21.        ,  22.        ,   8.16496581]])

The new linkage matrix ``mZ`` uses 1-indexing for all the
clusters (instead of 0-indexing). Also, the last column of
the original linkage matrix has been dropped.
*)

val to_tree : ?rd:bool -> z:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Convert a linkage matrix into an easy-to-use tree object.

The reference to the root `ClusterNode` object is returned (by default).

Each `ClusterNode` object has a ``left``, ``right``, ``dist``, ``id``,
and ``count`` attribute. The left and right attributes point to
ClusterNode objects that were combined to generate the cluster.
If both are None then the `ClusterNode` object is a leaf node, its count
must be 1, and its distance is meaningless but set to 0.

*Note: This function is provided for the convenience of the library
user. ClusterNodes are not used as input to any of the functions in this
library.*

Parameters
----------
Z : ndarray
    The linkage matrix in proper form (see the `linkage`
    function documentation).
rd : bool, optional
    When False (default), a reference to the root `ClusterNode` object is
    returned.  Otherwise, a tuple ``(r, d)`` is returned. ``r`` is a
    reference to the root node while ``d`` is a list of `ClusterNode`
    objects - one per original entry in the linkage matrix plus entries
    for all clustering steps.  If a cluster id is
    less than the number of samples ``n`` in the data that the linkage
    matrix describes, then it corresponds to a singleton cluster (leaf
    node).
    See `linkage` for more information on the assignment of cluster ids
    to clusters.

Returns
-------
tree : ClusterNode or tuple (ClusterNode, list of ClusterNode)
    If ``rd`` is False, a `ClusterNode`.
    If ``rd`` is True, a list of length ``2*n - 1``, with ``n`` the number
    of samples.  See the description of `rd` above for more details.

See Also
--------
linkage, is_valid_linkage, ClusterNode

Examples
--------
>>> from scipy.cluster import hierarchy
>>> x = np.random.rand(10).reshape(5, 2)
>>> Z = hierarchy.linkage(x)
>>> hierarchy.to_tree(Z)
<scipy.cluster.hierarchy.ClusterNode object at ...
>>> rootnode, nodelist = hierarchy.to_tree(Z, rd=True)
>>> rootnode
<scipy.cluster.hierarchy.ClusterNode object at ...
>>> len(nodelist)
9
*)

val ward : [>`Ndarray] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Perform Ward's linkage on a condensed distance matrix.

See `linkage` for more information on the return structure
and algorithm.

The following are common calling conventions:

1. ``Z = ward(y)``
   Performs Ward's linkage on the condensed distance matrix ``y``.

2. ``Z = ward(X)``
   Performs Ward's linkage on the observation matrix ``X`` using
   Euclidean distance as the distance metric.

Parameters
----------
y : ndarray
    A condensed distance matrix. A condensed
    distance matrix is a flat array containing the upper
    triangular of the distance matrix. This is the form that
    ``pdist`` returns.  Alternatively, a collection of
    m observation vectors in n dimensions may be passed as
    a m by n array.

Returns
-------
Z : ndarray
    The hierarchical clustering encoded as a linkage matrix. See
    `linkage` for more information on the return structure and
    algorithm.

See Also
--------
linkage: for advanced creation of hierarchical clusterings.
scipy.spatial.distance.pdist : pairwise distance metrics

Examples
--------
>>> from scipy.cluster.hierarchy import ward, fcluster
>>> from scipy.spatial.distance import pdist

First we need a toy dataset to play with::

    x x    x x
    x        x

    x        x
    x x    x x

>>> X = [[0, 0], [0, 1], [1, 0],
...      [0, 4], [0, 3], [1, 4],
...      [4, 0], [3, 0], [4, 1],
...      [4, 4], [3, 4], [4, 3]]

Then we get a condensed distance matrix from this dataset:

>>> y = pdist(X)

Finally, we can perform the clustering:

>>> Z = ward(y)
>>> Z
array([[ 0.        ,  1.        ,  1.        ,  2.        ],
       [ 3.        ,  4.        ,  1.        ,  2.        ],
       [ 6.        ,  7.        ,  1.        ,  2.        ],
       [ 9.        , 10.        ,  1.        ,  2.        ],
       [ 2.        , 12.        ,  1.29099445,  3.        ],
       [ 5.        , 13.        ,  1.29099445,  3.        ],
       [ 8.        , 14.        ,  1.29099445,  3.        ],
       [11.        , 15.        ,  1.29099445,  3.        ],
       [16.        , 17.        ,  5.77350269,  6.        ],
       [18.        , 19.        ,  5.77350269,  6.        ],
       [20.        , 21.        ,  8.16496581, 12.        ]])

The linkage matrix ``Z`` represents a dendrogram - see
`scipy.cluster.hierarchy.linkage` for a detailed explanation of its
contents.

We can use `scipy.cluster.hierarchy.fcluster` to see to which cluster
each initial point would belong given a distance threshold:

>>> fcluster(Z, 0.9, criterion='distance')
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12], dtype=int32)
>>> fcluster(Z, 1.1, criterion='distance')
array([1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8], dtype=int32)
>>> fcluster(Z, 3, criterion='distance')
array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], dtype=int32)
>>> fcluster(Z, 9, criterion='distance')
array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)

Also `scipy.cluster.hierarchy.dendrogram` can be used to generate a
plot of the dendrogram.
*)

val weighted : [>`Ndarray] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Perform weighted/WPGMA linkage on the condensed distance matrix.

See `linkage` for more information on the return
structure and algorithm.

Parameters
----------
y : ndarray
    The upper triangular of the distance matrix. The result of
    ``pdist`` is returned in this form.

Returns
-------
Z : ndarray
    A linkage matrix containing the hierarchical clustering. See
    `linkage` for more information on its structure.

See Also
--------
linkage : for advanced creation of hierarchical clusterings.
scipy.spatial.distance.pdist : pairwise distance metrics

Examples
--------
>>> from scipy.cluster.hierarchy import weighted, fcluster
>>> from scipy.spatial.distance import pdist

First we need a toy dataset to play with::

    x x    x x
    x        x

    x        x
    x x    x x

>>> X = [[0, 0], [0, 1], [1, 0],
...      [0, 4], [0, 3], [1, 4],
...      [4, 0], [3, 0], [4, 1],
...      [4, 4], [3, 4], [4, 3]]

Then we get a condensed distance matrix from this dataset:

>>> y = pdist(X)

Finally, we can perform the clustering:

>>> Z = weighted(y)
>>> Z
array([[ 0.        ,  1.        ,  1.        ,  2.        ],
       [ 6.        ,  7.        ,  1.        ,  2.        ],
       [ 3.        ,  4.        ,  1.        ,  2.        ],
       [ 9.        , 11.        ,  1.        ,  2.        ],
       [ 2.        , 12.        ,  1.20710678,  3.        ],
       [ 8.        , 13.        ,  1.20710678,  3.        ],
       [ 5.        , 14.        ,  1.20710678,  3.        ],
       [10.        , 15.        ,  1.20710678,  3.        ],
       [18.        , 19.        ,  3.05595762,  6.        ],
       [16.        , 17.        ,  3.32379407,  6.        ],
       [20.        , 21.        ,  4.06357713, 12.        ]])

The linkage matrix ``Z`` represents a dendrogram - see
`scipy.cluster.hierarchy.linkage` for a detailed explanation of its
contents.

We can use `scipy.cluster.hierarchy.fcluster` to see to which cluster
each initial point would belong given a distance threshold:

>>> fcluster(Z, 0.9, criterion='distance')
array([ 7,  8,  9,  1,  2,  3, 10, 11, 12,  4,  6,  5], dtype=int32)
>>> fcluster(Z, 1.5, criterion='distance')
array([3, 3, 3, 1, 1, 1, 4, 4, 4, 2, 2, 2], dtype=int32)
>>> fcluster(Z, 4, criterion='distance')
array([2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1], dtype=int32)
>>> fcluster(Z, 6, criterion='distance')
array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)

Also `scipy.cluster.hierarchy.dendrogram` can be used to generate a
plot of the dendrogram.
*)


end

module Vq : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module ClusterError : sig
type tag = [`ClusterError]
type t = [`BaseException | `ClusterError | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_exception : t -> [`BaseException] Obj.t
val with_traceback : tb:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Exception.with_traceback(tb) --
set self.__traceback__ to tb and return self.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Deque : sig
type tag = [`Deque]
type t = [`Deque | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val __getitem__ : key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return self[key].
*)

val __iter__ : [> tag] Obj.t -> Py.Object.t
(**
Implement iter(self).
*)

val __setitem__ : key:Py.Object.t -> value:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Set self[key] to value.
*)

val count : value:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
D.count(value) -> integer -- return number of occurrences of value
*)

val insert : index:Py.Object.t -> object_:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
D.insert(index, object) -- insert object before index
*)

val remove : value:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
D.remove(value) -- remove first occurrence of value.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Xrange : sig
type tag = [`Range]
type t = [`Object | `Range] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : Py.Object.t -> t
(**
range(stop) -> range object
range(start, stop[, step]) -> range object

Return an object that produces a sequence of integers from start (inclusive)
to stop (exclusive) by step.  range(i, j) produces i, i+1, i+2, ..., j-1.
start defaults to 0, and stop is omitted!  range(4) produces 0, 1, 2, 3.
These are exactly the valid indices for a list of 4 elements.
When step is given, it specifies the increment (or decrement).
*)

val __getitem__ : key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return self[key].
*)

val __iter__ : [> tag] Obj.t -> Py.Object.t
(**
Implement iter(self).
*)

val count : value:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
rangeobject.count(value) -> integer -- return number of occurrences of value
*)

val index : value:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
rangeobject.index(value) -> integer -- return index of value.
Raise ValueError if the value is not present.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val cdist : ?metric:[`Callable of Py.Object.t | `S of string] -> ?kwargs:(string * Py.Object.t) list -> xa:[>`Ndarray] Np.Obj.t -> xb:[>`Ndarray] Np.Obj.t -> Py.Object.t list -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute distance between each pair of the two collections of inputs.

See Notes for common calling conventions.

Parameters
----------
XA : ndarray
    An :math:`m_A` by :math:`n` array of :math:`m_A`
    original observations in an :math:`n`-dimensional space.
    Inputs are converted to float type.
XB : ndarray
    An :math:`m_B` by :math:`n` array of :math:`m_B`
    original observations in an :math:`n`-dimensional space.
    Inputs are converted to float type.
metric : str or callable, optional
    The distance metric to use.  If a string, the distance function can be
    'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
    'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
    'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
    'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
    'wminkowski', 'yule'.
*args : tuple. Deprecated.
    Additional arguments should be passed as keyword arguments
**kwargs : dict, optional
    Extra arguments to `metric`: refer to each metric documentation for a
    list of all possible arguments.

    Some possible arguments:

    p : scalar
    The p-norm to apply for Minkowski, weighted and unweighted.
    Default: 2.

    w : ndarray
    The weight vector for metrics that support weights (e.g., Minkowski).

    V : ndarray
    The variance vector for standardized Euclidean.
    Default: var(vstack([XA, XB]), axis=0, ddof=1)

    VI : ndarray
    The inverse of the covariance matrix for Mahalanobis.
    Default: inv(cov(vstack([XA, XB].T))).T

    out : ndarray
    The output array
    If not None, the distance matrix Y is stored in this array.
    Note: metric independent, it will become a regular keyword arg in a
    future scipy version

Returns
-------
Y : ndarray
    A :math:`m_A` by :math:`m_B` distance matrix is returned.
    For each :math:`i` and :math:`j`, the metric
    ``dist(u=XA[i], v=XB[j])`` is computed and stored in the
    :math:`ij` th entry.

Raises
------
ValueError
    An exception is thrown if `XA` and `XB` do not have
    the same number of columns.

Notes
-----
The following are common calling conventions:

1. ``Y = cdist(XA, XB, 'euclidean')``

   Computes the distance between :math:`m` points using
   Euclidean distance (2-norm) as the distance metric between the
   points. The points are arranged as :math:`m`
   :math:`n`-dimensional row vectors in the matrix X.

2. ``Y = cdist(XA, XB, 'minkowski', p=2.)``

   Computes the distances using the Minkowski distance
   :math:`||u-v||_p` (:math:`p`-norm) where :math:`p \geq 1`.

3. ``Y = cdist(XA, XB, 'cityblock')``

   Computes the city block or Manhattan distance between the
   points.

4. ``Y = cdist(XA, XB, 'seuclidean', V=None)``

   Computes the standardized Euclidean distance. The standardized
   Euclidean distance between two n-vectors ``u`` and ``v`` is

   .. math::

      \sqrt{\sum {(u_i-v_i)^2 / V[x_i]}}.

   V is the variance vector; V[i] is the variance computed over all
   the i'th components of the points. If not passed, it is
   automatically computed.

5. ``Y = cdist(XA, XB, 'sqeuclidean')``

   Computes the squared Euclidean distance :math:`||u-v||_2^2` between
   the vectors.

6. ``Y = cdist(XA, XB, 'cosine')``

   Computes the cosine distance between vectors u and v,

   .. math::

      1 - \frac{u \cdot v}
               {{ ||u|| }_2 { ||v|| }_2}

   where :math:`||*||_2` is the 2-norm of its argument ``*``, and
   :math:`u \cdot v` is the dot product of :math:`u` and :math:`v`.

7. ``Y = cdist(XA, XB, 'correlation')``

   Computes the correlation distance between vectors u and v. This is

   .. math::

      1 - \frac{(u - \bar{u}) \cdot (v - \bar{v})}
               {{ ||(u - \bar{u})|| }_2 { ||(v - \bar{v})|| }_2}

   where :math:`\bar{v}` is the mean of the elements of vector v,
   and :math:`x \cdot y` is the dot product of :math:`x` and :math:`y`.


8. ``Y = cdist(XA, XB, 'hamming')``

   Computes the normalized Hamming distance, or the proportion of
   those vector elements between two n-vectors ``u`` and ``v``
   which disagree. To save memory, the matrix ``X`` can be of type
   boolean.

9. ``Y = cdist(XA, XB, 'jaccard')``

   Computes the Jaccard distance between the points. Given two
   vectors, ``u`` and ``v``, the Jaccard distance is the
   proportion of those elements ``u[i]`` and ``v[i]`` that
   disagree where at least one of them is non-zero.

10. ``Y = cdist(XA, XB, 'chebyshev')``

   Computes the Chebyshev distance between the points. The
   Chebyshev distance between two n-vectors ``u`` and ``v`` is the
   maximum norm-1 distance between their respective elements. More
   precisely, the distance is given by

   .. math::

      d(u,v) = \max_i { |u_i-v_i| }.

11. ``Y = cdist(XA, XB, 'canberra')``

   Computes the Canberra distance between the points. The
   Canberra distance between two points ``u`` and ``v`` is

   .. math::

     d(u,v) = \sum_i \frac{ |u_i-v_i| }
                          { |u_i|+|v_i| }.

12. ``Y = cdist(XA, XB, 'braycurtis')``

   Computes the Bray-Curtis distance between the points. The
   Bray-Curtis distance between two points ``u`` and ``v`` is


   .. math::

        d(u,v) = \frac{\sum_i (|u_i-v_i|)}
                      {\sum_i (|u_i+v_i|)}

13. ``Y = cdist(XA, XB, 'mahalanobis', VI=None)``

   Computes the Mahalanobis distance between the points. The
   Mahalanobis distance between two points ``u`` and ``v`` is
   :math:`\sqrt{(u-v)(1/V)(u-v)^T}` where :math:`(1/V)` (the ``VI``
   variable) is the inverse covariance. If ``VI`` is not None,
   ``VI`` will be used as the inverse covariance matrix.

14. ``Y = cdist(XA, XB, 'yule')``

   Computes the Yule distance between the boolean
   vectors. (see `yule` function documentation)

15. ``Y = cdist(XA, XB, 'matching')``

   Synonym for 'hamming'.

16. ``Y = cdist(XA, XB, 'dice')``

   Computes the Dice distance between the boolean vectors. (see
   `dice` function documentation)

17. ``Y = cdist(XA, XB, 'kulsinski')``

   Computes the Kulsinski distance between the boolean
   vectors. (see `kulsinski` function documentation)

18. ``Y = cdist(XA, XB, 'rogerstanimoto')``

   Computes the Rogers-Tanimoto distance between the boolean
   vectors. (see `rogerstanimoto` function documentation)

19. ``Y = cdist(XA, XB, 'russellrao')``

   Computes the Russell-Rao distance between the boolean
   vectors. (see `russellrao` function documentation)

20. ``Y = cdist(XA, XB, 'sokalmichener')``

   Computes the Sokal-Michener distance between the boolean
   vectors. (see `sokalmichener` function documentation)

21. ``Y = cdist(XA, XB, 'sokalsneath')``

   Computes the Sokal-Sneath distance between the vectors. (see
   `sokalsneath` function documentation)


22. ``Y = cdist(XA, XB, 'wminkowski', p=2., w=w)``

   Computes the weighted Minkowski distance between the
   vectors. (see `wminkowski` function documentation)

23. ``Y = cdist(XA, XB, f)``

   Computes the distance between all pairs of vectors in X
   using the user supplied 2-arity function f. For example,
   Euclidean distance between the vectors could be computed
   as follows::

     dm = cdist(XA, XB, lambda u, v: np.sqrt(((u-v)**2).sum()))

   Note that you should avoid passing a reference to one of
   the distance functions defined in this library. For example,::

     dm = cdist(XA, XB, sokalsneath)

   would calculate the pair-wise distances between the vectors in
   X using the Python function `sokalsneath`. This would result in
   sokalsneath being called :math:`{n \choose 2}` times, which
   is inefficient. Instead, the optimized C version is more
   efficient, and we call it using the following syntax::

     dm = cdist(XA, XB, 'sokalsneath')

Examples
--------
Find the Euclidean distances between four 2-D coordinates:

>>> from scipy.spatial import distance
>>> coords = [(35.0456, -85.2672),
...           (35.1174, -89.9711),
...           (35.9728, -83.9422),
...           (36.1667, -86.7833)]
>>> distance.cdist(coords, coords, 'euclidean')
array([[ 0.    ,  4.7044,  1.6172,  1.8856],
       [ 4.7044,  0.    ,  6.0893,  3.3561],
       [ 1.6172,  6.0893,  0.    ,  2.8477],
       [ 1.8856,  3.3561,  2.8477,  0.    ]])


Find the Manhattan distance from a 3-D point to the corners of the unit
cube:

>>> a = np.array([[0, 0, 0],
...               [0, 0, 1],
...               [0, 1, 0],
...               [0, 1, 1],
...               [1, 0, 0],
...               [1, 0, 1],
...               [1, 1, 0],
...               [1, 1, 1]])
>>> b = np.array([[ 0.1,  0.2,  0.4]])
>>> distance.cdist(a, b, 'cityblock')
array([[ 0.7],
       [ 0.9],
       [ 1.3],
       [ 1.5],
       [ 1.5],
       [ 1.7],
       [ 2.1],
       [ 2.3]])
*)

val kmeans : ?iter:int -> ?thresh:float -> ?check_finite:bool -> obs:[>`Ndarray] Np.Obj.t -> k_or_guess:[`Ndarray of [>`Ndarray] Np.Obj.t | `I of int] -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * float)
(**
Performs k-means on a set of observation vectors forming k clusters.

The k-means algorithm adjusts the classification of the observations
into clusters and updates the cluster centroids until the position of
the centroids is stable over successive iterations. In this
implementation of the algorithm, the stability of the centroids is
determined by comparing the absolute value of the change in the average
Euclidean distance between the observations and their corresponding
centroids against a threshold. This yields
a code book mapping centroids to codes and vice versa.

Parameters
----------
obs : ndarray
   Each row of the M by N array is an observation vector. The
   columns are the features seen during each observation.
   The features must be whitened first with the `whiten` function.

k_or_guess : int or ndarray
   The number of centroids to generate. A code is assigned to
   each centroid, which is also the row index of the centroid
   in the code_book matrix generated.

   The initial k centroids are chosen by randomly selecting
   observations from the observation matrix. Alternatively,
   passing a k by N array specifies the initial k centroids.

iter : int, optional
   The number of times to run k-means, returning the codebook
   with the lowest distortion. This argument is ignored if
   initial centroids are specified with an array for the
   ``k_or_guess`` parameter. This parameter does not represent the
   number of iterations of the k-means algorithm.

thresh : float, optional
   Terminates the k-means algorithm if the change in
   distortion since the last k-means iteration is less than
   or equal to thresh.

check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.
    Default: True

Returns
-------
codebook : ndarray
   A k by N array of k centroids. The i'th centroid
   codebook[i] is represented with the code i. The centroids
   and codes generated represent the lowest distortion seen,
   not necessarily the globally minimal distortion.

distortion : float
   The mean (non-squared) Euclidean distance between the observations
   passed and the centroids generated. Note the difference to the standard
   definition of distortion in the context of the K-means algorithm, which
   is the sum of the squared distances.

See Also
--------
kmeans2 : a different implementation of k-means clustering
   with more methods for generating initial centroids but without
   using a distortion change threshold as a stopping criterion.

whiten : must be called prior to passing an observation matrix
   to kmeans.

Examples
--------
>>> from numpy import array
>>> from scipy.cluster.vq import vq, kmeans, whiten
>>> import matplotlib.pyplot as plt
>>> features  = array([[ 1.9,2.3],
...                    [ 1.5,2.5],
...                    [ 0.8,0.6],
...                    [ 0.4,1.8],
...                    [ 0.1,0.1],
...                    [ 0.2,1.8],
...                    [ 2.0,0.5],
...                    [ 0.3,1.5],
...                    [ 1.0,1.0]])
>>> whitened = whiten(features)
>>> book = np.array((whitened[0],whitened[2]))
>>> kmeans(whitened,book)
(array([[ 2.3110306 ,  2.86287398],    # random
       [ 0.93218041,  1.24398691]]), 0.85684700941625547)

>>> from numpy import random
>>> random.seed((1000,2000))
>>> codes = 3
>>> kmeans(whitened,codes)
(array([[ 2.3110306 ,  2.86287398],    # random
       [ 1.32544402,  0.65607529],
       [ 0.40782893,  2.02786907]]), 0.5196582527686241)

>>> # Create 50 datapoints in two clusters a and b
>>> pts = 50
>>> a = np.random.multivariate_normal([0, 0], [[4, 1], [1, 4]], size=pts)
>>> b = np.random.multivariate_normal([30, 10],
...                                   [[10, 2], [2, 1]],
...                                   size=pts)
>>> features = np.concatenate((a, b))
>>> # Whiten data
>>> whitened = whiten(features)
>>> # Find 2 clusters in the data
>>> codebook, distortion = kmeans(whitened, 2)
>>> # Plot whitened data and cluster centers in red
>>> plt.scatter(whitened[:, 0], whitened[:, 1])
>>> plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
>>> plt.show()
*)

val kmeans2 : ?iter:int -> ?thresh:float -> ?minit:string -> ?missing:string -> ?check_finite:bool -> data:[>`Ndarray] Np.Obj.t -> k:[`Ndarray of [>`Ndarray] Np.Obj.t | `I of int] -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Classify a set of observations into k clusters using the k-means algorithm.

The algorithm attempts to minimize the Euclidian distance between
observations and centroids. Several initialization methods are
included.

Parameters
----------
data : ndarray
    A 'M' by 'N' array of 'M' observations in 'N' dimensions or a length
    'M' array of 'M' one-dimensional observations.
k : int or ndarray
    The number of clusters to form as well as the number of
    centroids to generate. If `minit` initialization string is
    'matrix', or if a ndarray is given instead, it is
    interpreted as initial cluster to use instead.
iter : int, optional
    Number of iterations of the k-means algorithm to run. Note
    that this differs in meaning from the iters parameter to
    the kmeans function.
thresh : float, optional
    (not used yet)
minit : str, optional
    Method for initialization. Available methods are 'random',
    'points', '++' and 'matrix':

    'random': generate k centroids from a Gaussian with mean and
    variance estimated from the data.

    'points': choose k observations (rows) at random from data for
    the initial centroids.

    '++': choose k observations accordingly to the kmeans++ method
    (careful seeding)

    'matrix': interpret the k parameter as a k by M (or length k
    array for one-dimensional data) array of initial centroids.
missing : str, optional
    Method to deal with empty clusters. Available methods are
    'warn' and 'raise':

    'warn': give a warning and continue.

    'raise': raise an ClusterError and terminate the algorithm.
check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.
    Default: True

Returns
-------
centroid : ndarray
    A 'k' by 'N' array of centroids found at the last iteration of
    k-means.
label : ndarray
    label[i] is the code or index of the centroid the
    i'th observation is closest to.

See Also
--------
kmeans

References
----------
.. [1] D. Arthur and S. Vassilvitskii, 'k-means++: the advantages of
   careful seeding', Proceedings of the Eighteenth Annual ACM-SIAM Symposium
   on Discrete Algorithms, 2007.

Examples
--------
>>> from scipy.cluster.vq import kmeans2
>>> import matplotlib.pyplot as plt

Create z, an array with shape (100, 2) containing a mixture of samples
from three multivariate normal distributions.

>>> np.random.seed(12345678)
>>> a = np.random.multivariate_normal([0, 6], [[2, 1], [1, 1.5]], size=45)
>>> b = np.random.multivariate_normal([2, 0], [[1, -1], [-1, 3]], size=30)
>>> c = np.random.multivariate_normal([6, 4], [[5, 0], [0, 1.2]], size=25)
>>> z = np.concatenate((a, b, c))
>>> np.random.shuffle(z)

Compute three clusters.

>>> centroid, label = kmeans2(z, 3, minit='points')
>>> centroid
array([[-0.35770296,  5.31342524],
       [ 2.32210289, -0.50551972],
       [ 6.17653859,  4.16719247]])

How many points are in each cluster?

>>> counts = np.bincount(label)
>>> counts
array([52, 27, 21])

Plot the clusters.

>>> w0 = z[label == 0]
>>> w1 = z[label == 1]
>>> w2 = z[label == 2]
>>> plt.plot(w0[:, 0], w0[:, 1], 'o', alpha=0.5, label='cluster 0')
>>> plt.plot(w1[:, 0], w1[:, 1], 'd', alpha=0.5, label='cluster 1')
>>> plt.plot(w2[:, 0], w2[:, 1], 's', alpha=0.5, label='cluster 2')
>>> plt.plot(centroid[:, 0], centroid[:, 1], 'k*', label='centroids')
>>> plt.axis('equal')
>>> plt.legend(shadow=True)
>>> plt.show()
*)

val py_vq : ?check_finite:bool -> obs:[>`Ndarray] Np.Obj.t -> code_book:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Python version of vq algorithm.

The algorithm computes the euclidian distance between each
observation and every frame in the code_book.

Parameters
----------
obs : ndarray
    Expects a rank 2 array. Each row is one observation.
code_book : ndarray
    Code book to use. Same format than obs. Should have same number of
    features (eg columns) than obs.
check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.
    Default: True

Returns
-------
code : ndarray
    code[i] gives the label of the ith obversation, that its code is
    code_book[code[i]].
mind_dist : ndarray
    min_dist[i] gives the distance between the ith observation and its
    corresponding code.

Notes
-----
This function is slower than the C version but works for
all input types.  If the inputs have the wrong types for the
C versions of the function, this one is called as a last resort.

It is about 20 times slower than the C version.
*)

val py_vq2 : ?kwds:(string * Py.Object.t) list -> Py.Object.t list -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
`py_vq2` is deprecated, use `py_vq` instead!

 Python version of vq algorithm.

The algorithm computes the euclidian distance between each
observation and every frame in the code_book.

Parameters
----------
obs : ndarray
    Expects a rank 2 array. Each row is one observation.
code_book : ndarray
    Code book to use. Same format than obs. Should have same number of
    features (eg columns) than obs.
check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.
    Default: True

Returns
-------
code : ndarray
    code[i] gives the label of the ith obversation, that its code is
    code_book[code[i]].
mind_dist : ndarray
    min_dist[i] gives the distance between the ith observation and its
    corresponding code.

Notes
-----
This function is slower than the C version but works for
all input types.  If the inputs have the wrong types for the
C versions of the function, this one is called as a last resort.

It is about 20 times slower than the C version.
*)

val vq : ?check_finite:bool -> obs:[>`Ndarray] Np.Obj.t -> code_book:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Assign codes from a code book to observations.

Assigns a code from a code book to each observation. Each
observation vector in the 'M' by 'N' `obs` array is compared with the
centroids in the code book and assigned the code of the closest
centroid.

The features in `obs` should have unit variance, which can be
achieved by passing them through the whiten function.  The code
book can be created with the k-means algorithm or a different
encoding algorithm.

Parameters
----------
obs : ndarray
    Each row of the 'M' x 'N' array is an observation.  The columns are
    the 'features' seen during each observation. The features must be
    whitened first using the whiten function or something equivalent.
code_book : ndarray
    The code book is usually generated using the k-means algorithm.
    Each row of the array holds a different code, and the columns are
    the features of the code.

     >>> #              f0    f1    f2   f3
     >>> code_book = [
     ...             [  1.,   2.,   3.,   4.],  #c0
     ...             [  1.,   2.,   3.,   4.],  #c1
     ...             [  1.,   2.,   3.,   4.]]  #c2

check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.
    Default: True

Returns
-------
code : ndarray
    A length M array holding the code book index for each observation.
dist : ndarray
    The distortion (distance) between the observation and its nearest
    code.

Examples
--------
>>> from numpy import array
>>> from scipy.cluster.vq import vq
>>> code_book = array([[1.,1.,1.],
...                    [2.,2.,2.]])
>>> features  = array([[  1.9,2.3,1.7],
...                    [  1.5,2.5,2.2],
...                    [  0.8,0.6,1.7]])
>>> vq(features,code_book)
(array([1, 1, 0],'i'), array([ 0.43588989,  0.73484692,  0.83066239]))
*)

val whiten : ?check_finite:bool -> obs:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Normalize a group of observations on a per feature basis.

Before running k-means, it is beneficial to rescale each feature
dimension of the observation set with whitening. Each feature is
divided by its standard deviation across all observations to give
it unit variance.

Parameters
----------
obs : ndarray
    Each row of the array is an observation.  The
    columns are the features seen during each observation.

    >>> #         f0    f1    f2
    >>> obs = [[  1.,   1.,   1.],  #o0
    ...        [  2.,   2.,   2.],  #o1
    ...        [  3.,   3.,   3.],  #o2
    ...        [  4.,   4.,   4.]]  #o3

check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.
    Default: True

Returns
-------
result : ndarray
    Contains the values in `obs` scaled by the standard deviation
    of each column.

Examples
--------
>>> from scipy.cluster.vq import whiten
>>> features  = np.array([[1.9, 2.3, 1.7],
...                       [1.5, 2.5, 2.2],
...                       [0.8, 0.6, 1.7,]])
>>> whiten(features)
array([[ 4.17944278,  2.69811351,  7.21248917],
       [ 3.29956009,  2.93273208,  9.33380951],
       [ 1.75976538,  0.7038557 ,  7.21248917]])
*)


end

